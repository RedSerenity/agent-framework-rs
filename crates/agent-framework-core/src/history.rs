//! Conversation-history context providers.
//!
//! Upstream moved conversation history out of the thread/session entirely:
//! it is now just another [`ContextProvider`] — a [`HistoryProvider`] —
//! that prepends its stored messages ahead of a run (`before_run`) and
//! records the run's request + response messages after a successful run
//! (`after_run`). [`InMemoryHistoryProvider`] is the in-process default;
//! [`FileHistoryProvider`] persists to a JSON file on disk.
//!
//! [`Agent`](crate::agent::Agent) and
//! [`WorkflowAgent`](crate::workflow::WorkflowAgent) auto-attach a fresh
//! [`InMemoryHistoryProvider`] (via [`ensure_history_provider`]) to any
//! non-service-managed [`AgentSession`] that doesn't already carry a history
//! provider, so local multi-turn conversations keep accumulating history the
//! way the old `AgentThread` message store used to.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::session::AgentSession;
use crate::types::Message;

/// A [`ContextProvider`] that also manages conversation history.
///
/// This is a marker trait (over and above [`ContextProvider::is_history_provider`],
/// which drives runtime detection via trait objects): implementing it
/// documents that a type's `before_run`/`after_run` are the ones responsible
/// for a session's conversation history, distinguishing it from a generic
/// memory/RAG provider.
pub trait HistoryProvider: ContextProvider {}

/// The identity a message is matched on when aligning an incoming run against
/// already-stored history.
///
/// Mirrors upstream's `get_message_identity`: a message that carries a
/// (non-empty) `message_id` is identified by it alone, and one that does not is
/// identified by its role plus its contents. The two forms never compare equal, so an
/// id-bearing message is never confused with an id-less one that happens to
/// carry the same text.
#[derive(PartialEq)]
enum MessageIdentity<'a> {
    Id(&'a str),
    Contents(&'a crate::types::Role, &'a [crate::types::Content]),
}

fn message_identity(message: &Message) -> MessageIdentity<'_> {
    match real_message_id(message) {
        Some(id) => MessageIdentity::Id(id),
        None => MessageIdentity::Contents(&message.role, &message.contents),
    }
}

/// A message's id when it actually identifies something.
///
/// An empty string is not an identity: every message carrying one would
/// compare equal to every other, whatever its role or contents. The rest of the
/// crate already reads an empty id as absent — see the `!id.is_empty()` guard
/// in `agent::response_to_updates`'s `keep_provider_ids` — and matching that
/// here keeps a provider or caller that emits `Some("")` from collapsing an
/// entire conversation into one identity.
fn real_message_id(message: &Message) -> Option<&str> {
    message.message_id.as_deref().filter(|id| !id.is_empty())
}

/// Return the suffix of `incoming` that is not already present in `existing`,
/// so replaying a conversation does not store — or resend — it twice.
///
/// A caller that keeps its own transcript and replays all of it on every turn
/// (the AG-UI shape, and any client that tracks history itself) hands back
/// everything the provider already stored. Appending that unconditionally grows
/// history superlinearly — each turn re-storing the whole conversation on top of
/// the copy already there — and prepending it unconditionally sends every
/// replayed turn to the model twice.
///
/// **`incoming` must be messages that could be a replay** — a run's *input*.
/// Response messages were just generated and can never be a replay of stored
/// history, so they are never passed here: see [`new_run_messages`], which
/// aligns the input and appends the responses unconditionally. Aligning over
/// input and responses together would let a response that happens to reproduce
/// the stored tail swallow the genuinely new turn in front of it.
///
/// The stored run is located inside `incoming` by matching every message by
/// `MessageIdentity`; the messages after that block are the new ones. Where
/// it is looked for depends on what the provider holds, which is why
/// [`StoredHistory`] is a parameter rather than a guess: a complete history can
/// only be matched at offset `0`, while a trimmed window has to be searched
/// for. This function assumes [`StoredHistory::Complete`]; a windowed store
/// should call [`filter_new_messages_from`] instead.
///
/// When no alignment is found, **all** of `incoming` is returned: appending is
/// the behavior every provider had before this function existed, so a
/// conversation this cannot align is stored exactly as it used to be.
///
/// Two deliberate divergences from upstream, both refusing to drop a turn that
/// might be real:
///
/// - Upstream's fallback, when alignment fails, deduplicates by identity
///   against a set of everything stored. That drops a legitimately repeated
///   turn — two identical, id-less `"yes"` replies in one conversation collapse
///   to one, and the second turn's user message is lost from history
///   permanently. Here an unalignable run is simply appended.
/// - An alignment that consumes *all* of `incoming`, leaving nothing new, is
///   not treated as an alignment. Input that exactly repeats the stored tail is
///   ambiguous — a replay carrying no new turn, or a turn that genuinely
///   repeated itself verbatim — and it is read as the latter, which is what the
///   providers did before this function existed. Upstream reads it the other
///   way and stores nothing.
pub fn filter_new_messages<'a>(existing: &[Message], incoming: &'a [Message]) -> &'a [Message] {
    filter_new_messages_from(existing, incoming, StoredHistory::Complete)
}

/// What a provider's stored history is, which decides *which* occurrence of it
/// inside a replayed transcript is the one it actually holds.
///
/// The distinction only bites when the stored run occurs more than once in the
/// replay — a conversation that repeats an exchange verbatim — and the two
/// answers are opposites, so it is a caller's decision rather than a guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredHistory {
    /// Everything the conversation has said so far, in order: the provider
    /// drops nothing. The **first** occurrence is therefore the stored one, and
    /// everything after it is new — matching a later occurrence would discard
    /// the genuinely new turns in between.
    Complete,
    /// A retention-limited store's list, which *may* be a trimmed window of
    /// the most recent messages.
    ///
    /// A match at the start still wins, because a list that has not actually
    /// been trimmed yet — a first write that happened to fill the cap exactly —
    /// is still the complete conversation, and treating it as a window would
    /// drop the turns between two occurrences of it. Only when the stored run
    /// is *not* at the start is it searched for, and then the **last**
    /// occurrence is the retained window: matching an earlier one would re-send
    /// the whole middle of the transcript on every turn, messages the store is
    /// going to trim away again anyway.
    ///
    /// The ambiguous case — a genuine window whose content also opens the
    /// transcript — resolves to the anchored match, so it re-sends the middle
    /// rather than risking a lost turn. That is the right way round: the
    /// redundant writes are trimmed away, a dropped turn is not recoverable.
    Window,
}

/// [`filter_new_messages`] with an explicit [`StoredHistory`] shape.
pub fn filter_new_messages_from<'a>(
    existing: &[Message],
    incoming: &'a [Message],
    shape: StoredHistory,
) -> &'a [Message] {
    if existing.is_empty() || incoming.len() <= existing.len() || !could_be_a_replay(existing) {
        return incoming;
    }
    let matches = |start: usize| {
        incoming[start..start + existing.len()]
            .iter()
            .zip(existing)
            .all(|(a, b)| message_identity(a) == message_identity(b))
    };
    let found = match shape {
        // A complete history starts at the conversation's first message, so a
        // replay of it can only *begin* with it. Matching at a later offset
        // would mean the input carried turns from before the conversation
        // started — impossible — and a coincidental match there would silently
        // drop every genuinely new message in front of it.
        StoredHistory::Complete => matches(0).then_some(0),
        // A window may sit in the middle of the transcript, so it has to be
        // searched for — but an anchored match still wins, since an at-cap list
        // that has never actually been trimmed is still a complete history.
        StoredHistory::Window => matches(0).then_some(0).or_else(|| {
            (1..(incoming.len() - existing.len()))
                .rev()
                .find(|s| matches(*s))
        }),
    };
    match found {
        Some(start) => &incoming[start + existing.len()..],
        None => incoming,
    }
}

/// Whether stored history could be a *replay* at all when it turns up inside a
/// run's input, or whether a match could only ever be a coincidence.
///
/// Matching on content alone cannot tell a replayed transcript from new input
/// that happens to repeat it — with stored `[user("yes")]`, an input of
/// `[user("yes"), user("question")]` is equally well a caller replaying its one
/// stored turn or a caller saying "yes" again and asking something. Treating it
/// as a replay drops a real turn; treating it as new duplicates one. Neither
/// content nor length separates them, so this asks what kind of evidence the
/// stored messages carry:
///
/// - **A message id.** Ids are assigned, not guessed, so an id that matches is
///   the replayed message, full stop.
/// - **A non-user turn.** A replay is a *transcript*: it carries the assistant
///   (and tool) turns the conversation produced. A caller sending genuinely new
///   input sends its own turns, which are user messages — it does not compose
///   the assistant's replies. So stored history that is nothing but id-less user
///   messages is never read as a replay.
///
/// The cost is declining to deduplicate a genuine replay of a user-only,
/// id-less history — a store whose retention window happens to hold no
/// assistant turn, say — which then appends exactly as it did before any of
/// this existed. That is the safe direction: a redundant write is trimmed away,
/// a dropped turn is not recoverable.
fn could_be_a_replay(existing: &[Message]) -> bool {
    existing
        .iter()
        .any(|m| real_message_id(m).is_some() || m.role != crate::types::Role::user())
}

/// What a run adds to `existing`: the part of its **input** that is not a
/// replay of already-stored history (see [`filter_new_messages`]), followed by
/// **every** response message.
///
/// Splitting the two is load-bearing. Responses are generated by the run that
/// is reporting them, so they cannot be a replay of anything — but they can
/// coincidentally reproduce the stored tail. Aligning over the concatenation
/// would let that coincidence match, and the genuinely new input in front of it
/// would be dropped along with the stored block: stored `[q, a]` plus a new run
/// whose input is `q` and whose response opens with `a` would store neither.
pub fn new_run_messages(
    existing: &[Message],
    request_messages: &[Message],
    response_messages: &[Message],
) -> Vec<Message> {
    new_run_messages_from(
        existing,
        request_messages,
        response_messages,
        StoredHistory::Complete,
    )
}

/// [`new_run_messages`] for a provider whose stored history has a known
/// [`StoredHistory`] shape — a retention-limited store holds a
/// [`StoredHistory::Window`].
pub fn new_run_messages_from(
    existing: &[Message],
    request_messages: &[Message],
    response_messages: &[Message],
    shape: StoredHistory,
) -> Vec<Message> {
    filter_new_messages_from(existing, request_messages, shape)
        .iter()
        .chain(response_messages)
        .cloned()
        .collect()
}

/// Inject `stored` ahead of any context another provider has already added —
/// unless the run's own input already carries that stored run, which is
/// exactly what a caller replaying its own transcript sends.
///
/// Storing only the new suffix (see [`new_run_messages`]) stops history growing
/// on a replay, but the request is assembled the other way round: the agent
/// sends `ctx.messages` followed by `ctx.input_messages`. For a replaying
/// caller those hold the same turns, so injecting unconditionally sends the
/// model `q1, a1, q1, a1, q2` — every replayed turn twice, on every subsequent
/// run — even though this provider stored each of them only once. When the
/// input aligns against the stored run it is a superset of it, and injecting
/// nothing leaves the request complete.
pub fn inject_stored_history(ctx: &mut SessionContext, stored: Vec<Message>) {
    inject_stored_history_from(ctx, stored, StoredHistory::Complete)
}

/// [`inject_stored_history`] for a provider whose stored history has a known
/// [`StoredHistory`] shape — a retention-limited store holds a
/// [`StoredHistory::Window`].
pub fn inject_stored_history_from(
    ctx: &mut SessionContext,
    stored: Vec<Message>,
    shape: StoredHistory,
) {
    if stored.is_empty() {
        return;
    }
    // A shorter result means the input aligned against — and therefore already
    // contains — the stored run.
    if filter_new_messages_from(&stored, &ctx.input_messages, shape).len()
        < ctx.input_messages.len()
    {
        return;
    }
    let existing = std::mem::take(&mut ctx.messages);
    ctx.messages = stored.into_iter().chain(existing).collect();
}

/// Attach a fresh [`InMemoryHistoryProvider`] as the **first** context
/// provider on `session` when it is not service-managed and does not already
/// carry a history provider. A no-op for service-managed sessions (the
/// service owns history server-side) and for sessions that already have one
/// attached (detected via [`ContextProvider::is_history_provider`]).
pub fn ensure_history_provider(session: &mut AgentSession) {
    if session.service_session_id().is_none()
        && !session
            .context_providers
            .iter()
            .any(|p| p.is_history_provider())
    {
        session
            .context_providers
            .insert(0, Arc::new(InMemoryHistoryProvider::new()));
    }
}

/// In-memory [`HistoryProvider`]: keeps history in an `Arc<Mutex<Vec<Message>>>`,
/// shared across clones.
#[derive(Default, Clone)]
pub struct InMemoryHistoryProvider {
    messages: Arc<Mutex<Vec<Message>>>,
}

impl InMemoryHistoryProvider {
    /// An empty history provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// A history provider seeded with `messages`.
    pub fn with_messages(messages: Vec<Message>) -> Self {
        Self {
            messages: Arc::new(Mutex::new(messages)),
        }
    }

    /// The stored messages, in chronological order.
    pub fn list_messages(&self) -> Vec<Message> {
        self.messages.lock().unwrap().clone()
    }

    /// Serialize the stored history to `{"messages": [...]}`.
    pub fn to_dict(&self) -> Value {
        serde_json::json!({ "messages": self.list_messages() })
    }

    /// Reconstruct a provider from state produced by [`InMemoryHistoryProvider::to_dict`].
    pub fn from_dict(state: &Value) -> Result<Self> {
        let messages = match state.get("messages") {
            Some(v) if !v.is_null() => serde_json::from_value(v.clone()).map_err(|e| {
                Error::Serialization(format!("failed to restore history provider: {e}"))
            })?,
            _ => Vec::new(),
        };
        Ok(Self::with_messages(messages))
    }
}

#[async_trait]
impl ContextProvider for InMemoryHistoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let stored = self.messages.lock().unwrap().clone();
        inject_stored_history(ctx, stored);
        Ok(())
    }

    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_none() {
            let mut guard = self.messages.lock().unwrap();
            let new = new_run_messages(&guard, request_messages, response_messages);
            guard.extend(new);
        }
        Ok(())
    }

    fn is_history_provider(&self) -> bool {
        true
    }

    fn history_snapshot(&self) -> Option<Vec<Message>> {
        Some(self.list_messages())
    }
}

impl HistoryProvider for InMemoryHistoryProvider {}

/// A [`HistoryProvider`] that persists to a JSON file on disk, loading any
/// existing history from `path` on construction and rewriting the whole file
/// after every successful run.
///
/// Persistence is **atomic and concurrency-safe**: `after_run` serializes the
/// whole append→snapshot→write sequence behind an async `write_lock` (shared
/// across clones), writes to a temporary sibling file, and atomically renames
/// it into place. The in-memory history is only updated *after* the on-disk
/// write succeeds, so a failed write never diverges memory from disk, and two
/// concurrent runs sharing cloned providers can't lose each other's messages
/// via a snapshot/overwrite race.
#[derive(Clone)]
pub struct FileHistoryProvider {
    path: PathBuf,
    messages: Arc<Mutex<Vec<Message>>>,
    /// Serializes the append+snapshot+persist critical section across all
    /// clones so concurrent `after_run` calls can't interleave into a lost
    /// update. Held only in `after_run`; reads (`before_run`/`list_messages`)
    /// take the fast in-memory `messages` lock and never block on this.
    write_lock: Arc<tokio::sync::Mutex<()>>,
}

impl FileHistoryProvider {
    /// Open (or create) a file-backed history provider at `path`. A missing
    /// or empty file starts with no history; an existing file is parsed
    /// eagerly, so a malformed file fails the constructor rather than
    /// silently discarding history.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let messages = if path.exists() {
            let data = std::fs::read_to_string(&path)
                .map_err(|e| Error::other(format!("failed to read history file {path:?}: {e}")))?;
            if data.trim().is_empty() {
                Vec::new()
            } else {
                let value: Value = serde_json::from_str(&data).map_err(|e| {
                    Error::Serialization(format!("failed to parse history file {path:?}: {e}"))
                })?;
                match value.get("messages") {
                    Some(v) if !v.is_null() => serde_json::from_value(v.clone()).map_err(|e| {
                        Error::Serialization(format!("failed to parse history file {path:?}: {e}"))
                    })?,
                    _ => Vec::new(),
                }
            }
        } else {
            Vec::new()
        };
        Ok(Self {
            path,
            messages: Arc::new(Mutex::new(messages)),
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    /// The path this provider persists to.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// The stored messages, in chronological order.
    pub fn list_messages(&self) -> Vec<Message> {
        self.messages.lock().unwrap().clone()
    }

    /// Serialize the stored history to `{"messages": [...]}`.
    pub fn to_dict(&self) -> Value {
        serde_json::json!({ "messages": self.list_messages() })
    }

    /// Write `messages` as `{"messages": [...]}` via a temp-file-plus-rename so
    /// the destination is replaced **atomically**: serialize, write to a
    /// uniquely named temporary sibling file, then rename it over the
    /// destination. The rename is atomic on a POSIX filesystem, so a reader
    /// (or a crash) sees either the old file or the fully-written new one,
    /// never a truncated file.
    ///
    /// This guarantees atomic *replacement*, not fsync-level crash durability:
    /// like the sibling checkpoint writer, it does not `sync_all` the file or
    /// its directory, so a power loss immediately after the rename may still
    /// lose the last write. That is an intentional trade-off for these small,
    /// frequently-rewritten history files.
    async fn persist(&self, messages: &[Message]) -> Result<()> {
        let dict = serde_json::json!({ "messages": messages });
        let json = serde_json::to_string_pretty(&dict)
            .map_err(|e| Error::Serialization(format!("failed to serialize history: {e}")))?;
        // Temp file in the same directory so `rename` stays on one filesystem.
        // A uuid suffix keeps two providers on the same path from clobbering
        // each other's temp file.
        let file_name = self
            .path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("history.json");
        let tmp = self
            .path
            .with_file_name(format!("{file_name}.tmp.{}", uuid::Uuid::new_v4()));
        if let Err(e) = tokio::fs::write(&tmp, &json).await {
            // Don't leave the partial temp file behind on a failed write.
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Error::other(format!(
                "failed to write history temp file {tmp:?}: {e}"
            )));
        }
        tokio::fs::rename(&tmp, &self.path).await.map_err(|e| {
            // Best-effort cleanup of the temp file on a failed rename.
            let tmp = tmp.clone();
            tokio::spawn(async move {
                let _ = tokio::fs::remove_file(&tmp).await;
            });
            Error::other(format!(
                "failed to finalize history file {:?}: {e}",
                self.path
            ))
        })
    }
}

#[async_trait]
impl ContextProvider for FileHistoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let stored = self.messages.lock().unwrap().clone();
        inject_stored_history(ctx, stored);
        Ok(())
    }

    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_some() {
            return Ok(());
        }
        // Serialize the whole append→snapshot→persist sequence so two
        // concurrent runs (sharing cloned providers) can't interleave a
        // snapshot and an overwrite into a lost update.
        let _write = self.write_lock.lock().await;

        // Compute the next full history WITHOUT committing it to shared memory
        // yet: disk is the source of truth. We persist first and only update
        // the in-memory copy on success, so a failed write leaves memory and
        // disk consistent (the run's `after_run` returns the error and the
        // caller can retry) rather than diverging.
        let snapshot = {
            let guard = self.messages.lock().unwrap();
            let mut next = guard.clone();
            next.extend(new_run_messages(
                &guard,
                request_messages,
                response_messages,
            ));
            next
        };
        self.persist(&snapshot).await?;
        *self.messages.lock().unwrap() = snapshot;
        Ok(())
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl HistoryProvider for FileHistoryProvider {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[tokio::test]
    async fn before_run_prepends_stored_messages_ahead_of_existing_context_messages() {
        let provider = InMemoryHistoryProvider::with_messages(vec![
            Message::user("q1"),
            Message::assistant("a1"),
        ]);
        let mut ctx = SessionContext::new(vec![Message::user("q2")]);
        ctx.messages
            .push(Message::system("injected by another provider"));
        provider.before_run(&mut ctx).await.unwrap();
        let texts: Vec<String> = ctx.messages.iter().map(|m| m.text()).collect();
        assert_eq!(
            texts,
            vec![
                "q1".to_string(),
                "a1".to_string(),
                "injected by another provider".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn after_run_appends_only_on_success() {
        let provider = InMemoryHistoryProvider::new();
        provider
            .after_run(&[Message::user("hi")], &[Message::assistant("hello")], None)
            .await
            .unwrap();
        assert_eq!(provider.list_messages().len(), 2);

        // A failed run must not record anything.
        provider
            .after_run(
                &[Message::user("again")],
                &[],
                Some(&Error::service("boom")),
            )
            .await
            .unwrap();
        assert_eq!(provider.list_messages().len(), 2);
    }

    #[test]
    fn to_dict_from_dict_round_trips_messages() {
        let provider = InMemoryHistoryProvider::with_messages(vec![
            Message::user("q1"),
            Message::assistant("a1"),
        ]);
        let state = provider.to_dict();
        let restored = InMemoryHistoryProvider::from_dict(&state).unwrap();
        let msgs = restored.list_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text(), "q1");
        assert_eq!(msgs[1].text(), "a1");
    }

    #[test]
    fn from_dict_tolerates_a_missing_messages_key() {
        let restored = InMemoryHistoryProvider::from_dict(&serde_json::json!({})).unwrap();
        assert!(restored.list_messages().is_empty());
    }

    #[test]
    fn ensure_history_provider_attaches_once_and_skips_service_managed() {
        let mut local = AgentSession::new();
        ensure_history_provider(&mut local);
        assert_eq!(local.context_providers.len(), 1);
        assert!(local.context_providers[0].is_history_provider());
        // A second call must not attach a duplicate.
        ensure_history_provider(&mut local);
        assert_eq!(local.context_providers.len(), 1);

        let mut service = AgentSession::service("svc-1");
        ensure_history_provider(&mut service);
        assert!(service.context_providers.is_empty());
    }

    #[tokio::test]
    async fn file_history_provider_persists_and_reloads() {
        let dir = std::env::temp_dir().join(format!("afr-history-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");

        let provider = FileHistoryProvider::new(&path).unwrap();
        assert!(provider.list_messages().is_empty());
        provider
            .after_run(&[Message::user("hi")], &[Message::assistant("hello")], None)
            .await
            .unwrap();
        assert_eq!(provider.list_messages().len(), 2);

        // A fresh provider opened on the same path picks up the persisted
        // history.
        let reloaded = FileHistoryProvider::new(&path).unwrap();
        let msgs = reloaded.list_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text(), "hi");
        assert_eq!(msgs[1].text(), "hello");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn file_history_provider_concurrent_runs_do_not_lose_messages() {
        // Regression for the snapshot/overwrite race: many concurrent
        // `after_run` calls on cloned providers must all be durably recorded,
        // and the on-disk file must always be valid JSON (atomic rename).
        let dir = std::env::temp_dir().join(format!("afr-history-conc-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");

        let provider = FileHistoryProvider::new(&path).unwrap();
        const N: usize = 50;
        let mut handles = Vec::new();
        for i in 0..N {
            let p = provider.clone();
            handles.push(tokio::spawn(async move {
                p.after_run(
                    &[Message::user(format!("q{i}"))],
                    &[Message::assistant(format!("a{i}"))],
                    None,
                )
                .await
                .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        // Every run contributed a request + response message; none lost.
        assert_eq!(provider.list_messages().len(), N * 2);

        // The on-disk file is valid and holds the full history (atomic rename
        // means it is never a torn/partial write).
        let reloaded = FileHistoryProvider::new(&path).unwrap();
        assert_eq!(reloaded.list_messages().len(), N * 2);

        // No temp files left behind.
        let leftover: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(leftover.is_empty(), "temp files leaked: {leftover:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    fn texts(messages: &[Message]) -> Vec<String> {
        messages.iter().map(Message::text).collect()
    }

    #[test]
    fn filter_new_messages_returns_everything_when_nothing_is_stored() {
        let incoming = vec![Message::user("hi"), Message::assistant("hello")];
        assert_eq!(filter_new_messages(&[], &incoming).len(), 2);
    }

    #[test]
    fn filter_new_messages_drops_a_replayed_prefix() {
        let existing = vec![Message::user("hi"), Message::assistant("hello")];
        let incoming = vec![
            Message::user("hi"),
            Message::assistant("hello"),
            Message::user("more"),
            Message::assistant("sure"),
        ];
        assert_eq!(
            texts(filter_new_messages(&existing, &incoming)),
            vec!["more".to_string(), "sure".to_string()]
        );
    }

    /// A provider with a retention limit stores a *window* of the
    /// conversation, so the replayed transcript starts before what is stored
    /// and the window has to be searched for.
    #[test]
    fn filter_new_messages_aligns_a_trimmed_window() {
        let existing = vec![Message::user("q2"), Message::assistant("a2")];
        let incoming = vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q2"),
            Message::assistant("a2"),
            Message::user("q3"),
        ];
        assert_eq!(
            texts(filter_new_messages_from(
                &existing,
                &incoming,
                StoredHistory::Window
            )),
            vec!["q3".to_string()]
        );
        // A *complete* history can only be a replay when it comes first, so the
        // same input is entirely new to a store that keeps everything.
        assert_eq!(filter_new_messages(&existing, &incoming).len(), 5);
    }

    /// A retention-limited list whose length merely reaches the cap has not
    /// necessarily been trimmed — a first write that filled it exactly is still
    /// the complete conversation. The anchored match therefore wins even for a
    /// `Window`, or the turns between two occurrences would be lost
    /// (PR #16 review).
    #[test]
    fn a_window_prefers_an_anchored_match_over_a_later_one() {
        let existing = vec![Message::user("q"), Message::assistant("a")];
        let incoming = vec![
            Message::user("q"),
            Message::assistant("a"),
            Message::user("filler"),
            Message::user("q"),
            Message::assistant("a"),
            Message::user("new"),
        ];
        assert_eq!(
            texts(filter_new_messages_from(
                &existing,
                &incoming,
                StoredHistory::Window
            )),
            vec![
                "filler".to_string(),
                "q".to_string(),
                "a".to_string(),
                "new".to_string()
            ],
            "an at-cap list that opens the replay is a complete history, not a window"
        );

        // A genuine window — one that does *not* open the replay — is still
        // found by searching, and the last occurrence is the retained one.
        let earlier = vec![
            Message::user("q0"),
            Message::assistant("a0"),
            Message::user("q"),
            Message::assistant("a"),
            Message::user("filler"),
            Message::user("q"),
            Message::assistant("a"),
            Message::user("new"),
        ];
        assert_eq!(
            texts(filter_new_messages_from(
                &existing,
                &earlier,
                StoredHistory::Window
            )),
            vec!["new".to_string()]
        );
    }

    /// An empty string is not an identity — every message carrying one would
    /// otherwise compare equal to every other (PR #16 review).
    #[test]
    fn an_empty_message_id_is_not_an_identity() {
        let empty_id = |m: Message| Message {
            message_id: Some(String::new()),
            ..m
        };
        // Two unrelated messages, both with empty ids, must not align.
        let existing = vec![
            empty_id(Message::user("q")),
            empty_id(Message::assistant("a")),
        ];
        let incoming = vec![
            empty_id(Message::user("something else")),
            empty_id(Message::assistant("unrelated")),
            empty_id(Message::user("new")),
        ];
        assert_eq!(
            texts(filter_new_messages(&existing, &incoming)).len(),
            3,
            "empty ids must fall back to role and contents, not match everything"
        );

        // And an empty id is not evidence of a replay either: user-only stored
        // history carrying one is still left alone.
        let user_only = vec![empty_id(Message::user("yes"))];
        let repeat = vec![empty_id(Message::user("yes")), Message::user("question")];
        assert_eq!(texts(filter_new_messages(&user_only, &repeat)).len(), 2);
    }

    /// Stored history that is nothing but id-less user turns cannot be told
    /// apart from new input that repeats it, so it is never read as a replay
    /// (PR #16 review).
    #[test]
    fn id_less_user_only_history_is_never_read_as_a_replay() {
        let existing = vec![Message::user("yes")];
        let incoming = vec![Message::user("yes"), Message::user("question")];
        assert_eq!(
            texts(filter_new_messages(&existing, &incoming)),
            vec!["yes".to_string(), "question".to_string()],
            "saying 'yes' again is not a replay of having said it"
        );

        // The same shape *with* the assistant's reply in the stored history is
        // a transcript, and a caller sending it back is replaying.
        let with_reply = vec![Message::user("yes"), Message::assistant("go on")];
        let replayed = vec![
            Message::user("yes"),
            Message::assistant("go on"),
            Message::user("question"),
        ];
        assert_eq!(
            texts(filter_new_messages(&with_reply, &replayed)),
            vec!["question".to_string()]
        );

        // And a message id is evidence on its own, user-only or not.
        let with_id = vec![Message {
            message_id: Some("m1".to_string()),
            ..Message::user("yes")
        }];
        let replayed_by_id = vec![
            Message {
                message_id: Some("m1".to_string()),
                ..Message::user("yes")
            },
            Message::user("question"),
        ];
        assert_eq!(
            texts(filter_new_messages(&with_id, &replayed_by_id)),
            vec!["question".to_string()]
        );
    }

    /// A complete history matched at a later offset would drop every genuinely
    /// new message in front of the coincidence (PR #16 review).
    #[test]
    fn a_complete_history_never_matches_past_the_start() {
        let existing = vec![Message::user("yes")];
        let incoming = vec![
            Message::user("preface"),
            Message::user("yes"),
            Message::user("question"),
        ];
        assert_eq!(
            texts(filter_new_messages(&existing, &incoming)),
            vec![
                "preface".to_string(),
                "yes".to_string(),
                "question".to_string()
            ],
            "nothing may be dropped: the stored 'yes' is not where this input starts"
        );
    }

    /// A store that keeps everything takes the anchored match, so a repeat in
    /// the middle of a replay is a genuinely new turn it has never stored.
    #[test]
    fn a_complete_store_keeps_the_turns_between_two_occurrences() {
        let existing = vec![Message::user("q"), Message::assistant("a")];
        let incoming = vec![
            Message::user("q"),
            Message::assistant("a"),
            Message::user("filler"),
            Message::user("q"),
            Message::assistant("a"),
            Message::user("new"),
        ];
        assert_eq!(
            texts(filter_new_messages_from(
                &existing,
                &incoming,
                StoredHistory::Complete
            )),
            vec![
                "filler".to_string(),
                "q".to_string(),
                "a".to_string(),
                "new".to_string()
            ]
        );
    }
    /// The deliberate divergence from upstream: when the stored run cannot be
    /// aligned, everything is stored — never a set-based dedup that would drop
    /// a legitimately repeated turn.
    #[test]
    fn filter_new_messages_keeps_a_repeated_turn_it_cannot_align() {
        let existing = vec![Message::user("ping"), Message::assistant("pong")];
        let incoming = vec![Message::user("ping"), Message::assistant("pong!")];
        assert_eq!(
            texts(filter_new_messages(&existing, &incoming)),
            vec!["ping".to_string(), "pong!".to_string()]
        );
    }

    #[test]
    fn filter_new_messages_matches_on_message_id_when_present() {
        let with_id = |m: Message, id: &str| Message {
            message_id: Some(id.to_string()),
            ..m
        };
        let stored = with_id(Message::user("hi"), "m1");
        // Same id, different text: the id alone decides.
        let replayed = with_id(Message::user("edited after the fact"), "m1");
        let incoming = vec![replayed, Message::assistant("hello")];
        assert_eq!(
            texts(filter_new_messages(
                std::slice::from_ref(&stored),
                &incoming
            )),
            vec!["hello".to_string()]
        );

        // Different id, same text: a distinct message, so nothing aligns.
        let other = vec![with_id(Message::user("hi"), "m2")];
        assert_eq!(
            filter_new_messages(std::slice::from_ref(&stored), &other).len(),
            1
        );
    }

    /// The bug this guards: a caller that keeps its own transcript and replays
    /// all of it every turn used to have the whole conversation re-stored on
    /// top of the copy already there, growing history superlinearly and
    /// resending the duplicates to the model on the next run.
    #[tokio::test]
    async fn replaying_the_transcript_does_not_duplicate_stored_history() {
        let provider = InMemoryHistoryProvider::new();
        provider
            .after_run(&[Message::user("q1")], &[Message::assistant("a1")], None)
            .await
            .unwrap();

        // Turn two: the caller replays everything it has, plus the new turn.
        provider
            .after_run(
                &[
                    Message::user("q1"),
                    Message::assistant("a1"),
                    Message::user("q2"),
                ],
                &[Message::assistant("a2")],
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            texts(&provider.list_messages()),
            vec![
                "q1".to_string(),
                "a1".to_string(),
                "q2".to_string(),
                "a2".to_string()
            ]
        );
    }

    /// A response that happens to reproduce the stored tail must not be
    /// mistaken for a replay of it: alignment sees the run's input only, and
    /// response messages are always appended (PR #16 review).
    #[tokio::test]
    async fn a_response_repeating_stored_history_is_still_stored() {
        let provider = InMemoryHistoryProvider::with_messages(vec![
            Message::user("q"),
            Message::assistant("a"),
        ]);
        // A genuinely new turn whose input repeats "q" and whose (tool-loop)
        // response opens with "a" — concatenated, that is exactly the stored
        // run followed by one new message.
        provider
            .after_run(
                &[Message::user("q")],
                &[Message::assistant("a"), Message::assistant("b")],
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            texts(&provider.list_messages()),
            vec![
                "q".to_string(),
                "a".to_string(),
                "q".to_string(),
                "a".to_string(),
                "b".to_string()
            ],
            "the new turn must not be swallowed by its own response"
        );
    }

    /// `before_run` is the other half of the replay problem: prepending stored
    /// history to input that already contains it sends every replayed turn to
    /// the model twice (PR #16 review).
    #[tokio::test]
    async fn before_run_does_not_prepend_history_the_input_already_carries() {
        let provider = InMemoryHistoryProvider::with_messages(vec![
            Message::user("q1"),
            Message::assistant("a1"),
        ]);

        // The agent sends `ctx.messages` followed by `ctx.input_messages`, so
        // a replaying caller's input already carries the stored run and this
        // provider must contribute nothing.
        let mut replayed = SessionContext::new(vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q2"),
        ]);
        provider.before_run(&mut replayed).await.unwrap();
        assert!(
            replayed.messages.is_empty(),
            "stored history must not be injected on top of a replay of itself: {:?}",
            texts(&replayed.messages)
        );

        // A caller that tracks nothing itself still gets history injected.
        let mut incremental = SessionContext::new(vec![Message::user("q2")]);
        provider.before_run(&mut incremental).await.unwrap();
        assert_eq!(
            texts(&incremental.messages),
            vec!["q1".to_string(), "a1".to_string()]
        );
    }

    /// The append-only path — the shape the agent itself produces — is
    /// untouched, including a turn that repeats an earlier one verbatim (the
    /// case an alignment consuming all of `incoming` would otherwise swallow).
    #[tokio::test]
    async fn append_only_runs_still_accumulate_every_turn() {
        let provider = InMemoryHistoryProvider::new();
        for _ in 0..3 {
            provider
                .after_run(
                    &[Message::user("ping")],
                    &[Message::assistant("pong")],
                    None,
                )
                .await
                .unwrap();
        }
        assert_eq!(provider.list_messages().len(), 6);
    }

    #[tokio::test]
    async fn file_history_provider_does_not_duplicate_a_replayed_transcript() {
        let dir = std::env::temp_dir().join(format!("afr-history-dedup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");

        let provider = FileHistoryProvider::new(&path).unwrap();
        provider
            .after_run(&[Message::user("q1")], &[Message::assistant("a1")], None)
            .await
            .unwrap();
        provider
            .after_run(
                &[
                    Message::user("q1"),
                    Message::assistant("a1"),
                    Message::user("q2"),
                ],
                &[Message::assistant("a2")],
                None,
            )
            .await
            .unwrap();

        assert_eq!(texts(&provider.list_messages()).len(), 4);
        // Disk agrees with memory.
        let reloaded = FileHistoryProvider::new(&path).unwrap();
        assert_eq!(
            texts(&reloaded.list_messages()),
            vec![
                "q1".to_string(),
                "a1".to_string(),
                "q2".to_string(),
                "a2".to_string()
            ]
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
