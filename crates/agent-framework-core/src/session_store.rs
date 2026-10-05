//! Keyed storage for [`AgentSession`] snapshots.
//!
//! Ports upstream's experimental `SessionStore` / `FileSessionStore`
//! (`agent_framework/_sessions.py`). A store maps an opaque, caller-chosen id
//! to a session snapshot, and every read hands back an **independent working
//! copy**: a run that continues from a stored snapshot cannot mutate that
//! snapshot, so two requests may branch from the same point and store their
//! results under different ids. That is the property hosting protocols such
//! as the OpenAI Responses API's `previous_response_id` depend on.
//!
//! Stores never create sessions for ids they have not seen. That belongs to
//! whoever holds both the store and the agent — see
//! `agent_framework_hosting::AgentState::get_or_create_session`.
//!
//! # What a snapshot holds
//!
//! A snapshot is a session's *data*: its `session_id`, `service_session_id`,
//! [`SessionState`](crate::session::SessionState) bag, and — the one thing
//! upstream gets for free from keeping history in `session.state` — the
//! conversation held by any in-process history provider (see
//! [`ContextProvider::history_snapshot`](crate::memory::ContextProvider::history_snapshot)).
//! A restored session carries that history in a fresh
//! [`InMemoryHistoryProvider`] at the front of its `context_providers`.
//!
//! Other context providers attached to the session object are **not**
//! persisted: they are live objects, not data. Attach long-lived providers to
//! the agent (where upstream keeps them) and they apply to every restored
//! session automatically. A provider backed by external storage (a file,
//! Redis, a database) keeps its own history and is unaffected.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::history::InMemoryHistoryProvider;
use crate::session::AgentSession;
use crate::types::Message;

/// The snapshot format version written by [`FileSessionStore`]. A file with
/// any other version is refused rather than guessed at.
pub const SESSION_SNAPSHOT_VERSION: u32 = 1;

/// Storage for [`AgentSession`] snapshots keyed by an opaque id.
///
/// Implementations must return an independent copy from [`get`](Self::get):
/// mutating a returned session (or running an agent against it) must not
/// change what the store holds until the caller [`set`](Self::set)s it back.
/// Ids are opaque non-empty strings; an implementation applies whatever key
/// restrictions its backend needs, and must use its backend's normal
/// key-handling protections (parameterized queries and the like).
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// An independent copy of the session stored under `session_id`, or
    /// `None` when there is none.
    async fn get(&self, session_id: &str) -> Result<Option<AgentSession>>;

    /// Store `session` under `session_id`, replacing any existing entry.
    async fn set(&self, session_id: &str, session: &AgentSession) -> Result<()>;

    /// Delete the entry for `session_id`, if any.
    async fn delete(&self, session_id: &str) -> Result<()>;
}

/// Reject an empty id — the one rule every store shares.
pub fn validate_session_id(session_id: &str) -> Result<()> {
    if session_id.is_empty() {
        return Err(Error::Configuration(
            "session_id must be a non-empty string".into(),
        ));
    }
    Ok(())
}

/// A session's persistable data. See the [module docs](self).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSnapshot {
    /// [`AgentSession::to_dict`]'s `{session_id, service_session_id, state}`.
    pub session: Value,
    /// The in-process conversation history, when the session carried a
    /// provider that holds one.
    pub history: Option<Vec<Message>>,
}

impl SessionSnapshot {
    /// Capture `session`'s data. The result shares nothing with `session`.
    pub fn capture(session: &AgentSession) -> Self {
        let history = session
            .context_providers
            .iter()
            .find_map(|p| p.history_snapshot());
        Self {
            session: session.to_dict(),
            history,
        }
    }

    /// A new session holding this snapshot's data.
    pub fn restore(&self) -> Result<AgentSession> {
        let mut session = AgentSession::from_dict(&self.session)?;
        if let Some(history) = &self.history {
            session.context_providers.insert(
                0,
                Arc::new(InMemoryHistoryProvider::with_messages(history.clone())),
            );
        }
        Ok(session)
    }

    /// The versioned on-disk form.
    pub fn to_value(&self) -> Value {
        serde_json::json!({
            "type": "session",
            "version": SESSION_SNAPSHOT_VERSION,
            "session_id": self.session.get("session_id").cloned().unwrap_or(Value::Null),
            "service_session_id": self.session.get("service_session_id").cloned().unwrap_or(Value::Null),
            "state": self.session.get("state").cloned().unwrap_or_else(|| serde_json::json!({})),
            "history": self.history,
        })
    }

    /// Parse the form written by [`to_value`](Self::to_value), refusing an
    /// unknown `type` or `version` and a non-mapping `state`.
    pub fn from_value(value: &Value) -> Result<Self> {
        let invalid =
            |what: &str| Error::Serialization(format!("invalid session snapshot: {what}"));
        let map = value.as_object().ok_or_else(|| invalid("not an object"))?;
        if map.get("type").and_then(Value::as_str) != Some("session") {
            return Err(invalid("`type` must be \"session\""));
        }
        match map.get("version").and_then(Value::as_u64) {
            Some(v) if v == u64::from(SESSION_SNAPSHOT_VERSION) => {}
            other => {
                return Err(Error::Serialization(format!(
                    "unsupported session snapshot version {other:?}; expected {SESSION_SNAPSHOT_VERSION}"
                )))
            }
        }
        let session_id = map
            .get("session_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| invalid("`session_id` must be a non-empty string"))?;
        let service_session_id = match map.get("service_session_id") {
            None | Some(Value::Null) => Value::Null,
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(_) => return Err(invalid("`service_session_id` must be a string or null")),
        };
        let state = match map.get("state") {
            Some(state @ Value::Object(_)) => state.clone(),
            _ => return Err(invalid("`state` must be a mapping")),
        };
        let history = match map.get("history") {
            None | Some(Value::Null) => None,
            Some(h) => Some(
                serde_json::from_value(h.clone())
                    .map_err(|e| invalid(&format!("`history`: {e}")))?,
            ),
        };
        Ok(Self {
            session: serde_json::json!({
                "session_id": session_id,
                "service_session_id": service_session_id,
                "state": state,
            }),
            history,
        })
    }
}

// ---------------------------------------------------------------------------
// In-memory
// ---------------------------------------------------------------------------

/// In-memory [`SessionStore`]. Mirrors upstream's base `SessionStore`.
///
/// Holds snapshots, not live sessions, so every [`get`](SessionStore::get)
/// is an independent copy. There is **no eviction**: every id ever stored
/// stays resolvable for the life of the process. That is deliberate —
/// `previous_response_id` lets a caller continue from *any* earlier point in
/// a conversation, so every id handed out must stay resolvable. A store
/// backed by real storage owns its own TTL policy.
#[derive(Default, Clone)]
pub struct InMemorySessionStore {
    sessions: Arc<Mutex<HashMap<String, SessionSnapshot>>>,
}

impl InMemorySessionStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of stored sessions.
    pub fn len(&self) -> usize {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for InMemorySessionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemorySessionStore")
            .field("len", &self.len())
            .finish()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn get(&self, session_id: &str) -> Result<Option<AgentSession>> {
        validate_session_id(session_id)?;
        let snapshot = self
            .sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(session_id)
            .cloned();
        snapshot.map(|s| s.restore()).transpose()
    }

    async fn set(&self, session_id: &str, session: &AgentSession) -> Result<()> {
        validate_session_id(session_id)?;
        let snapshot = SessionSnapshot::capture(session);
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(session_id.to_string(), snapshot);
        Ok(())
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(session_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// File-backed
// ---------------------------------------------------------------------------

/// File-backed [`SessionStore`]: one JSON file per session beneath a
/// directory. Mirrors upstream's `FileSessionStore` (JSON format; upstream's
/// optional MessagePack encoding is not ported).
///
/// Writes go to a unique sibling temporary file that is then renamed into
/// place, so a reader never observes a partial snapshot. Operations on one
/// file are serialized within the process; writers in different processes
/// get last-writer-wins.
///
/// A file that does not parse is **quarantined** — renamed aside to
/// `.<name>.<uuid>.corrupt` — and the read fails, so a retry starts a new
/// session instead of failing forever. A file that parses but has the wrong
/// shape or version is refused without being moved, since it may belong to
/// a newer writer.
///
/// # Security
///
/// Snapshots are plaintext. Treat the directory as trusted application
/// storage, not a secret store. Ids are mapped to filenames through an
/// injective encoding (see [`crate::storage_keys`]), never used as paths
/// directly, so an id cannot escape the directory.
#[derive(Debug, Clone)]
pub struct FileSessionStore {
    root: PathBuf,
}

impl FileSessionStore {
    /// The longest id the file store accepts, in characters.
    pub const MAX_SESSION_ID_LENGTH: usize = 128;
    const ENCODED_PREFIX: &'static str = "~session-";
    const HASHED_PREFIX: &'static str = "~session-sha256-";
    const EXTENSION: &'static str = ".json";
    /// Past this, an encoded stem plus extension risks the common 255-byte
    /// filename limit, so the stem becomes a digest instead.
    const MAX_STEM_BYTES: usize = 200;

    /// Open a store rooted at `path`, creating the directory if needed.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let root = path.into();
        std::fs::create_dir_all(&root).map_err(|e| {
            Error::other(format!("failed to create session directory {root:?}: {e}"))
        })?;
        Ok(Self { root })
    }

    /// The directory this store writes to.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Validate an id for this store: non-empty and at most
    /// [`MAX_SESSION_ID_LENGTH`](Self::MAX_SESSION_ID_LENGTH) characters.
    pub fn validate_session_id(session_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        if session_id.chars().count() > Self::MAX_SESSION_ID_LENGTH {
            return Err(Error::Configuration(format!(
                "session_id must be at most {} characters",
                Self::MAX_SESSION_ID_LENGTH
            )));
        }
        Ok(())
    }

    /// The filename stem for `session_id`.
    ///
    /// A literal-safe id is used verbatim unless it starts with `.` (which
    /// would let `..` name the parent directory or hide the file); anything
    /// else is hex-encoded under a `~` prefix the literal alphabet excludes,
    /// so the two namespaces never meet. An encoding too long for a filename
    /// is replaced by its SHA-256 digest under a distinct prefix.
    fn file_stem(session_id: &str) -> String {
        let literal = crate::storage_keys::storage_key_segment(session_id, Self::ENCODED_PREFIX);
        let stem = if literal.starts_with('.') {
            let mut out = String::from(Self::ENCODED_PREFIX);
            for byte in session_id.as_bytes() {
                out.push_str(&format!("{byte:02x}"));
            }
            out
        } else {
            literal
        };
        if stem.len() <= Self::MAX_STEM_BYTES {
            return stem;
        }
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(session_id.as_bytes());
        let mut out = String::from(Self::HASHED_PREFIX);
        for byte in digest {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    fn file_path(&self, session_id: &str) -> Result<PathBuf> {
        Self::validate_session_id(session_id)?;
        let name = format!("{}{}", Self::file_stem(session_id), Self::EXTENSION);
        // Belt and braces: the encoding admits no separators, but a path
        // that is not a plain child of the root is refused regardless.
        let candidate = Path::new(&name);
        if candidate.components().count() != 1 || candidate.file_name().is_none() {
            return Err(Error::Configuration(format!(
                "session path escaped storage directory: {session_id:?}"
            )));
        }
        Ok(self.root.join(name))
    }

    /// The process-local lock serializing operations on `path`.
    fn file_lock(path: &Path) -> &'static Mutex<()> {
        const STRIPES: usize = 64;
        static LOCKS: OnceLock<Vec<Mutex<()>>> = OnceLock::new();
        let locks = LOCKS.get_or_init(|| (0..STRIPES).map(|_| Mutex::new(())).collect());
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        path.hash(&mut hasher);
        &locks[(hasher.finish() as usize) % STRIPES]
    }

    fn sibling(path: &Path, suffix: &str) -> PathBuf {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        path.with_file_name(format!(
            ".{name}.{}.{suffix}",
            uuid::Uuid::new_v4().simple()
        ))
    }

    fn read_blocking(path: &Path) -> Result<Option<SessionSnapshot>> {
        let _guard = Self::file_lock(path)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(Error::other(format!(
                    "failed to read session snapshot {path:?}: {e}"
                )))
            }
        };
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(parse_error) => {
                let quarantine = Self::sibling(path, "corrupt");
                return Err(match std::fs::rename(path, &quarantine) {
                    Ok(()) => Error::Serialization(format!(
                        "failed to deserialize session from {path:?} ({parse_error}); the corrupt \
                         snapshot was quarantined to {quarantine:?}; retry to create a new session"
                    )),
                    Err(e) => Error::Serialization(format!(
                        "failed to deserialize session from {path:?} ({parse_error}), and the \
                         corrupt snapshot could not be quarantined: {e}"
                    )),
                });
            }
        };
        SessionSnapshot::from_value(&value)
            .map(Some)
            .map_err(|e| Error::Serialization(format!("session snapshot {path:?}: {e}")))
    }

    fn write_blocking(path: &Path, bytes: &[u8]) -> Result<()> {
        let _guard = Self::file_lock(path)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let temp = Self::sibling(path, "tmp");
        let result = std::fs::write(&temp, bytes).and_then(|()| std::fs::rename(&temp, path));
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result.map_err(|e| Error::other(format!("failed to write session snapshot {path:?}: {e}")))
    }

    fn delete_blocking(path: &Path) -> Result<()> {
        let _guard = Self::file_lock(path)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::other(format!(
                "failed to delete session snapshot {path:?}: {e}"
            ))),
        }
    }
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T> + Send + 'static) -> Result<T> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| Error::other(format!("session store task failed: {e}")))?
}

#[async_trait]
impl SessionStore for FileSessionStore {
    async fn get(&self, session_id: &str) -> Result<Option<AgentSession>> {
        let path = self.file_path(session_id)?;
        let snapshot = blocking(move || Self::read_blocking(&path)).await?;
        snapshot.map(|s| s.restore()).transpose()
    }

    async fn set(&self, session_id: &str, session: &AgentSession) -> Result<()> {
        let path = self.file_path(session_id)?;
        let bytes = serde_json::to_vec(&SessionSnapshot::capture(session).to_value())?;
        blocking(move || Self::write_blocking(&path, &bytes)).await
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        let path = self.file_path(session_id)?;
        blocking(move || Self::delete_blocking(&path)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_with_history(messages: Vec<Message>) -> AgentSession {
        let mut session = AgentSession::new();
        session
            .context_providers
            .push(Arc::new(InMemoryHistoryProvider::with_messages(messages)));
        session.state.insert("counter", serde_json::json!(1));
        session
    }

    fn history_of(session: &AgentSession) -> Vec<Message> {
        session
            .context_providers
            .iter()
            .find_map(|p| p.history_snapshot())
            .unwrap_or_default()
    }

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!("af-session-store-{}", uuid::Uuid::new_v4()))
    }

    #[tokio::test]
    async fn in_memory_round_trips_state_and_history() {
        let store = InMemorySessionStore::new();
        let session = session_with_history(vec![Message::user("hi"), Message::assistant("hello")]);
        store.set("s1", &session).await.unwrap();

        let restored = store.get("s1").await.unwrap().unwrap();
        assert_eq!(restored.session_id(), session.session_id());
        assert_eq!(restored.state.get("counter"), Some(serde_json::json!(1)));
        assert_eq!(history_of(&restored).len(), 2);
        assert!(store.get("missing").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_read_is_an_independent_copy() {
        // The property `previous_response_id` branching rests on: running
        // against a restored session must not move the stored snapshot.
        let store = InMemorySessionStore::new();
        store
            .set("s1", &session_with_history(vec![Message::user("hi")]))
            .await
            .unwrap();

        let branch = store.get("s1").await.unwrap().unwrap();
        branch.state.insert("counter", serde_json::json!(99));
        let provider = branch.context_providers[0].clone();
        provider
            .after_run(&[Message::user("more")], &[Message::assistant("ok")], None)
            .await
            .unwrap();
        assert_eq!(history_of(&branch).len(), 3);

        let again = store.get("s1").await.unwrap().unwrap();
        assert_eq!(again.state.get("counter"), Some(serde_json::json!(1)));
        assert_eq!(history_of(&again).len(), 1);
    }

    #[tokio::test]
    async fn a_session_without_in_process_history_restores_without_one() {
        let store = InMemorySessionStore::new();
        store
            .set("s", &AgentSession::service("conv_1"))
            .await
            .unwrap();
        let restored = store.get("s").await.unwrap().unwrap();
        assert_eq!(restored.service_session_id(), Some("conv_1"));
        assert!(restored.context_providers.is_empty());
    }

    #[tokio::test]
    async fn delete_and_empty_ids() {
        let store = InMemorySessionStore::new();
        store.set("s", &AgentSession::new()).await.unwrap();
        store.delete("s").await.unwrap();
        store.delete("s").await.unwrap();
        assert!(store.get("s").await.unwrap().is_none());
        assert!(store.get("").await.is_err());
        assert!(store.set("", &AgentSession::new()).await.is_err());
    }

    #[tokio::test]
    async fn file_store_round_trips_across_instances() {
        let dir = temp_dir();
        let session = session_with_history(vec![Message::user("hi")]);
        FileSessionStore::new(&dir)
            .unwrap()
            .set("resp_abc", &session)
            .await
            .unwrap();

        // A second instance over the same directory: a cold start.
        let restored = FileSessionStore::new(&dir)
            .unwrap()
            .get("resp_abc")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(restored.session_id(), session.session_id());
        assert_eq!(restored.state.get("counter"), Some(serde_json::json!(1)));
        assert_eq!(history_of(&restored)[0].text(), "hi");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn hostile_ids_stay_inside_the_directory_and_do_not_collide() {
        let dir = temp_dir();
        let store = FileSessionStore::new(&dir).unwrap();
        let long = "é".repeat(FileSessionStore::MAX_SESSION_ID_LENGTH);
        let ids = [
            "..",
            ".",
            "../escape",
            "a/b",
            "A",
            "~session-41",
            long.as_str(),
        ];
        for (i, id) in ids.iter().enumerate() {
            let s = AgentSession::new().with_session_id(format!("n{i}"));
            store.set(id, &s).await.unwrap();
        }
        for (i, id) in ids.iter().enumerate() {
            let got = store.get(id).await.unwrap().unwrap();
            assert_eq!(got.session_id(), format!("n{i}"), "id {id:?}");
        }
        // Everything landed directly in the directory.
        let count = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(count, ids.len());
        assert!(!dir.parent().unwrap().join("escape.json").exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn file_store_rejects_overlong_ids() {
        let dir = temp_dir();
        let store = FileSessionStore::new(&dir).unwrap();
        let long = "x".repeat(FileSessionStore::MAX_SESSION_ID_LENGTH + 1);
        assert!(store.set(&long, &AgentSession::new()).await.is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_corrupt_snapshot_is_quarantined_and_a_retry_starts_fresh() {
        let dir = temp_dir();
        let store = FileSessionStore::new(&dir).unwrap();
        std::fs::write(dir.join("s1.json"), b"{not json").unwrap();

        let err = store.get("s1").await.unwrap_err().to_string();
        assert!(err.contains("quarantined"), "{err}");
        assert!(store.get("s1").await.unwrap().is_none());
        let corrupt = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".corrupt"))
            .count();
        assert_eq!(corrupt, 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_wrong_version_is_refused_and_left_in_place() {
        let dir = temp_dir();
        let store = FileSessionStore::new(&dir).unwrap();
        let path = dir.join("s1.json");
        std::fs::write(
            &path,
            br#"{"type":"session","version":2,"session_id":"x","state":{}}"#,
        )
        .unwrap();
        let err = store.get("s1").await.unwrap_err().to_string();
        assert!(err.contains("version"), "{err}");
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
