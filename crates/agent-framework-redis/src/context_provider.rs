//! A [`ContextProvider`] that stores and retrieves conversation memories in
//! Redis, scoped by application/agent/user/thread id.
//!
//! Mirrors the *scoping and prompt-injection* behavior of the Python
//! `agent_framework_redis.RedisProvider`: `after_run()` persists the
//! request/response exchange as JSON entries tagged with the provider's
//! scope, and `before_run()` retrieves matching entries and injects them into
//! the conversation as a single `user`-role [`Message`] prefixed by
//! [`DEFAULT_CONTEXT_PROMPT`] (same default text as Python's
//! `ContextProvider.DEFAULT_CONTEXT_PROMPT`).
//!
//! # Divergence from Python: BM25 full-text via RediSearch, but no vector search
//!
//! Python's `RedisProvider` is built on
//! [redisvl](https://github.com/redis/redis-vl-python) and RediSearch: it
//! maintains a `FT.CREATE`d index with `TAG`/`TEXT`/`VECTOR` fields, and
//! `before_run()` runs a server-side `TextQuery` (BM25 full-text) or
//! `HybridQuery` (BM25 + KNN vector similarity, when a vectorizer is
//! configured) against that index.
//!
//! RediSearch is a separate Redis module — bundled with **Redis Stack** but
//! not with plain open-source `redis-server` — so [`RedisContextProvider`]
//! detects it at runtime (once per provider instance, cached — see
//! `FT._LIST`/`with_force_scan_fallback`) and switches behavior accordingly:
//!
//! - **RediSearch available** (Redis Stack): each memory is written with
//!   `JSON.SET {key_prefix}:entry:{uuid} $ <entry>` instead of a plain
//!   `SET`, so it's visible to an `FT.CREATE ... ON JSON PREFIX 1
//!   {key_prefix}:entry: SCHEMA ...` index (created lazily, once, and
//!   tolerant of a concurrent creator via Redis's own "Index already
//!   exists" error) covering every `MemoryEntry` field: `content` is
//!   `TEXT` (BM25-scored full-text); `application_id`/`agent_id`/`user_id`/
//!   `thread_id`/`role`/`message_id`/`author_name` are `TAG` (exact-match
//!   scope filtering); `rank` is `NUMERIC SORTABLE`. `before_run()` runs a
//!   single `FT.SEARCH` combining ANDed `TAG` filters for the configured
//!   scope with an ORed `@content:(...)` clause over the same lowercased,
//!   stopword-filtered tokens the SCAN path uses (`tokenize_query`),
//!   `LIMIT`ed to the configured [`RedisContextProvider::with_limit`], and
//!   ranked by RediSearch's own default scorer (a BM25 variant — no
//!   explicit `SCORER` argument is sent, since the exact literal scorer
//!   name has changed across RediSearch versions — `BM25` was renamed
//!   `BM25STD` in Redis Open Source 8.4 — while the *default* scorer has
//!   been BM25-family since RediSearch 1.x, so omitting `SCORER` gets BM25
//!   scoring portably). All caller-supplied scope values and query tokens
//!   are backslash-escaped against RediSearch's reserved query-syntax
//!   characters before being interpolated into the query string
//!   (`escape_redisearch`), so they can never be misinterpreted as query
//!   operators.
//! - **RediSearch unavailable** (plain `redis-server`), or when
//!   [`RedisContextProvider::with_force_scan_fallback`] is set: the
//!   original SCAN-based behavior, documented in detail below, is used
//!   unchanged.
//!
//! Vector/hybrid (KNN) search is **still not ported** — that remains this
//! crate's one substantive gap versus Python, since it requires an
//! embedding model/vectorizer this crate has no opinion on. Bring your own
//! vector store (or embed client-side and extend this provider) if you need
//! semantic recall; BM25 full-text (when Redis Stack is available) or
//! token-match (otherwise) is what you get here.
//!
//! Because RediSearch requires documents to be stored as native JSON
//! (`JSON.SET`) rather than opaque strings (`SET`) to be indexable, the two
//! storage encodings are not cross-readable: entries written while
//! RediSearch was available are invisible to the plain `SCAN`+`MGET`
//! fallback (which only sees string-typed keys), and vice versa. This only
//! matters if a deployment's effective RediSearch usage *changes* after
//! memories already exist (e.g. migrating from OSS Redis to Redis Stack, or
//! flipping [`RedisContextProvider::with_force_scan_fallback`] mid-flight)
//! — capability is detected once per provider instance and does not
//! retroactively reconcile entries written under the other encoding.
//!
//! ## The SCAN fallback (RediSearch unavailable)
//!
//! 1. Stores each memory as its own key `{key_prefix}:entry:{uuid}` holding
//!    a JSON blob (content, role, scope fields, a client-assigned recency
//!    rank) — no index, no schema, no `FT.CREATE`/`overwrite_index`.
//! 2. On `before_run()`, enumerates candidates with `SCAN MATCH
//!    {key_prefix}:entry:*` (cursor-based, non-blocking — never `KEYS`),
//!    `MGET`s their values, and **filters entirely client-side**:
//!    - scope filter: exact-match AND over whichever of
//!      application/agent/user/thread id are configured (same semantics as
//!      Python's `Tag(k) == v` conjunction), then
//!    - *optional* naive full-text filter: if the query text is non-empty,
//!      split it into lowercased alphanumeric tokens, drop a small built-in
//!      stopword list (`the`, `is`, `of`, ...), and keep only entries whose
//!      content *contains* (substring, not word-boundary) at least one
//!      remaining token — a keyword check, not BM25 ranking or embeddings,
//!      then
//!    - sort by recency (most recent first) and take the configured
//!      `limit` (default 10, matching Python's `num_results` default).
//!
//! There is no relevance ranking beyond "did a token match" plus recency,
//! no stemming, and — because every fallback `before_run()` call does a full
//! `SCAN` over the prefix — retrieval is `O(entries under key_prefix)`
//! rather than `O(log n)`/index-accelerated. This is adequate for modest
//! memory volumes (demos, tests, small deployments); reach for Redis Stack
//! (giving you the `FT.SEARCH` path above) for anything larger.
//!
//! `application_id` participates in the scope filter here — unlike in the
//! sibling `agent-framework-mem0` crate's `Mem0Provider`, where it is
//! write-only metadata — matching the Python `RedisProvider`, which
//! includes `application_id` in its combined `Tag` filter for both reads
//! and writes.

use async_trait::async_trait;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, OnceCell};
use uuid::Uuid;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::storage_keys::storage_key_segment;
use agent_framework_core::types::{Message, Role};

use crate::internal::{map_redis_err, LazyConnection};

/// Default Redis key prefix, matching the Python provider's default `prefix`.
pub const DEFAULT_KEY_PREFIX: &str = "context";

/// Default context-injection header, byte-for-byte identical to Python's
/// `agent_framework.ContextProvider.DEFAULT_CONTEXT_PROMPT`.
pub const DEFAULT_CONTEXT_PROMPT: &str =
    "## Memories\nConsider the following memories when answering user questions:";

/// Default number of memories returned by `before_run()`, matching Python's
/// `RedisProvider._redis_search(..., num_results=10)` default.
pub const DEFAULT_LIMIT: usize = 10;

/// The wire format for one stored memory: written to (and read back from)
/// either a plain string key (`SET`, SCAN-fallback path) or a RedisJSON
/// document (`JSON.SET`, RediSearch path) at `{key_prefix}:entry:{uuid}`.
/// Private to this module — the storage simplification documented above is
/// entirely our own, so there is no Python shape to mirror here (contrast
/// with [`crate::RedisChatMessageStore`], whose per-message JSON *is* meant
/// to be wire-compatible-in-spirit with the Python store). Every field here
/// has a corresponding `SCHEMA` entry in [`ft_create_args`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MemoryEntry {
    content: String,
    role: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    application_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    thread_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    author_name: Option<String>,
    /// Client-assigned recency rank: `unix_millis * 1000 + index_in_batch`.
    /// Higher is more recent. Avoids a round trip through a Redis-side
    /// counter while still giving messages added in the same `after_run()`
    /// call (which may share a millisecond) a stable relative order. Used
    /// to sort SCAN-fallback hits; carried into the RediSearch schema too
    /// (`NUMERIC SORTABLE`) for potential future use, though `before_run()`'s
    /// RediSearch path currently sorts by relevance (BM25), not rank — see
    /// the module docs.
    rank: i64,
}

/// The scope a provider instance is configured with. `None` fields are
/// wildcards (unconstrained); `Some` fields must match exactly. Mirrors the
/// ANDed `Tag(k) == v for k, v in filters.items() if v` behavior of
/// Python's `RedisProvider._build_filter_from_dict`.
#[derive(Debug, Clone, Default, PartialEq)]
struct Scope {
    application_id: Option<String>,
    agent_id: Option<String>,
    user_id: Option<String>,
    thread_id: Option<String>,
}

impl Scope {
    fn is_empty(&self) -> bool {
        self.application_id.is_none()
            && self.agent_id.is_none()
            && self.user_id.is_none()
            && self.thread_id.is_none()
    }

    fn matches(&self, entry: &MemoryEntry) -> bool {
        fn field_matches(configured: &Option<String>, actual: &Option<String>) -> bool {
            match configured {
                None => true,
                Some(want) => actual.as_deref() == Some(want.as_str()),
            }
        }
        field_matches(&self.application_id, &entry.application_id)
            && field_matches(&self.agent_id, &entry.agent_id)
            && field_matches(&self.user_id, &entry.user_id)
            && field_matches(&self.thread_id, &entry.thread_id)
    }
}

/// Extremely common English function words, excluded from the query token
/// set before matching. Without this, a query like "What is the capital of
/// France?" would token-match almost any stored memory purely because both
/// share the word "the" — full-blown stopword lists / stemming are exactly
/// the machinery a real search engine (or RediSearch) provides and this
/// crate does not; this is the minimum needed to keep "simple" from also
/// meaning "matches everything". Also used, unmodified, to build the
/// RediSearch full-text query clause (`tokenize_query`) so both retrieval
/// paths agree on what counts as a "meaningful" token.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "not", "with", "by", "from", "as", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "what", "which", "who",
    "whom", "do", "does", "did", "have", "has", "had", "my", "your", "me", "about",
];

/// Lowercase, alphanumeric-only tokenization with the [`STOPWORDS`] filter
/// applied. Shared by [`select_recent`]'s client-side "does any token
/// appear" SCAN-fallback filter and [`build_ft_search_query`]'s
/// `@content:(tok1|tok2|...)` clause for the RediSearch path, so the two
/// retrieval paths agree on what a "matching token" means even though one
/// does a substring check and the other lets RediSearch's own indexer/BM25
/// scorer do the matching.
fn tokenize_query(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty() && !STOPWORDS.contains(t))
        .map(str::to_string)
        .collect()
}

/// Pure selection logic — no I/O, fully unit-testable: scope-filter, then
/// (if `query` is non-empty) keep only entries with at least one matching
/// lowercase, non-stopword token, then sort by recency and take the top
/// `limit`. Used only by the SCAN fallback; the RediSearch path
/// (`RedisContextProvider::ft_search`) does the equivalent filtering
/// server-side.
fn select_recent(
    mut entries: Vec<MemoryEntry>,
    scope: &Scope,
    query: Option<&str>,
    limit: usize,
) -> Vec<MemoryEntry> {
    entries.retain(|e| scope.matches(e));

    if let Some(q) = query {
        let tokens = tokenize_query(q);
        if !tokens.is_empty() {
            entries.retain(|e| {
                let content_lower = e.content.to_lowercase();
                tokens.iter().any(|t| content_lower.contains(t.as_str()))
            });
        }
    }

    entries.sort_by_key(|e| std::cmp::Reverse(e.rank));
    entries.truncate(limit);
    entries
}

fn is_storable_role(role: &Role) -> bool {
    let r = role.as_str();
    r == Role::USER || r == Role::ASSISTANT || r == Role::SYSTEM
}

/// Build the single `user`-role memory-injection message for `before_run()`
/// to push onto `SessionContext::messages`, or `None` when there is nothing
/// to inject (mirrors the old `Context::default()` "no-op" case).
fn format_context_message(context_prompt: &str, joined_memories: &str) -> Option<Message> {
    if joined_memories.is_empty() {
        None
    } else {
        Some(Message::user(format!(
            "{context_prompt}\n{joined_memories}"
        )))
    }
}

fn now_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

// region: RediSearch (FT.*) support — pure, hermetically-testable pieces

/// Characters RediSearch's query parser treats as syntactically special
/// (per RediSearch's documented query-escaping rules: commas, punctuation,
/// brackets, quotes, and whitespace break tokenization; `|` is the OR
/// operator; `\` is the escape character itself). Backslash-escaping every
/// occurrence in caller-supplied text (TAG values, query tokens) before
/// interpolating it into a query string guarantees it is matched literally
/// rather than parsed as query syntax — the "escape query text safely"
/// requirement for the RediSearch upgrade.
const REDISEARCH_SPECIAL_CHARS: &[char] = &[
    ',', '.', '<', '>', '{', '}', '[', ']', '"', '\'', ':', ';', '!', '@', '#', '$', '%', '^', '&',
    '*', '(', ')', '-', '+', '=', '~', '|', ' ', '\\',
];

/// Backslash-escape every RediSearch-reserved character in `value` (see
/// [`REDISEARCH_SPECIAL_CHARS`]).
fn escape_redisearch(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        if REDISEARCH_SPECIAL_CHARS.contains(&c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Build the `FT.CREATE {index_name} ...` argument list (everything after
/// the command name itself), managing one RediSearch index per key prefix
/// over the JSON documents written to `{key_prefix}:entry:*`. `ON JSON`
/// requires each entry to be written with `JSON.SET` rather than a plain
/// `SET` of serialized text — see the module docs for how writes switch
/// based on detected capability. The schema mirrors every field of
/// `MemoryEntry`: `content` is the only full-text (`TEXT`) field
/// (BM25-scored); the scope/identity fields are exact-match `TAG`s; `rank`
/// is `NUMERIC SORTABLE`.
fn ft_create_args(key_prefix: &str, index_name: &str) -> Vec<String> {
    const TAG_FIELDS: &[&str] = &[
        "role",
        "application_id",
        "agent_id",
        "user_id",
        "thread_id",
        "message_id",
        "author_name",
    ];

    let mut args: Vec<String> = vec![
        index_name.to_string(),
        "ON".to_string(),
        "JSON".to_string(),
        "PREFIX".to_string(),
        "1".to_string(),
        format!("{key_prefix}:entry:"),
        "SCHEMA".to_string(),
        "$.content".to_string(),
        "AS".to_string(),
        "content".to_string(),
        "TEXT".to_string(),
    ];

    for field in TAG_FIELDS {
        args.push(format!("$.{field}"));
        args.push("AS".to_string());
        args.push((*field).to_string());
        args.push("TAG".to_string());
    }

    args.push("$.rank".to_string());
    args.push("AS".to_string());
    args.push("rank".to_string());
    args.push("NUMERIC".to_string());
    args.push("SORTABLE".to_string());

    args
}

/// Build the `FT.SEARCH` query-string argument (everything the caller
/// supplies after `FT.SEARCH {index}`) for `scope` (ANDed exact-match
/// `TAG` filters, mirroring [`Scope::matches`]) combined with an OR of
/// `tokens` against the `content` field (mirrors the OR/"any token"
/// semantics of the SCAN-fallback's [`select_recent`]). Every scope value
/// and token is escaped with [`escape_redisearch`] first. Falls back to
/// `"*"` (match everything) only if both `scope` and `tokens` are empty —
/// unreachable in practice, since `before_run()` requires non-empty `tokens`
/// and `validate_filters()` requires a non-empty `scope` before this is
/// ever called, but kept total rather than partial.
fn build_ft_search_query(scope: &Scope, tokens: &[String]) -> String {
    let mut clauses = Vec::new();
    if let Some(v) = &scope.application_id {
        clauses.push(format!("@application_id:{{{}}}", escape_redisearch(v)));
    }
    if let Some(v) = &scope.agent_id {
        clauses.push(format!("@agent_id:{{{}}}", escape_redisearch(v)));
    }
    if let Some(v) = &scope.user_id {
        clauses.push(format!("@user_id:{{{}}}", escape_redisearch(v)));
    }
    if let Some(v) = &scope.thread_id {
        clauses.push(format!("@thread_id:{{{}}}", escape_redisearch(v)));
    }
    if !tokens.is_empty() {
        let ored = tokens
            .iter()
            .map(|t| escape_redisearch(t))
            .collect::<Vec<_>>()
            .join("|");
        clauses.push(format!("@content:({ored})"));
    }
    if clauses.is_empty() {
        "*".to_string()
    } else {
        clauses.join(" ")
    }
}

/// Parse a raw `FT.SEARCH ... RETURN 1 $` reply into [`MemoryEntry`]
/// values, in the order RediSearch returned them (its own relevance
/// ranking — no `SORTBY` is sent, so this is BM25-descending). RESP2 shape:
/// `[total, doc_id_1, [field_1, value_1, ...], doc_id_2, ...]`; with
/// `RETURN 1 $` each per-doc field array is exactly `["$", "<json>"]`,
/// where `<json>` is the complete original document (RediSearch's
/// documented behavior for returning the JSON root of an `ON JSON`-indexed
/// document). Anything that doesn't match this shape (unexpected reply
/// type, a `$` value that fails to parse as [`MemoryEntry`]) is skipped
/// rather than failing the whole call, mirroring
/// [`RedisContextProvider::scan_entries`]'s leniency for the SCAN fallback.
fn parse_ft_search_reply(value: &redis::Value) -> Vec<MemoryEntry> {
    let redis::Value::Array(items) = value else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut i = 1; // items[0] is the total result count.
    while i + 1 < items.len() {
        if let redis::Value::Array(fields) = &items[i + 1] {
            let mut j = 0;
            while j + 1 < fields.len() {
                if let redis::Value::BulkString(name) = &fields[j] {
                    if name == b"$" {
                        if let redis::Value::BulkString(raw) = &fields[j + 1] {
                            if let Ok(text) = std::str::from_utf8(raw) {
                                if let Ok(entry) = serde_json::from_str::<MemoryEntry>(text) {
                                    out.push(entry);
                                }
                            }
                        }
                    }
                }
                j += 2;
            }
        }
        i += 2;
    }
    out
}

/// Whether an `FT.CREATE` error means "this index already exists" — safe to
/// swallow, since [`RedisContextProvider::ensure_index`] creates
/// optimistically (no separate existence probe first) and multiple
/// provider instances/processes sharing a `key_prefix` will race to create
/// the same index. RediSearch's documented error text for this case is the
/// literal string `Index already exists`; `redis-rs` splits a reply's first
/// word off as its error code, so the same reply can also display as
/// `"Index": already exists` (observed against RediSearch 2.10), and both
/// forms are matched.
pub(crate) fn is_index_exists_error(message: &str) -> bool {
    let message = message.to_lowercase();
    message.contains("index already exists")
        || (message.contains("\"index\"") && message.contains("already exists"))
}

/// Probe whether the connected server has RediSearch loaded, via `FT._LIST`
/// (lists index names; succeeds — possibly with an empty array — iff the
/// module is present). Never propagates an error: any failure (unknown
/// command on plain/OSS Redis, a transient connection error, ...) is
/// treated as "not available" so callers always get a definite yes/no to
/// cache.
async fn probe_redisearch(conn: &mut redis::aio::MultiplexedConnection) -> bool {
    redis::cmd("FT._LIST")
        .query_async::<Vec<String>>(conn)
        .await
        .is_ok()
}

// endregion

/// Redis-backed [`ContextProvider`]: recency-based memory storage/retrieval
/// scoped by application/agent/user/thread id, upgraded to a real
/// `FT.SEARCH` BM25 full-text index when the connected server supports
/// RediSearch (Redis Stack). See the module docs for the RediSearch-vs-SCAN
/// behavior and the remaining vector-search divergence from Python's
/// `RedisProvider`.
///
/// ```no_run
/// use agent_framework_redis::RedisContextProvider;
/// use agent_framework_core::memory::{ContextProvider, SessionContext};
/// use agent_framework_core::types::Message;
///
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let provider = RedisContextProvider::new("redis://127.0.0.1:6379")?
///     .with_user_id("user-42")
///     .with_limit(5);
///
/// let request = vec![Message::user("I love hiking in the Cascades")];
/// provider.after_run(&request, &[], None).await?;
///
/// let mut ctx = SessionContext::new(vec![Message::user("Any outdoor hobbies?")]);
/// provider.before_run(&mut ctx).await?;
/// # Ok(())
/// # }
/// ```
pub struct RedisContextProvider {
    conn: LazyConnection,
    key_prefix: String,
    application_id: Option<String>,
    agent_id: Option<String>,
    user_id: Option<String>,
    thread_id: Option<String>,
    scope_to_per_operation_thread_id: bool,
    context_prompt: String,
    limit: usize,
    /// The per-operation thread id, captured from `SessionContext::session_id`
    /// on `before_run()` (the agent threads its scoping id through
    /// `SessionContext` — see [`agent_framework_core::memory::SessionContext`]).
    /// Cached here (rather than only living on the stack) so `after_run()` —
    /// which the trait does not pass a `SessionContext` to — can still scope
    /// its storage write to the same thread. Only consulted when
    /// `scope_to_per_operation_thread_id` is set.
    session_thread_id: Mutex<Option<String>>,
    /// When `true`, always use the SCAN fallback even if RediSearch is
    /// available (builder: [`Self::with_force_scan_fallback`]).
    force_scan_fallback: bool,
    /// Cached result of the one-time `FT._LIST` capability probe (skipped
    /// entirely when `force_scan_fallback` is set).
    search_capability: OnceCell<bool>,
    /// Set once this provider's RediSearch index has been created (or
    /// confirmed to already exist).
    index_ready: OnceCell<()>,
}

impl RedisContextProvider {
    /// Create a provider for `redis_url` with no scope configured yet (at
    /// least one of application/agent/user/thread id must be set via the
    /// builder methods before `before_run`/`after_run` are called).
    pub fn new(redis_url: impl Into<String>) -> Result<Self> {
        let redis_url = redis_url.into();
        let conn = LazyConnection::open(&redis_url)?;
        Ok(Self {
            conn,
            key_prefix: DEFAULT_KEY_PREFIX.to_string(),
            application_id: None,
            agent_id: None,
            user_id: None,
            thread_id: None,
            scope_to_per_operation_thread_id: false,
            context_prompt: DEFAULT_CONTEXT_PROMPT.to_string(),
            limit: DEFAULT_LIMIT,
            session_thread_id: Mutex::new(None),
            force_scan_fallback: false,
            search_capability: OnceCell::new(),
            index_ready: OnceCell::new(),
        })
    }

    /// Namespace Redis keys under `key_prefix` (builder style). Defaults to
    /// `"context"`.
    pub fn with_key_prefix(mut self, key_prefix: impl Into<String>) -> Self {
        self.key_prefix = key_prefix.into();
        self
    }

    /// Scope memories to an application id (builder style).
    pub fn with_application_id(mut self, application_id: impl Into<String>) -> Self {
        self.application_id = Some(application_id.into());
        self
    }

    /// Scope memories to an agent id (builder style).
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Scope memories to a user id (builder style).
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Scope memories to a thread id (builder style).
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// When `true`, the thread id used for scoping is captured from the
    /// `session_id` of the first [`SessionContext`] seen by
    /// [`ContextProvider::before_run`] instead of the static `thread_id`
    /// above, and a conflicting `session_id` on a later call is an error
    /// (builder style).
    pub fn with_scope_to_per_operation_thread_id(mut self, value: bool) -> Self {
        self.scope_to_per_operation_thread_id = value;
        self
    }

    /// Override the header prepended to injected memories (builder style).
    /// Defaults to [`DEFAULT_CONTEXT_PROMPT`].
    pub fn with_context_prompt(mut self, context_prompt: impl Into<String>) -> Self {
        self.context_prompt = context_prompt.into();
        self
    }

    /// Maximum memories returned per `before_run()` call (builder style).
    /// Defaults to [`DEFAULT_LIMIT`].
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Force the plain-`SCAN` retrieval path even when the connected server
    /// supports RediSearch (builder style). `false` by default: RediSearch
    /// is detected and used automatically. Useful for tests, for pinning
    /// behavior during a Redis Stack rollout, or for working around an
    /// operational RediSearch problem without a code change. See the
    /// module docs for why entries written under one path are not visible
    /// via the other.
    pub fn with_force_scan_fallback(mut self, force: bool) -> Self {
        self.force_scan_fallback = force;
        self
    }

    fn validate_filters(&self) -> Result<()> {
        if self.application_id.is_none()
            && self.agent_id.is_none()
            && self.user_id.is_none()
            && self.thread_id.is_none()
        {
            return Err(Error::Configuration(
                "At least one of the filters: agent_id, user_id, application_id, or thread_id is required."
                    .into(),
            ));
        }
        Ok(())
    }

    async fn effective_thread_id(&self) -> Option<String> {
        if self.scope_to_per_operation_thread_id {
            self.session_thread_id.lock().await.clone()
        } else {
            self.thread_id.clone()
        }
    }

    /// Record `session_id` (from `SessionContext::session_id`, on
    /// `before_run()`) as this provider's per-operation scoping thread id.
    /// Port of the old `ContextProvider::thread_created` hook, which is no
    /// longer part of the trait: the agent now threads the scoping id
    /// through `SessionContext` instead of a separate callback, so this is
    /// invoked directly from `before_run()`. Same semantics as the old
    /// method: the first non-`None` id seen wins, and — only when
    /// [`Self::with_scope_to_per_operation_thread_id`] is set — a later call
    /// with a *different* non-`None` id is a configuration error, since this
    /// provider can only be scoped to a single thread at a time.
    async fn record_session_thread_id(&self, session_id: Option<&str>) -> Result<()> {
        let mut guard = self.session_thread_id.lock().await;
        if self.scope_to_per_operation_thread_id {
            if let (Some(new_id), Some(existing)) = (session_id, guard.as_deref()) {
                if new_id != existing {
                    return Err(Error::other(
                        "RedisContextProvider can only be used with one thread, when scope_to_per_operation_thread_id is True.",
                    ));
                }
            }
        }
        if guard.is_none() {
            *guard = session_id.map(String::from);
        }
        Ok(())
    }

    async fn scope(&self) -> Scope {
        Scope {
            application_id: self.application_id.clone(),
            agent_id: self.agent_id.clone(),
            user_id: self.user_id.clone(),
            thread_id: self.effective_thread_id().await,
        }
    }

    /// The configured prefix, rendered as one unambiguous key segment.
    ///
    /// A raw prefix makes the key space ambiguous in two ways at once. A
    /// prefix containing `:entry:` puts one provider's entries inside
    /// another's `SCAN MATCH` pattern — so it reads, and `clear()` deletes,
    /// memories that are not its own — and a prefix containing a glob
    /// metacharacter (`*`, `?`, `[`) makes that pattern match something else
    /// again. Encoding the prefix closes both; a literal-safe prefix (the
    /// common case) is unchanged, so existing keys still resolve.
    fn key_segment(&self) -> String {
        storage_key_segment(&self.key_prefix, "~p-")
    }

    fn entry_key(&self, id: &str) -> String {
        format!("{}:entry:{}", self.key_segment(), id)
    }

    fn scan_pattern(&self) -> String {
        format!("{}:entry:*", self.key_segment())
    }

    /// RediSearch index name for this provider's `key_prefix` — one index
    /// per prefix, so distinct [`Self::with_key_prefix`] configurations
    /// never collide on the same server.
    fn index_name(&self) -> String {
        format!("{}_idx", self.key_segment())
    }

    /// `SCAN MATCH {key_prefix}:entry:*`, then `MGET` the matched keys and
    /// parse each JSON value. Keys that vanish between `SCAN` and `MGET`
    /// (e.g. concurrently cleared), whose value fails to parse, or whose
    /// value isn't a plain string (e.g. a RedisJSON document written via
    /// the RediSearch path — see the module docs) are silently skipped
    /// rather than failing the whole call.
    async fn scan_entries(&self) -> Result<Vec<MemoryEntry>> {
        let mut conn = self.conn.get().await?;
        let pattern = self.scan_pattern();

        let mut keys: Vec<String> = Vec::new();
        {
            let mut iter: redis::AsyncIter<'_, String> =
                conn.scan_match(&pattern).await.map_err(map_redis_err)?;
            while let Some(item) = iter.next_item().await {
                keys.push(item.map_err(map_redis_err)?);
            }
        }
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let raw: Vec<Option<String>> = conn.mget(&keys).await.map_err(map_redis_err)?;
        Ok(raw
            .into_iter()
            .flatten()
            .filter_map(|v| serde_json::from_str::<MemoryEntry>(&v).ok())
            .collect())
    }

    /// Whether this call should use RediSearch: `false` immediately if
    /// [`Self::with_force_scan_fallback`] was set (no network probe at
    /// all — forcing fallback must work even against a server that would
    /// otherwise answer `FT._LIST`); otherwise the result of
    /// [`probe_redisearch`], cached for the lifetime of this provider
    /// after the first call.
    async fn use_redisearch(&self, conn: &mut redis::aio::MultiplexedConnection) -> bool {
        if self.force_scan_fallback {
            return false;
        }
        *self
            .search_capability
            .get_or_init(move || async move { probe_redisearch(conn).await })
            .await
    }

    /// Idempotently create this provider's RediSearch index (`FT.CREATE`,
    /// tolerating "already exists"), memoized after the first successful
    /// attempt so steady-state calls skip the round trip entirely. Only
    /// called once [`Self::use_redisearch`] has confirmed RediSearch is
    /// available.
    async fn ensure_index(&self, conn: &mut redis::aio::MultiplexedConnection) -> Result<()> {
        // The *encoded* prefix, the one `entry_key` writes under. Passing the
        // raw prefix points the index at a namespace nothing is written to
        // whenever the prefix encodes, so every write succeeds and every
        // `FT.SEARCH` comes back empty — a silent retrieval failure rather
        // than an error.
        let args = ft_create_args(&self.key_segment(), &self.index_name());
        self.index_ready
            .get_or_try_init(move || async move {
                let mut cmd = redis::cmd("FT.CREATE");
                for arg in args {
                    cmd.arg(arg);
                }
                match cmd.query_async::<redis::Value>(conn).await {
                    Ok(_) => Ok(()),
                    Err(e) if is_index_exists_error(&e.to_string()) => Ok(()),
                    Err(e) => Err(map_redis_err(e)),
                }
            })
            .await?;
        Ok(())
    }

    /// Run `FT.SEARCH` for `scope`+`tokens`, returning matching entries in
    /// the order RediSearch returns them (its default relevance scoring,
    /// descending — see the module docs for why no explicit `SCORER` is
    /// sent). Ensures the index exists first (see [`Self::ensure_index`]).
    async fn ft_search(
        &self,
        conn: &mut redis::aio::MultiplexedConnection,
        scope: &Scope,
        tokens: &[String],
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        self.ensure_index(conn).await?;
        let query = build_ft_search_query(scope, tokens);
        let mut cmd = redis::cmd("FT.SEARCH");
        cmd.arg(self.index_name())
            .arg(&query)
            .arg("RETURN")
            .arg(1)
            .arg("$")
            .arg("LIMIT")
            .arg(0)
            .arg(limit);
        let reply = cmd
            .query_async::<redis::Value>(conn)
            .await
            .map_err(map_redis_err)?;
        Ok(parse_ft_search_reply(&reply))
    }
}

#[async_trait]
impl ContextProvider for RedisContextProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        self.validate_filters()?;
        // Replaces the old `thread_created` hook: the agent now threads the
        // per-operation scoping id through `SessionContext::session_id`
        // rather than a separate callback, so we capture it here, on every
        // `before_run()` call, exactly as `thread_created` used to be called
        // on every thread-creation event.
        self.record_session_thread_id(ctx.session_id.as_deref())
            .await?;

        let input_text = ctx
            .input_messages
            .iter()
            .map(Message::text)
            .filter(|t| !t.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if input_text.trim().is_empty() {
            // Python's RedisProvider unconditionally calls `_redis_search`,
            // which raises on empty text; we instead treat "nothing to
            // search for" as "no memories" — see module docs.
            return Ok(());
        }

        let scope = self.scope().await;
        if scope.is_empty() {
            // Unreachable given `validate_filters` above (it guarantees at
            // least one scope field), but keep `Scope` honest in isolation.
            return Ok(());
        }

        let mut conn = self.conn.get().await?;
        let hits = if self.use_redisearch(&mut conn).await {
            let tokens = tokenize_query(&input_text);
            self.ft_search(&mut conn, &scope, &tokens, self.limit)
                .await?
        } else {
            let entries = self.scan_entries().await?;
            select_recent(entries, &scope, Some(&input_text), self.limit)
        };

        let joined = hits
            .iter()
            .map(|e| e.content.as_str())
            .filter(|c| !c.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        if let Some(message) = format_context_message(&self.context_prompt, &joined) {
            ctx.messages.push(message);
        }

        Ok(())
    }

    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        _error: Option<&Error>,
    ) -> Result<()> {
        self.validate_filters()?;
        let scope = self.scope().await;
        let now = now_millis();

        let entries: Vec<MemoryEntry> = request_messages
            .iter()
            .chain(response_messages.iter())
            .enumerate()
            .filter(|(_, m)| is_storable_role(&m.role))
            .filter_map(|(i, m)| {
                let text = m.text();
                if text.trim().is_empty() {
                    return None;
                }
                Some(MemoryEntry {
                    content: text,
                    role: m.role.as_str().to_string(),
                    application_id: scope.application_id.clone(),
                    agent_id: scope.agent_id.clone(),
                    user_id: scope.user_id.clone(),
                    thread_id: scope.thread_id.clone(),
                    message_id: m.message_id.clone(),
                    author_name: m.author_name.clone(),
                    rank: now * 1000 + i as i64,
                })
            })
            .collect();

        if entries.is_empty() {
            return Ok(());
        }

        let mut conn = self.conn.get().await?;
        let mut pipe = redis::pipe();
        pipe.atomic();
        if self.use_redisearch(&mut conn).await {
            // RediSearch (ON JSON) can only index native RedisJSON
            // documents, not opaque strings — see the module docs.
            self.ensure_index(&mut conn).await?;
            for entry in &entries {
                let key = self.entry_key(&Uuid::new_v4().to_string());
                let payload = serde_json::to_string(entry)?;
                pipe.cmd("JSON.SET").arg(key).arg("$").arg(payload);
            }
        } else {
            for entry in &entries {
                let key = self.entry_key(&Uuid::new_v4().to_string());
                let payload = serde_json::to_string(entry)?;
                pipe.set(key, payload);
            }
        }
        let _: () = pipe.query_async(&mut conn).await.map_err(map_redis_err)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> RedisContextProvider {
        RedisContextProvider::new("redis://127.0.0.1:6379/0")
            .expect("valid redis url")
            .with_user_id("u1")
    }

    fn entry(content: &str, rank: i64) -> MemoryEntry {
        MemoryEntry {
            content: content.to_string(),
            role: "user".to_string(),
            application_id: None,
            agent_id: None,
            user_id: Some("u1".to_string()),
            thread_id: None,
            message_id: None,
            author_name: None,
            rank,
        }
    }

    // region: key construction

    #[test]
    fn entry_key_and_scan_pattern_use_key_prefix() {
        let p = provider();
        assert_eq!(p.entry_key("abc"), "context:entry:abc");
        assert_eq!(p.scan_pattern(), "context:entry:*");
    }

    #[test]
    fn custom_key_prefix_propagates() {
        let p = provider().with_key_prefix("myapp");
        assert_eq!(p.entry_key("abc"), "myapp:entry:abc");
        assert_eq!(p.scan_pattern(), "myapp:entry:*");
    }

    #[test]
    fn invalid_redis_url_is_rejected() {
        assert!(RedisContextProvider::new("not-a-redis-url").is_err());
    }

    #[test]
    fn index_name_derives_from_key_prefix() {
        let p = provider();
        assert_eq!(p.index_name(), "context_idx");
        let p = p.with_key_prefix("myapp");
        assert_eq!(p.index_name(), "myapp_idx");
    }

    // endregion

    // region: validate_filters

    #[test]
    fn validate_filters_rejects_no_scope() {
        let p = RedisContextProvider::new("redis://127.0.0.1:6379/0").unwrap();
        assert!(p.validate_filters().is_err());
    }

    #[test]
    fn validate_filters_accepts_any_single_scope_field() {
        let base = || RedisContextProvider::new("redis://127.0.0.1:6379/0").unwrap();
        assert!(base().with_user_id("u").validate_filters().is_ok());
        assert!(base().with_agent_id("a").validate_filters().is_ok());
        assert!(base().with_application_id("ap").validate_filters().is_ok());
        assert!(base().with_thread_id("t").validate_filters().is_ok());
    }

    // endregion

    // region: Scope matching (pure, no server)

    #[test]
    fn scope_with_no_fields_matches_everything() {
        let scope = Scope::default();
        assert!(scope.matches(&entry("hello", 1)));
    }

    #[test]
    fn scope_field_is_wildcard_when_unset() {
        let scope = Scope {
            user_id: Some("u1".to_string()),
            ..Default::default()
        };
        // agent_id/application_id/thread_id are None on both scope and
        // entry, so only user_id needs to match.
        assert!(scope.matches(&entry("hi", 1)));
    }

    #[test]
    fn scope_rejects_mismatched_field() {
        let scope = Scope {
            user_id: Some("someone-else".to_string()),
            ..Default::default()
        };
        assert!(!scope.matches(&entry("hi", 1)));
    }

    #[test]
    fn scope_requires_all_configured_fields_to_match() {
        let mut e = entry("hi", 1);
        e.agent_id = Some("agentA".to_string());
        let scope = Scope {
            user_id: Some("u1".to_string()),
            agent_id: Some("agentB".to_string()),
            ..Default::default()
        };
        assert!(!scope.matches(&e));
    }

    // endregion

    // region: tokenize_query (pure, no server)

    #[test]
    fn tokenize_query_lowercases_and_splits_on_non_alphanumeric() {
        assert_eq!(
            tokenize_query("Seattle, WA!"),
            vec!["seattle".to_string(), "wa".to_string()]
        );
    }

    #[test]
    fn tokenize_query_drops_stopwords() {
        assert_eq!(
            tokenize_query("What is the capital of France?"),
            vec!["capital".to_string(), "france".to_string()]
        );
    }

    #[test]
    fn tokenize_query_all_stopwords_yields_empty() {
        assert!(tokenize_query("is the of").is_empty());
    }

    // endregion

    // region: select_recent (pure, no server)

    #[test]
    fn select_recent_orders_by_rank_descending() {
        let entries = vec![entry("old", 1), entry("newest", 3), entry("mid", 2)];
        let scope = Scope::default();
        let hits = select_recent(entries, &scope, None, 10);
        assert_eq!(
            hits.iter().map(|e| e.content.as_str()).collect::<Vec<_>>(),
            vec!["newest", "mid", "old"]
        );
    }

    #[test]
    fn select_recent_respects_limit() {
        let entries = vec![entry("a", 1), entry("b", 2), entry("c", 3)];
        let hits = select_recent(entries, &Scope::default(), None, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].content, "c");
        assert_eq!(hits[1].content, "b");
    }

    #[test]
    fn select_recent_filters_by_scope() {
        let mut other_user = entry("secret", 5);
        other_user.user_id = Some("someone-else".to_string());
        let entries = vec![entry("mine", 1), other_user];

        let scope = Scope {
            user_id: Some("u1".to_string()),
            ..Default::default()
        };
        let hits = select_recent(entries, &scope, None, 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "mine");
    }

    #[test]
    fn select_recent_text_filter_matches_token_case_insensitively() {
        let entries = vec![
            entry("User likes outdoor activities", 1),
            entry("User lives in Seattle", 2),
            entry("Completely unrelated fact", 3),
        ];
        let hits = select_recent(entries, &Scope::default(), Some("SEATTLE weather"), 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "User lives in Seattle");
    }

    #[test]
    fn select_recent_text_filter_excludes_non_matching() {
        let entries = vec![entry("apples and oranges", 1)];
        let hits = select_recent(entries, &Scope::default(), Some("bananas"), 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn select_recent_none_query_skips_text_filter() {
        let entries = vec![entry("anything at all", 1)];
        let hits = select_recent(entries, &Scope::default(), None, 10);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn select_recent_blank_query_skips_text_filter() {
        let entries = vec![entry("anything at all", 1)];
        let hits = select_recent(entries, &Scope::default(), Some("   "), 10);
        assert_eq!(hits.len(), 1);
    }

    // endregion

    // region: format_context_message

    #[test]
    fn format_context_message_none_when_no_hits() {
        assert!(format_context_message(DEFAULT_CONTEXT_PROMPT, "").is_none());
    }

    #[test]
    fn format_context_message_builds_user_message_with_prompt_header() {
        let message = format_context_message(DEFAULT_CONTEXT_PROMPT, "A\nB").unwrap();
        assert_eq!(message.role, Role::user());
        assert_eq!(
            message.text(),
            "## Memories\nConsider the following memories when answering user questions:\nA\nB"
        );
    }

    // endregion

    // region: is_storable_role

    #[test]
    fn is_storable_role_allows_user_assistant_system() {
        assert!(is_storable_role(&Role::user()));
        assert!(is_storable_role(&Role::assistant()));
        assert!(is_storable_role(&Role::system()));
    }

    #[test]
    fn is_storable_role_rejects_tool() {
        assert!(!is_storable_role(&Role::tool()));
    }

    // endregion

    // region: escape_redisearch (pure, no server)

    #[test]
    fn escape_redisearch_leaves_plain_alphanumeric_untouched() {
        assert_eq!(escape_redisearch("hello123"), "hello123");
    }

    #[test]
    fn escape_redisearch_escapes_hyphen_in_uuid_like_value() {
        assert_eq!(
            escape_redisearch("thread-42-abc"),
            "thread\\-42\\-abc".to_string()
        );
    }

    #[test]
    fn escape_redisearch_escapes_email_like_value() {
        assert_eq!(
            escape_redisearch("user@example.com"),
            "user\\@example\\.com".to_string()
        );
    }

    #[test]
    fn escape_redisearch_escapes_pipe_and_braces() {
        assert_eq!(escape_redisearch("a|b{c}"), "a\\|b\\{c\\}".to_string());
    }

    #[test]
    fn escape_redisearch_escapes_backslash_itself() {
        assert_eq!(escape_redisearch("a\\b"), "a\\\\b".to_string());
    }

    #[test]
    fn escape_redisearch_escapes_whitespace() {
        assert_eq!(escape_redisearch("two words"), "two\\ words".to_string());
    }

    // endregion

    // region: ft_create_args (pure, no server)

    #[test]
    fn ft_create_args_starts_with_index_name_and_json_prefix() {
        let args = ft_create_args("context", "context_idx");
        assert_eq!(args[0], "context_idx");
        assert_eq!(args[1], "ON");
        assert_eq!(args[2], "JSON");
        assert_eq!(args[3], "PREFIX");
        assert_eq!(args[4], "1");
        assert_eq!(args[5], "context:entry:");
        assert_eq!(args[6], "SCHEMA");
    }

    #[test]
    fn ft_create_args_content_field_is_text() {
        let args = ft_create_args("context", "context_idx");
        let pos = args.iter().position(|a| a == "$.content").unwrap();
        assert_eq!(args[pos + 1], "AS");
        assert_eq!(args[pos + 2], "content");
        assert_eq!(args[pos + 3], "TEXT");
    }

    #[test]
    fn ft_create_args_scope_fields_are_tags() {
        let args = ft_create_args("context", "context_idx");
        for field in [
            "application_id",
            "agent_id",
            "user_id",
            "thread_id",
            "role",
            "message_id",
            "author_name",
        ] {
            let path = format!("$.{field}");
            let pos = args
                .iter()
                .position(|a| a == &path)
                .unwrap_or_else(|| panic!("missing schema field {field}"));
            assert_eq!(args[pos + 1], "AS");
            assert_eq!(args[pos + 2], field);
            assert_eq!(args[pos + 3], "TAG");
        }
    }

    #[test]
    fn ft_create_args_rank_field_is_numeric_sortable() {
        let args = ft_create_args("context", "context_idx");
        let pos = args.iter().position(|a| a == "$.rank").unwrap();
        assert_eq!(args[pos + 1], "AS");
        assert_eq!(args[pos + 2], "rank");
        assert_eq!(args[pos + 3], "NUMERIC");
        assert_eq!(args[pos + 4], "SORTABLE");
    }

    #[test]
    fn ft_create_args_uses_custom_key_prefix() {
        let args = ft_create_args("myapp", "myapp_idx");
        assert_eq!(args[0], "myapp_idx");
        assert_eq!(args[5], "myapp:entry:");
    }

    // endregion

    // region: build_ft_search_query (pure, no server)

    #[test]
    fn build_ft_search_query_single_scope_field_and_tokens() {
        let scope = Scope {
            user_id: Some("u1".to_string()),
            ..Default::default()
        };
        let tokens = vec!["seattle".to_string()];
        assert_eq!(
            build_ft_search_query(&scope, &tokens),
            "@user_id:{u1} @content:(seattle)"
        );
    }

    #[test]
    fn build_ft_search_query_ands_all_configured_scope_fields() {
        let scope = Scope {
            application_id: Some("app1".to_string()),
            agent_id: Some("agent1".to_string()),
            user_id: Some("u1".to_string()),
            thread_id: Some("t1".to_string()),
        };
        let query = build_ft_search_query(&scope, &[]);
        assert_eq!(
            query,
            "@application_id:{app1} @agent_id:{agent1} @user_id:{u1} @thread_id:{t1}"
        );
    }

    #[test]
    fn build_ft_search_query_ors_multiple_tokens() {
        let scope = Scope {
            user_id: Some("u1".to_string()),
            ..Default::default()
        };
        let tokens = vec!["capital".to_string(), "france".to_string()];
        assert_eq!(
            build_ft_search_query(&scope, &tokens),
            "@user_id:{u1} @content:(capital|france)"
        );
    }

    #[test]
    fn build_ft_search_query_escapes_special_characters_in_scope_values() {
        let scope = Scope {
            user_id: Some("user@example.com".to_string()),
            ..Default::default()
        };
        let query = build_ft_search_query(&scope, &[]);
        assert_eq!(query, "@user_id:{user\\@example\\.com}");
    }

    #[test]
    fn build_ft_search_query_no_scope_no_tokens_is_wildcard() {
        assert_eq!(build_ft_search_query(&Scope::default(), &[]), "*");
    }

    #[test]
    fn build_ft_search_query_no_tokens_omits_content_clause() {
        let scope = Scope {
            thread_id: Some("t1".to_string()),
            ..Default::default()
        };
        assert_eq!(build_ft_search_query(&scope, &[]), "@thread_id:{t1}");
    }

    // endregion

    // region: parse_ft_search_reply (pure, no server — hand-built redis::Value fixtures)

    #[test]
    fn parse_ft_search_reply_extracts_entries_from_dollar_field() {
        let entry_json = serde_json::to_string(&entry("hello", 1)).unwrap();
        let reply = redis::Value::Array(vec![
            redis::Value::Int(1),
            redis::Value::BulkString(b"context:entry:abc".to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"$".to_vec()),
                redis::Value::BulkString(entry_json.into_bytes()),
            ]),
        ]);
        let parsed = parse_ft_search_reply(&reply);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].content, "hello");
    }

    #[test]
    fn parse_ft_search_reply_preserves_order_across_multiple_docs() {
        let e1 = serde_json::to_string(&entry("first", 1)).unwrap();
        let e2 = serde_json::to_string(&entry("second", 2)).unwrap();
        let reply = redis::Value::Array(vec![
            redis::Value::Int(2),
            redis::Value::BulkString(b"k1".to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"$".to_vec()),
                redis::Value::BulkString(e1.into_bytes()),
            ]),
            redis::Value::BulkString(b"k2".to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"$".to_vec()),
                redis::Value::BulkString(e2.into_bytes()),
            ]),
        ]);
        let parsed = parse_ft_search_reply(&reply);
        assert_eq!(
            parsed
                .iter()
                .map(|e| e.content.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
    }

    #[test]
    fn parse_ft_search_reply_empty_results() {
        let reply = redis::Value::Array(vec![redis::Value::Int(0)]);
        assert!(parse_ft_search_reply(&reply).is_empty());
    }

    #[test]
    fn parse_ft_search_reply_non_array_top_level_yields_empty() {
        assert!(parse_ft_search_reply(&redis::Value::Nil).is_empty());
    }

    #[test]
    fn parse_ft_search_reply_skips_malformed_json_without_failing() {
        let reply = redis::Value::Array(vec![
            redis::Value::Int(1),
            redis::Value::BulkString(b"k1".to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"$".to_vec()),
                redis::Value::BulkString(b"not json".to_vec()),
            ]),
        ]);
        assert!(parse_ft_search_reply(&reply).is_empty());
    }

    #[test]
    fn parse_ft_search_reply_ignores_fields_other_than_dollar() {
        let reply = redis::Value::Array(vec![
            redis::Value::Int(1),
            redis::Value::BulkString(b"k1".to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"content".to_vec()),
                redis::Value::BulkString(b"hello".to_vec()),
            ]),
        ]);
        assert!(parse_ft_search_reply(&reply).is_empty());
    }

    // endregion

    // region: is_index_exists_error (pure, no server)

    #[test]
    fn is_index_exists_error_matches_redisearch_error_text() {
        assert!(is_index_exists_error("Index already exists"));
        assert!(is_index_exists_error(
            "An error was signalled by the server - ResponseError: Index already exists"
        ));
    }

    #[test]
    fn is_index_exists_error_case_insensitive() {
        assert!(is_index_exists_error("INDEX ALREADY EXISTS"));
    }

    #[test]
    fn is_index_exists_error_matches_the_redis_rs_split_form() {
        // redis-rs renders the reply's first word as an error code.
        assert!(is_index_exists_error(
            "redis error: \"Index\": already exists"
        ));
        assert!(!is_index_exists_error("\"Index\": unknown field"));
    }

    #[test]
    fn is_index_exists_error_rejects_unrelated_errors() {
        assert!(!is_index_exists_error("unknown command 'FT.CREATE'"));
        assert!(!is_index_exists_error("connection refused"));
    }

    // endregion

    // region: with_force_scan_fallback builder (pure, no server)

    #[test]
    fn force_scan_fallback_defaults_to_false() {
        assert!(!provider().force_scan_fallback);
    }

    #[test]
    fn with_force_scan_fallback_sets_flag() {
        assert!(
            provider()
                .with_force_scan_fallback(true)
                .force_scan_fallback
        );
    }

    // endregion

    // region: before_run() session_id scoping (async, no server: empty
    // input_messages makes before_run() return before any I/O, right after
    // recording the session thread id — same "pure Mutex state" coverage the
    // old thread_created tests had.

    #[tokio::test]
    async fn before_run_sets_session_thread_id() {
        let p = provider().with_scope_to_per_operation_thread_id(true);
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some("t1".to_string());
        p.before_run(&mut ctx).await.unwrap();
        assert_eq!(p.session_thread_id.lock().await.as_deref(), Some("t1"));
    }

    #[tokio::test]
    async fn before_run_does_not_overwrite_existing_session_thread_id() {
        let p = provider().with_scope_to_per_operation_thread_id(true);
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some("t1".to_string());
        p.before_run(&mut ctx).await.unwrap();
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some("t1".to_string());
        p.before_run(&mut ctx).await.unwrap();
        assert_eq!(p.session_thread_id.lock().await.as_deref(), Some("t1"));
    }

    #[tokio::test]
    async fn before_run_conflict_when_scoped() {
        let p = provider().with_scope_to_per_operation_thread_id(true);
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some("t1".to_string());
        p.before_run(&mut ctx).await.unwrap();
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some("t2".to_string());
        let err = p.before_run(&mut ctx).await.unwrap_err();
        assert!(err.to_string().contains("only be used with one thread"));
    }

    #[tokio::test]
    async fn before_run_allows_none_session_id_repeatedly() {
        let p = provider().with_scope_to_per_operation_thread_id(true);
        let mut ctx = SessionContext::new(vec![]);
        p.before_run(&mut ctx).await.unwrap();
        let mut ctx = SessionContext::new(vec![]);
        p.before_run(&mut ctx).await.unwrap();
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some("t1".to_string());
        p.before_run(&mut ctx).await.unwrap();
        assert_eq!(p.session_thread_id.lock().await.as_deref(), Some("t1"));
    }

    #[tokio::test]
    async fn before_run_without_scoping_never_conflicts() {
        let p = provider(); // scope_to_per_operation_thread_id defaults to false
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some("t1".to_string());
        p.before_run(&mut ctx).await.unwrap();
        // No conflict error even though the id changes, since scoping is off.
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some("t2".to_string());
        p.before_run(&mut ctx).await.unwrap();
    }

    // endregion

    // region: before_run()/after_run() input validation (async, no server: fails before any I/O)

    #[tokio::test]
    async fn before_run_fails_without_scope_configured() {
        let p = RedisContextProvider::new("redis://127.0.0.1:6379/0").unwrap();
        let mut ctx = SessionContext::new(vec![Message::user("hi")]);
        let err = p.before_run(&mut ctx).await.unwrap_err();
        assert!(err.to_string().contains("At least one of the filters"));
    }

    #[tokio::test]
    async fn after_run_fails_without_scope_configured() {
        let p = RedisContextProvider::new("redis://127.0.0.1:6379/0").unwrap();
        let err = p
            .after_run(&[Message::user("hi")], &[], None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("At least one of the filters"));
    }

    // endregion
}

#[cfg(test)]
mod key_scoping_tests {
    use super::*;

    fn provider_with(prefix: &str) -> RedisContextProvider {
        RedisContextProvider::new("redis://127.0.0.1:6379")
            .expect("provider")
            .with_key_prefix(prefix)
    }

    #[test]
    fn a_prefix_containing_the_entry_marker_cannot_capture_another_providers_entries() {
        // `SCAN MATCH a:entry:*` also matches `a:entry:b:entry:<uuid>`, so a
        // provider scoped to `a` would read — and `clear()` would delete —
        // the memories of one scoped to `a:entry:b`.
        let outer = provider_with("a");
        let inner = provider_with("a:entry:b");
        let captured = inner.entry_key("0000");
        let pattern = outer.scan_pattern();
        let literal_prefix = pattern.trim_end_matches('*');
        assert!(
            !captured.starts_with(literal_prefix),
            "{captured} is still inside {pattern}"
        );
    }

    #[test]
    fn a_literal_safe_prefix_keeps_its_existing_keys_and_index_name() {
        let p = provider_with("myapp");
        assert_eq!(p.entry_key("id1"), "myapp:entry:id1");
        assert_eq!(p.scan_pattern(), "myapp:entry:*");
        assert_eq!(p.index_name(), "myapp_idx");
    }
}

#[cfg(test)]
mod index_prefix_tests {
    use super::*;

    /// The RediSearch index must watch the namespace entries are actually
    /// written to. If `FT.CREATE ... PREFIX` gets the raw prefix while
    /// `entry_key` writes under the encoded one, every write succeeds and
    /// every search comes back empty — a silent retrieval failure, which is
    /// the worst shape this bug could take.
    #[test]
    fn the_index_prefix_matches_the_keys_entries_are_written_under() {
        let provider = RedisContextProvider::new("redis://127.0.0.1:6379")
            .expect("provider")
            .with_key_prefix("tenant:a");
        let args = ft_create_args(&provider.key_segment(), &provider.index_name());

        let prefix_arg = args
            .iter()
            .position(|a| a == "PREFIX")
            .map(|i| args[i + 2].clone())
            .expect("a PREFIX argument");
        let entry = provider.entry_key("0000");
        assert!(
            entry.starts_with(&prefix_arg),
            "entries are written to {entry}, index watches {prefix_arg}"
        );
        // And the prefix really is the encoded one, so this is not passing by
        // accident on a literal-safe name.
        assert!(prefix_arg.starts_with("~p-"), "{prefix_arg}");
    }
}
