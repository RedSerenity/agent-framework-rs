//! [`RedisHistoryProvider`]: the upstream-parity Redis history provider.
//!
//! Mirrors the Python `agent_framework_redis.RedisHistoryProvider`
//! (`_history_provider.py`): one Redis `LIST` of JSON-serialized messages per
//! conversation, with
//!
//! - **three connection sources** — a URL, explicit host settings
//!   (`host`/`port`/`ssl`/`username` plus an optional credential provider for
//!   Entra ID), or a *borrowed* connection the caller owns (upstream #8883) —
//!   exactly one of which must be supplied;
//! - **scoped keys** (the default, `key_format="scoped"`): the key carries
//!   independently encoded tenant, application, agent, provider-source and
//!   session segments, so two applications, tenants or agents sharing a
//!   Redis cannot read each other's history; and the deprecated
//!   **legacy** `{key_prefix}:{session_id}` format for migration;
//! - **retention** (`max_messages`, `0` = write nothing) and the
//!   `load_messages` / `store_inputs` / `store_outputs` switches of upstream's
//!   `HistoryProvider` base.
//!
//! [`RedisChatMessageStore`](crate::RedisChatMessageStore) remains for
//! existing callers: it is the legacy-format store with a URL-only
//! constructor, and its keys are unchanged.
//!
//! # Divergences
//!
//! - **Session-bound.** Upstream's provider is shared across sessions and
//!   reads the session id from each call. This port's
//!   [`ContextProvider::after_run`] receives no session, so — like every
//!   history provider in this workspace — one instance serves one
//!   conversation, whose id is fixed at build time
//!   ([`RedisHistoryProviderBuilder::session_id`]).
//! - **Key encoding.** Segments go through the core
//!   [`storage_key_segment`], which hex-encodes (upstream base32-encodes), so
//!   a scoped key is injective but not byte-identical to upstream's. The
//!   legacy format is encoded too (upstream joins it raw), keeping it
//!   injective; literal-safe ids — every UUID and the default prefix — are
//!   unchanged by the encoding.
//! - **`store_context_messages` / `store_context_from`** are not ported:
//!   [`ContextProvider::after_run`] does not receive the context messages
//!   other providers injected, so there is nothing for them to select.
//! - **`decode_responses` / client-type validation** has no counterpart: a
//!   borrowed [`MultiplexedConnection`] is always a standalone async client,
//!   and this provider decodes strings itself.

use std::sync::Arc;

use async_trait::async_trait;
use redis::aio::MultiplexedConnection;
use uuid::Uuid;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::history::{
    inject_stored_history_from, new_run_messages_from, HistoryProvider, StoredHistory,
};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::storage_keys::storage_key_segment;
use agent_framework_core::types::Message;

use crate::internal::{
    delete_key, list_len, ping, push_messages, read_messages, HostSettings, LazyConnection,
    StreamingCredentialsProvider,
};

/// Upstream's `RedisHistoryProvider.DEFAULT_SOURCE_ID`.
pub const DEFAULT_HISTORY_SOURCE_ID: &str = "redis_memory";

/// Upstream's default `key_prefix`.
pub const DEFAULT_HISTORY_KEY_PREFIX: &str = "chat_messages";

/// The literal segment marking a scoped key's layout version (upstream's
/// `"v2"`).
const SCOPED_KEY_VERSION: &str = "v2";
/// Upstream's `_ABSENT_SCOPE_SEGMENT`: an optional scope that was not given.
/// Starts with `~`, which no literal segment can, so it never collides with a
/// caller-supplied id.
const ABSENT_SCOPE_SEGMENT: &str = "~none";
const ENCODED_KEY_PREFIX: &str = "~redis-key-prefix-";
const ENCODED_TENANT_PREFIX: &str = "~redis-tenant-";
const ENCODED_APPLICATION_PREFIX: &str = "~redis-application-";
const ENCODED_AGENT_PREFIX: &str = "~redis-agent-";
const ENCODED_SOURCE_PREFIX: &str = "~redis-source-";
const ENCODED_SESSION_PREFIX: &str = "~redis-session-";

/// How [`RedisHistoryProvider`] lays out its Redis key. Mirrors upstream's
/// `key_format: Literal["scoped", "legacy"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedisKeyFormat {
    /// `{prefix}|v2|{tenant}|{application}|{agent}|{source}|{session}`, each
    /// segment independently encoded. Requires an application id.
    #[default]
    Scoped,
    /// `{prefix}:{session}` — the historical format, kept for migration.
    /// Deprecated upstream; building one logs a warning.
    Legacy,
}

/// Redis-backed history provider with scoped keys, mirroring upstream's
/// `RedisHistoryProvider`.
///
/// ```no_run
/// use agent_framework_redis::RedisHistoryProvider;
///
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let history = RedisHistoryProvider::builder()
///     .redis_url("redis://127.0.0.1:6379")
///     .application_id("support-bot")
///     .tenant_id("contoso")
///     .session_id("conversation-42")
///     .max_messages(200)
///     .build()?;
/// let stored = history.get_messages().await?;
/// # let _ = stored;
/// # Ok(())
/// # }
/// ```
pub struct RedisHistoryProvider {
    conn: LazyConnection,
    redis_url: Option<String>,
    session_id: String,
    source_id: String,
    key_prefix: String,
    tenant_id: Option<String>,
    application_id: Option<String>,
    agent_id: Option<String>,
    key_format: RedisKeyFormat,
    max_messages: Option<usize>,
    load_messages: bool,
    store_inputs: bool,
    store_outputs: bool,
}

impl std::fmt::Debug for RedisHistoryProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisHistoryProvider")
            .field("redis_key", &self.redis_key())
            .field("key_format", &self.key_format)
            .field("max_messages", &self.max_messages)
            .field("borrowed_connection", &self.conn.is_borrowed())
            .finish_non_exhaustive()
    }
}

/// Builder for [`RedisHistoryProvider`]; mirrors the upstream constructor's
/// keyword arguments.
#[derive(Default)]
pub struct RedisHistoryProviderBuilder {
    redis_url: Option<String>,
    host: HostSettings,
    connection: Option<MultiplexedConnection>,
    session_id: Option<String>,
    source_id: Option<String>,
    key_prefix: Option<String>,
    tenant_id: Option<String>,
    application_id: Option<String>,
    agent_id: Option<String>,
    key_format: RedisKeyFormat,
    max_messages: Option<usize>,
    load_messages: Option<bool>,
    store_inputs: Option<bool>,
    store_outputs: Option<bool>,
}

impl RedisHistoryProviderBuilder {
    /// Connect with a Redis URL (`redis://` or, with the `tls` feature,
    /// `rediss://`). Upstream's `redis_url`.
    pub fn redis_url(mut self, url: impl Into<String>) -> Self {
        self.redis_url = Some(url.into());
        self
    }

    /// Connect to an explicit host. Upstream's `host`; required with
    /// [`Self::credential_provider`].
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host.host = Some(host.into());
        self
    }

    /// Port for [`Self::host`]. Defaults to `6380`, the Azure Cache for Redis
    /// TLS port, as upstream.
    pub fn port(mut self, port: u16) -> Self {
        self.host.port = Some(port);
        self
    }

    /// TLS for [`Self::host`]. Defaults to `true`, as upstream; needs this
    /// crate's `tls` feature.
    pub fn ssl(mut self, ssl: bool) -> Self {
        self.host.ssl = Some(ssl);
        self
    }

    /// ACL username for [`Self::host`].
    pub fn username(mut self, username: impl Into<String>) -> Self {
        self.host.username = Some(username.into());
        self
    }

    /// Static password for [`Self::host`]. Not an upstream argument (redis-py
    /// reads it from the URL or the credential provider); offered so a host
    /// connection does not need a provider just to authenticate.
    pub fn password(mut self, password: impl Into<String>) -> Self {
        self.host.password = Some(password.into());
        self
    }

    /// Supply credentials from a (possibly rotating) provider — the Rust
    /// shape of redis-py's `CredentialProvider`, used for Microsoft Entra ID
    /// tokens. Requires [`Self::host`].
    pub fn credential_provider(mut self, provider: Arc<dyn StreamingCredentialsProvider>) -> Self {
        self.host.credential_provider = Some(provider);
        self
    }

    /// Use a caller-owned connection (upstream's borrowed `redis_client`,
    /// #8883). The provider never closes it.
    pub fn connection(mut self, connection: MultiplexedConnection) -> Self {
        self.connection = Some(connection);
        self
    }

    /// The conversation this provider serves. Defaults to a fresh
    /// `thread_{uuid}`.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Upstream's `source_id`; part of a scoped key. Defaults to
    /// [`DEFAULT_HISTORY_SOURCE_ID`].
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    /// Base key prefix. Defaults to [`DEFAULT_HISTORY_KEY_PREFIX`].
    pub fn key_prefix(mut self, key_prefix: impl Into<String>) -> Self {
        self.key_prefix = Some(key_prefix.into());
        self
    }

    /// Optional tenant boundary (scoped keys only; non-empty).
    pub fn tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Application boundary — required for scoped keys.
    pub fn application_id(mut self, application_id: impl Into<String>) -> Self {
        self.application_id = Some(application_id.into());
        self
    }

    /// Optional agent boundary (scoped keys only; non-empty).
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Key layout. Defaults to [`RedisKeyFormat::Scoped`].
    pub fn key_format(mut self, key_format: RedisKeyFormat) -> Self {
        self.key_format = key_format;
        self
    }

    /// Retain at most `max_messages` per conversation; `0` writes nothing at
    /// all (no payload reaches Redis, an AOF or a replica) and leaves stored
    /// history alone — [`RedisHistoryProvider::clear`] removes it.
    pub fn max_messages(mut self, max_messages: usize) -> Self {
        self.max_messages = Some(max_messages);
        self
    }

    /// Whether `before_run` loads stored history. Defaults to `true`.
    pub fn load_messages(mut self, load: bool) -> Self {
        self.load_messages = Some(load);
        self
    }

    /// Whether `after_run` stores the run's input messages. Defaults to
    /// `true`.
    pub fn store_inputs(mut self, store: bool) -> Self {
        self.store_inputs = Some(store);
        self
    }

    /// Whether `after_run` stores the run's response messages. Defaults to
    /// `true`.
    pub fn store_outputs(mut self, store: bool) -> Self {
        self.store_outputs = Some(store);
        self
    }

    /// Validate and build. No network I/O happens here.
    ///
    /// Errors (all [`Error::Configuration`]) mirror upstream's `ValueError`s:
    /// no or more than one connection source, a credential provider without a
    /// host, a scoped key without an application id, an empty tenant/agent
    /// id, scoped ids combined with the legacy format, or an empty session id.
    pub fn build(self) -> Result<RedisHistoryProvider> {
        let sources = usize::from(self.redis_url.is_some())
            + usize::from(self.host.is_set())
            + usize::from(self.connection.is_some());
        if sources == 0 {
            return Err(Error::Configuration(
                "either a connection, redis_url, or host/credential_provider must be provided"
                    .into(),
            ));
        }
        if sources > 1 {
            return Err(Error::Configuration(
                "connection, redis_url, and host/credential_provider are mutually exclusive".into(),
            ));
        }
        validate_scope(
            self.key_format,
            self.tenant_id.as_deref(),
            self.application_id.as_deref(),
            self.agent_id.as_deref(),
        )?;
        if self.key_format == RedisKeyFormat::Legacy {
            tracing::warn!(
                "RedisKeyFormat::Legacy is deprecated and will be removed in a future version; \
                 migrate persisted history to scoped keys"
            );
        }
        let session_id = self
            .session_id
            .unwrap_or_else(|| format!("thread_{}", Uuid::new_v4()));
        if session_id.is_empty() {
            return Err(Error::Configuration(
                "session_id must be a non-empty string".into(),
            ));
        }
        let conn = if let Some(connection) = self.connection {
            LazyConnection::borrowed(connection)
        } else if let Some(url) = &self.redis_url {
            LazyConnection::open(url)?
        } else {
            self.host.connect()?
        };
        Ok(RedisHistoryProvider {
            conn,
            redis_url: self.redis_url,
            session_id,
            source_id: self
                .source_id
                .unwrap_or_else(|| DEFAULT_HISTORY_SOURCE_ID.to_string()),
            key_prefix: self
                .key_prefix
                .unwrap_or_else(|| DEFAULT_HISTORY_KEY_PREFIX.to_string()),
            tenant_id: self.tenant_id,
            application_id: self.application_id,
            agent_id: self.agent_id,
            key_format: self.key_format,
            max_messages: self.max_messages,
            load_messages: self.load_messages.unwrap_or(true),
            store_inputs: self.store_inputs.unwrap_or(true),
            store_outputs: self.store_outputs.unwrap_or(true),
        })
    }
}

fn validate_scope(
    format: RedisKeyFormat,
    tenant_id: Option<&str>,
    application_id: Option<&str>,
    agent_id: Option<&str>,
) -> Result<()> {
    match format {
        RedisKeyFormat::Scoped => {
            if application_id.is_none_or(str::is_empty) {
                return Err(Error::Configuration(
                    "application_id must be a non-empty string when key_format is scoped".into(),
                ));
            }
            if tenant_id == Some("") {
                return Err(Error::Configuration(
                    "tenant_id must be non-empty when supplied".into(),
                ));
            }
            if agent_id == Some("") {
                return Err(Error::Configuration(
                    "agent_id must be non-empty when supplied".into(),
                ));
            }
        }
        RedisKeyFormat::Legacy => {
            if tenant_id.is_some() || application_id.is_some() || agent_id.is_some() {
                return Err(Error::Configuration(
                    "tenant_id, application_id, and agent_id cannot be used with the legacy key \
                     format"
                        .into(),
                ));
            }
        }
    }
    Ok(())
}

impl RedisHistoryProvider {
    /// Start building a provider.
    pub fn builder() -> RedisHistoryProviderBuilder {
        RedisHistoryProviderBuilder::default()
    }

    /// The conversation this provider serves.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Upstream's `source_id`.
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// The configured key prefix.
    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    /// The configured key format.
    pub fn key_format(&self) -> RedisKeyFormat {
        self.key_format
    }

    /// The tenant scope, if any.
    pub fn tenant_id(&self) -> Option<&str> {
        self.tenant_id.as_deref()
    }

    /// The application scope (always set for scoped keys).
    pub fn application_id(&self) -> Option<&str> {
        self.application_id.as_deref()
    }

    /// The agent scope, if any.
    pub fn agent_id(&self) -> Option<&str> {
        self.agent_id.as_deref()
    }

    /// The retention limit, if any.
    pub fn max_messages(&self) -> Option<usize> {
        self.max_messages
    }

    /// The Redis URL, when the provider was built from one.
    pub fn redis_url(&self) -> Option<&str> {
        self.redis_url.as_deref()
    }

    /// Whether the connection was supplied by (and is owned by) the caller.
    pub fn uses_borrowed_connection(&self) -> bool {
        self.conn.is_borrowed()
    }

    /// The Redis key holding this conversation. Mirrors upstream's
    /// `_redis_key`: a pipe-delimited scoped key or the colon-delimited legacy
    /// key, every caller-supplied segment rendered through
    /// [`storage_key_segment`] so no id can impersonate a separator.
    pub fn redis_key(&self) -> String {
        match self.key_format {
            RedisKeyFormat::Legacy => format!(
                "{}:{}",
                storage_key_segment(&self.key_prefix, "~p-"),
                storage_key_segment(&self.session_id, "~s-")
            ),
            RedisKeyFormat::Scoped => {
                let optional = |value: &Option<String>, prefix: &str| match value {
                    Some(v) => storage_key_segment(v, prefix),
                    None => ABSENT_SCOPE_SEGMENT.to_string(),
                };
                [
                    storage_key_segment(&self.key_prefix, ENCODED_KEY_PREFIX),
                    SCOPED_KEY_VERSION.to_string(),
                    optional(&self.tenant_id, ENCODED_TENANT_PREFIX),
                    storage_key_segment(
                        self.application_id.as_deref().unwrap_or_default(),
                        ENCODED_APPLICATION_PREFIX,
                    ),
                    optional(&self.agent_id, ENCODED_AGENT_PREFIX),
                    storage_key_segment(&self.source_id, ENCODED_SOURCE_PREFIX),
                    storage_key_segment(&self.session_id, ENCODED_SESSION_PREFIX),
                ]
                .join("|")
            }
        }
    }

    /// The stored messages, oldest first. Upstream's `get_messages`.
    pub async fn get_messages(&self) -> Result<Vec<Message>> {
        read_messages(&self.conn, &self.redis_key(), None, false).await
    }

    /// Persist `messages`, skipping any prefix that replays already-stored
    /// history, then trim to the retention limit. Upstream's `save_messages`
    /// (which filters through `filter_new_messages` the same way).
    pub async fn save_messages(&self, messages: Vec<Message>) -> Result<()> {
        self.store_run(&messages, &[]).await
    }

    /// Delete this conversation's history. Upstream's `clear`.
    pub async fn clear(&self) -> Result<()> {
        delete_key(&self.conn, &self.redis_key()).await
    }

    /// Number of stored messages (`LLEN`).
    pub async fn message_count(&self) -> Result<usize> {
        list_len(&self.conn, &self.redis_key()).await
    }

    /// `PING` the server.
    pub async fn ping(&self) -> bool {
        ping(&self.conn).await
    }

    /// Serialize the provider's *configuration* (not the messages, which Redis
    /// already holds). A borrowed connection cannot be serialized, so the
    /// `redis_url` is `null` for one; rebuild it with the same connection.
    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "redis_history_provider_state",
            "session_id": self.session_id,
            "source_id": self.source_id,
            "redis_url": self.redis_url,
            "key_prefix": self.key_prefix,
            "key_format": match self.key_format {
                RedisKeyFormat::Scoped => "scoped",
                RedisKeyFormat::Legacy => "legacy",
            },
            "tenant_id": self.tenant_id,
            "application_id": self.application_id,
            "agent_id": self.agent_id,
            "max_messages": self.max_messages,
            "load_messages": self.load_messages,
            "store_inputs": self.store_inputs,
            "store_outputs": self.store_outputs,
        })
    }

    /// Rebuild a URL-connected provider from [`Self::to_dict`] output.
    pub fn from_dict(state: &serde_json::Value) -> Result<Self> {
        let str_field = |name: &str| {
            state
                .get(name)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        };
        let bool_field = |name: &str| state.get(name).and_then(serde_json::Value::as_bool);
        let url = str_field("redis_url").ok_or_else(|| {
            Error::Configuration("state is missing 'redis_url' (borrowed connection?)".into())
        })?;
        let session_id = str_field("session_id")
            .ok_or_else(|| Error::Configuration("state is missing 'session_id'".into()))?;
        let mut builder = Self::builder().redis_url(url).session_id(session_id);
        if let Some(v) = str_field("source_id") {
            builder = builder.source_id(v);
        }
        if let Some(v) = str_field("key_prefix") {
            builder = builder.key_prefix(v);
        }
        builder = builder.key_format(match str_field("key_format").as_deref() {
            Some("legacy") => RedisKeyFormat::Legacy,
            Some("scoped") | None => RedisKeyFormat::Scoped,
            Some(other) => {
                return Err(Error::Configuration(format!(
                    "unknown key_format '{other}'"
                )))
            }
        });
        if let Some(v) = str_field("tenant_id") {
            builder = builder.tenant_id(v);
        }
        if let Some(v) = str_field("application_id") {
            builder = builder.application_id(v);
        }
        if let Some(v) = str_field("agent_id") {
            builder = builder.agent_id(v);
        }
        if let Some(v) = state
            .get("max_messages")
            .and_then(serde_json::Value::as_u64)
        {
            builder = builder.max_messages(usize::try_from(v).unwrap_or(usize::MAX));
        }
        if let Some(v) = bool_field("load_messages") {
            builder = builder.load_messages(v);
        }
        if let Some(v) = bool_field("store_inputs") {
            builder = builder.store_inputs(v);
        }
        if let Some(v) = bool_field("store_outputs") {
            builder = builder.store_outputs(v);
        }
        builder.build()
    }

    fn shape(&self, stored_len: usize) -> StoredHistory {
        match self.max_messages {
            Some(max) if stored_len >= max => StoredHistory::Window,
            _ => StoredHistory::Complete,
        }
    }

    async fn store_run(&self, request: &[Message], response: &[Message]) -> Result<()> {
        if (request.is_empty() && response.is_empty()) || self.max_messages == Some(0) {
            // Zero retention writes nothing — not even a read — so no payload
            // ever reaches Redis; stored history is left for `clear`.
            return Ok(());
        }
        let key = self.redis_key();
        let stored = read_messages(&self.conn, &key, None, false).await?;
        let new = new_run_messages_from(&stored, request, response, self.shape(stored.len()));
        push_messages(&self.conn, &key, &new, self.max_messages).await
    }
}

#[async_trait]
impl ContextProvider for RedisHistoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        if !self.load_messages {
            return Ok(());
        }
        let stored = self.get_messages().await?;
        let shape = self.shape(stored.len());
        inject_stored_history_from(ctx, stored, shape);
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
        let request: &[Message] = if self.store_inputs {
            request_messages
        } else {
            &[]
        };
        let response: &[Message] = if self.store_outputs {
            response_messages
        } else {
            &[]
        };
        self.store_run(request, response).await
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl HistoryProvider for RedisHistoryProvider {}

#[cfg(test)]
mod tests {
    use super::*;

    const URL: &str = "redis://127.0.0.1:6379/0";

    fn scoped() -> RedisHistoryProviderBuilder {
        RedisHistoryProvider::builder()
            .redis_url(URL)
            .application_id("app")
            .session_id("s1")
    }

    #[test]
    fn a_connection_source_is_required_and_exclusive() {
        let none = RedisHistoryProvider::builder()
            .application_id("app")
            .build();
        assert!(none.unwrap_err().to_string().contains("must be provided"));

        let two = scoped().host("h").ssl(false).build();
        assert!(two.unwrap_err().to_string().contains("mutually exclusive"));
    }

    #[test]
    fn a_credential_provider_needs_a_host() {
        struct Never;
        impl StreamingCredentialsProvider for Never {
            fn subscribe(
                &self,
            ) -> std::pin::Pin<
                Box<
                    dyn futures::Stream<Item = redis::RedisResult<redis::auth::BasicAuth>>
                        + Send
                        + 'static,
                >,
            > {
                Box::pin(futures::stream::empty())
            }
        }
        let err = RedisHistoryProvider::builder()
            .credential_provider(Arc::new(Never))
            .application_id("app")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("host is required"), "{err}");

        let ok = RedisHistoryProvider::builder()
            .credential_provider(Arc::new(Never))
            .host("127.0.0.1")
            .port(6379)
            .ssl(false)
            .application_id("app")
            .build();
        assert!(ok.is_ok());
    }

    #[test]
    fn host_settings_default_to_tls_on_6380() {
        // Without the `tls` feature that is refused up front rather than at
        // first use, so the default shape is observable without I/O.
        let result = RedisHistoryProvider::builder()
            .host("cache.example")
            .application_id("app")
            .build();
        if cfg!(feature = "tls") {
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().to_string().contains("tls"));
        }
    }

    #[test]
    fn scoped_keys_require_an_application_id() {
        let err = RedisHistoryProvider::builder()
            .redis_url(URL)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("application_id"), "{err}");
        let err = scoped().application_id("").build().unwrap_err();
        assert!(err.to_string().contains("application_id"), "{err}");
    }

    #[test]
    fn empty_optional_scopes_are_rejected() {
        assert!(scoped().tenant_id("").build().is_err());
        assert!(scoped().agent_id("").build().is_err());
    }

    #[test]
    fn legacy_keys_reject_scope_ids() {
        let err = RedisHistoryProvider::builder()
            .redis_url(URL)
            .key_format(RedisKeyFormat::Legacy)
            .tenant_id("t")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("legacy"), "{err}");
    }

    #[test]
    fn the_scoped_key_has_upstreams_layout() {
        let provider = scoped().build().unwrap();
        assert_eq!(
            provider.redis_key(),
            "chat_messages|v2|~none|app|~none|redis_memory|s1"
        );
        let provider = scoped()
            .tenant_id("t1")
            .agent_id("a1")
            .source_id("hist")
            .key_prefix("p")
            .build()
            .unwrap();
        assert_eq!(provider.redis_key(), "p|v2|t1|app|a1|hist|s1");
    }

    #[test]
    fn the_legacy_key_matches_the_chat_message_store() {
        let provider = RedisHistoryProvider::builder()
            .redis_url(URL)
            .key_format(RedisKeyFormat::Legacy)
            .session_id("a:b")
            .build()
            .unwrap();
        let store = crate::RedisChatMessageStore::new(URL, Some("a:b".into())).unwrap();
        assert_eq!(provider.redis_key(), store.redis_key());
    }

    #[test]
    fn scope_boundaries_are_injective() {
        // A separator inside one id must not shift another scope's boundary.
        let a = scoped().tenant_id("t|app").build().unwrap();
        let b = scoped()
            .tenant_id("t")
            .application_id("app|app")
            .build()
            .unwrap();
        assert_ne!(a.redis_key(), b.redis_key());
        // An absent tenant is not the tenant literally named "~none".
        let absent = scoped().build().unwrap();
        let named = scoped().tenant_id("~none").build().unwrap();
        assert_ne!(absent.redis_key(), named.redis_key());
        // Two applications never share a key.
        let other_app = scoped().application_id("app2").build().unwrap();
        assert_ne!(absent.redis_key(), other_app.redis_key());
        // Nor two agents, nor two sources.
        let agent = scoped().agent_id("x").build().unwrap();
        let source = scoped().source_id("x").build().unwrap();
        assert_ne!(agent.redis_key(), source.redis_key());
    }

    #[test]
    fn session_id_is_generated_and_non_empty() {
        let provider = RedisHistoryProvider::builder()
            .redis_url(URL)
            .application_id("app")
            .build()
            .unwrap();
        assert!(provider.session_id().starts_with("thread_"));
        assert!(scoped().session_id("").build().is_err());
    }

    #[test]
    fn defaults_mirror_upstream() {
        let p = scoped().build().unwrap();
        assert_eq!(p.source_id(), DEFAULT_HISTORY_SOURCE_ID);
        assert_eq!(p.key_prefix(), DEFAULT_HISTORY_KEY_PREFIX);
        assert_eq!(p.key_format(), RedisKeyFormat::Scoped);
        assert_eq!(p.max_messages(), None);
        assert!(p.load_messages && p.store_inputs && p.store_outputs);
        assert!(!p.uses_borrowed_connection());
        assert!(p.is_history_provider());
    }

    #[test]
    fn to_dict_round_trips() {
        let p = scoped()
            .tenant_id("t")
            .agent_id("a")
            .max_messages(5)
            .store_inputs(false)
            .build()
            .unwrap();
        let restored = RedisHistoryProvider::from_dict(&p.to_dict()).unwrap();
        assert_eq!(restored.redis_key(), p.redis_key());
        assert_eq!(restored.max_messages(), Some(5));
        assert!(!restored.store_inputs);
        assert!(restored.store_outputs);
    }

    #[tokio::test]
    async fn zero_retention_and_disabled_storage_perform_no_io() {
        // The URL points nowhere reachable; any I/O would error.
        let unreachable = "redis://127.0.0.1:1/0";
        let zero = RedisHistoryProvider::builder()
            .redis_url(unreachable)
            .application_id("app")
            .max_messages(0)
            .build()
            .unwrap();
        zero.after_run(&[Message::user("q")], &[Message::assistant("a")], None)
            .await
            .unwrap();

        let no_store = RedisHistoryProvider::builder()
            .redis_url(unreachable)
            .application_id("app")
            .store_inputs(false)
            .store_outputs(false)
            .load_messages(false)
            .build()
            .unwrap();
        no_store
            .after_run(&[Message::user("q")], &[Message::assistant("a")], None)
            .await
            .unwrap();
        let mut ctx = SessionContext::new(vec![Message::user("q")]);
        no_store.before_run(&mut ctx).await.unwrap();
        assert!(ctx.messages.is_empty());
    }
}
