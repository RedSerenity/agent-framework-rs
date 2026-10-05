//! [`ValkeyChatHistoryProvider`]: chat history in a Valkey (or Redis) list.
//!
//! Ports .NET's `Microsoft.Agents.AI.Valkey.ValkeyChatHistoryProvider`.
//! [Valkey](https://valkey.io) is the Linux Foundation fork of Redis and
//! speaks the same RESP protocol, so this provider reuses this crate's Redis
//! client (`redis-rs`) rather than a separate Valkey SDK (.NET uses
//! `Valkey.Glide`). It needs only list commands — no search module — so it
//! works against any Valkey or Redis server.
//!
//! Behavior carried over from .NET:
//!
//! - key `{key_prefix}:{conversation_id}`, `key_prefix` defaulting to
//!   [`DEFAULT_VALKEY_KEY_PREFIX`] (`"chat_history"`);
//! - `max_messages` must be greater than zero and trims the list to the most
//!   recent N after every write (`LTRIM -N -1`);
//! - `max_messages_to_retrieve` fetches only the tail (`LRANGE -N -1`), and
//!   `0` retrieves nothing — without a round trip, since `LRANGE key -0 -1`
//!   would return the whole list;
//! - a malformed stored entry is skipped with a warning instead of failing
//!   the run;
//! - [`ValkeyChatHistoryProvider::clear_messages`] and
//!   [`ValkeyChatHistoryProvider::message_count`];
//! - the conversation id is the provider's per-session state and must not be
//!   blank.
//!
//! # Divergences
//!
//! - **Session-bound.** .NET resolves the conversation id per session through
//!   a `stateInitializer`; this port's providers are one per conversation, so
//!   the id is given to [`ValkeyChatHistoryProvider::builder`].
//! - **Key encoding.** The two halves of the key go through the core
//!   [`storage_key_segment`] so a `:` inside an id cannot make two
//!   conversations share a list. Literal-safe ids (lowercase alphanumerics,
//!   `.`, `_`, `-` — every UUID) produce .NET's exact key.
//! - **Replay de-duplication.** .NET appends every request and response
//!   message; like every history provider in this workspace this one skips a
//!   prefix of the input that replays already-stored history, so a caller
//!   that resends its transcript does not duplicate it.
//! - The message filters (`ProvideOutputMessageFilter`, …) and the JSON
//!   serializer options are not ported; messages use the workspace's serde
//!   representation.

use async_trait::async_trait;
use redis::aio::MultiplexedConnection;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::history::{
    inject_stored_history_from, new_run_messages_from, HistoryProvider, StoredHistory,
};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::storage_keys::storage_key_segment;
use agent_framework_core::types::Message;

use crate::internal::{delete_key, list_len, ping, push_messages, read_messages, LazyConnection};

/// .NET's default `KeyPrefix`.
pub const DEFAULT_VALKEY_KEY_PREFIX: &str = "chat_history";

/// Valkey-backed history provider; see the [module docs](self).
///
/// ```no_run
/// use agent_framework_redis::ValkeyChatHistoryProvider;
///
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let history = ValkeyChatHistoryProvider::builder("conversation-42")
///     .url("redis://127.0.0.1:6379")
///     .max_messages(100)
///     .max_messages_to_retrieve(20)
///     .build()?;
/// println!("{} stored", history.message_count().await?);
/// # Ok(())
/// # }
/// ```
pub struct ValkeyChatHistoryProvider {
    conn: LazyConnection,
    url: Option<String>,
    conversation_id: String,
    key_prefix: String,
    max_messages: Option<usize>,
    max_messages_to_retrieve: Option<usize>,
}

impl std::fmt::Debug for ValkeyChatHistoryProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValkeyChatHistoryProvider")
            .field("key", &self.key())
            .field("max_messages", &self.max_messages)
            .field("max_messages_to_retrieve", &self.max_messages_to_retrieve)
            .finish_non_exhaustive()
    }
}

/// Builder for [`ValkeyChatHistoryProvider`]; mirrors
/// `ValkeyChatHistoryProviderOptions`.
pub struct ValkeyChatHistoryProviderBuilder {
    conversation_id: String,
    url: Option<String>,
    connection: Option<MultiplexedConnection>,
    key_prefix: Option<String>,
    max_messages: Option<usize>,
    max_messages_to_retrieve: Option<usize>,
}

impl ValkeyChatHistoryProviderBuilder {
    /// Connect with a URL. Valkey accepts the `redis://` (and, with the `tls`
    /// feature, `rediss://`) schemes; `valkey://` is not understood by the
    /// client, so use `redis://`.
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Use a caller-owned connection (.NET's `IConnectionMultiplexer`). Never
    /// closed by the provider.
    pub fn connection(mut self, connection: MultiplexedConnection) -> Self {
        self.connection = Some(connection);
        self
    }

    /// Key prefix. Defaults to [`DEFAULT_VALKEY_KEY_PREFIX`].
    pub fn key_prefix(mut self, key_prefix: impl Into<String>) -> Self {
        self.key_prefix = Some(key_prefix.into());
        self
    }

    /// Retain at most this many messages per conversation. Must be greater
    /// than zero; unset means unlimited.
    pub fn max_messages(mut self, max_messages: usize) -> Self {
        self.max_messages = Some(max_messages);
        self
    }

    /// Load at most this many (the most recent) messages; `0` loads nothing.
    pub fn max_messages_to_retrieve(mut self, max: usize) -> Self {
        self.max_messages_to_retrieve = Some(max);
        self
    }

    /// Validate and build. No I/O.
    pub fn build(self) -> Result<ValkeyChatHistoryProvider> {
        if self.conversation_id.trim().is_empty() {
            return Err(Error::Configuration(
                "conversation_id must not be null, empty, or whitespace".into(),
            ));
        }
        if self.max_messages == Some(0) {
            return Err(Error::Configuration(
                "max_messages must be greater than zero".into(),
            ));
        }
        let conn = match (self.connection, &self.url) {
            (Some(_), Some(_)) => {
                return Err(Error::Configuration(
                    "connection and url are mutually exclusive".into(),
                ))
            }
            (Some(connection), None) => LazyConnection::borrowed(connection),
            (None, Some(url)) => LazyConnection::open(url)?,
            (None, None) => {
                return Err(Error::Configuration(
                    "either a connection or a url must be provided".into(),
                ))
            }
        };
        Ok(ValkeyChatHistoryProvider {
            conn,
            url: self.url,
            conversation_id: self.conversation_id,
            key_prefix: self
                .key_prefix
                .unwrap_or_else(|| DEFAULT_VALKEY_KEY_PREFIX.to_string()),
            max_messages: self.max_messages,
            max_messages_to_retrieve: self.max_messages_to_retrieve,
        })
    }
}

impl ValkeyChatHistoryProvider {
    /// Start building a provider for `conversation_id` (.NET's
    /// `State.ConversationId`).
    pub fn builder(conversation_id: impl Into<String>) -> ValkeyChatHistoryProviderBuilder {
        ValkeyChatHistoryProviderBuilder {
            conversation_id: conversation_id.into(),
            url: None,
            connection: None,
            key_prefix: None,
            max_messages: None,
            max_messages_to_retrieve: None,
        }
    }

    /// The conversation id.
    pub fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    /// The key prefix.
    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    /// The retention limit.
    pub fn max_messages(&self) -> Option<usize> {
        self.max_messages
    }

    /// The retrieval limit.
    pub fn max_messages_to_retrieve(&self) -> Option<usize> {
        self.max_messages_to_retrieve
    }

    /// The list key: `{key_prefix}:{conversation_id}`, each half encoded with
    /// [`storage_key_segment`] (a no-op for literal-safe values).
    pub fn key(&self) -> String {
        format!(
            "{}:{}",
            storage_key_segment(&self.key_prefix, "~p-"),
            storage_key_segment(&self.conversation_id, "~c-")
        )
    }

    /// The history `before_run` would load: the stored messages (the last
    /// `max_messages_to_retrieve` when set), malformed entries skipped.
    pub async fn get_messages(&self) -> Result<Vec<Message>> {
        read_messages(&self.conn, &self.key(), self.max_messages_to_retrieve, true).await
    }

    /// Delete this conversation's history. .NET's `ClearMessagesAsync`.
    pub async fn clear_messages(&self) -> Result<()> {
        delete_key(&self.conn, &self.key()).await
    }

    /// Number of stored messages. .NET's `GetMessageCountAsync`.
    pub async fn message_count(&self) -> Result<usize> {
        list_len(&self.conn, &self.key()).await
    }

    /// `PING` the server.
    pub async fn ping(&self) -> bool {
        ping(&self.conn).await
    }

    /// The provider's per-session state, shaped like .NET's serialized
    /// `State` (`{"conversationId": …}`) plus the connection settings needed
    /// to rebuild it with [`Self::from_dict`].
    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "valkey_chat_history_state",
            "conversationId": self.conversation_id,
            "url": self.url,
            "key_prefix": self.key_prefix,
            "max_messages": self.max_messages,
            "max_messages_to_retrieve": self.max_messages_to_retrieve,
        })
    }

    /// Rebuild a URL-connected provider from [`Self::to_dict`] output.
    pub fn from_dict(state: &serde_json::Value) -> Result<Self> {
        let get_str = |name: &str| state.get(name).and_then(serde_json::Value::as_str);
        let get_usize = |name: &str| {
            state
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .map(|v| usize::try_from(v).unwrap_or(usize::MAX))
        };
        let conversation_id = get_str("conversationId")
            .ok_or_else(|| Error::Configuration("state is missing 'conversationId'".into()))?;
        let url =
            get_str("url").ok_or_else(|| Error::Configuration("state is missing 'url'".into()))?;
        let mut builder = Self::builder(conversation_id).url(url);
        if let Some(prefix) = get_str("key_prefix") {
            builder = builder.key_prefix(prefix);
        }
        if let Some(max) = get_usize("max_messages") {
            builder = builder.max_messages(max);
        }
        if let Some(max) = get_usize("max_messages_to_retrieve") {
            builder = builder.max_messages_to_retrieve(max);
        }
        builder.build()
    }
}

#[async_trait]
impl ContextProvider for ValkeyChatHistoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let stored = self.get_messages().await?;
        let len = stored.len();
        let cap = match (self.max_messages, self.max_messages_to_retrieve) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let shape = match cap {
            Some(cap) if len >= cap => StoredHistory::Window,
            _ => StoredHistory::Complete,
        };
        inject_stored_history_from(ctx, stored, shape);
        tracing::debug!(count = len, "ValkeyChatHistoryProvider: retrieved messages");
        Ok(())
    }

    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_some() || (request_messages.is_empty() && response_messages.is_empty()) {
            return Ok(());
        }
        let key = self.key();
        // The whole list, not the retrieval window: de-duplication must see
        // everything that is stored.
        let stored = read_messages(&self.conn, &key, None, true).await?;
        let shape = match self.max_messages {
            Some(max) if stored.len() >= max => StoredHistory::Window,
            _ => StoredHistory::Complete,
        };
        let new = new_run_messages_from(&stored, request_messages, response_messages, shape);
        push_messages(&self.conn, &key, &new, self.max_messages).await?;
        tracing::debug!(
            count = new.len(),
            "ValkeyChatHistoryProvider: stored messages"
        );
        Ok(())
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl HistoryProvider for ValkeyChatHistoryProvider {}

#[cfg(test)]
mod tests {
    use super::*;

    const URL: &str = "redis://127.0.0.1:6379/0";

    #[test]
    fn defaults_and_key_match_dotnet() {
        let p = ValkeyChatHistoryProvider::builder("conv-1")
            .url(URL)
            .build()
            .unwrap();
        assert_eq!(p.key_prefix(), "chat_history");
        assert_eq!(p.key(), "chat_history:conv-1");
        assert_eq!(p.max_messages(), None);
        assert_eq!(p.max_messages_to_retrieve(), None);
        assert!(p.is_history_provider());
    }

    #[test]
    fn blank_conversation_ids_are_rejected() {
        for id in ["", "   "] {
            assert!(ValkeyChatHistoryProvider::builder(id)
                .url(URL)
                .build()
                .is_err());
        }
    }

    #[test]
    fn max_messages_must_be_positive() {
        let err = ValkeyChatHistoryProvider::builder("c")
            .url(URL)
            .max_messages(0)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("greater than zero"), "{err}");
        // Zero is valid for retrieval: it means "load nothing".
        assert!(ValkeyChatHistoryProvider::builder("c")
            .url(URL)
            .max_messages_to_retrieve(0)
            .build()
            .is_ok());
    }

    #[test]
    fn exactly_one_connection_source() {
        assert!(ValkeyChatHistoryProvider::builder("c").build().is_err());
    }

    #[test]
    fn separators_in_ids_cannot_collide() {
        let a = ValkeyChatHistoryProvider::builder("a:b")
            .url(URL)
            .key_prefix("p")
            .build()
            .unwrap();
        let b = ValkeyChatHistoryProvider::builder("b")
            .url(URL)
            .key_prefix("p:a")
            .build()
            .unwrap();
        assert_ne!(a.key(), b.key());
    }

    #[test]
    fn state_round_trips() {
        let p = ValkeyChatHistoryProvider::builder("conv")
            .url(URL)
            .key_prefix("kp")
            .max_messages(10)
            .max_messages_to_retrieve(3)
            .build()
            .unwrap();
        let state = p.to_dict();
        assert_eq!(state["conversationId"], "conv");
        let back = ValkeyChatHistoryProvider::from_dict(&state).unwrap();
        assert_eq!(back.key(), p.key());
        assert_eq!(back.max_messages(), Some(10));
        assert_eq!(back.max_messages_to_retrieve(), Some(3));
    }

    #[tokio::test]
    async fn zero_retrieval_reads_nothing_without_io() {
        let p = ValkeyChatHistoryProvider::builder("c")
            .url("redis://127.0.0.1:1/0")
            .max_messages_to_retrieve(0)
            .build()
            .unwrap();
        let mut ctx = SessionContext::new(vec![Message::user("hi")]);
        p.before_run(&mut ctx).await.unwrap();
        assert!(ctx.messages.is_empty());
    }
}
