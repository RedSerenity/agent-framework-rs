//! Conversation history kept in session state.
//!
//! [`SessionStateHistoryProvider`] is the harness's default history provider.
//! It mirrors upstream Python's `InMemoryHistoryProvider`, which stores a
//! session's messages in `session.state[source_id]["messages"]` (default
//! source id `"in_memory"`). Core's
//! [`InMemoryHistoryProvider`](agent_framework_core::history::InMemoryHistoryProvider)
//! instead keeps them in an `Arc` attached to one session; keeping them in
//! the state bag is what lets the harness's
//! [`LoopAgent`](crate::loop_agent::LoopAgent) `fresh_context` rollback reset
//! the transcript along with the rest of the session — upstream's behavior —
//! and lets one provider instance serve every session, so the harness can
//! attach it to the agent once.
//!
//! It also hosts the **after-run compaction** phase of upstream
//! `create_harness_agent` (`after_compaction_strategy`, run by a
//! `CompactionProvider.after_run` there): after storing a run, the stored
//! transcript is compacted in place.

use std::sync::Arc;

use agent_framework_core::compaction::{compact, ApproxTokenizer, CompactionStrategy, Tokenizer};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::history::{inject_stored_history, new_run_messages, HistoryProvider};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::session::{AgentSession, SessionState};
use agent_framework_core::types::Message;
use async_trait::async_trait;
use serde_json::{Map, Value};

/// Default source id of [`SessionStateHistoryProvider`]. Mirrors upstream
/// `InMemoryHistoryProvider.DEFAULT_SOURCE_ID`.
pub const DEFAULT_HISTORY_SOURCE_ID: &str = "in_memory";

/// A [`HistoryProvider`] storing each session's transcript in its state bag.
#[derive(Clone)]
pub struct SessionStateHistoryProvider {
    source_id: String,
    after_compaction: Option<Arc<dyn CompactionStrategy>>,
    tokenizer: Arc<dyn Tokenizer>,
}

impl std::fmt::Debug for SessionStateHistoryProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionStateHistoryProvider")
            .field("source_id", &self.source_id)
            .field("after_compaction", &self.after_compaction.is_some())
            .finish()
    }
}

impl Default for SessionStateHistoryProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStateHistoryProvider {
    /// A provider under the default source id.
    pub fn new() -> Self {
        Self {
            source_id: DEFAULT_HISTORY_SOURCE_ID.into(),
            after_compaction: None,
            tokenizer: Arc::new(ApproxTokenizer),
        }
    }

    /// Override the source id (state key).
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// Compact the stored transcript with `strategy` after every run.
    pub fn after_compaction(mut self, strategy: Arc<dyn CompactionStrategy>) -> Self {
        self.after_compaction = Some(strategy);
        self
    }

    /// The tokenizer used by the after-run compaction.
    pub fn tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// The configured source id.
    pub fn get_source_id(&self) -> &str {
        &self.source_id
    }

    /// The messages stored for the session owning `state`.
    pub fn messages(&self, state: &SessionState) -> Result<Vec<Message>> {
        match state.get(&self.source_id) {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Object(map)) => match map.get("messages") {
                None | Some(Value::Null) => Ok(Vec::new()),
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| Error::Serialization(format!("invalid stored history: {e}"))),
            },
            Some(_) => Err(Error::Serialization(format!(
                "Session state for source_id '{}' must be a dict.",
                self.source_id
            ))),
        }
    }

    /// Replace the messages stored for the session owning `state`.
    pub fn save_messages(&self, state: &SessionState, messages: &[Message]) -> Result<()> {
        let mut map = match state.get(&self.source_id) {
            Some(Value::Object(map)) => map,
            _ => Map::new(),
        };
        map.insert(
            "messages".into(),
            serde_json::to_value(messages).map_err(|e| Error::Serialization(e.to_string()))?,
        );
        state.insert(self.source_id.clone(), Value::Object(map));
        Ok(())
    }
}

#[async_trait]
impl ContextProvider for SessionStateHistoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let Some(state) = ctx.session_state.clone() else {
            return Ok(());
        };
        let stored = self.messages(&state)?;
        inject_stored_history(ctx, stored);
        Ok(())
    }

    async fn after_run_in_session(
        &self,
        session: &AgentSession,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_some() {
            return Ok(());
        }
        let mut stored = self.messages(&session.state)?;
        let new = new_run_messages(&stored, request_messages, response_messages);
        stored.extend(new);
        if let Some(strategy) = &self.after_compaction {
            stored = compact(&stored, strategy.as_ref(), self.tokenizer.as_ref());
        }
        self.save_messages(&session.state, &stored)
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl HistoryProvider for SessionStateHistoryProvider {}
