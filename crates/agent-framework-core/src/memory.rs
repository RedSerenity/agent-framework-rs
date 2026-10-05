//! Context / memory providers.
//!
//! Rust equivalent of `agent_framework._memory`. A [`ContextProvider`] injects
//! extra instructions, messages, and tools into an agent invocation without
//! persisting them to the conversation history.
//!
//! Upstream renamed `ContextProvider.invoking`/`invoked` to `before_run`/
//! `after_run` and removed the `thread_created` hook entirely; `before_run`
//! mutates a [`SessionContext`] in place instead of returning a value. There
//! is no aggregate wrapper any more — consumers hold a
//! `Vec<Arc<dyn ContextProvider>>` and iterate it directly.

use async_trait::async_trait;

use crate::error::{Error, Result};
use crate::tools::ToolDefinition;
use crate::types::Message;

/// Per-invocation context a provider contributes to a run. Providers mutate
/// this in place in before_run. Rust equivalent of upstream SessionContext.
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    /// Local session identifier (from the thread), for provider scoping.
    pub session_id: Option<String>,
    /// Service-managed session/conversation id, when applicable.
    pub service_session_id: Option<String>,
    /// The run's input messages (read-only for providers).
    pub input_messages: Vec<Message>,
    /// Extra system instructions to inject (providers append via add_instructions).
    pub instructions: Option<String>,
    /// Extra context messages to inject ahead of history.
    pub messages: Vec<Message>,
    /// Extra tools to make available for this run.
    pub tools: Vec<ToolDefinition>,
}

impl SessionContext {
    pub fn new(input_messages: Vec<Message>) -> Self {
        Self {
            input_messages,
            ..Default::default()
        }
    }
    /// Append instructions, newline-concatenating with any already present.
    pub fn add_instructions(&mut self, s: impl Into<String>) {
        let s = s.into();
        self.instructions = match self.instructions.take() {
            Some(existing) => Some(format!("{existing}\n{s}")),
            None => Some(s),
        };
    }
}

/// A source of per-invocation context (memory, RAG, etc.).
/// Upstream renamed invoking/invoked -> before_run/after_run and REMOVED
/// thread_created. before_run mutates the SessionContext in place instead of
/// returning a Context.
#[async_trait]
pub trait ContextProvider: Send + Sync {
    /// Called before the model is invoked; mutate ctx to inject instructions,
    /// messages, and/or tools. Read ctx.input_messages / ctx.session_id.
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()>;

    /// Called after an invocation completes, on BOTH success and failure.
    /// On success, error is None and response_messages holds the output.
    /// On failure, error is Some and response_messages is empty.
    async fn after_run(
        &self,
        _request_messages: &[Message],
        _response_messages: &[Message],
        _error: Option<&Error>,
    ) -> Result<()> {
        Ok(())
    }

    /// Whether this provider manages conversation history (a
    /// [`HistoryProvider`](crate::history::HistoryProvider)). [`Agent`](crate::agent::Agent)
    /// and [`WorkflowAgent`](crate::workflow::WorkflowAgent) use this to
    /// detect an already-attached history provider among a session's
    /// `context_providers` and avoid auto-attaching a redundant
    /// [`InMemoryHistoryProvider`](crate::history::InMemoryHistoryProvider).
    /// Defaults to `false`; history providers override it to `true`.
    fn is_history_provider(&self) -> bool {
        false
    }

    /// The conversation history this provider holds **in process**, for a
    /// [`SessionStore`](crate::session_store::SessionStore) to snapshot.
    ///
    /// Upstream keeps in-memory history in `session.state`, so persisting a
    /// session persists its history for free. Here an
    /// [`InMemoryHistoryProvider`](crate::history::InMemoryHistoryProvider)
    /// owns it instead, so the store asks for it explicitly. Only a provider
    /// whose history would otherwise be lost with the process returns
    /// `Some`; one backed by external storage (a file, Redis, a database)
    /// returns `None`, because its history already outlives the session.
    fn history_snapshot(&self) -> Option<Vec<Message>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_instructions_sets_when_none() {
        let mut ctx = SessionContext::new(vec![]);
        assert!(ctx.instructions.is_none());
        ctx.add_instructions("be brief");
        assert_eq!(ctx.instructions.as_deref(), Some("be brief"));
    }

    #[test]
    fn add_instructions_newline_concatenates() {
        let mut ctx = SessionContext::new(vec![]);
        ctx.add_instructions("first");
        ctx.add_instructions("second");
        ctx.add_instructions("third");
        assert_eq!(ctx.instructions.as_deref(), Some("first\nsecond\nthird"));
    }

    #[test]
    fn new_sets_input_messages_and_defaults_rest() {
        let messages = vec![Message::user("hi")];
        let ctx = SessionContext::new(messages.clone());
        assert_eq!(ctx.input_messages.len(), messages.len());
        assert_eq!(ctx.input_messages[0].text(), "hi");
        assert!(ctx.session_id.is_none());
        assert!(ctx.service_session_id.is_none());
        assert!(ctx.instructions.is_none());
        assert!(ctx.messages.is_empty());
        assert!(ctx.tools.is_empty());
    }
}
