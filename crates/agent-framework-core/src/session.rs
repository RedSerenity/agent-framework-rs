//! Conversation sessions.
//!
//! Rust equivalent of upstream's `_sessions.py`. Upstream deleted
//! `_threads.py` and `_memory.py`, consolidating a lightweight `AgentSession`
//! (`{session_id, service_session_id, state}`) with memory/context providers.
//! Conversation history left the session entirely: it is now injected by a
//! [`HistoryProvider`](crate::history::HistoryProvider) — just another
//! [`ContextProvider`] — via `before_run`/`after_run`, instead of being owned
//! by a message store on the thread/session itself.
//!
//! A session is no longer "service-managed XOR locally-stored": it always
//! carries a `session_id`, optionally a `service_session_id` (when the
//! underlying service manages the conversation server-side), a bag of
//! free-form `state`, and the `context_providers` that run on every use.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::memory::ContextProvider;

/// The free-form state bag of an [`AgentSession`], shared **by reference**
/// across clones.
///
/// Upstream's `AgentSession.state` is a plain Python dict, and Python
/// sessions are reference types: every holder of the session sees the same
/// dict, including the *child* session `as_tool(propagate_session=True)`
/// hands a sub-agent (upstream shares the dict by reference while isolating
/// `service_session_id`). A by-value `HashMap` cannot express that, so the
/// state bag is an `Arc<Mutex<..>>` handle: cloning an [`AgentSession`]
/// (or a `SessionState`) yields a view onto the *same* bag, and mutations
/// through any clone are visible to all of them.
#[derive(Clone, Default)]
pub struct SessionState {
    inner: Arc<Mutex<HashMap<String, Value>>>,
}

impl SessionState {
    /// A fresh, empty state bag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `value` at `key`, returning the previous value if any.
    pub fn insert(&self, key: impl Into<String>, value: Value) -> Option<Value> {
        self.inner.lock().unwrap().insert(key.into(), value)
    }

    /// A clone of the value at `key`, if present.
    pub fn get(&self, key: &str) -> Option<Value> {
        self.inner.lock().unwrap().get(key).cloned()
    }

    /// Remove and return the value at `key`, if present.
    pub fn remove(&self, key: &str) -> Option<Value> {
        self.inner.lock().unwrap().remove(key)
    }

    /// Whether `key` is present.
    pub fn contains_key(&self, key: &str) -> bool {
        self.inner.lock().unwrap().contains_key(key)
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Whether the bag is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    /// A point-in-time copy of the whole bag (e.g. for serialization).
    pub fn snapshot(&self) -> HashMap<String, Value> {
        self.inner.lock().unwrap().clone()
    }

    /// Whether `other` is a view onto the same underlying bag.
    pub fn shares_storage_with(&self, other: &SessionState) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl From<HashMap<String, Value>> for SessionState {
    fn from(map: HashMap<String, Value>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(map)),
        }
    }
}

impl std::fmt::Debug for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.snapshot()).finish()
    }
}

/// A conversation session: a lightweight identity + state container.
///
/// Message history is **not** stored here any more — see
/// [`crate::history::HistoryProvider`] and
/// [`crate::history::ensure_history_provider`].
#[derive(Clone)]
pub struct AgentSession {
    session_id: String,
    service_session_id: Option<String>,
    /// Free-form session state (for context providers to persist per-session
    /// data across runs). Shared by reference across clones of this session —
    /// see [`SessionState`].
    pub state: SessionState,
    /// Context providers associated with this session (memory/RAG/history
    /// injection). Combined with an agent's own providers at request time —
    /// see [`Agent::combined_providers`](crate::agent::Agent).
    pub context_providers: Vec<Arc<dyn ContextProvider>>,
}

impl std::fmt::Debug for AgentSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSession")
            .field("session_id", &self.session_id)
            .field("service_session_id", &self.service_session_id)
            .field("state", &self.state)
            .field("context_providers", &self.context_providers.len())
            .finish()
    }
}

impl Default for AgentSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentSession {
    /// A fresh, local (non-service-managed) session with a newly generated
    /// `session_id`.
    pub fn new() -> Self {
        Self {
            session_id: Uuid::new_v4().to_string(),
            service_session_id: None,
            state: SessionState::new(),
            context_providers: Vec::new(),
        }
    }

    /// A **child** session for delegating part of this conversation to a
    /// sub-agent (the `as_tool(propagate_session)` path).
    ///
    /// The child keeps this session's `session_id` and shares its [`state`]
    /// bag by reference, but gets an **isolated** (cleared)
    /// `service_session_id` and no context providers. Isolating the
    /// server-side conversation pointer matters: after the parent's first
    /// model call, a service-managed session carries the parent conversation
    /// id (e.g. an OpenAI Responses `previous_response_id`); a child that
    /// inherited it would submit a follow-up onto a conversation whose
    /// tool call is still pending, which the server rejects. Mirrors
    /// upstream's `_agent_wrapper` child-session construction
    /// (`_agents.py`, microsoft/agent-framework#5875).
    ///
    /// [`state`]: AgentSession::state
    pub fn child(&self) -> AgentSession {
        AgentSession {
            session_id: self.session_id.clone(),
            service_session_id: None,
            state: self.state.clone(),
            context_providers: Vec::new(),
        }
    }

    /// A service-managed session identified by a conversation id.
    pub fn service(id: impl Into<String>) -> Self {
        Self {
            service_session_id: Some(id.into()),
            ..Self::new()
        }
    }

    /// Replace this session's local identifier.
    ///
    /// Hosting code keys sessions by an id the *protocol* chooses (an OpenAI
    /// Responses `resp_*` or `conv_*` id, say), so a session created for one
    /// takes that id rather than a random one. Mirrors upstream's
    /// `create_session(session_id=...)`.
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = id.into();
        self
    }

    /// Attach context providers to this session, replacing any previously set.
    pub fn with_context_providers(mut self, providers: Vec<Arc<dyn ContextProvider>>) -> Self {
        self.context_providers = providers;
        self
    }

    /// This session's local identifier.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The service-side conversation id, if any.
    pub fn service_session_id(&self) -> Option<&str> {
        self.service_session_id.as_deref()
    }

    /// Set the service-side conversation id explicitly.
    pub fn set_service_session_id(&mut self, id: impl Into<String>) {
        self.service_session_id = Some(id.into());
    }

    /// Adopt a service-managed conversation id returned by the chat service
    /// (e.g. an OpenAI Responses `previous_response_id` or an Azure AI thread
    /// id), so follow-up runs continue the same service conversation.
    ///
    /// Returns `true` when the id was newly adopted (it differed from what
    /// the session already carried), `false` when it was already current.
    pub fn try_adopt_service_session_id(&mut self, id: &str) -> bool {
        if self.service_session_id.as_deref() == Some(id) {
            return false;
        }
        self.service_session_id = Some(id.to_string());
        true
    }

    /// Serialize this session's `{session_id, service_session_id, state}` to
    /// JSON. Conversation history is deliberately **not** included — it lives
    /// in whichever [`HistoryProvider`](crate::history::HistoryProvider), if
    /// any, is attached to `context_providers`; serialize that separately
    /// (e.g. [`InMemoryHistoryProvider::to_dict`](crate::history::InMemoryHistoryProvider::to_dict)).
    pub fn to_dict(&self) -> Value {
        serde_json::json!({
            "session_id": self.session_id,
            "service_session_id": self.service_session_id,
            "state": self.state.snapshot(),
        })
    }

    /// Reconstruct a session from state produced by [`AgentSession::to_dict`].
    ///
    /// `context_providers` are **not** restored by this call — callers
    /// reattach their own (including any `HistoryProvider`, whose own state
    /// is serialized/restored independently).
    pub fn from_dict(state: &Value) -> Result<Self> {
        let session_id = state
            .get("session_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let service_session_id = state
            .get("service_session_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let session_state: HashMap<String, Value> = match state.get("state") {
            Some(v) if !v.is_null() => serde_json::from_value(v.clone()).map_err(|e| {
                Error::Serialization(format!("failed to restore session state: {e}"))
            })?,
            _ => HashMap::new(),
        };
        Ok(Self {
            session_id,
            service_session_id,
            state: SessionState::from(session_state),
            context_providers: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_session_has_a_generated_id_and_no_service_id() {
        let session = AgentSession::new();
        assert!(!session.session_id().is_empty());
        assert!(session.service_session_id().is_none());
        assert!(session.state.is_empty());
    }

    #[test]
    fn service_session_sets_service_session_id_and_still_has_a_local_id() {
        let session = AgentSession::service("svc-1");
        assert_eq!(session.service_session_id(), Some("svc-1"));
        assert!(!session.session_id().is_empty());
    }

    #[test]
    fn try_adopt_service_session_id_reports_whether_it_was_new() {
        let mut session = AgentSession::new();
        assert!(session.try_adopt_service_session_id("conv-1"));
        assert_eq!(session.service_session_id(), Some("conv-1"));
        // Adopting the same id again is a no-op, reported as such.
        assert!(!session.try_adopt_service_session_id("conv-1"));
        // A different id is adopted (and reported as newly adopted).
        assert!(session.try_adopt_service_session_id("conv-2"));
        assert_eq!(session.service_session_id(), Some("conv-2"));
    }

    #[test]
    fn to_dict_from_dict_round_trips_session_id_service_id_and_state() {
        let session = AgentSession::service("svc-9");
        session.state.insert("key", serde_json::json!("value"));
        let original_id = session.session_id().to_string();

        let state = session.to_dict();
        assert_eq!(state["session_id"], original_id);
        assert_eq!(state["service_session_id"], "svc-9");
        assert_eq!(state["state"]["key"], "value");
        // History is deliberately absent from the wire shape.
        assert!(state.get("messages").is_none());
        assert!(state.get("chat_message_store_state").is_none());

        let restored = AgentSession::from_dict(&state).unwrap();
        assert_eq!(restored.session_id(), original_id);
        assert_eq!(restored.service_session_id(), Some("svc-9"));
        assert_eq!(restored.state.get("key"), Some(serde_json::json!("value")));
        assert!(restored.context_providers.is_empty());
    }

    #[test]
    fn clones_share_the_state_bag_by_reference() {
        let session = AgentSession::new();
        let clone = session.clone();
        clone
            .state
            .insert("written-via-clone", serde_json::json!(1));
        assert_eq!(
            session.state.get("written-via-clone"),
            Some(serde_json::json!(1)),
            "a clone must be a view onto the same state bag"
        );
        assert!(session.state.shares_storage_with(&clone.state));
    }

    #[test]
    fn child_shares_id_and_state_but_isolates_the_service_pointer() {
        let parent = AgentSession::service("svc-parent");
        parent.state.insert("k", serde_json::json!("v"));

        let child = parent.child();
        assert_eq!(child.session_id(), parent.session_id());
        assert_eq!(
            child.service_session_id(),
            None,
            "the parent's server-side conversation pointer must not leak to the child"
        );
        assert!(child.context_providers.is_empty());

        // State is shared both ways.
        assert_eq!(child.state.get("k"), Some(serde_json::json!("v")));
        child.state.insert("from-child", serde_json::json!(2));
        assert_eq!(parent.state.get("from-child"), Some(serde_json::json!(2)));

        // And a service id the child adopts during its own run stays local
        // to the child rather than clobbering the parent's.
        let mut child = child;
        child.set_service_session_id("svc-child");
        assert_eq!(parent.service_session_id(), Some("svc-parent"));
    }

    #[test]
    fn from_dict_generates_a_session_id_when_absent() {
        let restored = AgentSession::from_dict(&serde_json::json!({})).unwrap();
        assert!(!restored.session_id().is_empty());
        assert!(restored.service_session_id().is_none());
        assert!(restored.state.is_empty());
    }
}
