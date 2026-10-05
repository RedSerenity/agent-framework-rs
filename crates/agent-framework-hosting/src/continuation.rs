//! The state [`AgentHost`](crate::AgentHost) keeps so a Responses request can
//! continue an earlier one.
//!
//! - **Agents** keep sessions in a
//!   [`SessionStore`](agent_framework_core::session_store::SessionStore):
//!   one snapshot per response id (immutable, so callers can branch from
//!   it) and one per conversation id (a mutable head).
//! - **Workflows** keep one checkpoint storage per conversation, as
//!   upstream DevUI does, so a run paused on a human-in-the-loop request can
//!   be resumed by a later request carrying the responses.
//!
//! Keys are scoped by entity id, so one entity's response id never restores
//! another entity's conversation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_framework_core::session_store::{InMemorySessionStore, SessionStore};
use agent_framework_core::storage_keys::storage_key_segment;
use agent_framework_core::workflow::{CheckpointStorage, InMemoryCheckpointStorage};

/// Builds a fresh checkpoint storage for a new workflow conversation.
pub(crate) type CheckpointStorageFactory =
    Arc<dyn Fn() -> Arc<dyn CheckpointStorage> + Send + Sync>;

pub(crate) struct Continuations {
    store: Arc<dyn SessionStore>,
    checkpoint_factory: CheckpointStorageFactory,
    head_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Scoped conversation key -> that conversation's checkpoint storage.
    workflow_storages: Mutex<HashMap<String, Arc<dyn CheckpointStorage>>>,
    /// Scoped workflow response key -> the conversation it belongs to, so
    /// `previous_response_id` can name a workflow conversation too.
    workflow_responses: Mutex<HashMap<String, String>>,
}

impl Default for Continuations {
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl Continuations {
    pub(crate) fn new(
        store: Option<Arc<dyn SessionStore>>,
        checkpoint_factory: Option<CheckpointStorageFactory>,
    ) -> Self {
        Self {
            store: store.unwrap_or_else(|| Arc::new(InMemorySessionStore::new())),
            checkpoint_factory: checkpoint_factory
                .unwrap_or_else(|| Arc::new(|| Arc::new(InMemoryCheckpointStorage::new()))),
            head_locks: Mutex::new(HashMap::new()),
            workflow_storages: Mutex::new(HashMap::new()),
            workflow_responses: Mutex::new(HashMap::new()),
        }
    }

    /// The store key for `id` under `entity`. Both halves go through the
    /// injective key encoding, so no `(entity, id)` pair can impersonate
    /// another across the separator.
    pub(crate) fn key(entity: &str, id: &str) -> String {
        format!(
            "{}:{}",
            storage_key_segment(entity, "~e-"),
            storage_key_segment(id, "~i-")
        )
    }

    pub(crate) fn sessions(&self) -> &Arc<dyn SessionStore> {
        &self.store
    }

    /// Serialize runs that advance one conversation head. Held across the
    /// whole read-run-write cycle, so two concurrent turns on a conversation
    /// cannot both start from the same head and lose one of their writes.
    pub(crate) async fn lock_head(&self, key: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = self
            .head_locks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(key.to_string())
            .or_default()
            .clone();
        lock.lock_owned().await
    }

    /// The checkpoint storage for a workflow conversation, created on first
    /// use.
    pub(crate) fn workflow_storage(&self, conversation_key: &str) -> Arc<dyn CheckpointStorage> {
        self.workflow_storages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(conversation_key.to_string())
            .or_insert_with(|| (self.checkpoint_factory)())
            .clone()
    }

    pub(crate) fn record_workflow_response(&self, response_key: String, conversation_id: String) {
        self.workflow_responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(response_key, conversation_id);
    }

    pub(crate) fn workflow_conversation_of(&self, response_key: &str) -> Option<String> {
        self.workflow_responses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(response_key)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_scoped_and_unambiguous() {
        assert_eq!(Continuations::key("weather", "resp_1"), "weather:resp_1");
        assert_ne!(
            Continuations::key("a.b", "c"),
            Continuations::key("a", "b.c")
        );
        assert_ne!(
            Continuations::key("a", "resp_1"),
            Continuations::key("b", "resp_1")
        );
    }
}
