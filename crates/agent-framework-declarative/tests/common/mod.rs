//! Shared test helpers: a scripted mock chat client (no network), copied from
//! the core crate's integration-test pattern.
//!
//! Each integration-test binary includes this module and uses a different
//! subset, so unused-item warnings here are expected.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;

/// A scripted chat client that returns queued responses in order.
#[derive(Clone)]
pub struct MockClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
    /// Every message list the client was asked to respond to.
    pub seen: Arc<Mutex<Vec<Vec<Message>>>>,
    model: Option<String>,
}

impl MockClient {
    /// Create a mock returning `responses` in order (then a filler response).
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            seen: Arc::new(Mutex::new(Vec::new())),
            model: None,
        }
    }

    /// A mock that always replies with the same text.
    pub fn always(text: &str) -> Self {
        Self::new(vec![ChatResponse::from_text(text)])
    }
}

#[async_trait]
impl ChatClient for MockClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(messages);
        let mut resps = self.responses.lock().unwrap();
        if resps.is_empty() {
            Ok(ChatResponse::from_text("(no more scripted responses)"))
        } else if resps.len() == 1 {
            Ok(resps[0].clone())
        } else {
            Ok(resps.remove(0))
        }
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = resp
            .messages
            .into_iter()
            .map(|m| {
                Ok(ChatResponseUpdate {
                    contents: m.contents,
                    role: Some(m.role),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }

    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
}

/// Build an agent registry entry backed by a mock that always replies `text`.
pub fn mock_agent(name: &str, text: &str) -> Agent {
    Agent::builder(MockClient::always(text)).name(name).build()
}

/// Read an upstream fixture under `tests/fixtures/upstream/`.
pub fn fixture(path: &str) -> String {
    let full = format!(
        "{}/tests/fixtures/upstream/{path}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("{full}: {e}"))
}

/// The declarative variables of a run.
pub async fn vars(run: &WorkflowRun) -> serde_json::Map<String, serde_json::Value> {
    agent_framework_declarative::flow::final_state(&run.shared_state())
        .await
        .expect("declarative state present")
}

/// The run's outputs as strings.
pub fn texts(run: &WorkflowRun) -> Vec<String> {
    run.outputs()
        .into_iter()
        .map(|v| match v {
            serde_json::Value::String(s) => s,
            other => other.to_string(),
        })
        .collect()
}

/// The single pending request: `(request_id, payload)`.
pub fn pending(run: &WorkflowRun) -> (String, serde_json::Value) {
    let p = run.pending_requests();
    assert_eq!(p.len(), 1, "expected one pending request, got {p:?}");
    (p[0].request_id.clone(), p[0].request_data.clone())
}

/// An agent that replies with scripted texts in order (repeating the last),
/// recording every message list it was asked to answer.
pub struct ScriptedAgent {
    name: String,
    replies: Mutex<std::collections::VecDeque<String>>,
    /// Every message list the agent received.
    pub calls: Arc<Mutex<Vec<Vec<Message>>>>,
    fail: bool,
}

impl ScriptedAgent {
    /// An agent named `name` replying with `replies` in order.
    pub fn new(name: &str, replies: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: false,
        })
    }

    /// An agent whose every run fails.
    pub fn failing(name: &str) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            replies: Mutex::new(Default::default()),
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: true,
        })
    }

    /// The text of the last message of each call.
    pub fn last_inputs(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|c| c.last().map(Message::text).unwrap_or_default())
            .collect()
    }

    /// How many times the agent ran.
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl SupportsAgentRun for ScriptedAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        self.calls.lock().unwrap().push(messages);
        if self.fail {
            return Err(Error::AgentExecution("scripted failure".into()));
        }
        let text = {
            let mut replies = self.replies.lock().unwrap();
            if replies.len() > 1 {
                replies.pop_front().unwrap()
            } else {
                replies.front().cloned().unwrap_or_default()
            }
        };
        Ok(AgentResponse {
            messages: vec![Message::new(Role::new("assistant"), text)],
            ..Default::default()
        })
    }

    fn id(&self) -> &str {
        &self.name
    }
}
