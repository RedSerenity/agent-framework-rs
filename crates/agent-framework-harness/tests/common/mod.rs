//! Shared offline test doubles: a scripted chat client, a scripted agent,
//! and temp-dir helpers.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_framework_core::agent::{AgentRunOptions, SupportsAgentRun};
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::Result;
use agent_framework_core::memory::SessionContext;
use agent_framework_core::session::{AgentSession, SessionState};
use agent_framework_core::tools::ToolDefinition;
use agent_framework_core::types::{
    AgentResponse, ChatOptions, ChatResponse, ChatResponseUpdate, Content,
    FunctionApprovalRequestContent, FunctionArguments, FunctionCallContent, Message, Role,
};
use async_trait::async_trait;
use serde_json::Value;

/// A chat client returning queued responses in order and recording every
/// call's messages and options.
#[derive(Clone, Default)]
pub struct MockClient {
    pub responses: Arc<Mutex<Vec<ChatResponse>>>,
    pub seen: Arc<Mutex<Vec<Vec<Message>>>>,
    pub seen_options: Arc<Mutex<Vec<ChatOptions>>>,
}

impl MockClient {
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            ..Default::default()
        }
    }

    pub fn push(&self, response: ChatResponse) {
        self.responses.lock().unwrap().push(response);
    }

    pub fn calls(&self) -> Vec<Vec<Message>> {
        self.seen.lock().unwrap().clone()
    }

    pub fn options(&self) -> Vec<ChatOptions> {
        self.seen_options.lock().unwrap().clone()
    }

    pub fn last_options(&self) -> ChatOptions {
        self.options().last().cloned().expect("at least one call")
    }

    pub fn last_messages(&self) -> Vec<Message> {
        self.calls().last().cloned().expect("at least one call")
    }
}

#[async_trait]
impl ChatClient for MockClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(messages);
        self.seen_options.lock().unwrap().push(options);
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            Ok(ChatResponse::from_text("(no more scripted responses)"))
        } else {
            Ok(responses.remove(0))
        }
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = response
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
        Ok(Box::pin(futures::stream::iter(updates)))
    }
}

/// A function call with object arguments.
pub fn call(call_id: &str, name: &str, args: Value) -> FunctionCallContent {
    let map: HashMap<String, Value> = match args {
        Value::Object(m) => m.into_iter().collect(),
        _ => HashMap::new(),
    };
    FunctionCallContent::new(call_id, name, Some(FunctionArguments::Object(map)))
}

/// A chat response whose assistant message requests `calls`.
pub fn tool_calls(calls: Vec<FunctionCallContent>) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            calls.into_iter().map(Content::FunctionCall).collect(),
        )],
        ..Default::default()
    }
}

/// An approval request for `call` with occurrence id `id`.
pub fn approval_request(id: &str, call: FunctionCallContent) -> FunctionApprovalRequestContent {
    FunctionApprovalRequestContent {
        id: id.to_string(),
        function_call: call,
    }
}

/// An agent returning queued responses and recording each run's input.
#[derive(Clone, Default)]
pub struct ScriptedAgent {
    pub responses: Arc<Mutex<Vec<AgentResponse>>>,
    pub inputs: Arc<Mutex<Vec<Vec<Message>>>>,
    pub name: Option<String>,
}

impl ScriptedAgent {
    pub fn new(responses: Vec<AgentResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            ..Default::default()
        }
    }

    pub fn inputs(&self) -> Vec<Vec<Message>> {
        self.inputs.lock().unwrap().clone()
    }
}

/// An assistant text response.
pub fn text_response(text: &str) -> AgentResponse {
    AgentResponse {
        messages: vec![Message::assistant(text)],
        ..Default::default()
    }
}

/// An assistant response carrying approval requests.
pub fn approval_response(requests: Vec<FunctionApprovalRequestContent>) -> AgentResponse {
    let mut contents: Vec<Content> = requests
        .iter()
        .map(|r| Content::FunctionCall(r.function_call.clone()))
        .collect();
    contents.extend(requests.into_iter().map(Content::FunctionApprovalRequest));
    AgentResponse {
        messages: vec![Message::with_contents(Role::assistant(), contents)],
        ..Default::default()
    }
}

#[async_trait]
impl SupportsAgentRun for ScriptedAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        self.run_with_options(messages, session, AgentRunOptions::default())
            .await
    }

    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
        _options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        self.inputs.lock().unwrap().push(messages);
        let mut responses = self.responses.lock().unwrap();
        Ok(if responses.is_empty() {
            text_response("done")
        } else {
            responses.remove(0)
        })
    }

    fn id(&self) -> &str {
        "scripted"
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// A fresh, unique temp directory (removed by [`TempDir`]'s drop).
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("harness-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A provider context for session `id` with `state`.
pub fn ctx(id: &str, state: &SessionState, input: Vec<Message>) -> SessionContext {
    let mut ctx = SessionContext::new(input);
    ctx.session_id = Some(id.to_string());
    ctx.session_state = Some(state.clone());
    ctx
}

/// Invoke the tool named `name` from `tools` with `args`.
pub async fn invoke(tools: &[ToolDefinition], name: &str, args: Value) -> Result<Value> {
    let tool = tools
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("tool {name} not found"));
    tool.executor
        .clone()
        .expect("executable")
        .invoke(args)
        .await
}

/// Like [`invoke`] but unwraps a string result.
pub async fn invoke_text(tools: &[ToolDefinition], name: &str, args: Value) -> String {
    match invoke(tools, name, args).await.unwrap() {
        Value::String(s) => s,
        other => other.to_string(),
    }
}
