//! An offline walk through the harness agent: a scripted chat client plays
//! the model, which adds a todo, then asks to write a file (approval
//! required), then finishes once the user approves.
//!
//! Run with: `cargo run -p agent-framework-harness --example harness_offline`

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::prelude::*;
use agent_framework_core::types::FunctionArguments;
use agent_framework_harness::file_access::{AgentFileStore, InMemoryAgentFileStore};
use agent_framework_harness::tool_approval::create_always_approve_tool_response;
use agent_framework_harness::HarnessAgent;
use async_trait::async_trait;
use serde_json::{json, Value};

/// Replays canned responses in order.
struct ScriptedModel(Mutex<Vec<ChatResponse>>);

#[async_trait]
impl ChatClient for ScriptedModel {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut queue = self.0.lock().unwrap();
        Ok(if queue.is_empty() {
            ChatResponse::from_text("(done)")
        } else {
            queue.remove(0)
        })
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let updates = response
            .messages
            .into_iter()
            .map(|m| {
                Ok(ChatResponseUpdate {
                    contents: m.contents,
                    role: Some(m.role),
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();
        Ok(Box::pin(futures::stream::iter(updates)))
    }
}

fn call(id: &str, name: &str, args: Value) -> ChatResponse {
    let args: HashMap<String, Value> = args
        .as_object()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                id,
                name,
                Some(FunctionArguments::Object(args)),
            ))],
        )],
        ..Default::default()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = ScriptedModel(Mutex::new(vec![
        call(
            "c1",
            "todos_add",
            json!({"todos": [{"title": "Write the summary"}]}),
        ),
        call(
            "c2",
            "file_access_write",
            json!({"file_name": "summary.md", "content": "# Summary\nAll good."}),
        ),
        ChatResponse::from_text("I wrote summary.md and the work is complete."),
    ]));
    let workspace = Arc::new(InMemoryAgentFileStore::new());
    let agent = HarnessAgent::builder(model)
        .name("offline-harness")
        .disable_web_search(true)
        .file_memory_store(Arc::new(InMemoryAgentFileStore::new()))
        .file_access_store(workspace.clone())
        .build()?;

    let mut session = agent.create_session();
    let response = agent
        .run(
            vec![Message::user("Summarize the project into summary.md")],
            Some(&mut session),
        )
        .await?;
    for request in response.user_input_requests() {
        println!(
            "approval needed: {}({:?})",
            request.function_call.name, request.function_call.arguments
        );
    }
    let request = response.user_input_requests()[0].clone();

    // Approve, and don't ask again for this tool in this session.
    let response = agent
        .run(
            vec![create_always_approve_tool_response(&request, None)],
            Some(&mut session),
        )
        .await?;
    println!("assistant: {}", response.text());
    println!("summary.md: {:?}", workspace.read("summary.md").await?);
    Ok(())
}
