//! End-to-end tests exercising agents, the tool loop, and workflows using a
//! mock chat client (no network).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::agent::AsToolOptions;
use agent_framework_core::prelude::*;
use agent_framework_core::types::{Content, FunctionArguments, FunctionCallContent, Role};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// A scripted chat client that returns queued responses in order.
#[derive(Clone)]
struct MockClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
    seen: Arc<Mutex<Vec<Vec<Message>>>>,
    seen_options: Arc<Mutex<Vec<ChatOptions>>>,
}

impl MockClient {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            seen: Arc::new(Mutex::new(Vec::new())),
            seen_options: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl MockClient {
    /// The `ChatOptions` of the most recent call, if any.
    fn last_options(&self) -> Option<ChatOptions> {
        self.seen_options.lock().unwrap().last().cloned()
    }
    /// Every call's `ChatOptions`, in order.
    fn all_options(&self) -> Vec<ChatOptions> {
        self.seen_options.lock().unwrap().clone()
    }
    /// Every call's message list, in order.
    fn all_seen(&self) -> Vec<Vec<Message>> {
        self.seen.lock().unwrap().clone()
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
        let mut resps = self.responses.lock().unwrap();
        if resps.is_empty() {
            Ok(ChatResponse::from_text("(no more scripted responses)"))
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
}

#[tokio::test]
async fn basic_agent_run() {
    let client = MockClient::new(vec![ChatResponse::from_text("Hello there!")]);
    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("Be nice.")
        .build();

    let response = agent.run_once("Hi").await.unwrap();
    assert_eq!(response.text(), "Hello there!");
    assert_eq!(
        response.messages[0].author_name.as_deref(),
        Some("assistant")
    );
}

#[tokio::test]
async fn agent_streaming_updates_thread() {
    let client = MockClient::new(vec![ChatResponse::from_text("streamed reply")]);
    let agent = Agent::builder(client).build();

    // Attach an explicit history provider so the test can inspect it directly
    // (its `Arc<Mutex<..>>` is shared with the clone passed into `run_stream`).
    let history = InMemoryHistoryProvider::new();
    let mut thread = AgentSession::new();
    thread.context_providers.push(Arc::new(history.clone()));

    let mut stream = agent
        .run_stream("hello", Some(thread.clone()), None)
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text());
    }
    assert_eq!(text, "streamed reply");
    // The shared history provider should now contain the user + assistant messages.
    assert_eq!(history.list_messages().len(), 2);
    let _ = &mut thread;
}

#[tokio::test]
async fn tool_loop_executes_function() {
    // First response asks to call `add`; second returns the final answer.
    let call = FunctionCallContent::new(
        "call_1",
        "add",
        Some(FunctionArguments::Raw(json!({"a": 2, "b": 3}).to_string())),
    );
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let answer = ChatResponse::from_text("The sum is 5.");
    let client = MockClient::new(vec![ask, answer]);

    let add = FunctionTool::new(
        "add",
        "Add two integers.",
        json!({
            "type": "object",
            "properties": { "a": {"type":"integer"}, "b": {"type":"integer"} },
            "required": ["a","b"]
        }),
        |args| async move {
            let a = args["a"].as_i64().unwrap_or(0);
            let b = args["b"].as_i64().unwrap_or(0);
            Ok(json!(a + b))
        },
    )
    .into_definition();

    let agent = Agent::builder(client).tool(add).build();
    let response = agent.run_once("What is 2 + 3?").await.unwrap();
    assert!(response.text().contains("5"), "got: {}", response.text());
    // The response should include the tool interaction messages.
    assert!(response.messages.iter().any(|m| m.role == Role::tool()
        && m.contents
            .iter()
            .any(|c| matches!(c, Content::FunctionResult(_)))));
}

#[tokio::test]
async fn sequential_workflow_chains_agents() {
    let a = Arc::new(
        Agent::builder(MockClient::new(vec![ChatResponse::from_text("step-A")]))
            .name("A")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;
    let b = Arc::new(
        Agent::builder(MockClient::new(vec![ChatResponse::from_text("step-B")]))
            .name("B")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;

    let workflow = agent_framework_core::workflow::SequentialBuilder::new()
        .participants(vec![a, b])
        .build()
        .unwrap();

    let result = workflow.run("start").await.unwrap();
    let output = result.last_output().expect("a final output");
    let conversation: Vec<Message> = serde_json::from_value(output).unwrap();
    let texts: Vec<String> = conversation.iter().map(|m| m.text()).collect();
    assert!(texts.contains(&"step-A".to_string()));
    assert!(texts.contains(&"step-B".to_string()));
}

#[tokio::test]
async fn concurrent_workflow_fans_out() {
    let a = Arc::new(
        Agent::builder(MockClient::new(vec![ChatResponse::from_text("from-A")]))
            .name("A")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;
    let b = Arc::new(
        Agent::builder(MockClient::new(vec![ChatResponse::from_text("from-B")]))
            .name("B")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;

    let workflow = agent_framework_core::workflow::ConcurrentBuilder::new()
        .participants(vec![a, b])
        .build()
        .unwrap();

    let result = workflow.run("question").await.unwrap();
    let output = result.last_output().expect("a final output");
    let conversation: Vec<Message> = serde_json::from_value(output).unwrap();
    let texts: Vec<String> = conversation.iter().map(|m| m.text()).collect();
    assert!(texts.iter().any(|t| t == "from-A"));
    assert!(texts.iter().any(|t| t == "from-B"));
}

#[tokio::test]
async fn workflow_function_executor() {
    use agent_framework_core::workflow::{FunctionExecutor, WorkflowBuilder};

    let doubler = FunctionExecutor::new("double", |msg, ctx| async move {
        let n = msg.as_i64().unwrap_or(0);
        ctx.send_message(json!(n * 2)).await?;
        Ok(())
    });
    let printer = FunctionExecutor::new("out", |msg, ctx| async move {
        ctx.yield_output(msg).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(doubler))
        .add_executor(Arc::new(printer))
        .set_start("double")
        .add_edge("double", "out")
        .build()
        .unwrap();

    let result = workflow.run(json!(21)).await.unwrap();
    assert_eq!(result.last_output(), Some(json!(42)));
}

#[test]
fn chat_options_merge() {
    let base = ChatOptions::new()
        .with_temperature(0.2)
        .with_instructions("base");
    let over = ChatOptions::new()
        .with_temperature(0.9)
        .with_instructions("more");
    let merged = base.merge(over);
    assert_eq!(merged.temperature, Some(0.9));
    assert_eq!(merged.instructions.as_deref(), Some("base\nmore"));
}

#[test]
fn function_call_merge_does_not_duplicate_name() {
    // A provider that repeats the full name in a continuation chunk must not
    // produce "addadd".
    let mut base =
        FunctionCallContent::new("c1", "add", Some(FunctionArguments::Raw("{\"a\":".into())));
    let cont = FunctionCallContent::new("", "add", Some(FunctionArguments::Raw("1}".into())));
    base.merge(&cont).unwrap();
    assert_eq!(base.name, "add");
    match base.arguments {
        Some(FunctionArguments::Raw(s)) => assert_eq!(s, "{\"a\":1}"),
        other => panic!("unexpected args: {other:?}"),
    }
}

/// SupportsAgentRun middleware that appends a suffix to every assistant message.
struct SuffixMiddleware;

#[async_trait]
impl Middleware<AgentContext> for SuffixMiddleware {
    async fn process(&self, ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
        let mut ctx = next.run(ctx).await?;
        if let Some(resp) = ctx.result.as_mut() {
            for m in &mut resp.messages {
                m.contents.push(Content::text(" [checked]"));
            }
        }
        Ok(ctx)
    }
}

#[tokio::test]
async fn middleware_applies_on_streaming_path() {
    let client = MockClient::new(vec![ChatResponse::from_text("answer")]);
    let agent = Agent::builder(client)
        .middleware(Arc::new(SuffixMiddleware))
        .build();

    // Streaming must honor the middleware just like `run` does.
    let mut stream = agent.run_stream("hi", None, None).await.unwrap();
    let mut text = String::new();
    while let Some(u) = stream.next().await {
        text.push_str(&u.unwrap().text());
    }
    assert!(text.contains("answer"), "got: {text}");
    assert!(
        text.contains("[checked]"),
        "middleware not applied on stream: {text}"
    );
}

#[tokio::test]
async fn tool_loop_reports_invalid_arguments() {
    // The model asks to call `add` with malformed JSON arguments; the loop must
    // report a tool error rather than invoking with null input.
    let bad_call = FunctionCallContent::new(
        "call_1",
        "add",
        Some(FunctionArguments::Raw("{ not json".into())),
    );
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(bad_call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse::from_text("done");

    let invoked = Arc::new(Mutex::new(false));
    let invoked_clone = invoked.clone();
    let add = FunctionTool::new(
        "add",
        "Add.",
        json!({"type":"object","properties":{}}),
        move |_args| {
            let invoked = invoked_clone.clone();
            async move {
                *invoked.lock().unwrap() = true;
                Ok(json!(0))
            }
        },
    )
    .into_definition();

    let agent = Agent::builder(MockClient::new(vec![ask, answer]))
        .tool(add)
        .build();
    let response = agent.run_once("add please").await.unwrap();

    // The tool must NOT have been invoked with bogus arguments.
    assert!(
        !*invoked.lock().unwrap(),
        "tool should not run on invalid args"
    );
    // A tool-error result should be present in the conversation.
    assert!(response.messages.iter().any(|m| m
        .contents
        .iter()
        .any(|c| matches!(c, Content::FunctionResult(fr) if fr.exception.is_some()))));
}

/// A context provider that records lifecycle-hook activity: whether
/// `after_run` fired, the error (if any) the last `after_run` carried, and
/// every `service_session_id` observed by `before_run` (upstream renamed
/// `invoking`/`invoked` to `before_run`/`after_run` and removed
/// `thread_created` entirely). Also injects an instruction so `before_run`
/// has an observable effect.
#[derive(Default, Clone)]
struct RecordingProvider {
    invoked: Arc<Mutex<bool>>,
    invoked_error: Arc<Mutex<Option<String>>>,
    service_session_ids: Arc<Mutex<Vec<Option<String>>>>,
}

#[async_trait]
impl ContextProvider for RecordingProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        // `session_id` is always `Some` (the session's own generated id);
        // `service_session_id` is the interesting signal here, reflecting
        // service-managed adoption.
        assert!(ctx.session_id.is_some(), "session_id is always populated");
        self.service_session_ids
            .lock()
            .unwrap()
            .push(ctx.service_session_id.clone());
        ctx.add_instructions("remember: be brief");
        Ok(())
    }
    async fn after_run(
        &self,
        _request: &[Message],
        _response: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        *self.invoked.lock().unwrap() = true;
        *self.invoked_error.lock().unwrap() = error.map(|e| e.to_string());
        Ok(())
    }
}

#[tokio::test]
async fn context_provider_invoked_hook_fires() {
    let provider = RecordingProvider::default();
    let invoked = provider.invoked.clone();

    let client = MockClient::new(vec![ChatResponse::from_text("ok")]);
    let agent = Agent::builder(client)
        .context_provider(Arc::new(provider))
        .build();

    let _ = agent.run_once("hi").await.unwrap();
    assert!(
        *invoked.lock().unwrap(),
        "after_run hook was not called after run"
    );
}

#[tokio::test]
async fn streaming_tool_replay_preserves_message_boundaries() {
    // Tool call, then final answer — two assistant messages that must NOT be
    // merged when the streamed updates are re-aggregated.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse::from_text("final answer");

    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("done")) },
    )
    .into_definition();

    let agent = Agent::builder(MockClient::new(vec![ask, answer]))
        .tool(noop)
        .build();

    let mut stream = agent.run_stream("go", None, None).await.unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    // Re-aggregate exactly as a downstream consumer would.
    let aggregated = AgentResponse::from_updates(updates);
    // The final answer must appear as its own assistant message, not merged
    // into the earlier tool-call message.
    let final_msg = aggregated.messages.last().unwrap();
    assert_eq!(final_msg.text(), "final answer");
    assert!(
        final_msg
            .contents
            .iter()
            .all(|c| !matches!(c, Content::FunctionCall(_))),
        "final message was merged with the tool-call message"
    );
}

#[tokio::test]
async fn streaming_tool_replay_preserves_usage_finish_reason_and_conversation_id() {
    // Usage, finish reason, and the service conversation id must survive the
    // tool-loop's run-then-replay streaming path, so aggregating the stream
    // yields the same metadata a non-streaming run() returns.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let mut usage = UsageDetails::new();
    usage.input_token_count = Some(11);
    usage.output_token_count = Some(7);
    let answer = ChatResponse {
        usage_details: Some(usage),
        finish_reason: Some(FinishReason::stop()),
        conversation_id: Some("conv-9".into()),
        ..ChatResponse::from_text("final answer")
    };

    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("done")) },
    )
    .into_definition();

    let agent = Agent::builder(MockClient::new(vec![ask, answer]))
        .tool(noop)
        .build();

    let mut stream = agent.run_stream("go", None, None).await.unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    let aggregated = AgentResponse::from_updates(updates);
    assert_eq!(aggregated.conversation_id.as_deref(), Some("conv-9"));
    let usage = aggregated
        .usage_details
        .as_ref()
        .expect("usage must survive the replay");
    assert_eq!(usage.output_token_count, Some(7));
    // The usage rode as a Content::Usage item and must have folded into
    // usage_details, not leaked into the final message's contents.
    assert!(aggregated
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .all(|c| !matches!(c, Content::Usage(_))));
    assert_eq!(aggregated.messages.last().unwrap().text(), "final answer");
}

#[tokio::test]
async fn per_run_conversation_id_survives_on_a_local_thread() {
    // A per-run ChatOptions::conversation_id on a LOCAL thread must reach the
    // provider (it was previously clobbered by the thread's absent service
    // id, silently starting a new service conversation).
    let client = MockClient::new(vec![ChatResponse::from_text("ok")]);
    let probe = client.clone();
    let agent = Agent::builder(client).build();
    let mut thread = agent.create_session();
    let options = AgentRunOptions::new().with_chat_options(ChatOptions {
        conversation_id: Some("conv-override".into()),
        ..Default::default()
    });
    agent
        .run_with_options(vec![Message::user("hi")], Some(&mut thread), options)
        .await
        .unwrap();
    assert_eq!(
        probe.last_options().unwrap().conversation_id.as_deref(),
        Some("conv-override")
    );
}

#[tokio::test]
async fn service_session_id_wins_over_per_run_conversation_id() {
    // Continuity contract: a service-managed thread's id drives the call even
    // when a per-run override is supplied.
    let resp = ChatResponse {
        conversation_id: Some("svc-1".into()),
        ..ChatResponse::from_text("ok")
    };
    let client = MockClient::new(vec![resp]);
    let probe = client.clone();
    let agent = Agent::builder(client).build();
    let mut thread = AgentSession::service("svc-1");
    let options = AgentRunOptions::new().with_chat_options(ChatOptions {
        conversation_id: Some("conv-override".into()),
        ..Default::default()
    });
    agent
        .run_with_options(vec![Message::user("hi")], Some(&mut thread), options)
        .await
        .unwrap();
    assert_eq!(
        probe.last_options().unwrap().conversation_id.as_deref(),
        Some("svc-1")
    );
}

#[tokio::test]
async fn middleware_stream_replay_preserves_conversation_id_and_usage() {
    // With agent middleware configured, run_stream replays the completed run;
    // the response's conversation id and usage must survive that replay.
    let mut usage = UsageDetails::new();
    usage.output_token_count = Some(3);
    let resp = ChatResponse {
        conversation_id: Some("conv-7".into()),
        usage_details: Some(usage),
        ..ChatResponse::from_text("answer")
    };
    let client = MockClient::new(vec![resp]);
    let agent = Agent::builder(client)
        .middleware(Arc::new(SuffixMiddleware))
        .build();

    let mut stream = agent.run_stream("hi", None, None).await.unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    let aggregated = AgentResponse::from_updates(updates);
    assert_eq!(aggregated.conversation_id.as_deref(), Some("conv-7"));
    assert_eq!(
        aggregated
            .usage_details
            .expect("usage survives")
            .output_token_count,
        Some(3)
    );
}

#[tokio::test]
async fn service_created_conversation_id_propagates_into_tool_followup() {
    // A service-managed client creates the thread on the first tool-call turn
    // and returns its conversation_id. The follow-up submission (carrying the
    // FunctionResultContent) must target that thread, and — since the service
    // now holds the history — send only the new tool results.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let first = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        conversation_id: Some("thread_new".into()),
        ..Default::default()
    };
    let second = ChatResponse::from_text("done");
    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("ok")) },
    )
    .into_definition();
    let probe = MockClient::new(vec![first, second]);
    let client = FunctionInvokingChatClient::new(probe.clone());

    let options = ChatOptions {
        tools: vec![noop],
        ..Default::default()
    };
    let resp = client
        .get_response(vec![Message::user("go")], options)
        .await
        .unwrap();
    assert_eq!(resp.text(), "done");

    let all_opts = probe.all_options();
    assert_eq!(all_opts.len(), 2, "expected two underlying calls");
    // First call had no conversation id; the follow-up carries the one the
    // service created.
    assert!(all_opts[0].conversation_id.is_none());
    assert_eq!(all_opts[1].conversation_id.as_deref(), Some("thread_new"));

    // The follow-up sends only the new tool results, not the re-accumulated
    // history (the service holds it server-side).
    let seen = probe.all_seen();
    let followup = &seen[1];
    assert!(
        followup.iter().all(|m| m.role == Role::tool()),
        "follow-up should carry only tool-result messages"
    );
}

#[tokio::test]
async fn duplicate_provider_message_ids_do_not_merge_on_replay() {
    // A service (e.g. Assistants) can reuse one run id for the tool-call turn
    // and the final answer. If the replay preserved that duplicate id,
    // aggregation would merge the final text into the tool-call message and
    // reorder it ahead of the tool result. The replay must keep the two
    // assistant messages distinct.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let mut tool_call_msg =
        Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)]);
    tool_call_msg.message_id = Some("run_dup".into());
    let ask = ChatResponse {
        messages: vec![tool_call_msg],
        ..Default::default()
    };
    let mut final_msg = Message::with_contents(Role::assistant(), vec![Content::text("final")]);
    final_msg.message_id = Some("run_dup".into()); // same id as the tool-call turn
    let answer = ChatResponse {
        messages: vec![final_msg],
        ..Default::default()
    };
    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("ok")) },
    )
    .into_definition();
    let agent = Agent::builder(MockClient::new(vec![ask, answer]))
        .tool(noop)
        .build();

    let mut stream = agent.run_stream("go", None, None).await.unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    let aggregated = AgentResponse::from_updates(updates);
    // Final answer stays its own message, after the tool result — not merged
    // into the tool-call message.
    let last = aggregated.messages.last().unwrap();
    assert_eq!(last.text(), "final");
    assert!(last
        .contents
        .iter()
        .all(|c| !matches!(c, Content::FunctionCall(_))));
}

#[tokio::test]
async fn provider_resolved_tool_calls_are_not_executed_locally() {
    // A response carrying a function call WITH its matching result in the
    // same response (e.g. Anthropic server-side web-search/MCP tool use) was
    // executed by the provider: the call must not enter the local tool loop,
    // which would emit a bogus "tool not found" and burn an extra iteration.
    let call = FunctionCallContent::new(
        "srv_1",
        "hosted_web_search",
        Some(FunctionArguments::Raw("{}".into())),
    );
    let resolved = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![
                Content::FunctionCall(call),
                Content::FunctionResult(FunctionResultContent {
                    call_id: "srv_1".into(),
                    result: Some(json!({"hits": 3})),
                    exception: None,
                }),
                Content::text("Found 3 results."),
            ],
        )],
        ..Default::default()
    };
    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("x")) },
    )
    .into_definition();
    // Exactly one scripted response: a second loop iteration would consume
    // the "(no more scripted responses)" fallback and change the text.
    let agent = Agent::builder(MockClient::new(vec![resolved]))
        .tool(noop)
        .build();

    let out = agent.run_once("go").await.unwrap();
    assert_eq!(out.text(), "Found 3 results.");
    // No synthetic error result was appended for the pre-resolved call.
    assert!(out
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(Content::as_function_result)
        .all(|fr| fr.exception.is_none()));
}

#[tokio::test]
async fn chat_level_tool_stream_replay_carries_finish_reason() {
    // AgentResponse has no finish_reason (matching upstream), so the
    // finish-reason half of the replay metadata is asserted at the
    // chat-client level, where ChatResponse::from_updates surfaces it.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse {
        finish_reason: Some(FinishReason::stop()),
        ..ChatResponse::from_text("done")
    };
    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("ok")) },
    )
    .into_definition();

    let client = FunctionInvokingChatClient::new(MockClient::new(vec![ask, answer]));
    let options = ChatOptions {
        tools: vec![noop],
        ..Default::default()
    };
    let mut stream = client
        .get_streaming_response(vec![Message::user("go")], options)
        .await
        .unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    let aggregated = ChatResponse::from_updates(updates);
    assert_eq!(aggregated.finish_reason, Some(FinishReason::stop()));
    assert_eq!(aggregated.messages.last().unwrap().text(), "done");
}

#[tokio::test]
async fn workflow_errors_on_max_iterations() {
    use agent_framework_core::workflow::{FunctionExecutor, WorkflowBuilder};

    // A single executor that sends to itself forever.
    let looper = FunctionExecutor::new("loop", |_msg, ctx| async move {
        ctx.send_message(json!(1)).await?;
        Ok(())
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(looper))
        .set_start("loop")
        .add_edge("loop", "loop")
        .set_max_iterations(5)
        .build()
        .unwrap();

    let result = workflow.run(json!(1)).await;
    assert!(
        result.is_err(),
        "expected a workflow error on iteration limit"
    );
}

// ---------------------------------------------------------------------------
// Structured output
// ---------------------------------------------------------------------------

#[test]
fn response_format_serializes_to_openai_shape() {
    // Text / JsonObject.
    assert_eq!(
        serde_json::to_value(ResponseFormat::Text).unwrap(),
        json!({ "type": "text" })
    );
    assert_eq!(
        serde_json::to_value(ResponseFormat::JsonObject).unwrap(),
        json!({ "type": "json_object" })
    );

    // JsonSchema nests under "json_schema", matching OpenAI's request object.
    let fmt = ResponseFormat::JsonSchema {
        name: "Person".into(),
        description: Some("a person".into()),
        schema: json!({ "type": "object", "properties": { "name": { "type": "string" } } }),
        strict: Some(true),
    };
    let value = serde_json::to_value(&fmt).unwrap();
    assert_eq!(value["type"], "json_schema");
    assert_eq!(value["json_schema"]["name"], "Person");
    assert_eq!(value["json_schema"]["description"], "a person");
    assert_eq!(value["json_schema"]["strict"], true);
    assert_eq!(value["json_schema"]["schema"]["type"], "object");

    // Round-trips through Deserialize.
    let back: ResponseFormat = serde_json::from_value(value).unwrap();
    assert_eq!(back, fmt);
}

#[test]
fn parse_json_reads_structured_value() {
    #[derive(serde::Deserialize, PartialEq, Debug)]
    struct Person {
        name: String,
        age: u32,
    }

    let resp = ChatResponse::from_text(r#"{"name":"Ada","age":36}"#);
    let person: Person = resp.parse_json().unwrap();
    assert_eq!(
        person,
        Person {
            name: "Ada".into(),
            age: 36
        }
    );

    // The same convenience exists on AgentResponse.
    let agent_resp =
        AgentResponse::from_chat_response(ChatResponse::from_text(r#"{"name":"Bob","age":5}"#));
    let person2: Person = agent_resp.parse_json().unwrap();
    assert_eq!(person2.name, "Bob");

    // Non-JSON text surfaces an error rather than panicking.
    assert!(ChatResponse::from_text("not json")
        .parse_json::<Person>()
        .is_err());
}

#[test]
fn response_format_builder_sugar_sets_option() {
    let agent = Agent::builder(MockClient::new(vec![])).response_format(ResponseFormat::JsonObject);
    // Build and confirm the option flows through (via a run that echoes options
    // is unnecessary; just assert the builder compiles and produces an agent).
    let _agent = agent.build();
}

// ---------------------------------------------------------------------------
// ToolMode serde
// ---------------------------------------------------------------------------

#[test]
fn tool_mode_serde_round_trip() {
    assert_eq!(serde_json::to_value(ToolMode::Auto).unwrap(), json!("auto"));
    assert_eq!(
        serde_json::to_value(ToolMode::required_any()).unwrap(),
        json!("required")
    );
    // Like Python's serialize_model, the pinned function name is not persisted
    // on the mode itself (the provider mapping applies it).
    assert_eq!(
        serde_json::to_value(ToolMode::required_function("get_weather")).unwrap(),
        json!("required")
    );
    assert_eq!(serde_json::to_value(ToolMode::None).unwrap(), json!("none"));

    assert_eq!(
        serde_json::from_value::<ToolMode>(json!("auto")).unwrap(),
        ToolMode::Auto
    );
    assert_eq!(
        serde_json::from_value::<ToolMode>(json!("required")).unwrap(),
        ToolMode::Required(None)
    );
    assert_eq!(
        serde_json::from_value::<ToolMode>(json!("none")).unwrap(),
        ToolMode::None
    );

    assert_eq!(
        ToolMode::required_function("f").required_function_name(),
        Some("f")
    );
    assert_eq!(ToolMode::Auto.required_function_name(), None);
}

// ---------------------------------------------------------------------------
// Update aggregation
// ---------------------------------------------------------------------------

#[test]
fn agent_update_aggregation() {
    let updates = vec![
        AgentResponseUpdate {
            contents: vec![Content::text("Hello")],
            role: Some(Role::assistant()),
            ..Default::default()
        },
        AgentResponseUpdate {
            contents: vec![Content::text(" world")],
            role: Some(Role::assistant()),
            ..Default::default()
        },
    ];
    let resp = AgentResponse::from_agent_run_response_updates(updates);
    assert_eq!(resp.text(), "Hello world");
}

// ---------------------------------------------------------------------------
// Function-approval flow
// ---------------------------------------------------------------------------

/// A tool requiring approval that records how many times it actually executed.
fn approval_tool(counter: Arc<Mutex<u32>>) -> ToolDefinition {
    FunctionTool::new(
        "get_secret",
        "Return the secret value.",
        json!({ "type": "object", "properties": {} }),
        move |_args| {
            let counter = counter.clone();
            async move {
                *counter.lock().unwrap() += 1;
                Ok(json!("42"))
            }
        },
    )
    .with_approval_mode(ApprovalMode::AlwaysRequire)
    .into_definition()
}

fn secret_call() -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "get_secret",
                Some(FunctionArguments::Raw("{}".into())),
            ))],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

#[tokio::test]
async fn approval_loop_approve_executes_and_answers() {
    let counter = Arc::new(Mutex::new(0));
    let tool = approval_tool(counter.clone());
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![
        secret_call(),
        ChatResponse::from_text("The secret is 42."),
    ]));
    let options = ChatOptions::new().with_tool(tool);

    // Request 1: the model asks for an approval-gated tool -> we get an approval
    // request back, and the tool has NOT run.
    let resp1 = client
        .get_response(vec![Message::user("what is the secret?")], options.clone())
        .await
        .unwrap();
    let requests = resp1.user_input_requests();
    assert_eq!(requests.len(), 1, "expected one approval request");
    assert_eq!(requests[0].function_call.call_id, "call_1");
    assert_eq!(*counter.lock().unwrap(), 0, "tool ran before approval");
    // The assistant message still carries the original function call too.
    assert_eq!(resp1.function_calls().len(), 1);

    // Request 2: approve -> the tool runs and the model produces a final answer.
    let approval = requests[0].create_response(true);
    let mut conversation = vec![Message::user("what is the secret?")];
    conversation.extend(resp1.messages.clone());
    conversation.push(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(approval)],
    ));
    let resp2 = client.get_response(conversation, options).await.unwrap();
    assert!(resp2.text().contains("42"), "got: {}", resp2.text());
    assert_eq!(*counter.lock().unwrap(), 1, "tool should run exactly once");
}

#[tokio::test]
async fn approval_loop_reject_skips_execution() {
    let counter = Arc::new(Mutex::new(0));
    let tool = approval_tool(counter.clone());
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![
        secret_call(),
        ChatResponse::from_text("Understood, I won't retrieve it."),
    ]));
    let options = ChatOptions::new().with_tool(tool);

    let resp1 = client
        .get_response(vec![Message::user("what is the secret?")], options.clone())
        .await
        .unwrap();
    let requests = resp1.user_input_requests();
    assert_eq!(requests.len(), 1);

    // Reject the call.
    let rejection = requests[0].create_response(false);
    let mut conversation = vec![Message::user("what is the secret?")];
    conversation.extend(resp1.messages.clone());
    conversation.push(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(rejection)],
    ));
    let resp2 = client.get_response(conversation, options).await.unwrap();

    assert!(resp2.text().contains("won't"), "got: {}", resp2.text());
    assert_eq!(*counter.lock().unwrap(), 0, "rejected tool must not run");
}

#[tokio::test]
async fn agent_surfaces_and_resolves_approval_round_trip() {
    let counter = Arc::new(Mutex::new(0));
    let tool = approval_tool(counter.clone());
    let agent = Agent::builder(MockClient::new(vec![
        secret_call(),
        ChatResponse::from_text("The secret is 42."),
    ]))
    .name("keeper")
    .tool(tool)
    .build();

    // Attach an explicit history provider so the test can inspect it directly
    // after the run.
    let history = InMemoryHistoryProvider::new();
    let mut thread = AgentSession::new();
    thread.context_providers.push(Arc::new(history.clone()));

    // First run pauses awaiting approval; the request is surfaced on the agent
    // response and persisted to the thread.
    let resp1 = agent
        .run(vec![Message::user("get the secret")], Some(&mut thread))
        .await
        .unwrap();
    assert_eq!(resp1.user_input_requests().len(), 1);
    let approval = resp1.user_input_requests()[0].create_response(true);

    // Supplying the approval response as new input resolves the exchange.
    let resp2 = agent
        .run(
            vec![Message::with_contents(
                Role::user(),
                vec![Content::FunctionApprovalResponse(approval)],
            )],
            Some(&mut thread),
        )
        .await
        .unwrap();
    assert!(resp2.text().contains("42"), "got: {}", resp2.text());
    assert_eq!(*counter.lock().unwrap(), 1);

    // The thread retains the full approval exchange.
    let recorded = history.list_messages();
    assert!(recorded.iter().any(|m| !m.user_input_requests().is_empty()));
}

/// `Agent` hands providers the session's state bag in `before_run` and the
/// session itself in `after_run_in_session`.
#[tokio::test]
async fn providers_see_the_session_state_and_the_session_after_the_run() {
    struct StateProbe {
        seen_after: Mutex<Option<String>>,
    }
    #[async_trait]
    impl ContextProvider for StateProbe {
        async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
            let state = ctx
                .session_state
                .clone()
                .expect("agent supplies the state bag");
            state.insert("probe", json!("written in before_run"));
            Ok(())
        }
        async fn after_run_in_session(
            &self,
            session: &AgentSession,
            _request: &[Message],
            _response: &[Message],
            _error: Option<&Error>,
        ) -> Result<()> {
            *self.seen_after.lock().unwrap() = Some(session.session_id().to_string());
            Ok(())
        }
    }
    let probe = Arc::new(StateProbe {
        seen_after: Mutex::new(None),
    });
    let agent = Agent::builder(MockClient::new(vec![ChatResponse::from_text("ok")]))
        .context_provider(probe.clone())
        .build();
    let mut session = agent.create_session();
    agent
        .run(vec![Message::user("hi")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(
        session.state.get("probe"),
        Some(json!("written in before_run"))
    );
    assert_eq!(
        probe.seen_after.lock().unwrap().as_deref(),
        Some(session.session_id())
    );
}

/// A resolved approval exchange persists in history; later runs on the same
/// session must not execute the approved (or rejected) call again, and the
/// conversation sent to the model must stay well-formed (each call paired
/// with exactly one result, no raw approval contents).
#[tokio::test]
async fn a_resolved_approval_is_not_re_executed_by_later_runs() {
    for approved in [true, false] {
        let counter = Arc::new(Mutex::new(0));
        let client = MockClient::new(vec![
            secret_call(),
            ChatResponse::from_text("first answer"),
            ChatResponse::from_text("second answer"),
            ChatResponse::from_text("third answer"),
        ]);
        let agent = Agent::builder(client.clone())
            .tool(approval_tool(counter.clone()))
            .build();
        let mut session = agent.create_session();
        let resp1 = agent
            .run(vec![Message::user("get the secret")], Some(&mut session))
            .await
            .unwrap();
        let approval = resp1.user_input_requests()[0].create_response(approved);
        let resp2 = agent
            .run(
                vec![Message::with_contents(
                    Role::user(),
                    vec![Content::FunctionApprovalResponse(approval)],
                )],
                Some(&mut session),
            )
            .await
            .unwrap();
        // The resolved result is part of the response, so history records it.
        assert!(resp2.messages[0]
            .contents
            .iter()
            .any(|c| matches!(c, Content::FunctionResult(r) if r.call_id == "call_1")));
        let expected = u32::from(approved);
        assert_eq!(*counter.lock().unwrap(), expected);
        for turn in ["again", "and again"] {
            agent
                .run(vec![Message::user(turn)], Some(&mut session))
                .await
                .unwrap();
        }
        assert_eq!(*counter.lock().unwrap(), expected, "approved={approved}");
        let sent = client.all_seen().last().cloned().unwrap();
        let contents: Vec<&Content> = sent.iter().flat_map(|m| m.contents.iter()).collect();
        assert!(!contents.iter().any(|c| matches!(
            c,
            Content::FunctionApprovalRequest(_) | Content::FunctionApprovalResponse(_)
        )));
        let calls = contents
            .iter()
            .filter(|c| matches!(c, Content::FunctionCall(_)))
            .count();
        let results = contents
            .iter()
            .filter(|c| matches!(c, Content::FunctionResult(_)))
            .count();
        assert_eq!((calls, results), (1, 1), "approved={approved}");
    }
}

// ---------------------------------------------------------------------------
// SupportsAgentRun-as-tool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_as_tool_is_callable_by_another_agent() {
    // Inner agent always answers "INNER-RESULT".
    let inner = Agent::builder(MockClient::new(vec![ChatResponse::from_text(
        "INNER-RESULT",
    )]))
    .name("researcher")
    .description("Performs research tasks.")
    .build();
    let research_tool = inner.as_tool(AsToolOptions::new().name("research"));
    assert_eq!(research_tool.name, "research");

    // Outer agent: the model calls `research`, then answers.
    let call = FunctionCallContent::new(
        "c1",
        "research",
        Some(FunctionArguments::Raw(
            json!({ "task": "find X" }).to_string(),
        )),
    );
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let outer = Agent::builder(MockClient::new(vec![ask, ChatResponse::from_text("Done.")]))
        .tool(research_tool)
        .build();

    let response = outer.run_once("do research").await.unwrap();
    assert!(response.text().contains("Done"), "got: {}", response.text());

    // The inner agent's output flowed back as the tool result.
    let saw_inner = response
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .any(|c| {
            matches!(c, Content::FunctionResult(fr)
            if fr.result.as_ref().and_then(|v| v.as_str()) == Some("INNER-RESULT"))
        });
    assert!(
        saw_inner,
        "inner agent result missing: {:?}",
        response.messages
    );
}

// ---------------------------------------------------------------------------
// Observability
//
// The span-capture smoke test lives in its own binary (`tests/observability.rs`)
// so the `chat` tracing callsite is first evaluated under the capturing
// subscriber — `tracing` caches callsite interest globally, so sharing a binary
// with tests that hit the callsite under the no-op subscriber would disable it.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observable_chat_client_is_transparent() {
    // Decorating a client must not change its observable behavior.
    let client = ObservableChatClient::new(
        MockClient::new(vec![ChatResponse::from_text("plain")]),
        "mock",
    );
    let resp = client
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();
    assert_eq!(resp.text(), "plain");
}

// ---------------------------------------------------------------------------
// Chat & function middleware
// ---------------------------------------------------------------------------

/// Chat middleware that rewrites every outgoing user message's text.
struct RewriteUserMessage;

#[async_trait]
impl Middleware<ChatContext> for RewriteUserMessage {
    async fn process(&self, mut ctx: ChatContext, next: Next<ChatContext>) -> Result<ChatContext> {
        for m in &mut ctx.messages {
            if m.role == Role::user() {
                *m = Message::user("REWRITTEN");
            }
        }
        next.run(ctx).await
    }
}

#[tokio::test]
async fn chat_middleware_rewrites_outgoing_message() {
    let client = MockClient::new(vec![ChatResponse::from_text("ok")]);
    let seen = client.seen.clone();
    let agent = Agent::builder(client)
        .chat_middleware(Arc::new(RewriteUserMessage))
        .build();

    let _ = agent.run_once("original").await.unwrap();

    let seen = seen.lock().unwrap();
    let last = seen.last().expect("the model should have been called");
    assert!(
        last.iter().any(|m| m.text() == "REWRITTEN"),
        "model did not see the rewritten message: {last:?}"
    );
}

/// Chat middleware that short-circuits with a canned response, never letting
/// the call reach the underlying client.
struct ShortCircuitChat;

#[async_trait]
impl Middleware<ChatContext> for ShortCircuitChat {
    async fn process(&self, mut ctx: ChatContext, _next: Next<ChatContext>) -> Result<ChatContext> {
        // Deliberately does not call `next.run(ctx)`: the underlying client
        // must never be invoked.
        ctx.result = Some(ChatResponse::from_text("canned"));
        ctx.terminate = true;
        Ok(ctx)
    }
}

#[tokio::test]
async fn chat_middleware_short_circuits_model_call() {
    let client = MockClient::new(vec![ChatResponse::from_text("should not be used")]);
    let seen = client.seen.clone();
    let agent = Agent::builder(client)
        .chat_middleware(Arc::new(ShortCircuitChat))
        .build();

    let response = agent.run_once("hi").await.unwrap();

    assert_eq!(response.text(), "canned");
    assert!(
        seen.lock().unwrap().is_empty(),
        "the underlying model must not have been called"
    );
}

/// Function middleware that rewrites arguments before execution.
struct RewriteArgsMiddleware;

#[async_trait]
impl Middleware<FunctionInvocationContext> for RewriteArgsMiddleware {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        if let Some(obj) = ctx.arguments.as_object_mut() {
            obj.insert("a".to_string(), json!(100));
        }
        next.run(ctx).await
    }
}

fn add_call(a: i64, b: i64) -> ChatResponse {
    let call = FunctionCallContent::new(
        "call_1",
        "add",
        Some(FunctionArguments::Raw(json!({"a": a, "b": b}).to_string())),
    );
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

#[tokio::test]
async fn function_middleware_rewrites_arguments() {
    let client = MockClient::new(vec![add_call(2, 3), ChatResponse::from_text("done")]);

    let seen_args: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let seen_args_clone = seen_args.clone();
    let add = FunctionTool::new(
        "add",
        "Add two integers.",
        json!({"type":"object","properties":{}}),
        move |args: Value| {
            let seen_args_clone = seen_args_clone.clone();
            async move {
                *seen_args_clone.lock().unwrap() = Some(args.clone());
                let a = args["a"].as_i64().unwrap_or(0);
                let b = args["b"].as_i64().unwrap_or(0);
                Ok(json!(a + b))
            }
        },
    )
    .into_definition();

    let agent = Agent::builder(client)
        .tool(add)
        .function_middleware(Arc::new(RewriteArgsMiddleware))
        .build();

    let _ = agent.run_once("add 2 and 3").await.unwrap();

    let seen = seen_args
        .lock()
        .unwrap()
        .clone()
        .expect("the tool should have run");
    assert_eq!(
        seen["a"],
        json!(100),
        "middleware did not rewrite the argument: {seen:?}"
    );
    assert_eq!(seen["b"], json!(3), "unrelated argument must be untouched");
}

/// Function middleware that blocks execution entirely by short-circuiting
/// with its own result.
struct BlockExecutionMiddleware;

#[async_trait]
impl Middleware<FunctionInvocationContext> for BlockExecutionMiddleware {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        _next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        ctx.result = Some(json!("blocked"));
        ctx.terminate = true;
        Ok(ctx)
    }
}

#[tokio::test]
async fn function_middleware_blocks_execution() {
    let client = MockClient::new(vec![add_call(2, 3), ChatResponse::from_text("done")]);

    let invoked = Arc::new(Mutex::new(false));
    let invoked_clone = invoked.clone();
    let add = FunctionTool::new(
        "add",
        "Add two integers.",
        json!({"type":"object","properties":{}}),
        move |_args| {
            let invoked_clone = invoked_clone.clone();
            async move {
                *invoked_clone.lock().unwrap() = true;
                Ok(json!(999))
            }
        },
    )
    .into_definition();

    let agent = Agent::builder(client)
        .tool(add)
        .function_middleware(Arc::new(BlockExecutionMiddleware))
        .build();

    let response = agent.run_once("add 2 and 3").await.unwrap();

    assert!(!*invoked.lock().unwrap(), "the tool must not have executed");
    assert!(
        response
            .messages
            .iter()
            .any(|m| m.contents.iter().any(|c| matches!(
                c,
                Content::FunctionResult(fr) if fr.result == Some(json!("blocked"))
            ))),
        "the blocked result should still flow through as the tool result: {:?}",
        response.messages
    );
}

/// Records `"{label}-before"`/`"{label}-after"` around `next.run(...)`, so two
/// instances reveal the pipeline's nesting order.
struct OrderRecorder {
    label: &'static str,
    log: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<FunctionInvocationContext> for OrderRecorder {
    async fn process(
        &self,
        ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}-before", self.label));
        let ctx = next.run(ctx).await?;
        self.log
            .lock()
            .unwrap()
            .push(format!("{}-after", self.label));
        Ok(ctx)
    }
}

#[tokio::test]
async fn function_middleware_order_is_onion_nested() {
    // Two function middlewares must nest onion-style — first registered is
    // outermost — matching the ordering convention `MiddlewarePipeline`
    // already establishes for agent middleware (`Next::run` walks the
    // registered list front-to-back, invoking the terminal only once every
    // middleware has called `next`).
    let client = MockClient::new(vec![
        ChatResponse {
            messages: vec![Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionCall(FunctionCallContent::new(
                    "call_1",
                    "noop",
                    Some(FunctionArguments::Raw("{}".into())),
                ))],
            )],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        },
        ChatResponse::from_text("done"),
    ]);

    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("ok")) },
    )
    .into_definition();

    let log = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder(client)
        .tool(noop)
        .function_middleware(Arc::new(OrderRecorder {
            label: "A",
            log: log.clone(),
        }))
        .function_middleware(Arc::new(OrderRecorder {
            label: "B",
            log: log.clone(),
        }))
        .build();

    let _ = agent.run_once("go").await.unwrap();

    let log = log.lock().unwrap().clone();
    assert_eq!(log, vec!["A-before", "B-before", "B-after", "A-after"]);
}

/// The replay flow end to end: a caller that keeps its own transcript and
/// replays it every turn must not have those turns sent to the model twice.
/// Storing only the new suffix is half the fix; the request is assembled as
/// injected-context + input, so the history provider also has to stay out of
/// the way when the input already carries what it holds (PR #16 review).
#[tokio::test]
async fn replayed_transcript_is_not_sent_to_the_model_twice() {
    let client = MockClient::new(vec![
        ChatResponse::from_text("a1"),
        ChatResponse::from_text("a2"),
    ]);
    let seen = client.seen.clone();
    let agent = Agent::builder(client).build();
    let mut session = agent.create_session();

    let first = agent
        .run(vec![Message::user("q1")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(first.text(), "a1");

    // Turn two: the caller replays everything it has, plus the new question.
    let replay = vec![
        Message::user("q1"),
        Message::assistant("a1"),
        Message::user("q2"),
    ];
    let _ = agent.run(replay, Some(&mut session)).await.unwrap();

    let requests = seen.lock().unwrap().clone();
    assert_eq!(requests.len(), 2);
    let second: Vec<String> = requests[1].iter().map(Message::text).collect();
    assert_eq!(
        second,
        vec!["q1".to_string(), "a1".to_string(), "q2".to_string()],
        "the replayed turns must reach the model once, not twice"
    );
}

/// Function middleware that refuses the call with the fail-closed signal
/// (an enforcement layer's "this must not run, and the run must not
/// continue").
struct RefusingMiddleware;

#[async_trait]
impl Middleware<FunctionInvocationContext> for RefusingMiddleware {
    async fn process(
        &self,
        _ctx: FunctionInvocationContext,
        _next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        Err(Error::middleware_failure("blocked by policy"))
    }
}

/// Function middleware that fails the ordinary way, which the loop absorbs.
struct FailingMiddleware;

#[async_trait]
impl Middleware<FunctionInvocationContext> for FailingMiddleware {
    async fn process(
        &self,
        _ctx: FunctionInvocationContext,
        _next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        Err(Error::tool("transient tool trouble"))
    }
}

fn tool_calls_response(calls: &[(&str, &str)]) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            calls
                .iter()
                .map(|(call_id, name)| {
                    Content::FunctionCall(FunctionCallContent::new(
                        *call_id,
                        *name,
                        Some(FunctionArguments::Raw("{}".into())),
                    ))
                })
                .collect(),
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

/// Upstream #7562: the loop absorbs every function-middleware error into a
/// tool-error result and keeps going, which leaves an enforcement layer no way
/// to fail closed. `Error::MiddlewareFailure` is the escape — it propagates.
#[tokio::test]
async fn middleware_failure_propagates_instead_of_becoming_a_tool_error() {
    let client = MockClient::new(vec![
        tool_calls_response(&[("call_1", "noop")]),
        ChatResponse::from_text("done"),
    ]);
    let seen = client.seen.clone();

    let agent = Agent::builder(client)
        .tool(noop_tool("noop"))
        .function_middleware(Arc::new(RefusingMiddleware))
        .build();

    let err = agent
        .run_once("go")
        .await
        .expect_err("a middleware failure must fail the run");
    assert!(
        err.is_middleware_failure(),
        "the fail-closed signal must reach the caller intact: {err:?}"
    );
    assert!(err.to_string().contains("blocked by policy"), "{err}");
    assert_eq!(
        seen.lock().unwrap().len(),
        1,
        "the loop must stop, not call the model again with a tool-error result"
    );
}

/// The negative control for the test above: an ordinary middleware error keeps
/// the absorb-and-continue contract the loop has always had.
#[tokio::test]
async fn ordinary_middleware_error_is_still_absorbed_into_a_tool_error() {
    let client = MockClient::new(vec![
        tool_calls_response(&[("call_1", "noop")]),
        ChatResponse::from_text("done"),
    ]);
    let seen = client.seen.clone();

    let agent = Agent::builder(client)
        .tool(noop_tool("noop"))
        .function_middleware(Arc::new(FailingMiddleware))
        .build();

    let response = agent.run_once("go").await.expect("run should not fail");
    assert_eq!(response.text(), "done");
    assert_eq!(
        seen.lock().unwrap().len(),
        2,
        "the model should have been called again with the tool-error result"
    );
    assert!(
        response.messages.iter().any(|m| m
            .contents
            .iter()
            .any(|c| matches!(c, Content::FunctionResult(fr) if fr.exception.is_some()))),
        "the absorbed error should surface as a tool-error result: {:?}",
        response.messages
    );
}

/// A failure in one call of a parallel batch takes the batch down with it: the
/// siblings still in flight are dropped rather than left to complete.
#[tokio::test]
async fn middleware_failure_cancels_sibling_calls_in_the_batch() {
    let client = MockClient::new(vec![
        tool_calls_response(&[("call_1", "slow"), ("call_2", "refused")]),
        ChatResponse::from_text("done"),
    ]);

    let finished = Arc::new(Mutex::new(false));
    let finished_clone = finished.clone();
    let slow = FunctionTool::new(
        "slow",
        "a tool that takes a while",
        json!({"type":"object","properties":{}}),
        move |_a| {
            let finished_clone = finished_clone.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                *finished_clone.lock().unwrap() = true;
                Ok(json!("ok"))
            }
        },
    )
    .into_definition();

    // Only the refused call's middleware fails; `slow` is left running when it
    // does.
    struct RefuseOne;
    #[async_trait]
    impl Middleware<FunctionInvocationContext> for RefuseOne {
        async fn process(
            &self,
            ctx: FunctionInvocationContext,
            next: Next<FunctionInvocationContext>,
        ) -> Result<FunctionInvocationContext> {
            if ctx.function_name == "refused" {
                return Err(Error::middleware_failure("blocked by policy"));
            }
            next.run(ctx).await
        }
    }

    let agent = Agent::builder(client)
        .tool(slow)
        .tool(noop_tool("refused"))
        .function_middleware(Arc::new(RefuseOne))
        .build();

    let err = agent
        .run_once("go")
        .await
        .expect_err("the batch must fail with the refused call");
    assert!(err.is_middleware_failure(), "{err:?}");
    assert!(
        !*finished.lock().unwrap(),
        "the sibling call should have been dropped, not awaited to completion"
    );
}

#[tokio::test]
async fn service_conversation_id_is_adopted_by_thread() {
    use std::sync::{Arc, Mutex};

    // A client that manages conversations service-side: returns a
    // conversation id and records the options of every request.
    struct ServiceClient {
        seen_options: Arc<Mutex<Vec<ChatOptions>>>,
    }
    #[async_trait::async_trait]
    impl ChatClient for ServiceClient {
        async fn get_response(
            &self,
            _messages: Vec<Message>,
            options: ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen_options.lock().unwrap().push(options);
            let mut resp = ChatResponse::from_text("ok");
            resp.conversation_id = Some("conv-1".to_string());
            Ok(resp)
        }
        async fn get_streaming_response(
            &self,
            messages: Vec<Message>,
            options: ChatOptions,
        ) -> Result<agent_framework_core::client::ChatStream> {
            let resp = self.get_response(messages, options).await?;
            let mut update = ChatResponseUpdate::text(resp.text());
            update.conversation_id = Some("conv-1".to_string());
            Ok(Box::pin(futures::stream::iter(vec![Ok(update)])))
        }
    }

    let seen_options = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder(ServiceClient {
        seen_options: seen_options.clone(),
    })
    .name("svc")
    .build();

    // Fresh agent threads start with an (empty) local store; the returned
    // service conversation id must still be adopted.
    let mut thread = agent.create_session();
    let response = agent
        .run(vec![Message::user("hi")], Some(&mut thread))
        .await
        .unwrap();
    assert_eq!(response.conversation_id.as_deref(), Some("conv-1"));
    assert_eq!(thread.service_session_id(), Some("conv-1"));

    // Turn two must carry the id back to the service.
    agent
        .run(vec![Message::user("again")], Some(&mut thread))
        .await
        .unwrap();
    let opts = seen_options.lock().unwrap();
    assert_eq!(opts.len(), 2);
    assert_eq!(opts[0].conversation_id, None);
    assert_eq!(opts[1].conversation_id.as_deref(), Some("conv-1"));
}

// ===========================================================================
// Task 5: as_tool name sanitization
// ===========================================================================

#[tokio::test]
async fn as_tool_sanitizes_agent_name() {
    let agent = Agent::builder(MockClient::new(vec![]))
        .name("My Weather Agent!! v2")
        .build();
    // Spaces/punctuation -> underscores, collapsed, trimmed.
    let tool = agent.as_tool(AsToolOptions::new());
    assert_eq!(tool.name, "My_Weather_Agent_v2");

    // An explicit name is used verbatim (mirrors Python `name or sanitize`).
    let tool2 = agent.as_tool(AsToolOptions::new().name("explicit name"));
    assert_eq!(tool2.name, "explicit name");

    // Leading digit gets an underscore prefix; all-invalid -> "agent".
    let numeric = Agent::builder(MockClient::new(vec![]))
        .name("9lives")
        .build();
    assert_eq!(numeric.as_tool(AsToolOptions::new()).name, "_9lives");
    let junk = Agent::builder(MockClient::new(vec![])).name("@@@").build();
    assert_eq!(junk.as_tool(AsToolOptions::new()).name, "agent");
}

// ===========================================================================
// Task 6: service-managed thread with no returned conversation id errors
// ===========================================================================

#[tokio::test]
async fn service_thread_without_conversation_id_errors() {
    // The client succeeds but returns no conversation id.
    let client = MockClient::new(vec![ChatResponse::from_text("hi")]);
    let agent = Agent::builder(client).name("svc").build();
    let mut thread = agent.create_session_with_service_id("svc-thread");
    let err = agent
        .run(vec![Message::user("hi")], Some(&mut thread))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::AgentExecution(_)));
    assert!(err
        .to_string()
        .contains("did not return a valid conversation id"));
}

// ===========================================================================
// Task 1: ContextProvider::before_run observes session/service session ids
// (upstream removed the `thread_created` hook entirely; the equivalent
// coverage is that `before_run` sees the correct `session_id` /
// `service_session_id` for a service-managed thread, and for a thread that
// newly adopts a service id mid-run).
// ===========================================================================

/// Echoes the request's conversation id back (keeps a service thread valid).
struct EchoServiceClient;
#[async_trait]
impl ChatClient for EchoServiceClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resp = ChatResponse::from_text("ok");
        resp.conversation_id = options.conversation_id.clone();
        Ok(resp)
    }
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let mut u = ChatResponseUpdate::text(resp.text());
        u.conversation_id = resp.conversation_id.clone();
        Ok(Box::pin(futures::stream::iter(vec![Ok(u)])))
    }
}

/// Returns a fresh conversation id for a previously-local thread to adopt.
struct AdoptServiceClient;
#[async_trait]
impl ChatClient for AdoptServiceClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resp = ChatResponse::from_text("ok");
        resp.conversation_id = Some("adopted-1".to_string());
        Ok(resp)
    }
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let mut u = ChatResponseUpdate::text(resp.text());
        u.conversation_id = Some("adopted-1".to_string());
        Ok(Box::pin(futures::stream::iter(vec![Ok(u)])))
    }
}

#[tokio::test]
async fn before_run_observes_service_session_id_for_service_thread() {
    let provider = RecordingProvider::default();
    let ids = provider.service_session_ids.clone();
    let agent = Agent::builder(EchoServiceClient)
        .context_provider(Arc::new(provider))
        .build();

    let mut thread = agent.create_session_with_service_id("svc-1");
    agent
        .run(vec![Message::user("hi")], Some(&mut thread))
        .await
        .unwrap();

    // `before_run` observes `service_session_id` set to the thread's service
    // id (no thread_created hook any more).
    assert_eq!(ids.lock().unwrap().clone(), vec![Some("svc-1".to_string())]);
}

#[tokio::test]
async fn before_run_service_session_id_reflects_service_id_adopted_on_a_prior_run() {
    let provider = RecordingProvider::default();
    let ids = provider.service_session_ids.clone();
    let agent = Agent::builder(AdoptServiceClient)
        .context_provider(Arc::new(provider))
        .build();

    // First run: a fresh local thread has no service session id yet.
    let mut thread = agent.create_session();
    agent
        .run(vec![Message::user("hi")], Some(&mut thread))
        .await
        .unwrap();
    assert_eq!(thread.service_session_id(), Some("adopted-1"));

    // Second run: the thread adopted a service id from the first run, so
    // `before_run` now observes it.
    agent
        .run(vec![Message::user("again")], Some(&mut thread))
        .await
        .unwrap();

    assert_eq!(
        ids.lock().unwrap().clone(),
        vec![None, Some("adopted-1".to_string())]
    );
}

// ===========================================================================
// Task 2: ContextProvider::after_run observes failures
// ===========================================================================

struct FailingClient;
#[async_trait]
impl ChatClient for FailingClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Err(Error::service("boom"))
    }
    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Err(Error::service("boom"))
    }
}

#[tokio::test]
async fn after_run_hook_observes_failure() {
    let provider = RecordingProvider::default();
    let invoked = provider.invoked.clone();
    let invoked_error = provider.invoked_error.clone();
    let agent = Agent::builder(FailingClient)
        .context_provider(Arc::new(provider))
        .build();

    let err = agent.run_once("hi").await.unwrap_err();
    assert!(err.to_string().contains("boom"));
    assert!(
        *invoked.lock().unwrap(),
        "after_run fired on the failure path"
    );
    let recorded = invoked_error.lock().unwrap().clone();
    assert!(
        recorded.is_some_and(|m| m.contains("boom")),
        "provider observed the run error"
    );
}

#[tokio::test]
async fn after_run_hook_observes_streaming_failure() {
    let provider = RecordingProvider::default();
    let invoked_error = provider.invoked_error.clone();
    let agent = Agent::builder(FailingClient)
        .context_provider(Arc::new(provider))
        .build();

    let err = agent.run_stream("hi", None, None).await.err().unwrap();
    assert!(err.to_string().contains("boom"));
    assert!(
        invoked_error
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|m| m.contains("boom")),
        "provider observed the streaming failure"
    );
}

// ===========================================================================
// Task 3: structured-output value auto-population
// ===========================================================================

#[tokio::test]
async fn structured_output_value_autofilled_on_agent_run() {
    let client = MockClient::new(vec![ChatResponse::from_text("{\"city\": \"Paris\"}")]);
    let agent = Agent::builder(client)
        .response_format(ResponseFormat::JsonObject)
        .build();
    let resp = agent.run_once("where?").await.unwrap();
    assert_eq!(resp.value, Some(json!({"city": "Paris"})));
}

#[tokio::test]
async fn structured_output_value_tolerates_non_json() {
    let client = MockClient::new(vec![ChatResponse::from_text("sorry, no idea")]);
    let agent = Agent::builder(client)
        .response_format(ResponseFormat::JsonObject)
        .build();
    let resp = agent.run_once("where?").await.unwrap();
    assert_eq!(resp.value, None);
}

#[tokio::test]
async fn structured_output_value_autofilled_on_bare_client() {
    use agent_framework_core::client::FunctionInvokingChatClient;
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![ChatResponse::from_text(
        "{\"n\": 5}",
    )]));
    let mut opts = ChatOptions::new();
    opts.response_format = Some(ResponseFormat::JsonObject);
    let resp = client
        .get_response(vec![Message::user("x")], opts)
        .await
        .unwrap();
    assert_eq!(resp.value, Some(json!({"n": 5})));
}

// ===========================================================================
// Task 7: session + history-provider persistence (agent-level)
// ===========================================================================

#[tokio::test]
async fn create_session_eagerly_attaches_a_history_provider() {
    // A fresh local session already carries a history provider (rather than
    // deferring attachment to the first `run`), so that a clone taken before
    // streaming observes the post-run write-back.
    let agent = Agent::builder(MockClient::new(vec![])).build();
    let session = agent.create_session();
    assert_eq!(session.context_providers.len(), 1);
    assert!(session.context_providers[0].is_history_provider());

    // A service-managed session gets no history provider (the service owns
    // history server-side).
    let svc_session = agent.create_session_with_service_id("svc-1");
    assert!(svc_session.context_providers.is_empty());
}

#[tokio::test]
async fn agent_session_and_history_provider_round_trip() {
    // `AgentSession::to_dict` and `InMemoryHistoryProvider::to_dict` are
    // serialized independently -- history is deliberately NOT part of the
    // session's own wire shape any more.
    let agent = Agent::builder(MockClient::new(vec![])).build();
    let mut session = AgentSession::new();
    let history = InMemoryHistoryProvider::with_messages(vec![
        Message::user("hi"),
        Message::assistant("hello"),
    ]);
    session.context_providers.push(Arc::new(history.clone()));

    let session_state = session.to_dict();
    let history_state = history.to_dict();

    let restored_session = agent.session_from_dict(&session_state).unwrap();
    assert_eq!(restored_session.session_id(), session.session_id());
    // `context_providers` (including the history provider) are not restored
    // by `AgentSession::from_dict`; callers reattach them explicitly.
    assert!(restored_session.context_providers.is_empty());

    let restored_history = InMemoryHistoryProvider::from_dict(&history_state).unwrap();
    let msgs = restored_history.list_messages();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].text(), "hi");
    assert_eq!(msgs[1].text(), "hello");
}

#[tokio::test]
async fn agent_create_session_with_service_id() {
    let agent = Agent::builder(MockClient::new(vec![])).build();
    let thread = agent.create_session_with_service_id("svc-9");
    assert_eq!(thread.service_session_id(), Some("svc-9"));
    // A service-managed session has no auto-attached history provider (the
    // service owns history server-side).
    assert!(thread.context_providers.is_empty());
}

// ---------------------------------------------------------------------------
// GAP 1.4 — trait-level streaming; GAP 1.5 — per-run options; Task 3 — declaration-only
// ---------------------------------------------------------------------------

/// A client that streams a fixed list of text deltas (real incremental
/// streaming, distinct from `MockClient`'s per-message replay).
#[derive(Clone)]
struct DeltaClient {
    deltas: Vec<String>,
}

#[async_trait]
impl ChatClient for DeltaClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(self.deltas.concat()))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        let updates: Vec<Result<ChatResponseUpdate>> = self
            .deltas
            .iter()
            .map(|d| Ok(ChatResponseUpdate::text(d.clone())))
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }
}

/// A client that records every `ChatOptions` it is handed (to assert per-run
/// option precedence and per-run tool visibility).
#[derive(Clone)]
struct RecordingClient {
    seen: Arc<Mutex<Vec<ChatOptions>>>,
}

impl RecordingClient {
    fn new() -> Self {
        Self {
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl ChatClient for RecordingClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(options);
        Ok(ChatResponse::from_text("ok"))
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
}

fn declaration_only_tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: String::new(),
        parameters: json!({ "type": "object", "properties": {} }),
        kind: ToolKind::Function,
        approval_mode: ApprovalMode::NeverRequire,
        executor: None,
    }
}

#[tokio::test]
async fn trait_default_run_stream_buffers_for_minimal_agent() {
    // A minimal custom agent implementing only `run` + `id` gets the trait's
    // default buffered `run_stream` for free.
    struct EchoAgent;
    #[async_trait]
    impl SupportsAgentRun for EchoAgent {
        async fn run(
            &self,
            messages: Vec<Message>,
            _thread: Option<&mut AgentSession>,
        ) -> Result<AgentResponse> {
            let text = messages.last().map(Message::text).unwrap_or_default();
            Ok(AgentResponse {
                messages: vec![Message::assistant(format!("echo: {text}"))],
                ..Default::default()
            })
        }
        fn id(&self) -> &str {
            "echo"
        }
    }

    let agent = EchoAgent;
    let mut stream = SupportsAgentRun::run_stream(&agent, vec![Message::user("hi")], None, None)
        .await
        .unwrap();
    let mut text = String::new();
    let mut count = 0;
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text());
        count += 1;
    }
    assert_eq!(text, "echo: hi");
    assert_eq!(count, 1, "one buffered update per response message");
}

#[tokio::test]
async fn chat_agent_trait_stream_yields_real_deltas() {
    // Agent's real streaming override forwards one update per model delta.
    let client = DeltaClient {
        deltas: vec!["Hel".into(), "lo ".into(), "world".into()],
    };
    let agent = Agent::builder(client).build();
    let mut stream = SupportsAgentRun::run_stream(&agent, vec![Message::user("hi")], None, None)
        .await
        .unwrap();
    let mut deltas = Vec::new();
    while let Some(update) = stream.next().await {
        deltas.push(update.unwrap().text());
    }
    assert_eq!(deltas.len(), 3, "one update per streamed delta");
    assert_eq!(deltas.concat(), "Hello world");
}

#[tokio::test]
async fn per_run_chat_options_override_agent_defaults() {
    // SupportsAgentRun default temperature 0.2; a per-run override of 0.9 must win, matching
    // Python's `run_chat_options & ChatOptions(...)`.
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let agent = Agent::builder(client).temperature(0.2).build();

    let options = AgentRunOptions::new().with_chat_options(ChatOptions {
        temperature: Some(0.9),
        ..Default::default()
    });
    let _ = agent
        .run_with_options(vec![Message::user("hi")], None, options)
        .await
        .unwrap();

    let recorded = seen.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(
        recorded[0].temperature,
        Some(0.9),
        "per-run temperature wins over the agent default"
    );
}

#[tokio::test]
async fn per_run_tools_are_visible_only_for_that_call() {
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let agent = Agent::builder(client)
        .tool(declaration_only_tool("base_tool"))
        .build();

    // Run 1: inject an extra per-run tool.
    let options = AgentRunOptions::new().with_tool(declaration_only_tool("run_tool"));
    let _ = agent
        .run_with_options(vec![Message::user("hi")], None, options)
        .await
        .unwrap();
    // Run 2: no per-run tools.
    let _ = agent.run(vec![Message::user("hi")], None).await.unwrap();

    let recorded = seen.lock().unwrap();
    let names =
        |i: usize| -> Vec<String> { recorded[i].tools.iter().map(|t| t.name.clone()).collect() };
    assert!(names(0).contains(&"base_tool".to_string()));
    assert!(
        names(0).contains(&"run_tool".to_string()),
        "per-run tool visible for that call"
    );
    assert!(
        !names(1).contains(&"run_tool".to_string()),
        "per-run tool must NOT leak into the next call"
    );
    assert!(names(1).contains(&"base_tool".to_string()));
}

#[tokio::test]
async fn declaration_only_tool_call_is_returned_to_caller() {
    // The model calls a known-but-declaration-only tool; the loop must return
    // the response with the FunctionCallContent intact (frontend-tool pattern).
    let call = FunctionCallContent::new(
        "c1",
        "frontend_tool",
        Some(FunctionArguments::Raw(json!({"x": 1}).to_string())),
    );
    let resp = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![resp]));
    // A real executable tool is present so the invocation loop actually engages;
    // the model instead calls the declaration-only tool, which the loop must
    // return unexecuted rather than error on.
    let real_tool = FunctionTool::new(
        "real",
        "",
        json!({ "type": "object", "properties": {} }),
        |_args: Value| async { Ok(Value::Null) },
    )
    .into_definition();
    let options = ChatOptions {
        tools: vec![real_tool, declaration_only_tool("frontend_tool")],
        ..Default::default()
    };
    let out = client
        .get_response(vec![Message::user("go")], options)
        .await
        .unwrap();

    let calls = out.function_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "frontend_tool");
    let has_result = out
        .messages
        .iter()
        .flat_map(|m| &m.contents)
        .any(|c| matches!(c, Content::FunctionResult(_)));
    assert!(!has_result, "declaration-only call must not be executed");
}

#[tokio::test]
async fn unknown_tool_call_is_not_declaration_only() {
    // A genuinely unknown tool name keeps the not-found error behavior (an
    // error result, loop continues), NOT the declaration-only early return.
    let call = FunctionCallContent::new("c1", "ghost_tool", None);
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse::from_text("done");
    // A real executable tool is present so the loop engages, but the model calls
    // a different, unknown tool.
    let real_tool = FunctionTool::new(
        "real",
        "",
        json!({ "type": "object", "properties": {} }),
        |_args: Value| async { Ok(Value::Null) },
    )
    .into_definition();
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![ask, answer]));
    let options = ChatOptions {
        tools: vec![real_tool],
        ..Default::default()
    };
    let out = client
        .get_response(vec![Message::user("go")], options)
        .await
        .unwrap();

    let has_error_result = out
        .messages
        .iter()
        .flat_map(|m| &m.contents)
        .any(|c| matches!(c, Content::FunctionResult(fr) if fr.exception.is_some()));
    assert!(
        has_error_result,
        "unknown tool yields an error result, not a declaration-only return"
    );
    assert_eq!(out.text(), "done");
}

// -- ToolSource: dynamic tool resolution per agent run --------------------

/// A [`ToolSource`] that returns a scripted sequence of tool lists, one per
/// `resolve_tools` call (the last is repeated once the script is
/// exhausted). Stands in for an MCP server whose catalog changes between
/// runs (e.g. after a `notifications/tools/list_changed`), without any real
/// transport.
struct StubToolSource {
    name: String,
    call_count: Arc<Mutex<usize>>,
    responses: Vec<Vec<ToolDefinition>>,
}

impl StubToolSource {
    fn new(name: &str, responses: Vec<Vec<ToolDefinition>>) -> Self {
        Self {
            name: name.to_string(),
            call_count: Arc::new(Mutex::new(0)),
            responses,
        }
    }
}

#[async_trait]
impl ToolSource for StubToolSource {
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
        let mut count = self.call_count.lock().unwrap();
        let idx = (*count).min(self.responses.len().saturating_sub(1));
        *count += 1;
        Ok(self.responses.get(idx).cloned().unwrap_or_default())
    }

    fn source_name(&self) -> &str {
        &self.name
    }
}

/// A [`ToolSource`] whose `resolve_tools` always fails — stands in for an
/// MCP server that is unreachable at run time.
struct FailingToolSource;

#[async_trait]
impl ToolSource for FailingToolSource {
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
        Err(Error::service("mcp server unreachable"))
    }
    fn source_name(&self) -> &str {
        "failing-source"
    }
}

#[tokio::test]
async fn tool_source_resolved_fresh_each_run_sees_catalog_change() {
    // Simulates a server whose tool list grows between runs (e.g. after a
    // notifications/tools/list_changed): the agent must re-resolve the
    // source on every run rather than resolving it once at build time.
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let source = Arc::new(StubToolSource::new(
        "mcp",
        vec![
            vec![declaration_only_tool("tool_a")],
            vec![
                declaration_only_tool("tool_a"),
                declaration_only_tool("tool_b"),
            ],
        ],
    ));
    let agent = Agent::builder(client).tool_source(source).build();

    let _ = agent.run(vec![Message::user("hi")], None).await.unwrap();
    let _ = agent
        .run(vec![Message::user("hi again")], None)
        .await
        .unwrap();

    let recorded = seen.lock().unwrap();
    assert_eq!(recorded.len(), 2);
    let names =
        |i: usize| -> Vec<String> { recorded[i].tools.iter().map(|t| t.name.clone()).collect() };
    assert_eq!(names(0), vec!["tool_a".to_string()]);
    assert_eq!(
        names(1),
        vec!["tool_a".to_string(), "tool_b".to_string()],
        "second run must see the source's updated catalog"
    );
}

#[tokio::test]
async fn tool_source_dedup_explicit_tool_wins_over_source_tool() {
    // The agent's own build-time tool named "shared" must win over a
    // same-named tool produced by a tool source (dedup against "explicit
    // tools", first wins).
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let explicit = ToolDefinition {
        description: "explicit".to_string(),
        ..declaration_only_tool("shared")
    };
    let source_tool = ToolDefinition {
        description: "from-source".to_string(),
        ..declaration_only_tool("shared")
    };
    let source = Arc::new(StubToolSource::new("mcp", vec![vec![source_tool]]));
    let agent = Agent::builder(client)
        .tool(explicit)
        .tool_source(source)
        .build();

    let _ = agent.run(vec![Message::user("hi")], None).await.unwrap();

    let recorded = seen.lock().unwrap();
    let shared: Vec<_> = recorded[0]
        .tools
        .iter()
        .filter(|t| t.name == "shared")
        .collect();
    assert_eq!(
        shared.len(),
        1,
        "only one 'shared' tool should survive dedup"
    );
    assert_eq!(
        shared[0].description, "explicit",
        "the explicit tool wins over the source's same-named tool"
    );
}

#[tokio::test]
async fn tool_source_dedup_first_registered_source_wins() {
    // Two sources both produce a "shared" tool; the first-registered
    // source's version must win.
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let first = Arc::new(StubToolSource::new(
        "first",
        vec![vec![ToolDefinition {
            description: "from-first".to_string(),
            ..declaration_only_tool("shared")
        }]],
    ));
    let second = Arc::new(StubToolSource::new(
        "second",
        vec![vec![ToolDefinition {
            description: "from-second".to_string(),
            ..declaration_only_tool("shared")
        }]],
    ));
    let agent = Agent::builder(client)
        .tool_source(first)
        .tool_source(second)
        .build();

    let _ = agent.run(vec![Message::user("hi")], None).await.unwrap();

    let recorded = seen.lock().unwrap();
    let shared: Vec<_> = recorded[0]
        .tools
        .iter()
        .filter(|t| t.name == "shared")
        .collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].description, "from-first");
}

#[tokio::test]
async fn tool_source_dedup_against_per_run_additional_tools() {
    // A per-run `additional_tools` entry must also win over a same-named
    // tool from a source (sources are resolved last).
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let source_tool = ToolDefinition {
        description: "from-source".to_string(),
        ..declaration_only_tool("shared")
    };
    let source = Arc::new(StubToolSource::new("mcp", vec![vec![source_tool]]));
    let agent = Agent::builder(client).tool_source(source).build();

    let per_run_tool = ToolDefinition {
        description: "per-run".to_string(),
        ..declaration_only_tool("shared")
    };
    let options = AgentRunOptions::new().with_tool(per_run_tool);
    let _ = agent
        .run_with_options(vec![Message::user("hi")], None, options)
        .await
        .unwrap();

    let recorded = seen.lock().unwrap();
    let shared: Vec<_> = recorded[0]
        .tools
        .iter()
        .filter(|t| t.name == "shared")
        .collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].description, "per-run");
}

#[tokio::test]
async fn failing_tool_source_propagates_error_out_of_run() {
    // Mirrors the Python reference's run()/run_stream(), which do not catch
    // a failure raised while connecting to an MCPTool at run time -- it
    // propagates out of the whole run rather than being swallowed.
    let client = MockClient::new(vec![ChatResponse::from_text("should not be reached")]);
    let agent = Agent::builder(client)
        .tool_source(Arc::new(FailingToolSource))
        .build();

    let err = agent
        .run(vec![Message::user("hi")], None)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Service(_)));
}

#[tokio::test]
async fn tool_source_tool_is_invokable_by_the_function_loop() {
    // A tool resolved from a ToolSource must be genuinely usable, not just
    // present in the assembled ChatOptions: the model calls it and the
    // function-invocation loop executes it like any other tool.
    let call = FunctionCallContent::new(
        "call_1",
        "double",
        Some(FunctionArguments::Raw(json!({"n": 21}).to_string())),
    );
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let answer = ChatResponse::from_text("42");
    let client = MockClient::new(vec![ask, answer]);

    let double = FunctionTool::new(
        "double",
        "Double a number.",
        json!({
            "type": "object",
            "properties": { "n": {"type": "integer"} },
            "required": ["n"]
        }),
        |args: Value| async move {
            let n = args["n"].as_i64().unwrap_or(0);
            Ok(json!(n * 2))
        },
    )
    .into_definition();
    let source = Arc::new(StubToolSource::new("mcp", vec![vec![double]]));

    let agent = Agent::builder(client).tool_source(source).build();
    let response = agent.run_once("double 21").await.unwrap();
    assert!(response.text().contains("42"), "got: {}", response.text());
    assert!(response.messages.iter().any(|m| m.role == Role::tool()
        && m.contents
            .iter()
            .any(|c| matches!(c, Content::FunctionResult(_)))));
}

// region: as_tool session propagation (upstream `propagate_session`, with the
// child-session isolation semantics of microsoft/agent-framework#5875)

/// A context provider that records the session identity (`session_id` +
/// `service_session_id`) of every run it participates in.
/// `(session_id, service_session_id)` as observed by a run.
type SeenSessionIdentity = (Option<String>, Option<String>);

#[derive(Default, Clone)]
struct SessionIdentityRecorder {
    seen: Arc<Mutex<Vec<SeenSessionIdentity>>>,
}

#[async_trait]
impl ContextProvider for SessionIdentityRecorder {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        self.seen
            .lock()
            .unwrap()
            .push((ctx.session_id.clone(), ctx.service_session_id.clone()));
        Ok(())
    }
}

/// A scripted coordinator client whose first response calls the `sub` tool
/// and whose second response is the final answer. `conversation_id` is echoed
/// on both responses (a service-managed session requires the service to
/// return one).
fn coordinator_client_calling_sub(conversation_id: Option<&str>) -> MockClient {
    let call = FunctionCallContent::new(
        "call_1",
        "sub",
        Some(FunctionArguments::Raw("{\"task\":\"do the thing\"}".into())),
    );
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        conversation_id: conversation_id.map(str::to_string),
        ..Default::default()
    };
    let done = ChatResponse {
        conversation_id: conversation_id.map(str::to_string),
        ..ChatResponse::from_text("done")
    };
    MockClient::new(vec![ask, done])
}

#[tokio::test]
async fn as_tool_propagate_session_shares_identity_and_isolates_service_pointer() {
    let sub_recorder = SessionIdentityRecorder::default();
    let sub_seen = sub_recorder.seen.clone();
    let sub_client = MockClient::new(vec![ChatResponse::from_text("sub answer")]);
    let sub_options = sub_client.seen_options.clone();
    let sub = Agent::builder(sub_client)
        .name("sub")
        .context_provider(Arc::new(sub_recorder))
        .build();

    let coordinator_client = coordinator_client_calling_sub(Some("svc-parent"));
    let coordinator_options = coordinator_client.seen_options.clone();
    let coordinator = Agent::builder(coordinator_client)
        .tool(sub.as_tool(AsToolOptions::new().name("sub").propagate_session(true)))
        .build();

    // A service-managed parent session: its server-side conversation pointer
    // must NOT leak into the sub-agent's own service calls.
    let mut parent = AgentSession::service("svc-parent");
    let parent_id = parent.session_id().to_string();

    let response = coordinator
        .run(vec![Message::user("go")], Some(&mut parent))
        .await
        .unwrap();
    assert_eq!(response.text(), "done");

    // The sub-agent ran on a *child* of the parent session: same session_id…
    let seen = sub_seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "the sub-agent ran exactly once");
    assert_eq!(
        seen[0].0.as_deref(),
        Some(parent_id.as_str()),
        "the parent's session identity must propagate to the sub-agent"
    );
    // …but an isolated (cleared) service_session_id.
    assert_eq!(
        seen[0].1, None,
        "the parent's service conversation pointer must not leak to the sub-agent"
    );
    // Confirmed at the wire level too: the sub-agent's provider client saw no
    // conversation id, while the coordinator's did.
    let sub_convs: Vec<Option<String>> = sub_options
        .lock()
        .unwrap()
        .iter()
        .map(|o| o.conversation_id.clone())
        .collect();
    assert!(sub_convs.iter().all(Option::is_none), "got: {sub_convs:?}");
    assert_eq!(
        coordinator_options.lock().unwrap()[0]
            .conversation_id
            .as_deref(),
        Some("svc-parent")
    );
    // The parent's own pointer is untouched.
    assert_eq!(parent.service_session_id(), Some("svc-parent"));
}

#[tokio::test]
async fn as_tool_without_propagate_session_runs_on_a_fresh_session() {
    let sub_recorder = SessionIdentityRecorder::default();
    let sub_seen = sub_recorder.seen.clone();
    let sub = Agent::builder(MockClient::new(vec![ChatResponse::from_text("sub answer")]))
        .name("sub")
        .context_provider(Arc::new(sub_recorder))
        .build();

    let coordinator = Agent::builder(coordinator_client_calling_sub(None))
        .tool(sub.as_tool(AsToolOptions::new().name("sub")))
        .build();

    let mut parent = AgentSession::new();
    let parent_id = parent.session_id().to_string();
    coordinator
        .run(vec![Message::user("go")], Some(&mut parent))
        .await
        .unwrap();

    let seen = sub_seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_ne!(
        seen[0].0.as_deref(),
        Some(parent_id.as_str()),
        "without propagate_session the sub-agent must get a fresh session"
    );
}

#[tokio::test]
async fn as_tool_state_written_by_the_sub_agent_run_is_visible_on_the_parent() {
    // The sub-agent's own tool writes into the (propagated) session state via
    // the invocation context; the parent must observe the write, because the
    // child session shares the parent's state bag by reference.
    struct StateWriter;
    #[async_trait]
    impl Tool for StateWriter {
        fn name(&self) -> &str {
            "remember"
        }
        fn description(&self) -> &str {
            "remember a fact"
        }
        fn parameters_schema(&self) -> Value {
            json!({ "type": "object", "properties": {} })
        }
        async fn invoke(&self, _arguments: Value) -> Result<Value> {
            Ok(Value::Null)
        }
        async fn invoke_in_context(
            &self,
            _arguments: Value,
            ctx: &FunctionInvocationContext,
        ) -> Result<Value> {
            let session = ctx.session.as_ref().expect("session propagated to tool");
            session.state.insert("fact", json!("blue"));
            Ok(json!("remembered"))
        }
    }

    let sub_call = FunctionCallContent::new(
        "call_sub_1",
        "remember",
        Some(FunctionArguments::Raw("{}".into())),
    );
    let sub_ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(sub_call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let sub = Agent::builder(MockClient::new(vec![
        sub_ask,
        ChatResponse::from_text("sub done"),
    ]))
    .name("sub")
    .tool(ToolDefinition::from_tool(Arc::new(StateWriter)))
    .build();

    let coordinator = Agent::builder(coordinator_client_calling_sub(None))
        .tool(sub.as_tool(AsToolOptions::new().name("sub").propagate_session(true)))
        .build();

    let mut parent = AgentSession::new();
    coordinator
        .run(vec![Message::user("go")], Some(&mut parent))
        .await
        .unwrap();

    assert_eq!(
        parent.state.get("fact"),
        Some(json!("blue")),
        "state written during the sub-agent's run must be visible on the parent session"
    );
}

#[tokio::test]
async fn as_tool_stream_callback_observes_sub_agent_updates() {
    let sub = Agent::builder(MockClient::new(vec![ChatResponse::from_text(
        "sub streamed answer",
    )]))
    .name("sub")
    .build();

    let streamed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = streamed.clone();
    let coordinator = Agent::builder(coordinator_client_calling_sub(None))
        .tool(
            sub.as_tool(AsToolOptions::new().name("sub").stream_callback(Arc::new(
                move |update: &AgentResponseUpdate| {
                    sink.lock().unwrap().push(update.text());
                },
            ))),
        )
        .build();

    let response = coordinator.run_once("go").await.unwrap();
    assert_eq!(response.text(), "done");
    let streamed = streamed.lock().unwrap();
    assert!(!streamed.is_empty(), "the stream callback never fired");
    assert_eq!(streamed.concat(), "sub streamed answer");
}

#[tokio::test]
async fn as_tool_approval_mode_gates_the_delegated_call() {
    let sub = Agent::builder(MockClient::new(vec![])).name("sub").build();
    let tool = sub.as_tool(
        AsToolOptions::new()
            .name("sub")
            .approval_mode(ApprovalMode::AlwaysRequire),
    );
    assert!(tool.requires_approval());

    // The coordinator's run surfaces an approval request instead of executing.
    let coordinator = Agent::builder(coordinator_client_calling_sub(None))
        .tool(tool)
        .build();
    let response = coordinator.run_once("go").await.unwrap();
    assert!(
        !response.user_input_requests().is_empty(),
        "an approval-gated agent tool must surface an approval request"
    );
}

// endregion

// region: progressive tool exposure (upstream FunctionInvocationContext.tools)

/// A tool that mutates the run's live tool list from inside its invocation.
struct ToolListMutator {
    name: String,
    add: Option<ToolDefinition>,
    remove: Vec<String>,
}

#[async_trait]
impl Tool for ToolListMutator {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "mutates the live tool list"
    }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn invoke(&self, _arguments: Value) -> Result<Value> {
        Ok(Value::Null)
    }
    async fn invoke_in_context(
        &self,
        _arguments: Value,
        ctx: &FunctionInvocationContext,
    ) -> Result<Value> {
        if let Some(tool) = &self.add {
            ctx.add_tools([tool.clone()])?;
        }
        ctx.remove_tools(self.remove.iter().map(String::as_str))?;
        Ok(json!("mutated"))
    }
}

fn noop_tool(name: &str) -> ToolDefinition {
    FunctionTool::new(
        name,
        "does nothing",
        json!({ "type": "object", "properties": {} }),
        |_args| async move { Ok(Value::Null) },
    )
    .into_definition()
}

fn tool_call_response(tool: &str) -> ChatResponse {
    let call = FunctionCallContent::new(
        format!("call_{tool}"),
        tool,
        Some(FunctionArguments::Raw("{}".into())),
    );
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

#[tokio::test]
async fn tool_added_mid_run_is_exposed_on_the_next_iteration() {
    // Iteration 1 calls `unlock`, whose execution adds `secret`; iteration 2
    // must see `secret` in the wire tool list (and not before).
    let client = MockClient::new(vec![
        tool_call_response("unlock"),
        ChatResponse::from_text("done"),
    ]);
    let options_seen = client.seen_options.clone();

    let unlock = ToolDefinition::from_tool(Arc::new(ToolListMutator {
        name: "unlock".into(),
        add: Some(noop_tool("secret")),
        remove: vec![],
    }));
    let agent = Agent::builder(client).tool(unlock).build();
    let response = agent.run_once("go").await.unwrap();
    assert_eq!(response.text(), "done");

    let seen = options_seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    let names =
        |o: &ChatOptions| -> Vec<String> { o.tools.iter().map(|t| t.name.clone()).collect() };
    assert!(
        !names(&seen[0]).contains(&"secret".to_string()),
        "iteration 1 must not yet see the added tool: {:?}",
        names(&seen[0])
    );
    assert!(
        names(&seen[1]).contains(&"secret".to_string()),
        "iteration 2 must see the added tool: {:?}",
        names(&seen[1])
    );
}

#[tokio::test]
async fn tool_removed_mid_run_disappears_from_the_next_iteration() {
    let client = MockClient::new(vec![
        tool_call_response("cleanup"),
        ChatResponse::from_text("done"),
    ]);
    let options_seen = client.seen_options.clone();

    let cleanup = ToolDefinition::from_tool(Arc::new(ToolListMutator {
        name: "cleanup".into(),
        add: None,
        remove: vec!["obsolete".into()],
    }));
    let agent = Agent::builder(client)
        .tool(cleanup)
        .tool(noop_tool("obsolete"))
        .build();
    agent.run_once("go").await.unwrap();

    let seen = options_seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen[0].tools.iter().any(|t| t.name == "obsolete"));
    assert!(
        !seen[1].tools.iter().any(|t| t.name == "obsolete"),
        "iteration 2 must not see the removed tool"
    );
}

#[tokio::test]
async fn adding_a_duplicate_tool_name_errors_and_leaves_the_list_unchanged() {
    let list = agent_framework_core::middleware::LiveToolList::new(vec![noop_tool("existing")]);
    let err = list
        .add_tools([noop_tool("existing"), noop_tool("fresh")])
        .unwrap_err();
    assert!(err.to_string().contains("existing"), "got: {err}");
    // Validation happens before mutation: the non-duplicate was not added.
    assert!(!list.contains("fresh"));
    assert!(list.contains("existing"));
}

#[tokio::test]
async fn tool_context_outside_a_run_has_no_live_tools() {
    let ctx = FunctionInvocationContext::new("f", json!({}));
    assert!(ctx.tools.is_none());
    assert!(ctx.add_tools([noop_tool("x")]).is_err());
    assert!(ctx.remove_tools(["x"]).is_err());
}

// endregion

// region: usage aggregation across the tool loop (upstream `UsageAggregator`,
// .NET #7539). The loop issues one model call per iteration and each reports
// only its own tokens, so the response it returns must carry their sum — not
// the final iteration's count alone.

/// The token counts for a single model call, used to script what each
/// iteration of the loop reports.
fn usage_of(input: u64, output: u64) -> UsageDetails {
    let mut usage = UsageDetails::new();
    usage.input_token_count = Some(input);
    usage.output_token_count = Some(output);
    usage.total_token_count = Some(input + output);
    usage
}

fn noop_call_with_usage(call_id: &str, usage: UsageDetails) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                call_id,
                "noop",
                Some(FunctionArguments::Raw("{}".into())),
            ))],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        usage_details: Some(usage),
        ..Default::default()
    }
}

fn noop_tool_definition() -> ToolDefinition {
    FunctionTool::new(
        "noop",
        "No-op.",
        json!({"type":"object"}),
        |_args| async move { Ok(json!("ok")) },
    )
    .into_definition()
}

#[tokio::test]
async fn tool_loop_sums_usage_across_every_iteration() {
    // Two tool-calling iterations, then a final answer: 3 model calls total.
    let answer = ChatResponse {
        usage_details: Some(usage_of(300, 30)),
        ..ChatResponse::from_text("done")
    };
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![
        noop_call_with_usage("call_1", usage_of(100, 10)),
        noop_call_with_usage("call_2", usage_of(200, 20)),
        answer,
    ]));

    let response = client
        .get_response(
            vec![Message::user("go")],
            ChatOptions::new().with_tool(noop_tool_definition()),
        )
        .await
        .unwrap();

    let usage = response.usage_details.expect("usage should be reported");
    assert_eq!(usage.input_token_count, Some(600));
    assert_eq!(usage.output_token_count, Some(60));
    assert_eq!(usage.total_token_count, Some(660));
}

#[tokio::test]
async fn tool_loop_usage_aggregate_survives_an_approval_pause() {
    // The approval gate returns mid-loop. A free tool runs first, so there is a
    // completed iteration whose usage the pause must carry out with it rather
    // than discard — reporting only the gated turn would under-count the run.
    let counter = Arc::new(Mutex::new(0));
    let mut gated = secret_call();
    gated.usage_details = Some(usage_of(200, 20));
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![
        noop_call_with_usage("call_0", usage_of(100, 10)),
        gated,
    ]));
    let options = ChatOptions::new()
        .with_tool(noop_tool_definition())
        .with_tool(approval_tool(counter));

    let response = client
        .get_response(vec![Message::user("what is the secret?")], options)
        .await
        .unwrap();

    assert_eq!(response.user_input_requests().len(), 1);
    let usage = response.usage_details.expect("usage should be reported");
    assert_eq!(usage.input_token_count, Some(300));
    assert_eq!(usage.output_token_count, Some(30));
}

#[tokio::test]
async fn tool_loop_reports_usage_from_the_iterations_that_had_it() {
    // A provider that reports usage on some turns but not others must not have
    // the reported turns erased by the silent ones, and a run where nothing was
    // reported must stay `None` rather than becoming a bogus zero.
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![
        noop_call_with_usage("call_1", usage_of(100, 10)),
        ChatResponse::from_text("done"),
    ]));
    let response = client
        .get_response(
            vec![Message::user("go")],
            ChatOptions::new().with_tool(noop_tool_definition()),
        )
        .await
        .unwrap();
    let usage = response
        .usage_details
        .expect("partial usage still reported");
    assert_eq!(usage.input_token_count, Some(100));
    assert_eq!(usage.output_token_count, Some(10));

    let mut silent_call = noop_call_with_usage("call_1", UsageDetails::new());
    silent_call.usage_details = None;
    let silent = FunctionInvokingChatClient::new(MockClient::new(vec![
        silent_call,
        ChatResponse::from_text("done"),
    ]));
    let response = silent
        .get_response(
            vec![Message::user("go")],
            ChatOptions::new().with_tool(noop_tool_definition()),
        )
        .await
        .unwrap();
    assert!(
        response.usage_details.is_none(),
        "no contributor reported usage, so none should be synthesized"
    );
}

// endregion

// region: function-invocation budgets (max_function_calls / max_duration_seconds)

/// A counting tool that always succeeds, for the budget tests.
fn counting_tool(counter: Arc<Mutex<u32>>) -> ToolDefinition {
    FunctionTool::new(
        "ping",
        "Return pong.",
        json!({ "type": "object", "properties": {} }),
        move |_args| {
            let counter = counter.clone();
            async move {
                *counter.lock().unwrap() += 1;
                Ok(json!("pong"))
            }
        },
    )
    .into_definition()
}

/// One assistant turn requesting `n` parallel `ping` calls.
fn ping_calls(n: usize) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            (0..n)
                .map(|i| {
                    Content::FunctionCall(FunctionCallContent::new(
                        format!("call_{i}"),
                        "ping",
                        Some(FunctionArguments::Raw("{}".into())),
                    ))
                })
                .collect(),
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

fn budgeted_client(
    responses: Vec<ChatResponse>,
    configure: impl FnOnce(&mut FunctionInvocationConfig),
) -> FunctionInvokingChatClient<MockClient> {
    let mut config = FunctionInvocationConfig::default();
    configure(&mut config);
    FunctionInvokingChatClient::new(MockClient::new(responses)).with_config(config)
}

/// A tool that records when each of its invocations is entered and left, so
/// a test can tell overlapping executions from sequential ones. It yields
/// once in the middle, which is where two concurrent invocations interleave.
fn overlap_tracking_tool(log: Arc<Mutex<Vec<&'static str>>>) -> ToolDefinition {
    FunctionTool::new(
        "ping",
        "Return pong.",
        json!({ "type": "object", "properties": {} }),
        move |_args| {
            let log = log.clone();
            async move {
                log.lock().unwrap().push("enter");
                tokio::task::yield_now().await;
                log.lock().unwrap().push("exit");
                Ok(json!("pong"))
            }
        },
    )
    .into_definition()
}

#[tokio::test]
async fn parallel_calls_overlap_by_default_and_serialize_when_asked() {
    // Concurrent (the default): both invocations are in flight at once, so
    // the second enters before the first leaves.
    let log = Arc::new(Mutex::new(Vec::new()));
    let client = budgeted_client(
        vec![ping_calls(2), ChatResponse::from_text("done")],
        |_config| {},
    );
    client
        .get_response(
            vec![Message::user("ping twice")],
            ChatOptions::new().with_tool(overlap_tracking_tool(log.clone())),
        )
        .await
        .unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        vec!["enter", "enter", "exit", "exit"],
        "parallel tool calls should overlap by default"
    );

    // Sequential: each invocation completes before the next begins, which is
    // the whole point — a tool with a side effect sees them in model order.
    let log = Arc::new(Mutex::new(Vec::new()));
    let client = budgeted_client(
        vec![ping_calls(2), ChatResponse::from_text("done")],
        |config| config.allow_concurrent_invocation = false,
    );
    let response = client
        .get_response(
            vec![Message::user("ping twice")],
            ChatOptions::new().with_tool(overlap_tracking_tool(log.clone())),
        )
        .await
        .unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        vec!["enter", "exit", "enter", "exit"],
        "with concurrency off no two invocations may overlap"
    );
    assert_eq!(response.text(), "done");
}

#[tokio::test]
async fn sequential_results_reach_the_model_in_model_order() {
    // Ordering of the *results* is not what sequencing buys — it holds
    // either way — but a regression that reordered them would be invisible
    // without this.
    let client = budgeted_client(
        vec![ping_calls(3), ChatResponse::from_text("done")],
        |config| config.allow_concurrent_invocation = false,
    );
    let counter = Arc::new(Mutex::new(0));
    let response = client
        .get_response(
            vec![Message::user("ping thrice")],
            ChatOptions::new().with_tool(counting_tool(counter.clone())),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "done");
    assert_eq!(*counter.lock().unwrap(), 3);
}

#[tokio::test]
async fn max_function_calls_disables_tools_and_still_answers() {
    let counter = Arc::new(Mutex::new(0));
    let inner = MockClient::new(vec![
        ping_calls(1),
        ping_calls(1),
        ChatResponse::from_text("done"),
    ]);
    let client =
        FunctionInvokingChatClient::new(inner.clone()).with_config(FunctionInvocationConfig {
            max_function_calls: Some(2),
            ..Default::default()
        });

    let response = client
        .get_response(
            vec![Message::user("ping twice")],
            ChatOptions::new().with_tool(counting_tool(counter.clone())),
        )
        .await
        .unwrap();

    // Reaching the limit is not a failure: the model is asked to answer with
    // what it has, and the caller gets that answer.
    assert_eq!(response.text(), "done");
    assert_eq!(
        *counter.lock().unwrap(),
        2,
        "exactly the budgeted calls ran"
    );
    // The third model call is the one made with tools off.
    let choices: Vec<_> = inner
        .all_options()
        .iter()
        .map(|o| o.tool_choice.clone())
        .collect();
    assert_eq!(choices.len(), 3);
    assert_eq!(choices[0], Some(ToolMode::Auto));
    assert_eq!(choices[1], Some(ToolMode::Auto));
    assert_eq!(
        choices[2],
        Some(ToolMode::None),
        "tools must be off once the call budget is spent"
    );
}

#[tokio::test]
async fn a_parallel_batch_completes_even_when_it_overshoots_the_budget() {
    // Documented as best-effort: the check is between batches, because a
    // half-executed batch would leave calls without results, which providers
    // reject.
    let counter = Arc::new(Mutex::new(0));
    let client = budgeted_client(
        vec![ping_calls(5), ChatResponse::from_text("done")],
        |config| config.max_function_calls = Some(2),
    );

    let response = client
        .get_response(
            vec![Message::user("ping")],
            ChatOptions::new().with_tool(counting_tool(counter.clone())),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "done");
    assert_eq!(*counter.lock().unwrap(), 5, "the whole batch ran");
}

#[tokio::test]
async fn an_unbudgeted_run_keeps_calling_tools() {
    // The negative control for both budget tests: the same script without a
    // budget executes every call the model asks for.
    let counter = Arc::new(Mutex::new(0));
    let inner = MockClient::new(vec![
        ping_calls(1),
        ping_calls(1),
        ping_calls(1),
        ChatResponse::from_text("done"),
    ]);
    let client = FunctionInvokingChatClient::new(inner.clone());

    let response = client
        .get_response(
            vec![Message::user("ping")],
            ChatOptions::new().with_tool(counting_tool(counter.clone())),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "done");
    assert_eq!(*counter.lock().unwrap(), 3);
    assert!(
        inner
            .all_options()
            .iter()
            .all(|o| o.tool_choice != Some(ToolMode::None)),
        "tools were never disabled without a budget"
    );
}

#[tokio::test]
async fn max_duration_seconds_stops_the_loop_after_a_slow_batch() {
    let slow = FunctionTool::new(
        "slow",
        "Take a while.",
        json!({ "type": "object", "properties": {} }),
        move |_args| async move {
            tokio::time::sleep(std::time::Duration::from_millis(60)).await;
            Ok(json!("ok"))
        },
    )
    .into_definition();
    let inner = MockClient::new(vec![
        ChatResponse {
            messages: vec![Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionCall(FunctionCallContent::new(
                    "call_0",
                    "slow",
                    Some(FunctionArguments::Raw("{}".into())),
                ))],
            )],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        },
        ChatResponse::from_text("out of time"),
    ]);
    let client =
        FunctionInvokingChatClient::new(inner.clone()).with_config(FunctionInvocationConfig {
            max_duration_seconds: Some(0.05),
            ..Default::default()
        });

    let response = client
        .get_response(
            vec![Message::user("go")],
            ChatOptions::new().with_tool(slow),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "out of time");
    assert_eq!(
        inner.all_options().last().unwrap().tool_choice,
        Some(ToolMode::None),
        "the wall-clock budget must disable tools"
    );
}

#[tokio::test]
async fn an_approval_resumed_past_the_budget_is_not_executed() {
    // The budget has to hold across the approval round trip: otherwise a run
    // that is out of budget still executes whatever a human approves later,
    // and an unattended approve-and-continue loop never hits a limit at all.
    let counter = Arc::new(Mutex::new(0));
    let tool = approval_tool(counter.clone());
    let client = budgeted_client(
        vec![secret_call(), ChatResponse::from_text("no budget left")],
        |config| config.max_duration_seconds = Some(0.05),
    );
    // The session is where the budget is parked between the two requests.
    let session = AgentSession::new();
    let mut options = ChatOptions::new().with_tool(tool);
    options.session = Some(session.clone());

    let resp1 = client
        .get_response(vec![Message::user("what is the secret?")], options.clone())
        .await
        .unwrap();
    let requests = resp1.user_input_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(*counter.lock().unwrap(), 0);

    // The human takes longer than the budget to answer.
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;

    let approval = requests[0].create_response(true);
    let mut conversation = vec![Message::user("what is the secret?")];
    conversation.extend(resp1.messages.clone());
    conversation.push(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(approval)],
    ));
    let resp2 = client.get_response(conversation, options).await.unwrap();

    assert_eq!(resp2.text(), "no budget left");
    assert_eq!(
        *counter.lock().unwrap(),
        0,
        "an approved call must not run once the budget is spent"
    );
}

#[tokio::test]
async fn an_approval_resumed_within_the_budget_still_executes() {
    // The negative control for the test above: the same shape, with time left.
    let counter = Arc::new(Mutex::new(0));
    let tool = approval_tool(counter.clone());
    let client = budgeted_client(
        vec![secret_call(), ChatResponse::from_text("The secret is 42.")],
        |config| config.max_duration_seconds = Some(30.0),
    );
    let session = AgentSession::new();
    let mut options = ChatOptions::new().with_tool(tool);
    options.session = Some(session.clone());

    let resp1 = client
        .get_response(vec![Message::user("what is the secret?")], options.clone())
        .await
        .unwrap();
    let approval = resp1.user_input_requests()[0].create_response(true);
    let mut conversation = vec![Message::user("what is the secret?")];
    conversation.extend(resp1.messages.clone());
    conversation.push(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(approval)],
    ));
    let resp2 = client.get_response(conversation, options).await.unwrap();

    assert!(resp2.text().contains("42"), "got: {}", resp2.text());
    assert_eq!(*counter.lock().unwrap(), 1);
    // And the finished run left no budget behind for the next one.
    assert!(
        !session
            .state
            .contains_key("__af_function_invocation_budget__"),
        "a terminal exit must clear the parked budget"
    );
}

#[tokio::test]
async fn the_call_budget_is_parked_across_an_approval_pause() {
    // One call runs before the pause and the approved call runs after it,
    // which together spend a budget of two — but only if the count survives
    // the pause. If it reset, the resumed leg would still have a full budget
    // and would go on calling tools.
    let executed = Arc::new(Mutex::new(0));
    let counter = executed.clone();
    let ping = counting_tool(executed.clone());
    let approval = approval_tool(executed.clone());
    let inner = MockClient::new(vec![
        ping_calls(1),
        secret_call(),
        ChatResponse::from_text("out of calls"),
    ]);
    let client =
        FunctionInvokingChatClient::new(inner.clone()).with_config(FunctionInvocationConfig {
            max_function_calls: Some(2),
            ..Default::default()
        });
    let session = AgentSession::new();
    let mut options = ChatOptions::new().with_tool(ping).with_tool(approval);
    options.session = Some(session.clone());

    let resp1 = client
        .get_response(vec![Message::user("go")], options.clone())
        .await
        .unwrap();
    let requests = resp1.user_input_requests();
    assert_eq!(requests.len(), 1, "paused on the approval-gated call");
    assert_eq!(*counter.lock().unwrap(), 1, "only `ping` ran");
    let parked = session
        .state
        .get("__af_function_invocation_budget__")
        .expect("the pause parks the budget");
    assert_eq!(parked["executed"], 1);

    let mut conversation = vec![Message::user("go")];
    conversation.extend(resp1.messages.clone());
    conversation.push(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(
            requests[0].create_response(true),
        )],
    ));
    let resp2 = client.get_response(conversation, options).await.unwrap();

    assert_eq!(resp2.text(), "out of calls");
    assert_eq!(*counter.lock().unwrap(), 2, "ping plus the approved call");
    // The discriminating assertion: the resumed leg's final model call has
    // tools off, which only happens because the *resumed* budget was already
    // at 2 after the approval replay. A budget that reset at the pause would
    // have been at 1 and asked for tools again.
    assert_eq!(
        inner.all_options().last().unwrap().tool_choice,
        Some(ToolMode::None)
    );
    assert!(
        !session
            .state
            .contains_key("__af_function_invocation_budget__"),
        "the finished run clears the parked budget"
    );
}

#[test]
fn budget_configuration_is_validated() {
    let config = FunctionInvocationConfig {
        max_function_calls: Some(0),
        ..Default::default()
    };
    assert!(config
        .validate()
        .unwrap_err()
        .to_string()
        .contains("max_function_calls"));

    let mut config = FunctionInvocationConfig {
        max_duration_seconds: Some(0.0),
        ..Default::default()
    };
    assert!(config.validate().is_err());
    config.max_duration_seconds = Some(-1.0);
    assert!(config.validate().is_err());
    // NaN is rejected by the same comparison, rather than silently meaning
    // "never expires".
    config.max_duration_seconds = Some(f64::NAN);
    assert!(config.validate().is_err());
    config.max_duration_seconds = Some(0.5);
    assert!(config.validate().is_ok());
}

// endregion

#[tokio::test]
async fn an_agent_can_set_its_tool_loop_budget() {
    // The builder wraps the client itself, so without this the budget is
    // unreachable for anyone using `Agent` — which is most callers.
    let counter = Arc::new(Mutex::new(0));
    let inner = MockClient::new(vec![ping_calls(1), ChatResponse::from_text("done")]);
    let agent = Agent::builder(inner.clone())
        .function_invocation_config(FunctionInvocationConfig {
            max_function_calls: Some(1),
            ..Default::default()
        })
        .tool(counting_tool(counter.clone()))
        .build();

    let response = agent.run_once("go").await.unwrap();
    assert_eq!(response.text(), "done");
    assert_eq!(*counter.lock().unwrap(), 1);
    assert_eq!(
        inner.all_options().last().unwrap().tool_choice,
        Some(ToolMode::None)
    );
}

#[tokio::test]
async fn a_spent_budget_stops_executing_even_when_the_provider_ignores_tool_choice() {
    // `tool_choice: none` is a hint to the provider, and this mock ignores it
    // exactly as a provider that does not honor it would. The budget has to
    // be a ceiling on *executions*, not just a request to stop asking, so the
    // loop must not execute the calls that come back anyway.
    let counter = Arc::new(Mutex::new(0));
    let inner = MockClient::new(vec![
        ping_calls(1),
        ping_calls(1),
        ChatResponse::from_text("done"),
    ]);
    let client =
        FunctionInvokingChatClient::new(inner.clone()).with_config(FunctionInvocationConfig {
            max_function_calls: Some(1),
            ..Default::default()
        });

    let _ = client
        .get_response(
            vec![Message::user("go")],
            ChatOptions::new().with_tool(counting_tool(counter.clone())),
        )
        .await
        .unwrap();

    assert_eq!(
        *counter.lock().unwrap(),
        1,
        "the budget is a ceiling on executions, not a hint"
    );
    assert_eq!(
        inner.all_options().len(),
        2,
        "one tool-calling iteration, then the final tools-off call"
    );
}

/// A chat client that takes `delay` to answer, wrapping another client.
///
/// The point is the *model call itself* consuming wall-clock budget, which no
/// slow-tool test can exercise: a tool's time is charged after its batch runs,
/// a provider's is charged while the loop is blocked waiting for it.
#[derive(Clone)]
struct SlowClient {
    inner: MockClient,
    delay: Duration,
}

#[async_trait::async_trait]
impl ChatClient for SlowClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        tokio::time::sleep(self.delay).await;
        self.inner.get_response(messages, options).await
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> agent_framework_core::error::Result<agent_framework_core::client::ChatStream> {
        self.inner.get_streaming_response(messages, options).await
    }
}

#[tokio::test]
async fn a_slow_model_call_spends_the_duration_budget_before_its_tools_run() {
    // The budget is a bound on the whole loop, and the model call is part of
    // the loop. Checking only *before* each request means a provider that
    // takes longer than the remaining budget and comes back asking for tools
    // gets its entire batch executed on a budget that expired while it was
    // thinking.
    let counter = Arc::new(Mutex::new(0));
    let inner = MockClient::new(vec![
        ping_calls(1),
        ChatResponse::from_text("answered without tools"),
    ]);
    let client = FunctionInvokingChatClient::new(SlowClient {
        inner: inner.clone(),
        delay: Duration::from_millis(60),
    })
    .with_config(FunctionInvocationConfig {
        max_duration_seconds: Some(0.05),
        ..Default::default()
    });

    let response = client
        .get_response(
            vec![Message::user("go")],
            ChatOptions::new().with_tool(counting_tool(counter.clone())),
        )
        .await
        .unwrap();

    assert_eq!(
        *counter.lock().unwrap(),
        0,
        "the budget expired during the model call, so its tools must not run"
    );
    assert_eq!(response.text(), "answered without tools");
    assert_eq!(
        inner.all_options().last().unwrap().tool_choice,
        Some(ToolMode::None),
        "the failsafe call asks the model to answer with what it has"
    );
}

#[test]
fn a_duration_budget_a_duration_cannot_hold_is_refused() {
    // `f64::INFINITY` is greater than zero, so a bare positivity check lets it
    // through — and `Duration::from_secs_f64` then *panics*, taking the
    // process down over a configuration value.
    for seconds in [f64::INFINITY, 1e30] {
        let config = FunctionInvocationConfig {
            max_duration_seconds: Some(seconds),
            ..Default::default()
        };
        let err = match config.validate() {
            Ok(()) => panic!("{seconds} should be refused"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("finite duration"), "{seconds}: {err}");
    }
    // A very large but representable budget is still fine.
    assert!(FunctionInvocationConfig {
        max_duration_seconds: Some(86_400.0),
        ..Default::default()
    }
    .validate()
    .is_ok());
}

#[tokio::test]
async fn a_failed_run_does_not_leave_its_budget_parked_on_the_session() {
    // A budget is parked only for an approval round trip. If the resumed leg
    // then fails, the parked clock outlives the run that started it, and the
    // next — unrelated — run on the same session resumes it: `started_millis`
    // from a run already over, so a fresh request can be spent before it
    // makes a single call.
    let session = AgentSession::new();
    let counter = Arc::new(Mutex::new(0));
    let approving = FunctionInvokingChatClient::new(MockClient::new(vec![secret_call()]))
        .with_config(FunctionInvocationConfig {
            max_duration_seconds: Some(30.0),
            ..Default::default()
        });
    let tool = approval_tool(counter.clone());

    // 1. A run that pauses for approval parks the budget.
    let mut options = ChatOptions::new().with_tool(tool.clone());
    options.session = Some(session.clone());
    let paused = approving
        .get_response(vec![Message::user("go")], options)
        .await
        .unwrap();
    assert!(session
        .state
        .get("__af_function_invocation_budget__")
        .is_some());

    // 2. The resumed leg fails at the provider.
    let approvals: Vec<Content> = paused
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(|c| match c {
            Content::FunctionApprovalRequest(req) => {
                Some(Content::FunctionApprovalResponse(req.create_response(true)))
            }
            _ => None,
        })
        .collect();
    assert!(!approvals.is_empty(), "the run paused for approval");
    let failing =
        FunctionInvokingChatClient::new(FailingClient).with_config(FunctionInvocationConfig {
            max_duration_seconds: Some(30.0),
            ..Default::default()
        });
    let mut resume_options = ChatOptions::new().with_tool(tool);
    resume_options.session = Some(session.clone());
    let err = failing
        .get_response(
            vec![Message::with_contents(Role::user(), approvals)],
            resume_options,
        )
        .await;
    assert!(err.is_err(), "the provider failed, so the run must fail");

    assert!(
        session
            .state
            .get("__af_function_invocation_budget__")
            .is_none(),
        "a failed run must not leave its budget behind for the next one"
    );
}

#[tokio::test]
async fn an_abandoned_approval_does_not_spend_the_next_runs_budget() {
    // A pending approval that is never answered leaves a parked budget on the
    // session, and its clock keeps running. A later, unrelated request on that
    // session has nothing to do with the paused run, so inheriting that clock
    // — and finding its own tools disabled before it makes a single call — is
    // the wrong outcome.
    let session = AgentSession::new();
    let counter = Arc::new(Mutex::new(0));
    let config = || FunctionInvocationConfig {
        max_duration_seconds: Some(0.05),
        ..Default::default()
    };

    let pausing =
        FunctionInvokingChatClient::new(MockClient::new(vec![secret_call()])).with_config(config());
    let mut options = ChatOptions::new().with_tool(approval_tool(counter.clone()));
    options.session = Some(session.clone());
    let _paused = pausing
        .get_response(vec![Message::user("go")], options)
        .await
        .unwrap();
    assert!(
        session
            .state
            .get("__af_function_invocation_budget__")
            .is_some(),
        "the paused run parked its budget"
    );

    // Long enough that the abandoned run's clock is now spent.
    tokio::time::sleep(Duration::from_millis(80)).await;

    // A fresh request on the same session, carrying no approval response.
    let ping_counter = Arc::new(Mutex::new(0));
    let fresh = FunctionInvokingChatClient::new(MockClient::new(vec![
        ping_calls(1),
        ChatResponse::from_text("done"),
    ]))
    .with_config(config());
    let mut fresh_options = ChatOptions::new().with_tool(counting_tool(ping_counter.clone()));
    fresh_options.session = Some(session.clone());
    let response = fresh
        .get_response(vec![Message::user("unrelated")], fresh_options)
        .await
        .unwrap();

    assert_eq!(
        *ping_counter.lock().unwrap(),
        1,
        "the new run gets its own clock, not the abandoned run's spent one"
    );
    assert_eq!(response.text(), "done");
}

#[tokio::test]
async fn a_spent_budget_keeps_what_the_provider_already_resolved() {
    // A provider that runs a hosted tool itself puts the call *and its
    // result* in the same response. When the budget expires during that
    // response, the hosted work is done and paid for — dropping the whole
    // response would leave the failsafe answering from a conversation missing
    // the thing the provider just looked up. Only the unresolved local call,
    // which will never run now, is stripped.
    let counter = Arc::new(Mutex::new(0));
    let mixed = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![
                Content::FunctionCall(FunctionCallContent::new(
                    "hosted_1",
                    "web_search",
                    Some(FunctionArguments::Raw("{}".into())),
                )),
                Content::FunctionResult(FunctionResultContent::new(
                    "hosted_1",
                    Some(json!("the capital is Paris")),
                )),
                Content::FunctionCall(FunctionCallContent::new(
                    "call_local",
                    "ping",
                    Some(FunctionArguments::Raw("{}".into())),
                )),
            ],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let inner = MockClient::new(vec![mixed, ChatResponse::from_text("Paris")]);
    let client = FunctionInvokingChatClient::new(SlowClient {
        inner: inner.clone(),
        delay: Duration::from_millis(60),
    })
    .with_config(FunctionInvocationConfig {
        max_duration_seconds: Some(0.05),
        ..Default::default()
    });

    let response = client
        .get_response(
            vec![Message::user("what is the capital of France?")],
            ChatOptions::new().with_tool(counting_tool(counter.clone())),
        )
        .await
        .unwrap();

    assert_eq!(*counter.lock().unwrap(), 0, "the local call must not run");

    // The messages the *final* model call actually received. Asserting on the
    // returned transcript alone is not enough: the transcript is assembled
    // from `carried`, so a fix that preserved the result only there would
    // pass this test while still asking the model to answer without it.
    let final_call = inner.all_seen().last().cloned().expect("a failsafe call");
    let sent: Vec<&Content> = final_call.iter().flat_map(|m| m.contents.iter()).collect();
    assert!(
        sent.iter()
            .any(|c| matches!(c, Content::FunctionResult(r) if r.call_id == "hosted_1")),
        "the model composing the answer must see the hosted result: {sent:?}"
    );
    assert!(
        !sent
            .iter()
            .any(|c| matches!(c, Content::FunctionCall(f) if f.call_id == "call_local")),
        "and must not see a call that will never be answered: {sent:?}"
    );

    let all: Vec<&Content> = response
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .collect();
    assert!(
        all.iter()
            .any(|c| matches!(c, Content::FunctionResult(r) if r.call_id == "hosted_1")),
        "the provider-resolved result survives: {all:?}"
    );
    assert!(
        all.iter()
            .any(|c| matches!(c, Content::FunctionCall(f) if f.call_id == "hosted_1")),
        "paired with the call it answers"
    );
    assert!(
        !all.iter()
            .any(|c| matches!(c, Content::FunctionCall(f) if f.call_id == "call_local")),
        "but the unresolved local call is not left dangling: {all:?}"
    );
}

#[tokio::test]
async fn a_call_that_never_reached_a_tool_does_not_spend_the_budget() {
    // `max_function_calls` bounds executions, and its own docs say so. A
    // hallucinated tool name produces a result without the executor — or the
    // middleware — ever running, so charging it would let one bad name from
    // the model spend a budget of one and force the tools-off failsafe before
    // the model got a chance to correct itself.
    let counter = Arc::new(Mutex::new(0));
    let bad_call = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_bad",
                "no_such_tool",
                Some(FunctionArguments::Raw("{}".into())),
            ))],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let inner = MockClient::new(vec![
        bad_call,
        ping_calls(1),
        ChatResponse::from_text("recovered"),
    ]);
    let client =
        FunctionInvokingChatClient::new(inner.clone()).with_config(FunctionInvocationConfig {
            max_function_calls: Some(1),
            ..Default::default()
        });

    let response = client
        .get_response(
            vec![Message::user("go")],
            ChatOptions::new().with_tool(counting_tool(counter.clone())),
        )
        .await
        .unwrap();

    assert_eq!(
        *counter.lock().unwrap(),
        1,
        "the real call still had its budget: the miss cost nothing"
    );
    assert_eq!(response.text(), "recovered");
    assert!(
        inner
            .all_options()
            .iter()
            .take(2)
            .all(|o| o.tool_choice != Some(ToolMode::None)),
        "the model was not forced into the tools-off failsafe by the miss"
    );
}
