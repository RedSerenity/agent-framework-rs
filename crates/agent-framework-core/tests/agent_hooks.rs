//! End-to-end tests for the agent-hooks enforcement (`agent_hooks`), modeled
//! on upstream's `tests/core/test_agent_hooks.py`. Offline: a scripted mock
//! chat client stands in for the model.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::agent_hooks::{
    resolver_fn, AgentContextBuilder, AgentHooks, AgentHooksOptions, ApprovalRequest,
    ApprovalResolution, CompositionConfig, EnforcementMode, InterceptionEmitter,
    InterceptionRecord, Interceptor, Verdict,
};
use agent_framework_core::middleware::{
    AgentContext, FnMiddleware, FunctionInvocationContext, Next,
};
use agent_framework_core::prelude::*;
use agent_framework_core::types::{DataContent, FunctionArguments, FunctionCallContent};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

// region Helpers

/// A scripted chat client recording every request.
#[derive(Clone, Default)]
struct MockClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
    seen: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl MockClient {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            seen: Arc::default(),
        }
    }

    fn calls(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    fn request(&self, i: usize) -> Vec<Message> {
        self.seen.lock().unwrap()[i].clone()
    }
}

#[async_trait]
impl ChatClient for MockClient {
    async fn get_response(&self, messages: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(messages);
        let mut responses = self.responses.lock().unwrap();
        Ok(if responses.is_empty() {
            ChatResponse::from_text("(no more scripted responses)")
        } else {
            responses.remove(0)
        })
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
        Ok(futures::stream::iter(updates).boxed())
    }

    fn model(&self) -> Option<&str> {
        Some("mock-model")
    }
}

type Rule = Arc<dyn Fn(&Value) -> Result<Verdict> + Send + Sync>;

/// Records every context it sees and answers with a rule (allow by default).
#[derive(Clone)]
struct Guard {
    contexts: Arc<Mutex<Vec<Value>>>,
    rule: Rule,
}

impl Guard {
    fn allow() -> Self {
        Self::with(|_| Ok(Verdict::allow()))
    }

    fn with(rule: impl Fn(&Value) -> Result<Verdict> + Send + Sync + 'static) -> Self {
        Self {
            contexts: Arc::default(),
            rule: Arc::new(rule),
        }
    }

    /// Answer `verdict` at `point`, allow elsewhere.
    fn at(point: &'static str, verdict: Verdict) -> Self {
        Self::with(move |ctx| {
            Ok(if ctx["interception_point"] == point {
                verdict.clone()
            } else {
                Verdict::allow()
            })
        })
    }

    fn points(&self) -> Vec<String> {
        self.contexts
            .lock()
            .unwrap()
            .iter()
            .map(|c| c["interception_point"].as_str().unwrap().to_string())
            .collect()
    }

    fn contexts_for(&self, point: &str) -> Vec<Value> {
        self.contexts
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c["interception_point"] == point)
            .cloned()
            .collect()
    }
}

#[async_trait]
impl Interceptor for Guard {
    async fn intercept(&self, context: Value) -> Result<Verdict> {
        self.contexts.lock().unwrap().push(context.clone());
        (self.rule)(&context)
    }
}

fn hooks(guard: &Guard) -> AgentHooks {
    AgentHooks::new(AgentHooksOptions::new().interceptor(guard.clone())).unwrap()
}

/// A weather tool recording the locations it was invoked with.
fn weather_tool(calls: Arc<Mutex<Vec<String>>>) -> ToolDefinition {
    FunctionTool::new(
        "weather_tool",
        "Get the weather for a location.",
        json!({"type": "object", "properties": {"location": {"type": "string"}}}),
        move |args: Value| {
            let calls = calls.clone();
            async move {
                let location = args["location"].as_str().unwrap_or_default().to_string();
                calls.lock().unwrap().push(location.clone());
                Ok(json!(format!("weather in {location}")))
            }
        },
    )
    .into_definition()
}

fn tool_call_response(location: &str) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "weather_tool",
                Some(FunctionArguments::Raw(format!(
                    "{{\"location\": \"{location}\"}}"
                ))),
            ))],
        )],
        ..Default::default()
    }
}

fn final_response(text: &str) -> ChatResponse {
    ChatResponse::from_text(text)
}

/// The function result the model saw on its second request.
fn tool_result_seen(client: &MockClient) -> Value {
    client
        .request(1)
        .iter()
        .flat_map(|m| m.function_results())
        .next()
        .and_then(|r| r.result.clone())
        .unwrap_or(Value::Null)
}

fn blocked_point(err: &Error) -> String {
    err.interception_blocked()
        .unwrap_or_else(|| panic!("expected InterceptionBlocked, got {err:?}"))
        .interception_point()
        .to_string()
}

const FULL_TOOL_RUN_POINTS: [&str; 10] = [
    "agent_startup",
    "input",
    "pre_model_call",
    "post_model_call",
    "pre_tool_call",
    "post_tool_call",
    "pre_model_call",
    "post_model_call",
    "output",
    "agent_shutdown",
];

// endregion

// region Factory and emission order

#[test]
fn factory_requires_interceptors() {
    let err = AgentHooks::new(AgentHooksOptions::new()).unwrap_err();
    assert!(err.to_string().contains("at least one interceptor"));
}

#[tokio::test]
async fn full_tool_run_emits_complete_ordered_session() {
    let guard = Guard::allow();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockClient::new(vec![tool_call_response("Seattle"), final_response("Sunny")]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .name("assistant")
        .tool(weather_tool(calls.clone()))
        .build();

    let response = agent.run_once("weather?").await.unwrap();
    assert_eq!(response.text(), "Sunny");
    assert_eq!(guard.points(), FULL_TOOL_RUN_POINTS);
    assert_eq!(*calls.lock().unwrap(), vec!["Seattle".to_string()]);

    let contexts = guard.contexts.lock().unwrap().clone();
    let session = contexts[0]["session"]["id"].clone();
    for (i, ctx) in contexts.iter().enumerate() {
        assert_eq!(ctx["sequence"], i, "sequence is the total order");
        assert_eq!(ctx["session"]["id"], session);
        assert_eq!(ctx["spec"], "agent-hooks/0.1");
        assert_eq!(ctx["agent"]["framework"], "agent-framework");
        assert_eq!(ctx["agent"]["name"], "assistant");
    }
    let pre_tool = &guard.contexts_for("pre_tool_call")[0];
    assert_eq!(pre_tool["tool_call"]["id"], "call_1");
    assert_eq!(pre_tool["tool_call"]["name"], "weather_tool");
    assert_eq!(pre_tool["target"], json!({"location": "Seattle"}));
    let post_tool = &guard.contexts_for("post_tool_call")[0];
    assert_eq!(
        post_tool["tool_result"],
        json!({"value": "weather in Seattle", "is_error": false})
    );
    let post_model = &guard.contexts_for("post_model_call")[0];
    assert_eq!(post_model["model"]["id"], "mock-model");
    assert_eq!(
        post_model["response"]["tool_calls"],
        json!([{"id": "call_1", "name": "weather_tool", "args": {"location": "Seattle"}}])
    );
    // The output covers the whole response, tool-loop messages included.
    let output = &guard.contexts_for("output")[0]["target"]["content"];
    assert_eq!(output.as_array().unwrap().len(), 3);
    assert_eq!(output[2], json!({"role": "assistant", "content": "Sunny"}));
    assert_eq!(
        guard.contexts_for("agent_shutdown")[0]["summary"]["reason"],
        "completed"
    );
}

#[tokio::test]
async fn input_projection_is_faithful_and_excludes_history() {
    let guard = Guard::allow();
    let client = MockClient::new(vec![final_response("one"), final_response("two")]);
    let agent = hooks(&guard)
        .agent_builder(client)
        .instructions("be brief")
        .build();
    let mut session = agent.create_session();
    agent
        .run(vec![Message::user("first")], Some(&mut session))
        .await
        .unwrap();
    agent
        .run(vec![Message::user("second")], Some(&mut session))
        .await
        .unwrap();

    let inputs = guard.contexts_for("input");
    // Only the run input is projected: no instructions, no prior history.
    assert_eq!(
        inputs[1]["input"],
        json!({"content": "second", "role": "user"})
    );

    agent
        .run(
            vec![Message::system("sys"), Message::assistant("prior")],
            Some(&mut session),
        )
        .await
        .unwrap();
    let inputs = guard.contexts_for("input");
    assert_eq!(
        inputs[2]["input"],
        json!({"content": [
            {"role": "system", "content": "sys"},
            {"role": "external", "content": "prior"}
        ], "role": "user"})
    );
    // pre_model_call sees the full request, history included.
    let pre = guard.contexts_for("pre_model_call");
    assert!(pre[1]["messages"].as_array().unwrap().len() >= 3);
}

#[tokio::test]
async fn rich_content_is_preserved_in_projections() {
    let guard = Guard::allow();
    let client = MockClient::new(vec![final_response("seen")]);
    let agent = hooks(&guard).agent_builder(client).build();
    let image = Content::Data(DataContent::from_bytes(b"png-bytes", "image/png"));
    agent
        .run_once(Message::with_contents(
            Role::user(),
            vec![Content::text("look"), image],
        ))
        .await
        .unwrap();
    let content = &guard.contexts_for("input")[0]["input"]["content"];
    assert_eq!(content[0], json!({"type": "text", "text": "look"}));
    assert_eq!(content[1]["type"], "data");
    assert_eq!(content[1]["media_type"], "image/png");
}

#[tokio::test]
async fn tools_are_projected_at_startup_and_per_model_call() {
    let guard = Guard::allow();
    let client = MockClient::new(vec![final_response("ok")]);
    let agent = hooks(&guard)
        .agent_builder(client)
        .tool(weather_tool(Arc::default()))
        .build();
    agent.run_once("hi").await.unwrap();
    assert_eq!(
        guard.contexts_for("agent_startup")[0]["agent_init"]["tools_registered"],
        json!(["weather_tool"])
    );
    assert_eq!(
        guard.contexts_for("pre_model_call")[0]["tools"],
        json!([{"name": "weather_tool", "description": "Get the weather for a location."}])
    );

    // A call offering no tools omits the optional field.
    let guard = Guard::allow();
    let agent = hooks(&guard)
        .agent_builder(MockClient::new(vec![final_response("ok")]))
        .build();
    agent.run_once("hi").await.unwrap();
    assert!(guard.contexts_for("pre_model_call")[0]
        .get("tools")
        .is_none());
}

// endregion

// region Deny before execution

#[tokio::test]
async fn input_deny_blocks_run_before_model_call() {
    let guard = Guard::at("input", Verdict::deny("blocked_input"));
    let client = MockClient::new(vec![final_response("never")]);
    let agent = hooks(&guard).agent_builder(client.clone()).build();
    let err = agent.run_once("attack").await.unwrap_err();
    assert_eq!(blocked_point(&err), "input");
    assert_eq!(
        err.interception_blocked()
            .unwrap()
            .record
            .verdict
            .reason
            .as_deref(),
        Some("blocked_input")
    );
    assert_eq!(client.calls(), 0);
    assert_eq!(guard.points(), ["agent_startup", "input", "agent_shutdown"]);
    assert_eq!(
        guard.contexts_for("agent_shutdown")[0]["summary"]["reason"],
        "error"
    );
}

#[tokio::test]
async fn pre_model_call_deny_blocks_model_dispatch() {
    let guard = Guard::at("pre_model_call", Verdict::deny("no_model"));
    let client = MockClient::new(vec![final_response("never")]);
    let agent = hooks(&guard).agent_builder(client.clone()).build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert_eq!(blocked_point(&err), "pre_model_call");
    assert_eq!(client.calls(), 0);
}

#[tokio::test]
async fn post_model_call_deny_discards_response() {
    let guard = Guard::at("post_model_call", Verdict::deny("bad_response"));
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockClient::new(vec![tool_call_response("Paris")]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .tool(weather_tool(calls.clone()))
        .build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert_eq!(blocked_point(&err), "post_model_call");
    // The denied tool-calling response was never acted on.
    assert!(calls.lock().unwrap().is_empty());
    assert!(!guard.points().contains(&"output".to_string()));
}

#[tokio::test]
async fn pre_tool_call_deny_blocks_tool_and_continues_loop() {
    let guard = Guard::at(
        "pre_tool_call",
        Verdict::deny("tool_blocked").with_message("not allowed"),
    );
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("Sorry")]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .tool(weather_tool(calls.clone()))
        .build();
    let response = agent.run_once("hi").await.unwrap();
    assert_eq!(response.text(), "Sorry");
    assert!(calls.lock().unwrap().is_empty(), "the tool never ran");
    assert_eq!(
        tool_result_seen(&client),
        json!({
            "error": "Tool call blocked by agent-hooks at pre_tool_call.",
            "reason": "tool_blocked",
            "message": "not allowed"
        })
    );
    // §6.2: no post_tool_call for a call that never dispatched.
    assert!(guard.contexts_for("post_tool_call").is_empty());
}

#[tokio::test]
async fn post_tool_call_deny_discards_result() {
    let guard = Guard::at("post_tool_call", Verdict::deny("result_blocked"));
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("Done")]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .tool(weather_tool(calls.clone()))
        .build();
    agent.run_once("hi").await.unwrap();
    assert_eq!(calls.lock().unwrap().len(), 1, "the tool ran");
    let seen = tool_result_seen(&client);
    assert_eq!(seen["reason"], "result_blocked");
    assert!(!seen.to_string().contains("weather in Paris"));
}

#[tokio::test]
async fn output_deny_blocks_response() {
    let guard = Guard::at("output", Verdict::deny("egress_blocked"));
    let agent = hooks(&guard)
        .agent_builder(MockClient::new(vec![final_response("secret")]))
        .build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert_eq!(blocked_point(&err), "output");
    assert_eq!(err.to_string(), "output blocked: deny (egress_blocked)");
}

// endregion

// region Transform write-back

#[tokio::test]
async fn input_transform_writes_back_into_run_messages_and_history() {
    let guard = Guard::at(
        "input",
        Verdict::transform("$target.content", json!("[redacted]")),
    );
    let client = MockClient::new(vec![final_response("ok")]);
    let agent = hooks(&guard).agent_builder(client.clone()).build();
    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session = AgentSession::new().with_context_providers(vec![history.clone()]);
    agent
        .run(
            vec![Message::user("my password is hunter2")],
            Some(&mut session),
        )
        .await
        .unwrap();
    let request = client.request(0);
    assert_eq!(request.last().unwrap().text(), "[redacted]");
    // The rewritten input is what becomes durable.
    let stored = history.list_messages();
    assert_eq!(stored[0].text(), "[redacted]");
    assert!(!stored.iter().any(|m| m.text().contains("hunter2")));
}

#[tokio::test]
async fn input_role_transform_is_written_back() {
    let guard = Guard::at("input", Verdict::transform("$target.role", json!("system")));
    let client = MockClient::new(vec![final_response("ok")]);
    let agent = hooks(&guard).agent_builder(client.clone()).build();
    agent.run_once("hi").await.unwrap();
    assert_eq!(client.request(0).last().unwrap().role, Role::system());
}

#[tokio::test]
async fn multi_message_input_role_transform_fails_closed() {
    let guard = Guard::at("input", Verdict::transform("$target.role", json!("system")));
    let client = MockClient::new(vec![final_response("ok")]);
    let agent = hooks(&guard).agent_builder(client.clone()).build();
    let err = agent
        .run_once(vec![Message::user("a"), Message::user("b")])
        .await
        .unwrap_err();
    assert!(err.is_middleware_failure(), "{err:?}");
    assert!(err.to_string().contains("cannot be written back"));
    assert_eq!(client.calls(), 0);
}

#[tokio::test]
async fn pre_model_call_transform_writes_back_into_request() {
    let guard = Guard::with(|ctx| {
        Ok(if ctx["interception_point"] == "pre_model_call" {
            let last = ctx["messages"].as_array().unwrap().len() - 1;
            Verdict::transform(format!("$target[{last}].content"), json!("rewritten"))
        } else {
            Verdict::allow()
        })
    });
    let client = MockClient::new(vec![final_response("ok")]);
    let agent = hooks(&guard).agent_builder(client.clone()).build();
    agent.run_once("original").await.unwrap();
    assert_eq!(client.request(0).last().unwrap().text(), "rewritten");
}

#[tokio::test]
async fn pre_tool_call_transform_writes_back_into_arguments() {
    let guard = Guard::at(
        "pre_tool_call",
        Verdict::transform("$target.location", json!("Oslo")),
    );
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("ok")]);
    let agent = hooks(&guard)
        .agent_builder(client)
        .tool(weather_tool(calls.clone()))
        .build();
    agent.run_once("hi").await.unwrap();
    assert_eq!(*calls.lock().unwrap(), vec!["Oslo".to_string()]);
    // post_tool_call brackets the effective (approved) arguments.
    assert_eq!(
        guard.contexts_for("post_tool_call")[0]["tool_call"]["args"],
        json!({"location": "Oslo"})
    );
}

#[tokio::test]
async fn pre_tool_call_transform_to_non_object_fails_closed() {
    let guard = Guard::at(
        "pre_tool_call",
        Verdict::transform("$target", json!("nope")),
    );
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("ok")]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .tool(weather_tool(calls.clone()))
        .build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert!(err.is_middleware_failure(), "{err:?}");
    assert!(err.to_string().contains("must produce an arguments object"));
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(client.calls(), 1, "the loop stopped");
}

#[tokio::test]
async fn post_tool_call_transform_writes_back_into_result() {
    let guard = Guard::at(
        "post_tool_call",
        Verdict::transform("$target", json!("scrubbed")),
    );
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("ok")]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .tool(weather_tool(Arc::default()))
        .build();
    agent.run_once("hi").await.unwrap();
    assert_eq!(tool_result_seen(&client), json!("scrubbed"));
}

#[tokio::test]
async fn output_transform_writes_back_into_response_and_history() {
    let guard = Guard::at(
        "output",
        Verdict::transform("$target.content", json!("safe answer")),
    );
    let agent = hooks(&guard)
        .agent_builder(MockClient::new(vec![final_response("leaky answer")]))
        .build();
    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session = AgentSession::new().with_context_providers(vec![history.clone()]);
    let response = agent
        .run(vec![Message::user("hi")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(response.text(), "safe answer");
    let stored = history.list_messages();
    assert_eq!(stored.last().unwrap().text(), "safe answer");
    assert!(!stored.iter().any(|m| m.text().contains("leaky")));
}

#[tokio::test]
async fn tool_call_name_transform_at_post_model_call_is_applied() {
    let guard = Guard::with(|ctx| {
        let has_calls = ctx["response"]["tool_calls"]
            .as_array()
            .is_some_and(|c| !c.is_empty());
        Ok(
            if ctx["interception_point"] == "post_model_call" && has_calls {
                Verdict::transform("$target.tool_calls[0].name", json!("safe_tool"))
            } else {
                Verdict::allow()
            },
        )
    });
    let safe_calls = Arc::new(Mutex::new(0));
    let safe_calls2 = safe_calls.clone();
    let safe_tool = FunctionTool::new("safe_tool", "safe", json!({"type": "object"}), move |_| {
        let safe_calls = safe_calls2.clone();
        async move {
            *safe_calls.lock().unwrap() += 1;
            Ok(json!("safe"))
        }
    })
    .into_definition();
    let weather_calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("ok")]);
    let agent = hooks(&guard)
        .agent_builder(client)
        .tool(weather_tool(weather_calls.clone()))
        .tool(safe_tool)
        .build();
    agent.run_once("hi").await.unwrap();
    assert_eq!(*safe_calls.lock().unwrap(), 1);
    assert!(weather_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn non_string_finish_reason_transform_fails_closed() {
    let guard = Guard::at(
        "post_model_call",
        Verdict::transform("$target.finish_reason", json!(5)),
    );
    let agent = hooks(&guard)
        .agent_builder(MockClient::new(vec![final_response("ok")]))
        .build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert!(err.to_string().contains("finish_reason a string"), "{err}");
}

// endregion

// region Streaming

#[tokio::test]
async fn streaming_buffers_until_all_verdicts_permit() {
    let guard = Guard::allow();
    let client = MockClient::new(vec![tool_call_response("Rome"), final_response("Warm")]);
    let agent = hooks(&guard)
        .agent_builder(client)
        .tool(weather_tool(Arc::default()))
        .build();
    let stream = agent.run_stream_once("weather?").await.unwrap();
    // Every verdict, through output and shutdown, precedes the first update.
    assert_eq!(guard.points(), FULL_TOOL_RUN_POINTS);
    let updates: Vec<_> = stream.collect().await;
    let text: String = updates.into_iter().map(|u| u.unwrap().text()).collect();
    assert!(text.contains("Warm"));
}

#[tokio::test]
async fn streaming_output_deny_releases_nothing() {
    let guard = Guard::at("output", Verdict::deny("egress_blocked"));
    let agent = hooks(&guard)
        .agent_builder(MockClient::new(vec![final_response("secret")]))
        .build();
    let err = agent.run_stream_once("hi").await.err().expect("denied");
    assert_eq!(blocked_point(&err), "output");
}

#[tokio::test]
async fn streaming_post_model_call_deny_releases_nothing() {
    let guard = Guard::at("post_model_call", Verdict::deny("nope"));
    let agent = hooks(&guard)
        .agent_builder(MockClient::new(vec![final_response("secret")]))
        .build();
    let err = agent.run_stream_once("hi").await.err().expect("denied");
    assert_eq!(blocked_point(&err), "post_model_call");
}

#[tokio::test]
async fn streaming_output_transform_rewrites_updates() {
    let guard = Guard::at(
        "output",
        Verdict::transform("$target.content", json!("clean")),
    );
    let agent = hooks(&guard)
        .agent_builder(MockClient::new(vec![final_response("dirty")]))
        .build();
    let updates: Vec<_> = agent.run_stream_once("hi").await.unwrap().collect().await;
    let text: String = updates.into_iter().map(|u| u.unwrap().text()).collect();
    assert_eq!(text, "clean");
}

// endregion

// region Error cleanup and fail-closed host errors

#[tokio::test]
async fn tool_exception_is_bracketed_with_error_post_tool_call() {
    let guard = Guard::allow();
    let failing = FunctionTool::new(
        "weather_tool",
        "fails",
        json!({"type": "object"}),
        |_| async { Err::<Value, _>(Error::tool("boom")) },
    )
    .into_definition();
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("ok")]);
    let agent = hooks(&guard).agent_builder(client).tool(failing).build();
    agent.run_once("hi").await.unwrap();
    let post = &guard.contexts_for("post_tool_call")[0];
    // Only the error's type tag crosses the boundary.
    assert_eq!(
        post["tool_result"],
        json!({"value": "tool", "is_error": true})
    );
}

#[tokio::test]
async fn interceptor_crash_at_tool_seam_fails_closed_and_halts_run() {
    let guard = Guard::with(|ctx| {
        if ctx["interception_point"] == "pre_tool_call" {
            Err(Error::other("guard crashed"))
        } else {
            Ok(Verdict::allow())
        }
    });
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("ok")]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .tool(weather_tool(calls.clone()))
        .build();
    let err = agent.run_once("hi").await.unwrap_err();
    // The tool-seam block surfaces as the block itself.
    assert_eq!(blocked_point(&err), "pre_tool_call");
    assert_eq!(
        err.interception_blocked()
            .unwrap()
            .record
            .verdict
            .reason
            .as_deref(),
        Some("host_error:interceptor_failed")
    );
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(client.calls(), 1, "the run halted instead of looping");
    assert_eq!(
        guard.contexts_for("agent_shutdown")[0]["summary"]["reason"],
        "error"
    );
}

#[tokio::test]
async fn interceptor_crash_at_input_fails_closed() {
    let guard = Guard::with(|ctx| {
        if ctx["interception_point"] == "input" {
            panic!("guard panicked")
        }
        Ok(Verdict::allow())
    });
    let client = MockClient::new(vec![final_response("never")]);
    let agent = hooks(&guard).agent_builder(client.clone()).build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert_eq!(
        err.interception_blocked()
            .unwrap()
            .record
            .verdict
            .reason
            .as_deref(),
        Some("host_error:interceptor_failed")
    );
    assert_eq!(client.calls(), 0);
}

#[tokio::test]
async fn interceptor_timeout_fails_closed() {
    struct Slow;
    #[async_trait]
    impl Interceptor for Slow {
        async fn intercept(&self, _: Value) -> Result<Verdict> {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok(Verdict::allow())
        }
    }
    let hooks = AgentHooks::new(
        AgentHooksOptions::new()
            .interceptor(Slow)
            .timeout(Some(Duration::from_millis(20))),
    )
    .unwrap();
    let agent = hooks
        .agent_builder(MockClient::new(vec![final_response("x")]))
        .build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert_eq!(
        err.interception_blocked()
            .unwrap()
            .record
            .verdict
            .reason
            .as_deref(),
        Some("host_error:interceptor_timeout")
    );
}

#[tokio::test]
async fn third_party_middleware_failure_is_bracketed_and_propagates() {
    let guard = Guard::allow();
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("ok")]);
    let agent = hooks(&guard)
        .agent_builder(client)
        .tool(weather_tool(Arc::default()))
        .function_middleware(Arc::new(FnMiddleware::new(
            |_ctx: FunctionInvocationContext, _next: Next<FunctionInvocationContext>| async {
                Err::<FunctionInvocationContext, _>(Error::middleware_failure("policy gate"))
            },
        )))
        .build();
    let err = agent.run_once("hi").await.unwrap_err();
    // Not laundered into an InterceptionBlocked: it propagates as raised.
    assert!(err.is_middleware_failure());
    assert!(err.interception_blocked().is_none());
    let post = &guard.contexts_for("post_tool_call")[0];
    assert_eq!(post["tool_result"]["is_error"], true);
    assert_eq!(post["tool_result"]["value"], "middleware_failure");
}

#[tokio::test]
async fn function_seam_short_circuit_result_is_bracketed_and_can_be_denied() {
    let substitute = || {
        Arc::new(FnMiddleware::new(
            |mut ctx: FunctionInvocationContext, _next: Next<FunctionInvocationContext>| async {
                ctx.result = Some(json!("substituted secret"));
                ctx.terminate = true;
                Ok(ctx)
            },
        ))
    };
    let guard = Guard::allow();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("ok")]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .tool(weather_tool(calls.clone()))
        .function_middleware(substitute())
        .build();
    agent.run_once("hi").await.unwrap();
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(
        guard.contexts_for("post_tool_call")[0]["target"],
        "substituted secret"
    );

    let guard = Guard::with(|ctx| {
        Ok(
            if ctx["interception_point"] == "post_tool_call"
                && ctx["target"].to_string().contains("secret")
            {
                Verdict::deny("no_secrets")
            } else {
                Verdict::allow()
            },
        )
    });
    let client = MockClient::new(vec![tool_call_response("Paris"), final_response("ok")]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .tool(weather_tool(Arc::default()))
        .function_middleware(substitute())
        .build();
    agent.run_once("hi").await.unwrap();
    assert_eq!(tool_result_seen(&client)["reason"], "no_secrets");
}

#[tokio::test]
async fn agent_seam_short_circuit_result_is_guarded() {
    let short_circuit = || {
        Arc::new(FnMiddleware::new(
            |mut ctx: AgentContext, _next: Next<AgentContext>| async {
                ctx.result = Some(AgentResponse {
                    messages: vec![Message::assistant("cached secret")],
                    ..Default::default()
                });
                ctx.terminate = true;
                Ok(ctx)
            },
        ))
    };
    let guard = Guard::allow();
    let client = MockClient::new(vec![]);
    let agent = hooks(&guard)
        .agent_builder(client.clone())
        .middleware(short_circuit())
        .build();
    assert_eq!(agent.run_once("hi").await.unwrap().text(), "cached secret");
    assert_eq!(client.calls(), 0);
    assert_eq!(
        guard.contexts_for("output")[0]["target"]["content"],
        "cached secret"
    );

    let guard = Guard::at("output", Verdict::deny("egress_blocked"));
    let agent = hooks(&guard)
        .agent_builder(MockClient::new(vec![]))
        .middleware(short_circuit())
        .build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert_eq!(blocked_point(&err), "output");
}

// endregion

// region Concurrency, session scoping and modes

#[tokio::test]
async fn concurrent_runs_are_isolated() {
    let records: Arc<Mutex<Vec<InterceptionRecord>>> = Arc::default();
    let sink = records.clone();
    let hooks = AgentHooks::new(
        AgentHooksOptions::new()
            .interceptor(Guard::allow())
            .record_sink(move |r| sink.lock().unwrap().push(r.clone())),
    )
    .unwrap();
    let agent = hooks
        .agent_builder(MockClient::new(vec![
            final_response("a"),
            final_response("b"),
        ]))
        .build();
    let (a, b) = tokio::join!(agent.run_once("one"), agent.run_once("two"));
    a.unwrap();
    b.unwrap();
    let records = records.lock().unwrap();
    let mut sessions: Vec<&str> = records.iter().map(|r| r.session_id.as_str()).collect();
    sessions.sort();
    sessions.dedup();
    assert_eq!(sessions.len(), 2, "one agent-hooks session per run");
    for session in sessions {
        let seqs: Vec<i64> = records
            .iter()
            .filter(|r| r.session_id == session)
            .map(|r| r.sequence)
            .collect();
        assert_eq!(seqs, (0..seqs.len() as i64).collect::<Vec<_>>());
    }
}

#[tokio::test]
async fn host_owned_session_spans_runs() {
    let guard = Guard::allow();
    let emitter = Arc::new(InterceptionEmitter::new().register(guard.clone()));
    let builder = Arc::new(AgentContextBuilder::new(
        "host-agent",
        "agent-framework",
        "host-session",
    ));
    let hooks = AgentHooks::from_emitter(emitter.clone(), builder);
    let agent = hooks
        .agent_builder(MockClient::new(vec![
            final_response("a"),
            final_response("b"),
        ]))
        .build();
    agent.run_once("one").await.unwrap();
    agent.run_once("two").await.unwrap();
    // Only per-run points; the host owns startup/shutdown.
    assert_eq!(
        guard.points(),
        ["input", "pre_model_call", "post_model_call", "output"].repeat(2)
    );
    let records = emitter.results();
    assert!(records.iter().all(|r| r.session_id == "host-session"));
    assert_eq!(records.last().unwrap().sequence, 7);
}

#[tokio::test]
async fn liftable_deny_is_resolved_through_the_approval_seam() {
    let records: Arc<Mutex<Vec<InterceptionRecord>>> = Arc::default();
    let sink = records.clone();
    let hooks = AgentHooks::new(
        AgentHooksOptions::new()
            .interceptor(Guard::at("output", Verdict::escalate("needs_review")))
            .resolver(resolver_fn(|req: ApprovalRequest| async move {
                assert_eq!(req.interception_point.as_str(), "output");
                Ok(ApprovalResolution::approve(&req, Verdict::allow()))
            }))
            .record_sink(move |r| sink.lock().unwrap().push(r.clone())),
    )
    .unwrap();
    let agent = hooks
        .agent_builder(MockClient::new(vec![final_response("reviewed")]))
        .build();
    assert_eq!(agent.run_once("hi").await.unwrap().text(), "reviewed");
    let records = records.lock().unwrap();
    let output = records
        .iter()
        .find(|r| r.interception_point.as_str() == "output")
        .unwrap();
    assert_eq!(output.resolved_by.as_deref(), Some("approval"));
}

#[tokio::test]
async fn evaluate_only_records_but_never_blocks() {
    let records: Arc<Mutex<Vec<InterceptionRecord>>> = Arc::default();
    let sink = records.clone();
    let hooks = AgentHooks::new(
        AgentHooksOptions::new()
            .interceptor(Guard::with(|ctx| {
                Ok(if ctx["interception_point"] == "output" {
                    Verdict::transform("$target.content", json!("rewritten"))
                } else {
                    Verdict::deny("everything")
                })
            }))
            .mode(EnforcementMode::EvaluateOnly)
            .record_sink(move |r| sink.lock().unwrap().push(r.clone())),
    )
    .unwrap();
    let agent = hooks
        .agent_builder(MockClient::new(vec![final_response("original")]))
        .build();
    // Denies are recorded but not enforced; transforms are not applied.
    assert_eq!(agent.run_once("hi").await.unwrap().text(), "original");
    let records = records.lock().unwrap();
    assert!(records.iter().all(|r| r.proceeds()));
    assert!(records
        .iter()
        .any(|r| r.verdict.reason.as_deref() == Some("everything")));
}

#[tokio::test]
async fn composition_profile_is_recorded_and_applied() {
    let records: Arc<Mutex<Vec<InterceptionRecord>>> = Arc::default();
    let sink = records.clone();
    let hooks = AgentHooks::new(
        AgentHooksOptions::new()
            .named_interceptor("deny-input", Guard::at("input", Verdict::deny("first")))
            .named_interceptor("audit", Guard::allow())
            .composition(CompositionConfig::run_all())
            .record_sink(move |r| sink.lock().unwrap().push(r.clone())),
    )
    .unwrap();
    let agent = hooks.agent_builder(MockClient::new(vec![])).build();
    agent.run_once("hi").await.unwrap_err();
    let records = records.lock().unwrap();
    let input = &records[1];
    assert_eq!(input.composition.profile.as_str(), "sequential/run_all");
    assert_eq!(input.verdicts.len(), 2, "run_all ran past the deny");
    assert_eq!(input.verdicts[1].name.as_deref(), Some("audit"));
    assert_eq!(input.decided_by, Some(0));
}

// endregion

// region Persistence is gated behind verdicts

#[tokio::test]
async fn denied_output_never_becomes_durable_history() {
    for streaming in [false, true] {
        let guard = Guard::with(|ctx| {
            Ok(
                if ctx["interception_point"] == "output"
                    && ctx["target"].to_string().contains("secret")
                {
                    Verdict::deny("egress_blocked")
                } else {
                    Verdict::allow()
                },
            )
        });
        let client = MockClient::new(vec![final_response("the secret"), final_response("fine")]);
        let agent = hooks(&guard).agent_builder(client.clone()).build();
        let history = Arc::new(InMemoryHistoryProvider::new());
        let session = AgentSession::new().with_context_providers(vec![history.clone()]);
        if streaming {
            assert!(agent
                .run_stream("first", Some(session.clone()), None)
                .await
                .is_err());
        } else {
            let mut s = session.clone();
            assert!(agent
                .run(vec![Message::user("first")], Some(&mut s))
                .await
                .is_err());
        }
        // Neither the denied output nor the denied turn's input persisted.
        assert!(history.list_messages().is_empty(), "streaming={streaming}");
        let mut s = session.clone();
        agent
            .run(vec![Message::user("second")], Some(&mut s))
            .await
            .unwrap();
        let second_request = client.request(1);
        assert!(!second_request.iter().any(|m| m.text().contains("secret")));
        assert_eq!(history.list_messages().len(), 2);
    }
}

#[tokio::test]
async fn nested_agents_with_their_own_installations_stay_isolated() {
    let inner_guard = Guard::allow();
    let inner = hooks(&inner_guard)
        .agent_builder(MockClient::new(vec![final_response("inner answer")]))
        .name("inner")
        .build();
    let outer_guard = Guard::allow();
    let outer_client = MockClient::new(vec![
        ChatResponse {
            messages: vec![Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionCall(FunctionCallContent::new(
                    "call_inner",
                    "inner",
                    Some(FunctionArguments::Raw("{\"task\": \"go\"}".into())),
                ))],
            )],
            ..Default::default()
        },
        final_response("outer answer"),
    ]);
    let outer = hooks(&outer_guard)
        .agent_builder(outer_client)
        .name("outer")
        .tool(inner.as_tool(Default::default()))
        .build();
    assert_eq!(
        outer.run_once("delegate").await.unwrap().text(),
        "outer answer"
    );
    assert_eq!(outer_guard.points(), FULL_TOOL_RUN_POINTS);
    assert_eq!(
        inner_guard.points(),
        [
            "agent_startup",
            "input",
            "pre_model_call",
            "post_model_call",
            "output",
            "agent_shutdown"
        ]
    );
    assert!(inner_guard
        .contexts
        .lock()
        .unwrap()
        .iter()
        .all(|c| c["agent"]["name"] == "inner"));
}

// endregion
