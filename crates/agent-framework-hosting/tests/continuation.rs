//! `/v1/responses` continuation: `previous_response_id` and `conversation`
//! for agents (session snapshots), and human-in-the-loop resume for
//! workflows (per-conversation checkpoints). Mirrors upstream
//! `hosting-responses` + `AgentState` and DevUI's `workflow_hil_response`.

mod common;

use std::sync::Arc;

use agent_framework_core::agent::Agent;
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::Result;
use agent_framework_core::session_store::FileSessionStore;
use agent_framework_core::types::{ChatOptions, ChatResponse, ChatResponseUpdate, Message, Role};
use agent_framework_core::workflow::{
    FunctionExecutor, RequestInfoExecutor, RequestResponse, Workflow, WorkflowBuilder,
};
use agent_framework_hosting::AgentHost;
use async_trait::async_trait;
use axum::http::StatusCode;
use futures::StreamExt;
use serde_json::{json, Value};

use common::{parse_sse_json, post_json, post_raw};

/// Replies with every user turn it was shown, so a reply proves which
/// history reached the model.
struct HistoryClient;

fn transcript(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|m| m.role == Role::user())
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("|")
}

#[async_trait]
impl ChatClient for HistoryClient {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(transcript(&messages)))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        _o: ChatOptions,
    ) -> Result<ChatStream> {
        let update = ChatResponseUpdate {
            contents: vec![agent_framework_core::types::Content::text(transcript(
                &messages,
            ))],
            role: Some(Role::assistant()),
            ..Default::default()
        };
        Ok(futures::stream::iter(vec![Ok(update)]).boxed())
    }
}

fn agent_host() -> AgentHost {
    AgentHost::new().agent("chat", Agent::builder(HistoryClient).name("chat").build())
}

async fn turn(app: axum::Router, extra: Value) -> (StatusCode, Value) {
    let mut body = json!({ "metadata": { "entity_id": "chat" } });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    post_json(app, "/v1/responses", &body).await
}

#[tokio::test]
async fn a_request_without_continuation_starts_fresh_each_time() {
    let app = agent_host().into_router();
    let (_, a) = turn(app.clone(), json!({ "input": "one" })).await;
    let (_, b) = turn(app, json!({ "input": "two" })).await;
    assert_eq!(a["output_text"], "one");
    assert_eq!(b["output_text"], "two");
    assert!(a["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(a["id"].as_str().unwrap().len(), "resp_".len() + 32);
    assert!(a.get("conversation").is_none());
}

#[tokio::test]
async fn previous_response_id_continues_and_branches() {
    let app = agent_host().into_router();
    let (_, first) = turn(app.clone(), json!({ "input": "one" })).await;
    let first_id = first["id"].as_str().unwrap();

    let (status, second) = turn(
        app.clone(),
        json!({ "input": "two", "previous_response_id": first_id }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(second["output_text"], "one|two");
    assert_eq!(second["previous_response_id"], first_id);

    // Branching from the first response again does not see "two": the
    // snapshot under `first_id` was not advanced by the second turn.
    let (_, branch) = turn(
        app.clone(),
        json!({ "input": "other", "previous_response_id": first_id }),
    )
    .await;
    assert_eq!(branch["output_text"], "one|other");

    // And the second response is itself continuable.
    let (_, third) = turn(
        app,
        json!({ "input": "three", "previous_response_id": second["id"] }),
    )
    .await;
    assert_eq!(third["output_text"], "one|two|three");
}

#[tokio::test]
async fn an_unknown_previous_response_id_is_a_400() {
    let (status, body) = turn(
        agent_host().into_router(),
        json!({ "input": "x", "previous_response_id": "resp_nope" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "previous_response_not_found");
}

#[tokio::test]
async fn conversation_is_a_mutable_head_in_either_shape() {
    let app = agent_host().into_router();
    let (_, a) = turn(
        app.clone(),
        json!({ "input": "one", "conversation": "conv_abc" }),
    )
    .await;
    assert_eq!(a["output_text"], "one");
    assert_eq!(a["conversation"]["id"], "conv_abc");

    let (_, b) = turn(
        app.clone(),
        json!({ "input": "two", "conversation": { "id": "conv_abc" } }),
    )
    .await;
    assert_eq!(b["output_text"], "one|two");

    // The deprecated spelling reaches the same head.
    let (_, c) = turn(
        app.clone(),
        json!({ "input": "three", "conversation_id": "conv_abc" }),
    )
    .await;
    assert_eq!(c["output_text"], "one|two|three");

    // A response inside a conversation is also a branch point.
    let (_, d) = turn(
        app,
        json!({ "input": "alt", "previous_response_id": a["id"] }),
    )
    .await;
    assert_eq!(d["output_text"], "one|alt");
}

#[tokio::test]
async fn concurrent_turns_on_one_conversation_both_land() {
    let app = agent_host().into_router();
    let turns: Vec<_> = (0..6)
        .map(|i| {
            let app = app.clone();
            tokio::spawn(async move {
                turn(
                    app,
                    json!({ "input": format!("t{i}"), "conversation": "conv_c" }),
                )
                .await
            })
        })
        .collect();
    for t in turns {
        assert_eq!(t.await.unwrap().0, StatusCode::OK);
    }
    let (_, last) = turn(app, json!({ "input": "end", "conversation": "conv_c" })).await;
    // Every turn is in the head: none overwrote another.
    assert_eq!(last["output_text"].as_str().unwrap().split('|').count(), 7);
}

#[tokio::test]
async fn malformed_or_combined_continuations_are_400s() {
    let app = agent_host().into_router();
    for extra in [
        json!({ "previous_response_id": "resp_a", "conversation": "conv_b" }),
        json!({ "previous_response_id": "" }),
        json!({ "previous_response_id": 7 }),
        json!({ "conversation": { "id": "" } }),
        json!({ "conversation": "conv_a", "conversation_id": "conv_a" }),
    ] {
        let mut extra = extra;
        extra["input"] = json!("x");
        let (status, body) = turn(app.clone(), extra.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}");
        assert_eq!(body["error"]["code"], "invalid_continuation");
    }
}

#[tokio::test]
async fn streaming_turns_are_continuable() {
    let app = agent_host().into_router();
    let body = json!({
        "input": "one",
        "stream": true,
        "conversation": "conv_s",
        "metadata": { "entity_id": "chat" },
    });
    let (status, text) = post_raw(app.clone(), "/v1/responses", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse_json(&text);
    let created = events
        .iter()
        .find(|e| e["type"] == "response.created")
        .unwrap();
    let done = events
        .iter()
        .find(|e| e["type"] == "response.completed")
        .unwrap();
    // One id across the stream, and the conversation on both payloads.
    assert_eq!(created["response"]["id"], done["response"]["id"]);
    assert_eq!(created["response"]["conversation"]["id"], "conv_s");
    assert_eq!(done["response"]["conversation"]["id"], "conv_s");

    let (_, second) = turn(
        app.clone(),
        json!({ "input": "two", "previous_response_id": done["response"]["id"] }),
    )
    .await;
    assert_eq!(second["output_text"], "one|two");
    let (_, head) = turn(app, json!({ "input": "three", "conversation": "conv_s" })).await;
    assert_eq!(head["output_text"], "one|three");
}

#[tokio::test]
async fn a_file_session_store_survives_a_restart() {
    let dir = std::env::temp_dir().join(format!("af-host-sessions-{}", uuid::Uuid::new_v4()));
    let host = || agent_host().with_session_store(Arc::new(FileSessionStore::new(&dir).unwrap()));
    let (_, first) = turn(host().into_router(), json!({ "input": "one" })).await;

    // A brand-new host over the same directory: a cold start.
    let (status, second) = turn(
        host().into_router(),
        json!({ "input": "two", "previous_response_id": first["id"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(second["output_text"], "one|two");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn response_ids_do_not_cross_entities() {
    let host = agent_host().agent("other", Agent::builder(HistoryClient).name("other").build());
    let app = host.into_router();
    let (_, first) = turn(app.clone(), json!({ "input": "secret" })).await;
    let body = json!({
        "input": "hi",
        "previous_response_id": first["id"],
        "metadata": { "entity_id": "other" },
    });
    let (status, _) = post_json(app, "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn request_options_reach_the_model() {
    struct OptionsClient;
    #[async_trait]
    impl ChatClient for OptionsClient {
        async fn get_response(&self, _m: Vec<Message>, o: ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse::from_text(format!(
                "{:?}/{:?}/{:?}",
                o.temperature, o.max_tokens, o.allow_multiple_tool_calls
            )))
        }
        async fn get_streaming_response(
            &self,
            _m: Vec<Message>,
            _o: ChatOptions,
        ) -> Result<ChatStream> {
            unreachable!()
        }
    }
    let app = AgentHost::new()
        .agent("opts", Agent::builder(OptionsClient).build())
        .into_router();
    let body = json!({
        "input": "x",
        "temperature": 0.5,
        "max_output_tokens": 42,
        "parallel_tool_calls": false,
        "metadata": { "entity_id": "opts" },
    });
    let (_, resp) = post_json(app, "/v1/responses", &body).await;
    assert_eq!(resp["output_text"], "Some(0.5)/Some(42)/Some(false)");
}

// ---------------------------------------------------------------------------
// Workflows: human-in-the-loop resume over HTTP
// ---------------------------------------------------------------------------

/// Asks a question per input and yields the answer once it arrives — twice,
/// so a run pauses, resumes, and pauses again.
fn hitl_workflow() -> Workflow {
    let asker = FunctionExecutor::new("asker", |msg, ctx| async move {
        if let Some(resp) = RequestResponse::from_message(&msg) {
            let answer = resp.data.as_str().unwrap_or_default().to_string();
            if answer == "first" {
                ctx.send_message(json!("second question")).await?;
            } else {
                ctx.yield_output(json!(format!("done: {answer}"))).await?;
            }
        } else {
            ctx.send_message(msg).await?;
        }
        Ok(())
    });
    WorkflowBuilder::new()
        .add_executor(Arc::new(asker))
        .add_executor(Arc::new(RequestInfoExecutor::new("ask")))
        .set_start("asker")
        .add_edge("asker", "ask")
        .build()
        .unwrap()
}

fn hil_input(request_id: &Value, answer: Value) -> Value {
    json!([{
        "type": "message",
        "role": "user",
        "content": [{ "type": "workflow_hil_response", "responses": { request_id.as_str().unwrap(): answer } }],
    }])
}

#[tokio::test]
async fn a_paused_workflow_resumes_across_requests() {
    let app = AgentHost::new()
        .workflow("hitl", hitl_workflow())
        .into_router();
    let post = |body: Value| {
        let app = app.clone();
        async move { post_json(app, "/v1/responses", &body).await }
    };

    let (status, first) = post(json!({
        "input": "first question",
        "metadata": { "entity_id": "hitl" },
    }))
    .await;
    assert_eq!(status, StatusCode::OK);
    let conversation = first["conversation"]["id"].as_str().unwrap().to_string();
    assert!(conversation.starts_with("conv_"));
    let pending = &first["pending_requests"][0];
    assert_eq!(pending["request_data"], "first question");

    // Answer via the conversation; the run resumes and pauses again.
    let (status, second) = post(json!({
        "input": hil_input(&pending["request_id"], json!("first")),
        "conversation": conversation,
        "metadata": { "entity_id": "hitl" },
    }))
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let pending = &second["pending_requests"][0];
    assert_eq!(pending["request_data"], "second question");

    // Answer via previous_response_id, with upstream's `{response: v}` wrap.
    let (status, third) = post(json!({
        "input": hil_input(&pending["request_id"], json!({ "response": "Ada" })),
        "previous_response_id": second["id"],
        "metadata": { "entity_id": "hitl" },
    }))
    .await;
    assert_eq!(status, StatusCode::OK, "{third}");
    assert_eq!(third["outputs"][0], "done: Ada");
    assert_eq!(third["conversation"]["id"], conversation);
    assert_eq!(third["previous_response_id"], second["id"]);
    assert!(third.get("pending_requests").is_none());
}

#[tokio::test]
async fn hil_responses_without_a_paused_run_are_rejected() {
    let app = AgentHost::new()
        .workflow("hitl", hitl_workflow())
        .into_router();
    let (status, body) = post_json(
        app.clone(),
        "/v1/responses",
        &json!({
            "input": hil_input(&json!("req-1"), json!("x")),
            "conversation": "conv_empty",
            "metadata": { "entity_id": "hitl" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "checkpoint_not_found");

    // A response for a request id the paused run never made.
    let (_, first) = post_json(
        app.clone(),
        "/v1/responses",
        &json!({ "input": "q", "metadata": { "entity_id": "hitl" } }),
    )
    .await;
    let (status, body) = post_json(
        app,
        "/v1/responses",
        &json!({
            "input": hil_input(&json!("not-a-request"), json!("x")),
            "conversation": first["conversation"]["id"],
            "metadata": { "entity_id": "hitl" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_hil_response");
}

#[tokio::test]
async fn conversations_keep_their_own_paused_runs() {
    let app = AgentHost::new()
        .workflow("hitl", hitl_workflow())
        .into_router();
    let start = |q: &'static str| {
        let app = app.clone();
        async move {
            let body = json!({ "input": q, "metadata": { "entity_id": "hitl" } });
            post_json(app, "/v1/responses", &body).await
        }
    };
    let (_, a) = start("qa").await;
    let (_, b) = start("qb").await;
    assert_ne!(a["conversation"]["id"], b["conversation"]["id"]);

    let (status, done) = post_json(
        app.clone(),
        "/v1/responses",
        &json!({
            "input": hil_input(&a["pending_requests"][0]["request_id"], json!("A")),
            "conversation": a["conversation"]["id"],
            "metadata": { "entity_id": "hitl" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{done}");
    assert_eq!(done["outputs"][0], "done: A");

    // b's request id is unknown to a's conversation.
    let (status, _) = post_json(
        app,
        "/v1/responses",
        &json!({
            "input": hil_input(&b["pending_requests"][0]["request_id"], json!("B")),
            "conversation": a["conversation"]["id"],
            "metadata": { "entity_id": "hitl" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
