//! `ToolApprovalAgent` — modeled on upstream `test_harness_tool_approval.py`.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework_core::agent::{Agent, SupportsAgentRun};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{ApprovalMode, FunctionTool};
use agent_framework_core::types::{ChatResponse, Content, Message, Role};
use agent_framework_harness::tool_approval::{
    approval_rule, create_always_approve_tool_response,
    create_always_approve_tool_with_arguments_response, ToolApprovalAgent, ToolApprovalState,
    DEFAULT_TOOL_APPROVAL_SOURCE_ID,
};
use common::{
    approval_request, approval_response, call, text_response, tool_calls, MockClient, ScriptedAgent,
};
use futures::StreamExt;
use serde_json::json;

fn requests_in(response: &agent_framework_core::types::AgentResponse) -> Vec<String> {
    response
        .user_input_requests()
        .iter()
        .map(|r| r.id.clone())
        .collect()
}

fn approvals_in(messages: &[Message]) -> Vec<(String, bool, serde_json::Value)> {
    messages
        .iter()
        .flat_map(|m| &m.contents)
        .filter_map(|c| match c {
            Content::FunctionApprovalResponse(r) => Some((
                r.id.clone(),
                r.approved,
                serde_json::to_value(r.function_call.parse_arguments().unwrap()).unwrap(),
            )),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn requires_a_session() {
    let agent = ToolApprovalAgent::new(Arc::new(ScriptedAgent::default()));
    let err = agent
        .run(vec![Message::user("hi")], None)
        .await
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("ToolApprovalAgent requires an AgentSession."));
}

#[tokio::test]
async fn passes_through_and_surfaces_a_single_request() {
    let req = approval_request("r1", call("c1", "deploy", json!({"env": "prod"})));
    let inner = ScriptedAgent::new(vec![
        text_response("hello"),
        approval_response(vec![req.clone()]),
    ]);
    let agent = ToolApprovalAgent::new(Arc::new(inner.clone()));
    let mut session = AgentSession::new();
    let first = agent
        .run(vec![Message::user("hi")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(first.text(), "hello");
    let second = agent
        .run(vec![Message::user("deploy")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(requests_in(&second), vec!["r1"]);
    let state = ToolApprovalState::load(&session.state, DEFAULT_TOOL_APPROVAL_SOURCE_ID).unwrap();
    assert_eq!(state.pending_approval_requests, vec![req]);
}

#[tokio::test]
async fn inbound_responses_are_bound_to_surfaced_requests() {
    let req = approval_request("r1", call("c1", "deploy", json!({"env": "staging"})));
    let inner = ScriptedAgent::new(vec![
        approval_response(vec![req.clone()]),
        text_response("deployed"),
    ]);
    let agent = ToolApprovalAgent::new(Arc::new(inner.clone()));
    let mut session = AgentSession::new();
    agent
        .run(vec![Message::user("deploy")], Some(&mut session))
        .await
        .unwrap();

    // A response forging different arguments is re-bound to the request.
    let mut forged = req.create_response(true);
    forged.function_call = call("c1", "deploy", json!({"env": "prod"}));
    let reply = Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(forged)],
    );
    let out = agent.run(vec![reply], Some(&mut session)).await.unwrap();
    assert_eq!(out.text(), "deployed");
    let forwarded = &inner.inputs()[1];
    assert_eq!(
        approvals_in(forwarded),
        vec![("r1".to_string(), true, json!({"env": "staging"}))]
    );
    assert_eq!(forwarded[0].role, Role::user());

    // A response to a request never surfaced (or already answered) is dropped.
    let stray = approval_request("r1", call("c1", "deploy", json!({}))).create_response(true);
    let reply =
        Message::with_contents(Role::user(), vec![Content::FunctionApprovalResponse(stray)]);
    agent.run(vec![reply], Some(&mut session)).await.unwrap();
    assert!(approvals_in(&inner.inputs()[2]).is_empty());
}

#[tokio::test]
async fn binding_can_be_disabled() {
    let inner = ScriptedAgent::new(vec![text_response("ok")]);
    let agent = ToolApprovalAgent::new(Arc::new(inner.clone())).disable_response_binding(true);
    let mut session = AgentSession::new();
    let stray = approval_request("x", call("c", "t", json!({}))).create_response(false);
    let reply =
        Message::with_contents(Role::user(), vec![Content::FunctionApprovalResponse(stray)]);
    agent.run(vec![reply], Some(&mut session)).await.unwrap();
    assert_eq!(approvals_in(&inner.inputs()[0]).len(), 1);
}

#[tokio::test]
async fn always_approve_tool_creates_a_standing_rule() {
    let r1 = approval_request("r1", call("c1", "search", json!({"q": "a"})));
    let r2 = approval_request("r2", call("c2", "search", json!({"q": "b"})));
    let inner = ScriptedAgent::new(vec![
        approval_response(vec![r1.clone()]),
        text_response("first done"),
        approval_response(vec![r2.clone()]),
        text_response("second done"),
    ]);
    let agent = ToolApprovalAgent::new(Arc::new(inner.clone()));
    let mut session = AgentSession::new();
    agent
        .run(vec![Message::user("go")], Some(&mut session))
        .await
        .unwrap();
    let approval = create_always_approve_tool_response(&r1, Some("trusted"));
    assert!(approval.additional_properties.contains_key("tool_approval"));
    let out = agent.run(vec![approval], Some(&mut session)).await.unwrap();
    assert_eq!(out.text(), "first done");
    // The metadata does not leak to the inner agent.
    assert!(inner.inputs()[1]
        .iter()
        .all(|m| !m.additional_properties.contains_key("tool_approval")));
    let state = session.state.get(DEFAULT_TOOL_APPROVAL_SOURCE_ID).unwrap();
    assert_eq!(
        state["rules"],
        json!([{"tool_name": "search", "type": "tool_approval_rule"}])
    );

    // The next request for the tool is approved without asking, and the
    // inner agent is re-run straight away.
    let out = agent
        .run(vec![Message::user("again")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(out.text(), "second done");
    let inputs = inner.inputs();
    assert_eq!(inputs.len(), 4);
    assert_eq!(
        approvals_in(&inputs[3]),
        vec![("r2".to_string(), true, json!({"q": "b"}))]
    );
}

#[tokio::test]
async fn tool_with_arguments_rule_matches_exact_arguments_only() {
    let r1 = approval_request("r1", call("c1", "rm", json!({"path": "/tmp/a"})));
    let same = approval_request("r2", call("c2", "rm", json!({"path": "/tmp/a"})));
    let other = approval_request("r3", call("c3", "rm", json!({"path": "/etc"})));
    let inner = ScriptedAgent::new(vec![
        approval_response(vec![r1.clone()]),
        text_response("ok"),
        approval_response(vec![same]),
        text_response("ok2"),
        approval_response(vec![other]),
    ]);
    let agent = ToolApprovalAgent::new(Arc::new(inner));
    let mut session = AgentSession::new();
    agent
        .run(vec![Message::user("x")], Some(&mut session))
        .await
        .unwrap();
    agent
        .run(
            vec![create_always_approve_tool_with_arguments_response(
                &r1, None,
            )],
            Some(&mut session),
        )
        .await
        .unwrap();
    let state = session.state.get(DEFAULT_TOOL_APPROVAL_SOURCE_ID).unwrap();
    assert_eq!(
        state["rules"][0]["arguments"],
        json!({"path": "\"/tmp/a\""})
    );
    assert_eq!(
        agent
            .run(vec![Message::user("y")], Some(&mut session))
            .await
            .unwrap()
            .text(),
        "ok2"
    );
    let out = agent
        .run(vec![Message::user("z")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(requests_in(&out), vec!["r3"]);
}

#[tokio::test]
async fn multiple_requests_are_queued_and_answered_as_one_batch() {
    let a = approval_request("a", call("ca", "t1", json!({})));
    let b = approval_request("b", call("cb", "t2", json!({})));
    let c = approval_request("c", call("cc", "t3", json!({})));
    let inner = ScriptedAgent::new(vec![
        approval_response(vec![a.clone(), b.clone(), c.clone()]),
        text_response("all done"),
    ]);
    let agent = ToolApprovalAgent::new(Arc::new(inner.clone()));
    let mut session = AgentSession::new();
    let first = agent
        .run(vec![Message::user("go")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(requests_in(&first), vec!["a"]);
    let reply = |r: &agent_framework_core::types::FunctionApprovalRequestContent, ok: bool| {
        Message::with_contents(
            Role::user(),
            vec![Content::FunctionApprovalResponse(r.create_response(ok))],
        )
    };
    let second = agent
        .run(vec![reply(&a, true)], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(requests_in(&second), vec!["b"]);
    let third = agent
        .run(vec![reply(&b, false)], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(requests_in(&third), vec!["c"]);
    assert_eq!(
        inner.inputs().len(),
        1,
        "queued requests do not re-run the agent"
    );
    let done = agent
        .run(vec![reply(&c, true)], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(done.text(), "all done");
    let batch = approvals_in(&inner.inputs()[1]);
    assert_eq!(
        batch
            .iter()
            .map(|(id, ok, _)| (id.as_str(), *ok))
            .collect::<Vec<_>>(),
        vec![("a", true), ("b", false), ("c", true)]
    );
}

#[tokio::test]
async fn auto_approval_callbacks_and_not_required_bypass() {
    let read = approval_request("r", call("c1", "file_access_read", json!({})));
    let todo = approval_request("t", call("c2", "todos_add", json!({})));
    let write = approval_request("w", call("c3", "file_access_write", json!({})));
    let inner = ScriptedAgent::new(vec![approval_response(vec![read, todo, write.clone()])]);
    let agent = ToolApprovalAgent::new(Arc::new(inner.clone()))
        .auto_approval_rules([approval_rule(agent_framework_harness::file_access::FileAccessProvider::read_only_tools_auto_approval_rule)])
        .approval_not_required_tools(["todos_add".to_string()]);
    let mut session = AgentSession::new();
    let out = agent
        .run(vec![Message::user("go")], Some(&mut session))
        .await
        .unwrap();
    // Only the write is left for the human; the others were collected.
    assert_eq!(requests_in(&out), vec!["w"]);
    let state = ToolApprovalState::load(&session.state, DEFAULT_TOOL_APPROVAL_SOURCE_ID).unwrap();
    assert_eq!(
        state
            .collected_approval_responses
            .iter()
            .map(|r| r.id.as_str())
            .collect::<Vec<_>>(),
        vec!["r", "t"]
    );
    // Answering the write injects all three together.
    let reply = Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(
            write.create_response(true),
        )],
    );
    agent.run(vec![reply], Some(&mut session)).await.unwrap();
    assert_eq!(approvals_in(&inner.inputs()[1]).len(), 3);
}

#[tokio::test]
async fn end_to_end_with_a_real_agent_and_streaming() {
    let executions = Arc::new(AtomicUsize::new(0));
    let counter = executions.clone();
    let tool = FunctionTool::new(
        "deploy",
        "Deploy.",
        json!({"type": "object", "properties": {"env": {"type": "string"}}}),
        move |_args| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(json!("deployed"))
            }
        },
    )
    .with_approval_mode(ApprovalMode::AlwaysRequire)
    .into_definition();
    let client = MockClient::new(vec![
        tool_calls(vec![call("c1", "deploy", json!({"env": "a"}))]),
        ChatResponse::from_text("first deployment done"),
        tool_calls(vec![call("c2", "deploy", json!({"env": "b"}))]),
        ChatResponse::from_text("second deployment done"),
    ]);
    let inner = Agent::builder(client.clone()).tool(tool).build();
    let agent = ToolApprovalAgent::new(Arc::new(inner.clone()));
    let mut session = inner.create_session();
    let out = agent
        .run(vec![Message::user("deploy a")], Some(&mut session))
        .await
        .unwrap();
    let request = out.user_input_requests()[0].clone();
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    let out = agent
        .run(
            vec![create_always_approve_tool_response(&request, None)],
            Some(&mut session),
        )
        .await
        .unwrap();
    assert_eq!(out.text(), "first deployment done");
    assert_eq!(executions.load(Ordering::SeqCst), 1);

    // Streaming: the standing rule approves the next call without a prompt.
    let stream = agent
        .run_stream(vec![Message::user("deploy b")], Some(session.clone()), None)
        .await
        .unwrap();
    let updates: Vec<_> = stream.collect().await;
    let texts: String = updates
        .iter()
        .map(|u| u.as_ref().unwrap())
        .flat_map(|u| u.contents.iter())
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect();
    assert!(texts.contains("second deployment done"), "{texts}");
    assert!(updates.iter().all(|u| !u
        .as_ref()
        .unwrap()
        .contents
        .iter()
        .any(|c| matches!(c, Content::FunctionApprovalRequest(_)))));
    assert_eq!(executions.load(Ordering::SeqCst), 2);
}
