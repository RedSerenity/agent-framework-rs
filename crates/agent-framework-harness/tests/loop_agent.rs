//! `LoopAgent` — modeled on upstream `test_harness_loop.py`.

mod common;

use std::sync::{Arc, Mutex};

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, ChatResponse, Message, Role, UsageDetails};
use agent_framework_harness::background_agents::{BackgroundAgent, BackgroundAgentsProvider};
use agent_framework_harness::loop_agent::{
    background_tasks_running, background_tasks_running_message, has_pending_approval_request,
    next_message_fn, record_feedback_fn, restore_session, should_continue_fn, todos_remaining,
    todos_remaining_message, HarnessProviders, JudgeOptions, JudgeVerdict, LoopAgent, LoopContext,
    LoopDecision, DEFAULT_JUDGE_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS, DEFAULT_NEXT_MESSAGE,
};
use agent_framework_harness::mode::{set_agent_mode, AgentModeProvider, DEFAULT_MODE_SOURCE_ID};
use agent_framework_harness::todo::{TodoItem, TodoProvider, TodoSessionStore, TodoStore};
use agent_framework_harness::util::SessionRef;
use common::{approval_request, approval_response, call, text_response, MockClient, ScriptedAgent};
use futures::StreamExt;
use serde_json::json;

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|m| format!("{}:{}", m.role.as_str(), m.text()))
        .collect()
}

type Observed = (usize, String, usize, Option<String>);

fn always() -> agent_framework_harness::loop_agent::ShouldContinueFn {
    should_continue_fn(|_ctx: &LoopContext| true)
}

#[test]
fn configuration_defaults_and_validation() {
    let agent = LoopAgent::new(Arc::new(ScriptedAgent::default()), always());
    assert_eq!(agent.get_max_iterations(), Some(DEFAULT_MAX_ITERATIONS));
    assert!(agent.clone().max_iterations(Some(0)).is_err());
    assert_eq!(
        agent.max_iterations(None).unwrap().get_max_iterations(),
        None
    );
    let judge = LoopAgent::with_judge(
        Arc::new(ScriptedAgent::default()),
        Arc::new(MockClient::default()),
        JudgeOptions::default(),
    )
    .unwrap();
    assert_eq!(
        judge.get_max_iterations(),
        Some(DEFAULT_JUDGE_MAX_ITERATIONS)
    );
}

#[tokio::test]
async fn stops_at_max_iterations_and_aggregates_the_transcript() {
    let inner = ScriptedAgent::new(vec![
        AgentResponse {
            usage_details: Some(UsageDetails {
                input_token_count: Some(1),
                ..Default::default()
            }),
            ..text_response("one")
        },
        AgentResponse {
            usage_details: Some(UsageDetails {
                input_token_count: Some(2),
                ..Default::default()
            }),
            ..text_response("two")
        },
        text_response("three"),
    ]);
    let agent = LoopAgent::new(Arc::new(inner.clone()), always())
        .max_iterations(Some(3))
        .unwrap()
        .inject_progress(false);
    let out = agent.run(vec![Message::user("task")], None).await.unwrap();
    assert_eq!(inner.inputs().len(), 3);
    assert_eq!(
        texts(&out.messages),
        vec![
            "assistant:one".to_string(),
            format!("user:{DEFAULT_NEXT_MESSAGE}"),
            "assistant:two".into(),
            format!("user:{DEFAULT_NEXT_MESSAGE}"),
            "assistant:three".into(),
        ]
    );
    assert_eq!(out.usage_details.unwrap().input_token_count, Some(3));
    let final_only = LoopAgent::new(
        Arc::new(ScriptedAgent::new(vec![
            text_response("a"),
            text_response("b"),
        ])),
        always(),
    )
    .max_iterations(Some(2))
    .unwrap()
    .return_final_only(true);
    assert_eq!(
        final_only
            .run(vec![Message::user("t")], None)
            .await
            .unwrap()
            .text(),
        "b"
    );
}

#[tokio::test]
async fn should_continue_receives_context_and_feedback_flows_to_callables() {
    let seen: Arc<Mutex<Vec<Observed>>> = Arc::default();
    let record = seen.clone();
    let inner = ScriptedAgent::new(vec![text_response("draft"), text_response("final DONE")]);
    let agent = LoopAgent::new(
        Arc::new(inner.clone()),
        Arc::new(move |ctx: LoopContext| {
            record.lock().unwrap().push((
                ctx.iteration,
                ctx.last_result.text(),
                ctx.original_messages.len(),
                ctx.feedback.clone(),
            ));
            let done = ctx.last_result.text().contains("DONE");
            Box::pin(async move {
                Ok(LoopDecision::from((
                    !done,
                    Some("needs polish".to_string()),
                )))
            })
        }),
    )
    .next_message(next_message_fn(|ctx| {
        Some(format!(
            "Feedback: {}",
            ctx.feedback.clone().unwrap_or_default()
        ))
    }))
    .record_feedback(record_feedback_fn(|ctx| {
        Some(format!("iter {}", ctx.iteration))
    }));
    let out = agent.run(vec![Message::user("write")], None).await.unwrap();
    assert_eq!(
        out.text(),
        "draftProgress so far:\n- iter 1Feedback: needs polishfinal DONE"
    );
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen[0], (1, "draft".into(), 1, None));
    assert_eq!(seen[1].0, 2);
    // Without a session the full log is injected ahead of the nudge.
    assert_eq!(
        texts(&inner.inputs()[1]),
        vec![
            "user:Progress so far:\n- iter 1",
            "user:Feedback: needs polish"
        ]
    );
}

#[tokio::test]
async fn progress_injection_uses_latest_entry_with_a_session() {
    let inner = ScriptedAgent::new(vec![
        text_response("r1"),
        text_response("r2"),
        text_response("r3"),
    ]);
    let agent = LoopAgent::new(Arc::new(inner.clone()), always())
        .max_iterations(Some(3))
        .unwrap();
    let mut session = AgentSession::new();
    agent
        .run(vec![Message::user("t")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(
        texts(&inner.inputs()[2]),
        vec![
            "user:Progress so far:\n- r2".to_string(),
            format!("user:{DEFAULT_NEXT_MESSAGE}")
        ]
    );
}

#[tokio::test]
async fn next_message_returning_none_reuses_previous_input() {
    let inner = ScriptedAgent::new(vec![text_response("a"), text_response("b")]);
    let agent = LoopAgent::new(Arc::new(inner.clone()), always())
        .max_iterations(Some(2))
        .unwrap()
        .next_message(next_message_fn(|_| None));
    agent.run(vec![Message::user("same")], None).await.unwrap();
    assert_eq!(texts(&inner.inputs()[1]), vec!["user:same"]);
}

#[tokio::test]
async fn fresh_context_restarts_from_original_plus_progress_and_restores_state() {
    let inner = ScriptedAgent::new(vec![text_response("p1"), text_response("p2")]);
    let mut session = AgentSession::new();
    session.state.insert("pre", json!(1));
    // Mutate state during the first iteration via should_continue.
    let agent = {
        let marker = session.state.clone();
        LoopAgent::new(
            Arc::new(inner.clone()),
            Arc::new(move |_ctx: LoopContext| {
                marker.insert("during", json!(true));
                Box::pin(async { Ok(true.into()) })
            }),
        )
        .max_iterations(Some(2))
        .unwrap()
        .fresh_context(true)
        .additional_instructions("Be thorough.")
        .next_message(next_message_fn(|_| Some("go on".into())))
        .record_feedback(record_feedback_fn(|ctx| Some(ctx.last_result.text())))
        .providers(HarnessProviders::default())
        .inject_progress(true)
        .return_final_only(false)
    };
    let _ = agent
        .run(vec![Message::user("task")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(
        texts(&inner.inputs()[1]),
        vec![
            "system:Be thorough.",
            "user:task",
            "user:Progress so far:\n- p1",
            "user:go on"
        ]
    );
    // In-loop state was rolled back before the second iteration.
    assert!(session.state.get("during").is_none());
    assert_eq!(session.state.get("pre"), Some(json!(1)));
}

#[tokio::test]
async fn restore_session_resets_state_and_service_id_in_place() {
    let mut session = AgentSession::new();
    session.state.insert("keep", json!("old"));
    let snapshot = session.to_dict();
    let sharer = session.state.clone();
    session.state.insert("keep", json!("new"));
    session.state.insert("extra", json!(1));
    session.set_service_session_id("conv-1");
    let id = session.session_id().to_string();
    restore_session(&mut session, &snapshot).unwrap();
    assert_eq!(session.session_id(), id);
    assert_eq!(session.service_session_id(), None);
    assert_eq!(
        sharer.get("keep"),
        Some(json!("old")),
        "holders of the state see the reset"
    );
    assert!(sharer.get("extra").is_none());
}

#[tokio::test]
async fn escape_hatch_stops_on_a_pending_approval_request() {
    let req = approval_request("r1", call("c1", "deploy", json!({})));
    let inner = ScriptedAgent::new(vec![
        text_response("thinking"),
        approval_response(vec![req]),
    ]);
    let agent = LoopAgent::new(Arc::new(inner.clone()), always());
    let out = agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(inner.inputs().len(), 2);
    assert!(has_pending_approval_request(&out));
    assert!(!has_pending_approval_request(&text_response("x")));
}

#[tokio::test]
async fn judge_loops_until_answered_and_feeds_reasoning_back() {
    let judge = MockClient::new(vec![
        ChatResponse::from_text(r#"{"answered": false, "reasoning": "missing sources"}"#),
        ChatResponse::from_text(r#"{"answered": true, "reasoning": "complete"}"#),
    ]);
    let inner = ScriptedAgent::new(vec![text_response("v1"), text_response("v2")]);
    let agent = LoopAgent::with_judge(
        Arc::new(inner.clone()),
        Arc::new(judge.clone()),
        JudgeOptions {
            criteria: vec!["Cite sources".into()],
            ..Default::default()
        },
    )
    .unwrap();
    let out = agent
        .run(vec![Message::user("research X")], None)
        .await
        .unwrap();
    assert!(out.text().ends_with("v2"));
    assert_eq!(inner.inputs().len(), 2);
    // Criteria became a system instruction for the agent...
    assert_eq!(
        texts(&inner.inputs()[0])[0],
        "system:Your response must satisfy all of the following criteria:\n- Cite sources"
    );
    // ...and were rendered into the judge instructions.
    let judge_call = &judge.calls()[0];
    assert!(judge_call[0]
        .text()
        .ends_with("The response must satisfy all of the following criteria:\n- Cite sources"));
    assert_eq!(
        judge_call[1].text(),
        "Evaluate the agent's work. The user's original request follows:"
    );
    assert_eq!(
        judge_call.last().unwrap().text(),
        "Has the original request been fully addressed?"
    );
    assert!(judge.options()[0].response_format.is_some());
    let nudge = inner.inputs()[1].last().unwrap().text();
    assert!(
        nudge.contains("Evaluator feedback: missing sources"),
        "{nudge}"
    );
}

#[tokio::test]
async fn judge_text_fallback_more_wins_and_markerless_keeps_looping() {
    let judge = MockClient::new(vec![
        ChatResponse::from_text("VERDICT: DONE but also VERDICT: MORE"),
        ChatResponse::from_text("no marker at all"),
        ChatResponse::from_text("All good.\nVERDICT: DONE"),
    ]);
    let inner = ScriptedAgent::new(vec![]);
    let agent = LoopAgent::with_judge(
        Arc::new(inner.clone()),
        Arc::new(judge),
        JudgeOptions::default(),
    )
    .unwrap();
    agent.run(vec![Message::user("q")], None).await.unwrap();
    assert_eq!(inner.inputs().len(), 3);
}

#[tokio::test]
async fn judge_custom_parser_owns_interpretation() {
    let judge = MockClient::new(vec![ChatResponse::from_text("yes")]);
    let inner = ScriptedAgent::new(vec![]);
    let agent = LoopAgent::with_judge(
        Arc::new(inner.clone()),
        Arc::new(judge),
        JudgeOptions {
            response_format: None,
            verdict_parser: Some(Arc::new(|r: ChatResponse| {
                let answered = r.text() == "yes";
                Box::pin(async move {
                    Ok(JudgeVerdict {
                        answered,
                        reasoning: String::new(),
                    })
                })
            })),
            instructions: Some("Judge. {{criteria}}".into()),
            ..Default::default()
        },
    )
    .unwrap();
    agent.run(vec![Message::user("q")], None).await.unwrap();
    assert_eq!(inner.inputs().len(), 1);
}

#[tokio::test]
async fn todos_remaining_helpers_follow_store_and_mode() {
    let todo = TodoProvider::new();
    let mode = AgentModeProvider::new();
    let session = AgentSession::new();
    let sref = SessionRef::from_session(&session);
    TodoSessionStore
        .save_state(&sref, &[TodoItem::new(1, "Write report", None)], 2, "todo")
        .await
        .unwrap();
    let ctx = |providers: HarnessProviders, session: Option<AgentSession>| LoopContext {
        iteration: 1,
        last_result: text_response("x"),
        messages: vec![],
        original_messages: vec![],
        session,
        providers,
        progress: vec![],
        feedback: None,
    };
    let providers = HarnessProviders {
        todo: Some(todo.clone()),
        mode: Some(mode.clone()),
        background_agents: None,
    };
    let any_mode = todos_remaining(None).unwrap();
    assert!(
        any_mode(ctx(providers.clone(), Some(session.clone())))
            .await
            .unwrap()
            .continue_loop
    );
    assert!(
        !any_mode(ctx(providers.clone(), None))
            .await
            .unwrap()
            .continue_loop
    );
    assert!(
        !any_mode(ctx(HarnessProviders::default(), Some(session.clone())))
            .await
            .unwrap()
            .continue_loop
    );
    let execute_only = todos_remaining(Some(vec!["Execute".into()])).unwrap();
    assert!(
        !execute_only(ctx(providers.clone(), Some(session.clone())))
            .await
            .unwrap()
            .continue_loop
    );
    set_agent_mode(
        &session.state,
        "execute",
        DEFAULT_MODE_SOURCE_ID,
        None,
        true,
    )
    .unwrap();
    assert!(
        execute_only(ctx(providers.clone(), Some(session.clone())))
            .await
            .unwrap()
            .continue_loop
    );
    assert!(todos_remaining(Some(vec![])).is_err());
    let message = todos_remaining_message()(ctx(providers.clone(), Some(session.clone())))
        .await
        .unwrap()
        .unwrap();
    assert!(message[0].text().starts_with("You still have 1 open todo item(s) that must be addressed before you can finish:\n- Write report"));
    assert!(
        todos_remaining_message()(ctx(HarnessProviders::default(), Some(session.clone())))
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn background_helpers_report_running_tasks() {
    struct Slow;
    #[async_trait::async_trait]
    impl SupportsAgentRun for Slow {
        async fn run(
            &self,
            _m: Vec<Message>,
            _s: Option<&mut AgentSession>,
        ) -> agent_framework_core::Result<AgentResponse> {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            Ok(text_response("late"))
        }
        fn id(&self) -> &str {
            "slow"
        }
        fn name(&self) -> Option<&str> {
            Some("slow")
        }
    }
    let provider =
        BackgroundAgentsProvider::new(vec![BackgroundAgent::new(Arc::new(Slow))]).unwrap();
    let session = AgentSession::new();
    let mut c = common::ctx(session.session_id(), &session.state, vec![]);
    agent_framework_core::memory::ContextProvider::before_run(&provider, &mut c)
        .await
        .unwrap();
    let providers = HarnessProviders {
        background_agents: Some(provider.clone()),
        ..Default::default()
    };
    let ctx = || LoopContext {
        iteration: 1,
        last_result: text_response("x"),
        messages: vec![],
        original_messages: vec![],
        session: Some(session.clone()),
        providers: providers.clone(),
        progress: vec![],
        feedback: None,
    };
    assert!(
        !background_tasks_running()(ctx())
            .await
            .unwrap()
            .continue_loop
    );
    assert!(background_tasks_running_message()(ctx())
        .await
        .unwrap()
        .is_none());
    common::invoke_text(
        &c.tools,
        "background_agents_start_task",
        json!({"agent_name": "slow", "input": "x", "description": "Slow job"}),
    )
    .await;
    assert!(
        background_tasks_running()(ctx())
            .await
            .unwrap()
            .continue_loop
    );
    let message = background_tasks_running_message()(ctx())
        .await
        .unwrap()
        .unwrap();
    assert!(message[0].text().contains("- #1 (slow): Slow job"));
    provider
        .release_session(session.session_id(), true, None)
        .await
        .unwrap();
}

#[tokio::test]
async fn streaming_reyields_updates_and_injects_nudges_as_user_updates() {
    let inner = ScriptedAgent::new(vec![text_response("s1"), text_response("s2")]);
    let agent = LoopAgent::new(Arc::new(inner.clone()), always())
        .max_iterations(Some(2))
        .unwrap()
        .inject_progress(false);
    let updates: Vec<_> = agent
        .run_stream(vec![Message::user("t")], None, None)
        .await
        .unwrap()
        .collect()
        .await;
    let rendered: Vec<String> = updates
        .into_iter()
        .map(|u| u.unwrap())
        .map(|u| {
            format!(
                "{}:{}",
                u.role
                    .as_ref()
                    .map(|r| r.as_str().to_string())
                    .unwrap_or_default(),
                u.text()
            )
        })
        .collect();
    assert_eq!(
        rendered,
        vec![
            "assistant:s1".to_string(),
            format!("user:{DEFAULT_NEXT_MESSAGE}"),
            "assistant:s2".into()
        ]
    );
    let req = approval_request("r", call("c", "t", json!({})));
    let inner = ScriptedAgent::new(vec![approval_response(vec![req])]);
    let agent = LoopAgent::new(Arc::new(inner.clone()), always());
    let updates: Vec<_> = agent
        .run_stream(vec![Message::user("t")], None, None)
        .await
        .unwrap()
        .collect()
        .await;
    assert_eq!(
        inner.inputs().len(),
        1,
        "streaming escape hatch stops the loop"
    );
    assert!(!updates.is_empty());
    let _ = Role::user();
}
