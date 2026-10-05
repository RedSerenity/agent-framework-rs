//! `BackgroundAgentsProvider` — modeled on upstream
//! `test_harness_background_agents.py`.

mod common;

use std::sync::Arc;
use std::time::Duration;

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::session::{AgentSession, SessionState};
use agent_framework_core::types::{AgentResponse, Message};
use agent_framework_harness::background_agents::{
    BackgroundAgent, BackgroundAgentsProvider, BackgroundTaskInfo, BackgroundTaskStatus,
    BACKGROUND_AGENTS_TOOL_NAMES,
};
use agent_framework_harness::util::SessionRef;
use async_trait::async_trait;
use common::{ctx, invoke_text, text_response, ScriptedAgent};
use serde_json::json;

/// An agent that answers after `delay`, or fails when `fail` is set.
struct TimedAgent {
    name: String,
    delay: Duration,
    fail: bool,
}

#[async_trait]
impl SupportsAgentRun for TimedAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        tokio::time::sleep(self.delay).await;
        if self.fail {
            return Err(Error::AgentExecution("boom".into()));
        }
        Ok(text_response(&format!("result for {}", messages[0].text())))
    }
    fn id(&self) -> &str {
        &self.name
    }
    fn name(&self) -> Option<&str> {
        Some(&self.name)
    }
}

fn timed(name: &str, delay_ms: u64, fail: bool) -> BackgroundAgent {
    BackgroundAgent::new(Arc::new(TimedAgent {
        name: name.into(),
        delay: Duration::from_millis(delay_ms),
        fail,
    }))
}

async fn tools(provider: &BackgroundAgentsProvider, state: &SessionState) -> SessionContext {
    let mut c = ctx("parent", state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    c
}

#[test]
fn constructor_validates_agents_and_timeout() {
    assert!(BackgroundAgentsProvider::new(vec![]).is_err());
    let unnamed = BackgroundAgent::new(Arc::new(ScriptedAgent::default()));
    assert!(BackgroundAgentsProvider::new(vec![unnamed])
        .unwrap_err()
        .to_string()
        .contains("non-empty name"));
    let dup = BackgroundAgentsProvider::new(vec![
        timed("Research", 0, false),
        timed("research", 0, false),
    ])
    .unwrap_err();
    assert!(dup
        .to_string()
        .contains("Duplicate background agent name: 'research'"));
    let provider = BackgroundAgentsProvider::new(vec![timed("a", 0, false)]).unwrap();
    assert_eq!(provider.get_wait_timeout(), Duration::from_secs(300));
    assert!(provider.clone().wait_timeout_seconds(0).is_err());
    assert_eq!(provider.get_source_id(), "background_agents");
}

#[tokio::test]
async fn injects_six_tools_and_instructions_with_listing() {
    let researcher = BackgroundAgent::new(Arc::new(TimedAgent {
        name: "researcher".into(),
        delay: Duration::ZERO,
        fail: false,
    }))
    .description("Finds things");
    let provider =
        BackgroundAgentsProvider::new(vec![researcher, timed("writer", 0, false)]).unwrap();
    let state = SessionState::new();
    let c = tools(&provider, &state).await;
    let names: Vec<&str> = c.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, BACKGROUND_AGENTS_TOOL_NAMES.to_vec());
    let instructions = c.instructions.unwrap();
    assert!(instructions.starts_with("## Background Agents"));
    assert!(instructions
        .ends_with("Available background agents:\n- researcher: Finds things\n- writer"));
    // The wait timeout is not model-settable.
    let wait = &c.tools[1];
    assert!(wait.parameters["properties"].get("timeout").is_none());
    let custom = BackgroundAgentsProvider::new(vec![timed("a", 0, false)])
        .unwrap()
        .instructions("Agents: {background_agents}");
    assert_eq!(
        custom.get_instructions(),
        "Agents: Available background agents:\n- a"
    );
}

#[tokio::test]
async fn start_wait_results_continue_and_clear() {
    let provider = BackgroundAgentsProvider::new(vec![timed("worker", 10, false)]).unwrap();
    let state = SessionState::new();
    let c = tools(&provider, &state).await;
    let t = &c.tools;
    assert_eq!(
        invoke_text(t, "background_agents_get_all_tasks", json!({})).await,
        "No tasks."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_start_task",
            json!({"agent_name": "nobody", "input": "x", "description": "d"})
        )
        .await,
        "Error: No background agent found with name 'nobody'. Available agents: worker"
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_start_task",
            json!({"agent_name": "WORKER", "input": "find", "description": "Find it"})
        )
        .await,
        "Background task 1 started on agent 'WORKER'."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_start_task",
            json!({"agent_name": "worker", "input": "two", "description": "Second"})
        )
        .await,
        "Background task 2 started on agent 'worker'."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "Task 1 is still running."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_clear_completed_task",
            json!({"task_id": 1})
        )
        .await,
        "Error: Task 1 is still running. Wait for it to complete before clearing."
    );
    let waited = invoke_text(
        t,
        "background_agents_wait_for_first_completion",
        json!({"task_ids": [1]}),
    )
    .await;
    assert_eq!(waited, "Task 1 finished with status: completed.");
    assert_eq!(
        invoke_text(
            t,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "result for find"
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_get_task_results",
            json!({"task_id": 9})
        )
        .await,
        "Error: No task found with ID 9."
    );
    // The serialized state follows upstream's shape.
    let saved = state.get("background_agents").unwrap();
    assert_eq!(saved["next_task_id"], 3);
    assert_eq!(
        saved["tasks"][0],
        json!({"id": 1, "agent_name": "WORKER", "description": "Find it", "status": "completed", "result_text": "result for find"})
    );
    // Continue a finished task on its own session.
    assert_eq!(
        invoke_text(
            t,
            "background_agents_continue_task",
            json!({"task_id": 1, "text": "more"})
        )
        .await,
        "Task 1 continued with new input."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_continue_task",
            json!({"task_id": 1, "text": "more"})
        )
        .await,
        "Error: Task 1 is still running. Wait for it to complete before continuing."
    );
    invoke_text(
        t,
        "background_agents_wait_for_first_completion",
        json!({"task_ids": [1, 2]}),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    let all = invoke_text(t, "background_agents_get_all_tasks", json!({})).await;
    assert_eq!(
        all,
        "Tasks:\n- Task 1 [completed] (WORKER): Find it\n- Task 2 [completed] (worker): Second"
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "result for more"
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_clear_completed_task",
            json!({"task_id": 1})
        )
        .await,
        "Task 1 cleared."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_clear_completed_task",
            json!({"task_id": 1})
        )
        .await,
        "Error: No task found with ID 1."
    );
    // The status message is injected on the next run.
    let next = tools(&provider, &state).await;
    assert_eq!(
        next.messages[0].text(),
        "### Current background tasks\n- Task 2 [completed] (worker): Second"
    );
}

#[tokio::test]
async fn wait_validates_inputs_and_reports_non_running() {
    let provider = BackgroundAgentsProvider::new(vec![timed("w", 0, false)]).unwrap();
    let state = SessionState::new();
    let c = tools(&provider, &state).await;
    let t = &c.tools;
    assert_eq!(
        invoke_text(
            t,
            "background_agents_wait_for_first_completion",
            json!({"task_ids": []})
        )
        .await,
        "Error: No task IDs provided."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_wait_for_first_completion",
            json!({"task_ids": [7]})
        )
        .await,
        "Error: None of the specified task IDs correspond to running tasks."
    );
    invoke_text(
        t,
        "background_agents_start_task",
        json!({"agent_name": "w", "input": "x", "description": "d"}),
    )
    .await;
    invoke_text(
        t,
        "background_agents_wait_for_first_completion",
        json!({"task_ids": [1]}),
    )
    .await;
    assert_eq!(
        invoke_text(
            t,
            "background_agents_wait_for_first_completion",
            json!({"task_ids": [1]})
        )
        .await,
        "Task 1 is not running; current status: completed."
    );
}

#[tokio::test]
async fn wait_timeout_returns_without_stopping_the_task() {
    let provider = BackgroundAgentsProvider::new(vec![timed("slow", 300, false)])
        .unwrap()
        .wait_timeout(Duration::from_millis(30))
        .unwrap();
    let state = SessionState::new();
    let c = tools(&provider, &state).await;
    let t = &c.tools;
    invoke_text(
        t,
        "background_agents_start_task",
        json!({"agent_name": "slow", "input": "x", "description": "d"}),
    )
    .await;
    let out = invoke_text(
        t,
        "background_agents_wait_for_first_completion",
        json!({"task_ids": [1]}),
    )
    .await;
    assert!(
        out.starts_with("No background task completed within 0.03 seconds."),
        "{out}"
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "Task 1 is still running."
    );
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(
        invoke_text(
            t,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "result for x"
    );
}

#[tokio::test]
async fn failed_and_lost_tasks() {
    let provider = BackgroundAgentsProvider::new(vec![timed("bad", 0, true)]).unwrap();
    let state = SessionState::new();
    let c = tools(&provider, &state).await;
    let t = &c.tools;
    invoke_text(
        t,
        "background_agents_start_task",
        json!({"agent_name": "bad", "input": "x", "description": "d"}),
    )
    .await;
    assert_eq!(
        invoke_text(
            t,
            "background_agents_wait_for_first_completion",
            json!({"task_ids": [1]})
        )
        .await,
        "Task 1 finished with status: failed."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_get_task_results",
            json!({"task_id": 1})
        )
        .await,
        "Task failed: agent execution error: boom"
    );
    // A task recorded as running with no runtime handle (e.g. after a
    // restart) is reported lost and cannot be continued.
    let mut info = BackgroundTaskInfo::new(5, "bad", "orphan");
    info.status = BackgroundTaskStatus::Running;
    let mut saved = state.get("background_agents").unwrap();
    saved["tasks"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::to_value(&info).unwrap());
    state.insert("background_agents", saved);
    assert_eq!(
        invoke_text(
            t,
            "background_agents_get_task_results",
            json!({"task_id": 5})
        )
        .await,
        "Task state was lost (reference unavailable)."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_continue_task",
            json!({"task_id": 5, "text": "x"})
        )
        .await,
        "Error: Task 5 cannot be continued because its session was lost. Start a new task instead."
    );
}

#[tokio::test]
async fn release_session_cancels_clears_and_is_idempotent() {
    let provider = BackgroundAgentsProvider::new(vec![timed("slow", 10_000, false)]).unwrap();
    let state = SessionState::new();
    let c = tools(&provider, &state).await;
    let t = &c.tools;
    invoke_text(
        t,
        "background_agents_start_task",
        json!({"agent_name": "slow", "input": "x", "description": "d"}),
    )
    .await;
    assert!(provider
        .release_session("parent", false, None)
        .await
        .is_err());
    provider
        .release_session("parent", true, Some(Duration::from_secs(1)))
        .await
        .unwrap();
    provider
        .release_session("parent", true, None)
        .await
        .unwrap();
    // Tools bound to the released runtime refuse new work.
    assert_eq!(
        invoke_text(
            t,
            "background_agents_start_task",
            json!({"agent_name": "slow", "input": "x", "description": "d"})
        )
        .await,
        "Error: Session is being released; cannot start a new background task."
    );
    assert_eq!(
        invoke_text(
            t,
            "background_agents_clear_completed_task",
            json!({"task_id": 1})
        )
        .await,
        "Error: Session is being released; cannot clear tasks."
    );
    // A fresh run gets a fresh runtime; the orphaned task reads as lost.
    let next = tools(&provider, &state).await;
    assert!(next.messages[0].text().contains("[lost]"));
    let session = SessionRef {
        session_id: "parent".into(),
        state: state.clone(),
    };
    assert!(provider.running_tasks(&session).is_empty());
}

#[tokio::test]
async fn task_info_serialization_omits_unset_texts() {
    let info = BackgroundTaskInfo::new(1, "a", "d");
    assert_eq!(
        serde_json::to_value(&info).unwrap(),
        json!({"id": 1, "agent_name": "a", "description": "d", "status": "running"})
    );
    assert_eq!(BackgroundTaskStatus::Lost.as_str(), "lost");
}
