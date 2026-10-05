//! Delegating work to background sub-agents.
//!
//! Rust equivalent of upstream `agent_framework._harness._background_agents`
//! (`BackgroundAgentsProvider`, `BackgroundTaskInfo`, `BackgroundTaskStatus`).
//!
//! Each background task runs one sub-agent on its own session as a spawned
//! tokio task. Serializable task metadata lives in the parent session's
//! state (`{"next_task_id": n, "tasks": [...]}` under the source id); the
//! live task handles and child sessions are per-provider runtime state keyed
//! by session id, which — exactly as upstream — cannot survive a process
//! restart: a task recorded as running whose handle is gone is reported as
//! [`BackgroundTaskStatus::Lost`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::agent::{Agent, SupportsAgentRun};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{ApprovalMode, ToolDefinition};
use agent_framework_core::types::Message;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::sync::watch;

use crate::util::{empty_object_schema, function_tool, parse_args, SessionRef};

/// Default source id of [`BackgroundAgentsProvider`]. Mirrors upstream
/// `DEFAULT_BACKGROUND_AGENTS_SOURCE_ID`.
pub const DEFAULT_BACKGROUND_AGENTS_SOURCE_ID: &str = "background_agents";

/// Default maximum seconds the wait tool blocks. Mirrors upstream
/// `DEFAULT_BACKGROUND_AGENTS_WAIT_TIMEOUT_SECONDS`.
pub const DEFAULT_BACKGROUND_AGENTS_WAIT_TIMEOUT_SECONDS: u64 = 300;

/// Default instructions; `{background_agents}` is replaced with the agent
/// listing. Mirrors upstream `DEFAULT_BACKGROUND_AGENTS_INSTRUCTIONS`.
pub const DEFAULT_BACKGROUND_AGENTS_INSTRUCTIONS: &str = concat!(
    "## Background Agents\n\n",
    "You have access to background agents that can perform work on your behalf.\n\n",
    "- Use the `background_agents_*` tools to start tasks on background agents and check their results.\n",
    "- Creating a background task does not block, and background tasks run concurrently.\n",
    "- Important: Always wait for outstanding tasks to finish before you finish processing.\n",
    "- Important: After retrieving results from a completed task, clear it with background_agents_clear_completed_task to free memory, unless you plan to continue it with background_agents_continue_task.\n\n",
    "{background_agents}",
);

/// The names of the six tools [`BackgroundAgentsProvider`] contributes.
pub const BACKGROUND_AGENTS_TOOL_NAMES: [&str; 6] = [
    "background_agents_start_task",
    "background_agents_wait_for_first_completion",
    "background_agents_get_task_results",
    "background_agents_get_all_tasks",
    "background_agents_continue_task",
    "background_agents_clear_completed_task",
];

/// Status of a background task. Mirrors upstream `BackgroundTaskStatus`
/// (serialized as `"running"`, `"completed"`, `"failed"`, `"lost"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackgroundTaskStatus {
    /// Still executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished with an error (or was canceled).
    Failed,
    /// Recorded as running, but its runtime handle is gone.
    Lost,
}

impl BackgroundTaskStatus {
    /// The serialized value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Lost => "lost",
        }
    }
}

/// Metadata for one background task. Mirrors upstream `BackgroundTaskInfo`;
/// `result_text` / `error_text` are omitted from the serialized form when
/// unset, as upstream's `to_dict()` does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackgroundTaskInfo {
    /// The task id, unique within the parent session.
    pub id: i64,
    /// The name the task was started with.
    pub agent_name: String,
    /// The model-supplied description.
    pub description: String,
    /// The current status.
    pub status: BackgroundTaskStatus,
    /// The sub-agent's response text, once completed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_text: Option<String>,
    /// The failure text, once failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_text: Option<String>,
}

impl BackgroundTaskInfo {
    /// A new running task.
    pub fn new(id: i64, agent_name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id,
            agent_name: agent_name.into(),
            description: description.into(),
            status: BackgroundTaskStatus::Running,
            result_text: None,
            error_text: None,
        }
    }
}

/// One agent available for background delegation: the agent plus the
/// description shown in the instructions listing.
///
/// [`SupportsAgentRun`] has no `description` accessor, so it travels
/// alongside the agent; `From<Agent>` captures an [`Agent`]'s own.
#[derive(Clone)]
pub struct BackgroundAgent {
    agent: Arc<dyn SupportsAgentRun>,
    description: Option<String>,
}

impl BackgroundAgent {
    /// Wrap any agent (no description).
    pub fn new(agent: Arc<dyn SupportsAgentRun>) -> Self {
        Self {
            agent,
            description: None,
        }
    }

    /// Set the description shown in the agent listing.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl From<Agent> for BackgroundAgent {
    fn from(agent: Agent) -> Self {
        let description = agent.description().map(str::to_string);
        Self {
            agent: Arc::new(agent),
            description,
        }
    }
}

impl From<Arc<dyn SupportsAgentRun>> for BackgroundAgent {
    fn from(agent: Arc<dyn SupportsAgentRun>) -> Self {
        Self::new(agent)
    }
}

/// What a finished task left behind.
type Outcome = std::result::Result<String, String>;

/// A spawned task: its completion signal, outcome slot, and abort handle.
struct InFlight {
    done: watch::Receiver<bool>,
    outcome: Arc<Mutex<Option<(Outcome, AgentSession)>>>,
    abort: tokio::task::AbortHandle,
}

impl InFlight {
    fn is_finished(&self) -> bool {
        *self.done.borrow() || self.abort.is_finished()
    }
}

/// Non-serializable per-session runtime state. Mirrors upstream
/// `_RuntimeState`.
#[derive(Default)]
struct RuntimeState {
    in_flight: HashMap<i64, InFlight>,
    sessions: HashMap<i64, AgentSession>,
    closed: bool,
}

type Runtime = Arc<Mutex<RuntimeState>>;

#[derive(Deserialize)]
struct StartArgs {
    agent_name: String,
    input: String,
    description: String,
}

#[derive(Deserialize)]
struct WaitArgs {
    task_ids: Vec<i64>,
}

#[derive(Deserialize)]
struct TaskIdArgs {
    task_id: i64,
}

#[derive(Deserialize)]
struct ContinueArgs {
    task_id: i64,
    text: String,
}

/// Context provider letting an agent delegate work to background sub-agents.
///
/// Mirrors upstream `BackgroundAgentsProvider` (experimental upstream).
/// Tools (all `never_require`):
///
/// - `background_agents_start_task` — start a task on a named agent.
/// - `background_agents_wait_for_first_completion` — wait until the first of
///   the given tasks completes or the configured timeout expires (a timeout
///   leaves the tasks running).
/// - `background_agents_get_task_results` — a task's output.
/// - `background_agents_get_all_tasks` — list every task.
/// - `background_agents_continue_task` — send follow-up input to a finished
///   task's session.
/// - `background_agents_clear_completed_task` — forget a finished task.
///
/// **Security:** supplied agents receive text from the parent (possibly
/// derived from untrusted context) and their output is fed back into the
/// parent's context; only supply agents you have vetted and trust.
#[derive(Clone)]
pub struct BackgroundAgentsProvider {
    source_id: String,
    agents: Arc<Vec<(String, BackgroundAgent)>>,
    instructions: String,
    wait_timeout: Duration,
    runtimes: Arc<Mutex<HashMap<String, Runtime>>>,
    state_lock: Arc<Mutex<()>>,
}

impl std::fmt::Debug for BackgroundAgentsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundAgentsProvider")
            .field("source_id", &self.source_id)
            .field(
                "agents",
                &self
                    .agents
                    .iter()
                    .map(|(_, a)| a.agent.display_name())
                    .collect::<Vec<_>>(),
            )
            .field("wait_timeout", &self.wait_timeout)
            .finish_non_exhaustive()
    }
}

impl BackgroundAgentsProvider {
    /// A provider over `agents` with the default source id, instructions and
    /// wait timeout.
    ///
    /// Errors when `agents` is empty, an agent has no (non-blank) name, or
    /// two names collide case-insensitively.
    pub fn new(agents: Vec<BackgroundAgent>) -> Result<Self> {
        if agents.is_empty() {
            return Err(Error::Configuration(
                "At least one background agent must be provided.".into(),
            ));
        }
        let mut list: Vec<(String, BackgroundAgent)> = Vec::new();
        for agent in agents {
            let name = agent.agent.name().unwrap_or_default().to_string();
            if name.trim().is_empty() {
                return Err(Error::Configuration(
                    "All background agents must have a non-empty name.".into(),
                ));
            }
            let key = name.to_lowercase();
            if list.iter().any(|(k, _)| *k == key) {
                return Err(Error::Configuration(format!(
                    "Duplicate background agent name: '{name}'. Agent names must be unique (case-insensitive)."
                )));
            }
            list.push((key, agent));
        }
        let mut provider = Self {
            source_id: DEFAULT_BACKGROUND_AGENTS_SOURCE_ID.into(),
            agents: Arc::new(list),
            instructions: String::new(),
            wait_timeout: Duration::from_secs(DEFAULT_BACKGROUND_AGENTS_WAIT_TIMEOUT_SECONDS),
            runtimes: Arc::new(Mutex::new(HashMap::new())),
            state_lock: Arc::new(Mutex::new(())),
        };
        provider.instructions =
            provider.render_instructions(DEFAULT_BACKGROUND_AGENTS_INSTRUCTIONS);
        Ok(provider)
    }

    /// Override the source id (session-state key).
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// Override the instructions; a `{background_agents}` placeholder is
    /// replaced with the agent listing.
    pub fn instructions(mut self, instructions: &str) -> Self {
        self.instructions = self.render_instructions(instructions);
        self
    }

    /// Maximum seconds the wait tool blocks. Must be positive.
    pub fn wait_timeout_seconds(mut self, seconds: u64) -> Result<Self> {
        if seconds == 0 {
            return Err(Error::Configuration(
                "wait_timeout_seconds must be a positive integer.".into(),
            ));
        }
        self.wait_timeout = Duration::from_secs(seconds);
        Ok(self)
    }

    /// Like [`wait_timeout_seconds`](Self::wait_timeout_seconds) but with
    /// sub-second precision (tests use short timeouts).
    pub fn wait_timeout(mut self, timeout: Duration) -> Result<Self> {
        if timeout.is_zero() {
            return Err(Error::Configuration(
                "wait_timeout_seconds must be a positive integer.".into(),
            ));
        }
        self.wait_timeout = timeout;
        Ok(self)
    }

    /// The configured source id.
    pub fn get_source_id(&self) -> &str {
        &self.source_id
    }

    /// The rendered instructions.
    pub fn get_instructions(&self) -> &str {
        &self.instructions
    }

    /// The configured wait timeout.
    pub fn get_wait_timeout(&self) -> Duration {
        self.wait_timeout
    }

    fn render_instructions(&self, template: &str) -> String {
        let mut lines = vec!["Available background agents:".to_string()];
        for (_, agent) in self.agents.iter() {
            let mut line = format!("- {}", agent.agent.name().unwrap_or_default());
            if let Some(d) = agent.description.as_deref().filter(|d| !d.is_empty()) {
                line.push_str(": ");
                line.push_str(d);
            }
            lines.push(line);
        }
        template.replace("{background_agents}", &lines.join("\n"))
    }

    fn find_agent(&self, name: &str) -> Option<&BackgroundAgent> {
        let key = name.to_lowercase();
        self.agents.iter().find(|(k, _)| *k == key).map(|(_, a)| a)
    }

    /// Get (or replace a closed) runtime for `session_id`.
    fn runtime(&self, session_id: &str) -> Runtime {
        let mut runtimes = self.runtimes.lock().unwrap();
        match runtimes.get(session_id) {
            Some(r) if !r.lock().unwrap().closed => r.clone(),
            _ => {
                let r: Runtime = Arc::new(Mutex::new(RuntimeState::default()));
                runtimes.insert(session_id.to_string(), r.clone());
                r
            }
        }
    }

    /// The task list persisted in `session`'s state (no refresh).
    pub fn tasks(&self, session: &SessionRef) -> Vec<BackgroundTaskInfo> {
        let state = self.load_state(session);
        Self::parse_tasks(&state)
    }

    /// The tasks persisted as running in `session`'s state, after
    /// refreshing their status from the runtime (used by the loop helpers).
    pub fn running_tasks(&self, session: &SessionRef) -> Vec<BackgroundTaskInfo> {
        let runtime = self.runtime(&session.session_id);
        self.refresh(session, &runtime)
            .into_iter()
            .filter(|t| t.status == BackgroundTaskStatus::Running)
            .collect()
    }

    fn load_state(&self, session: &SessionRef) -> Map<String, Value> {
        match session.state.get(&self.source_id) {
            Some(Value::Object(map)) => map,
            _ => {
                let mut initial = Map::new();
                initial.insert("next_task_id".into(), json!(1));
                initial.insert("tasks".into(), json!([]));
                session
                    .state
                    .insert(self.source_id.clone(), Value::Object(initial.clone()));
                initial
            }
        }
    }

    fn parse_tasks(state: &Map<String, Value>) -> Vec<BackgroundTaskInfo> {
        state
            .get("tasks")
            .and_then(Value::as_array)
            .map(|tasks| {
                tasks
                    .iter()
                    .filter_map(|t| serde_json::from_value(t.clone()).ok())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn save(
        &self,
        session: &SessionRef,
        mut state: Map<String, Value>,
        tasks: &[BackgroundTaskInfo],
    ) {
        state.insert(
            "tasks".into(),
            Value::Array(
                tasks
                    .iter()
                    .filter_map(|t| serde_json::to_value(t).ok())
                    .collect(),
            ),
        );
        session
            .state
            .insert(self.source_id.clone(), Value::Object(state));
    }

    /// Fold a finished in-flight task into `info`. Mirrors upstream
    /// `_finalize_task`.
    fn finalize(info: &mut BackgroundTaskInfo, runtime: &mut RuntimeState) {
        let Some(in_flight) = runtime.in_flight.remove(&info.id) else {
            return;
        };
        let outcome = in_flight.outcome.lock().unwrap().take();
        match outcome {
            Some((Ok(text), session)) => {
                info.status = BackgroundTaskStatus::Completed;
                info.result_text = Some(text);
                runtime.sessions.insert(info.id, session);
            }
            Some((Err(error), session)) => {
                info.status = BackgroundTaskStatus::Failed;
                info.error_text = Some(error);
                runtime.sessions.insert(info.id, session);
            }
            None => {
                info.status = BackgroundTaskStatus::Failed;
                info.error_text = Some("Task was canceled.".into());
            }
        }
    }

    /// Refresh running tasks' status and persist changes. Mirrors upstream
    /// `_refresh_task_state`.
    fn refresh(&self, session: &SessionRef, runtime: &Runtime) -> Vec<BackgroundTaskInfo> {
        let _guard = self.state_lock.lock().unwrap();
        let state = self.load_state(session);
        let mut tasks = Self::parse_tasks(&state);
        let mut changed = false;
        let mut rt = runtime.lock().unwrap();
        for info in &mut tasks {
            if info.status != BackgroundTaskStatus::Running {
                continue;
            }
            match rt.in_flight.get(&info.id) {
                None => {
                    info.status = BackgroundTaskStatus::Lost;
                    changed = true;
                }
                Some(f) if f.is_finished() => {
                    Self::finalize(info, &mut rt);
                    changed = true;
                }
                Some(_) => {}
            }
        }
        drop(rt);
        if changed {
            self.save(session, state, &tasks);
        }
        tasks
    }

    /// Spawn `agent` on `sub_session` with `input`, tracking it as
    /// `task_id`. Errors when the runtime is closed.
    fn spawn(
        runtime: &Runtime,
        task_id: i64,
        agent: Arc<dyn SupportsAgentRun>,
        mut sub_session: AgentSession,
        input: String,
    ) -> std::result::Result<(), String> {
        let mut rt = runtime.lock().unwrap();
        if rt.closed {
            return Err("Session runtime is closed; cannot start background task.".into());
        }
        let (tx, rx) = watch::channel(false);
        let outcome: Arc<Mutex<Option<(Outcome, AgentSession)>>> = Arc::new(Mutex::new(None));
        let slot = outcome.clone();
        let handle = tokio::spawn(async move {
            let result = agent
                .run(vec![Message::user(input)], Some(&mut sub_session))
                .await;
            let outcome = match result {
                Ok(response) => Ok(response.text()),
                Err(e) => Err(e.to_string()),
            };
            *slot.lock().unwrap() = Some((outcome, sub_session));
            let _ = tx.send(true);
        });
        rt.in_flight.insert(
            task_id,
            InFlight {
                done: rx,
                outcome,
                abort: handle.abort_handle(),
            },
        );
        Ok(())
    }

    /// Release all runtime state for `session_id`, preventing leaks.
    ///
    /// With `cancel_running`, pending tasks are aborted and awaited for up
    /// to `timeout` (`None` = indefinitely); without it, pending tasks make
    /// this an error. Idempotent. Mirrors upstream `release_session`.
    pub async fn release_session(
        &self,
        session_id: &str,
        cancel_running: bool,
        timeout: Option<Duration>,
    ) -> Result<()> {
        let runtime = match self.runtimes.lock().unwrap().get(session_id) {
            Some(r) => r.clone(),
            None => return Ok(()),
        };
        let receivers: Vec<watch::Receiver<bool>> = {
            let mut rt = runtime.lock().unwrap();
            if rt.closed {
                return Ok(());
            }
            let pending: Vec<&InFlight> =
                rt.in_flight.values().filter(|f| !f.is_finished()).collect();
            if !pending.is_empty() && !cancel_running {
                return Err(Error::AgentExecution(format!(
                    "Cannot release session {session_id}: {} tasks still running.",
                    pending.len()
                )));
            }
            for f in &pending {
                f.abort.abort();
            }
            let receivers = pending.iter().map(|f| f.done.clone()).collect();
            rt.closed = true;
            receivers
        };
        let drain = async {
            for mut rx in receivers {
                let _ = rx.wait_for(|done| *done).await;
            }
        };
        let finished = match timeout {
            Some(t) => tokio::time::timeout(t, drain).await.is_ok(),
            None => {
                drain.await;
                true
            }
        };
        if !finished {
            tracing::warn!("Session release timed out before all tasks finished; abandoning them.");
        }
        {
            let mut rt = runtime.lock().unwrap();
            rt.in_flight.clear();
            rt.sessions.clear();
        }
        let mut runtimes = self.runtimes.lock().unwrap();
        if runtimes
            .get(session_id)
            .is_some_and(|r| Arc::ptr_eq(r, &runtime))
        {
            runtimes.remove(session_id);
        }
        Ok(())
    }

    fn tools(&self, session: SessionRef, runtime: Runtime) -> Vec<ToolDefinition> {
        let mut tools = Vec::new();

        let p = self.clone();
        let (s, rt) = (session.clone(), runtime.clone());
        tools.push(function_tool(
            BACKGROUND_AGENTS_TOOL_NAMES[0],
            "Start a background task on a named agent. Returns a confirmation with the task ID.",
            json!({
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string"},
                    "input": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["agent_name", "input", "description"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let (p, s, rt) = (p.clone(), s.clone(), rt.clone());
                async move {
                    let args: StartArgs = parse_args(BACKGROUND_AGENTS_TOOL_NAMES[0], args)?;
                    if rt.lock().unwrap().closed {
                        return Ok(Value::String(
                            "Error: Session is being released; cannot start a new background task.".into(),
                        ));
                    }
                    let Some(bg) = p.find_agent(&args.agent_name).cloned() else {
                        let available = p
                            .agents
                            .iter()
                            .map(|(_, a)| a.agent.name().unwrap_or_default().to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        return Ok(Value::String(format!(
                            "Error: No background agent found with name '{}'. Available agents: {available}",
                            args.agent_name
                        )));
                    };
                    let _guard = p.state_lock.lock().unwrap();
                    let mut state = p.load_state(&s);
                    let task_id = state.get("next_task_id").and_then(Value::as_i64).unwrap_or(1);
                    let sub_session = bg.agent.create_session();
                    if let Err(e) = Self::spawn(&rt, task_id, bg.agent.clone(), sub_session.clone(), args.input) {
                        return Ok(Value::String(format!("Error: {e}")));
                    }
                    rt.lock().unwrap().sessions.insert(task_id, sub_session);
                    state.insert("next_task_id".into(), json!(task_id + 1));
                    let mut tasks = Self::parse_tasks(&state);
                    tasks.push(BackgroundTaskInfo::new(task_id, &args.agent_name, &args.description));
                    p.save(&s, state, &tasks);
                    Ok(Value::String(format!(
                        "Background task {task_id} started on agent '{}'.",
                        args.agent_name
                    )))
                }
            },
        ));

        let p = self.clone();
        let (s, rt) = (session.clone(), runtime.clone());
        tools.push(function_tool(
            BACKGROUND_AGENTS_TOOL_NAMES[1],
            "Wait until the first of the specified tasks completes or the configured timeout expires.\n\nReturns the completed task's ID. On timeout, the tasks remain running and this tool\ncan be called again to continue waiting.",
            json!({
                "type": "object",
                "properties": {"task_ids": {"type": "array", "items": {"type": "integer"}}},
                "required": ["task_ids"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let (p, s, rt) = (p.clone(), s.clone(), rt.clone());
                async move {
                    let args: WaitArgs = parse_args(BACKGROUND_AGENTS_TOOL_NAMES[1], args)?;
                    let waitable: Vec<(i64, watch::Receiver<bool>)> = {
                        let guard = rt.lock().unwrap();
                        if guard.closed {
                            return Ok(Value::String(
                                "Error: Session is being released; cannot wait for background tasks.".into(),
                            ));
                        }
                        if args.task_ids.is_empty() {
                            return Ok(Value::String("Error: No task IDs provided.".into()));
                        }
                        args.task_ids
                            .iter()
                            .filter_map(|id| guard.in_flight.get(id).map(|f| (*id, f.done.clone())))
                            .collect()
                    };
                    if waitable.is_empty() {
                        let tasks = p.refresh(&s, &rt);
                        if let Some(done) = tasks
                            .iter()
                            .find(|t| args.task_ids.contains(&t.id) && t.status != BackgroundTaskStatus::Running)
                        {
                            return Ok(Value::String(format!(
                                "Task {} is not running; current status: {}.",
                                done.id,
                                done.status.as_str()
                            )));
                        }
                        return Ok(Value::String(
                            "Error: None of the specified task IDs correspond to running tasks.".into(),
                        ));
                    }
                    let waits = waitable.into_iter().map(|(id, mut rx)| {
                        Box::pin(async move {
                            let _ = rx.wait_for(|done| *done).await;
                            id
                        })
                    });
                    let completed_id = match tokio::time::timeout(p.wait_timeout, futures::future::select_all(waits)).await {
                        Ok((id, _, _)) => id,
                        Err(_) => {
                            return Ok(Value::String(format!(
                                "No background task completed within {} seconds. The tasks are still running; call this tool again if you wish to continue waiting.",
                                format_seconds(p.wait_timeout)
                            )))
                        }
                    };
                    let status = {
                        let _guard = p.state_lock.lock().unwrap();
                        let state = p.load_state(&s);
                        let mut tasks = Self::parse_tasks(&state);
                        let mut status = None;
                        if let Some(info) = tasks.iter_mut().find(|t| t.id == completed_id) {
                            let mut guard = rt.lock().unwrap();
                            if guard.in_flight.contains_key(&completed_id) {
                                Self::finalize(info, &mut guard);
                            }
                            status = Some(info.status);
                        }
                        p.save(&s, state, &tasks);
                        status
                    };
                    Ok(Value::String(format!(
                        "Task {completed_id} finished with status: {}.",
                        status.map(|s| s.as_str()).unwrap_or("Unknown")
                    )))
                }
            },
        ));

        let p = self.clone();
        let (s, rt) = (session.clone(), runtime.clone());
        tools.push(function_tool(
            BACKGROUND_AGENTS_TOOL_NAMES[2],
            "Get the text output of a background task by its ID.",
            json!({
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let (p, s, rt) = (p.clone(), s.clone(), rt.clone());
                async move {
                    let args: TaskIdArgs = parse_args(BACKGROUND_AGENTS_TOOL_NAMES[2], args)?;
                    let tasks = p.refresh(&s, &rt);
                    let id = args.task_id;
                    let Some(info) = tasks.iter().find(|t| t.id == id) else {
                        return Ok(Value::String(format!("Error: No task found with ID {id}.")));
                    };
                    Ok(Value::String(match info.status {
                        BackgroundTaskStatus::Completed => info
                            .result_text
                            .clone()
                            .filter(|t| !t.is_empty())
                            .unwrap_or_else(|| "(no output)".into()),
                        BackgroundTaskStatus::Failed => format!(
                            "Task failed: {}",
                            info.error_text
                                .clone()
                                .filter(|t| !t.is_empty())
                                .unwrap_or_else(|| "Unknown error".into())
                        ),
                        BackgroundTaskStatus::Lost => {
                            "Task state was lost (reference unavailable).".into()
                        }
                        BackgroundTaskStatus::Running => format!("Task {id} is still running."),
                    }))
                }
            },
        ));

        let p = self.clone();
        let (s, rt) = (session.clone(), runtime.clone());
        tools.push(function_tool(
            BACKGROUND_AGENTS_TOOL_NAMES[3],
            "List all background tasks with their IDs, statuses, agent names, and descriptions.",
            empty_object_schema(),
            ApprovalMode::NeverRequire,
            move |_args| {
                let (p, s, rt) = (p.clone(), s.clone(), rt.clone());
                async move {
                    let tasks = p.refresh(&s, &rt);
                    if tasks.is_empty() {
                        return Ok(Value::String("No tasks.".into()));
                    }
                    let mut lines = vec!["Tasks:".to_string()];
                    lines.extend(tasks.iter().map(task_line));
                    Ok(Value::String(lines.join("\n")))
                }
            },
        ));

        let p = self.clone();
        let (s, rt) = (session.clone(), runtime.clone());
        tools.push(function_tool(
            BACKGROUND_AGENTS_TOOL_NAMES[4],
            "Send follow-up input to a completed or failed task to resume its work.",
            json!({
                "type": "object",
                "properties": {"task_id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["task_id", "text"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let (p, s, rt) = (p.clone(), s.clone(), rt.clone());
                async move {
                    let args: ContinueArgs = parse_args(BACKGROUND_AGENTS_TOOL_NAMES[4], args)?;
                    if rt.lock().unwrap().closed {
                        return Ok(Value::String(
                            "Error: Session is being released; cannot continue a background task.".into(),
                        ));
                    }
                    let id = args.task_id;
                    let mut tasks = p.refresh(&s, &rt);
                    let Some(info) = tasks.iter_mut().find(|t| t.id == id) else {
                        return Ok(Value::String(format!("Error: No task found with ID {id}.")));
                    };
                    match info.status {
                        BackgroundTaskStatus::Lost => {
                            return Ok(Value::String(format!(
                                "Error: Task {id} cannot be continued because its session was lost. Start a new task instead."
                            )))
                        }
                        BackgroundTaskStatus::Running => {
                            return Ok(Value::String(format!(
                                "Error: Task {id} is still running. Wait for it to complete before continuing."
                            )))
                        }
                        _ => {}
                    }
                    let Some(bg) = p.find_agent(&info.agent_name).cloned() else {
                        return Ok(Value::String(format!(
                            "Error: Agent '{}' is no longer available.",
                            info.agent_name
                        )));
                    };
                    let Some(sub_session) = rt.lock().unwrap().sessions.get(&id).cloned() else {
                        return Ok(Value::String(format!(
                            "Error: Session for task {id} is no longer available."
                        )));
                    };
                    if let Err(e) = Self::spawn(&rt, id, bg.agent.clone(), sub_session, args.text) {
                        return Ok(Value::String(format!("Error: {e}")));
                    }
                    info.status = BackgroundTaskStatus::Running;
                    info.result_text = None;
                    info.error_text = None;
                    let _guard = p.state_lock.lock().unwrap();
                    let state = p.load_state(&s);
                    p.save(&s, state, &tasks);
                    Ok(Value::String(format!("Task {id} continued with new input.")))
                }
            },
        ));

        let p = self.clone();
        let (s, rt) = (session, runtime);
        tools.push(function_tool(
            BACKGROUND_AGENTS_TOOL_NAMES[5],
            "Remove a completed or failed task and release its session to free memory.",
            json!({
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let (p, s, rt) = (p.clone(), s.clone(), rt.clone());
                async move {
                    let args: TaskIdArgs = parse_args(BACKGROUND_AGENTS_TOOL_NAMES[5], args)?;
                    if rt.lock().unwrap().closed {
                        return Ok(Value::String(
                            "Error: Session is being released; cannot clear tasks.".into(),
                        ));
                    }
                    let id = args.task_id;
                    let tasks = p.refresh(&s, &rt);
                    let Some(info) = tasks.iter().find(|t| t.id == id) else {
                        return Ok(Value::String(format!("Error: No task found with ID {id}.")));
                    };
                    if info.status == BackgroundTaskStatus::Running {
                        return Ok(Value::String(format!(
                            "Error: Task {id} is still running. Wait for it to complete before clearing."
                        )));
                    }
                    let remaining: Vec<BackgroundTaskInfo> = tasks.into_iter().filter(|t| t.id != id).collect();
                    {
                        let mut guard = rt.lock().unwrap();
                        guard.in_flight.remove(&id);
                        guard.sessions.remove(&id);
                    }
                    let _guard = p.state_lock.lock().unwrap();
                    let state = p.load_state(&s);
                    p.save(&s, state, &remaining);
                    Ok(Value::String(format!("Task {id} cleared.")))
                }
            },
        ));
        tools
    }
}

fn task_line(t: &BackgroundTaskInfo) -> String {
    format!(
        "- Task {} [{}] ({}): {}",
        t.id,
        t.status.as_str(),
        t.agent_name,
        t.description
    )
}

/// Render a duration in seconds the way upstream's int seconds print.
fn format_seconds(d: Duration) -> String {
    if d.subsec_nanos() == 0 {
        d.as_secs().to_string()
    } else {
        format!("{}", d.as_secs_f64())
    }
}

#[async_trait]
impl ContextProvider for BackgroundAgentsProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session = SessionRef::from_context(ctx, "BackgroundAgentsProvider")?;
        let runtime = self.runtime(&session.session_id);
        ctx.add_instructions(self.instructions.clone());
        ctx.tools
            .extend(self.tools(session.clone(), runtime.clone()));
        let tasks = self.refresh(&session, &runtime);
        if !tasks.is_empty() {
            let mut lines = vec!["### Current background tasks".to_string()];
            lines.extend(tasks.iter().map(task_line));
            ctx.messages.push(Message::user(lines.join("\n")));
        }
        Ok(())
    }
}
