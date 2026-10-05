//! `evaluate_agent` / `evaluate_workflow` orchestration.
//!
//! Mirrors upstream's "Public orchestration functions", "Workflow extraction
//! helpers" and "Internal helpers" regions of `_evaluation.py`.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;

use super::{
    to_eval_item, ConversationSplitter, EvalCheck, EvalItem, EvalQuery, EvalResults, EvalRunStatus,
    Evaluator, ExpectedToolCall, LocalEvaluator, ResultCounts,
};
use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::types::{AgentResponse, Message};
use crate::workflow::{Workflow, WorkflowEvent, WorkflowRun};

// ---------------------------------------------------------------------------
// Evaluator resolution
// ---------------------------------------------------------------------------

/// One entry of an evaluator list: a provider, or a bare check that is
/// grouped into a [`LocalEvaluator`] with its neighbours.
#[derive(Clone)]
enum EvaluatorEntry {
    Evaluator(Arc<dyn Evaluator>),
    Check(Arc<dyn EvalCheck>),
}

/// Normalize entries into concrete evaluators. Mirrors upstream's
/// `_resolve_evaluators`: each run of consecutive bare checks is wrapped in
/// one [`LocalEvaluator`], keeping the providers' relative order.
fn resolve_evaluators(entries: &[EvaluatorEntry]) -> Vec<Arc<dyn Evaluator>> {
    let mut resolved: Vec<Arc<dyn Evaluator>> = Vec::new();
    let mut pending: Vec<Arc<dyn EvalCheck>> = Vec::new();
    for entry in entries {
        match entry {
            EvaluatorEntry::Evaluator(ev) => {
                if !pending.is_empty() {
                    resolved.push(Arc::new(LocalEvaluator::from_checks(std::mem::take(
                        &mut pending,
                    ))));
                }
                resolved.push(ev.clone());
            }
            EvaluatorEntry::Check(check) => pending.push(check.clone()),
        }
    }
    if !pending.is_empty() {
        resolved.push(Arc::new(LocalEvaluator::from_checks(pending)));
    }
    resolved
}

/// The ` (<provider>)` suffix upstream appends to eval names when more than
/// one provider runs.
fn suffix(ev: &dyn Evaluator, count: usize) -> String {
    if count > 1 {
        format!(" ({})", ev.name())
    } else {
        String::new()
    }
}

/// Run every evaluator over `items` concurrently. Mirrors upstream's
/// `_run_evaluators`; the first error wins, as with `asyncio.gather`.
async fn run_evaluators(
    evaluators: &[Arc<dyn Evaluator>],
    items: &[EvalItem],
    eval_name: &str,
) -> Result<Vec<EvalResults>> {
    let count = evaluators.len();
    futures::future::try_join_all(evaluators.iter().map(|ev| {
        let name = format!("{eval_name}{}", suffix(ev.as_ref(), count));
        async move { ev.evaluate(items, &name).await }
    }))
    .await
}

macro_rules! evaluator_setters {
    () => {
        /// Add an evaluation provider.
        pub fn evaluator(mut self, evaluator: impl Evaluator + 'static) -> Self {
            self.evaluators
                .push(EvaluatorEntry::Evaluator(Arc::new(evaluator)));
            self
        }

        /// Add an already-shared evaluation provider.
        pub fn evaluator_arc(mut self, evaluator: Arc<dyn Evaluator>) -> Self {
            self.evaluators.push(EvaluatorEntry::Evaluator(evaluator));
            self
        }

        /// Add a bare check. Consecutive checks (with no provider between
        /// them) are grouped into one [`LocalEvaluator`], as upstream does
        /// with bare callables in its `evaluators` list.
        pub fn check(mut self, check: impl EvalCheck + 'static) -> Self {
            self.evaluators.push(EvaluatorEntry::Check(Arc::new(check)));
            self
        }

        /// Set the display name (upstream `eval_name`).
        pub fn eval_name(mut self, name: impl Into<String>) -> Self {
            self.eval_name = Some(name.into());
            self
        }

        /// Split strategy stamped on every item, overriding each
        /// evaluator's default (upstream `conversation_split`).
        pub fn conversation_split(mut self, split: impl ConversationSplitter + 'static) -> Self {
            self.conversation_split = Some(Arc::new(split));
            self
        }

        /// Run each query this many times (default 1) to measure
        /// consistency. Must be at least 1.
        pub fn num_repetitions(mut self, n: usize) -> Self {
            self.num_repetitions = n;
            self
        }

        /// Add one ground-truth expected output; one per query.
        pub fn expected_output(mut self, expected: impl Into<String>) -> Self {
            self.expected_output
                .get_or_insert_with(Vec::new)
                .push(expected.into());
            self
        }

        /// Add ground-truth expected outputs, one per query.
        pub fn expected_outputs<I, S>(mut self, expected: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.expected_output
                .get_or_insert_with(Vec::new)
                .extend(expected.into_iter().map(Into::into));
            self
        }

        /// Add one test query.
        pub fn query(mut self, query: impl Into<String>) -> Self {
            self.queries.get_or_insert_with(Vec::new).push(query.into());
            self
        }

        /// Add test queries.
        pub fn queries<I, S>(mut self, queries: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.queries
                .get_or_insert_with(Vec::new)
                .extend(queries.into_iter().map(Into::into));
            self
        }
    };
}

fn check_repetitions(n: usize) -> Result<()> {
    if n < 1 {
        return Err(Error::Configuration(format!(
            "num_repetitions must be >= 1, got {n}."
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// evaluate_agent
// ---------------------------------------------------------------------------

/// A request to run an agent against test queries and evaluate the results.
/// The builder form of upstream's `evaluate_agent(...)` keyword arguments.
///
/// Two modes:
///
/// - **Run + evaluate** — set [`agent`](Self::agent) and
///   [`queries`](Self::queries): each query is sent to the agent as one user
///   message, [`num_repetitions`](Self::num_repetitions) times.
/// - **Pre-existing responses** — set [`responses`](Self::responses) (one per
///   query) and [`queries`](Self::queries): the agent is not run, but if set
///   its function tools are still attached to the items.
///
/// Expected outputs and expected tool calls, when set, must have one entry
/// per query and are stamped on the items (repeated across repetitions).
///
/// ```no_run
/// use agent_framework_core::evaluation::{tool_call_args_match, EvaluateAgent, ExpectedToolCall};
/// use agent_framework_core::prelude::*;
/// use serde_json::json;
///
/// # async fn demo(agent: Agent) -> Result<()> {
/// let results = EvaluateAgent::new()
///     .agent(&agent)
///     .query("What's the weather in NYC?")
///     .expected_tool_calls_for_query([
///         ExpectedToolCall::new("get_weather").with_arguments([("location", json!("NYC"))]),
///     ])
///     .check(tool_call_args_match)
///     .run()
///     .await?;
/// results[0].raise_for_status(None)?;
/// # Ok(())
/// # }
/// ```
pub struct EvaluateAgent<'a> {
    agent: Option<&'a dyn SupportsAgentRun>,
    queries: Option<Vec<String>>,
    expected_output: Option<Vec<String>>,
    expected_tool_calls: Option<Vec<Vec<ExpectedToolCall>>>,
    responses: Option<Vec<AgentResponse>>,
    evaluators: Vec<EvaluatorEntry>,
    eval_name: Option<String>,
    context: Option<String>,
    conversation_split: Option<Arc<dyn ConversationSplitter>>,
    num_repetitions: usize,
}

impl Default for EvaluateAgent<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> EvaluateAgent<'a> {
    /// An empty request (one repetition, no evaluators).
    pub fn new() -> Self {
        Self {
            agent: None,
            queries: None,
            expected_output: None,
            expected_tool_calls: None,
            responses: None,
            evaluators: Vec::new(),
            eval_name: None,
            context: None,
            conversation_split: None,
            num_repetitions: 1,
        }
    }

    evaluator_setters!();

    /// The agent to run (and to take tool definitions from).
    pub fn agent(mut self, agent: &'a dyn SupportsAgentRun) -> Self {
        self.agent = Some(agent);
        self
    }

    /// Add the expected tool calls for the next query (one list per query).
    pub fn expected_tool_calls_for_query(
        mut self,
        calls: impl IntoIterator<Item = ExpectedToolCall>,
    ) -> Self {
        self.expected_tool_calls
            .get_or_insert_with(Vec::new)
            .push(calls.into_iter().collect());
        self
    }

    /// Add expected tool calls, one list per query.
    pub fn expected_tool_calls<I, C>(mut self, calls: I) -> Self
    where
        I: IntoIterator<Item = C>,
        C: IntoIterator<Item = ExpectedToolCall>,
    {
        self.expected_tool_calls
            .get_or_insert_with(Vec::new)
            .extend(calls.into_iter().map(|c| c.into_iter().collect()));
        self
    }

    /// Add one pre-existing response to evaluate without running the agent.
    pub fn response(mut self, response: AgentResponse) -> Self {
        self.responses.get_or_insert_with(Vec::new).push(response);
        self
    }

    /// Add pre-existing responses (one per query) to evaluate without
    /// running the agent.
    pub fn responses(mut self, responses: impl IntoIterator<Item = AgentResponse>) -> Self {
        self.responses
            .get_or_insert_with(Vec::new)
            .extend(responses);
        self
    }

    /// Grounding context attached to every item (for groundedness
    /// evaluators).
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Run the agent (unless responses were given) and evaluate. Returns one
    /// [`EvalResults`] per provider.
    ///
    /// # Errors
    ///
    /// [`Error::Configuration`] when `num_repetitions` is 0; when expected
    /// outputs / expected tool calls / responses do not match the query
    /// count; when responses are given without queries; when queries are
    /// given without an agent or responses; or when neither is given. Agent
    /// and evaluator errors propagate.
    pub async fn run(self) -> Result<Vec<EvalResults>> {
        check_repetitions(self.num_repetitions)?;
        let queries = self.queries;
        if let (Some(expected), Some(queries)) = (&self.expected_output, &queries) {
            if expected.len() != queries.len() {
                return Err(Error::Configuration(format!(
                    "Got {} queries but {} expected_output values.",
                    queries.len(),
                    expected.len()
                )));
            }
        }
        if let (Some(calls), Some(queries)) = (&self.expected_tool_calls, &queries) {
            if calls.len() != queries.len() {
                return Err(Error::Configuration(format!(
                    "Got {} queries but {} expected_tool_calls lists.",
                    queries.len(),
                    calls.len()
                )));
            }
        }

        let mut items: Vec<EvalItem> = Vec::new();
        match (self.responses, queries, self.agent) {
            (Some(responses), Some(queries), agent) => {
                if queries.len() != responses.len() {
                    return Err(Error::Configuration(format!(
                        "Got {} queries but {} responses.",
                        queries.len(),
                        responses.len()
                    )));
                }
                for (query, response) in queries.into_iter().zip(&responses) {
                    items.push(to_eval_item(
                        EvalQuery::Text(query),
                        response,
                        agent,
                        None,
                        self.context.clone(),
                    ));
                }
            }
            (Some(_), None, _) => {
                return Err(Error::Configuration(
                    "Provide 'queries' alongside 'responses' so the conversation can be \
                     constructed for evaluation. For Responses API evaluation by response ID, \
                     use evaluate_traces(response_ids=...) from agent-framework-foundry."
                        .into(),
                ));
            }
            (None, Some(queries), Some(agent)) => {
                for _ in 0..self.num_repetitions {
                    for query in &queries {
                        let response = agent.run(vec![Message::user(query.clone())], None).await?;
                        items.push(to_eval_item(
                            EvalQuery::Text(query.clone()),
                            &response,
                            Some(agent),
                            None,
                            self.context.clone(),
                        ));
                    }
                }
            }
            (None, Some(_), None) => {
                return Err(Error::Configuration(
                    "Provide 'agent' when using 'queries' to run the agent. To evaluate \
                     pre-existing responses without an agent, use 'responses' instead."
                        .into(),
                ));
            }
            (None, None, _) => {
                return Err(Error::Configuration(
                    "Provide either 'queries' (with 'agent') or 'responses' (or both).".into(),
                ));
            }
        }

        if let Some(expected) = &self.expected_output {
            if !expected.is_empty() {
                for (i, item) in items.iter_mut().enumerate() {
                    item.expected_output = Some(expected[i % expected.len()].clone());
                }
            }
        }
        if let Some(calls) = &self.expected_tool_calls {
            if !calls.is_empty() {
                for (i, item) in items.iter_mut().enumerate() {
                    item.expected_tool_calls = Some(calls[i % calls.len()].clone());
                }
            }
        }
        if let Some(split) = &self.conversation_split {
            for item in &mut items {
                item.split_strategy = Some(split.clone());
            }
        }

        let name = self.eval_name.unwrap_or_else(|| match self.agent {
            Some(agent) => format!(
                "Eval: {}",
                agent
                    .name()
                    .filter(|n| !n.is_empty())
                    .unwrap_or_else(|| agent.id())
            ),
            None => "Eval: agent".to_string(),
        });
        let evaluators = resolve_evaluators(&self.evaluators);
        run_evaluators(&evaluators, &items, &name).await
    }
}

/// Run an agent against test queries and evaluate the results. Mirrors
/// upstream's `evaluate_agent`; see [`EvaluateAgent`] for the options.
pub async fn evaluate_agent(request: EvaluateAgent<'_>) -> Result<Vec<EvalResults>> {
    request.run().await
}

// ---------------------------------------------------------------------------
// Workflow extraction helpers
// ---------------------------------------------------------------------------

/// Per-agent query/response data pulled out of a workflow run. Mirrors
/// upstream's `_AgentEvalData` (minus the agent reference — see
/// [`EvaluateWorkflow`]).
struct AgentEvalData {
    key: String,
    query: EvalQuery,
    response: AgentResponse,
}

/// Upstream's filter for internal framework executors.
fn is_internal_executor(id: &str) -> bool {
    id.starts_with('_')
        || matches!(
            id.to_lowercase().as_str(),
            "input-conversation" | "end-conversation" | "end"
        )
}

/// Normalize a workflow input into a conversation, as the built-in
/// orchestration executors do: a string is one user message, an array is a
/// message list, an object is one message.
fn input_conversation(input: &Value) -> Option<Vec<Message>> {
    match input {
        Value::String(s) => Some(vec![Message::user(s.clone())]),
        Value::Array(_) => serde_json::from_value(input.clone()).ok(),
        Value::Object(_) => serde_json::from_value::<Message>(input.clone())
            .ok()
            .map(|m| vec![m]),
        _ => None,
    }
}

/// The per-agent query: the user messages of the input conversation, else the
/// whole input conversation, else the input rendered as text. Mirrors
/// upstream's `user_msgs or full_conversation` / `str(input_data)`.
fn agent_query(input: &Value) -> EvalQuery {
    match input_conversation(input) {
        Some(conversation) => {
            let users: Vec<Message> = conversation
                .iter()
                .filter(|m| m.role.as_str() == "user")
                .cloned()
                .collect();
            EvalQuery::Messages(if users.is_empty() {
                conversation
            } else {
                users
            })
        }
        None => EvalQuery::Text(value_text(input)),
    }
}

fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Walk a run's events and extract per-agent query/response pairs. Mirrors
/// upstream's `_extract_agent_eval_data`, reading the
/// [`WorkflowEvent::AgentRun`] events every built-in agent-running executor
/// emits.
fn extract_agent_eval_data(run: &WorkflowRun, input: &Value) -> Vec<AgentEvalData> {
    let mut results = Vec::new();
    for event in run.events() {
        let WorkflowEvent::AgentRun {
            executor_id,
            response,
        } = event
        else {
            continue;
        };
        if is_internal_executor(executor_id) {
            tracing::debug!(
                executor_id,
                "skipping internal executor during eval data extraction"
            );
            continue;
        }
        let Ok(response) = serde_json::from_value::<AgentResponse>(response.clone()) else {
            continue;
        };
        let key = response
            .messages
            .iter()
            .find_map(|m| m.author_name.clone())
            .filter(|a| !a.is_empty())
            .unwrap_or_else(|| executor_id.clone());
        if is_internal_executor(&key) {
            continue;
        }
        results.push(AgentEvalData {
            key,
            query: agent_query(input),
            response,
        });
    }
    results
}

/// The original user query of a run. Mirrors upstream's
/// `_extract_overall_query` over the first executor's input: a string is
/// itself, a message list is its user messages' text, a string list is
/// space-joined, anything else is rendered as text.
fn extract_overall_query(input: &Value) -> Option<String> {
    match input {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Array(items) if !items.is_empty() => {
            if let Ok(messages) = serde_json::from_value::<Vec<Message>>(input.clone()) {
                Some(
                    messages
                        .iter()
                        .filter(|m| m.role.as_str() == "user")
                        .map(Message::text)
                        .collect::<Vec<_>>()
                        .join(" "),
                )
            } else if items.iter().all(Value::is_string) {
                Some(items.iter().map(value_text).collect::<Vec<_>>().join(" "))
            } else {
                Some(input.to_string())
            }
        }
        Value::Object(_) => match serde_json::from_value::<Message>(input.clone()) {
            Ok(m) if m.role.as_str() == "user" => Some(m.text()),
            Ok(_) => Some(String::new()),
            Err(_) => Some(input.to_string()),
        },
        other => Some(other.to_string()),
    }
}

/// Build the item for the workflow's overall output. Mirrors upstream's
/// `_build_overall_item`: the last output, read as a message list (its
/// assistant text joined into one reply), an agent response, or any other
/// value rendered as text. `None` when the run produced no output.
fn build_overall_item(query: String, run: &WorkflowRun) -> Option<EvalItem> {
    let final_output = run.outputs().pop()?;
    let messages = match &final_output {
        Value::Array(items) if !items.is_empty() => {
            serde_json::from_value::<Vec<Message>>(final_output.clone()).ok()
        }
        _ => None,
    };
    let response = if let Some(messages) = messages {
        let text = messages
            .iter()
            .filter(|m| m.role.as_str() == "assistant")
            .map(Message::text)
            .collect::<Vec<_>>()
            .join(" ");
        AgentResponse {
            messages: vec![Message::assistant(text)],
            ..AgentResponse::default()
        }
    } else if let Some(response) = final_output
        .as_object()
        .filter(|o| o.contains_key("messages"))
        .and_then(|_| serde_json::from_value::<AgentResponse>(final_output.clone()).ok())
    {
        response
    } else {
        AgentResponse {
            messages: vec![Message::assistant(value_text(&final_output))],
            ..AgentResponse::default()
        }
    };
    Some(to_eval_item(
        EvalQuery::Text(query),
        &response,
        None,
        None,
        None,
    ))
}

// ---------------------------------------------------------------------------
// evaluate_workflow
// ---------------------------------------------------------------------------

/// A request to evaluate a multi-agent workflow with a per-agent breakdown.
/// The builder form of upstream's `evaluate_workflow(...)` keyword
/// arguments.
///
/// Two modes:
///
/// - **Run + evaluate** — set [`queries`](Self::queries): the workflow runs
///   once per query (times [`num_repetitions`](Self::num_repetitions)).
///   Queries take precedence over a supplied result, as upstream.
/// - **Post-hoc** — pass a finished run with
///   [`workflow_result`](Self::workflow_result).
///
/// Each provider evaluates every agent's interactions separately (into
/// [`EvalResults::sub_results`]) and, when
/// [`include_overall`](Self::include_overall) (the default), the workflow's
/// final output; without an overall evaluation the returned result
/// aggregates the sub-results' counts.
///
/// ## Divergences from upstream
///
/// - **Per-agent data comes from [`WorkflowEvent::AgentRun`] events.**
///   Upstream pairs `executor_invoked` / `executor_completed` events and
///   reads each `AgentExecutorResponse`. This engine's executor events carry
///   no payloads; every built-in agent-running executor instead emits an
///   `AgentRun` event with the agent's response.
/// - **Agents are keyed by participant name.** Upstream keys by executor id,
///   because each participant is its own executor there. The built-in Rust
///   orchestrators (group chat, handoff, Magentic) run every participant
///   inside one executor, so the key is the response's author name (the
///   agent's name), falling back to the executor id.
/// - **Each agent's query is the run input's user messages.** Upstream takes
///   the user messages of the agent's full input conversation, which the
///   Rust `AgentRun` event does not carry. In every built-in orchestration
///   those are the run input's user messages; they differ only when user
///   input is added mid-run (handoff's interactive mode).
/// - **Post-hoc mode needs the run input**, for the same reason: pass the
///   value given to [`Workflow::run`] alongside the [`WorkflowRun`].
/// - **No tool definitions on per-agent items.** Upstream looks each executor
///   up in `workflow.executors` to read its agent's tools; Rust executors are
///   opaque trait objects.
/// - The default eval name uses the workflow's [`name`](Workflow::name)
///   (else `"Workflow"`), where upstream prints the Python class name.
pub struct EvaluateWorkflow<'a> {
    workflow: &'a Workflow,
    workflow_result: Option<(&'a WorkflowRun, Value)>,
    queries: Option<Vec<String>>,
    expected_output: Option<Vec<String>>,
    evaluators: Vec<EvaluatorEntry>,
    eval_name: Option<String>,
    include_overall: bool,
    include_per_agent: bool,
    conversation_split: Option<Arc<dyn ConversationSplitter>>,
    num_repetitions: usize,
}

impl<'a> EvaluateWorkflow<'a> {
    /// A request over `workflow` (overall and per-agent evaluation on, one
    /// repetition, no evaluators).
    pub fn new(workflow: &'a Workflow) -> Self {
        Self {
            workflow,
            workflow_result: None,
            queries: None,
            expected_output: None,
            evaluators: Vec::new(),
            eval_name: None,
            include_overall: true,
            include_per_agent: true,
            conversation_split: None,
            num_repetitions: 1,
        }
    }

    evaluator_setters!();

    /// Evaluate a finished run (post-hoc mode). `input` is the value the run
    /// was started with.
    pub fn workflow_result(mut self, run: &'a WorkflowRun, input: impl Into<Value>) -> Self {
        self.workflow_result = Some((run, input.into()));
        self
    }

    /// Whether to evaluate the workflow's final output (default `true`).
    pub fn include_overall(mut self, include: bool) -> Self {
        self.include_overall = include;
        self
    }

    /// Whether to evaluate each agent individually (default `true`).
    pub fn include_per_agent(mut self, include: bool) -> Self {
        self.include_per_agent = include;
        self
    }

    /// Run (unless a result was given) and evaluate. Returns one
    /// [`EvalResults`] per provider, each with per-agent `sub_results`.
    ///
    /// # Errors
    ///
    /// [`Error::Configuration`] when neither queries nor a result is given;
    /// when expected outputs are given without queries or with a different
    /// count; when `num_repetitions` is 0; or when the run yielded no agent
    /// data and no overall output to evaluate. Workflow and evaluator errors
    /// propagate.
    pub async fn run(self) -> Result<Vec<EvalResults>> {
        if self.workflow_result.is_none() && self.queries.is_none() {
            return Err(Error::Configuration(
                "Provide either 'workflow_result' or 'queries'.".into(),
            ));
        }
        if let Some(expected) = &self.expected_output {
            let Some(queries) = &self.queries else {
                return Err(Error::Configuration(
                    "Provide 'queries' when using 'expected_output'; 'expected_output' is not \
                     supported with 'workflow_result' only."
                        .into(),
                ));
            };
            if expected.len() != queries.len() {
                return Err(Error::Configuration(format!(
                    "Got {} queries but {} expected_output values.",
                    queries.len(),
                    expected.len()
                )));
            }
        }
        check_repetitions(self.num_repetitions)?;

        let wf_name = self.eval_name.clone().unwrap_or_else(|| {
            format!(
                "Workflow Eval: {}",
                self.workflow.name().unwrap_or("Workflow")
            )
        });
        let evaluators = resolve_evaluators(&self.evaluators);

        let mut all_agent_data: Vec<AgentEvalData> = Vec::new();
        let mut overall_items: Vec<EvalItem> = Vec::new();

        if let Some(queries) = &self.queries {
            for _ in 0..self.num_repetitions {
                for (qi, query) in queries.iter().enumerate() {
                    let input = Value::String(query.clone());
                    let run = self.workflow.run(input.clone()).await?;
                    all_agent_data.extend(extract_agent_eval_data(&run, &input));
                    if self.include_overall {
                        if let Some(mut item) = build_overall_item(query.clone(), &run) {
                            if let Some(expected) = &self.expected_output {
                                item.expected_output = Some(expected[qi].clone());
                            }
                            overall_items.push(item);
                        }
                    }
                }
            }
        } else if let Some((run, input)) = &self.workflow_result {
            all_agent_data = extract_agent_eval_data(run, input);
            if self.include_overall {
                if let Some(query) = extract_overall_query(input).filter(|q| !q.is_empty()) {
                    if let Some(item) = build_overall_item(query, run) {
                        overall_items.push(item);
                    }
                }
            }
        }

        // Group per-agent items by key, in first-appearance order (the map
        // that holds the sub-results sorts them by name).
        let mut agent_items: Vec<(String, Vec<EvalItem>)> = Vec::new();
        if self.include_per_agent {
            for data in all_agent_data {
                let item = to_eval_item(data.query, &data.response, None, None, None);
                match agent_items.iter_mut().find(|(k, _)| *k == data.key) {
                    Some((_, items)) => items.push(item),
                    None => agent_items.push((data.key, vec![item])),
                }
            }
        }

        if agent_items.is_empty() && overall_items.is_empty() {
            return Err(no_agent_data());
        }

        if let Some(split) = &self.conversation_split {
            for item in agent_items
                .iter_mut()
                .flat_map(|(_, items)| items.iter_mut())
                .chain(overall_items.iter_mut())
            {
                item.split_strategy = Some(split.clone());
            }
        }

        let count = evaluators.len();
        let mut all_results = Vec::with_capacity(count);
        for ev in &evaluators {
            let sfx = suffix(ev.as_ref(), count);
            let mut sub_results: BTreeMap<String, EvalResults> = BTreeMap::new();
            for (key, items) in &agent_items {
                let result = ev
                    .evaluate(items, &format!("{wf_name} — {key}{sfx}"))
                    .await?;
                sub_results.insert(key.clone(), result);
            }

            let mut overall = if self.include_overall && !overall_items.is_empty() {
                ev.evaluate(&overall_items, &format!("{wf_name} — overall{sfx}"))
                    .await?
            } else if !sub_results.is_empty() {
                let passed = sub_results.values().map(EvalResults::passed).sum();
                let failed = sub_results.values().map(EvalResults::failed).sum();
                let all_completed = sub_results
                    .values()
                    .all(|s| s.status == EvalRunStatus::Completed);
                EvalResults::new(ev.name())
                    .with_eval_id("aggregate")
                    .with_run_id("aggregate")
                    .with_status(if all_completed {
                        EvalRunStatus::Completed
                    } else {
                        EvalRunStatus::Partial
                    })
                    .with_result_counts(ResultCounts::new(passed, failed, 0))
            } else {
                return Err(no_agent_data());
            };
            overall.sub_results = sub_results;
            all_results.push(overall);
        }
        Ok(all_results)
    }
}

fn no_agent_data() -> Error {
    Error::Configuration(
        "No agent executor data found in the workflow result. Ensure the workflow uses \
         AgentExecutor-based agents."
            .into(),
    )
}

/// Evaluate a multi-agent workflow with a per-agent breakdown. Mirrors
/// upstream's `evaluate_workflow`; see [`EvaluateWorkflow`] for the options
/// and divergences.
pub async fn evaluate_workflow(request: EvaluateWorkflow<'_>) -> Result<Vec<EvalResults>> {
    request.run().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn internal_executors_are_recognized() {
        assert!(is_internal_executor("_internal"));
        assert!(is_internal_executor("End"));
        assert!(is_internal_executor("input-conversation"));
        assert!(is_internal_executor("end-conversation"));
        assert!(!is_internal_executor("writer"));
    }

    #[test]
    fn overall_query_extraction_mirrors_upstream() {
        assert_eq!(extract_overall_query(&json!("hi")), Some("hi".into()));
        assert_eq!(
            extract_overall_query(&json!(["a", "b"])),
            Some("a b".into())
        );
        let msgs = serde_json::to_value(vec![
            Message::system("sys"),
            Message::user("one"),
            Message::user("two"),
        ])
        .unwrap();
        assert_eq!(extract_overall_query(&msgs), Some("one two".into()));
        assert_eq!(extract_overall_query(&json!(42)), Some("42".into()));
        assert_eq!(extract_overall_query(&Value::Null), None);
    }

    #[test]
    fn agent_query_prefers_user_messages() {
        let input = serde_json::to_value(vec![Message::system("s"), Message::user("u")]).unwrap();
        match agent_query(&input) {
            EvalQuery::Messages(m) => {
                assert_eq!(m.len(), 1);
                assert_eq!(m[0].text(), "u");
            }
            other => panic!("unexpected {other:?}"),
        }
        let only_system = serde_json::to_value(vec![Message::system("s")]).unwrap();
        assert!(matches!(agent_query(&only_system), EvalQuery::Messages(m) if m.len() == 1));
        assert_eq!(agent_query(&json!(7)), EvalQuery::Text("7".into()));
    }
}
