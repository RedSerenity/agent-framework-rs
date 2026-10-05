//! Provider-agnostic evaluation of agents and workflows.
//!
//! Rust port of upstream `agent_framework._evaluation`. It defines the core
//! evaluation data model and the orchestration that runs an agent (or a
//! workflow) over test queries and hands the resulting interactions to one or
//! more evaluation providers:
//!
//! - [`EvalItem`] — one query/response interaction, derived from a full
//!   conversation through a [`ConversationSplitter`] (the built-in
//!   [`ConversationSplit::LastTurn`] / [`ConversationSplit::Full`], or any
//!   closure).
//! - [`Evaluator`] — the provider trait (Microsoft Foundry's `FoundryEvals`
//!   in `agent-framework-foundry`, the in-process [`LocalEvaluator`], or your
//!   own scorer).
//! - [`EvalResults`] / [`EvalItemResult`] / [`EvalScoreResult`] /
//!   [`RubricScore`] — provider-neutral results with CI-gate assertions
//!   ([`EvalResults::raise_for_status`],
//!   [`EvalResults::assert_score_at_least`], …) that fail with
//!   [`Error::EvalNotPassed`].
//! - [`LocalEvaluator`] and the built-in checks ([`keyword_check`],
//!   [`tool_called_check`], [`tool_calls_present`], [`tool_call_args_match`])
//!   for fast, API-free evaluation, plus [`evaluator`] / [`async_evaluator`]
//!   to turn a plain function into a check.
//! - [`EvaluateAgent`] / [`evaluate_agent`] and [`EvaluateWorkflow`] /
//!   [`evaluate_workflow`] — the orchestration entry points.
//!
//! ```no_run
//! use agent_framework_core::evaluation::{
//!     keyword_check, tool_called_check, EvaluateAgent, LocalEvaluator,
//! };
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo(agent: Agent) -> Result<()> {
//! let local = LocalEvaluator::new()
//!     .with_check(keyword_check(["weather", "temperature"]))
//!     .with_check(tool_called_check(["get_weather"]));
//! let results = EvaluateAgent::new()
//!     .agent(&agent)
//!     .queries(["What's the weather in Seattle?"])
//!     .evaluator(local)
//!     .run()
//!     .await?;
//! results[0].raise_for_status(None)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Deliberate divergences from upstream
//!
//! - Upstream's keyword-argument entry points (`evaluate_agent(agent=…,
//!   queries=…, …)`) are builders here ([`EvaluateAgent`],
//!   [`EvaluateWorkflow`]); the free functions [`evaluate_agent`] /
//!   [`evaluate_workflow`] take a built request. Singular-or-list arguments
//!   are expressed as a singular setter (`query`) plus a plural one
//!   (`queries`).
//! - Upstream accepts a mixed list of `Evaluator`s and bare check callables
//!   and groups each run of consecutive callables into one `LocalEvaluator`.
//!   The builders keep that exact grouping, through separate
//!   [`EvaluateAgent::evaluator`] and [`EvaluateAgent::check`] calls.
//! - The `@evaluator` decorator introspects a Python function's parameter
//!   names to decide which item fields to pass. Rust cannot, so
//!   [`evaluator`] hands the function an [`EvalFields`] view with every
//!   supported field and the function reads what it needs; the name is
//!   explicit rather than taken from `__name__`.
//! - String statuses become enums ([`EvalItemStatus`], [`EvalRunStatus`]) and
//!   the `dict[str, int]` counts become [`ResultCounts`]. Maps keyed by name
//!   (`per_evaluator`, `sub_results`) are [`BTreeMap`]s, so they iterate in
//!   name order rather than insertion order.
//! - Errors upstream raises (`ValueError`, `TypeError`, `EvalNotPassedError`)
//!   are returned as [`Error`] values:
//!   `ValueError` → [`Error::Configuration`],
//!   a malformed function-evaluator return value →
//!   [`Error::Other`], and `EvalNotPassedError` →
//!   [`Error::EvalNotPassed`].
//! - Python's `@experimental(EVALS)` marker has no Rust equivalent; the
//!   surface is simply documented as mirroring an experimental upstream
//!   feature.

mod checks;
mod legacy;
mod orchestrate;
mod pyrepr;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::tools::{ToolDefinition, ToolKind};
use crate::types::{AgentResponse, Message};

pub use checks::{
    async_evaluator, evaluator, keyword_check, tool_call_args_match, tool_called_check,
    tool_calls_present, CheckResult, EvalCheck, EvalFields, EvalOutcome, FunctionEvaluator,
    IntoEvalOutcome, KeywordCheck, LocalEvaluator, ToolCalledCheck, ToolCalledMode,
};
#[allow(deprecated)]
pub use legacy::AgentEvalConverter;
pub use orchestrate::{evaluate_agent, evaluate_workflow, EvaluateAgent, EvaluateWorkflow};

// ---------------------------------------------------------------------------
// Conversation splitting
// ---------------------------------------------------------------------------

/// A strategy for splitting a conversation into `(query, response)` messages.
///
/// Mirrors upstream's `ConversationSplitter` protocol. Implemented by the
/// built-in [`ConversationSplit`] strategies and by any
/// `Fn(&[Message]) -> (Vec<Message>, Vec<Message>)` closure, so a custom
/// split — e.g. just before a memory-retrieval tool call, to evaluate recall
/// quality — is a plain closure:
///
/// ```
/// use agent_framework_core::evaluation::{split_last_turn, EvalItem};
/// use agent_framework_core::types::{Content, Message};
///
/// let split_before_memory = |conversation: &[Message]| {
///     for (i, msg) in conversation.iter().enumerate() {
///         if msg.contents.iter().any(|c| {
///             matches!(c, Content::FunctionCall(fc) if fc.name == "retrieve_memory")
///         }) {
///             return (conversation[..i].to_vec(), conversation[i..].to_vec());
///         }
///     }
///     split_last_turn(conversation)
/// };
/// let item = EvalItem::new(vec![Message::user("Hi"), Message::assistant("Hello!")]);
/// let (query, response) = item.split_messages_with(&split_before_memory);
/// assert_eq!((query.len(), response.len()), (1, 1));
/// ```
pub trait ConversationSplitter: Send + Sync {
    /// Split `conversation` into `(query_messages, response_messages)`.
    fn split(&self, conversation: &[Message]) -> (Vec<Message>, Vec<Message>);
}

impl<F> ConversationSplitter for F
where
    F: Fn(&[Message]) -> (Vec<Message>, Vec<Message>) + Send + Sync,
{
    fn split(&self, conversation: &[Message]) -> (Vec<Message>, Vec<Message>) {
        self(conversation)
    }
}

/// Built-in conversation split strategies. Mirrors upstream's
/// `ConversationSplit` enum (whose members are themselves callable).
///
/// - [`LastTurn`](Self::LastTurn) (the default): split at the last user
///   message. Everything up to and including it is the query; everything
///   after is the response. Evaluates whether the agent answered the *latest*
///   question well.
/// - [`Full`](Self::Full): the first user message (and anything before it,
///   such as system messages) is the query; the entire remainder is the
///   response. Evaluates whether the *whole trajectory* served the original
///   request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationSplit {
    /// Split at the last user message (`"last_turn"`).
    #[default]
    LastTurn,
    /// Split after the first user message (`"full"`).
    Full,
}

impl ConversationSplit {
    /// The upstream string value (`"last_turn"` / `"full"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            ConversationSplit::LastTurn => "last_turn",
            ConversationSplit::Full => "full",
        }
    }
}

impl ConversationSplitter for ConversationSplit {
    fn split(&self, conversation: &[Message]) -> (Vec<Message>, Vec<Message>) {
        match self {
            ConversationSplit::LastTurn => split_last_turn(conversation),
            ConversationSplit::Full => split_full(conversation),
        }
    }
}

fn is_role(message: &Message, role: &str) -> bool {
    message.role.as_str() == role
}

/// Split at the last user message — the default strategy, usable as a
/// fallback inside custom splitters. Mirrors upstream's `_split_last_turn`
/// (exposed there as `EvalItem._split_last_turn_static`).
///
/// With no user message at all, the query is empty and the whole
/// conversation is the response.
pub fn split_last_turn(conversation: &[Message]) -> (Vec<Message>, Vec<Message>) {
    match conversation.iter().rposition(|m| is_role(m, "user")) {
        Some(i) => (conversation[..=i].to_vec(), conversation[i + 1..].to_vec()),
        None => (Vec::new(), conversation.to_vec()),
    }
}

/// Split after the first user message (evaluates the whole trajectory).
/// Mirrors upstream's `_split_full`.
pub fn split_full(conversation: &[Message]) -> (Vec<Message>, Vec<Message>) {
    match conversation.iter().position(|m| is_role(m, "user")) {
        Some(i) => (conversation[..=i].to_vec(), conversation[i + 1..].to_vec()),
        None => (Vec::new(), conversation.to_vec()),
    }
}

// ---------------------------------------------------------------------------
// Expected tool calls and eval items
// ---------------------------------------------------------------------------

/// A tool call an agent is expected to make. Mirrors upstream's
/// `ExpectedToolCall` dataclass.
///
/// Pure data: the evaluator decides the matching semantics (order, extras,
/// argument checking). `arguments: None` means "don't check arguments".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedToolCall {
    /// The tool/function name (e.g. `"get_weather"`).
    pub name: String,
    /// Expected arguments; `None` means "don't check arguments" (or "no
    /// arguments").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Map<String, Value>>,
}

impl ExpectedToolCall {
    /// An expected call to `name`, with arguments unchecked.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: None,
        }
    }

    /// Set the expected arguments (a subset match: every pair given here must
    /// appear in the actual call; extra actual arguments are fine).
    ///
    /// ```
    /// use agent_framework_core::evaluation::ExpectedToolCall;
    /// use serde_json::json;
    ///
    /// let call = ExpectedToolCall::new("get_weather").with_arguments([("location", json!("NYC"))]);
    /// assert_eq!(call.arguments.unwrap()["location"], "NYC");
    /// ```
    pub fn with_arguments<I, K>(mut self, arguments: I) -> Self
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        self.arguments = Some(arguments.into_iter().map(|(k, v)| (k.into(), v)).collect());
        self
    }
}

/// The query side of an agent interaction: a bare string (becomes one user
/// message) or explicit input messages. Mirrors upstream's
/// `query: str | Sequence[Message]` parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalQuery {
    /// A single user query.
    Text(String),
    /// The full input messages.
    Messages(Vec<Message>),
}

impl EvalQuery {
    fn into_messages(self) -> Vec<Message> {
        match self {
            EvalQuery::Text(text) => vec![Message::user(text)],
            EvalQuery::Messages(messages) => messages,
        }
    }
}

impl From<&str> for EvalQuery {
    fn from(value: &str) -> Self {
        EvalQuery::Text(value.to_string())
    }
}

impl From<String> for EvalQuery {
    fn from(value: String) -> Self {
        EvalQuery::Text(value)
    }
}

impl From<&String> for EvalQuery {
    fn from(value: &String) -> Self {
        EvalQuery::Text(value.clone())
    }
}

impl From<Vec<Message>> for EvalQuery {
    fn from(value: Vec<Message>) -> Self {
        EvalQuery::Messages(value)
    }
}

/// A single item to be evaluated. Mirrors upstream's `EvalItem`.
///
/// [`conversation`](Self::conversation) is the single source of truth:
/// [`query`](Self::query) and [`response`](Self::response) are derived from
/// it through the split strategy (the item's own
/// [`split_strategy`](Self::split_strategy), else
/// [`ConversationSplit::LastTurn`]).
#[derive(Clone, Default)]
pub struct EvalItem {
    /// The full conversation.
    pub conversation: Vec<Message>,
    /// Function-tool definitions available to the agent, for evaluator logic
    /// (upstream: typed `FunctionTool` objects). Hosted tools are never
    /// included.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Optional grounding context document.
    pub context: Option<String>,
    /// Optional expected output for ground-truth comparison.
    pub expected_output: Option<String>,
    /// Expected tool calls, for tool-correctness evaluation.
    pub expected_tool_calls: Option<Vec<ExpectedToolCall>>,
    /// How [`query`](Self::query) and [`response`](Self::response) are
    /// derived. `None` means [`ConversationSplit::LastTurn`].
    pub split_strategy: Option<Arc<dyn ConversationSplitter>>,
}

impl fmt::Debug for EvalItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalItem")
            .field("conversation", &self.conversation)
            .field("tools", &self.tools)
            .field("context", &self.context)
            .field("expected_output", &self.expected_output)
            .field("expected_tool_calls", &self.expected_tool_calls)
            .field("split_strategy", &self.split_strategy.is_some())
            .finish()
    }
}

impl EvalItem {
    /// An item over `conversation`, with nothing else set.
    pub fn new(conversation: Vec<Message>) -> Self {
        Self {
            conversation,
            ..Self::default()
        }
    }

    /// Build an item from one agent interaction: the query (a string becomes
    /// a user message) followed by the response's messages.
    ///
    /// Mirrors upstream's provider-neutral `_to_eval_item` without the
    /// `agent` / `tools` arguments — chain [`with_tools`](Self::with_tools)
    /// or [`with_agent_tools`](Self::with_agent_tools) for those.
    pub fn from_agent_response(query: impl Into<EvalQuery>, response: &AgentResponse) -> Self {
        to_eval_item(query.into(), response, None, None, None)
    }

    /// Set the tool definitions. Only function tools are kept; an empty list
    /// leaves [`tools`](Self::tools) unset.
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        let tools = function_tools(tools);
        self.tools = (!tools.is_empty()).then_some(tools);
        self
    }

    /// Set the tool definitions from an agent's
    /// [`default_tools`](SupportsAgentRun::default_tools).
    pub fn with_agent_tools(self, agent: &dyn SupportsAgentRun) -> Self {
        self.with_tools(agent.default_tools())
    }

    /// Set the grounding context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set the expected output.
    pub fn with_expected_output(mut self, expected_output: impl Into<String>) -> Self {
        self.expected_output = Some(expected_output.into());
        self
    }

    /// Set the expected tool calls.
    pub fn with_expected_tool_calls(
        mut self,
        calls: impl IntoIterator<Item = ExpectedToolCall>,
    ) -> Self {
        self.expected_tool_calls = Some(calls.into_iter().collect());
        self
    }

    /// Set the split strategy used to derive the query and response.
    pub fn with_split_strategy(mut self, split: impl ConversationSplitter + 'static) -> Self {
        self.split_strategy = Some(Arc::new(split));
        self
    }

    fn effective_split(&self) -> &dyn ConversationSplitter {
        match &self.split_strategy {
            Some(split) => split.as_ref(),
            None => &ConversationSplit::LastTurn,
        }
    }

    /// The user query text: the space-joined text of the user messages on the
    /// query side of the split, trimmed.
    pub fn query(&self) -> String {
        let (query, _) = self.effective_split().split(&self.conversation);
        joined_text(&query, "user")
    }

    /// The agent response text: the space-joined text of the assistant
    /// messages on the response side of the split, trimmed.
    pub fn response(&self) -> String {
        let (_, response) = self.effective_split().split(&self.conversation);
        joined_text(&response, "assistant")
    }

    /// Split the conversation with the item's own strategy (else
    /// [`ConversationSplit::LastTurn`]).
    pub fn split_messages(&self) -> (Vec<Message>, Vec<Message>) {
        self.effective_split().split(&self.conversation)
    }

    /// Split the conversation with an explicit strategy, overriding the
    /// item's own. Upstream: `split_messages(split=…)`.
    pub fn split_messages_with(
        &self,
        split: &dyn ConversationSplitter,
    ) -> (Vec<Message>, Vec<Message>) {
        split.split(&self.conversation)
    }

    /// Split a multi-turn conversation into one item per user turn.
    ///
    /// Each user message starts a new turn. Item *n* holds the conversation
    /// up to (not including) the user message that starts turn *n + 1*, so
    /// with the default last-turn split each response is evaluated with its
    /// full preceding context. `tools` and `context` are shared across all
    /// items. A conversation without user messages yields no items.
    pub fn per_turn_items(
        conversation: &[Message],
        tools: Option<Vec<ToolDefinition>>,
        context: Option<String>,
    ) -> Vec<EvalItem> {
        let user_indices: Vec<usize> = conversation
            .iter()
            .enumerate()
            .filter(|(_, m)| is_role(m, "user"))
            .map(|(i, _)| i)
            .collect();
        (0..user_indices.len())
            .map(|turn| {
                let end = user_indices
                    .get(turn + 1)
                    .copied()
                    .unwrap_or(conversation.len());
                EvalItem {
                    conversation: conversation[..end].to_vec(),
                    tools: tools.clone(),
                    context: context.clone(),
                    ..EvalItem::default()
                }
            })
            .collect()
    }
}

fn joined_text(messages: &[Message], role: &str) -> String {
    messages
        .iter()
        .filter(|m| is_role(m, role))
        .map(Message::text)
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn function_tools(tools: impl IntoIterator<Item = ToolDefinition>) -> Vec<ToolDefinition> {
    tools
        .into_iter()
        .filter(|t| t.kind == ToolKind::Function)
        .collect()
}

/// Build a provider-neutral [`EvalItem`] from an agent interaction. Mirrors
/// upstream's `_to_eval_item`: explicit (non-empty) `tools` win over the
/// agent's; only function tools are kept; an empty result leaves `tools`
/// unset.
pub(crate) fn to_eval_item(
    query: EvalQuery,
    response: &AgentResponse,
    agent: Option<&dyn SupportsAgentRun>,
    tools: Option<Vec<ToolDefinition>>,
    context: Option<String>,
) -> EvalItem {
    let mut conversation = query.into_messages();
    conversation.extend(response.messages.iter().cloned());

    let typed_tools = match (tools, agent) {
        (Some(tools), _) if !tools.is_empty() => function_tools(tools),
        (_, Some(agent)) => function_tools(agent.default_tools()),
        _ => Vec::new(),
    };

    EvalItem {
        conversation,
        tools: (!typed_tools.is_empty()).then_some(typed_tools),
        context,
        ..EvalItem::default()
    }
}

// ---------------------------------------------------------------------------
// Scores and results
// ---------------------------------------------------------------------------

/// A single rubric dimension's score from a rubric-based evaluator. Mirrors
/// upstream's `RubricScore`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RubricScore {
    /// Dimension id (matches the rubric definition).
    pub id: String,
    /// The score, or `None` when the dimension was not applicable.
    pub score: Option<i64>,
    /// Whether the dimension applied to this item.
    pub applicable: bool,
    /// Dimension weight (mirrors the rubric definition).
    pub weight: i64,
    /// Short rationale produced by the evaluator.
    pub reason: String,
}

/// The result of a single evaluator on a single item. Mirrors upstream's
/// `EvalScoreResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalScoreResult {
    /// Evaluator name (e.g. `"relevance"`).
    pub name: String,
    /// Numeric score.
    pub score: f64,
    /// Whether the item passed this evaluator's threshold.
    #[serde(default)]
    pub passed: Option<bool>,
    /// Raw evaluator output (rationale, metadata).
    #[serde(default)]
    pub sample: Option<Value>,
    /// Per-dimension scores for rubric evaluators; `None` otherwise.
    #[serde(default)]
    pub dimensions: Option<Vec<RubricScore>>,
}

impl EvalScoreResult {
    /// A score with no pass/fail verdict, sample or dimensions.
    pub fn new(name: impl Into<String>, score: f64) -> Self {
        Self {
            name: name.into(),
            score,
            passed: None,
            sample: None,
            dimensions: None,
        }
    }
}

/// Per-item status. Upstream uses the strings `"pass"`, `"fail"` and
/// `"error"` (some providers say `"errored"`, which parses as
/// [`Error`](Self::Error)).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum EvalItemStatus {
    /// `"pass"`.
    Pass,
    /// `"fail"`.
    Fail,
    /// `"error"` / `"errored"` — an infrastructure failure, not a quality one.
    Error,
    /// Any other provider-specific status.
    Other(String),
}

impl From<String> for EvalItemStatus {
    fn from(value: String) -> Self {
        EvalItemStatus::from(value.as_str())
    }
}

impl From<&str> for EvalItemStatus {
    fn from(value: &str) -> Self {
        match value {
            "pass" => EvalItemStatus::Pass,
            "fail" => EvalItemStatus::Fail,
            "error" | "errored" => EvalItemStatus::Error,
            other => EvalItemStatus::Other(other.to_string()),
        }
    }
}

impl From<EvalItemStatus> for String {
    fn from(value: EvalItemStatus) -> Self {
        value.to_string()
    }
}

impl fmt::Display for EvalItemStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            EvalItemStatus::Pass => "pass",
            EvalItemStatus::Fail => "fail",
            EvalItemStatus::Error => "error",
            EvalItemStatus::Other(s) => s,
        })
    }
}

/// Per-item result from an evaluation run. Mirrors upstream's
/// `EvalItemResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalItemResult {
    /// Provider-assigned item identifier.
    pub item_id: String,
    /// Pass / fail / error.
    pub status: EvalItemStatus,
    /// Per-evaluator results for this item.
    #[serde(default)]
    pub scores: Vec<EvalScoreResult>,
    /// Error category when the item errored (e.g. `"QueryExtractionError"`).
    #[serde(default)]
    pub error_code: Option<String>,
    /// Human-readable error detail.
    #[serde(default)]
    pub error_message: Option<String>,
    /// Responses API response id, if applicable.
    #[serde(default)]
    pub response_id: Option<String>,
    /// The query/input that was evaluated.
    #[serde(default)]
    pub input_text: Option<String>,
    /// The response/output that was evaluated.
    #[serde(default)]
    pub output_text: Option<String>,
    /// Token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`, …).
    #[serde(default)]
    pub token_usage: Option<BTreeMap<String, i64>>,
    /// Additional provider-specific data.
    #[serde(default)]
    pub metadata: Option<Value>,
}

impl EvalItemResult {
    /// A result with the given id and status and nothing else.
    pub fn new(item_id: impl Into<String>, status: impl Into<EvalItemStatus>) -> Self {
        Self {
            item_id: item_id.into(),
            status: status.into(),
            scores: Vec::new(),
            error_code: None,
            error_message: None,
            response_id: None,
            input_text: None,
            output_text: None,
            token_usage: None,
            metadata: None,
        }
    }

    /// Whether this item errored (infrastructure failure, not quality).
    pub fn is_error(&self) -> bool {
        self.status == EvalItemStatus::Error
    }

    /// Whether this item passed all evaluators.
    pub fn is_passed(&self) -> bool {
        self.status == EvalItemStatus::Pass
    }

    /// Whether this item failed at least one evaluator.
    pub fn is_failed(&self) -> bool {
        self.status == EvalItemStatus::Fail
    }
}

/// Run status of an evaluation. Upstream uses the strings `"completed"`,
/// `"failed"`, `"canceled"`, `"timeout"` (polling exceeded the deadline) and
/// `"partial"` (an aggregate whose sub-runs did not all complete).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum EvalRunStatus {
    /// `"completed"`.
    Completed,
    /// `"failed"`.
    Failed,
    /// `"canceled"`.
    Canceled,
    /// `"timeout"`.
    Timeout,
    /// `"partial"`.
    Partial,
    /// Any other (e.g. in-progress) provider status.
    Other(String),
}

impl From<String> for EvalRunStatus {
    fn from(value: String) -> Self {
        EvalRunStatus::from(value.as_str())
    }
}

impl From<&str> for EvalRunStatus {
    fn from(value: &str) -> Self {
        match value {
            "completed" => EvalRunStatus::Completed,
            "failed" => EvalRunStatus::Failed,
            "canceled" => EvalRunStatus::Canceled,
            "timeout" => EvalRunStatus::Timeout,
            "partial" => EvalRunStatus::Partial,
            other => EvalRunStatus::Other(other.to_string()),
        }
    }
}

impl From<EvalRunStatus> for String {
    fn from(value: EvalRunStatus) -> Self {
        value.to_string()
    }
}

impl fmt::Display for EvalRunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            EvalRunStatus::Completed => "completed",
            EvalRunStatus::Failed => "failed",
            EvalRunStatus::Canceled => "canceled",
            EvalRunStatus::Timeout => "timeout",
            EvalRunStatus::Partial => "partial",
            EvalRunStatus::Other(s) => s,
        })
    }
}

/// Pass/fail/error counts. Upstream: the `result_counts` / `per_evaluator`
/// `dict[str, int]`s (keys `passed`, `failed`, `errored`, and Foundry's
/// `total`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultCounts {
    /// Passing results.
    #[serde(default)]
    pub passed: u64,
    /// Failing results.
    #[serde(default)]
    pub failed: u64,
    /// Errored results.
    #[serde(default)]
    pub errored: u64,
    /// A provider-reported total, when the provider supplies one. Not used by
    /// [`EvalResults::total`], which is `passed + failed` as upstream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

impl ResultCounts {
    /// Counts with the given passed/failed/errored values and no reported
    /// total.
    pub fn new(passed: u64, failed: u64, errored: u64) -> Self {
        Self {
            passed,
            failed,
            errored,
            total: None,
        }
    }
}

/// Results from an evaluation run by a single provider. Mirrors upstream's
/// `EvalResults`.
///
/// `result_counts: None` plays the role of upstream's empty/absent
/// `result_counts` dict (which is falsy) in [`all_passed`](Self::all_passed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalResults {
    /// Name of the provider that produced these results.
    pub provider: String,
    /// The evaluation definition id (provider-specific).
    #[serde(default)]
    pub eval_id: String,
    /// The evaluation run id (provider-specific).
    #[serde(default)]
    pub run_id: String,
    /// Run status.
    pub status: EvalRunStatus,
    /// Pass/fail counts, populated when completed.
    #[serde(default)]
    pub result_counts: Option<ResultCounts>,
    /// URL to view results in the provider's portal.
    #[serde(default)]
    pub report_url: Option<String>,
    /// Error details when the run failed.
    #[serde(default)]
    pub error: Option<String>,
    /// Per-evaluator counts, keyed by evaluator name.
    #[serde(default)]
    pub per_evaluator: BTreeMap<String, ResultCounts>,
    /// Per-item results, when the provider supports per-item retrieval.
    #[serde(default)]
    pub items: Vec<EvalItemResult>,
    /// Per-agent breakdown for workflow evaluations, keyed by agent/executor
    /// name.
    #[serde(default)]
    pub sub_results: BTreeMap<String, EvalResults>,
}

impl EvalResults {
    /// Empty, completed results from `provider` (upstream's constructor
    /// defaults: `eval_id = run_id = ""`, `status = "completed"`).
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            eval_id: String::new(),
            run_id: String::new(),
            status: EvalRunStatus::Completed,
            result_counts: None,
            report_url: None,
            error: None,
            per_evaluator: BTreeMap::new(),
            items: Vec::new(),
            sub_results: BTreeMap::new(),
        }
    }

    /// Set the evaluation definition id.
    pub fn with_eval_id(mut self, eval_id: impl Into<String>) -> Self {
        self.eval_id = eval_id.into();
        self
    }

    /// Set the run id.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = run_id.into();
        self
    }

    /// Set the run status.
    pub fn with_status(mut self, status: impl Into<EvalRunStatus>) -> Self {
        self.status = status.into();
        self
    }

    /// Set the result counts.
    pub fn with_result_counts(mut self, counts: ResultCounts) -> Self {
        self.result_counts = Some(counts);
        self
    }

    /// Set the report URL.
    pub fn with_report_url(mut self, url: impl Into<String>) -> Self {
        self.report_url = Some(url.into());
        self
    }

    /// Set the error detail.
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Set the per-item results.
    pub fn with_items(mut self, items: Vec<EvalItemResult>) -> Self {
        self.items = items;
        self
    }

    /// Add one per-agent sub-result.
    pub fn with_sub_result(mut self, name: impl Into<String>, results: EvalResults) -> Self {
        self.sub_results.insert(name.into(), results);
        self
    }

    fn counts(&self) -> ResultCounts {
        self.result_counts.unwrap_or_default()
    }

    /// Number of passing results.
    pub fn passed(&self) -> u64 {
        self.counts().passed
    }

    /// Number of failing results.
    pub fn failed(&self) -> u64 {
        self.counts().failed
    }

    /// Number of errored results.
    pub fn errored(&self) -> u64 {
        self.counts().errored
    }

    /// Total results (`passed + failed`, as upstream — errored results are
    /// not counted).
    pub fn total(&self) -> u64 {
        self.passed() + self.failed()
    }

    /// Whether every result passed with no failures or errors.
    ///
    /// `false` unless the run completed. For workflow evaluations with
    /// sub-results, every sub-result must also pass; a parent without its
    /// own counts then defers entirely to its sub-results.
    pub fn all_passed(&self) -> bool {
        if self.status != EvalRunStatus::Completed {
            return false;
        }
        let clean = self.failed() == 0 && self.errored() == 0 && self.total() > 0;
        if !self.sub_results.is_empty() {
            let own_passed = if self.result_counts.is_some() {
                clean
            } else {
                true
            };
            return own_passed && self.sub_results.values().all(EvalResults::all_passed);
        }
        clean
    }

    /// Fail with [`Error::EvalNotPassed`] unless [`all_passed`](Self::all_passed).
    ///
    /// The message names the run, counts, report URL, error, failed
    /// sub-results and errored items, unless `msg` overrides it. Similar to
    /// `requests.Response.raise_for_status()` — call it after evaluation in
    /// CI pipelines or test suites.
    pub fn raise_for_status(&self, msg: Option<&str>) -> Result<()> {
        if self.all_passed() {
            return Ok(());
        }
        let errored = self.errored();
        let mut detail = match msg {
            Some(m) => m.to_string(),
            None => format!(
                "Eval run {} {}: {} passed, {} failed.",
                self.run_id,
                self.status,
                self.passed(),
                self.failed()
            ),
        };
        if errored > 0 {
            detail.push_str(&format!(" {errored} errored."));
        }
        if let Some(url) = &self.report_url {
            detail.push_str(&format!(" See {url} for details."));
        }
        if let Some(error) = &self.error {
            detail.push_str(&format!(" Error: {error}"));
        }
        if !self.sub_results.is_empty() {
            let failed: Vec<&str> = self
                .sub_results
                .iter()
                .filter(|(_, sub)| !sub.all_passed())
                .map(|(name, _)| name.as_str())
                .collect();
            if !failed.is_empty() {
                detail.push_str(&format!(" Failed: {}.", failed.join(", ")));
            }
        }
        let errored_items: Vec<String> = self
            .items
            .iter()
            .filter(|i| i.is_error())
            .map(|i| {
                format!(
                    "{}: {}",
                    i.item_id,
                    i.error_code.as_deref().unwrap_or("unknown")
                )
            })
            .collect();
        if !errored_items.is_empty() {
            detail.push_str(&format!(" Errored items: {}.", errored_items.join(", ")));
        }
        Err(Error::EvalNotPassed(detail))
    }

    fn walk<'a>(&'a self, visit: &mut dyn FnMut(&'a EvalResults)) {
        visit(self);
        for sub in self.sub_results.values() {
            sub.walk(visit);
        }
    }

    /// Assert every item's score (optionally only `evaluator`'s) is
    /// `>= min_score`, including workflow sub-results. Designed for CI gates
    /// on rubric evaluators, e.g. `results.assert_score_at_least(0.8, None, None)`.
    pub fn assert_score_at_least(
        &self,
        min_score: f64,
        evaluator: Option<&str>,
        msg: Option<&str>,
    ) -> Result<()> {
        let mut offenders: Vec<String> = Vec::new();
        self.walk(&mut |results| {
            for item in &results.items {
                for score in &item.scores {
                    if evaluator.is_some_and(|e| e != score.name) {
                        continue;
                    }
                    if score.score < min_score {
                        offenders.push(format!(
                            "{}/{}={:.3}",
                            item.item_id, score.name, score.score
                        ));
                    }
                }
            }
        });
        if offenders.is_empty() {
            return Ok(());
        }
        let detail = match msg {
            Some(m) => m.to_string(),
            None => format!(
                "{} score(s) below threshold {}{}: {}{}",
                offenders.len(),
                pyrepr::float(min_score),
                evaluator.map(|e| format!(" for {e}")).unwrap_or_default(),
                offenders[..offenders.len().min(5)].join(", "),
                more_suffix(offenders.len()),
            ),
        };
        Err(Error::EvalNotPassed(detail))
    }

    /// Assert every item's score for the rubric dimension `dimension_id` is
    /// `>= min_score`, including workflow sub-results.
    ///
    /// Non-applicable dimensions are skipped; with `require_applicable`, an
    /// item that produced no applicable score for the dimension fails too.
    /// An applicable dimension with no score counts as below the threshold.
    pub fn assert_dimension_score_at_least(
        &self,
        dimension_id: &str,
        min_score: f64,
        evaluator: Option<&str>,
        require_applicable: bool,
        msg: Option<&str>,
    ) -> Result<()> {
        let mut offenders: Vec<String> = Vec::new();
        let mut missing_items: Vec<String> = Vec::new();
        self.walk(&mut |results| {
            for item in &results.items {
                let mut found_applicable = false;
                for score in &item.scores {
                    if evaluator.is_some_and(|e| e != score.name) {
                        continue;
                    }
                    let Some(dimensions) = &score.dimensions else {
                        continue;
                    };
                    for rs in dimensions {
                        if rs.id != dimension_id || !rs.applicable {
                            continue;
                        }
                        found_applicable = true;
                        if rs.score.is_none_or(|s| (s as f64) < min_score) {
                            offenders.push(format!(
                                "{}/{}/{}={}",
                                item.item_id,
                                score.name,
                                dimension_id,
                                rs.score
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| "None".to_string())
                            ));
                        }
                    }
                }
                if require_applicable && !found_applicable {
                    missing_items.push(item.item_id.clone());
                }
            }
        });
        let mut problems: Vec<String> = Vec::new();
        if !offenders.is_empty() {
            problems.push(format!(
                "{} dimension score(s) for '{}' below {}: {}{}",
                offenders.len(),
                dimension_id,
                pyrepr::float(min_score),
                offenders[..offenders.len().min(5)].join(", "),
                more_suffix(offenders.len()),
            ));
        }
        if !missing_items.is_empty() {
            problems.push(format!(
                "Dimension '{}' not applicable on {} item(s): {}",
                dimension_id,
                missing_items.len(),
                missing_items[..missing_items.len().min(5)].join(", "),
            ));
        }
        if problems.is_empty() {
            return Ok(());
        }
        Err(Error::EvalNotPassed(
            msg.map(str::to_string)
                .unwrap_or_else(|| problems.join("; ")),
        ))
    }

    /// Assert no item ended in fail or error status, including workflow
    /// sub-results.
    pub fn assert_no_failed_items(&self, msg: Option<&str>) -> Result<()> {
        let mut bad: Vec<String> = Vec::new();
        self.walk(&mut |results| {
            for item in &results.items {
                if item.is_failed() || item.is_error() {
                    bad.push(format!("{}:{}", item.item_id, item.status));
                }
            }
        });
        if bad.is_empty() {
            return Ok(());
        }
        let detail = match msg {
            Some(m) => m.to_string(),
            None => format!(
                "{} item(s) failed or errored: {}{}",
                bad.len(),
                bad[..bad.len().min(5)].join(", "),
                more_suffix(bad.len()),
            ),
        };
        Err(Error::EvalNotPassed(detail))
    }
}

fn more_suffix(count: usize) -> String {
    if count > 5 {
        format!(" (+{} more)", count - 5)
    } else {
        String::new()
    }
}

// ---------------------------------------------------------------------------
// Evaluator trait
// ---------------------------------------------------------------------------

/// An evaluation provider. Mirrors upstream's `Evaluator` protocol.
///
/// Any backend — Microsoft Foundry, a local LLM-as-judge, custom scorers —
/// implements this. The provider owns its connection details, metric
/// selection and execution, and may auto-detect capabilities from the items
/// (e.g. run tool evaluators only when [`EvalItem::tools`] is set).
///
/// ```
/// use agent_framework_core::evaluation::{EvalItem, EvalResults, Evaluator, ResultCounts};
/// use agent_framework_core::error::Result;
///
/// struct NonEmpty;
///
/// #[async_trait::async_trait]
/// impl Evaluator for NonEmpty {
///     fn name(&self) -> &str {
///         "non-empty"
///     }
///     async fn evaluate(&self, items: &[EvalItem], eval_name: &str) -> Result<EvalResults> {
///         let passed = items.iter().filter(|i| !i.response().is_empty()).count() as u64;
///         let failed = items.len() as u64 - passed;
///         Ok(EvalResults::new(self.name())
///             .with_run_id(eval_name)
///             .with_result_counts(ResultCounts::new(passed, failed, 0)))
///     }
/// }
/// ```
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// The provider name, used in result labels and eval-name suffixes.
    fn name(&self) -> &str;

    /// Evaluate a batch of items and return results.
    async fn evaluate(&self, items: &[EvalItem], eval_name: &str) -> Result<EvalResults>;
}

#[async_trait]
impl<T: Evaluator + ?Sized> Evaluator for Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }

    async fn evaluate(&self, items: &[EvalItem], eval_name: &str) -> Result<EvalResults> {
        (**self).evaluate(items, eval_name).await
    }
}

#[cfg(test)]
mod tests;
