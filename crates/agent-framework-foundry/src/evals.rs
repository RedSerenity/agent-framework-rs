//! Microsoft Foundry Evals: an [`Evaluator`] backed by Foundry's built-in
//! evaluators.
//!
//! Rust port of upstream `agent_framework_foundry._foundry_evals`.
//! [`FoundryEvals`] plugs into the provider-agnostic
//! [`EvaluateAgent`](agent_framework_core::evaluation::EvaluateAgent) /
//! [`EvaluateWorkflow`](agent_framework_core::evaluation::EvaluateWorkflow)
//! runners; [`evaluate_traces`] and [`evaluate_foundry_target`] are the
//! Foundry-only entry points (evaluate stored responses / OTel traces, or let
//! Foundry invoke a registered target itself).
//!
//! The `builtin.*` evaluators are reached through the OpenAI **Evals API**
//! on the project endpoint (`{endpoint}/openai/v1/evals`): create an eval
//! definition with testing criteria, start a run over an inline JSONL
//! dataset (or a responses/traces/target data source), poll it to a terminal
//! state, then page through its output items for per-item scores.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_azure::AzureCliCredential;
//! use agent_framework_core::evaluation::EvaluateAgent;
//! use agent_framework_core::prelude::*;
//! use agent_framework_foundry::{FoundryEvals, FOUNDRY_SCOPE};
//!
//! # async fn demo(agent: Agent) -> Result<()> {
//! let evals = FoundryEvals::with_token_credential(
//!     "https://my-project.services.ai.azure.com/api/projects/p",
//!     "gpt-4o",
//!     Arc::new(AzureCliCredential::new(FOUNDRY_SCOPE)),
//! )
//! .with_evaluators([FoundryEvals::RELEVANCE, FoundryEvals::TASK_ADHERENCE]);
//! let results = EvaluateAgent::new()
//!     .agent(&agent)
//!     .query("What's the weather in Seattle?")
//!     .evaluator(evals)
//!     .run()
//!     .await?;
//! results[0].raise_for_status(None)?;
//! println!("{:?}", results[0].report_url);
//! # Ok(())
//! # }
//! ```
//!
//! ## Divergences from upstream
//!
//! - Upstream drives the `openai` SDK's `AsyncOpenAI` client (from a
//!   `FoundryChatClient`, an `AIProjectClient`, or one auto-created from the
//!   environment). This port speaks the same REST routes directly with
//!   `reqwest`, authenticating with an API key or a [`TokenCredential`]
//!   (bearer, scoped to [`FOUNDRY_SCOPE`](crate::FOUNDRY_SCOPE)), like the
//!   rest of this crate. [`FoundryEvals::from_env`] stands in for the
//!   zero-config constructor.
//! - [`evaluate_traces`] / [`evaluate_foundry_target`] take a
//!   [`FoundryEvals`] for the connection, judge model and polling settings,
//!   where upstream takes `client` / `project_client` / `model` /
//!   `poll_interval` / `timeout` keyword arguments.
//! - Output-item pages are fetched with `limit=100` and the `after` cursor.
//!   A transport or HTTP failure while paging propagates, as an `openai`
//!   SDK error does upstream; an output item with an unexpected shape is
//!   logged and ends the listing with the items gathered so far (upstream
//!   catches `AttributeError`/`KeyError`/`TypeError`).
//! - An output-item result with no numeric `score` reads as `0.0`.

use std::sync::Arc;
use std::time::Duration;

use agent_framework_azure::TokenCredential;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::evaluation::{
    ConversationSplit, ConversationSplitter, EvalItem, EvalItemResult, EvalResults, EvalRunStatus,
    EvalScoreResult, Evaluator, ResultCounts, RubricScore,
};
use agent_framework_core::types::{Content, FunctionArguments, Message};
use async_trait::async_trait;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Evaluator names
// ---------------------------------------------------------------------------

/// A reference to a rubric evaluator that already exists in Foundry. Mirrors
/// upstream's `GeneratedEvaluatorRef`.
///
/// agent-framework only *references* the persisted evaluator by name; it
/// never creates or modifies the definition. Pin a [`version`](Self::version)
/// for reproducible runs — an unpinned reference resolves to whatever is
/// current at execution time, and a warning is logged when one is used.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GeneratedEvaluatorRef {
    /// Evaluator name as stored in the project (e.g.
    /// `"reservation-policy-rubric"`).
    pub name: String,
    /// Pinned version; `None` means "latest" (discouraged for CI).
    pub version: Option<String>,
    /// Human-readable name for result summaries; defaults to `name`.
    pub display_name: Option<String>,
}

impl GeneratedEvaluatorRef {
    /// A reference pinned to `version`.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: Some(version.into()),
            display_name: None,
        }
    }

    /// A versionless reference (resolves to the latest version at run time).
    /// Discouraged for reproducible runs.
    pub fn latest(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
            display_name: None,
        }
    }

    /// Set the display name.
    pub fn with_display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = Some(display_name.into());
        self
    }
}

/// One evaluator to run: a built-in name (short, e.g. `"relevance"`, or
/// fully qualified, e.g. `"builtin.relevance"`) or a
/// [`GeneratedEvaluatorRef`]. Upstream: `str | GeneratedEvaluatorRef`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EvaluatorSpec {
    /// A built-in evaluator name.
    Builtin(String),
    /// A pre-existing rubric evaluator.
    Generated(GeneratedEvaluatorRef),
}

impl EvaluatorSpec {
    fn label(&self) -> &str {
        match self {
            EvaluatorSpec::Builtin(name) => name,
            EvaluatorSpec::Generated(r) => &r.name,
        }
    }
}

impl From<&str> for EvaluatorSpec {
    fn from(value: &str) -> Self {
        EvaluatorSpec::Builtin(value.to_string())
    }
}

impl From<String> for EvaluatorSpec {
    fn from(value: String) -> Self {
        EvaluatorSpec::Builtin(value)
    }
}

impl From<GeneratedEvaluatorRef> for EvaluatorSpec {
    fn from(value: GeneratedEvaluatorRef) -> Self {
        EvaluatorSpec::Generated(value)
    }
}

/// Agent evaluators: query/response go in as conversation arrays.
const AGENT_EVALUATORS: &[&str] = &[
    "builtin.intent_resolution",
    "builtin.task_adherence",
    "builtin.task_completion",
    "builtin.task_navigation_efficiency",
    "builtin.tool_call_accuracy",
    "builtin.tool_selection",
    "builtin.tool_input_accuracy",
    "builtin.tool_output_utilization",
    "builtin.tool_call_success",
];

/// Evaluators that require tool definitions.
const TOOL_EVALUATORS: &[&str] = &[
    "builtin.tool_call_accuracy",
    "builtin.tool_selection",
    "builtin.tool_input_accuracy",
    "builtin.tool_output_utilization",
    "builtin.tool_call_success",
];

/// Evaluators that accept tool definitions in their data mapping (the tool
/// evaluators plus these).
const EXTRA_TOOL_DEFINITION_EVALUATORS: &[&str] = &[
    "builtin.intent_resolution",
    "builtin.task_adherence",
    "builtin.task_completion",
    "builtin.task_navigation_efficiency",
];

/// Evaluators that require a ground truth.
const GROUND_TRUTH_EVALUATORS: &[&str] = &["builtin.similarity"];

/// Short name → fully-qualified built-in evaluator name.
const BUILTIN_EVALUATORS: &[(&str, &str)] = &[
    ("intent_resolution", "builtin.intent_resolution"),
    ("task_adherence", "builtin.task_adherence"),
    ("task_completion", "builtin.task_completion"),
    (
        "task_navigation_efficiency",
        "builtin.task_navigation_efficiency",
    ),
    ("tool_call_accuracy", "builtin.tool_call_accuracy"),
    ("tool_selection", "builtin.tool_selection"),
    ("tool_input_accuracy", "builtin.tool_input_accuracy"),
    ("tool_output_utilization", "builtin.tool_output_utilization"),
    ("tool_call_success", "builtin.tool_call_success"),
    ("coherence", "builtin.coherence"),
    ("fluency", "builtin.fluency"),
    ("relevance", "builtin.relevance"),
    ("groundedness", "builtin.groundedness"),
    ("response_completeness", "builtin.response_completeness"),
    ("similarity", "builtin.similarity"),
    ("violence", "builtin.violence"),
    ("sexual", "builtin.sexual"),
    ("self_harm", "builtin.self_harm"),
    ("hate_unfairness", "builtin.hate_unfairness"),
];

/// Evaluators used when none are configured.
const DEFAULT_EVALUATORS: &[&str] = &["relevance", "coherence", "task_adherence"];

/// Added to the defaults when any item carries tools.
const DEFAULT_TOOL_EVALUATORS: &[&str] = &["tool_call_accuracy"];

fn py_list<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items
        .iter()
        .map(|s| {
            format!(
                "'{}'",
                s.as_ref().replace('\\', "\\\\").replace('\'', "\\'")
            )
        })
        .collect();
    format!("[{}]", inner.join(", "))
}

/// Resolve a short evaluator name to its `builtin.*` form. Mirrors
/// upstream's `_resolve_evaluator`: a qualified name passes through (with a
/// warning when it is not a known built-in), an unknown short name is an
/// error.
pub fn resolve_evaluator(name: &str) -> Result<String> {
    if let Some(short) = name.strip_prefix("builtin.") {
        if !BUILTIN_EVALUATORS.iter().any(|(s, _)| *s == short) {
            tracing::warn!(
                evaluator = name,
                "evaluator is not in the known built-in list; if this is a new evaluator, \
                 consider updating BUILTIN_EVALUATORS"
            );
        }
        return Ok(name.to_string());
    }
    match BUILTIN_EVALUATORS.iter().find(|(s, _)| *s == name) {
        Some((_, qualified)) => Ok(qualified.to_string()),
        None => {
            let mut available: Vec<&str> = BUILTIN_EVALUATORS.iter().map(|(s, _)| *s).collect();
            available.sort_unstable();
            Err(Error::Configuration(format!(
                "Unknown evaluator '{name}'. Available: {}",
                py_list(&available)
            )))
        }
    }
}

// ---------------------------------------------------------------------------
// Wire conversion
// ---------------------------------------------------------------------------

fn arguments_json(arguments: &Option<FunctionArguments>) -> Value {
    match arguments {
        None => json!({}),
        Some(FunctionArguments::Object(map)) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            )
        }
        Some(FunctionArguments::Raw(raw)) => {
            serde_json::from_str(raw).unwrap_or_else(|_| json!({"_raw_arguments": "[unparseable]"}))
        }
    }
}

/// Convert one message to the Foundry Evals wire format. Mirrors upstream's
/// `_convert_message`.
///
/// Text becomes `{"type": "text"}`; data and URI content become
/// `{"type": "input_image", "image_url"}` (plus `"detail": "auto"` when a
/// media type is known); function calls become `{"type": "tool_call"}` with
/// parsed arguments (`{}` when absent, `{"_raw_arguments": "[unparseable]"}`
/// when not JSON). Function results become one `role: "tool"` message each
/// (a JSON-string result is parsed). A message with nothing convertible
/// becomes a single empty text item.
pub fn convert_message(message: &Message) -> Vec<Value> {
    let mut content_items: Vec<Value> = Vec::new();
    let mut tool_results: Vec<(String, Value)> = Vec::new();

    for content in &message.contents {
        match content {
            Content::Text(t) if !t.text.is_empty() => {
                content_items.push(json!({"type": "text", "text": t.text}));
            }
            Content::Data(d) if !d.uri.is_empty() => {
                let mut image = json!({"type": "input_image", "image_url": d.uri});
                if d.media_type.as_deref().is_some_and(|m| !m.is_empty()) {
                    image["detail"] = json!("auto");
                }
                content_items.push(image);
            }
            Content::Uri(u) if !u.uri.is_empty() => {
                let mut image = json!({"type": "input_image", "image_url": u.uri});
                if !u.media_type.is_empty() {
                    image["detail"] = json!("auto");
                }
                content_items.push(image);
            }
            Content::FunctionCall(fc) => content_items.push(json!({
                "type": "tool_call",
                "tool_call_id": fc.call_id,
                "name": fc.name,
                "arguments": arguments_json(&fc.arguments),
            })),
            Content::FunctionResult(fr) => {
                let result = match &fr.result {
                    Some(Value::String(s)) => {
                        serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
                    }
                    Some(other) => other.clone(),
                    None => Value::Null,
                };
                tool_results.push((fr.call_id.clone(), result));
            }
            _ => {}
        }
    }

    if !tool_results.is_empty() {
        return tool_results
            .into_iter()
            .map(|(call_id, result)| {
                json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": [{"type": "tool_result", "tool_result": result}],
                })
            })
            .collect();
    }
    if !content_items.is_empty() {
        return vec![json!({"role": message.role.as_str(), "content": content_items})];
    }
    vec![json!({"role": message.role.as_str(), "content": [{"type": "text", "text": ""}]})]
}

/// Convert messages to the Foundry Evals wire format. Mirrors upstream's
/// `_convert_messages`.
pub fn convert_messages(messages: &[Message]) -> Vec<Value> {
    messages.iter().flat_map(convert_message).collect()
}

// ---------------------------------------------------------------------------
// Testing criteria and schema
// ---------------------------------------------------------------------------

/// Build the `testing_criteria` for an eval definition. Mirrors upstream's
/// `_build_testing_criteria`.
///
/// With `include_data_mapping` (the JSONL dataset path) each criterion maps
/// item fields: agent evaluators and generated rubric evaluators take the
/// conversation arrays, quality evaluators the plain strings; groundedness
/// adds `context`, similarity adds `ground_truth`, and tool-aware
/// evaluators add `tool_definitions` when `include_tool_definitions`.
pub fn build_testing_criteria(
    evaluators: &[EvaluatorSpec],
    model: &str,
    include_data_mapping: bool,
    include_tool_definitions: bool,
) -> Result<Vec<Value>> {
    let mut criteria = Vec::with_capacity(evaluators.len());
    for spec in evaluators {
        match spec {
            EvaluatorSpec::Generated(r) => {
                let mut entry = json!({
                    "type": "azure_ai_evaluator",
                    "name": r.display_name.as_deref().unwrap_or(&r.name),
                    "evaluator_name": r.name,
                    "initialization_parameters": {"deployment_name": model},
                });
                match &r.version {
                    Some(version) => entry["evaluator_version"] = json!(version),
                    None => tracing::warn!(
                        evaluator = %r.name,
                        "GeneratedEvaluatorRef has no pinned version; the eval run will resolve \
                         to whichever version is current at execution time. Pin the version for \
                         reproducible runs."
                    ),
                }
                if include_data_mapping {
                    let mut mapping = json!({
                        "query": "{{item.query_messages}}",
                        "response": "{{item.response_messages}}",
                    });
                    if include_tool_definitions {
                        mapping["tool_definitions"] = json!("{{item.tool_definitions}}");
                    }
                    entry["data_mapping"] = mapping;
                }
                criteria.push(entry);
            }
            EvaluatorSpec::Builtin(name) => {
                let qualified = resolve_evaluator(name)?;
                let short = if name.starts_with("builtin.") {
                    name.rsplit('.').next().unwrap_or(name)
                } else {
                    name
                };
                let mut entry = json!({
                    "type": "azure_ai_evaluator",
                    "name": short,
                    "evaluator_name": qualified,
                    "initialization_parameters": {"deployment_name": model},
                });
                if include_data_mapping {
                    let q = qualified.as_str();
                    let mut mapping = if AGENT_EVALUATORS.contains(&q) {
                        json!({
                            "query": "{{item.query_messages}}",
                            "response": "{{item.response_messages}}",
                        })
                    } else {
                        json!({"query": "{{item.query}}", "response": "{{item.response}}"})
                    };
                    if q == "builtin.groundedness" {
                        mapping["context"] = json!("{{item.context}}");
                    }
                    if GROUND_TRUTH_EVALUATORS.contains(&q) {
                        mapping["ground_truth"] = json!("{{item.ground_truth}}");
                    }
                    if include_tool_definitions
                        && (TOOL_EVALUATORS.contains(&q)
                            || EXTRA_TOOL_DEFINITION_EVALUATORS.contains(&q))
                    {
                        mapping["tool_definitions"] = json!("{{item.tool_definitions}}");
                    }
                    entry["data_mapping"] = mapping;
                }
                criteria.push(entry);
            }
        }
    }
    Ok(criteria)
}

/// Build the `item_schema` of a custom JSONL eval definition. Mirrors
/// upstream's `_build_item_schema`.
pub fn build_item_schema(has_context: bool, has_tools: bool, has_ground_truth: bool) -> Value {
    let mut properties = json!({
        "query": {"type": "string"},
        "response": {"type": "string"},
        "query_messages": {"type": "array"},
        "response_messages": {"type": "array"},
    });
    if has_context {
        properties["context"] = json!({"type": "string"});
    }
    if has_ground_truth {
        properties["ground_truth"] = json!({"type": "string"});
    }
    if has_tools {
        properties["tool_definitions"] = json!({"type": "array"});
    }
    json!({"type": "object", "properties": properties, "required": ["query", "response"]})
}

fn items_have_tools(items: &[EvalItem]) -> bool {
    items
        .iter()
        .any(|i| i.tools.as_ref().is_some_and(|t| !t.is_empty()))
}

/// Apply defaults when no evaluators are configured: relevance, coherence and
/// task adherence, plus tool-call accuracy when any item carries tools.
/// Mirrors upstream's `_resolve_default_evaluators`.
fn resolve_default_evaluators(
    evaluators: Option<&[EvaluatorSpec]>,
    items: Option<&[EvalItem]>,
) -> Vec<EvaluatorSpec> {
    if let Some(evaluators) = evaluators {
        return evaluators.to_vec();
    }
    let mut result: Vec<EvaluatorSpec> = DEFAULT_EVALUATORS.iter().map(|&s| s.into()).collect();
    if items.is_some_and(items_have_tools) {
        result.extend(DEFAULT_TOOL_EVALUATORS.iter().map(|&s| s.into()));
    }
    result
}

/// Drop tool-only evaluators when no item has tools; error when nothing
/// remains. Generated rubric evaluators are kept regardless. Mirrors
/// upstream's `_filter_tool_evaluators`.
fn filter_tool_evaluators(
    evaluators: Vec<EvaluatorSpec>,
    items: &[EvalItem],
) -> Result<Vec<EvaluatorSpec>> {
    if items_have_tools(items) {
        return Ok(evaluators);
    }
    let is_tool_only = |spec: &EvaluatorSpec| -> Result<bool> {
        match spec {
            EvaluatorSpec::Generated(_) => Ok(false),
            EvaluatorSpec::Builtin(name) => {
                Ok(TOOL_EVALUATORS.contains(&resolve_evaluator(name)?.as_str()))
            }
        }
    };
    let mut kept = Vec::new();
    let mut removed = Vec::new();
    for spec in &evaluators {
        if is_tool_only(spec)? {
            removed.push(spec.label().to_string());
        } else {
            kept.push(spec.clone());
        }
    }
    if kept.is_empty() {
        let labels: Vec<&str> = evaluators.iter().map(EvaluatorSpec::label).collect();
        return Err(Error::Configuration(format!(
            "All requested evaluators {} require tool definitions, but no items have tools. \
             Either add tool definitions to your items or choose evaluators that do not \
             require tools.",
            py_list(&labels)
        )));
    }
    if !removed.is_empty() {
        tracing::info!(removed = %py_list(&removed), "removed tool evaluators (no items have tools)");
    }
    Ok(kept)
}

// ---------------------------------------------------------------------------
// Result parsing
// ---------------------------------------------------------------------------

/// Python `str(v)` of a JSON value.
fn value_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `int(v)`.
fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::Bool(b) => Some(i64::from(*b)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Parse a list of rubric dimension entries, skipping malformed ones (no
/// `id`, `weight` or `applicable`, or a non-integer weight). Mirrors
/// upstream's `_parse_dimension_entries`.
fn parse_dimension_entries(raw: &Value) -> Vec<RubricScore> {
    let Some(entries) = raw.as_array() else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let entry = entry.as_object()?;
            let id = entry.get("id").filter(|v| !v.is_null())?;
            let applicable = entry.get("applicable").filter(|v| !v.is_null())?;
            let weight = entry.get("weight").filter(|v| !v.is_null())?;
            let Some(weight) = py_int(weight) else {
                tracing::debug!(?entry, "skipping malformed rubric dimension entry");
                return None;
            };
            let score = match entry.get("score") {
                Some(Value::Number(n)) => {
                    n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64))
                }
                Some(Value::Bool(b)) => Some(i64::from(*b)),
                _ => None,
            };
            Some(RubricScore {
                id: value_str(id),
                score,
                applicable: truthy(applicable),
                weight,
                reason: match entry.get("reason") {
                    None | Some(Value::Null) => String::new(),
                    Some(r) => value_str(r),
                },
            })
        })
        .collect()
}

/// Property keys that may carry rubric breakdowns: the documented
/// `dimension_scores`, then the preview-era `rubric_scores`.
const RUBRIC_DIMENSION_KEYS: &[&str] = &["dimension_scores", "rubric_scores"];

/// Extract rubric dimensions from an evaluator result's raw `sample`: under
/// `properties.<key>` or `<key>` at the top level, `dimension_scores` taking
/// priority. `None` when absent (not a rubric evaluator). Mirrors upstream's
/// `_extract_rubric_scores`.
pub fn extract_rubric_scores(sample: &Value) -> Option<Vec<RubricScore>> {
    let object = sample.as_object()?;
    let mut containers: Vec<&Map<String, Value>> = Vec::new();
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        containers.push(properties);
    }
    containers.push(object);
    for container in containers {
        for key in RUBRIC_DIMENSION_KEYS {
            if let Some(raw) = container.get(*key) {
                let parsed = parse_dimension_entries(raw);
                if !parsed.is_empty() {
                    return Some(parsed);
                }
            }
        }
    }
    None
}

fn non_empty_str(v: Option<&Value>) -> Option<String> {
    v.and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Convert one `output_items` entry. Mirrors the body of upstream's
/// `_fetch_output_items` loop; `None` when the entry is not an object with an
/// `id`.
fn parse_output_item(oi: &Value) -> Option<EvalItemResult> {
    let id = oi.get("id")?.as_str()?;
    let status = oi.get("status").and_then(Value::as_str).unwrap_or_default();
    let mut item = EvalItemResult::new(id, status);

    if let Some(results) = oi.get("results").and_then(Value::as_array) {
        for r in results {
            let sample = r.get("sample").filter(|s| !s.is_null()).cloned();
            item.scores.push(EvalScoreResult {
                name: r.get("name").map(value_str).unwrap_or_default(),
                score: r.get("score").and_then(Value::as_f64).unwrap_or(0.0),
                passed: r.get("passed").and_then(Value::as_bool),
                dimensions: sample.as_ref().and_then(extract_rubric_scores),
                sample,
            });
        }
    }

    if let Some(sample) = oi.get("sample").filter(|s| s.is_object()) {
        if let Some(err) = sample.get("error").filter(|e| e.is_object()) {
            let code = non_empty_str(err.get("code"));
            let message = non_empty_str(err.get("message"));
            if code.is_some() || message.is_some() {
                item.error_code = code;
                item.error_message = message;
            }
        }
        if let Some(usage) = sample.get("usage").filter(|u| u.is_object()) {
            if usage.get("total_tokens").is_some_and(truthy) {
                let mut tokens = std::collections::BTreeMap::new();
                for key in [
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cached_tokens",
                ] {
                    if let Some(n) = usage.get(key).and_then(Value::as_i64) {
                        tokens.insert(key.to_string(), n);
                    }
                }
                item.token_usage = Some(tokens);
            }
        }
        let joined = |key: &str, role: &str| -> Option<String> {
            let parts: Vec<String> = sample
                .get(key)?
                .as_array()?
                .iter()
                .filter(|m| m.get("role").and_then(Value::as_str) == Some(role))
                .map(|m| match m.get("content") {
                    None | Some(Value::Null) => String::new(),
                    Some(c) => value_str(c),
                })
                .collect();
            (!parts.is_empty()).then(|| parts.join(" "))
        };
        item.input_text = joined("input", "user");
        item.output_text = joined("output", "assistant");
    }

    if let Some(ds) = oi.get("datasource_item").and_then(Value::as_object) {
        item.response_id = ds
            .get("resp_id")
            .filter(|v| truthy(v))
            .or_else(|| ds.get("response_id").filter(|v| truthy(v)))
            .map(value_str);
    }
    Some(item)
}

/// `result_counts` of a run, or `None` when absent. Mirrors upstream's
/// `_extract_result_counts`.
fn extract_result_counts(run: &Value) -> Option<ResultCounts> {
    let counts = run.get("result_counts").filter(|c| c.is_object())?;
    let get = |k: &str| counts.get(k).and_then(Value::as_u64).unwrap_or(0);
    Some(ResultCounts {
        passed: get("passed"),
        failed: get("failed"),
        errored: get("errored"),
        total: counts.get("total").and_then(Value::as_u64),
    })
}

/// Per-criterion counts of a run. Mirrors upstream's
/// `_extract_per_evaluator`.
fn extract_per_evaluator(run: &Value) -> std::collections::BTreeMap<String, ResultCounts> {
    run.get("per_testing_criteria_results")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|e| {
                    let name = non_empty_str(e.get("testing_criteria"))?;
                    let get = |k: &str| e.get(k).and_then(Value::as_u64).unwrap_or(0);
                    Some((name, ResultCounts::new(get("passed"), get("failed"), 0)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The error text of a failed run: a string as-is, an object's `message`,
/// else the JSON.
fn run_error(run: &Value) -> Option<String> {
    match run.get("error")? {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(non_empty_str(other.get("message")).unwrap_or_else(|| other.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Auth {
    ApiKey(String),
    Credential(Arc<dyn TokenCredential>),
}

#[derive(Clone)]
struct EvalsTransport {
    http: reqwest::Client,
    base_url: String,
    api_version: Option<String>,
    auth: Auth,
    token_scope: String,
}

impl EvalsTransport {
    fn url(&self, path: &str, query: &[(&str, String)]) -> String {
        let mut url = format!("{}{}", self.base_url, path);
        let mut params: Vec<String> = query
            .iter()
            .map(|(k, v)| format!("{k}={}", encode_query(v)))
            .collect();
        if let Some(v) = &self.api_version {
            params.push(format!("api-version={}", encode_query(v)));
        }
        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }
        url
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> Result<Value> {
        let request = match &self.auth {
            Auth::ApiKey(key) => request.header("api-key", key),
            Auth::Credential(credential) => {
                request.bearer_auth(credential.get_token_for_scope(&self.token_scope).await?)
            }
        };
        let resp = request
            .send()
            .await
            .map_err(|e| Error::service(format!("Foundry evals request failed: {e}")))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::service_status(
                status.as_u16(),
                format!("Foundry evals API error {status}: {text}"),
                None,
            ));
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text)
            .map_err(|e| Error::service(format!("Foundry evals returned invalid JSON: {e}")))
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value> {
        self.send(self.http.post(self.url(path, &[])).json(body))
            .await
    }

    async fn get(&self, path: &str, query: &[(&str, String)]) -> Result<Value> {
        self.send(self.http.get(self.url(path, query))).await
    }

    /// `evals.create(...)`; returns the eval id.
    async fn create_eval(&self, body: Value) -> Result<String> {
        let created = self.post("/evals", &body).await?;
        id_of(&created, "eval")
    }

    /// `evals.runs.create(...)`; returns the run id.
    async fn create_run(&self, eval_id: &str, body: Value) -> Result<String> {
        let created = self.post(&format!("/evals/{eval_id}/runs"), &body).await?;
        id_of(&created, "eval run")
    }

    /// Page through `evals.runs.output_items.list(...)`.
    async fn fetch_output_items(&self, eval_id: &str, run_id: &str) -> Result<Vec<EvalItemResult>> {
        let mut items = Vec::new();
        let mut after: Option<String> = None;
        loop {
            let mut query = vec![("limit", "100".to_string())];
            if let Some(a) = &after {
                query.push(("after", a.clone()));
            }
            let page = self
                .get(
                    &format!("/evals/{eval_id}/runs/{run_id}/output_items"),
                    &query,
                )
                .await?;
            let Some(data) = page.get("data").and_then(Value::as_array) else {
                tracing::warn!(run_id, "could not fetch output_items: no `data` array");
                return Ok(items);
            };
            for oi in data {
                match parse_output_item(oi) {
                    Some(item) => items.push(item),
                    None => {
                        tracing::warn!(run_id, "could not fetch output_items: malformed item");
                        return Ok(items);
                    }
                }
            }
            let has_more = page
                .get("has_more")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let last_id = page
                .get("last_id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| {
                    data.last()
                        .and_then(|d| d.get("id"))
                        .and_then(Value::as_str)
                        .map(str::to_string)
                });
            match (has_more, last_id) {
                (true, Some(last)) if !data.is_empty() => after = Some(last),
                _ => return Ok(items),
            }
        }
    }

    /// Poll a run until it is terminal or `timeout` passes. Mirrors
    /// upstream's `_poll_eval_run`.
    async fn poll_eval_run(
        &self,
        eval_id: &str,
        run_id: &str,
        poll_interval: Duration,
        timeout: Duration,
        provider: &str,
        fetch_output_items: bool,
    ) -> Result<EvalResults> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let run = self
                .get(&format!("/evals/{eval_id}/runs/{run_id}"), &[])
                .await?;
            let status = run
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if matches!(status, "completed" | "failed" | "canceled") {
                let error = if status == "failed" {
                    run_error(&run)
                } else {
                    None
                };
                let items = if fetch_output_items && status == "completed" {
                    self.fetch_output_items(eval_id, run_id).await?
                } else {
                    Vec::new()
                };
                let mut results = EvalResults::new(provider)
                    .with_eval_id(eval_id)
                    .with_run_id(run_id)
                    .with_status(status)
                    .with_items(items);
                results.result_counts = extract_result_counts(&run);
                results.report_url = non_empty_str(run.get("report_url"));
                results.error = error;
                results.per_evaluator = extract_per_evaluator(&run);
                return Ok(results);
            }
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return Ok(EvalResults::new(provider)
                    .with_eval_id(eval_id)
                    .with_run_id(run_id)
                    .with_status(EvalRunStatus::Timeout));
            }
            let remaining = deadline - now;
            tracing::debug!(run_id, status, ?remaining, "eval run still in progress");
            tokio::time::sleep(poll_interval.min(remaining)).await;
        }
    }
}

fn id_of(value: &Value, what: &str) -> Result<String> {
    value
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Error::service(format!("Foundry evals: created {what} has no `id`")))
}

/// Percent-encode a query value (RFC 3986 unreserved characters pass).
fn encode_query(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// FoundryEvals
// ---------------------------------------------------------------------------

/// An [`Evaluator`] backed by Microsoft Foundry's built-in evaluators.
/// Mirrors upstream's `FoundryEvals` (provider name `"Microsoft Foundry"`).
///
/// **Evaluator selection:** by default runs relevance, coherence and task
/// adherence, adding tool-call accuracy when any item carries tool
/// definitions; override with [`with_evaluators`](Self::with_evaluators).
/// Tool-only evaluators are dropped when no item has tools (an error when
/// nothing would remain).
///
/// Each [`evaluate`](Evaluator::evaluate) call creates an eval definition
/// with a custom JSONL item schema, starts a run over the items (split into
/// query/response with each item's own strategy, else this evaluator's
/// [`conversation_split`](Self::with_conversation_split)), polls it every
/// [`poll_interval`](Self::with_poll_interval) (default 5 s) for up to
/// [`timeout`](Self::with_timeout) (default 180 s; past it the result's
/// status is [`EvalRunStatus::Timeout`]), and fetches per-item results.
#[derive(Clone)]
pub struct FoundryEvals {
    transport: EvalsTransport,
    model: String,
    evaluators: Option<Vec<EvaluatorSpec>>,
    conversation_split: Arc<dyn ConversationSplitter>,
    poll_interval: Duration,
    timeout: Duration,
}

impl std::fmt::Debug for FoundryEvals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryEvals")
            .field("base_url", &self.transport.base_url)
            .field("model", &self.model)
            .field("evaluators", &self.evaluators)
            .field("poll_interval", &self.poll_interval)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl FoundryEvals {
    // Agent behavior
    /// `intent_resolution`.
    pub const INTENT_RESOLUTION: &'static str = "intent_resolution";
    /// `task_adherence`.
    pub const TASK_ADHERENCE: &'static str = "task_adherence";
    /// `task_completion`.
    pub const TASK_COMPLETION: &'static str = "task_completion";
    /// `task_navigation_efficiency`.
    pub const TASK_NAVIGATION_EFFICIENCY: &'static str = "task_navigation_efficiency";
    // Tool usage
    /// `tool_call_accuracy`.
    pub const TOOL_CALL_ACCURACY: &'static str = "tool_call_accuracy";
    /// `tool_selection`.
    pub const TOOL_SELECTION: &'static str = "tool_selection";
    /// `tool_input_accuracy`.
    pub const TOOL_INPUT_ACCURACY: &'static str = "tool_input_accuracy";
    /// `tool_output_utilization`.
    pub const TOOL_OUTPUT_UTILIZATION: &'static str = "tool_output_utilization";
    /// `tool_call_success`.
    pub const TOOL_CALL_SUCCESS: &'static str = "tool_call_success";
    // Quality
    /// `coherence`.
    pub const COHERENCE: &'static str = "coherence";
    /// `fluency`.
    pub const FLUENCY: &'static str = "fluency";
    /// `relevance`.
    pub const RELEVANCE: &'static str = "relevance";
    /// `groundedness`.
    pub const GROUNDEDNESS: &'static str = "groundedness";
    /// `response_completeness`.
    pub const RESPONSE_COMPLETENESS: &'static str = "response_completeness";
    /// `similarity`.
    pub const SIMILARITY: &'static str = "similarity";
    // Safety
    /// `violence`.
    pub const VIOLENCE: &'static str = "violence";
    /// `sexual`.
    pub const SEXUAL: &'static str = "sexual";
    /// `self_harm`.
    pub const SELF_HARM: &'static str = "self_harm";
    /// `hate_unfairness`.
    pub const HATE_UNFAIRNESS: &'static str = "hate_unfairness";

    fn build(endpoint: impl Into<String>, model: impl Into<String>, auth: Auth) -> Self {
        let endpoint = endpoint.into();
        Self {
            transport: EvalsTransport {
                http: reqwest::Client::new(),
                base_url: format!("{}/openai/v1", endpoint.trim_end_matches('/')),
                api_version: None,
                auth,
                token_scope: crate::FOUNDRY_SCOPE.to_string(),
            },
            model: model.into(),
            evaluators: None,
            conversation_split: Arc::new(ConversationSplit::LastTurn),
            poll_interval: Duration::from_secs(5),
            timeout: Duration::from_secs(180),
        }
    }

    /// Evaluate against a Foundry project `endpoint`, judging with the
    /// `model` deployment, authenticating with an API key (`api-key`
    /// header).
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self::build(endpoint, model, Auth::ApiKey(api_key.into()))
    }

    /// Evaluate against a Foundry project `endpoint`, judging with the
    /// `model` deployment, authenticating with a Microsoft Entra ID bearer
    /// token requested for [`FOUNDRY_SCOPE`](crate::FOUNDRY_SCOPE).
    pub fn with_token_credential(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self::build(endpoint, model, Auth::Credential(credential))
    }

    /// Build from the environment, as
    /// [`FoundryChatClient::from_env`](crate::FoundryChatClient::from_env)
    /// does: `FOUNDRY_ENDPOINT` (alias `FOUNDRY_PROJECT_ENDPOINT`), then
    /// `FOUNDRY_API_KEY` or else a `DefaultAzureCredential`. The judge model
    /// is `FOUNDRY_MODEL`, else `"gpt-4o"` (upstream's default when no model
    /// is given).
    ///
    /// # Errors
    /// [`Error::Configuration`] when no endpoint variable is set.
    pub fn from_env() -> Result<Self> {
        Self::from_env_vars(|k| std::env::var(k).ok())
    }

    fn from_env_vars(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let endpoint = get("FOUNDRY_ENDPOINT")
            .or_else(|| get("FOUNDRY_PROJECT_ENDPOINT"))
            .ok_or_else(|| {
                Error::Configuration(
                    "FOUNDRY_ENDPOINT (or FOUNDRY_PROJECT_ENDPOINT) is not set".into(),
                )
            })?;
        let model = get("FOUNDRY_MODEL").unwrap_or_else(|| "gpt-4o".to_string());
        Ok(match get("FOUNDRY_API_KEY") {
            Some(key) => Self::new(endpoint, model, key),
            None => Self::with_token_credential(
                endpoint,
                model,
                Arc::new(agent_framework_azure::DefaultAzureCredential::new(
                    crate::FOUNDRY_SCOPE,
                )),
            ),
        })
    }

    /// Override the base URL the Evals routes hang off (default
    /// `{endpoint}/openai/v1`).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.transport.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Send an `api-version` query parameter on every request (none by
    /// default; the v1 route is path-versioned).
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.transport.api_version = Some(api_version.into());
        self
    }

    /// Override the Entra ID scope requested for the bearer token.
    pub fn with_token_scope(mut self, scope: impl Into<String>) -> Self {
        self.transport.token_scope = scope.into();
        self
    }

    /// Override the judge model deployment.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Choose the evaluators (built-in names or [`GeneratedEvaluatorRef`]s)
    /// instead of the smart defaults.
    pub fn with_evaluators<I, E>(mut self, evaluators: I) -> Self
    where
        I: IntoIterator<Item = E>,
        E: Into<EvaluatorSpec>,
    {
        self.evaluators = Some(evaluators.into_iter().map(Into::into).collect());
        self
    }

    /// Default split strategy for items without their own (default
    /// [`ConversationSplit::LastTurn`]).
    pub fn with_conversation_split(mut self, split: impl ConversationSplitter + 'static) -> Self {
        self.conversation_split = Arc::new(split);
        self
    }

    /// Time between status polls (default 5 s).
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Maximum time to wait for a run (default 180 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The judge model deployment.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The configured evaluators, or `None` for the smart defaults.
    pub fn evaluators(&self) -> Option<&[EvaluatorSpec]> {
        self.evaluators.as_deref()
    }

    /// The JSONL row for one item. Mirrors the per-item body of upstream's
    /// `_evaluate_via_dataset`.
    fn dataset_row(&self, item: &EvalItem) -> Value {
        let (query_msgs, response_msgs) = match &item.split_strategy {
            Some(split) => item.split_messages_with(split.as_ref()),
            None => item.split_messages_with(self.conversation_split.as_ref()),
        };
        let text_of = |msgs: &[Message], role: &str| {
            msgs.iter()
                .filter(|m| m.role.as_str() == role)
                .map(Message::text)
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string()
        };
        let mut row = json!({
            "query": text_of(&query_msgs, "user"),
            "response": text_of(&response_msgs, "assistant"),
            "query_messages": convert_messages(&query_msgs),
            "response_messages": convert_messages(&response_msgs),
        });
        if let Some(tools) = item.tools.as_ref().filter(|t| !t.is_empty()) {
            row["tool_definitions"] = tools
                .iter()
                .map(|t| {
                    json!({"name": t.name, "description": t.description, "parameters": t.parameters})
                })
                .collect();
        }
        if let Some(context) = item.context.as_ref().filter(|c| !c.is_empty()) {
            row["context"] = json!(context);
        }
        if let Some(expected) = &item.expected_output {
            row["ground_truth"] = json!(expected);
        }
        row
    }
}

#[async_trait]
impl Evaluator for FoundryEvals {
    fn name(&self) -> &str {
        "Microsoft Foundry"
    }

    async fn evaluate(&self, items: &[EvalItem], eval_name: &str) -> Result<EvalResults> {
        let resolved = resolve_default_evaluators(self.evaluators.as_deref(), Some(items));
        let resolved = filter_tool_evaluators(resolved, items)?;

        let rows: Vec<Value> = items.iter().map(|i| self.dataset_row(i)).collect();
        let has = |key: &str| rows.iter().any(|r| r.get(key).is_some());
        let (has_context, has_ground_truth, has_tools) =
            (has("context"), has("ground_truth"), has("tool_definitions"));

        let eval_id = self
            .transport
            .create_eval(json!({
                "name": eval_name,
                "data_source_config": {
                    "type": "custom",
                    "item_schema": build_item_schema(has_context, has_tools, has_ground_truth),
                    "include_sample_schema": true,
                },
                "testing_criteria": build_testing_criteria(&resolved, &self.model, true, has_tools)?,
            }))
            .await?;
        let content: Vec<Value> = rows.into_iter().map(|r| json!({"item": r})).collect();
        let run_id = self
            .transport
            .create_run(
                &eval_id,
                json!({
                    "name": format!("{eval_name} Run"),
                    "data_source": {
                        "type": "jsonl",
                        "source": {"type": "file_content", "content": content},
                    },
                }),
            )
            .await?;
        self.transport
            .poll_eval_run(
                &eval_id,
                &run_id,
                self.poll_interval,
                self.timeout,
                self.name(),
                true,
            )
            .await
    }
}

// ---------------------------------------------------------------------------
// evaluate_traces / evaluate_foundry_target
// ---------------------------------------------------------------------------

/// Options for [`evaluate_traces`]. Mirrors upstream `evaluate_traces`'
/// keyword arguments (minus the connection, which comes from the
/// [`FoundryEvals`]).
#[derive(Debug, Clone)]
pub struct EvaluateTraces {
    evaluators: Option<Vec<EvaluatorSpec>>,
    response_ids: Vec<String>,
    trace_ids: Vec<String>,
    agent_id: Option<String>,
    lookback_hours: u32,
    eval_name: String,
}

impl Default for EvaluateTraces {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluateTraces {
    /// Defaults: the default evaluators, 24 hours of lookback, eval name
    /// `"Agent Framework Trace Eval"`.
    pub fn new() -> Self {
        Self {
            evaluators: None,
            response_ids: Vec::new(),
            trace_ids: Vec::new(),
            agent_id: None,
            lookback_hours: 24,
            eval_name: "Agent Framework Trace Eval".to_string(),
        }
    }

    /// Evaluator names (default relevance, coherence, task adherence).
    pub fn evaluators<I, E>(mut self, evaluators: I) -> Self
    where
        I: IntoIterator<Item = E>,
        E: Into<EvaluatorSpec>,
    {
        self.evaluators = Some(evaluators.into_iter().map(Into::into).collect());
        self
    }

    /// Evaluate specific Responses API responses.
    pub fn response_ids<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.response_ids.extend(ids.into_iter().map(Into::into));
        self
    }

    /// Evaluate specific OTel trace ids from App Insights.
    pub fn trace_ids<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.trace_ids.extend(ids.into_iter().map(Into::into));
        self
    }

    /// Filter traces by agent id (used with the lookback window).
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Hours of trace history to evaluate (default 24).
    pub fn lookback_hours(mut self, hours: u32) -> Self {
        self.lookback_hours = hours;
        self
    }

    /// Display name for the evaluation.
    pub fn eval_name(mut self, name: impl Into<String>) -> Self {
        self.eval_name = name.into();
        self
    }
}

/// Evaluate agent behavior from stored responses or OTel traces. Mirrors
/// upstream's `evaluate_traces`.
///
/// With response ids, Foundry retrieves those Responses API responses
/// (provider label `"foundry"`, as upstream); otherwise it evaluates the
/// given trace ids and/or an agent's traces within the lookback window
/// (provider label `"Microsoft Foundry"`). The connection, judge model and
/// polling settings come from `evals`; its own evaluator list is not used.
///
/// # Errors
/// [`Error::Configuration`] when no response ids, trace ids or agent id are
/// given, or an evaluator name is unknown. Service errors propagate.
pub async fn evaluate_traces(evals: &FoundryEvals, request: EvaluateTraces) -> Result<EvalResults> {
    let evaluators = resolve_default_evaluators(request.evaluators.as_deref(), None);
    let transport = &evals.transport;

    if !request.response_ids.is_empty() {
        let eval_id = transport
            .create_eval(json!({
                "name": request.eval_name,
                "data_source_config": {"type": "azure_ai_source", "scenario": "responses"},
                "testing_criteria": build_testing_criteria(&evaluators, &evals.model, false, false)?,
            }))
            .await?;
        let content: Vec<Value> = request
            .response_ids
            .iter()
            .map(|rid| json!({"item": {"resp_id": rid}}))
            .collect();
        let run_id = transport
            .create_run(
                &eval_id,
                json!({
                    "name": format!("{} Run", request.eval_name),
                    "data_source": {
                        "type": "azure_ai_responses",
                        "item_generation_params": {
                            "type": "response_retrieval",
                            "data_mapping": {"response_id": "{{item.resp_id}}"},
                            "source": {"type": "file_content", "content": content},
                        },
                    },
                }),
            )
            .await?;
        return transport
            .poll_eval_run(
                &eval_id,
                &run_id,
                evals.poll_interval,
                evals.timeout,
                "foundry",
                true,
            )
            .await;
    }

    if request.trace_ids.is_empty() && request.agent_id.is_none() {
        return Err(Error::Configuration(
            "Provide at least one of: response_ids, trace_ids, or agent_id".into(),
        ));
    }

    let mut trace_source = json!({
        "type": "azure_ai_traces",
        "lookback_hours": request.lookback_hours,
    });
    if !request.trace_ids.is_empty() {
        trace_source["trace_ids"] = json!(request.trace_ids);
    }
    if let Some(agent_id) = &request.agent_id {
        trace_source["agent_id"] = json!(agent_id);
    }
    let eval_id = transport
        .create_eval(json!({
            "name": request.eval_name,
            "data_source_config": {"type": "azure_ai_source", "scenario": "traces"},
            "testing_criteria": build_testing_criteria(&evaluators, &evals.model, false, false)?,
        }))
        .await?;
    let run_id = transport
        .create_run(
            &eval_id,
            json!({
                "name": format!("{} Run", request.eval_name),
                "data_source": trace_source,
            }),
        )
        .await?;
    transport
        .poll_eval_run(
            &eval_id,
            &run_id,
            evals.poll_interval,
            evals.timeout,
            "Microsoft Foundry",
            true,
        )
        .await
}

/// Options for [`evaluate_foundry_target`]. Mirrors upstream
/// `evaluate_foundry_target`'s keyword arguments (minus the connection).
#[derive(Debug, Clone)]
pub struct EvaluateFoundryTarget {
    target: Value,
    test_queries: Vec<String>,
    evaluators: Option<Vec<EvaluatorSpec>>,
    eval_name: String,
}

impl EvaluateFoundryTarget {
    /// Evaluate `target` (e.g. `{"type": "azure_ai_agent", "name":
    /// "my-agent"}`) over `test_queries`; eval name defaults to
    /// `"Agent Framework Target Eval"`.
    pub fn new<I, S>(target: Value, test_queries: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            target,
            test_queries: test_queries.into_iter().map(Into::into).collect(),
            evaluators: None,
            eval_name: "Agent Framework Target Eval".to_string(),
        }
    }

    /// Evaluator names (default relevance, coherence, task adherence).
    pub fn evaluators<I, E>(mut self, evaluators: I) -> Self
    where
        I: IntoIterator<Item = E>,
        E: Into<EvaluatorSpec>,
    {
        self.evaluators = Some(evaluators.into_iter().map(Into::into).collect());
        self
    }

    /// Display name for the evaluation.
    pub fn eval_name(mut self, name: impl Into<String>) -> Self {
        self.eval_name = name.into();
        self
    }
}

/// Evaluate a Foundry-registered agent or model deployment: Foundry invokes
/// the target with each query, captures the output and evaluates it. Mirrors
/// upstream's `evaluate_foundry_target` — for scheduled evals, red teaming
/// and CI quality gates.
///
/// # Errors
/// [`Error::Configuration`] when the target has no `type` key or an
/// evaluator name is unknown. Service errors propagate.
pub async fn evaluate_foundry_target(
    evals: &FoundryEvals,
    request: EvaluateFoundryTarget,
) -> Result<EvalResults> {
    if request.target.get("type").is_none() {
        return Err(Error::Configuration(
            "target dict must include a 'type' key (e.g., 'azure_ai_agent').".into(),
        ));
    }
    let evaluators = resolve_default_evaluators(request.evaluators.as_deref(), None);
    let transport = &evals.transport;
    let eval_id = transport
        .create_eval(json!({
            "name": request.eval_name,
            "data_source_config": {"type": "azure_ai_source", "scenario": "target_completions"},
            "testing_criteria": build_testing_criteria(&evaluators, &evals.model, false, false)?,
        }))
        .await?;
    let content: Vec<Value> = request
        .test_queries
        .iter()
        .map(|q| json!({"item": {"query": q}}))
        .collect();
    let run_id = transport
        .create_run(
            &eval_id,
            json!({
                "name": format!("{} Run", request.eval_name),
                "data_source": {
                    "type": "azure_ai_target_completions",
                    "target": request.target,
                    "source": {"type": "file_content", "content": content},
                },
            }),
        )
        .await?;
    transport
        .poll_eval_run(
            &eval_id,
            &run_id,
            evals.poll_interval,
            evals.timeout,
            "Microsoft Foundry",
            true,
        )
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::FunctionTool;
    use agent_framework_core::types::{
        DataContent, FunctionCallContent, FunctionResultContent, UriContent,
    };
    use std::collections::HashMap;

    fn tool_item() -> EvalItem {
        EvalItem::new(vec![Message::user("q"), Message::assistant("a")]).with_tools([
            FunctionTool::new(
                "get_weather",
                "Get weather",
                json!({"type": "object"}),
                |_| async { Ok(Value::Null) },
            )
            .into_definition(),
        ])
    }

    #[test]
    fn resolve_evaluator_names() {
        assert_eq!(resolve_evaluator("relevance").unwrap(), "builtin.relevance");
        assert_eq!(
            resolve_evaluator("builtin.relevance").unwrap(),
            "builtin.relevance"
        );
        assert_eq!(
            resolve_evaluator("builtin.brand_new").unwrap(),
            "builtin.brand_new"
        );
        let err = resolve_evaluator("nope").unwrap_err().to_string();
        assert!(err.contains("Unknown evaluator 'nope'. Available: ['coherence', 'fluency'"));
        for (short, _) in BUILTIN_EVALUATORS {
            assert!(resolve_evaluator(short).is_ok());
        }
    }

    #[test]
    fn evaluator_sets_are_consistent() {
        let qualified: Vec<&str> = BUILTIN_EVALUATORS.iter().map(|(_, q)| *q).collect();
        for set in [
            AGENT_EVALUATORS,
            TOOL_EVALUATORS,
            EXTRA_TOOL_DEFINITION_EVALUATORS,
            GROUND_TRUTH_EVALUATORS,
        ] {
            for name in set {
                assert!(qualified.contains(name), "{name}");
            }
        }
        for name in TOOL_EVALUATORS {
            assert!(AGENT_EVALUATORS.contains(name));
        }
        for name in DEFAULT_EVALUATORS.iter().chain(DEFAULT_TOOL_EVALUATORS) {
            assert!(resolve_evaluator(name).is_ok());
        }
        for c in [
            FoundryEvals::INTENT_RESOLUTION,
            FoundryEvals::TOOL_CALL_SUCCESS,
            FoundryEvals::RESPONSE_COMPLETENESS,
            FoundryEvals::HATE_UNFAIRNESS,
        ] {
            assert!(resolve_evaluator(c).is_ok());
        }
    }

    fn fc(call_id: &str, name: &str, args: Option<FunctionArguments>) -> Content {
        Content::FunctionCall(FunctionCallContent::new(call_id, name, args))
    }

    #[test]
    fn convert_message_shapes() {
        assert_eq!(
            convert_message(&Message::user("Hello, world!")),
            vec![json!({"role": "user", "content": [{"type": "text", "text": "Hello, world!"}]})]
        );
        let call = Message::with_contents(
            "assistant",
            vec![fc(
                "call_1",
                "get_weather",
                Some(FunctionArguments::Raw("{\"location\": \"Seattle\"}".into())),
            )],
        );
        let tc = &convert_message(&call)[0]["content"][0];
        assert_eq!(tc["type"], "tool_call");
        assert_eq!(tc["tool_call_id"], "call_1");
        assert_eq!(tc["arguments"], json!({"location": "Seattle"}));
        let zero = Message::with_contents("assistant", vec![fc("call_3", "summary", None)]);
        assert_eq!(
            convert_message(&zero)[0]["content"][0]["arguments"],
            json!({})
        );
        let mut args = HashMap::new();
        args.insert("query".to_string(), json!("flights"));
        let mixed = Message::with_contents(
            "assistant",
            vec![
                Content::text("Let me check that."),
                fc("c2", "search", Some(FunctionArguments::Object(args))),
            ],
        );
        let out = convert_message(&mixed);
        assert_eq!(
            out[0]["content"][0],
            json!({"type": "text", "text": "Let me check that."})
        );
        assert_eq!(
            out[0]["content"][1]["arguments"],
            json!({"query": "flights"})
        );
    }

    #[test]
    fn convert_tool_results_and_images() {
        let results = Message::with_contents(
            "tool",
            vec![
                Content::FunctionResult(FunctionResultContent::new(
                    "call_1",
                    Some(json!("72°F, sunny")),
                )),
                Content::FunctionResult(FunctionResultContent::new(
                    "call_2",
                    Some(json!({"temp": 72})),
                )),
            ],
        );
        let out = convert_message(&results);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["tool_call_id"], "call_1");
        assert_eq!(
            out[0]["content"],
            json!([{"type": "tool_result", "tool_result": "72°F, sunny"}])
        );
        assert_eq!(out[1]["content"][0]["tool_result"], json!({"temp": 72}));

        assert_eq!(
            convert_message(&Message::with_contents("user", vec![])),
            vec![json!({"role": "user", "content": [{"type": "text", "text": ""}]})]
        );
        let data = Message::with_contents(
            "user",
            vec![Content::Data(DataContent::from_bytes(
                b"\x89PNG",
                "image/png",
            ))],
        );
        let part = &convert_message(&data)[0]["content"][0];
        assert!(part["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
        assert_eq!(part["detail"], "auto");
        let uri = |media: &str| {
            Message::with_contents(
                "user",
                vec![Content::Uri(UriContent {
                    uri: "https://example.com/p.png".into(),
                    media_type: media.into(),
                })],
            )
        };
        assert_eq!(
            convert_message(&uri("image/png"))[0]["content"][0]["detail"],
            "auto"
        );
        assert!(convert_message(&uri(""))[0]["content"][0]
            .get("detail")
            .is_none());
    }

    #[test]
    fn testing_criteria_mappings() {
        let specs = |names: &[&str]| {
            names
                .iter()
                .map(|&n| EvaluatorSpec::from(n))
                .collect::<Vec<_>>()
        };
        let c = build_testing_criteria(&specs(&["relevance", "coherence"]), "gpt-4o", false, false)
            .unwrap();
        assert_eq!(c[0]["evaluator_name"], "builtin.relevance");
        assert_eq!(
            c[0]["initialization_parameters"],
            json!({"deployment_name": "gpt-4o"})
        );
        assert!(c[0].get("data_mapping").is_none());

        let c = build_testing_criteria(&specs(&["relevance", "groundedness"]), "m", true, false)
            .unwrap();
        assert_eq!(
            c[0]["data_mapping"],
            json!({"query": "{{item.query}}", "response": "{{item.response}}"})
        );
        assert_eq!(c[1]["data_mapping"]["context"], "{{item.context}}");

        let c = build_testing_criteria(
            &specs(&["relevance", "builtin.tool_call_accuracy"]),
            "m",
            true,
            true,
        )
        .unwrap();
        assert!(c[0]["data_mapping"].get("tool_definitions").is_none());
        assert_eq!(c[1]["name"], "tool_call_accuracy");
        assert_eq!(c[1]["data_mapping"]["query"], "{{item.query_messages}}");
        assert_eq!(
            c[1]["data_mapping"]["tool_definitions"],
            "{{item.tool_definitions}}"
        );

        let c = build_testing_criteria(
            &specs(&["task_adherence", "task_navigation_efficiency"]),
            "m",
            true,
            true,
        )
        .unwrap();
        for e in &c {
            assert_eq!(e["data_mapping"]["response"], "{{item.response_messages}}");
            assert_eq!(
                e["data_mapping"]["tool_definitions"],
                "{{item.tool_definitions}}"
            );
        }
        let c = build_testing_criteria(&specs(&["similarity"]), "m", true, false).unwrap();
        assert_eq!(
            c[0]["data_mapping"]["ground_truth"],
            "{{item.ground_truth}}"
        );
        assert!(build_testing_criteria(&specs(&["bogus"]), "m", false, false).is_err());
    }

    #[test]
    fn generated_evaluator_refs() {
        let pinned: EvaluatorSpec = GeneratedEvaluatorRef::new("my-rubric", "1").into();
        let c =
            build_testing_criteria(std::slice::from_ref(&pinned), "gpt-4o", true, false).unwrap();
        assert_eq!(
            c[0],
            json!({
                "type": "azure_ai_evaluator",
                "name": "my-rubric",
                "evaluator_name": "my-rubric",
                "evaluator_version": "1",
                "initialization_parameters": {"deployment_name": "gpt-4o"},
                "data_mapping": {"query": "{{item.query_messages}}", "response": "{{item.response_messages}}"},
            })
        );
        let named: EvaluatorSpec = GeneratedEvaluatorRef::new("my-rubric", "2")
            .with_display_name("My Rubric")
            .into();
        let c = build_testing_criteria(&[named], "m", false, false).unwrap();
        assert_eq!(c[0]["name"], "My Rubric");
        let c = build_testing_criteria(std::slice::from_ref(&pinned), "m", true, true).unwrap();
        assert_eq!(
            c[0]["data_mapping"]["tool_definitions"],
            "{{item.tool_definitions}}"
        );
        let c = build_testing_criteria(
            &[GeneratedEvaluatorRef::latest("r").into()],
            "m",
            false,
            false,
        )
        .unwrap();
        assert!(c[0].get("evaluator_version").is_none());
        let mixed = vec!["relevance".into(), pinned, "task_adherence".into()];
        let c = build_testing_criteria(&mixed, "m", true, false).unwrap();
        let names: Vec<&str> = c.iter().map(|e| e["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["relevance", "my-rubric", "task_adherence"]);
    }

    #[test]
    fn item_schema() {
        let s = build_item_schema(false, false, false);
        assert!(s["properties"].get("context").is_none());
        assert_eq!(s["required"], json!(["query", "response"]));
        let s = build_item_schema(true, true, true);
        for key in ["context", "tool_definitions", "ground_truth"] {
            assert!(s["properties"].get(key).is_some(), "{key}");
        }
    }

    #[test]
    fn default_and_filtered_evaluators() {
        let plain = vec![EvalItem::new(vec![Message::user("q")])];
        let names =
            |v: &[EvaluatorSpec]| v.iter().map(|s| s.label().to_string()).collect::<Vec<_>>();
        assert_eq!(
            names(&resolve_default_evaluators(None, Some(&plain))),
            ["relevance", "coherence", "task_adherence"]
        );
        let tools = vec![tool_item()];
        assert_eq!(
            names(&resolve_default_evaluators(None, Some(&tools)))
                .last()
                .unwrap(),
            "tool_call_accuracy"
        );
        let explicit = vec![EvaluatorSpec::from("fluency")];
        assert_eq!(
            names(&resolve_default_evaluators(Some(&explicit), Some(&tools))),
            ["fluency"]
        );

        let both = vec!["relevance".into(), "tool_call_accuracy".into()];
        assert_eq!(
            names(&filter_tool_evaluators(both.clone(), &tools).unwrap()).len(),
            2
        );
        assert_eq!(
            names(&filter_tool_evaluators(both, &plain).unwrap()),
            ["relevance"]
        );
        let err = filter_tool_evaluators(vec!["tool_selection".into()], &plain)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("All requested evaluators ['tool_selection'] require tool definitions")
        );
        let rubric: Vec<EvaluatorSpec> = vec![
            GeneratedEvaluatorRef::new("r", "1").into(),
            "tool_selection".into(),
        ];
        assert_eq!(
            names(&filter_tool_evaluators(rubric, &plain).unwrap()),
            ["r"]
        );
        assert!(
            filter_tool_evaluators(vec![GeneratedEvaluatorRef::latest("r").into()], &plain).is_ok()
        );
    }

    #[test]
    fn rubric_score_extraction() {
        let sample = json!({"properties": {"rubric_scores": [
            {"id": "policy", "score": 4, "applicable": true, "weight": 1, "reason": "ok"},
            {"id": "safety", "score": null, "applicable": false, "weight": 1, "reason": "n/a"},
            {"id": "bad-no-weight", "score": 2, "applicable": true, "reason": "x"},
        ]}});
        let dims = extract_rubric_scores(&sample).unwrap();
        assert_eq!(dims.len(), 2);
        assert_eq!(
            dims[0],
            RubricScore {
                id: "policy".into(),
                score: Some(4),
                applicable: true,
                weight: 1,
                reason: "ok".into()
            }
        );
        assert_eq!(dims[1].score, None);
        assert!(!dims[1].applicable);

        let top = json!({"rubric_scores": [{"id": "a", "score": 3, "applicable": true, "weight": 1, "reason": "r"}]});
        assert_eq!(extract_rubric_scores(&top).unwrap()[0].id, "a");
        let canonical = json!({"properties": {
            "dimension_scores": [{"id": "intent", "score": 5, "applicable": true, "weight": 9, "reason": "x"}],
            "rubric_scores": [{"id": "legacy", "score": 1, "applicable": true, "weight": 1}],
        }});
        let dims = extract_rubric_scores(&canonical).unwrap();
        assert_eq!((dims[0].id.as_str(), dims[0].weight), ("intent", 9));
        assert!(extract_rubric_scores(&Value::Null).is_none());
        assert!(extract_rubric_scores(&json!({})).is_none());
        assert!(extract_rubric_scores(&json!({"properties": {}})).is_none());
    }

    #[test]
    fn output_item_parsing() {
        let oi = json!({
            "id": "oi_abc123",
            "status": "pass",
            "results": [{"name": "relevance", "score": 0.85, "passed": true, "sample": null}],
            "sample": {
                "error": {"code": "", "message": ""},
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cached_tokens": 0},
                "input": [{"role": "user", "content": "What is the weather?"}],
                "output": [{"role": "assistant", "content": "It is sunny."}],
            },
            "datasource_item": {"resp_id": "resp_xyz"},
        });
        let item = parse_output_item(&oi).unwrap();
        assert!(item.is_passed());
        assert_eq!(item.scores[0].score, 0.85);
        assert_eq!(item.scores[0].passed, Some(true));
        assert!(item.scores[0].dimensions.is_none());
        assert_eq!(item.response_id.as_deref(), Some("resp_xyz"));
        assert_eq!(item.input_text.as_deref(), Some("What is the weather?"));
        assert_eq!(item.output_text.as_deref(), Some("It is sunny."));
        assert_eq!(item.token_usage.as_ref().unwrap()["total_tokens"], 150);
        assert!(item.error_code.is_none());

        let errored = json!({
            "id": "oi_err1", "status": "error", "results": [],
            "sample": {"error": {"code": "QueryExtractionError", "message": "Query list cannot be empty"},
                       "usage": null, "input": [], "output": []},
            "datasource_item": {},
        });
        let item = parse_output_item(&errored).unwrap();
        assert!(item.is_error());
        assert_eq!(item.error_code.as_deref(), Some("QueryExtractionError"));
        assert_eq!(
            item.error_message.as_deref(),
            Some("Query list cannot be empty")
        );
        assert!(item.input_text.is_none());
        assert!(parse_output_item(&json!({"status": "pass"})).is_none());
    }

    #[test]
    fn run_metadata_extraction() {
        let run = json!({
            "status": "failed",
            "error": {"code": "x", "message": "Model deployment unavailable"},
            "result_counts": {"passed": 1, "failed": 2, "errored": 3, "total": 6},
            "per_testing_criteria_results": [{"testing_criteria": "relevance", "passed": 1, "failed": 2}, {"testing_criteria": "", "passed": 9}],
        });
        assert_eq!(
            run_error(&run).as_deref(),
            Some("Model deployment unavailable")
        );
        assert_eq!(
            run_error(&json!({"error": "plain"})).as_deref(),
            Some("plain")
        );
        assert_eq!(run_error(&json!({"error": null})), None);
        let counts = extract_result_counts(&run).unwrap();
        assert_eq!(
            (counts.passed, counts.failed, counts.errored, counts.total),
            (1, 2, 3, Some(6))
        );
        assert!(extract_result_counts(&json!({"result_counts": null})).is_none());
        let per = extract_per_evaluator(&run);
        assert_eq!(per.len(), 1);
        assert_eq!(per["relevance"], ResultCounts::new(1, 2, 0));
    }

    #[test]
    fn env_configuration() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(key, _)| *key == k)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert!(FoundryEvals::from_env_vars(env(&[])).is_err());
        let e = FoundryEvals::from_env_vars(env(&[
            ("FOUNDRY_PROJECT_ENDPOINT", "https://x/"),
            ("FOUNDRY_API_KEY", "k"),
        ]))
        .unwrap();
        assert_eq!(e.model(), "gpt-4o");
        assert_eq!(e.transport.base_url, "https://x/openai/v1");
        let e = FoundryEvals::from_env_vars(env(&[
            ("FOUNDRY_ENDPOINT", "https://y"),
            ("FOUNDRY_MODEL", "m"),
            ("FOUNDRY_API_KEY", "k"),
        ]))
        .unwrap();
        assert_eq!(e.model(), "m");
        assert_eq!(e.name(), "Microsoft Foundry");
        assert!(e.evaluators().is_none());
        let e = e.with_evaluators(["relevance", "coherence"]);
        assert_eq!(e.evaluators().unwrap().len(), 2);
    }

    #[test]
    fn dataset_rows() {
        let evals = FoundryEvals::new("https://x", "m", "k");
        let item = tool_item().with_context("doc").with_expected_output("gt");
        let row = evals.dataset_row(&item);
        assert_eq!(row["query"], "q");
        assert_eq!(row["response"], "a");
        assert_eq!(row["query_messages"][0]["role"], "user");
        assert_eq!(row["tool_definitions"][0]["name"], "get_weather");
        assert_eq!(row["context"], "doc");
        assert_eq!(row["ground_truth"], "gt");
        let bare = evals.dataset_row(&EvalItem::new(vec![Message::user("q")]).with_context(""));
        assert!(
            bare.get("context").is_none()
                && bare.get("tool_definitions").is_none()
                && bare.get("ground_truth").is_none()
        );

        // The evaluator's split applies only when the item has none.
        let convo = vec![
            Message::user("A"),
            Message::assistant("B"),
            Message::user("C"),
            Message::assistant("D"),
        ];
        let full = FoundryEvals::new("https://x", "m", "k")
            .with_conversation_split(ConversationSplit::Full);
        assert_eq!(
            full.dataset_row(&EvalItem::new(convo.clone()))["response"],
            "B D"
        );
        let own = EvalItem::new(convo).with_split_strategy(ConversationSplit::LastTurn);
        assert_eq!(full.dataset_row(&own)["response"], "D");
    }

    #[test]
    fn query_encoding() {
        assert_eq!(encode_query("a b/c"), "a%20b%2Fc");
        let t = FoundryEvals::new("https://x", "m", "k")
            .with_api_version("2025-01-01")
            .transport;
        assert_eq!(
            t.url("/evals", &[("after", "o 1".into())]),
            "https://x/openai/v1/evals?after=o%201&api-version=2025-01-01"
        );
    }
}
