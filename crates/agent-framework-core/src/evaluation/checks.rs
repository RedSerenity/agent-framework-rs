//! Local, API-free checks and the [`LocalEvaluator`] that runs them.
//!
//! Mirrors upstream's "Local evaluation checks", "Function evaluator" and
//! "LocalEvaluator" regions of `_evaluation.py`.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::pyrepr;
use super::{
    EvalItem, EvalItemResult, EvalItemStatus, EvalResults, EvalRunStatus, EvalScoreResult,
    Evaluator, ExpectedToolCall, ResultCounts,
};
use crate::error::{Error, Result};
use crate::tools::{BoxFuture, ToolDefinition};
use crate::types::{Content, FunctionArguments, Message};

/// The result of one check on one item. Mirrors upstream's `CheckResult`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckResult {
    /// Whether the check passed.
    pub passed: bool,
    /// Human-readable explanation.
    pub reason: String,
    /// Name of the check that produced this result.
    pub check_name: String,
}

impl CheckResult {
    /// Build a check result.
    pub fn new(passed: bool, reason: impl Into<String>, check_name: impl Into<String>) -> Self {
        Self {
            passed,
            reason: reason.into(),
            check_name: check_name.into(),
        }
    }
}

/// A check run against each [`EvalItem`] by a [`LocalEvaluator`].
///
/// Mirrors upstream's `EvalCheck` type alias (a sync or async callable
/// `EvalItem -> CheckResult`). Any `Fn(&EvalItem) -> CheckResult` closure or
/// function is a check — including [`tool_calls_present`] and
/// [`tool_call_args_match`] themselves. For async or fallible checks, or
/// checks that return a score, use [`evaluator`] / [`async_evaluator`] or
/// implement this trait directly. An `Err` aborts the whole evaluation, as
/// an exception raised by an upstream check does.
#[async_trait]
pub trait EvalCheck: Send + Sync {
    /// Run the check against one item.
    async fn check(&self, item: &EvalItem) -> Result<CheckResult>;
}

#[async_trait]
impl<F> EvalCheck for F
where
    F: Fn(&EvalItem) -> CheckResult + Send + Sync,
{
    async fn check(&self, item: &EvalItem) -> Result<CheckResult> {
        Ok(self(item))
    }
}

// ---------------------------------------------------------------------------
// keyword_check
// ---------------------------------------------------------------------------

/// Check that the response contains every keyword. Built by
/// [`keyword_check`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeywordCheck {
    keywords: Vec<String>,
    case_sensitive: bool,
}

impl KeywordCheck {
    /// Match case-sensitively (default: case-insensitive).
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }
}

/// Check that the response contains all `keywords`. Mirrors upstream's
/// `keyword_check(*keywords, case_sensitive=False)`; check name
/// `"keyword_check"`.
///
/// ```
/// use agent_framework_core::evaluation::keyword_check;
/// let check = keyword_check(["weather", "temperature"]).case_sensitive(false);
/// ```
pub fn keyword_check<I, S>(keywords: I) -> KeywordCheck
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    KeywordCheck {
        keywords: keywords.into_iter().map(Into::into).collect(),
        case_sensitive: false,
    }
}

#[async_trait]
impl EvalCheck for KeywordCheck {
    async fn check(&self, item: &EvalItem) -> Result<CheckResult> {
        let response = item.response();
        let text = if self.case_sensitive {
            response
        } else {
            response.to_lowercase()
        };
        let missing: Vec<&String> = self
            .keywords
            .iter()
            .filter(|k| {
                let needle = if self.case_sensitive {
                    (*k).clone()
                } else {
                    k.to_lowercase()
                };
                !text.contains(&needle)
            })
            .collect();
        Ok(if missing.is_empty() {
            CheckResult::new(true, "All keywords found", "keyword_check")
        } else {
            CheckResult::new(
                false,
                format!("Missing keywords: {}", pyrepr::string_list(&missing)),
                "keyword_check",
            )
        })
    }
}

// ---------------------------------------------------------------------------
// tool_called_check
// ---------------------------------------------------------------------------

/// Whether [`tool_called_check`] requires every tool or any one of them.
/// Upstream: `mode: Literal["all", "any"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCalledMode {
    /// Every named tool must be called (default).
    #[default]
    All,
    /// At least one named tool must be called.
    Any,
}

/// Check that specific tools were called. Built by [`tool_called_check`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCalledCheck {
    tool_names: Vec<String>,
    mode: ToolCalledMode,
}

impl ToolCalledCheck {
    /// Set the matching mode (default [`ToolCalledMode::All`]).
    pub fn mode(mut self, mode: ToolCalledMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Check that the named tools were called anywhere in the conversation.
/// Mirrors upstream's `tool_called_check(*tool_names, mode="all")`; check
/// name `"tool_called"`.
pub fn tool_called_check<I, S>(tool_names: I) -> ToolCalledCheck
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    ToolCalledCheck {
        tool_names: tool_names.into_iter().map(Into::into).collect(),
        mode: ToolCalledMode::All,
    }
}

fn function_calls(
    conversation: &[Message],
) -> impl Iterator<Item = &crate::types::FunctionCallContent> {
    conversation.iter().flat_map(|m| {
        m.contents.iter().filter_map(|c| match c {
            Content::FunctionCall(fc) if !fc.name.is_empty() => Some(fc),
            _ => None,
        })
    })
}

#[async_trait]
impl EvalCheck for ToolCalledCheck {
    async fn check(&self, item: &EvalItem) -> Result<CheckResult> {
        const NAME: &str = "tool_called";
        let expected: HashSet<&str> = self.tool_names.iter().map(String::as_str).collect();
        let mut called: BTreeSet<String> = BTreeSet::new();
        let sorted = |set: &BTreeSet<String>| {
            pyrepr::string_list(&set.iter().map(String::as_str).collect::<Vec<_>>())
        };
        for fc in function_calls(&item.conversation) {
            called.insert(fc.name.clone());
            match self.mode {
                ToolCalledMode::All => {
                    if expected.iter().all(|e| called.contains(*e)) {
                        return Ok(CheckResult::new(
                            true,
                            format!("All expected tools called: {}", sorted(&called)),
                            NAME,
                        ));
                    }
                }
                ToolCalledMode::Any => {
                    let found: BTreeSet<String> = called
                        .iter()
                        .filter(|c| expected.contains(c.as_str()))
                        .cloned()
                        .collect();
                    if !found.is_empty() {
                        return Ok(CheckResult::new(
                            true,
                            format!("Expected tool found: {}", sorted(&found)),
                            NAME,
                        ));
                    }
                }
            }
        }
        Ok(match self.mode {
            ToolCalledMode::All => {
                let missing: Vec<&String> = self
                    .tool_names
                    .iter()
                    .filter(|t| !called.contains(*t))
                    .collect();
                if missing.is_empty() {
                    CheckResult::new(
                        true,
                        format!("All expected tools called: {}", sorted(&called)),
                        NAME,
                    )
                } else {
                    CheckResult::new(
                        false,
                        format!(
                            "Expected tools not called: {} (called: {})",
                            pyrepr::string_list(&missing),
                            sorted(&called)
                        ),
                        NAME,
                    )
                }
            }
            ToolCalledMode::Any => CheckResult::new(
                false,
                format!(
                    "None of expected tools called: {} (called: {})",
                    pyrepr::string_list(&self.tool_names),
                    sorted(&called)
                ),
                NAME,
            ),
        })
    }
}

// ---------------------------------------------------------------------------
// Expected-tool-call checks
// ---------------------------------------------------------------------------

/// `(name, arguments)` for every function call in the conversation. Mirrors
/// upstream's `_extract_tool_calls`: absent or blank arguments are `{}`, a
/// JSON-object string is parsed, and anything unparseable (or a non-object)
/// is `None`. Object keys are sorted for deterministic reasons.
fn extract_tool_calls(item: &EvalItem) -> Vec<(String, Option<Map<String, Value>>)> {
    function_calls(&item.conversation)
        .map(|fc| {
            let args = match &fc.arguments {
                None => Some(Map::new()),
                Some(FunctionArguments::Object(map)) => {
                    let mut entries: Vec<(&String, &Value)> = map.iter().collect();
                    entries.sort_by(|a, b| a.0.cmp(b.0));
                    Some(
                        entries
                            .into_iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    )
                }
                Some(FunctionArguments::Raw(raw)) => {
                    let stripped = raw.trim();
                    if stripped.is_empty() {
                        Some(Map::new())
                    } else {
                        match serde_json::from_str::<Value>(stripped) {
                            Ok(Value::Object(map)) => Some(map),
                            _ => None,
                        }
                    }
                }
            };
            (fc.name.clone(), args)
        })
        .collect()
}

/// Check that every expected tool was called at least once (unordered,
/// extras OK, arguments not checked), using [`EvalItem::expected_tool_calls`].
/// Mirrors upstream's `tool_calls_present`; check name
/// `"tool_calls_present"`. Passes when no calls are expected.
pub fn tool_calls_present(item: &EvalItem) -> CheckResult {
    const NAME: &str = "tool_calls_present";
    let expected: &[ExpectedToolCall] = item.expected_tool_calls.as_deref().unwrap_or(&[]);
    if expected.is_empty() {
        return CheckResult::new(true, "No expected tool calls specified.", NAME);
    }
    let actual: BTreeSet<String> = extract_tool_calls(item)
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    let actual_repr = pyrepr::string_list(&actual.iter().map(String::as_str).collect::<Vec<_>>());
    let (found, missing): (Vec<&str>, Vec<&str>) = expected
        .iter()
        .map(|e| e.name.as_str())
        .partition(|n| actual.contains(*n));
    if missing.is_empty() {
        CheckResult::new(
            true,
            format!(
                "All expected tools called: {} (called: {actual_repr})",
                pyrepr::string_list(&found)
            ),
            NAME,
        )
    } else {
        CheckResult::new(
            false,
            format!(
                "Missing tool calls: {} (called: {actual_repr})",
                pyrepr::string_list(&missing)
            ),
            NAME,
        )
    }
}

/// Check that expected tool calls match on name and arguments. Mirrors
/// upstream's `tool_call_args_match`; check name `"tool_call_args_match"`.
///
/// For each expected call, some actual call with that name must exist; when
/// [`ExpectedToolCall::arguments`] is set, one of those calls must contain
/// every expected key with an equal value (a subset match, compared with
/// Python `==` semantics so `1 == 1.0`). Passes only when every expected call
/// matches; passes trivially when none are expected.
pub fn tool_call_args_match(item: &EvalItem) -> CheckResult {
    const NAME: &str = "tool_call_args_match";
    let expected: &[ExpectedToolCall] = item.expected_tool_calls.as_deref().unwrap_or(&[]);
    if expected.is_empty() {
        return CheckResult::new(true, "No expected tool calls specified.", NAME);
    }
    let actual_calls = extract_tool_calls(item);
    let mut matched = 0usize;
    let mut details: Vec<String> = Vec::new();

    for exp in expected {
        let matching: Vec<&Option<Map<String, Value>>> = actual_calls
            .iter()
            .filter(|(n, _)| *n == exp.name)
            .map(|(_, a)| a)
            .collect();
        if matching.is_empty() {
            details.push(format!("  {}: not called", exp.name));
            continue;
        }
        let Some(expected_args) = &exp.arguments else {
            matched += 1;
            details.push(format!("  {}: called (args not checked)", exp.name));
            continue;
        };
        let found = matching.iter().any(|actual| {
            actual.as_ref().is_some_and(|actual| {
                expected_args
                    .iter()
                    .all(|(k, v)| actual.get(k).is_some_and(|a| pyrepr::py_eq(a, v)))
            })
        });
        if found {
            matched += 1;
            details.push(format!("  {}: args match", exp.name));
        } else {
            let actual_list: Vec<String> = matching
                .iter()
                .map(|a| match a {
                    Some(map) => pyrepr::value(&Value::Object(map.clone())),
                    None => "None".to_string(),
                })
                .collect();
            details.push(format!(
                "  {}: args mismatch (actual: [{}])",
                exp.name,
                actual_list.join(", ")
            ));
        }
    }

    let passed = matched == expected.len();
    let reason = format!(
        "Tool call args match: {matched}/{}\n{}",
        expected.len(),
        details.join("\n")
    );
    CheckResult::new(passed, reason, NAME)
}

// ---------------------------------------------------------------------------
// Function evaluators
// ---------------------------------------------------------------------------

/// The item fields handed to a function evaluator. Upstream's `@evaluator`
/// passes these by parameter name; Rust passes them all and the function
/// reads what it needs.
#[derive(Debug, Clone)]
pub struct EvalFields {
    /// [`EvalItem::query`].
    pub query: String,
    /// [`EvalItem::response`].
    pub response: String,
    /// [`EvalItem::expected_output`], or `""` when unset.
    pub expected_output: String,
    /// [`EvalItem::expected_tool_calls`], or empty when unset.
    pub expected_tool_calls: Vec<ExpectedToolCall>,
    /// [`EvalItem::conversation`].
    pub conversation: Vec<Message>,
    /// [`EvalItem::tools`].
    pub tools: Option<Vec<ToolDefinition>>,
    /// [`EvalItem::context`].
    pub context: Option<String>,
}

impl EvalFields {
    /// Extract the fields from an item. Mirrors upstream's
    /// `_resolve_function_args` field map.
    pub fn from_item(item: &EvalItem) -> Self {
        Self {
            query: item.query(),
            response: item.response(),
            expected_output: item.expected_output.clone().unwrap_or_default(),
            expected_tool_calls: item.expected_tool_calls.clone().unwrap_or_default(),
            conversation: item.conversation.clone(),
            tools: item.tools.clone(),
            context: item.context.clone(),
        }
    }
}

/// A function evaluator's return value, before coercion to a
/// [`CheckResult`]. Mirrors the return types upstream's `_coerce_result`
/// accepts.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalOutcome {
    /// Pass/fail directly.
    Bool(bool),
    /// A score; `>= 0.5` passes.
    Score(f64),
    /// A dict with a `score` key (plus optional `passed`, `threshold`,
    /// `reason`) or a `passed` key (plus optional `reason`).
    Map(Map<String, Value>),
    /// A ready-made result, returned as-is.
    Check(CheckResult),
    /// Any other value; coercion fails naming this Python-style type name.
    Unsupported(String),
}

/// Conversion of a function evaluator's return value into an
/// [`EvalOutcome`]: implemented for `bool`, the numeric types,
/// [`CheckResult`], JSON objects/values, and `Result` of any of these (an
/// `Err` aborts the evaluation).
pub trait IntoEvalOutcome {
    /// Convert into an outcome.
    fn into_eval_outcome(self) -> Result<EvalOutcome>;
}

impl IntoEvalOutcome for EvalOutcome {
    fn into_eval_outcome(self) -> Result<EvalOutcome> {
        Ok(self)
    }
}

impl IntoEvalOutcome for bool {
    fn into_eval_outcome(self) -> Result<EvalOutcome> {
        Ok(EvalOutcome::Bool(self))
    }
}

macro_rules! numeric_outcome {
    ($($t:ty),*) => {$(
        impl IntoEvalOutcome for $t {
            fn into_eval_outcome(self) -> Result<EvalOutcome> {
                Ok(EvalOutcome::Score(self as f64))
            }
        }
    )*};
}
numeric_outcome!(f64, f32, i32, i64, u32, u64, usize);

impl IntoEvalOutcome for CheckResult {
    fn into_eval_outcome(self) -> Result<EvalOutcome> {
        Ok(EvalOutcome::Check(self))
    }
}

impl IntoEvalOutcome for Map<String, Value> {
    fn into_eval_outcome(self) -> Result<EvalOutcome> {
        Ok(EvalOutcome::Map(self))
    }
}

impl IntoEvalOutcome for Value {
    fn into_eval_outcome(self) -> Result<EvalOutcome> {
        Ok(match self {
            Value::Bool(b) => EvalOutcome::Bool(b),
            Value::Number(n) => EvalOutcome::Score(n.as_f64().unwrap_or(f64::NAN)),
            Value::Object(map) => EvalOutcome::Map(map),
            Value::Null => EvalOutcome::Unsupported("NoneType".into()),
            Value::String(_) => EvalOutcome::Unsupported("str".into()),
            Value::Array(_) => EvalOutcome::Unsupported("list".into()),
        })
    }
}

impl<T: IntoEvalOutcome> IntoEvalOutcome for Result<T> {
    fn into_eval_outcome(self) -> Result<EvalOutcome> {
        self?.into_eval_outcome()
    }
}

/// Python truthiness of a JSON value.
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

/// Python `float(v)` of a JSON value.
fn py_float(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Python `str(v)` of a JSON value.
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => pyrepr::value(other),
    }
}

/// Convert a function evaluator's outcome into a [`CheckResult`]. Mirrors
/// upstream's `_coerce_result`:
///
/// - `bool` → pass/fail with reason `"passed"` / `"failed"`;
/// - a number → passes when `>= 0.5`, reason `score=<value:.3>`;
/// - a dict with `score` → an explicit `passed` wins, else
///   `score >= threshold` (default 0.5); reason from `reason` or
///   `score=<value:.3>`;
/// - a dict with only `passed` (bool or int) → that, reason from `reason` or
///   `"passed"` / `"failed"`;
/// - anything else → an error.
pub(crate) fn coerce_result(outcome: EvalOutcome, check_name: &str) -> Result<CheckResult> {
    match outcome {
        EvalOutcome::Check(c) => Ok(c),
        EvalOutcome::Bool(b) => Ok(CheckResult::new(
            b,
            if b { "passed" } else { "failed" },
            check_name,
        )),
        EvalOutcome::Score(score) => Ok(CheckResult::new(
            score >= 0.5,
            format!("score={score:.3}"),
            check_name,
        )),
        EvalOutcome::Map(d) => {
            if let Some(raw_score) = d.get("score") {
                let score = py_float(raw_score).ok_or_else(|| {
                    Error::Other(format!(
                        "Function evaluator '{check_name}' returned dict with non-numeric 'score' value: {}",
                        pyrepr::value(raw_score)
                    ))
                })?;
                let passed = match d.get("passed") {
                    Some(p) => truthy(p),
                    None => {
                        let threshold = match d.get("threshold") {
                            Some(t) => py_float(t).ok_or_else(|| {
                                Error::Other(format!(
                                    "Function evaluator '{check_name}' returned dict with non-numeric 'threshold' value: {}",
                                    pyrepr::value(t)
                                ))
                            })?,
                            None => 0.5,
                        };
                        score >= threshold
                    }
                };
                let reason = d
                    .get("reason")
                    .map(py_str)
                    .unwrap_or_else(|| format!("score={score:.3}"));
                return Ok(CheckResult::new(passed, reason, check_name));
            }
            if let Some(passed_val) = d.get("passed") {
                let is_int = matches!(passed_val, Value::Number(n) if n.is_i64() || n.is_u64());
                if !matches!(passed_val, Value::Bool(_)) && !is_int {
                    return Err(Error::Other(format!(
                        "Function evaluator '{check_name}' returned dict with non-boolean 'passed' value: {}",
                        pyrepr::value(passed_val)
                    )));
                }
                let passed = truthy(passed_val);
                let reason = d
                    .get("reason")
                    .map(py_str)
                    .unwrap_or_else(|| if passed { "passed" } else { "failed" }.to_string());
                return Ok(CheckResult::new(passed, reason, check_name));
            }
            Err(unsupported(check_name, "dict"))
        }
        EvalOutcome::Unsupported(type_name) => Err(unsupported(check_name, &type_name)),
    }
}

fn unsupported(check_name: &str, type_name: &str) -> Error {
    Error::Other(format!(
        "Function evaluator '{check_name}' returned unsupported type {type_name}. \
         Expected bool, float, dict, or CheckResult."
    ))
}

type EvalFn = Arc<dyn Fn(EvalFields) -> BoxFuture<Result<EvalOutcome>> + Send + Sync>;

/// A plain function wrapped as an [`EvalCheck`]. Built by [`evaluator`] /
/// [`async_evaluator`]; mirrors what upstream's `@evaluator` decorator
/// returns.
#[derive(Clone)]
pub struct FunctionEvaluator {
    name: String,
    func: EvalFn,
}

impl std::fmt::Debug for FunctionEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionEvaluator")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl FunctionEvaluator {
    /// The check name reported in results.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl EvalCheck for FunctionEvaluator {
    async fn check(&self, item: &EvalItem) -> Result<CheckResult> {
        let outcome = (self.func)(EvalFields::from_item(item)).await?;
        coerce_result(outcome, &self.name)
    }
}

/// Wrap a synchronous function as a named [`EvalCheck`]. Mirrors upstream's
/// `@evaluator(name=…)` / `evaluator(fn, name=…)`.
///
/// The function receives every supported field as [`EvalFields`] and may
/// return `bool`, a number (`>= 0.5` passes), a JSON object with `score` or
/// `passed` (see [`EvalOutcome::Map`]), a [`CheckResult`], or a `Result` of
/// any of these.
///
/// ```
/// use agent_framework_core::evaluation::{evaluator, LocalEvaluator};
///
/// let mentions_weather = evaluator("mentions_weather", |f| f.response.to_lowercase().contains("weather"));
/// let not_too_long = evaluator("length_check", |f| f.response.len() < 2000);
/// let local = LocalEvaluator::new().with_check(mentions_weather).with_check(not_too_long);
/// ```
pub fn evaluator<F, R>(name: impl Into<String>, func: F) -> FunctionEvaluator
where
    F: Fn(EvalFields) -> R + Send + Sync + 'static,
    R: IntoEvalOutcome,
{
    let func = Arc::new(func);
    FunctionEvaluator {
        name: name.into(),
        func: Arc::new(move |fields| {
            let outcome = func(fields).into_eval_outcome();
            Box::pin(std::future::ready(outcome))
        }),
    }
}

/// Wrap an async function as a named [`EvalCheck`] — e.g. an LLM-as-judge
/// that awaits a model call. Same contract as [`evaluator`].
///
/// ```
/// use agent_framework_core::evaluation::async_evaluator;
///
/// let judge = async_evaluator("llm_judge", |fields| async move {
///     // Score with a model here; return a float in [0, 1].
///     if fields.response.is_empty() { 0.0 } else { 1.0 }
/// });
/// ```
pub fn async_evaluator<F, Fut, R>(name: impl Into<String>, func: F) -> FunctionEvaluator
where
    F: Fn(EvalFields) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = R> + Send + 'static,
    R: IntoEvalOutcome + 'static,
{
    let func = Arc::new(func);
    FunctionEvaluator {
        name: name.into(),
        func: Arc::new(move |fields| {
            let fut = func(fields);
            Box::pin(async move { fut.await.into_eval_outcome() })
        }),
    }
}

// ---------------------------------------------------------------------------
// LocalEvaluator
// ---------------------------------------------------------------------------

/// An [`Evaluator`] that runs checks in-process, without API calls. Mirrors
/// upstream's `LocalEvaluator` (provider name `"Local"`).
///
/// Every check runs against every item (concurrently per item). An item
/// passes only when at least one check ran and every check passed — with no
/// checks an item fails, since a pass would carry no evidence. Each check
/// becomes an [`EvalScoreResult`] scored `1.0`/`0.0`; per-check counts land
/// in [`EvalResults::per_evaluator`], and every failure reason is joined into
/// [`EvalResults::error`].
#[derive(Clone, Default)]
pub struct LocalEvaluator {
    checks: Vec<Arc<dyn EvalCheck>>,
}

impl std::fmt::Debug for LocalEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalEvaluator")
            .field("checks", &self.checks.len())
            .finish()
    }
}

impl LocalEvaluator {
    /// An evaluator with no checks yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// An evaluator over already-shared checks.
    pub fn from_checks(checks: impl IntoIterator<Item = Arc<dyn EvalCheck>>) -> Self {
        Self {
            checks: checks.into_iter().collect(),
        }
    }

    /// Add a check.
    pub fn with_check(mut self, check: impl EvalCheck + 'static) -> Self {
        self.checks.push(Arc::new(check));
        self
    }

    /// Add an already-shared check.
    pub fn with_check_arc(mut self, check: Arc<dyn EvalCheck>) -> Self {
        self.checks.push(check);
        self
    }

    /// The number of checks.
    pub fn len(&self) -> usize {
        self.checks.len()
    }

    /// Whether there are no checks.
    pub fn is_empty(&self) -> bool {
        self.checks.is_empty()
    }
}

#[async_trait]
impl Evaluator for LocalEvaluator {
    fn name(&self) -> &str {
        "Local"
    }

    async fn evaluate(&self, items: &[EvalItem], eval_name: &str) -> Result<EvalResults> {
        let mut passed = 0u64;
        let mut failed = 0u64;
        let mut per_check: BTreeMap<String, ResultCounts> = BTreeMap::new();
        let mut failure_reasons: Vec<String> = Vec::new();
        let mut result_items: Vec<EvalItemResult> = Vec::with_capacity(items.len());

        for (idx, item) in items.iter().enumerate() {
            let check_results =
                futures::future::try_join_all(self.checks.iter().map(|c| c.check(item))).await?;
            let mut item_passed = !check_results.is_empty();
            let mut scores: Vec<EvalScoreResult> = Vec::with_capacity(check_results.len());
            for result in check_results {
                let counts = per_check.entry(result.check_name.clone()).or_default();
                if result.passed {
                    counts.passed += 1;
                } else {
                    counts.failed += 1;
                    item_passed = false;
                    failure_reasons.push(format!("{}: {}", result.check_name, result.reason));
                }
                scores.push(EvalScoreResult {
                    name: result.check_name.clone(),
                    score: if result.passed { 1.0 } else { 0.0 },
                    passed: Some(result.passed),
                    sample: (!result.reason.is_empty()).then(|| json!({"reason": result.reason})),
                    dimensions: None,
                });
            }
            if item_passed {
                passed += 1;
            } else {
                failed += 1;
            }
            let mut item_result = EvalItemResult::new(
                idx.to_string(),
                if item_passed {
                    EvalItemStatus::Pass
                } else {
                    EvalItemStatus::Fail
                },
            );
            item_result.scores = scores;
            item_result.input_text = Some(item.query());
            item_result.output_text = Some(item.response());
            result_items.push(item_result);
        }

        Ok(EvalResults {
            provider: self.name().to_string(),
            eval_id: "local".to_string(),
            run_id: eval_name.to_string(),
            status: EvalRunStatus::Completed,
            result_counts: Some(ResultCounts::new(passed, failed, 0)),
            report_url: None,
            error: (!failure_reasons.is_empty()).then(|| failure_reasons.join("; ")),
            per_evaluator: per_check,
            items: result_items,
            sub_results: BTreeMap::new(),
        })
    }
}
