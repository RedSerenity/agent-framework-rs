//! Declarative workflow state: scoped variables stored in the workflow's
//! [`SharedState`], and PowerFx evaluation against them.
//!
//! Port of upstream `DeclarativeWorkflowState` (`_declarative_base.py`) and
//! the path policy of `_state_path.py`.
//!
//! # Scopes
//!
//! | Path prefix | Storage | Writable |
//! |---|---|---|
//! | `Workflow.Inputs.*` | `Inputs` | no (read-only) |
//! | `Workflow.Outputs.*` | `Outputs` | yes |
//! | `Local.*` | `Local` | yes |
//! | `System.*` | `System` (`ConversationId`, `LastMessage`, `LastMessageText`, `LastMessageId`, `conversations`) | yes |
//! | `Agent.*` | `Agent` (last agent result) | yes |
//! | `Conversation.*` | `Conversation` (`messages`, `history`) | yes |
//! | anything else (`Foo.*`) | `Custom.Foo` | yes |
//!
//! PowerFx expressions see the roots `Workflow` (`Inputs`/`Outputs`),
//! `Local`, `System`, `Agent`, `Conversation`, `inputs` (an alias of
//! `Workflow.Inputs`), every custom namespace, and `Env` (see [`EnvConfig`]).
//!
//! Path segments are plain object keys: like upstream (where only dict keys
//! are traversed and attribute access is restricted to identifiers), arrays
//! are **not** indexed by path segments, and empty segments are rejected.
//!
//! The whole state lives under the single [`DECLARATIVE_STATE_KEY`] entry of
//! the run's [`SharedState`], so it is checkpointed with the run and staged
//! per superstep exactly like upstream's `State`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use agent_framework_core::workflow::SharedState;
use serde_json::{json, Map, Value as Json};

use crate::env::{EnvSource, ProcessEnv};
use crate::powerfx::limits::{validate_state, BudgetTally, StateBudget, StateLimitError};
use crate::powerfx::{Engine, PowerFxError, PowerFxErrorKind, Record, Symbols, Value};

/// The [`SharedState`] key holding the declarative variables.
pub const DECLARATIVE_STATE_KEY: &str = "_declarative_workflow_state";
/// The state-data key holding `Foreach` iteration state.
pub(crate) const LOOP_STATE_KEY: &str = "_declarative_loop_state";

/// Errors from declarative state operations.
#[derive(Debug, thiserror::Error)]
pub enum StateError {
    /// A path was empty, had empty segments, or targeted a read-only scope.
    #[error("{0}")]
    InvalidPath(String),
    /// The state would exceed its [`StateBudget`].
    #[error(transparent)]
    Limit(#[from] StateLimitError),
    /// A PowerFx expression failed to parse or evaluate.
    #[error("failed to evaluate {expression:?}: {source}")]
    Expression {
        /// The offending expression (truncated to 80 characters).
        expression: String,
        /// The PowerFx error.
        #[source]
        source: PowerFxError,
    },
}

impl From<StateError> for agent_framework_core::error::Error {
    fn from(e: StateError) -> Self {
        agent_framework_core::error::Error::Workflow(format!("declarative action error: {e}"))
    }
}

/// Configuration of the PowerFx `Env` symbol (upstream `DeclarativeEnvConfig`).
///
/// `values` are always exposed under `Env.<name>`. The process environment is
/// consulted only when `restrict_to_configuration` is `false` **and** the name
/// is in `referenced_names` (the `Env.NAME` references discovered in the
/// workflow's `=` expressions), so unrelated variables never enter scope.
/// When nothing resolves, `Env` is not bound at all and `=Env.X` evaluates to
/// `null`, as upstream.
#[derive(Clone)]
pub struct EnvConfig {
    /// Caller-supplied configuration values.
    pub values: BTreeMap<String, String>,
    /// When `true` (the default) only `values` are exposed.
    pub restrict_to_configuration: bool,
    /// `Env.NAME` references found in the workflow definition.
    pub referenced_names: BTreeSet<String>,
    /// Where the fallback reads variables (the process environment by default).
    pub source: Arc<dyn EnvSource + Send + Sync>,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            values: BTreeMap::new(),
            restrict_to_configuration: true,
            referenced_names: BTreeSet::new(),
            source: Arc::new(ProcessEnv),
        }
    }
}

impl std::fmt::Debug for EnvConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvConfig")
            .field("values", &self.values.keys().collect::<Vec<_>>())
            .field("restrict_to_configuration", &self.restrict_to_configuration)
            .field("referenced_names", &self.referenced_names)
            .finish()
    }
}

impl EnvConfig {
    /// Resolve the `Env` symbol mapping.
    pub fn resolve(&self) -> BTreeMap<String, String> {
        let mut resolved = self.values.clone();
        if self.restrict_to_configuration {
            return resolved;
        }
        for name in &self.referenced_names {
            if resolved.contains_key(name) {
                continue;
            }
            if let Some(v) = self.source.get(name) {
                resolved.insert(name.clone(), v);
            }
        }
        resolved
    }
}

/// Discover `Env.NAME` references inside `=` expressions anywhere in a
/// workflow definition (upstream `discover_env_references`).
pub fn discover_env_references(node: &Json) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut stack = vec![node];
    while let Some(v) = stack.pop() {
        match v {
            Json::String(s) if s.starts_with('=') => scan_env_refs(s, &mut names),
            Json::Array(items) => stack.extend(items.iter()),
            Json::Object(map) => stack.extend(map.values()),
            _ => {}
        }
    }
    names
}

fn scan_env_refs(s: &str, out: &mut BTreeSet<String>) {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i + 4 <= chars.len() {
        let boundary = i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '_');
        if boundary && chars[i..i + 4] == ['E', 'n', 'v', '.'] {
            let mut j = i + 4;
            if j < chars.len() && (chars[j].is_ascii_alphabetic() || chars[j] == '_') {
                while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                    j += 1;
                }
                out.insert(chars[i + 4..j].iter().collect());
                i = j;
                continue;
            }
        }
        i += 1;
    }
}

/// Runtime configuration shared by every executor of a workflow.
#[derive(Debug, Clone, Default)]
pub struct StateConfig {
    /// The `Env` symbol configuration.
    pub env: EnvConfig,
    /// The state budget (upstream `_powerfx_limits.py`).
    pub budget: StateBudget,
    /// The PowerFx engine (expression limits).
    pub engine: Engine,
}

/// The in-memory view of the declarative variables for one executor
/// invocation.
#[derive(Debug, Clone)]
pub struct DeclarativeState {
    data: Map<String, Json>,
    config: StateConfig,
}

/// Render a JSON value the way upstream's Python `str()` does for scalars
/// (`None`, `True`/`False`); arrays and objects render as compact JSON.
pub fn py_str(v: &Json) -> String {
    match v {
        Json::Null => "None".to_string(),
        Json::Bool(true) => "True".to_string(),
        Json::Bool(false) => "False".to_string(),
        Json::String(s) => s.clone(),
        Json::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Python truthiness of a JSON value (upstream evaluates conditions with
/// `bool(result)`).
pub fn py_truthy(v: &Json) -> bool {
    match v {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Json::String(s) => !s.is_empty(),
        Json::Array(a) => !a.is_empty(),
        Json::Object(o) => !o.is_empty(),
    }
}

fn split_path(path: &str) -> Option<Vec<&str>> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.iter().any(|p| p.is_empty()) {
        None
    } else {
        Some(parts)
    }
}

struct StateSymbols<'a> {
    data: &'a Map<String, Json>,
    env: Option<Json>,
}

impl Symbols for StateSymbols<'_> {
    fn lookup(&self, name: &str) -> Option<Value> {
        let field = |k: &str| {
            self.data
                .get(k)
                .map(Value::from_json)
                .unwrap_or(Value::Record(Record::new()))
        };
        match name {
            "Workflow" => {
                let mut r = Record::new();
                r.insert("Inputs", field("Inputs"));
                r.insert("Outputs", field("Outputs"));
                Some(Value::Record(r))
            }
            "Local" | "System" | "Agent" | "Conversation" => Some(field(name)),
            "inputs" => Some(field("Inputs")),
            "Env" => self.env.as_ref().map(Value::from_json),
            _ => self
                .data
                .get("Custom")
                .and_then(|c| c.get(name))
                .map(Value::from_json),
        }
    }
}

impl DeclarativeState {
    /// An uninitialized state (no scopes yet).
    pub fn new(config: StateConfig) -> Self {
        Self {
            data: Map::new(),
            config,
        }
    }

    /// Load the state from `shared`, initializing it when absent (upstream
    /// `get_state_data` initializes lazily too).
    pub async fn load(shared: &SharedState, config: StateConfig) -> Result<Self, StateError> {
        let mut state = Self::new(config);
        match shared.get(DECLARATIVE_STATE_KEY).await {
            Some(Json::Object(map)) => {
                validate_state(&Json::Object(map.clone()), state.config.budget)?;
                state.data = map;
            }
            _ => state.initialize(None)?,
        }
        Ok(state)
    }

    /// Whether the state was present in `shared` (upstream `is_initialized`).
    pub async fn is_initialized_in(shared: &SharedState) -> bool {
        matches!(
            shared.get(DECLARATIVE_STATE_KEY).await,
            Some(Json::Object(_))
        )
    }

    /// Write the state back to `shared`.
    pub async fn save(&self, shared: &SharedState) {
        shared
            .set(DECLARATIVE_STATE_KEY, Json::Object(self.data.clone()))
            .await;
    }

    /// The raw state data (scopes keyed `Inputs`, `Outputs`, `Local`, …).
    pub fn data(&self) -> &Map<String, Json> {
        &self.data
    }

    /// The configuration in effect.
    pub fn config(&self) -> &StateConfig {
        &self.config
    }

    /// Replace the raw state data after validating its budget.
    pub fn set_data(&mut self, data: Map<String, Json>) -> Result<(), StateError> {
        validate_state(&Json::Object(data.clone()), self.config.budget)?;
        self.data = data;
        Ok(())
    }

    /// Reset every scope, binding `inputs` as `Workflow.Inputs`.
    pub fn initialize(&mut self, inputs: Option<Map<String, Json>>) -> Result<(), StateError> {
        let conversation_id = uuid::Uuid::new_v4().to_string();
        let data = json!({
            "Inputs": Json::Object(inputs.unwrap_or_default()),
            "Outputs": {},
            "Local": {},
            "System": {
                "ConversationId": conversation_id,
                "LastMessage": {"Text": "", "Id": ""},
                "LastMessageText": "",
                "LastMessageId": "",
                "conversations": {
                    conversation_id.clone(): {"id": conversation_id, "messages": []}
                },
            },
            "Agent": {},
            "Conversation": {"messages": [], "history": []},
            "Custom": {},
        });
        let Json::Object(map) = data else {
            unreachable!()
        };
        self.set_data(map)
    }

    /// Read a dotted path; `None` when any segment is missing.
    pub fn get(&self, path: &str) -> Option<&Json> {
        let parts = split_path(path)?;
        let (mut obj, rest): (&Json, &[&str]) = match parts[0] {
            "Workflow" if parts.len() > 1 => match parts[1] {
                "Inputs" => (self.data.get("Inputs")?, &parts[2..]),
                "Outputs" => (self.data.get("Outputs")?, &parts[2..]),
                _ => return None,
            },
            "Local" | "System" | "Agent" | "Conversation" => {
                (self.data.get(parts[0])?, &parts[1..])
            }
            ns => (self.data.get("Custom")?.get(ns)?, &parts[1..]),
        };
        for part in rest {
            obj = obj.as_object()?.get(*part)?;
        }
        Some(obj)
    }

    /// Read a path, cloning, with `null` for missing values.
    pub fn get_value(&self, path: &str) -> Json {
        self.get(path).cloned().unwrap_or(Json::Null)
    }

    /// Write a dotted path, creating intermediate objects.
    ///
    /// Errors for `Workflow.Inputs.*` (read-only), bare namespaces, unknown
    /// `Workflow.*` sub-scopes, empty segments, a non-object intermediate,
    /// and writes that would exceed the state budget (the previous state is
    /// kept).
    pub fn set(&mut self, path: &str, value: Json) -> Result<(), StateError> {
        let parts = split_path(path).ok_or_else(|| {
            StateError::InvalidPath(format!(
                "Invalid path {path:?}: empty segments are not allowed"
            ))
        })?;
        let (scope_key, custom_ns, rest): (&str, Option<&str>, &[&str]) = match parts[0] {
            "Workflow" => {
                if parts.len() == 1 {
                    return Err(StateError::InvalidPath(
                        "Cannot set 'Workflow' directly; use 'Workflow.Outputs.*'".into(),
                    ));
                }
                match parts[1] {
                    "Inputs" => {
                        return Err(StateError::InvalidPath(
                            "Cannot modify Workflow.Inputs - they are read-only".into(),
                        ))
                    }
                    "Outputs" => ("Outputs", None, &parts[2..]),
                    other => {
                        return Err(StateError::InvalidPath(format!(
                            "Unknown Workflow namespace: {other}"
                        )))
                    }
                }
            }
            "Local" | "System" | "Agent" | "Conversation" => (parts[0], None, &parts[1..]),
            ns => ("Custom", Some(ns), &parts[1..]),
        };
        if rest.is_empty() {
            return Err(StateError::InvalidPath(format!(
                "Cannot replace entire namespace '{}'",
                parts[0]
            )));
        }
        let backup = self.data.clone();
        let result = (|| {
            let mut target = self
                .data
                .entry(scope_key.to_string())
                .or_insert_with(|| Json::Object(Map::new()));
            if let Some(ns) = custom_ns {
                target = object_mut(target, scope_key)?
                    .entry(ns.to_string())
                    .or_insert_with(|| Json::Object(Map::new()));
            }
            for part in &rest[..rest.len() - 1] {
                target = object_mut(target, path)?
                    .entry(part.to_string())
                    .or_insert_with(|| Json::Object(Map::new()));
            }
            object_mut(target, path)?.insert(rest[rest.len() - 1].to_string(), value);
            Ok::<(), StateError>(())
        })();
        let result = result.and_then(|()| {
            validate_state(&Json::Object(self.data.clone()), self.config.budget)
                .map_err(StateError::from)
        });
        if result.is_err() {
            self.data = backup;
        }
        result
    }

    /// Append to a list at `path`, creating it when absent.
    pub fn append(&mut self, path: &str, value: Json) -> Result<(), StateError> {
        if split_path(path).is_none() {
            return Err(StateError::InvalidPath(format!(
                "Invalid path {path:?}: empty segments are not allowed"
            )));
        }
        match self.get(path).cloned() {
            None | Some(Json::Null) => self.set(path, Json::Array(vec![value])),
            Some(Json::Array(mut items)) => {
                items.push(value);
                self.set(path, Json::Array(items))
            }
            Some(_) => Err(StateError::InvalidPath(format!(
                "Cannot append to non-list at path '{path}'"
            ))),
        }
    }

    /// Remove a top-level `Local` variable if present.
    pub fn clear_local(&mut self, name: &str) {
        if let Some(Json::Object(local)) = self.data.get_mut("Local") {
            local.remove(name);
        }
    }

    fn symbols(&self) -> Result<StateSymbols<'_>, StateError> {
        let env = self.config.env.resolve();
        let env_json = if env.is_empty() {
            None
        } else {
            Some(Json::Object(
                env.into_iter().map(|(k, v)| (k, Json::String(v))).collect(),
            ))
        };
        // Budget the projected symbols (state, the `inputs` alias, Env), as
        // upstream does before handing symbols to PowerFx.
        let mut tally = BudgetTally::new(self.config.budget);
        tally.visit(&Json::Object(self.data.clone()))?;
        if let Some(inputs) = self.data.get("Inputs") {
            tally.visit(inputs)?;
        }
        if let Some(env) = &env_json {
            tally.visit(env)?;
        }
        Ok(StateSymbols {
            data: &self.data,
            env: env_json,
        })
    }

    /// Evaluate an expression string.
    ///
    /// Strings starting with `=` are PowerFx; anything else is returned
    /// unchanged. A reference to an undefined root name yields `null` (the
    /// upstream "isn't recognized" fallback); every other failure — syntax
    /// errors, unknown functions, type and runtime errors, limits — is an
    /// error.
    pub fn eval(&self, expression: &str) -> Result<Json, StateError> {
        let Some(formula) = expression.strip_prefix('=') else {
            return Ok(Json::String(expression.to_string()));
        };
        let symbols = self.symbols()?;
        match self.config.engine.eval_json(formula, &symbols) {
            Ok(v) => Ok(v),
            Err(e) if e.kind() == PowerFxErrorKind::UnknownName => {
                tracing::debug!("PowerFx: undefined name in expression; returning null");
                Ok(Json::Null)
            }
            Err(source) => Err(StateError::Expression {
                expression: expression.chars().take(80).collect(),
                source,
            }),
        }
    }

    /// Evaluate strings recursively inside objects and arrays; other values
    /// are returned as-is.
    pub fn eval_if_expression(&self, value: &Json) -> Result<Json, StateError> {
        Ok(match value {
            Json::String(s) => self.eval(s)?,
            Json::Object(map) => {
                let mut out = Map::new();
                for (k, v) in map {
                    out.insert(k.clone(), self.eval_if_expression(v)?);
                }
                Json::Object(out)
            }
            Json::Array(items) => Json::Array(
                items
                    .iter()
                    .map(|v| self.eval_if_expression(v))
                    .collect::<Result<_, _>>()?,
            ),
            other => other.clone(),
        })
    }

    /// Replace `{Scope.path}` tokens with the value at that path (upstream
    /// `interpolate_string`): the root must be an identifier
    /// (`[A-Za-z][A-Za-z0-9_]*`), later segments any non-empty key without
    /// braces, whitespace, or dots. Unresolved paths become `""`; tokens that
    /// do not look like paths (`{Ctrl+C}`) stay literal. Values render with
    /// [`py_str`].
    pub fn interpolate_string(&self, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '{' {
                if let Some((path, end)) = match_path_token(&chars, i) {
                    if let Some(v) = self.get(&path).filter(|v| !v.is_null()) {
                        out.push_str(&py_str(v));
                    }
                    i = end;
                    continue;
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        out
    }

    /// .NET-style template formatting: every `{expr}` hole is evaluated as a
    /// PowerFx expression (paths, `MessageText(...)`, …). Holes that fail to
    /// parse are left literal. Used for the .NET `SetTextVariable.value` form.
    pub fn format_template(&self, text: &str) -> Result<String, StateError> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '{' {
                let mut depth = 0usize;
                let mut j = i;
                let mut close = None;
                let mut in_str = false;
                while j < chars.len() {
                    match chars[j] {
                        '"' => in_str = !in_str,
                        '{' if !in_str => depth += 1,
                        '}' if !in_str => {
                            depth -= 1;
                            if depth == 0 {
                                close = Some(j);
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                if let Some(end) = close {
                    let inner: String = chars[i + 1..end].iter().collect();
                    if !inner.trim().is_empty() && self.config.engine.check(inner.trim()).is_ok() {
                        let v = self.eval(&format!("={}", inner.trim()))?;
                        if !v.is_null() {
                            out.push_str(&py_str(&v));
                        }
                        i = end + 1;
                        continue;
                    }
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        Ok(out)
    }
}

fn object_mut<'a>(v: &'a mut Json, path: &str) -> Result<&'a mut Map<String, Json>, StateError> {
    if v.is_null() {
        *v = Json::Object(Map::new());
    }
    v.as_object_mut().ok_or_else(|| {
        StateError::InvalidPath(format!(
            "Cannot set {path:?}: an intermediate value is not an object"
        ))
    })
}

/// Match `{Root(.seg)*}` at `start`; returns the path and the index after `}`.
fn match_path_token(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut i = start + 1;
    if i >= chars.len() || !chars[i].is_ascii_alphabetic() {
        return None;
    }
    while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
        i += 1;
    }
    loop {
        if i < chars.len() && chars[i] == '.' {
            let seg_start = i + 1;
            let mut j = seg_start;
            while j < chars.len()
                && !matches!(chars[j], '{' | '}' | '.')
                && !chars[j].is_whitespace()
            {
                j += 1;
            }
            if j == seg_start {
                return None;
            }
            i = j;
        } else {
            break;
        }
    }
    if i < chars.len() && chars[i] == '}' {
        Some((chars[start + 1..i].iter().collect(), i + 1))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> DeclarativeState {
        let mut s = DeclarativeState::new(StateConfig::default());
        s.initialize(Some(
            json!({"name": "Alice", "age": 25})
                .as_object()
                .cloned()
                .unwrap(),
        ))
        .unwrap();
        s
    }

    #[test]
    fn scopes_route_reads_and_writes() {
        let mut s = state();
        assert_eq!(s.get_value("Workflow.Inputs.name"), json!("Alice"));
        s.set("Local.x", json!(1)).unwrap();
        s.set("Workflow.Outputs.r", json!("ok")).unwrap();
        s.set("Custom1.a.b", json!(true)).unwrap();
        assert_eq!(s.get_value("Local.x"), json!(1));
        assert_eq!(s.get_value("Workflow.Outputs.r"), json!("ok"));
        assert_eq!(s.get_value("Custom1.a.b"), json!(true));
        assert_eq!(s.data()["Custom"]["Custom1"]["a"]["b"], json!(true));
        assert!(s.get("Local.missing.deep").is_none());
        assert!(s.get("Local..x").is_none());
    }

    #[test]
    fn invalid_writes_are_rejected() {
        let mut s = state();
        for bad in [
            "Workflow.Inputs.name",
            "Workflow",
            "Workflow.Other.x",
            "Local",
            "Local.",
            "Local..a",
            "",
        ] {
            assert!(s.set(bad, json!(1)).is_err(), "{bad}");
        }
        s.set("Local.s", json!("text")).unwrap();
        assert!(s.set("Local.s.inner", json!(1)).is_err());
        assert!(s.append("Local.s", json!(1)).is_err());
    }

    #[test]
    fn append_creates_and_extends_lists() {
        let mut s = state();
        s.append("Local.items", json!(1)).unwrap();
        s.append("Local.items", json!(2)).unwrap();
        assert_eq!(s.get_value("Local.items"), json!([1, 2]));
    }

    #[test]
    fn eval_semantics() {
        let mut s = state();
        s.set("Local.x", json!(5)).unwrap();
        assert_eq!(s.eval("plain").unwrap(), json!("plain"));
        assert_eq!(s.eval("").unwrap(), json!(""));
        assert_eq!(s.eval("=Local.x * 2").unwrap(), json!(10));
        assert_eq!(s.eval("=inputs.name").unwrap(), json!("Alice"));
        assert_eq!(s.eval("=Local.UndefinedVar").unwrap(), json!(null));
        assert_eq!(s.eval("=Undefined.Root").unwrap(), json!(null));
        assert_eq!(s.eval("=Env.KEY").unwrap(), json!(null));
        assert!(s.eval("=1 +").is_err());
        assert!(s.eval("=NoSuchFunction(1)").is_err());
        assert_eq!(
            s.eval_if_expression(&json!({"a": "=Local.x", "b": [1, "=1+1"]}))
                .unwrap(),
            json!({"a": 5, "b": [1, 2]})
        );
    }

    #[test]
    fn env_symbol_respects_configuration() {
        let mut config = StateConfig::default();
        config.env.values.insert("KEY".into(), "v".into());
        config.env.referenced_names.insert("OTHER".into());
        config.env.source = Arc::new(|k: &str| (k == "OTHER").then(|| "o".to_string()));
        let mut s = DeclarativeState::new(config.clone());
        s.initialize(None).unwrap();
        assert_eq!(s.eval("=Env.KEY").unwrap(), json!("v"));
        assert_eq!(s.eval("=Env.OTHER").unwrap(), json!(null));
        config.env.restrict_to_configuration = false;
        let mut s = DeclarativeState::new(config);
        s.initialize(None).unwrap();
        assert_eq!(s.eval("=Env.KEY & Env.OTHER").unwrap(), json!("vo"));
    }

    #[test]
    fn env_reference_discovery_only_scans_expressions() {
        let def = json!({"a": "=Env.FIRST & Env.SECOND", "b": ["=xEnv.NOPE", "Env.PLAIN"]});
        let names = discover_env_references(&def);
        assert_eq!(
            names.into_iter().collect::<Vec<_>>(),
            vec!["FIRST", "SECOND"]
        );
    }

    #[test]
    fn interpolation_matches_upstream_rules() {
        let mut s = state();
        s.set("Local.T", json!({"TicketId": "TKT-1"})).unwrap();
        s.set("Local.flag", json!(true)).unwrap();
        assert_eq!(
            s.interpolate_string(
                "Created #{Local.T.TicketId} {Local.flag} {Local.none} {Ctrl+C} {x"
            ),
            "Created #TKT-1 True  {Ctrl+C} {x"
        );
    }

    #[test]
    fn template_formatting_evaluates_expressions() {
        let mut s = state();
        s.set("Local.m", json!([{"role": "assistant", "text": "facts"}]))
            .unwrap();
        assert_eq!(
            s.format_template("Use {MessageText(Local.m)} for {Workflow.Inputs.name}; {not valid}")
                .unwrap(),
            "Use facts for Alice; {not valid}"
        );
    }

    #[test]
    fn budget_rejects_oversized_writes_and_keeps_previous_value() {
        let mut config = StateConfig::default();
        config.budget.max_depth = 5;
        let mut s = DeclarativeState::new(config);
        s.initialize(None).unwrap();
        s.set("Local.value", json!("before")).unwrap();
        let err = s.set("Local.value", json!([[[[0]]]])).unwrap_err();
        assert!(err.to_string().contains("depth budget"), "{err}");
        assert_eq!(s.get_value("Local.value"), json!("before"));
    }

    #[test]
    fn budget_is_checked_before_evaluation() {
        let mut config = StateConfig::default();
        config.budget.max_nodes = 60;
        let mut s = DeclarativeState::new(config);
        s.initialize(None).unwrap();
        // The `inputs` alias doubles Inputs in the projection.
        s.set("Local.a", json!([1, 2, 3, 4, 5])).unwrap();
        let mut big = s.clone();
        big.data
            .get_mut("Local")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("b".into(), json!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
        assert!(big.eval("=1 + 1").is_err());
    }

    #[test]
    fn python_rendering_helpers() {
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(py_str(&json!(false)), "False");
        assert_eq!(py_str(&json!(3)), "3");
        assert!(!py_truthy(&json!(0)));
        assert!(!py_truthy(&json!("")));
        assert!(py_truthy(&json!([0])));
    }
}
