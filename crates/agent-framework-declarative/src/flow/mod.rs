//! Upstream-format declarative workflows: `kind: Workflow` documents with a
//! `trigger` (or top-level `actions`), evaluated with PowerFx `=expressions`.
//!
//! This is the Rust port of upstream's `agent_framework_declarative._workflows`
//! (Python) / `Microsoft.Agents.AI.Workflows.Declarative` (.NET). Each action
//! becomes an executor on the core graph engine
//! ([`agent_framework_core::workflow`]), so runs support streaming,
//! human-in-the-loop pauses, and checkpointing at action boundaries.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_core::prelude::*;
//! use agent_framework_declarative::flow::WorkflowFactory;
//!
//! # async fn demo(agent: Arc<dyn SupportsAgentRun>) -> std::result::Result<(), Box<dyn std::error::Error>> {
//! let yaml = r#"
//! kind: Workflow
//! trigger:
//!   kind: OnConversationStart
//!   id: demo
//!   actions:
//!     - kind: SetVariable
//!       variable: Local.Greeting
//!       value: =Concat("Hello, ", Workflow.Inputs.name, "!")
//!     - kind: SendActivity
//!       activity: =Local.Greeting
//!     - kind: InvokeAzureAgent
//!       agent: { name: Writer }
//!       input: { messages: =UserMessage(Local.Greeting) }
//! "#;
//! let workflow = WorkflowFactory::new()
//!     .with_agent("Writer", agent)
//!     .create_workflow_from_yaml(yaml)?;
//! let run = workflow.run(serde_json::json!({"name": "Ada"})).await?;
//! println!("{:?}", run.outputs());
//! # Ok(())
//! # }
//! ```
//!
//! # Supported actions
//!
//! | Kind | Notes |
//! |---|---|
//! | `SetValue`, `SetVariable`, `SetTextVariable`, `SetMultipleVariables`, `ResetVariable`, `ClearAllVariables` | state writes; `SetTextVariable` also accepts the .NET `value` template |
//! | `SendActivity` | yields text output (`=expr` evaluated, otherwise `{Path}` interpolation) |
//! | `ParseValue` | with `valueType` conversion |
//! | `EditTable`, `EditTableV2` | Python (`table`/`operation`) and .NET (`itemsVariable`/`changeType`) shapes |
//! | `CreateConversation`, `AddConversationMessage`, `CopyConversationMessages`, `RetrieveConversationMessage`, `RetrieveConversationMessages` | conversations live in `System.conversations` (the last four are .NET actions upstream Python lacks) |
//! | `If`, `ConditionGroup`, `Foreach`, `GotoAction`, `BreakLoop`, `ContinueLoop` | graph structure, see *Graph structure* below |
//! | `EndWorkflow`, `EndDialog`, `EndConversation`, `CancelDialog`, `CancelAllDialogs` | stop the current path |
//! | `InvokeAzureAgent` | any registered [`SupportsAgentRun`] agent; external loop HITL |
//! | `Question`, `RequestExternalInput`, `RequestHumanInput`, `WaitForHumanInput` | pause via the core request/response mechanism |
//! | `InvokeFunctionTool` | registered [`Tool`]s, optional approval |
//! | `HttpRequestAction` | through an [`HttpRequestHandler`] |
//! | `InvokeMcpTool` | through an [`McpToolHandler`], optional header-bound approval |
//!
//! Unknown action kinds are skipped with a warning, as upstream does, unless
//! [`WorkflowFactory::strict_actions`] is enabled.
//!
//! # Graph structure
//!
//! Every action becomes an executor (ids are the action `id`, else
//! generated as upstream: `{Kind}_{n}` / `{parent}_{Kind}_{n}`), joined by
//! plain edges except after a terminator (`GotoAction`, `BreakLoop`,
//! `ContinueLoop`, `End*`, `Cancel*`). `If`/`ConditionGroup` become an
//! `<id>_eval` node that sends the index of the first true condition, with
//! conditional edges into each branch and a pass-through (`<id>_else_pass` /
//! `<id>_default`) when there is no else; every branch exit continues to the
//! next action. `Foreach` becomes `<id>_init` → body → `<id>_next` → body,
//! leaving through `<id>_exit`; `BreakLoop`/`ContinueLoop` signal `<id>_next`.
//! `GotoAction` adds an edge to its target (back edges make loops). A fixed
//! [`ENTRY_ID`] node receives the run input. One action runs per superstep,
//! so `maxTurns` (default 100) bounds the number of actions executed.
//!
//! Build-time validation mirrors upstream: duplicate or reserved ids, missing
//! required fields (with upstream's alternates), `ConditionGroup`
//! `else`/`default`, self-targeting or unknown goto targets, loop signals
//! outside a loop, and HTTP/MCP actions without a handler are errors.
//! Strictly more permissive than upstream: unreachable actions (dead code
//! after a terminator) are pruned instead of rejected, a goto may target an
//! `If`/`ConditionGroup`/`Foreach` id, and `Foreach.items` satisfies
//! `source`.
//!
//! # Input
//!
//! The run input initializes the state: an object becomes `Workflow.Inputs`;
//! text becomes `Workflow.Inputs.input` and `System.LastMessage.Text`; a
//! serialized chat message (or array of them) splits into the last user
//! message (as above) and `Conversation.messages` history.
//!
//! # Output
//!
//! `SendActivity`, auto-sent agent replies, tool results and MCP outputs are
//! yielded as JSON strings; read them from
//! [`WorkflowRun::outputs`](agent_framework_core::workflow::WorkflowRun::outputs).
//! The final variables are in the run's shared state under
//! [`DECLARATIVE_STATE_KEY`] (see [`final_state`]).
//!
//! # Divergences from upstream
//!
//! * Expressions run on this crate's [`powerfx`](crate::powerfx) interpreter,
//!   not Microsoft's engine; see its docs for the supported subset.
//! * Internal messages are JSON tagged with `"$declarative"`; request
//!   payloads carry a `"type"` discriminator (`ExternalInputRequest`,
//!   `AgentExternalInputRequest`, `ToolApprovalRequest`,
//!   `MCPToolApprovalRequest`) where upstream uses dataclass types, and the
//!   engine assigns its own request ids (the payload's `request_id` is for
//!   correlation/binding only).
//! * Values are rendered with Python `str()` semantics for scalars
//!   (`True`, `None`, `3.0`) but as compact JSON for objects and arrays.
//! * State lives in JSON objects, whose keys are sorted; mapping order is
//!   kept only where it is observable upstream through action definitions
//!   (`input.arguments` text, HTTP/MCP headers and query parameters).
//! * `${VAR}` interpolation is **not** applied to upstream-format workflows
//!   (upstream has none; use `=Env.VAR`).
//! * YAML is parsed as YAML 1.2 (`yes`/`no` are strings), whereas PyYAML
//!   uses YAML 1.1.
//! * Multi-turn continuation across separate `run()` calls (upstream keeps
//!   state on the workflow instance) does not apply: every core run starts
//!   with fresh shared state. Resume a paused run with
//!   [`WorkflowRun::send_response`](agent_framework_core::workflow::WorkflowRun::send_response)
//!   or from a checkpoint instead.

mod agents;
mod builder;
mod executor;
mod external_input;
mod http;
mod mcp;
mod messages;
mod state;
mod tools;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::tools::{Tool, ToolDefinition};
use agent_framework_core::workflow::{CheckpointStorage, SharedState, Workflow};
use serde_json::{Map, Value as Json};

pub use agents::EXTERNAL_LOOP_STATE_KEY;
pub use builder::ENTRY_ID;
#[cfg(feature = "http")]
pub use http::DefaultHttpRequestHandler;
pub use http::{HttpRequestError, HttpRequestHandler, HttpRequestInfo, HttpRequestResult};
#[cfg(feature = "mcp")]
pub use mcp::DefaultMcpToolHandler;
pub use mcp::{
    McpToolError, McpToolHandler, McpToolInvocation, McpToolResult, LIST_TOOLS_TOOL_NAME,
};
pub use messages::MESSAGE_TAG;
pub use state::{
    discover_env_references, py_str, py_truthy, DeclarativeState, EnvConfig, StateConfig,
    StateError, DECLARATIVE_STATE_KEY,
};

use crate::env::{EnvSource, ProcessEnv};
use crate::error::{DeclarativeError, Result};
use crate::loader::DeclarativeLoader;
use crate::powerfx::limits::StateBudget;
use crate::powerfx::{Engine, ExpressionLimits};

/// Whether a parsed document is an upstream-format workflow (it has
/// `trigger` or `actions`), as opposed to this crate's Rust-native
/// [`WorkflowSpec`](crate::WorkflowSpec).
pub fn is_upstream_workflow(def: &Json) -> bool {
    def.as_object()
        .is_some_and(|m| m.contains_key("trigger") || m.contains_key("actions"))
}

/// Convert parsed YAML to JSON, stringifying non-string mapping keys and
/// dropping YAML tags.
pub fn yaml_to_json(v: serde_yaml::Value) -> Json {
    match v {
        serde_yaml::Value::Null => Json::Null,
        serde_yaml::Value::Bool(b) => Json::Bool(b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Json::from(i)
            } else if let Some(u) = n.as_u64() {
                Json::from(u)
            } else {
                n.as_f64()
                    .and_then(serde_json::Number::from_f64)
                    .map(Json::Number)
                    .unwrap_or(Json::Null)
            }
        }
        serde_yaml::Value::String(s) => Json::String(s),
        serde_yaml::Value::Sequence(items) => {
            Json::Array(items.into_iter().map(yaml_to_json).collect())
        }
        serde_yaml::Value::Mapping(m) => {
            let mut out = Map::new();
            for (k, v) in m {
                out.insert(yaml_key_string(&k), yaml_to_json(v));
            }
            Json::Object(out)
        }
        serde_yaml::Value::Tagged(t) => yaml_to_json(t.value),
    }
}

/// The JSON object key for a YAML mapping key (non-strings rendered the way
/// upstream's Python `str()` would).
pub(crate) fn yaml_key_string(k: &serde_yaml::Value) -> String {
    match yaml_to_json(k.clone()) {
        Json::String(s) => s,
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Null => "None".into(),
        other => other.to_string(),
    }
}

/// Read the declarative variables of a finished (or paused) run.
pub async fn final_state(shared: &SharedState) -> Option<Map<String, Json>> {
    match shared.get(DECLARATIVE_STATE_KEY).await {
        Some(Json::Object(m)) => Some(m),
        _ => None,
    }
}

/// Convert a declarative `inputs:` block to a JSON Schema (upstream
/// `_convert_inputs_to_json_schema`): fields are required unless
/// `required: false`, and type aliases (`int`, `str`, `float`, `bool`,
/// `list`, `dict`) map to JSON Schema types.
pub fn inputs_json_schema(inputs: &Json) -> Json {
    let mut properties = Map::new();
    let mut required = Vec::new();
    let map_type = |t: &str, simple: bool| -> String {
        match t {
            "string" | "str" => "string",
            "integer" | "int" => "integer",
            "number" | "float" => "number",
            "boolean" | "bool" => "boolean",
            "array" | "list" if !simple => "array",
            "object" | "dict" if !simple => "object",
            other if !simple => other,
            _ => "string",
        }
        .to_string()
    };
    for (name, def) in inputs.as_object().into_iter().flatten() {
        if let Json::Object(d) = def {
            let t = d.get("type").map(py_str).unwrap_or_else(|| "string".into());
            let mut prop = Map::new();
            prop.insert("type".into(), Json::String(map_type(&t, false)));
            for k in ["description", "default", "enum"] {
                if let Some(v) = d.get(k) {
                    prop.insert(k.into(), v.clone());
                }
            }
            if d.get("required").map(py_truthy).unwrap_or(true) {
                required.push(Json::String(name.clone()));
            }
            properties.insert(name.clone(), Json::Object(prop));
        } else {
            properties.insert(
                name.clone(),
                serde_json::json!({"type": map_type(&py_str(def), true)}),
            );
            required.push(Json::String(name.clone()));
        }
    }
    let mut schema = Map::new();
    schema.insert("type".into(), Json::String("object".into()));
    schema.insert("properties".into(), Json::Object(properties));
    if !required.is_empty() {
        schema.insert("required".into(), Json::Array(required));
    }
    Json::Object(schema)
}

/// Builds an agent from a workflow `agents:` entry.
pub(crate) type AgentMaker<'a> =
    &'a dyn Fn(&Json, Option<&Path>) -> Result<Arc<dyn SupportsAgentRun>>;

/// Builds runnable [`Workflow`]s from upstream-format declarative YAML
/// (upstream `WorkflowFactory`).
pub struct WorkflowFactory {
    agents: HashMap<String, Arc<dyn SupportsAgentRun>>,
    tools: HashMap<String, Arc<dyn Tool>>,
    http: Option<Arc<dyn HttpRequestHandler>>,
    mcp: Option<Arc<dyn McpToolHandler>>,
    configuration: BTreeMap<String, String>,
    restrict_env_to_configuration: bool,
    env_source: Arc<dyn EnvSource + Send + Sync>,
    max_iterations: Option<usize>,
    checkpoint_storage: Option<Arc<dyn CheckpointStorage>>,
    expression_limits: ExpressionLimits,
    state_budget: StateBudget,
    agent_loader: Option<Arc<DeclarativeLoader>>,
    strict_actions: bool,
}

impl Default for WorkflowFactory {
    fn default() -> Self {
        Self {
            agents: HashMap::new(),
            tools: HashMap::new(),
            http: None,
            mcp: None,
            configuration: BTreeMap::new(),
            restrict_env_to_configuration: true,
            env_source: Arc::new(ProcessEnv),
            max_iterations: None,
            checkpoint_storage: None,
            expression_limits: ExpressionLimits::default(),
            state_budget: StateBudget::default(),
            agent_loader: None,
            strict_actions: false,
        }
    }
}

/// A function tool backed by an async closure over the JSON arguments.
struct FnTool<F> {
    name: String,
    func: F,
}

#[async_trait::async_trait]
impl<F, Fut> Tool for FnTool<F>
where
    F: Fn(Json) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = agent_framework_core::error::Result<Json>> + Send,
{
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        ""
    }
    fn parameters_schema(&self) -> Json {
        serde_json::json!({"type": "object", "properties": {}})
    }
    async fn invoke(&self, arguments: Json) -> agent_framework_core::error::Result<Json> {
        (self.func)(arguments).await
    }
}

impl WorkflowFactory {
    /// A factory with no agents, tools, or handlers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent for `InvokeAzureAgent` (upstream `register_agent`).
    pub fn with_agent(mut self, name: impl Into<String>, agent: Arc<dyn SupportsAgentRun>) -> Self {
        self.agents.insert(name.into(), agent);
        self
    }

    /// Register an agent (mutating form).
    pub fn register_agent(&mut self, name: impl Into<String>, agent: Arc<dyn SupportsAgentRun>) {
        self.agents.insert(name.into(), agent);
    }

    /// Register a tool for `InvokeFunctionTool` under `name`.
    pub fn with_tool(mut self, name: impl Into<String>, tool: Arc<dyn Tool>) -> Self {
        self.tools.insert(name.into(), tool);
        self
    }

    /// Register an executable [`ToolDefinition`] under its own name; a
    /// definition without an executor is ignored with a warning.
    pub fn with_tool_definition(mut self, def: ToolDefinition) -> Self {
        match def.executor {
            Some(exec) => {
                self.tools.insert(def.name, exec);
            }
            None => tracing::warn!("tool '{}' has no executor; not registered", def.name),
        }
        self
    }

    /// Register an async function `(arguments) -> result` under `name`
    /// (upstream `register_tool` with a Python callable).
    pub fn with_function<F, Fut>(mut self, name: impl Into<String>, func: F) -> Self
    where
        F: Fn(Json) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = agent_framework_core::error::Result<Json>>
            + Send
            + 'static,
    {
        let name = name.into();
        self.tools
            .insert(name.clone(), Arc::new(FnTool { name, func }));
        self
    }

    /// The handler for `HttpRequestAction` (required when the workflow uses it).
    pub fn with_http_request_handler(mut self, handler: Arc<dyn HttpRequestHandler>) -> Self {
        self.http = Some(handler);
        self
    }

    /// The handler for `InvokeMcpTool` (required when the workflow uses it).
    pub fn with_mcp_tool_handler(mut self, handler: Arc<dyn McpToolHandler>) -> Self {
        self.mcp = Some(handler);
        self
    }

    /// Values exposed as `Env.<name>` (upstream `configuration`).
    pub fn with_configuration<I, K, V>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.configuration
            .extend(values.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// When `false`, `Env.NAME` falls back to the environment for names the
    /// workflow references (default `true`: configuration only).
    pub fn restrict_env_to_configuration(mut self, restrict: bool) -> Self {
        self.restrict_env_to_configuration = restrict;
        self
    }

    /// The environment used for the `Env` fallback (process env by default).
    pub fn with_env_source<E: EnvSource + Send + Sync + 'static>(mut self, env: E) -> Self {
        self.env_source = Arc::new(env);
        self
    }

    /// Maximum supersteps; overrides the YAML `maxTurns` (default 100).
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Checkpoint storage for pause/resume across processes.
    pub fn with_checkpoint_storage(mut self, storage: Arc<dyn CheckpointStorage>) -> Self {
        self.checkpoint_storage = Some(storage);
        self
    }

    /// PowerFx expression length / depth limits.
    pub fn with_expression_limits(mut self, limits: ExpressionLimits) -> Self {
        self.expression_limits = limits;
        self
    }

    /// The state budget (upstream `_powerfx_limits.py`).
    pub fn with_state_budget(mut self, budget: StateBudget) -> Self {
        self.state_budget = budget;
        self
    }

    /// The loader used for agents defined inline under the workflow's
    /// `agents:` key (`file:` references or `kind: Prompt` definitions).
    pub fn with_agent_loader(mut self, loader: DeclarativeLoader) -> Self {
        self.agent_loader = Some(Arc::new(loader));
        self
    }

    /// Fail on unknown action kinds instead of skipping them.
    pub fn strict_actions(mut self, strict: bool) -> Self {
        self.strict_actions = strict;
        self
    }

    /// Build from a YAML file; relative agent `file:` references resolve
    /// against its directory.
    pub fn create_workflow_from_yaml_path(&self, path: impl AsRef<Path>) -> Result<Workflow> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| {
            DeclarativeError::Invalid(format!(
                "Workflow YAML file not found or unreadable: {}: {e}",
                path.display()
            ))
        })?;
        let raw = parse_yaml_raw(&text)?;
        self.build_with_default_loader(&yaml_to_json(raw.clone()), Some(&raw), path.parent())
    }

    /// Build from a YAML string.
    pub fn create_workflow_from_yaml(&self, yaml: &str) -> Result<Workflow> {
        let raw = parse_yaml_raw(yaml)?;
        self.build_with_default_loader(&yaml_to_json(raw.clone()), Some(&raw), None)
    }

    /// Build from a parsed definition.
    pub fn create_workflow_from_definition(
        &self,
        def: &Json,
        base_path: Option<&Path>,
    ) -> Result<Workflow> {
        self.build_with_default_loader(def, None, base_path)
    }

    fn build_with_default_loader(
        &self,
        def: &Json,
        raw: Option<&serde_yaml::Value>,
        base_path: Option<&Path>,
    ) -> Result<Workflow> {
        let loader = self.agent_loader.clone();
        self.build(def, raw, base_path, &|agent_def, base| {
            let default_loader;
            let loader = match &loader {
                Some(l) => l.as_ref(),
                None => {
                    default_loader = DeclarativeLoader::new();
                    &default_loader
                }
            };
            agent_from_definition(loader, agent_def, base)
        })
    }

    pub(crate) fn build(
        &self,
        def: &Json,
        raw: Option<&serde_yaml::Value>,
        base_path: Option<&Path>,
        make_agent: AgentMaker<'_>,
    ) -> Result<Workflow> {
        let Json::Object(map) = def else {
            return Err(DeclarativeError::Invalid(
                "Workflow definition must be a dictionary".into(),
            ));
        };
        let (actions, raw_actions) = match (map.get("actions"), map.get("trigger")) {
            (Some(a), _) => (a.clone(), raw.and_then(|r| r.get("actions"))),
            (None, Some(Json::Object(t))) if t.contains_key("actions") => (
                t["actions"].clone(),
                raw.and_then(|r| r.get("trigger"))
                    .and_then(|t| t.get("actions")),
            ),
            _ => {
                return Err(DeclarativeError::Invalid(
                    "Workflow definition must have 'actions' field or 'trigger.actions' field"
                        .into(),
                ))
            }
        };
        let Json::Array(actions) = actions else {
            return Err(DeclarativeError::Invalid(
                "Workflow 'actions' must be a list".into(),
            ));
        };
        let name = map
            .get("name")
            .filter(|n| py_truthy(n))
            .map(py_str)
            .or_else(|| {
                map.get("trigger")
                    .and_then(|t| t.get("id"))
                    .filter(|n| py_truthy(n))
                    .map(py_str)
            })
            .unwrap_or_else(|| "declarative_workflow".into());
        let description = map
            .get("description")
            .and_then(Json::as_str)
            .map(str::to_string);

        let mut agents = self.agents.clone();
        if let Some(Json::Object(defs)) = map.get("agents") {
            for (agent_name, agent_def) in defs {
                if agents.contains_key(agent_name) {
                    continue;
                }
                let agent = make_agent(agent_def, base_path).map_err(|e| {
                    DeclarativeError::Invalid(format!("Failed to create agent '{agent_name}': {e}"))
                })?;
                agents.insert(agent_name.clone(), agent);
            }
        }

        let max_iterations = match self.max_iterations {
            Some(m) => Some(m),
            None => match map.get("maxTurns") {
                None | Some(Json::Null) => None,
                Some(v) => match v.as_u64().filter(|n| *n > 0) {
                    Some(n) => Some(n as usize),
                    None => {
                        return Err(DeclarativeError::Invalid(format!(
                        "Invalid max_iterations/maxTurns value: {v}. Expected a positive integer."
                    )))
                    }
                },
            },
        };
        if self.max_iterations == Some(0) {
            return Err(DeclarativeError::Invalid(
                "Invalid max_iterations/maxTurns value: 0. Expected a positive integer.".into(),
            ));
        }

        let env = EnvConfig {
            values: self.configuration.clone(),
            restrict_to_configuration: self.restrict_env_to_configuration,
            referenced_names: discover_env_references(def),
            source: self.env_source.clone(),
        };
        let rt = Arc::new(executor::Runtime {
            agents,
            tools: self.tools.clone(),
            http: self.http.clone(),
            mcp: self.mcp.clone(),
            state: StateConfig {
                env,
                budget: self.state_budget,
                engine: Engine::with_limits(self.expression_limits),
            },
        });
        builder::build_graph(
            &actions,
            raw_actions,
            rt,
            builder::BuildOptions {
                name,
                description,
                max_iterations,
                checkpoint_storage: self.checkpoint_storage.clone(),
                strict_actions: self.strict_actions,
            },
        )
    }
}

/// Parse YAML, keeping the raw tree (for mapping key order).
pub(crate) fn parse_yaml_raw(yaml: &str) -> Result<serde_yaml::Value> {
    serde_yaml::from_str(yaml).map_err(|e| DeclarativeError::Parse(format!("Invalid YAML: {e}")))
}

/// Build an agent from a workflow `agents:` entry (upstream
/// `_create_agent_from_def`).
pub(crate) fn agent_from_definition(
    loader: &DeclarativeLoader,
    def: &Json,
    base: Option<&Path>,
) -> Result<Arc<dyn SupportsAgentRun>> {
    if let Some(file) = def.get("file").and_then(Json::as_str) {
        let mut path = PathBuf::from(file);
        if let (Some(base), false) = (base, path.is_absolute()) {
            path = base.join(path);
        }
        let yaml = std::fs::read_to_string(&path).map_err(|e| {
            DeclarativeError::Invalid(format!("cannot read agent file {}: {e}", path.display()))
        })?;
        return Ok(Arc::new(loader.load_agent(&yaml)?));
    }
    if def.get("kind").is_some() {
        let yaml =
            serde_yaml::to_string(def).map_err(|e| DeclarativeError::Serialize(e.to_string()))?;
        return Ok(Arc::new(loader.load_agent(&yaml)?));
    }
    if def.get("connection").is_some() {
        return Err(DeclarativeError::Invalid(
            "Connection-based agents must be provided via the agent registry. Create the agent \
             using the appropriate client and register it on the factory."
                .into(),
        ));
    }
    Err(DeclarativeError::Invalid(format!(
        "Invalid agent definition. Expected 'file', 'kind', or 'connection': {def}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_upstream_shape() {
        assert!(is_upstream_workflow(
            &json!({"kind": "Workflow", "trigger": {}})
        ));
        assert!(is_upstream_workflow(&json!({"actions": []})));
        assert!(!is_upstream_workflow(
            &json!({"kind": "Workflow", "type": "sequential"})
        ));
    }

    #[test]
    fn yaml_conversion_stringifies_keys() {
        let v: serde_yaml::Value = serde_yaml::from_str("1: a\ntrue: b\nc: {Local.X}").unwrap();
        assert_eq!(
            yaml_to_json(v),
            json!({"1": "a", "True": "b", "c": {"Local.X": null}})
        );
    }

    #[test]
    fn input_schema_conversion() {
        let schema = inputs_json_schema(&json!({
            "age": {"type": "int", "description": "d"},
            "name": "str",
            "opt": {"type": "list", "required": false}
        }));
        assert_eq!(
            schema,
            json!({
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "description": "d"},
                    "name": {"type": "string"},
                    "opt": {"type": "array"}
                },
                "required": ["age", "name"]
            })
        );
    }
}
