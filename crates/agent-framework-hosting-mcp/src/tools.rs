//! Agent- and workflow-backed MCP tools. Ports upstream's `AgentMCPTool` and
//! `WorkflowMCPTool`.

use std::sync::Arc;

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::workflow::Workflow;
use agent_framework_hosting::{AgentState, WorkflowState};
use async_trait::async_trait;
use serde_json::{json, Map, Value};

use crate::conversion::{mcp_from_run, mcp_to_run};
use crate::McpHostError;

/// Something an [`McpServer`](crate::McpServer) can list and call.
///
/// Implemented by [`AgentMcpTool`] and [`WorkflowMcpTool`]; implement it to
/// serve any other tool from the same server.
#[async_trait]
pub trait McpToolProvider: Send + Sync {
    /// The MCP `Tool` definitions this provider serves (`name`,
    /// `description`, `inputSchema`, ...).
    async fn list_tools(&self) -> Result<Vec<Value>, McpHostError>;

    /// Run the tool `name` with `arguments`, returning its content blocks.
    ///
    /// Returns [`McpHostError::UnknownTool`] for a name this provider does
    /// not serve, so a server can try the next provider.
    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<&Map<String, Value>>,
    ) -> Result<Vec<Value>, McpHostError>;
}

/// Replace every run of characters outside `[A-Za-z0-9_.-]` with `_` and
/// trim `_` from the ends, as upstream derives a tool name.
fn sanitize_tool_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut in_run = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
            out.push(c);
            in_run = false;
        } else if !in_run {
            out.push('_');
            in_run = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// Expose one agent as one MCP tool. Mirrors upstream `AgentMCPTool`.
///
/// The tool takes one required string argument (`task` by default) as the
/// request, plus any extra `parameters` the application declares. Arguments
/// declared as `chat_option_parameters` are forwarded as chat options (see
/// [`mcp_to_run`]). With a `session_id_parameter`, that argument keys an
/// [`AgentState`] session, so repeated calls with the same value continue
/// one conversation.
///
/// The adapter generates the schema *and* parses calls against it, so the
/// two cannot drift.
pub struct AgentMcpTool {
    state: Arc<AgentState>,
    name: Option<String>,
    description: Option<String>,
    argument_name: String,
    argument_description: Option<String>,
    parameters: Map<String, Value>,
    chat_option_parameters: Map<String, Value>,
    required: Vec<String>,
    session_id_parameter: Option<String>,
}

impl AgentMcpTool {
    /// A tool for `agent` with default settings.
    pub fn new(agent: Arc<dyn SupportsAgentRun>) -> Self {
        Self::from_state(Arc::new(AgentState::new(agent)))
    }

    /// A tool over an existing [`AgentState`] (sharing its session store and
    /// target resolution).
    pub fn from_state(state: Arc<AgentState>) -> Self {
        Self {
            state,
            name: None,
            description: None,
            argument_name: "task".into(),
            argument_description: None,
            parameters: Map::new(),
            chat_option_parameters: Map::new(),
            required: Vec::new(),
            session_id_parameter: None,
        }
    }

    /// The tool name. Defaults to the agent's name, sanitized.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The tool description. Defaults to empty.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// The request argument's name (default `task`).
    pub fn argument_name(mut self, name: impl Into<String>) -> Self {
        self.argument_name = name.into();
        self
    }

    /// The request argument's description (default `Task for <tool>`).
    pub fn argument_description(mut self, description: impl Into<String>) -> Self {
        self.argument_description = Some(description.into());
        self
    }

    /// An extra application-owned argument and its JSON schema. Its value is
    /// kept on the message's `mcp_arguments`, not forwarded to the model.
    pub fn parameter(mut self, name: impl Into<String>, schema: Value) -> Self {
        self.parameters.insert(name.into(), schema);
        self
    }

    /// An argument forwarded as a chat option, with its JSON schema.
    pub fn chat_option_parameter(mut self, name: impl Into<String>, schema: Value) -> Self {
        self.chat_option_parameters.insert(name.into(), schema);
        self
    }

    /// Mark an extra or chat-option argument as required.
    pub fn required(mut self, name: impl Into<String>) -> Self {
        self.required.push(name.into());
        self
    }

    /// Key sessions by the extra string argument `name` (which is then
    /// required): calls carrying the same value continue one conversation.
    pub fn session_id_parameter(mut self, name: impl Into<String>) -> Self {
        self.session_id_parameter = Some(name.into());
        self
    }

    /// Check the configuration, as upstream's constructor does: names must
    /// not overlap, and required and session parameters must be declared.
    pub fn validate(&self) -> Result<(), McpHostError> {
        let config = |m: &str| Err(McpHostError::Configuration(m.to_string()));
        if self.parameters.contains_key(&self.argument_name)
            || self
                .chat_option_parameters
                .contains_key(&self.argument_name)
        {
            return config(&format!(
                "Main argument '{}' must not be repeated in additional parameters.",
                self.argument_name
            ));
        }
        if self
            .parameters
            .keys()
            .any(|k| self.chat_option_parameters.contains_key(k))
        {
            return config(
                "Additional parameters and chat option parameters must have distinct names.",
            );
        }
        if let Some(sid) = &self.session_id_parameter {
            if !self.parameters.contains_key(sid) {
                return config("session_id_parameter must name an additional parameter.");
            }
        }
        let undefined: Vec<&String> = self
            .required
            .iter()
            .filter(|r| {
                !self.parameters.contains_key(*r) && !self.chat_option_parameters.contains_key(*r)
            })
            .collect();
        if !undefined.is_empty() {
            return config(&format!(
                "Required parameters are not defined: {undefined:?}"
            ));
        }
        Ok(())
    }

    fn tool_name(&self, agent: &dyn SupportsAgentRun) -> Result<String, McpHostError> {
        if let Some(name) = &self.name {
            return Ok(name.clone());
        }
        let raw = agent.name().ok_or_else(|| {
            McpHostError::Configuration(
                "MCP tool name requires either an override or an agent name.".into(),
            )
        })?;
        let name = sanitize_tool_name(raw);
        Ok(if name.is_empty() {
            "agent".into()
        } else {
            name
        })
    }

    fn definition(&self, agent: &dyn SupportsAgentRun) -> Result<Value, McpHostError> {
        self.validate()?;
        let name = self.tool_name(agent)?;
        let mut properties = Map::new();
        properties.insert(
            self.argument_name.clone(),
            json!({
                "type": "string",
                "description": self
                    .argument_description
                    .clone()
                    .unwrap_or_else(|| format!("Task for {name}")),
            }),
        );
        properties.extend(self.parameters.clone());
        properties.extend(self.chat_option_parameters.clone());

        // Declaration order, as upstream: extra parameters, then options.
        let mut required = vec![Value::String(self.argument_name.clone())];
        for key in self
            .parameters
            .keys()
            .chain(self.chat_option_parameters.keys())
        {
            let is_required = self.required.contains(key)
                || self.session_id_parameter.as_deref() == Some(key.as_str());
            if is_required {
                required.push(Value::String(key.clone()));
            }
        }
        Ok(json!({
            "name": name,
            "description": self.description.clone().unwrap_or_default(),
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": false,
            },
        }))
    }
}

#[async_trait]
impl McpToolProvider for AgentMcpTool {
    async fn list_tools(&self) -> Result<Vec<Value>, McpHostError> {
        let agent = self.state.get_target().await?;
        Ok(vec![self.definition(&*agent)?])
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<&Map<String, Value>>,
    ) -> Result<Vec<Value>, McpHostError> {
        let agent = self.state.get_target().await?;
        self.validate()?;
        if name != self.tool_name(&*agent)? {
            return Err(McpHostError::UnknownTool(name.to_string()));
        }
        let option_names: Vec<&str> = self
            .chat_option_parameters
            .keys()
            .map(String::as_str)
            .collect();
        let run = mcp_to_run(arguments, &self.argument_name, &option_names)?;

        let Some(sid_param) = &self.session_id_parameter else {
            let response = agent
                .run_with_options(run.messages, None, run.options)
                .await?;
            return mcp_from_run(&response);
        };
        let session_id = arguments
            .and_then(|a| a.get(sid_param))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                McpHostError::InvalidArguments(format!(
                    "MCP tool argument '{sid_param}' must be a non-empty string."
                ))
            })?;
        let mut session = self.state.get_or_create_session(session_id).await?;
        let response = agent
            .run_with_options(run.messages, Some(&mut session), run.options)
            .await?;
        self.state.set_session(session_id, &session).await?;
        mcp_from_run(&response)
    }
}

/// Expose one workflow as one MCP tool. Mirrors upstream `WorkflowMCPTool`.
///
/// Upstream derives the input schema from the start executor's Python input
/// type. Rust workflow messages are untyped JSON, so the schema is declared
/// with [`input_schema`](Self::input_schema) (default `{"type": "string"}`).
/// An object schema is used as the tool's whole input; any other is wrapped
/// as the single property `argument_name` (default `input`).
///
/// A run that pauses on a human-in-the-loop request is an error, as
/// upstream: this adapter does not manage continuation.
pub struct WorkflowMcpTool {
    state: Arc<WorkflowState>,
    name: Option<String>,
    description: Option<String>,
    argument_name: String,
    input_schema: Value,
}

impl WorkflowMcpTool {
    /// A tool for `workflow`.
    pub fn new(workflow: Workflow) -> Self {
        Self::from_state(Arc::new(WorkflowState::new(workflow)))
    }

    /// A tool over an existing [`WorkflowState`].
    pub fn from_state(state: Arc<WorkflowState>) -> Self {
        Self {
            state,
            name: None,
            description: None,
            argument_name: "input".into(),
            input_schema: json!({ "type": "string" }),
        }
    }

    /// The tool name. Defaults to the workflow's name (or id), sanitized.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The tool description. Defaults to the workflow's description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// The property name wrapping a non-object input (default `input`).
    pub fn argument_name(mut self, name: impl Into<String>) -> Self {
        self.argument_name = name.into();
        self
    }

    /// The JSON schema of the workflow's start input.
    pub fn input_schema(mut self, schema: Value) -> Self {
        self.input_schema = schema;
        self
    }

    fn is_object_input(&self) -> bool {
        self.input_schema.get("type").and_then(Value::as_str) == Some("object")
    }

    fn tool_name(&self, workflow: &Workflow) -> String {
        if let Some(name) = &self.name {
            return name.clone();
        }
        let name = sanitize_tool_name(workflow.name().unwrap_or(workflow.id()));
        if name.is_empty() {
            "workflow".into()
        } else {
            name
        }
    }

    fn definition(&self, workflow: &Workflow) -> Value {
        let schema = if self.is_object_input() {
            self.input_schema.clone()
        } else {
            json!({
                "type": "object",
                "properties": { self.argument_name.clone(): self.input_schema },
                "required": [self.argument_name],
                "additionalProperties": false,
            })
        };
        json!({
            "name": self.tool_name(workflow),
            "description": self
                .description
                .clone()
                .or_else(|| workflow.description().map(str::to_string))
                .unwrap_or_default(),
            "inputSchema": schema,
        })
    }
}

#[async_trait]
impl McpToolProvider for WorkflowMcpTool {
    async fn list_tools(&self) -> Result<Vec<Value>, McpHostError> {
        let workflow = self.state.get_target().await?;
        Ok(vec![self.definition(&workflow)])
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: Option<&Map<String, Value>>,
    ) -> Result<Vec<Value>, McpHostError> {
        let workflow = self.state.get_target().await?;
        if name != self.tool_name(&workflow) {
            return Err(McpHostError::UnknownTool(name.to_string()));
        }
        let input = if self.is_object_input() {
            Value::Object(arguments.cloned().unwrap_or_default())
        } else {
            arguments
                .and_then(|a| a.get(&self.argument_name))
                .cloned()
                .ok_or_else(|| {
                    McpHostError::InvalidArguments(format!(
                        "MCP tool arguments must include '{}'.",
                        self.argument_name
                    ))
                })?
        };
        let run = workflow.run(input).await?;
        if !run.pending_requests().is_empty() {
            return Err(McpHostError::InvalidArguments(
                "The workflow requires external input. WorkflowMcpTool does not manage \
                 human-in-the-loop continuation; handle it in the application contract."
                    .into(),
            ));
        }
        let mut blocks = Vec::new();
        for output in run.outputs() {
            match output {
                Value::String(text) => blocks.push(json!({ "type": "text", "text": text })),
                other => {
                    // An agent response or message a workflow yielded is
                    // converted as one; anything else is its JSON text.
                    if let Ok(resp) = serde_json::from_value::<
                        agent_framework_core::types::AgentResponse,
                    >(other.clone())
                    {
                        if !resp.messages.is_empty() {
                            blocks.extend(mcp_from_run(&resp)?);
                            continue;
                        }
                    }
                    blocks.push(json!({ "type": "text", "text": other.to_string() }));
                }
            }
        }
        Ok(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_sanitized_as_upstream() {
        assert_eq!(sanitize_tool_name("Weather Bot!"), "Weather_Bot");
        assert_eq!(sanitize_tool_name("a.b-c_d"), "a.b-c_d");
        assert_eq!(sanitize_tool_name("??"), "");
        assert_eq!(sanitize_tool_name("x  y"), "x_y");
    }
}
