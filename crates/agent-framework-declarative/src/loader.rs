//! The [`DeclarativeLoader`]: parse specs, interpolate environment variables,
//! and build [`Agent`]s and [`Workflow`]s using the core builders.

use std::sync::Arc;

use agent_framework_core::agent::Agent;
use agent_framework_core::tools::{
    empty_schema, hosted_code_interpreter, hosted_file_search, hosted_mcp, hosted_web_search,
    ApprovalMode, ToolDefinition, ToolKind,
};
use agent_framework_core::types::{ChatOptions, ResponseFormat, ToolMode};
use agent_framework_core::workflow::{
    AgentExecutor, Case, ConcurrentBuilder, Default as SwitchDefault, GroupChatBuilder,
    HandoffBuilder, SequentialBuilder, Workflow, WorkflowBuilder,
};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::agent::{AgentSpec, ApprovalModeSpec, ModelOptions, ToolSpec};
use crate::env::{EnvSource, ProcessEnv};
use crate::error::{DeclarativeError, Result};
use crate::registry::{AgentRegistry, ChatClientFactory, PredicateRegistry, ToolRegistry};
use crate::workflow::{OrchestrationType, WorkflowSpec};

/// Loads declarative agent and workflow specs, resolving providers, tools, and
/// referenced agents through the registries it holds.
///
/// ```no_run
/// use agent_framework_declarative::{ChatClientFactory, DeclarativeLoader};
/// # use std::sync::Arc;
/// # use agent_framework_core::prelude::*;
/// # fn make_client() -> Arc<dyn ChatClient> { unimplemented!() }
/// let loader = DeclarativeLoader::new().with_client_factory(
///     ChatClientFactory::new().with("OpenAI.Chat", |_model| Ok(make_client())),
/// );
/// let agent = loader.load_agent("kind: Prompt\nname: A\nmodel:\n  id: gpt-4\n  provider: OpenAI\n  apiType: Chat")?;
/// # let _ = agent;
/// # Ok::<(), agent_framework_declarative::DeclarativeError>(())
/// ```
pub struct DeclarativeLoader {
    clients: ChatClientFactory,
    tools: ToolRegistry,
    predicates: PredicateRegistry,
    env: Box<dyn EnvSource + Send + Sync>,
}

impl Default for DeclarativeLoader {
    fn default() -> Self {
        Self {
            clients: ChatClientFactory::new(),
            tools: ToolRegistry::new(),
            predicates: PredicateRegistry::new(),
            env: Box::new(ProcessEnv),
        }
    }
}

impl DeclarativeLoader {
    /// Create a loader with empty registries and the process environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chat-client factory registry.
    pub fn with_client_factory(mut self, factory: ChatClientFactory) -> Self {
        self.clients = factory;
        self
    }

    /// Set the native-tool registry.
    pub fn with_tool_registry(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    /// Set the predicate registry consulted for workflow `predicate:` edges.
    pub fn with_predicates(mut self, predicates: PredicateRegistry) -> Self {
        self.predicates = predicates;
        self
    }

    /// Use a custom environment source for `${VAR}` interpolation (e.g. a fixed
    /// map in tests).
    pub fn with_env<E: EnvSource + Send + Sync + 'static>(mut self, env: E) -> Self {
        self.env = Box::new(env);
        self
    }

    /// Mutable access to the client-factory registry.
    pub fn client_factory_mut(&mut self) -> &mut ChatClientFactory {
        &mut self.clients
    }

    /// Mutable access to the tool registry.
    pub fn tool_registry_mut(&mut self) -> &mut ToolRegistry {
        &mut self.tools
    }

    /// Mutable access to the predicate registry.
    pub fn predicate_registry_mut(&mut self) -> &mut PredicateRegistry {
        &mut self.predicates
    }

    // --- parsing ---------------------------------------------------------

    fn parse_interpolated<T: DeserializeOwned>(&self, yaml: &str) -> Result<T> {
        let mut value: serde_yaml::Value =
            serde_yaml::from_str(yaml).map_err(|e| DeclarativeError::Parse(e.to_string()))?;
        crate::env::interpolate_value(&mut value, self.env.as_ref())?;
        serde_yaml::from_value(value).map_err(|e| DeclarativeError::Parse(e.to_string()))
    }

    /// Parse an [`AgentSpec`] from YAML with environment interpolation applied.
    pub fn load_agent_spec(&self, yaml: &str) -> Result<AgentSpec> {
        self.parse_interpolated(yaml)
    }

    /// Parse a [`WorkflowSpec`] from YAML with environment interpolation applied.
    pub fn load_workflow_spec(&self, yaml: &str) -> Result<WorkflowSpec> {
        self.parse_interpolated(yaml)
    }

    // --- agents ----------------------------------------------------------

    /// Parse and build a [`Agent`] from a YAML agent spec.
    pub fn load_agent(&self, yaml: &str) -> Result<Agent> {
        let spec = self.load_agent_spec(yaml)?;
        self.build_agent(&spec)
    }

    /// Build a [`Agent`] from an already-parsed [`AgentSpec`].
    pub fn build_agent(&self, spec: &AgentSpec) -> Result<Agent> {
        spec.validate_kind()?;
        let client = self.clients.resolve(spec.model.as_ref())?;

        let mut options = ChatOptions::new();
        if let Some(model) = &spec.model {
            if let Some(id) = &model.id {
                options.model = Some(id.clone());
            }
            if let Some(model_options) = &model.options {
                apply_options(&mut options, model_options)?;
            }
        }
        if let Some(output_schema) = &spec.output_schema {
            let name = spec.effective_name().unwrap_or("response").to_string();
            options.response_format = Some(ResponseFormat::JsonSchema {
                name,
                description: None,
                schema: output_schema.to_json_schema(),
                strict: Some(output_schema.strict),
            });
        }

        let mut tool_defs = Vec::with_capacity(spec.tools.len());
        for tool in &spec.tools {
            tool_defs.push(self.build_tool(tool)?);
        }

        let mut builder = Agent::builder(client);
        if let Some(name) = spec.effective_name() {
            builder = builder.name(name.to_string());
        }
        if let Some(description) = &spec.description {
            builder = builder.description(description.clone());
        }
        if let Some(instructions) = spec.combined_instructions() {
            builder = builder.instructions(instructions);
        }
        builder = builder.chat_options(options);
        for def in tool_defs {
            builder = builder.tool(def);
        }
        Ok(builder.build())
    }

    // --- tools -----------------------------------------------------------

    fn build_tool(&self, spec: &ToolSpec) -> Result<ToolDefinition> {
        match spec.kind.to_ascii_lowercase().as_str() {
            "function" => self.build_function_tool(spec),
            "custom" => self.build_custom_tool(spec),
            "web_search" => {
                let mut def = hosted_web_search();
                override_meta(&mut def, spec);
                Ok(def)
            }
            "file_search" => {
                let mut def = hosted_file_search(spec.maximum_result_count);
                override_meta(&mut def, spec);
                // Providers (e.g. the OpenAI Responses client) read the
                // required store ids from the definition's parameters.
                if let Some(ids) = spec.vector_store_ids.as_deref().filter(|v| !v.is_empty()) {
                    def.parameters = serde_json::json!({ "vector_store_ids": ids });
                }
                Ok(def)
            }
            "code_interpreter" => {
                let mut def = hosted_code_interpreter();
                override_meta(&mut def, spec);
                Ok(def)
            }
            "mcp" => self.build_mcp_tool(spec),
            _ => Err(DeclarativeError::UnsupportedKind {
                what: "tool",
                kind: spec.kind.clone(),
                expected: vec![
                    "function",
                    "custom",
                    "web_search",
                    "file_search",
                    "code_interpreter",
                    "mcp",
                ],
            }),
        }
    }

    fn build_function_tool(&self, spec: &ToolSpec) -> Result<ToolDefinition> {
        let name = spec
            .name
            .clone()
            .ok_or_else(|| DeclarativeError::missing_field(tool_context(spec), "name"))?;
        let description = spec.description.clone().unwrap_or_default();
        let parameters = spec.parameters.as_ref().map(|p| p.to_json_schema());

        match self.resolve_binding(spec) {
            // Bound to a native tool: expose the spec's identity, execute the
            // native closure.
            Some(native) => Ok(ToolDefinition {
                name,
                description: if description.is_empty() {
                    native.description.clone()
                } else {
                    description
                },
                parameters: parameters.unwrap_or_else(|| native.parameters.clone()),
                kind: ToolKind::Function,
                approval_mode: native.approval_mode,
                executor: native.executor.clone(),
            }),
            // Unbound: a non-executable declaration.
            None => Ok(ToolDefinition {
                name,
                description,
                parameters: parameters.unwrap_or_else(empty_schema),
                kind: ToolKind::Function,
                approval_mode: ApprovalMode::NeverRequire,
                executor: None,
            }),
        }
    }

    fn build_custom_tool(&self, spec: &ToolSpec) -> Result<ToolDefinition> {
        let name = spec
            .name
            .clone()
            .ok_or_else(|| DeclarativeError::missing_field(tool_context(spec), "name"))?;
        match self.resolve_binding(spec) {
            Some(native) => Ok(ToolDefinition {
                name,
                description: spec
                    .description
                    .clone()
                    .unwrap_or_else(|| native.description.clone()),
                parameters: native.parameters.clone(),
                kind: ToolKind::Function,
                approval_mode: native.approval_mode,
                executor: native.executor.clone(),
            }),
            None => Err(DeclarativeError::Invalid(format!(
                "custom tool {name:?} requires a native tool binding registered in the ToolRegistry"
            ))),
        }
    }

    fn build_mcp_tool(&self, spec: &ToolSpec) -> Result<ToolDefinition> {
        let name = spec
            .name
            .clone()
            .ok_or_else(|| DeclarativeError::missing_field(tool_context(spec), "name"))?;
        let url = spec
            .url
            .clone()
            .ok_or_else(|| DeclarativeError::missing_field(tool_context(spec), "url"))?;
        let mut def = hosted_mcp(name, url, spec.allowed_tools.clone());
        if let Some(description) = &spec.description {
            def.description = description.clone();
        }
        if let Some(approval) = &spec.approval_mode {
            def.approval_mode = mcp_approval_mode(approval);
        }
        Ok(def)
    }

    /// Resolve a native tool for a function/custom spec: try each binding key,
    /// then the tool's own name.
    fn resolve_binding(&self, spec: &ToolSpec) -> Option<ToolDefinition> {
        if let Some(bindings) = &spec.bindings {
            for key in bindings.keys() {
                if let Some(tool) = self.tools.get(key) {
                    return Some(tool.clone());
                }
            }
        }
        spec.name
            .as_deref()
            .and_then(|name| self.tools.get(name).cloned())
    }

    // --- workflows -------------------------------------------------------

    /// Parse and build a [`Workflow`] from a YAML workflow document.
    ///
    /// Dispatches on the document's shape:
    ///
    /// * documents with `trigger` or `actions` are **upstream-format**
    ///   declarative workflows (PowerFx actions) and are built by a
    ///   [`WorkflowFactory`](crate::flow::WorkflowFactory) pre-populated by
    ///   [`Self::workflow_factory`]; `${VAR}` interpolation is not applied to
    ///   them (upstream uses `=Env.VAR`);
    /// * anything else is this crate's Rust-native [`WorkflowSpec`].
    ///
    /// Agents are resolved from `agents`; upstream-format workflows may also
    /// define agents inline under `agents:`, built with this loader.
    pub fn load_workflow(&self, yaml: &str, agents: &AgentRegistry) -> Result<Workflow> {
        let raw = crate::flow::parse_yaml_raw(yaml)?;
        let def = crate::flow::yaml_to_json(raw.clone());
        if crate::flow::is_upstream_workflow(&def) {
            return self
                .workflow_factory(agents)
                .build(&def, Some(&raw), None, &|def, base| {
                    crate::flow::agent_from_definition(self, def, base)
                });
        }
        let spec = self.load_workflow_spec(yaml)?;
        self.build_workflow(&spec, agents)
    }

    /// A [`WorkflowFactory`](crate::flow::WorkflowFactory) for
    /// upstream-format workflows, pre-populated with `agents` and every
    /// executable tool in this loader's [`ToolRegistry`] (for
    /// `InvokeFunctionTool`). Add HTTP/MCP handlers, `Env` configuration,
    /// checkpointing, or limits on the returned factory.
    pub fn workflow_factory(&self, agents: &AgentRegistry) -> crate::flow::WorkflowFactory {
        let mut factory = crate::flow::WorkflowFactory::new();
        for (id, agent) in agents.iter() {
            factory.register_agent(id, agent.clone());
        }
        for (name, tool) in self.tools.iter() {
            if let Some(exec) = &tool.executor {
                factory = factory.with_tool(name, exec.clone());
            }
        }
        factory
    }

    /// Build a [`Workflow`] from an already-parsed [`WorkflowSpec`].
    pub fn build_workflow(&self, spec: &WorkflowSpec, agents: &AgentRegistry) -> Result<Workflow> {
        spec.validate_kind()?;
        match spec.r#type {
            Some(kind) => self.build_orchestration(spec, agents, kind),
            None => self.build_graph(spec, agents),
        }
    }

    fn build_orchestration(
        &self,
        spec: &WorkflowSpec,
        agents: &AgentRegistry,
        kind: OrchestrationType,
    ) -> Result<Workflow> {
        if spec.participants.is_empty() {
            return Err(DeclarativeError::Invalid(
                "orchestration workflow requires a non-empty 'participants' list".into(),
            ));
        }
        let resolved: Vec<(
            String,
            Arc<dyn agent_framework_core::agent::SupportsAgentRun>,
        )> = spec
            .participants
            .iter()
            .map(|id| Ok((id.clone(), agents.get(id)?)))
            .collect::<Result<_>>()?;

        let workflow = match kind {
            OrchestrationType::Sequential => {
                let mut builder =
                    SequentialBuilder::new().participants(resolved.iter().map(|(_, a)| a.clone()));
                if let Some(name) = &spec.name {
                    builder = builder.name(name.clone());
                }
                builder.build()?
            }
            OrchestrationType::Concurrent => {
                let mut builder =
                    ConcurrentBuilder::new().participants(resolved.iter().map(|(_, a)| a.clone()));
                if let Some(name) = &spec.name {
                    builder = builder.name(name.clone());
                }
                builder.build()?
            }
            OrchestrationType::GroupChat => {
                let mut builder = GroupChatBuilder::new();
                for (id, agent) in &resolved {
                    builder = builder.participant(id.clone(), agent.clone());
                }
                if spec.round_robin.unwrap_or(true) {
                    builder = builder.round_robin();
                } else {
                    return Err(DeclarativeError::Invalid(
                        "group_chat shorthand requires round_robin: true (an LLM manager cannot be \
                         expressed declaratively); use builders directly for a custom manager"
                            .into(),
                    ));
                }
                if let Some(max_rounds) = spec.max_rounds {
                    builder = builder.max_rounds(max_rounds);
                }
                if let Some(name) = &spec.name {
                    builder = builder.name(name.clone());
                }
                builder.build()?
            }
            OrchestrationType::Handoff => {
                let mut builder = HandoffBuilder::new();
                for (id, agent) in &resolved {
                    builder = builder.participant(id.clone(), agent.clone());
                }
                let initial = spec.start.clone().unwrap_or_else(|| resolved[0].0.clone());
                builder = builder.initial_agent(initial);
                for edge in &spec.handoffs {
                    builder = builder.add_handoff(edge.from.clone()).to(edge.to.clone());
                }
                if spec.autonomous.unwrap_or(true) {
                    builder = builder.autonomous();
                }
                if let Some(max_iterations) = spec.max_iterations {
                    builder = builder.max_iterations(max_iterations);
                }
                if let Some(name) = &spec.name {
                    builder = builder.name(name.clone());
                }
                builder.build()?
            }
        };
        Ok(workflow)
    }

    fn build_graph(&self, spec: &WorkflowSpec, agents: &AgentRegistry) -> Result<Workflow> {
        if spec.nodes.is_empty() {
            return Err(DeclarativeError::Invalid(
                "workflow must set 'type' (orchestration shorthand) or define 'nodes' (graph)"
                    .into(),
            ));
        }
        let mut builder = WorkflowBuilder::new();
        for node in &spec.nodes {
            let agent = agents.get(&node.agent)?;
            let executor = AgentExecutor::new(node.id.clone(), agent).with_output(node.output);
            builder = builder.add_executor(Arc::new(executor));
        }

        let start = spec
            .start
            .clone()
            .ok_or_else(|| DeclarativeError::Invalid("graph workflow requires 'start'".into()))?;
        builder = builder.set_start(start);

        for edge in &spec.edges {
            match self.build_condition(edge.condition.as_deref(), edge.predicate.as_deref())? {
                Some(condition) => {
                    builder = builder.add_conditional_edge_async(
                        edge.from.clone(),
                        edge.to.clone(),
                        move |value: &Value| condition(value),
                    );
                }
                None => builder = builder.add_edge(edge.from.clone(), edge.to.clone()),
            }
        }
        for fan_out in &spec.fan_out {
            builder = builder.add_fan_out(fan_out.from.clone(), fan_out.to.clone());
        }
        for fan_in in &spec.fan_in {
            builder = builder.add_fan_in(fan_in.from.clone(), fan_in.to.clone());
        }
        for switch in &spec.switch {
            let mut cases = Vec::with_capacity(switch.cases.len());
            for case in &switch.cases {
                let condition = self
                    .build_condition(case.condition.as_deref(), case.predicate.as_deref())?
                    .ok_or_else(|| {
                        DeclarativeError::Invalid(
                            "switch case requires 'condition' or 'predicate'".into(),
                        )
                    })?;
                let built = match &case.label {
                    Some(label) => Case::labeled_async(
                        move |v: &Value| condition(v),
                        case.to.clone(),
                        label.clone(),
                    ),
                    None => Case::new_async(move |v: &Value| condition(v), case.to.clone()),
                };
                cases.push(built);
            }
            builder = builder.add_switch(
                switch.from.clone(),
                cases,
                SwitchDefault::new(switch.default.clone()),
            );
        }

        if let Some(max_iterations) = spec.max_iterations {
            builder = builder.set_max_iterations(max_iterations);
        }
        if let Some(name) = &spec.name {
            builder = builder.name(name.clone());
        }
        Ok(builder.build()?)
    }

    fn build_condition(
        &self,
        condition: Option<&str>,
        predicate: Option<&str>,
    ) -> Result<Option<agent_framework_core::workflow::Condition>> {
        match (condition, predicate) {
            (Some(_), Some(_)) => Err(DeclarativeError::Invalid(
                "edge/case specifies both 'condition' and 'predicate'; choose one".into(),
            )),
            (Some(expr), None) => Ok(Some(crate::condition::parse(expr)?)),
            (None, Some(name)) => Ok(Some(self.predicates.get(name)?)),
            (None, None) => Ok(None),
        }
    }
}

/// Apply spec [`ModelOptions`] onto a [`ChatOptions`], mirroring the upstream
/// `_parse_chat_options` mapping.
fn apply_options(options: &mut ChatOptions, model_options: &ModelOptions) -> Result<()> {
    if let Some(v) = model_options.temperature {
        options.temperature = Some(v);
    }
    if let Some(v) = model_options.top_p {
        options.top_p = Some(v);
    }
    if let Some(v) = model_options.frequency_penalty {
        options.frequency_penalty = Some(v);
    }
    if let Some(v) = model_options.presence_penalty {
        options.presence_penalty = Some(v);
    }
    if let Some(v) = model_options.max_output_tokens {
        options.max_tokens = Some(v);
    }
    if let Some(v) = model_options.seed {
        options.seed = Some(v);
    }
    if let Some(v) = &model_options.stop_sequences {
        options.stop = Some(v.clone());
    }
    if let Some(v) = model_options.allow_multiple_tool_calls {
        options.allow_multiple_tool_calls = Some(v);
    }
    if let Some(v) = model_options.top_k {
        options
            .additional_properties
            .insert("top_k".to_string(), json!(v));
    }
    if let Some(mode) = &model_options.chat_tool_mode {
        options.tool_choice = Some(match mode.to_ascii_lowercase().as_str() {
            "auto" => ToolMode::Auto,
            "required" => ToolMode::required_any(),
            "none" => ToolMode::None,
            other => {
                return Err(DeclarativeError::Invalid(format!(
                    "unknown chatToolMode {other:?}; expected 'auto', 'required', or 'none'"
                )))
            }
        });
    }
    for (key, value) in &model_options.additional_properties {
        options
            .additional_properties
            .insert(key.clone(), value.clone());
    }
    Ok(())
}

/// Overwrite a tool definition's name/description from the spec when present.
fn override_meta(def: &mut ToolDefinition, spec: &ToolSpec) {
    if let Some(name) = &spec.name {
        def.name = name.clone();
    }
    if let Some(description) = &spec.description {
        def.description = description.clone();
    }
}

/// Map an [`ApprovalModeSpec`] onto the core binary [`ApprovalMode`].
fn mcp_approval_mode(approval: &ApprovalModeSpec) -> ApprovalMode {
    match approval.kind().map(str::to_ascii_lowercase).as_deref() {
        Some("always") => ApprovalMode::AlwaysRequire,
        Some("specify") => match approval {
            ApprovalModeSpec::Detailed(detail)
                if detail
                    .always_require_approval_tools
                    .as_ref()
                    .is_some_and(|tools| !tools.is_empty()) =>
            {
                ApprovalMode::AlwaysRequire
            }
            _ => ApprovalMode::NeverRequire,
        },
        _ => ApprovalMode::NeverRequire,
    }
}

fn tool_context(spec: &ToolSpec) -> String {
    match &spec.name {
        Some(name) => format!("tool {name:?} (kind: {})", spec.kind),
        None => format!("tool (kind: {})", spec.kind),
    }
}
