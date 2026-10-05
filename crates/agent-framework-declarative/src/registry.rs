//! Registries the loader consults to resolve providers, tools, agents, and
//! predicates referenced by name from specs.

use std::collections::HashMap;
use std::sync::Arc;

use agent_framework_core::agent::{Agent, SupportsAgentRun};
use agent_framework_core::client::ChatClient;
use agent_framework_core::tools::ToolDefinition;
use agent_framework_core::workflow::Condition;
use serde_json::Value;

use crate::agent::ModelSpec;
use crate::error::{DeclarativeError, Result};

/// The error type a client factory closure may return. Boxed so the
/// declarative crate stays decoupled from any provider crate's error types.
pub type FactoryError = Box<dyn std::error::Error + Send + Sync>;

/// The result of a client factory closure.
pub type ClientFactoryResult = std::result::Result<Arc<dyn ChatClient>, FactoryError>;

type ClientFactoryFn = Arc<dyn Fn(&ModelSpec) -> ClientFactoryResult + Send + Sync>;

/// A registry of per-provider closures that construct a [`ChatClient`] from a
/// spec's `model` block.
///
/// The crate never depends on the provider crates directly; callers register a
/// closure per provider string (`"openai"`, `"OpenAI.Chat"`, `"anthropic"`,
/// `"azure_openai"`, …). Lookup mirrors the upstream loader: `provider.apiType`
/// is tried first, then `provider`, then a registered default.
#[derive(Clone, Default)]
pub struct ChatClientFactory {
    factories: HashMap<String, ClientFactoryFn>,
    default: Option<ClientFactoryFn>,
}

impl ChatClientFactory {
    /// Create an empty factory registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory for an exact provider key (e.g. `"OpenAI.Chat"` or
    /// `"openai"`). Returns `self` for chaining.
    pub fn with<F>(mut self, provider: impl Into<String>, factory: F) -> Self
    where
        F: Fn(&ModelSpec) -> ClientFactoryResult + Send + Sync + 'static,
    {
        self.register(provider, factory);
        self
    }

    /// Register a fallback factory used when no provider-specific match is
    /// found (or when the spec has no `model.provider`).
    pub fn with_default<F>(mut self, factory: F) -> Self
    where
        F: Fn(&ModelSpec) -> ClientFactoryResult + Send + Sync + 'static,
    {
        self.default = Some(Arc::new(factory));
        self
    }

    /// Register a factory for an exact provider key (mutating form).
    pub fn register<F>(&mut self, provider: impl Into<String>, factory: F)
    where
        F: Fn(&ModelSpec) -> ClientFactoryResult + Send + Sync + 'static,
    {
        self.factories.insert(provider.into(), Arc::new(factory));
    }

    /// Whether any factory (or a default) is registered.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty() && self.default.is_none()
    }

    /// Resolve a client for `model`, trying `provider.apiType`, then `provider`,
    /// then the default factory.
    pub(crate) fn resolve(&self, model: Option<&ModelSpec>) -> Result<Arc<dyn ChatClient>> {
        // Build the ordered list of candidate keys.
        let mut keys: Vec<String> = Vec::new();
        if let Some(model) = model {
            if let (Some(p), Some(a)) = (&model.provider, &model.api_type) {
                keys.push(format!("{p}.{a}"));
            }
            if let Some(p) = &model.provider {
                keys.push(p.clone());
            }
        }
        for key in &keys {
            if let Some(factory) = self.factories.get(key) {
                return invoke(factory, model);
            }
        }
        if let Some(default) = &self.default {
            return invoke(default, model);
        }
        let attempted = keys
            .first()
            .cloned()
            .unwrap_or_else(|| "<none>".to_string());
        Err(DeclarativeError::NoClientFactory(attempted))
    }
}

fn invoke(factory: &ClientFactoryFn, model: Option<&ModelSpec>) -> Result<Arc<dyn ChatClient>> {
    let empty = ModelSpec {
        id: None,
        provider: None,
        api_type: None,
        connection: None,
        options: None,
    };
    let model = model.unwrap_or(&empty);
    factory(model).map_err(|e| DeclarativeError::Invalid(format!("client factory failed: {e}")))
}

/// A registry of native Rust tools, referenced by name from function/custom
/// tool specs. Inline function-spec tools that are not bound here become
/// non-executable *declaration-only* [`ToolDefinition`]s.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
}

impl ToolRegistry {
    /// Create an empty tool registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a native tool, keyed by its [`ToolDefinition::name`].
    pub fn register(&mut self, tool: impl Into<ToolDefinition>) {
        let tool = tool.into();
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Register a native tool under an explicit `name` (overriding the tool's
    /// own name as the registry key).
    pub fn register_as(&mut self, name: impl Into<String>, tool: impl Into<ToolDefinition>) {
        self.tools.insert(name.into(), tool.into());
    }

    /// Builder form of [`ToolRegistry::register`].
    pub fn with(mut self, tool: impl Into<ToolDefinition>) -> Self {
        self.register(tool);
        self
    }

    /// Builder form of [`ToolRegistry::register_as`].
    pub fn with_named(mut self, name: impl Into<String>, tool: impl Into<ToolDefinition>) -> Self {
        self.register_as(name, tool);
        self
    }

    /// Look up a native tool by name.
    pub(crate) fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// Iterate over `(registry key, tool)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ToolDefinition)> {
        self.tools.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// A registry of pre-built agents, referenced by id from a
/// [`WorkflowSpec`](crate::WorkflowSpec).
#[derive(Clone, Default)]
pub struct AgentRegistry {
    agents: HashMap<String, Arc<dyn SupportsAgentRun>>,
}

impl AgentRegistry {
    /// Create an empty agent registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent under `id`.
    pub fn register(&mut self, id: impl Into<String>, agent: Arc<dyn SupportsAgentRun>) {
        self.agents.insert(id.into(), agent);
    }

    /// Register a concrete [`Agent`] under `id`.
    pub fn register_chat_agent(&mut self, id: impl Into<String>, agent: Agent) {
        self.agents.insert(id.into(), Arc::new(agent));
    }

    /// Builder form of [`AgentRegistry::register`].
    pub fn with(mut self, id: impl Into<String>, agent: Arc<dyn SupportsAgentRun>) -> Self {
        self.register(id, agent);
        self
    }

    /// Iterate over `(id, agent)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Arc<dyn SupportsAgentRun>)> {
        self.agents.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Look up an agent by id.
    pub(crate) fn get(&self, id: &str) -> Result<Arc<dyn SupportsAgentRun>> {
        self.agents
            .get(id)
            .cloned()
            .ok_or_else(|| DeclarativeError::UnknownReference {
                kind: "agent",
                name: id.to_string(),
                registry: "agent",
            })
    }
}

/// A registry of named predicates that workflow edges/cases may reference via
/// `predicate:`, for routing logic beyond the [`condition`](crate::condition)
/// mini-language.
#[derive(Clone, Default)]
pub struct PredicateRegistry {
    predicates: HashMap<String, Condition>,
}

impl PredicateRegistry {
    /// Create an empty predicate registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a synchronous predicate under `name`.
    pub fn register<F>(&mut self, name: impl Into<String>, predicate: F)
    where
        F: Fn(&Value) -> bool + Send + Sync + 'static,
    {
        self.predicates.insert(
            name.into(),
            Arc::new(move |v: &Value| {
                let result = predicate(v);
                Box::pin(async move { result }) as agent_framework_core::tools::BoxFuture<bool>
            }),
        );
    }

    /// Builder form of [`PredicateRegistry::register`].
    pub fn with<F>(mut self, name: impl Into<String>, predicate: F) -> Self
    where
        F: Fn(&Value) -> bool + Send + Sync + 'static,
    {
        self.register(name, predicate);
        self
    }

    /// Look up a predicate by name.
    pub(crate) fn get(&self, name: &str) -> Result<Condition> {
        self.predicates
            .get(name)
            .cloned()
            .ok_or_else(|| DeclarativeError::UnknownReference {
                kind: "predicate",
                name: name.to_string(),
                registry: "predicate",
            })
    }
}
