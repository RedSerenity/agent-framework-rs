//! # agent-framework-foundry
//!
//! An Azure AI Foundry [`ChatClient`] and Prompt Agent surface for
//! `agent-framework-rs`, built on the **Responses API**
//! (`POST {endpoint}/openai/v1/responses`) rather than the older Agents
//! threads/runs data plane (removed upstream, along with the `azure-ai-agents`
//! SDK it wrapped — see this crate's own history / `UPSTREAM_DRIFT.md`).
//!
//! [`FoundryChatClient`] does not speak HTTP/SSE itself: it wraps an
//! [`agent_framework_azure::responses::AzureOpenAIResponsesClient`]
//! configured for a Foundry project endpoint (Microsoft Entra ID bearer auth
//! or a static API key), with
//! [`without_api_version`](agent_framework_azure::responses::AzureOpenAIResponsesClient::without_api_version)
//! applied, since the Foundry v1 GA route is path-versioned
//! (`{endpoint}/openai/v1/responses`, no `?api-version=` query parameter).
//! All request/response conversion, streaming, and error classification are
//! reused verbatim from that client (which itself reuses
//! `agent_framework_openai::responses`), so wire fidelity comes for free
//! rather than being re-implemented here.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_azure::AzureCliCredential;
//! use agent_framework_core::prelude::*;
//! use agent_framework_foundry::{FoundryChatClient, FOUNDRY_SCOPE};
//!
//! # async fn demo() -> Result<()> {
//! let credential = Arc::new(AzureCliCredential::new(FOUNDRY_SCOPE));
//! let client = FoundryChatClient::with_token_credential(
//!     "https://my-project.services.ai.azure.com",
//!     "gpt-4o",
//!     credential,
//! );
//! let agent = Agent::builder(client).instructions("You are concise.").build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```
//!
//! ## Evaluation
//!
//! [`FoundryEvals`] is an
//! [`Evaluator`](agent_framework_core::evaluation::Evaluator) backed by
//! Foundry's built-in (`builtin.*`) and generated rubric evaluators, run
//! through the project's OpenAI Evals API; [`evaluate_traces`] and
//! [`evaluate_foundry_target`] evaluate stored responses / traces and
//! Foundry-invoked targets. See the [`evals`] module.
//!
//! ## Prompt Agents
//!
//! [`FoundryAgent`] realizes a Foundry **Prompt Agent** *client-side*: it
//! pairs a [`FoundryChatClient`] with a [`PromptAgentDefinition`]
//! (name/model/instructions/tools) and runs it over the Responses API through
//! an inner [`agent_framework_core::agent::Agent`]. [`FoundryAgent::to_prompt_agent`]
//! hands back the definition it was built from, mirroring upstream's
//! `Agent.to_prompt_agent()` -> `PromptAgentDefinition`.
//!
//! **This does not bind to a *server-hosted* agent** by id/name (the Foundry
//! Agents control plane, e.g. `AIProjectClient.agents.get(...)`) — see the
//! [`FoundryAgent`] docs for why that's a documented extension point rather
//! than something wired up here.
//!
//! ```no_run
//! # use agent_framework_foundry::FoundryChatClient;
//! use agent_framework_core::prelude::*;
//! use agent_framework_foundry::FoundryAgent;
//!
//! # async fn demo(client: FoundryChatClient) -> Result<()> {
//! let agent = FoundryAgent::builder(client)
//!     .name("rust-example-agent")
//!     .instructions("You are a helpful, concise assistant.")
//!     .build();
//! let reply = agent.run(vec![Message::user("Say hi")], None).await?;
//! println!("{}", reply.text());
//!
//! // Round-trips the definition the agent was built from.
//! let definition = agent.to_prompt_agent();
//! assert_eq!(definition.name, "rust-example-agent");
//! # Ok(())
//! # }
//! ```

pub mod embeddings;
pub mod evals;
pub mod memory;
mod tool_definition_wire;

pub use embeddings::FoundryEmbeddingClient;
pub use evals::{
    evaluate_foundry_target, evaluate_traces, EvaluateFoundryTarget, EvaluateTraces, EvaluatorSpec,
    FoundryEvals, GeneratedEvaluatorRef,
};
pub use memory::FoundryMemoryProvider;

use std::sync::Arc;

use agent_framework_azure::responses::AzureOpenAIResponsesClient;
use agent_framework_azure::TokenCredential;
use agent_framework_core::agent::{Agent, AgentRunOptions, AgentRunStream, SupportsAgentRun};
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::ToolDefinition;
use agent_framework_core::types::{AgentResponse, ChatOptions, ChatResponse, Message};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// The Entra ID scope (audience) for the Azure AI Foundry data plane.
///
/// [`AzureOpenAIResponsesClient`] (the transport [`FoundryChatClient`]
/// delegates to) has no per-request scope override — it calls
/// [`TokenCredential::get_token`], trusting the credential to already be
/// bound to the right audience. Construct the credential with this scope,
/// e.g. `AzureCliCredential::new(FOUNDRY_SCOPE)` or
/// `DefaultAzureCredential::new(FOUNDRY_SCOPE)`.
pub const FOUNDRY_SCOPE: &str = "https://ai.azure.com/.default";

// ---------------------------------------------------------------------------
// FoundryChatClient
// ---------------------------------------------------------------------------

/// A [`ChatClient`] for the Azure AI Foundry project Responses API.
///
/// A thin wrapper around [`AzureOpenAIResponsesClient`]: constructs it
/// against the Foundry project `endpoint` and `model` (deployment) with
/// [`without_api_version`](AzureOpenAIResponsesClient::without_api_version)
/// applied (see the [module docs](self)). All request building, response
/// parsing, SSE streaming, and HTTP-error classification are reused verbatim
/// from that client — this type adds nothing beyond Foundry-shaped
/// constructors and a plain delegating [`ChatClient`] impl.
#[derive(Clone)]
pub struct FoundryChatClient {
    inner: AzureOpenAIResponsesClient,
    endpoint: String,
    model: String,
}

impl std::fmt::Debug for FoundryChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryChatClient")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl FoundryChatClient {
    /// Create a client authenticating with a static API key (`api-key`
    /// header).
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let endpoint = endpoint.into();
        let model = model.into();
        let inner = AzureOpenAIResponsesClient::new(endpoint.clone(), model.clone(), api_key)
            .without_api_version()
            .without_implicit_encrypted_reasoning();
        Self {
            inner,
            endpoint,
            model,
        }
    }

    /// Create a client authenticating via a [`TokenCredential`] (Microsoft
    /// Entra ID `Authorization: Bearer <token>`) — the primary Foundry auth
    /// path. The credential should already be scoped to [`FOUNDRY_SCOPE`],
    /// e.g. `AzureCliCredential::new(FOUNDRY_SCOPE)`.
    pub fn with_token_credential(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        let endpoint = endpoint.into();
        let model = model.into();
        let inner = AzureOpenAIResponsesClient::with_token_credential(
            endpoint.clone(),
            model.clone(),
            credential,
        )
        .without_api_version()
        .without_implicit_encrypted_reasoning();
        Self {
            inner,
            endpoint,
            model,
        }
    }

    /// Build a client from environment variables.
    ///
    /// Reads `FOUNDRY_ENDPOINT` (alias `FOUNDRY_PROJECT_ENDPOINT`) and
    /// `FOUNDRY_MODEL` — the `FOUNDRY_*`/`model` naming mirrors upstream's
    /// shift away from the old `AZURE_AI_*`/`model_deployment_name`
    /// convention. When `FOUNDRY_API_KEY` is set it authenticates with that
    /// key; otherwise it falls back to a
    /// [`DefaultAzureCredential`](agent_framework_azure::DefaultAzureCredential)
    /// scoped to [`FOUNDRY_SCOPE`] (a chain that tries a managed identity,
    /// then client-secret env vars, then the Azure CLI).
    ///
    /// # Errors
    /// [`Error::Configuration`] when neither endpoint variable, or
    /// `FOUNDRY_MODEL`, is set.
    pub fn from_env() -> Result<Self> {
        Self::from_env_vars(|key| std::env::var(key).ok())
    }

    /// Implementation of [`from_env`](Self::from_env), parameterized over an
    /// environment lookup function so the parsing/validation logic is
    /// testable against an in-memory map instead of real process env vars.
    fn from_env_vars(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let endpoint = get("FOUNDRY_ENDPOINT")
            .or_else(|| get("FOUNDRY_PROJECT_ENDPOINT"))
            .ok_or_else(|| {
                Error::Configuration(
                    "FOUNDRY_ENDPOINT (or FOUNDRY_PROJECT_ENDPOINT) is not set".into(),
                )
            })?;
        let model = get("FOUNDRY_MODEL")
            .ok_or_else(|| Error::Configuration("FOUNDRY_MODEL is not set".into()))?;
        Ok(match get("FOUNDRY_API_KEY") {
            Some(api_key) => Self::new(endpoint, model, api_key),
            None => {
                let credential: Arc<dyn TokenCredential> = Arc::new(
                    agent_framework_azure::DefaultAzureCredential::new(FOUNDRY_SCOPE),
                );
                Self::with_token_credential(endpoint, model, credential)
            }
        })
    }

    /// Override the full base URL requests are built against — see
    /// [`AzureOpenAIResponsesClient::with_base_url`].
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner = self.inner.with_base_url(base_url);
        self
    }

    /// The Foundry project endpoint this client targets.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The model deployment name this client targets.
    pub fn model_name(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl ChatClient for FoundryChatClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.inner.get_response(messages, options).await
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        self.inner.get_streaming_response(messages, options).await
    }

    fn model(&self) -> Option<&str> {
        self.inner.model()
    }
}

// ---------------------------------------------------------------------------
// PromptAgentDefinition
// ---------------------------------------------------------------------------

/// The serializable definition of a Foundry **Prompt Agent**: a name, a
/// model, optional instructions/description, and its tools.
///
/// Maps to upstream's `PromptAgentDefinition` (the declarative shape a Prompt
/// Agent is created from, and that `Agent.to_prompt_agent()` emits). Used
/// here to build a [`FoundryAgent`] ([`FoundryAgent::from_definition`]) and to
/// read one back ([`FoundryAgent::to_prompt_agent`]).
///
/// [`ToolDefinition`] itself does not implement `Serialize`/`Deserialize` (a
/// function tool carries a `dyn Tool` local executor, which isn't
/// serializable), so `tools` round-trips through a private wire shape
/// capturing only the declarative fields a definition needs — see
/// `tool_definition_wire`. A tool
/// deserialized back out of a `PromptAgentDefinition` always has
/// `executor: None`: a definition describes what a tool *is*, not a live
/// local implementation to call it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptAgentDefinition {
    /// The agent's name.
    pub name: String,
    /// The model deployment this agent runs on.
    pub model: String,
    /// The system prompt / instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// A human-readable description of what the agent does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The tools available to the agent.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        with = "tool_definition_wire::vec"
    )]
    pub tools: Vec<ToolDefinition>,
}

impl PromptAgentDefinition {
    /// Create a definition with just a name and model — the minimum needed
    /// to run a Prompt Agent.
    pub fn new(name: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            model: model.into(),
            instructions: None,
            description: None,
            tools: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// FoundryAgent
// ---------------------------------------------------------------------------

/// A [`SupportsAgentRun`] that runs a Foundry **Prompt Agent** over the
/// Responses API.
///
/// Built from a [`FoundryChatClient`] plus a [`PromptAgentDefinition`]
/// (name/instructions/model/tools): internally it constructs a core
/// [`Agent`] over that client with the definition applied, and every
/// [`SupportsAgentRun`] method delegates straight through to it.
/// [`to_prompt_agent`](Self::to_prompt_agent) hands back the definition the
/// agent was built from, so it round-trips through
/// [`FoundryAgent::from_definition`].
///
/// This realizes a Prompt Agent **client-side**, entirely through the
/// stateless Responses API — it does not create, fetch, or bind to a
/// *server-hosted* agent by id/name on the Foundry Agents control plane
/// (`AIProjectClient.agents.*`). Wiring up that control plane, so
/// [`FoundryAgent`] could target a Prompt Agent or Hosted Agent that already
/// exists on the service (e.g. a future `agent_id`/`with_existing_agent`
/// constructor), is a documented extension point, not implemented here.
#[derive(Clone)]
pub struct FoundryAgent {
    inner: Agent,
    definition: PromptAgentDefinition,
}

impl std::fmt::Debug for FoundryAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryAgent")
            .field("definition", &self.definition)
            .finish_non_exhaustive()
    }
}

impl FoundryAgent {
    /// Start building a [`FoundryAgent`] over `client`.
    pub fn builder(client: FoundryChatClient) -> FoundryAgentBuilder {
        FoundryAgentBuilder::new(client)
    }

    /// Build a [`FoundryAgent`] directly from a [`PromptAgentDefinition`].
    ///
    /// The inner [`Agent`]'s id and name are both set to `definition.name`
    /// (the core [`AgentBuilder`](agent_framework_core::agent::AgentBuilder)
    /// otherwise defaults `id` to a random UUID).
    pub fn from_definition(client: FoundryChatClient, definition: PromptAgentDefinition) -> Self {
        let mut builder = Agent::builder(client)
            .id(definition.name.clone())
            .name(definition.name.clone())
            .model(definition.model.clone())
            .tools(definition.tools.clone());
        if let Some(instructions) = &definition.instructions {
            builder = builder.instructions(instructions.clone());
        }
        if let Some(description) = &definition.description {
            builder = builder.description(description.clone());
        }
        Self {
            inner: builder.build(),
            definition,
        }
    }

    /// The [`PromptAgentDefinition`] this agent was built from — round-trips
    /// through [`FoundryAgent::from_definition`].
    pub fn to_prompt_agent(&self) -> PromptAgentDefinition {
        self.definition.clone()
    }

    /// The wrapped core [`Agent`], for functionality (e.g. [`Agent::as_tool`])
    /// not exposed directly on [`FoundryAgent`].
    pub fn inner(&self) -> &Agent {
        &self.inner
    }
}

#[async_trait]
impl SupportsAgentRun for FoundryAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        self.inner.run(messages, session).await
    }

    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        self.inner
            .run_with_options(messages, session, options)
            .await
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        self.inner.run_stream(messages, session, options).await
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    fn display_name(&self) -> String {
        self.inner.display_name()
    }

    fn create_session(&self) -> AgentSession {
        self.inner.create_session()
    }

    fn default_tools(&self) -> Vec<ToolDefinition> {
        self.inner.default_tools()
    }
}

/// Builds a [`FoundryAgent`] from a [`FoundryChatClient`] plus the pieces of
/// a [`PromptAgentDefinition`] (name/instructions/model/tools).
pub struct FoundryAgentBuilder {
    client: FoundryChatClient,
    name: Option<String>,
    model: String,
    instructions: Option<String>,
    description: Option<String>,
    tools: Vec<ToolDefinition>,
}

impl FoundryAgentBuilder {
    fn new(client: FoundryChatClient) -> Self {
        let model = client.model_name().to_string();
        Self {
            client,
            name: None,
            model,
            instructions: None,
            description: None,
            tools: Vec::new(),
        }
    }

    /// Set the agent's name (default: the client's configured model name).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Override the model deployment (default: the client's configured
    /// model).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the system prompt / instructions.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set a human-readable description of what the agent does.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add one tool.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add several tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Build the [`FoundryAgent`].
    pub fn build(self) -> FoundryAgent {
        let name = self.name.unwrap_or_else(|| self.model.clone());
        let definition = PromptAgentDefinition {
            name,
            model: self.model,
            instructions: self.instructions,
            description: self.description,
            tools: self.tools,
        };
        FoundryAgent::from_definition(self.client, definition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_azure::StaticTokenCredential;
    use agent_framework_core::tools::{hosted_code_interpreter, FunctionTool};
    use serde_json::json;

    fn client() -> FoundryChatClient {
        FoundryChatClient::with_token_credential(
            "https://my-project.services.ai.azure.com",
            "gpt-4o",
            Arc::new(StaticTokenCredential::new("test-token")),
        )
    }

    #[test]
    fn client_reports_endpoint_and_model() {
        let c = client();
        assert_eq!(c.endpoint(), "https://my-project.services.ai.azure.com");
        assert_eq!(c.model_name(), "gpt-4o");
        assert_eq!(c.model(), Some("gpt-4o"));
    }

    // -- from_env -------------------------------------------------------

    #[test]
    fn from_env_errors_without_endpoint() {
        let err = FoundryChatClient::from_env_vars(|key| match key {
            "FOUNDRY_MODEL" => Some("gpt-4o".into()),
            _ => None,
        })
        .unwrap_err();
        assert!(matches!(err, Error::Configuration(_)), "{err:?}");
        assert!(err.to_string().contains("FOUNDRY_ENDPOINT"));
    }

    #[test]
    fn from_env_errors_without_model() {
        let err = FoundryChatClient::from_env_vars(|key| match key {
            "FOUNDRY_ENDPOINT" => Some("https://proj.services.ai.azure.com".into()),
            _ => None,
        })
        .unwrap_err();
        assert!(matches!(err, Error::Configuration(_)), "{err:?}");
        assert!(err.to_string().contains("FOUNDRY_MODEL"));
    }

    #[test]
    fn from_env_accepts_project_endpoint_alias() {
        let c = FoundryChatClient::from_env_vars(|key| match key {
            "FOUNDRY_PROJECT_ENDPOINT" => Some("https://proj.services.ai.azure.com".into()),
            "FOUNDRY_MODEL" => Some("gpt-4o".into()),
            "FOUNDRY_API_KEY" => Some("key-123".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(c.endpoint(), "https://proj.services.ai.azure.com");
        assert_eq!(c.model_name(), "gpt-4o");
    }

    #[test]
    fn from_env_with_api_key_authenticates_by_key() {
        let c = FoundryChatClient::from_env_vars(|key| match key {
            "FOUNDRY_ENDPOINT" => Some("https://proj.services.ai.azure.com".into()),
            "FOUNDRY_MODEL" => Some("gpt-4o".into()),
            "FOUNDRY_API_KEY" => Some("key-123".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(c.model_name(), "gpt-4o");
    }

    #[test]
    fn from_env_without_api_key_falls_back_to_default_azure_credential() {
        // No FOUNDRY_API_KEY: this must still construct successfully (via
        // DefaultAzureCredential) rather than erroring.
        let c = FoundryChatClient::from_env_vars(|key| match key {
            "FOUNDRY_ENDPOINT" => Some("https://proj.services.ai.azure.com".into()),
            "FOUNDRY_MODEL" => Some("gpt-4o".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(c.endpoint(), "https://proj.services.ai.azure.com");
    }

    // -- PromptAgentDefinition serde round trip --------------------------

    #[test]
    fn prompt_agent_definition_round_trips_through_json() {
        let tool = FunctionTool::new(
            "get_weather",
            "Get the weather",
            json!({"type": "object", "properties": {}}),
            |_| async { Ok(json!("ok")) },
        )
        .into_definition();
        let definition = PromptAgentDefinition {
            name: "weather-bot".into(),
            model: "gpt-4o".into(),
            instructions: Some("Be terse.".into()),
            description: Some("Answers weather questions.".into()),
            tools: vec![tool, hosted_code_interpreter()],
        };

        let json = serde_json::to_string(&definition).unwrap();
        let back: PromptAgentDefinition = serde_json::from_str(&json).unwrap();

        assert_eq!(back.name, "weather-bot");
        assert_eq!(back.model, "gpt-4o");
        assert_eq!(back.instructions.as_deref(), Some("Be terse."));
        assert_eq!(
            back.description.as_deref(),
            Some("Answers weather questions.")
        );
        assert_eq!(back.tools.len(), 2);
        assert_eq!(back.tools[0].name, "get_weather");
        assert!(back.tools[0].executor.is_none());
    }

    #[test]
    fn prompt_agent_definition_omits_empty_optional_fields() {
        let definition = PromptAgentDefinition::new("bare", "gpt-4o");
        let json = serde_json::to_value(&definition).unwrap();
        assert!(json.get("instructions").is_none());
        assert!(json.get("description").is_none());
        assert!(json.get("tools").is_none());
    }

    // -- FoundryAgent -----------------------------------------------------

    #[test]
    fn from_definition_to_prompt_agent_round_trips() {
        let definition = PromptAgentDefinition {
            name: "rust-example-agent".into(),
            model: "gpt-4o".into(),
            instructions: Some("You are concise.".into()),
            description: Some("An example agent.".into()),
            tools: vec![hosted_code_interpreter()],
        };
        let agent = FoundryAgent::from_definition(client(), definition.clone());

        let round_tripped = agent.to_prompt_agent();
        assert_eq!(round_tripped.name, definition.name);
        assert_eq!(round_tripped.model, definition.model);
        assert_eq!(round_tripped.instructions, definition.instructions);
        assert_eq!(round_tripped.description, definition.description);
        assert_eq!(round_tripped.tools.len(), 1);

        assert_eq!(agent.id(), "rust-example-agent");
        assert_eq!(agent.name(), Some("rust-example-agent"));
    }

    #[test]
    fn builder_defaults_name_to_the_client_model() {
        let agent = FoundryAgent::builder(client()).build();
        assert_eq!(agent.to_prompt_agent().name, "gpt-4o");
        assert_eq!(agent.to_prompt_agent().model, "gpt-4o");
    }

    #[test]
    fn builder_collects_instructions_description_and_tools() {
        let agent = FoundryAgent::builder(client())
            .name("named-agent")
            .instructions("Be helpful.")
            .description("A test agent.")
            .tool(hosted_code_interpreter())
            .build();
        let definition = agent.to_prompt_agent();
        assert_eq!(definition.name, "named-agent");
        assert_eq!(definition.instructions.as_deref(), Some("Be helpful."));
        assert_eq!(definition.description.as_deref(), Some("A test agent."));
        assert_eq!(definition.tools.len(), 1);
    }
}
