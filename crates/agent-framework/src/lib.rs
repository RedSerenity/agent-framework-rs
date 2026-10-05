//! # agent-framework
//!
//! A Rust implementation of the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
//! for building AI agents and multi-agent workflows.
//!
//! This is the umbrella crate: it re-exports [`agent_framework_core`] and, via
//! cargo features, the providers:
//!
//! | Feature | Crate | Default |
//! | --- | --- | --- |
//! | `openai` | [`agent_framework_openai`] — OpenAI Chat Completions + Responses API | yes |
//! | `anthropic` | [`agent_framework_anthropic`] — Anthropic (Claude) Messages API | no |
//! | `ollama` | [`agent_framework_ollama`] — Ollama (local/remote, OpenAI-compatible) | no |
//! | `gemini` | [`agent_framework_gemini`] — Google Gemini `generateContent` API | no |
//! | `mistral` | [`agent_framework_mistral`] — Mistral AI Chat Completions API | no |
//! | `foundry-local` | [`agent_framework_foundry_local`] — Microsoft Foundry Local (OpenAI-compatible localhost endpoint) | no |
//! | `bedrock` | [`agent_framework_bedrock`] — AWS Bedrock Converse API (SigV4-signed) | no |
//! | `github-copilot` | [`agent_framework_github_copilot`] — GitHub Copilot chat API | no |
//! | `azure` | [`agent_framework_azure`] — Azure OpenAI (api-key / Entra ID) | no |
//! | `mcp` | [`agent_framework_mcp`] — Model Context Protocol tools (stdio, HTTP, websocket) | no |
//! | `a2a` | [`agent_framework_a2a`] — Agent2Agent protocol client | no |
//! | `declarative` | [`agent_framework_declarative`] — YAML/JSON agents & workflows | no |
//! | `hosting` | [`agent_framework_hosting`] — serve agents over HTTP (DevUI-style, A2A, OpenAI-compatible) | no |
//! | `hosting-mcp` | [`agent_framework_hosting_mcp`] — expose agents and workflows as MCP tools (stdio / streamable HTTP server) | no |
//! | `redis` | [`agent_framework_redis`] — Redis/Valkey history providers, context provider & RediSearch vector store | no |
//! | `mem0` | [`agent_framework_mem0`] — Mem0 long-term memory provider | no |
//! | `foundry` | [`agent_framework_foundry`] — Azure AI Foundry Responses API chat client + Prompt Agents | no |
//! | `azure-ai-search` | [`agent_framework_azure_ai_search`] — Azure AI Search memory + vector store | no |
//! | `cosmos` | [`agent_framework_cosmos`] — Cosmos DB NoSQL message store, workflow checkpoints, and vector store | no |
//! | `qdrant` | [`agent_framework_qdrant`] — Qdrant vector store (REST) | no |
//! | `copilotstudio` | [`agent_framework_copilotstudio`] — Copilot Studio agents | no |
//! | `purview` | [`agent_framework_purview`] — Purview compliance middleware | no |
//! | `harness` | [`agent_framework_harness`] — batteries-included harness agent (todos, modes, file access, file memory, background agents, tool approval, looping) | no |
//!
//! `full` enables everything.
//!
//! ```no_run
//! use agent_framework::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = OpenAIChatCompletionClient::from_env("gpt-4o-mini")?;
//! let agent = Agent::builder(client)
//!     .name("assistant")
//!     .instructions("You are a helpful assistant.")
//!     .build();
//!
//! let response = agent.run_once("What is the capital of France?").await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```

#![doc(html_root_url = "https://docs.rs/agent-framework")]

pub use agent_framework_core::*;

/// The OpenAI provider (enabled by the default `openai` feature).
#[cfg(feature = "openai")]
pub use agent_framework_openai as openai;

/// The Anthropic (Claude) provider (enable the `anthropic` feature).
#[cfg(feature = "anthropic")]
pub use agent_framework_anthropic as anthropic;

/// The Ollama provider (enable the `ollama` feature).
#[cfg(feature = "ollama")]
pub use agent_framework_ollama as ollama;

/// The Google Gemini provider (enable the `gemini` feature).
#[cfg(feature = "gemini")]
pub use agent_framework_gemini as gemini;

/// The Mistral AI provider (enable the `mistral` feature).
#[cfg(feature = "mistral")]
pub use agent_framework_mistral as mistral;

/// The Microsoft Foundry Local provider (enable the `foundry-local` feature).
#[cfg(feature = "foundry-local")]
pub use agent_framework_foundry_local as foundry_local;

/// The AWS Bedrock provider (enable the `bedrock` feature).
#[cfg(feature = "bedrock")]
pub use agent_framework_bedrock as bedrock;

/// The GitHub Copilot provider (enable the `github-copilot` feature).
#[cfg(feature = "github-copilot")]
pub use agent_framework_github_copilot as github_copilot;

/// The Azure OpenAI provider (enable the `azure` feature).
#[cfg(feature = "azure")]
pub use agent_framework_azure as azure;

/// Model Context Protocol tools (enable the `mcp` feature).
#[cfg(feature = "mcp")]
pub use agent_framework_mcp as mcp;

/// Agent2Agent protocol client (enable the `a2a` feature).
#[cfg(feature = "a2a")]
pub use agent_framework_a2a as a2a;

/// Declarative YAML/JSON agents and workflows (enable the `declarative` feature).
#[cfg(feature = "declarative")]
pub use agent_framework_declarative as declarative;

/// HTTP hosting: DevUI-style, A2A, and OpenAI-compatible serving (enable the `hosting` feature).
#[cfg(feature = "hosting")]
pub use agent_framework_hosting as hosting;

/// Agents and workflows served as MCP tools (enable the `hosting-mcp` feature).
#[cfg(feature = "hosting-mcp")]
pub use agent_framework_hosting_mcp as hosting_mcp;

/// Redis-backed history providers (incl. Valkey), context provider and
/// RediSearch vector store (enable the `redis` feature).
#[cfg(feature = "redis")]
pub use agent_framework_redis as redis;

/// Mem0 long-term memory context provider (enable the `mem0` feature).
#[cfg(feature = "mem0")]
pub use agent_framework_mem0 as mem0;

/// Azure AI Foundry Responses API chat client + Prompt Agents (enable the
/// `foundry` feature).
#[cfg(feature = "foundry")]
pub use agent_framework_foundry as foundry;

/// Azure AI Search context provider and vector store (enable the
/// `azure-ai-search` feature).
#[cfg(feature = "azure-ai-search")]
pub use agent_framework_azure_ai_search as azure_ai_search;

/// Azure Cosmos DB NoSQL chat-message store, checkpoint storage, and
/// vector store (enable the `cosmos` feature).
#[cfg(feature = "cosmos")]
pub use agent_framework_cosmos as cosmos;

/// Qdrant vector store over the REST API (enable the `qdrant` feature).
#[cfg(feature = "qdrant")]
pub use agent_framework_qdrant as qdrant;

/// Microsoft Copilot Studio agent client (enable the `copilotstudio` feature).
#[cfg(feature = "copilotstudio")]
pub use agent_framework_copilotstudio as copilotstudio;

/// Microsoft Purview compliance middleware (enable the `purview` feature).
#[cfg(feature = "purview")]
pub use agent_framework_purview as purview;

/// The batteries-included harness agent: todos, modes, sandboxed file
/// access, file-backed memory, background sub-agents, tool-approval policy,
/// and looping (enable the `harness` feature).
#[cfg(feature = "harness")]
pub use agent_framework_harness as harness;

/// Commonly used imports for building agents and workflows.
pub mod prelude {
    pub use agent_framework_core::prelude::*;

    #[cfg(feature = "openai")]
    pub use agent_framework_openai::{
        OpenAIChatClient, OpenAIChatCompletionClient, OpenAIEmbeddingClient,
    };

    #[cfg(feature = "anthropic")]
    pub use agent_framework_anthropic::{
        AnthropicBedrockClient, AnthropicClient, AnthropicFoundryClient, AnthropicVertexClient,
    };

    #[cfg(feature = "ollama")]
    pub use agent_framework_ollama::{OllamaChatClient, OllamaEmbeddingClient};

    #[cfg(feature = "gemini")]
    pub use agent_framework_gemini::GeminiChatClient;

    #[cfg(feature = "mistral")]
    pub use agent_framework_mistral::{MistralChatClient, MistralEmbeddingClient};

    #[cfg(feature = "foundry-local")]
    pub use agent_framework_foundry_local::FoundryLocalChatClient;

    #[cfg(feature = "bedrock")]
    pub use agent_framework_bedrock::{BedrockChatClient, BedrockEmbeddingClient};

    #[cfg(feature = "github-copilot")]
    pub use agent_framework_github_copilot::GitHubCopilotChatClient;

    #[cfg(feature = "azure")]
    pub use agent_framework_azure::{
        AzureOpenAIClient, AzureOpenAIEmbeddingClient, StaticTokenCredential, TokenCredential,
    };

    #[cfg(feature = "mcp")]
    pub use agent_framework_mcp::{McpStdioTool, McpStreamableHttpTool, McpWebsocketTool};

    #[cfg(feature = "a2a")]
    pub use agent_framework_a2a::{A2AAgent, A2AClient};

    #[cfg(feature = "declarative")]
    pub use agent_framework_declarative::DeclarativeLoader;

    #[cfg(feature = "hosting")]
    pub use agent_framework_hosting::AgentHost;

    #[cfg(feature = "redis")]
    pub use agent_framework_redis::{
        RedisChatMessageStore, RedisContextProvider, RedisHistoryProvider, RedisVectorStore,
        ValkeyChatHistoryProvider,
    };

    #[cfg(feature = "mem0")]
    pub use agent_framework_mem0::Mem0Provider;

    #[cfg(feature = "foundry")]
    pub use agent_framework_foundry::{FoundryAgent, FoundryChatClient, FoundryEmbeddingClient};

    #[cfg(feature = "azure-ai-search")]
    pub use agent_framework_azure_ai_search::{
        AzureAISearchCollection, AzureAISearchProvider, AzureAISearchStore,
    };

    #[cfg(feature = "cosmos")]
    pub use agent_framework_cosmos::CosmosChatMessageStore;

    #[cfg(feature = "qdrant")]
    pub use agent_framework_qdrant::{QdrantCollection, QdrantStore};

    #[cfg(feature = "copilotstudio")]
    pub use agent_framework_copilotstudio::CopilotStudioAgent;

    #[cfg(feature = "purview")]
    pub use agent_framework_purview::{PurviewAgentMiddleware, PurviewChatMiddleware};
}
