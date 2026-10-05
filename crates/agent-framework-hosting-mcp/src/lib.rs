//! # agent-framework-hosting-mcp
//!
//! Expose agents and workflows as Model Context Protocol tools. Ports
//! upstream's `agent-framework-hosting-mcp` package:
//!
//! - [`mcp_to_run`] / [`mcp_from_run`] — convert MCP `tools/call` arguments
//!   into run arguments, and an agent response into MCP content blocks.
//! - [`AgentMcpTool`] — one agent as one MCP tool, generating the tool's
//!   schema and parsing calls against it so the two cannot drift, with
//!   optional per-argument [`AgentState`](agent_framework_hosting::AgentState)
//!   sessions.
//! - [`WorkflowMcpTool`] — one workflow as one MCP tool.
//!
//! Upstream leaves the server itself to the MCP Python SDK. This workspace
//! has no MCP server SDK, so [`McpServer`] supplies a minimal one —
//! `initialize`, `ping`, `tools/list`, `tools/call` — over stdio
//! ([`McpServer::serve_stdio`]) or streamable HTTP
//! ([`McpServer::into_router`]). Any [`McpToolProvider`] can be served.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use agent_framework_core::agent::SupportsAgentRun;
//! # use agent_framework_hosting_mcp::{AgentMcpTool, McpServer};
//! # use serde_json::json;
//! # async fn demo(agent: Arc<dyn SupportsAgentRun>) -> std::io::Result<()> {
//! let tool = AgentMcpTool::new(agent)
//!     .name("run_agent")
//!     .argument_description("The request for the hosted agent.")
//!     .chat_option_parameter(
//!         "reasoning_effort",
//!         json!({ "type": "string", "enum": ["low", "medium", "high"] }),
//!     );
//! let app = McpServer::new("my-agents", "1.0.0").tool(tool).into_router("/mcp");
//! let listener = tokio::net::TcpListener::bind("127.0.0.1:8000").await?;
//! axum::serve(listener, app).await
//! # }
//! ```

pub mod conversion;
pub mod server;
pub mod tools;

pub use conversion::{mcp_from_messages, mcp_from_run, mcp_to_run, McpRunArgs};
pub use server::McpServer;
pub use tools::{AgentMcpTool, McpToolProvider, WorkflowMcpTool};

/// Errors from MCP tool adapters.
#[derive(Debug)]
pub enum McpHostError {
    /// The tool was called with arguments that do not satisfy its contract.
    InvalidArguments(String),
    /// Agent output could not be represented as MCP content.
    InvalidContent(String),
    /// The adapter is misconfigured (overlapping or undeclared parameters).
    Configuration(String),
    /// No tool of that name is served by this provider.
    UnknownTool(String),
    /// The agent or workflow run failed.
    Run(agent_framework_core::Error),
}

impl std::fmt::Display for McpHostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArguments(m) | Self::InvalidContent(m) | Self::Configuration(m) => {
                f.write_str(m)
            }
            Self::UnknownTool(name) => write!(f, "Unknown MCP tool: {name}"),
            Self::Run(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for McpHostError {}

impl From<agent_framework_core::Error> for McpHostError {
    fn from(e: agent_framework_core::Error) -> Self {
        Self::Run(e)
    }
}
