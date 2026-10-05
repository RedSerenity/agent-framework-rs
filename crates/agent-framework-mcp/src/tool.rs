//! High-level MCP tools: [`McpStdioTool`], [`McpStreamableHttpTool`], and
//! [`McpWebsocketTool`].
//!
//! Each owns a lazily connected, reconnectable session (see
//! [`crate::session`]) and turns the server's catalog into ready-to-use
//! [`ToolDefinition`]s whose executors call back into that session.
//!
//! Two ways to wire one into a [`agent_framework_core::agent::Agent`]:
//!
//! - **Static** (frozen at build time): `mcp.tool_definitions().await` once,
//!   up front, and hand the result to `Agent::builder().tools(..)`. The
//!   agent never notices a later server-side tool-catalog change.
//! - **Dynamic** (resolved on every run): all three types implement
//!   [`agent_framework_core::tools::ToolSource`], so
//!   `Agent::builder().tool_source(Arc::new(mcp))` connects lazily on
//!   the agent's first run and re-resolves the tool list on every
//!   subsequent run from a cache that self-invalidates on the server's
//!   `notifications/tools/list_changed` (see [`McpClient::list_tools_cached`]).
//!
//! # What the catalog becomes
//!
//! As upstream's `MCPTool`:
//!
//! - every server tool becomes a function named by its normalized name,
//!   behind [`tool_name_prefix`](McpStdioTool::tool_name_prefix) when set;
//! - with [`load_prompts`](McpStdioTool::load_prompts) (the default), every
//!   server prompt becomes a function too, returning the rendered prompt;
//! - [`allowed_tools`](McpStdioTool::allowed_tools) and per-tool
//!   [`McpApprovalMode`] names match a tool's **raw remote name** (or its
//!   prefixed name when the remote name is already normalized), and a name
//!   that matches two remote tools is an error;
//! - with [`progressive_disclosure`](McpStdioTool::progressive_disclosure),
//!   the model starts with `list_mcp_tools` / `load_tool` / `unload_tool`
//!   (plus [`always_load`](McpStdioTool::always_load) tools) and loads the
//!   rest into the run as it needs them.
//!
//! A call forwards only the arguments the tool declares (plus
//! [`additional_tool_argument_names`](McpStdioTool::additional_tool_argument_names)),
//! echoes the tool's `_meta`, and reconnects and retries once if the
//! connection was lost underneath it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::OnceCell;

use agent_framework_core::error::Result;
use agent_framework_core::tools::{ToolDefinition, ToolSource};
use agent_framework_core::types::Message;

use crate::client::McpClient;
use crate::protocol::{
    normalize_mcp_name, role_and_content_to_chat_message, McpLogLevel, PromptDescriptor,
    ToolResultContent,
};
use crate::sampling::{Root, SamplingHandler};
use crate::session::{
    AdditionalArgumentNames, Common, McpSession, PromptResultParser, ToolResultParser,
    TransportFactory,
};
use crate::transport::{
    McpStdioTransport, McpStreamableHttpTransport, McpTransport, McpWebsocketTransport, StdioEnv,
};

pub use crate::session::McpApprovalMode;

/// Map an MCP `prompts/get` message into a core [`Message`] — mirrors
/// the Python reference's `_mcp_prompt_message_to_chat_message`.
fn prompt_message_to_chat_message(msg: &crate::protocol::PromptMessage) -> Message {
    role_and_content_to_chat_message(&msg.role, &msg.content)
}

/// Drop any prompt whose normalized name collides with an earlier one in the
/// same listing (first occurrence wins), warning on each skip. The returned
/// descriptors keep their original `name` for [`McpStdioTool::get_prompt`].
fn dedup_prompts_by_normalized_name(prompts: Vec<PromptDescriptor>) -> Vec<PromptDescriptor> {
    let mut seen = std::collections::HashSet::new();
    prompts
        .into_iter()
        .filter(|p| {
            let local = normalize_mcp_name(&p.name);
            let fresh = seen.insert(local.clone());
            if !fresh {
                tracing::warn!(
                    prompt = %p.name,
                    local_name = %local,
                    "MCP prompt name collides (after normalization) with another prompt from \
                     the same server; skipping the later one"
                );
            }
            fresh
        })
        .collect()
}

/// The builder methods and session-backed operations every wrapper shares.
/// Each wrapper supplies `fn transport_factory(&self) -> TransportFactory`.
macro_rules! mcp_tool_common {
    ($ty:ident) => {
        impl $ty {
            /// Set a human-readable description for this tool source.
            pub fn description(mut self, description: impl Into<String>) -> Self {
                self.common.description = Some(description.into());
                self
            }

            /// Restrict the tools produced to these names: a tool's raw
            /// remote name, or its prefixed name when the remote name is
            /// already normalized.
            pub fn allowed_tools<I, S>(mut self, tools: I) -> Self
            where
                I: IntoIterator<Item = S>,
                S: Into<String>,
            {
                self.common.allowed_tools = Some(tools.into_iter().map(Into::into).collect());
                self
            }

            /// Set the approval policy applied to produced [`ToolDefinition`]s.
            pub fn approval_mode(mut self, mode: McpApprovalMode) -> Self {
                self.common.approval_mode = mode;
                self
            }

            /// Register the handler for server-initiated
            /// `sampling/createMessage` requests. See
            /// [`McpClient::sampling_handler`] and
            /// [`crate::chat_client_sampling_handler_with`].
            pub fn sampling_handler(mut self, handler: SamplingHandler) -> Self {
                self.common.sampling_handler = Some(handler);
                self
            }

            /// Register a static list of filesystem roots. See [`McpClient::roots`].
            pub fn roots<I>(mut self, roots: I) -> Self
            where
                I: IntoIterator<Item = Root>,
            {
                self.common.roots = Some(roots.into_iter().collect());
                self
            }

            /// Whether to load tools from the server (default `true`). When
            /// `false`, [`ToolSource::resolve_tools`] returns an empty list
            /// without connecting, and calling a tool is an error.
            pub fn load_tools(mut self, load_tools: bool) -> Self {
                self.common.load_tools = load_tools;
                self
            }

            /// Whether to load prompts from the server (default `true`):
            /// they become functions alongside the tools, and
            /// [`Self::prompts`] lists them. When `false`, neither happens
            /// and no prompt round trip is made.
            pub fn load_prompts(mut self, load_prompts: bool) -> Self {
                self.common.load_prompts = load_prompts;
                self
            }

            /// Prefix every generated function name: the normalized prefix,
            /// trailing `_.-` stripped, then `_`. Lets two servers with
            /// overlapping tool names share one agent.
            pub fn tool_name_prefix(mut self, prefix: impl Into<String>) -> Self {
                self.common.tool_name_prefix = Some(prefix.into());
                self
            }

            /// Choose the model-visible value of a result carrying both
            /// `content` and `structuredContent` (default
            /// [`ToolResultContent::StructuredFirst`], as upstream).
            pub fn tool_result_content(mut self, mode: ToolResultContent) -> Self {
                self.common.result_content = mode;
                self
            }

            /// Replace result parsing entirely (upstream's
            /// `parse_tool_results`); overrides [`Self::tool_result_content`].
            pub fn parse_tool_results(mut self, parser: ToolResultParser) -> Self {
                self.common.result_parser = Some(parser);
                self
            }

            /// Replace prompt-result parsing for prompts called as tools
            /// (upstream's `parse_prompt_results`).
            pub fn parse_prompt_results(mut self, parser: PromptResultParser) -> Self {
                self.common.prompt_parser = Some(parser);
                self
            }

            /// Argument names forwarded to `tools/call` beyond those a tool
            /// declares; everything else the model sends is dropped.
            pub fn additional_tool_argument_names(
                mut self,
                names: AdditionalArgumentNames,
            ) -> Self {
                self.common.extra_arguments = names;
                self
            }

            /// Expose only `list_mcp_tools`, `load_tool` and `unload_tool`
            /// (plus [`Self::always_load`] tools) up front; the model loads
            /// the tools it needs into the run. Requires
            /// [`Self::load_tools`]`(true)`.
            pub fn progressive_disclosure(mut self, enabled: bool) -> Self {
                self.common.progressive = enabled;
                self
            }

            /// Tools visible from the start under progressive disclosure.
            pub fn always_load<I, S>(mut self, names: I) -> Self
            where
                I: IntoIterator<Item = S>,
                S: Into<String>,
            {
                self.common.always_load = names.into_iter().map(Into::into).collect();
                self
            }

            /// Ask the server, after connecting, to send log messages at
            /// `level` and above (when it declares `logging`). They are
            /// re-emitted through `tracing` with target `mcp_server`.
            pub fn logging_level(mut self, level: McpLogLevel) -> Self {
                self.common.logging_level = Some(level);
                self
            }

            async fn session(&self) -> Result<Arc<McpSession>> {
                self.common.validate()?;
                let session = self
                    .session
                    .get_or_init(|| async {
                        McpSession::new(self.common.clone(), self.transport_factory())
                    })
                    .await
                    .clone();
                Ok(session)
            }

            /// Connect to the server and perform the `initialize` handshake.
            ///
            /// Idempotent and safe to call concurrently. Later calls
            /// reconnect transparently if the connection was lost.
            pub async fn connect(&self) -> Result<()> {
                self.session().await?.client().await.map(|_| ())
            }

            /// The connected client, connecting first if needed.
            pub async fn client(&self) -> Result<Arc<McpClient>> {
                self.session().await?.client().await
            }

            /// Connect (if needed) and return the functions the server's
            /// catalog yields (see the crate docs), from a live
            /// `tools/list` round trip — regardless of [`Self::load_tools`].
            /// For a cached, per-run alternative use this type as a
            /// [`ToolSource`].
            pub async fn tool_definitions(&self) -> Result<Vec<ToolDefinition>> {
                self.session().await?.tool_definitions(false).await
            }

            /// Connect (if needed) and list the server's prompts, cached until
            /// `notifications/prompts/list_changed`. Empty without a round
            /// trip when [`Self::load_prompts`] is `false` or the server did
            /// not declare `prompts`.
            pub async fn prompts(&self) -> Result<Vec<PromptDescriptor>> {
                if !self.common.load_prompts {
                    return Ok(Vec::new());
                }
                let prompts = self.session().await?.list_prompts().await?;
                Ok(dedup_prompts_by_normalized_name(prompts))
            }

            /// Connect (if needed) and fetch a rendered prompt's messages,
            /// mapped into core [`Message`]s — mirrors Python's
            /// `MCPTool.get_prompt`.
            pub async fn get_prompt(&self, name: &str, arguments: Value) -> Result<Vec<Message>> {
                let result = self.session().await?.get_prompt(name, arguments).await?;
                Ok(result
                    .messages
                    .iter()
                    .map(prompt_message_to_chat_message)
                    .collect())
            }

            /// The configured tool-source name.
            pub fn name(&self) -> &str {
                &self.common.name
            }

            /// The configured description, if any.
            pub fn description_text(&self) -> Option<&str> {
                self.common.description.as_deref()
            }

            /// Close the underlying session, if connected (best effort,
            /// idempotent).
            pub async fn close(&self) -> Result<()> {
                if let Some(session) = self.session.get() {
                    session.close().await?;
                }
                Ok(())
            }
        }

        /// Resolved per agent run: lazily connects on first call and serves
        /// functions from the cached catalog, invalidated automatically by
        /// `notifications/tools/list_changed`. Returns an empty list without
        /// connecting if [`load_tools`]($ty::load_tools) is `false`.
        /// Connection and listing failures propagate.
        #[async_trait]
        impl ToolSource for $ty {
            async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
                if !self.common.load_tools {
                    return Ok(Vec::new());
                }
                self.session().await?.tool_definitions(true).await
            }

            fn source_name(&self) -> &str {
                &self.common.name
            }
        }
    };
}

/// An MCP tool backed by a stdio-connected child process.
///
/// ```no_run
/// # use agent_framework_mcp::McpStdioTool;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let mcp = McpStdioTool::new("filesystem", "npx")
///     .args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
///     .description("Local filesystem access");
/// let tools = mcp.tool_definitions().await?;
/// # let _ = tools;
/// # Ok(())
/// # }
/// ```
pub struct McpStdioTool {
    common: Common,
    command: String,
    args: Vec<String>,
    env: Option<HashMap<String, String>>,
    /// Inherit the full parent process environment (default `false`, i.e. the
    /// child sees only a minimal baseline plus [`Self::env`]). See [`StdioEnv`].
    inherit_parent_env: bool,
    /// Named parent variables to pass through to the child by value when not
    /// inheriting the whole parent environment.
    inherit_env_vars: Vec<String>,
    cwd: Option<PathBuf>,
    request_timeout: Option<Duration>,
    session: OnceCell<Arc<McpSession>>,
}

impl McpStdioTool {
    /// Create a tool that spawns `command` (with no arguments) as an MCP
    /// server over stdio when connected.
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            common: Common::new(name.into()),
            command: command.into(),
            args: Vec::new(),
            env: None,
            inherit_parent_env: false,
            inherit_env_vars: Vec::new(),
            cwd: None,
            request_timeout: None,
            session: OnceCell::new(),
        }
    }

    /// Set the command-line arguments passed to the server process.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Add environment variables for the server process.
    ///
    /// **Secure by default:** the child does *not* inherit the parent process
    /// environment unless [`Self::inherit_parent_environment`] is enabled — it
    /// sees only a minimal baseline (PATH, temp-dir, locale) plus these
    /// explicit variables. This keeps host secrets (API keys, cloud
    /// credentials, tokens) from being disclosed to the MCP server. To pass a
    /// specific parent variable through, use [`Self::inherit_env_var`].
    pub fn env<I, K, V>(mut self, env: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.env = Some(env.into_iter().map(|(k, v)| (k.into(), v.into())).collect());
        self
    }

    /// Inherit the **full** parent process environment (default `false`).
    ///
    /// Convenient but re-exposes every host secret to the server; prefer
    /// [`Self::inherit_env_var`] for the specific variables the server needs.
    pub fn inherit_parent_environment(mut self, inherit: bool) -> Self {
        self.inherit_parent_env = inherit;
        self
    }

    /// Pass a single named parent environment variable through to the child by
    /// value (opt-in; ignored when [`Self::inherit_parent_environment`] is on).
    pub fn inherit_env_var(mut self, name: impl Into<String>) -> Self {
        self.inherit_env_vars.push(name.into());
        self
    }

    /// Pass several named parent environment variables through to the child.
    pub fn inherit_env_vars<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inherit_env_vars
            .extend(names.into_iter().map(Into::into));
        self
    }

    /// Set the server process's working directory.
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Set a per-request timeout applied while awaiting a response to any
    /// JSON-RPC request sent to this server. Unset (the default) waits
    /// indefinitely.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    fn transport_factory(&self) -> TransportFactory {
        let command = self.command.clone();
        let args = self.args.clone();
        let mut env = StdioEnv::new()
            .inherit_parent_environment(self.inherit_parent_env)
            .inherit_vars(self.inherit_env_vars.clone());
        if let Some(vars) = &self.env {
            env = env.vars(vars.clone());
        }
        let cwd = self.cwd.clone();
        let timeout = self.request_timeout;
        Arc::new(move || {
            let (command, args, env, cwd) =
                (command.clone(), args.clone(), env.clone(), cwd.clone());
            Box::pin(async move {
                let mut transport =
                    McpStdioTransport::spawn(&command, &args, &env, cwd.as_deref()).await?;
                if let Some(timeout) = timeout {
                    transport = transport.with_request_timeout(timeout);
                }
                Ok(Arc::new(transport) as Arc<dyn McpTransport>)
            })
        })
    }
}

mcp_tool_common!(McpStdioTool);

/// An MCP tool backed by a streamable-HTTP server.
///
/// ```no_run
/// # use agent_framework_mcp::McpStreamableHttpTool;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let mcp = McpStreamableHttpTool::new("web-api", "https://api.example.com/mcp")
///     .headers([("Authorization", "Bearer token")])
///     .description("Web API operations");
/// let tools = mcp.tool_definitions().await?;
/// # let _ = tools;
/// # Ok(())
/// # }
/// ```
pub struct McpStreamableHttpTool {
    common: Common,
    url: String,
    headers: Vec<(String, String)>,
    timeout: Option<Duration>,
    session: OnceCell<Arc<McpSession>>,
}

impl McpStreamableHttpTool {
    /// Create a tool that talks to the MCP server at `url` over streamable HTTP.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            common: Common::new(name.into()),
            url: url.into(),
            headers: Vec::new(),
            timeout: None,
            session: OnceCell::new(),
        }
    }

    /// Add custom headers (e.g. `Authorization`) sent with every request.
    pub fn headers<I, K, V>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.headers
            .extend(headers.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// Set a per-request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Alias of [`Self::timeout`], named as on the other two wrappers and as
    /// upstream's `request_timeout`.
    pub fn request_timeout(self, timeout: Duration) -> Self {
        self.timeout(timeout)
    }

    fn transport_factory(&self) -> TransportFactory {
        let url = self.url.clone();
        let headers = self.headers.clone();
        let timeout = self.timeout;
        Arc::new(move || {
            let (url, headers) = (url.clone(), headers.clone());
            Box::pin(async move {
                let map = McpStreamableHttpTransport::header_map(&headers)?;
                let transport = McpStreamableHttpTransport::new(url, map, timeout)?;
                Ok(Arc::new(transport) as Arc<dyn McpTransport>)
            })
        })
    }
}

mcp_tool_common!(McpStreamableHttpTool);

/// An MCP tool backed by a WebSocket server (`ws://` or `wss://`), using the
/// `"mcp"` subprotocol.
pub struct McpWebsocketTool {
    common: Common,
    url: String,
    headers: Vec<(String, String)>,
    request_timeout: Option<Duration>,
    session: OnceCell<Arc<McpSession>>,
}

impl McpWebsocketTool {
    /// Create a tool that talks to the MCP server at `url` (`ws://` or
    /// `wss://`) over a WebSocket.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            common: Common::new(name.into()),
            url: url.into(),
            headers: Vec::new(),
            request_timeout: None,
            session: OnceCell::new(),
        }
    }

    /// Add custom headers (e.g. `Authorization`) sent on the WebSocket upgrade
    /// request.
    pub fn headers<I, K, V>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.headers
            .extend(headers.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// Set a per-request timeout applied while awaiting a response.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    fn transport_factory(&self) -> TransportFactory {
        let url = self.url.clone();
        let headers = self.headers.clone();
        let timeout = self.request_timeout;
        Arc::new(move || {
            let (url, headers) = (url.clone(), headers.clone());
            Box::pin(async move {
                let mut transport = McpWebsocketTransport::connect(&url, &headers).await?;
                if let Some(timeout) = timeout {
                    transport = transport.with_request_timeout(timeout);
                }
                Ok(Arc::new(transport) as Arc<dyn McpTransport>)
            })
        })
    }
}

mcp_tool_common!(McpWebsocketTool);

#[cfg(test)]
mod tests;
