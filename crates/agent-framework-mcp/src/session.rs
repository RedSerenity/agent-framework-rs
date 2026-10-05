//! The connection core shared by [`McpStdioTool`](crate::McpStdioTool),
//! [`McpStreamableHttpTool`](crate::McpStreamableHttpTool) and
//! [`McpWebsocketTool`](crate::McpWebsocketTool): one reconnectable
//! [`McpClient`] plus everything that turns the server's catalog into
//! [`ToolDefinition`]s — name prefixing, allow/approval matching, prompts as
//! tools, result-content policy, argument filtering, and progressive
//! disclosure. Mirrors upstream `MCPTool` (`_mcp.py`).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use tokio::sync::RwLock;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::middleware::FunctionInvocationContext;
use agent_framework_core::tools::{ApprovalMode, BoxFuture, Tool, ToolDefinition};

use crate::client::McpClient;
use crate::protocol::{
    normalize_mcp_name, CallToolResult, ContentBlock, GetPromptResult, McpLogLevel,
    PromptDescriptor, ToolDescriptor, ToolResultContent,
};
use crate::sampling::{Root, SamplingHandler};
use crate::transport::McpTransport;

/// The `clientInfo.name` this crate sends during `initialize`.
const CLIENT_NAME: &str = "agent-framework-rs";
/// The `clientInfo.version` this crate sends during `initialize`.
const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

const PROGRESSIVE_LIST: &str = "list_mcp_tools";
const PROGRESSIVE_LOAD: &str = "load_tool";
const PROGRESSIVE_UNLOAD: &str = "unload_tool";
/// JSON-RPC "method not found": a server that does not implement `ping`.
const METHOD_NOT_FOUND: i64 = -32601;

/// Builds a fresh transport for each (re)connection.
pub(crate) type TransportFactory =
    Arc<dyn Fn() -> BoxFuture<Result<Arc<dyn McpTransport>>> + Send + Sync>;

/// Replaces the default parsing of a `tools/call` result into the value the
/// model sees (upstream's `parse_tool_results`). Overrides
/// [`ToolResultContent`] entirely.
pub type ToolResultParser = Arc<dyn Fn(&CallToolResult) -> Result<Value> + Send + Sync>;

/// Replaces the default parsing of a `prompts/get` result (upstream's
/// `parse_prompt_results`).
pub type PromptResultParser = Arc<dyn Fn(&GetPromptResult) -> Result<Value> + Send + Sync>;

/// Approval policy for tools produced from an MCP server.
///
/// Mirrors upstream's `approval_mode`: `"always_require"`,
/// `"never_require"`, or a per-tool mapping. Names in the per-tool sets match
/// a tool's **raw remote name**, or its prefixed local name when the remote
/// name is already in normalized form — a normalized-only alias does not
/// match, as upstream has it. Rust's [`ApprovalMode`] has no unset state, so a
/// tool in neither set resolves to [`ApprovalMode::NeverRequire`].
#[derive(Debug, Clone, Default)]
pub enum McpApprovalMode {
    /// No tool produced by this server requires approval (default).
    #[default]
    NeverRequireAll,
    /// Every tool produced by this server requires approval before it runs.
    AlwaysRequireAll,
    /// Approval is required only for the named tools. `never_require` names
    /// resolve to [`ApprovalMode::NeverRequire`], as does every unnamed tool;
    /// listing them still matters, because a name that matches more than one
    /// remote tool is rejected as ambiguous.
    PerTool {
        always_require: HashSet<String>,
        never_require: HashSet<String>,
    },
}

impl McpApprovalMode {
    /// Require approval for every tool.
    pub fn always_require_all() -> Self {
        Self::AlwaysRequireAll
    }

    /// Require approval for no tool (the default).
    pub fn never_require_all() -> Self {
        Self::NeverRequireAll
    }

    /// Require approval only for the named tools.
    pub fn per_tool<I, S>(always_require: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::PerTool {
            always_require: always_require.into_iter().map(Into::into).collect(),
            never_require: HashSet::new(),
        }
    }

    /// Per-tool sets for both directions (upstream's `MCPSpecificApproval`).
    pub fn specific<A, N, S, T>(always_require: A, never_require: N) -> Self
    where
        A: IntoIterator<Item = S>,
        N: IntoIterator<Item = T>,
        S: Into<String>,
        T: Into<String>,
    {
        Self::PerTool {
            always_require: always_require.into_iter().map(Into::into).collect(),
            never_require: never_require.into_iter().map(Into::into).collect(),
        }
    }

    fn resolve(&self, candidates: &[String]) -> ApprovalMode {
        match self {
            Self::NeverRequireAll => ApprovalMode::NeverRequire,
            Self::AlwaysRequireAll => ApprovalMode::AlwaysRequire,
            Self::PerTool { always_require, .. } => {
                if candidates.iter().any(|c| always_require.contains(c)) {
                    ApprovalMode::AlwaysRequire
                } else {
                    ApprovalMode::NeverRequire
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn resolve_for_test(&self, name: &str) -> ApprovalMode {
        self.resolve(&[name.to_string()])
    }

    fn configured_names(&self) -> impl Iterator<Item = &String> {
        let (a, n) = match self {
            Self::PerTool {
                always_require,
                never_require,
            } => (Some(always_require), Some(never_require)),
            _ => (None, None),
        };
        a.into_iter().flatten().chain(n.into_iter().flatten())
    }
}

/// Extra argument names forwarded to `tools/call` beyond those a tool
/// declares in its `inputSchema.properties`. Mirrors upstream's
/// `additional_tool_argument_names`: global names apply to every tool, and
/// per-tool names (keyed by remote tool name) to that tool only.
#[derive(Debug, Clone, Default)]
pub struct AdditionalArgumentNames {
    global: HashSet<String>,
    per_tool: HashMap<String, HashSet<String>>,
}

impl AdditionalArgumentNames {
    /// Names forwarded for every tool.
    pub fn global<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            global: names.into_iter().map(Into::into).collect(),
            per_tool: HashMap::new(),
        }
    }

    /// Also forward `names` for the tool whose remote name is `tool` (`"*"`
    /// means every tool, as upstream's reserved key).
    pub fn for_tool<I, S>(mut self, tool: impl Into<String>, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let tool = tool.into();
        let names = names.into_iter().map(Into::into);
        if tool == "*" {
            self.global.extend(names);
        } else {
            self.per_tool.entry(tool).or_default().extend(names);
        }
        self
    }

    fn resolve(&self, remote: &str) -> HashSet<String> {
        let mut names = self.global.clone();
        if let Some(extra) = self.per_tool.get(remote) {
            names.extend(extra.iter().cloned());
        }
        names
    }
}

/// Options every MCP tool wrapper shares.
#[derive(Clone)]
pub(crate) struct Common {
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) allowed_tools: Option<HashSet<String>>,
    pub(crate) approval_mode: McpApprovalMode,
    pub(crate) sampling_handler: Option<SamplingHandler>,
    pub(crate) roots: Option<Vec<Root>>,
    pub(crate) load_tools: bool,
    pub(crate) load_prompts: bool,
    pub(crate) tool_name_prefix: Option<String>,
    pub(crate) result_content: ToolResultContent,
    pub(crate) result_parser: Option<ToolResultParser>,
    pub(crate) prompt_parser: Option<PromptResultParser>,
    pub(crate) extra_arguments: AdditionalArgumentNames,
    pub(crate) progressive: bool,
    pub(crate) always_load: HashSet<String>,
    pub(crate) logging_level: Option<McpLogLevel>,
}

impl Common {
    pub(crate) fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            allowed_tools: None,
            approval_mode: McpApprovalMode::default(),
            sampling_handler: None,
            roots: None,
            load_tools: true,
            load_prompts: true,
            tool_name_prefix: None,
            result_content: ToolResultContent::default(),
            result_parser: None,
            prompt_parser: None,
            extra_arguments: AdditionalArgumentNames::default(),
            progressive: false,
            always_load: HashSet::new(),
            logging_level: None,
        }
    }

    /// Check option combinations, as upstream's constructor does.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.progressive && !self.load_tools {
            return Err(Error::Configuration(
                "progressive disclosure requires load_tools(true)".into(),
            ));
        }
        Ok(())
    }

    fn local_name(&self, normalized: &str) -> String {
        build_prefixed_name(normalized, self.tool_name_prefix.as_deref())
    }
}

/// The exposed name for `normalized` under `prefix`: the normalized prefix
/// with trailing `_.-` stripped, `_`, and the name with leading `_.-`
/// stripped. Mirrors upstream `_build_prefixed_mcp_name`.
pub(crate) fn build_prefixed_name(normalized: &str, prefix: Option<&str>) -> String {
    let Some(prefix) = prefix.filter(|p| !p.is_empty()) else {
        return normalized.to_string();
    };
    let prefix = normalize_mcp_name(prefix);
    let prefix = prefix.trim_end_matches(['_', '.', '-']);
    if prefix.is_empty() {
        return normalized.to_string();
    }
    let name = normalized.trim_start_matches(['_', '.', '-']);
    if name.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}_{name}")
    }
}

/// The configuration names that may identify a function: always its raw
/// remote name, and its local name only when the remote name was already
/// normalized (so a normalized-only alias never matches). Upstream
/// `_mcp_config_candidate_names`.
fn candidate_names(local: &str, normalized: &str, remote: &str) -> Vec<String> {
    let mut names = vec![remote.to_string()];
    if normalized == remote && local != remote {
        names.push(local.to_string());
    }
    names
}

/// One function generated from the server's catalog.
#[derive(Clone)]
struct Generated {
    definition: ToolDefinition,
    remote: String,
    candidates: Vec<String>,
}

impl Generated {
    fn matches(&self, names: &HashSet<String>) -> bool {
        self.candidates.iter().any(|c| names.contains(c))
    }
}

/// A live connection to one MCP server, shared by every tool built from it.
pub(crate) struct McpSession {
    common: Common,
    factory: TransportFactory,
    client: RwLock<Option<Arc<McpClient>>>,
    connect_lock: tokio::sync::Mutex<()>,
    ping_available: AtomicBool,
    /// Tools loaded through progressive disclosure, by local name.
    loaded: Mutex<HashSet<String>>,
}

impl McpSession {
    pub(crate) fn new(common: Common, factory: TransportFactory) -> Arc<Self> {
        Arc::new(Self {
            common,
            factory,
            client: RwLock::new(None),
            connect_lock: tokio::sync::Mutex::new(()),
            ping_available: AtomicBool::new(true),
            loaded: Mutex::new(HashSet::new()),
        })
    }

    /// The current client, connecting first when there is none or the last
    /// one's connection is gone.
    pub(crate) async fn client(&self) -> Result<Arc<McpClient>> {
        if let Some(c) = self.client.read().await.as_ref() {
            if !c.is_closed() {
                return Ok(c.clone());
            }
        }
        self.reconnect(None).await
    }

    /// Replace the client. With `stale`, only if the current client is still
    /// that one — so concurrent callers that all saw one connection drop
    /// reconnect once between them.
    async fn reconnect(&self, stale: Option<&Arc<McpClient>>) -> Result<Arc<McpClient>> {
        let _guard = self.connect_lock.lock().await;
        if let Some(current) = self.client.read().await.as_ref() {
            let replaced_already = stale.is_some_and(|s| !Arc::ptr_eq(s, current));
            if replaced_already || (stale.is_none() && !current.is_closed()) {
                return Ok(current.clone());
            }
            let _ = current.close().await;
        }
        let transport = (self.factory)().await?;
        let mut client = McpClient::new(transport);
        if let Some(handler) = &self.common.sampling_handler {
            client = client.sampling_handler(handler.clone());
        }
        if let Some(roots) = &self.common.roots {
            client = client.roots(roots.clone());
        }
        client.initialize(CLIENT_NAME, CLIENT_VERSION).await?;
        if let Some(level) = self.common.logging_level {
            if client.supports_logging().await {
                if let Err(e) = client.set_logging_level(level).await {
                    tracing::warn!(error = %e, "failed to set MCP server log level");
                }
            }
        }
        let client = Arc::new(client);
        *self.client.write().await = Some(client.clone());
        self.ping_available.store(true, Ordering::Release);
        Ok(client)
    }

    /// Run `op` against the client, reconnecting and retrying once when the
    /// connection was lost underneath it — upstream's retry on
    /// `ClosedResourceError` / "session terminated". Any other failure is
    /// returned as is.
    async fn with_reconnect<T, F, Fut>(&self, op: F) -> Result<T>
    where
        F: Fn(Arc<McpClient>) -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let client = self.client().await?;
        match op(client.clone()).await {
            Err(e) if client.is_closed() => {
                tracing::info!(error = %e, "MCP connection lost; reconnecting once");
                let fresh = self.reconnect(Some(&client)).await?;
                op(fresh).await
            }
            other => other,
        }
    }

    /// Check the connection with `ping` before a call and reconnect when it
    /// fails, as upstream's `_ensure_connected` does. A server that answers
    /// "method not found" is not pinged again.
    async fn ensure_connected(&self) -> Result<()> {
        let client = self.client().await?;
        if !self.ping_available.load(Ordering::Acquire) {
            return Ok(());
        }
        match client.ping().await {
            Ok(()) => Ok(()),
            Err(e) if e.to_string().contains(&METHOD_NOT_FOUND.to_string()) => {
                self.ping_available.store(false, Ordering::Release);
                Ok(())
            }
            Err(e) => {
                tracing::info!(error = %e, "MCP ping failed; reconnecting");
                self.reconnect(Some(&client)).await.map(|_| ())
            }
        }
    }

    pub(crate) async fn close(&self) -> Result<()> {
        if let Some(c) = self.client.write().await.take() {
            c.close().await?;
        }
        Ok(())
    }

    pub(crate) async fn list_prompts(&self) -> Result<Vec<PromptDescriptor>> {
        self.with_reconnect(|c| async move { c.list_prompts_cached().await })
            .await
    }

    pub(crate) async fn get_prompt(&self, name: &str, arguments: Value) -> Result<GetPromptResult> {
        self.with_reconnect(|c| {
            let arguments = arguments.clone();
            async move { c.get_prompt(name, arguments).await }
        })
        .await
    }

    /// Every function the server's catalog yields, before `allowed_tools`
    /// and progressive disclosure: tools, then prompts as tools.
    async fn generate(self: &Arc<Self>, cached: bool) -> Result<Vec<Generated>> {
        if !cached {
            self.ensure_connected().await?;
        }
        let descriptors = self
            .with_reconnect(|c| async move {
                if cached {
                    c.list_tools_cached().await
                } else {
                    c.list_tools().await
                }
            })
            .await?;
        let mut functions = Vec::new();
        let mut remote_by_local: HashMap<String, String> = HashMap::new();
        for descriptor in &descriptors {
            let generated = self.tool_function(descriptor);
            if let Some(existing) = remote_by_local.get(&generated.definition.name) {
                if existing != &descriptor.name {
                    return Err(Error::tool(format!(
                        "MCP server advertised multiple tools that map to the same local function \
                         name: {existing:?} and {:?} both map to {:?}.",
                        descriptor.name, generated.definition.name
                    )));
                }
                continue;
            }
            remote_by_local.insert(generated.definition.name.clone(), descriptor.name.clone());
            functions.push(generated);
        }
        if self.common.load_prompts {
            for prompt in self.list_prompts().await? {
                let generated = self.prompt_function(&prompt);
                // A prompt never shadows a tool (or an earlier prompt).
                if remote_by_local.contains_key(&generated.definition.name) {
                    continue;
                }
                remote_by_local.insert(generated.definition.name.clone(), prompt.name.clone());
                functions.push(generated);
            }
        }
        self.validate_config_names(&functions)?;
        Ok(functions)
    }

    /// Reject a configured name that identifies more than one remote name.
    fn validate_config_names(&self, functions: &[Generated]) -> Result<()> {
        let configured: HashSet<&String> = self
            .common
            .allowed_tools
            .iter()
            .flatten()
            .chain(self.common.approval_mode.configured_names())
            .chain(self.common.always_load.iter())
            .collect();
        if configured.is_empty() {
            return Ok(());
        }
        let mut remote_by_name: HashMap<&str, &str> = HashMap::new();
        for f in functions {
            for name in &f.candidates {
                if !configured.contains(name) {
                    continue;
                }
                match remote_by_name.insert(name.as_str(), f.remote.as_str()) {
                    Some(previous) if previous != f.remote => {
                        return Err(Error::Configuration(format!(
                            "MCP configuration name {name:?} is ambiguous: it matches raw remote \
                             names {previous:?} and {:?}. Use an unambiguous name or a different \
                             tool_name_prefix.",
                            f.remote
                        )));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn tool_function(self: &Arc<Self>, d: &ToolDescriptor) -> Generated {
        let normalized = normalize_mcp_name(&d.name);
        let local = self.common.local_name(&normalized);
        let candidates = candidate_names(&local, &normalized, &d.name);
        // Some servers omit `properties` for zero-argument tools, which
        // providers reject; fill it in, as upstream does.
        let mut schema = d.input_schema.clone();
        if schema.get("type").and_then(Value::as_str) == Some("object")
            && schema.get("properties").is_none()
        {
            schema["properties"] = json!({});
        }
        let executor: Arc<dyn Tool> = Arc::new(McpToolExecutor {
            local_name: local.clone(),
            description: d.description.clone().unwrap_or_default(),
            parameters: schema,
            remote_name: d.name.clone(),
            declared: d.declared_parameters(),
            meta: d.meta.clone().filter(Value::is_object),
            session: self.clone(),
        });
        let mut definition = ToolDefinition::from_tool(executor);
        definition.approval_mode = self.common.approval_mode.resolve(&candidates);
        Generated {
            definition,
            remote: d.name.clone(),
            candidates,
        }
    }

    fn prompt_function(self: &Arc<Self>, p: &PromptDescriptor) -> Generated {
        let normalized = normalize_mcp_name(&p.name);
        let local = self.common.local_name(&normalized);
        let candidates = candidate_names(&local, &normalized, &p.name);
        let executor: Arc<dyn Tool> = Arc::new(McpPromptExecutor {
            local_name: local.clone(),
            description: p.description.clone().unwrap_or_default(),
            parameters: prompt_input_schema(p),
            remote_name: p.name.clone(),
            session: self.clone(),
        });
        let mut definition = ToolDefinition::from_tool(executor);
        definition.approval_mode = self.common.approval_mode.resolve(&candidates);
        Generated {
            definition,
            remote: p.name.clone(),
            candidates,
        }
    }

    /// The functions after `allowed_tools`.
    async fn filtered(self: &Arc<Self>, cached: bool) -> Result<Vec<Generated>> {
        let functions = self.generate(cached).await?;
        Ok(match &self.common.allowed_tools {
            None => functions,
            Some(allowed) => functions
                .into_iter()
                .filter(|f| f.matches(allowed))
                .collect(),
        })
    }

    /// The model-facing tool list: every allowed function, or under
    /// progressive disclosure the three loader tools plus `always_load` and
    /// previously loaded functions.
    pub(crate) async fn tool_definitions(
        self: &Arc<Self>,
        cached: bool,
    ) -> Result<Vec<ToolDefinition>> {
        self.common.validate()?;
        let functions = self.filtered(cached).await?;
        if !self.common.progressive {
            return Ok(functions.into_iter().map(|f| f.definition).collect());
        }
        let loaders = self.loader_names();
        let loaded = self
            .loaded
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut out = self.loader_tools();
        out.extend(
            functions
                .into_iter()
                .filter(|f| {
                    (f.matches(&self.common.always_load) || loaded.contains(&f.definition.name))
                        && !loaders.contains(&f.definition.name)
                })
                .map(|f| f.definition),
        );
        Ok(out)
    }

    // -- progressive disclosure ---------------------------------------------

    fn loader_name(&self, base: &str) -> String {
        build_prefixed_name(base, self.common.tool_name_prefix.as_deref())
    }

    fn loader_names(&self) -> HashSet<String> {
        [PROGRESSIVE_LIST, PROGRESSIVE_LOAD, PROGRESSIVE_UNLOAD]
            .into_iter()
            .map(|b| self.loader_name(b))
            .collect()
    }

    fn loader_tools(self: &Arc<Self>) -> Vec<ToolDefinition> {
        let tool_arg = json!({
            "type": "object",
            "properties": {
                "tool": {
                    "oneOf": [
                        { "type": "string" },
                        { "type": "array", "items": { "type": "string" } },
                    ],
                    "description": "The MCP tool name, or MCP tool names, to {verb}.",
                },
            },
            "required": ["tool"],
        });
        let with_verb = |verb: &str| {
            let mut schema = tool_arg.clone();
            schema["properties"]["tool"]["description"] =
                json!(format!("The MCP tool name, or MCP tool names, to {verb}."));
            schema
        };
        [
            (
                PROGRESSIVE_LIST,
                "List the MCP tools that can be loaded from this server.",
                json!({ "type": "object", "properties": {} }),
                LoaderKind::List,
            ),
            (
                PROGRESSIVE_LOAD,
                "Load an MCP tool from this server so it can be called on the next iteration.",
                with_verb("load"),
                LoaderKind::Load,
            ),
            (
                PROGRESSIVE_UNLOAD,
                "Unload an MCP tool from this server when it is no longer relevant.",
                with_verb("unload"),
                LoaderKind::Unload,
            ),
        ]
        .into_iter()
        .map(|(base, description, parameters, kind)| {
            let mut def = ToolDefinition::from_tool(Arc::new(LoaderTool {
                name: self.loader_name(base),
                description,
                parameters,
                kind,
                session: self.clone(),
            }));
            def.approval_mode = ApprovalMode::NeverRequire;
            def
        })
        .collect()
    }

    /// Resolve a requested name against the allowed functions: by
    /// configuration name first, then by local name.
    fn resolve_progressive<'a>(
        &self,
        functions: &'a [Generated],
        name: &str,
    ) -> std::result::Result<&'a Generated, String> {
        let wanted = HashSet::from([name.to_string()]);
        let mut matches: Vec<&Generated> =
            functions.iter().filter(|f| f.matches(&wanted)).collect();
        for f in functions.iter().filter(|f| f.definition.name == name) {
            if !matches.iter().any(|m| std::ptr::eq(*m, f)) {
                matches.push(f);
            }
        }
        match matches.as_slice() {
            [] => {
                let loaders = self.loader_names();
                let available: Vec<&str> = functions
                    .iter()
                    .map(|f| f.definition.name.as_str())
                    .filter(|n| !loaders.contains(*n))
                    .collect();
                let available = if available.is_empty() {
                    "none".to_string()
                } else {
                    available.join(", ")
                };
                Err(format!(
                    "MCP tool '{name}' is not available. Available tools: {available}."
                ))
            }
            [one] => Ok(one),
            _ => Err(format!("MCP tool name '{name}' is ambiguous.")),
        }
    }

    fn requested_names(arguments: &Value) -> Result<Vec<String>> {
        let bad =
            || Error::tool("Progressive MCP tool request must be a string or a list of strings.");
        match arguments.get("tool") {
            Some(Value::String(s)) => Ok(vec![s.clone()]),
            Some(Value::Array(items)) => items
                .iter()
                .map(|i| i.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(bad),
            _ => Err(bad()),
        }
    }

    async fn progressive_list(self: &Arc<Self>, ctx: &FunctionInvocationContext) -> Result<Value> {
        let functions = self.filtered(true).await?;
        let loaders = self.loader_names();
        let loaded = self
            .loaded
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let live = |name: &str| ctx.tools.as_ref().is_some_and(|t| t.contains(name));
        Ok(Value::Array(
            functions
                .iter()
                .filter(|f| !loaders.contains(&f.definition.name))
                .map(|f| {
                    let always = f.matches(&self.common.always_load);
                    let name = &f.definition.name;
                    json!({
                        "name": name,
                        "remote_name": f.remote,
                        "description": f.definition.description,
                        "parameters": f.definition.parameters,
                        "approval_mode": match f.definition.approval_mode {
                            ApprovalMode::AlwaysRequire => "always_require",
                            ApprovalMode::NeverRequire => "never_require",
                        },
                        "loaded": always || loaded.contains(name) || live(name),
                        "always_loaded": always,
                    })
                })
                .collect(),
        ))
    }

    async fn progressive_load(
        self: &Arc<Self>,
        arguments: &Value,
        ctx: &FunctionInvocationContext,
    ) -> Result<Value> {
        let Some(live) = ctx.tools.as_ref() else {
            return Err(Error::tool(
                "load_tool can only be used inside an agent function-calling run.",
            ));
        };
        let functions = self.filtered(true).await?;
        let loaders = self.loader_names();
        let mut messages = Vec::new();
        let mut to_load: Vec<ToolDefinition> = Vec::new();
        for name in Self::requested_names(arguments)? {
            let f = match self.resolve_progressive(&functions, &name) {
                Ok(f) => f,
                Err(m) => {
                    messages.push(m);
                    continue;
                }
            };
            let local = &f.definition.name;
            if loaders.contains(local) {
                messages.push(self.loader_collision(local, &loaders));
            } else if live.contains(local) {
                self.loaded
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(local.clone());
                messages.push(format!("MCP tool '{local}' is already available."));
            } else if to_load.iter().any(|d| &d.name == local) {
                messages.push(format!("MCP tool '{local}' is already queued to load."));
            } else {
                to_load.push(f.definition.clone());
                messages.push(format!(
                    "Loaded MCP tool '{local}'. It is available on the next model iteration."
                ));
            }
        }
        if messages.is_empty() {
            return Ok(json!("No MCP tools requested."));
        }
        if !to_load.is_empty() {
            let names: Vec<String> = to_load.iter().map(|d| d.name.clone()).collect();
            live.add_tools(to_load)
                .map_err(|e| Error::tool(e.to_string()))?;
            self.loaded
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(names);
        }
        Ok(json!(messages.join("\n")))
    }

    async fn progressive_unload(
        self: &Arc<Self>,
        arguments: &Value,
        ctx: &FunctionInvocationContext,
    ) -> Result<Value> {
        let Some(live) = ctx.tools.as_ref() else {
            return Err(Error::tool(
                "unload_tool can only be used inside an agent function-calling run.",
            ));
        };
        let functions = self.filtered(true).await?;
        let loaders = self.loader_names();
        let loaded = self
            .loaded
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut messages = Vec::new();
        let mut to_unload: Vec<String> = Vec::new();
        for name in Self::requested_names(arguments)? {
            if loaders.contains(&name) {
                messages.push(format!("MCP loader tool '{name}' cannot be unloaded."));
                continue;
            }
            let f = match self.resolve_progressive(&functions, &name) {
                Ok(f) => f,
                Err(m) => {
                    messages.push(m);
                    continue;
                }
            };
            let local = &f.definition.name;
            if loaders.contains(local) {
                messages.push(self.loader_collision(local, &loaders));
            } else if f.matches(&self.common.always_load) {
                messages.push(format!(
                    "MCP tool '{local}' is configured in always_load and cannot be unloaded."
                ));
            } else if to_unload.contains(local) {
                messages.push(format!("MCP tool '{local}' is already queued to unload."));
            } else if !loaded.contains(local) && !live.contains(local) {
                messages.push(format!("MCP tool '{local}' is not currently loaded."));
            } else {
                to_unload.push(local.clone());
                messages.push(format!(
                    "Unloaded MCP tool '{local}'. It will be removed on the next model iteration."
                ));
            }
        }
        if messages.is_empty() {
            return Ok(json!("No MCP tools requested."));
        }
        if !to_unload.is_empty() {
            live.remove_tools(to_unload.iter().map(String::as_str));
            let mut loaded = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
            for name in &to_unload {
                loaded.remove(name);
            }
        }
        Ok(json!(messages.join("\n")))
    }

    fn loader_collision(&self, local: &str, loaders: &HashSet<String>) -> String {
        let mut names: Vec<&String> = loaders.iter().collect();
        names.sort();
        let names: Vec<&str> = names.into_iter().map(String::as_str).collect();
        format!(
            "MCP tool '{local}' conflicts with progressive disclosure loader tool name(s): {}. \
             Set tool_name_prefix or exclude the colliding MCP tool.",
            names.join(", ")
        )
    }

    // -- calls ------------------------------------------------------------------

    async fn call_tool(
        &self,
        remote: &str,
        arguments: Value,
        declared: &HashSet<String>,
        meta: Option<Value>,
    ) -> Result<Value> {
        if !self.common.load_tools {
            return Err(Error::tool(
                "Tools are not loaded for this server, please set load_tools(true).",
            ));
        }
        // Forward only what the tool declares plus configured extras, so an
        // argument the model invented never reaches the server.
        let extras = self.common.extra_arguments.resolve(remote);
        let arguments = match arguments {
            Value::Object(map) => Value::Object(
                map.into_iter()
                    .filter(|(k, _)| k != "_meta" && (declared.contains(k) || extras.contains(k)))
                    .collect::<Map<String, Value>>(),
            ),
            Value::Null => json!({}),
            other => other,
        };
        let result = self
            .with_reconnect(|c| {
                let arguments = arguments.clone();
                let meta = meta.clone();
                async move { c.call_tool_with_meta(remote, arguments, meta).await }
            })
            .await?;
        if result.is_error {
            return Err(Error::tool(match &self.common.result_parser {
                Some(parse) => match parse(&result)? {
                    Value::String(s) => s,
                    other => other.to_string(),
                },
                None => result.error_message(),
            }));
        }
        match &self.common.result_parser {
            Some(parse) => parse(&result),
            None => Ok(result.to_value_with(self.common.result_content)),
        }
    }

    async fn call_prompt(&self, remote: &str, arguments: Value) -> Result<Value> {
        if !self.common.load_prompts {
            return Err(Error::tool(
                "Prompts are not loaded for this server, please set load_prompts(true).",
            ));
        }
        let result = self.get_prompt(remote, arguments).await?;
        match &self.common.prompt_parser {
            Some(parse) => parse(&result),
            None => Ok(prompt_result_value(&result)),
        }
    }
}

/// A prompt's arguments as a JSON schema: every argument a string, described
/// when the server describes it, required when it says so. Upstream
/// `_get_input_model_from_mcp_prompt`.
fn prompt_input_schema(p: &PromptDescriptor) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for arg in p.arguments.iter().flatten() {
        let mut schema = json!({ "type": "string" });
        if let Some(d) = &arg.description {
            schema["description"] = json!(d);
        }
        properties.insert(arg.name.clone(), schema);
        if arg.required.unwrap_or(false) {
            required.push(json!(arg.name));
        }
    }
    let mut schema = json!({ "type": "object", "properties": properties });
    if !required.is_empty() {
        schema["required"] = Value::Array(required);
    }
    schema
}

/// A `prompts/get` result as one string, as upstream's
/// `_parse_prompt_result_from_mcp`: each message's text (media and blobs as
/// JSON), one part returned bare and several as a JSON array of strings.
fn prompt_result_value(result: &GetPromptResult) -> Value {
    let parts: Vec<String> = result
        .messages
        .iter()
        .map(|m| match m.content_block() {
            ContentBlock::Text(t) => t,
            ContentBlock::Image { data, mime_type } => {
                json!({ "type": "image", "data": data, "mimeType": mime_type }).to_string()
            }
            ContentBlock::Audio { data, mime_type } => {
                json!({ "type": "audio", "data": data, "mimeType": mime_type }).to_string()
            }
            ContentBlock::Resource(r) => match r.get("text").and_then(Value::as_str) {
                Some(text) => text.to_string(),
                None => json!({
                    "type": "blob",
                    "data": r.get("blob"),
                    "mimeType": r.get("mimeType"),
                })
                .to_string(),
            },
            other => other.to_json().to_string(),
        })
        .collect();
    match parts.as_slice() {
        [] => json!(""),
        [one] => json!(one),
        many => json!(serde_json::to_string(many).unwrap_or_default()),
    }
}

/// Forwards `invoke` to `tools/call` with the server's raw tool name. The
/// remote name is fixed here, never taken from arguments, so a model cannot
/// redirect a call to another tool.
struct McpToolExecutor {
    local_name: String,
    description: String,
    parameters: Value,
    remote_name: String,
    declared: HashSet<String>,
    meta: Option<Value>,
    session: Arc<McpSession>,
}

#[async_trait]
impl Tool for McpToolExecutor {
    fn name(&self) -> &str {
        &self.local_name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.parameters.clone()
    }
    async fn invoke(&self, arguments: Value) -> Result<Value> {
        self.session
            .call_tool(
                &self.remote_name,
                arguments,
                &self.declared,
                self.meta.clone(),
            )
            .await
    }
}

/// A server prompt exposed as a tool, returning the rendered prompt text.
struct McpPromptExecutor {
    local_name: String,
    description: String,
    parameters: Value,
    remote_name: String,
    session: Arc<McpSession>,
}

#[async_trait]
impl Tool for McpPromptExecutor {
    fn name(&self) -> &str {
        &self.local_name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.parameters.clone()
    }
    async fn invoke(&self, arguments: Value) -> Result<Value> {
        self.session.call_prompt(&self.remote_name, arguments).await
    }
}

#[derive(Clone, Copy)]
enum LoaderKind {
    List,
    Load,
    Unload,
}

/// One of the three progressive-disclosure tools.
struct LoaderTool {
    name: String,
    description: &'static str,
    parameters: Value,
    kind: LoaderKind,
    session: Arc<McpSession>,
}

#[async_trait]
impl Tool for LoaderTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        self.description
    }
    fn parameters_schema(&self) -> Value {
        self.parameters.clone()
    }
    async fn invoke(&self, arguments: Value) -> Result<Value> {
        self.invoke_in_context(
            arguments,
            &FunctionInvocationContext::new(&self.name, Value::Null),
        )
        .await
    }
    async fn invoke_in_context(
        &self,
        arguments: Value,
        ctx: &FunctionInvocationContext,
    ) -> Result<Value> {
        match self.kind {
            LoaderKind::List => self.session.progressive_list(ctx).await,
            LoaderKind::Load => self.session.progressive_load(&arguments, ctx).await,
            LoaderKind::Unload => self.session.progressive_unload(&arguments, ctx).await,
        }
    }
}
