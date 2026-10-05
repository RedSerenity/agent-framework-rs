//! `InvokeMcpTool` and the MCP handler abstraction (port of
//! `_executors_mcp.py` and `_mcp_handler.py`, mirroring .NET's
//! `IMcpToolHandler`).
//!
//! A workflow containing `InvokeMcpTool` only builds when an
//! [`McpToolHandler`] is supplied to the
//! [`WorkflowFactory`](super::WorkflowFactory). The executor evaluates
//! `serverUrl`, `toolName` (both required, non-empty), `serverLabel`,
//! `arguments` (nulls preserved), `headers` (empty values dropped),
//! `connection.name`, `requireApproval` and `output.autoSend`, then:
//!
//! * on success stores the parsed outputs (text JSON-parsed when possible,
//!   data/URI content as its URI) as a list at `output.result`, a single
//!   `tool`-role message with all outputs at `output.messages`, auto-sends
//!   the rendered outputs (strings newline-joined; a single non-string as
//!   JSON; otherwise the JSON list), and with `conversationId` appends an
//!   assistant message carrying the outputs to that conversation;
//! * on a tool error stores `"Error: <message>"` at `output.result` and
//!   continues (parity with .NET `AssignErrorAsync`).
//!
//! # Approval
//!
//! With `requireApproval`, the run pauses with a
//! `{"type": "MCPToolApprovalRequest", request_id, tool_name, server_url,
//! server_label, arguments, header_names, connection_name, metadata,
//! header_binding}` request. Header **values** are never placed in the
//! request; instead `header_binding` is an HMAC-SHA256 (keyed by a secret
//! kept in workflow state, never in the payload) over the request id and the
//! lower-cased, sorted headers. On resume the headers are re-evaluated and
//! re-bound; if they changed (or cannot be verified) a fresh approval is
//! requested instead of invoking the tool. Rejection stores
//! `"Error: MCP tool invocation was not approved by user."`.
//!
//! The 64-hex-character key is drawn from two v4 UUIDs (244 random bits)
//! where upstream uses `secrets.token_hex(32)`.

use agent_framework_core::error::Error as CoreError;
use agent_framework_core::types::{Content, Message, Role};
use agent_framework_core::workflow::{RequestResponse, WorkflowContext};
use async_trait::async_trait;
use hmac::{Hmac, Mac};
use serde_json::{json, Map, Value as Json};
use sha2::Sha256;

use super::executor::{field, path_ref, ActionResult, DeclarativeExecutor};
use super::messages::{action_complete, message_to_json};
use super::state::{py_str, py_truthy, DeclarativeState, StateError};
use super::tools::parse_approval;

/// The reserved tool name that lists the server's tools instead of calling one.
pub const LIST_TOOLS_TOOL_NAME: &str = "tools/list";

const HEADER_BINDING_KEY: &str = "_declarative_mcp_header_binding_key";

/// An MCP tool call for an [`McpToolHandler`] (upstream `MCPToolInvocation`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct McpToolInvocation {
    /// Absolute MCP server URL.
    pub server_url: String,
    /// The tool to invoke (or [`LIST_TOOLS_TOOL_NAME`]).
    pub tool_name: String,
    /// Optional human-readable label.
    pub server_label: Option<String>,
    /// Evaluated tool arguments.
    pub arguments: Map<String, Json>,
    /// Outbound headers (e.g. authentication).
    pub headers: Vec<(String, String)>,
    /// Optional connection name for handlers that resolve credentials.
    pub connection_name: Option<String>,
}

/// The outcome of an MCP tool call (upstream `MCPToolResult`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct McpToolResult {
    /// Tool output contents.
    pub outputs: Vec<Content>,
    /// Whether the tool reported an error.
    pub is_error: bool,
    /// The error message when `is_error`.
    pub error_message: Option<String>,
}

impl McpToolResult {
    /// An error result carrying `Error: <message>` text.
    pub fn error(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            outputs: vec![Content::text(format!("Error: {message}"))],
            is_error: true,
            error_message: Some(message),
        }
    }
}

/// Failures an [`McpToolHandler`] may raise.
#[derive(Debug, thiserror::Error)]
pub enum McpToolError {
    /// A tool/transport/MCP-protocol failure: normalized into an error
    /// result stored at `output.result` (upstream's narrow catch).
    #[error("{0}")]
    Tool(String),
    /// Any other failure: propagated, failing the action.
    #[error("{0}")]
    Other(String),
}

/// Dispatches `InvokeMcpTool` calls. Implementations own allow-listing,
/// SSRF guards, authentication and connection resolution.
#[async_trait]
pub trait McpToolHandler: Send + Sync {
    /// Invoke the tool described by `invocation`.
    async fn invoke_tool(
        &self,
        invocation: McpToolInvocation,
    ) -> Result<McpToolResult, McpToolError>;
}

/// Parse outputs for `output.result` (upstream `_parse_outputs`).
pub(crate) fn parse_outputs(outputs: &[Content]) -> Vec<Json> {
    outputs
        .iter()
        .map(|c| match c {
            Content::Text(t) => {
                serde_json::from_str(&t.text).unwrap_or_else(|_| Json::String(t.text.clone()))
            }
            Content::Data(d) => Json::String(d.uri.clone()),
            Content::Uri(u) => Json::String(u.uri.clone()),
            other => Json::String(serde_json::to_string(other).unwrap_or_default()),
        })
        .collect()
}

/// Render outputs for auto-send (upstream `_format_outputs_for_send`).
pub(crate) fn format_outputs_for_send(parsed: &[Json]) -> String {
    if parsed.is_empty() {
        return String::new();
    }
    if parsed.iter().all(Json::is_string) {
        return parsed
            .iter()
            .filter_map(Json::as_str)
            .collect::<Vec<_>>()
            .join("\n");
    }
    if parsed.len() == 1 {
        return parsed[0].to_string();
    }
    Json::Array(parsed.to_vec()).to_string()
}

/// JSON with Python's `ensure_ascii=True` escaping.
fn ascii_json(v: &Json) -> String {
    let s = v.to_string();
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut buf = [0u16; 2];
            for unit in c.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

fn action_error(msg: impl Into<String>) -> CoreError {
    CoreError::Workflow(format!("declarative action error: {}", msg.into()))
}

fn truthy_flag(v: &Json) -> bool {
    match v {
        Json::Bool(b) => *b,
        Json::String(s) => matches!(s.trim().to_lowercase().as_str(), "true" | "1" | "yes"),
        other => py_truthy(other),
    }
}

impl DeclarativeExecutor {
    fn mcp_required_text(&self, state: &DeclarativeState, key: &str) -> Result<String, CoreError> {
        let raw = field(&self.def, key)
            .ok_or_else(|| action_error(format!("InvokeMcpTool requires a '{key}' field.")))?;
        match state.eval_if_expression(raw)? {
            Json::String(s) if !s.is_empty() => Ok(s),
            _ => Err(action_error(format!(
                "InvokeMcpTool '{key}' evaluated to an empty value."
            ))),
        }
    }

    fn mcp_flag(
        &self,
        state: &DeclarativeState,
        v: Option<&Json>,
        default: bool,
    ) -> Result<bool, StateError> {
        match v.filter(|v| !v.is_null()) {
            None => Ok(default),
            Some(v) => Ok(truthy_flag(&state.eval_if_expression(v)?)),
        }
    }

    fn mcp_output_path(&self, key: &str) -> Option<String> {
        path_ref(field(&self.def, "output").and_then(|o| o.get(key)))
    }

    fn mcp_auto_send(&self, state: &DeclarativeState) -> Result<bool, StateError> {
        match field(&self.def, "output") {
            Some(Json::Object(o)) => self.mcp_flag(state, o.get("autoSend"), true),
            _ => Ok(true),
        }
    }

    fn mcp_conversation_id(&self, state: &DeclarativeState) -> Result<Option<String>, StateError> {
        let Some(Json::String(expr)) = field(&self.def, "conversationId") else {
            return Ok(None);
        };
        if expr.is_empty() {
            return Ok(None);
        }
        Ok(match state.eval(expr)? {
            Json::Null => None,
            v => Some(py_str(&v)).filter(|s| !s.is_empty()),
        })
    }

    async fn bind_headers(
        &self,
        ctx: &WorkflowContext,
        request_id: &str,
        headers: &[(String, String)],
        create_key: bool,
    ) -> Result<Option<String>, CoreError> {
        let shared = ctx.shared_state();
        let key = match shared.get(HEADER_BINDING_KEY).await {
            Some(Json::String(k)) => k,
            Some(_) => return Err(action_error("Invalid MCP approval header binding state.")),
            None if create_key => {
                let k = format!(
                    "{}{}",
                    uuid::Uuid::new_v4().simple(),
                    uuid::Uuid::new_v4().simple()
                );
                shared
                    .set(HEADER_BINDING_KEY, Json::String(k.clone()))
                    .await;
                k
            }
            None => return Ok(None),
        };
        if key.len() != 64 || !key.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
            return Err(action_error("Invalid MCP approval header binding state."));
        }
        let mut canonical: Vec<(String, String)> = headers
            .iter()
            .map(|(k, v)| (k.to_lowercase(), v.clone()))
            .collect();
        canonical.sort_by(|a, b| a.0.cmp(&b.0));
        let payload = ascii_json(&json!([
            request_id,
            canonical
                .iter()
                .map(|(k, v)| json!([k, v]))
                .collect::<Vec<_>>()
        ]));
        let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes())
            .map_err(|e| action_error(e.to_string()))?;
        mac.update(payload.as_bytes());
        let digest = mac.finalize().into_bytes();
        Ok(Some(digest.iter().map(|b| format!("{b:02x}")).collect()))
    }

    async fn request_mcp_approval(
        &self,
        ctx: &WorkflowContext,
        invocation: &McpToolInvocation,
        metadata: Json,
    ) -> ActionResult {
        let request_id = uuid::Uuid::new_v4().to_string();
        let binding = if invocation.headers.is_empty() {
            None
        } else {
            self.bind_headers(ctx, &request_id, &invocation.headers, true)
                .await?
        };
        let mut header_names: Vec<&str> =
            invocation.headers.iter().map(|(k, _)| k.as_str()).collect();
        header_names.sort_unstable();
        ctx.request_info(json!({
            "type": "MCPToolApprovalRequest",
            "request_id": request_id,
            "tool_name": invocation.tool_name,
            "server_url": invocation.server_url,
            "server_label": invocation.server_label,
            "arguments": invocation.arguments,
            "header_names": header_names,
            "connection_name": invocation.connection_name,
            "metadata": metadata,
            "header_binding": binding,
        }))
        .await
    }

    async fn call_mcp(&self, invocation: McpToolInvocation) -> Result<McpToolResult, CoreError> {
        let handler = self
            .rt
            .mcp
            .clone()
            .ok_or_else(|| action_error("no McpToolHandler is configured"))?;
        match handler.invoke_tool(invocation).await {
            Ok(r) => Ok(r),
            Err(McpToolError::Tool(msg)) => Ok(McpToolResult::error(msg)),
            Err(McpToolError::Other(msg)) => Err(action_error(msg)),
        }
    }

    async fn process_mcp_result(
        &self,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
        result: McpToolResult,
        auto_send: bool,
        conversation_id: Option<String>,
    ) -> ActionResult {
        let result_path = self.mcp_output_path("result");
        if result.is_error {
            if let Some(path) = result_path {
                let msg = result
                    .error_message
                    .unwrap_or_else(|| "MCP tool invocation failed.".into());
                state.set(&path, Json::String(format!("Error: {msg}")))?;
            }
            return Ok(());
        }
        let parsed = parse_outputs(&result.outputs);
        if let Some(path) = &result_path {
            if !parsed.is_empty() {
                state.set(path, Json::Array(parsed.clone()))?;
            }
        }
        if let Some(path) = self.mcp_output_path("messages") {
            let m = Message::with_contents(Role::new(Role::TOOL), result.outputs.clone());
            state.set(&path, message_to_json(&m))?;
        }
        if auto_send && !parsed.is_empty() {
            ctx.yield_output(Json::String(format_outputs_for_send(&parsed)))
                .await?;
        }
        if let Some(id) = conversation_id {
            let m = Message::with_contents(Role::new(Role::ASSISTANT), result.outputs);
            state.append(
                &format!("System.conversations.{id}.messages"),
                message_to_json(&m),
            )?;
        }
        Ok(())
    }

    pub(crate) async fn invoke_mcp_tool(
        &self,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let server_url = self.mcp_required_text(state, "serverUrl")?;
        let tool_name = self.mcp_required_text(state, "toolName")?;
        let server_label = match field(&self.def, "serverLabel").filter(|v| !v.is_null()) {
            Some(v) => match state.eval_if_expression(v)? {
                Json::Null => None,
                other => Some(py_str(&other)).filter(|s| !s.is_empty()),
            },
            None => None,
        };
        let mut arguments = Map::new();
        if let Some(Json::Object(m)) = field(&self.def, "arguments") {
            for (k, v) in m {
                if !k.is_empty() {
                    arguments.insert(k.clone(), state.eval_if_expression(v)?);
                }
            }
        }
        let headers = self.eval_headers(state)?;
        let invocation = McpToolInvocation {
            server_url,
            tool_name,
            server_label,
            arguments,
            headers,
            connection_name: self.connection_name(state)?,
        };
        let require_approval = self.mcp_flag(state, field(&self.def, "requireApproval"), false)?;
        let auto_send = self.mcp_auto_send(state)?;
        let conversation_id = self.mcp_conversation_id(state)?;
        if require_approval {
            return self
                .request_mcp_approval(
                    ctx,
                    &invocation,
                    json!({"conversation_id": conversation_id}),
                )
                .await;
        }
        let result = self.call_mcp(invocation).await?;
        self.process_mcp_result(state, ctx, result, auto_send, conversation_id)
            .await?;
        ctx.send_message(action_complete()).await
    }

    pub(crate) async fn mcp_approval_response(
        &self,
        state: &mut DeclarativeState,
        response: RequestResponse,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let (approved, _reason) = parse_approval(&response.data)?;
        let original = &response.original_request;
        let metadata = original.get("metadata").cloned().unwrap_or(json!({}));
        let conversation_id = metadata
            .get("conversation_id")
            .and_then(Json::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let auto_send = self.mcp_auto_send(state)?;
        if !approved {
            if let Some(path) = self.mcp_output_path("result") {
                state.set(
                    &path,
                    json!("Error: MCP tool invocation was not approved by user."),
                )?;
            }
            return ctx.send_message(action_complete()).await;
        }
        let text = |k: &str| original.get(k).and_then(Json::as_str).map(str::to_string);
        let invocation = McpToolInvocation {
            server_url: text("server_url").unwrap_or_default(),
            tool_name: text("tool_name").unwrap_or_default(),
            server_label: text("server_label"),
            arguments: original
                .get("arguments")
                .and_then(Json::as_object)
                .cloned()
                .unwrap_or_default(),
            headers: self.eval_headers(state)?,
            connection_name: text("connection_name"),
        };
        let had_header_names = original
            .get("header_names")
            .and_then(Json::as_array)
            .is_some_and(|a| !a.is_empty());
        if !invocation.headers.is_empty() || had_header_names {
            let binding = original.get("header_binding").and_then(Json::as_str);
            let request_id = text("request_id").unwrap_or_default();
            let expected = self
                .bind_headers(ctx, &request_id, &invocation.headers, false)
                .await?;
            let verified = match (binding, expected) {
                (Some(b), Some(e)) => constant_time_eq(b.as_bytes(), e.as_bytes()),
                _ => false,
            };
            if !verified {
                tracing::warn!(
                    "InvokeMcpTool: MCP header context changed or could not be verified; requesting fresh approval."
                );
                return self.request_mcp_approval(ctx, &invocation, metadata).await;
            }
        }
        let result = self.call_mcp(invocation).await?;
        self.process_mcp_result(state, ctx, result, auto_send, conversation_id)
            .await?;
        ctx.send_message(action_complete()).await
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(feature = "mcp")]
pub use default_handler::DefaultMcpToolHandler;

#[cfg(feature = "mcp")]
mod default_handler {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Arc;

    use agent_framework_mcp::{McpClient, McpStreamableHttpTransport};
    use tokio::sync::Mutex;

    const DEFAULT_CACHE_MAX_SIZE: usize = 32;

    type CacheKey = (String, Option<String>, Option<String>, String);
    type Cache = (HashMap<CacheKey, Arc<McpClient>>, VecDeque<CacheKey>);

    /// The default [`McpToolHandler`], backed by `agent-framework-mcp`'s
    /// streamable-HTTP client.
    ///
    /// Caches one initialized [`McpClient`] per `(server_url, server_label,
    /// connection_name, headers)` in a bounded (32 entry, oldest-evicted)
    /// cache, so distinct auth headers never share a session. Header *names*
    /// are lower-cased for the key only. [`LIST_TOOLS_TOOL_NAME`] is
    /// answered client-side with a `{"tools": [{name, description,
    /// inputSchema, outputSchema}]}` JSON catalog and rejects arguments.
    /// Connection failures and tool errors become error results. It performs
    /// **no** URL filtering or SSRF protection.
    pub struct DefaultMcpToolHandler {
        cache: Mutex<Cache>,
        max_size: usize,
    }

    impl Default for DefaultMcpToolHandler {
        fn default() -> Self {
            Self::new()
        }
    }

    impl DefaultMcpToolHandler {
        /// A handler with the default cache size.
        pub fn new() -> Self {
            Self::with_cache_size(DEFAULT_CACHE_MAX_SIZE)
        }

        /// A handler with a custom cache size (minimum 1).
        pub fn with_cache_size(max_size: usize) -> Self {
            Self {
                cache: Mutex::new((HashMap::new(), VecDeque::new())),
                max_size: max_size.max(1),
            }
        }

        fn key(inv: &McpToolInvocation) -> CacheKey {
            let mut headers: Vec<(String, String)> = inv
                .headers
                .iter()
                .map(|(k, v)| (k.to_lowercase(), v.clone()))
                .collect();
            headers.sort();
            (
                inv.server_url.clone(),
                inv.server_label.clone(),
                inv.connection_name.clone(),
                json!(headers).to_string(),
            )
        }

        async fn client(&self, inv: &McpToolInvocation) -> Result<Arc<McpClient>, String> {
            let key = Self::key(inv);
            let mut guard = self.cache.lock().await;
            if let Some(c) = guard.0.get(&key) {
                return Ok(c.clone());
            }
            let mut header_map = reqwest::header::HeaderMap::new();
            for (k, v) in &inv.headers {
                let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                    .map_err(|e| e.to_string())?;
                let value = reqwest::header::HeaderValue::from_str(v).map_err(|e| e.to_string())?;
                header_map.append(name, value);
            }
            let transport =
                McpStreamableHttpTransport::new(inv.server_url.clone(), header_map, None)
                    .map_err(|e| e.to_string())?;
            let client = Arc::new(McpClient::new(Arc::new(transport)));
            client
                .initialize("agent-framework-declarative", env!("CARGO_PKG_VERSION"))
                .await
                .map_err(|e| e.to_string())?;
            if guard.0.len() >= self.max_size {
                if let Some(old) = guard.1.pop_front() {
                    if let Some(c) = guard.0.remove(&old) {
                        let _ = c.close().await;
                    }
                }
            }
            guard.0.insert(key.clone(), client.clone());
            guard.1.push_back(key);
            Ok(client)
        }
    }

    #[async_trait]
    impl McpToolHandler for DefaultMcpToolHandler {
        async fn invoke_tool(
            &self,
            invocation: McpToolInvocation,
        ) -> Result<McpToolResult, McpToolError> {
            if invocation.tool_name == LIST_TOOLS_TOOL_NAME && !invocation.arguments.is_empty() {
                return Ok(McpToolResult::error(format!(
                    "The reserved MCP '{LIST_TOOLS_TOOL_NAME}' operation does not accept tool arguments."
                )));
            }
            let client = match self.client(&invocation).await {
                Ok(c) => c,
                Err(e) => {
                    return Ok(McpToolResult::error(format!(
                        "Failed to connect to MCP server: {e}"
                    )))
                }
            };
            if invocation.tool_name == LIST_TOOLS_TOOL_NAME {
                let tools = client
                    .list_tools()
                    .await
                    .map_err(|e| McpToolError::Tool(e.to_string()))?;
                let payload = json!({
                    "tools": tools.iter().map(|t| json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                        "outputSchema": t.output_schema,
                    })).collect::<Vec<_>>()
                });
                let text = serde_json::to_string_pretty(&payload)
                    .map_err(|e| McpToolError::Other(e.to_string()))?;
                return Ok(McpToolResult {
                    outputs: vec![Content::text(text)],
                    is_error: false,
                    error_message: None,
                });
            }
            let result = client
                .call_tool(
                    &invocation.tool_name,
                    Json::Object(invocation.arguments.clone()),
                )
                .await
                .map_err(|e| McpToolError::Tool(e.to_string()))?;
            if result.is_error {
                return Ok(McpToolResult::error(result.error_message()));
            }
            Ok(McpToolResult {
                outputs: result.content.iter().map(|c| c.to_core_content()).collect(),
                is_error: false,
                error_message: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_parsing_and_rendering() {
        let outputs = vec![Content::text("{\"a\": 1}"), Content::text("plain")];
        let parsed = parse_outputs(&outputs);
        assert_eq!(parsed, vec![json!({"a": 1}), json!("plain")]);
        assert_eq!(format_outputs_for_send(&parsed), "[{\"a\":1},\"plain\"]");
        assert_eq!(format_outputs_for_send(&[json!("a"), json!("b")]), "a\nb");
        assert_eq!(format_outputs_for_send(&[json!(42)]), "42");
        assert_eq!(format_outputs_for_send(&[]), "");
    }

    #[test]
    fn ascii_json_escapes_like_python() {
        assert_eq!(
            ascii_json(&json!(["é", "😀"])),
            "[\"\\u00e9\",\"\\ud83d\\ude00\"]"
        );
    }

    #[test]
    fn flags_accept_strings() {
        assert!(truthy_flag(&json!("Yes")));
        assert!(!truthy_flag(&json!("no")));
        assert!(truthy_flag(&json!(1)));
        assert!(constant_time_eq(b"ab", b"ab"));
        assert!(!constant_time_eq(b"ab", b"ac"));
    }
}
