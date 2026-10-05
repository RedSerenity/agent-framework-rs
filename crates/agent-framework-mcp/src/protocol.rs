//! JSON-RPC 2.0 message shapes and MCP-specific payload types.
//!
//! This is a "typed-enough" layer: request/notification envelopes and the
//! handful of MCP result shapes we care about (`initialize`, `tools/list`,
//! `tools/call`) get real structs; everything else stays as [`Value`] so we
//! don't have to model the entire MCP schema to talk to a server.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicI64, Ordering};

use agent_framework_core::types::{Content, DataContent, Message, Role, UriContent};

/// The MCP protocol version this client requests during `initialize`.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// Protocol versions this client understands well enough to proceed if a
/// server negotiates down to one of them.
pub const COMPATIBLE_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// A monotonically increasing JSON-RPC request id generator, starting at 1.
#[derive(Debug, Default)]
pub struct IdGenerator(AtomicI64);

impl IdGenerator {
    pub fn new() -> Self {
        Self(AtomicI64::new(0))
    }

    /// Return the next id.
    pub fn next(&self) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst) + 1
    }
}

/// Build a JSON-RPC request envelope: `{"jsonrpc":"2.0","id":..,"method":..,"params":..}`.
pub fn build_request(id: i64, method: &str, params: Value) -> Value {
    let mut obj = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
    });
    if !params.is_null() {
        obj["params"] = params;
    }
    obj
}

/// Build a JSON-RPC notification envelope (no `id`): `{"jsonrpc":"2.0","method":..,"params":..}`.
pub fn build_notification(method: &str, params: Value) -> Value {
    let mut obj = json!({
        "jsonrpc": "2.0",
        "method": method,
    });
    if !params.is_null() {
        obj["params"] = params;
    }
    obj
}

/// A JSON-RPC error object (the `error` field of a response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MCP error {}: {}", self.code, self.message)
    }
}

/// A parsed incoming JSON-RPC message, classified by shape.
#[derive(Debug)]
pub enum IncomingMessage {
    /// A response (success or error) correlated to one of our outgoing requests.
    Response {
        id: i64,
        result: Result<Value, RpcError>,
    },
    /// A notification from the server: no `id`, no response expected.
    Notification { method: String, params: Value },
    /// A request FROM the server (e.g. sampling, roots). Not supported by this
    /// client; transports log and ignore these.
    ServerRequest {
        id: Value,
        method: String,
        params: Value,
    },
    /// Valid JSON that didn't match any recognized JSON-RPC shape.
    Malformed(Value),
}

/// Classify a raw JSON value as a JSON-RPC response, notification, or
/// server-initiated request.
pub fn parse_incoming(value: Value) -> IncomingMessage {
    let has_result_or_error = value.get("result").is_some() || value.get("error").is_some();
    if has_result_or_error {
        if let Some(id) = value.get("id").and_then(Value::as_i64) {
            if let Some(err) = value.get("error") {
                let rpc_err: RpcError = serde_json::from_value(err.clone()).unwrap_or(RpcError {
                    code: -32603,
                    message: "unrecognized error shape from MCP server".to_string(),
                    data: Some(err.clone()),
                });
                return IncomingMessage::Response {
                    id,
                    result: Err(rpc_err),
                };
            }
            let result = value.get("result").cloned().unwrap_or(Value::Null);
            return IncomingMessage::Response {
                id,
                result: Ok(result),
            };
        }
        return IncomingMessage::Malformed(value);
    }
    if let Some(method) = value.get("method").and_then(Value::as_str) {
        let params = value.get("params").cloned().unwrap_or(Value::Null);
        if let Some(id) = value.get("id").cloned() {
            return IncomingMessage::ServerRequest {
                id,
                method: method.to_string(),
                params,
            };
        }
        return IncomingMessage::Notification {
            method: method.to_string(),
            params,
        };
    }
    IncomingMessage::Malformed(value)
}

/// `clientInfo`/`serverInfo`: an MCP implementation's name and version.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Implementation {
    pub name: String,
    #[serde(default)]
    pub version: String,
}

/// The result of a successful `initialize` request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    #[serde(default)]
    pub capabilities: Value,
    pub server_info: Implementation,
    #[serde(default)]
    pub instructions: Option<String>,
}

impl InitializeResult {
    /// Whether the server declared the `prompts` capability, i.e.
    /// `capabilities.prompts` is present in the `initialize` response.
    ///
    /// Used to short-circuit [`crate::McpClient::list_prompts`] without a
    /// round trip — the Rust equivalent of the Python reference's
    /// try/except around `session.list_prompts()`, which logs and treats a
    /// failure the same way, but checks the negotiated capability up front
    /// instead of discarding an expected error.
    pub fn supports_prompts(&self) -> bool {
        self.capabilities.get("prompts").is_some()
    }
}

/// A tool descriptor as returned by `tools/list`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDescriptor {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default = "empty_object_schema")]
    pub input_schema: Value,
    #[serde(default)]
    pub output_schema: Option<Value>,
    /// The tool's `_meta`, echoed back on `tools/call` (some MCP proxies
    /// require it, as upstream notes).
    #[serde(default, rename = "_meta")]
    pub meta: Option<Value>,
    /// Behavioural hints (`readOnlyHint`, `destructiveHint`, ...), verbatim.
    #[serde(default)]
    pub annotations: Option<Value>,
    /// Execution properties, e.g. `{"taskSupport": "required"}`.
    #[serde(default)]
    pub execution: Option<Value>,
}

impl ToolDescriptor {
    /// The names declared in `inputSchema.properties`.
    pub fn declared_parameters(&self) -> std::collections::HashSet<String> {
        self.input_schema
            .get("properties")
            .and_then(Value::as_object)
            .map(|p| p.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// `execution.taskSupport` (`"required"`, `"optional"`, `"forbidden"`).
    pub fn task_support(&self) -> Option<&str> {
        self.execution
            .as_ref()
            .and_then(|e| e.get("taskSupport"))
            .and_then(Value::as_str)
    }
}

fn empty_object_schema() -> Value {
    json!({ "type": "object", "properties": {} })
}

/// One page of `tools/list` results.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListToolsResult {
    #[serde(default)]
    pub tools: Vec<ToolDescriptor>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

/// One content block inside a `tools/call` result.
///
/// MCP defines more block kinds than this (annotations, etc.); anything we
/// don't specifically model is preserved verbatim in [`ContentBlock::Unknown`]
/// so no information is silently dropped.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text(String),
    Image {
        data: String,
        mime_type: String,
    },
    Audio {
        data: String,
        mime_type: String,
    },
    Resource(Value),
    ResourceLink {
        uri: String,
        mime_type: Option<String>,
        name: Option<String>,
    },
    Unknown(Value),
}

impl ContentBlock {
    /// Parse a single content block from its raw JSON representation.
    pub fn from_value(v: &Value) -> ContentBlock {
        match v.get("type").and_then(Value::as_str) {
            Some("text") => ContentBlock::Text(
                v.get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ),
            Some("image") => ContentBlock::Image {
                data: v
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                mime_type: v
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            },
            Some("audio") => ContentBlock::Audio {
                data: v
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                mime_type: v
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            },
            Some("resource") => {
                ContentBlock::Resource(v.get("resource").cloned().unwrap_or(Value::Null))
            }
            Some("resource_link") => ContentBlock::ResourceLink {
                uri: v
                    .get("uri")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                mime_type: v
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                name: v.get("name").and_then(Value::as_str).map(str::to_string),
            },
            _ => ContentBlock::Unknown(v.clone()),
        }
    }

    /// Reconstruct the block's JSON representation (used when preserving
    /// multi-block results as a structured array).
    pub fn to_json(&self) -> Value {
        match self {
            ContentBlock::Text(text) => json!({ "type": "text", "text": text }),
            ContentBlock::Image { data, mime_type } => {
                json!({ "type": "image", "data": data, "mimeType": mime_type })
            }
            ContentBlock::Audio { data, mime_type } => {
                json!({ "type": "audio", "data": data, "mimeType": mime_type })
            }
            ContentBlock::Resource(resource) => json!({ "type": "resource", "resource": resource }),
            ContentBlock::ResourceLink {
                uri,
                mime_type,
                name,
            } => {
                let mut obj = json!({ "type": "resource_link", "uri": uri });
                if let Some(m) = mime_type {
                    obj["mimeType"] = json!(m);
                }
                if let Some(n) = name {
                    obj["name"] = json!(n);
                }
                obj
            }
            ContentBlock::Unknown(v) => v.clone(),
        }
    }

    /// Convert this block into a core [`Content`] item.
    ///
    /// Mirrors the relevant arms of the Python reference's
    /// `_mcp_type_to_ai_content`: text stays text; image/audio become
    /// [`Content::Data`] with the base64 payload wrapped into a proper
    /// `data:<mime>;base64,<payload>` URI (this crate's established
    /// convention for bare-base64 wire payloads — see
    /// `agent-framework-a2a`'s `FileWithBytes` handling); a resource link
    /// becomes [`Content::Uri`]; an embedded resource becomes text or data
    /// depending on whether it carries `text` or `blob`. Infallible: any
    /// shape this crate doesn't specifically recognize degrades to its raw
    /// JSON text rather than being dropped.
    pub fn to_core_content(&self) -> Content {
        match self {
            ContentBlock::Text(text) => Content::text(text.clone()),
            ContentBlock::Image { data, mime_type } | ContentBlock::Audio { data, mime_type } => {
                Content::Data(DataContent {
                    uri: format!("data:{mime_type};base64,{data}"),
                    media_type: Some(mime_type.clone()),
                })
            }
            ContentBlock::ResourceLink { uri, mime_type, .. } => Content::Uri(UriContent {
                uri: uri.clone(),
                media_type: mime_type
                    .clone()
                    .unwrap_or_else(|| "application/json".to_string()),
            }),
            ContentBlock::Resource(resource) => {
                if let Some(text) = resource.get("text").and_then(Value::as_str) {
                    Content::text(text.to_string())
                } else if let Some(blob) = resource.get("blob").and_then(Value::as_str) {
                    let mime = resource
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("application/octet-stream");
                    Content::Data(DataContent {
                        uri: format!("data:{mime};base64,{blob}"),
                        media_type: Some(mime.to_string()),
                    })
                } else {
                    Content::text(resource.to_string())
                }
            }
            ContentBlock::Unknown(v) => Content::text(v.to_string()),
        }
    }
}

/// Build a core [`Message`] from an MCP role string plus a single raw
/// content value. Shared by prompt-message (`prompts/get`) and
/// sampling-message (`sampling/createMessage`) mapping: both carry a role
/// and exactly one content block each, unlike `tools/call`'s content array.
pub(crate) fn role_and_content_to_chat_message(role: &str, content: &Value) -> Message {
    Message::with_contents(
        Role::new(role.to_string()),
        vec![ContentBlock::from_value(content).to_core_content()],
    )
}

/// An MCP logging level (RFC 5424 severities), for `logging/setLevel` and
/// `notifications/message`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum McpLogLevel {
    Debug,
    Info,
    Notice,
    Warning,
    Error,
    Critical,
    Alert,
    Emergency,
}

impl McpLogLevel {
    /// The wire name (`"debug"`, `"warning"`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Notice => "notice",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
            Self::Alert => "alert",
            Self::Emergency => "emergency",
        }
    }

    /// Parse a wire name.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "debug" => Self::Debug,
            "info" => Self::Info,
            "notice" => Self::Notice,
            "warning" => Self::Warning,
            "error" => Self::Error,
            "critical" => Self::Critical,
            "alert" => Self::Alert,
            "emergency" => Self::Emergency,
            _ => return None,
        })
    }
}

/// How the model-visible value of a tool result is chosen when the result
/// carries both `content` and `structuredContent`. Mirrors upstream's
/// `MCPToolResultContentMode`; servers disagree on whether the two are
/// duplicates or complementary, so the caller picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolResultContent {
    /// `structuredContent` when present, else `content` (upstream's default).
    #[default]
    StructuredFirst,
    /// `content` when non-empty, else `structuredContent`.
    ContentFirst,
    /// Ignore `structuredContent`.
    ContentOnly,
    /// Ignore `content`.
    StructuredOnly,
    /// The `content` blocks followed by serialized `structuredContent` as a
    /// text block.
    Both,
}

/// The result of a `tools/call` request.
#[derive(Debug, Clone, Default)]
pub struct CallToolResult {
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
    pub structured_content: Option<Value>,
    /// The result's `_meta` (information-flow labels and the like).
    pub meta: Option<Value>,
}

impl CallToolResult {
    /// Parse a `tools/call` result from its raw JSON representation.
    pub fn from_value(v: &Value) -> Self {
        let content = v
            .get("content")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().map(ContentBlock::from_value).collect())
            .unwrap_or_default();
        let is_error = v.get("isError").and_then(Value::as_bool).unwrap_or(false);
        let structured_content = v.get("structuredContent").cloned();
        Self {
            content,
            is_error,
            structured_content,
            meta: v.get("_meta").cloned(),
        }
    }

    /// Map the result's content into a single JSON value for handing back to
    /// a model, independent of whether it was an error:
    ///
    /// - A single text block becomes a JSON string, or the value it parses as
    ///   if the text itself is valid JSON.
    /// - No content but a `structuredContent` payload returns that payload.
    /// - Anything else (zero or multiple / non-text blocks) becomes a JSON
    ///   array preserving each block's shape.
    ///
    /// This is [`ToolResultContent::ContentFirst`]; see
    /// [`Self::to_value_with`] for the other policies.
    pub fn to_value(&self) -> Value {
        self.to_value_with(ToolResultContent::ContentFirst)
    }

    /// The model-visible value under an explicit `mode` for a result that
    /// may carry both `content` and `structuredContent`. Mirrors upstream's
    /// `tool_result_content` (#7866). An empty selection is `null`.
    pub fn to_value_with(&self, mode: ToolResultContent) -> Value {
        let content = self.content_value();
        let structured = self.structured_content.clone();
        let chosen = match mode {
            ToolResultContent::StructuredFirst => structured.or(content),
            ToolResultContent::ContentFirst => content.or(structured),
            ToolResultContent::ContentOnly => content,
            ToolResultContent::StructuredOnly => structured,
            ToolResultContent::Both => match structured {
                None => content,
                Some(structured) => {
                    let mut blocks: Vec<Value> =
                        self.content.iter().map(ContentBlock::to_json).collect();
                    blocks.push(json!({ "type": "text", "text": structured.to_string() }));
                    Some(Value::Array(blocks))
                }
            },
        };
        chosen.unwrap_or(Value::Null)
    }

    /// The `content` blocks as one value, or `None` when there are none: a
    /// lone text block is its text (parsed if it is JSON), anything else an
    /// array of blocks.
    fn content_value(&self) -> Option<Value> {
        match self.content.as_slice() {
            [] => None,
            [ContentBlock::Text(text)] => Some(
                serde_json::from_str::<Value>(text).unwrap_or_else(|_| Value::String(text.clone())),
            ),
            blocks => Some(Value::Array(
                blocks.iter().map(ContentBlock::to_json).collect(),
            )),
        }
    }

    /// A human-readable message extracted from the text blocks, used when
    /// `is_error` is true and we need a message for [`agent_framework_core::error::Error::Tool`].
    pub fn error_message(&self) -> String {
        let joined = self
            .content
            .iter()
            .filter_map(|c| match c {
                ContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if joined.is_empty() {
            "MCP tool call reported an error".to_string()
        } else {
            joined
        }
    }
}

/// Normalize an MCP tool/prompt name to the identifier pattern most model
/// providers require (`A-Za-z0-9_.-`), replacing any other character with `-`.
///
/// Mirrors `_normalize_mcp_name` in the Python reference implementation.
pub fn normalize_mcp_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

// ---------------------------------------------------------------------
// Prompts (`prompts/list` / `prompts/get`)
// ---------------------------------------------------------------------

/// One argument a [`PromptDescriptor`] accepts.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptArgument {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub required: Option<bool>,
}

/// A prompt descriptor as returned by `prompts/list`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptDescriptor {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub arguments: Option<Vec<PromptArgument>>,
}

/// One page of `prompts/list` results.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListPromptsResult {
    #[serde(default)]
    pub prompts: Vec<PromptDescriptor>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

/// One message inside a `prompts/get` result.
///
/// Unlike a `tools/call` result, an MCP prompt message carries exactly one
/// content block, not an array — `content` is kept as a raw [`Value`] here
/// (parse with [`Self::content_block`]) rather than modeled as
/// [`ContentBlock`] directly, so this type's `Deserialize` impl can stay
/// derived.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptMessage {
    /// `"user"` or `"assistant"`.
    pub role: String,
    pub content: Value,
}

impl PromptMessage {
    /// Parse this message's raw content into a [`ContentBlock`].
    pub fn content_block(&self) -> ContentBlock {
        ContentBlock::from_value(&self.content)
    }
}

/// The result of a `prompts/get` request.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GetPromptResult {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub messages: Vec<PromptMessage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_generator_starts_at_one_and_increments() {
        let gen = IdGenerator::new();
        assert_eq!(gen.next(), 1);
        assert_eq!(gen.next(), 2);
        assert_eq!(gen.next(), 3);
    }

    #[test]
    fn build_request_includes_params_when_present() {
        let req = build_request(7, "tools/call", json!({"name": "echo"}));
        assert_eq!(req["jsonrpc"], "2.0");
        assert_eq!(req["id"], 7);
        assert_eq!(req["method"], "tools/call");
        assert_eq!(req["params"]["name"], "echo");
    }

    #[test]
    fn build_request_omits_null_params() {
        let req = build_request(1, "ping", Value::Null);
        assert!(req.get("params").is_none());
    }

    #[test]
    fn build_notification_has_no_id() {
        let note = build_notification("notifications/initialized", json!({}));
        assert!(note.get("id").is_none());
        assert_eq!(note["method"], "notifications/initialized");
    }

    #[test]
    fn parse_incoming_classifies_response() {
        let msg = json!({"jsonrpc":"2.0","id":3,"result":{"ok":true}});
        match parse_incoming(msg) {
            IncomingMessage::Response { id, result } => {
                assert_eq!(id, 3);
                assert_eq!(result.unwrap(), json!({"ok": true}));
            }
            other => panic!("expected Response, got {other:?}"),
        }
    }

    #[test]
    fn parse_incoming_classifies_error_response() {
        let msg = json!({"jsonrpc":"2.0","id":3,"error":{"code":-32601,"message":"not found"}});
        match parse_incoming(msg) {
            IncomingMessage::Response { id, result } => {
                assert_eq!(id, 3);
                let err = result.unwrap_err();
                assert_eq!(err.code, -32601);
                assert_eq!(err.message, "not found");
            }
            other => panic!("expected Response, got {other:?}"),
        }
    }

    #[test]
    fn parse_incoming_classifies_notification() {
        let msg =
            json!({"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info"}});
        match parse_incoming(msg) {
            IncomingMessage::Notification { method, params } => {
                assert_eq!(method, "notifications/message");
                assert_eq!(params["level"], "info");
            }
            other => panic!("expected Notification, got {other:?}"),
        }
    }

    #[test]
    fn parse_incoming_classifies_server_request() {
        let msg =
            json!({"jsonrpc":"2.0","id":"srv-1","method":"sampling/createMessage","params":{}});
        match parse_incoming(msg) {
            IncomingMessage::ServerRequest { method, .. } => {
                assert_eq!(method, "sampling/createMessage");
            }
            other => panic!("expected ServerRequest, got {other:?}"),
        }
    }

    #[test]
    fn call_tool_result_single_text_block_becomes_string() {
        let result = CallToolResult::from_value(&json!({
            "content": [{"type": "text", "text": "hello world"}],
            "isError": false,
        }));
        assert_eq!(result.to_value(), json!("hello world"));
    }

    #[test]
    fn call_tool_result_single_text_block_parses_json() {
        let result = CallToolResult::from_value(&json!({
            "content": [{"type": "text", "text": "42"}],
            "isError": false,
        }));
        assert_eq!(result.to_value(), json!(42));
    }

    #[test]
    fn call_tool_result_multi_block_preserves_structure() {
        let result = CallToolResult::from_value(&json!({
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image", "data": "aGk=", "mimeType": "image/png"},
            ],
            "isError": false,
        }));
        assert_eq!(
            result.to_value(),
            json!([
                {"type": "text", "text": "first"},
                {"type": "image", "data": "aGk=", "mimeType": "image/png"},
            ])
        );
    }

    #[test]
    fn call_tool_result_is_error_flag_parsed() {
        let result = CallToolResult::from_value(&json!({
            "content": [{"type": "text", "text": "boom"}],
            "isError": true,
        }));
        assert!(result.is_error);
        assert_eq!(result.error_message(), "boom");
    }

    #[test]
    fn call_tool_result_empty_content_falls_back_to_structured_content() {
        let result = CallToolResult::from_value(&json!({
            "content": [],
            "structuredContent": {"count": 3},
        }));
        assert_eq!(result.to_value(), json!({"count": 3}));
    }

    #[test]
    fn normalize_mcp_name_replaces_disallowed_chars() {
        assert_eq!(
            normalize_mcp_name("weather/get current"),
            "weather-get-current"
        );
        assert_eq!(normalize_mcp_name("valid_Name-1.0"), "valid_Name-1.0");
    }

    #[test]
    fn initialize_result_supports_prompts_checks_capabilities() {
        let with_prompts: InitializeResult = serde_json::from_value(json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {"prompts": {}},
            "serverInfo": {"name": "s", "version": "1"},
        }))
        .unwrap();
        assert!(with_prompts.supports_prompts());

        let without_prompts: InitializeResult = serde_json::from_value(json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "s", "version": "1"},
        }))
        .unwrap();
        assert!(!without_prompts.supports_prompts());
    }

    #[test]
    fn content_block_to_core_content_maps_text() {
        let block = ContentBlock::Text("hello".to_string());
        assert_eq!(block.to_core_content(), Content::text("hello"));
    }

    #[test]
    fn content_block_to_core_content_wraps_image_bytes_as_data_uri() {
        let block = ContentBlock::Image {
            data: "aGVsbG8=".to_string(),
            mime_type: "image/png".to_string(),
        };
        match block.to_core_content() {
            Content::Data(d) => {
                assert_eq!(d.uri, "data:image/png;base64,aGVsbG8=");
                assert_eq!(d.media_type.as_deref(), Some("image/png"));
            }
            other => panic!("expected Content::Data, got {other:?}"),
        }
    }

    #[test]
    fn content_block_to_core_content_wraps_audio_bytes_as_data_uri() {
        let block = ContentBlock::Audio {
            data: "d2F2ZQ==".to_string(),
            mime_type: "audio/wav".to_string(),
        };
        match block.to_core_content() {
            Content::Data(d) => assert_eq!(d.uri, "data:audio/wav;base64,d2F2ZQ=="),
            other => panic!("expected Content::Data, got {other:?}"),
        }
    }

    #[test]
    fn content_block_to_core_content_maps_resource_link() {
        let block = ContentBlock::ResourceLink {
            uri: "file:///tmp/report.pdf".to_string(),
            mime_type: Some("application/pdf".to_string()),
            name: Some("report.pdf".to_string()),
        };
        match block.to_core_content() {
            Content::Uri(u) => {
                assert_eq!(u.uri, "file:///tmp/report.pdf");
                assert_eq!(u.media_type, "application/pdf");
            }
            other => panic!("expected Content::Uri, got {other:?}"),
        }
    }

    #[test]
    fn content_block_to_core_content_maps_text_resource() {
        let block = ContentBlock::Resource(json!({"uri": "file:///a.txt", "text": "hi"}));
        assert_eq!(block.to_core_content(), Content::text("hi"));
    }

    #[test]
    fn content_block_to_core_content_maps_blob_resource() {
        let block = ContentBlock::Resource(
            json!({"uri": "file:///a.bin", "blob": "AAAA", "mimeType": "application/octet-stream"}),
        );
        match block.to_core_content() {
            Content::Data(d) => assert_eq!(d.uri, "data:application/octet-stream;base64,AAAA"),
            other => panic!("expected Content::Data, got {other:?}"),
        }
    }

    #[test]
    fn content_block_to_core_content_unknown_falls_back_to_raw_json_text() {
        let block = ContentBlock::Unknown(json!({"type": "future_kind", "x": 1}));
        assert_eq!(
            block.to_core_content(),
            Content::text(json!({"type": "future_kind", "x": 1}).to_string())
        );
    }

    #[test]
    fn role_and_content_to_chat_message_builds_single_content_message() {
        let msg =
            role_and_content_to_chat_message("assistant", &json!({"type": "text", "text": "hi"}));
        assert_eq!(msg.role, Role::assistant());
        assert_eq!(msg.contents, vec![Content::text("hi")]);
    }

    #[test]
    fn list_prompts_result_parses_page_with_cursor() {
        let page: ListPromptsResult = serde_json::from_value(json!({
            "prompts": [{"name": "greet", "description": "Say hello", "arguments": [
                {"name": "name", "required": true},
            ]}],
            "nextCursor": "page2",
        }))
        .unwrap();
        assert_eq!(page.prompts.len(), 1);
        assert_eq!(page.prompts[0].name, "greet");
        assert_eq!(page.prompts[0].arguments.as_ref().unwrap()[0].name, "name");
        assert_eq!(page.next_cursor.as_deref(), Some("page2"));
    }

    #[test]
    fn get_prompt_result_parses_messages() {
        let result: GetPromptResult = serde_json::from_value(json!({
            "description": "A greeting prompt",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": "Say hi to Ada"}},
            ],
        }))
        .unwrap();
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(
            result.messages[0].content_block(),
            ContentBlock::Text("Say hi to Ada".to_string())
        );
    }
}
