//! A minimal MCP server for [`McpToolProvider`]s: the JSON-RPC dispatcher
//! plus stdio and streamable-HTTP transports.
//!
//! Upstream leaves the server to the MCP Python SDK and ships only the tool
//! adapters. There is no MCP server SDK in this workspace, so this module
//! provides the small part of one the adapters need: the `initialize`
//! handshake, `ping`, `tools/list` and `tools/call`. Prompts, resources,
//! sampling and server-initiated messages are not served — a client that
//! asks gets JSON-RPC "method not found".

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use agent_framework_mcp::{COMPATIBLE_PROTOCOL_VERSIONS, PROTOCOL_VERSION};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use serde_json::{json, Map, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

use crate::tools::McpToolProvider;
use crate::McpHostError;

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

/// The `Mcp-Session-Id` header of the streamable-HTTP transport.
pub const SESSION_HEADER: &str = "mcp-session-id";

/// An MCP server over a set of [`McpToolProvider`]s.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use agent_framework_core::agent::SupportsAgentRun;
/// # use agent_framework_hosting_mcp::{AgentMcpTool, McpServer};
/// # async fn demo(agent: Arc<dyn SupportsAgentRun>) -> std::io::Result<()> {
/// let server = McpServer::new("weather-server", "1.0.0")
///     .tool(AgentMcpTool::new(agent).description("Answers weather questions"));
/// server.serve_stdio().await
/// # }
/// ```
#[derive(Clone)]
pub struct McpServer {
    inner: Arc<Inner>,
}

struct Inner {
    name: String,
    version: String,
    instructions: Option<String>,
    providers: Vec<Arc<dyn McpToolProvider>>,
    /// Streamable-HTTP sessions issued by `initialize` and not yet deleted.
    sessions: Mutex<HashSet<String>>,
}

impl McpServer {
    /// A server announcing itself as `name` / `version` in `serverInfo`.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                name: name.into(),
                version: version.into(),
                instructions: None,
                providers: Vec::new(),
                sessions: Mutex::new(HashSet::new()),
            }),
        }
    }

    fn inner_mut(&mut self) -> &mut Inner {
        Arc::get_mut(&mut self.inner).expect("configure an McpServer before cloning or serving it")
    }

    /// Instructions returned from `initialize`.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.inner_mut().instructions = Some(instructions.into());
        self
    }

    /// Serve `provider`'s tools.
    pub fn tool(self, provider: impl McpToolProvider + 'static) -> Self {
        self.tool_arc(Arc::new(provider))
    }

    /// Serve a shared provider's tools.
    pub fn tool_arc(mut self, provider: Arc<dyn McpToolProvider>) -> Self {
        self.inner_mut().providers.push(provider);
        self
    }

    /// Handle one JSON-RPC message, returning the response to send (none
    /// for a notification).
    pub async fn handle_message(&self, message: Value) -> Option<Value> {
        let Some(obj) = message.as_object() else {
            return Some(error_response(
                Value::Null,
                INVALID_REQUEST,
                "Invalid Request",
            ));
        };
        let id = obj.get("id").cloned();
        let Some(method) = obj.get("method").and_then(Value::as_str) else {
            // A response to a request we never sent; nothing to answer.
            if obj.contains_key("result") || obj.contains_key("error") {
                return None;
            }
            return Some(error_response(
                id.unwrap_or(Value::Null),
                INVALID_REQUEST,
                "Invalid Request",
            ));
        };
        let params = obj.get("params").cloned().unwrap_or(Value::Null);
        let Some(id) = id else {
            // Notifications (`notifications/initialized`, cancellations, …)
            // need no answer.
            return None;
        };
        Some(match self.dispatch(method, &params).await {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err((code, message)) => error_response(id, code, &message),
        })
    }

    async fn dispatch(&self, method: &str, params: &Value) -> Result<Value, (i64, String)> {
        match method {
            "initialize" => Ok(self.initialize(params)),
            "ping" => Ok(json!({})),
            "tools/list" => self.list_tools().await,
            "tools/call" => self.call_tool(params).await,
            other => Err((METHOD_NOT_FOUND, format!("Method not found: {other}"))),
        }
    }

    fn initialize(&self, params: &Value) -> Value {
        // Answer in the client's version when we speak it, else in ours.
        let requested = params.get("protocolVersion").and_then(Value::as_str);
        let version = requested
            .filter(|v| COMPATIBLE_PROTOCOL_VERSIONS.contains(v))
            .unwrap_or(PROTOCOL_VERSION);
        let mut result = json!({
            "protocolVersion": version,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": self.inner.name, "version": self.inner.version },
        });
        if let Some(instructions) = &self.inner.instructions {
            result["instructions"] = json!(instructions);
        }
        result
    }

    async fn list_tools(&self) -> Result<Value, (i64, String)> {
        let mut tools = Vec::new();
        for provider in &self.inner.providers {
            tools.extend(
                provider
                    .list_tools()
                    .await
                    .map_err(|e| (INTERNAL_ERROR, e.to_string()))?,
            );
        }
        Ok(json!({ "tools": tools }))
    }

    async fn call_tool(&self, params: &Value) -> Result<Value, (i64, String)> {
        let name = params.get("name").and_then(Value::as_str).ok_or((
            INVALID_PARAMS,
            "tools/call requires a string `name`".to_string(),
        ))?;
        let arguments: Option<Map<String, Value>> = match params.get("arguments") {
            None | Some(Value::Null) => None,
            Some(Value::Object(m)) => Some(m.clone()),
            Some(_) => return Err((INVALID_PARAMS, "`arguments` must be an object".into())),
        };
        for provider in &self.inner.providers {
            match provider.call_tool(name, arguments.as_ref()).await {
                Err(McpHostError::UnknownTool(_)) => continue,
                Ok(content) => return Ok(json!({ "content": content, "isError": false })),
                // A tool that ran and failed is a *result* the model can see
                // and react to, per the MCP spec — not a protocol error.
                Err(e) => {
                    return Ok(json!({
                        "content": [{ "type": "text", "text": e.to_string() }],
                        "isError": true,
                    }))
                }
            }
        }
        Err((INVALID_PARAMS, format!("Unknown tool: {name}")))
    }

    /// Serve newline-delimited JSON-RPC on this process's stdin/stdout until
    /// stdin closes — the transport an MCP client uses when it launches this
    /// program as a subprocess.
    pub async fn serve_stdio(&self) -> std::io::Result<()> {
        self.serve_io(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// Serve newline-delimited JSON-RPC over any reader/writer pair.
    /// Requests are handled one at a time, in order.
    pub async fn serve_io<R, W>(&self, reader: R, mut writer: W) -> std::io::Result<()>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut lines = BufReader::new(reader).lines();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let reply = match serde_json::from_str::<Value>(&line) {
                Ok(message) => self.handle_message(message).await,
                Err(e) => Some(error_response(
                    Value::Null,
                    PARSE_ERROR,
                    &format!("Parse error: {e}"),
                )),
            };
            if let Some(reply) = reply {
                let mut bytes = serde_json::to_vec(&reply)?;
                bytes.push(b'\n');
                writer.write_all(&bytes).await?;
                writer.flush().await?;
            }
        }
        Ok(())
    }

    /// An `axum` router speaking MCP's streamable-HTTP transport at `path`.
    ///
    /// `POST` carries one JSON-RPC message: a request is answered with an
    /// `application/json` body (the transport allows JSON instead of an SSE
    /// stream), a notification or response with `202 Accepted`. `initialize`
    /// issues an `Mcp-Session-Id`; later requests must carry a live one, and
    /// `DELETE` ends it. `GET` is `405`: this server sends no
    /// server-initiated messages. Wrap the router in
    /// [`HostingSecurity`](agent_framework_hosting::HostingSecurity) for
    /// authentication and the `Host`/`Origin` checks the transport's
    /// security guidance calls for.
    pub fn into_router(self, path: &str) -> Router {
        Router::new()
            .route(
                path,
                post(http_post)
                    .delete(http_delete)
                    .get(|| async { StatusCode::METHOD_NOT_ALLOWED }),
            )
            .with_state(self)
    }

    fn session_live(&self, id: &str) -> bool {
        self.inner
            .sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(id)
    }
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn json_response(status: StatusCode, body: &Value, session: Option<&str>) -> Response {
    let mut response = (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response();
    if let Some(id) = session.and_then(|s| HeaderValue::from_str(s).ok()) {
        response.headers_mut().insert(SESSION_HEADER, id);
    }
    response
}

async fn http_post(State(server): State<McpServer>, headers: HeaderMap, body: Bytes) -> Response {
    let message: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                &error_response(Value::Null, PARSE_ERROR, &format!("Parse error: {e}")),
                None,
            )
        }
    };
    let is_initialize = message.get("method").and_then(Value::as_str) == Some("initialize");
    let session = headers.get(SESSION_HEADER).and_then(|v| v.to_str().ok());

    if is_initialize {
        let reply = server.handle_message(message).await;
        let id = uuid::Uuid::new_v4().simple().to_string();
        server
            .inner
            .sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.clone());
        return match reply {
            Some(reply) => json_response(StatusCode::OK, &reply, Some(&id)),
            None => StatusCode::ACCEPTED.into_response(),
        };
    }

    match session {
        // The transport: a request without a session id after initialization
        // is a 400; one naming an unknown or ended session is a 404, which
        // tells the client to start a new session.
        None => {
            return json_response(
                StatusCode::BAD_REQUEST,
                &error_response(
                    Value::Null,
                    INVALID_REQUEST,
                    "Missing Mcp-Session-Id header",
                ),
                None,
            )
        }
        Some(id) if !server.session_live(id) => {
            return json_response(
                StatusCode::NOT_FOUND,
                &error_response(Value::Null, INVALID_REQUEST, "Session not found"),
                None,
            )
        }
        Some(_) => {}
    }
    match server.handle_message(message).await {
        Some(reply) => json_response(StatusCode::OK, &reply, session),
        None => StatusCode::ACCEPTED.into_response(),
    }
}

async fn http_delete(State(server): State<McpServer>, headers: HeaderMap) -> StatusCode {
    let Some(id) = headers.get(SESSION_HEADER).and_then(|v| v.to_str().ok()) else {
        return StatusCode::BAD_REQUEST;
    };
    let removed = server
        .inner
        .sessions
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(id);
    if removed {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}
