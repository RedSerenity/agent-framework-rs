//! WebSocket transport: connects to an MCP server over a WebSocket using the
//! "mcp" subprotocol and frames each JSON-RPC message as one text frame.
//!
//! Structurally this mirrors [`super::stdio::McpStdioTransport`]: a background
//! task reads frames off the socket and routes responses back to their
//! waiting caller by request id, notifications are logged, and
//! server-initiated requests are answered via whatever handler
//! [`McpTransport::set_server_request_handler`] installed.

use std::collections::HashMap;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::Uri;
use tokio_tungstenite::tungstenite::{ClientRequestBuilder, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use agent_framework_core::error::{Error, Result};

use crate::protocol::{self, IdGenerator, IncomingMessage, RpcError};
use crate::sampling::{BoxedNotificationHandler, BoxedServerRequestHandler};
use crate::transport::McpTransport;

/// The WebSocket subprotocol MCP servers expect during the handshake.
const MCP_SUBPROTOCOL: &str = "mcp";

type WsConn = WebSocketStream<MaybeTlsStream<TcpStream>>;
type WsSink = SplitSink<WsConn, Message>;
type WsSource = SplitStream<WsConn>;
type PendingMap = StdMutex<HashMap<i64, oneshot::Sender<std::result::Result<Value, RpcError>>>>;

/// An MCP transport backed by a WebSocket connection.
///
/// Connects with the `"mcp"` subprotocol advertised via `Sec-WebSocket-Protocol`
/// and, optionally, custom headers on the upgrade request (e.g. `Authorization`).
///
/// Note: `wss://` (TLS) connections require a process-wide `rustls`
/// `CryptoProvider` to be installed (e.g.
/// `rustls::crypto::ring::default_provider().install_default()`, called once
/// at process startup) — this crate does not install one itself so as not to
/// override a choice made elsewhere in the process. Plain `ws://` connections
/// (used by this crate's own tests) are unaffected.
pub struct McpWebsocketTransport {
    inner: std::sync::Arc<WsInner>,
    /// Applied to every [`McpTransport::call`] while awaiting its response.
    /// Unset (the default) waits indefinitely. See [`Self::with_request_timeout`].
    request_timeout: Option<Duration>,
}

struct WsInner {
    sink: AsyncMutex<WsSink>,
    pending: PendingMap,
    next_id: IdGenerator,
    reader_task: StdMutex<Option<JoinHandle<()>>>,
    /// Handler for server-initiated requests (`ping`, `sampling/createMessage`,
    /// `roots/list`), installed via [`McpTransport::set_server_request_handler`].
    server_request_handler: StdMutex<Option<BoxedServerRequestHandler>>,
    /// Handler for server notifications (e.g. `notifications/tools/list_changed`),
    /// installed via [`McpTransport::set_notification_handler`].
    notification_handler: StdMutex<Option<BoxedNotificationHandler>>,
    /// Set once the socket has closed (see [`McpTransport::is_closed`]).
    closed: std::sync::atomic::AtomicBool,
}

impl McpWebsocketTransport {
    /// Connect to the MCP server at `url` (`ws://` or `wss://`), negotiating
    /// the `"mcp"` subprotocol and sending `headers` (e.g. `Authorization`) on
    /// the upgrade request.
    pub async fn connect(url: &str, headers: &[(String, String)]) -> Result<Self> {
        let uri: Uri = url
            .parse()
            .map_err(|e| Error::Configuration(format!("invalid MCP websocket URL '{url}': {e}")))?;

        let mut builder = ClientRequestBuilder::new(uri).with_sub_protocol(MCP_SUBPROTOCOL);
        for (k, v) in headers {
            builder = builder.with_header(k.clone(), v.clone());
        }
        let request = builder
            .into_client_request()
            .map_err(|e| Error::Configuration(format!("invalid MCP websocket request: {e}")))?;

        let (stream, _response) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| {
                Error::service(format!("failed to connect to MCP websocket server: {e}"))
            })?;

        let (sink, source) = stream.split();
        let inner = std::sync::Arc::new(WsInner {
            sink: AsyncMutex::new(sink),
            pending: StdMutex::new(HashMap::new()),
            next_id: IdGenerator::new(),
            reader_task: StdMutex::new(None),
            server_request_handler: StdMutex::new(None),
            notification_handler: StdMutex::new(None),
            closed: std::sync::atomic::AtomicBool::new(false),
        });

        let reader_task = spawn_reader(source, inner.clone());
        *inner.reader_task.lock().unwrap() = Some(reader_task);

        Ok(Self {
            inner,
            request_timeout: None,
        })
    }

    /// Set a per-request timeout applied while awaiting a response to any
    /// JSON-RPC request sent over this transport. Mirrors
    /// [`crate::McpStreamableHttpTransport`]'s `timeout` option; unset (the
    /// default) waits indefinitely. A request that times out is removed from
    /// the pending-response table, so a late reply from the server is
    /// discarded rather than mis-delivered to a later call reusing the id
    /// space.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }
}

/// Encode `message` as one JSON text frame and send it. A free function
/// (rather than an `McpWebsocketTransport` method) so the background reader
/// task — which only holds `Arc<WsInner>`, not the outer transport handle —
/// can use it too, to write responses to server-initiated requests.
async fn write_text(inner: &WsInner, message: &Value) -> Result<()> {
    let text = serde_json::to_string(message)
        .map_err(|e| Error::service(format!("failed to encode MCP message: {e}")))?;
    let mut sink = inner.sink.lock().await;
    sink.send(Message::text(text))
        .await
        .map_err(|e| Error::service(format!("failed to write to MCP websocket: {e}")))
}

#[async_trait]
impl McpTransport for McpWebsocketTransport {
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.inner.next_id.next();
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().unwrap().insert(id, tx);

        let request = protocol::build_request(id, method, params);
        if let Err(e) = write_text(&self.inner, &request).await {
            self.inner.pending.lock().unwrap().remove(&id);
            return Err(e);
        }

        let await_response = async {
            match rx.await {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(rpc_err)) => Err(Error::service(rpc_err.to_string())),
                Err(_) => Err(Error::service(
                    "MCP server closed the connection before responding",
                )),
            }
        };
        match self.request_timeout {
            None => await_response.await,
            Some(timeout) => match tokio::time::timeout(timeout, await_response).await {
                Ok(result) => result,
                Err(_) => {
                    self.inner.pending.lock().unwrap().remove(&id);
                    Err(Error::service(format!(
                        "MCP request '{method}' timed out after {timeout:?}"
                    )))
                }
            },
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let notification = protocol::build_notification(method, params);
        write_text(&self.inner, &notification).await
    }

    async fn close(&self) -> Result<()> {
        let mut sink = self.inner.sink.lock().await;
        // Best effort: a server that already dropped the connection makes
        // this fail, which is not itself an error worth surfacing.
        let _ = sink.send(Message::Close(None)).await;
        Ok(())
    }

    fn set_server_request_handler(&self, handler: BoxedServerRequestHandler) {
        *self.inner.server_request_handler.lock().unwrap() = Some(handler);
    }

    fn set_notification_handler(&self, handler: BoxedNotificationHandler) {
        *self.inner.notification_handler.lock().unwrap() = Some(handler);
    }

    fn is_closed(&self) -> bool {
        self.inner.closed.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl Drop for WsInner {
    fn drop(&mut self) {
        if let Ok(mut task) = self.reader_task.lock() {
            if let Some(task) = task.take() {
                task.abort();
            }
        }
        // Fail any requests still waiting on a response so callers don't hang.
        if let Ok(mut pending) = self.pending.lock() {
            for (_, tx) in pending.drain() {
                let _ = tx.send(Err(RpcError {
                    code: -1,
                    message: "MCP websocket transport dropped".to_string(),
                    data: None,
                }));
            }
        }
    }
}

/// Spawn the background task that reads text frames from the websocket and
/// routes them: responses go to their waiting caller by id, notifications and
/// server-initiated requests are logged and otherwise ignored.
fn spawn_reader(mut source: WsSource, inner: std::sync::Arc<WsInner>) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match source.next().await {
                Some(Ok(Message::Text(text))) => {
                    let trimmed = text.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(value) => route_incoming(&inner, value),
                        Err(e) => {
                            tracing::warn!(error = %e, text = %trimmed, "MCP: non-JSON text frame from websocket server");
                        }
                    }
                }
                Some(Ok(Message::Close(_))) => {
                    tracing::debug!("MCP websocket server closed the connection");
                    break;
                }
                // Pings/pongs are answered by tokio-tungstenite itself; binary
                // and raw frames are not part of MCP's websocket framing.
                Some(Ok(
                    Message::Ping(_) | Message::Pong(_) | Message::Binary(_) | Message::Frame(_),
                )) => {}
                Some(Err(e)) => {
                    tracing::warn!(error = %e, "MCP: error reading from websocket");
                    break;
                }
                None => {
                    tracing::debug!("MCP websocket stream ended");
                    break;
                }
            }
        }
        // The server is gone; unblock anyone still waiting on a response.
        // Marked closed first, so a caller woken below already sees it.
        inner
            .closed
            .store(true, std::sync::atomic::Ordering::Release);
        let mut pending = inner.pending.lock().unwrap();
        for (_, tx) in pending.drain() {
            let _ = tx.send(Err(RpcError {
                code: -1,
                message: "MCP websocket connection closed".to_string(),
                data: None,
            }));
        }
    })
}

/// Identical routing logic to the stdio transport's `route_incoming`:
/// classify the message and either resolve a pending response, log a
/// notification, or answer a server-initiated request.
fn route_incoming(inner: &std::sync::Arc<WsInner>, value: Value) {
    match protocol::parse_incoming(value) {
        IncomingMessage::Response { id, result } => {
            if let Some(tx) = inner.pending.lock().unwrap().remove(&id) {
                let _ = tx.send(result);
            } else {
                tracing::debug!(id, "MCP: response for unknown/already-resolved request id");
            }
        }
        IncomingMessage::Notification { method, params } => {
            tracing::debug!(method = %method, params = %params, "MCP server notification");
            dispatch_notification(inner.clone(), method, params);
        }
        IncomingMessage::ServerRequest { id, method, params } => {
            spawn_server_request_response(inner.clone(), id, method, params);
        }
        IncomingMessage::Malformed(v) => {
            tracing::warn!(raw = %v, "MCP: unrecognized JSON-RPC message shape");
        }
    }
}

/// Dispatch a server notification (e.g. `notifications/tools/list_changed`)
/// to whatever handler [`McpTransport::set_notification_handler`] installed,
/// in its own task so a slow handler doesn't block the reader loop from
/// noticing other incoming messages in the meantime. No response is expected
/// or sent; a notification with no handler registered is simply dropped
/// (already logged by the caller).
fn dispatch_notification(inner: std::sync::Arc<WsInner>, method: String, params: Value) {
    tokio::spawn(async move {
        let handler = inner.notification_handler.lock().unwrap().clone();
        if let Some(handler) = handler {
            handler(method, params).await;
        }
    });
}

/// Compute the response to a server-initiated request (via whatever handler
/// is registered — see [`McpTransport::set_server_request_handler`]) and
/// send it back over the socket, in its own task so a slow handler (e.g. a
/// sampling handler calling out to an LLM) doesn't block the reader loop
/// from noticing other incoming messages in the meantime.
fn spawn_server_request_response(
    inner: std::sync::Arc<WsInner>,
    id: Value,
    method: String,
    params: Value,
) {
    tokio::spawn(async move {
        let handler = inner.server_request_handler.lock().unwrap().clone();
        let result = match handler {
            Some(h) => h(method.clone(), params).await,
            None => Err(RpcError {
                code: -32601,
                message: format!("Method not found: {method}"),
                data: None,
            }),
        };
        let envelope = match result {
            Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
            Err(err) => json!({ "jsonrpc": "2.0", "id": id, "error": err }),
        };
        if let Err(e) = write_text(&inner, &envelope).await {
            tracing::warn!(
                error = %e,
                "MCP: failed to write response for a server-initiated request"
            );
        }
    });
}
