//! Pluggable JSON-RPC transports for MCP.
//!
//! [`McpTransport`] hides how a JSON-RPC message actually reaches the server
//! (a child process's stdio, an HTTP POST, …) behind two operations: a
//! correlated request/response `call`, and a fire-and-forget `notify`.
//! [`crate::McpClient`] is written entirely against this trait.

pub mod http;
pub mod stdio;
pub mod websocket;

use async_trait::async_trait;
use serde_json::Value;

use agent_framework_core::error::Result;

use crate::sampling::{BoxedNotificationHandler, BoxedServerRequestHandler};

/// A JSON-RPC 2.0 transport for MCP messages.
///
/// Implementations own request-id bookkeeping and response correlation, and
/// any background I/O required to notice server-initiated messages: a
/// server-initiated request is routed to whatever handler
/// [`Self::set_server_request_handler`] installed (see
/// [`crate::McpClient`], which installs one automatically), computing and
/// writing back a JSON-RPC response; anything else (notifications) is
/// logged and otherwise ignored.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a JSON-RPC request and return its `result` value.
    ///
    /// Resolves to `Err` if the server returns a JSON-RPC error response, or
    /// if the underlying transport fails before a response is received.
    async fn call(&self, method: &str, params: Value) -> Result<Value>;

    /// Send a JSON-RPC notification. No response is expected or awaited.
    async fn notify(&self, method: &str, params: Value) -> Result<()>;

    /// Best-effort, idempotent shutdown of the transport.
    async fn close(&self) -> Result<()>;

    /// Register the handler responsible for computing responses to
    /// server-initiated requests (`ping`, `sampling/createMessage`,
    /// `roots/list`). [`crate::McpClient`] installs one automatically at
    /// construction, so callers normally never need to call this directly.
    /// Replaces any previously registered handler.
    ///
    /// The default implementation does nothing, for transports (e.g. test
    /// mocks) that never see server-initiated requests.
    fn set_server_request_handler(&self, handler: BoxedServerRequestHandler) {
        let _ = handler;
    }

    /// Register the handler invoked for every notification received from the
    /// server (no response expected). [`crate::McpClient`] installs one
    /// automatically at construction time to invalidate its tools/prompts
    /// cache on `notifications/tools/list_changed` /
    /// `notifications/prompts/list_changed` — see
    /// [`crate::McpClient::list_tools_cached`] /
    /// [`crate::McpClient::list_prompts_cached`]. Replaces any previously
    /// registered handler.
    ///
    /// The default implementation does nothing, for transports (e.g. test
    /// mocks) that never see server-initiated notifications.
    fn set_notification_handler(&self, handler: BoxedNotificationHandler) {
        let _ = handler;
    }

    /// Whether the connection is gone for good: the stdio server exited,
    /// the websocket closed, or the HTTP server ended the session (`404` on
    /// a request carrying its session id). A closed transport never
    /// recovers; the owner reconnects with a new one. The MCP tool wrappers
    /// read this after a failed call to decide whether reconnecting and
    /// retrying once is worthwhile, as upstream does on
    /// `ClosedResourceError` / "session terminated".
    ///
    /// The default is `false`, for transports that cannot tell.
    fn is_closed(&self) -> bool {
        false
    }
}

pub use http::McpStreamableHttpTransport;
pub use stdio::{McpStdioTransport, StdioEnv};
pub use websocket::McpWebsocketTransport;
