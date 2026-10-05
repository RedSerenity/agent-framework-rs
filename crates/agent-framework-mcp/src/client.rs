//! [`McpClient`]: JSON-RPC/MCP methods layered over any [`McpTransport`].

use std::sync::{Arc, RwLock as StdRwLock};

use serde_json::{json, Value};
use tokio::sync::RwLock;

use agent_framework_core::error::{Error, Result};

use crate::protocol::{
    CallToolResult, GetPromptResult, Implementation, InitializeResult, ListPromptsResult,
    ListToolsResult, McpLogLevel, PromptDescriptor, ToolDescriptor, COMPATIBLE_PROTOCOL_VERSIONS,
    PROTOCOL_VERSION,
};
use crate::sampling::{dispatch_server_request, Root, SamplingHandler, ServerRequestHandlers};
use crate::transport::McpTransport;

/// The notification methods [`McpClient::new`] listens for to invalidate its
/// tools/prompts caches. See [`McpClient::list_tools_cached`] /
/// [`McpClient::list_prompts_cached`].
const TOOLS_LIST_CHANGED: &str = "notifications/tools/list_changed";
const PROMPTS_LIST_CHANGED: &str = "notifications/prompts/list_changed";

/// Safety cap on `tools/list`/`prompts/list` pagination, so a server that
/// never stops returning a `nextCursor` cannot spin the client forever.
const LOG_MESSAGE: &str = "notifications/message";

const MAX_LIST_PAGES: usize = 10_000;

/// Re-emit a server's `notifications/message` through `tracing`, at the
/// level it mapped from (upstream's `logging_callback`: debug→DEBUG,
/// info/notice→INFO, warning→WARNING, everything above→ERROR/CRITICAL).
fn log_server_message(params: &Value) {
    let level = params
        .get("level")
        .and_then(Value::as_str)
        .and_then(McpLogLevel::parse)
        .unwrap_or(McpLogLevel::Info);
    let logger = params.get("logger").and_then(Value::as_str).unwrap_or("");
    let data = params.get("data").cloned().unwrap_or(Value::Null);
    let data = match &data {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    match level {
        McpLogLevel::Debug => tracing::debug!(target: "mcp_server", logger, "{data}"),
        McpLogLevel::Info | McpLogLevel::Notice => {
            tracing::info!(target: "mcp_server", logger, "{data}")
        }
        McpLogLevel::Warning => tracing::warn!(target: "mcp_server", logger, "{data}"),
        _ => tracing::error!(target: "mcp_server", logger, level = level.as_str(), "{data}"),
    }
}

/// A connected MCP session: `initialize`, `ping`, `tools/list`, `tools/call`,
/// `prompts/list`, `prompts/get`, layered over any [`McpTransport`], plus
/// server-initiated `sampling/createMessage` / `roots/list` support.
///
/// `McpClient` is transport-agnostic: construct it with an `Arc<dyn McpTransport>`
/// (see [`crate::McpStdioTransport`] / [`crate::McpStreamableHttpTransport`]), or
/// use [`crate::McpStdioTool`] / [`crate::McpStreamableHttpTool`], which each own
/// one for you and hand out ready-to-use [`agent_framework_core::tools::ToolDefinition`]s.
///
/// [`Self::list_tools_cached`] / [`Self::list_prompts_cached`] cache their
/// result across calls, invalidated automatically when the server sends
/// `notifications/tools/list_changed` / `notifications/prompts/list_changed`
/// (see [`Self::new`]) — this is what lets a
/// [`agent_framework_core::tools::ToolSource`] resolve on every agent run
/// cheaply instead of always performing a live round trip.
pub struct McpClient {
    transport: Arc<dyn McpTransport>,
    initialize_result: RwLock<Option<InitializeResult>>,
    handlers: Arc<StdRwLock<ServerRequestHandlers>>,
    /// Cache for [`Self::list_tools_cached`], invalidated on a
    /// `notifications/tools/list_changed` notification from the server.
    tools_cache: Arc<RwLock<Option<Vec<ToolDescriptor>>>>,
    /// Cache for [`Self::list_prompts_cached`], invalidated on a
    /// `notifications/prompts/list_changed` notification from the server.
    prompts_cache: Arc<RwLock<Option<Vec<PromptDescriptor>>>>,
    /// Serializes cache *misses* in [`Self::list_tools_cached`] so concurrent
    /// resolvers don't each fire their own `tools/list` round trip. A separate
    /// lock (rather than holding the `tools_cache` write lock across the
    /// fetch) because the `list_changed` notification handler takes the cache
    /// write lock and can run *inside* our own fetch's transport processing —
    /// holding the cache lock across the round trip would deadlock.
    tools_fetch_lock: tokio::sync::Mutex<()>,
    /// [`Self::list_prompts_cached`]'s counterpart to `tools_fetch_lock`.
    prompts_fetch_lock: tokio::sync::Mutex<()>,
    /// Bumped by every `tools/list_changed` notification; a fetch only
    /// publishes its result to `tools_cache` when no invalidation raced it.
    tools_generation: Arc<std::sync::atomic::AtomicU64>,
    /// `prompts/list_changed` counterpart to `tools_generation`.
    prompts_generation: Arc<std::sync::atomic::AtomicU64>,
}

impl McpClient {
    /// Wrap a transport in a fresh, uninitialized session.
    ///
    /// Installs the server-request dispatcher into `transport` immediately
    /// (see [`McpTransport::set_server_request_handler`]), so
    /// [`Self::sampling_handler`] / [`Self::roots`] take effect for any
    /// server-initiated request the transport sees from this point on, even
    /// ones that arrive before [`Self::initialize`] completes. Also installs
    /// a notification handler (see [`McpTransport::set_notification_handler`])
    /// that invalidates [`Self::list_tools_cached`] / [`Self::list_prompts_cached`]
    /// on `notifications/tools/list_changed` / `notifications/prompts/list_changed`.
    pub fn new(transport: Arc<dyn McpTransport>) -> Self {
        let handlers: Arc<StdRwLock<ServerRequestHandlers>> = Arc::default();
        let dispatch_handlers = handlers.clone();
        transport.set_server_request_handler(Arc::new(move |method: String, params: Value| {
            let handlers = dispatch_handlers.clone();
            Box::pin(async move { dispatch_server_request(&handlers, &method, params).await })
        }));

        let tools_cache: Arc<RwLock<Option<Vec<ToolDescriptor>>>> = Arc::new(RwLock::new(None));
        let prompts_cache: Arc<RwLock<Option<Vec<PromptDescriptor>>>> = Arc::new(RwLock::new(None));
        let tools_generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let prompts_generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let notif_tools_cache = tools_cache.clone();
        let notif_prompts_cache = prompts_cache.clone();
        let notif_tools_generation = tools_generation.clone();
        let notif_prompts_generation = prompts_generation.clone();
        transport.set_notification_handler(Arc::new(move |method: String, params: Value| {
            let tools_cache = notif_tools_cache.clone();
            let prompts_cache = notif_prompts_cache.clone();
            let tools_generation = notif_tools_generation.clone();
            let prompts_generation = notif_prompts_generation.clone();
            Box::pin(async move {
                // Bump the generation *before* clearing so an in-flight fetch
                // that started earlier observes the change and declines to
                // publish its (possibly stale) result.
                match method.as_str() {
                    TOOLS_LIST_CHANGED => {
                        tools_generation.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                        *tools_cache.write().await = None;
                    }
                    PROMPTS_LIST_CHANGED => {
                        prompts_generation.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                        *prompts_cache.write().await = None;
                    }
                    LOG_MESSAGE => log_server_message(&params),
                    other => tracing::debug!(method = %other, "unhandled MCP notification"),
                }
            })
        }));

        Self {
            transport,
            initialize_result: RwLock::new(None),
            handlers,
            tools_cache,
            prompts_cache,
            tools_fetch_lock: tokio::sync::Mutex::new(()),
            prompts_fetch_lock: tokio::sync::Mutex::new(()),
            tools_generation,
            prompts_generation,
        }
    }

    /// Register the handler for server-initiated `sampling/createMessage`
    /// requests. Declares the `sampling` capability on the next
    /// [`Self::initialize`] call — set this up before initializing, matching
    /// the `mcp` Python SDK, which derives the capability from whichever
    /// callback was passed to the session at construction time.
    pub fn sampling_handler(self, handler: SamplingHandler) -> Self {
        self.handlers.write().unwrap().sampling = Some(handler);
        self
    }

    /// Register a static list of filesystem roots, answered when the server
    /// sends `roots/list`. Declares the `roots` capability on the next
    /// [`Self::initialize`] call, the same way [`Self::sampling_handler`] does.
    pub fn roots(self, roots: Vec<Root>) -> Self {
        self.handlers.write().unwrap().roots = Some(roots);
        self
    }

    /// The `capabilities` object sent during `initialize`: `sampling`/`roots`
    /// are included only when a handler is registered via
    /// [`Self::sampling_handler`] / [`Self::roots`] — matching the `mcp`
    /// Python SDK, which derives `ClientCapabilities` from whichever
    /// callbacks were supplied. Unlike that SDK (which always sets
    /// `roots.listChanged: true` whenever any roots callback is present,
    /// regardless of whether it ever actually sends that notification),
    /// this always advertises `listChanged: false`: this crate's root list
    /// is static, so that's the honest answer.
    fn client_capabilities(&self) -> Value {
        let handlers = self.handlers.read().unwrap();
        let mut capabilities = serde_json::Map::new();
        if handlers.sampling.is_some() {
            capabilities.insert("sampling".to_string(), json!({}));
        }
        if handlers.roots.is_some() {
            capabilities.insert("roots".to_string(), json!({ "listChanged": false }));
        }
        Value::Object(capabilities)
    }

    /// Perform the MCP `initialize` request followed by the
    /// `notifications/initialized` notification.
    ///
    /// Idempotent: once a handshake has succeeded, later calls return the
    /// cached result without re-sending anything. The client always *sends*
    /// [`PROTOCOL_VERSION`], but accepts whatever `protocolVersion` the
    /// server responds with — including an older one — logging a warning if
    /// it isn't one this client specifically recognizes.
    pub async fn initialize(
        &self,
        client_name: &str,
        client_version: &str,
    ) -> Result<InitializeResult> {
        if let Some(existing) = self.initialize_result.read().await.clone() {
            return Ok(existing);
        }

        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": self.client_capabilities(),
            "clientInfo": Implementation {
                name: client_name.to_string(),
                version: client_version.to_string(),
            },
        });
        let raw = self.transport.call("initialize", params).await?;
        let result: InitializeResult = serde_json::from_value(raw)
            .map_err(|e| Error::service(format!("invalid MCP initialize response: {e}")))?;

        if !COMPATIBLE_PROTOCOL_VERSIONS.contains(&result.protocol_version.as_str()) {
            tracing::warn!(
                server_protocol_version = %result.protocol_version,
                client_protocol_version = PROTOCOL_VERSION,
                "MCP server negotiated an unrecognized protocol version; proceeding anyway"
            );
        }

        self.transport
            .notify("notifications/initialized", json!({}))
            .await?;

        *self.initialize_result.write().await = Some(result.clone());
        Ok(result)
    }

    /// `ping` — verify the server is still responsive.
    pub async fn ping(&self) -> Result<()> {
        self.require_initialized().await?;
        self.transport.call("ping", json!({})).await?;
        Ok(())
    }

    /// List every tool the server exposes, transparently following
    /// `nextCursor` pagination until the server stops returning one.
    pub async fn list_tools(&self) -> Result<Vec<ToolDescriptor>> {
        self.require_initialized().await?;
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let raw = self.transport.call("tools/list", params).await?;
            let page: ListToolsResult = serde_json::from_value(raw)
                .map_err(|e| Error::service(format!("invalid MCP tools/list response: {e}")))?;
            tools.extend(page.tools);
            match page.next_cursor {
                Some(next) if !next.is_empty() => cursor = Some(next),
                _ => return Ok(tools),
            }
        }
        Err(Error::service(
            "MCP tools/list pagination exceeded the maximum page count",
        ))
    }

    /// [`Self::list_tools`], cached after the first successful call: later
    /// calls return the cached list without a round trip, until a
    /// `notifications/tools/list_changed` notification from the server (see
    /// [`Self::new`]) invalidates it, or the call fails (nothing is cached
    /// on `Err`, so the next call retries rather than sticking with a
    /// failure).
    ///
    /// Used by `agent-framework-mcp`'s tool wrappers to implement
    /// [`agent_framework_core::tools::ToolSource::resolve_tools`], so a tool
    /// source can be resolved on every agent run without paying for a live
    /// `tools/list` round trip each time.
    pub async fn list_tools_cached(&self) -> Result<Vec<ToolDescriptor>> {
        if let Some(cached) = self.tools_cache.read().await.clone() {
            return Ok(cached);
        }
        // Serialize misses so concurrent resolvers share one round trip (see
        // the `tools_fetch_lock` field docs for why this is not done by
        // holding the cache write lock across the fetch).
        let _fetch = self.tools_fetch_lock.lock().await;
        if let Some(cached) = self.tools_cache.read().await.clone() {
            return Ok(cached);
        }
        let generation = self
            .tools_generation
            .load(std::sync::atomic::Ordering::Acquire);
        let tools = self.list_tools().await?;
        // Publish only if no `list_changed` invalidation raced the fetch;
        // otherwise the next call refetches a fresh list. The generation
        // recheck happens while HOLDING the cache write lock — checked
        // before acquiring it, an invalidation landing in between would be
        // overwritten by this stale publish. (The lock is never held across
        // the fetch itself, so the notification handler cannot deadlock.)
        {
            let mut cache = self.tools_cache.write().await;
            if self
                .tools_generation
                .load(std::sync::atomic::Ordering::Acquire)
                == generation
            {
                *cache = Some(tools.clone());
            }
        }
        Ok(tools)
    }

    /// `tools/call` — invoke a tool and return the raw result, including
    /// `is_error`, without converting an error result into an `Err`.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<CallToolResult> {
        self.call_tool_with_meta(name, arguments, None).await
    }

    /// [`Self::call_tool`] with request `_meta` (a progress token, trace
    /// context, or metadata a proxy requires echoed from `tools/list`).
    pub async fn call_tool_with_meta(
        &self,
        name: &str,
        arguments: Value,
        meta: Option<Value>,
    ) -> Result<CallToolResult> {
        self.require_initialized().await?;
        let arguments = if arguments.is_null() {
            json!({})
        } else {
            arguments
        };
        let mut params = json!({ "name": name, "arguments": arguments });
        if let Some(meta) = meta.filter(|m| m.as_object().is_some_and(|o| !o.is_empty())) {
            params["_meta"] = meta;
        }
        let raw = self.transport.call("tools/call", params).await?;
        Ok(CallToolResult::from_value(&raw))
    }

    /// Whether the server declared the `logging` capability.
    pub async fn supports_logging(&self) -> bool {
        self.initialize_result
            .read()
            .await
            .as_ref()
            .is_some_and(|r| r.capabilities.get("logging").is_some())
    }

    /// `logging/setLevel` — ask the server to send `notifications/message`
    /// at `level` and above. Those messages are re-emitted through
    /// `tracing` (target `mcp_server`).
    pub async fn set_logging_level(&self, level: McpLogLevel) -> Result<()> {
        self.require_initialized().await?;
        self.transport
            .call("logging/setLevel", json!({ "level": level.as_str() }))
            .await?;
        Ok(())
    }

    /// Whether the underlying connection is gone for good (see
    /// [`McpTransport::is_closed`]); a closed client must be replaced.
    pub fn is_closed(&self) -> bool {
        self.transport.is_closed()
    }

    /// `tools/call`, mapped to a single JSON value suitable for handing back
    /// to a model: a lone text block becomes a string (or the value it parses
    /// as, if it's valid JSON); `isError: true` becomes `Err(Error::Tool(..))`.
    pub async fn call_tool_value(&self, name: &str, arguments: Value) -> Result<Value> {
        let result = self.call_tool(name, arguments).await?;
        if result.is_error {
            return Err(Error::tool(result.error_message()));
        }
        Ok(result.to_value())
    }

    /// List every prompt the server exposes, transparently following
    /// `nextCursor` pagination until the server stops returning one.
    ///
    /// Short-circuits to an empty list, without issuing any request, if the
    /// server didn't declare the `prompts` capability during `initialize` —
    /// the Rust equivalent of the Python reference's try/except around
    /// `session.list_prompts()` (which logs and treats a failure the same
    /// way), checking the negotiated capability up front instead of
    /// discarding an expected error.
    pub async fn list_prompts(&self) -> Result<Vec<PromptDescriptor>> {
        self.require_initialized().await?;
        if !self.supports_prompts().await {
            return Ok(Vec::new());
        }
        let mut prompts = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let raw = self.transport.call("prompts/list", params).await?;
            let page: ListPromptsResult = serde_json::from_value(raw)
                .map_err(|e| Error::service(format!("invalid MCP prompts/list response: {e}")))?;
            prompts.extend(page.prompts);
            match page.next_cursor {
                Some(next) if !next.is_empty() => cursor = Some(next),
                _ => return Ok(prompts),
            }
        }
        Err(Error::service(
            "MCP prompts/list pagination exceeded the maximum page count",
        ))
    }

    /// [`Self::list_prompts`], cached the same way as
    /// [`Self::list_tools_cached`]: reused until a
    /// `notifications/prompts/list_changed` notification invalidates it, or
    /// the call fails.
    pub async fn list_prompts_cached(&self) -> Result<Vec<PromptDescriptor>> {
        if let Some(cached) = self.prompts_cache.read().await.clone() {
            return Ok(cached);
        }
        // Same miss-serialization + raced-invalidation rules as
        // [`Self::list_tools_cached`].
        let _fetch = self.prompts_fetch_lock.lock().await;
        if let Some(cached) = self.prompts_cache.read().await.clone() {
            return Ok(cached);
        }
        let generation = self
            .prompts_generation
            .load(std::sync::atomic::Ordering::Acquire);
        let prompts = self.list_prompts().await?;
        // Recheck-while-holding-the-lock, as in `list_tools_cached`.
        {
            let mut cache = self.prompts_cache.write().await;
            if self
                .prompts_generation
                .load(std::sync::atomic::Ordering::Acquire)
                == generation
            {
                *cache = Some(prompts.clone());
            }
        }
        Ok(prompts)
    }

    /// `prompts/get` — fetch a rendered prompt's messages.
    ///
    /// `arguments` should be a JSON object of string values per the MCP
    /// spec (`GetPromptRequest.params.arguments?: {[key: string]: string}`);
    /// `Value::Null` is sent as `{}`, mirroring [`Self::call_tool`].
    ///
    /// Unlike [`Self::list_prompts`], this does not pre-check the `prompts`
    /// capability: a caller invoking a specific prompt by name presumably
    /// already knows it exists, and a server that doesn't support prompts at
    /// all will simply answer with a JSON-RPC error, surfaced here as
    /// [`Error::Service`] — mirroring how the Python reference's
    /// `get_prompt` lets the underlying request fail naturally.
    pub async fn get_prompt(&self, name: &str, arguments: Value) -> Result<GetPromptResult> {
        self.require_initialized().await?;
        let arguments = if arguments.is_null() {
            json!({})
        } else {
            arguments
        };
        let params = json!({ "name": name, "arguments": arguments });
        let raw = self.transport.call("prompts/get", params).await?;
        serde_json::from_value(raw)
            .map_err(|e| Error::service(format!("invalid MCP prompts/get response: {e}")))
    }

    /// Whether the server declared the `prompts` capability during
    /// [`Self::initialize`]. `false` before `initialize` has completed.
    pub async fn supports_prompts(&self) -> bool {
        self.initialize_result
            .read()
            .await
            .as_ref()
            .map(InitializeResult::supports_prompts)
            .unwrap_or(false)
    }

    /// The server's `serverInfo`, once [`Self::initialize`] has completed.
    pub async fn server_info(&self) -> Option<Implementation> {
        self.initialize_result
            .read()
            .await
            .as_ref()
            .map(|r| r.server_info.clone())
    }

    /// The protocol version negotiated with the server, once
    /// [`Self::initialize`] has completed.
    pub async fn protocol_version(&self) -> Option<String> {
        self.initialize_result
            .read()
            .await
            .as_ref()
            .map(|r| r.protocol_version.clone())
    }

    /// Whether [`Self::initialize`] has completed successfully.
    pub async fn is_initialized(&self) -> bool {
        self.initialize_result.read().await.is_some()
    }

    /// Gracefully close the underlying transport (best effort, idempotent):
    /// kills the child process for stdio, or `DELETE`s the session for
    /// streamable HTTP.
    pub async fn close(&self) -> Result<()> {
        self.transport.close().await
    }

    async fn require_initialized(&self) -> Result<()> {
        if self.is_initialized().await {
            Ok(())
        } else {
            Err(Error::service(
                "MCP client is not initialized; call initialize() first",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::CreateMessageResult;
    use async_trait::async_trait;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Mutex;

    /// A canned, in-memory transport for exercising [`McpClient`] without any I/O.
    struct MockTransport {
        responses: Mutex<HashMap<String, VecDeque<Value>>>,
        /// Every `(method, params)` pair passed to `call`, in order — lets
        /// tests assert on what a higher-level `McpClient` method actually
        /// sent (e.g. the `capabilities` object during `initialize`).
        sent: Mutex<Vec<(String, Value)>>,
        /// The handler [`McpClient::new`] installed via
        /// [`McpTransport::set_notification_handler`], captured so tests can
        /// simulate the server sending a notification via
        /// [`Self::fire_notification`].
        notification_handler: Mutex<Option<crate::sampling::BoxedNotificationHandler>>,
        /// When set to `(trigger_method, notif_method, params)`, the next
        /// [`McpTransport::call`] whose method equals `trigger_method` fires
        /// the notification *before returning* — simulating a server
        /// notification racing an in-flight request.
        notify_during_call: Mutex<Option<(String, String, Value)>>,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                responses: Mutex::new(HashMap::new()),
                sent: Mutex::new(Vec::new()),
                notification_handler: Mutex::new(None),
                notify_during_call: Mutex::new(None),
            }
        }

        /// Arrange for `notif_method` to fire mid-flight during the next
        /// `call(trigger_method, ...)`. See `notify_during_call`.
        fn notify_during(&self, trigger_method: &str, notif_method: &str, params: Value) {
            *self.notify_during_call.lock().unwrap() =
                Some((trigger_method.to_string(), notif_method.to_string(), params));
        }

        fn push(&self, method: &str, value: Value) {
            self.responses
                .lock()
                .unwrap()
                .entry(method.to_string())
                .or_default()
                .push_back(value);
        }

        /// The params of the most recent `call(method, ..)`, if any.
        fn last_params(&self, method: &str) -> Option<Value> {
            self.sent
                .lock()
                .unwrap()
                .iter()
                .rev()
                .find(|(m, _)| m == method)
                .map(|(_, p)| p.clone())
        }

        /// Simulate the server sending a notification (e.g.
        /// `notifications/tools/list_changed`): invokes whatever handler
        /// [`McpClient::new`] registered via
        /// [`McpTransport::set_notification_handler`], the same way a real
        /// transport's reader loop would on seeing one on the wire.
        async fn fire_notification(&self, method: &str, params: Value) {
            let handler = self.notification_handler.lock().unwrap().clone();
            let handler = handler.expect(
                "no notification handler registered; construct the McpClient before firing",
            );
            handler(method.to_string(), params).await;
        }
    }

    #[async_trait]
    impl McpTransport for MockTransport {
        async fn call(&self, method: &str, params: Value) -> Result<Value> {
            self.sent.lock().unwrap().push((method.to_string(), params));
            let response = {
                let mut guard = self.responses.lock().unwrap();
                let queue = guard
                    .get_mut(method)
                    .unwrap_or_else(|| panic!("no canned responses queued for method '{method}'"));
                queue
                    .pop_front()
                    .ok_or_else(|| Error::service(format!("no more canned responses for {method}")))
            };
            // Simulate a server notification landing while this very call is
            // still in flight (as a real transport's reader loop dispatches
            // mid-response) — used to exercise raced-invalidation handling.
            let fire = {
                let mut pending = self.notify_during_call.lock().unwrap();
                match pending.take() {
                    Some((trigger, notif_method, notif_params)) if trigger == method => {
                        Some((notif_method, notif_params))
                    }
                    other => {
                        *pending = other;
                        None
                    }
                }
            };
            if let Some((notif_method, notif_params)) = fire {
                self.fire_notification(&notif_method, notif_params).await;
            }
            response
        }

        async fn notify(&self, _method: &str, _params: Value) -> Result<()> {
            Ok(())
        }

        async fn close(&self) -> Result<()> {
            Ok(())
        }

        fn set_notification_handler(&self, handler: crate::sampling::BoxedNotificationHandler) {
            *self.notification_handler.lock().unwrap() = Some(handler);
        }
    }

    fn init_result() -> Value {
        json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "serverInfo": {"name": "mock-server", "version": "0.0.1"},
        })
    }

    #[tokio::test]
    async fn initialize_caches_result_and_is_idempotent() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let client = McpClient::new(mock);

        let first = client.initialize("c", "1").await.unwrap();
        assert_eq!(first.server_info.name, "mock-server");

        // Second call must not attempt to pull another canned response (there
        // isn't one) — it should just return the cached result.
        let second = client.initialize("c", "1").await.unwrap();
        assert_eq!(second.protocol_version, "2025-06-18");
    }

    #[tokio::test]
    async fn methods_require_initialize_first() {
        let mock = Arc::new(MockTransport::new());
        let client = McpClient::new(mock);
        let err = client.list_tools().await.unwrap_err();
        assert!(matches!(err, Error::Service(_)));
    }

    #[tokio::test]
    async fn list_tools_follows_cursor_pagination() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push(
            "tools/list",
            json!({
                "tools": [{"name": "a", "inputSchema": {"type":"object","properties":{}}}],
                "nextCursor": "page2",
            }),
        );
        mock.push(
            "tools/list",
            json!({
                "tools": [{"name": "b", "inputSchema": {"type":"object","properties":{}}}],
            }),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        let tools = client.list_tools().await.unwrap();
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[tokio::test]
    async fn call_tool_value_maps_single_text_block_to_string() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push(
            "tools/call",
            json!({"content": [{"type": "text", "text": "hello world"}], "isError": false}),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        let value = client
            .call_tool_value("echo", json!({"text": "hello world"}))
            .await
            .unwrap();
        assert_eq!(value, json!("hello world"));
    }

    #[tokio::test]
    async fn call_tool_value_parses_json_text_block() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push(
            "tools/call",
            json!({"content": [{"type": "text", "text": "42"}], "isError": false}),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        let value = client
            .call_tool_value("add", json!({"a": 40, "b": 2}))
            .await
            .unwrap();
        assert_eq!(value, json!(42));
    }

    #[tokio::test]
    async fn call_tool_value_maps_is_error_to_tool_error() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push(
            "tools/call",
            json!({"content": [{"type": "text", "text": "boom"}], "isError": true}),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        let err = client
            .call_tool_value("broken", json!({}))
            .await
            .unwrap_err();
        match err {
            Error::Tool(msg) => assert_eq!(msg, "boom"),
            other => panic!("expected Error::Tool, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn ping_and_server_info_after_initialize() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push("ping", json!({}));
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        client.ping().await.unwrap();
        let info = client.server_info().await.unwrap();
        assert_eq!(info.name, "mock-server");
        assert_eq!(client.protocol_version().await.unwrap(), "2025-06-18");
    }

    // -- Capability declaration ------------------------------------------

    #[tokio::test]
    async fn initialize_omits_sampling_and_roots_capabilities_by_default() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();
        let sent = mock.last_params("initialize").unwrap();
        assert_eq!(sent["capabilities"], json!({}));
    }

    #[tokio::test]
    async fn initialize_declares_sampling_capability_only_when_handler_set() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let handler: SamplingHandler =
            Arc::new(|_| Box::pin(async { Ok(CreateMessageResult::text("assistant", "hi", "m")) }));
        let client = McpClient::new(mock.clone()).sampling_handler(handler);
        client.initialize("c", "1").await.unwrap();
        let sent = mock.last_params("initialize").unwrap();
        assert_eq!(sent["capabilities"], json!({"sampling": {}}));
    }

    #[tokio::test]
    async fn initialize_declares_roots_capability_only_when_set() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let client = McpClient::new(mock.clone()).roots(vec![Root::new("file:///tmp")]);
        client.initialize("c", "1").await.unwrap();
        let sent = mock.last_params("initialize").unwrap();
        assert_eq!(
            sent["capabilities"],
            json!({"roots": {"listChanged": false}})
        );
    }

    #[tokio::test]
    async fn initialize_declares_both_capabilities_when_both_set() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let handler: SamplingHandler =
            Arc::new(|_| Box::pin(async { Ok(CreateMessageResult::text("assistant", "hi", "m")) }));
        let client = McpClient::new(mock.clone())
            .sampling_handler(handler)
            .roots(vec![Root::new("file:///tmp")]);
        client.initialize("c", "1").await.unwrap();
        let sent = mock.last_params("initialize").unwrap();
        assert_eq!(sent["capabilities"]["sampling"], json!({}));
        assert_eq!(sent["capabilities"]["roots"]["listChanged"], json!(false));
    }

    // -- Prompts ----------------------------------------------------------

    fn init_result_with_prompts() -> Value {
        json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {"prompts": {}},
            "serverInfo": {"name": "mock-server", "version": "0.0.1"},
        })
    }

    #[tokio::test]
    async fn list_prompts_short_circuits_without_prompts_capability() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();
        assert!(!client.supports_prompts().await);
        let prompts = client.list_prompts().await.unwrap();
        assert!(prompts.is_empty());
        // No `prompts/list` request should ever have been sent.
        assert!(mock.last_params("prompts/list").is_none());
    }

    #[tokio::test]
    async fn list_prompts_follows_cursor_pagination_when_supported() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result_with_prompts());
        mock.push(
            "prompts/list",
            json!({
                "prompts": [{"name": "greet"}],
                "nextCursor": "page2",
            }),
        );
        mock.push(
            "prompts/list",
            json!({
                "prompts": [{"name": "farewell"}],
            }),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        assert!(client.supports_prompts().await);
        let prompts = client.list_prompts().await.unwrap();
        assert_eq!(
            prompts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            vec!["greet", "farewell"]
        );
    }

    #[tokio::test]
    async fn get_prompt_maps_messages() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result_with_prompts());
        mock.push(
            "prompts/get",
            json!({
                "description": "A greeting",
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": "Say hi to Ada"}},
                ],
            }),
        );
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();
        let result = client
            .get_prompt("greet", json!({"name": "Ada"}))
            .await
            .unwrap();
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
        let sent = mock.last_params("prompts/get").unwrap();
        assert_eq!(sent["name"], "greet");
        assert_eq!(sent["arguments"]["name"], "Ada");
    }

    #[tokio::test]
    async fn get_prompt_null_arguments_sent_as_empty_object() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result_with_prompts());
        mock.push("prompts/get", json!({"messages": []}));
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();
        client.get_prompt("greet", Value::Null).await.unwrap();
        let sent = mock.last_params("prompts/get").unwrap();
        assert_eq!(sent["arguments"], json!({}));
    }

    // -- Tools/prompts caching + list_changed invalidation ---------------

    #[tokio::test]
    async fn list_tools_cached_reuses_result_until_invalidated() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push("tools/list", json!({"tools": [{"name": "a"}]}));
        mock.push(
            "tools/list",
            json!({"tools": [{"name": "a"}, {"name": "b"}]}),
        );
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();

        // First call performs a live round trip and caches the result.
        let first = client.list_tools_cached().await.unwrap();
        assert_eq!(
            first.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["a"]
        );

        // Second call is a cache hit: if it were live, `MockTransport::call`
        // would panic (only one `tools/list` response was left queued after
        // consuming the second one) or return the second canned list; it
        // must return the identical first list without consuming anything.
        let second = client.list_tools_cached().await.unwrap();
        assert_eq!(
            second.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["a"],
            "cache hit must not perform a live round trip"
        );

        // The server signals a change; the next call must refetch and
        // observe the new list.
        mock.fire_notification("notifications/tools/list_changed", json!({}))
            .await;
        let third = client.list_tools_cached().await.unwrap();
        assert_eq!(
            third.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"],
            "cache must refetch after a list_changed notification"
        );
    }

    #[tokio::test]
    async fn concurrent_cache_misses_share_a_single_fetch() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        // Exactly ONE tools/list response queued: if both concurrent misses
        // fetched, the second would find the queue empty and error.
        mock.push("tools/list", json!({"tools": [{"name": "a"}]}));
        let client = Arc::new(McpClient::new(mock.clone()));
        client.initialize("c", "1").await.unwrap();

        let (c1, c2) = (client.clone(), client.clone());
        let (r1, r2) = tokio::join!(
            tokio::spawn(async move { c1.list_tools_cached().await }),
            tokio::spawn(async move { c2.list_tools_cached().await }),
        );
        let names = |r: std::result::Result<Result<Vec<ToolDescriptor>>, _>| {
            r.expect("join")
                .expect("list")
                .iter()
                .map(|t| t.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(r1), vec!["a"]);
        assert_eq!(names(r2), vec!["a"]);
        let fetches = mock
            .sent
            .lock()
            .unwrap()
            .iter()
            .filter(|(m, _)| m == "tools/list")
            .count();
        assert_eq!(fetches, 1, "concurrent misses must share one round trip");
    }

    #[tokio::test]
    async fn invalidation_racing_an_in_flight_fetch_is_not_lost() {
        // A list_changed notification that lands while a fetch is in flight
        // (dispatched by the transport mid-call, exactly like a real reader
        // loop) must prevent that fetch from publishing a possibly-stale
        // cache — the NEXT call refetches instead of serving the raced list.
        // This also proves the notification handler cannot deadlock against
        // an in-flight `list_tools_cached` (the reason the fetch does not
        // hold the cache write lock across the round trip).
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push("tools/list", json!({"tools": [{"name": "raced"}]}));
        mock.push("tools/list", json!({"tools": [{"name": "fresh"}]}));
        let client = Arc::new(McpClient::new(mock.clone()));
        client.initialize("c", "1").await.unwrap();

        mock.notify_during("tools/list", "notifications/tools/list_changed", json!({}));
        let first = client.list_tools_cached().await.unwrap();
        // The in-flight fetch still returns what it fetched...
        assert_eq!(first[0].name, "raced");
        // ...but must NOT have cached it: the raced invalidation wins and the
        // next call performs a fresh round trip.
        let second = client.list_tools_cached().await.unwrap();
        assert_eq!(
            second[0].name, "fresh",
            "a fetch raced by list_changed must not publish a stale cache"
        );
    }

    #[tokio::test]
    async fn list_prompts_cached_reuses_result_until_invalidated() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result_with_prompts());
        mock.push("prompts/list", json!({"prompts": [{"name": "greet"}]}));
        mock.push(
            "prompts/list",
            json!({"prompts": [{"name": "greet"}, {"name": "farewell"}]}),
        );
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();

        let first = client.list_prompts_cached().await.unwrap();
        assert_eq!(first.len(), 1);

        let second = client.list_prompts_cached().await.unwrap();
        assert_eq!(
            second.len(),
            1,
            "cache hit must not perform a live round trip"
        );

        mock.fire_notification("notifications/prompts/list_changed", json!({}))
            .await;
        let third = client.list_prompts_cached().await.unwrap();
        assert_eq!(
            third.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            vec!["greet", "farewell"],
            "cache must refetch after a list_changed notification"
        );
    }

    #[tokio::test]
    async fn list_tools_changed_notification_does_not_invalidate_prompts_cache() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result_with_prompts());
        mock.push("tools/list", json!({"tools": [{"name": "a"}]}));
        mock.push("prompts/list", json!({"prompts": [{"name": "greet"}]}));
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();

        client.list_tools_cached().await.unwrap();
        client.list_prompts_cached().await.unwrap();

        // A tools-only notification must not force a prompts refetch (there
        // is only one `prompts/list` response queued; a spurious second
        // fetch would panic in `MockTransport::call`).
        mock.fire_notification("notifications/tools/list_changed", json!({}))
            .await;
        client.list_prompts_cached().await.unwrap();
    }

    #[tokio::test]
    async fn unrelated_notification_does_not_invalidate_either_cache() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push("tools/list", json!({"tools": [{"name": "a"}]}));
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();

        client.list_tools_cached().await.unwrap();
        mock.fire_notification("notifications/message", json!({"level": "info"}))
            .await;
        // Only one `tools/list` response was ever queued; a second live call
        // here would panic in `MockTransport::call`.
        client.list_tools_cached().await.unwrap();
    }
}
