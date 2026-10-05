//! Server-initiated request support: `sampling/createMessage`, `roots/list`,
//! and `ping`.
//!
//! Every transport routes an incoming server-initiated ("client-bound")
//! request to [`dispatch_server_request`], which:
//!
//! - answers `ping` unconditionally with an empty result, before consulting
//!   any handler — mirroring the `mcp` Python SDK's `ClientSession`, which
//!   intercepts `ping` before any user callback ever sees it;
//! - delegates `sampling/createMessage` to whatever [`SamplingHandler`]
//!   [`crate::McpClient::sampling_handler`] (or the equivalent tool-wrapper
//!   builder method) registered;
//! - delegates `roots/list` to whatever static list
//!   [`crate::McpClient::roots`] registered;
//! - maps anything else — including sampling/roots when no handler is
//!   registered — to a JSON-RPC "method not found" error, never silence.

use std::sync::{Arc, RwLock as StdRwLock};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::tools::BoxFuture;
use agent_framework_core::types::{ChatOptions, ChatResponse, Content, DataContent, Message};

use crate::protocol::{role_and_content_to_chat_message, RpcError};

const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

// ---------------------------------------------------------------------
// sampling/createMessage
// ---------------------------------------------------------------------

/// A handler for server-initiated `sampling/createMessage` requests: the
/// server asks the client to run a completion (typically via whatever LLM
/// the *client* has access to, not the server) and return the result.
///
/// Register one with [`McpClient::sampling_handler`](crate::McpClient::sampling_handler)
/// or the equivalent builder method on [`McpStdioTool`](crate::McpStdioTool) /
/// [`McpStreamableHttpTool`](crate::McpStreamableHttpTool) /
/// [`McpWebsocketTool`](crate::McpWebsocketTool). See
/// [`chat_client_sampling_handler`] for a ready-made adapter backed by any
/// [`ChatClient`].
pub type SamplingHandler =
    Arc<dyn Fn(CreateMessageParams) -> BoxFuture<Result<CreateMessageResult>> + Send + Sync>;

/// One message in a `sampling/createMessage` request's `messages` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingMessage {
    /// `"user"` or `"assistant"`.
    pub role: String,
    /// A single content block — `{"type": "text", ...}`, `{"type": "image", ...}`,
    /// or `{"type": "audio", ...}` per the MCP spec. Unlike `tools/call`,
    /// sampling messages carry exactly one content item, not an array.
    pub content: Value,
}

impl SamplingMessage {
    /// Parse this message's raw content into a [`crate::ContentBlock`].
    pub fn content_block(&self) -> crate::protocol::ContentBlock {
        crate::protocol::ContentBlock::from_value(&self.content)
    }
}

/// Parameters for a server-initiated `sampling/createMessage` request.
///
/// Models the MCP 2025-06-18 baseline this crate speaks (see
/// [`crate::PROTOCOL_VERSION`]): `messages`, `modelPreferences`,
/// `systemPrompt`, `includeContext`, `temperature`, `maxTokens`,
/// `stopSequences`, `metadata`. The `tools`/`toolChoice` fields later SDK
/// revisions add for tool-use during sampling are out of scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateMessageParams {
    pub messages: Vec<SamplingMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_preferences: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_context: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Required per spec (`CreateMessageRequestParams.maxTokens: int`, no
    /// default): a request missing it fails to deserialize and is answered
    /// with a JSON-RPC "invalid params" error rather than silently
    /// proceeding without a token budget.
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// The result of a `sampling/createMessage` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateMessageResult {
    pub role: String,
    /// A single content block, same shape as [`SamplingMessage::content`].
    pub content: Value,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

impl CreateMessageResult {
    /// Build a text result — the common case.
    pub fn text(
        role: impl Into<String>,
        text: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            role: role.into(),
            content: json!({ "type": "text", "text": text.into() }),
            model: model.into(),
            stop_reason: None,
        }
    }
}

/// Decides whether a server's `sampling/createMessage` request may run.
pub type SamplingApproval = Arc<dyn Fn(&CreateMessageParams) -> BoxFuture<bool> + Send + Sync>;

/// The safety limits [`chat_client_sampling_handler_with`] applies before
/// forwarding a server's sampling request to a chat client. Mirrors
/// upstream's `sampling_approval_callback` / `sampling_max_tokens` /
/// `sampling_max_requests`.
///
/// An MCP server is an untrusted third party, and forwarding its prompts to
/// your model unreviewed is a confused-deputy risk, so the default **denies
/// every request**: set [`approval`](Self::approval) — or
/// [`approve_all`](Self::approve_all) as an explicit opt-in — to allow any.
/// Requests are then rate limited per handler (one handler serves one
/// session) and their `maxTokens` capped.
#[derive(Clone)]
pub struct SamplingGuard {
    approval: Option<SamplingApproval>,
    max_tokens: Option<u32>,
    max_requests: Option<u32>,
}

impl Default for SamplingGuard {
    fn default() -> Self {
        Self {
            approval: None,
            max_tokens: Some(4096),
            max_requests: Some(25),
        }
    }
}

impl SamplingGuard {
    /// Approve a request when `approval` resolves to `true`.
    pub fn approval<F, Fut>(mut self, approval: F) -> Self
    where
        F: Fn(&CreateMessageParams) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = bool> + Send + 'static,
    {
        self.approval = Some(Arc::new(move |p| Box::pin(approval(p))));
        self
    }

    /// Approve every request (explicit opt-in to unreviewed sampling).
    pub fn approve_all(self) -> Self {
        self.approval(|_| async { true })
    }

    /// Cap `maxTokens` at `cap` (default 4096); `None` removes the cap.
    pub fn max_tokens(mut self, cap: Option<u32>) -> Self {
        self.max_tokens = cap;
        self
    }

    /// Refuse requests past `limit` (default 25); `None` removes the limit.
    pub fn max_requests(mut self, limit: Option<u32>) -> Self {
        self.max_requests = limit;
        self
    }
}

/// [`chat_client_sampling_handler_with`] under the default
/// [`SamplingGuard`] — which **denies every request** until an approval is
/// configured. This matches upstream, where sampling without a
/// `sampling_approval_callback` is refused.
pub fn chat_client_sampling_handler(client: Arc<dyn ChatClient>) -> SamplingHandler {
    chat_client_sampling_handler_with(client, SamplingGuard::default())
}

/// Build a [`SamplingHandler`] backed by any [`ChatClient`], guarded by
/// `guard`: the per-session limit is checked first, then the approval, then
/// `maxTokens` is capped — upstream's order. A refusal is an error the
/// server receives as a JSON-RPC error.
pub fn chat_client_sampling_handler_with(
    client: Arc<dyn ChatClient>,
    guard: SamplingGuard,
) -> SamplingHandler {
    let inner = unguarded_chat_client_sampling_handler(client);
    let count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    Arc::new(move |mut params: CreateMessageParams| {
        let inner = inner.clone();
        let guard = guard.clone();
        let count = count.clone();
        Box::pin(async move {
            tracing::warn!(
                messages = params.messages.len(),
                max_tokens = params.max_tokens,
                "MCP server sent a sampling/createMessage request"
            );
            if let Some(limit) = guard.max_requests {
                let used = count.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                if used >= limit {
                    return Err(Error::service(
                        "Sampling rate limit exceeded for this MCP session.",
                    ));
                }
            }
            let approved = match &guard.approval {
                None => {
                    return Err(Error::service(
                        "Sampling request denied. MCP sampling is disabled by default for \
                         untrusted servers; configure a SamplingGuard approval that approves \
                         the request to enable it.",
                    ))
                }
                Some(approve) => approve(&params).await,
            };
            if !approved {
                return Err(Error::service(
                    "Sampling request denied by the SamplingGuard approval.",
                ));
            }
            if let Some(cap) = guard.max_tokens {
                if params.max_tokens > cap {
                    tracing::warn!(
                        requested = params.max_tokens,
                        cap,
                        "capping MCP sampling maxTokens"
                    );
                    params.max_tokens = cap;
                }
            }
            inner(params).await
        })
    })
}

/// Build a [`SamplingHandler`] backed by any [`ChatClient`]: converts the
/// server's `sampling/createMessage` request into core [`Message`]s
/// (`messages`/`systemPrompt`/`maxTokens`/`temperature`/`stopSequences` map
/// onto [`ChatOptions`]), calls [`ChatClient::get_response`], and maps the
/// reply back into a [`CreateMessageResult`].
///
/// Mirrors the Python reference's `MCPTool.sampling_callback`: prefers the
/// first text (or, failing that, image/audio) content item of the
/// response's first message, and never sets `stopReason` — Python's own
/// callback doesn't either, relying on the field's `None` default.
fn unguarded_chat_client_sampling_handler(client: Arc<dyn ChatClient>) -> SamplingHandler {
    Arc::new(move |params: CreateMessageParams| {
        let client = client.clone();
        Box::pin(async move {
            let messages: Vec<Message> = params
                .messages
                .iter()
                .map(|m| role_and_content_to_chat_message(&m.role, &m.content))
                .collect();

            let mut options = ChatOptions::new();
            options.instructions = params.system_prompt.clone();
            options.temperature = params.temperature;
            options.max_tokens = Some(params.max_tokens);
            options.stop = params.stop_sequences.clone();

            let response = client.get_response(messages, options).await.map_err(|e| {
                Error::service(format!("sampling handler: chat client call failed: {e}"))
            })?;

            let model = response
                .model
                .clone()
                .or_else(|| client.model().map(str::to_string))
                .unwrap_or_else(|| "unknown".to_string());
            let content = first_sampling_result_content(&response)?;

            Ok(CreateMessageResult {
                role: "assistant".to_string(),
                content,
                model,
                stop_reason: None,
            })
        })
    })
}

/// Extract a result content block (preferring text, falling back to
/// image/audio) from a chat response's messages, in order — mirrors
/// Python's `next(content for content in mcp_contents if isinstance(content, (TextContent, ImageContent)))`.
fn first_sampling_result_content(response: &ChatResponse) -> Result<Value> {
    for message in &response.messages {
        for content in &message.contents {
            match content {
                Content::Text(t) => return Ok(json!({ "type": "text", "text": t.text })),
                Content::Data(d) => {
                    if let Some((kind, mime, data)) = image_or_audio_block(d) {
                        return Ok(json!({ "type": kind, "data": data, "mimeType": mime }));
                    }
                }
                _ => {}
            }
        }
    }
    Err(Error::service(
        "sampling handler: chat client response had no text or image/audio content to return",
    ))
}

/// If `data` carries an `image/*` or `audio/*` media type, split its
/// `data:<mime>;base64,<payload>` URI back into `(kind, mime, payload)`.
fn image_or_audio_block(data: &DataContent) -> Option<(&'static str, String, &str)> {
    let mime = data.media_type.clone()?;
    let kind = if mime.starts_with("image/") {
        "image"
    } else if mime.starts_with("audio/") {
        "audio"
    } else {
        return None;
    };
    let payload = data
        .uri
        .split_once(";base64,")
        .map(|(_, payload)| payload)?;
    Some((kind, mime, payload))
}

// ---------------------------------------------------------------------
// roots/list
// ---------------------------------------------------------------------

/// A filesystem root the client exposes to the server via `roots/list`.
///
/// Mirrors the MCP `Root` type: a `file://` URI plus an optional
/// human-readable name. Register a static list with
/// [`McpClient::roots`](crate::McpClient::roots) or the equivalent
/// tool-wrapper builder method; the `roots` capability is then advertised
/// during `initialize`, and a `roots/list` request from the server is
/// answered from this list.
///
/// This crate only supports a *static* root list: there is no
/// `notifications/roots/list_changed` support, so `listChanged` is always
/// advertised as `false` — unlike the `mcp` Python SDK, which (per its own
/// `# TODO` comment questioning it) advertises `listChanged: true`
/// unconditionally whenever any roots callback is registered, regardless of
/// whether it actually sends that notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Root {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Root {
    /// Build a root from a `file://` URI.
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            name: None,
        }
    }

    /// Set the root's human-readable name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

// ---------------------------------------------------------------------
// Server-request dispatch (shared by all three transports)
// ---------------------------------------------------------------------

/// The handlers [`crate::McpClient`] currently has registered.
#[derive(Default)]
pub(crate) struct ServerRequestHandlers {
    pub(crate) sampling: Option<SamplingHandler>,
    pub(crate) roots: Option<Vec<Root>>,
}

/// The transport-facing callback installed via
/// [`crate::transport::McpTransport::set_server_request_handler`]: computes
/// the JSON-RPC response (success value or [`RpcError`]) for one
/// server-initiated request. Transports write the resulting envelope back
/// over the wire themselves.
pub type BoxedServerRequestHandler =
    Arc<dyn Fn(String, Value) -> BoxFuture<std::result::Result<Value, RpcError>> + Send + Sync>;

// ---------------------------------------------------------------------
// Notification dispatch (shared by all three transports)
// ---------------------------------------------------------------------

/// The transport-facing callback installed via
/// [`crate::transport::McpTransport::set_notification_handler`]: invoked for
/// every notification received from the server (no `id`, no response
/// expected/sent). [`crate::McpClient`] installs one automatically at
/// construction time to invalidate its cached `tools/list`/`prompts/list`
/// results on `notifications/tools/list_changed` /
/// `notifications/prompts/list_changed` — see
/// [`crate::McpClient::list_tools_cached`] /
/// [`crate::McpClient::list_prompts_cached`]. Unlike
/// [`BoxedServerRequestHandler`], there is nothing to write back, so this
/// returns `()` rather than a `Result`.
pub type BoxedNotificationHandler = Arc<dyn Fn(String, Value) -> BoxFuture<()> + Send + Sync>;

fn method_not_found(method: &str) -> RpcError {
    RpcError {
        code: METHOD_NOT_FOUND,
        message: format!("Method not found: {method}"),
        data: None,
    }
}

/// Compute the JSON-RPC response for one server-initiated request, given the
/// handlers currently registered on [`crate::McpClient`]. See the module
/// docs for the exact dispatch order/semantics.
pub(crate) async fn dispatch_server_request(
    handlers: &Arc<StdRwLock<ServerRequestHandlers>>,
    method: &str,
    params: Value,
) -> std::result::Result<Value, RpcError> {
    match method {
        "ping" => Ok(json!({})),
        "sampling/createMessage" => {
            let handler = handlers.read().unwrap().sampling.clone();
            let Some(handler) = handler else {
                return Err(method_not_found(method));
            };
            let params: CreateMessageParams =
                serde_json::from_value(params).map_err(|e| RpcError {
                    code: INVALID_PARAMS,
                    message: format!("invalid sampling/createMessage params: {e}"),
                    data: None,
                })?;
            let result = handler(params).await.map_err(|e| RpcError {
                code: INTERNAL_ERROR,
                message: e.to_string(),
                data: None,
            })?;
            serde_json::to_value(result).map_err(|e| RpcError {
                code: INTERNAL_ERROR,
                message: format!("failed to encode sampling/createMessage result: {e}"),
                data: None,
            })
        }
        "roots/list" => {
            let roots = handlers.read().unwrap().roots.clone();
            match roots {
                Some(roots) => Ok(json!({ "roots": roots })),
                None => Err(method_not_found(method)),
            }
        }
        other => Err(method_not_found(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::Role;
    use async_trait::async_trait;

    fn handlers(
        sampling: Option<SamplingHandler>,
        roots: Option<Vec<Root>>,
    ) -> Arc<StdRwLock<ServerRequestHandlers>> {
        Arc::new(StdRwLock::new(ServerRequestHandlers { sampling, roots }))
    }

    #[tokio::test]
    async fn dispatch_ping_always_answers_empty_result_regardless_of_handlers() {
        let h = handlers(None, None);
        let result = dispatch_server_request(&h, "ping", Value::Null)
            .await
            .unwrap();
        assert_eq!(result, json!({}));
    }

    #[tokio::test]
    async fn dispatch_unknown_method_is_method_not_found_error() {
        let h = handlers(None, None);
        let err = dispatch_server_request(&h, "elicitation/create", json!({}))
            .await
            .unwrap_err();
        assert_eq!(err.code, METHOD_NOT_FOUND);
        assert!(err.message.contains("elicitation/create"));
    }

    #[tokio::test]
    async fn dispatch_sampling_without_handler_is_method_not_found() {
        let h = handlers(None, None);
        let err = dispatch_server_request(&h, "sampling/createMessage", json!({}))
            .await
            .unwrap_err();
        assert_eq!(err.code, METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn dispatch_sampling_with_handler_returns_its_result() {
        let handler: SamplingHandler = Arc::new(|params: CreateMessageParams| {
            Box::pin(async move {
                assert_eq!(params.max_tokens, 50);
                Ok(CreateMessageResult::text(
                    "assistant",
                    "hi there",
                    "test-model",
                ))
            })
        });
        let h = handlers(Some(handler), None);
        let params = json!({
            "messages": [{"role": "user", "content": {"type": "text", "text": "hello"}}],
            "maxTokens": 50,
        });
        let result = dispatch_server_request(&h, "sampling/createMessage", params)
            .await
            .unwrap();
        assert_eq!(result["content"]["text"], "hi there");
        assert_eq!(result["model"], "test-model");
    }

    #[tokio::test]
    async fn dispatch_sampling_invalid_params_is_invalid_params_error() {
        let handler: SamplingHandler =
            Arc::new(|_| Box::pin(async { unreachable!("handler should not run") }));
        let h = handlers(Some(handler), None);
        // Missing the required `maxTokens` field.
        let params = json!({"messages": []});
        let err = dispatch_server_request(&h, "sampling/createMessage", params)
            .await
            .unwrap_err();
        assert_eq!(err.code, INVALID_PARAMS);
    }

    #[tokio::test]
    async fn dispatch_sampling_handler_error_is_internal_error() {
        let handler: SamplingHandler =
            Arc::new(|_| Box::pin(async { Err(Error::service("boom")) }));
        let h = handlers(Some(handler), None);
        let params = json!({"messages": [], "maxTokens": 10});
        let err = dispatch_server_request(&h, "sampling/createMessage", params)
            .await
            .unwrap_err();
        assert_eq!(err.code, INTERNAL_ERROR);
        assert!(err.message.contains("boom"));
    }

    #[tokio::test]
    async fn dispatch_roots_without_registration_is_method_not_found() {
        let h = handlers(None, None);
        let err = dispatch_server_request(&h, "roots/list", Value::Null)
            .await
            .unwrap_err();
        assert_eq!(err.code, METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn dispatch_roots_with_registration_returns_them() {
        let h = handlers(None, Some(vec![Root::new("file:///tmp").with_name("Temp")]));
        let result = dispatch_server_request(&h, "roots/list", Value::Null)
            .await
            .unwrap();
        assert_eq!(result["roots"][0]["uri"], "file:///tmp");
        assert_eq!(result["roots"][0]["name"], "Temp");
    }

    /// A stub `ChatClient` returning a canned response, for exercising
    /// [`chat_client_sampling_handler`] without any real provider.
    struct StubChatClient {
        text: &'static str,
        model: &'static str,
    }

    #[async_trait]
    impl ChatClient for StubChatClient {
        async fn get_response(
            &self,
            messages: Vec<Message>,
            options: ChatOptions,
        ) -> Result<ChatResponse> {
            // Exercise that the adapter actually forwards the mapped fields.
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0].role, Role::user());
            assert_eq!(options.max_tokens, Some(64));
            assert_eq!(options.instructions.as_deref(), Some("be terse"));
            Ok(ChatResponse {
                model: Some(self.model.to_string()),
                ..ChatResponse::from_text(self.text)
            })
        }
        async fn get_streaming_response(
            &self,
            _messages: Vec<Message>,
            _options: ChatOptions,
        ) -> Result<agent_framework_core::client::ChatStream> {
            unreachable!("not exercised by these tests")
        }
        fn model(&self) -> Option<&str> {
            Some(self.model)
        }
    }

    #[tokio::test]
    async fn chat_client_sampling_handler_round_trips_a_text_response() {
        let client: Arc<dyn ChatClient> = Arc::new(StubChatClient {
            text: "Paris is the capital of France.",
            model: "stub-model",
        });
        let handler =
            chat_client_sampling_handler_with(client, SamplingGuard::default().approve_all());
        let params = CreateMessageParams {
            messages: vec![SamplingMessage {
                role: "user".to_string(),
                content: json!({"type": "text", "text": "What is the capital of France?"}),
            }],
            model_preferences: None,
            system_prompt: Some("be terse".to_string()),
            include_context: None,
            temperature: Some(0.2),
            max_tokens: 64,
            stop_sequences: None,
            metadata: None,
        };
        let result = handler(params).await.unwrap();
        assert_eq!(result.role, "assistant");
        assert_eq!(result.content["type"], "text");
        assert_eq!(result.content["text"], "Paris is the capital of France.");
        assert_eq!(result.model, "stub-model");
        assert!(result.stop_reason.is_none());
    }

    /// Records the `max_tokens` each forwarded request carried.
    struct CountingClient(std::sync::Mutex<Vec<Option<u32>>>);

    #[async_trait]
    impl ChatClient for CountingClient {
        async fn get_response(&self, _m: Vec<Message>, o: ChatOptions) -> Result<ChatResponse> {
            self.0.lock().unwrap().push(o.max_tokens);
            Ok(ChatResponse::from_text("ok"))
        }
        async fn get_streaming_response(
            &self,
            _m: Vec<Message>,
            _o: ChatOptions,
        ) -> Result<agent_framework_core::client::ChatStream> {
            unreachable!()
        }
    }

    fn request(max_tokens: u32) -> CreateMessageParams {
        CreateMessageParams {
            messages: vec![SamplingMessage {
                role: "user".to_string(),
                content: json!({"type": "text", "text": "hi"}),
            }],
            model_preferences: None,
            system_prompt: None,
            include_context: None,
            temperature: None,
            max_tokens,
            stop_sequences: None,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn the_default_guard_denies_without_reaching_the_model() {
        let client = Arc::new(CountingClient(Default::default()));
        let handler = chat_client_sampling_handler(client.clone());
        let err = handler(request(10)).await.unwrap_err().to_string();
        assert!(err.contains("disabled by default"), "{err}");
        assert!(client.0.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn the_guard_caps_tokens_limits_requests_and_honours_refusals() {
        let client = Arc::new(CountingClient(Default::default()));
        let handler = chat_client_sampling_handler_with(
            client.clone(),
            SamplingGuard::default()
                .approval(|p: &CreateMessageParams| {
                    let ok = p.max_tokens != 7;
                    async move { ok }
                })
                .max_requests(Some(3)),
        );
        handler(request(10_000)).await.unwrap();
        handler(request(100)).await.unwrap();
        let refused = handler(request(7)).await.unwrap_err().to_string();
        assert!(refused.contains("denied by the SamplingGuard"), "{refused}");
        // The refused request still counted against the limit, as upstream.
        let limited = handler(request(100)).await.unwrap_err().to_string();
        assert!(limited.contains("rate limit"), "{limited}");
        assert_eq!(*client.0.lock().unwrap(), vec![Some(4096), Some(100)]);
    }

    #[tokio::test]
    async fn chat_client_sampling_handler_errors_when_no_usable_content() {
        struct EmptyClient;
        #[async_trait]
        impl ChatClient for EmptyClient {
            async fn get_response(
                &self,
                _messages: Vec<Message>,
                _options: ChatOptions,
            ) -> Result<ChatResponse> {
                Ok(ChatResponse::default())
            }
            async fn get_streaming_response(
                &self,
                _messages: Vec<Message>,
                _options: ChatOptions,
            ) -> Result<agent_framework_core::client::ChatStream> {
                unreachable!()
            }
        }
        let handler = chat_client_sampling_handler_with(
            Arc::new(EmptyClient),
            SamplingGuard::default().approve_all(),
        );
        let params = CreateMessageParams {
            messages: vec![SamplingMessage {
                role: "user".to_string(),
                content: json!({"type": "text", "text": "hi"}),
            }],
            model_preferences: None,
            system_prompt: None,
            include_context: None,
            temperature: None,
            max_tokens: 16,
            stop_sequences: None,
            metadata: None,
        };
        let err = handler(params).await.unwrap_err();
        assert!(matches!(err, Error::Service(_)));
    }

    #[test]
    fn root_builder_sets_name() {
        let root = Root::new("file:///workspace").with_name("Workspace");
        assert_eq!(root.uri, "file:///workspace");
        assert_eq!(root.name.as_deref(), Some("Workspace"));
        let value = serde_json::to_value(&root).unwrap();
        assert_eq!(
            value,
            json!({"uri": "file:///workspace", "name": "Workspace"})
        );
    }

    #[test]
    fn create_message_result_text_helper() {
        let result = CreateMessageResult::text("assistant", "hi", "m1");
        assert_eq!(result.content, json!({"type": "text", "text": "hi"}));
    }
}
