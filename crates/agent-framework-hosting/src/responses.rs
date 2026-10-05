//! Reusable OpenAI-Responses request/response conversion.
//!
//! Framework-agnostic OpenAI-Responses-shape types plus the two conversion
//! functions that translate between them and `agent-framework-core` types:
//! [`responses_to_run`] (request input → [`Message`]s) and
//! [`responses_from_run`] (a completed [`AgentResponse`] → [`ResponseObject`]).
//! Mirrors the Python `hosting-responses` package
//! (`responses_to_run`/`responses_from_run`; UPSTREAM_DRIFT.md §14): a small,
//! standalone conversion surface any host — not just [`crate::devui`] — can use
//! to speak the OpenAI Responses wire shape.
//!
//! A response's `output` is heterogeneous, as OpenAI's is: an assistant
//! message and a function call are sibling [`OutputItem`]s rather than a
//! message with calls attached, so a turn that declares client-side calls
//! carries each one's id, name and arguments for the caller to execute. A
//! call the agent answered itself is not one of those and is not
//! serialized. Coming back the other way, a `function_call_output` input
//! item becomes the `FunctionResultContent` that completes the round trip.
//!
//! [`crate::devui`] is the only current caller; it layers DevUI-specific
//! concerns (entity routing, the `~4-chars-per-token` usage estimate for runs
//! that report no usage, SSE framing) on top of this module.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use agent_framework_core::agent::AgentRunOptions;
use agent_framework_core::types::{
    AgentResponse, ChatOptions, Content, FinishReason, FunctionArguments, FunctionCallContent,
    FunctionResultContent, Message, Role, UsageDetails,
};

/// `POST /v1/responses` request — a subset of DevUI's `AgentFrameworkRequest`
/// (itself an OpenAI `ResponseCreateParams` superset). Unknown fields are
/// ignored.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ResponsesRequest {
    /// OpenAI `model`; accepted as an entity-id fallback.
    #[serde(default)]
    pub model: Option<String>,
    /// The user input: a string, an OpenAI input-items array, or a structured
    /// object (for workflows).
    #[serde(default)]
    pub input: Value,
    /// Whether to stream SSE. DevUI defaults this to `false`.
    #[serde(default)]
    pub stream: bool,
    /// Routing metadata; DevUI reads the entity id from `metadata.entity_id`.
    #[serde(default)]
    pub metadata: Option<Map<String, Value>>,
    /// Advanced routing; `extra_body.entity_id` is accepted as a fallback.
    #[serde(default)]
    pub extra_body: Option<Map<String, Value>>,
    /// Continue from an earlier response: its stored session is restored as
    /// a working copy, and the earlier snapshot is left untouched. Mutually
    /// exclusive with `conversation`. Kept as a raw value so a malformed one
    /// is reported, not silently dropped — see [`responses_session_id`].
    #[serde(default)]
    pub previous_response_id: Option<Value>,
    /// Continue a conversation: a `conv_*` id string or `{"id": ...}`. Unlike
    /// a response id this names a mutable head, advanced by every run.
    #[serde(default)]
    pub conversation: Option<Value>,
    /// Deprecated spelling of `conversation`, accepted only on its own.
    #[serde(default)]
    pub conversation_id: Option<Value>,
    /// Extra instructions for this run.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus-sampling mass.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Output-token cap; maps to `ChatOptions::max_tokens`.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Maps to `ChatOptions::allow_multiple_tool_calls`.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// End-user identifier.
    #[serde(default)]
    pub user: Option<String>,
}

/// Which continuation mechanism a request used. See
/// [`responses_session_id`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponsesContinuation {
    /// `previous_response_id`: an immutable snapshot to branch from.
    PreviousResponse(String),
    /// `conversation` (or the deprecated `conversation_id`): a mutable head.
    Conversation(String),
}

impl ResponsesContinuation {
    /// The id, whichever mechanism carried it.
    pub fn id(&self) -> &str {
        match self {
            Self::PreviousResponse(id) | Self::Conversation(id) => id,
        }
    }

    /// Whether this is a conversation id (upstream's `is_conversation_id`).
    pub fn is_conversation(&self) -> bool {
        matches!(self, Self::Conversation(_))
    }
}

/// The session a Responses request continues, if any. Mirrors upstream's
/// `responses_session_id`, which returns `(session_id, is_conversation_id)`.
///
/// `previous_response_id`, `conversation` and `conversation_id` are mutually
/// exclusive, and each must be a non-empty string (`conversation` may also be
/// `{"id": "..."}`). `conversation_id` is the deprecated spelling and is
/// honored only on its own. An id without the usual `resp_` / `conv_` prefix
/// is accepted, as upstream accepts it (with a warning there; a `tracing`
/// debug line here).
///
/// # Errors
/// A message suitable for a `400` when a field is malformed or two
/// mechanisms are combined.
pub fn responses_session_id(
    request: &ResponsesRequest,
) -> std::result::Result<Option<ResponsesContinuation>, String> {
    let present = |v: &Option<Value>| v.as_ref().is_some_and(|v| !v.is_null());
    let supplied = [
        present(&request.previous_response_id),
        present(&request.conversation),
        present(&request.conversation_id),
    ];
    if supplied.iter().filter(|s| **s).count() > 1 {
        return Err(
            "`previous_response_id`, `conversation`, and `conversation_id` are mutually exclusive"
                .into(),
        );
    }
    let non_empty = |v: Option<&Value>| {
        v.and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    if supplied[0] {
        let id = non_empty(request.previous_response_id.as_ref())
            .ok_or("`previous_response_id` must be a non-empty string")?;
        if !id.starts_with("resp_") {
            tracing::debug!(%id, "`previous_response_id` lacks the `resp_` prefix; continuing");
        }
        return Ok(Some(ResponsesContinuation::PreviousResponse(id)));
    }
    let id = if supplied[1] {
        let value = request.conversation.as_ref();
        let value = match value {
            Some(Value::Object(m)) => m.get("id"),
            other => other,
        };
        non_empty(value).ok_or(
            "`conversation` must be a non-empty string or an object with a non-empty string `id`",
        )?
    } else if supplied[2] {
        non_empty(request.conversation_id.as_ref())
            .ok_or("`conversation_id` must be a non-empty string")?
    } else {
        return Ok(None);
    };
    if !id.starts_with("conv_") {
        tracing::debug!(%id, "conversation id lacks the `conv_` prefix; continuing");
    }
    Ok(Some(ResponsesContinuation::Conversation(id)))
}

/// A Responses-shaped response id: `resp_` and 32 hex digits. Mirrors
/// upstream's `create_response_id`. The full UUID matters: these ids key
/// stored sessions, so a short one that collides would hand one caller
/// another's conversation.
pub fn create_response_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

/// A Responses-shaped conversation id: `conv_` and 32 hex digits. Mirrors
/// upstream's `create_conversation_id`.
pub fn create_conversation_id() -> String {
    format!("conv_{}", uuid::Uuid::new_v4().simple())
}

/// The per-run options a Responses request carries, as [`AgentRunOptions`].
///
/// The other half of upstream's `responses_to_run`, which returns options
/// alongside the messages: `max_output_tokens` becomes `max_tokens` and
/// `parallel_tool_calls` becomes `allow_multiple_tool_calls`, as there.
/// Transport fields (`input`, `stream`, the continuation ids) and this
/// crate's routing fields (`model`, `metadata`, `extra_body`) are not
/// options and are not copied.
pub fn responses_run_options(request: &ResponsesRequest) -> AgentRunOptions {
    let any = request.instructions.is_some()
        || request.temperature.is_some()
        || request.top_p.is_some()
        || request.max_output_tokens.is_some()
        || request.parallel_tool_calls.is_some()
        || request.user.is_some();
    if !any {
        // Empty options keep agents that do not support per-run options
        // from warning about options nobody sent.
        return AgentRunOptions::default();
    }
    AgentRunOptions::default().with_chat_options(ChatOptions {
        instructions: request.instructions.clone(),
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_output_tokens,
        allow_multiple_tool_calls: request.parallel_tool_calls,
        user: request.user.clone(),
        ..Default::default()
    })
}

/// The `conversation` field of a response object: `{"id": "conv_..."}`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ConversationRef {
    pub id: String,
}

impl ResponsesRequest {
    /// Resolve the target entity id.
    ///
    /// `metadata.entity_id`/`extra_body.entity_id`/`model`-as-fallback is
    /// [`crate::devui`]'s routing convention, not part of the OpenAI Responses
    /// wire shape; it lives here only because it is an inherent method on
    /// [`ResponsesRequest`], which this module owns.
    pub fn entity_id(&self) -> Option<String> {
        fn get<'a>(m: &'a Option<Map<String, Value>>, k: &str) -> Option<&'a str> {
            m.as_ref().and_then(|m| m.get(k)).and_then(Value::as_str)
        }
        get(&self.metadata, "entity_id")
            .or_else(|| get(&self.extra_body, "entity_id"))
            .map(str::to_string)
            .or_else(|| self.model.clone())
    }
}

/// Per-response token usage — mirrors OpenAI `ResponseUsage`.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens_details: OutputTokensDetails,
}

#[derive(Debug, Clone, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u64,
}

/// A single `output_text` content part — mirrors OpenAI `ResponseOutputText`.
#[derive(Debug, Clone, Serialize)]
pub struct OutputText {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub text: String,
    pub annotations: Vec<Value>,
}

impl OutputText {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content_type: "output_text",
            text: text.into(),
            annotations: Vec::new(),
        }
    }
}

/// Assemble a response's `output`: the assistant message when there is text
/// to carry, then one item per declared function call.
///
/// A turn that is only a call gets no message item. OpenAI omits it, and an
/// empty assistant message would read to a client as a blank answer rather
/// than as work to do.
pub(crate) fn output_items(
    text: &str,
    calls: &[&agent_framework_core::types::FunctionCallContent],
    message_id: String,
    status: &'static str,
) -> Vec<OutputItem> {
    let mut items = Vec::with_capacity(calls.len() + 1);
    if !text.is_empty() || calls.is_empty() {
        items.push(OutputItem::Message(
            OutputMessage::assistant_text(message_id, text).with_status(status),
        ));
    }
    items.extend(
        calls
            .iter()
            .map(|c| OutputItem::FunctionCall(OutputFunctionCall::new(c))),
    );
    items
}

/// One entry of a response's `output` array.
///
/// OpenAI's Responses output is heterogeneous: an assistant message and a
/// function call are *sibling items*, not a message with calls attached (the
/// shape Chat Completions uses). Untagged, because each variant already
/// carries its own `type` discriminator.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum OutputItem {
    Message(OutputMessage),
    FunctionCall(OutputFunctionCall),
}

impl OutputItem {
    /// This item as an assistant message, or `None` if it is a call.
    pub fn as_message(&self) -> Option<&OutputMessage> {
        match self {
            Self::Message(message) => Some(message),
            Self::FunctionCall(_) => None,
        }
    }

    /// This item as a function call, or `None` if it is a message.
    pub fn as_function_call(&self) -> Option<&OutputFunctionCall> {
        match self {
            Self::FunctionCall(call) => Some(call),
            Self::Message(_) => None,
        }
    }
}

impl From<OutputMessage> for OutputItem {
    fn from(message: OutputMessage) -> Self {
        Self::Message(message)
    }
}

/// A function call the caller is expected to execute — mirrors OpenAI
/// `ResponseFunctionToolCall`.
#[derive(Debug, Clone, Serialize)]
pub struct OutputFunctionCall {
    #[serde(rename = "type")]
    pub item_type: &'static str,
    pub id: String,
    /// The provider's id for the call, which the caller echoes back on the
    /// result. Distinct from `id`, which names this output item.
    pub call_id: String,
    pub name: String,
    /// A JSON **string**, as the wire format has it — not an object.
    pub arguments: String,
    pub status: &'static str,
}

impl OutputFunctionCall {
    pub fn new(call: &agent_framework_core::types::FunctionCallContent) -> Self {
        Self {
            item_type: "function_call",
            id: crate::util::msg_id(),
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            arguments: crate::util::arguments_string(call),
            status: "completed",
        }
    }
}

/// An assistant message output item — mirrors OpenAI `ResponseOutputMessage`.
#[derive(Debug, Clone, Serialize)]
pub struct OutputMessage {
    #[serde(rename = "type")]
    pub item_type: &'static str,
    pub id: String,
    pub role: &'static str,
    pub content: Vec<OutputText>,
    pub status: &'static str,
}

impl OutputMessage {
    pub fn assistant_text(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            item_type: "message",
            id: id.into(),
            role: "assistant",
            content: vec![OutputText::new(text)],
            status: "completed",
        }
    }

    /// Mark this item with the status of the response carrying it.
    ///
    /// An item's status has to agree with its response's: a `completed`
    /// message inside an `incomplete` response tells a client that inspects
    /// item status the opposite of what the response says, and the item is
    /// the more specific of the two. [`Self::assistant_text`] builds a
    /// completed item because that is the common case; this is how a
    /// truncated or filtered one says so.
    pub fn with_status(mut self, status: &'static str) -> Self {
        self.status = status;
        self
    }
}

/// Why a response stopped short of a complete answer — mirrors OpenAI
/// `Response.incomplete_details`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IncompleteDetails {
    pub reason: String,
}

/// Whether a run ended abnormally — the question `status` answers.
///
/// Completion is an **allowlist**: an absent reason, `stop`, `tool_calls`
/// and its deprecated predecessor `function_call`. `stop` is a complete
/// answer and the two tool reasons are a turn that continues;
/// every other reason is something a provider went out of its way to report.
/// `FinishReason` is an open string and providers use it — Anthropic's
/// converter deliberately preserves `model_context_window_exceeded` — so
/// treating the unfamiliar ones as completions is the false success this
/// whole area exists to remove.
pub fn is_incomplete(finish_reason: Option<&FinishReason>) -> bool {
    !matches!(
        finish_reason.map(FinishReason::as_str),
        None | Some(FinishReason::STOP)
            | Some(FinishReason::TOOL_CALLS)
            | Some(FinishReason::FUNCTION_CALL)
    )
}

/// The OpenAI-Responses `incomplete_details.reason` for a run, when the
/// schema has a name for it.
///
/// This is deliberately **narrower** than [`is_incomplete`]. The Responses
/// schema admits exactly two values here, and a generated client whose enum
/// is strict can reject an entire response over a third — so an unfamiliar
/// provider reason must not be smuggled into this field, and must not be
/// relabelled as one of the two either, since a wrong familiar value is
/// worse than an absent one. Such a run is still reported `incomplete`; its
/// raw reason travels in [`ResponseObject::x_finish_reason`], which is an
/// extension a strict client ignores rather than a value it must parse.
///
/// The naming differs from the core vocabulary on one of the two: a
/// token-budget cut-off is `max_output_tokens` here, while the
/// chat-completions `finish_reason` for the same event is `length`.
pub fn incomplete_reason(finish_reason: Option<&FinishReason>) -> Option<&'static str> {
    match finish_reason?.as_str() {
        FinishReason::CONTENT_FILTER => Some("content_filter"),
        FinishReason::LENGTH => Some("max_output_tokens"),
        _ => None,
    }
}

/// The aggregated final response — mirrors OpenAI `Response`.
///
/// Divergences from DevUI: adds a convenience top-level `output_text` (the
/// aggregated assistant text); for workflow entities adds `outputs` (the raw
/// workflow output values) and `pending_requests` (outstanding request-info
/// entries), since DevUI's text-only aggregation would otherwise drop them.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: f64,
    pub model: String,
    pub status: &'static str,
    /// Present only when `status` is `"incomplete"` *and* the Responses
    /// schema has a name for the reason, as OpenAI has it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    /// **Extension, not OpenAI.** The provider's own finish reason, verbatim,
    /// whenever a run ended abnormally. It exists because
    /// `incomplete_details.reason` is a two-value enum a strict client may
    /// police, while a provider reason like `model_context_window_exceeded`
    /// is worth keeping: an unknown *field* is ignored by such clients,
    /// whereas an unknown *enum value* can sink the whole response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_finish_reason: Option<String>,
    pub output: Vec<OutputItem>,
    /// The response this one continued, when the request named one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// The conversation this response belongs to, when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pending_requests: Vec<Value>,
    pub parallel_tool_calls: bool,
    pub tool_choice: &'static str,
    pub tools: Vec<Value>,
}

impl ResponseObject {
    /// A skeleton `in_progress` response used by the `response.created` /
    /// `response.in_progress` streaming events.
    pub fn in_progress(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "response",
            created_at: crate::util::now_ts(),
            model: model.into(),
            status: "in_progress",
            incomplete_details: None,
            x_finish_reason: None,
            output: Vec::new(),
            previous_response_id: None,
            conversation: None,
            output_text: None,
            usage: None,
            outputs: None,
            pending_requests: Vec::new(),
            parallel_tool_calls: false,
            tool_choice: "none",
            tools: Vec::new(),
        }
    }

    /// Record the continuation this response belongs to: `conversation`
    /// for a conversation, `previous_response_id` for a branch. Mirrors
    /// upstream `responses_from_run(..., conversation_id=...)`.
    pub fn with_continuation(mut self, continuation: Option<&ResponsesContinuation>) -> Self {
        match continuation {
            Some(ResponsesContinuation::Conversation(id)) => {
                self.conversation = Some(ConversationRef { id: id.clone() });
            }
            Some(ResponsesContinuation::PreviousResponse(id)) => {
                self.previous_response_id = Some(id.clone());
            }
            None => {}
        }
        self
    }
}

/// Build an OpenAI-style error body: `{"error": {"message", "type", "code"}}`.
///
/// Mirrors DevUI's `OpenAIError.create`.
pub fn openai_error(message: impl Into<String>, error_type: &str, code: Option<&str>) -> Value {
    serde_json::json!({
        "error": {
            "message": message.into(),
            "type": error_type,
            "code": code,
        }
    })
}

// ---------------------------------------------------------------------------
// Conversion: ResponsesRequest -> Vec<Message>
// ---------------------------------------------------------------------------

/// Parse a [`ResponsesRequest`]'s OpenAI-style `input` into chat messages for
/// an agent run. Mirrors `hosting-responses`' `responses_to_run`.
///
/// Accepts a bare string, an array of input items (OpenAI `{type:"message",
/// content:[…]}` or `{role, content}`), or falls back to a stringified value.
pub fn responses_to_run(request: &ResponsesRequest) -> Vec<Message> {
    input_to_messages(&request.input)
}

/// Shared with [`crate::devui`]'s workflow input handling, which needs to
/// convert a bare `input` value (not a full [`ResponsesRequest`]) to messages.
pub(crate) fn input_to_messages(input: &Value) -> Vec<Message> {
    match input {
        Value::String(s) => vec![Message::user(s.clone())],
        Value::Null => vec![Message::user(String::new())],
        Value::Array(items) => {
            let msgs: Vec<Message> = items.iter().filter_map(item_to_message).collect();
            if msgs.is_empty() {
                vec![Message::user(String::new())]
            } else {
                msgs
            }
        }
        obj @ Value::Object(_) => item_to_message(obj)
            .map(|m| vec![m])
            .unwrap_or_else(|| vec![Message::user(obj.to_string())]),
        other => vec![Message::user(other.to_string())],
    }
}

/// Convert one input item into a chat message, if it carries text — or a
/// function call or its result, which are items in their own right.
///
/// The Responses protocol does not put a tool result inside a message the
/// way Chat Completions does: a caller that executes a `function_call`
/// output item sends back a top-level `function_call_output` item carrying
/// `call_id` and `output`, with no `role` and no `content`. Read as a
/// message that would be an empty user turn, and the result — the entire
/// point of the round trip — would be lost. `function_call` is accepted on
/// the way in for the same reason: a caller replaying the turn it was
/// given must not have the call silently dropped from the conversation.
fn item_to_message(item: &Value) -> Option<Message> {
    match item {
        Value::String(s) => Some(Message::user(s.clone())),
        Value::Object(map) => {
            match map.get("type").and_then(Value::as_str) {
                Some("function_call_output") => {
                    let call_id = map.get("call_id").and_then(Value::as_str)?;
                    return Some(single(
                        Role::tool(),
                        Content::FunctionResult(FunctionResultContent::new(
                            call_id,
                            Some(function_result_value(map.get("output"))),
                        )),
                    ));
                }
                Some("function_call") => {
                    let call_id = map.get("call_id").and_then(Value::as_str)?;
                    return Some(single(
                        Role::assistant(),
                        Content::FunctionCall(FunctionCallContent::new(
                            call_id,
                            map.get("name").and_then(Value::as_str).unwrap_or_default(),
                            map.get("arguments")
                                .and_then(Value::as_str)
                                .map(|a| FunctionArguments::Raw(a.to_string())),
                        )),
                    ));
                }
                _ => {}
            }
            let role = map
                .get("role")
                .and_then(Value::as_str)
                .map(role_from)
                .unwrap_or_else(Role::user);
            let text = map.get("content").map(content_text).unwrap_or_default();
            Some(Message::new(role, text))
        }
        _ => None,
    }
}

fn single(role: Role, content: Content) -> Message {
    Message {
        role,
        contents: vec![content],
        author_name: None,
        message_id: None,
        additional_properties: Default::default(),
    }
}

/// A `function_call_output`'s `output` as a result value.
///
/// The wire field is typed as a string, and tools overwhelmingly return
/// JSON in it, so a string that parses is stored as the structured value it
/// represents and anything else stays a string — a tool expecting its own
/// shape back should not be handed an opaque blob.
fn function_result_value(output: Option<&Value>) -> Value {
    match output {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(other) => other.clone(),
        None => Value::String(String::new()),
    }
}

/// Extract text from an OpenAI `content` value (string or array of parts).
fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                p.get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| p.as_str().map(str::to_string))
            })
            .collect::<Vec<_>>()
            .join(""),
        other => other.to_string(),
    }
}

fn role_from(role: &str) -> Role {
    match role {
        "user" => Role::user(),
        "assistant" => Role::assistant(),
        "system" => Role::system(),
        "tool" => Role::tool(),
        other => Role::new(other),
    }
}

// ---------------------------------------------------------------------------
// Conversion: AgentResponse -> ResponseObject
// ---------------------------------------------------------------------------

/// Build the aggregated OpenAI-Responses output object for a completed
/// [`AgentResponse`]. Mirrors `hosting-responses`' `responses_from_run`.
///
/// Maps `resp`'s text into a single assistant `output`/`output_text`, and its
/// [`UsageDetails`], when present, into [`Usage`]. `id`/`model` are threaded
/// through as-is (callers, e.g. [`crate::devui`], own id generation and
/// entity-to-model resolution). When `resp` carries no usage details, `usage`
/// is `None`; callers that want a token-count estimate (as DevUI does) fill
/// one in afterward.
pub fn responses_from_run(resp: &AgentResponse, id: &str, model: &str) -> ResponseObject {
    let text = resp.text();
    let mid = crate::util::msg_id();
    // A turn the model was cut off in — an Azure OpenAI content-filter block,
    // or the token budget running out — reads as an ordinary finished answer
    // in its text alone. Reporting it as `completed` left a caller matching
    // the provider's canned refusal string as the only way to tell, so the
    // status and `incomplete_details` carry it instead.
    let calls = crate::util::function_calls_of(&resp.messages);
    let incomplete = is_incomplete(resp.finish_reason.as_ref());
    let status = if incomplete {
        "incomplete"
    } else {
        "completed"
    };
    // Only the two the schema names reach `incomplete_details`; the raw
    // reason rides the extension field so nothing is lost either way.
    let detail = incomplete_reason(resp.finish_reason.as_ref());
    let raw_reason = incomplete
        .then(|| resp.finish_reason.as_ref().map(|r| r.as_str().to_string()))
        .flatten();
    ResponseObject {
        id: id.to_string(),
        object: "response",
        created_at: crate::util::now_ts(),
        model: model.to_string(),
        status,
        incomplete_details: detail.map(|reason| IncompleteDetails {
            reason: reason.to_string(),
        }),
        x_finish_reason: raw_reason,
        // The item's status follows the response's: the two disagreeing is
        // worse than either being wrong alone.
        //
        // A declaration-only call is a sibling item, not an attachment to
        // the message, and a turn that is *only* a call carries no message
        // item at all — an empty assistant message would read as a blank
        // answer rather than as a call to execute.
        output: output_items(&text, &calls, mid, status),
        previous_response_id: None,
        conversation: None,
        output_text: Some(text),
        usage: resp.usage_details.as_ref().map(usage_from_details),
        outputs: None,
        pending_requests: Vec::new(),
        parallel_tool_calls: false,
        tool_choice: "none",
        tools: Vec::new(),
    }
}

/// Map a core [`UsageDetails`] onto the OpenAI-Responses [`Usage`] shape.
fn usage_from_details(u: &UsageDetails) -> Usage {
    let input = u.input_token_count.unwrap_or(0);
    let output = u.output_token_count.unwrap_or(0);
    let total = u.total_token_count.unwrap_or(input + output);
    Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        input_tokens_details: InputTokensDetails { cached_tokens: 0 },
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(input: Value) -> ResponsesRequest {
        ResponsesRequest {
            input,
            ..Default::default()
        }
    }

    #[test]
    fn entity_id_prefers_metadata_then_extra_body_then_model() {
        let mut r = req(Value::Null);
        r.model = Some("from-model".to_string());
        assert_eq!(r.entity_id(), Some("from-model".to_string()));

        let mut extra = Map::new();
        extra.insert(
            "entity_id".to_string(),
            Value::String("from-extra".to_string()),
        );
        r.extra_body = Some(extra);
        assert_eq!(r.entity_id(), Some("from-extra".to_string()));

        let mut meta = Map::new();
        meta.insert(
            "entity_id".to_string(),
            Value::String("from-meta".to_string()),
        );
        r.metadata = Some(meta);
        assert_eq!(r.entity_id(), Some("from-meta".to_string()));
    }

    fn run_finishing_with(reason: Option<FinishReason>) -> AgentResponse {
        AgentResponse {
            messages: vec![Message::assistant("here is what I can say")],
            finish_reason: reason,
            ..Default::default()
        }
    }

    /// The Responses schema names exactly two reasons. A provider's own
    /// string must still be reported — as `incomplete` status plus the
    /// extension field — but must not be smuggled into the enum, where a
    /// strict generated client could reject the whole response over it.
    #[test]
    fn an_unfamiliar_reason_is_incomplete_without_an_invalid_detail() {
        let obj = responses_from_run(
            &run_finishing_with(Some(FinishReason::new("model_context_window_exceeded"))),
            "resp_1",
            "claude",
        );
        assert_eq!(obj.status, "incomplete");
        assert!(
            obj.incomplete_details.is_none(),
            "the schema has no name for this reason, so the field stays absent"
        );
        assert_eq!(
            obj.x_finish_reason.as_deref(),
            Some("model_context_window_exceeded"),
            "but the reason itself is not lost"
        );
        // And it must serialize without an invalid enum value anywhere.
        let json = serde_json::to_value(&obj).unwrap();
        assert!(json.get("incomplete_details").is_none());
        assert_eq!(
            json["x_finish_reason"],
            serde_json::json!("model_context_window_exceeded")
        );
    }

    /// The two the schema does name keep using it.
    #[test]
    fn a_schema_named_reason_still_fills_incomplete_details() {
        for (reason, expected) in [
            (FinishReason::CONTENT_FILTER, "content_filter"),
            (FinishReason::LENGTH, "max_output_tokens"),
        ] {
            let obj = responses_from_run(
                &run_finishing_with(Some(FinishReason::new(reason))),
                "resp_1",
                "gpt-4o",
            );
            assert_eq!(obj.status, "incomplete");
            assert_eq!(
                obj.incomplete_details.as_ref().map(|d| d.reason.as_str()),
                Some(expected)
            );
            assert_eq!(obj.x_finish_reason.as_deref(), Some(reason));
        }
    }

    /// A completed turn carries neither.
    #[test]
    fn a_completed_turn_carries_no_finish_reason_extension() {
        let obj = responses_from_run(
            &run_finishing_with(Some(FinishReason::new(FinishReason::STOP))),
            "resp_1",
            "gpt-4o",
        );
        assert_eq!(obj.status, "completed");
        assert!(obj.incomplete_details.is_none());
        assert!(obj.x_finish_reason.is_none());
    }

    /// An item's status has to agree with its response's. A `completed`
    /// message inside an `incomplete` response tells a client that reads item
    /// status the opposite of what the response says.
    #[test]
    fn the_output_item_status_follows_the_response_status() {
        let cut_off = responses_from_run(
            &run_finishing_with(Some(FinishReason::new(FinishReason::LENGTH))),
            "resp_1",
            "gpt-4o",
        );
        assert_eq!(cut_off.status, "incomplete");
        assert_eq!(cut_off.output[0].as_message().unwrap().status, "incomplete");

        let finished = responses_from_run(
            &run_finishing_with(Some(FinishReason::new(FinishReason::STOP))),
            "resp_1",
            "gpt-4o",
        );
        assert_eq!(finished.status, "completed");
        assert_eq!(finished.output[0].as_message().unwrap().status, "completed");
    }

    /// The other half of that allowlist: the two reasons that really are
    /// completions must not start being reported as incomplete.
    #[test]
    fn stop_and_tool_calls_remain_completions() {
        // `function_call` is the deprecated spelling of `tool_calls` and,
        // like it, marks a turn that succeeded — reporting it as incomplete
        // told clients a working tool call had been cut off.
        for reason in [
            FinishReason::STOP,
            FinishReason::TOOL_CALLS,
            FinishReason::FUNCTION_CALL,
        ] {
            let obj = responses_from_run(
                &run_finishing_with(Some(FinishReason::new(reason))),
                "resp_1",
                "gpt-4o",
            );
            assert_eq!(obj.status, "completed", "{reason} is a completion");
            assert!(obj.incomplete_details.is_none());
        }
    }

    /// A turn the Azure OpenAI content filter cut off used to be reported as
    /// `completed`, leaving the canned refusal text as the only signal.
    #[test]
    fn a_content_filtered_run_is_reported_incomplete() {
        let obj = responses_from_run(
            &run_finishing_with(Some(FinishReason::new(FinishReason::CONTENT_FILTER))),
            "resp_1",
            "gpt-4o",
        );
        assert_eq!(obj.status, "incomplete");
        assert_eq!(
            obj.incomplete_details,
            Some(IncompleteDetails {
                reason: "content_filter".to_string()
            })
        );
    }

    /// `length` is OpenAI's `max_output_tokens` on this surface — the one
    /// place the two vocabularies disagree.
    #[test]
    fn a_length_capped_run_is_reported_as_max_output_tokens() {
        let obj = responses_from_run(
            &run_finishing_with(Some(FinishReason::new(FinishReason::LENGTH))),
            "resp_1",
            "gpt-4o",
        );
        assert_eq!(obj.status, "incomplete");
        assert_eq!(
            obj.incomplete_details,
            Some(IncompleteDetails {
                reason: "max_output_tokens".to_string()
            })
        );
    }

    /// The negative controls: a turn that genuinely ended, one that continues
    /// into tools, and one whose provider reported nothing are all complete —
    /// and none of them grows an `incomplete_details` key.
    #[test]
    fn a_run_that_was_not_cut_off_stays_completed() {
        for reason in [
            Some(FinishReason::stop()),
            Some(FinishReason::tool_calls()),
            None,
        ] {
            let obj = responses_from_run(&run_finishing_with(reason.clone()), "resp_1", "gpt-4o");
            assert_eq!(obj.status, "completed", "{reason:?}");
            assert_eq!(obj.incomplete_details, None, "{reason:?}");
            let encoded = serde_json::to_value(&obj).unwrap();
            assert!(encoded.get("incomplete_details").is_none(), "{reason:?}");
        }
    }

    #[test]
    fn responses_to_run_string_input() {
        let messages = responses_to_run(&req(Value::String("hello".to_string())));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::user());
        assert_eq!(messages[0].text(), "hello");
    }

    #[test]
    fn responses_to_run_array_input() {
        let input = serde_json::json!([
            { "role": "system", "content": "be terse" },
            { "role": "user", "content": [ { "type": "input_text", "text": "hi" } ] },
        ]);
        let messages = responses_to_run(&req(input));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::system());
        assert_eq!(messages[0].text(), "be terse");
        assert_eq!(messages[1].role, Role::user());
        assert_eq!(messages[1].text(), "hi");
    }

    #[test]
    fn responses_to_run_null_input_yields_empty_user_message() {
        let messages = responses_to_run(&req(Value::Null));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::user());
        assert_eq!(messages[0].text(), "");
    }

    #[test]
    fn responses_from_run_maps_text_and_usage() {
        let resp = AgentResponse {
            messages: vec![Message::assistant("hello there")],
            usage_details: Some(UsageDetails {
                input_token_count: Some(3),
                output_token_count: Some(5),
                total_token_count: Some(8),
                ..Default::default()
            }),
            ..Default::default()
        };
        let obj = responses_from_run(&resp, "resp_123", "my-agent");
        assert_eq!(obj.id, "resp_123");
        assert_eq!(obj.model, "my-agent");
        assert_eq!(obj.status, "completed");
        assert_eq!(obj.output_text.as_deref(), Some("hello there"));
        assert_eq!(obj.output.len(), 1);
        assert_eq!(
            obj.output[0].as_message().unwrap().content[0].text,
            "hello there"
        );
        let usage = obj.usage.expect("usage present");
        assert_eq!(usage.input_tokens, 3);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 8);
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
    }

    #[test]
    fn responses_from_run_without_usage_details_leaves_usage_none() {
        let resp = AgentResponse {
            messages: vec![Message::assistant("hi")],
            usage_details: None,
            ..Default::default()
        };
        let obj = responses_from_run(&resp, "resp_1", "m");
        assert!(obj.usage.is_none());
    }
}
