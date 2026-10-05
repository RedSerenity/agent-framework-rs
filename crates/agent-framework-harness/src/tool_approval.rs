//! Standing ("don't ask again") tool approvals and queued approval prompts.
//!
//! Rust equivalent of upstream `agent_framework._harness._tool_approval`
//! (`ToolApprovalMiddleware`, `ToolApprovalRule`, `ToolApprovalState`,
//! `create_always_approve_tool_response`, …) and of .NET's
//! `ToolApprovalAgent`.
//!
//! # Divergence: a decorator agent, not agent middleware
//!
//! Upstream implements this as an `AgentMiddleware` whose `call_next()`
//! re-runs the *whole* agent (context providers included) and which reads
//! `context.session`. A Rust [`AgentContext`](agent_framework_core::middleware::AgentContext)
//! carries no session and its pipeline runs *after* the context providers,
//! so re-invocation there could not reload history. [`ToolApprovalAgent`]
//! therefore wraps any [`SupportsAgentRun`] — the shape .NET's
//! `ToolApprovalAgent` decorator has — and re-runs the inner agent on the
//! same session.
//!
//! # Divergence: where the "always approve" scope travels
//!
//! Upstream marks a `function_approval_response` content with
//! `additional_properties["tool_approval"] = {"always_approve": scope}`.
//! Rust's [`FunctionApprovalResponseContent`] has no property bag, so the
//! scope rides on the carrying [`Message`]'s `additional_properties`, keyed
//! by response id: `{"tool_approval": {"<id>": {"always_approve": scope,
//! "reason": …}}}`. Use [`create_always_approve_tool_response`] /
//! [`create_always_approve_tool_with_arguments_response`] (or
//! [`mark_always_approve`]) rather than building it by hand.

use std::collections::HashSet;
use std::sync::Arc;

use agent_framework_core::agent::{AgentRunOptions, AgentRunStream, SupportsAgentRun};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::session::{AgentSession, SessionState};
use agent_framework_core::tools::BoxFuture;
use agent_framework_core::types::{
    AgentResponse, AgentResponseUpdate, Content, FunctionApprovalRequestContent,
    FunctionApprovalResponseContent, FunctionCallContent, Message, Role,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Default session-state key of [`ToolApprovalAgent`]. Mirrors upstream
/// `DEFAULT_TOOL_APPROVAL_SOURCE_ID`.
pub const DEFAULT_TOOL_APPROVAL_SOURCE_ID: &str = "tool_approval";
/// The message property carrying always-approve metadata. Mirrors upstream
/// `ALWAYS_APPROVE_PROPERTY`.
pub const ALWAYS_APPROVE_PROPERTY: &str = "tool_approval";
/// The metadata key naming the scope. Mirrors upstream
/// `ALWAYS_APPROVE_SCOPE_PROPERTY`.
pub const ALWAYS_APPROVE_SCOPE_PROPERTY: &str = "always_approve";
/// Scope value: approve every future call to the tool. Upstream
/// `ALWAYS_APPROVE_TOOL`.
pub const ALWAYS_APPROVE_TOOL: &str = "tool";
/// Scope value: approve future calls with exactly these arguments. Upstream
/// `ALWAYS_APPROVE_TOOL_WITH_ARGUMENTS`.
pub const ALWAYS_APPROVE_TOOL_WITH_ARGUMENTS: &str = "tool_with_arguments";

const RULES_KEY: &str = "rules";
const QUEUED_KEY: &str = "queued_approval_requests";
const COLLECTED_KEY: &str = "collected_approval_responses";
const PENDING_KEY: &str = "pending_approval_requests";

/// The scope of a standing approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolApprovalScope {
    /// Every future call to the tool.
    Tool,
    /// Future calls to the tool with exactly the same arguments.
    ToolWithArguments,
}

impl ToolApprovalScope {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tool => ALWAYS_APPROVE_TOOL,
            Self::ToolWithArguments => ALWAYS_APPROVE_TOOL_WITH_ARGUMENTS,
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            ALWAYS_APPROVE_TOOL => Some(Self::Tool),
            ALWAYS_APPROVE_TOOL_WITH_ARGUMENTS => Some(Self::ToolWithArguments),
            _ => None,
        }
    }
}

/// A heuristic callback that can auto-approve a function call that would
/// otherwise require approval. Mirrors upstream `ToolApprovalRuleCallback`
/// (sync or async there; build a sync one with [`approval_rule`]).
///
/// **Security:** a rule that matches by name approves *any* local tool with
/// that name, not just the one it was written for — make sure no unrelated
/// tool collides with a name a rule approves.
pub type ToolApprovalRuleCallback =
    Arc<dyn Fn(&FunctionCallContent) -> BoxFuture<bool> + Send + Sync>;

/// Wrap a synchronous predicate as a [`ToolApprovalRuleCallback`].
pub fn approval_rule<F>(f: F) -> ToolApprovalRuleCallback
where
    F: Fn(&FunctionCallContent) -> bool + Send + Sync + 'static,
{
    Arc::new(move |call| {
        let approved = f(call);
        Box::pin(async move { approved })
    })
}

/// A standing rule approving future matching tool calls. Mirrors upstream
/// `ToolApprovalRule`; serialized as `{"tool_name", "type":
/// "tool_approval_rule", "arguments"?, "server_label"?}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolApprovalRule {
    /// The function tool name the rule applies to.
    pub tool_name: String,
    /// Canonicalized arguments (`name → compact sorted JSON`); `None`
    /// approves every call, `Some({})` only no-argument calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<std::collections::BTreeMap<String, String>>,
    /// Hosted-tool server boundary. Rust function calls carry none, so rules
    /// created here always have `None`; kept for state-format parity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_label: Option<String>,
    #[serde(rename = "type", default = "rule_type", skip_deserializing)]
    kind: String,
}

fn rule_type() -> String {
    "tool_approval_rule".into()
}

impl ToolApprovalRule {
    /// A rule for `tool_name` (trimmed, must be non-empty).
    pub fn new(
        tool_name: &str,
        arguments: Option<std::collections::BTreeMap<String, String>>,
    ) -> Result<Self> {
        let name = tool_name.trim();
        if name.is_empty() {
            return Err(Error::Configuration(
                "Tool approval rule tool_name must be a non-empty string.".into(),
            ));
        }
        Ok(Self {
            tool_name: name.to_string(),
            arguments,
            server_label: None,
            kind: rule_type(),
        })
    }

    /// Whether `call` matches this rule.
    pub fn matches(&self, call: &FunctionCallContent) -> bool {
        if self.tool_name != call.name || self.server_label.is_some() {
            return false;
        }
        match &self.arguments {
            None => true,
            Some(expected) => &serialize_arguments(call) == expected,
        }
    }
}

/// Canonicalize a call's arguments for exact matching (each value rendered
/// as compact, key-sorted JSON). Mirrors upstream `_serialize_arguments`.
pub fn serialize_arguments(
    call: &FunctionCallContent,
) -> std::collections::BTreeMap<String, String> {
    call.parse_arguments()
        .unwrap_or_default()
        .into_iter()
        .map(|(k, v)| (k, serde_json::to_string(&v).unwrap_or_default()))
        .collect()
}

/// Session-backed state of [`ToolApprovalAgent`]. Mirrors upstream
/// `ToolApprovalState`, plus `pending_approval_requests` — the requests
/// surfaced to the caller, used to bind inbound responses (see
/// [`ToolApprovalAgent::disable_response_binding`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolApprovalState {
    /// Standing approval rules.
    pub rules: Vec<ToolApprovalRule>,
    /// Requests held back to be surfaced one at a time.
    pub queued_approval_requests: Vec<FunctionApprovalRequestContent>,
    /// Responses gathered but not yet sent to the inner agent.
    pub collected_approval_responses: Vec<FunctionApprovalResponseContent>,
    /// Requests surfaced to the caller and not yet answered.
    pub pending_approval_requests: Vec<FunctionApprovalRequestContent>,
}

fn parse_list<T: serde::de::DeserializeOwned>(
    map: &Map<String, Value>,
    key: &str,
) -> Result<Vec<T>> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| Error::Serialization(format!("invalid tool approval state '{key}': {e}"))),
    }
}

impl ToolApprovalState {
    /// Load from session state (initializing an empty entry when absent).
    pub fn load(state: &SessionState, source_id: &str) -> Result<Self> {
        match state.get(source_id) {
            None | Some(Value::Null) => {
                let empty = Self::default();
                empty.save(state, source_id);
                Ok(empty)
            }
            Some(Value::Object(map)) => Ok(Self {
                rules: parse_list(&map, RULES_KEY)?,
                queued_approval_requests: parse_list(&map, QUEUED_KEY)?,
                collected_approval_responses: parse_list(&map, COLLECTED_KEY)?,
                pending_approval_requests: parse_list(&map, PENDING_KEY)?,
            }),
            Some(other) => Err(Error::Serialization(format!(
                "Session state for '{source_id}' must be a mapping, got {}.",
                crate::util::json_type_name(&other)
            ))),
        }
    }

    /// Persist into session state, preserving unrelated keys a caller may
    /// have stored alongside (mirrors upstream `_save_state`).
    pub fn save(&self, state: &SessionState, source_id: &str) {
        let mut map = match state.get(source_id) {
            Some(Value::Object(map)) => map,
            _ => Map::new(),
        };
        map.insert(
            RULES_KEY.into(),
            serde_json::to_value(&self.rules).unwrap_or(json!([])),
        );
        map.insert(
            QUEUED_KEY.into(),
            serde_json::to_value(&self.queued_approval_requests).unwrap_or(json!([])),
        );
        map.insert(
            COLLECTED_KEY.into(),
            serde_json::to_value(&self.collected_approval_responses).unwrap_or(json!([])),
        );
        map.insert(
            PENDING_KEY.into(),
            serde_json::to_value(&self.pending_approval_requests).unwrap_or(json!([])),
        );
        state.insert(source_id, Value::Object(map));
    }

    fn add_rule_if_missing(&mut self, rule: ToolApprovalRule) {
        if !self.rules.iter().any(|r| {
            r.tool_name == rule.tool_name
                && r.server_label == rule.server_label
                && r.arguments == rule.arguments
        }) {
            self.rules.push(rule);
        }
    }

    fn matches_rule(&self, call: &FunctionCallContent) -> bool {
        self.rules.iter().any(|r| r.matches(call))
    }

    fn record_pending(
        &mut self,
        requests: impl IntoIterator<Item = FunctionApprovalRequestContent>,
    ) {
        for request in requests {
            if !self
                .pending_approval_requests
                .iter()
                .any(|p| p.id == request.id)
            {
                self.pending_approval_requests.push(request);
            }
        }
    }
}

/// Record on `message` that the approval response `response_id` it carries
/// should also create a standing rule of `scope`.
pub fn mark_always_approve(
    message: &mut Message,
    response_id: &str,
    scope: ToolApprovalScope,
    reason: Option<&str>,
) {
    let mut metadata = Map::new();
    metadata.insert(ALWAYS_APPROVE_SCOPE_PROPERTY.into(), json!(scope.as_str()));
    if let Some(reason) = reason {
        metadata.insert("reason".into(), json!(reason));
    }
    let entry = message
        .additional_properties
        .entry(ALWAYS_APPROVE_PROPERTY.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !entry.is_object() {
        *entry = Value::Object(Map::new());
    }
    if let Value::Object(map) = entry {
        map.insert(response_id.to_string(), Value::Object(metadata));
    }
}

fn always_approve_message(
    request: &FunctionApprovalRequestContent,
    scope: ToolApprovalScope,
    reason: Option<&str>,
) -> Message {
    let response = request.create_response(true);
    let mut message = Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(response)],
    );
    mark_always_approve(&mut message, &request.id, scope, reason);
    message
}

/// A user message approving `request` and recording a standing rule for the
/// whole tool. Mirrors upstream `create_always_approve_tool_response`.
pub fn create_always_approve_tool_response(
    request: &FunctionApprovalRequestContent,
    reason: Option<&str>,
) -> Message {
    always_approve_message(request, ToolApprovalScope::Tool, reason)
}

/// A user message approving `request` and recording a standing rule for the
/// tool with exactly these arguments. Mirrors upstream
/// `create_always_approve_tool_with_arguments_response`.
pub fn create_always_approve_tool_with_arguments_response(
    request: &FunctionApprovalRequestContent,
    reason: Option<&str>,
) -> Message {
    always_approve_message(request, ToolApprovalScope::ToolWithArguments, reason)
}

fn always_approve_scope(message: &Message, response_id: &str) -> Option<ToolApprovalScope> {
    message
        .additional_properties
        .get(ALWAYS_APPROVE_PROPERTY)?
        .get(response_id)?
        .get(ALWAYS_APPROVE_SCOPE_PROPERTY)?
        .as_str()
        .and_then(ToolApprovalScope::parse)
}

/// Whether `content` is a user-input request other than a function approval
/// (upstream `_has_non_approval_user_input`).
fn is_other_user_input(content: &Content) -> bool {
    matches!(content, Content::OauthConsentRequest(_))
}

fn has_other_user_input(messages: &[Message]) -> bool {
    messages
        .iter()
        .flat_map(|m| &m.contents)
        .any(is_other_user_input)
}

fn approval_requests(messages: &[Message]) -> Vec<FunctionApprovalRequestContent> {
    messages
        .iter()
        .flat_map(|m| &m.contents)
        .filter_map(|c| match c {
            Content::FunctionApprovalRequest(r) => Some(r.clone()),
            _ => None,
        })
        .collect()
}

fn remove_approval_requests(messages: &mut Vec<Message>, remove_ids: &HashSet<String>) {
    for message in messages.iter_mut() {
        message.contents.retain(
            |c| !matches!(c, Content::FunctionApprovalRequest(r) if remove_ids.contains(&r.id)),
        );
    }
    messages.retain(|m| !m.contents.is_empty());
}

/// Coordinates standing approval rules and queued approval prompts around an
/// inner agent.
///
/// Mirrors upstream `ToolApprovalMiddleware` / .NET `ToolApprovalAgent`
/// (see the [module docs](self) for the decorator divergence). Per run:
///
/// 1. **Inbound**: every `FunctionApprovalResponse` in the input is bound to
///    the request this agent surfaced (responses to unknown requests are
///    dropped, and the bound request's call replaces whatever call the
///    response carried — so a caller cannot widen an approval by editing its
///    arguments); an always-approve marker on an approval adds a
///    [`ToolApprovalRule`]; the response is *collected* rather than
///    forwarded.
/// 2. Queued requests now matching a rule (or an auto-approval callback) are
///    approved; if any queued request remains, the next one is returned to
///    the caller without running the inner agent.
/// 3. Otherwise the collected responses are injected as one user message and
///    the inner agent runs. Requests in its response matching a rule or
///    callback are approved automatically; if more than one remains, the
///    first is returned and the rest are queued (unless the response also
///    asks for other user input, in which case the batch is returned
///    intact). When every request was auto-approved the inner agent is
///    re-run straight away.
///
/// Requires an [`AgentSession`] (state lives in `session.state`).
#[derive(Clone)]
pub struct ToolApprovalAgent {
    inner: Arc<dyn SupportsAgentRun>,
    source_id: String,
    auto_approval_rules: Vec<ToolApprovalRuleCallback>,
    approval_not_required_tools: Arc<HashSet<String>>,
    response_binding: bool,
}

impl ToolApprovalAgent {
    /// Wrap `inner`.
    pub fn new(inner: Arc<dyn SupportsAgentRun>) -> Self {
        Self {
            inner,
            source_id: DEFAULT_TOOL_APPROVAL_SOURCE_ID.into(),
            auto_approval_rules: Vec::new(),
            approval_not_required_tools: Arc::new(HashSet::new()),
            response_binding: true,
        }
    }

    /// Override the session-state key.
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// Add heuristic auto-approval callbacks, evaluated after standing rules.
    pub fn auto_approval_rules(
        mut self,
        rules: impl IntoIterator<Item = ToolApprovalRuleCallback>,
    ) -> Self {
        self.auto_approval_rules.extend(rules);
        self
    }

    /// Names of tools known **not** to require approval.
    ///
    /// The core function-invocation loop defers a whole batch of calls when
    /// any one requires approval, surfacing an approval request for every
    /// call in it. Requests for tools named here are approved automatically,
    /// so the human only sees the calls that actually need sign-off — the
    /// equivalent of .NET's `UseApprovalNotRequiredFunctionBypassing` (and
    /// of upstream Python hiding already-approved siblings).
    pub fn approval_not_required_tools(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.approval_not_required_tools = Arc::new(names.into_iter().collect());
        self
    }

    /// Disable binding inbound approval responses to surfaced requests (the
    /// equivalent of .NET's `DisableApprovalResponseBinding`). With binding
    /// off, responses are forwarded as supplied.
    pub fn disable_response_binding(mut self, disable: bool) -> Self {
        self.response_binding = !disable;
        self
    }

    /// The wrapped agent.
    pub fn inner(&self) -> &Arc<dyn SupportsAgentRun> {
        &self.inner
    }

    async fn auto_approves(&self, state: &ToolApprovalState, call: &FunctionCallContent) -> bool {
        if state.matches_rule(call) || self.approval_not_required_tools.contains(&call.name) {
            return true;
        }
        for rule in &self.auto_approval_rules {
            if rule(call).await {
                return true;
            }
        }
        false
    }

    /// Step 1: bind, record rules for, and collect inbound responses.
    fn prepare_inbound(
        &self,
        messages: Vec<Message>,
        state: &mut ToolApprovalState,
    ) -> Result<Vec<Message>> {
        let mut prepared = Vec::with_capacity(messages.len());
        for mut message in messages {
            if !message
                .contents
                .iter()
                .any(|c| matches!(c, Content::FunctionApprovalResponse(_)))
            {
                prepared.push(message);
                continue;
            }
            let mut kept = Vec::new();
            for content in std::mem::take(&mut message.contents) {
                let Content::FunctionApprovalResponse(mut response) = content else {
                    kept.push(content);
                    continue;
                };
                if self.response_binding {
                    let Some(index) = state
                        .pending_approval_requests
                        .iter()
                        .position(|p| p.id == response.id)
                    else {
                        tracing::warn!(
                            id = %response.id,
                            "dropping a function approval response that matches no pending approval request"
                        );
                        continue;
                    };
                    let request = state.pending_approval_requests.remove(index);
                    response.function_call = request.function_call;
                }
                if response.approved {
                    if let Some(scope) = always_approve_scope(&message, &response.id) {
                        let arguments = match scope {
                            ToolApprovalScope::Tool => None,
                            ToolApprovalScope::ToolWithArguments => {
                                Some(serialize_arguments(&response.function_call))
                            }
                        };
                        state.add_rule_if_missing(ToolApprovalRule::new(
                            &response.function_call.name,
                            arguments,
                        )?);
                    }
                }
                state.collected_approval_responses.push(response);
            }
            message
                .additional_properties
                .remove(ALWAYS_APPROVE_PROPERTY);
            if !kept.is_empty() {
                message.contents = kept;
                prepared.push(message);
            }
        }
        Ok(prepared)
    }

    /// Step 2: approve queued requests that a rule now covers.
    async fn drain_auto_approvable_queue(&self, state: &mut ToolApprovalState) {
        let queued = std::mem::take(&mut state.queued_approval_requests);
        for request in queued {
            if self.auto_approves(state, &request.function_call).await {
                state
                    .collected_approval_responses
                    .push(request.create_response(true));
            } else {
                state.queued_approval_requests.push(request);
            }
        }
    }

    fn inject_collected(messages: Vec<Message>, state: &mut ToolApprovalState) -> Vec<Message> {
        if state.collected_approval_responses.is_empty() {
            return messages;
        }
        let responses = std::mem::take(&mut state.collected_approval_responses)
            .into_iter()
            .map(Content::FunctionApprovalResponse)
            .collect();
        let mut out = vec![Message::with_contents(Role::user(), responses)];
        out.extend(messages);
        out
    }

    /// Step 3: auto-approve, queue, and strip outbound requests. Returns
    /// whether every request was auto-approved. Mirrors upstream
    /// `_process_outbound_messages`.
    async fn process_outbound(
        &self,
        messages: &mut Vec<Message>,
        state: &mut ToolApprovalState,
        preserve_batch: bool,
    ) -> bool {
        let requests = approval_requests(messages);
        if requests.is_empty() {
            return false;
        }
        let mut auto_approved = HashSet::new();
        let mut unresolved = Vec::new();
        for request in requests {
            if self.auto_approves(state, &request.function_call).await {
                state
                    .collected_approval_responses
                    .push(request.create_response(true));
                auto_approved.insert(request.id.clone());
            } else {
                unresolved.push(request);
            }
        }
        if auto_approved.is_empty() && (preserve_batch || unresolved.len() <= 1) {
            return false;
        }
        let mut remove = auto_approved;
        if !preserve_batch {
            for request in unresolved.iter().skip(1) {
                remove.insert(request.id.clone());
                state.queued_approval_requests.push(request.clone());
            }
        }
        remove_approval_requests(messages, &remove);
        unresolved.is_empty()
    }

    fn require_session<'a>(
        &self,
        session: Option<&'a mut AgentSession>,
    ) -> Result<&'a mut AgentSession> {
        session.ok_or_else(|| {
            Error::AgentExecution("ToolApprovalAgent requires an AgentSession.".into())
        })
    }

    /// Steps 1–2. `Some(response)` short-circuits with a queued request.
    async fn prologue(
        &self,
        messages: Vec<Message>,
        state: &mut ToolApprovalState,
        session_state: &SessionState,
    ) -> Result<std::result::Result<Vec<Message>, FunctionApprovalRequestContent>> {
        let messages = self.prepare_inbound(messages, state)?;
        self.drain_auto_approvable_queue(state).await;
        if !state.queued_approval_requests.is_empty() {
            let next = state.queued_approval_requests.remove(0);
            state.record_pending([next.clone()]);
            state.save(session_state, &self.source_id);
            return Ok(Err(next));
        }
        Ok(Ok(messages))
    }
}

#[async_trait]
impl SupportsAgentRun for ToolApprovalAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        self.run_with_options(messages, session, AgentRunOptions::default())
            .await
    }

    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        let session = self.require_session(session)?;
        let session_state = session.state.clone();
        let mut state = ToolApprovalState::load(&session_state, &self.source_id)?;
        let mut messages = match self.prologue(messages, &mut state, &session_state).await? {
            Ok(m) => m,
            Err(request) => {
                return Ok(AgentResponse {
                    messages: vec![Message::with_contents(
                        Role::assistant(),
                        vec![Content::FunctionApprovalRequest(request)],
                    )],
                    ..Default::default()
                })
            }
        };
        loop {
            messages = Self::inject_collected(messages, &mut state);
            state.save(&session_state, &self.source_id);
            let mut result = self
                .inner
                .run_with_options(messages, Some(&mut *session), options.clone())
                .await?;
            let preserve = has_other_user_input(&result.messages);
            let all_auto = self
                .process_outbound(&mut result.messages, &mut state, preserve)
                .await;
            state.record_pending(approval_requests(&result.messages));
            state.save(&session_state, &self.source_id);
            if !all_auto || preserve {
                return Ok(result);
            }
            messages = Vec::new();
        }
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let session = session.ok_or_else(|| {
            Error::AgentExecution("ToolApprovalAgent requires an AgentSession.".into())
        })?;
        let options = options.unwrap_or_default();
        let session_state = session.state.clone();
        let mut state = ToolApprovalState::load(&session_state, &self.source_id)?;
        let first = match self.prologue(messages, &mut state, &session_state).await? {
            Ok(m) => m,
            Err(request) => {
                let update = AgentResponseUpdate {
                    contents: vec![Content::FunctionApprovalRequest(request)],
                    role: Some(Role::assistant()),
                    ..Default::default()
                };
                return Ok(futures::stream::iter(vec![Ok(update)]).boxed());
            }
        };
        let (tx, rx) = futures::channel::mpsc::unbounded::<Result<AgentResponseUpdate>>();
        let this = self.clone();
        tokio::spawn(async move {
            let mut messages = first;
            loop {
                messages = Self::inject_collected(messages, &mut state);
                state.save(&session_state, &this.source_id);
                let mut inner = match this
                    .inner
                    .run_stream(messages, Some(session.clone()), Some(options.clone()))
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.unbounded_send(Err(e));
                        return;
                    }
                };
                // Stream non-approval updates live; buffer from the first
                // approval request on, so auto-approved requests never reach
                // the caller (upstream `_process_stream`).
                let mut buffered: Vec<AgentResponseUpdate> = Vec::new();
                let mut streamed_user_input: Vec<Content> = Vec::new();
                let mut buffering = false;
                while let Some(update) = inner.next().await {
                    let update = match update {
                        Ok(u) => u,
                        Err(e) => {
                            let _ = tx.unbounded_send(Err(e));
                            return;
                        }
                    };
                    let has_request = update
                        .contents
                        .iter()
                        .any(|c| matches!(c, Content::FunctionApprovalRequest(_)));
                    if !buffering && !has_request {
                        streamed_user_input.extend(
                            update
                                .contents
                                .iter()
                                .filter(|c| is_other_user_input(c))
                                .cloned(),
                        );
                    }
                    if has_request {
                        buffering = true;
                    }
                    if buffering {
                        buffered.push(update);
                    } else if tx.unbounded_send(Ok(update)).is_err() {
                        return;
                    }
                }
                if buffered.is_empty() {
                    state.save(&session_state, &this.source_id);
                    return;
                }
                let mut contents = streamed_user_input;
                contents.extend(buffered.iter().flat_map(|u| u.contents.clone()));
                let mut response_messages =
                    vec![Message::with_contents(Role::assistant(), contents)];
                let preserve = has_other_user_input(&response_messages);
                let all_auto = this
                    .process_outbound(&mut response_messages, &mut state, preserve)
                    .await;
                let remaining: HashSet<String> = approval_requests(&response_messages)
                    .into_iter()
                    .map(|r| r.id)
                    .collect();
                state.record_pending(approval_requests(&response_messages));
                state.save(&session_state, &this.source_id);
                for mut update in buffered {
                    update.contents.retain(|c| match c {
                        Content::FunctionApprovalRequest(r) => remaining.contains(&r.id),
                        _ => true,
                    });
                    if !update.contents.is_empty() && tx.unbounded_send(Ok(update)).is_err() {
                        return;
                    }
                }
                if !all_auto || preserve {
                    return;
                }
                messages = Vec::new();
            }
        });
        Ok(rx.boxed())
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    fn create_session(&self) -> AgentSession {
        self.inner.create_session()
    }
}
