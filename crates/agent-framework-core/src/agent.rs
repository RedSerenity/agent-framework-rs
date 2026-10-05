//! Agents: the [`SupportsAgentRun`] trait and the concrete [`Agent`].
//!
//! Rust equivalent of `agent_framework._agents`.

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use tracing::Instrument;
use uuid::Uuid;

use crate::client::{ChatClient, FunctionInvokingChatClient};
use crate::compaction::{CompactionProvider, CompactionStrategy};
use crate::error::{Error, Result};
use crate::history::ensure_history_provider;
use crate::memory::{ContextProvider, SessionContext};
use crate::middleware::{AgentContext, ChatContext, MiddlewarePipeline, Terminal};
use crate::session::AgentSession;
use crate::tools::{ToolDefinition, ToolSource};
use crate::types::{
    prepare_messages, AgentResponse, AgentResponseUpdate, ChatOptions, ChatResponse, IntoMessages,
    Message, ResponseFormat,
};

/// A boxed stream of agent run updates.
pub type AgentRunStream = Pin<Box<dyn Stream<Item = Result<AgentResponseUpdate>> + Send>>;

/// Per-run option overrides for a single [`SupportsAgentRun::run_with_options`] /
/// [`SupportsAgentRun::run_stream`] call, merged over the agent's build-time defaults.
///
/// Mirrors upstream `run`/`run_stream` per-call keyword arguments and .NET
/// `AgentRunOptions`: the per-run [`ChatOptions`] take precedence over the
/// agent's defaults (via [`ChatOptions::merge`], matching Python's
/// `run_chat_options & ChatOptions(...)`), and
/// [`additional_tools`](Self::additional_tools) are appended to the tool list
/// for that call only.
#[derive(Debug, Clone, Default)]
pub struct AgentRunOptions {
    /// Chat-option overrides merged over the agent's defaults (per-run wins).
    pub chat_options: Option<ChatOptions>,
    /// Extra tools available only for this run, appended to the agent's tools.
    ///
    /// Declaration-only tools (no executor) surface their calls back to the
    /// caller instead of being executed locally — this is how a hosting
    /// frontend injects client-side tools (see the AG-UI router).
    pub additional_tools: Vec<ToolDefinition>,
}

impl AgentRunOptions {
    /// Empty options (no overrides).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-run chat-option overrides (merged over the agent defaults,
    /// per-run winning).
    pub fn with_chat_options(mut self, options: ChatOptions) -> Self {
        self.chat_options = Some(options);
        self
    }

    /// Append a tool available only for this run.
    pub fn with_tool(mut self, tool: ToolDefinition) -> Self {
        self.additional_tools.push(tool);
        self
    }

    /// Append multiple tools available only for this run.
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.additional_tools.extend(tools);
        self
    }

    /// Whether these options carry no overrides at all. Used by the default
    /// [`SupportsAgentRun::run_with_options`] to decide whether to warn about ignoring
    /// options it cannot honor.
    pub fn is_empty(&self) -> bool {
        self.chat_options.is_none() && self.additional_tools.is_empty()
    }
}

/// Map a completed run into buffered agent updates — one update per message,
/// each with a distinct `message_id` so that re-aggregation via
/// [`AgentResponse::from_updates`] keeps the message boundaries. Shared by
/// the default [`SupportsAgentRun::run_stream`] and [`Agent`]'s middleware-path
/// replay.
///
/// Response-level metadata survives the replay: `response_id` and
/// `conversation_id` ride on every update, and `usage_details` rides the
/// final update as a [`Content::Usage`] item (aggregation folds it back into
/// [`AgentResponse::usage_details`], never into message contents) — the
/// same contract as the tool-loop replay in `FunctionInvokingChatClient`.
pub(crate) fn response_to_updates(response: AgentResponse) -> Vec<Result<AgentResponseUpdate>> {
    let AgentResponse {
        messages,
        response_id,
        conversation_id,
        usage_details,
        finish_reason,
        ..
    } = response;
    let last = messages.len().saturating_sub(1);
    // Keep provider message ids only when all present and distinct; otherwise
    // positional ids for every message. A service (e.g. Assistants) can reuse
    // one run id across the tool-call and final assistant messages, and
    // `AgentResponse::from_updates` keys by id — a duplicate would merge
    // the final answer into the tool-call message, so streamed and
    // non-streamed responses would differ.
    let keep_provider_ids = {
        let mut seen = std::collections::HashSet::new();
        messages.iter().all(|m| {
            m.message_id
                .as_ref()
                .is_some_and(|id| !id.is_empty() && seen.insert(id.as_str()))
        })
    };
    let mut updates: Vec<Result<AgentResponseUpdate>> = messages
        .into_iter()
        .enumerate()
        .map(|(i, m)| {
            let message_id = if keep_provider_ids {
                m.message_id.clone()
            } else {
                Some(format!("msg-{i}"))
            };
            let mut contents = m.contents;
            if i == last {
                if let Some(usage) = usage_details.clone() {
                    contents.push(crate::types::Content::Usage(crate::types::UsageContent {
                        details: usage,
                    }));
                }
            }
            Ok(AgentResponseUpdate {
                contents,
                role: Some(m.role),
                author_name: m.author_name,
                message_id,
                response_id: response_id.clone(),
                conversation_id: conversation_id.clone(),
                // Only the final update carries it, exactly as `client.rs`
                // does for `ChatResponse`: `AgentResponse::from_updates`
                // takes the last reason it sees, and tagging every update
                // would claim each message ended the turn.
                finish_reason: (i == last).then(|| finish_reason.clone()).flatten(),
                ..Default::default()
            })
        })
        .collect();
    // A reason with no messages is still worth emitting — a turn the content
    // filter stopped before any text is exactly that shape.
    if updates.is_empty()
        && (usage_details.is_some() || response_id.is_some() || finish_reason.is_some())
    {
        let contents = usage_details
            .map(|u| {
                vec![crate::types::Content::Usage(crate::types::UsageContent {
                    details: u,
                })]
            })
            .unwrap_or_default();
        updates.push(Ok(AgentResponseUpdate {
            contents,
            role: Some(crate::types::Role::assistant()),
            response_id,
            conversation_id,
            finish_reason,
            ..Default::default()
        }));
    }
    updates
}

/// Sanitize an agent name into a valid tool/function identifier, mirroring
/// Python's `_sanitize_agent_name` (`_agents.py:53-87`).
///
/// Every character that is not ASCII alphanumeric or `_` is replaced with `_`;
/// runs of `_` are collapsed to one; leading/trailing `_` are trimmed. An
/// all-invalid name (e.g. `"@@@"`) becomes `"agent"`, and a name that would
/// start with a digit is prefixed with `_`. `None` maps to `None`.
fn sanitize_agent_name(agent_name: Option<&str>) -> Option<String> {
    let name = agent_name?;
    let replaced: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // Collapse consecutive underscores into one.
    let mut collapsed = String::with_capacity(replaced.len());
    let mut prev_underscore = false;
    for c in replaced.chars() {
        if c == '_' {
            if !prev_underscore {
                collapsed.push('_');
            }
            prev_underscore = true;
        } else {
            collapsed.push(c);
            prev_underscore = false;
        }
    }
    let trimmed = collapsed.trim_matches('_');
    if trimmed.is_empty() {
        return Some("agent".to_string());
    }
    let mut result = trimmed.to_string();
    if result.starts_with(|c: char| c.is_ascii_digit()) {
        result.insert(0, '_');
    }
    Some(result)
}

/// The common interface implemented by all agents.
#[async_trait]
pub trait SupportsAgentRun: Send + Sync {
    /// Run the agent to completion.
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse>;

    /// Run the agent to completion, applying per-run [`AgentRunOptions`] over
    /// the agent's build-time defaults.
    ///
    /// The default implementation ignores `options` and delegates to
    /// [`SupportsAgentRun::run`], emitting a `tracing::warn!` when non-empty options are
    /// supplied — mirroring upstream agents that silently drop kwargs they do
    /// not understand. Agents that support per-run overrides (notably
    /// [`Agent`]) override this.
    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        if !options.is_empty() {
            tracing::warn!(
                agent = %self.id(),
                "agent does not support per-run options; ignoring them"
            );
        }
        self.run(messages, session).await
    }

    /// Run the agent and stream incremental [`AgentResponseUpdate`]s.
    ///
    /// The default implementation is a **buffered fallback**: it runs to
    /// completion via [`SupportsAgentRun::run_with_options`] and yields the response's
    /// messages as updates. Agents with a real streaming backend (notably
    /// [`Agent`], [`WorkflowAgent`](crate::workflow::WorkflowAgent), and the
    /// A2A client agent) override this to stream incrementally.
    ///
    /// `session` is taken **by value**: the returned stream owns it and
    /// drives its context providers (including any history provider) once
    /// the stream is fully consumed. When a provider's storage is shared (as
    /// [`InMemoryHistoryProvider`](crate::history::InMemoryHistoryProvider)'s
    /// is, via `Arc`), the write-back is
    /// observable through a clone taken before streaming.
    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let mut owned = session;
        let response = self
            .run_with_options(messages, owned.as_mut(), options.unwrap_or_default())
            .await?;
        Ok(futures::stream::iter(response_to_updates(response)).boxed())
    }

    /// A stable identifier for this agent.
    fn id(&self) -> &str;

    /// The optional human-readable name.
    fn name(&self) -> Option<&str> {
        None
    }

    /// The display name: `name` if set, else `id`.
    fn display_name(&self) -> String {
        self.name()
            .map(str::to_string)
            .unwrap_or_else(|| self.id().to_string())
    }

    /// A fresh session for a new conversation.
    fn create_session(&self) -> AgentSession {
        AgentSession::new()
    }
}

/// The primary concrete agent: pairs a chat client with instructions, default
/// options, tools, context providers, and middleware.
///
/// Cheaply cloneable (the client, context providers, and middleware are shared
/// via `Arc`), which is what makes [`Agent::as_tool`] possible.
#[derive(Clone)]
pub struct Agent {
    id: String,
    name: Option<String>,
    description: Option<String>,
    client: Arc<dyn ChatClient>,
    chat_options: ChatOptions,
    context_providers: Vec<Arc<dyn ContextProvider>>,
    agent_middleware: MiddlewarePipeline<AgentContext>,
    /// Middleware run around the underlying chat-client call (mirrors
    /// Python's `use_chat_middleware`). See [`Agent::call_chat_client`].
    chat_middleware: MiddlewarePipeline<ChatContext>,
    /// Dynamic tool sources (e.g. MCP servers), resolved fresh on every run
    /// and appended after the agent's static/context/per-run tools. See
    /// [`ToolSource`] and [`Agent::prepare_request`].
    tool_sources: Vec<Arc<dyn ToolSource>>,
}

/// A callback receiving each [`AgentResponseUpdate`] streamed by an agent
/// running as a tool — see [`AsToolOptions::stream_callback`].
pub type AgentToolStreamCallback = Arc<dyn Fn(&AgentResponseUpdate) + Send + Sync>;

/// Options for [`Agent::as_tool`].
#[derive(Clone, Default)]
pub struct AsToolOptions {
    /// The tool name. Defaults to the agent's name (else its id).
    pub name: Option<String>,
    /// The tool description. Defaults to the agent's description (else empty).
    pub description: Option<String>,
    /// The single string argument's name. Defaults to `"task"`.
    pub arg_name: Option<String>,
    /// The argument's description. Defaults to `"Task for {tool_name}"`.
    pub arg_description: Option<String>,
    /// Whether calls to this delegated tool require human approval before
    /// executing (default: no). Mirrors upstream `as_tool(approval_mode=…)`.
    pub approval_mode: crate::tools::ApprovalMode,
    /// Observe the sub-agent's streamed updates as they arrive. When set,
    /// the wrapper runs the sub-agent via `run_stream` and invokes the
    /// callback on every update before aggregating the final response.
    /// Mirrors upstream `as_tool(stream_callback=…)`.
    pub stream_callback: Option<AgentToolStreamCallback>,
    /// Forward the **parent** agent's session to the sub-agent, so both
    /// share the same session identity and state bag.
    ///
    /// The sub-agent receives a [`AgentSession::child`] of the parent's
    /// session: same `session_id`, shared `state`, but an **isolated**
    /// `service_session_id` — the parent's server-side conversation pointer
    /// (whose tool call is still pending mid-run) must not leak into the
    /// sub-agent's own service calls. Mirrors upstream
    /// `as_tool(propagate_session=True)` with the child-session isolation
    /// fix (microsoft/agent-framework#5875). Defaults to `false` (a fresh
    /// session per call).
    pub propagate_session: bool,
}

impl std::fmt::Debug for AsToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsToolOptions")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("arg_name", &self.arg_name)
            .field("arg_description", &self.arg_description)
            .field("approval_mode", &self.approval_mode)
            .field("stream_callback", &self.stream_callback.is_some())
            .field("propagate_session", &self.propagate_session)
            .finish()
    }
}

impl AsToolOptions {
    pub fn new() -> Self {
        Self::default()
    }
    /// Set the tool name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// Set the tool description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    /// Set the argument name (default `"task"`).
    pub fn arg_name(mut self, arg_name: impl Into<String>) -> Self {
        self.arg_name = Some(arg_name.into());
        self
    }
    /// Set the argument description.
    pub fn arg_description(mut self, arg_description: impl Into<String>) -> Self {
        self.arg_description = Some(arg_description.into());
        self
    }
    /// Require human approval before every call to the delegated tool.
    pub fn approval_mode(mut self, mode: crate::tools::ApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }
    /// Observe the sub-agent's streamed updates (see
    /// [`AsToolOptions::stream_callback`]).
    pub fn stream_callback(mut self, callback: AgentToolStreamCallback) -> Self {
        self.stream_callback = Some(callback);
        self
    }
    /// Forward the parent agent's session to the sub-agent (see
    /// [`AsToolOptions::propagate_session`]).
    pub fn propagate_session(mut self, propagate: bool) -> Self {
        self.propagate_session = propagate;
        self
    }
}

impl Agent {
    /// Start building an agent from a chat client. The client is automatically
    /// wrapped with [`FunctionInvokingChatClient`] so local tools are executed.
    pub fn builder(client: impl ChatClient + 'static) -> AgentBuilder {
        AgentBuilder::new(client)
    }

    /// The agent's default instructions.
    pub fn instructions(&self) -> Option<&str> {
        self.chat_options.instructions.as_deref()
    }

    /// Run and stream incremental updates — an ergonomic wrapper over the
    /// object-safe [`SupportsAgentRun::run_stream`] trait method (the real streaming
    /// implementation), accepting `impl IntoMessages`.
    ///
    /// The session's context providers (including any history provider) are
    /// driven when the stream completes; because provider storage is shared
    /// via `Arc`, updates are visible on the original session once the
    /// returned stream is fully consumed. Pass per-run [`AgentRunOptions`] to
    /// override the agent's defaults for this call only.
    pub async fn run_stream(
        &self,
        messages: impl IntoMessages,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        SupportsAgentRun::run_stream(self, messages.into_messages(), session, options).await
    }

    /// Ergonomic streaming run with a fresh session and no per-run options
    /// (mirrors [`Agent::run_once`]).
    pub async fn run_stream_once(&self, messages: impl IntoMessages) -> Result<AgentRunStream> {
        SupportsAgentRun::run_stream(self, messages.into_messages(), None, None).await
    }

    /// Ergonomic run without an explicit session.
    pub async fn run_once(&self, messages: impl IntoMessages) -> Result<AgentResponse> {
        self.run(messages.into_messages(), None).await
    }

    /// The real streaming implementation, shared by the [`SupportsAgentRun::run_stream`]
    /// trait impl. Kept as an inherent helper so the trait method stays a thin
    /// forwarder.
    async fn run_stream_impl(
        &self,
        input: Vec<Message>,
        session: Option<AgentSession>,
        run_options: AgentRunOptions,
    ) -> Result<AgentRunStream> {
        let mut session = session.unwrap_or_else(|| self.create_session());
        let (final_messages, options) = self
            .prepare_request(&input, &mut session, &run_options)
            .await?;

        // When agent middleware is configured, route the run through the same
        // pipeline as `run` (so guardrails/rewrites/termination apply) and then
        // emit the resulting messages as updates. Token-level streaming is only
        // used when there is no agent middleware to honor.
        if self.has_middleware() {
            let (response, persisted_input) = match self
                .run_core(final_messages, options, input.len(), true)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    for cp in self.combined_providers(&session) {
                        let _ = cp.after_run(&input, &[], Some(&e)).await;
                    }
                    return Err(e);
                }
            };
            self.update_session_conversation_id(&mut session, response.conversation_id.as_deref())?;
            let persisted_input = persisted_input.as_deref().unwrap_or(&input);
            for cp in self.combined_providers(&session) {
                cp.after_run(persisted_input, &response.messages, None)
                    .await?;
            }
            // Distinct message ids keep boundaries when re-aggregated; the
            // response's conversation/response ids and usage ride along so
            // service-managed continuity survives the middleware replay.
            return Ok(futures::stream::iter(response_to_updates(response)).boxed());
        }

        let (final_messages, options) = self
            .apply_chat_middleware_pre_call(final_messages, options)
            .await?;
        // Capture the structured-output format before `options` is consumed, so
        // the stream's terminal aggregation can auto-populate `value` (task 3).
        let response_format = options.response_format.clone();
        let agent_name = self.name.clone();
        let providers = self.combined_providers(&session);
        let inner = match self
            .client
            .get_streaming_response(final_messages, options)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                // Failure before the stream opens: let providers observe it.
                for cp in &providers {
                    let _ = cp.after_run(&input, &[], Some(&e)).await;
                }
                return Err(e);
            }
        };

        // Wrap the inner stream: forward mapped updates, then update the session.
        let stream = async_stream_forward(
            inner,
            agent_name,
            session,
            input,
            providers,
            response_format,
        );
        Ok(stream.boxed())
    }

    /// Assemble the final message list and options for a request, applying
    /// context providers (including history) over the session.
    async fn prepare_request(
        &self,
        input: &[Message],
        session: &mut AgentSession,
        run_options: &AgentRunOptions,
    ) -> Result<(Vec<Message>, ChatOptions)> {
        // Auto-attach a fresh `InMemoryHistoryProvider` to a non-service-managed
        // session that doesn't already carry a history provider, so local
        // multi-turn conversations keep accumulating history the way the old
        // `AgentThread` message store used to.
        ensure_history_provider(session);
        let service_session_id = session.service_session_id().map(str::to_string);

        // Merge per-run chat-option overrides over the agent's defaults, with
        // the per-run side winning (mirrors Python's `run_chat_options &
        // ChatOptions(...)`, whose right-hand side takes precedence — see
        // `ChatOptions::merge`).
        let mut options = match &run_options.chat_options {
            Some(overrides) => self.chat_options.clone().merge(overrides.clone()),
            None => self.chat_options.clone(),
        };
        // A service-managed session's id drives continuity and wins; on a
        // local session, a per-run / agent-default `conversation_id` override
        // survives (previously it was unconditionally cleared here, silently
        // starting a new service conversation despite the documented per-run
        // precedence).
        options.conversation_id = service_session_id
            .clone()
            .or(options.conversation_id.take());

        // Context provider injection: run every provider's `before_run` over a
        // shared `SessionContext`, then fold the result into the request
        // (provider instructions AFTER the agent's own; provider messages —
        // including, via the auto-attached/explicit `HistoryProvider`, thread
        // history — PREPENDED ahead of the run's own input; provider tools
        // appended, deduplicated by name).
        let providers = self.combined_providers(session);
        let mut ctx = SessionContext::new(input.to_vec());
        ctx.session_id = Some(session.session_id().to_string());
        ctx.service_session_id = service_session_id.clone();
        for provider in &providers {
            provider.before_run(&mut ctx).await?;
        }
        if let Some(instr) = ctx.instructions {
            options.instructions = Some(match options.instructions.take() {
                Some(base) => format!("{base}\n{instr}"),
                None => instr,
            });
        }
        let mut history = ctx.messages;

        // Deduplicate by name: a tool may be defined on the agent and also
        // injected by a context provider. Providers rejecting duplicate
        // tool names would otherwise fail the request.
        for t in ctx.tools {
            if !options.tools.iter().any(|existing| existing.name == t.name) {
                options.tools.push(t);
            }
        }

        // Append per-run additional tools (deduplicated by name), available for
        // this call only. Declaration-only tools (no executor) surface their
        // calls back to the caller (frontend-tool pattern) via the
        // function-invocation loop's declaration-only handling in `client.rs`.
        for t in &run_options.additional_tools {
            if !options.tools.iter().any(|existing| existing.name == t.name) {
                options.tools.push(t.clone());
            }
        }

        // Resolve dynamic tool sources (e.g. MCP servers) fresh for this run,
        // appended after every tool assembled above (dedup by name against
        // those tools plus any earlier source already appended in this same
        // loop; first registrant wins) — mirrors the Python reference's
        // `existing_names` skip when (re)loading MCP tools/prompts
        // (`_mcp.py:654,696`). A source's failure propagates out of the whole
        // run; see [`ToolSource::resolve_tools`].
        for source in &self.tool_sources {
            let resolved = source.resolve_tools().await?;
            for t in resolved {
                if options.tools.iter().any(|existing| existing.name == t.name) {
                    tracing::warn!(
                        source = source.source_name(),
                        tool = %t.name,
                        "tool source produced a tool whose name collides with an existing \
                         tool; skipping"
                    );
                    continue;
                }
                options.tools.push(t);
            }
        }

        history.extend(input.iter().cloned());
        let instructions = options.instructions.take();
        let final_messages = prepare_messages(history, instructions.as_deref());

        // Hand the run's session to the function-invocation loop (which pops
        // it before the wire client sees the options), so invoked tools can
        // read it from `FunctionInvocationContext::session` — the channel
        // behind `as_tool` + `propagate_session`. The clone shares the
        // session's state bag by reference (see `SessionState`).
        options.session = Some(session.clone());
        Ok((final_messages, options))
    }

    /// Run the agent middleware pipeline with a terminal that calls the chat
    /// client, returning the aggregated response. Shared by `run` and the
    /// middleware path of `run_stream` (thread updates are handled by callers).
    ///
    /// Wrapped in an `invoke_agent` span (OTel GenAI semconv). The plain
    /// token-streaming path does not go through here; that path is observed at
    /// the chat-client decorator level (see [`crate::observability`]).
    ///
    /// `input_len` is the number of trailing `final_messages` that are the
    /// caller's run input (see [`AgentContext::input_start`]). Alongside the
    /// response, returns the run input middleware asked to be persisted in
    /// place of the caller's original (see [`AgentContext::persisted_input`]).
    async fn run_core(
        &self,
        final_messages: Vec<Message>,
        options: ChatOptions,
        input_len: usize,
        is_streaming: bool,
    ) -> Result<(AgentResponse, Option<Vec<Message>>)> {
        let span = crate::observability::agent_span(
            self.name.as_deref().unwrap_or(self.id.as_str()),
            &self.id,
        );
        async move {
            let result = self
                .run_core_inner(final_messages, options, input_len, is_streaming)
                .await;
            let span = tracing::Span::current();
            match &result {
                Ok((response, _)) => {
                    if let Some(usage) = &response.usage_details {
                        if let Some(input) = usage.input_token_count {
                            span.record(crate::observability::attr::INPUT_TOKENS, input);
                        }
                        if let Some(output) = usage.output_token_count {
                            span.record(crate::observability::attr::OUTPUT_TOKENS, output);
                        }
                    }
                }
                Err(err) => {
                    crate::observability::record_error(&span, err);
                }
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn run_core_inner(
        &self,
        final_messages: Vec<Message>,
        options: ChatOptions,
        input_len: usize,
        is_streaming: bool,
    ) -> Result<(AgentResponse, Option<Vec<Message>>)> {
        let tools = options.tools.clone();
        let client = self.client.clone();
        let chat_middleware = self.chat_middleware.clone();
        let terminal: Terminal<AgentContext> = Box::new(move |mut ctx: AgentContext| {
            let client = client.clone();
            let options = options.clone();
            let chat_middleware = chat_middleware.clone();
            Box::pin(async move {
                if ctx.terminate {
                    return Ok(ctx);
                }
                let response = Self::call_chat_client(
                    &client,
                    &chat_middleware,
                    ctx.messages.clone(),
                    options,
                    ctx.is_streaming,
                )
                .await?;
                ctx.result = Some(AgentResponse::from_chat_response(response));
                Ok(ctx)
            }) as crate::tools::BoxFuture<Result<AgentContext>>
        });

        let mut ctx = AgentContext::new(final_messages, is_streaming);
        ctx.input_start = ctx.messages.len().saturating_sub(input_len);
        ctx.agent_id = Some(self.id.clone());
        ctx.agent_name = self.name.clone();
        ctx.tools = tools;
        let ctx = self.agent_middleware.execute(ctx, terminal).await?;
        let persisted_input = ctx.persisted_input;
        let mut response = ctx.result.ok_or_else(|| {
            crate::error::Error::AgentExecution("agent produced no result".into())
        })?;

        if let Some(name) = &self.name {
            for m in &mut response.messages {
                if m.author_name.is_none() {
                    m.author_name = Some(name.clone());
                }
            }
        }
        Ok((response, persisted_input))
    }

    /// Invoke the chat client once, routed through the chat-middleware
    /// pipeline (mirrors Python's `use_chat_middleware`).
    ///
    /// Middleware may mutate `messages`/`chat_options` before the call, then
    /// observe (or override, via [`ChatContext::result`]) the response after
    /// calling `next.run(...)`. A middleware that sets `terminate = true`
    /// without invoking `next` short-circuits the call entirely: the
    /// underlying client is never invoked, and [`ChatContext::result`] (if
    /// set) becomes the returned response.
    async fn call_chat_client(
        client: &Arc<dyn ChatClient>,
        chat_middleware: &MiddlewarePipeline<ChatContext>,
        messages: Vec<Message>,
        options: ChatOptions,
        is_streaming: bool,
    ) -> Result<ChatResponse> {
        if chat_middleware.is_empty() {
            return client.get_response(messages, options).await;
        }
        let client = client.clone();
        let terminal: Terminal<ChatContext> = Box::new(move |mut ctx: ChatContext| {
            let client = client.clone();
            Box::pin(async move {
                if ctx.terminate {
                    return Ok(ctx);
                }
                let response = client
                    .get_response(ctx.messages.clone(), ctx.chat_options.clone())
                    .await?;
                ctx.result = Some(response);
                Ok(ctx)
            }) as crate::tools::BoxFuture<Result<ChatContext>>
        });
        let ctx = ChatContext::new(messages, options, is_streaming);
        let ctx = chat_middleware.execute(ctx, terminal).await?;
        ctx.result.ok_or_else(|| {
            crate::error::Error::AgentExecution("chat middleware produced no result".into())
        })
    }

    /// Apply chat middleware to a *streaming* call's `messages`/`chat_options`
    /// before the real network call.
    ///
    /// Unlike [`Agent::call_chat_client`], this only honors *pre-call*
    /// mutation: a real token stream can't flow back through
    /// [`ChatContext::result`] (typed for a complete [`ChatResponse`]), so any
    /// middleware logic placed *after* `next.run(...)` observes
    /// `ctx.result == None` and cannot post-process individual streamed
    /// tokens, and `terminate`/`result` short-circuiting is not honored here.
    /// This mirrors upstream Python's `use_chat_middleware`, whose streaming
    /// path likewise hands middleware an unconsumed async generator rather
    /// than driving it through the pipeline. Full interception (including
    /// short-circuiting) for chat middleware is available via
    /// [`Agent::run`]/[`Agent::run_once`], and via `run_stream` too
    /// when at least one agent middleware is also configured (that path
    /// funnels through [`Agent::run_core`] and replays the result as
    /// updates).
    async fn apply_chat_middleware_pre_call(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<(Vec<Message>, ChatOptions)> {
        if self.chat_middleware.is_empty() {
            return Ok((messages, options));
        }
        let terminal: Terminal<ChatContext> = Box::new(|ctx| Box::pin(async move { Ok(ctx) }));
        let ctx = ChatContext::new(messages, options, true);
        let ctx = self.chat_middleware.execute(ctx, terminal).await?;
        Ok((ctx.messages, ctx.chat_options))
    }

    /// Whether this agent has any agent-level middleware configured.
    fn has_middleware(&self) -> bool {
        !self.agent_middleware.is_empty()
    }

    /// The effective context providers for a run: the session's, combined
    /// with the agent's own. There is no aggregate wrapper any more — callers
    /// iterate this list directly.
    fn combined_providers(&self, session: &AgentSession) -> Vec<Arc<dyn ContextProvider>> {
        let mut providers = session.context_providers.clone();
        providers.extend(self.context_providers.iter().cloned());
        providers
    }

    /// Reconcile a run's conversation id with the session, mirroring Python's
    /// `_update_thread_with_type_and_conversation_id` (`_agents.py:1204-1234`).
    ///
    /// * No id returned while the session *is* service-managed → the service
    ///   doesn't support service-managed sessions for this request, so surface
    ///   an [`Error::AgentExecution`] (matches Python raising
    ///   `AgentExecutionException`, GAP item 14).
    /// * An id returned that the session newly adopts is simply recorded on
    ///   the session; there is no `thread_created` hook any more (upstream
    ///   removed it — see [`crate::memory::ContextProvider`]).
    fn update_session_conversation_id(
        &self,
        session: &mut AgentSession,
        response_conversation_id: Option<&str>,
    ) -> Result<()> {
        match response_conversation_id {
            None => {
                if session.service_session_id().is_some() {
                    return Err(Error::AgentExecution(
                        "Service did not return a valid conversation id when using a service \
                         managed thread."
                            .into(),
                    ));
                }
                Ok(())
            }
            Some(cid) => {
                session.try_adopt_service_session_id(cid);
                Ok(())
            }
        }
    }
}

/// State carried while forwarding a chat stream as agent updates.
type ForwardFinish = Option<(
    AgentSession,
    Vec<Message>,
    Vec<Arc<dyn ContextProvider>>,
    Option<ResponseFormat>,
)>;

/// Forward an inner chat stream as agent updates and update the session on end.
fn async_stream_forward(
    inner: crate::client::ChatStream,
    agent_name: Option<String>,
    session: AgentSession,
    input: Vec<Message>,
    providers: Vec<Arc<dyn ContextProvider>>,
    response_format: Option<ResponseFormat>,
) -> impl Stream<Item = Result<AgentResponseUpdate>> + Send {
    let finish: ForwardFinish = Some((session, input, providers, response_format));
    futures::stream::unfold(
        (
            inner,
            Vec::<crate::types::ChatResponseUpdate>::new(),
            false,
            finish,
        ),
        move |(mut inner, mut collected, done, mut finish)| {
            let agent_name = agent_name.clone();
            async move {
                if done {
                    return None;
                }
                match inner.next().await {
                    Some(Ok(update)) => {
                        collected.push(update.clone());
                        let mut au = AgentResponseUpdate::from_chat_update(&update);
                        if au.author_name.is_none() {
                            au.author_name = agent_name.clone();
                        }
                        Some((Ok(au), (inner, collected, false, finish)))
                    }
                    Some(Err(e)) => {
                        // Failure mid-stream: let context providers observe the
                        // error before surfacing it. The stream error takes
                        // precedence, so the hooks' results are discarded.
                        if let Some((_session, input, providers, _rf)) = finish.take() {
                            for cp in &providers {
                                let _ = cp.after_run(&input, &[], Some(&e)).await;
                            }
                        }
                        Some((Err(e), (inner, collected, true, None)))
                    }
                    None => {
                        // Stream finished: reconcile the conversation id and fire
                        // the context providers' completion hook (which records
                        // history, for any attached `HistoryProvider`). Surface
                        // any failure as the final item rather than dropping it.
                        if let Some((mut session, input, providers, response_format)) =
                            finish.take()
                        {
                            let response = ChatResponse::from_updates_with_format(
                                collected.clone(),
                                response_format.as_ref(),
                            );
                            match response.conversation_id.as_deref() {
                                None => {
                                    if session.service_session_id().is_some() {
                                        return Some((
                                            Err(Error::AgentExecution(
                                                "Service did not return a valid conversation id \
                                                 when using a service managed thread."
                                                    .into(),
                                            )),
                                            (inner, collected, true, None),
                                        ));
                                    }
                                }
                                Some(cid) => {
                                    session.try_adopt_service_session_id(cid);
                                }
                            }
                            for cp in providers {
                                if let Err(e) = cp.after_run(&input, &response.messages, None).await
                                {
                                    return Some((Err(e), (inner, collected, true, None)));
                                }
                            }
                        }
                        None
                    }
                }
            }
        },
    )
}

#[async_trait]
impl SupportsAgentRun for Agent {
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
        let mut owned_session;
        let session: &mut AgentSession = match session {
            Some(s) => s,
            None => {
                owned_session = self.create_session();
                &mut owned_session
            }
        };

        let (final_messages, chat_options) =
            self.prepare_request(&messages, session, &options).await?;
        let (response, persisted_input) = match self
            .run_core(final_messages, chat_options, messages.len(), false)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                // Failure path: let context providers observe the error.
                // The run's error takes precedence over any hook failure, so
                // the hook result is intentionally discarded.
                for cp in self.combined_providers(session) {
                    let _ = cp.after_run(&messages, &[], Some(&e)).await;
                }
                return Err(e);
            }
        };

        // Persist / validate the service-managed conversation id before
        // firing the completion hooks (tasks: service-session adoption +
        // missing-id error).
        self.update_session_conversation_id(session, response.conversation_id.as_deref())?;

        // Fire the context providers' success completion hook (this is what
        // records history, for any attached `HistoryProvider`).
        let persisted_input = persisted_input.as_deref().unwrap_or(&messages);
        for cp in self.combined_providers(session) {
            cp.after_run(persisted_input, &response.messages, None)
                .await?;
        }

        Ok(response)
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        self.run_stream_impl(messages, session, options.unwrap_or_default())
            .await
    }

    fn id(&self) -> &str {
        &self.id
    }
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn create_session(&self) -> AgentSession {
        // A service-managed conversation id yields a service session;
        // otherwise eagerly attach a fresh `InMemoryHistoryProvider` (rather
        // than relying solely on `prepare_request`'s auto-attach) so history
        // is observable across clones (e.g. during streaming): the only way a
        // caller observes the post-stream write-back through a clone taken
        // beforehand is if the provider (and therefore its `Arc`) already
        // exists at clone time.
        //
        // The agent's own context providers are NOT copied onto the session
        // here: [`Agent::combined_providers`] merges the session's providers
        // with the agent's own at request time, so copying them here would
        // double-invoke the agent's providers for every run against this
        // session.
        match &self.chat_options.conversation_id {
            Some(id) => AgentSession::service(id.clone()),
            None => {
                let mut session = AgentSession::new();
                ensure_history_provider(&mut session);
                session
            }
        }
    }
}

/// Builder for [`Agent`].
pub struct AgentBuilder {
    id: Option<String>,
    name: Option<String>,
    description: Option<String>,
    instructions: Option<String>,
    /// The raw, caller-supplied client. Wrapping in [`FunctionInvokingChatClient`]
    /// is deferred to [`AgentBuilder::build`] so that builder-collected
    /// function middleware can be threaded into the wrapper's constructor.
    client: Arc<dyn ChatClient>,
    chat_options: ChatOptions,
    context_providers: Vec<Arc<dyn ContextProvider>>,
    agent_middleware: Vec<Arc<crate::middleware::AgentMiddleware>>,
    chat_middleware: Vec<Arc<crate::middleware::ChatMiddleware>>,
    function_middleware: Vec<Arc<crate::middleware::FunctionMiddleware>>,
    /// Governs the tool-loop spans; `None` leaves the wrapper reading the
    /// environment. See [`AgentBuilder::observability_config`].
    observability: Option<crate::observability::ObservabilityConfig>,
    function_invocation_config: Option<crate::tools::FunctionInvocationConfig>,
    tool_sources: Vec<Arc<dyn ToolSource>>,
}

impl AgentBuilder {
    fn new(client: impl ChatClient + 'static) -> Self {
        Self {
            id: None,
            name: None,
            description: None,
            instructions: None,
            client: Arc::new(client),
            chat_options: ChatOptions::new(),
            context_providers: Vec::new(),
            agent_middleware: Vec::new(),
            chat_middleware: Vec::new(),
            function_middleware: Vec::new(),
            observability: None,
            function_invocation_config: None,
            tool_sources: Vec::new(),
        }
    }

    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.chat_options.model = Some(model.into());
        self
    }
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.chat_options.temperature = Some(temperature);
        self
    }
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.chat_options.max_tokens = Some(max_tokens);
        self
    }
    /// Request a structured-output response format (e.g. a JSON schema).
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.chat_options.response_format = Some(format);
        self
    }
    /// Add a tool available to the agent.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.chat_options.tools.push(tool);
        self
    }
    /// Add multiple tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.chat_options.tools.extend(tools);
        self
    }
    /// Register a dynamic tool source (e.g. an MCP server wrapper), resolved
    /// fresh on every run and appended after the agent's static/context/
    /// per-run tools (dedup by name against those and any earlier-registered
    /// source; first registrant wins). Call repeatedly to register more than
    /// one source. See [`ToolSource`], resolved internally by every
    /// [`Agent`] run.
    pub fn tool_source(mut self, source: Arc<dyn ToolSource>) -> Self {
        self.tool_sources.push(source);
        self
    }
    /// Add a single context provider (repeatable; providers run in
    /// registration order, agent providers after any thread-level ones).
    pub fn context_provider(mut self, provider: Arc<dyn ContextProvider>) -> Self {
        self.context_providers.push(provider);
        self
    }
    /// Set the context provider list, replacing any previously registered.
    pub fn context_providers(mut self, providers: Vec<Arc<dyn ContextProvider>>) -> Self {
        self.context_providers = providers;
        self
    }
    /// Attach conversation-history compaction, via a
    /// [`CompactionProvider`] wrapping
    /// `strategy` (with the default `ApproxTokenizer`; use
    /// [`AgentBuilder::context_provider`] with
    /// [`CompactionProvider::with_tokenizer`](crate::compaction::CompactionProvider::with_tokenizer)
    /// for a custom tokenizer).
    ///
    /// Registered as one of the agent's own context providers, which
    /// `Agent::combined_providers` always runs *after* the session's —
    /// including the auto-attached (or explicitly attached)
    /// [`HistoryProvider`](crate::history::HistoryProvider), which lives on
    /// the session — so compaction sees, and can shrink, the full
    /// history-prepended message list for the run. Not calling this leaves
    /// the default behavior unchanged: the full history is sent every run.
    pub fn with_compaction(self, strategy: impl CompactionStrategy + 'static) -> Self {
        self.context_provider(Arc::new(CompactionProvider::new(strategy)))
    }
    /// Add an agent middleware.
    pub fn middleware(mut self, mw: Arc<crate::middleware::AgentMiddleware>) -> Self {
        self.agent_middleware.push(mw);
        self
    }
    /// Add a chat middleware, run around the underlying chat-client call on
    /// every request (repeatable, like [`AgentBuilder::middleware`]).
    /// See the chat-client call pipeline for exactly what it can observe
    /// and mutate.
    pub fn chat_middleware(mut self, mw: Arc<crate::middleware::ChatMiddleware>) -> Self {
        self.chat_middleware.push(mw);
        self
    }
    /// Add a function-invocation middleware, run around every local tool call
    /// (repeatable). Plumbed down into the [`FunctionInvokingChatClient`]
    /// this builder wraps the underlying client with.
    pub fn function_middleware(mut self, mw: Arc<crate::middleware::FunctionMiddleware>) -> Self {
        self.function_middleware.push(mw);
        self
    }
    /// Set the [`ObservabilityConfig`](crate::observability::ObservabilityConfig)
    /// for the `execute_tool` spans this agent's tool loop emits — content
    /// capture and the GenAI semantic-convention version.
    ///
    /// The builder wraps the caller's client in a
    /// [`FunctionInvokingChatClient`] itself, so this is the only way to reach
    /// that wrapper's own config. Pass the same value given to an
    /// [`ObservableChatClient`](crate::observability::ObservableChatClient)
    /// around the same client, so one trace reports one convention version
    /// across its chat and tool spans. Left unset, the tool loop reads the
    /// environment, as it always has.
    pub fn observability_config(
        mut self,
        config: crate::observability::ObservabilityConfig,
    ) -> Self {
        self.observability = Some(config);
        self
    }

    /// Set the [`FunctionInvocationConfig`](crate::tools::FunctionInvocationConfig)
    /// for this agent's tool loop: iteration cap, the per-request call and
    /// wall-clock budgets, the consecutive-error limit, and how unknown calls
    /// and tool errors are reported.
    ///
    /// Like [`observability_config`](Self::observability_config), this is the
    /// only way to reach that config: the builder wraps the caller's client in
    /// a [`FunctionInvokingChatClient`] itself, so configuring one before
    /// handing it over would just be wrapped again.
    ///
    /// ```no_run
    /// # use agent_framework_core::prelude::*;
    /// # fn demo<C: ChatClient + 'static>(client: C) {
    /// let config = FunctionInvocationConfig {
    ///     // Bound an unattended run three ways.
    ///     max_iterations: 10,
    ///     max_function_calls: Some(25),
    ///     max_duration_seconds: Some(120.0),
    ///     ..Default::default()
    /// };
    /// let agent = Agent::builder(client).function_invocation_config(config).build();
    /// # let _ = agent;
    /// # }
    /// ```
    pub fn function_invocation_config(
        mut self,
        config: crate::tools::FunctionInvocationConfig,
    ) -> Self {
        self.function_invocation_config = Some(config);
        self
    }

    /// Override the whole chat options object (advanced).
    pub fn chat_options(mut self, options: ChatOptions) -> Self {
        // Preserve tools/instructions collected so far by merging.
        self.chat_options = options.merge(self.chat_options);
        self
    }

    /// Build the agent.
    pub fn build(mut self) -> Agent {
        if let Some(instr) = self.instructions.take() {
            self.chat_options.instructions = Some(match self.chat_options.instructions.take() {
                Some(existing) => format!("{instr}\n{existing}"),
                None => instr,
            });
        }
        if self.chat_options.model.is_none() {
            self.chat_options.model = self.client.model().map(str::to_string);
        }
        // Wrap the raw client in `FunctionInvokingChatClient` now that all
        // builder-collected function middleware is known.
        let mut invoking = FunctionInvokingChatClient::new(self.client)
            .with_function_middleware(self.function_middleware);
        if let Some(config) = self.observability {
            invoking = invoking.with_observability_config(config);
        }
        if let Some(config) = self.function_invocation_config {
            invoking = invoking.with_config(config);
        }
        let client: Arc<dyn ChatClient> = Arc::new(invoking);
        Agent {
            id: self.id.unwrap_or_else(|| Uuid::new_v4().to_string()),
            name: self.name,
            description: self.description,
            client,
            chat_options: self.chat_options,
            context_providers: self.context_providers,
            agent_middleware: MiddlewarePipeline::new(self.agent_middleware),
            chat_middleware: MiddlewarePipeline::new(self.chat_middleware),
            tool_sources: self.tool_sources,
        }
    }
}

impl Agent {
    /// The agent description, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Create a new **service-managed** session bound to `service_session_id`,
    /// mirroring Python's `get_new_thread(service_thread_id=…)`
    /// (`_agents.py:1078-1082`).
    ///
    /// The agent's own context providers are NOT copied onto the returned
    /// session; see the note on [`Agent::create_session`].
    pub fn create_session_with_service_id(
        &self,
        service_session_id: impl Into<String>,
    ) -> AgentSession {
        AgentSession::service(service_session_id)
    }

    /// Reconstruct a session from state (as produced by
    /// [`AgentSession::to_dict`]), mirroring Python's
    /// `BaseAgent.deserialize_thread` (`_agents.py:378-392`).
    ///
    /// Conversation history is **not** part of this state (see
    /// [`AgentSession::to_dict`]); reattach a [`crate::history::HistoryProvider`]
    /// (e.g. via [`crate::history::InMemoryHistoryProvider::from_dict`]) to
    /// `context_providers` separately when restoring a conversation. The
    /// agent's own context providers are NOT copied onto the returned
    /// session; see the note on [`Agent::create_session`].
    pub fn session_from_dict(&self, state: &Value) -> Result<AgentSession> {
        AgentSession::from_dict(state)
    }

    /// Wrap this agent as a [`ToolDefinition`] usable by another agent's
    /// `.tool(...)`. Mirrors Python `BaseAgent.as_tool`.
    ///
    /// The tool takes a single string argument (default name `"task"`) and,
    /// on each call, runs this agent and returns the response text. By
    /// default each call runs **statelessly** (a fresh session per call);
    /// with [`AsToolOptions::propagate_session`] the parent agent's session
    /// is forwarded instead (as an [`AgentSession::child`]). Set
    /// [`AsToolOptions::stream_callback`] to observe the sub-agent's
    /// streamed updates, and [`AsToolOptions::approval_mode`] to gate calls
    /// behind human approval.
    ///
    /// A run that ends with pending user-input requests (function-approval
    /// requests from the sub-agent's own tools) cannot be satisfied from
    /// within a tool call and surfaces as a tool error — mirroring
    /// upstream's `UserInputRequiredException`.
    ///
    /// ```no_run
    /// # use agent_framework_core::prelude::*;
    /// # use agent_framework_core::agent::AsToolOptions;
    /// # fn demo(researcher: Agent, coordinator_client: impl ChatClient + 'static) {
    /// let research_tool = researcher.as_tool(AsToolOptions::new().name("research"));
    /// let coordinator = Agent::builder(coordinator_client)
    ///     .tool(research_tool)
    ///     .build();
    /// # let _ = coordinator;
    /// # }
    /// ```
    pub fn as_tool(&self, options: AsToolOptions) -> ToolDefinition {
        // Mirror Python `name or _sanitize_agent_name(self.name)`: an explicit
        // name is used verbatim; a derived name is sanitized into a valid tool
        // identifier. Falls back to the agent id when no name is available.
        let tool_name = options
            .name
            .or_else(|| sanitize_agent_name(self.name.as_deref()))
            .unwrap_or_else(|| self.id.clone());
        let description = options
            .description
            .or_else(|| self.description.clone())
            .unwrap_or_default();
        let arg_name = options.arg_name.unwrap_or_else(|| "task".to_string());
        let arg_description = options
            .arg_description
            .unwrap_or_else(|| format!("Task for {tool_name}"));
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                arg_name.clone(): { "type": "string", "description": arg_description }
            },
            "required": [arg_name.clone()],
            "additionalProperties": false,
        });
        ToolDefinition {
            name: tool_name.clone(),
            description: description.clone(),
            parameters: schema.clone(),
            kind: crate::tools::ToolKind::Function,
            approval_mode: options.approval_mode,
            executor: Some(Arc::new(AgentAsTool {
                agent: Arc::new(self.clone()),
                name: tool_name,
                description,
                parameters: schema,
                arg_key: arg_name,
                propagate_session: options.propagate_session,
                stream_callback: options.stream_callback,
            })),
        }
    }
}

/// The [`Tool`] behind [`Agent::as_tool`]: delegates each call to the wrapped
/// agent. Reads the parent run's session from the invocation context (via
/// [`Tool::invoke_in_context`]) when `propagate_session` is enabled.
///
/// [`Tool`]: crate::tools::Tool
struct AgentAsTool {
    agent: Arc<Agent>,
    name: String,
    description: String,
    parameters: Value,
    arg_key: String,
    propagate_session: bool,
    stream_callback: Option<AgentToolStreamCallback>,
}

impl AgentAsTool {
    async fn run_task(
        &self,
        arguments: Value,
        parent_session: Option<&AgentSession>,
    ) -> Result<Value> {
        let task = arguments
            .get(&self.arg_key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        // With `propagate_session`, run the sub-agent on a *child* of the
        // parent's session: shared identity + state, isolated server-side
        // conversation pointer (see `AgentSession::child`). Without it (or
        // when the call arrives without a session — e.g. a direct
        // `Tool::invoke`), the sub-agent runs on a fresh session per call.
        let mut child = if self.propagate_session {
            parent_session.map(AgentSession::child)
        } else {
            None
        };

        let response = match &self.stream_callback {
            Some(callback) => {
                let mut stream = SupportsAgentRun::run_stream(
                    self.agent.as_ref(),
                    task.into_messages(),
                    child.clone(),
                    None,
                )
                .await?;
                let mut updates = Vec::new();
                while let Some(update) = stream.next().await {
                    let update = update?;
                    callback(&update);
                    updates.push(update);
                }
                AgentResponse::from_updates(updates)
            }
            None => {
                SupportsAgentRun::run(self.agent.as_ref(), task.into_messages(), child.as_mut())
                    .await?
            }
        };

        // Pending user-input (approval) requests cannot be answered from
        // within a tool call; surface them as a tool error (upstream raises
        // `UserInputRequiredException` here).
        if !response.user_input_requests().is_empty() {
            return Err(Error::tool(format!(
                "agent tool '{}' ended its run with pending user-input requests, \
                 which cannot be satisfied from within a tool call",
                self.name
            )));
        }
        Ok(Value::String(response.text()))
    }
}

#[async_trait]
impl crate::tools::Tool for AgentAsTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.parameters.clone()
    }

    async fn invoke(&self, arguments: Value) -> Result<Value> {
        self.run_task(arguments, None).await
    }

    async fn invoke_in_context(
        &self,
        arguments: Value,
        ctx: &crate::middleware::FunctionInvocationContext,
    ) -> Result<Value> {
        self.run_task(arguments, ctx.session.as_ref()).await
    }
}

#[cfg(test)]
mod tests {
    /// `response_to_updates` feeds the buffered `run_stream` and the
    /// middleware streaming path. It dropped `finish_reason` entirely, so a
    /// content-filtered turn reached hosting as a normal `stop` on exactly
    /// the paths that do not go through `client.rs`.
    #[test]
    fn response_to_updates_carries_the_finish_reason_on_the_last_update_only() {
        let response = super::AgentResponse {
            messages: vec![
                crate::types::Message::assistant("partial"),
                crate::types::Message::assistant("cut off here"),
            ],
            finish_reason: Some(crate::types::FinishReason::new(
                crate::types::FinishReason::CONTENT_FILTER,
            )),
            ..Default::default()
        };
        let updates: Vec<_> = super::response_to_updates(response)
            .into_iter()
            .map(|u| u.expect("update"))
            .collect();
        assert_eq!(updates.len(), 2);
        assert!(
            updates[0].finish_reason.is_none(),
            "only the turn's last update ended the turn"
        );
        assert_eq!(
            updates[1].finish_reason.as_ref().map(|r| r.as_str()),
            Some(crate::types::FinishReason::CONTENT_FILTER)
        );
    }

    /// A turn the filter stopped before any text is a reason with no
    /// messages; it must still reach the stream rather than vanishing.
    #[test]
    fn a_finish_reason_with_no_messages_still_produces_an_update() {
        let response = super::AgentResponse {
            finish_reason: Some(crate::types::FinishReason::new(
                crate::types::FinishReason::CONTENT_FILTER,
            )),
            ..Default::default()
        };
        let updates: Vec<_> = super::response_to_updates(response)
            .into_iter()
            .map(|u| u.expect("update"))
            .collect();
        assert_eq!(updates.len(), 1);
        assert_eq!(
            updates[0].finish_reason.as_ref().map(|r| r.as_str()),
            Some(crate::types::FinishReason::CONTENT_FILTER)
        );
    }

    use super::*;
    use crate::client::ChatStream;
    use crate::compaction::Truncation;
    use crate::types::{ChatResponse, ChatResponseUpdate};
    use futures::stream;
    use std::sync::Mutex;

    /// A chat client that records the full message list of every request it
    /// receives and always replies with the same canned text.
    #[derive(Clone, Default)]
    struct RecordingClient {
        received: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    impl RecordingClient {
        fn requests(&self) -> Vec<Vec<Message>> {
            self.received.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ChatClient for RecordingClient {
        async fn get_response(
            &self,
            messages: Vec<Message>,
            _options: ChatOptions,
        ) -> Result<ChatResponse> {
            self.received.lock().unwrap().push(messages);
            Ok(ChatResponse::from_text("ok"))
        }

        async fn get_streaming_response(
            &self,
            messages: Vec<Message>,
            options: ChatOptions,
        ) -> Result<ChatStream> {
            let resp = self.get_response(messages, options).await?;
            let updates: Vec<Result<ChatResponseUpdate>> = resp
                .messages
                .into_iter()
                .map(|m| {
                    Ok(ChatResponseUpdate {
                        contents: m.contents,
                        role: Some(m.role),
                        ..Default::default()
                    })
                })
                .collect();
            Ok(Box::pin(stream::iter(updates)))
        }
    }

    #[tokio::test]
    async fn without_compaction_sends_the_full_accumulated_history() {
        let client = RecordingClient::default();
        let agent = Agent::builder(client.clone()).build();
        let mut session = agent.create_session();

        agent
            .run(vec![Message::user("turn 1")], Some(&mut session))
            .await
            .unwrap();
        agent
            .run(vec![Message::user("turn 2")], Some(&mut session))
            .await
            .unwrap();
        agent
            .run(vec![Message::user("turn 3")], Some(&mut session))
            .await
            .unwrap();

        let requests = client.requests();
        assert_eq!(requests.len(), 3);

        // Third request: the full history accumulated by turns 1 and 2 (2
        // user + 2 assistant messages) plus this turn's own input.
        let last = requests.last().unwrap();
        assert_eq!(last.len(), 5);
        assert_eq!(last[0].text(), "turn 1");
        assert_eq!(last[2].text(), "turn 2");
        assert_eq!(last.last().unwrap().text(), "turn 3");
    }

    #[tokio::test]
    async fn with_compaction_sends_only_the_compacted_message_set() {
        let client = RecordingClient::default();
        let agent = Agent::builder(client.clone())
            .with_compaction(Truncation::new(2))
            .build();
        let mut session = agent.create_session();

        agent
            .run(vec![Message::user("turn 1")], Some(&mut session))
            .await
            .unwrap();
        agent
            .run(vec![Message::user("turn 2")], Some(&mut session))
            .await
            .unwrap();
        agent
            .run(vec![Message::user("turn 3")], Some(&mut session))
            .await
            .unwrap();

        let requests = client.requests();
        assert_eq!(requests.len(), 3);

        // Third request: compaction caps the *stored history* (4 messages
        // by then) at 2 before this turn's own input is appended, so the
        // outgoing request is 3 messages, and the oldest turn is gone.
        let last = requests.last().unwrap();
        assert_eq!(last.len(), 3);
        assert!(last.iter().all(|m| m.text() != "turn 1"));
        assert_eq!(last[0].text(), "turn 2");
        assert_eq!(last.last().unwrap().text(), "turn 3");
    }

    #[tokio::test]
    async fn with_compaction_runs_after_the_history_provider_in_combined_providers() {
        // Direct check on `combined_providers` ordering: the agent-level
        // provider attached by `with_compaction` must come after the
        // session's auto-attached history provider.
        let client = RecordingClient::default();
        let agent = Agent::builder(client)
            .with_compaction(Truncation::new(2))
            .build();
        let session = agent.create_session();

        let providers = agent.combined_providers(&session);
        assert_eq!(providers.len(), 2);
        assert!(providers[0].is_history_provider());
        assert!(!providers[1].is_history_provider());
    }
}
