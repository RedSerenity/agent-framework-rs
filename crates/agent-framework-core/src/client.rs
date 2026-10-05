//! The chat client trait and the automatic function-invocation loop.
//!
//! Rust equivalent of `agent_framework._clients` plus the tool loop from
//! `_tools.use_function_invocation`.

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::Instrument;

use crate::error::{Error, Result};
use crate::middleware::{FunctionInvocationContext, LiveToolList, MiddlewarePipeline, Terminal};
use crate::session::AgentSession;
use crate::tools::{FunctionInvocationConfig, ToolDefinition, ToolKind};
use crate::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, EmbeddingGenerationOptions,
    FunctionApprovalRequestContent, FunctionApprovalResponseContent, FunctionCallContent,
    FunctionResultContent, GeneratedEmbeddings, Message, Role, ToolMode, UsageContent,
    UsageDetails,
};

/// A boxed stream of streaming chat updates.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatResponseUpdate>> + Send>>;

/// The interface every chat client implements.
///
/// Implementors provide [`ChatClient::get_response`] and
/// [`ChatClient::get_streaming_response`]; the framework layers tool invocation
/// and middleware on top via [`FunctionInvokingChatClient`].
#[async_trait]
pub trait ChatClient: Send + Sync {
    /// Get a complete (non-streaming) response.
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse>;

    /// Get a streaming response as a sequence of updates.
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream>;

    /// The default model id for this client, if any.
    fn model(&self) -> Option<&str> {
        None
    }
}

/// Blanket impl so `Arc<dyn ChatClient>` and wrappers are usable as clients.
#[async_trait]
impl<T: ChatClient + ?Sized> ChatClient for Arc<T> {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        (**self).get_response(messages, options).await
    }
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        (**self).get_streaming_response(messages, options).await
    }
    fn model(&self) -> Option<&str> {
        (**self).model()
    }
}

/// The interface every embedding client implements.
///
/// Rust equivalent of upstream's `SupportsGetEmbeddings` protocol /
/// `BaseEmbeddingClient` (`_clients.py`): generate one embedding per input
/// string, batched in a single request. Vectors are `Vec<f32>` — see the
/// note on [`crate::types::Embedding`] about upstream's genericity.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Generate embeddings for the given values (one per value, in order).
    async fn get_embeddings(
        &self,
        values: Vec<String>,
        options: Option<EmbeddingGenerationOptions>,
    ) -> Result<GeneratedEmbeddings>;

    /// The default embedding model id for this client, if any.
    fn model(&self) -> Option<&str> {
        None
    }
}

/// Blanket impl so `Arc<dyn EmbeddingClient>` and wrappers are usable as
/// clients.
#[async_trait]
impl<T: EmbeddingClient + ?Sized> EmbeddingClient for Arc<T> {
    async fn get_embeddings(
        &self,
        values: Vec<String>,
        options: Option<EmbeddingGenerationOptions>,
    ) -> Result<GeneratedEmbeddings> {
        (**self).get_embeddings(values, options).await
    }
    fn model(&self) -> Option<&str> {
        (**self).model()
    }
}

/// Wraps a [`ChatClient`] to automatically execute local tool calls in a loop,
/// mirroring `use_function_invocation`.
pub struct FunctionInvokingChatClient<C: ChatClient> {
    inner: C,
    config: FunctionInvocationConfig,
    /// Governs the `execute_tool` spans this client emits: content capture and
    /// the GenAI semantic-convention version. Resolved from the environment
    /// once at construction rather than re-read per tool call, so a caller who
    /// selects a version explicitly (via [`Self::with_observability_config`])
    /// gets that version on tool spans too, instead of a trace that mixes
    /// conventions between its chat and tool spans.
    observability: crate::observability::ObservabilityConfig,
    /// Middleware run around every individual tool call (mirrors Python's
    /// function-middleware pipeline, driven here instead of by a
    /// `use_function_invocation` decorator).
    function_middleware: MiddlewarePipeline<FunctionInvocationContext>,
}

impl<C: ChatClient> FunctionInvokingChatClient<C> {
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            config: FunctionInvocationConfig::default(),
            function_middleware: MiddlewarePipeline::default(),
            observability: crate::observability::ObservabilityConfig::from_env(),
        }
    }

    /// Set the [`ObservabilityConfig`](crate::observability::ObservabilityConfig)
    /// governing this client's `execute_tool` spans. Pass the same config given
    /// to an [`ObservableChatClient`](crate::observability::ObservableChatClient)
    /// wrapping the same stack, so one trace reports one semantic-convention
    /// version throughout.
    pub fn with_observability_config(
        mut self,
        observability: crate::observability::ObservabilityConfig,
    ) -> Self {
        self.observability = observability;
        self
    }

    /// Override the function-invocation configuration.
    pub fn with_config(mut self, config: FunctionInvocationConfig) -> Self {
        self.config = config;
        self
    }

    /// Configure the function-invocation middleware pipeline run around every
    /// tool call: middleware may inspect/rewrite
    /// [`FunctionInvocationContext::arguments`], short-circuit execution by
    /// setting [`FunctionInvocationContext::result`] (and either not calling
    /// `next`, or setting `terminate = true`), or observe a propagated
    /// execution error by matching on the `Result` returned from their own
    /// `next.run(...)` call. Replaces any previously configured middleware.
    pub fn with_function_middleware(
        mut self,
        middleware: Vec<Arc<crate::middleware::FunctionMiddleware>>,
    ) -> Self {
        self.function_middleware = MiddlewarePipeline::new(middleware);
        self
    }

    /// A reference to the wrapped client.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    async fn inner_get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.inner.get_response(messages, options).await
    }
}

/// Extract the executable tools from the options into a name→tool map.
fn executable_tools(options: &ChatOptions) -> Vec<ToolDefinition> {
    options
        .tools
        .iter()
        .filter(|t| t.is_executable())
        .cloned()
        .collect()
}

/// Whether `tool` is a *declaration-only* function tool: a known function with
/// no local executor. Mirrors Python's `AIFunction.declaration_only`. A call to
/// such a tool is returned to the caller unexecuted (the frontend-tool pattern
/// that makes AG-UI client-side tools work). Hosted tools (web search, MCP, …)
/// are deliberately excluded — they are not function tools and a call whose
/// name matches none of the local function tools is treated as unknown, not
/// declaration-only, exactly as Python's `_get_tool_map` omits them.
fn is_declaration_only(tool: &ToolDefinition) -> bool {
    tool.kind == ToolKind::Function && tool.executor.is_none()
}

/// Fold one model call's usage into a running aggregate.
///
/// The tool loop issues *several* model calls per logical `get_response`, and
/// each reports only its own tokens. Without accumulation the returned response
/// carries the last iteration's usage alone, so a run that called tools five
/// times under-reports its cost by roughly a factor of five — and because this
/// port's OTel layer reads `usage_details`, the `gen_ai.usage.*` metrics
/// under-report with it. Mirrors upstream's `UsageAggregator` (.NET #7539).
///
/// `None` means *not reported* rather than zero: an aggregate only carries a
/// count once some contributor reported one, and stays `None` when no iteration
/// reported usage at all.
fn accumulate_usage(aggregate: &mut Option<UsageDetails>, incoming: Option<&UsageDetails>) {
    let Some(incoming) = incoming else {
        return;
    };
    match aggregate {
        Some(current) => current.add_assign(incoming),
        None => *aggregate = Some(incoming.clone()),
    }
}

/// The exact rejection payload Python emits for a denied tool call.
const REJECTION_MESSAGE: &str = "Error: Tool call invocation was rejected by user.";

/// The payload a call approved *after* the run's budget was spent receives in
/// place of an execution.
const BUDGET_EXHAUSTED_MESSAGE: &str =
    "Error: Tool call was not executed: the request's function-invocation budget is spent.";

/// Where a run's budget is parked in [`AgentSession::state`] while it waits
/// for an approval. Reserved: a caller's own state keys must not collide with
/// it.
const BUDGET_STATE_KEY: &str = "__af_function_invocation_budget__";

/// Milliseconds since the Unix epoch.
fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The run's tool-call and wall-clock budgets
/// ([`FunctionInvocationConfig::max_function_calls`] and
/// [`FunctionInvocationConfig::max_duration_seconds`]).
///
/// Both are *graceful*: when one is spent the loop disables tools and lets the
/// model answer with what it has, rather than failing the run. Both are also
/// checked between batches rather than inside one, so a batch of parallel
/// calls always completes — a half-executed batch would leave calls without
/// results, which providers reject.
///
/// # Why this lives in the session
///
/// A budget that reset on every `get_response` would be no bound at all on
/// the case it most needs to cover: an approval round trip is a *separate*
/// request, so a run that pauses for a human and resumes would start each leg
/// with a full budget, and an unattended loop of approve-and-continue would
/// never hit either limit. So it is parked in [`AgentSession::state`] at the
/// one point the loop pauses across requests — the approval deferral — and
/// cleared at every terminal exit, so an unrelated later run starts fresh.
///
/// A caller with no session gets a per-request budget, which is the most that
/// can be tracked when there is nowhere to park it; that is also upstream's
/// behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InvocationBudget {
    /// Wall-clock start, as Unix milliseconds rather than an `Instant`: it
    /// has to survive a round trip through the session's JSON state, which an
    /// `Instant` cannot.
    started_millis: u64,
    executed: usize,
    max_calls: Option<usize>,
    max_duration: Option<Duration>,
}

impl InvocationBudget {
    /// Resume the session's budget, or start one.
    ///
    /// `resuming_approval` says whether this request actually carries the
    /// approval responses the parked budget was parked for. It has to: a
    /// caller who abandons a pending approval and starts an unrelated request
    /// on the same session would otherwise inherit that run's elapsed clock
    /// and call count, and find the new request's tools disabled before it
    /// made a single call. The parked state belongs to one paused run, and
    /// a request that is not resuming it discards it.
    fn resume_or_start(
        config: &FunctionInvocationConfig,
        session: Option<&AgentSession>,
        resuming_approval: bool,
    ) -> Self {
        let mut budget = Self {
            started_millis: epoch_millis(),
            executed: 0,
            max_calls: config.max_function_calls,
            // `FunctionInvocationConfig::validate` — which every run calls
            // before reaching here — rejects everything `try_from_secs_f64`
            // does. The fallback is not dead weight for all that:
            // `from_secs_f64` would *panic* on such a value, so a budget
            // built by some future path that skipped validation degrades to
            // an effectively unbounded one instead of taking the process
            // down.
            max_duration: config
                .max_duration_seconds
                .map(|seconds| Duration::try_from_secs_f64(seconds).unwrap_or(Duration::MAX)),
        };
        if !resuming_approval {
            // Not a resumption, so anything parked is from a run this request
            // has nothing to do with.
            Self::clear(session);
            return budget;
        }
        // The limits always come from the live config, never from the parked
        // state: a caller who lowered a limit between runs means it, and a
        // resumed run must not keep spending against the old one.
        if let Some(parked) = session.and_then(|s| s.state.get(BUDGET_STATE_KEY)) {
            if let Some(started) = parked.get("started_millis").and_then(Value::as_u64) {
                budget.started_millis = started;
            }
            if let Some(executed) = parked.get("executed").and_then(Value::as_u64) {
                budget.executed = executed as usize;
            }
        }
        budget
    }

    /// Park this budget for the approval round trip to come.
    fn park(&self, session: Option<&AgentSession>) {
        if let Some(session) = session {
            session.state.insert(
                BUDGET_STATE_KEY,
                serde_json::json!({
                    "started_millis": self.started_millis,
                    "executed": self.executed,
                }),
            );
        }
    }

    /// Drop any parked budget: this run is over, and the next one starts with
    /// a full budget.
    fn clear(session: Option<&AgentSession>) {
        if let Some(session) = session {
            session.state.remove(BUDGET_STATE_KEY);
        }
    }

    /// Charge `count` executed tool bodies against the call budget.
    ///
    /// Only calls that actually ran are charged: a call deferred on an
    /// approval has executed nothing yet, and charging it at deferral *and*
    /// again on the replay would spend the budget twice for one execution.
    fn record(&mut self, count: usize) {
        self.executed = self.executed.saturating_add(count);
    }

    /// Why the budget is spent, or `None` while it is not.
    fn spent(&self) -> Option<String> {
        if let Some(max) = self.max_calls {
            if self.executed >= max {
                return Some(format!(
                    "maximum function calls reached ({}/{max})",
                    self.executed
                ));
            }
        }
        if let Some(max) = self.max_duration {
            // Saturating: a clock that moved backwards between requests
            // reads as no time elapsed rather than as an enormous negative
            // that wraps into an instantly-spent budget.
            let elapsed = Duration::from_millis(epoch_millis().saturating_sub(self.started_millis));
            if elapsed >= max {
                return Some(format!(
                    "maximum duration reached ({:.2}s/{:.2}s)",
                    elapsed.as_secs_f64(),
                    max.as_secs_f64()
                ));
            }
        }
        None
    }
}

/// Execute a single requested tool call through the function-middleware
/// pipeline, with the actual invocation (wrapped in an `execute_tool` span)
/// as the pipeline's terminal handler.
///
/// Returns `(is_error, result)`. `terminate_on_unknown` turns an unknown-tool
/// call into a hard error (propagated) rather than an error result. Unknown
/// tools and unparseable arguments are rejected before middleware ever sees
/// them (there is no function to hand the pipeline in that case); once a
/// [`FunctionInvocationContext`] is built, middleware can rewrite
/// `arguments`, short-circuit by setting `result` (without calling `next`, or
/// with `terminate = true`), or observe an execution error by matching on the
/// `Result` their own `next.run(...)` call returns. A propagated error is
/// converted to the same `(true, FunctionResultContent { exception: .. })`
/// shape the direct-error path used before middleware existed, so
/// `include_detailed_errors` behaves identically either way.
///
/// The single exception is [`Error::MiddlewareFailure`], which is propagated
/// rather than absorbed: it is the fail-closed signal an enforcement layer
/// returns to stop the run instead of letting the model retry around a
/// tool-error result. Because the parallel batch is driven by `try_join_all`,
/// propagating it also drops (cancels) the sibling calls still in flight.
/// The ambient state one tool call runs against: everything that comes from
/// the client and the run rather than from the call itself.
struct ToolCallEnv<'a> {
    function_middleware: &'a MiddlewarePipeline<FunctionInvocationContext>,
    session: Option<&'a crate::session::AgentSession>,
    live_tools: Option<&'a LiveToolList>,
    observability: &'a crate::observability::ObservabilityConfig,
}

/// What came of one tool call.
struct ToolCallOutcome {
    /// Whether the call reached the invocation pipeline at all.
    ///
    /// The budget charges executions, not *attempts*: a hallucinated tool name
    /// or unparseable arguments produce a result without the executor — or
    /// even the middleware — ever running, and charging those would let one
    /// bad name from the model spend a `max_function_calls: 1` budget and
    /// force the tools-off failsafe before the model got a chance to correct
    /// itself. A call that entered the pipeline counts even if middleware
    /// terminated it before the executor: middleware ran, and that is work the
    /// caller asked for.
    executed: bool,
    is_error: bool,
    content: FunctionResultContent,
}

async fn execute_tool_call(
    tool: Option<ToolDefinition>,
    call: &FunctionCallContent,
    include_detailed_errors: bool,
    terminate_on_unknown: bool,
    env: ToolCallEnv<'_>,
) -> Result<ToolCallOutcome> {
    let ToolCallEnv {
        function_middleware,
        session,
        live_tools,
        observability,
    } = env;
    match tool {
        None => {
            if terminate_on_unknown {
                return Err(Error::tool(format!("unknown tool: {}", call.name)));
            }
            Ok(ToolCallOutcome {
                executed: false,
                is_error: true,
                content: FunctionResultContent {
                    call_id: call.call_id.clone(),
                    result: None,
                    exception: Some(format!("tool '{}' not found", call.name)),
                },
            })
        }
        Some(def) => {
            // Reject unparseable arguments rather than silently invoking the tool
            // with null/default input.
            let args = match call.parse_arguments() {
                Ok(m) => Value::Object(m.into_iter().collect()),
                Err(e) => {
                    let msg = if include_detailed_errors {
                        format!("invalid tool arguments: {e}")
                    } else {
                        "invalid tool arguments".to_string()
                    };
                    return Ok(ToolCallOutcome {
                        executed: false,
                        is_error: true,
                        content: FunctionResultContent {
                            call_id: call.call_id.clone(),
                            result: None,
                            exception: Some(msg),
                        },
                    });
                }
            };
            let obs_config = observability.clone();
            let exec = def.executor.as_ref().unwrap().clone();
            let tool_name = def.name.clone();
            let description = def.description.clone();
            let call_id = call.call_id.clone();
            let terminal: Terminal<FunctionInvocationContext> = Box::new(move |mut ctx| {
                Box::pin(async move {
                    if ctx.terminate {
                        return Ok(ctx);
                    }
                    let span = crate::observability::tool_span_ex(
                        &tool_name,
                        &call_id,
                        Some(&description),
                    );
                    crate::observability::record_tool_arguments(&span, &ctx.arguments, &obs_config);
                    #[cfg(feature = "otel-metrics")]
                    let started = std::time::Instant::now();
                    let outcome = async {
                        let result = exec.invoke_in_context(ctx.arguments.clone(), &ctx).await;
                        if let Err(e) = &result {
                            crate::observability::record_error(&tracing::Span::current(), e);
                        }
                        result
                    }
                    .instrument(span.clone())
                    .await;
                    #[cfg(feature = "otel-metrics")]
                    crate::observability::metrics::record_function_invocation_duration(
                        &tool_name,
                        started.elapsed(),
                        outcome
                            .as_ref()
                            .err()
                            .map(crate::observability::error_type)
                            .as_deref(),
                    );
                    if let Ok(value) = &outcome {
                        crate::observability::record_tool_result(&span, value, &obs_config);
                    }
                    ctx.result = Some(outcome?);
                    Ok(ctx)
                }) as crate::tools::BoxFuture<Result<FunctionInvocationContext>>
            });

            let ctx = FunctionInvocationContext::new(call.name.clone(), args)
                .with_session(session.cloned())
                .with_tools(live_tools.cloned());
            match function_middleware.execute(ctx, terminal).await {
                Ok(ctx) => Ok(ToolCallOutcome {
                    executed: true,
                    is_error: false,
                    content: FunctionResultContent {
                        call_id: call.call_id.clone(),
                        result: Some(ctx.result.unwrap_or(Value::Null)),
                        exception: None,
                    },
                }),
                // The one error the loop does not absorb: middleware that
                // refuses a call outright (a guardrail, a policy or
                // authorization gate) needs the run to fail closed rather than
                // hand the model an error string it can retry around. Every
                // other error keeps the absorb-and-continue contract below.
                Err(e) if e.is_middleware_failure() => Err(e),
                Err(e) => {
                    let msg = if include_detailed_errors {
                        format!("{e}")
                    } else {
                        "tool execution failed".to_string()
                    };
                    // The pipeline ran and the tool failed, which is an
                    // execution: charged like any other.
                    Ok(ToolCallOutcome {
                        executed: true,
                        is_error: true,
                        content: FunctionResultContent {
                            call_id: call.call_id.clone(),
                            result: None,
                            exception: Some(msg),
                        },
                    })
                }
            }
        }
    }
}

/// Collect all function-approval responses present in a conversation.
fn collect_approval_responses(messages: &[Message]) -> Vec<FunctionApprovalResponseContent> {
    let mut out = Vec::new();
    for msg in messages {
        for content in &msg.contents {
            if let Content::FunctionApprovalResponse(resp) = content {
                out.push(resp.clone());
            }
        }
    }
    out
}

/// Drop approval contents that an earlier run already resolved: an approval
/// request or response followed, later in the conversation, by a function
/// result for its call.
///
/// Persisted history replays a resolved approval exchange on every later
/// run — the assistant's call plus its approval request, the approval
/// response the caller sent, then (see the tool loop) the result its
/// execution produced. Treating such a response as fresh input would execute
/// the approved call again on every turn, and the request is redundant next
/// to the call it was raised for. Only a result *after* the content counts,
/// so a provider reusing a `call_id` for a new call that is awaiting approval
/// is not mistaken for an answered one. Messages left empty are removed.
fn strip_resolved_approval_contents(messages: &mut Vec<Message>) {
    let mut resolved: Vec<(usize, usize)> = Vec::new();
    for (mi, msg) in messages.iter().enumerate() {
        for (ci, content) in msg.contents.iter().enumerate() {
            let call_id = match content {
                Content::FunctionApprovalResponse(resp) => resp.function_call.call_id.as_str(),
                Content::FunctionApprovalRequest(req) => req.function_call.call_id.as_str(),
                _ => continue,
            };
            if call_id.is_empty() {
                continue;
            }
            let answered_later = msg.contents[ci + 1..]
                .iter()
                .chain(messages[mi + 1..].iter().flat_map(|m| m.contents.iter()))
                .filter_map(Content::as_function_result)
                .any(|r| r.call_id == call_id);
            if answered_later {
                resolved.push((mi, ci));
            }
        }
    }
    if resolved.is_empty() {
        return;
    }
    for (mi, ci) in resolved.into_iter().rev() {
        messages[mi].contents.remove(ci);
    }
    messages.retain(|m| !m.contents.is_empty());
}

/// Rewrite approval request/response contents in place, mirroring Python's
/// `_replace_approval_contents_with_results`.
///
/// * A [`FunctionApprovalRequestContent`] becomes its embedded
///   [`FunctionCallContent`], unless an equal call is *outstanding at that
///   point in the conversation* (a replayed duplicate), in which case the
///   request is removed instead.
/// * An approved [`FunctionApprovalResponseContent`] becomes the corresponding
///   result (correlated strictly by call id) and the message role becomes
///   `tool`.
/// * A rejected response becomes a [`FunctionResultContent`] carrying the
///   rejection payload, and the message role becomes `tool`.
fn replace_approval_contents_with_results(
    messages: &mut [Message],
    approved_results: &HashMap<String, FunctionResultContent>,
) {
    /// A call currently awaiting its result. `from_request` marks entries that
    /// exist because an approval request expanded, so a *real* copy of the same
    /// call arriving later is recognized as the duplicate instead.
    struct Outstanding {
        call: FunctionCallContent,
        from_request: bool,
    }

    // One ordered walk. The outstanding-call set is *derived* as the walk
    // decides each content — a call becomes outstanding when its (kept)
    // declaration passes by, and stops being outstanding when the content that
    // answers it passes by. Three earlier revisions instead maintained this
    // set as separate bookkeeping around a whole-list pre-scan, and every one
    // was wrong the same way: some site updated the mirror on an event that
    // does not change what is outstanding, or missed one that does. A
    // pre-scan is also order-blind — it nets a call against a result that
    // only arrives *after* the replayed request, expanding the request into a
    // second declaration — so deriving in order fixes a real timing bug, not
    // just the structure.
    let mut outstanding: Vec<Outstanding> = Vec::new();
    // Nearest-preceding outstanding call sharing the id — the same pairing
    // rule the compaction module settled on.
    fn retire(outstanding: &mut Vec<Outstanding>, call_id: &str) {
        if call_id.is_empty() {
            return;
        }
        if let Some(position) = outstanding
            .iter()
            .rposition(|entry| entry.call.call_id == call_id)
        {
            outstanding.remove(position);
        }
    }

    for msg in messages.iter_mut() {
        let mut to_remove: Vec<usize> = Vec::new();
        let mut set_role_tool = false;

        for (idx, content) in msg.contents.iter_mut().enumerate() {
            match content {
                Content::FunctionCall(fc) => {
                    if fc.call_id.is_empty() {
                        continue;
                    }
                    // A real call matching an expanded request is the duplicate
                    // now: the expansion already declared it. Keep exactly one,
                    // and mark the survivor as real so it is not itself treated
                    // as expendable.
                    if let Some(position) = outstanding
                        .iter()
                        .position(|entry| entry.from_request && entry.call.same_invocation(fc))
                    {
                        outstanding[position].from_request = false;
                        to_remove.push(idx);
                    } else {
                        outstanding.push(Outstanding {
                            call: fc.clone(),
                            from_request: false,
                        });
                    }
                }
                Content::FunctionResult(fr) => {
                    retire(&mut outstanding, &fr.call_id);
                }
                Content::FunctionApprovalRequest(req) => {
                    // Suppressed only for the *same invocation* — ids are
                    // reused, so a request differing in name or arguments is a
                    // fresh call, not a replay. Nothing is consumed on a match:
                    // dropping a request answers no call.
                    if outstanding
                        .iter()
                        .any(|entry| entry.call.same_invocation(&req.function_call))
                    {
                        to_remove.push(idx);
                    } else {
                        if !req.function_call.call_id.is_empty() {
                            outstanding.push(Outstanding {
                                call: req.function_call.clone(),
                                from_request: true,
                            });
                        }
                        *content = Content::FunctionCall(req.function_call.clone());
                    }
                }
                Content::FunctionApprovalResponse(resp) => {
                    let call_id = resp.function_call.call_id.clone();
                    // Mirrors how the results were keyed above.
                    let result_key = resp
                        .function_call
                        .id
                        .clone()
                        .unwrap_or_else(|| call_id.clone());
                    let mut answered = false;
                    if resp.approved {
                        if let Some(result) = approved_results.get(&result_key) {
                            *content = Content::FunctionResult(result.clone());
                            answered = true;
                        }
                    } else {
                        *content = Content::FunctionResult(FunctionResultContent {
                            call_id: call_id.clone(),
                            result: Some(Value::String(REJECTION_MESSAGE.to_string())),
                            exception: None,
                        });
                        answered = true;
                    }
                    // A response that became a result answers its call, exactly
                    // as a literal result would.
                    if answered {
                        retire(&mut outstanding, &call_id);
                        set_role_tool = true;
                    }
                }
                _ => {}
            }
        }

        for idx in to_remove.into_iter().rev() {
            msg.contents.remove(idx);
        }
        if set_role_tool {
            msg.role = Role::tool();
        }
    }
}

#[async_trait]
impl<C: ChatClient> ChatClient for FunctionInvokingChatClient<C> {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        mut options: ChatOptions,
    ) -> Result<ChatResponse> {
        // After the tool loop settles, auto-populate `ChatResponse.value` from
        // the final text when a structured `response_format` was requested
        // (mirrors Python `try_parse_value`). This is the central non-streaming
        // fill point: it covers a bare `FunctionInvokingChatClient` and every
        // `Agent` run (whose client is always wrapped in one). The tool
        // loop is run inside an `async move` block so its interior `return`s
        // funnel through this single fill/return path.
        let response_format = options.response_format.clone();
        let mut response: ChatResponse = async move {
            self.config.validate()?;
            // Pop the agent-session side channel before the inner provider
            // client ever sees the options (mirrors upstream's
            // `effective_client_kwargs.pop("session")`); it is handed to
            // invoked tools via `FunctionInvocationContext::session`.
            let session = options.session.take();

            // Default tool choice to auto when tools are present and unset.
            if !options.tools.is_empty() && options.tool_choice.is_none() {
                options.tool_choice = Some(ToolMode::Auto);
            }

            if executable_tools(&options).is_empty() || !self.config.enabled {
                return self.inner_get_response(messages, options).await;
            }

            // The run's live tool list (progressive tool exposure): handed to
            // every invocation via `FunctionInvocationContext::tools`, and
            // re-snapshotted into the wire options at the top of every model
            // iteration — so `add_tools`/`remove_tools` from middleware or
            // tools take effect on the NEXT iteration, never the in-flight
            // batch (mirrors upstream `_middleware.py` add_tools/remove_tools
            // semantics).
            let live_tools = LiveToolList::new(std::mem::take(&mut options.tools));

            let mut conversation = messages;
            // Approval exchanges an earlier run already resolved come back
            // with persisted history; they must not execute again.
            strip_resolved_approval_contents(&mut conversation);
            let mut carried: Vec<Message> = Vec::new();
            let mut consecutive_errors = 0usize;
            // Started here rather than at the first execution: the clock is a
            // bound on the whole run, and the first model round trip is part
            // of what it bounds. A run resuming from an approval picks the
            // parked budget back up instead of starting over.
            // Checked against the request's *own* input: only a request
            // carrying approval responses is resuming the paused run whose
            // budget is parked on this session.
            let resuming_approval = !collect_approval_responses(&conversation).is_empty();
            let mut budget = InvocationBudget::resume_or_start(
                &self.config,
                session.as_ref(),
                resuming_approval,
            );
            // Usage summed over every model call this loop makes, applied to
            // whichever response is returned so the caller sees the cost of the
            // whole run and not just its final iteration.
            let mut aggregated_usage: Option<UsageDetails> = None;

            for _ in 0..self.config.max_iterations {
                options.tools = live_tools.snapshot();
                let tools = executable_tools(&options);
                // Checked before the approval replay below, not only after
                // execution: a run resumed from an approval whose budget was
                // already spent must not execute the approved call either.
                let budget_spent = budget.spent();
                if let Some(reason) = budget_spent.as_deref() {
                    tracing::info!(
                        reason,
                        "function-invocation budget spent; disabling tools for this request"
                    );
                    options.tool_choice = Some(ToolMode::None);
                }
                // Process any function-approval responses supplied in the input:
                // execute the approved calls and splice their results into the
                // conversation (mirrors Python's `_collect_approval_responses` +
                // `_replace_approval_contents_with_results`).
                let approval_responses = collect_approval_responses(&conversation);
                if !approval_responses.is_empty() {
                    let mut approved_results: HashMap<String, FunctionResultContent> =
                        HashMap::new();
                    let mut had_error = false;
                    let mut executed = 0usize;
                    for resp in &approval_responses {
                        if !resp.approved {
                            continue;
                        }
                        let call = &resp.function_call;
                        // The budget was spent while this approval was
                        // outstanding. The call still needs a result — leaving
                        // the approval unresolved would send approval content
                        // back to the provider — so it gets one saying why it
                        // did not run, the same shape a rejection gets.
                        if budget_spent.is_some() {
                            let key = call.id.clone().unwrap_or_else(|| call.call_id.clone());
                            approved_results.insert(
                                key,
                                FunctionResultContent::new(
                                    call.call_id.clone(),
                                    Some(Value::String(BUDGET_EXHAUSTED_MESSAGE.to_string())),
                                ),
                            );
                            continue;
                        }
                        let tool = tools.iter().find(|t| t.name == call.name).cloned();
                        let outcome = execute_tool_call(
                            tool,
                            call,
                            self.config.include_detailed_errors,
                            self.config.terminate_on_unknown_calls,
                            ToolCallEnv {
                                function_middleware: &self.function_middleware,
                                session: session.as_ref(),
                                live_tools: Some(&live_tools),
                                observability: &self.observability,
                            },
                        )
                        .await
                        .inspect_err(|_| InvocationBudget::clear(session.as_ref()))?;
                        let ToolCallOutcome {
                            executed: ran,
                            is_error,
                            content,
                        } = outcome;
                        had_error |= is_error;
                        // Keyed by occurrence id when the call carries one, so
                        // two approvals pending under the same provider
                        // `call_id` get their own results instead of one
                        // overwriting the other. Falls back to `call_id` for a
                        // call that predates occurrence ids.
                        let key = call.id.clone().unwrap_or_else(|| content.call_id.clone());
                        approved_results.insert(key, content);
                        if ran {
                            executed += 1;
                        }
                    }
                    budget.record(executed);
                    // Surface the resolved results in the response, so a
                    // history provider records each approved call's result
                    // (or its rejection) next to the approval that produced
                    // it. Without them, persisted history would hold a bare
                    // approval response, and every later run on the session
                    // would execute the approved call again.
                    let resolved: Vec<Content> = approval_responses
                        .iter()
                        .filter_map(|resp| {
                            let call = &resp.function_call;
                            if resp.approved {
                                let key = call.id.clone().unwrap_or_else(|| call.call_id.clone());
                                approved_results
                                    .get(&key)
                                    .cloned()
                                    .map(Content::FunctionResult)
                            } else {
                                Some(Content::FunctionResult(FunctionResultContent::new(
                                    call.call_id.clone(),
                                    Some(Value::String(REJECTION_MESSAGE.to_string())),
                                )))
                            }
                        })
                        .collect();
                    if !resolved.is_empty() {
                        carried.push(Message::with_contents(Role::tool(), resolved));
                    }
                    replace_approval_contents_with_results(&mut conversation, &approved_results);
                    if had_error {
                        consecutive_errors += 1;
                        if consecutive_errors > self.config.max_consecutive_errors_per_request {
                            options.tool_choice = Some(ToolMode::None);
                        }
                    }
                }

                // Out of budget: leave the loop, having resolved any inbound
                // approvals above, and let the failsafe below make one final
                // tools-disabled call. Setting `tool_choice` and continuing to
                // iterate would not be a ceiling at all — a provider that
                // ignores the hint (or a model that emits a call anyway) would
                // have its calls executed, which is the one thing a spent
                // budget must not allow.
                //
                // Re-checked rather than reusing `budget_spent`, because the
                // approval replay just above may itself have spent the last of
                // it; without this the resumed leg would get one more full
                // tool-calling iteration for free.
                if let Some(reason) = budget.spent() {
                    if budget_spent.is_none() {
                        tracing::info!(
                            reason,
                            "function-invocation budget spent; disabling tools for this request"
                        );
                    }
                    options.tool_choice = Some(ToolMode::None);
                    break;
                }

                let response = self
                    .inner_get_response(conversation.clone(), options.clone())
                    .await
                    // An error ends the run, so a budget parked by an earlier
                    // approval must not outlive it: the next, unrelated run on
                    // this session would otherwise resume a clock that started
                    // in a run already over, and could be spent before making
                    // a single call.
                    .inspect_err(|_| InvocationBudget::clear(session.as_ref()))?;
                accumulate_usage(&mut aggregated_usage, response.usage_details.as_ref());

                // A call whose result is already present in the same response
                // was executed by the provider (e.g. Anthropic server-side
                // web-search/code-execution/MCP `server_tool_use` blocks,
                // which arrive paired with their `*_tool_result`). Executing
                // it locally would produce a bogus "tool not found" — only
                // unresolved calls enter the local tool loop.
                let resolved_call_ids: std::collections::HashSet<&str> = response
                    .messages
                    .iter()
                    .flat_map(|m| m.contents.iter())
                    .filter_map(Content::as_function_result)
                    .map(|fr| fr.call_id.as_str())
                    .collect();
                let calls: Vec<_> = response
                    .messages
                    .iter()
                    .flat_map(|m| m.contents.iter())
                    .filter_map(Content::as_function_call)
                    .filter(|fc| !resolved_call_ids.contains(fc.call_id.as_str()))
                    .cloned()
                    .collect();

                if calls.is_empty() {
                    // The run is over, so nothing is left to budget: a later,
                    // unrelated run on this session starts with a full one.
                    InvocationBudget::clear(session.as_ref());
                    // Prepend the accumulated tool-interaction messages so the final
                    // assistant message stays last.
                    let mut final_resp = response;
                    let mut msgs = std::mem::take(&mut carried);
                    msgs.append(&mut final_resp.messages);
                    final_resp.messages = msgs;
                    final_resp.usage_details = aggregated_usage;
                    return Ok(final_resp);
                }

                // The wall-clock budget is checked again *here*, after the
                // model call, because that call is itself part of the elapsed
                // time it bounds. A provider response that takes longer than
                // the remaining budget and comes back asking for tools would
                // otherwise have its whole batch executed — the check above
                // ran before the request, when the budget was still alive —
                // which is the one thing a spent budget must not allow.
                //
                // Placed before the approval and declaration-only branches for
                // the same reason the top-of-loop check precedes them: once
                // the budget is spent this loop stops asking for tools at all,
                // and opening a human-approval round trip whose calls could
                // only come back unexecuted is worse than ending the run.
                // What the response already achieved is kept; only the local
                // calls that will now never run are dropped. A provider that
                // ran a hosted tool itself — an Anthropic server-side web
                // search, say — put the call *and its result* in this same
                // response, and that work is done and paid for: discarding it
                // would make the failsafe answer from a conversation missing
                // the very thing it just looked up. Stripping only the
                // unresolved calls keeps that, and still leaves no unanswered
                // function call behind; the failsafe below then asks the
                // model once with tools off.
                if let Some(reason) = budget.spent() {
                    if budget_spent.is_none() {
                        tracing::info!(
                            reason,
                            "function-invocation budget spent while the model was responding; \
                             disabling tools for this request"
                        );
                    }
                    let resolved_owned: std::collections::HashSet<String> = resolved_call_ids
                        .iter()
                        .map(|id| (*id).to_string())
                        .collect();
                    let mut kept = response;
                    for message in kept.messages.iter_mut() {
                        message.contents.retain(|content| match content {
                            Content::FunctionCall(fc) => resolved_owned.contains(&fc.call_id),
                            _ => true,
                        });
                    }
                    // A message left with nothing in it would be an empty turn
                    // in the conversation, which some providers reject.
                    kept.messages.retain(|m| !m.contents.is_empty());
                    // Into the *conversation* as well as the transcript,
                    // mirroring what the normal path does with a response it
                    // keeps. `carried` is only what the caller gets back;
                    // the failsafe's model call reads `conversation`, so
                    // extending `carried` alone would preserve the hosted
                    // result in the returned messages while still asking the
                    // model to answer without it — the exact outcome this is
                    // supposed to prevent.
                    match kept.conversation_id.clone() {
                        // Service-managed: the provider already holds this
                        // response in its own stored history, so forwarding
                        // the id is what makes the result visible to the
                        // failsafe. Re-sending the messages would duplicate
                        // it.
                        Some(cid) => options.conversation_id = Some(cid),
                        // Stateless: the model sees only what we send.
                        None => conversation.extend(kept.messages.iter().cloned()),
                    }
                    carried.extend(kept.messages);
                    options.tool_choice = Some(ToolMode::None);
                    break;
                }

                // Human-in-the-loop gate: if *any* requested tool requires approval,
                // defer *all* calls (matching Python) and return an assistant message
                // that carries the original calls plus one approval request each.
                let needs_approval = calls.iter().any(|c| {
                    tools
                        .iter()
                        .find(|t| t.name == c.name)
                        .map(ToolDefinition::requires_approval)
                        .unwrap_or(false)
                });
                if needs_approval {
                    // Owned before `response` moves below: the stamping walk
                    // needs the same filter `calls` was built with.
                    let resolved_owned: std::collections::HashSet<String> = resolved_call_ids
                        .iter()
                        .map(|id| (*id).to_string())
                        .collect();
                    let mut resp = response;
                    // This is the one moment a call stops being answered within
                    // its turn: it now has to survive a round trip and come back
                    // matched to an approval. A provider `call_id` cannot carry
                    // that on its own — providers reuse ids, so two approvals
                    // pending at once under one id are indistinguishable and
                    // approving either could resolve the other. Mint an
                    // occurrence id here and stamp it on the call itself, so the
                    // request, the response derived from it, and the replayed
                    // call all carry the same identity.
                    let mut identified_calls = calls.clone();
                    let approval_contents: Vec<Content> = identified_calls
                        .iter_mut()
                        .map(|c| {
                            let occurrence_id = c.ensure_occurrence_id().to_string();
                            Content::FunctionApprovalRequest(FunctionApprovalRequestContent {
                                id: occurrence_id,
                                function_call: c.clone(),
                            })
                        })
                        .collect();
                    // The response's own copies of the calls must carry the id
                    // too: they are what a caller replays back, and a replay
                    // without the id would fall back to structural matching and
                    // reintroduce the ambiguity the id exists to remove.
                    //
                    // Matched **positionally, consuming each id**, rather than
                    // by searching. `calls` was collected from these same
                    // messages in this same order, so the nth function call
                    // here is the nth entry of `identified_calls`. A search
                    // would reintroduce exactly the bug the ids exist to fix:
                    // for two calls sharing a `call_id` — the case this whole
                    // mechanism is for — every copy would match the *first*
                    // entry and be stamped with one id, so two approval
                    // requests would carry distinct ids while the replayed
                    // calls carried the same one.
                    let mut ids = identified_calls.iter().map(|c| c.id.clone());
                    for message in resp.messages.iter_mut() {
                        for content in message.contents.iter_mut() {
                            if let Content::FunctionCall(fc) = content {
                                // `calls` skipped provider-resolved calls, so
                                // this walk must skip them too — otherwise a
                                // resolved call appearing first consumes the
                                // id belonging to the unresolved call after
                                // it, and every later stamp is off by one.
                                if resolved_owned.contains(fc.call_id.as_str()) {
                                    continue;
                                }
                                if let Some(id) = ids.next() {
                                    fc.id = id;
                                }
                            }
                        }
                    }
                    if let Some(m) = resp
                        .messages
                        .iter_mut()
                        .rev()
                        .find(|m| m.role == Role::assistant())
                    {
                        m.contents.extend(approval_contents);
                    } else {
                        resp.messages
                            .push(Message::with_contents(Role::assistant(), approval_contents));
                    }
                    // The one exit that is a *pause*, not an end: park the
                    // budget so the resumed leg keeps spending the same one.
                    budget.park(session.as_ref());
                    let mut msgs = std::mem::take(&mut carried);
                    msgs.append(&mut resp.messages);
                    resp.messages = msgs;
                    resp.usage_details = aggregated_usage;
                    return Ok(resp);
                }

                // Declaration-only calls: a call targeting a KNOWN tool that has
                // no local executor (declaration-only — e.g. an AG-UI frontend
                // tool, or a per-run `additional_tools` entry) terminates the
                // loop and returns the response with the `FunctionCallContent`
                // intact, so the caller can execute it. Mirrors Python's
                // `_try_execute_function_calls` `declaration_only` branch
                // (`_tools.py:1396-1420`): if *any* requested call is
                // declaration-only, the whole response is returned unexecuted.
                // A genuinely unknown tool name is NOT declaration-only and
                // keeps today's not-found handling in `execute_tool_call`.
                let has_declaration_only = calls.iter().any(|c| {
                    options
                        .tools
                        .iter()
                        .any(|t| t.name == c.name && is_declaration_only(t))
                });
                if has_declaration_only {
                    InvocationBudget::clear(session.as_ref());
                    let mut resp = response;
                    let mut msgs = std::mem::take(&mut carried);
                    msgs.append(&mut resp.messages);
                    resp.messages = msgs;
                    resp.usage_details = aggregated_usage;
                    return Ok(resp);
                }

                // Record the assistant message(s) that requested the calls.
                carried.extend(response.messages.iter().cloned());
                let response_conversation_id = response.conversation_id.clone();

                // The model may emit several parallel tool calls, and
                // I/O-bound tools should not be serialized — unless the
                // caller asked for that, which
                // `FunctionInvocationConfig::allow_concurrent_invocation`
                // turns off.
                let invocations = calls.iter().map(|call| {
                    let tool = tools.iter().find(|t| t.name == call.name).cloned();
                    let call = call.clone();
                    let include_detailed_errors = self.config.include_detailed_errors;
                    let terminate_on_unknown = self.config.terminate_on_unknown_calls;
                    let function_middleware = self.function_middleware.clone();
                    let session = session.clone();
                    let live_tools = live_tools.clone();
                    let observability = self.observability.clone();
                    async move {
                        execute_tool_call(
                            tool,
                            &call,
                            include_detailed_errors,
                            terminate_on_unknown,
                            ToolCallEnv {
                                function_middleware: &function_middleware,
                                session: session.as_ref(),
                                live_tools: Some(&live_tools),
                                observability: &observability,
                            },
                        )
                        .await
                    }
                });

                let outcomes = if self.config.allow_concurrent_invocation {
                    futures::future::try_join_all(invocations)
                        .await
                        .inspect_err(|_| InvocationBudget::clear(session.as_ref()))?
                } else {
                    // One at a time, in the order the model emitted them. A
                    // failure stops the batch, so the calls after it never
                    // run: a caller who sequenced these asked for each to see
                    // the previous one's effect, and running the rest after
                    // one was refused is the opposite of that.
                    let invocations: Vec<_> = invocations.collect();
                    let mut sequential = Vec::with_capacity(invocations.len());
                    for invocation in invocations {
                        match invocation.await {
                            Ok(outcome) => sequential.push(outcome),
                            Err(err) => {
                                // Same as the concurrent path: the run ends
                                // here, so the budget is discarded rather
                                // than charged for the calls that did run.
                                InvocationBudget::clear(session.as_ref());
                                return Err(err);
                            }
                        }
                    }
                    sequential
                };
                budget.record(outcomes.iter().filter(|o| o.executed).count());
                let mut result_contents: Vec<Content> = Vec::with_capacity(outcomes.len());
                let mut had_error = false;
                for ToolCallOutcome {
                    is_error, content, ..
                } in outcomes
                {
                    had_error |= is_error;
                    result_contents.push(Content::FunctionResult(content));
                }

                if had_error {
                    consecutive_errors += 1;
                    if consecutive_errors > self.config.max_consecutive_errors_per_request {
                        // Give up on tools and let the model answer directly.
                        options.tool_choice = Some(ToolMode::None);
                    }
                } else {
                    consecutive_errors = 0;
                }

                let tool_message = Message::with_contents(Role::tool(), result_contents);
                carried.push(tool_message.clone());
                match response_conversation_id {
                    // A service-managed client that created (or continued) the
                    // conversation now holds the history server-side. Propagate
                    // its id so the follow-up tool-output submission targets the
                    // right thread — without this, Assistants / Azure AI reject
                    // the submission because `conversation_id` is still `None` —
                    // and send ONLY the new tool results next turn rather than
                    // re-sending the whole history (mirrors Python
                    // `_tools.py:1635-1637, 1695-1699`).
                    Some(cid) => {
                        options.conversation_id = Some(cid);
                        conversation = vec![tool_message];
                    }
                    // Stateless client (e.g. Chat Completions): accumulate and
                    // re-send the full history each turn.
                    None => {
                        conversation.extend(response.messages);
                        conversation.push(tool_message);
                    }
                }
            }

            // Failsafe: one final call with tools disabled.
            InvocationBudget::clear(session.as_ref());
            options.tool_choice = Some(ToolMode::None);
            let mut final_resp = self.inner_get_response(conversation, options).await?;
            accumulate_usage(&mut aggregated_usage, final_resp.usage_details.as_ref());
            let mut msgs = std::mem::take(&mut carried);
            msgs.append(&mut final_resp.messages);
            final_resp.messages = msgs;
            final_resp.usage_details = aggregated_usage;
            Ok(final_resp)
        }
        .await?;
        response.try_parse_value(response_format.as_ref());
        Ok(response)
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let tools = executable_tools(&options);
        if tools.is_empty() || !self.config.enabled {
            return self.inner.get_streaming_response(messages, options).await;
        }
        // With tools, run the full loop then stream the aggregated result.
        // Each message is replayed as its own update with a stable, distinct
        // `message_id` so that consumers re-aggregating via
        // `ChatResponse::from_updates` keep the messages separate rather than
        // merging the tool-call and final assistant messages by role.
        let response = self.get_response(messages, options).await?;
        // Response-level metadata must survive the replay so re-aggregation
        // (and the agent's thread adoption) sees it: ids on every update,
        // and usage/finish-reason on the final one (usage rides as a
        // `Content::Usage` item, which `absorb_update` folds into
        // `usage_details` rather than the message contents — the same shape
        // providers use for their terminal stream chunk).
        let conversation_id = response.conversation_id.clone();
        let response_id = response.response_id.clone();
        let finish_reason = response.finish_reason.clone();
        let usage_details = response.usage_details.clone();
        let last = response.messages.len().saturating_sub(1);
        // Keep the provider message ids only when they're all present and
        // distinct; otherwise use positional ids for every message. A service
        // (e.g. Assistants) can reuse one run id for both the tool-call turn
        // and the final assistant turn, and `ChatResponse::from_updates` keys
        // messages by id — a duplicate would merge the final answer into the
        // tool-call message, ahead of the tool result.
        let keep_provider_ids = {
            let mut seen = std::collections::HashSet::new();
            response.messages.iter().all(|m| {
                m.message_id
                    .as_ref()
                    .is_some_and(|id| !id.is_empty() && seen.insert(id.as_str()))
            })
        };
        let mut updates: Vec<Result<ChatResponseUpdate>> = response
            .messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| {
                let message_id = if keep_provider_ids {
                    m.message_id.clone()
                } else {
                    Some(format!("replay-{i}"))
                };
                let mut contents = m.contents;
                let is_last = i == last;
                if is_last {
                    if let Some(usage) = usage_details.clone() {
                        contents.push(Content::Usage(UsageContent { details: usage }));
                    }
                }
                Ok(ChatResponseUpdate {
                    contents,
                    role: Some(m.role),
                    author_name: m.author_name,
                    message_id,
                    conversation_id: conversation_id.clone(),
                    response_id: response_id.clone(),
                    finish_reason: is_last.then(|| finish_reason.clone()).flatten(),
                    ..Default::default()
                })
            })
            .collect();
        // A messageless response (unusual, but possible) still carries its
        // terminal metadata in one trailing update.
        if updates.is_empty() && (usage_details.is_some() || finish_reason.is_some()) {
            let contents = usage_details
                .map(|u| vec![Content::Usage(UsageContent { details: u })])
                .unwrap_or_default();
            updates.push(Ok(ChatResponseUpdate {
                contents,
                role: Some(Role::assistant()),
                conversation_id,
                response_id,
                finish_reason,
                ..Default::default()
            }));
        }
        Ok(stream::iter(updates).boxed())
    }

    fn model(&self) -> Option<&str> {
        self.inner.model()
    }
}

// ---------------------------------------------------------------------------
// Retry / backoff layer
// ---------------------------------------------------------------------------

/// Which errors a [`RetryPolicy`] considers retryable.
#[derive(Clone)]
pub enum RetryOn {
    /// The built-in default predicate (see [`RetryPolicy`] docs for the exact
    /// rule): retries HTTP `408`/`429`/`5xx` ([`Error::ServiceStatus`]) and
    /// transport-ish [`Error::Service`] failures (timeouts / connection
    /// errors). Never retries [`Error::ServiceInvalidAuth`],
    /// [`Error::ServiceInvalidRequest`], or [`Error::ServiceContentFilter`] —
    /// authentication/authorization failures, malformed requests, and
    /// content-filter refusals are non-transient, so retrying would just
    /// repeat the same rejection.
    Default,
    /// A fully custom predicate deciding, per error, whether to retry.
    Predicate(Arc<dyn Fn(&Error) -> bool + Send + Sync>),
}

impl std::fmt::Debug for RetryOn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetryOn::Default => f.write_str("RetryOn::Default"),
            RetryOn::Predicate(_) => f.write_str("RetryOn::Predicate(..)"),
        }
    }
}

impl RetryOn {
    /// A custom retry predicate.
    pub fn predicate<F>(f: F) -> Self
    where
        F: Fn(&Error) -> bool + Send + Sync + 'static,
    {
        RetryOn::Predicate(Arc::new(f))
    }

    fn should_retry(&self, err: &Error) -> bool {
        match self {
            RetryOn::Default => default_should_retry(err),
            RetryOn::Predicate(p) => p(err),
        }
    }
}

/// The default retryability rule used by [`RetryOn::Default`].
///
/// Retries when either:
/// * the error is an [`Error::ServiceStatus`] whose status is `408`
///   (Request Timeout), `429` (Too Many Requests), or any `5xx`; or
/// * the error is an [`Error::Service`] whose (lowercased) message contains one
///   of the transport-failure markers the provider clients emit — `"request
///   failed"` (the prefix wrapping every `reqwest` send error: DNS, connect,
///   timeout, reset), `"timed out"`, `"timeout"`, `"connection"`, or `"stream
///   error"`.
///
/// Everything else (4xx other than 408/429, parse errors, tool/workflow errors,
/// non-transport service errors) is treated as non-retryable. This explicitly
/// includes [`Error::ServiceInvalidAuth`], [`Error::ServiceInvalidRequest`],
/// and [`Error::ServiceContentFilter`] — authentication/authorization
/// failures, malformed requests, and content-filter refusals are
/// non-transient, so retrying would just repeat the same rejection. None of
/// the three carry a status via [`Error::status`], so they fall through to
/// the final `_ => false` below (there's no dedicated match arm for them:
/// merging one in would just duplicate that `false`, which `clippy` flags as
/// `match_same_arms`).
fn default_should_retry(err: &Error) -> bool {
    if let Some(status) = err.status() {
        return status == 408 || status == 429 || (500..600).contains(&status);
    }
    match err {
        Error::Service(msg) => {
            let m = msg.to_lowercase();
            m.contains("request failed")
                || m.contains("timed out")
                || m.contains("timeout")
                || m.contains("connection")
                || m.contains("stream error")
        }
        _ => false,
    }
}

/// Policy controlling [`RetryingChatClient`] backoff.
///
/// Delays grow exponentially from [`initial_delay`](Self::initial_delay) by
/// [`backoff_multiplier`](Self::backoff_multiplier) per attempt, are capped at
/// [`max_delay`](Self::max_delay), and are then reduced by up to
/// [`jitter`](Self::jitter) (a fraction of the delay). When the failing error
/// carries a server `Retry-After` (see [`Error::retry_after`]) that value is
/// used instead of the computed backoff (still capped by `max_delay`, and not
/// jittered — it is an explicit server instruction).
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    /// Maximum number of *retries* after the initial attempt (default `3`, so
    /// up to four total attempts).
    pub max_retries: usize,
    /// Base delay before the first retry (default `500ms`).
    pub initial_delay: Duration,
    /// Upper bound on any single delay, also capping a server `Retry-After`
    /// (default `30s`).
    pub max_delay: Duration,
    /// Exponential growth factor applied per retry (default `2.0`).
    pub backoff_multiplier: f64,
    /// Jitter as a fraction in `0.0..=1.0` (default `0.3`): the computed delay
    /// is multiplied by `1 - jitter * r` for a per-attempt random `r` in
    /// `[0, 1)`. `0.0` disables jitter (fully deterministic delays).
    pub jitter: f64,
    /// Which errors to retry (default [`RetryOn::Default`]).
    pub retry_on: RetryOn,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: 0.3,
            retry_on: RetryOn::Default,
        }
    }
}

impl RetryPolicy {
    /// A policy with the given retry count and otherwise-default backoff.
    pub fn with_max_retries(max_retries: usize) -> Self {
        Self {
            max_retries,
            ..Self::default()
        }
    }

    /// Set the base delay before the first retry.
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the per-delay cap (also caps a server `Retry-After`).
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set the exponential growth factor.
    pub fn backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Set the jitter fraction (clamped to `0.0..=1.0`).
    pub fn jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// Set the retryability rule.
    pub fn retry_on(mut self, retry_on: RetryOn) -> Self {
        self.retry_on = retry_on;
        self
    }

    /// The delay to wait before a retry, given the 1-based `attempt` number
    /// (attempt `1` is the first retry) and the error that triggered it.
    fn delay_for(&self, attempt: usize, err: &Error) -> Duration {
        // A server-advised `Retry-After` wins over computed backoff (capped by
        // `max_delay`, not jittered — it is an explicit instruction).
        if let Some(secs) = err.retry_after() {
            let capped = secs.min(self.max_delay.as_secs_f64()).max(0.0);
            return Duration::from_secs_f64(capped);
        }
        let exp = self.backoff_multiplier.powi((attempt - 1) as i32);
        let base = self.initial_delay.as_secs_f64() * exp;
        let capped = base.min(self.max_delay.as_secs_f64());
        let jittered = capped * jitter_factor(self.jitter);
        Duration::from_secs_f64(jittered.max(0.0))
    }
}

/// A cheap jitter multiplier in `[1 - jitter, 1.0]`, without a `rand`
/// dependency: entropy comes from the current wall-clock nanoseconds mixed
/// with a process-lifetime counter (so repeated calls within the same
/// nanosecond still differ). `jitter <= 0` returns `1.0` (no jitter).
fn jitter_factor(jitter: f64) -> f64 {
    let jitter = jitter.clamp(0.0, 1.0);
    if jitter == 0.0 {
        return 1.0;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mixed = nanos ^ COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    // Map to [0, 1) via the top 53 bits (f64 mantissa width).
    let r = (mixed >> 11) as f64 / ((1u64 << 53) as f64);
    1.0 - jitter * r
}

/// A [`ChatClient`] decorator that retries transient failures with exponential
/// backoff, honoring a server `Retry-After` when present.
///
/// Wraps any inner [`ChatClient`] and re-issues the request per its
/// [`RetryPolicy`]. For streaming, only the *initial connection* is retried:
/// if establishing the stream (or its very first item, before anything is
/// yielded to the consumer) fails with a retryable error, the connection is
/// re-attempted; once the first update flows, later stream errors propagate
/// unchanged.
///
/// ```no_run
/// # use std::time::Duration;
/// # use agent_framework_core::client::{RetryingChatClient, RetryPolicy};
/// # use agent_framework_core::prelude::*;
/// # fn demo(inner: impl ChatClient + 'static) {
/// let client = RetryingChatClient::new(inner)
///     .with_policy(RetryPolicy::with_max_retries(5).initial_delay(Duration::from_millis(200)));
/// # let _ = client;
/// # }
/// ```
pub struct RetryingChatClient<C: ChatClient> {
    inner: C,
    policy: RetryPolicy,
}

impl<C: ChatClient> RetryingChatClient<C> {
    /// Wrap `inner` with the default [`RetryPolicy`].
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            policy: RetryPolicy::default(),
        }
    }

    /// Set the retry policy (builder-style).
    pub fn with_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// A reference to the wrapped client.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// A reference to the active retry policy.
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    /// Sleep before a retry, emitting a tracing warning describing the attempt.
    async fn backoff(&self, attempt: usize, err: &Error) {
        let delay = self.policy.delay_for(attempt, err);
        tracing::warn!(
            attempt,
            max_retries = self.policy.max_retries,
            delay_ms = delay.as_millis() as u64,
            retry_after = err.retry_after(),
            status = err.status(),
            error = %err,
            "retrying chat request after transient error"
        );
        tokio::time::sleep(delay).await;
    }
}

#[async_trait]
impl<C: ChatClient> ChatClient for RetryingChatClient<C> {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut attempt = 0usize;
        loop {
            match self
                .inner
                .get_response(messages.clone(), options.clone())
                .await
            {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    if attempt >= self.policy.max_retries || !self.policy.retry_on.should_retry(&e)
                    {
                        return Err(e);
                    }
                    attempt += 1;
                    self.backoff(attempt, &e).await;
                }
            }
        }
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let mut attempt = 0usize;
        loop {
            let established = self
                .inner
                .get_streaming_response(messages.clone(), options.clone())
                .await;
            match established {
                // The stream opened: peek its first item. An error there (with
                // nothing yet yielded to the consumer) is still an
                // initial-connection failure and is eligible for retry; any Ok
                // item — or a non-retryable / retries-exhausted error — is
                // handed back with the rest of the stream chained after it.
                Ok(mut stream) => match stream.next().await {
                    Some(Err(e))
                        if attempt < self.policy.max_retries
                            && self.policy.retry_on.should_retry(&e) =>
                    {
                        attempt += 1;
                        self.backoff(attempt, &e).await;
                        continue;
                    }
                    Some(first) => {
                        let head = stream::once(async move { first });
                        return Ok(head.chain(stream).boxed());
                    }
                    None => return Ok(stream::empty().boxed()),
                },
                // The stream never opened (e.g. a non-success HTTP status).
                Err(e) => {
                    if attempt >= self.policy.max_retries || !self.policy.retry_on.should_retry(&e)
                    {
                        return Err(e);
                    }
                    attempt += 1;
                    self.backoff(attempt, &e).await;
                }
            }
        }
    }

    fn model(&self) -> Option<&str> {
        self.inner.model()
    }
}

#[cfg(test)]
mod approval_replacement_tests {
    use super::*;
    use crate::types::{FunctionApprovalRequestContent, FunctionArguments};

    fn call(call_id: &str) -> FunctionCallContent {
        FunctionCallContent::new(call_id, "get_weather", None)
    }

    fn approval_request(call_id: &str) -> Message {
        Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionApprovalRequest(
                FunctionApprovalRequestContent {
                    id: format!("req_{call_id}"),
                    function_call: call(call_id),
                },
            )],
        )
    }

    fn function_calls(messages: &[Message]) -> Vec<&str> {
        messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .filter_map(Content::as_function_call)
            .map(|fc| fc.call_id.as_str())
            .collect()
    }

    /// A call carrying an explicit occurrence id.
    fn identified_call(call_id: &str, occurrence: &str) -> FunctionCallContent {
        let mut c = call(call_id);
        c.id = Some(occurrence.to_string());
        c
    }

    fn identified_request(call_id: &str, occurrence: &str) -> Message {
        Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionApprovalRequest(
                FunctionApprovalRequestContent {
                    id: occurrence.to_string(),
                    function_call: identified_call(call_id, occurrence),
                },
            )],
        )
    }

    // region: occurrence identity

    #[test]
    fn two_pending_approvals_sharing_a_call_id_stay_distinct() {
        // The case occurrence ids exist for. Providers reuse `call_id`, so two
        // approvals outstanding at once under one id were indistinguishable:
        // the second request looked like a replay of the first and was
        // dropped, leaving one of the two calls never declared. With distinct
        // occurrence ids both are genuine invocations and both must expand.
        let mut messages = vec![
            identified_request("c1", "af-call-one"),
            identified_request("c1", "af-call-two"),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(function_calls(&messages), vec!["c1", "c1"]);
    }

    #[test]
    fn a_replayed_request_for_the_same_occurrence_is_still_deduped() {
        // The counterpart: same occurrence id really is the same invocation,
        // so a replay collapses exactly as it did before.
        let mut messages = vec![
            identified_request("c1", "af-call-one"),
            identified_request("c1", "af-call-one"),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(function_calls(&messages), vec!["c1"]);
    }

    #[test]
    fn approved_results_are_matched_by_occurrence_id() {
        // Two approvals under one `call_id`, each with its own result. Keying
        // by `call_id` alone would let one result answer both.
        let mut approved: HashMap<String, FunctionResultContent> = HashMap::new();
        approved.insert(
            "af-call-one".into(),
            FunctionResultContent {
                call_id: "c1".into(),
                result: Some(Value::String("first".into())),
                exception: None,
            },
        );
        approved.insert(
            "af-call-two".into(),
            FunctionResultContent {
                call_id: "c1".into(),
                result: Some(Value::String("second".into())),
                exception: None,
            },
        );

        let mut messages = vec![Message::with_contents(
            Role::assistant(),
            vec![
                Content::FunctionApprovalResponse(FunctionApprovalResponseContent {
                    approved: true,
                    id: "af-call-one".into(),
                    function_call: identified_call("c1", "af-call-one"),
                }),
                Content::FunctionApprovalResponse(FunctionApprovalResponseContent {
                    approved: true,
                    id: "af-call-two".into(),
                    function_call: identified_call("c1", "af-call-two"),
                }),
            ],
        )];
        replace_approval_contents_with_results(&mut messages, &approved);

        let results: Vec<&str> = messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .filter_map(Content::as_function_result)
            .filter_map(|fr| fr.result.as_ref().and_then(Value::as_str))
            .collect();
        assert_eq!(results, vec!["first", "second"]);
    }

    #[test]
    fn a_legacy_approval_without_an_occurrence_id_still_resolves() {
        // State written before occurrence ids existed must keep working: the
        // result is keyed by `call_id` and found by the structural fallback.
        let mut approved: HashMap<String, FunctionResultContent> = HashMap::new();
        approved.insert(
            "c1".into(),
            FunctionResultContent {
                call_id: "c1".into(),
                result: Some(Value::String("legacy".into())),
                exception: None,
            },
        );
        let mut messages = vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionApprovalResponse(
                FunctionApprovalResponseContent {
                    approved: true,
                    id: "req_c1".into(),
                    function_call: call("c1"),
                },
            )],
        )];
        replace_approval_contents_with_results(&mut messages, &approved);
        let results: Vec<&str> = messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .filter_map(Content::as_function_result)
            .filter_map(|fr| fr.result.as_ref().and_then(Value::as_str))
            .collect();
        assert_eq!(results, vec!["legacy"]);
    }

    // endregion

    #[test]
    fn an_approval_request_in_a_separate_message_does_not_duplicate_the_call() {
        // The round-trip shape: a hosting layer replays the stored function
        // call and its approval request as two *separate* assistant messages.
        // Deduping per-message never fired, so the request restored a second
        // copy of the call and only one copy received a result — the provider
        // then rejects the unanswered one.
        let mut messages = vec![
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call("c1"))]),
            approval_request("c1"),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(function_calls(&messages), vec!["c1"]);
    }

    #[test]
    fn two_approval_requests_for_one_call_expand_only_once() {
        let mut messages = vec![approval_request("c1"), approval_request("c1")];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(function_calls(&messages), vec!["c1"]);
    }

    #[test]
    fn a_single_approval_request_still_expands() {
        let mut messages = vec![approval_request("c1")];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(function_calls(&messages), vec!["c1"]);
    }

    #[test]
    fn a_second_unanswered_call_reusing_an_id_still_suppresses_the_request() {
        // The completed c1 pair must not mask the *second*, still-unanswered c1
        // call: expanding the replayed request here would hand the provider two
        // unanswered copies. "Any result exists for this id" cannot tell these
        // apart from the reuse-after-completion case above; a call/result count
        // can.
        let mut messages = vec![
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call("c1"))]),
            Message::with_contents(
                Role::tool(),
                vec![Content::FunctionResult(FunctionResultContent {
                    call_id: "c1".into(),
                    result: Some(Value::String("sunny".into())),
                    exception: None,
                })],
            ),
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call("c1"))]),
            approval_request("c1"),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(function_calls(&messages), vec!["c1", "c1"]);
    }

    #[test]
    fn a_different_invocation_reusing_an_id_is_not_suppressed() {
        // Same call_id, different arguments — a distinct invocation. Judging by
        // id alone dropped the new call while its approval response still
        // executed, attaching that result to the older call on the wire.
        let older = FunctionCallContent::new(
            "c1",
            "get_weather",
            Some(FunctionArguments::Raw("{\"city\":\"old\"}".into())),
        );
        let newer = FunctionCallContent::new(
            "c1",
            "get_weather",
            Some(FunctionArguments::Raw("{\"city\":\"new\"}".into())),
        );
        let mut messages = vec![
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(older)]),
            Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionApprovalRequest(
                    FunctionApprovalRequestContent {
                        id: "req_1".into(),
                        function_call: newer,
                    },
                )],
            ),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());

        let cities: Vec<String> = messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .filter_map(Content::as_function_call)
            .filter_map(|c| {
                c.parse_arguments()
                    .ok()?
                    .get("city")?
                    .as_str()
                    .map(str::to_string)
            })
            .collect();
        assert!(
            cities.contains(&"new".to_string()),
            "the new invocation must survive, got {cities:?}"
        );
    }

    #[test]
    fn two_replayed_requests_for_one_outstanding_call_both_collapse() {
        // Removing the *request* answers nothing, so the call stays
        // outstanding. Consuming the anchor let the second replayed copy expand
        // into a duplicate declaration with only one result to answer it.
        let mut messages = vec![
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call("c1"))]),
            approval_request("c1"),
            approval_request("c1"),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(function_calls(&messages), vec!["c1"]);
    }

    #[test]
    fn a_completed_approval_cycle_does_not_suppress_the_next_one() {
        // The first request expands and its response answers it. A later cycle
        // reusing the same invocation must still expand: a stale outstanding
        // entry suppressed it while its own response still converted, leaving
        // two results for the old call and no declaration for the new one.
        let approved = FunctionApprovalRequestContent {
            id: "req_1".into(),
            function_call: call("c1"),
        }
        .create_response(false);
        let mut messages = vec![
            approval_request("c1"),
            Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionApprovalResponse(approved)],
            ),
            approval_request("c1"),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(
            function_calls(&messages),
            vec!["c1", "c1"],
            "the second cycle needs its own declaration"
        );
    }

    #[test]
    fn a_result_arriving_after_the_request_still_suppresses_it() {
        // Order matters: at the moment the replayed request is seen, the call
        // is still unanswered — the result only arrives later in the list. A
        // whole-list pre-scan nets the call against that future result,
        // concludes nothing is outstanding, and expands the request into a
        // second declaration with a single result between them.
        let mut messages = vec![
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call("c1"))]),
            approval_request("c1"),
            Message::with_contents(
                Role::tool(),
                vec![Content::FunctionResult(FunctionResultContent {
                    call_id: "c1".into(),
                    result: Some(Value::String("sunny".into())),
                    exception: None,
                })],
            ),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(function_calls(&messages), vec!["c1"]);
    }

    #[test]
    fn a_request_replayed_before_its_call_still_collapses_to_one() {
        // The replay order is not guaranteed: the approval request can precede
        // the stored call. Whichever comes second is the duplicate.
        let mut messages = vec![
            approval_request("c1"),
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call("c1"))]),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        assert_eq!(function_calls(&messages), vec!["c1"]);
    }

    #[test]
    fn an_already_answered_call_does_not_suppress_a_reused_call_id() {
        // Reusing a call id for a later invocation is supported: the completed
        // pair must not suppress the fresh request, which would drop the new
        // call and leave its result attached to the old one.
        let mut messages = vec![
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call("c1"))]),
            Message::with_contents(
                Role::tool(),
                vec![Content::FunctionResult(FunctionResultContent {
                    call_id: "c1".into(),
                    result: Some(Value::String("sunny".into())),
                    exception: None,
                })],
            ),
            approval_request("c1"),
        ];
        replace_approval_contents_with_results(&mut messages, &HashMap::new());
        // Both the completed call and the freshly restored one are present.
        assert_eq!(function_calls(&messages), vec!["c1", "c1"]);
    }
}
