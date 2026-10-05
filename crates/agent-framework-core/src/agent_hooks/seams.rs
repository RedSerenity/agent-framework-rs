//! The three enforcement seams and the per-run state they share.
//!
//! Private by design (upstream keeps its middleware trio private and ships
//! it as one indivisible `MiddlewareBundle`; .NET keeps its decorators
//! `internal`): installing only part of them would enforce only part of the
//! control contract, so the only way to obtain them is
//! [`AgentHooks::agent_builder`](super::AgentHooks::agent_builder), which
//! installs all three.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::codecs::{
    finish_reason_str, usage_to_wire, InputCodec, ModelRequestCodec, ModelResponseCodec,
    OutputCodec, ToolArgumentsCodec, ToolResultCodec,
};
use super::protocol::{
    AgentContextBuilder, InterceptionBlocked, InterceptionEmitter, HOST_ERROR_PREFIX,
};
use super::AgentHooksConfig;
use crate::client::{ChatClient, ChatStream};
use crate::error::{Error, Result};
use crate::middleware::{AgentContext, FunctionInvocationContext, Middleware, Next};
use crate::types::{AgentResponse, ChatOptions, ChatResponse, ChatResponseUpdate, Message};

/// The `framework` recorded on agent contexts.
pub(crate) const FRAMEWORK_NAME: &str = "agent-framework";

const TRIO_REQUIRED: &str = "middleware was invoked without an active agent-hooks run. The \
     agent-hooks seams must be installed as one unit, via AgentHooks::agent_builder(client).";

const FOREIGN_TRIO: &str = "found an active agent-hooks run owned by a different agent-hooks \
     installation. Stacking multiple installations on one agent is not supported: emissions \
     would silently bind to the wrong emitter. Install exactly one per agent.";

/// Per-run enforcement state shared by the three seams (upstream `_RunState`
/// in a `ContextVar`; .NET `AgentHooksRunState` in an `AsyncLocal`). Here it
/// rides a tokio task-local scoped around the agent pipeline's descent, so
/// concurrent runs are isolated and a nested guarded agent's scope shadows
/// its parent's for exactly the nested run.
pub(crate) struct RunState {
    pub(crate) emitter: Arc<InterceptionEmitter>,
    pub(crate) builder: Arc<AgentContextBuilder>,
    /// Whether the session (and its startup/shutdown bracket) is host-owned.
    pub(crate) session_scoped: bool,
    pub(crate) config: Arc<AgentHooksConfig>,
    /// Set when a tool-seam `host_error:*` block halted the run: the agent
    /// seam surfaces it as the run's error (one deny surface at every seam).
    pub(crate) halted: Mutex<Option<InterceptionBlocked>>,
}

tokio::task_local! {
    static RUN_STATE: Arc<RunState>;
}

/// The run state owned by `config`, or the fail-closed error for a partial
/// or foreign install.
fn current_state(config: &Arc<AgentHooksConfig>, seam: &str) -> Result<Arc<RunState>> {
    let state = RUN_STATE
        .try_with(Arc::clone)
        .map_err(|_| Error::MiddlewareFailure(format!("agent-hooks {seam} {TRIO_REQUIRED}")))?;
    if !Arc::ptr_eq(&state.config, config) {
        return Err(Error::MiddlewareFailure(format!(
            "agent-hooks {seam} {FOREIGN_TRIO}"
        )));
    }
    Ok(state)
}

fn as_blocked(err: Error) -> std::result::Result<InterceptionBlocked, Error> {
    match err {
        Error::InterceptionBlocked(b) => Ok(*b),
        other => Err(other),
    }
}

fn is_host_error(block: &InterceptionBlocked) -> bool {
    block
        .record
        .verdict
        .reason
        .as_deref()
        .is_some_and(|r| r.starts_with(HOST_ERROR_PREFIX))
}

// region Agent seam

/// Run bracket: `agent_startup`, `input`, `output`, `agent_shutdown`
/// (upstream `_AgentHooksAgentMiddleware`; .NET `AgentHooksAgent`).
///
/// Streaming needs no extra machinery here: an [`Agent`](crate::agent::Agent)
/// with agent middleware runs streaming requests through the same pipeline
/// and only replays the (verdicted) response as updates after it returns, so
/// no partial content ever egresses ahead of the `output` verdict and a deny
/// releases zero updates — upstream's fail-closed buffering, by construction.
pub(crate) struct AgentSeam {
    pub(crate) config: Arc<AgentHooksConfig>,
}

impl AgentSeam {
    fn new_run_state(&self, ctx: &AgentContext) -> RunState {
        let config = &self.config;
        if let (Some(emitter), Some(builder)) = (&config.emitter, &config.builder) {
            return RunState {
                emitter: emitter.clone(),
                builder: builder.clone(),
                session_scoped: true,
                config: config.clone(),
                halted: Mutex::new(None),
            };
        }
        let agent_id = ctx
            .agent_id
            .clone()
            .or_else(|| ctx.agent_name.clone())
            .unwrap_or_else(|| "agent".into());
        let builder = AgentContextBuilder::new(
            agent_id,
            FRAMEWORK_NAME,
            uuid::Uuid::new_v4().simple().to_string(),
        )
        .with_agent_name(ctx.agent_name.clone());
        let mut emitter = InterceptionEmitter::new()
            .with_mode(config.mode)
            .with_timeout(config.timeout)
            .with_identity_provider(config.identity_provider.clone());
        if let Some(resolver) = &config.resolver {
            emitter = emitter.with_resolver(resolver.clone());
        }
        if let Some(composition) = config.composition {
            emitter = emitter.with_composition(composition);
        }
        if let Some(sink) = &config.record_sink {
            emitter = emitter.with_record_sink(sink.clone());
        }
        for (name, interceptor) in &config.interceptors {
            emitter = emitter.register_arc(name.clone(), interceptor.clone());
        }
        RunState {
            emitter: Arc::new(emitter),
            builder: Arc::new(builder),
            session_scoped: false,
            config: config.clone(),
            halted: Mutex::new(None),
        }
    }

    /// Emit `agent_startup` (per-run sessions) and `input`; apply input
    /// transforms to the run input (and to what the session persists).
    async fn emit_run_start(state: &RunState, ctx: &mut AgentContext) -> Result<()> {
        if !state.session_scoped {
            let names = ctx.tools.iter().map(|t| t.name.clone()).collect();
            state
                .emitter
                .emit(state.builder.agent_startup(names))
                .await?;
        }
        let start = ctx.input_start.min(ctx.messages.len());
        let mut input: Vec<Message> = ctx.messages[start..].to_vec();
        let (content, role) = InputCodec::to_wire(&input);
        let before = json!({ "content": content, "role": role });
        let outcome = state
            .emitter
            .emit(state.builder.input(content, &role))
            .await?;
        if InputCodec::write_back(&mut input, &before, &outcome.target)? {
            ctx.messages.truncate(start);
            ctx.messages.extend(input.iter().cloned());
            ctx.persisted_input = Some(input);
        }
        Ok(())
    }

    /// Emit `output` over the assembled response; apply output transforms.
    async fn emit_output(state: &RunState, response: &mut AgentResponse) -> Result<bool> {
        let before = OutputCodec::to_wire(response);
        let outcome = state
            .emitter
            .emit(state.builder.output(before.clone()))
            .await?;
        OutputCodec::write_back(response, &before, &outcome.target)
    }

    /// Best-effort `agent_shutdown` (per-run sessions only; blocks there are
    /// record-only).
    async fn emit_shutdown(state: &RunState, reason: &str) {
        if !state.session_scoped {
            state
                .emitter
                .emit_unchecked(state.builder.agent_shutdown(reason))
                .await;
        }
    }

    async fn run(
        state: Arc<RunState>,
        mut ctx: AgentContext,
        next: Next<AgentContext>,
    ) -> Result<AgentContext> {
        let result = async {
            Self::emit_run_start(&state, &mut ctx).await?;
            let mut ctx = match next.run(ctx).await {
                Ok(ctx) => ctx,
                Err(Error::MiddlewareFailure(msg)) => {
                    // A fail-closed abort from behind the function-invocation
                    // loop: surface this run's own tool-seam block as the block
                    // itself; any other failure propagates exactly as raised.
                    return Err(match state.halted.lock().unwrap().take() {
                        Some(block) => Error::InterceptionBlocked(Box::new(block)),
                        None => Error::MiddlewareFailure(msg),
                    });
                }
                Err(e) => return Err(e),
            };
            if let Some(block) = state.halted.lock().unwrap().take() {
                // The enforcement layer failed mid-run even though something
                // between the seams swallowed the abort: still fail closed.
                return Err(Error::InterceptionBlocked(Box::new(block)));
            }
            // A substituted (short-circuit) result still egresses, so it
            // passes the output point too; no result means nothing egresses.
            if let Some(response) = ctx.result.as_mut() {
                Self::emit_output(&state, response).await?;
            }
            Ok(ctx)
        }
        .await;
        let reason = if result.is_ok() { "completed" } else { "error" };
        Self::emit_shutdown(&state, reason).await;
        result
    }
}

#[async_trait]
impl Middleware<AgentContext> for AgentSeam {
    async fn process(&self, ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
        let state = Arc::new(self.new_run_state(&ctx));
        RUN_STATE
            .scope(state.clone(), Self::run(state, ctx, next))
            .await
    }
}

// endregion

// region Chat seam

/// Model bracket: `pre_model_call` and `post_model_call` around every
/// individual model service call (upstream `_AgentHooksChatMiddleware`;
/// .NET `AgentHooksChatClient`).
///
/// A [`ChatClient`] decorator on the raw client, *below*
/// [`FunctionInvokingChatClient`](crate::client::FunctionInvokingChatClient)
/// — as in .NET. (This port's chat-middleware pipeline wraps the whole tool
/// loop rather than each service call, so a chat middleware could not
/// bracket calls individually the way upstream Python's does.)
pub(crate) struct ChatSeam<C> {
    pub(crate) inner: C,
    pub(crate) config: Arc<AgentHooksConfig>,
}

impl<C: ChatClient> ChatSeam<C> {
    fn model_id(&self, options: &ChatOptions) -> String {
        options
            .model
            .clone()
            .or_else(|| self.inner.model().map(str::to_string))
            .unwrap_or_else(|| {
                std::any::type_name::<C>()
                    .rsplit("::")
                    .next()
                    .unwrap_or("ChatClient")
                    .to_string()
            })
    }

    /// The per-call effective tool set (`{name, description?}`), omitted when
    /// the call offers no tools.
    fn tools(options: &ChatOptions) -> Option<Value> {
        (!options.tools.is_empty()).then(|| {
            Value::Array(
                options
                    .tools
                    .iter()
                    .map(|t| {
                        let mut entry = json!({ "name": t.name });
                        if !t.description.is_empty() {
                            entry["description"] = json!(t.description);
                        }
                        entry
                    })
                    .collect(),
            )
        })
    }

    async fn pre_model_call(
        &self,
        state: &RunState,
        model_id: &str,
        messages: Vec<Message>,
        options: &ChatOptions,
    ) -> Result<Vec<Message>> {
        let before = ModelRequestCodec::to_wire(&messages);
        let outcome = state
            .emitter
            .emit(state.builder.pre_model_call(
                model_id,
                Value::Array(before.clone()),
                Self::tools(options),
                None,
            ))
            .await?;
        ModelRequestCodec::write_back(messages, &before, &outcome.target)
    }

    /// Emit `post_model_call`; apply transforms. Returns whether it changed.
    async fn post_model_call(
        state: &RunState,
        model_id: &str,
        response: &mut ChatResponse,
    ) -> Result<bool> {
        let before = ModelResponseCodec::to_wire(response);
        let outcome = state
            .emitter
            .emit(state.builder.post_model_call(
                response.model.as_deref().unwrap_or(model_id),
                before["content"].clone(),
                before["tool_calls"].clone(),
                &finish_reason_str(response.finish_reason.as_ref()),
                usage_to_wire(response.usage_details.as_ref()),
                response.response_id.clone(),
            ))
            .await?;
        ModelResponseCodec::write_back(response, &before, &outcome.target)
    }
}

/// Re-derive stream updates from a verdicted response (one per message;
/// response metadata on every update, the finish reason on the last).
fn response_to_chat_updates(response: &ChatResponse) -> Vec<Result<ChatResponseUpdate>> {
    let last = response.messages.len().saturating_sub(1);
    response
        .messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            Ok(ChatResponseUpdate {
                contents: m.contents.clone(),
                role: Some(m.role.clone()),
                author_name: m.author_name.clone(),
                message_id: m.message_id.clone().or_else(|| Some(format!("msg-{i}"))),
                response_id: response.response_id.clone(),
                conversation_id: response.conversation_id.clone(),
                model: response.model.clone(),
                finish_reason: if i == last {
                    response.finish_reason.clone()
                } else {
                    None
                },
                ..Default::default()
            })
        })
        .collect()
}

#[async_trait]
impl<C: ChatClient> ChatClient for ChatSeam<C> {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let state = current_state(&self.config, "chat")?;
        let model_id = self.model_id(&options);
        let messages = self
            .pre_model_call(&state, &model_id, messages, &options)
            .await?;
        let mut response = self.inner.get_response(messages, options).await?;
        // §6.1: a denied response is never incorporated (the error replaces it).
        Self::post_model_call(&state, &model_id, &mut response).await?;
        Ok(response)
    }

    /// Fail-closed by buffering (spec §12.1): the model stream is fully
    /// consumed, `post_model_call` is applied to the assembled response, and
    /// only then are the updates released — re-derived from the response
    /// when a transform changed it, so egress never diverges from the
    /// verdict. A deny releases nothing.
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let state = current_state(&self.config, "chat")?;
        let model_id = self.model_id(&options);
        let messages = self
            .pre_model_call(&state, &model_id, messages, &options)
            .await?;
        let mut inner = self.inner.get_streaming_response(messages, options).await?;
        let mut buffered = Vec::new();
        while let Some(update) = inner.next().await {
            buffered.push(update?);
        }
        let mut response = ChatResponse::from_updates(buffered.clone());
        let changed = Self::post_model_call(&state, &model_id, &mut response).await?;
        let released: Vec<Result<ChatResponseUpdate>> = if changed {
            response_to_chat_updates(&response)
        } else {
            buffered.into_iter().map(Ok).collect()
        };
        Ok(futures::stream::iter(released).boxed())
    }

    fn model(&self) -> Option<&str> {
        self.inner.model()
    }
}

// endregion

// region Function seam

/// Tool bracket: `pre_tool_call` and `post_tool_call` around every
/// host-executed tool call (upstream `_AgentHooksFunctionMiddleware`; .NET
/// `AgentHooksFunctionMiddleware`).
///
/// A policy deny blocks the call: the tool is not executed (or its result is
/// discarded) and a tool-error payload becomes the result the model sees,
/// so the loop continues. A `host_error:*` deny additionally halts the run
/// via [`Error::MiddlewareFailure`] (the loop's fail-closed escape), and the
/// agent seam surfaces the block itself as [`Error::InterceptionBlocked`].
/// Any other failure inside the enforcement layer (write-back, projection)
/// halts the run as [`Error::MiddlewareFailure`].
///
/// Approval requests need no pass-through here: this port's
/// function-invocation loop resolves approvals before a call ever enters the
/// function-middleware pipeline, so an unapproved tool never reaches this
/// seam and the approved replay enters through `pre_tool_call`.
pub(crate) struct FunctionSeam {
    pub(crate) config: Arc<AgentHooksConfig>,
}

impl FunctionSeam {
    /// Enforce a tool-seam deny: surface a tool error and, on host errors,
    /// halt the run.
    fn block(
        state: &RunState,
        mut ctx: FunctionInvocationContext,
        block: InterceptionBlocked,
        point: &str,
    ) -> Result<FunctionInvocationContext> {
        let verdict = &block.record.verdict;
        let mut payload = json!({
            "error": format!("Tool call blocked by agent-hooks at {point}."),
            "reason": verdict.reason.clone().unwrap_or_else(|| "deny".into()),
        });
        if let Some(message) = &verdict.message {
            payload["message"] = json!(message);
        }
        ctx.result = Some(payload);
        Self::maybe_halt(state, block, point)?;
        Ok(ctx)
    }

    fn maybe_halt(state: &RunState, block: InterceptionBlocked, point: &str) -> Result<()> {
        if is_host_error(&block) {
            let reason = block.record.verdict.reason.clone().unwrap_or_default();
            *state.halted.lock().unwrap() = Some(block);
            return Err(Error::MiddlewareFailure(format!(
                "agent-hooks {point} failed closed: {reason}"
            )));
        }
        Ok(())
    }

    /// Abort fail-closed on an unexpected failure inside the enforcement
    /// layer (upstream `_halt_on_enforcement_failure`).
    fn halt(err: Error, point: &str) -> Error {
        match err {
            Error::MiddlewareFailure(msg) => Error::MiddlewareFailure(msg),
            other => Error::MiddlewareFailure(format!(
                "agent-hooks {point} enforcement failed: {}",
                crate::observability::error_type(&other)
            )),
        }
    }
}

#[async_trait]
impl Middleware<FunctionInvocationContext> for FunctionSeam {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        let state = current_state(&self.config, "function")?;
        let call_id = ctx
            .metadata
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
        let name = ctx.function_name.clone();

        let before = ToolArgumentsCodec::to_wire(&ctx.arguments);
        let args = match state
            .emitter
            .emit(
                state
                    .builder
                    .pre_tool_call(&call_id, &name, Value::Object(before.clone())),
            )
            .await
        {
            Ok(outcome) => {
                match ToolArgumentsCodec::write_back(&ctx.arguments, &before, &outcome.target) {
                    Ok((native, effective)) => {
                        if let Some(native) = native {
                            // Execute exactly the approved arguments.
                            ctx.arguments = native;
                        }
                        Value::Object(effective)
                    }
                    Err(e) => return Err(Self::halt(e, "pre_tool_call")),
                }
            }
            // §6.2: the tool is not dispatched and no post_tool_call is emitted.
            Err(e) => match as_blocked(e) {
                Ok(block) => return Self::block(&state, ctx, block, "pre_tool_call"),
                Err(e) => return Err(Self::halt(e, "pre_tool_call")),
            },
        };

        let ctx = match next.run(ctx).await {
            Ok(ctx) => ctx,
            Err(err) => {
                // The invocation errored: the contract still brackets it
                // (is_error = true). Only the error's type tag crosses the
                // boundary (spec §6.3/§14).
                let ctx_json = state.builder.post_tool_call(
                    &call_id,
                    &name,
                    args,
                    json!(crate::observability::error_type(&err)),
                    true,
                    None,
                );
                if let Err(e) = state.emitter.emit(ctx_json).await {
                    match as_blocked(e) {
                        // A policy deny over an already-errored call changes
                        // nothing; a host error still halts the run.
                        Ok(block) => Self::maybe_halt(&state, block, "post_tool_call")?,
                        Err(e) => return Err(Self::halt(e, "post_tool_call")),
                    }
                }
                return Err(err);
            }
        };

        // A short-circuited (substituted) result still enters the transcript,
        // so it is bracketed like any other.
        let value = ToolResultCodec::to_wire(ctx.result.as_ref());
        let emitted = state
            .emitter
            .emit(
                state
                    .builder
                    .post_tool_call(&call_id, &name, args, value.clone(), false, None),
            )
            .await;
        match emitted {
            Ok(outcome) => {
                let mut ctx = ctx;
                ctx.result =
                    ToolResultCodec::write_back(ctx.result.take(), &value, &outcome.target);
                Ok(ctx)
            }
            // §6.1: the result is discarded as if the call had errored.
            Err(e) => match as_blocked(e) {
                Ok(block) => Self::block(&state, ctx, block, "post_tool_call"),
                Err(e) => Err(Self::halt(e, "post_tool_call")),
            },
        }
    }
}

// endregion

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_hooks::{interceptor_fn, AgentHooks, AgentHooksOptions, Verdict};
    use crate::middleware::MiddlewarePipeline;

    struct Echo;

    #[async_trait]
    impl ChatClient for Echo {
        async fn get_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse::from_text("echo"))
        }
        async fn get_streaming_response(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<ChatStream> {
            Ok(futures::stream::iter(vec![Ok(ChatResponseUpdate::text("echo"))]).boxed())
        }
    }

    fn config() -> Arc<AgentHooksConfig> {
        let allow = interceptor_fn(|_| async { Ok(Verdict::allow()) });
        AgentHooks::new(AgentHooksOptions::new().interceptor(allow))
            .unwrap()
            .config
    }

    fn state_for(config: Arc<AgentHooksConfig>, emitter: InterceptionEmitter) -> Arc<RunState> {
        Arc::new(RunState {
            emitter: Arc::new(emitter),
            builder: Arc::new(AgentContextBuilder::new("a", FRAMEWORK_NAME, "s")),
            session_scoped: true,
            config,
            halted: Mutex::new(None),
        })
    }

    #[tokio::test]
    async fn seams_without_run_state_fail_closed() {
        let chat = ChatSeam {
            inner: Echo,
            config: config(),
        };
        let err = chat
            .get_response(vec![Message::user("x")], ChatOptions::default())
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("without an active agent-hooks run"));

        let pipeline: MiddlewarePipeline<FunctionInvocationContext> =
            MiddlewarePipeline::new(vec![Arc::new(FunctionSeam { config: config() })]);
        let invoked = Arc::new(Mutex::new(false));
        let flag = invoked.clone();
        let err = pipeline
            .execute(
                FunctionInvocationContext::new("tool", json!({})),
                Box::new(move |ctx| {
                    *flag.lock().unwrap() = true;
                    Box::pin(async move { Ok(ctx) })
                }),
            )
            .await
            .err()
            .unwrap();
        assert!(err.is_middleware_failure());
        assert!(!*invoked.lock().unwrap(), "the tool is never dispatched");
    }

    #[tokio::test]
    async fn seams_refuse_a_foreign_installations_run_state() {
        let state = state_for(config(), InterceptionEmitter::new());
        let foreign = ChatSeam {
            inner: Echo,
            config: config(),
        };
        let err = RUN_STATE
            .scope(state, async {
                foreign
                    .get_response(vec![Message::user("x")], ChatOptions::default())
                    .await
            })
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("different agent-hooks installation"));
    }

    #[tokio::test]
    async fn streaming_chat_seam_buffers_and_rederives_on_transform() {
        let cfg = config();
        let transform = interceptor_fn(|ctx: Value| async move {
            Ok(if ctx["interception_point"] == "post_model_call" {
                Verdict::transform("$target.content", json!("rewritten"))
            } else {
                Verdict::allow()
            })
        });
        let state = state_for(cfg.clone(), InterceptionEmitter::new().register(transform));
        let chat = ChatSeam {
            inner: Echo,
            config: cfg,
        };
        let updates: Vec<_> = RUN_STATE
            .scope(state.clone(), async {
                chat.get_streaming_response(vec![Message::user("x")], ChatOptions::default())
                    .await
                    .unwrap()
                    .collect()
                    .await
            })
            .await;
        let text: String = updates
            .into_iter()
            .map(|u| u.unwrap().text_content())
            .collect();
        assert_eq!(text, "rewritten");
        let points: Vec<_> = state
            .emitter
            .results()
            .iter()
            .map(|r| r.interception_point.as_str())
            .collect();
        assert_eq!(points, ["pre_model_call", "post_model_call"]);
    }
}
