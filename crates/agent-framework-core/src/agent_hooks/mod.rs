//! AGENT-HOOKS-0.1 enforcement for agents (experimental).
//!
//! Rust equivalent of upstream `agent_framework._agent_hooks`
//! (`create_agent_hooks_middleware` / `create_agent_hooks_middleware_from_emitter`)
//! and .NET `Microsoft.Agents.AI.AgentHooks`
//! (`AgentHooksChatClientExtensions.AsAIAgentWithAgentHooks`). It implements
//! the [agent-hooks](https://github.com/responsibleai/agent-hooks) control
//! contract as one coherent feature on the framework's three seams:
//!
//! - `agent_startup` / `input` / `output` / `agent_shutdown` ride the agent
//!   seam (an agent middleware),
//! - `pre_model_call` / `post_model_call` ride the chat seam (a chat-client
//!   decorator below the function-invocation loop, so every model service
//!   call is bracketed individually — the .NET placement),
//! - `pre_tool_call` / `post_tool_call` ride the function seam (a function
//!   middleware).
//!
//! The seams are private and installed as one indivisible unit by
//! [`AgentHooks::agent_builder`], which returns an
//! [`AgentBuilder`] with all three already in place and *first* in their
//! pipelines: middleware added to the returned builder runs inside the
//! enforcement boundary of the agent and function seams (outer position is
//! outer trust; the final `output` point still guards whatever egresses).
//! A function middleware that substitutes a tool result, for example, is
//! bracketed by `post_tool_call`.
//!
//! The wire contract itself — verdicts, composition profiles, the approval
//! and identity seams, the context builder and the emitter — lives in
//! [`protocol`] (upstream's external `agent-hooks-sdk`).
//!
//! # Enforcement semantics (`EnforcementMode::Enforce`)
//!
//! - Every point is emitted **before** the guarded action runs (pre points)
//!   or before its result is incorporated (post points). Emission failures
//!   (interceptor error/panic/timeout, invalid context) synthesize
//!   `host_error:*` denies and are treated as blocks — the feature never
//!   fails open.
//! - `transform` verdicts are written back into the native values (run
//!   input, request messages, tool arguments, tool results, chat response,
//!   agent response) through per-point codecs, so the framework executes
//!   exactly what the interceptors approved. Rich content is preserved as
//!   content objects, never flattened to text.
//! - A deny at `input`, `pre_model_call`, `post_model_call` or `output`
//!   terminates the run with [`Error::InterceptionBlocked`] carrying the
//!   record. For streaming runs the error surfaces before any update is
//!   released (see below).
//! - A deny at `pre_tool_call` / `post_tool_call` blocks the tool call: the
//!   tool is not executed (or its result is discarded) and a tool-error
//!   payload `{"error", "reason", "message"?}` is what the model sees, so the
//!   loop continues. A `host_error:*` deny at the tool seam additionally
//!   halts the run (the enforcement layer itself failed) and surfaces as
//!   [`Error::InterceptionBlocked`] too; other failures inside the
//!   enforcement layer at the tool seam (an unappliable transform) halt it
//!   as [`Error::MiddlewareFailure`].
//! - Short-circuits are guarded: a result substituted by inner middleware
//!   still passes `output` / `post_tool_call` before it egresses or enters
//!   the transcript.
//!
//! # Persistence and streaming
//!
//! Upstream defers durable history writes behind the covering verdict with a
//! run persistence gate. In this port the same guarantees hold by
//! construction: context providers (and so history providers) persist only
//! after a run *succeeds*, and every verdict is applied before the run
//! returns — so denied content never becomes durable, transformed output is
//! persisted post-transform, and a transformed run input is persisted in its
//! rewritten form (via [`AgentContext::persisted_input`](crate::middleware::AgentContext::persisted_input)).
//! Nested agents (sub-agents invoked as tools) persist at their own run
//! boundaries, so an outer deny never discards fully-permitted inner
//! history. Likewise, an [`Agent`] with agent middleware
//! replays streaming runs from the completed (verdicted) response, so
//! streaming is fail-closed by buffering exactly like upstream: no partial
//! content egresses ahead of a verdict and a deny releases zero updates.
//!
//! # Session scoping
//!
//! By default each run is one agent-hooks session: a fresh emitter and
//! context builder per run, bracketed by `agent_startup` / `agent_shutdown`.
//! A host owning a longer-lived session builds its own
//! [`InterceptionEmitter`] and [`AgentContextBuilder`] and uses
//! [`AgentHooks::from_emitter`]; only the per-run points are then emitted
//! and the host owns the session boundaries.
//!
//! # Known limitations
//!
//! - Tools executed by the model provider itself (hosted MCP, web search,
//!   code interpreter) never pass through the function seam; their calls and
//!   outputs are surfaced in the `post_model_call` content projection, where
//!   interceptors can deny/transform the response that carries them.
//! - `agent_startup.tools_registered` is the tool set resolved at run start,
//!   which in this port already includes context-provider tools and
//!   resolved tool sources (they are resolved before the agent pipeline).
//! - Run cancellation is drop-based: dropping a run future mid-flight emits
//!   no `agent_shutdown` (upstream emits `reason: "cancelled"`).
//! - Upstream's telemetry feature bit (`mark_feature_used`) has no
//!   equivalent here.
//!
//! # Example
//!
//! ```no_run
//! use agent_framework_core::agent_hooks::{interceptor_fn, AgentHooks, AgentHooksOptions, Verdict};
//! use agent_framework_core::prelude::*;
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let egress_guard = interceptor_fn(|ctx: serde_json::Value| async move {
//!     if ctx["interception_point"] == "output" && ctx["target"].to_string().contains("secret") {
//!         return Ok(Verdict::deny("egress_blocked"));
//!     }
//!     Ok(Verdict::allow())
//! });
//! let hooks = AgentHooks::new(AgentHooksOptions::new().interceptor(egress_guard))?;
//! let agent = hooks.agent_builder(client).name("assistant").build();
//! match agent.run_once("Hello!").await {
//!     Err(Error::InterceptionBlocked(blocked)) => println!("blocked: {blocked}"),
//!     other => println!("{}", other?.text()),
//! }
//! # Ok(())
//! # }
//! ```

mod codecs;
pub mod protocol;
mod seams;

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

pub use protocol::{
    canonical_json, context_identity, interceptor_fn, resolver_fn, AgentContextBuilder,
    ApprovalOutcome, ApprovalRedactor, ApprovalRequest, ApprovalResolution, ApprovalResolver,
    CompositionConfig, CompositionProfile, Decision, EmitOutcome, EnforcementMode, Evidence,
    FnApprovalResolver, FnInterceptor, HostError, IdentityProvider, InterceptionBlocked,
    InterceptionEmitter, InterceptionPoint, InterceptionRecord, Interceptor, OnApproval,
    RecordSink, SynthesisPolicy, Transform, Verdict, VerdictSummary, VerdictWarning,
    DEFAULT_TIMEOUT, JCS_SHA256, SPEC_VERSION,
};

use crate::agent::{Agent, AgentBuilder};
use crate::client::ChatClient;
use crate::error::{Error, Result};

/// Configuration shared by the seams of one installation. Its `Arc` identity
/// is the ownership token: a seam only binds to run state created by its own
/// installation (upstream compares the bundle's config by identity; .NET
/// uses `ReferenceEquals` on `AgentHooksConfiguration`).
pub(crate) struct AgentHooksConfig {
    pub(crate) interceptors: Vec<(Option<String>, Arc<dyn Interceptor>)>,
    pub(crate) resolver: Option<Arc<dyn ApprovalResolver>>,
    pub(crate) mode: EnforcementMode,
    pub(crate) composition: Option<CompositionConfig>,
    pub(crate) identity_provider: Option<IdentityProvider>,
    pub(crate) timeout: Option<Duration>,
    pub(crate) record_sink: Option<RecordSink>,
    /// Host-owned session: when set, only the per-run points are emitted.
    pub(crate) emitter: Option<Arc<InterceptionEmitter>>,
    /// Host-owned session: the builder matching `emitter`.
    pub(crate) builder: Option<Arc<AgentContextBuilder>>,
}

/// Options for per-run-session enforcement. Mirrors the keyword arguments of
/// upstream `create_agent_hooks_middleware` and .NET `AgentHooksOptions`.
///
/// Defaults: `enforce` mode, no resolver, the SDK's default composition
/// (`sequential/first_deny`, `on_approval: stop`), `jcs-sha256` identity, a
/// 5-second timeout, no record sink. At least one interceptor is required.
#[derive(Clone)]
pub struct AgentHooksOptions {
    interceptors: Vec<(Option<String>, Arc<dyn Interceptor>)>,
    resolver: Option<Arc<dyn ApprovalResolver>>,
    mode: EnforcementMode,
    composition: Option<CompositionConfig>,
    identity_provider: Option<IdentityProvider>,
    timeout: Option<Duration>,
    record_sink: Option<RecordSink>,
}

impl Default for AgentHooksOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for AgentHooksOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentHooksOptions")
            .field(
                "interceptors",
                &self.interceptors.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            )
            .field("resolver", &self.resolver.is_some())
            .field("mode", &self.mode)
            .field("composition", &self.composition)
            .field("identity_provider", &self.identity_provider)
            .field("timeout", &self.timeout)
            .field("record_sink", &self.record_sink.is_some())
            .finish()
    }
}

impl AgentHooksOptions {
    /// Options with the defaults and no interceptors yet.
    pub fn new() -> Self {
        Self {
            interceptors: Vec::new(),
            resolver: None,
            mode: EnforcementMode::Enforce,
            composition: None,
            identity_provider: Some(IdentityProvider::JcsSha256),
            timeout: Some(DEFAULT_TIMEOUT),
            record_sink: None,
        }
    }

    /// Register an interceptor (upstream's sequence form).
    pub fn interceptor(self, interceptor: impl Interceptor + 'static) -> Self {
        self.interceptor_arc(None, Arc::new(interceptor))
    }

    /// Register an interceptor under a payload-free name recorded on the
    /// records' verdict summaries (upstream's mapping form; .NET
    /// `AddInterceptor(interceptor, name)`).
    pub fn named_interceptor(
        self,
        name: impl Into<String>,
        interceptor: impl Interceptor + 'static,
    ) -> Self {
        self.interceptor_arc(Some(name.into()), Arc::new(interceptor))
    }

    /// Register a shared interceptor with an optional name.
    pub fn interceptor_arc(
        mut self,
        name: Option<String>,
        interceptor: Arc<dyn Interceptor>,
    ) -> Self {
        self.interceptors.push((name, interceptor));
        self
    }

    /// The optional approval resolver consulted for liftable denies.
    pub fn resolver(mut self, resolver: impl ApprovalResolver + 'static) -> Self {
        self.resolver = Some(Arc::new(resolver));
        self
    }

    /// `Enforce` (default) honours verdicts; `EvaluateOnly` records them
    /// without acting.
    pub fn mode(mut self, mode: EnforcementMode) -> Self {
        self.mode = mode;
        self
    }

    /// The composition profile and knobs (default: the SDK default).
    pub fn composition(mut self, composition: CompositionConfig) -> Self {
        self.composition = Some(composition);
        self
    }

    /// The identity provider: `Some(IdentityProvider::JcsSha256)` (default),
    /// a custom provider, or `None` for identity-unbound records.
    pub fn identity_provider(mut self, provider: Option<IdentityProvider>) -> Self {
        self.identity_provider = provider;
        self
    }

    /// The per-interceptor/resolver timeout (default 5 s); `None` disables
    /// it.
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// A callback receiving every interception record.
    pub fn record_sink(
        mut self,
        sink: impl Fn(&InterceptionRecord) + Send + Sync + 'static,
    ) -> Self {
        self.record_sink = Some(Arc::new(sink));
        self
    }
}

/// One agent-hooks installation: the agent, chat and function seams as an
/// indivisible unit. Mirrors upstream's `MiddlewareBundle` returned by
/// `create_agent_hooks_middleware` and .NET `AsAIAgentWithAgentHooks`.
///
/// Install it on exactly one agent via [`AgentHooks::agent_builder`]. The
/// installation is cheap to clone; clones share one identity.
#[derive(Clone)]
pub struct AgentHooks {
    config: Arc<AgentHooksConfig>,
}

impl fmt::Debug for AgentHooks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentHooks")
            .field("interceptors", &self.config.interceptors.len())
            .field("mode", &self.config.mode)
            .field("host_owned_session", &self.config.emitter.is_some())
            .finish()
    }
}

impl AgentHooks {
    /// Enforcement with one agent-hooks session per run (upstream
    /// `create_agent_hooks_middleware`).
    ///
    /// Errors with [`Error::Configuration`] when no interceptor is registered
    /// (an emitter with zero interceptors fails closed on every emission).
    pub fn new(options: AgentHooksOptions) -> Result<Self> {
        if options.interceptors.is_empty() {
            return Err(Error::Configuration(
                "agent-hooks enforcement requires at least one interceptor (an emitter with zero \
                 interceptors fails closed on every emission)."
                    .into(),
            ));
        }
        Ok(Self {
            config: Arc::new(AgentHooksConfig {
                interceptors: options.interceptors,
                resolver: options.resolver,
                mode: options.mode,
                composition: options.composition,
                identity_provider: options.identity_provider,
                timeout: options.timeout,
                record_sink: options.record_sink,
                emitter: None,
                builder: None,
            }),
        })
    }

    /// Enforcement bound to a host-owned session (upstream
    /// `create_agent_hooks_middleware_from_emitter`): the fully configured
    /// `emitter` and matching `builder` serve every run, only the per-run
    /// points (`input` through `output`) are emitted, and the host owns the
    /// `agent_startup` / `agent_shutdown` boundaries. (Upstream's
    /// missing-argument `ValueError` is enforced by the types here.)
    pub fn from_emitter(
        emitter: Arc<InterceptionEmitter>,
        builder: Arc<AgentContextBuilder>,
    ) -> Self {
        Self {
            config: Arc::new(AgentHooksConfig {
                interceptors: Vec::new(),
                resolver: None,
                mode: emitter.mode(),
                composition: None,
                identity_provider: None,
                timeout: None,
                record_sink: None,
                emitter: Some(emitter),
                builder: Some(builder),
            }),
        }
    }

    /// An [`AgentBuilder`] over `client` with all three seams installed.
    ///
    /// The chat seam decorates `client` (supply the raw client: the agent
    /// adds its own function-invocation loop *above* the seam), and the agent
    /// and function seams are registered first in their pipelines. Configure
    /// the agent further (name, tools, providers, more middleware) on the
    /// returned builder.
    pub fn agent_builder(&self, client: impl ChatClient + 'static) -> AgentBuilder {
        Agent::builder(seams::ChatSeam {
            inner: client,
            config: self.config.clone(),
        })
        .middleware(Arc::new(seams::AgentSeam {
            config: self.config.clone(),
        }))
        .function_middleware(Arc::new(seams::FunctionSeam {
            config: self.config.clone(),
        }))
    }
}
