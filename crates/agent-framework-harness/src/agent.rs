//! The batteries-included harness agent.
//!
//! Rust equivalent of upstream Python `create_harness_agent`
//! (`_harness/_agent.py`) and .NET `HarnessAgent` / `HarnessAgentOptions` /
//! `ChatClientHarnessExtensions.AsHarnessAgent`.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agent_framework_core::agent::{Agent, AgentRunOptions, AgentRunStream, SupportsAgentRun};
use agent_framework_core::client::ChatClient;
use agent_framework_core::compaction::{
    CompactionProvider, CompactionStrategy, TokenBudget, Tokenizer,
};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::ContextProvider;
use agent_framework_core::middleware::{AgentMiddleware, ChatMiddleware, FunctionMiddleware};
use agent_framework_core::session::AgentSession;
use agent_framework_core::skills::SkillsProvider;
use agent_framework_core::tools::{
    hosted_web_search, ApprovalMode, FunctionInvocationConfig, ToolDefinition,
};
use agent_framework_core::types::{AgentResponse, ChatOptions, Message};
use async_trait::async_trait;
use tracing::Instrument;

use crate::background_agents::{
    BackgroundAgent, BackgroundAgentsProvider, BACKGROUND_AGENTS_TOOL_NAMES,
    DEFAULT_BACKGROUND_AGENTS_WAIT_TIMEOUT_SECONDS,
};
use crate::file_access::{AgentFileStore, FileAccessProvider, FileSystemAgentFileStore};
use crate::file_memory::{FileMemoryProvider, FILE_MEMORY_TOOL_NAMES};
use crate::history::SessionStateHistoryProvider;
use crate::loop_agent::{
    HarnessProviders, LoopAgent, NextMessageFn, ShouldContinueFn, DEFAULT_MAX_ITERATIONS,
};
use crate::mode::AgentModeProvider;
use crate::todo::TodoProvider;
use crate::tool_approval::{ToolApprovalAgent, ToolApprovalRuleCallback};

/// The harness's built-in operating guidelines. Mirrors upstream
/// `DEFAULT_HARNESS_INSTRUCTIONS` (and .NET `HarnessAgent.DefaultInstructions`).
pub const DEFAULT_HARNESS_INSTRUCTIONS: &str = "You are a helpful AI assistant that uses tools to complete tasks.

## General guidelines

- Think through the task before acting. Break complex work into clear steps.
- Use the tools available to you to gather information, perform actions, and verify results.
- Explain your reasoning and thought process as you work through tasks.
- Explain what you learned and what you are going to do next between tool calls, so the user can follow along with your thought process.
- Avoid making more than 4 tool calls in a row without explaining what you are doing.
- If a tool call fails or returns unexpected results, adapt your approach rather than repeating the same call.
- When you have completed the task, present a clear and concise summary of what you did and what you found.
";

/// Default telemetry provider name of a harness agent. Mirrors upstream
/// `HARNESS_AGENT_PROVIDER_NAME`.
pub const HARNESS_AGENT_PROVIDER_NAME: &str = "microsoft.agent_framework.harness";

/// The default file-memory directory, relative to the working directory.
pub const DEFAULT_FILE_MEMORY_DIRECTORY: &str = "agent-file-memory";

/// Assemble the final instructions from harness + agent instructions:
/// `"{harness}\n\n{agent}"` trimmed, `None` when empty. `harness = None`
/// uses [`DEFAULT_HARNESS_INSTRUCTIONS`]; `Some("")` omits it. Mirrors
/// upstream `_assemble_instructions`.
pub fn assemble_instructions(
    harness_instructions: Option<&str>,
    agent_instructions: Option<&str>,
) -> Option<String> {
    let harness = harness_instructions.unwrap_or(DEFAULT_HARNESS_INSTRUCTIONS);
    let combined = format!("{harness}\n\n{}", agent_instructions.unwrap_or_default());
    let trimmed = combined.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Emit one warning per process the first time an experimental harness
/// feature is enabled (upstream emits an `ExperimentalWarning` once).
fn warn_experimental(params: &[&str]) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if params.is_empty() || WARNED.swap(true, Ordering::SeqCst) {
        return;
    }
    let joined = params
        .iter()
        .map(|p| format!("'{p}'"))
        .collect::<Vec<_>>()
        .join(", ");
    tracing::warn!(
        "[HARNESS] HarnessAgent option(s) {joined} enable experimental harness features that may change or be removed in future versions without notice."
    );
}

/// Adapter so an `Arc<dyn CompactionStrategy>` can feed APIs taking one by
/// value.
struct SharedStrategy(Arc<dyn CompactionStrategy>);

impl CompactionStrategy for SharedStrategy {
    fn compact(&self, messages: &[Message], tokenizer: &dyn Tokenizer) -> Vec<Message> {
        self.0.compact(messages, tokenizer)
    }
}

/// Adapter so an `Arc<dyn Tokenizer>` can feed APIs taking one by value.
struct SharedTokenizer(Arc<dyn Tokenizer>);

impl Tokenizer for SharedTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        self.0.count_tokens(text)
    }
}

/// Builder (options) for [`HarnessAgent`].
///
/// Every option mirrors a `create_harness_agent` keyword argument (or a
/// .NET `HarnessAgentOptions` property). Not ported, because core has no
/// counterpart: `shell_executor` / `shell_environment_provider_options`
/// (the pre-release `agent-framework-tools` shell), `skills_paths`
/// (SKILL.md discovery — pass a [`SkillsProvider`] instead), and
/// `MessageInjectionMiddleware`.
pub struct HarnessAgentBuilder {
    client: Box<dyn FnOnce() -> agent_framework_core::agent::AgentBuilder + Send>,
    id: Option<String>,
    name: Option<String>,
    description: Option<String>,
    harness_instructions: Option<String>,
    agent_instructions: Option<String>,
    tools: Vec<ToolDefinition>,
    max_context_window_tokens: Option<usize>,
    max_output_tokens: Option<usize>,
    history_provider: Option<Arc<dyn ContextProvider>>,
    disable_compaction: bool,
    before_compaction_strategy: Option<Arc<dyn CompactionStrategy>>,
    after_compaction_strategy: Option<Arc<dyn CompactionStrategy>>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    disable_todo: bool,
    todo_provider: Option<TodoProvider>,
    disable_mode: bool,
    mode_provider: Option<AgentModeProvider>,
    disable_file_memory: bool,
    file_memory_store: Option<Arc<dyn AgentFileStore>>,
    file_access_store: Option<Arc<dyn AgentFileStore>>,
    file_access_session_scoped: bool,
    file_access_disable_write_tools: bool,
    file_access_disable_readonly_tool_approval: bool,
    file_access_disable_write_tool_approval: bool,
    skills_provider: Option<Arc<SkillsProvider>>,
    background_agents: Vec<BackgroundAgent>,
    background_agents_instructions: Option<String>,
    background_agents_wait_timeout_seconds: u64,
    disable_web_search: bool,
    disable_tool_auto_approval: bool,
    auto_approval_rules: Vec<ToolApprovalRuleCallback>,
    disable_approval_not_required_function_bypassing: bool,
    disable_approval_response_binding: bool,
    loop_should_continue: Option<ShouldContinueFn>,
    loop_next_message: Option<NextMessageFn>,
    loop_max_iterations: Option<usize>,
    otel_provider_name: Option<String>,
    context_providers: Vec<Arc<dyn ContextProvider>>,
    agent_middleware: Vec<Arc<AgentMiddleware>>,
    chat_middleware: Vec<Arc<ChatMiddleware>>,
    function_middleware: Vec<Arc<FunctionMiddleware>>,
    default_options: Option<ChatOptions>,
    function_invocation_config: Option<FunctionInvocationConfig>,
}

macro_rules! setter {
    ($(#[$doc:meta])* $name:ident: $ty:ty) => {
        $(#[$doc])*
        pub fn $name(mut self, value: $ty) -> Self {
            self.$name = value;
            self
        }
    };
}

macro_rules! str_setter {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        pub fn $name(mut self, value: impl Into<String>) -> Self {
            self.$name = Some(value.into());
            self
        }
    };
}

macro_rules! opt_setter {
    ($(#[$doc:meta])* $name:ident: $ty:ty) => {
        $(#[$doc])*
        pub fn $name(mut self, value: $ty) -> Self {
            self.$name = Some(value);
            self
        }
    };
}

impl HarnessAgentBuilder {
    fn new(client: impl ChatClient + 'static) -> Self {
        Self {
            client: Box::new(move || Agent::builder(client)),
            id: None,
            name: None,
            description: None,
            harness_instructions: None,
            agent_instructions: None,
            tools: Vec::new(),
            max_context_window_tokens: None,
            max_output_tokens: None,
            history_provider: None,
            disable_compaction: false,
            before_compaction_strategy: None,
            after_compaction_strategy: None,
            tokenizer: None,
            disable_todo: false,
            todo_provider: None,
            disable_mode: false,
            mode_provider: None,
            disable_file_memory: false,
            file_memory_store: None,
            file_access_store: None,
            file_access_session_scoped: false,
            file_access_disable_write_tools: false,
            file_access_disable_readonly_tool_approval: false,
            file_access_disable_write_tool_approval: false,
            skills_provider: None,
            background_agents: Vec::new(),
            background_agents_instructions: None,
            background_agents_wait_timeout_seconds: DEFAULT_BACKGROUND_AGENTS_WAIT_TIMEOUT_SECONDS,
            disable_web_search: false,
            disable_tool_auto_approval: false,
            auto_approval_rules: Vec::new(),
            disable_approval_not_required_function_bypassing: false,
            disable_approval_response_binding: false,
            loop_should_continue: None,
            loop_next_message: None,
            loop_max_iterations: Some(DEFAULT_MAX_ITERATIONS),
            otel_provider_name: None,
            context_providers: Vec::new(),
            agent_middleware: Vec::new(),
            chat_middleware: Vec::new(),
            function_middleware: Vec::new(),
            default_options: None,
            function_invocation_config: None,
        }
    }

    str_setter!(/// Agent id (a UUID when omitted).
        id);
    str_setter!(/// Agent name.
        name);
    str_setter!(/// Agent description.
        description);
    str_setter!(/// Override the harness operating guidelines
        /// ([`DEFAULT_HARNESS_INSTRUCTIONS`]); `""` omits them.
        harness_instructions);
    str_setter!(/// Task-specific instructions appended after the harness ones.
        agent_instructions);
    opt_setter!(/// The model's context-window size in tokens. With
        /// [`max_output_tokens`](Self::max_output_tokens) it builds the
        /// default token-budget compaction for both phases.
        max_context_window_tokens: usize);
    opt_setter!(/// Maximum output tokens per response; also the default
        /// `max_tokens` chat option.
        max_output_tokens: usize);
    opt_setter!(/// A custom history provider (attached to each session lacking
        /// one). Default: [`SessionStateHistoryProvider`].
        history_provider: Arc<dyn ContextProvider>);
    setter!(/// Skip compaction entirely.
        disable_compaction: bool);
    opt_setter!(/// Custom before-run compaction (runs even without token params).
        before_compaction_strategy: Arc<dyn CompactionStrategy>);
    opt_setter!(/// Custom after-run compaction of the stored transcript (runs even
        /// without token params; applies to the default history provider).
        after_compaction_strategy: Arc<dyn CompactionStrategy>);
    opt_setter!(/// Tokenizer for the compaction strategies.
        tokenizer: Arc<dyn Tokenizer>);
    setter!(/// Skip the [`TodoProvider`].
        disable_todo: bool);
    opt_setter!(/// A custom [`TodoProvider`] (ignored when disabled).
        todo_provider: TodoProvider);
    setter!(/// Skip the [`AgentModeProvider`].
        disable_mode: bool);
    opt_setter!(/// A custom [`AgentModeProvider`] (ignored when disabled).
        mode_provider: AgentModeProvider);
    setter!(/// Skip the [`FileMemoryProvider`] (on by default).
        disable_file_memory: bool);
    opt_setter!(/// The store backing file memory (default: a
        /// [`FileSystemAgentFileStore`] at `{cwd}/agent-file-memory`).
        file_memory_store: Arc<dyn AgentFileStore>);
    opt_setter!(/// Opt into a [`FileAccessProvider`] over this store
        /// (experimental).
        file_access_store: Arc<dyn AgentFileStore>);
    setter!(/// Confine file access to a per-session working folder.
        file_access_session_scoped: bool);
    setter!(/// Advertise only the read-only file-access tools.
        file_access_disable_write_tools: bool);
    setter!(/// Run the read-only file-access tools without approval.
        file_access_disable_readonly_tool_approval: bool);
    setter!(/// Run the write file-access tools without approval.
        file_access_disable_write_tool_approval: bool);
    opt_setter!(/// A skills provider (opt-in). **Security:** content from an
        /// external skill source is untrusted input.
        skills_provider: Arc<SkillsProvider>);
    str_setter!(/// Instructions override for the background-agents provider
        /// (`{background_agents}` placeholder supported).
        background_agents_instructions);
    setter!(/// Maximum seconds the background wait tool blocks (default 300;
        /// must be positive).
        background_agents_wait_timeout_seconds: u64);
    setter!(/// Skip the hosted web-search tool (added by default, like .NET's
        /// `HostedWebSearchTool`).
        disable_web_search: bool);
    setter!(/// Do not wire the [`ToolApprovalAgent`] (callers then answer every
        /// approval request themselves).
        disable_tool_auto_approval: bool);
    setter!(/// Surface approval requests for sibling calls that do not need
        /// approval (.NET `DisableApprovalNotRequiredFunctionBypassing`).
        disable_approval_not_required_function_bypassing: bool);
    setter!(/// Forward inbound approval responses unbound (.NET
        /// `DisableApprovalResponseBinding`).
        disable_approval_response_binding: bool);
    opt_setter!(/// Enable looping (experimental): the agent is re-run while this
        /// says so, outermost of all decorators.
        loop_should_continue: ShouldContinueFn);
    opt_setter!(/// The loop's next-message callable (only with
        /// `loop_should_continue`).
        loop_next_message: NextMessageFn);
    setter!(/// The loop's iteration cap (`None` = unbounded; default 10).
        loop_max_iterations: Option<usize>);
    str_setter!(/// Telemetry provider name (default
        /// [`HARNESS_AGENT_PROVIDER_NAME`]).
        otel_provider_name);
    opt_setter!(/// Provider-specific chat options (temperature, max_tokens, …).
        default_options: ChatOptions);
    opt_setter!(/// The function-invocation loop configuration (.NET
        /// `MaximumIterationsPerRequest` and friends).
        function_invocation_config: FunctionInvocationConfig);

    /// Add a tool.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Add a background agent (experimental). Names must be non-empty and
    /// unique (case-insensitive). **Security:** only supply vetted agents.
    pub fn background_agent(mut self, agent: impl Into<BackgroundAgent>) -> Self {
        self.background_agents.push(agent.into());
        self
    }

    /// Add heuristic auto-approval callbacks (evaluated after standing rules).
    pub fn auto_approval_rules(
        mut self,
        rules: impl IntoIterator<Item = ToolApprovalRuleCallback>,
    ) -> Self {
        self.auto_approval_rules.extend(rules);
        self
    }

    /// Add a context provider after the built-in ones.
    pub fn context_provider(mut self, provider: Arc<dyn ContextProvider>) -> Self {
        self.context_providers.push(provider);
        self
    }

    /// Add an agent middleware (runs inside the decorators).
    pub fn middleware(mut self, middleware: Arc<AgentMiddleware>) -> Self {
        self.agent_middleware.push(middleware);
        self
    }

    /// Add a chat middleware.
    pub fn chat_middleware(mut self, middleware: Arc<ChatMiddleware>) -> Self {
        self.chat_middleware.push(middleware);
        self
    }

    /// Add a function-invocation middleware.
    pub fn function_middleware(mut self, middleware: Arc<FunctionMiddleware>) -> Self {
        self.function_middleware.push(middleware);
        self
    }

    /// Validate and assemble the agent. Mirrors upstream
    /// `create_harness_agent`'s validation: token params must be positive and
    /// `max_output_tokens < max_context_window_tokens`; the background wait
    /// timeout must be positive; background agent names must be valid.
    pub fn build(self) -> Result<HarnessAgent> {
        if self.max_context_window_tokens == Some(0) {
            return Err(Error::Configuration(
                "max_context_window_tokens must be positive.".into(),
            ));
        }
        if self.max_output_tokens == Some(0) {
            return Err(Error::Configuration(
                "max_output_tokens must be positive.".into(),
            ));
        }
        if let (Some(ctx), Some(out)) = (self.max_context_window_tokens, self.max_output_tokens) {
            if out >= ctx {
                return Err(Error::Configuration(
                    "max_output_tokens must be less than max_context_window_tokens.".into(),
                ));
            }
        }
        let mut experimental = Vec::new();
        if !self.background_agents.is_empty() {
            experimental.push("background_agents");
        }
        if self.file_access_store.is_some() {
            experimental.push("file_access_store");
        }
        if self.loop_should_continue.is_some() {
            experimental.push("loop_should_continue");
        }
        warn_experimental(&experimental);

        // Compaction: the token-budget default (upstream's
        // `ContextWindowCompactionStrategy`: keep the newest messages within
        // `context - output` tokens) serves both phases unless overridden.
        let (before, after) = if self.disable_compaction {
            (None, None)
        } else {
            let default: Option<Arc<dyn CompactionStrategy>> =
                match (self.max_context_window_tokens, self.max_output_tokens) {
                    (Some(ctx), Some(out)) => Some(Arc::new(TokenBudget::new(ctx - out))),
                    _ => None,
                };
            (
                self.before_compaction_strategy
                    .clone()
                    .or_else(|| default.clone()),
                self.after_compaction_strategy.clone().or(default),
            )
        };

        // History: attached to each session (see `HarnessAgent::attach`).
        let history: Arc<dyn ContextProvider> = match self.history_provider {
            Some(custom) => {
                if after.is_some() {
                    tracing::warn!(
                        "after-run compaction applies to the harness's default history provider only; it is skipped for a custom history provider"
                    );
                }
                custom
            }
            None => {
                let mut provider = SessionStateHistoryProvider::new();
                if let Some(strategy) = after.clone() {
                    provider = provider.after_compaction(strategy);
                }
                if let Some(tokenizer) = self.tokenizer.clone() {
                    provider = provider.tokenizer(tokenizer);
                }
                Arc::new(provider)
            }
        };

        // Context providers, in upstream's order (history lives on the
        // session, which core runs first).
        let mut providers: Vec<Arc<dyn ContextProvider>> = Vec::new();
        if let Some(strategy) = before.clone() {
            let provider = match self.tokenizer.clone() {
                Some(t) => {
                    CompactionProvider::with_tokenizer(SharedStrategy(strategy), SharedTokenizer(t))
                }
                None => CompactionProvider::new(SharedStrategy(strategy)),
            };
            providers.push(Arc::new(provider));
        }
        let todo = (!self.disable_todo).then(|| self.todo_provider.unwrap_or_default());
        if let Some(todo) = &todo {
            providers.push(Arc::new(todo.clone()));
        }
        let mode = (!self.disable_mode).then(|| self.mode_provider.unwrap_or_default());
        if let Some(mode) = &mode {
            providers.push(Arc::new(mode.clone()));
        }
        let file_memory = if self.disable_file_memory {
            None
        } else {
            let store: Arc<dyn AgentFileStore> = match self.file_memory_store {
                Some(store) => store,
                None => {
                    let root = std::env::current_dir()
                        .map_err(|e| {
                            Error::Configuration(format!(
                                "cannot resolve the working directory: {e}"
                            ))
                        })?
                        .join(DEFAULT_FILE_MEMORY_DIRECTORY);
                    Arc::new(FileSystemAgentFileStore::new(root)?)
                }
            };
            Some(FileMemoryProvider::new(store))
        };
        if let Some(memory) = &file_memory {
            providers.push(Arc::new(memory.clone()));
        }
        let file_access = self.file_access_store.map(|store| {
            FileAccessProvider::new(store)
                .disable_write_tools(self.file_access_disable_write_tools)
                .disable_readonly_tool_approval(self.file_access_disable_readonly_tool_approval)
                .disable_write_tool_approval(self.file_access_disable_write_tool_approval)
                .session_scoped(self.file_access_session_scoped)
        });
        if let Some(access) = &file_access {
            providers.push(Arc::new(access.clone()));
        }
        if let Some(skills) = &self.skills_provider {
            providers.push(skills.clone());
        }
        let background = if self.background_agents.is_empty() {
            None
        } else {
            let mut provider = BackgroundAgentsProvider::new(self.background_agents)?
                .wait_timeout_seconds(self.background_agents_wait_timeout_seconds)?;
            if let Some(instructions) = &self.background_agents_instructions {
                provider = provider.instructions(instructions);
            }
            Some(provider)
        };
        if let Some(background) = &background {
            providers.push(Arc::new(background.clone()));
        }
        providers.extend(self.context_providers);

        // Tools: hosted web search first (unless disabled), then the caller's.
        let mut tools = Vec::new();
        if !self.disable_web_search {
            tools.push(hosted_web_search());
        }
        tools.extend(self.tools);

        // Names of tools known not to need approval (for the bypass).
        let mut never_require: HashSet<String> = HashSet::new();
        if todo.is_some() {
            never_require.extend(
                [
                    "todos_add",
                    "todos_complete",
                    "todos_remove",
                    "todos_get_remaining",
                    "todos_get_all",
                ]
                .map(String::from),
            );
        }
        if mode.is_some() {
            never_require.extend(["mode_set", "mode_get"].map(String::from));
        }
        if file_memory.is_some() {
            never_require.extend(FILE_MEMORY_TOOL_NAMES.map(String::from));
        }
        if background.is_some() {
            never_require.extend(BACKGROUND_AGENTS_TOOL_NAMES.map(String::from));
        }
        if self.skills_provider.is_some() {
            never_require.extend(["load_skill", "read_skill_resource"].map(String::from));
        }
        if let Some(access) = &file_access {
            for (name, mode) in access.tool_approval_modes() {
                if mode == ApprovalMode::NeverRequire {
                    never_require.insert(name.to_string());
                }
            }
        }
        for tool in &tools {
            if tool.requires_approval() {
                // The caller's tool shadows a built-in of the same name (the
                // agent's own tools come first), so it must keep prompting.
                never_require.remove(&tool.name);
            } else if tool.is_executable() {
                never_require.insert(tool.name.clone());
            }
        }

        // Chat options.
        let mut options = self.default_options.unwrap_or_default();
        if let Some(out) = self.max_output_tokens {
            if options.max_tokens.is_none() {
                options.max_tokens = Some(u32::try_from(out).unwrap_or(u32::MAX));
            }
        }

        let mut builder = (self.client)()
            .chat_options(options)
            .tools(tools)
            .context_providers(providers.clone());
        if let Some(instructions) = assemble_instructions(
            self.harness_instructions.as_deref(),
            self.agent_instructions.as_deref(),
        ) {
            builder = builder.instructions(instructions);
        }
        if let Some(id) = self.id {
            builder = builder.id(id);
        }
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        if let Some(description) = self.description {
            builder = builder.description(description);
        }
        for mw in self.agent_middleware {
            builder = builder.middleware(mw);
        }
        for mw in self.chat_middleware {
            builder = builder.chat_middleware(mw);
        }
        for mw in self.function_middleware {
            builder = builder.function_middleware(mw);
        }
        if let Some(config) = self.function_invocation_config {
            builder = builder.function_invocation_config(config);
        }
        let inner = builder.build();

        let mut outer: Arc<dyn SupportsAgentRun> = Arc::new(inner.clone());
        let tool_approval = !self.disable_tool_auto_approval;
        if tool_approval {
            let mut approval = ToolApprovalAgent::new(outer)
                .auto_approval_rules(self.auto_approval_rules)
                .disable_response_binding(self.disable_approval_response_binding);
            if !self.disable_approval_not_required_function_bypassing {
                approval = approval.approval_not_required_tools(never_require);
            }
            outer = Arc::new(approval);
        }
        let harness_providers = HarnessProviders {
            todo: todo.clone(),
            mode: mode.clone(),
            background_agents: background.clone(),
        };
        let looping = self.loop_should_continue.is_some();
        if let Some(should_continue) = self.loop_should_continue {
            let mut loop_agent = LoopAgent::new(outer, should_continue)
                .max_iterations(self.loop_max_iterations)?
                .providers(harness_providers.clone());
            if let Some(next) = self.loop_next_message {
                loop_agent = loop_agent.next_message(next);
            }
            outer = Arc::new(loop_agent);
        }

        Ok(HarnessAgent {
            outer,
            inner,
            history,
            providers: harness_providers,
            file_access,
            file_memory,
            context_providers: providers,
            otel_provider_name: self
                .otel_provider_name
                .unwrap_or_else(|| HARNESS_AGENT_PROVIDER_NAME.to_string()),
            tool_approval,
            looping,
            before_compaction: before.is_some(),
            after_compaction: after.is_some(),
        })
    }
}

/// A pre-configured, batteries-included agent.
///
/// Rust equivalent of upstream `create_harness_agent` / .NET `HarnessAgent`.
/// [`HarnessAgent::builder`] assembles, from a chat client:
///
/// - an inner [`Agent`] with the harness instructions
///   ([`DEFAULT_HARNESS_INSTRUCTIONS`] + agent instructions), the hosted
///   web-search tool (unless disabled) plus the caller's tools, and context
///   providers in upstream's order: compaction (when configured),
///   [`TodoProvider`], [`AgentModeProvider`], [`FileMemoryProvider`] (on by
///   default, rooted at `{cwd}/agent-file-memory`), [`FileAccessProvider`]
///   (opt-in), a skills provider (opt-in), [`BackgroundAgentsProvider`]
///   (opt-in), then the caller's providers;
/// - a [`SessionStateHistoryProvider`] (or a custom history provider)
///   attached to every session it runs, first;
/// - a [`ToolApprovalAgent`] around it (unless disabled), and outermost a
///   [`LoopAgent`] when `loop_should_continue` is set.
///
/// # Divergences
///
/// - History is persisted at the end of each run, not after every model
///   call: core has no per-service-call persistence, so a crash mid-run
///   loses that run's history (upstream's
///   `require_per_service_call_history_persistence`).
/// - Before-run compaction runs once per run over the loaded history (core
///   [`CompactionProvider`]), not per model call inside the tool loop.
/// - The web-search tool is added unconditionally unless disabled, as .NET
///   does; upstream Python adds it only for clients implementing
///   `SupportsWebSearchTool` (core cannot detect that capability).
/// - Running without a session creates a throwaway one (the approval
///   decorator needs a session; upstream raises instead).
/// - Telemetry: core instruments agents itself; the provider name is
///   recorded on a `harness_agent` tracing span around each run.
#[derive(Clone)]
pub struct HarnessAgent {
    outer: Arc<dyn SupportsAgentRun>,
    inner: Agent,
    history: Arc<dyn ContextProvider>,
    providers: HarnessProviders,
    file_access: Option<FileAccessProvider>,
    file_memory: Option<FileMemoryProvider>,
    context_providers: Vec<Arc<dyn ContextProvider>>,
    otel_provider_name: String,
    tool_approval: bool,
    looping: bool,
    before_compaction: bool,
    after_compaction: bool,
}

impl std::fmt::Debug for HarnessAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarnessAgent")
            .field("id", &self.inner.id())
            .field("name", &self.inner.name())
            .field("context_providers", &self.context_providers.len())
            .field("tool_approval", &self.tool_approval)
            .field("looping", &self.looping)
            .finish_non_exhaustive()
    }
}

impl HarnessAgent {
    /// Start configuring a harness agent over `client`.
    pub fn builder(client: impl ChatClient + 'static) -> HarnessAgentBuilder {
        HarnessAgentBuilder::new(client)
    }

    /// The assembled inner [`Agent`] (no decorators).
    pub fn inner_agent(&self) -> &Agent {
        &self.inner
    }

    /// The agent-level context providers, in order (the history provider is
    /// attached to sessions instead — see [`history_provider`](Self::history_provider)).
    pub fn context_providers(&self) -> &[Arc<dyn ContextProvider>] {
        &self.context_providers
    }

    /// The history provider attached to each session.
    pub fn history_provider(&self) -> &Arc<dyn ContextProvider> {
        &self.history
    }

    /// The wired [`TodoProvider`], if any.
    pub fn todo_provider(&self) -> Option<&TodoProvider> {
        self.providers.todo.as_ref()
    }

    /// The wired [`AgentModeProvider`], if any.
    pub fn mode_provider(&self) -> Option<&AgentModeProvider> {
        self.providers.mode.as_ref()
    }

    /// The wired [`BackgroundAgentsProvider`], if any.
    pub fn background_agents_provider(&self) -> Option<&BackgroundAgentsProvider> {
        self.providers.background_agents.as_ref()
    }

    /// The wired [`FileAccessProvider`], if any.
    pub fn file_access_provider(&self) -> Option<&FileAccessProvider> {
        self.file_access.as_ref()
    }

    /// The wired [`FileMemoryProvider`], if any.
    pub fn file_memory_provider(&self) -> Option<&FileMemoryProvider> {
        self.file_memory.as_ref()
    }

    /// The harness providers handed to loop callbacks.
    pub fn harness_providers(&self) -> &HarnessProviders {
        &self.providers
    }

    /// The telemetry provider name.
    pub fn otel_provider_name(&self) -> &str {
        &self.otel_provider_name
    }

    /// Whether the [`ToolApprovalAgent`] decorator is wired.
    pub fn has_tool_approval(&self) -> bool {
        self.tool_approval
    }

    /// Whether the [`LoopAgent`] decorator is wired.
    pub fn has_loop(&self) -> bool {
        self.looping
    }

    /// Whether before-run / after-run compaction is configured.
    pub fn compaction_phases(&self) -> (bool, bool) {
        (self.before_compaction, self.after_compaction)
    }

    /// Attach the harness history provider to `session` (first) when it is
    /// local and carries no history provider yet.
    pub fn attach(&self, session: &mut AgentSession) {
        if session.service_session_id().is_none()
            && !session
                .context_providers
                .iter()
                .any(|p| p.is_history_provider())
        {
            session.context_providers.insert(0, self.history.clone());
        }
    }

    /// Run with a fresh session.
    pub async fn run_once(
        &self,
        messages: impl agent_framework_core::types::IntoMessages,
    ) -> Result<AgentResponse> {
        self.run(messages.into_messages(), None).await
    }

    fn span(&self) -> tracing::Span {
        tracing::info_span!(
            "harness_agent",
            provider = %self.otel_provider_name,
            agent = %self.inner.display_name()
        )
    }
}

#[async_trait]
impl SupportsAgentRun for HarnessAgent {
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
        let mut owned;
        let session = match session {
            Some(s) => s,
            None => {
                owned = self.create_session();
                &mut owned
            }
        };
        self.attach(session);
        self.outer
            .run_with_options(messages, Some(session), options)
            .instrument(self.span())
            .await
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let mut session = session.unwrap_or_else(|| self.create_session());
        self.attach(&mut session);
        self.outer
            .run_stream(messages, Some(session), options)
            .instrument(self.span())
            .await
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    fn create_session(&self) -> AgentSession {
        let mut session = AgentSession::new();
        self.attach(&mut session);
        session
    }
}

/// Extension mirroring .NET `ChatClientHarnessExtensions.AsHarnessAgent`.
pub trait ChatClientHarnessExt: ChatClient + Sized + 'static {
    /// Start a [`HarnessAgent`] over this client.
    #[allow(clippy::wrong_self_convention)] // mirrors .NET `AsHarnessAgent`
    fn as_harness_agent(self) -> HarnessAgentBuilder {
        HarnessAgent::builder(self)
    }
}

impl<C: ChatClient + Sized + 'static> ChatClientHarnessExt for C {}
