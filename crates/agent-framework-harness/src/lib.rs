//! # agent-framework-harness
//!
//! A batteries-included, coding-agent-style **harness agent** for
//! `agent-framework-rs`: todos, operating modes, sandboxed file access,
//! file-backed memory, background sub-agents, a "don't ask again"
//! tool-approval policy, and agent looping — assembled around any
//! [`ChatClient`](agent_framework_core::client::ChatClient) by
//! [`HarnessAgent::builder`](agent::HarnessAgent::builder).
//!
//! Rust port of upstream Python `agent_framework._harness`
//! (`create_harness_agent` and friends) and .NET
//! `Microsoft.Agents.AI.Harness` (`HarnessAgent`, `HarnessAgentOptions`,
//! `AsHarnessAgent`), shipped — like .NET's — as its own package.
//!
//! | Module | Upstream | What it provides |
//! | --- | --- | --- |
//! | [`agent`] | `_agent.py`, `HarnessAgent.cs` | [`HarnessAgent`] and its builder |
//! | [`todo`](mod@todo) | `_todo.py` | [`TodoProvider`] + session/file stores |
//! | [`mode`] | `_mode.py` | [`AgentModeProvider`], [`get_agent_mode`], [`set_agent_mode`] |
//! | [`file_access`] | `_file_access.py` | [`AgentFileStore`], in-memory and sandboxed filesystem stores, [`FileAccessProvider`] |
//! | [`file_memory`] | `_file_memory.py` | [`FileMemoryProvider`] |
//! | [`memory`] | `_memory.py` | [`MemoryContextProvider`] (topic memory, transcripts, extraction) |
//! | [`background_agents`] | `_background_agents.py` | [`BackgroundAgentsProvider`] |
//! | [`tool_approval`] | `_tool_approval.py` | [`ToolApprovalAgent`], standing approval rules |
//! | [`loop_agent`] | `_loop.py` | [`LoopAgent`], judge loops, todo/background loop helpers |
//! | [`history`] | `InMemoryHistoryProvider` | [`SessionStateHistoryProvider`] |
//! | [`paths`] | `_file_access.py` / `_filesystem.py` helpers | path normalization, line editing, globbing, bounded regex search |
//!
//! ## Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_core::prelude::*;
//! use agent_framework_harness::agent::HarnessAgent;
//! use agent_framework_harness::file_access::FileSystemAgentFileStore;
//!
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let agent = HarnessAgent::builder(client)
//!     .name("research-agent")
//!     .agent_instructions("Focus on academic sources.")
//!     .max_context_window_tokens(200_000)
//!     .max_output_tokens(32_000)
//!     .file_access_store(Arc::new(FileSystemAgentFileStore::new("./workspace")?))
//!     .build()?;
//!
//! let mut session = agent.create_session();
//! let response = agent
//!     .run(vec![Message::user("Plan a weekend trip to Seattle")], Some(&mut session))
//!     .await?;
//! for request in response.user_input_requests() {
//!     println!("approve {}?", request.function_call.name);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Structural divergences from upstream
//!
//! - **Decorators instead of agent middleware.** Upstream's
//!   `ToolApprovalMiddleware` and `AgentLoopMiddleware` re-run the whole
//!   agent from inside the agent-middleware pipeline and read
//!   `context.session`. A Rust `AgentContext` has no session and runs after
//!   the context providers, so both are agent decorators here
//!   ([`ToolApprovalAgent`],
//!   [`LoopAgent`]) — the shape .NET uses.
//! - **Session access in providers** comes from two additive core hooks:
//!   [`SessionContext::session_state`](agent_framework_core::memory::SessionContext::session_state)
//!   and [`ContextProvider::after_run_in_session`](agent_framework_core::memory::ContextProvider::after_run_in_session).
//! - **History** is persisted at the end of each run (core has no
//!   per-service-call persistence) in session state, mirroring upstream's
//!   state-backed `InMemoryHistoryProvider`.
//! - **Not ported** (no core counterpart): the shell tool and
//!   `ShellEnvironmentProvider` (pre-release `agent-framework-tools`),
//!   `skills_paths` SKILL.md discovery (pass a
//!   [`SkillsProvider`](agent_framework_core::skills::SkillsProvider)),
//!   `MessageInjectionMiddleware`, upstream's turn-scoped
//!   `after_run_once_per_turn` hook, and feature-usage telemetry.
//!
//! Each module documents its own, finer-grained divergences.

pub mod agent;
pub mod background_agents;
pub mod file_access;
pub mod file_memory;
pub mod history;
pub mod loop_agent;
pub mod memory;
pub mod mode;
pub mod paths;
pub mod todo;
pub mod tool_approval;
pub mod util;

pub use agent::{
    ChatClientHarnessExt, HarnessAgent, HarnessAgentBuilder, DEFAULT_HARNESS_INSTRUCTIONS,
};
pub use background_agents::{BackgroundAgent, BackgroundAgentsProvider};
pub use file_access::{
    AgentFileStore, FileAccessProvider, FileSystemAgentFileStore, InMemoryAgentFileStore,
};
pub use file_memory::FileMemoryProvider;
pub use history::SessionStateHistoryProvider;
pub use loop_agent::{LoopAgent, LoopContext, LoopDecision};
pub use memory::{MemoryContextProvider, MemoryFileStore};
pub use mode::{get_agent_mode, set_agent_mode, AgentModeProvider};
pub use todo::{TodoFileStore, TodoProvider, TodoSessionStore};
pub use tool_approval::{
    create_always_approve_tool_response, create_always_approve_tool_with_arguments_response,
    ToolApprovalAgent, ToolApprovalRule,
};
pub use util::SessionRef;
