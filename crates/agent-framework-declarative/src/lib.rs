//! # agent-framework-declarative
//!
//! Load [`Agent`](agent_framework_core::agent::Agent)s and
//! [`Workflow`](agent_framework_core::workflow::Workflow)s from declarative
//! YAML/JSON specifications, mirroring the Microsoft Agent Framework
//! `agent-framework-declarative` (Python) package.
//!
//! The crate is intentionally **provider-agnostic**: it never depends on the
//! OpenAI/Azure/Anthropic crates. Instead you register a
//! [`ChatClientFactory`] closure per provider string, a [`ToolRegistry`] of
//! native Rust tools, and (for workflows) an [`AgentRegistry`] of pre-built
//! agents, then call [`DeclarativeLoader::load_agent`] /
//! [`DeclarativeLoader::load_workflow`].
//!
//! ## Agent specs
//!
//! Agent specs follow the official schema vocabulary (`kind: Prompt`, `name`,
//! `instructions`, `model.id`/`provider`/`apiType`/`connection`/`options`,
//! `tools`, `outputSchema`, `template`, …). String fields support `${VAR}` /
//! `${VAR:-default}` environment interpolation, and — like upstream — `=`
//! PowerFx expressions in identity/connection/instruction fields (with
//! `=Env.NAME` gated behind [`DeclarativeLoader::with_safe_mode`]).
//!
//! ## Workflow specs
//!
//! [`DeclarativeLoader::load_workflow`] accepts two document shapes and
//! dispatches on which one it is given:
//!
//! * **Upstream declarative workflows** — the format the Python and .NET
//!   Agent Framework accept: `kind: Workflow` with a `trigger` (or top-level
//!   `actions`) of `SetVariable`, `If`, `ConditionGroup`, `Foreach`,
//!   `GotoAction`, `InvokeAzureAgent`, `Question`, `InvokeFunctionTool`,
//!   `HttpRequestAction`, `InvokeMcpTool`, … actions whose `=` values are
//!   PowerFx expressions. They compile onto the core graph engine with
//!   upstream's semantics; see [`flow`] (and [`flow::WorkflowFactory`] for
//!   HTTP/MCP handlers, `Env` configuration, checkpointing and limits) and
//!   the [`powerfx`] interpreter for the supported expression language.
//! * **Rust-native [`WorkflowSpec`]** — this crate's own schema that drives
//!   the `WorkflowBuilder` and orchestration builders directly, via
//!   orchestration shorthand (`type: sequential | concurrent | group_chat |
//!   handoff`) or an explicit node/edge graph. See [`workflow`].
//!
//! ## Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_core::prelude::*;
//! use agent_framework_declarative::{ChatClientFactory, DeclarativeLoader};
//!
//! # fn make_client() -> Arc<dyn ChatClient> { unimplemented!() }
//! # async fn demo() -> Result<()> {
//! let loader = DeclarativeLoader::new().with_client_factory(
//!     ChatClientFactory::new().with("OpenAI.Chat", |_model| Ok(make_client())),
//! );
//!
//! let yaml = r#"
//! kind: Prompt
//! name: Assistant
//! instructions: You are a helpful assistant.
//! model:
//!   id: gpt-4.1-mini
//!   provider: OpenAI
//!   apiType: Chat
//!   options:
//!     temperature: 0.7
//! "#;
//!
//! let agent = loader.load_agent(yaml).unwrap();
//! let response = agent.run_once("Hello!").await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

pub mod agent;
pub mod condition;
pub mod env;
pub mod error;
pub mod flow;
pub mod loader;
pub mod powerfx;
pub mod registry;
pub mod workflow;

pub use agent::{
    AgentSpec, ApprovalModeDetail, ApprovalModeSpec, ConnectionSpec, ModelOptions, ModelSpec,
    PropertySchema, PropertySpec, TemplateFormatSpec, TemplateParserSpec, TemplateSpec, ToolSpec,
};
pub use env::{EnvSource, ProcessEnv};
pub use error::{DeclarativeError, Result};
pub use flow::WorkflowFactory;
pub use loader::DeclarativeLoader;
pub use registry::{
    AgentRegistry, ChatClientFactory, ClientFactoryResult, FactoryError, PredicateRegistry,
    ToolRegistry,
};
pub use workflow::{
    CaseSpec, EdgeSpec, FanInSpec, FanOutSpec, HandoffEdgeSpec, NodeSpec, OrchestrationType,
    SwitchSpec, WorkflowSpec,
};
