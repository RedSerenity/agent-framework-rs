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
//! ## SupportsAgentRun specs
//!
//! SupportsAgentRun specs follow the official schema vocabulary (`kind: Prompt`, `name`,
//! `instructions`, `model.id`/`provider`/`apiType`/`connection`/`options`,
//! `tools`, `outputSchema`, …). String fields support `${VAR}` /
//! `${VAR:-default}` environment interpolation.
//!
//! ## Workflow specs
//!
//! The upstream declarative *workflow* schema is a Power Platform / Copilot
//! Studio imperative DSL that does not map onto this port's graph engine. This
//! crate therefore defines a documented Rust-native [`WorkflowSpec`] that drives
//! the existing `WorkflowBuilder` and orchestration builders — either via
//! orchestration shorthand (`type: sequential | concurrent | group_chat |
//! handoff`) or an explicit node/edge graph. See [`workflow`] for details.
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
    PropertySchema, PropertySpec, ToolSpec,
};
pub use env::{EnvSource, ProcessEnv};
pub use error::{DeclarativeError, Result};
pub use loader::DeclarativeLoader;
pub use registry::{
    AgentRegistry, ChatClientFactory, ClientFactoryResult, FactoryError, PredicateRegistry,
    ToolRegistry,
};
pub use workflow::{
    CaseSpec, EdgeSpec, FanInSpec, FanOutSpec, HandoffEdgeSpec, NodeSpec, OrchestrationType,
    SwitchSpec, WorkflowSpec,
};
