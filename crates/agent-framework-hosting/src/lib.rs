//! # agent-framework-hosting
//!
//! Serve `agent-framework-rs` agents and workflows over HTTP. Three independent,
//! composable [`axum`] surfaces:
//!
//! - **DevUI-style API** ([`AgentHost`]) — entity discovery and
//!   OpenAI-Responses-flavored execution, mirroring the Python
//!   `agent_framework_devui` server:
//!   `GET /health`, `GET /v1/entities`, `GET /v1/entities/{id}/info`,
//!   `POST /v1/responses` (JSON or SSE), plus an embedded single-file debug
//!   page at `GET /` and `GET /ui`. See [`devui`].
//! - **A2A hosting** ([`a2a::A2ARouter`]) — the SupportsAgentRun-to-SupportsAgentRun protocol:
//!   `GET /.well-known/agent-card.json` and a JSON-RPC 2.0 `POST /`.
//! - **OpenAI Chat Completions** ([`openai_compat::OpenAiRouter`]) —
//!   `POST /v1/chat/completions` (JSON or SSE), for OpenAI-Chat clients.
//! - **AG-UI protocol** ([`agui::AgUiRouter`]) — CopilotKit's SupportsAgentRun-User
//!   Interaction protocol: `POST {path}` streaming camelCase SSE events
//!   (`RUN_STARTED` → `TEXT_MESSAGE_*` / `TOOL_CALL_*` → `RUN_FINISHED`),
//!   mirroring the Python `agent_framework_ag_ui` package.
//!
//! The OpenAI-Responses request/response types and the
//! [`responses::responses_to_run`]/[`responses::responses_from_run`]
//! conversion functions that back the DevUI API are a standalone, reusable
//! module ([`responses`]), mirroring the Python `hosting-responses` package;
//! any host can depend on it without pulling in DevUI's routing/SSE layer.
//!
//! Each surface builds a plain [`axum::Router`] you can nest into your own app,
//! or run directly with [`AgentHost::serve`].
//!
//! ```no_run
//! use agent_framework_core::agent::Agent;
//! use agent_framework_hosting::{AgentHost, a2a::A2ARouter, openai_compat::OpenAiRouter};
//!
//! # async fn demo(assistant: Agent) -> std::io::Result<()> {
//! // DevUI host with one agent.
//! let host = AgentHost::new().agent("assistant", assistant.clone());
//!
//! // Compose the A2A and OpenAI surfaces alongside it.
//! let app = host
//!     .into_router()
//!     .merge(OpenAiRouter::for_agent("assistant", assistant.clone()).into_router())
//!     .nest(
//!         "/a2a",
//!         A2ARouter::for_agent("assistant", assistant, "http://localhost:8080/a2a").into_router(),
//!     );
//!
//! let listener = tokio::net::TcpListener::bind(("127.0.0.1", 8080)).await?;
//! axum::serve(listener, app).await
//! # }
//! ```
//!
//! ## Securing a multi-surface app
//!
//! [`AgentHost::with_bearer_token`]/[`AgentHost::with_allowed_hosts`] guard only
//! the DevUI routes built by [`AgentHost::into_router`] — **not** any
//! [`OpenAiRouter`](openai_compat::OpenAiRouter),
//! [`A2ARouter`](a2a::A2ARouter), or [`AgUiRouter`](agui::AgUiRouter) merged or
//! nested onto it. To protect every execution endpoint in a composed app, build
//! the whole router first and wrap it with [`HostingSecurity`] (see its docs for
//! a full example).
//!
//! ## Divergences from the reference
//! Per-surface divergences (streaming realized by run-to-completion,
//! metadata-derived A2A skills, omitted fields) are documented on each module.
//! The `/v1/responses` surface is **stateful**: `previous_response_id` and
//! `conversation` continue agent sessions kept in a
//! [`SessionStore`](agent_framework_core::session_store::SessionStore), and a
//! paused workflow resumes from its conversation's checkpoints (see
//! [`devui`]). [`AgentState`] and [`WorkflowState`] expose the same building
//! blocks for applications that own their routes, as upstream's
//! `agent-framework-hosting` package does. The OpenAI chat-completions, A2A
//! and AG-UI surfaces remain stateless per request.

pub mod a2a;
pub mod agui;
mod continuation;
pub mod devui;
pub mod openai_compat;
pub mod registry;
pub mod responses;
pub mod security;
pub mod state;

mod sse;
mod ui;
mod util;

pub use registry::{AgentHost, AgentRegistration, IntoAgentRegistration};

// Re-export the reusable composed-router security layer.
pub use security::HostingSecurity;
pub use state::{AgentState, WorkflowState};

// Re-export the DevUI model types for callers building responses/clients.
pub use devui::models::{DiscoveryResponse, EntityInfo, HealthResponse};

// Re-export the reusable OpenAI-Responses conversion surface (mirrors
// upstream `hosting-responses`; UPSTREAM_DRIFT.md §14).
pub use responses::{
    create_conversation_id, create_response_id, responses_from_run, responses_run_options,
    responses_session_id, responses_to_run, ConversationRef, OutputFunctionCall, OutputItem,
    ResponseObject, ResponsesContinuation, ResponsesRequest,
};
