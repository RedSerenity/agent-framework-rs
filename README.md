# agent-framework-rs

A Rust implementation of the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
for building AI agents and multi-agent workflows.

This project ports the framework's architecture — chat clients, agents, tools,
threads, memory, middleware, a graph-based workflow engine, a family of
prebuilt multi-agent orchestrations, and the surrounding ecosystem (A2A,
AG-UI, declarative specs, HTTP hosting, memory backends, compliance
middleware) — to idiomatic, async Rust, with the goal of complete feature
parity with the Python and .NET implementations.

> Status: **at parity for the core and nearly all of the ecosystem.** The
> core runtime with retries and three middleware levels; OpenAI (chat +
> Responses API), Azure OpenAI, Azure AI Foundry, Anthropic, and Copilot
> Studio providers; a full Entra ID credential chain; MCP tools (stdio /
> HTTP / WebSocket, with prompts, sampling, and roots); the workflow engine
> (signature-validated checkpointing, human-in-the-loop, shared state,
> validation, visualization, sub-workflows); every orchestration (sequential,
> concurrent, group chat, handoff, Magentic with plan-review and
> stall-intervention HITL, workflow-as-agent); a complete A2A client; YAML
> agents/workflows; HTTP hosting (DevUI-style + embedded debug page, A2A,
> AG-UI, OpenAI-compatible); Redis, Mem0, Cosmos DB, and Azure AI Search
> memory/storage (including an Azure AI Search vector store with portable
> filters); and Purview compliance middleware are implemented and tested. See [PARITY.md](PARITY.md) for the detailed, grounded matrix and
> the [roadmap](#roadmap) for the short list of what's left.

## Highlights

- **Agents** — a `ChatAgent` with instructions, tools, default options,
  conversation threads, memory/context providers, and middleware at all three
  levels: agent-run, chat-client-call, and per-tool-invocation.
- **Automatic tool calling** — the function-invocation loop executes local
  tools and feeds results back to the model until it produces a final answer,
  bounded three ways: model round trips (`max_iterations`), total tool
  executions (`max_function_calls`), and wall-clock time
  (`max_duration_seconds`). The last two are cumulative across approval round
  trips, so a run that pauses for a human cannot reset its own budget.
- **Retries** — `RetryingChatClient` wraps any client with a `RetryPolicy`:
  exponential backoff with jitter, retryable-status classification, and
  server `Retry-After` honored over the computed delay.
- **Human-in-the-loop everywhere** — tool-call approval
  (`ApprovalMode::AlwaysRequire`), workflow `request_info` pauses, Magentic
  plan review before execution, and Magentic stall intervention mid-run.
- **Structured output** — request a JSON-Schema-conforming response with
  `ResponseFormat::json_schema` and parse it with `response.parse_json::<T>()`.
  Works on every provider — for Anthropic (no native `response_format`) it's
  folded into the system prompt, which the Python and .NET references don't
  do at all.
- **Streaming** — token-by-token streaming for both chat clients and agents.
- **Workflows** — a Pregel-style, superstep graph engine (`WorkflowBuilder`)
  with single edges, conditions, fan-out, fan-in, and switch/case routing,
  plus checkpointing (in-memory and file-backed, with graph-signature
  validation on resume), human-in-the-loop request/response, run-scoped
  shared state, graph validation, Mermaid/DOT visualization, and sub-workflow
  composition.
- **Orchestration** — prebuilt `SequentialBuilder`, `ConcurrentBuilder`,
  `GroupChatBuilder` (round-robin, custom, or LLM-managed), `HandoffBuilder`,
  and `MagenticBuilder` patterns, plus `WorkflowAgent` to expose any workflow
  as an `Agent`.
- **Providers** — OpenAI (Chat Completions + Responses API), Azure OpenAI
  (API key or Entra ID), Azure AI Foundry persistent agents, Anthropic, and
  Copilot Studio (Direct-to-Engine), all behind the `ChatClient`/`Agent`
  abstractions; plus a hand-rolled Entra ID credential chain
  (`AzureCliCredential`, `ClientSecretCredential`, `ManagedIdentityCredential`,
  `ChainedTokenCredential`).
- **MCP** — stdio, streamable HTTP, and WebSocket transports; tools, prompts,
  server-initiated sampling (`chat_client_sampling_handler` answers
  `sampling/createMessage` with your model), and roots.
- **A2A** — call any Agent2Agent server as a local `Agent` (`A2AAgent`), with
  card discovery, multi-turn context continuity, streaming, push-notification
  config, task resubscription, and the authenticated extended card.
- **Declarative** — load agents from the official YAML spec vocabulary and
  workflows from a Rust-native spec, with env interpolation and pluggable
  provider/tool registries.
- **Hosting** — serve agents/workflows over HTTP with axum: a DevUI-style API
  (`/v1/entities`, `/v1/responses` with SSE) with an embedded zero-dependency
  debug page at `/`, A2A serving (agent card + JSON-RPC), the AG-UI protocol
  (`AgUiRouter` SSE events for CopilotKit frontends), and an OpenAI-compatible
  `/v1/chat/completions`.
- **Memory & storage** — Redis-backed history and long-term memory (BM25 via
  RediSearch, SCAN fallback on plain Redis), Mem0, Azure Cosmos DB message
  store, workflow checkpoints and vector store (master key or Entra ID), and
  Azure AI Search both as a context provider and as a full vector store. Any
  collection can be handed to an agent as tools with
  `VectorCollectionContextProvider`.
- **Vector stores** — a provider-agnostic `VectorStore`/`VectorCollection`
  pair with a **portable filter** (`Filter::gte("year", 2020)`) each connector
  translates into its own dialect, so the same retrieval code runs against the
  in-memory store in a test and against Azure AI Search or Azure Cosmos DB in
  production. `VectorCollectionContextProvider` turns any of them into agent
  tools, with writes gated on approval by default.
- **Compliance** — Purview middleware evaluates every content item of each
  prompt and response against Microsoft Graph `processContent` — text,
  attachments, tool calls and tool results alike — and blocks on DLP
  verdicts.
- **Observability** — `ObservableChatClient` emits `tracing` spans following
  OpenTelemetry GenAI semantic conventions, ready to bridge into any OTel
  exporter.

## Workspace layout

| Crate | Description |
| --- | --- |
| [`agent-framework-core`](crates/agent-framework-core) | Core abstractions: types, chat client + retries, agents, tools, threads, memory, middleware, observability, workflows & orchestration. |
| [`agent-framework-openai`](crates/agent-framework-openai) | OpenAI Chat Completions and Responses API clients (also for OpenAI-compatible endpoints). |
| [`agent-framework-anthropic`](crates/agent-framework-anthropic) | Anthropic (Claude) Messages API client. |
| [`agent-framework-azure`](crates/agent-framework-azure) | Azure OpenAI client + the Entra ID credential chain (CLI / client-secret / managed-identity / chained). |
| [`agent-framework-foundry`](crates/agent-framework-foundry) | Azure AI Foundry Responses API chat client (`FoundryChatClient`), client-side Prompt Agents (`FoundryAgent`), embeddings (`FoundryEmbeddingClient`) over either a Models inference endpoint or a project's OpenAI deployments, and managed memory (`FoundryMemoryProvider`). |
| [`agent-framework-azure-ai-search`](crates/agent-framework-azure-ai-search) | Azure AI Search: context provider (semantic + optional vector query) and vector store (index management, document CRUD, filtered vector and keyword-hybrid search). |
| [`agent-framework-mcp`](crates/agent-framework-mcp) | MCP client: stdio/HTTP/WebSocket transports, tools, prompts, sampling, roots. |
| [`agent-framework-a2a`](crates/agent-framework-a2a) | Agent2Agent protocol client: `A2AAgent` + `A2AClient` (full task surface). |
| [`agent-framework-declarative`](crates/agent-framework-declarative) | Declarative YAML/JSON agents and workflows with provider/tool registries. |
| [`agent-framework-hosting`](crates/agent-framework-hosting) | HTTP serving (axum): DevUI-style API + embedded debug UI, A2A, AG-UI, OpenAI-compatible; stateful `/v1/responses` (`previous_response_id`, conversations, workflow resume). |
| [`agent-framework-hosting-mcp`](crates/agent-framework-hosting-mcp) | Agents and workflows as MCP tools, with a minimal stdio / streamable-HTTP MCP server. |
| [`agent-framework-redis`](crates/agent-framework-redis) | Redis-backed `ChatMessageStore` and long-term-memory `ContextProvider` (RediSearch BM25). |
| [`agent-framework-mem0`](crates/agent-framework-mem0) | Mem0 hosted-API long-term-memory `ContextProvider`. |
| [`agent-framework-cosmos`](crates/agent-framework-cosmos) | Azure Cosmos DB NoSQL `ChatMessageStore`, workflow checkpoints, and vector store (master key or Entra ID, REST). |
| [`agent-framework-copilotstudio`](crates/agent-framework-copilotstudio) | Microsoft Copilot Studio agent client (Direct-to-Engine). |
| [`agent-framework-purview`](crates/agent-framework-purview) | Microsoft Purview compliance middleware (`processContent` DLP checks). |
| [`agent-framework`](crates/agent-framework) | Umbrella crate re-exporting the core plus everything above behind cargo features. |

## Quick start

```toml
[dependencies]
agent-framework = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIClient::from_env("gpt-4o-mini")?; // reads OPENAI_API_KEY

    let agent = ChatAgent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of France?").await?;
    println!("{}", response.text());
    Ok(())
}
```

### Tools

```rust
use agent_framework::prelude::*;
use serde_json::json;

let get_weather = AiFunction::new(
    "get_weather",
    "Get the current weather for a city.",
    json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
    }),
    |args| async move {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(json!({ "city": city, "temperature_c": 21, "condition": "sunny" }))
    },
)
.into_definition();

let agent = ChatAgent::builder(client).tool(get_weather).build();
let reply = agent.run_once("What's the weather in Paris?").await?;
```

### Workflows

```rust
use std::sync::Arc;
use agent_framework::prelude::*;
use agent_framework::workflow::SequentialBuilder;

let workflow = SequentialBuilder::new()
    .participants(vec![writer as Arc<dyn Agent>, editor])
    .build()?;

let result = workflow.run("Write about Rust").await?;
let final_output = result.last_output();
```

## Feature flags

The umbrella `agent-framework` crate re-exports [`agent-framework-core`]
unconditionally, plus each companion crate behind a cargo feature:

| Feature | Crate | Default |
| --- | --- | --- |
| `openai` | [`agent-framework-openai`](crates/agent-framework-openai) — OpenAI Chat Completions + Responses API | yes |
| `anthropic` | [`agent-framework-anthropic`](crates/agent-framework-anthropic) — Anthropic (Claude) Messages API | no |
| `azure` | [`agent-framework-azure`](crates/agent-framework-azure) — Azure OpenAI (api-key / Entra ID) | no |
| `mcp` | [`agent-framework-mcp`](crates/agent-framework-mcp) — Model Context Protocol tools (stdio, HTTP, websocket) | no |
| `a2a` | [`agent-framework-a2a`](crates/agent-framework-a2a) — Agent2Agent protocol client | no |
| `declarative` | [`agent-framework-declarative`](crates/agent-framework-declarative) — YAML/JSON agents & workflows | no |
| `hosting` | [`agent-framework-hosting`](crates/agent-framework-hosting) — serve agents over HTTP (DevUI-style, A2A, AG-UI, OpenAI-compatible) | no |
| `hosting-mcp` | [`agent-framework-hosting-mcp`](crates/agent-framework-hosting-mcp) — serve agents and workflows as MCP tools | no |
| `redis` | [`agent-framework-redis`](crates/agent-framework-redis) — Redis chat-message store & context provider | no |
| `mem0` | [`agent-framework-mem0`](crates/agent-framework-mem0) — Mem0 long-term memory provider | no |
| `foundry` | [`agent-framework-foundry`](crates/agent-framework-foundry) — Azure AI Foundry Responses API chat client, Prompt Agents, embeddings, and managed memory | no |
| `azure-ai-search` | [`agent-framework-azure-ai-search`](crates/agent-framework-azure-ai-search) — Azure AI Search memory | no |
| `cosmos` | [`agent-framework-cosmos`](crates/agent-framework-cosmos) — Cosmos DB NoSQL message store, checkpoints, and vector store | no |
| `copilotstudio` | [`agent-framework-copilotstudio`](crates/agent-framework-copilotstudio) — Copilot Studio agents | no |
| `purview` | [`agent-framework-purview`](crates/agent-framework-purview) — Purview compliance middleware | no |
| `otel-metrics` | GenAI metrics (token-usage / operation-duration / function-invocation histograms) via the `opentelemetry` API crate | no |
| `full` | all of the above except `otel-metrics` | no |

```toml
# Everything:
agent-framework = { version = "0.1", features = ["full"] }
# Just OpenAI (the default) plus Anthropic:
agent-framework = { version = "0.1", features = ["anthropic"] }
```

## Examples

99 runnable examples live in [`examples/`](examples), organized by topic
(agents, providers, workflows, orchestrations, mcp, hosting, memory,
observability, a2a, declarative, compliance). See
[`examples/README.md`](examples/README.md) for the full gallery — every
example with a one-line description and what it needs to run. The offline
ones need no API key or network access; the rest read their provider's
credentials from environment variables, and skip gracefully when unset.

All of them run the same way, no `--features` flag needed:

```bash
cargo run -p agent-framework-examples --example <name>
```

A few highlights:

```bash
OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example quickstart
cargo run -p agent-framework-examples --example streaming_sse              # offline -- real SSE token streaming end to end
cargo run -p agent-framework-examples --example typed_tools                # offline -- JSON Schema derived from a Rust struct
cargo run -p agent-framework-examples --example checkpoint_resume_fanin    # offline -- checkpoint mid-fan-in, then resume
cargo run -p agent-framework-examples --example custom_chat_client         # offline -- implement ChatClient for your own backend
cargo run -p agent-framework-examples --example mcp_http_transport         # offline -- a real MCP server, served in-process
OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example magentic
```

## Design

The crate mirrors the Python framework's module structure so concepts map
one-to-one:

| Python (`agent_framework`) | Rust |
| --- | --- |
| `_types` | `types` (`ChatMessage`, `Content`, `ChatResponse`, `ChatOptions`, `ResponseFormat`, …) |
| `_clients` | `client` (`ChatClient`, `FunctionInvokingChatClient`, `RetryingChatClient`) |
| `_agents` | `agent` (`Agent`, `ChatAgent`, `as_tool`) |
| `_tools` | `tools` (`Tool`, `AiFunction`, hosted tools, `ApprovalMode`) |
| `_threads` | `threads` (`AgentThread`, `ChatMessageStore`) |
| `_memory` | `memory` (`ContextProvider`, `AggregateContextProvider`) |
| `_middleware` | `middleware` (agent / chat / function pipelines) |
| `observability` | `observability` (`ObservableChatClient`, OTel GenAI spans) |
| `_workflows` | `workflow` (`Workflow`, `WorkflowBuilder`, `Executor`, checkpointing, HITL, shared state, validation, viz, sub-workflows) |
| `_workflows._sequential` / `_concurrent` / `_group_chat` / `_handoff` / `_magentic` / `_agent` | `workflow::orchestration` (`SequentialBuilder`, `ConcurrentBuilder`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `WorkflowAgent`) |
| `_mcp` | [`agent-framework-mcp`](crates/agent-framework-mcp) (`McpStdioTool`, `McpStreamableHttpTool`, `McpWebsocketTool`, prompts, sampling, roots) |
| `a2a` package | [`agent-framework-a2a`](crates/agent-framework-a2a) (`A2AAgent`, `A2AClient`) |
| `declarative` package | [`agent-framework-declarative`](crates/agent-framework-declarative) (`DeclarativeLoader`) |
| `devui` / `ag-ui` packages | [`agent-framework-hosting`](crates/agent-framework-hosting) (`AgentHost`, `A2ARouter`, `AgUiRouter`, `OpenAiRouter`, embedded debug UI) |
| `redis` / `mem0` packages | [`agent-framework-redis`](crates/agent-framework-redis) / [`agent-framework-mem0`](crates/agent-framework-mem0) |
| `foundry` / `azure-ai-search` packages | [`agent-framework-foundry`](crates/agent-framework-foundry) / [`agent-framework-azure-ai-search`](crates/agent-framework-azure-ai-search) |
| `copilotstudio` / `purview` packages | [`agent-framework-copilotstudio`](crates/agent-framework-copilotstudio) / [`agent-framework-purview`](crates/agent-framework-purview) |
| (.NET `Microsoft.Agents.AI.CosmosNoSql`) | [`agent-framework-cosmos`](crates/agent-framework-cosmos) |

Cross-cutting behavior implemented in Python via class decorators
(`use_function_invocation`, `use_*_middleware`) is expressed in Rust as wrapper
types (`FunctionInvokingChatClient`, `RetryingChatClient`,
`ObservableChatClient`) and explicit middleware pipelines.

## Roadmap

See [PARITY.md](PARITY.md) for the feature matrix and
[GAP_ANALYSIS.md](GAP_ANALYSIS.md) for the audited gap list and its status
section (the current source of truth). The remaining gaps:

- [ ] Workflow depth: typed executor routing / multiple handlers,
      `AgentExecutorRequest`-style envelopes, sub-workflow request
      interception, orchestration `with_request_info`/`with_checkpointing`
      options, event origin/warning events, viz file export
- [ ] Cross-language wire compatibility: `type`-tagged message payloads,
      `raw_representation`/`additional_properties` on content types
- [ ] AG-UI: the client (`AGUIChatClient`) and predictive-state events
      (`STATE_SNAPSHOT`/`STATE_DELTA`, `confirm_changes`)
- [ ] DevUI parity: conversations API, run cancellation, `/meta`,
      directory-based entity discovery, auth; the React frontend
- [ ] A2A serving: streaming, non-terminal task lifecycle, file/data parts,
      push-notification config, `tasks/resubscribe`, extended card
- [ ] Providers: `AzureOpenAIAssistantsClient` wrapper, the new Foundry
      Prompt-Agent client; `as_mcp_server`
- [ ] MCP client: standalone GET-based SSE listening, automatic reconnect,
      elicitation
- [ ] Redis provider: embeddings/vector-KNN and hybrid search (BM25
      full-text ships on Redis Stack)
- [ ] Cosmos DB: `TransactionalBatch`, hierarchical partition keys, TTL
      (Entra ID auth and the Cosmos-backed workflow-checkpoint store ship)
- [ ] Vector connectors upstream added alongside the portable filter:
      PostgreSQL/pgvector, Qdrant, Redis HASH/JSON, and SQL Server / Azure SQL
      native vector (the last needs a TDS driver this workspace does not have)
- [ ] Azure: a Content Understanding context provider, Foundry evaluations,
      and the image half of Foundry embeddings (the text half, over both
      endpoints, ships; the Foundry memory provider ships)
- [ ] The upstream Copilot-Studio declarative *workflow* DSL (declarative
      agents already follow the official schema)
- [ ] Purview: protection-scopes precheck/caching, background
      content-activity logging, JWT-derived identity fallback
- [ ] OTel SDK exporter wiring stays the application's job by design
      (spans and `otel-metrics` histograms are emitted and bridge-ready)
- [ ] Remaining ecosystem: ChatKit, the `lab` experimental packages,
      DurableTask/Azure Functions hosting

Done — everything else, including:

- [x] Core data model, function-invocation loop, approval flow, structured
      output on every provider (auto-parsed into `response.value`), and
      `RetryingChatClient` retries with `Retry-After` support
- [x] `ChatAgent` with threads (serialize/deserialize + store factory),
      memory, `as_tool`, per-run `AgentRunOptions`, and agent / chat /
      function middleware
- [x] Trait-level `run_stream`: real token SSE through hosting (DevUI-style,
      AG-UI, OpenAI-compatible), incremental agent updates inside
      orchestrations, streaming A2A
- [x] `AiFunction::typed` — parameter schemas derived from Rust types
      (schemars), plus invocation limits and hosted-tool config setters
- [x] Providers: OpenAI (chat + Responses + **Assistants**), Azure OpenAI
      (chat + **Responses**), Azure AI Foundry (incl. Bing grounding /
      file-search configs), Anthropic (betas + hosted tools + citations),
      Copilot Studio; Entra ID credential chain incl.
      `DefaultAzureCredential`
- [x] Multimodal input (images/audio/files) and citation annotations on
      OpenAI + Anthropic; granular service errors (auth / invalid-request /
      content-filter)
- [x] MCP: stdio + HTTP + WebSocket, tools, prompts, sampling, roots, and
      first-class `ToolSource` integration with `list_changed` reloads
- [x] Workflow engine incl. signature-validated checkpointing (fan-in state
      included), concurrent supersteps, HITL, shared state, validation,
      viz, sub-workflows
- [x] All orchestrations incl. Magentic plan-review and stall-intervention
      HITL
- [x] A2A client (full task surface) and A2A serving (card + JSON-RPC)
- [x] Declarative YAML agents; Rust-native workflow specs
- [x] Hosting: DevUI-style API + embedded debug page, AG-UI (with client
      tools injected per-run), OpenAI-compatible
- [x] Redis (BM25), Mem0, Cosmos DB, Azure AI Search; Purview middleware
- [x] Observability: OTel GenAI spans (request + tool attributes) and
      optional GenAI metrics histograms (`otel-metrics`)
- [x] 99 runnable examples in [`examples/`](examples)

## Development

```bash
cargo build            # build all crates
cargo test              # run the test suite (no network required)
cargo clippy --all-targets
cargo fmt --check
```

## License

MIT. This is an independent port and is not affiliated with or endorsed by
Microsoft.
