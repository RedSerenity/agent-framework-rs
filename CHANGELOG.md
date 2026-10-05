# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor bumps
may break APIs).

## [Unreleased]

### Added

- **`agent-framework-harness`** — a new crate porting upstream's harness
  agent (Python `agent_framework._harness`, `create_harness_agent`; .NET
  `Microsoft.Agents.AI.Harness`), also reachable as
  `agent_framework::harness` behind the umbrella crate's new `harness`
  feature. `HarnessAgent::builder(client)` (or `client.as_harness_agent()`)
  assembles an `Agent` with the harness instructions, hosted web search,
  compaction, and the default providers, wrapped in a tool-approval
  decorator and, optionally, a loop:
  - `TodoProvider` (`todos_add` / `todos_complete` / `todos_remove` /
    `todos_get_remaining` / `todos_get_all`) over session-state or
    per-session JSON-file stores;
  - `AgentModeProvider` (`plan` / `execute`, `mode_set` / `mode_get`,
    `get_agent_mode` / `set_agent_mode` with external-change notification);
  - `FileAccessProvider` over an `AgentFileStore` — in-memory, or a
    `FileSystemAgentFileStore` sandboxed to its root (rejects `..`,
    absolute and drive-rooted paths, symlinks or reparse points on any
    segment and on the root itself, `O_NOFOLLOW` opens, links skipped in
    listings and searches) — with eight `file_access_*` tools that require
    approval by default, line-addressed editing, and linear-time,
    deadline-bounded regex search; optional per-session scoping;
  - `FileMemoryProvider` (seven `file_memory_*` tools, description
    sidecars, an auto-maintained `memories.md` index injected each run);
  - `MemoryContextProvider` + `MemoryFileStore` (topic files, `MEMORY.md`,
    JSONL transcript archive, LLM extraction and consolidation);
  - `BackgroundAgentsProvider` (six `background_agents_*` tools running
    sub-agents as tokio tasks, with lost-task detection and
    `release_session`);
  - `ToolApprovalAgent` ("always approve" standing rules for a tool or
    tool+arguments, heuristic auto-approval callbacks, queued one-at-a-time
    prompts, inbound-response binding to surfaced requests, and bypass of
    sibling calls that need no approval);
  - `LoopAgent` (`should_continue` / `next_message` / progress log,
    `fresh_context`, judge loops via `LoopAgent::with_judge`, and the
    `todos_remaining` / `background_tasks_running` helpers);
  - `SessionStateHistoryProvider`, the harness's default, state-backed
    history (with optional after-run compaction).
- Core: `SessionContext::session_state`, a by-reference handle on the run's
  `AgentSession::state` that `Agent` now passes to every provider's
  `before_run`, and `ContextProvider::after_run_in_session`, a session-aware
  completion hook (defaulting to `after_run`) that `Agent` now calls. Both
  are additive.

### Fixed

- Core: a resolved approval exchange replayed from persisted history no
  longer re-executes the approved tool on every later run of the session.
  The tool loop now drops approval requests/responses already answered by a
  later function result, and an approval-resuming response carries the
  resolved results (or rejections) so history records them.

## [0.9.0] — 2026-09-30

Four values the code already had and never read, a Foundry surface it could
not reach, and tool calls that neither hosting surface put on the wire.
Nearly all of it failed in the permissive direction: a cut-off turn reported
as a completed one, a blocked prompt reported as an answer, a hosted MCP
allowlist of none reported as all.

**Breaking, in three places.** `AgentResponse` and `AgentResponseUpdate` each
gain a `finish_reason` field, so a struct literal for either without
`..Default::default()` needs it. `agent_framework_hosting::ResponseObject`
gains `incomplete_details` on the same terms, and its `output` changes from
`Vec<OutputMessage>` to `Vec<OutputItem>` now that a response can carry
function calls beside its message — `OutputItem::as_message()` recovers the
old view.

**One behaviour change worth calling out.** The in-memory vector store's
`hamming` score is now the *fraction* of differing coordinates rather than the
raw count, matching upstream. Ranking is unchanged; a `score_threshold` tuned
against the old raw count needs dividing by the vector width.

### Added

- **Microsoft Foundry managed memory** (`FoundryMemoryProvider`, `foundry`
  crate). A `ContextProvider` that searches a Foundry memory store for
  relevant memories before a run and writes the turn back after it, spoken
  directly against the project data plane
  (`{endpoint}/memory_stores/{name}:search_memories` and `:update_memories`,
  `api-version=v1`, bearer-scoped to `FOUNDRY_SCOPE`). Memories are isolated
  by scope — `with_scope` pins one, otherwise the session id is used, and
  nothing is read or written when neither is available. Search and update
  cursors chain incrementally, the user-profile fetch happens once per scope,
  and every service failure is logged rather than raised, since `after_run`
  also runs on the agent's own failure path. Provider state is keyed by scope,
  so a provider shared by `Arc` across runs cannot let one session read
  another's profile or resume its cursor; because `after_run` is handed no
  session, a provider that has served more than one session declines to write
  rather than guess a scope — set `with_scope` when sharing one.
- **Foundry embeddings from a project endpoint.** `FoundryEmbeddingClient`
  previously spoke only the Foundry *Models* inference endpoint, so a Foundry
  project holding an embedding deployment could not be embedded against
  without provisioning a second surface. `with_project_endpoint` derives the
  resource-scoped `{resource}/openai/v1/embeddings` route from a project
  endpoint (`openai_model_base_url`), path-versioned and Entra-only. `from_env`
  reads `FOUNDRY_PROJECT_ENDPOINT` (and `FOUNDRY_ENDPOINT`, the name
  `FoundryChatClient` takes first) and still prefers `FOUNDRY_MODELS_ENDPOINT`
  when both are set (upstream #8454).
- **Tool calls on both hosting surfaces.** Core leaves a
  `FunctionCallContent` intact for the caller to execute, but neither host
  put it on the wire: `/v1/chat/completions` and `/v1/responses` serialized
  text only, so a client was told a call had been requested and given no id,
  name or arguments to act on. `/v1/chat/completions` now emits
  `message.tool_calls` (with `content: null` for a call-only turn, as OpenAI
  sends) and, when streaming, `delta.tool_calls` fragments keyed by a stable
  per-call `index`. `/v1/responses` now emits `function_call` output items
  as siblings of the assistant message, announced once over the stream as
  `response.output_item.added` and then fed
  `response.function_call_arguments.delta`. A call streamed across several
  updates is announced once, with its fragments reassembled, and the
  terminal payload reuses the ids the client already saw. Arguments go out
  as a JSON *string* on both surfaces, as the wire format has it. Consequently a `tool_calls` / `function_call` finish reason
  now passes through whenever the response actually carries a call, instead
  of degrading to `stop`; it still degrades when the turn declares none.

  Only calls that are still **unanswered** are serialized. Core keeps a
  call and its `FunctionResultContent` together in the response — after a
  local tool ran, and when a provider executed a hosted tool itself — and
  advertising those would ask the client to re-run work already done. The
  streaming surfaces hold calls until the stream ends to make that
  decision, because with local tools the result arrives in a later update
  than the call; a call's arguments therefore arrive in one delta rather
  than forming incrementally.

  The inbound half works too: `/v1/chat/completions` reads `tool_calls` and
  `tool_call_id` off the follow-up request, and `/v1/responses` reads
  `function_call` and `function_call_output` input items, so an executed
  call reaches the agent as a `FunctionCallContent` / `FunctionResultContent`
  pair rather than as empty text.
- `agent_framework_hosting::{OutputItem, OutputFunctionCall}`.
- `AgentResponse::finish_reason` / `AgentResponseUpdate::finish_reason`, and
  `incomplete_details` on the hosting `ResponseObject`.

### Fixed

- **A truncated Responses stream lost its terminal metadata** (`openai`
  crate). `parse_responses_event` handled `response.completed` but not
  `response.incomplete`, so a stream cut off by the content filter or the
  token budget delivered no finish reason, usage or response id — exactly
  the information the distinct event name exists to report. Both are now
  handled on the same arm.

- **An empty hosted-MCP allowlist enabled every tool on the server**
  (Anthropic). `allowed_tools: Some(vec![])` means "expose none", and was
  treated as "no allowlist" — leaving the API default, which enables all of
  them. Now encoded as `tool_configuration: {"enabled": false}`. The OpenAI,
  Azure and Foundry paths were already correct (upstream #8576).
- **A cut-off turn was reported as a completed one.** `finish_reason` never
  reached the agent types, so an Azure OpenAI content-filter block or a
  token-budget truncation was indistinguishable from a finished answer. The
  OpenAI-compatible surface no longer hardcodes `finish_reason: "stop"`, and
  the Responses surface emits `status: "incomplete"` with
  `incomplete_details.reason` (`content_filter` / `max_output_tokens`) and a
  `response.incomplete` terminal event (upstream #8478).
- **Hamming distance could not distinguish integers above 2^24.** Stored
  vector coordinates were narrowed to `f32` before comparison, so distinct
  large integers — ids, timestamps, hashes — compared equal and a
  non-matching record scored as an exact match. Coordinates are now compared
  at `f64` (upstream #8637).
- **Hamming could not distinguish 64-bit integers.** Widening stored vector
  coordinates to `f64` moved the collision boundary from 2^24 to 2^53; it did
  not remove it, since `f64` cannot represent `2^53 + 1` either. Hamming asks
  whether two coordinates are the *same*, and a store keyed on ids, timestamps
  or hashes lives exactly there — so stored coordinates now keep the integer
  JSON carried and Hamming compares integers as integers, while the
  arithmetic metrics keep using `f64`.
- **`finish_reason` was dropped by `response_to_updates`.** The buffered
  `SupportsAgentRun::run_stream` and `Agent` streaming through agent
  middleware rebuild updates from an `AgentResponse`, and that helper did not
  carry the reason across — so a content-filtered turn still reached hosting
  as `stop`/`completed` on those paths. It now rides the final update, and a
  reason with no messages emits one, matching what `client.rs` does for
  `ChatResponse`.
- **An incomplete response carried a completed output item.**
  `OutputMessage::assistant_text` hardcodes `status: "completed"`, so a
  truncated or filtered response contained a message item claiming it
  finished — on the buffered and DevUI streaming paths alike. The item's
  status now follows its response's.
- **A run with no searchable input lost its memories.** An instruction-only
  run, or one carrying only system turns, returned from
  `FoundryMemoryProvider::before_run` before injecting the profile already
  fetched for its scope. The contextual search is still skipped (there is
  nothing to search with); the injection is not.
- **The memory provider held its state lock across HTTP calls**, so one slow
  scope stalled every other session sharing the provider. It now snapshots
  under the lock, releases it for the request, and re-acquires to record the
  result.
- **An unfamiliar finish reason was reported as a completed turn.** The
  Responses surface mapped only `content_filter` and `length`, defaulting
  everything else to `completed`. `FinishReason` is an open string and
  providers use it — the Anthropic converter deliberately preserves
  `model_context_window_exceeded` — so the abnormal endings that matter most
  were the ones swallowed. Completion is now an allowlist (absent, `stop`,
  `tool_calls`); every other reason is reported incomplete and passes through
  verbatim.
- **The scope cache never came back down after a burst.** Keeping in-flight
  scopes alive (below) lets the cache exceed its bound, but eviction ran
  only when inserting a *new* scope and dropped at most one slot — so a
  burst of concurrent sessions stayed resident permanently. Trimming now
  runs on every touch and loops until the bound is met, making the overrun
  temporary, which is the only thing that made it acceptable.
- **Evicting a scope with a request in flight defeated per-scope locking.**
  The LRU dropped the least-recently-used slot unconditionally, so a slow
  run could have its slot evicted, the next run for that scope would build
  a *second* mutex, and the two would race and fork the cursor — the exact
  failure per-scope locking exists to prevent. Only a slot nobody holds is
  now evictable; when every candidate is busy the cache runs briefly over
  capacity instead.
- **A tool finish reason was advertised without the tool calls.** Reporting
  `tool_calls` / `function_call` tells a client to execute the call in
  `message.tool_calls`, and the OpenAI-compatible host serializes text only
  — so the client was handed an instruction with no id, name or arguments.
  Those two reasons now degrade to `stop` with the real one in
  `x_finish_reason`, until the surface can serialize the calls it is
  promising. (Recorded as a capability gap, not a permanent answer.)
- **Concurrent turns on one memory scope forked the update chain.** The
  provider snapshotted `previous_update_id`, released its lock for the
  request and wrote back, so two turns for the same scope both resumed from
  the same cursor and one branch fell out of the service's incremental
  chain. The provider now holds **one lock per scope** rather than one per
  provider: the snapshot → request → commit sequence is atomic within a
  scope, and unrelated scopes still never wait on each other. The same
  change removes the duplicate profile fetch the previous release note
  described as merely idempotent.
- **A `function_call` turn was reported as a failure.** OpenAI's finish
  vocabulary has five values, not four: `function_call` is the deprecated
  spelling of `tool_calls` and, like it, marks a turn that *succeeded*. It
  was missing from core's `FinishReason` constants, so the Responses surface
  called it incomplete and the Chat Completions surface rewrote it to
  `length` — telling clients a working tool call had been truncated.
  `FinishReason::FUNCTION_CALL` now exists and both surfaces treat it as the
  completion it is.
- **The DevUI stream closed an incomplete response with
  `response.completed`.** The terminal event was chosen from
  `incomplete_details`, which the fix below makes absent for an unfamiliar
  provider reason — so precisely the cut-off turns this path exists to flag
  were announced as successes. It now follows the response's `status`.
- **A provider-only reason could reach the Chat Completions `finish_reason`
  enum.** That wire field is a closed set, and a strict generated client can
  reject a response over a value outside it. An unfamiliar reason is now
  reported as `length` — the only legal value that says "not a complete
  answer" without asserting a cause — and carried verbatim in a
  non-standard `x_finish_reason` beside it, on both the buffered and
  streaming paths.
- **A failed concurrent profile fetch could erase a scope's memories
  permanently.** Releasing the state lock for the request (below) lets two
  first runs for one scope overlap; the failing one committed an empty
  profile over the successful one's and set `initialized`, so nothing ever
  re-fetched it. A failure now leaves the cached profile untouched.
- **The scope cache grew without bound.** With the session fallback a scope
  is a session, so a long-lived agent accumulated one `ScopeState` per
  session forever. It is now an LRU bounded by `DEFAULT_MAX_CACHED_SCOPES`
  (512), overridable with `with_max_cached_scopes`; eviction costs a scope
  only a re-fetched profile and a restarted cursor.
- **`incomplete_details.reason` could carry a value the schema forbids.**
  The Responses schema names exactly two reasons, and a strict generated
  client can reject a whole response over a third — so a provider string
  like `model_context_window_exceeded` no longer goes in that field. Such a
  run is still reported `incomplete`, and the raw reason moves to the new
  `x_finish_reason` extension, which a strict client ignores rather than
  has to parse.
- **The project embeddings route dropped the OpenAI `user` option.** The
  per-route allowlist added below pruned Inference-only fields but also
  pruned `user`, which the derived `/openai/v1/embeddings` endpoint *does*
  accept and which `agent-framework-openai` and `agent-framework-azure` both
  forward on the same surface. The two routes now differ in both directions:
  `input_type` is Models-only, `user` is project-only.
- **The project embeddings route forwarded Azure AI Inference-only fields.**
  Selecting the project route changed the URL but not the payload, so
  `input_type` — an Inference field with no OpenAI equivalent — was still
  sent to the derived `/openai/v1/embeddings` endpoint, which rejects it.
  Request properties and the `extra-parameters` pass-through header are now
  chosen per route.
- **A failed contextual search discarded the cached profile.** In
  `FoundryMemoryProvider::before_run`, a transient failure of the second
  search returned early and dropped the static memories the first had
  already fetched. It is now treated as an empty contextual result, so the
  known-good profile is still injected.
- **A blank `FOUNDRY_PROJECT_ENDPOINT` shadowed the `FOUNDRY_ENDPOINT`
  alias.** The fallback was selected before the blank check, so an env
  template declaring the optional variable as `""` suppressed a perfectly
  good alias. Each candidate is now filtered before the fallback.
- **A whitespace-only instruction became a contentless system message.**
  `prepare_messages` skipped `""` but not `" "` or `"\n"`. A real instruction
  still prepends verbatim (upstream #8524).

### Documentation

- `StandardMagenticManager` now documents that its cached task ledger is
  shared by every run holding the same manager, and that the two readers of
  that cache are the plan-review and stall-intervention requests — so
  concurrent runs can show a human reviewer the other run's plan. Prefer
  `standard_manager`, which takes the manager by value (upstream #8581).


## [0.8.0] — 2026-09-21

An Azure Cosmos DB vector store and a provider that hands any vector
collection to an agent as tools, plus four faults in the Purview middleware
that each let content reach the model without being evaluated — or, in one
case, evaluated as its own escaping.

**Breaking, in four places.** `Error::ServiceContentFilter` gains a `detail`
field, so a struct-variant destructuring without `..` stops compiling.
`FunctionInvocationConfig` gains `allow_concurrent_invocation`, so a struct
literal without `..Default::default()` needs it. `DlpAction` gains
`RestrictAccess`, so an exhaustive match over it needs an arm. And
`ProcessConversationMetadata::content` is now a `PurviewContent` rather than
a `PurviewTextContent`, because an attachment is submitted as binary.

**Two behaviour changes in the Purview middleware.** It now **fails closed**
when no Entra user id can be resolved: previously it reported the content as
allowed, which is an unevaluated message reported as a cleared one.
`ignore_exceptions` still trades that for availability, deliberately and for
every error rather than only this one. And it submits **every content item**
rather than each message's text, so a deployment will see more
`processContent` calls and, correctly, more verdicts.

**One behaviour change in core serialization**, unrelated to Purview:
`FunctionResultContent::exception` now serializes to a fixed marker, so a
conversation persisted from here on keeps the *fact* of a tool failure but not
its diagnostic text. Conversations already stored are unaffected, the model
still sees the real text, and Purview still evaluates the real text — the
evaluation path reads the live value rather than the serialized one.

### Added

- **An Azure Cosmos DB for NoSQL vector store**: `CosmosVectorStore` and
  `CosmosVectorCollection` in `agent-framework-cosmos`. A container becomes a
  collection — the `vectorEmbeddingPolicy` and `indexingPolicy` at creation,
  point reads and upserts in the item's own partition, and a `VectorDistance`
  query with every filter literal bound as a parameter.

  Four details are decisions rather than defaults. The vector surface needs
  its own api-version (`DEFAULT_VECTOR_API_VERSION`, `2020-07-15`): the
  `2018-12-31` the history and checkpoint stores speak predates vector search,
  so a container created under it comes back without the policy it asked for.
  A distance function Cosmos cannot compute is **refused** rather than
  substituted, since Cosmos computes cosine as a *similarity* and a silent
  substitution ranks by a metric the caller's `higher_is_closer` then reads
  backwards. `skip` rides in the limit, because Cosmos will not take `OFFSET`
  beside an `ORDER BY VectorDistance`. And an ordered filter on an untyped
  field carries a SQL type guard, because Cosmos SQL orders *across* types.

- **`VectorCollectionContextProvider`** (`agent_framework_core::vectors`):
  generates search / get / upsert / delete tools over any `VectorCollection`,
  so the in-memory, Azure AI Search and Cosmos DB stores become things an
  agent can use rather than things a caller drives.

  Writes require approval by default and reads do not. The `scope_filter` is
  applied everywhere — conjoined into search, checked on the records a read,
  write or delete touches, and checked against the *existing* record on an
  upsert so a scoped agent cannot overwrite another group's record by naming
  its key. An out-of-scope record reads as absent rather than as a refusal,
  since "you may not read that" confirms it exists. It remains grouping rather
  than an authorization boundary, and the module docs say so in those words.

  The model never authors an embedding: `embed_from_field` names the field the
  vector is derived from, and the vector field is not in the tool schema at
  all. Without it there is no upsert tool rather than one that cannot work —
  the same reason a multi-vector collection gets no upsert tool, since one
  embedding source cannot maintain several vectors and a whole-document write
  would drop the rest.

- **Azure content-filter detail** on `Error::ServiceContentFilter`:
  `ContentFilterDetail` carries the policy code, the `param`, and the
  per-category verdicts from Azure's nested `innererror`, reaching Azure
  OpenAI chat, Responses and embeddings. Every value is an open string —
  upstream parses the code and severity into enums, which raise on a value
  Azure has not shipped yet.

- **`reasoning_details` on Chat Completions**, in both directions. A
  reasoning-capable provider on that surface (DeepSeek in thinking mode,
  OpenRouter, vLLM) requires the payload back on the next request of the same
  turn, so a conversation that used reasoning could not be continued. It rides
  in `TextReasoningContent::protected_data`, and fragments split across a
  stream concatenate in order.

- **`FunctionInvocationConfig::allow_concurrent_invocation`**: runs one model
  response's tool calls in model order when `false`. The point is the side
  effects rather than the result order — a tool holding a non-reentrant handle
  behaves differently when two invocations overlap.

- **`FUNCTION_INVOCATION_ERROR_MARKER`** and
  `FunctionResultContent::is_error`.

### Fixed

- **Purview evaluated a fraction of each message.** It submitted
  `Message::text()` and nothing else, so a function result — where exfiltrated
  data actually appears — an attachment, a reasoning block and a tool call all
  reached the model unevaluated. Every content item is mapped now, one request
  per entry: an attachment's bytes go as Graph `binaryContent` (handing a
  classifier the base64 *string* reads as gibberish and passes every policy),
  and everything else is serialized whole. Only `usage` is skipped. This port
  had a hole on top of upstream's: `text()` returns `""` whenever a refusal is
  present, so a partly-declined turn submitted nothing at all.

- **Purview could not parse a verdict carrying `restrictAccess`**, losing it
  entirely. The action is named now; it is deliberately not a block on its
  own, since its `restrictionAction` selects the mode. The enums stay strict:
  an unrecognized value fails the parse, which the default
  `ignore_exceptions = false` turns into a stopped request, rather than
  landing in a catch-all that `should_block` would answer `false` for.

- **Purview reported an unevaluated message as allowed** when no user id
  resolved. See the behaviour note above.

- **A failing tool's diagnostics were persisted verbatim** into every history
  store and workflow checkpoint. `FunctionResultContent::exception` is
  host-internal text this framework does not control — a connection string or
  a SQL error quoting the row it failed on are ordinary things to find in one.
  It serializes as a fixed marker now; failure state survives, deserialization
  is untouched, and the model is unaffected because every provider converter
  reads the field directly.

- **`gen_ai.client.operation.duration` omitted failed calls**, which leaves no
  error rate at all and skews the latency distribution. All three failure
  paths record with `error.type` now, including the stream that fails midway —
  which ends before its completion arm and so was missing entirely.

- **A streamed `function_call` fragment could land on the wrong call.** An
  untagged continuation delta went to the *first* in-flight call rather than
  the one still streaming; an untagged chunk could be absorbed by a call
  already carrying an occurrence id; two calls sharing a provider `call_id`
  but carrying *different* occurrence ids merged, appending one call's
  arguments onto the other's; and a failed merge dropped the fragment silently.

- **A handoff dropped the user's attachments.** An image or file is the
  request as much as the words beside it. Multimodal content is kept on user
  messages and still dropped elsewhere, since providers reject input-only
  parts replayed on an assistant turn. The same pass stops an approval
  *response* riding on a user message through to the next agent.

- **Concurrent checkpoint saves collided on one temp file.** Two saves of the
  same id wrote into the same file, so the survivor could be a blend of both,
  and the loser's rename failed with `NotFound`.

- **A Cosmos id containing `%` stored but could not be read back.** The id is
  percent-encoded as a path segment now while the signature stays over the raw
  link, as Cosmos's auth scheme requires. Previously a point read 404'd and a
  delete reported success while leaving the record.

- **A percent-escaped data URI was evaluated as its own escaping.** RFC 2397's
  data segment is URL characters, so the payload may arrive escaped; decoding
  that as base64 fails and the item went to Purview as text. Payloads are
  percent-decoded and unwrapped before the base64 decode.

### Changed

- The Cosmos chat store documents that a thread id **selects** history rather
  than protecting it: distinct ids prevent accidental overlap without
  restricting a client whose credentials already authorize the container.

## [0.7.0] — 2026-09-15

Portable vector filters and an Azure AI Search vector store, bounds on the
tool loop, and four fixes to isolation and replay.

**Breaking, in five places.** `VectorSearchOptions::filter` is now a
`FilterExpression` rather than a provider-dialect `String`; the string form
moved to `provider_filter` / `with_provider_filter`.
`McpStreamableHttpTransport::new` returns a `Result` (it validates the URL).
`SecretString` no longer implements `Serialize`, so a `#[derive(Serialize)]`
on a struct holding one stops compiling — deliberately; see below. Every
workflow graph signature changes, so a checkpoint written by 0.6.x is refused
on resume with a message naming the scheme change (and
`run_from_checkpoint_unchecked` still resumes it). And `VectorSearchResult`
gains a `score_kind` field, so code constructing one with struct-literal
syntax needs updating (it does not derive `Default`). Reading one is
unaffected.

Two behaviour changes to look for. Redis keys are now derived through an
injective encoding: a literal-safe prefix and session id — the default prefix
and any UUID — produce exactly the keys they always did, so existing data is
unaffected, but an identifier containing `:`, uppercase, or a glob character
now addresses a different key than before. And an Anthropic reasoning content
with no signature is no longer sent as a thinking block; with extended
thinking enabled the API rejects one, so this turns a 400 into a request that
works.

### Added

- **Portable vector filters** (`agent_framework_core::vectors::filters`):
  `Filter` leaves and `FilterGroup` nodes across 18 operators, validated at
  construction (operand shape per operator, depth ≤ 8, ≤ 64 nodes), with a
  namespaced escape hatch (`azure_ai_search.match`) for anything a connector
  defines itself. `InMemoryVectorStore` evaluates them instead of refusing
  every filter.

  The semantics are upstream's: a missing field is a non-match for every
  operator except `exists` (so `ne` and `not(eq)` differ), a boolean never
  equals a number, and numbers compare by **value** — a record that
  round-tripped through JSON as `1` matches `eq: 1.0` — with integers compared
  as integers, since routing them through `f64` to get that rule rounds past
  2^53 and makes two adjacent 64-bit ids equal. An expression that never went
  through a constructor (`Deserialize` bypasses them) is reported as a
  configuration error by `matches` rather than panicking or silently becoming
  match-all.

- **An Azure AI Search vector store**: `AzureAISearchStore` and
  `AzureAISearchCollection` in `agent-framework-azure-ai-search`. Index
  create/exists/delete from a `VectorStoreCollectionDefinition` (EDM types,
  vector profiles, hnsw vs exhaustiveKnn, metric mapping), document
  upsert/get/delete, vector search with the portable filter translated to
  OData, keyword-hybrid search, and index aliases. `build_index()` and
  `prepare_filter()` are public so a caller driving the REST API themselves
  can reuse either.

  Details that decide whether results are right: filtering is `preFilter` and
  `k` covers `skip + top`, so a selective filter does not return fewer than
  `top`; `upsert` sends `upload`, honouring the trait's insert-or-*replace*
  contract rather than merging; an indexing batch is bounded by serialized
  payload size as well as the 1,000-action limit, because a 1536-dimension
  embedding is ~18 KB and a full batch would be ~18 MB; a **207** partial
  success is raised rather than read as success; and an operator Azure cannot
  express is refused with the alternative named, never silently dropped.

- **Bounds on the function-invocation loop**:
  `FunctionInvocationConfig::max_function_calls` caps total tool executions
  per request and `max_duration_seconds` caps elapsed wall time.
  `AgentBuilder::function_invocation_config` reaches the whole config from an
  agent for the first time, `max_iterations` included.

  Both bounds are cumulative across approval round trips — parked in the
  session, resumed only by a request that actually carries the approval
  responses, and discarded when a run ends or fails — and both are graceful:
  tools are disabled and the model answers with what it has. The call budget
  charges only calls that reached a tool, so a hallucinated tool name costs
  nothing. The clock is re-checked after each model call, so a slow provider
  response cannot buy a free tool batch. And a spent budget stops *executing*
  rather than only setting `tool_choice`, which a provider may ignore; when it
  expires mid-response, work the provider already did (a hosted tool call and
  its result) is kept and handed to the final model call.

- `agent_framework_core::storage_keys::storage_key_segment`, the derivation
  behind the Redis key fix below.

- `agent_framework_azure::DEFAULT_CHAT_API_VERSION` /
  `DEFAULT_EMBEDDING_API_VERSION`, both public.

### Fixed

- **MCP HTTP headers leaked across a redirect.** Headers attached to an
  `McpStreamableHttpTool` went onto a client that follows redirects and strips
  only `Authorization`, `Cookie` and `Proxy-Authorization` on a cross-*host*
  hop — so an `X-Api-Key` reached whatever host the server redirected to, and
  even `Authorization` survived a redirect to a different port or scheme. The
  `Mcp-Session-Id` was both sent to and adopted from a redirect target. The
  transport now follows redirects itself, scoped to scheme/host/port, and
  preserves the POST method and body across every hop. Session teardown goes
  the same way.

- **Anthropic extended-thinking signatures were dropped**, on both the
  buffered and streaming paths, and `redacted_thinking` blocks vanished
  entirely. With extended thinking enabled the Messages API requires a
  replayed thinking block to carry its signature, so a conversation that used
  it could not be continued — the tool-result turn was rejected. Signatures
  now ride in `protected_data`, a decoded block is replayed verbatim (the
  signature covers the exact text), and two thinking blocks in one streamed
  message stay separate instead of coalescing into one that neither signature
  covers.

- **Redis keys were ambiguous.** `{key_prefix}:{session_id}` addressed one
  list for `("chat", "a:b")` and `("chat:a", "b")`, so two conversations that
  should be isolated shared a history — one tenant reading another's, where
  the prefix is the tenant boundary. The context provider had a second
  variant: a prefix containing `:entry:` put one provider's entries inside
  another's `SCAN MATCH`, which `clear()` would then delete.

- **The workflow graph signature was not injective.** Executor ids were joined
  with `,` and `->`, so a fan-out to `["x", "y"]` and one to `"x,y"` signed
  identically and a checkpoint from either was accepted for the other. Now
  JSON-encoded; the scheme tag is `v2`, and a `v1` checkpoint is reported as a
  scheme mismatch rather than as a graph change.

- **`SecretString` serialized its secret in cleartext.** The derive is
  removed rather than masked, because a masked `Serialize` round-trips the
  mask back as the value.

- **The Azure chat api-version was pinned to GA `2024-10-21`**, which rejects
  request fields this client sends (`store` first among them) with
  "Unrecognized request argument supplied". Now `2024-12-01-preview` for chat,
  matching upstream, with embeddings left on GA as upstream also has it.

### Changed

- `VectorSearchResult` gains `score_kind` and `higher_is_closer(..)`. A
  collection's distance function is a *request*, and on several services it
  does not describe what comes back: Azure AI Search returns a
  higher-is-better relevance score whatever metric the profile declares, and a
  reciprocal-rank-fusion score for a hybrid query. A caller reading direction
  from `DistanceFunction::higher_is_closer` — false for the distance metrics,
  which are the usual declaration — would have ranked every result backwards.
  `InMemoryVectorStore` leaves it `None`, meaning the declared function does
  describe the score. Mirrors upstream's `score_kind` result metadata.

- `VectorStoreCollectionDefinition` gains `try_get_field` and
  `storage_name_for`.

- `AzureAISearchStore::collection` returns the concrete collection type, for
  the surfaces (`search_hybrid`, `build_index`) the object-safe
  `VectorCollection` trait cannot declare.

## [0.6.0] — 2026-09-09

Provider refusals handled properly end to end, Entra ID authentication for
Cosmos DB, and the first vector-store abstractions.

**This one can break a compiling caller**, unlike 0.4 and 0.5. Two public
structs gained a field: `TextContent::refusal` and `FunctionCallContent::id`.
Neither is `#[non_exhaustive]`, so any code constructing them with
struct-literal syntax needs updating. `TextContent` derives `Default`, so a
`..Default::default()` literal keeps working; `FunctionCallContent` does not,
so its literals must add `id` (or move to `FunctionCallContent::new`, which
takes the same three arguments as before). Nothing was removed, and no
function signature changed.

The behavior change to check for is refusal handling. A provider decline used
to arrive as ordinary text, so `Message::text()`, `ChatResponse::text()` and
`AgentResponse::text()` returned the refusal prose as though it were the
answer — and `parse_json` tried to parse it. They now return `""` for a
refused turn, and the decline is read deliberately through `refusal_text()`.
Code that displayed whatever `text()` returned will show an empty string
where it used to show "I can't help with that"; branch on `has_refusal()` to
render a decline. This is the fix, not a regression, but it is visible.

### Added

- **Microsoft Entra ID authentication for `agent-framework-cosmos`.**
  `CosmosChatMessageStore::with_token_credential` and
  `CosmosCheckpointStorage::with_token_credential` take any
  `agent_framework_azure::TokenCredential`, so a Cosmos store authenticates
  with a managed identity, a workload identity, or the Azure CLI instead of a
  master key — and a Cosmos account with `disableLocalAuth` set, which has no
  key to give, becomes usable at all.
  `CosmosChatMessageStore::from_state_with_token_credential` restores a
  serialized store, since a credential cannot round-trip through a state blob
  the way a key does. The token scope defaults to
  `https://cosmos.azure.com/.default` — Cosmos DB's data-plane audience is
  service-wide, not per account — and `AZURE_COSMOS_AAD_SCOPE_OVERRIDE`
  overrides it, matching the official `azure-cosmos` SDK.

  Note that Cosmos DB's Entra RBAC grants data-plane actions only:
  `ensure_created` cannot succeed with a token whatever role the principal
  holds, so the database and container must be provisioned out of band. The
  call now says so rather than surfacing a bare `403`.

- **`InMemoryVectorStore` validates vectors before scoring**: non-numeric
  elements, wrong-width stored vectors, and non-finite values are skipped
  rather than reshaped or ranked, and a non-finite query vector is rejected
  outright (it would score every record identically and produce a score that
  cannot serialize as JSON).

- **Vector-store abstractions** (`agent_framework_core::vectors`):
  `VectorStoreField`, `VectorStoreCollectionDefinition`, `IndexKind`,
  `DistanceFunction`, `VectorSearchOptions`, `VectorSearchResult`, the
  object-safe `VectorCollection` and `VectorStore` traits, and an
  `InMemoryVectorStore`. Records are `serde_json::Value` objects keyed by
  field name; a definition maps logical names to storage names. No provider
  crate implements the traits yet.

- **`preserve_first_user()`** on the `Truncation`, `SlidingWindow` and
  `TokenBudget` compaction strategies: retain the earliest user message
  whatever the budget, so a long conversation cannot lose the request it is
  about. Off by default; a preserved message is kept regardless of budget, so
  the result may exceed the configured limit by one.

- **`FunctionCallContent::id`**, a framework-generated occurrence id
  (`af-call-<uuid>`) minted when a call is deferred for approval, plus
  `ensure_occurrence_id()` and `same_invocation()`. Approvals now bind to a
  specific occurrence rather than to the provider `call_id`, which providers
  reuse.

- **`TextContent::refusal`** and `TextContent::refusal(..)`, plus
  `Message::has_refusal()` / `Message::refusal_text()`.

### Fixed

- **Provider refusals no longer read as answers.** An OpenAI refusal was
  parsed into plain text, so `Message::text()` returned it as though the model
  had answered and `parse_json` would try to parse it as the requested output.
  Refusal text is now marked, `Message::text()` returns `""` when one is
  present, and streamed coalescing keeps refusal and ordinary text in separate
  content items. The Chat Completions parser also used to emit a refusal only
  when there was no content, hiding a model that answered part of a request
  and declined part; both are now kept. Streamed refusals are parsed too
  (`delta.refusal` on Chat Completions, `response.refusal.delta` on
  Responses) — they were dropped outright, so a streamed decline reached the
  caller as an empty response. And `structured_output_text` withholds a
  refused turn, so `parse_json` / `value` cannot be populated from a refusal
  (nor fall back to an older turn's JSON and serve it as this run's answer).
  The guard is applied at the response level too — `ChatResponse::text`,
  `AgentResponse::text` and both streaming updates — because a tool loop
  accumulates earlier assistant turns, so blanking only the refusing message
  still handed back an intermediate aside as the final answer. On the two
  response types that guard is anchored to the run's **final** assistant
  turn rather than to any accumulated one: a provider may decline part of a
  request while still calling a tool for the rest (OpenAI puts `refusal` and
  `tool_calls` on the same message, and the parser keeps both), and an
  `any`-style check let that carried turn blank the successful answer that
  followed it. A run that ends on a refusal is still withheld whole.
  `has_refusal()` / `refusal_text()` are available on each. The **outbound**
  path preserves the marker too: replaying a refusal as history now uses the
  assistant message's own `refusal` field (Chat Completions) and a
  `{"type": "refusal"}` output content part (Responses), instead of folding
  it into ordinary content and telling the provider the assistant answered.

- **An empty vector-store storage-name override is rejected.** A
  `VectorStoreField` whose `name` was empty was refused, but one carrying
  `storage_name: Some("")` was accepted and produced the same broken result:
  the field is written under an empty JSON key. `InMemoryVectorStore`
  tolerates that, so it passed locally and would have failed at a real
  provider's collection creation or first write. An absent override is
  unaffected — it still falls back to the field name.

- **Two approvals pending under one provider `call_id` are no longer
  conflated.** They were matched structurally, so a second pending approval
  for the same call looked like a replay and was dropped, and one result
  answered both. Occurrence ids now decide identity when present, with the
  structural rule as a fallback so approvals stored before this keep
  resolving.

- **Purview asks for inline evaluation.** Every `processContent` request now
  carries `Prefer: evaluateInline`. Without it the service may evaluate
  content offline and return no actionable verdict, which would leave the
  blocking middleware with nothing to decide on — enforcement would quietly
  become a no-op rather than fail.

- **A token count reported as zero is no longer dropped from the OpenAI usage
  breakdown.** `parse_usage` skipped any `completion_tokens_details.*` /
  `prompt_tokens_details.*` value equal to `0`, mirroring a truthiness bug in
  upstream's Python (since fixed there), so a provider reporting zero audio,
  accepted-prediction, rejected-prediction or cached tokens produced no entry
  at all — indistinguishable from a provider that does not report that count.
  Both now appear as `Some(0)` and `None` respectively in
  `UsageDetails::additional_counts`, which is what the GenAI metrics layer
  reads. Non-integer values are still ignored. The Responses path and the
  typed fields (`reasoning_output_token_count`, `cache_read_input_token_count`)
  were already correct, so this also removes a silent disagreement between the
  two paths about the same response.

### Changed

- `CosmosChatMessageStore::serialize` omits `key` and emits
  `"auth": "token_credential"` for a credential-authenticated store (a
  master-key store's state is unchanged). `from_state` on such a blob now
  returns an error naming `from_state_with_token_credential` rather than
  reporting a missing field.

## [0.5.0] — 2026-08-31

Embeddings for two more providers, and a Gemini finish-reason fix.

Nothing was removed and no existing public item changed shape, so a compiling
caller should keep compiling. The bump is a minor one for the new surface —
two embedding clients, both re-exported from the prelude — rather than for an
incompatibility. Pre-1.0 that is still a compatibility break to cargo, so
dependants pinned to `0.4` need to move to `0.5` to pick this up.

The one behavior change is in the Gemini client: finish reasons that used to
arrive as raw lowercased strings now arrive as the canonical values, and
`FINISH_REASON_UNSPECIFIED` now reads as absent. Code matching on the old
strings needs updating — see **Fixed** below.

### Added

- **`BedrockEmbeddingClient`** (`agent-framework-bedrock`) — Amazon Titan Text
  Embeddings through Bedrock Runtime `InvokeModel`
  (`POST /model/{modelId}/invoke`), SigV4-signed by the same `sigv4` module
  the Converse chat client uses.
  - Titan's body carries a single `inputText` rather than a batch, so a batch
    of *n* values costs *n* signed requests, issued concurrently and
    reassembled in input order, with `inputTextTokenCount` summed across the
    batch into `UsageDetails::input_token_count`.
  - `with_endpoint` overrides the base URL — a PrivateLink interface
    endpoint, a region the host derivation does not cover, or a local test
    server. Both the `Host` *and* any path prefix on that URL are folded into
    the canonical request before signing, so the signature always covers the
    path actually requested; an endpoint or proxy mounted under a sub-path
    works rather than failing AWS authentication.
  - Credentials are the static or `AWS_*`-environment ones the chat client
    already takes. Upstream reaches Bedrock through boto3 and inherits its
    whole credential chain; this crate signs directly and does not.
  - Titan's `normalize` knob is forwarded through
    `EmbeddingGenerationOptions::additional_properties`.
- **`FoundryEmbeddingClient`** (`agent-framework-foundry`) — the Azure AI
  Foundry **Models** inference endpoint (`POST {models_endpoint}/embeddings`),
  with `api-key` or Entra `TokenCredential` auth. Distinct from
  `AzureOpenAIEmbeddingClient`, which is deployment-scoped; the body and
  response are OpenAI-shaped, so parsing and error classification are shared
  with `agent-framework-openai` rather than duplicated.
  - **Text inputs only.** Upstream also accepts image `Content` and splits a
    batch across `ImageEmbeddingsClient` (`/images/embeddings`). The core
    `EmbeddingClient` trait takes `Vec<String>`, so an image input cannot be
    expressed at that boundary at all — closing this needs a trait widening
    across every provider, not a change to this client.
    `FOUNDRY_IMAGE_EMBEDDING_MODEL` is correspondingly not read.
  - `DEFAULT_API_VERSION` reproduces the default in `azure-ai-inference`
    1.0.0b9 (the version upstream's `foundry` package pins), since upstream
    sends no explicit `api-version` of its own. It is the one value here taken
    from an SDK default rather than a verified service contract, so
    `with_api_version` overrides it.
  - Entra tokens are requested for `DEFAULT_SCOPE`
    (`https://cognitiveservices.azure.com/.default`), the Azure AI Services
    **data plane** audience — *not* `FOUNDRY_SCOPE`
    (`https://ai.azure.com/.default`), which is the Foundry **project**
    audience `FoundryChatClient` needs for the Responses API. The Models
    inference endpoint rejects a token minted for the project scope.
    `with_scope` overrides it.
  - `extra_parameters` is **expanded** into the request body under each
    entry's own name, matching what upstream's `model_extras` mapping does,
    rather than sent as a literal `extra_parameters` field the model would
    ignore. Expanding also sends `extra-parameters: pass-through`, without
    which Azure AI Inference rejects body fields outside its schema.
    `encoding_format` and `input_type` continue to forward verbatim.
  - `additional_properties` is an **allowlist**, matching the kwargs upstream
    builds and the sibling OpenAI/Azure clients: only `encoding_format`,
    `input_type` and `extra_parameters` reach the wire. Copying arbitrary
    entries through would put fields outside the inference schema on the
    request, which the service rejects — so an options struct reused across
    providers (carrying, say, OpenAI's `user`) would 4xx on an option that is
    merely irrelevant here. `extra_parameters` remains the escape hatch for
    anything genuinely model-specific.
  - `from_env` requires only the endpoint and model, matching upstream's
    `required_fields`. With no `FOUNDRY_MODELS_API_KEY` it falls back to
    `DefaultAzureCredential` at the scope above, so a managed-identity or
    `az login` environment works keyless — the same fallback
    `FoundryChatClient::from_env` already had.
- Both clients are re-exported from `agent_framework::prelude` under their
  existing `bedrock` and `foundry` features. No new default dependencies;
  `agent-framework-foundry` gains `reqwest` and `agent-framework-openai`.

### Fixed

- **Base64 embedding responses were rejected.** Every OpenAI-shaped embedding
  client here (`OpenAIEmbeddingClient`, `AzureOpenAIEmbeddingClient`,
  `MistralEmbeddingClient`, `OllamaEmbeddingClient`, and the new
  `FoundryEmbeddingClient`) forwards `encoding_format` verbatim, but the
  shared response parser accepted only a numeric `embedding` array. Asking for
  `encoding_format: "base64"` — a documented, forwarded option — therefore
  failed a perfectly successful response. The parser now decodes the base64
  form (packed little-endian `f32`s) as well, and reports undecodable or
  misaligned payloads rather than silently truncating. Pre-existing since
  embeddings were introduced, not new in 0.5.0.
- **Gemini finish reasons.** Five of the names in the Gemini API's
  `FinishReason` enum were not in `agent-framework-gemini`'s mapping table and
  fell through its passthrough arm as lowercased raw strings, so a caller
  matching on the canonical value saw `"language"`, `"image_recitation"`,
  `"image_prohibited_content"`, `"malformed_function_call"` or
  `"unexpected_tool_call"` where `content_filter` or `tool_calls` was meant.
  All five now map, matching upstream's table (#7837).
- **Gemini `FINISH_REASON_UNSPECIFIED`.** Proto3's "field never set" was being
  reported as a finish reason named `finish_reason_unspecified`. It now reads
  as *absent*, like a response carrying no `finishReason` at all — so a turn
  that ends in a function call is upgraded to `tool_calls` as it already was
  when the field was omitted entirely.

## [0.4.0] — 2026-08-26

Two optional integrations: Entra ID credentials from the official Azure SDK
for Rust (GA'd in May 2026), and a ready-made OTLP export pipeline for the
spans and GenAI metrics this framework already emits.

Nothing in the existing API was removed or changed, and the default build
gains no dependencies, so nothing here should break a compiling caller. The
minor bump reflects the size of the new surface and the build requirement
below rather than an incompatibility — but pre-1.0 it is still a
compatibility break to cargo, so dependants pinned to `0.3` need to move to
`0.4` to pick this up.

> **Build requirement for the new features.** Enabling `entra-sdk` or
> `otel-export` requires a **C toolchain (`cmake`)**. Both pull `aws-lc-rs`,
> which is the TLS provider for the reqwest 0.13 that `azure_core` and
> `opentelemetry-otlp` depend on — a separate major version from the reqwest
> 0.12 used elsewhere here, whose features cargo cannot unify with it. Builds
> that do not enable either feature are unaffected.

### Added

- **Entra ID credentials from the official Azure SDK for Rust**, behind
  `agent-framework-azure`'s new optional `entra-sdk` feature.
  `SdkTokenCredential` adapts any [`azure_identity`] credential onto this
  crate's `TokenCredential`, so SDK and hand-rolled credentials are
  interchangeable wherever a credential is accepted. This is additive: the
  hand-rolled chain remains the default and is unchanged when the feature is
  off. It exists to reach credential types this crate does not implement —
  `ClientCertificateCredential`, `ClientAssertionCredential`,
  `AzurePipelinesCredential`, `AzureDeveloperCliCredential` — and to let
  Microsoft own IMDS quirks, sovereign-cloud endpoints and token lifetimes
  under a semver guarantee.
  - `azure_core` depends on **reqwest 0.13**, a different major from the 0.12
    the workspace uses. Cargo does not unify features across
    semver-incompatible versions, so the workspace's `rustls-tls` does not
    apply to the SDK's client and `azure_core` carries its own
    `reqwest_rustls`. Without it reqwest 0.13 resolves with no TLS backend:
    everything compiles and every unit test passes, but every Entra token
    request fails at run time. Enabling it means `aws-lc-rs` (a C library
    needing `cmake`) is the TLS provider on that side, so builds with this
    feature need a C toolchain. Verified both ways against the live Entra
    endpoint — see the ignored `tls_probe` test in `agent-framework-azure`.

- **A ready-made OTLP export pipeline**, behind `agent-framework-core`'s new
  optional `otel-export` feature (re-exported by the umbrella crate).
  `observability::export::OtelExport` builds an OTLP exporter, tracer
  provider, meter provider and the `tracing`↔OpenTelemetry bridge in one call,
  wired to the GenAI conventions this crate already emits. Enabling it implies
  `otel-metrics`, since the pipeline installs the `MeterProvider` those
  histograms record through. This is the only feature that pulls an OTel
  *SDK* — the default build stays API-only, so consumers who don't ask for
  export never inherit the SDK or its version churn.
  - `OtelPipeline::install` claims the global subscriber for applications that
    have none; `OtelPipeline::tracing_layer` returns the bridge layer for
    applications composing their own. `install` is a no-op when traces are
    disabled, so a metrics-only pipeline leaves the global subscriber free for
    the application's own logging rather than taking it irreversibly.
  - `OtelPipeline::shutdown` returns its outcome instead of logging it, and
    attempts both providers before reporting. Logging would go nowhere in the
    setup this module recommends: `install` builds a subscriber of an
    `EnvFilter` and the OpenTelemetry layer alone, so there is no formatting
    layer for the event to reach, and the one layer present feeds the tracer
    provider being shut down. A failed flush means telemetry was dropped —
    which happens whenever the collector is unreachable — and the caller can
    now see it.
  - Transport is OTLP-over-HTTP/protobuf on reqwest's *blocking* client, not
    gRPC and not the async client. gRPC would add the tonic/hyper stack; the
    async client panics with "there is no reactor running" when the batch span
    processor flushes, because that processor and the periodic metric reader
    each export from a background thread with no tokio runtime on it.
  - `opentelemetry-otlp` brings its own reqwest 0.13 and so needs its own TLS
    feature (`reqwest-rustls`) for the same reason as `azure_core` above;
    without it the exporter reaches plain-HTTP collectors only and fails
    against any HTTPS endpoint.
  - `examples/observability/otel_export.rs` demonstrates both routes and is
    compiled by CI, so the wiring cannot drift — replacing the previous
    ` ```ignore ` snippet in the module docs, which was never compiled.

### Fixed

- **`observability`: corrected stale documentation on the third GenAI
  histogram.** The module docs and
  `metrics::record_function_invocation_duration` both stated that
  `agent_framework.function.invocation.duration` was defined but "not yet
  called anywhere in this crate", and listed switching to `tool_span_ex` and
  adding `record_tool_arguments`/`record_tool_result` as outstanding
  follow-ups. All of that had in fact landed —
  `FunctionInvokingChatClient` times each tool invocation and records the
  histogram, and uses all three span helpers. A reader following the old docs
  would have concluded tool-call timing was unavailable, or wired up a second
  recording of it. Documentation only; no behavior change.

## [0.3.0] — 2026-08-24

Upstream-alignment passes against `microsoft/agent-framework`, moving the
baseline from `4b1afd90` (2026-08-07) to `a63d462` (2026-08-24). Six
upstream changes are ported and 100+ intervening commits are triaged in
[`ALIGNMENT_PROGRESS.md`](./ALIGNMENT_PROGRESS.md), which records why each
one that does not apply does not.

This release **breaks API** (pre-1.0, so a minor bump): the observability
recording functions take an `&ObservabilityConfig` in place of a
`capture_content: bool`, `chat_span` takes the semantic-convention flag, and
`gen_ai.system` is no longer emitted alongside `gen_ai.provider.name`. See
**Changed** below.

### Changed

- **The GenAI semantic-convention version is now selectable, and the provider
  tag follows it** (upstream #7673, [BREAKING] there). `gen_ai.system` was
  renamed to `gen_ai.provider.name` above the OTel v1.36.0 baseline, and this
  port emitted *both* names on every chat span, so a consumer pinned to the
  baseline saw an attribute its version does not define. `ObservabilityConfig`
  now reads `OTEL_SEMCONV_STABILITY_OPT_IN` and exposes
  `use_latest_experimental_gen_ai_semconv()`: unset means the latest
  conventions (upstream's default too), and a list omitting
  `gen_ai_latest_experimental` selects the baseline. Exactly one provider
  attribute is emitted — on spans and on the metrics attributes — and the four
  above-baseline attributes (`gen_ai.usage.cache_creation.input_tokens`,
  `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.usage.reasoning.output_tokens`, `gen_ai.tool.definitions`) plus
  `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` are withheld at the
  baseline. Under the default nothing changes except that `gen_ai.system` is
  no longer emitted alongside `gen_ai.provider.name`.
  - API: `record_request`, `record_response`, `record_tool_arguments` and
    `record_tool_result` take `&ObservabilityConfig` in place of a
    `capture_content: bool`; `chat_span` takes the semconv flag;
    `ObservableChatClient` gained `with_observability_config`
    (`with_content_capture` still works and now sets the flag on the config).

### Added

- **`Error::MiddlewareFailure`, a fail-closed signal for function middleware**
  (upstream #7562). The function-invocation loop absorbs every error a tool or
  its middleware produces into a tool-error result, hands it to the model and
  keeps looping — the right default for a tool failure the model can route
  around, but it left an enforcement layer (a guardrail, a policy or
  authorization gate) no way to stop a run: refusing a call just produced an
  error string the model could try again. Middleware returning
  `Error::middleware_failure(..)` is now propagated instead of absorbed, and
  because the parallel batch is driven by `try_join_all`, propagating it also
  drops the sibling calls still in flight. Every other error keeps the
  absorb-and-continue contract unchanged.

### Fixed

- **Replaying a conversation duplicated stored history** (upstream #7242). A
  history provider is handed a run's input plus its response, so a caller that
  keeps its own transcript and replays all of it each turn handed back
  everything already stored — and each provider appended it unconditionally.
  History grew superlinearly, and since `before_run` prepends it to the
  request, the duplicated turns were resent to the model on every later run.
  `agent_framework_core::history::filter_new_messages` now aligns the stored
  run inside the incoming one (by `message_id`, or by role and contents when
  there is none) and stores only what follows it; `InMemoryHistoryProvider`,
  `FileHistoryProvider`, `RedisChatMessageStore` and `CosmosChatMessageStore`
  all use it, the last two reading their stored history first (and skipping
  that read entirely when there is nothing to store, or when a Redis store is
  configured to retain nothing). A run that cannot be aligned is appended
  exactly as before — unlike upstream, no set-based fallback drops a turn that
  merely repeats an earlier one. Alignment sees a run's **input** only:
  response messages were just generated and can never be a replay, so they are
  always stored, even when one happens to reproduce the stored tail.
- **A replayed transcript was also sent to the model twice.** Storing only the
  new suffix fixed history growth, but the request is assembled the other way
  round — injected context first, then the caller's input — so a history
  provider that unconditionally injected what it held sent `q1, a1, q1, a1,
  q2` for a caller replaying `q1, a1, q2`. All four history providers now
  inject nothing when the run's input already carries the stored run
  (`inject_stored_history`).
  `StoredHistory::{Complete, Window}` selects where a stored run is looked for:
  a complete history is matched at the start only — it begins at the
  conversation's first message, so a replay of it can only begin with it — while
  a window is searched for, preferring a match at the start (an at-cap list that
  has never actually been trimmed is still complete) and otherwise taking the
  last occurrence. The Redis store asks for `Window` only when its list is at
  its cap. Alignment also requires *evidence* that stored history could be a
  replay at all — a matching message id, or a non-user turn, since a replay is
  a transcript and carries the assistant's replies. Stored history that is
  nothing but id-less user messages is indistinguishable from new input that
  repeats it, and is left alone. An empty `message_id` counts as no id at all,
  matching the `!id.is_empty()` guard the crate already applies elsewhere.
- **Tool spans could report a different semconv version than chat spans.** The
  function-invocation loop rebuilt an `ObservabilityConfig` from the
  environment for every tool call, so a client configured explicitly for one
  convention version emitted tool spans under whatever the environment said.
  `FunctionInvokingChatClient` now carries the config, settable with
  `with_observability_config` and resolved from the environment once at
  construction. `AgentBuilder::observability_config` reaches that wrapper,
  which the builder constructs itself.
- **A Redis retention limit of zero retained everything** (upstream #7470).
  `RedisChatMessageStore::with_max_messages(0)` is a request to retain
  nothing — unlimited is expressed by not calling it at all — but trimming to
  `-(max)` emits `LTRIM key 0 -1` for a limit of zero, which is Redis's "keep
  the whole list", so the trim ran on every save and did nothing.
  `add_messages` now returns before serializing, so no payload reaches Redis
  (or an AOF or a replica) even briefly. Stored history is deliberately left
  alone rather than deleted: the key carries no per-provider discriminator, so
  two stores sharing a prefix and session id address the same list, and
  deleting it would drop a co-located store's history. Use `clear` to remove
  history. A negative limit — the other half of upstream's fix — is
  unrepresentable here, since `max_messages` is a `usize`.
- **Gemini 3 thought signatures were dropped across an approval round trip**
  (upstream #7546). Gemini 3 rejects a `functionCall` part that lacks the
  `thoughtSignature` it was issued with. Signatures were paired to calls by
  adjacency alone, and any intervening content cleared the held signature —
  so a `FunctionApprovalResponse` sitting between a reasoning carrier and its
  call dropped it, and a call replayed in a later message could never be
  signed at all. Both turns then failed with a 400. Content that emits no
  wire Part no longer clears the signature, and a `call_id -> signature` map
  accumulated as the conversation is emitted signs a later replay. Precedence
  is unchanged: the call's own `protected_data`, then an adjacent carrier,
  then the map. The map is written by the emit walk rather than a pre-pass, so
  the pairing rules have a single implementation — a pre-pass that restated
  them laxly would re-sign the very calls adjacency had refused.
- **Stateless Responses requests never asked for the encrypted reasoning
  item.** A reasoning item is only replayable on the next turn of a `store:
  false` tool loop if it carries `encrypted_content`, and the service only
  returns that when the request's `include` asks for it. This port set
  `include` nowhere, so the replay path in `messages_to_input` — which
  re-emits the item verbatim and drops one lacking `id`/`encrypted_content` —
  had nothing valid to re-send. Both Responses clients (OpenAI and Azure
  OpenAI) now add `reasoning.encrypted_content` when a request carries no
  service-side-storage indicator, matching upstream. A caller's own `include`
  entries are preserved and never duplicated.
- **Foundry opts out of the above**, matching upstream #7536: it does not want
  encrypted reasoning unless asked for by name. New
  `AzureOpenAIResponsesClient::without_implicit_encrypted_reasoning`, which
  `FoundryChatClient` sets on its transport. An explicitly requested
  `reasoning.encrypted_content` is still honored.
- **The tool loop reported only its last iteration's token usage.** Each model
  call in the function-invocation loop reports its own usage, and every exit
  path returned the final call's `ChatResponse` untouched — so a run that
  called tools five times reported roughly a fifth of the tokens it spent, and
  the `gen_ai.usage.*` OTel metrics (which read `usage_details`) under-reported
  with it. Usage is now summed across every iteration and applied to whichever
  response the loop returns, including the approval pause, the
  declaration-only hand-back, and the tools-disabled failsafe. A run where no
  iteration reported usage still reports none rather than a synthesized zero.
  (upstream #7539)

## [0.2.0] — 2026-08-08

Upstream-alignment pass against `microsoft/agent-framework` `4b1afd90`
(2026-08-07), re-baselining from `beb65b21` (2026-07-13). See
[`ALIGNMENT_PROGRESS.md`](./ALIGNMENT_PROGRESS.md) for the full triage of the
112 intervening upstream commits, including the items deliberately left open.

### Fixed

- **Structured output was parsed from the wrong text.** The JSON value was
  built from every message joined with separators and included reasoning
  content, so a tool result could be mistaken for the answer, chain-of-thought
  was folded into the payload, and a JSON document split across streaming text
  chunks had separators injected into it. Now taken from the last non-empty
  assistant message's `text` contents, concatenated with no separator.
  (upstream #6990)
- **Anthropic streaming double-counted tokens.** Anthropic streams cumulative
  usage snapshots and the port summed them as if they were increments, so a
  response reporting 25 input tokens aggregated to 50. (upstream #7162)
- **OpenAI Chat Completions rejected some author names.** A `author_name`
  containing `/`, `|`, `\`, `<` or `>` was sent verbatim and failed the whole
  request with a 400; one containing a space was silently dropped. Names are
  now sanitized to `[a-zA-Z0-9_]` and truncated to 64 characters, matching the
  Python and .NET clients. (upstream #7126)
- **Gemini 3 function-call replays lost `thought_signature`.** Gemini 3
  requires the signature echoed when a call is replayed. Both placements are
  handled: on the function-call part itself (the usual one) and on a preceding
  thought part (backfill only). Reasoning content is no longer sent back as a
  part, matching upstream. (upstream #7095)
- **Approval replacement is now a single ordered walk.** The outstanding-call
  set is derived as the walk decides each content, rather than maintained as
  separate bookkeeping around an order-blind pre-scan — which netted a call
  against a result arriving *after* a replayed request and expanded the
  request into a duplicate declaration.
- **Approval round-trips could duplicate a function call.** The restored call
  was deduped against only the message being scanned, but a hosting layer
  replays the stored call and its approval request as two separate messages, so
  a second copy was restored and left unanswered — which the Responses API
  rejects with "No tool output found for function call ...". (upstream #7271)
- **Compaction could emit conversations providers reject.** A function call and
  its result are now retained or dropped together — previously `TokenBudget`
  could keep a tool result whose call fell outside the budget (a tool message
  answering nothing), and `SelectiveToolResult` deleted stale results while
  their assistant `tool_calls` entries remained (an unanswered call). Either
  half alone is a 400 on the next request. (upstream #7406)
- **Compaction could reduce a conversation to system messages only.**
  `Truncation`/`SlidingWindow` with a budget at or below the system prefix left
  no turn for the model to answer; the most recent non-system message is now
  retained even when that exceeds the limit. (upstream #7219)

### Changed

- **BREAKING: mem0 retrieval scope no longer inherits the storage scope.**
  `Mem0Provider::before_run` searched with the storage `user_id`/`agent_id`, so
  a provider configured with a shared `agent_id` retrieved memories written by
  every user of that agent and injected them into the current user's
  conversation. Retrieval now uses only `with_search_user_id` /
  `with_search_agent_id` / `with_search_application_id`; with none set, nothing
  is retrieved and a warning is logged once. Code that retrieved via
  `with_user_id` alone must add `with_search_user_id`. (upstream #7531)
- **`SelectiveToolResult` now replaces a stale tool result's payload with
  `OMITTED_TOOL_RESULT` instead of deleting the result content.** Deleting it
  orphaned the matching function call; replacing the payload sheds the same
  bulk while keeping the exchange valid. Mirrors the intent of upstream's
  `ToolResultCompactionStrategy`, which replaces stale tool groups with a
  compact stand-in rather than removing them.

### Added

- **Bedrock Converse image blocks.** The Bedrock converter previously dropped
  all `Data`/`Uri` content; inline images in Converse's accepted formats
  (`png`/`jpeg`/`gif`/`webp`) are now emitted as `{"image": ...}` blocks,
  closing a parity gap with upstream's Bedrock client.
- **`Content::renders_on_every_provider`.** The wire-visibility contract that
  compaction's minimum-retention logic depends on now lives on `Content`,
  pinned by contract tests in each provider crate — a converter change that
  invalidates a row fails a test next to the converter instead of surfacing as
  a compaction bug. `Uri` content is excluded (Bedrock has no remote-URL image
  source), and `Data` images are bounded by Bedrock's format set.
- `DataContent::from_uri` / `DataContent::media_type_from_uri`: validating
  construction from a `data:` URI, rejecting a missing scheme, missing `,`, or
  non-base64 declaration instead of silently mis-slicing it. (upstream #6916)
- `TextReasoningContent::protected_data`, mirroring upstream's
  `Content.protected_data`, for provider-opaque reasoning replay tokens.
- OpenAI cache-**write** token counts on both the Chat Completions and
  Responses surfaces, populating `UsageDetails::cache_creation_input_token_count`
  and so the `gen_ai.usage.cache_creation.input_tokens` OTel attribute.
  (upstream #7369)

## [0.1.1] — 2026-07-13

First published release on crates.io — identical in content to 0.1.0.

- The v0.1.0 release-pipeline run failed at the publish step (the crates.io
  token secret was misnamed), after the `v0.1.0` tag had already been
  pushed, so 0.1.0 was never published and its version number is burned.
- Release workflow: publish to crates.io **before** tagging and creating
  the GitHub Release, so a failed publish no longer burns the version —
  the run can simply be retried after fixing the cause.

## [0.1.0] — 2026-07-13

First release: a Rust implementation of the Microsoft Agent Framework at
structural parity with upstream (`microsoft/agent-framework`) as of upstream
commit `beb65b21` (2026-07-13). 22 crates.

### Core (`agent-framework-core`)

- Data model: `Message`, the `Content` union (text, reasoning, data/URI,
  function call/result, hosted tool calls/results, usage, approvals),
  `ChatResponse`/`AgentResponse` (+ streaming updates and aggregation),
  `ChatOptions` with Python `&`-merge semantics, structured output
  (`ResponseFormat` + `parse_json`), typed `UsageDetails` with
  cache/reasoning counts, **embeddings** (`Embedding`, `GeneratedEmbeddings`,
  `EmbeddingGenerationOptions`, `EmbeddingClient`).
- `ChatClient` trait + `FunctionInvokingChatClient`: parallel tool
  execution, human-in-the-loop approval gating, declaration-only (frontend)
  tools, **progressive tool exposure** (live `FunctionInvocationContext::tools`
  with `add_tools`/`remove_tools`), retry layer (`RetryingChatClient`)
  honoring `Retry-After`.
- Agents: `Agent`/`AgentBuilder`, three-level middleware (agent/chat/
  function), per-run options, dynamic `ToolSource`s (MCP catalog changes),
  `as_tool` (with `propagate_session` child-session semantics,
  `stream_callback`, `approval_mode`).
- Sessions: `AgentSession` + shared-by-reference `SessionState`,
  `HistoryProvider` (in-memory/file), context providers
  (`before_run`/`after_run`), history compaction (four strategies +
  `CompactionProvider`), skills (progressive disclosure), settings
  (`SecretString`, `load_setting`).
- Workflow engine: Pregel-style supersteps, checkpointing (+ resume,
  graph-signature validation), human-in-the-loop pause/resume, output
  designation, async edge conditions, shared state, sub-workflows,
  Mermaid/DOT visualization.
- Orchestrations: Sequential, Concurrent, GroupChat, Handoff (enforced mesh
  topology), Magentic (plan review + stall intervention HITL),
  `WorkflowAgent`, post-agent approval (`AgentApprovalExecutor`).
- Observability: OTel GenAI-semconv spans and (feature-gated) metrics.

### Providers

- OpenAI (Chat Completions + Responses + **embeddings**), Azure OpenAI
  (api-key + Entra ID credential chain, Responses, **embeddings**),
  Anthropic (incl. Bedrock/Vertex/Foundry cloud transports), AWS Bedrock
  (Converse, dependency-free SigV4), Foundry (Responses + Prompt Agents),
  Foundry Local, Gemini, Mistral (chat + **embeddings**), Ollama (chat +
  **embeddings**), GitHub Copilot (token exchange), Copilot Studio.

### Integrations & hosting

- MCP (stdio/HTTP/websocket, sampling, prompts, roots), A2A client + serving,
  declarative agents/workflows (Rust-native `WorkflowSpec`), hosting crate
  (DevUI-style API, AG-UI protocol, OpenAI-compat endpoint, security
  middleware), Redis / Mem0 / Azure AI Search context providers, Cosmos DB
  message store + checkpoint storage, Purview compliance middleware.

### Known divergences from upstream

Documented in `ALIGNMENT_PROGRESS.md` / `PARITY.md`: streaming is expressed
as Rust method pairs (`run`/`run_stream`); the declarative *workflow* DSL is
Rust-native rather than Power Platform; `durabletask`, the `@experimental`
harness/security/evaluation modules, and the Claude Agent SDK wrapper are out
of scope; DevUI's bundled-frontend routes are partial.
