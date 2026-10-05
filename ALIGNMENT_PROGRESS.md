# Alignment progress against current upstream

Tracks the re-baselining of `agent-framework-rs` onto current upstream, as
catalogued in [`UPSTREAM_DRIFT.md`](./UPSTREAM_DRIFT.md). Section numbers under
the `68136ee` heading refer to that document. Every item recorded as landed was
independently verified (full workspace build + `cargo test` + clippy
`--all-targets` + rustfmt, all green) before commit.

**Current upstream baseline: `dc8e226` (2026-09-28); feature parity push checked against `b9d24c8` (2026-10-05).** Sections are newest
first; each records the upstream revision it was checked against.

## Stateful Responses hosting (checked against `b9d24c8`, 2026-10-05)

First item of a parity push against current upstream `main`. An independent
re-audit found the `/v1/responses` host stateless — no `previous_response_id`,
no conversations, and no way to resume a workflow paused on a
human-in-the-loop request over HTTP — where upstream ships three packages for
exactly this: core `SessionStore` / `FileSessionStore`,
`agent-framework-hosting` (`AgentState` / `WorkflowState`), and
`agent-framework-hosting-responses` (continuation helpers), plus DevUI's
per-conversation checkpoint resume.

### What landed

| Upstream | Rust |
|---|---|
| `SessionStore`, `FileSessionStore` (`_sessions.py`) | `core::session_store` — trait + in-memory + file. Independent copies on read; atomic writes; injective id→filename mapping with a digest stem past filename limits; version check; corrupt-file quarantine. Snapshots carry in-process history via the new `ContextProvider::history_snapshot`, because Rust keeps history in the provider where upstream keeps it in `session.state`. MessagePack not ported. |
| `AgentState`, `WorkflowState` (`hosting/_state.py`) | `hosting::state` — instance / builder / cached-or-not factory targets; per-id serialized session creation. |
| `responses_session_id`, `create_response_id`, `create_conversation_id`, the options half of `responses_to_run` | `hosting::responses` — same validation (mutually exclusive, non-empty, `{id}` form, deprecated `conversation_id`), same option remaps. |
| `workflow.run(checkpoint_storage=...)` runtime override | `Workflow::run_with_checkpointing`; `run_from_checkpoint` now keeps writing to the storage it resumed from. Without that, a run that paused twice could not be resumed the second time — pinned by `runtime_checkpoint_storage_survives_two_pauses`. |
| DevUI `_execute_workflow` (`workflow_hil_response`, per-conversation `InMemoryCheckpointStorage`, `extra_body.checkpoint_id`) | `AgentHost`'s `/v1/responses` workflow path. |

Two deliberate divergences. An unknown `previous_response_id` is a `400
previous_response_not_found` (OpenAI's behaviour) rather than upstream's
example `get_or_create`, which turns a typo into a silently fresh
conversation; an unknown `conversation` still starts one. And every agent
response is stored under its own id *and* advances the conversation head,
so any response is a branch point — upstream's example stores one or the
other.

Verified: `cargo test --workspace --all-features` (**2177 passing, 0
failing**), clippy `--all-targets --all-features -D warnings`, `cargo fmt
--check`, `cargo doc` with `-D warnings`. The workflow-resume tests were
probed by removing the runtime storage override: three fail.

### Landed later in the same parity push

Each merged with the full workspace green (`cargo test --workspace
--all-features`, clippy `-D warnings`, fmt):

- **MCP hosting** — new `agent-framework-hosting-mcp` crate (`AgentMcpTool`,
  `WorkflowMcpTool`, `mcp_to_run`/`mcp_from_run`, minimal stdio +
  streamable-HTTP `McpServer`).
- **MCP client parity with upstream `MCPTool`** — prefixes, prompts as
  tools, raw-name matching with ambiguity errors, result-content modes,
  argument filtering, `_meta` echo, progressive disclosure, logging,
  `SamplingGuard` (deny by default), reconnect-and-retry. (Correction to the
  re-audit: upstream's client has no `resources/*` either.)
- **Evaluation** (core `evaluation`, Foundry evaluators).
- **Agent hooks** (core `agent_hooks`, AGENT-HOOKS-0.1).
- **Harness agent** — new `agent-framework-harness` crate; also fixed
  approved tool calls re-executing when history replays an approval.
- **Upstream-format declarative workflows** — PowerFx subset interpreter and
  every upstream action kind.
- **Redis vector store, Redis/Valkey history providers, Qdrant** (live-tested
  against Redis, Redis Stack and Qdrant).

### Still open

MCP tasks, per-run MCP `header_provider`, OTel context in `_meta`;
`security.py` (information-flow labels); functional workflows; OpenAI
computer-use / shell tools; the `claude` and `typesafe` packages; sandboxes
(Hyperlight, Monty, shell tools, LocalCodeAct); hosting for Telegram,
ChatKit and Foundry; DevUI's conversations API and React UI; A2A server push
notifications / resubscribe; `lab`; and the ~130 upstream commits after
`dc8e226`. Connector ports for Postgres, SQL Server, MongoDB, DocumentDB,
DuckDB, Cosmos memory and Content Understanding were in flight when this
push stopped — see the CHANGELOG for which landed.

## Tool-call serialization on both hosting surfaces (same upstream baseline, `dc8e226`)

The round below recorded "neither hosting surface serializes tool calls" as
a capability gap and declined to close it inside a review cycle. The
repository owner overruled that: close it. This is that work.

### What was broken

Core deliberately leaves a `FunctionCallContent` intact for the *caller* to
execute — that is the whole point of a client-side tool. Both hosting
surfaces then serialized only the text:

- `/v1/chat/completions` built its message from `resp.text()`, so a turn
  whose only content was a call became `content: ""` with no `tool_calls`.
- `/v1/responses` built exactly one assistant text item, buffered and
  streaming alike.

The client was told a call had been requested and given nothing to act on:
no `call_id`, no name, no arguments. The previous round's mitigation was to
degrade the `tool_calls` finish reason to `stop`, which kept the surface
honest but left the capability missing.

### What landed

| Surface | Buffered | Streaming |
|---|---|---|
| `/v1/chat/completions` | `message.tool_calls` with `id` / `type` / `function.{name,arguments}`; `content: null` for a call-only turn, as OpenAI sends | `delta.tool_calls` fragments keyed by a stable per-`call_id` `index`; the first sighting identifies the call, later ones carry arguments only |
| `/v1/responses` | `function_call` output items as *siblings* of the message, each with its own item `id` distinct from `call_id` | `response.output_item.added` once per call, then `response.function_call_arguments.delta` per fragment |

Four details were worth getting right rather than approximating:

**Arguments are a JSON string, not an object.** Both wire formats type the
field as a string and clients `JSON.parse` it. `FunctionArguments` is either
a raw string (what a provider streams, possibly a fragment) or a parsed
object, so `util::arguments_string` re-serializes the object case and maps
absent arguments to `"{}"` rather than `null`.

**A streamed call is one call, not one per fragment.** A provider may stream
a single call's arguments across several updates. OpenAI's contract is that
fragments sharing an `index` (chat completions) or an `item_id` (responses)
concatenate, so both paths remember the calls they have already announced
and re-announce nothing. Re-sending `id`/`name` per fragment would have a
strict client either ignore them or read them as further calls.

**The tool finish reason now passes through** whenever the response actually
carries a call — the promise can be kept. Without one it still degrades to
`stop` with the provider's reason in `x_finish_reason`, because a turn that
reports `tool_calls` while declaring none is still an instruction with
nothing behind it.

**The two Responses paths differ on the message item, deliberately.** The
buffered path omits it for a call-only turn, as OpenAI does — an empty
assistant message reads as a blank answer rather than as work to do. The
streaming path keeps it, because the preamble already announced it at
`output_index` 0 before any content was known, and every call event numbered
itself from there; dropping it would shift each call one index away from the
event that announced it. The terminal payload also reuses the item ids the
client already saw, or it cannot correlate the two.

### Codex review on the same PR — five findings, all real

The outbound half above shipped first and drew a review. All five findings
held up, and three of them were in what had just landed.

| Finding | Verdict | Fix |
|---|---|---|
| **P1** A resolved call is re-advertised | **Real, and mine.** Core keeps a `FunctionCallContent` *and* its `FunctionResultContent` in the response — after a local tool ran, and when a provider executed a hosted tool itself. `function_calls_of` collected every historical call, so the client was asked to re-run work already done. `FunctionInvokingChatClient` filters exactly this way, for exactly this reason, twenty lines from code I had read: the precedent was already in the repo. | Both surfaces now serialize only calls with no matching result. |
| **P1** `function_call_output` is not parsed | **Real.** The Responses protocol returns a tool result as a *top-level* item with `call_id` and `output` — no `role`, no `content` — so `item_to_message` turned it into an empty user turn and dropped the result. | `function_call_output` → `FunctionResultContent`, `function_call` → `FunctionCallContent`. |
| **P1** Chat Completions tool-result messages are not parsed | **Real.** The client's follow-up replays the assistant turn with `tool_calls` and adds a `role: "tool"` message with `tool_call_id`; `IncomingMessage` read neither, and every provider converter builds its wire tool messages from `FunctionResultContent` rather than tool-role text. | `IncomingMessage` gains both fields and rebuilds the call/result contents. |
| **P2** An argumentless announcement emits `{}` | **Real, and mine.** A streamed call is often announced before any arguments exist — this repository's own Responses parser builds exactly that, `arguments: None`. Mapping it to `"{}"` put a literal `{}` at the head of the fragment sequence, so a client concatenating deltas parsed `{}{"city":"Oslo"}`. | `arguments_delta` returns `None` there; `"{}"` stays the *buffered* default, where it is the whole value rather than the first of several. |
| **P2** The client ignores `response.incomplete` | **Real, and from the pass above**, which added the event to the host without adding the arm to `parse_responses_event`. A client pointed at this endpoint lost the finish reason, usage and response id for every truncated or filtered stream — exactly what the distinct event name exists to report. | Handled on the same arm as `response.completed`. |

**The first finding turned out to be worse than reported, and the obvious
fix did not close it.** Filtering per update works on the buffered path,
where the whole response is in hand. On the streaming path the result
arrives *after* the call: with local tools,
`FunctionInvokingChatClient::get_streaming_response` runs the entire loop
and then replays each message as its own update, so the call update always
precedes the tool-result update answering it. A per-update filter has
already put the call on the wire, and no later event can recall it — so the
common case, an agent with local tools, still told the client to re-execute
them. A test written to the reported shape passed; one written to the
replay shape failed.

So the streaming paths now **hold calls until the stream ends** and emit
only those still unanswered. The cost is that a call's arguments arrive in
one delta rather than forming incrementally — a presentation detail, since
a client cannot execute a call before it has the whole argument object.
Duplicating a tool's side effects is not a presentation detail, which is
what settles the trade.

### Verification

Twenty-seven tests across both rounds: buffered serialization and
`content: null`, index stability and fragment reassembly, the function call
output items, the announce-then-fill event sequence with monotonic sequence
numbers and id correlation into the terminal payload, the resolved-call
filter on both surfaces and both paths, both inbound round trips (JSON and
non-JSON results), the `response.incomplete` terminal arm, and six
no-regression tests pinning that an ordinary turn is unchanged.

Sixteen mutation probes, each reintroducing one specific bug and each
failing exactly the test written for it and only that test. Full workspace:
**2147 passing, 0 failing**, clippy `-D warnings`, rustfmt and `cargo doc`
clean.

`ResponseObject::output` changes type from `Vec<OutputMessage>` to
`Vec<OutputItem>`, which is a breaking change for callers that read it;
`OutputItem::as_message()` recovers the old view.

The lesson, and it is the same one this session keeps relearning: the fix
was written to the shape the report described rather than to the invariant
it named. "A resolved call must never be advertised" does not stop being
true because the result arrives late, and the test that would have caught
it was the one modelling how the framework actually streams — not the one
modelling the example in the finding.

## Review round on PR #28 (same upstream baseline, `dc8e226`)

Copilot's review on [#28](https://github.com/CodeHalwell/agent-framework-rs/pull/28)
raised four findings, two High and two Medium. **All four were real**, and
two of them were in the two passes above rather than in older code — worth
recording plainly, because both are the same failure mode: a fix that moved
a boundary and was written up as if it had removed one.

| Finding | Verdict | Fix |
|---|---|---|
| `Value::as_f64` still rounds JSON integers above 2^53, so Hamming equality is not preserved | **Real, and mine.** The pass above widened stored coordinates from `f32` to `f64` and described the class as closed. It was not: `f64` cannot represent `2^53 + 1`, so `2^53` and `2^53 + 1` still collapse and the non-matching record still scores as exact. The write-up was more confident than the change. | Stored coordinates now keep the exact integer JSON carried alongside the `f64` (`Coord`), and Hamming compares in integer space (`i128`, covering all of `i64` and `u64`) while every arithmetic metric keeps using `f64`. Pinned at the 2^53 boundary; disabling the integer path reproduces the old `0.0`-instead-of-`0.5`. The query stays `f32`, which is right for embeddings — the 64-bit case is ids, and those live in the *stored* side. |
| A shared provider's session-derived scope leaks across runs | **Real, and mine.** The new memory provider documented "one provider serves one logical session" and left nothing enforcing it, while the natural wiring — `Arc` on an agent — shares it. Run B's `before_run` would overwrite the latched session while run A was generating, so A's `after_run` wrote A's messages under **B's scope**; sequential sessions also shared one profile and one cursor. Filing one user's conversation against another's is the worst shape a memory bug takes, and the doc comment was the only thing standing in the way. | State is now a map keyed by scope, so profiles and cursors cannot cross. For `after_run`, which is handed no session at all, the provider writes under a derived scope only while it has seen exactly one; a second session makes any write a guess, so it warns and declines. A configured `with_scope` is unambiguous and keeps working — and the docs now say to set it on a shared provider instead of merely describing the hazard. |
| `response_to_updates` drops `finish_reason` | **Real, and mine.** The pass above added the field and threaded it through `client.rs`, but `agent.rs`'s helper destructures `AgentResponse` with `..` and rebuilds updates with `..Default::default()`. So the buffered `run_stream` and middleware streaming paths still lost it, and hosting still said `stop` there — the half of the fix that was tested was the half that worked. | The reason rides the final update, and a reason with no messages now emits one, exactly as `client.rs` does. |
| A blank `FOUNDRY_PROJECT_ENDPOINT` blocks the `FOUNDRY_ENDPOINT` alias | **Real, and mine.** `.ok().or_else(...).filter(...)` checks for blank *after* choosing, so a present-but-empty primary satisfies the fallback and is then discarded, and the alias is never read. The `models_endpoint` line two above it filters correctly, which is what makes this a slip rather than a design choice. | Each candidate is filtered before the fallback. |

**A second Codex round on the fixed head raised three more, all real**, and
all the same shape as the first batch — a change that handled the cases it
was written for and defaulted the rest wrongly:

| Finding | Verdict | Fix |
|---|---|---|
| The Responses surface reports an unfamiliar finish reason as `completed` | **Real.** `incomplete_reason` mapped `content_filter` and `length` and sent everything else to `None`. But `FinishReason` is an open string and providers use it — this port's own Anthropic converter goes out of its way to preserve `model_context_window_exceeded`, with a test pinning that it is no longer flattened. So the abnormal endings most worth reporting were exactly the ones turned back into false successes. Worse, the sibling function `finish_reason_of` on the chat-completions surface already got this right, and its doc comment states the principle — an unfamiliar value a client ignores is recoverable, a wrong familiar one is not. The two surfaces disagreed. | Completion is now an allowlist: absent, `stop` and `tool_calls`. Everything else is incomplete, with OpenAI's spelling for the two it names and the provider's own string otherwise. |
| The project embeddings route forwards Inference-only request fields | **Real.** Selecting `Route::ProjectOpenAI` changed the URL and nothing else, so `input_type` — an Azure AI Inference field with no OpenAI equivalent — still went to the derived `/openai/v1/embeddings`, which rejects it. The `extra-parameters` pass-through header had the same problem. | Supported properties and that header are now chosen per route. |
| A failed contextual search discards the cached profile | **Real.** The error branch returned early, which also skipped injecting the static memories the *first* search had already fetched — so a transient failure of the second request threw away known-good context, in a provider whose whole error posture is to degrade rather than fail. | Treated as an empty contextual result; the profile is still injected. |

**A third round on that head raised three more.** All real, and one of them
is a divergence this port now makes deliberately:

| Finding | Verdict | Fix |
|---|---|---|
| An `incomplete` response carries a `completed` output item | **Real.** `OutputMessage::assistant_text` hardcodes the item status, and both `responses_from_run` and the DevUI streaming path build their item with it. A client reading item status was told the opposite of what the response said, by the more specific of the two. | The item's status follows the response's, on both paths. |
| A run with no searchable input loses its memories | **Real, and it was *this port's own* inconsistency.** Upstream returns early here too, so the behaviour was faithful — but the fix one round earlier made the *search-failed* path inject the cached profile, and this path still did not. Two degraded paths, the same profile in hand, opposite answers. Resolved toward injecting in both: a **deliberate divergence from upstream**, recorded here, on the grounds that the profile is already fetched and dropping it silently loses managed memory on a valid run. | Skip the request, keep the injection. |
| The state lock is held across HTTP calls | **Real, and known when written.** It was accepted for simplicity — but the round before had just made this provider safe to *share*, which makes serializing unrelated scopes behind one lock exactly the wrong trade. | Snapshot under the lock, release, request, re-acquire to record. Two concurrent first-runs for one scope may now both fetch the profile: an idempotent read, against stalling every other session. |

A fourth round raised **one** more, and it is the mirror image of the
round before: splitting the embedding properties per route pruned
`input_type` correctly, and pruned `user` with it — but `user` is a field
OpenAI *defines*, which the derived project endpoint accepts and the Models
endpoint does not, and which `agent-framework-openai` and
`agent-framework-azure` already forward on exactly that surface. Having
noticed that the two routes differ, the fix had only looked in one
direction. The lists now differ both ways, and the test asserts both.

A fifth round raised three more, **two of them created by the fixes two
rounds earlier** — which is the honest shape of this exchange and worth
recording rather than smoothing over:

| Finding | Verdict | Fix |
|---|---|---|
| A failed concurrent fetch erases the profile | **Real, and caused by my own fix.** Releasing the lock for the request made two first runs for one scope possible; I called that "an idempotent read" in the code comment. It is not idempotent *on failure*: the failing run committed an empty profile over the successful one's and set `initialized`, so nothing re-fetched it and the scope lost its memories for good. | A failure no longer writes `static_memories` at all, and the run reads the entry back rather than trusting its own result. **The first version of this test was vacuous** — with equal delays the harmful ordering is a coin toss, and the test passed against the reintroduced bug. It now forces the failing fetch to commit last. |
| The scope cache grows without bound | **Real.** Partitioning state by scope — the round-two fix — turned a fixed-size struct into a map with one entry per session and no eviction. | An LRU bounded at 512, overridable. Eviction costs only a re-fetch and a restarted cursor, both already the first-run path. |
| `incomplete_details.reason` can hold a schema-invalid value | **Real, and in tension with round three.** Round three said: do not swallow abnormal reasons. I satisfied that by passing the provider's string through `incomplete_details.reason` — a field the Responses schema defines as a two-value enum, which a strict generated client can reject outright. Both constraints hold together: the *status* carries the abnormality, the *enum* stays legal. | `is_incomplete` (the status question) is now separate from `incomplete_reason` (the two schema names). An unfamiliar reason yields `incomplete` with no `incomplete_details`, and travels in a new `x_finish_reason` extension. The round-three test asserting the old location was retired, its successor asserting strictly more. |

A sixth round raised two more, **one of them again created by the round
before it**: separating `is_incomplete` from `incomplete_reason` meant an
`incomplete` response can now carry no `incomplete_details`, and the DevUI
terminal event was still keyed on that optional field — so it announced
`response.completed` around exactly the payloads the separation existed to
flag. It now follows `status`. The second extends the same closed-enum
reasoning to the *chat-completions* surface, which had the identical
problem and which my own doc comment there had argued the other way about:
an unfamiliar reason is reported as `length` with the raw value in
`x_finish_reason`, the wrong-familiar-value objection having lost its force
once the real value stopped being discarded.

A seventh round raised two, and they are one mistake: OpenAI's finish
vocabulary has **five** values and core's `FinishReason` named four.
`function_call` — the deprecated spelling of `tool_calls`, and like it a
*successful* turn — was therefore classed as abnormal by the new
`is_incomplete` allowlist and rewritten to `length` by the new
chat-completions mapping, both of which told clients a working tool call
had been cut off. The `finish_reason_of` doc comment had listed all five
correctly while its match handled four, which is the whole bug in one line.
`FinishReason::FUNCTION_CALL` now exists and both surfaces use it.

A third finding in that round — **not delivered as a notification, and
found only by reading the threads directly** — was the one that mattered:
two turns on one scope both snapshot `previous_update_id`, both post, and
one falls out of the service's incremental chain.

That is the third round in which this provider's concurrency model was
wrong, and the three are one mistake seen from three sides. One lock per
*provider* is wrong in both positions: held across the requests it
serializes unrelated scopes (round five), released across them it lets two
runs for one scope race (rounds six and seven). The unit the operations
actually need is the **scope**. The provider now takes one lock per scope —
the provider-wide lock is held only long enough to hand back the scope's —
so snapshot → request → commit is atomic within a scope while scopes stay
parallel. That is a root-cause fix rather than a fourth point patch, and it
retires the duplicate-profile-fetch race this document previously excused
as "an idempotent read".

An eighth round raised two. One — serialize contextual searches per scope —
was **already closed by the per-scope locking above**, which landed after
the commit it reviewed; the same guard covers the search path and the
update path, which is the point of fixing the shape rather than the site.

The other is the sharpest finding of the whole exchange, because it is
about a *promise* rather than a value. Reporting `tool_calls` tells a
client to go and execute the call in `message.tool_calls` — and this host
serializes text and nothing else, so the client is handed an instruction
with no id, name or arguments. Before this pass the surface reported
`stop` for everything, so the bug arrived *with* the finish-reason work:
making the reason honest made it promise something the surface could not
keep. Both tool reasons now degrade to `stop` with the real one in
`x_finish_reason`. Note this partly reverses the round-seven fix, on a
better argument: that round established `function_call` should not be
rewritten to `length`, and this one establishes that the destination is
`stop`, not the reason itself.

A ninth round raised two. One is a defect in the round-seven fix, and the
code comment I wrote there states the opposite of the truth: it called an
eviction-while-held harmless, reasoning that the run "completes against
state nobody reads again". It does not — the *next* run for that scope
builds a second mutex, races the first, and both resume one cursor, which
is the fork per-scope locking exists to prevent. Only a slot with no other
holder is evictable now (`Arc::strong_count == 1`), and a cache whose every
candidate is busy runs briefly over capacity, because the bound exists to
stop unbounded growth rather than to be honoured at the cost of
correctness.

The other extends the tool-call gap below to the Responses surface, where
it is **pre-existing and untouched by this pass**: `responses_from_run` has
always built a text-only output item, and a tool turn reported `completed`
before this work as it does after. Recorded, not fixed here.

A tenth round raised one, and it is the other half of the ninth. Pinning
in-flight slots lets the cache exceed its bound — fine, provided the
overrun is temporary, and it was not. Eviction ran only when inserting a
new scope and dropped at most one slot, so a touch of an existing scope
trimmed nothing and a new-scope touch removed one and added one. A burst of
concurrent sessions therefore stayed resident for good, which is the
unbounded growth the cache was added to prevent, reached by a different
road. Trimming now runs on every touch and loops until the bound is met.

Worth recording that the first probe for this fix **did not compile** — the
mutation put a `break` inside an `if` — and produced no output rather than
a failure, which reads identically to a passing probe if the output is
skimmed. The second, valid mutation failed at four entries against two.
That is the second vacuous probe in this exchange; a probe that proves
nothing is worse than none, because it is recorded as evidence.

### Capability gap recorded, then closed

**Neither hosting surface serialized tool calls.** Core keeps
`FunctionCallContent` intact for the caller to execute, and
`completion_object` read only `resp.text()`; the streaming path was the
same. So a declaration-only call reached the client with no id, name or
arguments, and the `tool_calls` finish reason had to be degraded to `stop`
to avoid instructing a client to run something it could not see.

Recorded here as a feature deliberately not bolted on at the end of a
review cycle — and then built, at the owner's instruction, in the round
above. Both surfaces now carry the calls; see
*[Tool-call serialization on both hosting surfaces](#tool-call-serialization-on-both-hosting-surfaces)*.

Thirty-five tests added across the twenty-three findings. Six review rounds;
**nine of the sixteen findings were in code written earlier in the same
session**, four of them introduced by the fix for a previous round. The
pattern is worth stating rather than burying: each fix was locally correct
and globally incomplete, because it reasoned about the case in front of it
and not about the invariant it had just changed. Three of the fixes were probed by
mutation — disabling the integer comparison, the ambiguity detection, or the
latch each reproduces the reported symptom exactly. Full workspace:
**2100 passing, 0 failing**, clippy `-D warnings`, rustfmt and `cargo doc`
clean.

The lesson worth carrying: three of these four are boundary-moved-not-removed
or fix-threaded-partway, and in each case the *prose* claimed more than the
diff delivered. A reviewer caught what the tests did not because the tests
were written to the same belief as the code.

## Verification pass + the Foundry memory provider (same upstream baseline, `dc8e226`)

A follow-up over the pass above: re-check its six changes against primary
sources, then spend the rest on the standing-gap list. The re-check held —
and it turned up that the list's own top entry had been closed off for the
wrong reason.

### What the re-check confirmed

The riskiest call in the pass above was encoding an empty hosted-MCP
allowlist as `tool_configuration: {"enabled": false}` rather than upstream's
literal `allowed_tools: []`. Anthropic's MCP connector documentation settles
both halves of it. Its migration table off the deprecated beta reads
`tool_configuration.enabled: false` as a first-class "no tools" and — the
line that matters — describes **"No `tool_configuration` (all tools
enabled)"**, which is precisely the inversion the fix removes. So the
diagnosis was right and the encoding is the documented one, not merely the
defensible one. The other five were re-read against their diffs; the Hamming
normalization's one unguarded edge (`a.len()` as a divisor) is unreachable
because a zero-dimension vector field is rejected at definition validation
(`vectors.rs:383`).

One genuinely new thing came out of that reading, and it is **not** a bug
here: `mcp-client-2025-04-04`, the beta this crate sends, is deprecated in
favour of `mcp-client-2025-11-20`, which moves tool configuration out of
`mcp_servers[].tool_configuration` and into an `mcp_toolset` entry in
`tools[]`. Upstream Python sends the same deprecated flag
(`_chat_client.py`'s `BETA_FLAGS`), so this port is faithful and moving alone
would be a divergence, not a fix. Recorded below with the mapping so that
whoever follows upstream across has it ready.

### The standing-gap list was wrong about why it was stuck

Two Azure items — the Foundry memory provider and Content Understanding —
were recorded as externally blocked on the same premise: the SDKs carrying
their wire contracts "are not available in this environment", so "the REST
paths, the api-version and the long-running-operation shape would all be
guesses."

That premise does not hold: the package index is reachable from this
environment. `azure-ai-projects` and `azure-ai-contentunderstanding` both
download and unpack, and their generated request builders state the contract
outright. The gap was never external; it was an untested assumption, and it
had been carried forward across passes as settled.

### Ported this pass

| Gap | Change | Rust site |
|---|---|---|
| **Foundry managed memory had no provider.** Flagged two passes running as "the most tractable Azure item left", then shelved as unportable. With `azure-ai-projects` readable the contract is explicit: `POST {project_endpoint}/memory_stores/{name}:search_memories` and `:update_memories`, `api-version=v1`, bearer-scoped to `https://ai.azure.com/.default` — already this crate's `FOUNDRY_SCOPE`. Bodies are `{scope, items?, previous_search_id?}` and `{scope, items?, previous_update_id?, update_delay?}` with nulls dropped (the SDK's own `{k: v for k, v in body.items() if v is not None}`); `items` are `{"type":"message","role":…,"content":…}`; the search answers `{search_id, memories[].memory_item.content}`. Three things are faithful rather than invented. The **incremental cursors** only advance on a search that returned something, so an empty answer does not reset where the next one resumes. The **static (user-profile) fetch** runs once per provider and its latch is set even when it *fails*, so an unreachable store costs one request rather than one per run. And **every failure is swallowed and logged**: retrieval and storage are enhancements, and `after_run` also runs on the agent's own failure path, where raising would replace the real error with this one. The one structural divergence is forced by the trait: Python's hooks get a per-run `state` dict and a `SessionContext`, while Rust's `after_run` gets neither — so the cursors, the latch and the session id live in one `Mutex`-guarded struct on the provider, with the session id latched during `before_run` so `after_run` can still resolve a scope. That is the same shape `Mem0Provider` already uses for the same signature gap. Upstream's `begin_update_memories` returns an LRO poller it never polls, reading only `update_id`; the single POST here is exactly that much of the operation. | `foundry/memory.rs` (new), `foundry/lib.rs` |

Ten tests, five of them loopback against a fake data plane on a real socket,
pinning the routes, the bearer header, the null-dropping, both cursors, the
once-only static fetch, and that a 500 leaves the run intact. Both halves of
the risky behaviour were probed by mutation: removing the once-only latch
breaks the cursor test with the right symptom, and propagating the search
error instead of logging it breaks the failure test. Full workspace:
**2093 passing, 0 failing**, clippy `-D warnings`, rustfmt and `cargo doc`
clean.

### Re-triaged, with the reason corrected

| Gap | Status | Assessment |
|---|---|---|
| Foundry — memory provider | ✅ **Closed above** | Was never externally blocked. |
| Azure AI Content Understanding | ❌, unblocked | `azure-ai-contentunderstanding` reads the same way: `/analyzers`, `/analyzers/{analyzer}`, api-versions `2025-11-01` (GA) and `2026-06-01-preview`. What makes it the larger job is size, not mystery — ~1400 lines upstream across a context provider, a file-search backend pair, and content detection. Now the top item with nothing in its way. |
| Foundry — evaluations | ❌ | Same SDK, so also readable now; still large, and upstream is still moving it. The "blocked" half of the old reason is gone; the "moving target" half stands. |
| Anthropic hosted MCP — `mcp-client-2025-11-20` | ❌, deliberate | Not a defect: upstream sends the same deprecated `mcp-client-2025-04-04`, and this port matches it. The mapping when upstream moves: no `tool_configuration` → an `mcp_toolset` with neither `default_config` nor `configs`; `enabled: false` → `default_config.enabled: false`; `allowed_tools: [...]` → `default_config.enabled: false` plus those tools enabled in `configs`. Note the new beta also *requires* the `mcp_toolset` entry — `mcp_servers` alone is rejected — so this is a two-part change, not a rename. |
| Switch/case predicate cannot report failure (#8490) | ❌ | Re-examined and still declined, now with the blast radius measured: `Condition` returns `bool`, and the `Selection` it feeds returns `Vec<String>`, so neither has an error channel. Making a predicate fallible means widening both public type aliases and every builder that takes one. Worth doing deliberately, not as a side effect of a verification pass. |

## Post-`6606bef` drift + Azure-ecosystem review (checked against `dc8e226`, 2026-09-28)

Upstream moved **89 non-merge commits** in this window (2026-09-21 → 09-28).
**Six land on this port.** Four of the six are one shape: *a value the code
already had, and never read.* An explicit empty allowlist read as "no
allowlist"; a finish reason was carried to the edge of the agent types and
dropped there; stored vector coordinates were narrowed before being compared
for equality; a blank instruction was checked for emptiness but not for
whitespace. None of these fail loudly, and three of them fail in the
permissive direction — which is why the window reads as small and is not.

The Azure surface accounts for three: a Foundry **project** can now be used
for embeddings at all, an Azure OpenAI content-filter block is no longer
reported to hosting clients as a normal completion, and the standing Azure
table gains one new upstream connector.

### Ported this pass (6 upstream changes, all with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #8576 | **An empty hosted-MCP allowlist enabled every tool on the server.** `allowed_tools: Some(vec![])` means "expose none of this server's tools". The Anthropic converter treated it like `None` and omitted `tool_configuration` entirely, which leaves the API default in place — and the API default is *all tools enabled*. So the one input a caller uses to lock a server down was the input that unlocked it, and a test pinned that behaviour ("…`_is_omitted`"). It is now encoded rather than dropped. Deliberately as `enabled: false` rather than upstream's literal `allowed_tools: []`: only `enabled` has documented semantics for "no tools" on Anthropic's `tool_configuration`, whereas an empty array is undocumented there and could be read back as unset — which would reintroduce the bug in the one place it must not recur. Upstream emits the empty array because the OpenAI/Foundry shape it fixes has no `enabled` field. **The other two providers were already right** and are pinned as negative controls: `openai/responses.rs` emits `[]`, and Foundry and Azure both delegate to it rather than converting for themselves. | `anthropic/convert.rs` (`tools_to_anthropic`) |
| #8478 | **A turn the model was cut off in was reported as a turn that finished.** Upstream's bug was a hosting loop that never read `AgentResponseUpdate.finish_reason`; here the field **did not exist on the agent types at all**. `AgentResponse::from_chat_response` mapped every other field across and dropped this one, and `into_chat_update` dropped it again on the streaming path — so the aggregation that computes it threw it away immediately afterwards. The consequence is the same in both languages and worst on Azure: `content_filter` (an Azure OpenAI content-filter block) and `length` produce a response whose *text* reads like a finished answer, so an application's only signal was matching the provider's canned refusal string. Three surfaces were wrong as a result. `openai_compat.rs` hardcoded `"finish_reason": "stop"` on both the buffered and streaming paths, asserting normal completion to every OpenAI-compatible client. `responses_from_run` hardcoded `status: "completed"` and had no `incomplete_details` field. And DevUI's terminal event was always `response.completed`. All four now carry the real reason, with `length` mapped to `max_output_tokens` on the Responses surface (the one point the two OpenAI vocabularies disagree) and an unfamiliar provider reason passed through rather than flattened to `stop` — a value a client ignores is recoverable, a wrong familiar one is not. The inbound half already worked: `finish_reason_from_response` has parsed Azure's `incomplete_details.reason` since the Responses client was built, so this closes a round trip rather than opening one. | `core/types/response.rs` (`AgentResponse`, `AgentResponseUpdate`), `hosting/responses.rs` (`IncompleteDetails`, `incomplete_reason`), `hosting/openai_compat.rs` (`finish_reason_of`), `hosting/devui/mod.rs` |
| #8637 | **Hamming distance could not tell two large integers apart.** Stored vectors were read as `f64` and then narrowed to `f32` before scoring. `f32` carries 24 bits of mantissa, so any two distinct integers above 2^24 — ids, timestamps, hashes, which is exactly what a Hamming collection holds — compared **equal**, and the record that did not match came back scored as an exact match. Hamming is where this surfaces because it is the one metric asking whether coordinates are the *same* rather than how far apart they are. Coordinates are now read at `f64` and the query widened once, which also stops every other metric losing precision it was never meant to lose. Two details ride along. The score is now **normalized** by the vector width, as upstream has it: ranking was unaffected (dividing by a constant is monotonic) but a raw count made `score_threshold` mean a different thing on every collection. And the sibling test that pinned the *old* narrowing — it fed `1e39`, finite in `f64` and infinite in `f32`, to prove a non-finite stored vector is dropped — was rewritten rather than deleted: the property it protects (a score that cannot be serialized is never ranked) is real, but its route is now arithmetic overflow, so it squares `1e200` instead, and a second test pins that `1e39` is ranked normally now. Worth noting the port was *already* ahead of this class elsewhere: `vectors/filters.rs` documents refusing `f64` for integer comparison for the same reason. | `core/vectors.rs` (`score_vectors`, `InMemoryCollection::search`) |
| #8454 | **A Foundry project could not be used for embeddings.** `FoundryEmbeddingClient` spoke only the Foundry **Models** inference endpoint, a separately-provisioned surface. The endpoint a Foundry user actually has is the *project* endpoint — the one `FoundryChatClient` already takes — so a project holding an embedding deployment still could not be embedded against from here. Upstream's route is now derived: `https://<res>.services.ai.azure.com/api/projects/<proj>` becomes `https://<res>.openai.azure.com/openai/v1`, scoped to the **resource** rather than the project, which is why the project path is dropped. Three details are right rather than plausible. The project route is **path-versioned**, so it must not carry the Models endpoint's `?api-version=` — the same split `FoundryChatClient` handles with `without_api_version`, and the reason `url()` now branches instead of formatting one string. The token audience stays `cognitiveservices.azure.com` (the derived host is Azure OpenAI data plane), *not* `FOUNDRY_SCOPE`. And the project route is **Entra-only**, which is why there is no api-key counterpart. `from_env` prefers the Models endpoint when both are set, so an environment that already worked is untouched, and accepts `FOUNDRY_ENDPOINT` beside `FOUNDRY_PROJECT_ENDPOINT` so one Foundry environment configures both clients — an alias upstream does not need and this crate does, because its chat client reads the other name first. | `foundry/embeddings.rs` (`openai_model_base_url`, `Route`, `with_project_endpoint`, `from_env`) |
| #8524 | **A whitespace-only instruction became a contentless system turn.** `prepare_messages` skipped `""` but not `" "` or `"\n"`, so a blank instruction — the shape an unset options default or an instruction merge produces — was prepended to the conversation as a system message some providers bill for and others reject. A real instruction still prepends **verbatim**, whitespace included; the trim decides only whether to prepend. | `core/types/message.rs` (`prepare_messages`) |
| #8581 | **A shared Magentic manager interleaves two runs' plans.** Upstream created a manager per build. Rust's `build(self)` consumes the builder, so the case upstream was fixing cannot arise here — but two others can, and they surface identically: a caller can hand one `Arc` to two builders, and `Workflow::run` takes `&self`, so one Magentic workflow can have two runs in flight. `StandardMagenticManager` caches the decomposed task ledger, and that cache is read back by exactly the two surfaces where being wrong is expensive — the plan-review request and the stall-intervention request — so a human reviewer can be shown, and asked to approve, the *other* run's facts and plan. The runs' own execution state is unaffected; it lives per-run on the orchestrator. **Documented rather than fixed**, which is the honest scope: the fix is to move the cache per-run, and `standard_manager` (which takes the manager by value, so each builder owns one) is already the shape that avoids it. Recorded below as a standing gap. | `core/workflow/orchestration/magentic.rs` (docs on `StandardMagenticManager`, `manager`, `standard_manager`) |

Verified across all six: full workspace build, `cargo test --workspace
--all-features` (**2082 passing, 0 failing**), `cargo clippy --all-targets
--all-features` under `-D warnings` (CI's own flag) clean, `cargo fmt --check`
clean, `cargo doc --workspace` clean.

Each behavioural test was probed against the code it pins rather than merely
written beside it. Restoring the `f32` narrowing fails the Hamming test with
exactly the old wrong answer (`0.0` where the record differs, against the
expected `0.5`). Restoring `if !instr.is_empty()` fails the blank-instruction
test on the first whitespace case. Zeroing either `finish_reason` hand-off —
the buffered one or the streamed one — fails its own test while the
negative control still passes. The Anthropic allowlist test is the previous
behaviour's own test rewritten in place, so it fails against the code it used
to pass against, which is the strongest form this probe takes. Negative
controls throughout: an *absent* allowlist still emits no `tool_configuration`,
a `stop` / `tool_calls` / unreported run still serializes with no
`incomplete_details` key, a run with no finish reason still grows no
`finish_reason` key, the Foundry Models route still carries its
`?api-version=`, and a real instruction still prepends with its whitespace
intact.

One test-only fault was found and fixed while adding the Foundry tests: the
crate's `temp_env_absent` helper mutates process-global environment variables
and had only ever had one caller, so a second one raced it under the default
multi-threaded harness — passing alone and failing in the suite. It now takes
a lock, and recovers a poisoned one so a single panicking test does not fail
every other.

### The Azure-ecosystem review

Rechecked against the standing table in the previous section; **one row is
new** and two changed.

| Azure surface | Upstream | Here | Change this pass |
|---|---|---|---|
| Foundry — embeddings | ✅ | ✅ (was 🚧) | **Closed for text.** Project endpoints now work (above). What remains of the old 🚧 is the *image* half only: upstream splits a batch across an image-embeddings endpoint, which the core `EmbeddingClient` signature (`Vec<String>`) cannot express without widening a shared trait. That is a core change, not a Foundry one. |
| Azure OpenAI — content filter, reported to hosting clients | ✅ | ✅ (was ❌, untracked) | **Closed.** A filtered turn reaches an OpenAI-compatible client as `content_filter` / `incomplete` rather than as a normal stop (above). No table row tracked this before, because the inbound parse was right and only the outbound half was wrong. |
| Azure SQL / SQL Server native vector store | ✅ (`sql-server`, #8686, new this window) | ❌ | **New gap.** SQL Server 2025 / Azure SQL's native `VECTOR` type as a `VectorStore` pair. Genuinely portable in shape — the `VectorStore`/`VectorCollection` traits and the portable filter compiler both already exist, and the Cosmos and AI Search collections are the template — but blocked in this workspace the same way MongoDB and DocumentDB are: it speaks **TDS**, and there is no TDS driver here. Unlike those two the block is soft (`tiberius` is a pure-Rust async TDS client that supports Azure SQL and Entra auth), so this is a dependency decision rather than an impossibility. Recorded, not attempted: a connector aimed at a wire protocol this workspace cannot exercise would be untestable. |
| Azure DocumentDB | ✅ | ❌ | Unchanged. #8654 (null fields in membership filters) lands on a connector that does not exist here, still blocked on the MongoDB wire protocol. |
| Foundry hosting | ✅ | ❌ | Unchanged. #8565 (no `response.completed` after a cancelled run) is a `foundry_hosting` change; standing gap. |

Everything else in the standing table is unchanged. The largest unblocked
Azure item remains **Azure AI Content Understanding** — a REST surface, so
portable, and still the biggest one not waiting on something else.

### Standing gaps this pass surfaced (not closed)

- **A switch/case predicate cannot report failure** (#8490). Upstream stopped
  swallowing predicate exceptions, which were routing a broken predicate's
  message to the default branch — silent misrouting rather than a visible
  failure. The port has no analogue to *remove*, because `Condition` returns
  `bool`: there is no error channel, so a fallible predicate (one that
  deserializes the payload, say) has nowhere to put the error and must return
  `false` — which produces exactly the behaviour upstream just fixed. Closing
  it means widening `Condition` to `Result<bool>`, a breaking change across
  every builder method that takes one.
- **Magentic's task-ledger cache is per-manager, not per-run** (#8581, above).
  Documented this pass; the fix is to move the cache onto the run.
- **The in-memory vector search clones every record before paging** (#8544).
  Upstream narrowed its `deepcopy` to the requested page. Here the whole
  collection is cloned out under the lock before scoring, so the equivalent
  saving is larger — but it is a performance property of a store meant for
  tests and development, not a correctness one.

## Post-`061dc28` drift + Azure-ecosystem review (checked against `6606bef`, 2026-09-21)

Upstream moved **129 non-merge commits** in this window (2026-09-14 → 09-21).
**Twelve land on this port**, and the Azure surface accounts for five of
them: the Cosmos DB vector store it set out to build, the content-filter
detail beside it, and three separate faults on the Purview middleware — which
is the one surface in this workspace whose whole job is to not let something
through, and which turned out to be evaluating a fraction of each message,
unable to parse a verdict carrying an action added this same window, and
reporting an unevaluated message as a cleared one. Those three, plus a fourth
elsewhere (a failing tool's diagnostics persisted verbatim into every history
store), make this a window of *silent* failures rather than missing features:
nothing here was reported by a user or would appear in a log.

### The mirror sync held for a second week

The fork-sync fix from two passes ago (restore the fork's own
`.github/workflows/**` after merging, so the pushed ref contains no change
under it) has now carried upstream through two full windows with no manual
intervention. Recorded once more because three consecutive passes before it
opened with the opposite; assume it works from here and stop reporting it
unless it breaks.

### Ported this pass (12 upstream changes, all with regression tests)

Fourteen rows: the Purview review found three separate faults under one
upstream PR, and they are worth reading apart.

| Upstream | Change | Rust site |
|---|---|---|
| #8186 | **An Azure Cosmos DB for NoSQL vector store.** The largest Azure item on the books after last pass's AI Search work, and the one that makes `core::vectors` worth having on two providers rather than one. A container becomes a collection: the `vectorEmbeddingPolicy` and `indexingPolicy` at creation, point reads and upserts in the item's own partition (`/id`, so a read by key never fans out), and a `VectorDistance` query with every filter literal bound as a parameter. Four details are right rather than plausible. **The vector surface needs its own api-version** — `2018-12-31`, which the history and checkpoint stores speak, predates vector search entirely, so a container created under it would come back without the policy it asked for and the search would be a syntax error; `DEFAULT_VECTOR_API_VERSION` is split out and public, the same split the Azure OpenAI crate needed last pass for the same reason. **A distance function Cosmos cannot compute is refused**, not substituted: Cosmos computes cosine as a *similarity*, so accepting `cosine_distance` would return a score whose direction the caller's own `higher_is_closer` reads backwards — and because the three it does have each agree with the portable name's direction, a result needs no `score_kind` override at all. **`skip` rides in the limit** rather than being dropped from an already-truncated page, since Cosmos will not take `OFFSET` beside an `ORDER BY VectorDistance`. And an **ordered filter on an untyped field carries a SQL type guard**: Cosmos SQL orders *across* types, so `loose > 5` would otherwise match every string in that field. Upstream refuses an ordered filter on an untyped field outright; here the type hint is optional and usually unset, so the guard the refusal exists to provide is emitted instead. | `cosmos/vector_store.rs` (new), `cosmos/client.rs` (container read/create/delete/list), `cosmos/tests/loopback.rs`, `examples/memory/cosmos_vector_store.rs` |
| #8421 | **A vector collection, handed to an agent as tools.** Upstream's new `VectorCollectionContextProvider`, which is what turns the two Azure stores this port now has into something a model can use rather than something a caller drives. Four decisions carry the weight. Writes require approval by default and reads do not, matching upstream — a delete the model gets wrong is not recoverable from the conversation. The scope filter is applied *everywhere*: conjoined into search, and checked on the records a read, write or delete touches. An out-of-scope record reads as **absent** rather than as a refusal, because "you may not read that" confirms it exists; an out-of-scope write is **refused** rather than rewritten into scope, which would let the model launder a record through the provider. The model never authors an embedding — `embed_from_field` names the field the vector is derived from, and the vector field is not in the tool schema at all; without it there is no upsert tool rather than one that cannot work (upstream needs no equivalent, because its collection owns an embedding generator). And tool names take an optional prefix: upstream's plain `search`/`get` collide the moment an agent holds two of these. The `VectorStoreHistoryProvider` half of the same PR is not ported — it is a second schema and a second lifecycle, and the port already has four history providers. | `core/vectors/provider.rs` (new), `examples/memory/vector_collection_tools.rs` |
| #8370 | **Purview was evaluating a fraction of each message.** It submitted `Message::text()` and nothing else, so a function result — which is exactly where exfiltrated data shows up — an attachment, a reasoning block and a tool call all reached the model unevaluated. Every content item is now mapped, one `processContent` request per entry as upstream: an attachment's bytes go as Graph `binaryContent` (handing a classifier the base64 *string* instead reads as gibberish and passes every policy, which is worse than sending nothing because it looks like coverage — and the data-URI parse has to accept RFC 2397's `;parameter=value` segments or that is exactly what happens), and everything else is serialized whole, which keeps `additional_properties` under evaluation too. Only `usage` is skipped. The port had a hole **on top of** upstream's: `text()` returns `""` whenever a refusal is present (the #7992 marking), so a partly-declined turn submitted nothing at all. | `purview/processor.rs` (`map_content`, `build_requests`), `purview/models.rs` (`PurviewBinaryContent`, `PurviewContent`) |
| #8370 | **A verdict carrying `restrictAccess` could not be parsed.** Upstream added that action this same window, and `DlpAction` named only `blockAccess`/`other`, so a response carrying it failed to deserialize and the whole verdict was lost. `restrictAccess` is now named — and, as upstream has it, is deliberately *not* a block on its own: it carries a `restrictionAction` that selects the enforcement mode, and only `block` withholds content. The enums stay **strict**, which is the correction to a first attempt at this that made both `#[serde(other)]` catch-alls; Codex caught it on the PR. A catch-all is a policy bypass here: `should_block` would answer `false` for a *blocking* action Graph adds later, and nothing could recover it, because a verdict is not an error and `ignore_exceptions` never reaches one. Failing the parse raises an error instead, which the default `ignore_exceptions = false` turns into a stopped request — fail-closed, which is the posture a DLP middleware exists to hold. | `purview/models.rs` |
| #8370 | **No user id reported "allowed".** Purview evaluates policy for a specific user, so no resolvable user means nothing was evaluated — which is not the same as nothing being found, and reporting it as a clear verdict is the one failure mode this middleware exists to prevent. It now fails, as upstream's does; `ignore_exceptions` still trades that for availability, explicitly and for every error rather than only this one. The crate docs gained the security section upstream wrote alongside it: both identity sources are host-supplied and unverified, so a host that populates either from anything that crossed a trust boundary lets that party select a weaker policy. | `purview/processor.rs`, `purview/lib.rs`, `purview/middleware.rs` |
| #8235 | **A failing tool's diagnostics were persisted forever.** `FunctionResultContent::exception` is host-internal text whose contents this framework does not control — it may come from a tool, middleware, a provider or a caller, and a connection string, a SQL error quoting the row it failed on, or a stack trace naming an internal host are all ordinary things to find in one. Serialization is exactly what persists a conversation (the Redis, Cosmos and file history stores; workflow checkpoints), so every one of those was durably storing it. It now serializes as a fixed marker. Failure *state* survives, which is what a resumed workflow reads; deserialization is untouched, so a conversation written before this keeps what it stored; and the model is unaffected either way, because every provider converter reads the field directly rather than through `serde`. One deliberate divergence: the AG-UI router still sends the diagnostic to its frontend, since this port's tool loop puts a failure's only text in `exception` and leaves `result` empty — redacting there would leave a frontend with nothing rather than with something safer. | `core/types/content.rs` (`FUNCTION_INVOCATION_ERROR_MARKER`) |
| #8393 | **An Azure content-filter refusal said only that it was one.** The classification was right and the detail was gone: a caller held a message string and could not tell a self-harm block from a jailbreak detection, or a prompt block from a completion block — which is the whole of what an application does with this error. Azure's nested `innererror` (the policy code, the per-category verdicts, the `param`) now rides on `Error::ServiceContentFilter`, reaching Azure OpenAI chat, Responses and embeddings, all three of which share the classifier. Every value is an **open string**: upstream parses the code and the severity into enums, which raise on a value Azure has not shipped yet — the bug its own fix closed for the code and still has for the severity. A plain OpenAI refusal, which carries nothing beyond the marker, grows no empty detail. | `core/error.rs` (`ContentFilterDetail`, `ContentFilterCategory`), `openai/lib.rs` (`parse_content_filter_detail`) |
| #8405 (+#8382) | **Chat Completions dropped reasoning in both directions.** Upstream's fix is about the payload surviving a turn that was split into two assistant messages; this port emits one message per turn, so it never had that — it had the whole mechanism missing instead. A reasoning-capable provider on this surface (DeepSeek in thinking mode, OpenRouter, vLLM) returns a `reasoning_details` array and **requires it back** on the next request of the same turn, which in a tool loop is the tool-result follow-up, so a conversation that used reasoning could not be continued. It now rides in `TextReasoningContent::protected_data` — the same field the Anthropic thinking signature and the Gemini thought signature use, because it is the same kind of thing — on both the buffered and streaming paths, and is replayed on the message that carries the turn. Sent only when the provider gave us one: OpenAI proper does not know the field and answers an unrecognized request argument with an error. A payload that is not JSON (an Anthropic signature shares the field) is dropped rather than sent, since that would turn a working request into a 400. | `openai/convert.rs` (`reasoning_details_of`, `parse_reasoning_details`), `openai/lib.rs` (stream) |
| #8337 | **A streamed tool-call fragment could land on the wrong call.** The port already matched by `call_id` across every in-flight call, so upstream's main fix had nothing to land on — but both of its follow-ups did, in a slightly different shape. An *untagged* continuation delta (the ones some providers only stamp on the first chunk) merged into the **first** function call rather than the one still being streamed, so with two calls in flight every such delta went to whichever opened first. And an untagged chunk whose `call_id` matched a call that already carried an occurrence id was merged anyway, though sharing a `call_id` is not proof of being the same occurrence and a provider is free to reuse one. A third bug was the port's own: a failed merge discarded the fragment silently instead of appending it. | `core/types/response.rs` (`merge_target_for_call`) |
| #8347 | **Error latency was missing from the metric that measures latency.** `gen_ai.client.operation.duration` recorded only successful calls, which leaves no error rate at all — a histogram with no failed observations cannot be divided by one — and skews the distribution the wrong way, since a call that times out after thirty seconds is exactly the observation a p99 needs. It now records on all three failure paths with `error.type`: the buffered call, the stream that fails to open, and the stream that fails midway, which ends the stream before its completion arm can run and so was missing entirely. No token histogram on a failure: the call reported no usage, and a zero is not "not reported". The test that pinned the old behavior is rewritten rather than removed. | `core/observability.rs` (`metrics::record_chat_error`) |
| #8453 | **`allow_concurrent_invocation`.** One model response's calls run in model order when it is `false`. The point is the side effects rather than the result order, which holds either way: a tool holding a non-reentrant handle or writing to a shared resource behaves differently when two of its invocations overlap, and "create the record, then update it" in one batch means them in that order. A failure then stops the batch — the calls after it never run — where the concurrent path cancels siblings already in flight. | `core/tools.rs`, `core/client.rs` |
| #8551 | **A handoff dropped the user's attachments.** An image or a file is the request as much as the words beside it, and the receiving agent was getting only the words. Multimodal content is kept on user messages and still dropped elsewhere, because providers treat those parts as input-only and reject them replayed on an assistant turn. The same rewrite closes the other half, which the port had and upstream did not: an approval *response* rides on a user message, which passed through untouched — handing the next agent an approval for a call it never made. | `core/workflow/orchestration/handoff.rs` (`clean_conversation`) |
| #7757 | **Concurrent checkpoint saves collided on one temp file.** The temp path was derived from the checkpoint id alone, so two saves of the same id wrote into the same file — the survivor could be a blend of both — and the loser's rename then failed with `NotFound`, because the winner had already moved that file away. Each save now writes its own file; the renames are atomic, so a reader sees one complete checkpoint or the other, which is what concurrent saves of one id mean. The regression test reproduces the exact upstream error against the old code. | `core/workflow/checkpoint.rs` |
| #8391 | **A thread id selects history; it does not protect it.** Upstream's docs clarification, which lands here because the property is the same: two stores over one account, database and container with the same thread id read, write and clear the same history — the thing that makes a conversation resumable — and therefore distinct ids prevent accidental overlap without restricting a client whose Cosmos credentials already authorize the container. | `cosmos/chat_message_store.rs` (docs) |

Verified across all twelve: full workspace build, `cargo test --workspace
--all-features` (**2025 passing**, 111 more than the 1,914 this pass started
from, including 37 hermetic loopback tests against a fake Cosmos DB account),
`cargo clippy --all-targets --all-features` under `-D warnings` (CI's own
flag) clean, `cargo fmt --check` clean, `cargo doc` clean.

Each behavioral test was probed against the code it pins rather than merely
written beside it. Restoring the shared checkpoint temp path fails the
concurrent-save test with the same `NotFound` upstream reported. Opening
either Purview action enum to a catch-all fails the fail-closed test.
Dropping the vector-field name from the generated search options fails the
multi-vector test with the exact "no vector field to search" error. Reverting the
merge-target helper fails the untagged-delta test *and* the reused-`call_id`
test, and does so by producing exactly the old wrong answer. The two
metrics tests and the four Purview fail-open tests were the previous
behavior's own tests, rewritten in place — each of them fails against the
code it used to pass against, which is the strongest form this probe takes.
Negative controls throughout: a plain OpenAI content-filter body still grows
no detail, a message with no reasoning payload still grows no
`reasoning_details` field, a declared-type field still needs no SQL type
guard, an omitted non-key field still reaches Cosmos without being
materialized as null, and a successful tool result still serializes without
an `exception` key.

#### Twenty-one corrections from review

Across ten review rounds, Codex and Copilot raised twenty-seven distinct
findings; **twenty-one were right** and are folded into the rows above. Two were
*interactions between changes in this same pass* — a correct change meeting
another correct change — which is the class a per-change review cannot see,
and the reason this section exists rather than a line saying review was
clean.

One shape accounted for four of the twelve on its own: **a payload that does
not live in the field the reader reads.** A tool failure's text sits in
`exception`, a citation's in `snippet`, a reasoning item's in
`protected_data` — and the Purview mapper read `text`. Each was found
separately, one round apart, which is the argument for fixing a class rather
than an instance: the last of them (a reasoning item carrying *both* text and
payload) was the ordinary post-streaming shape, because `coalesce_text` folds
a later fragment's payload onto the accumulated text.

* **The Purview catch-all was a bypass, not a fix** (row above, rewritten).
  The first attempt optimized for the `ignore_exceptions = true` case and
  made the default worse.
* **A multi-vector collection could not be searched at all.** The provider
  took the first declared vector field for upserts and named none for
  search, so on a collection with two vectors every search call failed with
  "no vector field to search". `vector_field` names it and `build` refuses
  the ambiguity where the caller can still see it.
* **Cosmos refused an upsert omitting any declared field.** Both sibling
  stores require only the key, and the comment in the Azure AI Search
  collection argues explicitly that omission means "this field is absent" —
  so the Cosmos rule broke the portability promise the `VectorStore` pair
  exists for, and contradicted the upsert tool in the same PR, whose schema
  requires only the key and the embedding source.
* **The exception redaction blinded the DLP check** — the first of the two
  self-inflicted interactions. Purview evaluates a tool result by serializing
  it, and serializing is exactly what now replaces `exception` with a marker,
  so a failing tool's diagnostic reached the policy as the word
  `FunctionInvocationError`. Both changes are right on their own and wrong
  together: the text is the likeliest place for a connection string to
  appear, which is why it is redacted *and* why it must be evaluated. Purview
  now builds an evaluation-only view carrying the real text; persistence is
  untouched.
* **Reasoning carried in `protected_data` went unevaluated** — the second.
  The `reasoning_details` support added in this same pass stores its payload
  with an empty `text`, and Purview's mapper skipped a reasoning item on
  `text.is_empty()`. So the pass introduced a content type and a hole for it
  in the same breath, against the module's own "everything but `usage`"
  contract.
* **Two streamed calls sharing a `call_id` merged when both were
  identified.** The guard excluded an *untagged* delta from an identified
  call but said nothing about two different occurrence ids, so the second
  call's arguments were appended onto the first's — reproduced as
  `{"x":1}{"y":2}` before fixing.
* **The generated key schema always said `string`.** A collection keyed by an
  integer stores `42`, and `InMemoryVectorStore` keys on
  `Value::to_string()`, so a model told `string` sends `"42"`, which does not
  collide — it never matches, and every read and delete silently finds
  nothing. The item type now comes from the key field's declared type.
* **The Cosmos example deleted a pre-existing container.** Its cleanup was
  unconditional on a fixed name, so running it against an account that
  already had `af-vector-demo` destroyed it and its records — under a comment
  reading "leave the account as we found it". It now uses a fresh name per
  run, so cleanup can only remove what that run created.
* **A citation's snippet went unevaluated.** The Purview mapper sent a text
  item's `text` and dropped its annotations, and a citation quotes its source
  in `snippet` — so if the quoted span was the sensitive part, everything
  around it was checked and it was not. An annotated item now serializes
  whole; plain text still goes as itself, which is cheaper and is what
  upstream does.
* **Cosmos refused a portable null equality.** `Filter::eq(field, null)` is
  constructible and, as `filters.rs` documents in as many words, "works as a
  null test" in the in-memory evaluator — while the Cosmos scalar guard
  rejected it outright. The same filter therefore errored on one backend and
  matched on another. It now translates to the `IS_NULL` form the `is_null`
  operator already emits, presence semantics included; non-scalar literals
  are still refused.
* **Auxiliary fields went unevaluated whenever text was present.** The fix
  for the two findings above keyed on the text being *empty*, which is the
  wrong question: a citation's snippet and a reasoning item's
  `protected_data` are data whether or not there is text beside them, and
  after stream aggregation there usually is. Both branches now ask whether
  anything beside the text holds data. A bare text item still goes as
  itself.
* **A read/delete-only provider was refused without a vector field.**
  `build` resolved the vector field unconditionally, but only search and
  upsert touch a vector — `get` and `delete` work by key. So a collection
  with no vector field, or several and none named, could not get a
  read/delete provider even though the underlying `VectorCollection` serves
  one fine. Resolved only when a tool that needs it is enabled.
* **Mixed `between` bounds compiled to a false predicate.** On an untyped
  field, `between(f, 1, "z")` emitted `IS_NUMBER(f) AND IS_STRING(f)` — which
  no record satisfies — so an invalid comparison came back as an empty page
  while the in-memory evaluator reported the bounds as incomparable.
  Differing bound types are refused now, and matching ones emit one guard
  rather than one per bound.
* **A scope filter the local tools could not honor was accepted.** `get`,
  `delete` and `upsert` check the scope by running `matches` over a record in
  this process, where search hands the filter to the store. So a provider
  operator (`azure_ai_search.match`), which `matches` errors on, would fail
  those three on every call while search worked; and a vector-field
  predicate, which they can never answer — they fetch without vectors, and
  `upsert` checks before deriving one — would quietly report every record as
  out of scope. Both refused at `build` now, and still allowed for a
  search-only provider, which never evaluates the filter locally.
* **An upsert could overwrite another group's record by naming its key.**
  The scope check ran on the *payload*, which says nothing about the record
  it lands on — an upsert replaces the whole document. So a scoped agent
  could destroy a record `get` and `delete` refuse to even show it, by
  submitting an in-scope payload at that key. Laundering by key collision
  rather than by payload, and the one hole in the scope story that was a
  genuine hole rather than a documented limit. The existing record is read
  and checked first now, as delete already did.
* **Delete reported absent keys as deleted.** The unscoped path skipped the
  read, so `deleted` was simply the input length — contradicting the
  response's own documented meaning two lines below it, where an
  already-absent key counts as *not* deleted. Both paths read first now, and
  a key named twice counts once.
* **A streamed reasoning payload was replayed as its last fragment only.**
  `reasoning_details` arrives over several deltas, each parsed into its own
  content — and `coalesce_text` leaves them separate *because* each carries
  `protected_data`. The request builder took the last, so a provider
  requiring the whole payload back got the tail of the reasoning and nothing
  before it. Fragments now concatenate in stream order. The comment asserting
  "a provider emits the payload once per turn" was true of the buffered path
  and false of the streamed one, which is where a turn gets split.
* **A Cosmos id containing `%` stored but could not be read back.** Cosmos
  forbids `/`, `\\`, `?` and `#` in an id and allows everything else, so
  `a%2Fb` is legal and upserts fine — the id rides in the request body. A
  point read then built the URI from the raw link, and the server decoded
  `%2F` into a slash: a different resource, 404. Worse, `delete_document`
  treats 404 as "already gone", so the delete reported success and left the
  record. The id is percent-encoded as a path segment now, while the
  signature stays over the raw link as Cosmos's auth scheme requires — a
  divergence the client's `resource_link`/`url_path` split already existed
  for.
* **A percent-escaped data URI was evaluated as its own escaping.** RFC
  2397's data segment is URL characters, so `%63%32%56%6A%63%6D%56%30` is a
  legal spelling of `c2VjcmV0` — "secret". Decoding that as base64 fails, and
  the mapper then submitted the *serialized URI* as text: Purview classified
  percent-encoded gibberish while any compliant consumer downstream read the
  real bytes. The payload is percent-decoded and unwrapped before the base64
  decode now. The same shape as the media-type finding earlier in this list —
  a decoder stricter than the producers it has to read, in a check where
  failing to decode means failing to evaluate.
* **A multi-vector upsert destroyed the sibling embeddings.** The generated
  upsert derives exactly one vector — the one `embed_from_field` names — and
  an upsert replaces the whole document, so on a collection with several
  vector fields it dropped the others on every write and never set them on a
  new record. Reachable only because naming a `vector_field` (an earlier
  correction in this same list) made those collections buildable at all: a
  fix opening a door onto a second bug. The upsert tool is refused there now;
  search, get and delete never write a vector and are unaffected.
* **An embedding source declared non-string could never work.** `build`
  checked that `embed_from_field` names a *data* field but not its declared
  type, so naming an `int` field produced a schema asking the model for an
  integer and an executor reading it with `as_str` — every schema-valid
  upsert failing at runtime. Refused at `build` now; an *undeclared* type is
  still left alone, since guessing would reject a good untyped text field.

The two rounds that followed each found the *same* thing one level out: a
validation stricter than the configuration needed (the vector field) and a
translation quietly disagreeing with the portable evaluator (the `between`
bounds). Both are the classes already named above, which is the argument for
reading a review for its class rather than its instance.

Four findings were rejected, all of them duplicates or wrong rather than
matters of taste. Two were a reviewer repeating, against a commit that had
already moved, a finding fixed in the round before.

A third claimed the Cosmos membership filters diverge from the portable
evaluator on stored nulls — that `any_of(f, [null])` and `none_of(f, [1])`
both match a present null in memory while Cosmos refuses them. They do not:
`filters.rs` guards every operator but `Eq`/`Ne` with `if actual.is_null() {
return Ok(false) }`, so a present null is a non-match for `In`/`NotIn` on
both sides, and the Cosmos emit already mirrors it exactly — as it does the
*other* half, where `Ne` emits `IS_NULL(x) OR x != p` precisely because a
stored null does match there. Applying the suggestion would have created the
divergence it was written to prevent. The neighbouring null finding, on
`Eq`/`Ne`, was right and was fixed; being right about one is not evidence
about the next.

The fourth, a Codex P1 claiming Cosmos `VectorDistance` returns a
lower-is-closer distance for cosine and dot product, is **wrong**, and was
rejected against
Microsoft's documentation rather than on judgement: `VECTORDISTANCE`
"returns the similarity score", the container policy reference gives cosine
as "-1 (least similar) to +1 (most similar)" and euclidean as "0 (most
similar) to +inf (least similar)", and the vector-search page's bare
`ORDER BY VectorDistance(...)` is documented as sorting "most-similar to
least-similar". All three metrics therefore agree with the portable name's
direction, `higher_is_closer` from the definition is already right, and
setting `score_kind` to a distance would have inverted cosine and dot-product
scores — introducing the bug the finding was written to prevent.

### Already ahead of upstream (3)

Recorded because each was a deliberate earlier decision that a later refactor
could undo without noticing.

| Upstream | Why it does not land |
|---|---|
| #8371 | **HTTP cookie persistence.** Upstream configured its internally-created MCP, A2A, AG-UI and declarative clients to reject response cookies. This workspace pins `reqwest` with `default-features = false` and never enables its `cookies` feature, so no cookie jar exists to persist into — the property upstream had to configure is structural here. Worth keeping structural: adding the feature for one crate would quietly give every transport a jar. |
| #8303 | **MCP sessions bound to request identity.** Upstream's shared `httpx` client let a session established under one caller's identity be reused under another's. Each `McpStreamableHttpTransport` here owns its client and its headers for its lifetime, so there is no identity to change out from under a session — the same structural reason #8039 did not land last pass. The *origin* dimension of the same concern did land, in the #8285 window. |
| #8354 | **Encrypted reasoning inflating compaction token counts.** Upstream's counter serializes each message and tokenizes the JSON, so an opaque `protected_data` blob was counted as though the model would read it. This port's `count_content_tokens` matches on the content variants that carry real text and never sees `protected_data` at all, so there is nothing to exclude. The same design also means its `reasoning_details` payload, added above, costs nothing in the budget — which is correct, since the provider does not tokenize it either. |

### Not applicable (114)

Grouped by why, rather than one row each. The commits touching subsystems this
port has were read as diffs; the routine remainder (CI, dependency bumps,
docs, samples, release version bumps) was classified from its file scope and
subject.

| Upstream | Why not |
|---|---|
| #5735, #7687, #7815, #7991, #8334, #8375, #8377, #8402, #8406, #8425, #8427, #8432, #8434, #8458, #8474, #8529, #8531 | **.NET only.** The GitHub Copilot SDK bump and per-session token, per-run `ChatClientAgent` tools, `AsIChatClient`, approval binding on replay, cached/reasoning token counts in Foundry Hosting, hosted session key boundaries, promoting `AgentSessionStore` into Abstractions, provider-backed MCP session scoping, workflow URI canonization and cookie-jar redirection tests, `OpenTelemetryAgent.DefaultSourceName`, disabling function tools on `ModeProvider`, harness middleware for dynamic tools, and the release version bump. |
| #8238, #8394 | **FIDES.** Tightening security-label enforcement and restricting argument labels against owned input integrity. `agent_framework.security` has no Rust counterpart; it is an `@experimental` surface upstream is still reshaping weekly, which is the standing reason not to pin it. |
| #8230, #8380, #8459, #8404, #8507, #8294 | **Sandboxed code execution and shell tools.** Hyperlight output cleanup and its test stabilization, CodeAct tool parameter schemas, MCP skill-archive digest verification, regex validation for shell and file access, and separating hosted from local shell calls. This port has neither a sandbox, an archive loader, nor a shell-tools crate. |
| #8430, #8387 | **Skills as files.** Upstream replaced its regex frontmatter parser with a real YAML one, and threaded runtime arguments through skill callbacks. This port's skills are values constructed in code — there is no `SKILL.md` loader and no script callback — so neither has a landing site. The frontmatter work is worth revisiting if a loader is ever added; its whole point is that a hand-rolled parser gets block scalars and CRLF wrong. |
| #8058, #8149, #8163, #8279, #8368, #8373, #8433 | **AG-UI depth.** Keeping workflow reasoning in thread snapshots, deduplicating client-replayed transcripts on resume, `checkpoint_id` on interrupt metadata, closing open tool calls before an approval interrupt, sanitizing workflow error messages, and preserving approval context across resume paths. `agent-framework-hosting::agui` streams one run to completion and keeps no snapshot store, so each lands on machinery the router does not have — the standing AG-UI depth gap, unchanged. |
| #8263, #8363, #8372, #8409, #8525, #8117 | **`foundry_hosting`.** Hosted session key boundaries, parallelized pre-model storage reads, request-scoped agent factories, test-state isolation, preemption-test synchronization, and call-level compaction summaries in hosted responses. Hosting agents *on* Foundry infrastructure, against a server contract this repo does not have — still the single most frequent source of rows in this table. |
| #8184, #8185, #8381, #8460 | **Non-Azure and SDK-blocked vector connectors.** MongoDB, Azure DocumentDB, a Qdrant readiness race, and Cosmos memory retrieval compatibility. The first two need a MongoDB wire-protocol client this workspace does not have; the last wraps the separate Azure Cosmos DB Agent Memory Toolkit rather than the Cosmos data plane. See the Azure review below. |
| #8260, #8366, #8435, #8508, #8509, #8510, #8547 | **Declarative-workflow depth.** Portable YAML path loading, preserving `MessageText` through formula preprocessing, keeping internal routing kwargs out of the `Agent.run` splat, `autoSend` expression evaluation, expression results in `SendActivity` output, reference-traversal caching, and `InvokeAzureAgent` input arguments in agent text. `agent-framework-declarative` reads an agent manifest; it has no formula engine, no expression evaluator and no workflow-routing layer, which is where all seven live. |
| #8328, #8351, #8398, #8311, #8272, #8262, #8214, #8261 | **Python runtime shapes.** Postponed `@response_handler` annotations, `Literal` handling in `is_instance_of`, OTel context-detach errors in graph streams, a kwargs collision with `__global__` executor ids, checkpoint dicts whose keys collide after `str()`, malformed Base64 checkpoint payloads, `FileCheckpointStorage` save/load encoding symmetry, and concurrent deletion of a checkpoint file. Rust has no postponed annotations, no runtime AST, no untyped kwargs bag, `serde` types the checkpoint payload with no Base64 layer and no `str()`-keyed dict, file I/O is bytes rather than locale-encoded text, and `delete` already treats `NotFound` as success. The *other* `FileCheckpointStorage` fix in the same family, #7757, did land. |
| #8231, #8287, #8300, #8309, #8345, #8389, #7999, #8420, #8258, #8384, #8329 | **Workflow and orchestration shapes this port does not have.** Waking workflow streaming on iteration completion, the `WorkflowAgent` return type and empty-stream response, functional-workflow HITL state and response-type validation, restoring runtime tools on nested resume, response-update metadata forwarding, custom result-parser failures, registering built-in orchestration types for checkpoint restore, stable orchestration workflow names, and keeping compaction configuration when `HandoffBuilder` clones participants. Each was read against the Rust engine: the functional-workflow API, the result-parser hook and the type registry have no counterpart, and the rest are already the behavior here (a typed `WorkflowAgent` return, a stream that ends by ending, metadata carried by `absorb_update`). |
| #8274, #8386, #8399, #8401 | **MCP.** Not duplicating `structuredContent` when `content` is present, ambiguous configuration-name matching, annotation override scope, and deprecating the sampling callback. The duplication bug does not exist here — `CallToolResult::text` has always preferred content and fallen back to structured — so what remains of the first is upstream's new five-mode policy knob, which is configuration rather than a fix; the port's behavior equals its `content_first` mode. The other three land on a configuration registry, an annotation-override layer and a sampling callback this port does not have. |
| #8288, #8428, #8449, #8451, #8450, #8536, #8560, #8396, #8353, #7944, #8358, #8078, #8424, #8392, #8395, #8319 | **Surfaces and hooks with no counterpart.** Middleware repairing function arguments, mixed function-call batch classification and stateless pause batches (both on the `_harness` tool-approval middleware), instrumentation message-event control, per-tool `AgentModeProvider` exposure and empty-mode rejection, expected keys in tool-argument evaluation, provider-invalidated responses, instruction-order preservation on partial dedup, synthetic compaction summaries, dropping a continuation token after a resumed background stream, Responses `function_call_output` parsing for Host transports, streamed image-generation results, per-call Mistral transport options, Monty callbacks, and a Serply MCP sample. This port has no harness, no mode provider, no `ContinuationToken`, no Host transport channel, no hosted image generation, no Mistral transport-options bag (its client takes no per-call transport arguments), and no Monty bridge. Two deserve a note: instruction dedup does not exist here because an agent joins instructions into one string and prepends it at exactly one point, so the order inversion upstream fixed cannot arise; and #8078's preserved hosted result landed here in the #8285 window. |
| #8505, #8497, #8266, #8249, #8469, #8470, #8471, #8472, #8473, #8512 | **Dependency floors and lockfiles.** Python 3.15 resolution, cross-project dependency updates, ruff, `ty`, uv, pyright, `huggingface-hub`, the lab package's own pins, and requiring `azure-monitor-opentelemetry>=1.8.10` for Foundry HTTPX auto-instrumentation. This port pins no provider or exporter SDK and exports over OTLP, so none has a landing site. The last one is a reminder of a standing gap rather than a change: see the Azure review. |
| #7874, #8362, #8419, #8439, #8271, #8476, #8477, #8480, #8481, #8482, #8483, #8484, #8485, #8498 | DevUI Aspire traces, issue triage, public-API compatibility checks and their fork-PR/merge-ref handling, dependabot and action bumps, the additional-languages policy, and the Python 1.19.0 release bump. |

### The Azure-ecosystem review

The whole Azure surface, upstream against this port. This pass changed five
things and found one new gap; the rest is the standing picture.

| Azure surface | Upstream | Here | Assessment |
|---|---|---|---|
| Azure OpenAI — chat completions, Responses, embeddings | ✅ | ✅ | Complete on both auth modes. A content-filter refusal now carries **which policy and which category fired**, which is the part an application acts on (above). |
| Microsoft Entra ID credentials | ✅ (`azure-identity`) | ✅ | `DefaultAzureCredential`, managed identity, workload identity, client secret and the Azure CLI, all with per-scope caching, plus an `azure_core`-backed path. Unchanged. |
| Azure AI Foundry — chat client | ✅ | ✅ | Foundry project Responses API, delegating to the Azure Responses client rather than re-implementing it. Unchanged. |
| Foundry — Prompt Agents | ✅ | 🚧 | `FoundryAgent` realizes a Prompt Agent client-side; binding to a server-hosted agent by id on the Agents control plane remains a documented extension point. Unchanged. |
| Foundry — embeddings | ✅ | 🚧 | Text inputs only; upstream also splits a batch across an image-embeddings endpoint, which the core `EmbeddingClient` signature cannot express without widening a shared trait. Unchanged. |
| Foundry — memory provider | ✅ | ❌ | Flagged last pass as the most tractable Azure item left, and **deliberately not attempted this pass**. Its two operations (`search_memories`, `begin_update_memories`) are reached through `azure-ai-projects`' `beta.memory_stores`, and that SDK is not available in this environment — so the REST paths, the api-version and the long-running-operation shape would all be guesses. A connector aimed at a guessed URL is worse than a gap, because it looks finished. Reopen when the wire surface can be read rather than inferred; the provider itself is only ~280 lines and the `ContextProvider` shape beside it (`Mem0Provider`, `AzureAISearchProvider`) is the right one. |
| Foundry — evaluations | ✅ | ❌ | Graders and evaluator runs over stored responses, built on the same `azure-ai-projects` evals client and blocked the same way, on top of being large and still moving. Unchanged. |
| Foundry hosting (`foundry_hosting`) | ✅ | ❌ | Standing gap: hosting agents *on* Foundry infrastructure, against a server contract this repo does not have. Six rows in this window's not-applicable table. |
| Foundry Local | ✅ | ✅ | OpenAI-compatible localhost endpoint. Unchanged. |
| Azure AI Search — context provider | ✅ | ✅ | Hybrid/semantic retrieval. Unchanged. |
| Azure AI Search — vector store | ✅ | ✅ | Built last pass, and now reachable from an agent through the new `VectorCollectionContextProvider` without writing a tool by hand. |
| Azure AI Search — Knowledge Base ("agentic") retrieval | ✅ | ❌ | Standing documented gap: upstream's second retrieval mode, a different service surface from the one the provider speaks. |
| Azure Cosmos DB — history provider | ✅ | ✅ | Including Entra ID auth. Gained upstream's storage-scope note this pass. |
| Azure Cosmos DB — checkpoint storage | ❌ | ✅ | This port is ahead: workflow checkpoints in Cosmos have no upstream Python equivalent. |
| Azure Cosmos DB — vector store | ✅ | ✅ | **Built this pass**, and the largest Azure item that was open. |
| Azure Cosmos DB — memory provider | ✅ (`azure-cosmos-memory`) | ❌ | Externally blocked, unchanged: it wraps the separate Azure Cosmos DB Agent Memory Toolkit rather than the Cosmos data plane, so porting it means porting that toolkit first. Upstream's #8460 compatibility fix this window lands on that wrapper. |
| Azure DocumentDB — vector store | ✅ (`azure-documentdb`, #8185) | ❌ | **New gap this window.** Azure's MongoDB-compatible offering, so the connector speaks the MongoDB wire protocol rather than a REST surface — which is the whole cost of it: this workspace has no MongoDB driver, and adding one is a larger decision than adding a connector. The sibling `mongodb` connector (#8184) is the same shape and the same blocker. Worth noting that `core::vectors` is now the *only* thing these need, which was not true two passes ago. |
| Azure AI Content Understanding | ✅ (`azure-contentunderstanding`) | ❌ | Unchanged, and now the largest Azure item that is neither blocked nor guessed at: a `ContextProvider` running Content Understanding analyzers over documents, audio and video, plus a file-search backend pair. A REST surface, so genuinely portable. The strongest candidate for next pass. |
| Microsoft Purview | ✅ | ✅ | **Three findings this pass**, all above: it evaluated only each message's text, a new enforcement action made a verdict unparseable, and an unresolvable user reported "allowed". Complete against upstream now. |
| Azure Monitor / Application Insights | ✅ (exporter package) | 🚧 | Unchanged. Spans and metrics export over OTLP, which Azure Monitor accepts through its OTLP-capable collector; the dedicated exporter (connection-string auth, its own ingestion endpoint, and now the HTTPX auto-instrumentation #8512 relies on to join client and service traces) is not built. |

Two things changed in the *shape* of this table rather than in a single row.
Azure storage is no longer the weak side of the surface — history,
checkpoints and vectors are all built on Cosmos now, and both Azure vector
stores are reachable from an agent rather than only from a caller. And what
remains divides cleanly into three kinds: blocked on an SDK whose wire
surface cannot be read here (Foundry memory, Foundry evals), blocked on a
dependency this workspace does not carry (DocumentDB, MongoDB), and simply
not built yet (Content Understanding, Knowledge Base retrieval, the
Application Insights exporter, Foundry hosting). Only the third kind is work
this port can decide to do.

## Post-`010a43a` drift + Azure-ecosystem review (checked against `061dc28`, 2026-09-14)

Upstream moved **116 non-merge commits** in this window (2026-09-07 → 09-14).
**Eight land on this port**, and they are unusually substantial for one
window: upstream made its vector-store filter portable and built the first
connector on top of it, added a second and third bound to the tool loop, and
landed three separate isolation fixes (MCP headers, Redis keys, `SecretString`)
whose Rust equivalents were all present and all exposed. Alongside the drift
triage, this pass ran a review of the whole Azure surface — the largest gap it
found is now built (the Azure AI Search vector store), and one silent
misconfiguration is fixed (the chat api-version).

### The mirror sync worked, unattended, for the first time

Every window since 2026-08-16 had to be recovered by hand: the fork-sync
failed on any upstream commit touching `.github/workflows/**`, which upstream
touches most days. The fix landed in the mirror repo last pass (restore the
fork's own workflow directory after merging, so the pushed ref contains no
change under it). It has now held for a full week — the mirror carried
upstream through 2026-09-14 with no manual intervention, and this window was
readable the moment the pass started. Nothing to do here; recorded because
three consecutive passes had to open with the opposite.

### Ported this pass (8, all with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #8115 | **Portable vector filters.** Upstream turned the vector-store filter from a provider dialect into an expression tree, which is what makes a collection swappable at all: `core::vectors` landed here last pass with `VectorSearchOptions::filter` as a raw provider string, so a caller who wrote one was pinned to that provider. Built as `core::vectors::filters`: 18 operators across `Filter` leaves and `FilterGroup` nodes, validated at construction (operand shape per operator, depth ≤ 8, ≤ 64 nodes), plus an evaluator implementing upstream's documented semantics that `InMemoryVectorStore` now uses instead of refusing every filter. Three semantics are load-bearing and easy to get subtly wrong: a missing field is a non-match for every operator *except* `exists` (so `ne` and `not(eq)` differ, which the module docs and a test both pin); a boolean never equals a number; and two numbers compare **by value**, so a record that round-tripped through JSON as `1` still matches `eq: 1.0` — `serde_json`'s own `==` says it does not, and the mismatch would silently return nothing. Provider-specific operators are namespaced (`azure_ai_search.match`), so a misspelled standard operator is rejected at construction rather than at search time. Upstream's `Param` late-binding machinery is not ported: it exists to fill filter values from an agent tool call that Python cannot type, which a Rust caller does with the arguments it already holds. | `core/vectors/filters.rs` (new), `core/vectors.rs` (`VectorSearchOptions::filter` / `provider_filter`, `InMemoryVectorStore::search`) |
| #8153 | **An Azure AI Search vector store.** The largest Azure gap this pass's review found, and the reason the filter work came first. The crate was a read-only `ContextProvider` against an index someone else built; it now owns one. `AzureAISearchStore` / `AzureAISearchCollection` speak the Search REST API directly (as the context provider beside them already did, rather than through an SDK): index create/exists/delete from a collection definition — EDM types, vector profiles, hnsw vs exhaustiveKnn, the metric mapping — document upsert/get/delete over the indexing-batch endpoint, vector and keyword-hybrid search, and index aliases. Three details are right rather than plausible. Filtering is `preFilter`: a post-filter discards matches from an already-truncated neighbor list, so a selective filter returns far fewer than `top`, and `k` covers `skip + top` for the same reason. An indexing batch can return HTTP **207** with per-document failures — a success status, so a plain `is_success()` check reports a half-written batch as a complete one; the per-document results are read and any failure raised with the service's own message. And an operator Azure cannot express is refused rather than dropped: `ne`/`exists`/`is_null`/`is_not_null` cannot preserve missing-versus-null semantics there, and `starts_with`/`ends_with`/`contains_text` have no literal form, so each names the alternative (a NOT group, `azure_ai_search.match`) instead of silently widening the result set. Filter literals double embedded quotes, so a value cannot become part of the expression. | `azure-ai-search/vector_store.rs` (new), `azure-ai-search/tests/vector_store_loopback.rs` (new), `examples/memory/azure_ai_search_vector_store.rs` |
| #7587/#7772 | **Bounds on the tool loop.** Upstream added `max_duration_seconds` alongside the `max_function_calls` it already had; this port had neither, so the only bound was `max_iterations` — which caps model round trips and says nothing about how many tools run per trip or how long any of them takes. Both are added, both graceful (tools off, model answers with what it has), both validated (zero, negative and NaN refused rather than silently meaning "never"). Three things a transcription would have missed. The budget lives in the **session**, not the request: an approval round trip is a separate `get_response`, so a per-request budget would reset at exactly the pause it most needs to survive, and an unattended approve-and-continue loop would never hit a limit. A spent budget **stops execution**, not just asking — `tool_choice: none` is a hint, and a provider that ignores it would have its calls executed, which is the one thing a ceiling must not allow — so the loop breaks to its existing tools-disabled failsafe, re-checking after the approval replay since that replay can spend the last of it. And an approval arriving past the budget still gets a **result** saying it did not run, where upstream simply skips execution and leaves approval content unresolved in the conversation. `AgentBuilder::function_invocation_config` makes all of this reachable from an agent at all — the builder wraps the caller's client itself, which had left `max_iterations` unreachable too. | `core/tools.rs` (`FunctionInvocationConfig`), `core/client.rs` (`InvocationBudget`), `core/agent.rs` (builder) |
| #8285 | **MCP headers scoped to their origin.** Upstream moved its Secure MCP proxy's static headers behind an origin-scoped hook. This port had the same exposure and a slightly worse version of it: headers went onto a default `reqwest::Client`, which follows redirects and strips exactly three header names (`Authorization`, `Cookie`, `Proxy-Authorization`) on a cross-*host* hop — so an `X-Api-Key`, the shape most MCP servers actually use, went to whatever host the server redirected to, and because the check is host-only even `Authorization` survived a redirect to a different port or scheme. The `Mcp-Session-Id` was in the same position both ways: sent to a redirect target, and *adopted* from one, so any host the server redirected to could fix this client's session. The transport now follows redirects itself with `Policy::none()`, comparing scheme/host/port (ports defaulted from the scheme, so `https://h` and `https://h:443` are one origin) and re-attaching headers and session id only while same-origin. Method and body are preserved across every hop: `reqwest` would have degraded a 301/302/303 to a GET, which turns a JSON-RPC request into something no MCP server can answer. `McpStreamableHttpTransport::new` now returns a `Result` and rejects a URL with no origin to scope to. | `mcp/transport/http.rs` (`url_origin`, `post`, `capture_session_id`), `mcp/tests/http_loopback.rs` |
| #8171 | **Anthropic extended-thinking signatures.** Upstream fixed a narrow merge bug; this port had the whole mechanism missing — it parsed a thinking block's text and dropped its `signature`, never handled `signature_delta` while streaming, and emitted unsigned thinking blocks outbound. With extended thinking on, the Messages API requires a replayed thinking block to carry its signature and validates it against the block's exact text, so a conversation that used extended thinking could not be continued at all: the next request, usually the tool-result turn, was rejected. `redacted_thinking` was worse — it fell through the catch-all and vanished, breaking the same replay with no trace. Signatures now ride in `protected_data` (the field Gemini thought signatures already use) on both paths, with the streamed `signature_delta` carried as an empty-text fragment so the existing `coalesce_text` merge lands it on the block it signs. Outbound, a decoded block is replayed **verbatim** from its raw form rather than re-serialized, because the signature covers the exact text; an unsigned reasoning content is dropped rather than sent, since the API rejects it and emitting one turns a working request into a 400. | `anthropic/convert.rs` (`thinking_block`, `parse_content_blocks`), `anthropic/lib.rs` (stream `signature_delta`) |
| #7948 | **The graph signature was not injective.** Only the sub-fix in upstream's third commit lands — the fan-in buffer carry and the sibling-cancellation race it is mostly about are both already handled here (see *Already ahead* below) — but it lands on a bigger surface than upstream's. `compute_graph_signature` joined executor ids with `,` and `->`, and ids are only required to be non-empty: a fan-out to `["x", "y"]` and one to the single target `"x,y"` produced the same descriptor *and* the same node list, so two different graphs shared one signature and a checkpoint from either would be accepted for the other — resumed onto a topology it was never written for, which is what the signature exists to prevent. Now JSON-encoded throughout, including the node list, which upstream's own fix did not have to cover. Every signature changes, so the scheme tag is bumped to `v2` and a `v1` checkpoint is reported as a *scheme* mismatch — "the graph itself may well be identical" — rather than as a graph change the reader will go looking for and not find. | `core/workflow/runner.rs` (`edge_group_descriptor`, `compute_graph_signature`, `check_graph_signature`) |
| #8127 | **`SecretString` serialized its secret.** Upstream rewrote its own away from a `str` subclass because every path but `repr()` silently produced the value; the Rust type already masked `Debug` and `Display` but derived `Serialize`, so `serde_json::to_string` on any struct holding one wrote the secret in cleartext — no call site to audit, which is the entire failure the type exists to prevent, and it made the careful masking beside it beside the point. The impl is **removed** rather than masked: a masked `Serialize` round-trips `"***"` back as the value and silently replaces the secret with the mask. Without it, `#[derive(Serialize)]` on a struct holding one fails to compile, which is where the author decides what to write instead. `Deserialize` stays — reading a secret in from config is what the type is for. | `core/settings.rs` |
| #8236 | **Redis keys were ambiguous.** `{key_prefix}:{session_id}` is only unambiguous while no identifier contains the separator, and nothing constrains one: `prefix="chat"` + `session="a:b"` and `prefix="chat:a"` + `session="b"` address the same list, so two conversations that should be isolated share a history — and where the prefix is a tenant boundary, that is one tenant reading another's. `core::storage_keys::storage_key_segment` renders each segment unambiguously before joining. The context provider had a second variant: a prefix containing `:entry:` puts one provider's entries inside another's `SCAN MATCH a:entry:*`, so it reads — and `clear()` deletes — memories that are not its own, and a prefix containing a glob metacharacter changes what that pattern selects at all. A literal-safe value passes through unchanged, which covers the default prefix and every UUID session id, so existing data stays addressable and this is not a migration; only the ambiguous identifiers move, which are the ones that were colliding. Two documented divergences: hex rather than base32-with-a-SHA-256-fallback (these are Redis keys, whose limit is 512 MB rather than a filesystem's 255 bytes, so the cap guards nothing and dropping it keeps the derivation injective rather than merely collision-resistant), and upstream's tenant/application/agent segments are not ported because with an injective encoding a caller composes those into `key_prefix` and gets the same isolation — which is exactly what they could not safely do before. | `core/storage_keys.rs` (new), `redis/chat_message_store.rs`, `redis/context_provider.rs` |

### Also ported this pass: the Azure review

The review is below; it produced one code change beyond the vector store.

| Gap | Change | Rust site |
|---|---|---|
| **The Azure chat api-version was pinned to GA.** One `DEFAULT_API_VERSION = "2024-10-21"` covered both the chat-completions and the embeddings client. Upstream carries two, and the difference is not bookkeeping: its chat default is `2024-12-01-preview` while its embedding default stays on GA `2024-10-21`. Azure OpenAI answers a request field its api-version does not know with "Unrecognized request argument supplied" rather than ignoring it, and this client sends `store` whenever a caller sets it — a field `2024-10-21` does not have. So the pin quietly narrowed what the Azure client could express relative to the OpenAI client it shares its request builder with, and did so as a runtime rejection at the first caller to use the feature. Split into `DEFAULT_CHAT_API_VERSION` / `DEFAULT_EMBEDDING_API_VERSION`, both public so a caller pinning a validated version can see what they are moving away from. | `azure/lib.rs`, `azure/embeddings.rs` |

Verified across all nine: full workspace build, `cargo test --workspace
--all-features` (**1914 passing**, 104 of them new, including 11 hermetic
loopback tests against a fake Azure AI Search service and 2 against a pair of
loopback MCP servers), `cargo clippy --all-targets --all-features` under
`-D warnings` (CI's own flag) clean, `cargo fmt --check` clean, `cargo doc`
clean. Every behavioral test was probed against the code it pins: making
number equality fall back to `serde_json`'s fails the integer/float filter
test; switching the Azure search body to `postFilter` fails the pre-filter
test; dropping the budget-spent break lets the mock's ignored `tool_choice`
run a second batch and fails the ceiling test; widening the MCP origin check
to always match fails the cross-origin test while the same-origin control
still passes; disabling the `signature_delta` arm fails the streaming
signature test; restoring the joined graph rendering fails the collision test;
and making the storage-key derivation the identity fails all three Redis
collision tests while both existing-keys-unchanged tests still pass.

### Already ahead of upstream (2, plus half of a third)

These fixes are for bugs this port does not have — recorded because each was a
deliberate earlier decision, and a later refactor could undo one without
noticing. The third row is #7948 again: its key-encoding sub-fix landed (see
above), its main change had nothing to land on.

| Upstream | Why it does not land |
|---|---|
| #8269 | **Replayed history persisted twice in the Cosmos provider.** Upstream's `CosmosHistoryProvider.save_messages` now reads existing history and filters the replayed prefix before writing. This port did that in the #7242 window, and did it for *all four* history providers rather than the two upstream had touched then — `cosmos/chat_message_store.rs` has called `filter_new_messages` since. Upstream is catching up to the port here, not the other way round. |
| #8237 | **Anthropic request-parsing state was shared across requests.** Upstream held the active tool-call id, its content type, and the tool-name aliases on the client, so two concurrent requests corrupted each other's parse; the fix moves them into a per-request dataclass. This port never had the bug, and not by luck: streaming state lives in an `SseState` owned by the stream's `unfold`, so the borrow checker would reject the shared-mutable shape upstream had. Nothing to port. |
| #7948 (main change) | **Fan-in buffers through a checkpoint restore.** Upstream now checkpoints edge-runner delivery state and resets every runner on restore. `WorkflowCheckpoint::fanin_state` has carried partially-satisfied barriers since it was written, and `restore` replaces `self.fanin` wholesale, which is both halves of upstream's fix. The sibling-cancellation race that the rest of the PR addresses cannot arise here either: deliveries run under `join_all` inside the superstep rather than as detached tasks, so dropping the joining future cancels the rest — structured concurrency instead of an explicit cancel-and-await helper. Only the key-encoding sub-fix landed (above). |

### Not applicable (106)

Grouped by why, rather than one row each. The commits touching subsystems this
port has were read as diffs; the routine remainder (CI, dependency bumps,
docs, samples, release version bumps) was classified from its file scope and
subject.

| Upstream | Why not |
|---|---|
| #8301, #8307, #8252, #8295, #8297, #8270, #8227, #8259, #8253, #8164, #8229, #8202, #8239, #8198, #7844, #8151, #8165, #8190, #8166, #8146, #8159, #8020, #8082 | **.NET only.** Header delimiters, workflow formula state races, Hyperlight fingerprints, PowerShell exit codes, redirect header forwarding, LocalCodeAct OS validation, file-skill path revalidation, hosting storage isolation, hosted-agent workflow outputs, and dependency/analyzer bumps. |
| #8142, #8141, #8139, #8138, #8187 | **FIDES.** Policy-approval binding and recovery, confidentiality through security tools, label enforcement on expanded variables, per-session security state, and keeping MCP labels subordinate to local policy. `agent_framework.security` has no Rust counterpart; it is an `@experimental` surface that upstream is still reshaping weekly, which is the standing reason not to pin it. |
| #8176, #8174, #8289, #8233, #8290, #8118 | **Sandboxed code execution and MCP skill archives.** Bounding Hyperlight output attachments, the 0.6 sandbox bump, LocalCodeAct approval parity, local-shell approval binding, restricting skill archives to ZIP, and inline-skill argument errors. This port has skills but no sandbox and no archive loader. |
| #8199, #8197, #8116, #8129, #8158, #8130, #8005, #7808, #8128 | **AG-UI depth.** Reserved HA session ids, rejecting empty scope-resolver results, multimodal messages in chat-client requests, MCP Host payloads in snapshots and their metadata, scoped internal session ids, MCP Host history conversion, tool call/result ordering in the message split, and `emit_messages_snapshot`. `agent-framework-hosting::agui` streams one run to completion and keeps no snapshot store or scope resolver, so each lands on machinery the router does not have — the standing AG-UI depth gap, unchanged. |
| #8331, #8145, #8219, #8126, #7951, #8278, #8215, #7798, #8224, #8246, #8225, #7963, #8312 | **Python runtime shapes.** Awaiting `Task`/`Future` stream sources; preserving a `Param` unset sentinel's identity; lazy provider imports; `httpx` resource cleanup after a failed MCP connection; an `AttributeError` on exit with a caller-supplied client; counting exceptions raised *inside* a returned coroutine; validating that `on_checkpoint_save` returned a dict; `ContextVar` leaks in abandoned streams; deduplicating MessagePack file-history writes; MCP request ownership and connection-lifetime kwargs; mixed workflow invocation kwargs; and forwarding workflow run kwargs to the GroupChat orchestrator. A Rust `async fn` returns a `Result` rather than a coroutine that can fail later, `serde` types the checkpoint payload, `FileHistoryProvider` rewrites its whole file atomically, and there is no untyped kwargs bag to forward — per-run configuration is `AgentRunOptions`, which the orchestrator already receives. |
| #8155, #8154, #8156 | **New non-Azure vector connectors** — PostgreSQL/pgvector, Qdrant, and Redis HASH/JSON. Each is a substantial new crate, and each was *blocked* until this pass: with no portable filter there was nothing for them to translate. They are now the most tractable ecosystem work on the books; see the roadmap. |
| #8045, #8188, #8152, #8162, #8172, #7839, #7765, #8122, #7517, #8206, #8087, #8123, #8144, #8136 | **Packages and surfaces with no Rust counterpart.** `FoundryCheckpointStore` deserialization restrictions, isolating Lab, the Monty bridge (×2), DevUI frontend CVEs, declarative-workflow DevUI input, the workflow HTTP request handler's query strings, Claude prompt-history roles, GitHub Copilot workspace file hooks (this port has the chat client, not the agent), PowerShell session state, summarizer tool-trajectory input (no summarizing compaction strategy here), shared path normalization (this port's equivalent is the new `storage_keys` module, built for the Redis keys above), native issue types, and code owners. |
| #8323, #8335, #8324, #8316, #8317, #8318, #8321, #8322, #8325, #7942, #8296, #8222, #8213, #8208, #8210, #8211, #8212, #8207, #8209, #8064, #8175, #7583, #7584, #7887, #8069, #8110, #8113, #8112, #8111, #8100, #8167, #8179, #8190 | Docs and ADR typo fixes, dependabot and action bumps, ruff/type-checker bumps, code owners, and the Python 1.18.0 release version bump. |

### The Azure-ecosystem review

The whole Azure surface, upstream against this port. Two items changed as a
result (the vector store and the api-version split, both above); the rest is
the standing picture, and three of the gaps below are new to this document.

| Azure surface | Upstream | Here | Assessment |
|---|---|---|---|
| Azure OpenAI — chat completions, Responses, embeddings | ✅ | ✅ | Complete, with api-key and Entra auth on all three. The api-version defaults were the one thing wrong, and are fixed above. |
| Microsoft Entra ID credentials | ✅ (`azure-identity`) | ✅ | `DefaultAzureCredential`, managed identity, workload identity, client secret, and the Azure CLI, all with per-scope caching, plus an `azure_core`-backed path. |
| Azure AI Foundry — chat client | ✅ | ✅ | Foundry project Responses API, delegating to the Azure Responses client rather than re-implementing it. |
| Foundry — Prompt Agents | ✅ | 🚧 | `FoundryAgent` realizes a Prompt Agent client-side; binding to a server-hosted agent by id on the Foundry Agents control plane remains a documented extension point. |
| Foundry — embeddings | ✅ | 🚧 | Text inputs only; upstream also splits a batch across an image-embeddings endpoint, which the core `EmbeddingClient` signature cannot express without widening a shared trait. |
| Foundry — memory provider | ✅ (`_memory_provider.py`) | ❌ | **New gap.** Foundry's managed memory as a `ContextProvider`. Portable in principle — it is a REST surface on the Projects data plane — and the closest analogue here (`Mem0Provider`, `AzureAISearchProvider`) shows the shape. The most tractable Azure item left. |
| Foundry — evaluations | ✅ (`_foundry_evals.py`) | ❌ | **New gap.** Graders and evaluator runs over stored responses (~600 lines upstream, built on `azure-ai-projects`' evals client). Large, and upstream is still moving it (it left core this window), so pinning it now would buy a rewrite. |
| Foundry hosting (`foundry_hosting`) | ✅ | ❌ | Standing gap, and the single most frequent source of "not applicable" rows in this document. Hosting agents *on* Foundry infrastructure, against a server contract this repo does not have. |
| Foundry Local | ✅ | ✅ | OpenAI-compatible localhost endpoint. |
| Azure AI Search — context provider | ✅ | ✅ | Hybrid/semantic retrieval. |
| Azure AI Search — vector store | ✅ | ✅ | **Built this pass.** |
| Azure AI Search — Knowledge Base ("agentic") retrieval | ✅ | ❌ | Standing documented gap: upstream's second retrieval mode, a different service surface from the one the provider speaks. |
| Azure Cosmos DB — history provider | ✅ | ✅ | Including Entra ID auth, added last pass; this window's upstream fix (#8269) was already here. |
| Azure Cosmos DB — checkpoint storage | ❌ (upstream has a Foundry one) | ✅ | This port is ahead: workflow checkpoints in Cosmos have no upstream Python equivalent. |
| Azure Cosmos DB — memory provider | ✅ (`azure-cosmos-memory`) | ❌ | **New gap**, and an externally-blocked one: it wraps the separate Azure Cosmos DB Agent Memory Toolkit rather than the Cosmos data plane, so porting it means porting that toolkit first. |
| Azure AI Content Understanding | ✅ (`azure-contentunderstanding`) | ❌ | **New gap.** A `ContextProvider` that runs Content Understanding analyzers over documents, audio and video and injects the extracted fields, plus a file-search backend pair. A REST surface, so genuinely portable; the largest remaining Azure item that is not blocked on something else. |
| Microsoft Purview | ✅ | ✅ | Prompt and response policy enforcement as middleware, including the inline-evaluation header added last pass. |
| Azure Monitor / Application Insights | ✅ (exporter package) | 🚧 | Spans and metrics export over OTLP, which Azure Monitor accepts through its OTLP-capable collector; the dedicated Application Insights exporter (connection-string auth, its own ingestion endpoint) is not built. |

Nothing in the Azure surface is *wrong* after this pass; what is left is
missing rather than misbehaving, and the three items marked new are each a
self-contained crate.



Upstream moved **83 non-merge commits** in this window (2026-08-31 → 09-07).
**Six land on this port**: a usage-accounting bug it inherited from upstream
and has now fixed, plus the five changes that land on surfaces this port has
but needed more than a transcription (refusal marking, occurrence-bound
approvals, the compaction preserve-first-user option, core vector stores, and
Purview's inline-evaluation header). Alongside the drift triage, this pass ran
a review of the whole Azure surface and closed its largest gap (Cosmos DB
authentication).

### The sync was still broken; the window was recovered by hand

The mirror did not advance on its own: its scheduled sync failed on **every**
run from 2026-09-01 through 09-07, seven consecutive scheduled failures, and
the window below only became readable because it was synced manually.

The cause was not the one the previous pass fixed. That pass moved the sync
onto the fork-sync API (`POST /repos/{owner}/{repo}/merge-upstream`) on the
premise that a server-side merge would sidestep GitHub's refusal to let
GITHUB_TOKEN write `.github/workflows/**`. It does not — the endpoint applies
the identical rule, and every run returned `422`:

> refusing to allow a GitHub App to create or update workflow
> `.github/workflows/codeql-analysis.yml` without `workflows` permission

Because upstream touches a workflow file regularly, one such commit entering
the window stops the mirror indefinitely rather than for a single day, which
is why three successive windows were affected.

Fixed in the mirror repo (`CodeHalwell/agent-framework`
[PR #6](https://github.com/CodeHalwell/agent-framework/pull/6), branch
`claude/amazing-mayer-h5ps6a`): the sync is a plain git merge again, and the
pushed ref update is made to contain no change under `.github/workflows/**` —
after merging, that directory is restored to the fork's own version, so the
restriction has nothing to apply to. Upstream history still merges in full, so
the merge base advances and a window is never re-attempted; only upstream's CI
*files* are frozen, which costs this mirror's one consumer nothing. A
`SYNC_PAT` secret carrying the `workflow` scope, if added, is used instead and
drops the exclusion. Verified against a local two-repo simulation of both
shapes upstream produces (a workflow file modified plus one added alongside a
source change; and a workflow file edited on the same lines by both sides).

Until that PR merges, each window still depends on someone syncing the mirror
by hand. The next pass should confirm the 06:00 UTC run went green before
triaging.

### Ported this pass (7, all with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7964 | **A token count reported as zero was dropped from the usage breakdown.** Python's walrus guard `if tokens := usage.completion_tokens_details.audio_tokens:` is falsy for `0`, so a provider reporting *zero* audio / accepted-prediction / rejected-prediction / cached tokens produced no entry at all — indistinguishable from a provider that does not break that count out. This port had transcribed the guard faithfully as `if v > 0` in `add_usage_detail`, **with a test pinning it** ("Zero-valued counts are skipped (truthy guard)"), so the bug was not merely inherited but protected. Upstream's fix is `is not None`; here it is dropping the comparison, leaving `Value::as_u64` as the only filter — so a non-integer is still ignored and the guard has not been widened into accepting junk. The distinction matters downstream: `additional_counts` feeds the GenAI metrics layer, which can only express "not reported" by omission, so a real zero and a missing field collapsed into the same reading. Scoped to Chat Completions: `parse_responses_usage` never had the guard, and the typed fields (`reasoning_output_token_count`, `cache_read_input_token_count`) never did either — which is why the two paths silently disagreed about the same response until now. | `openai/convert.rs` (`add_usage_detail`) |

The other five from this window are below; the Cosmos DB authentication work
that follows them came out of the Azure review rather than this window.

### Ported this pass (Azure review)

| Gap | Change | Rust site |
|---|---|---|
| **Cosmos DB was master-key-only.** `agent-framework-cosmos` signed every request with an HMAC-SHA256 master key and had no Entra ID path, while .NET's `Microsoft.Agents.AI.CosmosNoSql` accepts an already-authenticated `CosmosClient` and therefore any `TokenCredential`. That is not a cosmetic gap: an Azure account with `disableLocalAuth` set — a common governance default, since key-based auth cannot be attributed to a principal or scoped by role — has no master key to give, so the crate could not be used on such an account at all, and a deployment that could use it had to carry a key. The credential machinery already existed one crate away (`agent_framework_azure::credentials`: `DefaultAzureCredential`, `ManagedIdentityCredential`, `WorkloadIdentityCredential`, `ClientSecretCredential`, `AzureCliCredential`, all caching per scope), so this was wiring rather than new infrastructure. `CosmosAuth` now selects between the two modes at the one point that builds the `Authorization` header. Cosmos does **not** take an Entra token as `Authorization: Bearer` — it reuses the master-key envelope as `type=aad&ver=1.0&sig={token}`, percent-encoded — which is the detail a hand-rolled client gets wrong silently, so it is pinned by both a unit test on the header builder and a loopback test on the real outbound request. The default scope is `https://cosmos.azure.com/.default`, matching `AAD_DEFAULT_SCOPE` in the official `azure-cosmos` SDK, with `AZURE_COSMOS_AAD_SCOPE_OVERRIDE` honored as that SDK does. The first cut derived it from the account endpoint instead — which reads plausibly and is what a per-account service would use — and PR #21 review caught that Entra rejects `https://<account>.documents.azure.com/.default` as an invalid scope outright, so every credential-backed store would have failed at token acquisition. | `cosmos/auth.rs` (`aad_authorization_header`), `cosmos/client.rs` (`CosmosAuth`, `with_token_credential`, `authorization`, `management_error`, `normalize_endpoint`), `cosmos/chat_message_store.rs`, `cosmos/checkpoint_storage.rs` |

Two consequences are handled rather than left to surface as confusing
failures:

- **`ensure_created` cannot work under Entra ID.** Cosmos DB's Entra RBAC
  covers data-plane actions only; creating a database or container is a
  control-plane (ARM) operation, so it is refused with `403` no matter which
  role the principal holds — including Owner. The raw body says only "Request
  blocked by Auth", which reads like a missing role assignment rather than an
  operation no assignment can grant. `management_error` attaches the
  explanation and the way out (provision out of band, or use a master key),
  and does so *only* under credential auth: a `403` on the master-key path
  means a wrong or revoked key, and a negative-control test pins that it is
  not re-labelled.
- **A credential cannot be serialized.** `serialize()` embeds the master key
  when there is one; under Entra ID there is no value to embed, so it emits
  `"auth": "token_credential"` and no `key` — the one shape of this blob that
  is not itself a secret. `from_state` then fails naming
  `from_state_with_token_credential`, which takes the credential back in
  (the same shape as .NET's `CreateFromSerializedState(CosmosClient, ...)`).
  Master-key state is also accepted there, which is the supported way to move
  an existing conversation off a key without losing its thread id.

Verified: full workspace build, `cargo test --workspace --all-features`
(**1726 passing**, 14 of them new), `cargo clippy --all-targets --all-features`
under `-D warnings` (CI's own flag) clean, `cargo fmt --check` clean. The new
tests were probed against the behavior they pin: degrading `management_error`
to the plain error mapping fails the Entra diagnostic test while the
master-key negative control still passes; emitting `Bearer {token}` instead of
the AAD envelope fails the two header-shape tests; dropping the endpoint trim
before deriving the scope fails the scope-default test (and only it — the
explicit-override test still passes); and reinstating the `> 0` truthy guard
fails both usage tests, while the non-integer negative control passes either
way by design.

### Not applicable (77)

Grouped by why, rather than one row each. The commits touching subsystems this
port has were read as diffs; the routine remainder (CI, dependency bumps,
docs, samples, release version bumps) was classified from its file scope and
subject.

| Upstream | Why not |
|---|---|
| #7790, #8023, #7704, #8095 | **Python runtime shapes with no Rust counterpart.** Recursion/cycle handling and scalar fast paths in `SerializationMixin`; a `__getstate__`/pickle path so runtime raw representations survive checkpointing; naming the real error when an anyio cancel scope masks an MCP init failure; and initializing `_close_http_client` on the user-supplied-client path. Serialization here is `serde` over owned types (a `serde_json::Value` cannot be cyclic and needs no pickle protocol), there are no cancel scopes, and client ownership is settled by the type system. |
| #8039, #7855, #7835, #7747 | **Python-shaped again, one layer up.** Scoping MCP provider headers per tool across a *shared* `httpx` client with mutable request-hook lists; wrapping Anthropic and Gemini SDK exceptions in `ChatClientException`; rejecting MCP servers passed as provider agent tools; projecting the per-call effective tool set on an agent-hooks `pre_model_call`. This port's MCP tools each own their `reqwest` client and carry their own headers, so there is no shared hook list to scope or to mutate mid-iteration; it depends on no provider SDK, so there are no SDK exceptions to wrap; and it has no agent-hooks/harness loop. |
| #7984, #8046, #8032, #7671, #7998, #7986, #7983, #3932, #8027, #7995, #7954, #8029 | **.NET only.** Foundry hosted-response streamed annotations, declarative workflow input serialization, A2A run modes and task-state tracking, `file_access_read_lines`/`AgentFileStore`, removing the `Azure.AI.OpenAI` and deprecated Bedrock MEAI dependencies, Cosmos emulator test reliability, and dependency bumps. The A2A pair lands on serving-side task lifecycle, already tracked as an open gap. |
| #8097, #8071, #8009, #8019, #7997, #8006, #7965, #8031 | **Packages with no Rust counterpart.** ChatKit, `lab`, the DevUI frontend, `foundry_hosting` (×3 — still the single most frequent source of "not applicable" here), and moving Foundry eval serialization out of core. |
| #8052, #7967, #8002, #7960 | **Dependency-floor moves.** FastAPI 0.141, OpenAI SDK 3.x, the GitHub Copilot SDK, and migrating Mistral onto the official SDK. This port speaks each provider's REST surface over `reqwest` and pins no provider SDK, so an SDK floor has no landing site. The Mistral one was read for wire-visible behavior alongside the SDK swap and carries none: the port's client continues to reuse `agent_framework_openai::convert::parse_response`. |
| #8003, #8011, #7980, #7776, #7906 | **AG-UI depth.** Surfacing workflow intermediate events as reasoning, stamping the checkpoint owner on every save, preserving parallel `function_result` contents, workflow-as-agent approval resumes, and Responses replay metadata across continuations. `agent-framework-hosting::agui` streams one run to completion and keeps no snapshot store, checkpoint owner, or continuation state, so each of these lands on machinery the router does not have — the standing AG-UI depth gap, unchanged. |
| #8065, #8085, #8062, #8084, #8090, #8101, #8102, #8076, #8077, #8074, #8043, #8072, #8073, #8070, #8059, #8066, #8067, #8060, #7582, #8037, #8036, #8047, #7876, #7777, #8038, #8033, #8035, #8004, #8007, #7937, #6046, #7883, #8030, #7972, #7935, #7680 | CI and dependabot configuration, action bumps, docs and sample additions, code owners, public-API analyzers, and the Python 1.17.0 / .NET 1.20.0 release bumps. |

### Also ported this pass: the five deltas this window opened

Each lands on a surface this port has, and each needed more than a
transcription. All five are built.

| Upstream | Change | Rust site |
|---|---|---|
| #7992 | **A provider refusal read as the answer.** The port parsed an OpenAI refusal into plain `TextContent` on both the Chat Completions and Responses paths — upstream's *previous* behavior. So `response.text()` handed back "I can't help with that" as though the model had answered, and `parse_json` would try to parse it as the requested JSON. `TextContent::refusal` now marks it (a typed `bool` rather than upstream's `additional_properties["model_output_kind"]` bag, since it is the only marker this port needs); `Message::text` returns `""` when one is present, with `has_refusal()` / `refusal_text()` to read it deliberately. Two consequences fell out. The Chat Completions parser used to emit the refusal *only when there was no content*, which hid a model that answered part of a request and declined part — both are now kept as separate items. And `coalesce_text` merged any two adjacent text items, so a streamed refusal following ordinary text folded into it under the first fragment's flag; the merge is now gated on the flag matching, which is the same boundary upstream splits on. | `core/types/content.rs`, `core/types/message.rs`, `core/types/response.rs` (`coalesce_text`), `openai/convert.rs`, `openai/responses.rs` |
| #7383/#7988 | **Two approvals pending under one provider `call_id` were indistinguishable.** Providers reuse `call_id`, which is harmless while a call is answered inside its turn but not across an approval round trip. The port matched approvals structurally (`call_id` + name + arguments), so two *simultaneously* pending approvals for the same call collapsed: the second request looked like a replay and was dropped, and one result answered both. `FunctionCallContent::id` is now a framework-generated occurrence id (`af-call-<uuid>`), minted at the one moment a call stops being answered within its turn — the approval deferral — and stamped on the call, its request, and the response's own copies so a replay carries it. `same_invocation` replaces the `==` comparisons in the invocation loop: occurrence ids decide when both sides have one, and it falls back to the structural rule otherwise, so approvals stored before this keep resolving (upstream's staged migration, without the deprecation warnings — nothing here has shipped a stored occurrence-less approval to warn about). Results are keyed by occurrence id with the same fallback. `merge` carries the id across streamed fragments for the same reason it carries `protected_data`. | `core/types/content.rs` (`id`, `ensure_occurrence_id`, `same_invocation`, `merge`), `core/client.rs` (approval deferral, `replace_approval_contents_with_results`) |
| #7912 | **Truncation could drop the request the conversation is about.** Upstream added `preserve_first_user_group` to its truncation strategy; the annotation-model halves of that commit (summary reconciliation across the middleware boundary) have no landing site here, since this port implements the reduced-list model and has no summarizing strategy. The portable half is built as `preserve_first_user()` on `Truncation`, `SlidingWindow` **and** `TokenBudget` — upstream offers it on one strategy, but all three drop the oldest turns and the exposure is identical in each; a caller's choice between them is about how to bound context, not about whether the opening request matters. Protected messages are kept regardless of budget, as upstream documents, so the result may exceed the limit by one. Off by default. This port has no group model, so it protects the earliest user *message*; a user turn carries no tool calls, so that cannot orphan a call/result pair. | `core/compaction.rs` (`first_user_index`, `push_preserved_first_user`, all three strategies) |
| #8014 | **Core vector-store abstractions**, a new upstream surface with no Rust counterpart. Built as `core::vectors`: `VectorStoreField` / `VectorStoreCollectionDefinition` (with validation at construction — no key, two keys, duplicate names, two fields renamed onto one storage name, a vector field without dimensions), `IndexKind` / `DistanceFunction` as open value wrappers carrying upstream's constants plus `higher_is_closer()` (upstream's `DISTANCE_FUNCTION_DIRECTION_HELPER` — get the direction wrong and a search returns the *worst* matches first), `VectorSearchOptions` / `VectorSearchResult`, the `VectorCollection` and `VectorStore` traits, and an `InMemoryVectorStore` for tests. Roughly half of upstream's 1,923-line module is a model layer — a decorator that registers a Python class, walks its annotations and generates encoders between instances and flat mappings — which `serde` makes redundant, so records are `serde_json::Value` objects keyed by field name and both traits stay object-safe. The one piece of that layer that *is* load-bearing is the logical-name/storage-name split, since it is a property of the store rather than of the Rust type: `to_storage` / `from_storage` do that renaming. Upstream's lambda-AST filter parsing has no Rust counterpart (there is no runtime AST), so a filter is the provider's own expression. | `core/vectors.rs` (new module) |
| #7976 | **Purview never asked for inline evaluation.** `process_inline` turns out not to be a body field at all: both upstream implementations translate it to a `Prefer: evaluateInline` request header, and set it when the protection-scope cache is cold. This crate holds no cache, so every request is that case — and it is the only case that can work here, because the middleware blocks a prompt or a response on the verdict in the reply, and an offline evaluation would leave it with nothing to decide on and silently turn enforcement into a no-op. Sent unconditionally as a constant rather than modelled as a field that could only hold one value. | `purview/client.rs` (`PREFER_EVALUATE_INLINE`), `purview/models.rs` (docs) |

Verified across all five: full workspace build, `cargo test --workspace
--all-features` (**1776 passing**, 48 of them new), `cargo clippy
--all-targets --all-features` under `-D warnings` clean, `cargo fmt --check`
clean. Each behavioral test was probed against the code it pins — degrading
`same_invocation` to the structural rule fails the two-pending-approvals test
and the identity test; keying results by `call_id` alone fails the
matched-results test; the two refusal tests that previously pinned
refusal-as-answer were rewritten and fail against the new parser only in their
old form; and the compaction and vector tests carry negative controls
(`preserve_first_user` off by default on every strategy, ordinary text still
coalescing, an unknown distance function refusing to guess a direction).

## Post-`e6d8d99` drift (checked against `d8d07eb`, 2026-08-29)

Upstream moved **20 non-merge commits** in this window (2026-08-26 → 08-29).
**One lands on this port**, in the Gemini finish-reason mapping — and it turned
out to be a wider gap than the upstream commit that surfaced it, because the
Rust map had been transcribed from a shorter version of upstream's than the one
that shipped. The other 19 are .NET, samples, dependency bumps, or
Python-shaped problems that cannot arise here — four of those because Rust's
ownership rules make the aliasing bug upstream fixed unrepresentable.

### Ported this pass (1 fix + 4 missing mappings, with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7837 | **Gemini dropped four finish-reason mappings and reported the proto default as a real reason.** Upstream's own bug was the unmapped-value fallback (`_FINISH_REASON_MAP.get(reason)` with no default), which this port never had — `map_finish_reason` has always ended in `other => FinishReason::new(other.to_lowercase())`, and `parse_stream_chunk` attaches `usageMetadata` unconditionally, so the usage-attach cascade upstream describes (usage rides along only when the finish reason is truthy) cannot fire here either. Both are now pinned rather than left implicit, following the #7850 precedent. What the commit *did* expose is that the port's map is a shorter transcription of upstream's: `LANGUAGE`, `IMAGE_PROHIBITED_CONTENT` and `IMAGE_RECITATION` (→ `content_filter`) and `MALFORMED_FUNCTION_CALL` / `UNEXPECTED_TOOL_CALL` (→ `tool_calls`) were falling through the passthrough arm as lowercased raw strings, so a caller matching on the canonical `content_filter` saw `"language"` instead. Separately, `FINISH_REASON_UNSPECIFIED` — proto3's "field never set" — was surfacing as a finish reason named `finish_reason_unspecified`; upstream maps it to `None`, and `map_finish_reason` now returns `Option<FinishReason>` so it (and the empty string) takes the absent path. That routes it through `finalize_finish_reason`'s `has_call` branch, so an unspecified turn that ends in a function call is upgraded to `tool_calls` exactly like a turn with no `finishReason` at all. | `gemini/convert.rs` (`map_finish_reason`, `finalize_finish_reason`) |

Verified: full workspace build, `cargo test --workspace --all-features`
(**1678 passing**, 3 of them new), `cargo clippy --all-targets --all-features`
under `-D warnings` (CI's own flag) clean, `cargo fmt --check` clean. The three
new tests were confirmed to fail against the prior behavior — the mapping and
`FINISH_REASON_UNSPECIFIED` tests against the old match arms, the streaming one
against a usage attachment gated on `finish_reason.is_some()` (upstream's own
cascade), probed on its own to confirm the usage assertion is load-bearing
rather than passing on the finish-reason change beside it. In each probe the
new tests were the only ones in the crate to fail.
`map_finish_reason_passes_unmapped_values_through` passes against the old code
by design: it pins behavior that was already correct.

### Not applicable (19)

| Upstream | Why not |
|---|---|
| #7847 | **Workflow checkpoints could be mutated outside their storage.** Python's `InMemoryCheckpointStorage.load`/`list`/`latest` handed back the stored object itself, and `State.get`/`set`/`export_state`/`import_state` passed values by reference, so a caller that mutated what it read silently rewrote persisted state; the fix deep-copies at every boundary. Neither aliasing is representable here. `CheckpointStorage::save` takes `WorkflowCheckpoint` **by value** and `load`/`list` return owned values cloned out of the map, and `SharedState::get` returns an owned `Option<Value>` while `set` takes `impl Into<Value>` — a `serde_json::Value` clone *is* the deep copy Python had to ask for. There is no shared handle for a caller to hold, so there is nothing to defend and no test that could fail. |
| #7903 | **`Content.__deepcopy__` now discards `_SHALLOW_COPY_FIELDS` instead of sharing them**, because those fields hold LLM SDK objects (proto/gRPC responses) that are neither safe to deep-copy nor safe to share between copies. This port's nearest fields, `raw_representation` and `protected_data`, are a `serde_json::Value` and a `String` — inert data that `derive(Clone)` copies correctly. Discarding them on clone would be actively wrong here: `raw_representation` carries the OpenAI Responses reasoning item that must be replayed verbatim on the follow-up tool-call turn, and `protected_data` the Gemini thought signature; a clone that dropped either would make the model reject the replayed turn. A deliberate divergence, and one the type system already makes safe. |
| #7901 | **`SerializationMixin.from_dict` mutated its caller's dict**, merging into a nested `dict` in place rather than building a new one. Deserialization here is `serde`: `from_dict`-shaped entry points take `&Value` or an owned `Value` and construct a fresh struct. There is no caller-owned mutable map to write through. |
| #7875 | **Azure AI Search declared every knowledge source as `searchIndex`**, so a mixed knowledge base failed retrieval; the fix reads each source's real `kind` back and sends `KnowledgeSourceParams` per source. Entirely inside the Knowledge-Base ("agentic") retrieval mode, which `agent-framework-azure-ai-search` does not implement — the crate ports the *semantic* mode only, a divergence documented at the top of `lib.rs`. |
| #7770 | **AG-UI service-session snapshot replay** — canonical-form comparison of stored vs. incoming snapshot messages, contiguous-overlap detection, unanswered-tool-call tracking, and a `service_session_id_from_thread_id` compatibility switch. All of it belongs to the Python package's thread-snapshot store and service-session machinery. `agent-framework-hosting::agui` has neither: it streams one run to completion and, as its module docs state, emits no `STATE_SNAPSHOT` / `STATE_DELTA` / `MESSAGES_SNAPSHOT` events and keeps no per-thread snapshot store. |
| #7908, #7911 | **A timeout for wait-for-first-completion** in the Python (`_harness/_background_agents.py`) and .NET background-agent hosts. This port has no harness agent loop and no background-agent host — the same reason #7289 and #7860 were not applicable in earlier passes. |
| #7896 | **.NET removes the retired OpenAI Assistants integration tests.** Already reflected here: `OpenAIAssistantsClient` was deleted in the `68136ee` re-baseline (see the `a63d462` section's predecessor below), for the same reason — upstream removed the API. Recorded rather than skipped because it is the last visible trace of a surface this port used to ship. |
| #7932, #7913, #7891 | .NET: a duplicate Foundry `AgentHost` port binding, a workflow file-input sample, and an A2A client-server sample simplification. No framework change reaching this port (and no Foundry hosted-agents host here). |
| #7924, #7921, #7915 | Python 1.16.0 version bumps, an agentserver dependency upgrade, and Python-wide code ownership. Release and repo hygiene. |
| #7889, #7888, #7886 | Dependency bumps (`Dapr.AI.Microsoft.Extensions`, `CommunityToolkit.VectorData.InMemory`, `Azure.AI.AgentServer.Invocations`). |
| #7869 | .NET contributing-docs update for CFS users. |
| (fork) | `Fix sync-upstream workflow: grant workflows:write permission` — a workflow-permission fix in the mirror fork itself, not an upstream framework change. |

Still deferred from earlier passes: **#7768** (pin GitHub Actions to
full-length commit SHAs) remains blocked for the same reason — resolving each
action's SHA means reading repositories outside this session's GitHub scope.

## Post-`a63d462` drift (checked against `e6d8d99`, 2026-08-26)

Upstream moved **17 non-merge commits** in this window (2026-08-24 → 08-26).
**None require a code change here.** Four are Python, one of which (#7850) is a
bug this port never had; the rest are .NET, samples, dependency bumps, or
Python-shaped problems that cannot arise here. The mirror is current — its last
`microsoft:main` merge is ~2h behind the newest commit triaged.

### Ported this pass (0 behavioral changes; 3 regression tests)

#7850 needed no fix, but it pins a behavior worth protecting, so it is recorded
as a test-only change following the precedent set by `7f4cc296` last pass
(correct by construction, pinned rather than left implicit).

| Upstream | Change | Rust site |
|---|---|---|
| #7850 | **Unmapped Anthropic and Mistral finish reasons were dropped.** Both Python clients looked their finish-reason map up without a default, so any provider value the map did not cover — Anthropic's `model_context_window_exceeded`, Mistral's `error` — reached the caller as *no finish reason at all*. This port has always passed unmapped values through: `map_stop_reason` ends in `other => FinishReason::new(other)` and is the single resolution site for both the buffered and streaming Anthropic paths, while the Mistral client reuses `agent_framework_openai::convert::parse_response`, which builds the reason straight from the wire string. `FinishReason` is an open string enum, so there was nothing to widen. Three regression tests now pin it, because in two of the three sites the behavior is *inherited* rather than written locally, and upstream's own bug is exactly what a later "tidy the match into a lookup table" refactor would reintroduce. | `anthropic/convert.rs` (`map_stop_reason_passes_unmapped_values_through`), `anthropic/lib.rs` (`stream_passes_an_unmapped_stop_reason_through`), `mistral/convert.rs` (`parse_response_passes_an_unmapped_finish_reason_through`) |

Verified: full workspace build, `cargo test --workspace --all-features`
(**1663 passing**, 3 of them new), `cargo clippy --all-targets --all-features`
under `-D warnings` (CI's own flag) clean, `cargo fmt --check` clean. Each new
test was confirmed to fail against the lossy behavior — the Anthropic pair
against a fallback arm rewritten to collapse unknown values, the Mistral one
against a lookup-without-default in the shared OpenAI parser — and each was the
*only* test in its crate to fail, so they pin the passthrough specifically and
not the documented mappings alongside it.

### Not applicable (16)

| Upstream | Why not |
|---|---|
| #7860 | **The harness agent-loop marker leaked into provider options.** `RawAgent` now pops the loop-iteration key off the merged options so it stays on `SessionContext.options` for `after_run` provider scoping without reaching the client. This port has no harness agent loop at all — the same reason #7289 was not applicable last pass — so there is no marker to leak. |
| #7703 | **Programmatic OTel service name, resource attributes, and OTLP exporter config**, added as keyword arguments to `configure_otel_providers()` / `ObservabilitySettings`. That entry point configures the OpenTelemetry **SDK**: tracer/meter providers and OTLP exporters. This port deliberately ships none of that — it emits `tracing` spans the application bridges itself (e.g. via `tracing-opentelemetry`) and touches the `opentelemetry` **API** crate only for its two metrics instruments, with `opentelemetry_sdk` a test-only dependency. There is no provider-setup function here to add parameters to. A documented divergence, not a gap. |
| #7705 | **Streaming broke when the Azure GenAI instrumentor replaced the raw response.** The instrumentor substitutes an object exposing neither `.parse()` nor `.headers` for the OpenAI Python SDK's raw-response wrapper, so `_open_event_stream()` must probe `.parse` with `callable()` and parse the *inner* unparsed response the wrapper holds. Both halves are Python-SDK object-graph plumbing: this port speaks HTTP directly over `reqwest` and parses SSE frames itself, so there is no SDK response wrapper for an instrumentor to swap out. |
| #7846 | **ChatKit attachment handling** — `python/samples/05-end-to-end/chatkit-integration/` only. Samples, and there is no ChatKit crate (a tracked ecosystem gap). |
| #5860 | **.NET preserves the Responses `logprobs` field**, adding it to `CreateResponse`/`Response` and echoing the request's value onto the response. Those are `Microsoft.Agents.AI.Hosting.OpenAI`'s full, stateful Responses-service models. This port's `hosting::responses` mirrors Python's `hosting-responses` *conversion* surface instead: `ResponsesRequest` is a documented subset (`model`/`input`/`stream`/`metadata`/`extra_body`) modelling no sampling parameters at all, and `ResponseObject` echoes none of the request-parameter fields (`instructions`, `max_output_tokens`, `max_tool_calls`, `top_logprobs`, …). Adding `logprobs` alone would be arbitrary; the request-echo surface as a whole, and the stateful service (`InMemoryResponsesService`) it belongs to, are out of scope — hosted runs here are stateless by design. |
| #7843, #7861, #7792 | .NET **samples** only (verified against each commit's file list): AG-UI hosted web search moved onto the Responses API, A2A function-tool sample simplification, and a Mem0Sharp in-memory sample. No framework change. |
| #7842, #7817 | .NET Foundry hosted-agent response cancellation, and recovery-test stabilization. This port has no Foundry hosted-agents host. |
| #7864, #7858 | .NET static-analysis annotations — DevUI aggregator and Zip Slip false positives. |
| #7878, #7870, #7826, #7868 | .NET package rename (`CommunityToolkit.VectorData.CosmosNoSql` → `AzureCosmosDB`), an ASP.NET OpenAPI dependency upgrade, an `Aspire.Hosting` bump, and a dependabot cooldown setting. Dependency and repo hygiene. |

Still deferred from last pass: **#7768** (pin GitHub Actions to full-length
commit SHAs) remains blocked for the same reason — resolving each action's SHA
means reading repositories outside this session's GitHub scope.

## Post-`e1326eb` drift (checked against `a63d462`, 2026-08-24)

The `e1326eb` pass below was triaged against a fork mirror that had stopped
advancing on 2026-08-20; the sync then caught up and moved **31 further
non-merge commits** (through 2026-08-24). One lands on this port.

### Ported this pass (1, with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7673 | **The GenAI semantic-convention version was never selectable, and the provider tag was emitted twice.** `gen_ai.system` was renamed to `gen_ai.provider.name` above the OTel v1.36.0 baseline. This port emitted **both** names on every chat span (and `gen_ai.provider.name` on the metrics attributes), so a consumer pinned to the baseline saw an attribute its version does not define, and one on the latest saw a name that had been renamed away. Upstream's fix makes the version an explicit input: `OTEL_SEMCONV_STABILITY_OPT_IN`, a comma-separated opt-in list in OpenTelemetry's standard format, whose `gen_ai_latest_experimental` token selects the conventions above the baseline — defaulting, when unset, to *opted in*, which upstream documents as a deliberate departure from OpenTelemetry's own default. `ObservabilityConfig` now carries that value and derives `use_latest_experimental_gen_ai_semconv()` / `emit_tool_call_attributes()`; exactly one provider attribute is emitted, and the four above-baseline attributes (`cache_creation.input_tokens`, `cache_read.input_tokens`, `reasoning.output_tokens`, `tool.definitions`) plus `gen_ai.tool.call.arguments`/`result` are withheld at the baseline. Under the default the only visible change is that `gen_ai.system` no longer rides along beside `gen_ai.provider.name`. | `core/observability.rs` (`GEN_AI_LATEST_EXPERIMENTAL_OPT_IN`, `ObservabilityConfig`, `chat_span`, `record_request`, `record_response`, `record_tool_arguments`, `record_tool_result`, `ObservableChatClient`, `metrics::record_chat_completion`), `core/client.rs` (tool-span call sites) |

The recording functions took a `capture_content: bool`; they now take
`&ObservabilityConfig`, because the second gate is not a property of the call
site. That keeps both gates explicit and, unlike reading the environment inside
the recorders, leaves them free of hidden global state on a per-response path.
`ObservableChatClient::with_content_capture` still works and sets the flag on
the config; `with_observability_config` sets both.

Verified: full workspace build, `cargo test --workspace --all-features`
(**1645 passing**, 4 of them new), `cargo clippy --all-targets --all-features`
under `-D warnings` (CI's own flag) clean, `cargo fmt --check` clean. All four
new tests were confirmed to fail against ungated code.

### Not applicable (30)

| Upstream | Why not |
|---|---|
| #7799, #7801 | **MCP tool argument shadowing the remote tool name**, and its documentation follow-up. Python's generated MCP function held the remote tool name as a keyword-only parameter *default*, and model-supplied arguments are splatted into that function — so an argument named `_remote_tool_name` bound to the parameter and redirected the call to a different remote tool. This port's MCP tools capture the remote name in the tool struct and pass arguments as one `serde_json::Value` map to `call_tool(name, arguments)`; there is no splat and no parameter for an argument to bind to. |
| #7289 | **Turn-scoped `after_run` providers deferred to the agent-loop boundary.** Each `AgentLoopMiddleware` iteration is a full agent run, so `CompactionProvider.after_run` fired per iteration and rewrote persisted history mid-task; providers can now opt into once-per-turn semantics. The port has no harness agent loop (a documented "remaining" item), so nothing drives several runs inside one turn and there is no per-iteration re-fire to defer. |
| #7625 | **GitHub Copilot telemetry config forwarding**, plus the Python settings-machinery fixes it needed (parameterized generics and `Literal` arms in runtime annotation checks). Both halves are Python-shaped: this port's `agent-framework-github-copilot` targets the OpenAI-compatible chat endpoint rather than the Copilot Agent SDK's session API — the same reason #7155 and #7300 were not applicable — and its settings are typed struct fields, not runtime-inspected annotations. |
| #7779 | **DevUI forwards `function_invocation_kwargs` to `agent.run`.** No `function_invocation_kwargs` concept here: tools receive a typed `Value` plus a `FunctionInvocationContext`. |
| #7734 | **FoundryEvals always emits an `arguments` field for tool calls**, in `_evaluation.py`. No evaluation crate. |
| #7423 | **A2UI (Agent-to-UI) support in the AG-UI adapter.** No AG-UI crate. |
| #7670, #7370, #7649 | **Foundry hosted-agent resiliency, steerable hosted agents, hosted state persistence** (Python and .NET). This port has no Foundry hosted-agents host. |
| #7768 | **Pin GitHub Actions to full-length commit SHAs**, across upstream's own 24 workflow files. This is repo hygiene rather than framework parity; worth adopting for this repo's two workflows, but resolving each action's SHA means reading repositories outside this session's GitHub scope, so it is left as a follow-up rather than guessed at. |
| #7812, #7814, #7795, #7813, #7804, #7754, #7678 | Release version bumps (Python 1.15.0, .NET 1.19.0), release-tag resolution, a DevFlow command fix, codeowners, README/doc edits. |
| #7774, #6441, #1893, #7709, #7639, #7778, #7829 | .NET: the MCP long-running-task migration to the 2026-07-28 Tasks extension, GitHub Copilot `ReasoningSummary` passthrough, Azure Blob Storage session persistence, a feature-usage bitmask, and dependency bumps. |
| #7780, #7781, #7782, #7783, #7784 | Python tooling bumps (`uv`, `ruff`, `ty`, `mypy`, `flit-core`). |

## Post-`5c06755` drift (checked against `e1326eb`, 2026-08-20)

Upstream moved **38 non-merge commits** in this window (2026-08-16 → 08-20).
Two land on this port: a bug it shares with upstream, and a capability its
middleware contract lacks. The other 36 are .NET, AG-UI, samples, dependency
bumps, or Python-shaped problems that cannot arise here — several of those
because the port's payloads are untyped `serde_json::Value`s rather than
dynamically resolved Python types.

### Ported this pass (1 fix + 1 capability, both with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7242 | **Replaying a conversation duplicated stored history.** A history provider is handed a run's input messages plus its response messages, so a caller that keeps its own transcript and replays all of it every turn (the AG-UI shape, and any client tracking history itself) hands back everything the provider already stored. Every provider appended it unconditionally, so history grew superlinearly — and because `before_run` prepends stored history to the request, the duplicated turns were resent to the model on every later run. `filter_new_messages` locates the stored run inside the incoming one and returns only what follows it. Matching is by `message_id` where a message has one and by role + contents where it does not, mirroring upstream's `get_message_identity`; the scan is not anchored at offset 0, so a provider whose stored history is a *trimmed window* (a retention limit having dropped the oldest messages) still aligns. Applied to all four history providers, not just the two upstream touched: `RedisChatMessageStore` and `CosmosChatMessageStore` have the identical shape and the identical bug, and now read their stored history before writing — one extra round trip per run, the same read `before_run` already makes. | `core/history.rs` (`filter_new_messages`, both providers' `after_run`), `redis/chat_message_store.rs`, `cosmos/chat_message_store.rs` |
| #7562 | **Function middleware had no way to fail closed.** The invocation loop converts every error a tool or its middleware produces into a `FunctionResultContent { exception, .. }`, hands it to the model and keeps looping. That is right for a tool failure the model can recover from, but an enforcement layer — a guardrail, a policy or authorization gate — needs the opposite: when it refuses a call, the run must stop, not hand the model an error string it can retry around. `Error::MiddlewareFailure` is the escape, and the only error the loop propagates rather than absorbs. Because the parallel batch runs under `try_join_all`, propagating it also drops the siblings still in flight — upstream's "cancel the in-flight batch" without needing a cancellation mechanism of its own. | `core/error.rs` (`MiddlewareFailure`, `middleware_failure`, `is_middleware_failure`), `core/client.rs` (`execute_tool_call`), `core/observability.rs` (`error_type`) |

Two review findings on PR #16 extended the fix past what upstream's own
patch covers, both confirmed by probe before fixing:

- **Alignment must not see response messages.** Concatenating a run's input and
  its responses before aligning let a response that coincidentally reproduced
  the stored tail match as a replay, swallowing the genuinely new input in
  front of it — stored `[q, a]` plus a run whose input is `q` and whose
  response opens with `a` stored nothing but the tail. Responses are generated
  by the run reporting them and can never be a replay, so `new_run_messages`
  aligns the input alone and appends every response.
- **The request side duplicated too.** Storing only the new suffix left the
  other half of the problem untouched: the agent sends injected context
  followed by the caller's input, so a provider that injected unconditionally
  sent `q1, a1, q1, a1, q2` for a caller replaying `q1, a1, q2` — verified end
  to end against the messages the model actually received. All four providers
  now inject nothing when the input already aligns against what they hold; the
  first fix covered only the two core ones, and a third review round caught
  that the Redis and Cosmos stores still had the unconditional injection.

A second review round raised two more, both fixed:

- **Which occurrence of a stored run to align on depends on what the provider
  holds.** A forward scan takes the first match, which is right for a store
  that keeps everything — a later match would discard the genuinely new turns
  in between — but wrong for a retention-limited one, whose stored list is a
  *window* of the most recent messages: there the last match is the window, and
  taking an earlier one re-pushes the whole middle of a replayed transcript on
  every turn. `StoredHistory::{Complete, Window}` makes it the caller's
  decision, and the Redis store picks `Window` only when its list is *at* its
  cap, since only a trimmed list can be a window. A fourth round then caught
  that a complete history must be matched at the **start** and nowhere else: it
  begins at the conversation's first message, so a replay can only begin with
  it, and a coincidental match further in silently dropped every genuinely new
  message in front of it — stored `[yes]` against an input of
  `[preface, yes, question]` kept only `question`, and `before_run` then
  injected nothing, so the stored turn never reached the model either. The
  search remains for `Window`, but prefers an anchored match there too: a fifth
  round pointed out that a list merely *reaching* its cap has not necessarily
  been trimmed — a first write that filled it exactly is still the complete
  conversation — and searching such a list loses the turns between two
  occurrences of it. The ambiguity that leaves (a genuine window whose content
  also opens the transcript) resolves to re-sending the middle, which the trim
  discards, rather than to a dropped turn, which is not recoverable. A sixth
  round then reached the ambiguity underneath all of this: content alone cannot
  tell a replayed transcript from new input that repeats it, so alignment now
  requires *evidence* — a matching message id, or a non-user turn in the stored
  block, since a replay is a transcript and carries the assistant's replies
  while new input carries only the caller's own. Stored history that is nothing
  but id-less user messages is left alone.
- **A configured semconv version has to reach tool spans.** The tool loop
  rebuilt an `ObservabilityConfig` from the environment per call, so a client
  configured for one convention version could emit chat spans under it and tool
  spans under another. `FunctionInvokingChatClient` now carries the config
  (`with_observability_config`), resolved once at construction, and
  `AgentBuilder::observability_config` reaches that wrapper — the builder
  constructs it internally, so without a way through it the setter was
  unreachable on the main path.
- **The injection fix had to reach the remote stores.** It landed in the two
  core providers and stopped there, and the end-to-end test covered only the
  in-memory one, so a replay through a Redis- or Cosmos-backed session was
  still sent to the model twice. `inject_stored_history` is now public and used
  by all four, with a `StoredHistory`-aware variant for the Redis store.

The remote stores also no longer read before writing when there is nothing to
write (an empty run, or a Redis store configured to retain nothing, which is
documented to leave Redis untouched); both cases are pinned by pointing a store
at an address nothing is listening on and asserting the run still succeeds.
Their read-then-write sequence is not atomic, and the Cosmos read can lag a
just-landed write on an account with session or eventual consistency; both are
documented at the call sites rather than closed, since the fallback in each
case is the duplicate that the blind append they replace produced every time.

Two deliberate divergences in the dedup, both refusing to drop a turn that
might be real. Upstream falls back, when alignment fails, to deduplicating by
identity against a set of everything stored; that collapses two identical,
id-less `"yes"` turns into one and loses the second permanently. And upstream
treats an alignment consuming *all* of the incoming run — a run whose messages
exactly repeat the stored tail — as a replay carrying nothing new, storing
nothing; since a provider only reaches `after_run` by completing a real run,
this port reads it as a turn that genuinely repeated itself and stores it. In
both cases the port's behavior is what it was before the fix (append), so
neither can regress a conversation that used to be stored correctly.

The fail-closed signal is carried by the error *type* rather than by who
produced it, which is the one place it is looser than upstream's exception
class: a tool executor returning `Error::MiddlewareFailure` propagates the same
way. That is documented on the variant rather than guarded against — the
alternative (a marker only the pipeline can set) would need a wrapper type
threaded through every middleware signature for no practical gain.

Verified: full workspace build, `cargo test --workspace --all-features`
(**1659 passing**, 28 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean. The two Redis tests run against a real
`redis-server` spawned by the existing integration harness; the Cosmos test
asserts the write count on the loopback server, so it fails if a replayed
message is written a second time. Both fixes were probed against the code they
fix: 5 of the history tests and both fail-closed tests fail without them (the
fail-closed pair by hanging on the 30-second sibling call, which is the
cancellation the fix buys). The remaining tests are negative controls — an
append-only run still accumulates every turn, an unalignable run is still
stored whole, and an ordinary middleware error is still absorbed into a
tool-error result and the loop still continues.

### Not applicable (36)

Grouped by why, rather than one row each:

| Upstream | Why not |
|---|---|
| #7684, #7500, #7636 | **Python type resolution.** Coercing JSON workflow-resume payloads into declared annotations, restricting request-info type-name resolution to caller-provided mappings, and a global checkpoint type registry. All three exist because Python resolves a payload's type *by name* at runtime. This port's `PendingRequest.request_data` and `RequestResponse.data` are `serde_json::Value`; there is no declared response type to coerce to, no type name in the payload to resolve, and no import to restrict — the executor deserializes what it asked for. |
| #7730 | **Structured instructions coerced to their `repr` when merged.** Upstream's `instructions` is declared `str` but widened by some clients to provider-native structured blocks, and three merge paths joined it with an f-string. `ChatOptions::instructions` is `Option<String>` here, so there is no non-string value to stringify; the newline concatenation in `ChatOptions::merge` and `prepare_request` is correct for every value the type admits. |
| #7755 | **`HandoffBuilder` clones dropped `Agent.additional_properties`.** Upstream's `HandoffAgentExecutor` rebuilds each participant agent to attach handoff tools. This port's `HandoffBuilder` holds `Arc<dyn SupportsAgentRun>` participants and never rebuilds them, and `Agent` carries no `additional_properties` field to lose. |
| #7557 | **Fan-in dropped all but the first trace context.** Upstream's workflow messages carry `trace_contexts` / `source_span_ids` lists for distributed-trace linking across a fan-in. This port's workflow engine propagates no trace context on messages at all — a standing gap in workflow observability, not a bug in aggregation, and one this commit does not close. |
| #7761 | **A2A input handling in orchestrations.** Three-part change: reject an empty A2A invocation explicitly, translate a remote `INPUT_REQUIRED` task into the `user_input_request` content contract so a group chat pauses on it, and restore that pending input from a checkpoint. The first half is already satisfied: `A2AAgent::run` errors on an empty `messages` list rather than inventing input (upstream was raising a bare `ValueError` and now raises `AgentInvalidRequestException` with session context; this port's message is already specific). The rest is blocked — the port's `A2AAgent` surfaces an `INPUT_REQUIRED` task's status message as ordinary chat messages, and there is no `user_input_request` content classification to translate the task into, so pausing an orchestration on remote input is a design task (tracked below), not a port. The two core-workflow hunks in this commit ride on the same classification. |
| #7766, #7510, #7662 | **AG-UI.** Unchanged predictive-state snapshots, tool-message IDs across snapshots, run continuity. No AG-UI crate. |
| #7606 | **A2A preview consent URLs**, in `foundry_hosting`. This port has no Foundry hosted-agents host. |
| #7698, #7695, #7693, #7706, #7746, #7740, #7762 | Harness blog samples, skill-script argument guidance, docs link fixes, spec/review-process guidance, engineering-system metadata, codeowners. |
| #7722, #7741, #7764, #7742, #7564, #7668, #7295, #7737, #7731, #7641, #7721, #7713, #7674, #7412, #7648, #7650 | .NET: A2A streaming artifacts, AG-UI history and SDK bumps, agent-hooks interception (the .NET half of #7515, already tracked as open), Foundry hosted samples and identity pass-through, `IServiceProvider` overloads, harness tool descriptions, session-persisted routing, release/build/version chores, declarative samples, Cosmos chat-history retrieval, and opt-in concurrent tool invocation (this port's loop is concurrent by default). |
| #7644, #7645 | Dependency bumps confined to Python tooling (`ty`, `flit`). |

### Standing gaps, reconfirmed (not closed)

- **Workflow trace propagation.** #7557 is the first upstream commit in this
  window to touch machinery — per-message `trace_contexts` carried across
  edges and merged at a fan-in — that the port's workflow engine does not have
  at all. Agent and tool spans are instrumented; workflow message flow is not.
- **`user_input_request` content classification.** #7761 needs an A2A
  `INPUT_REQUIRED` task to become a content item an orchestration recognizes
  as a request for caller input. The port has request-info events but no
  content-level classification for them, so an A2A participant cannot pause a
  group chat for remote input. Left open rather than half-built.

## Post-`2eb8fbb` drift (checked against `5c06755`, 2026-08-16)

Upstream moved **29 non-merge commits** in this window (2026-08-10 → 08-16),
the largest batch since the daily sync was repaired. Two are real bugs this
port shared and has now fixed; one is a mapping the port already had, now
pinned by a test; the rest are not applicable, and three of those land on
subsystems already tracked as open gaps.

### Ported this pass (2 fixes + 1 pinned mapping, all with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7470 | **A Redis retention limit of zero retained everything.** The documented sentinel for *unlimited* is not calling `with_max_messages` at all, so `0` must retain nothing — it retained every message instead. Trimming to `-(max)` emits `LTRIM key 0 -1` for `max == 0`, which is Redis's "keep the whole list", while the `len > max` guard is true for any non-empty list: the trim ran on every save and did nothing. `add_messages` now short-circuits *before* serializing, so no payload reaches Redis — or an AOF or replica — even briefly. It deliberately does **not** delete the key: `redis_key` is `{key_prefix}:{session_id}` with no per-provider discriminator, so two stores sharing a prefix and session id address the same list, and deleting would drop a co-located store's just-written history. Removing stored history is what `clear` is for. Upstream's other half — rejecting a *negative* limit, which emitted `LTRIM key 5 -1` and deleted the five oldest messages on every save — cannot arise here: `max_messages` is a `usize`. | `redis/chat_message_store.rs` (`add_messages`, `with_max_messages`) |
| #7546 | **Gemini 3 thought signatures were dropped across an approval round trip.** Gemini 3 rejects a request whose `functionCall` parts lack the `thoughtSignature` they were issued with. This port paired a signature to its call by *adjacency* only — a reasoning carrier immediately preceding the call — and cleared the held signature on any intervening content. Two failures followed, both ending in a 400 on the next turn: content that emits no Part at all (a `FunctionApprovalResponse`) cleared the signature merely by sitting between the carrier and its call; and a call replayed in a later message, with no carrier beside it, could never be signed. The clear now happens only for content that actually reaches the wire, and a `call_id -> signature` map accumulated across the conversation is the final fallback, which is what lets a later replay find its signature. Precedence is unchanged and matches upstream: the call's own `protected_data` wins, then an adjacent carrier, then the map. | `gemini/convert.rs` (`message_contents_to_parts`, `messages_to_gemini`) |
| #7597 | **Mistral prompt-cache usage** — already mapped, now asserted. Upstream's Mistral package hand-rolls its own usage mapping and had to grow `prompt_tokens_details.cached_tokens`; this port's `parse_response` delegates to the OpenAI parser, whose usage handling already covers it, so the field was never dropped. The delegation is the only reason there is no bug, so it is now a tested contract rather than an inherited accident. Upstream's explicit `isinstance(..., int) and not isinstance(..., bool)` guard has no counterpart here — `Value::as_u64` rejects strings, floats, and bools structurally — which is also pinned. | `mistral/convert.rs` (tests only; no behavior change) |

The Gemini fix diverges from upstream deliberately. Upstream caches signatures
in a bounded (256-entry, LRU) map on the *client*, which spans conversations
and therefore needs eviction and a `max_tracked_thought_signatures` knob.
Scoping the map to the conversation being converted is naturally bounded by the
history resent on every stateless request, needs no eviction policy, and cannot
leak a signature between conversations. The tradeoff, recorded rather than
hidden: a signature whose carrier has been dropped from history entirely is not
recoverable here, where upstream's client-lifetime cache would still hold it.

The map is written by the emit walk itself rather than gathered by a pre-pass.
The first cut used a pre-pass shaped like `collect_call_names`, and PR #15
review caught the flaw: a pre-pass has to *restate* the pairing rules, and that
restatement omitted the clear on wire-visible content. Because the map is
consulted only after adjacency has deliberately declined to sign a call, the
laxer map silently overrode that decision and re-signed calls the converter had
just refused — `[reasoning(sig), text, call]` reached the wire signed. Since a
replayed call always follows the turn that issued it, one forward pass suffices,
so the rules now have exactly one implementation and cannot drift apart.

Verified: full workspace build, `cargo test --workspace --all-features`
(**1627 passing**, 10 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean. The two Redis tests run against a real
`redis-server` spawned by the existing integration harness. All six
bug-fixing tests were confirmed to fail against the code they fix — including
the two covering the review finding, checked against a reinstated lax map; the
remaining four are negative controls (an unsigned call stays unsigned, one
carrier never signs a second call, non-integer cached tokens are ignored) and
pass either way by design.

### Not applicable (26)

Grouped by why, rather than one row each:

| Upstream | Why not |
|---|---|
| #7486, #7608, #7533-adjacent FHA work | **Foundry hosted-agents host.** `_OutputItemTracker` duplicate-call suppression, FHA session-id translation. This port has no `foundry_hosting` equivalent — `agent-framework-foundry` is the client only. |
| #7655, #7594, #6646 | **AG-UI.** URL-source attachments, approval lifecycle/resume hardening, workflow checkpointing in the AG-UI adapter. No AG-UI crate. |
| #7652 | **DevUI frontend** streamed-tool-call dedup, in the bundled web UI this port does not ship. |
| #7622 | **MCP archive rejection warnings.** Raises `debug` to `warning` in `_ArchiveEntryLoader`, part of the file-based skill discovery subsystem this port lacks entirely (see the standing gap below). |
| #7631, #7607 | **Approval storage and approve-for-session scoping.** Both build on an `AgentSessionStateBag`-backed permission store this port has no equivalent of; related to the standing declaration-only gap. |
| #7521 | **[BREAKING] Require building functional workflow instances.** Python's `@executor`-decorated functions must now be built into instances before use — a Python-decorator ergonomics constraint. This port's `WorkflowBuilder` already requires constructed executors; there is no unbuilt form to reject. |
| #7550 | **JSON parsing for declarative workflows** — the Power-Platform-style declarative *workflow* DSL, already tracked as a deliberate divergence. |
| #7635 | **Cosmos memory provider** calling a renamed `add_cosmos` toolkit API — a Python-package rename with no Rust counterpart. |
| #7404 | **ClaudeAgent SDK client reuse** — the `agent-framework-claude` subprocess shim, a standing roadmap item. |
| #7450, #7602 | **BackgroundAgentsProvider `release_session`** (Python and .NET) — no background-agents provider in this port. |
| #7509, #7558, #7621, #7661, #7660, #7646, #7666, #7572, #7612, #7609, #7552 | Workspace glob matching, feature-usage telemetry, agentserver/package version bumps, code-owner enforcement, nuget config, .NET sample style, and .NET-only hosting/telemetry/diagnostics changes. |
| #7529, #7493, #7541, #7554, #7545 | Dependency bumps confined to Python tooling and the DevUI frontend (postcss ×2, pyrefly, js-yaml, zuban). |

### Standing gaps, reconfirmed (not closed)

This window added evidence to two gaps already on the books, and neither was
half-implemented to improve the table:

- **File-based skills.** #7622 is the third upstream commit in a row
  (after #7540 and #7507) to harden a discovery walk this port does not have:
  `SkillsProvider` builds `Skill` values in memory from caller-supplied
  strings. Adding a file source means adopting the whole security boundary —
  path traversal, symlink escape, archive size and format limits — not just a
  directory walk.
- **Declaration-only sibling calls.** #7631 and #7607 both extend the session
  state bag and approval-response binding that upstream's #7388 workaround
  depends on. That machinery is still absent here, so the gap widened rather
  than closed.

## Post-`266206e` drift (checked against `2eb8fbb`, 2026-08-11)

Upstream moved **8 commits** in this window — the first batch to arrive after
the fork's daily sync was repaired (it had failed on every run since it was
added, so the three preceding passes were all triaged against a mirror that
had stopped advancing on 2026-08-06).

**Nothing in the batch needed porting.** Five are dependency bumps confined to
Python tooling and the DevUI frontend (postcss ×2, pyrefly, js-yaml, zuban).
The three substantive commits are all Python and all land on subsystems this
port does not have:

| Upstream | Why not applicable |
|---|---|
| #7550 | **JSON parsing for declarative workflows.** Rewrites how an agent's free-text output is coerced to JSON — fenced-block extraction, then a scan for the last decodable object in prose — inside `_executors_agents.py`. That file belongs to the Power-Platform-style declarative *workflow* DSL, where an action captures an agent's output into a typed variable. This port's declarative crate is a spec model that compiles YAML into a `WorkflowBuilder` graph and passes conversation messages between agent nodes; it never JSON-decodes agent prose. Already tracked as an open roadmap item ("the upstream Copilot-Studio declarative *workflow* DSL"). |
| #7533 | **FHA migrated to `responses==2.0.0b1`, plus a Foundry state store.** Confined to the `foundry_hosting` package. This port has no Foundry hosted-agents host — `agent-framework-foundry` is the persistent-agents *client* only. |
| #7536 | **Encrypted reasoning made opt-in for Foundry chat.** Strips `reasoning.encrypted_content` from the `include` that upstream's base Responses client adds implicitly. Not applicable as written — but verifying *why* surfaced a real gap in the opposite direction, fixed below. |

### Ported this pass (1, with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7536 (inverted) | **Stateless Responses requests never asked for the encrypted reasoning item.** Upstream's Responses client appends `reasoning.encrypted_content` to `include` whenever a request carries no service-side-storage indicator (`_chat_client.py:1414-1418`); #7536 is Foundry opting *out* of that default. This port set `include` nowhere at all, so it never opted *in*. That quietly defeated machinery it already had: `messages_to_input` re-emits a reasoning item verbatim for a `store: false` tool-loop replay, and drops one that lacks `id`/`encrypted_content` as having "no valid input form" — but the item could never carry `encrypted_content`, because the request never asked for it. `responses_include` now builds the array once, shared by both Responses clients, and Foundry turns the implicit add off via `AzureOpenAIResponsesClient::without_implicit_encrypted_reasoning`, which is #7536's behavior. | `openai/responses.rs` (`responses_include`, `ENCRYPTED_REASONING_INCLUDE`, `build_body`), `azure/responses.rs` (`build_body`, the new builder), `foundry/lib.rs` (both constructors) |

Semantics mirror upstream exactly: a caller's own `include` entries are always
preserved; an explicitly named `reasoning.encrypted_content` is honored even
with the implicit add disabled (the switch governs only what is added
unprompted) and is never duplicated; the trigger is the service-side-storage
indicator rather than `store`; and an empty `include` is omitted rather than
sent as `[]`.

Verified: full workspace build, `cargo test --workspace --all-features`
(**1614 passing**, 8 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean. The Foundry opt-out is asserted on the real
outbound body through the hermetic loopback server, not on the flag, and was
confirmed to fail without the wiring.

## Post-`4b1afd90` drift (checked against `266206e`, 2026-08-07)

Upstream moved **6 commits** after the `4b1afd90` baseline. All six are .NET;
no Python commit landed in this window. One was ported, five are not
applicable — and two of those five mark subsystems this port simply does not
have, which is recorded here as a gap rather than dressed up as parity.

### Ported this pass (1, with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #7539 | **Usage aggregation across a looping run.** Upstream's invariant: when a component re-invokes an inner agent or chat client several times within one logical run, the usage it returns must cover the whole run. This port's function-invocation loop violated it — every exit path returned the final iteration's `ChatResponse` untouched, so a five-iteration tool run reported roughly a fifth of the tokens it spent, and the `gen_ai.usage.*` OTel metrics (which read `usage_details`) under-reported with it. Confirmed by probe before fixing: a two-call run reporting 100 then 200 input tokens surfaced 200. `accumulate_usage` now folds each iteration's usage into a running aggregate applied to whichever response the loop returns — the no-more-calls exit, the approval pause, the declaration-only hand-back, and the tools-disabled failsafe. `UsageDetails::add_assign` already carried the null-aware semantics upstream's `UsageAggregator.Combine` specifies, so `None` still means *not reported* rather than zero. | `core/client.rs` (`accumulate_usage`, the `get_response` tool loop) |

Verified: full workspace build, `cargo test --workspace --all-features`
(**1606 passing**, 3 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean. All three new tests were confirmed to fail
against the pre-fix loop, so none of them is vacuous.

### Not applicable (5)

| Upstream | Why not |
|---|---|
| #7535 | **Declarative fenced-string parsing.** Replaces a backtracking regex in `TrimJsonDelimiter` with a linear scan, to deny a malformed fenced input the chance to trigger catastrophic backtracking. This port pulls in no regex engine at all (`regex` appears nowhere in the tree) and has no fenced-code-block trimmer — the declarative crate parses YAML and conditions directly. There is nothing here to harden. |
| #7567 | **`AgentIsolationKeyProvider` rename.** A .NET hosting rename across A2A task stores, AG-UI endpoints, and session stores. This port has no isolation-key concept in `agent-framework-hosting`. |
| #7525 | **Single source of conversation history for a hosted agent.** Rewires .NET's `AgentSessionStore` / `AgentFrameworkResponseHandler` for Foundry hosting. This port's hosting crate has no `AgentSessionStore` equivalent. |
| #7540 | **Hardened file skill discovery.** Skips symlinked `SKILL.md` files and symlinked subdirectories during discovery, and fails closed on paths it cannot inspect. Not applicable *because the port has no filesystem skill source at all* — `skills.rs` builds `Skill` values in memory from caller-supplied strings, so there is no discovery walk to harden. See the gap note below. |
| #7388 | **`InvocableFunctionBypassingChatClient`.** See the gap note below. |

### Gaps this pass surfaced (not closed)

Two upstream changes landed on subsystems this port lacks. Neither is a
regression, and neither was half-implemented to make the table look better:

- **File-based skills.** Upstream (both languages) discovers skills from disk:
  `SKILL.md` with YAML frontmatter, resource and script files found by
  scanning the skill directory, and a security boundary around path traversal
  and symlink escape. This port's `SkillsProvider` is in-memory only, so
  #7540 and the earlier #7507 (Windows junction detection) have no landing
  site. Adding a file source means adopting that whole security boundary, not
  just a directory walk.
- **Sibling backend calls dropped beside a declaration-only call.** When one
  response mixes an invocable tool call with a declaration-only (frontend)
  one, this port returns the whole response unexecuted — the same limitation
  #7388 works around in .NET. Upstream's fix is an opt-in decorator that
  stashes the invocable calls in the session state bag and re-injects them
  next turn as pre-approved approval responses. That mechanism depends on an
  `AgentSessionStateBag` and approval-response binding this port does not
  have, so it is a design task rather than a port, and is left open
  deliberately.

## Post-`beb65b21` drift (checked against `4b1afd90`, 2026-08-07)

Upstream moved **112 commits touching `python/packages`** in the ~3.5 weeks
between `beb65b21` (2026-07-13) and `4b1afd90` (2026-08-07). Each was triaged
against the subsystems this port actually implements. The outcome splits four
ways; the counts are the honest picture, not a claim of completeness.

### Ported this pass (6, each with regression tests)

| Upstream | Change | Rust site |
|---|---|---|
| #6990 | **Structured-output text selection.** The JSON value was parsed from *every* message's text, joined with `"\n"`/`" "`, including reasoning content. Three separate corruptions: a tool result or user echo carrying JSON could be mistaken for the answer; a reasoning model's chain-of-thought was prepended to the payload; and a JSON document split across text chunks had separators injected into it (`{"na` + `me":1}` → `{"na me":1}`). Now mirrors upstream's `_last_non_empty_assistant_message_text` — last non-empty **assistant** message, `text` contents only, joined with no separator. | `core/types/response.rs` (`structured_output_text`, used by both `parse_json`s and `try_parse_value`) |
| #6916 | **Data-URI validation.** No validating constructor existed; a malformed `data:` URI was silently mis-sliced or ignored. Added `DataContent::from_uri` / `media_type_from_uri`, rejecting a missing `data:` scheme, a missing `,`, or a non-`;base64` declaration. | `core/types/content.rs` |
| #7126/#7127 | **Chat Completions `author_name` sanitization.** The API validates `name` against `^[^\s<\|\\/>]+$`. The port only skipped names containing whitespace, so `"a/b"` was sent and 400'd the entire request, while `"My Agent"` was dropped rather than sanitized. Now mirrors upstream/.NET `SanitizeAuthorName`: strip outside `[a-zA-Z0-9_]`, omit when empty, truncate to 64. | `openai/convert.rs` |
| #7369 | **OpenAI cache-*write* tokens.** Only cache *reads* were parsed. Added `cache_write_tokens` on both surfaces, populating the typed `cache_creation_input_token_count` (which is what this port's OTel layer reads, so the `gen_ai.usage.cache_creation.input_tokens` attribute now flows without the extra key-mapping table upstream needs). | `openai/convert.rs`, `openai/responses.rs` |
| #7162 | **Anthropic streaming token double-count.** Anthropic streams *cumulative* usage snapshots: `message_delta` repeats the input/cache counts `message_start` already reported. Because `absorb_update` sums every usage content, a stream reporting 25 input tokens aggregated to 50. Added a per-stream `StreamUsageAccumulator` threaded through `SseState` that emits increments. | `anthropic/convert.rs`, `anthropic/lib.rs` |
| #7095 | **Gemini 3 `thought_signature` replay.** Not supported at all. Gemini 3 requires the signature echoed when a function call is replayed, or the follow-up turn is rejected. Two placements exist and both are now handled: on the **function-call part itself** (Gemini 3's usual placement — `FunctionCallContent::protected_data`, which wins) and on a **preceding thought part** (`TextReasoningContent::protected_data`, used only to backfill a call carrying none, mirroring upstream's "backfill only when the raw Part lacks one"). Reasoning is no longer sent back as a part, matching upstream. | `core/types/content.rs`, `gemini/convert.rs` |

Verified: full workspace build, `cargo test --workspace --all-features`
(**1508 passing**, 24 of them new), `cargo clippy --all-targets --all-features`
clean, `cargo fmt --check` clean.

### Compaction cluster (follow-up pass)

Upstream's `_compaction.py` is an annotation-driven system that groups messages
into spans and flags them `_excluded`; this port deliberately implements a
smaller "return a reduced list" model (see the module docs). So the cluster was
triaged by *invariant* rather than by patch — what does upstream's fix
guarantee, and does this port's much simpler design guarantee it too? Two did
not, and both produced conversations providers reject outright:

| Upstream | Invariant | What was wrong here |
|---|---|---|
| #7406 | A function call and its result are retained or dropped **together**. | Confirmed broken two ways by probe. `TokenBudget` dropped an expensive call-bearing message while keeping its cheap result — a tool message answering nothing. `SelectiveToolResult` deleted stale results outright while their assistant `tool_calls` entries stayed — an unanswered call. Either half alone is a 400 on the next request, not merely a worse completion. Upstream enforces this by linking call and result into one indivisible span *before* any strategy runs; with no span model here, `drop_orphaned_tool_exchanges` enforces the identical observable guarantee as a repair pass over the retained set. |
| #7219 | Compaction never yields a projection with nothing for the model to answer. | `Truncation::new(1)` / `SlidingWindow::new(0)` over a conversation with a system prefix returned system messages *only*. `ensure_non_system_message` reinstates the most recent non-system turn, accepting a result over the limit exactly as upstream does. |

`SelectiveToolResult` changed shape as a result: it now **replaces** a stale
result's payload with `OMITTED_TOOL_RESULT` instead of deleting the content.
That keeps the exchange paired (no orphan to repair) while still shedding the
bulk, and matches the intent of upstream's `ToolResultCompactionStrategy`,
which likewise replaces stale tool groups with a compact stand-in rather than
removing them — upstream summarizes the group with an LLM; this port, having no
summarizing strategy, substitutes a fixed marker. Three existing tests asserted
the old delete-and-drop behavior over fixtures with unpaired results; they were
rebuilt on realistic paired conversations.

Ordering is load-bearing and documented at the call site: the orphan repair runs
*first*, because it can itself strip a conversation down to system-only (when
the sole non-system message was an orphaned result), which the minimum-retention
pass then catches.

Not applicable in this cluster:

- **#7124** (token counts inflated by `\uXXXX` escapes on non-ASCII text) —
  upstream counted tokens off a JSON serialization with `ensure_ascii=True`;
  this port counts message text directly, so the inflation never existed.
  Pinned with a regression test rather than changed.
- **#7391** (ignore `_excluded` tool results) — depends on upstream's
  exclusion-marking model, which this port does not implement.
- **#7396 / #7375** (bound tool-result summaries; bound summarization input
  before the provider call) — both govern the LLM-backed `Summarization`
  strategy, which this port does not have.

### mem0 storage/retrieval scope separation (#7531)

The port had the same cross-user memory leak upstream fixed: `before_run`
searched with `self.user_id` / `self.agent_id` — the *storage* scope. A provider
configured with a shared `agent_id` (one agent serving many users, the ordinary
deployment shape) therefore retrieved memories written by **every** user of that
agent and injected them into the current user's conversation.

`Mem0Provider` now separates the two scopes exactly as upstream does:

- **Storage** — `with_application_id` / `with_agent_id` / `with_user_id`, stamped
  onto memories written by `after_run`, never used to retrieve.
- **Retrieval** — `with_search_application_id` / `with_search_agent_id` /
  `with_search_user_id`, each queried as its **own** request and the results
  merged (Mem0 ANDs the entries of a single `filters` object, so one combined
  query would return only memories tagged with *both* — dropping exactly the
  agent-wide memories written by other users that `search_agent_id` exists for;
  upstream fans the partitions out for the same reason). Used only by
  `before_run`, and never inheriting from
  the storage scope. With no retrieval scope set, `before_run` retrieves nothing
  and warns once. `search_application_id` narrows a search only as a fallback
  when neither user nor agent retrieval scope is set, matching upstream.

This is a deliberate **behavior change**, as it was upstream: code that
retrieved memories via `with_user_id` alone must now also set
`with_search_user_id`. Agent-wide retrieval has to be requested explicitly,
which is the entire point — the leak came from it being implicit. Seven
loopback tests configured only a storage scope and were updated; three new tests
pin the isolation.

### Approvals: duplicate call on round-trip (#7271) — fixed

`replace_approval_contents_with_results` deduped a restored function call
against **only the message being scanned**. On an approval round trip a hosting
layer replays the stored `function_call` item and its approval request as two
*separate* assistant messages, so the per-message check never fired and the
approval request restored a second copy of the call. Only one copy received the
function result; the provider then rejects the orphan with "No tool output
found for function call ...".

Now collects pending call ids across all messages, excludes ids that already
carry a result (reusing a call id for a later invocation is supported, and a
completed pair must not suppress a fresh request), and records each restored id
so two approval requests for the same call cannot both expand. Four regression
tests cover the round-trip shape, the double-request case, the single-request
case that must still expand, and the reused-id case.

### Approvals: blocked on a missing core field (#7462)

Upstream stopped serializing **local** function approvals as MCP input items on
the OpenAI Responses path — local approvals are resolved in-process, and only
*hosted* (MCP) decisions have a matching approval request on the provider. It
distinguishes the two with `_is_hosted_tool_approval`, which tests
`function_call.additional_properties["server_label"]`.

This port cannot express that test: `FunctionCallContent` has only
`call_id` / `name` / `arguments` — no `additional_properties` — so hosted and
local approvals are indistinguishable, and `messages_to_input` serializes every
approval content as `mcp_approval_request` / `mcp_approval_response`.

In practice the local case is mostly shielded by layering, since
`FunctionInvokingChatClient` converts local approvals into calls and results
*before* the Responses client serializes anything, and an approved hosted
approval survives untouched (no local tool produces a result for it, so the
conversion is skipped). One narrower divergence is real and shares the same
blocker: a **rejected** approval is unconditionally converted into a local
rejection result, so a rejected *hosted* MCP approval never reaches the provider
as `mcp_approval_response {approve: false}`. Closing either needs
`additional_properties` on `FunctionCallContent` first — a core type change
worth doing deliberately rather than as a side effect.

### Provider cluster — triaged, six of seven not applicable

Worked through the provider fixes as a batch. Only one turned out to need code,
and it needed none: the rest do not apply to this port's architecture. Recorded
individually so the next pass does not re-derive them.

| Upstream | Verdict |
|---|---|
| #7199 — raw JSON-Schema `response_format` passed through unwrapped | **Structurally impossible here.** Upstream's bug needs a raw dict in the `response_format` slot; this port's `ResponseFormat` is a closed typed enum (`Text` / `JsonObject` / `JsonSchema{..}`) whose `Serialize` impl always builds the correct envelope. |
| #7163 — GPT-5.6 prompt-cache breakpoints | **Half already available, half blocked.** The request-wide `prompt_cache_options` reaches the wire today through `ChatOptions::additional_properties`, which `apply_options` merges into the body — no typed field needed, now pinned by a test. The *per-content* `prompt_cache_breakpoint` marker is blocked: it lives on `Content.additional_properties`, which this port's content types do not have. |
| #7283 — Foundry agent inheriting `OPENAI_CHAT_MODEL` | **N/A.** This port reads `FOUNDRY_MODEL` and never consults `OPENAI_CHAT_MODEL`, and the agent-*reference* request path the bug lives on is a documented unimplemented extension point here (`FoundryAgent` realizes a Prompt Agent client-side, where sending a model is correct). |
| #7417 — CopilotStudio `LineTooLong` on large activities | **N/A.** aiohttp's 512 KB per-line read buffer is the cause; this port speaks Direct-to-Engine over `reqwest`, which has no equivalent per-line cap. |
| #7155 — forward `GitHubCopilotOptions` verbatim to `create_session` | **N/A.** Different surface: this port's GitHub Copilot client is the OpenAI-compatible `POST /chat/completions` endpoint, not the Copilot Agent SDK's session API. |
| #7300 — forward Copilot input attachments as inline blobs | **N/A**, same reason as #7155 (`copilot_session.send(..., attachments=...)`). |
| #7278 — Azure AI Search query-source identity | **N/A for now.** The `x-ms-query-source-authorization` header applies to *agentic* Knowledge Base retrieval; this port's provider implements classic index search (hybrid/semantic/vector) only. Agentic retrieval is a feature gap, not a bug — worth its own decision. |

The recurring blocker is worth calling out on its own: **three separate items
now hinge on this port's content types lacking `additional_properties`** —
#7462 (hosted vs. local approvals), the rejected-hosted-approval divergence
found alongside it, and #7163's per-content cache breakpoints. Adding
`additional_properties` to `Content` / `FunctionCallContent` would unblock all
three at once and is the highest-leverage next piece of work in this area.

### Verified already satisfied — no action (the port was level or ahead)

- **#6809** function-call name lost when a streaming delta carries it late —
  `FunctionCallContent::merge` already fills an empty name from `other`.
- **#7488** Gemini thought summaries surfaced as reasoning content — already
  done; upstream was catching up to the port here.
- **#7292** OpenAI Responses native `instructions` — the port already sends the
  top-level field rather than prepending a system message.
- **#7060** per-run `additional_beta_flags` leaking into the Anthropic request
  body — `compute_beta_flags` already removes the key from
  `additional_properties`.
- **#7105** finish-reason normalization — `map_stop_reason` already maps
  `guardrail_intervened` → `content_filter` and passes unknown reasons through.

### Not applicable — architectural divergence

- **#6822** (Ollama parallel tool calls colliding on `call_id`) and the Ollama
  half of **#7105** (`done_reason` normalization): this port's Ollama client
  targets Ollama's **OpenAI-compatible** `/v1/chat/completions` surface, which
  returns real per-call ids and OpenAI-shaped `finish_reason`s. Upstream's bugs
  live in the native `/api/chat` path, which the port does not have.
- The **AG-UI** cluster (~10 commits), **harness/skills/evaluation**
  graduations, the **durabletask / Azure Functions** extraction,
  **foundry-hosting**, and the **telegram / chatkit / monty / hyperlight / lab**
  packages: no Rust counterpart, already documented under "Remaining" below.

### Triaged as relevant but NOT yet ported

Carried forward as the next pass's work — none is closed, and the list is
roughly in descending value order:

- **Approvals cluster** (#7407, #7408, #7410, #7345, #7090): decisions
  preserved under OpenAI continuation, tool content returned after invocation
  limits, provider-injected approvals deferred to in-run execution,
  resume/replay, auto-approval name-collision warnings. (#7271 is done — see
  above; #7462 is blocked — see below.)
- **Workflow checkpointing**: full replayability (#7374, BREAKING), sub-workflow
  restore preserving sub-workflow state (#7097), checkpoint encoding (#6579).
- **Sessions**: `SessionStore` moved into core + Foundry Responses session
  persistence (#7306), cross-session origin attribution (#7041), hosted session
  snapshot isolation (#7141).
- **Core**: agent-hooks interception contract (#7515, new experimental
  feature), declaration-only streaming metadata (#7409), stateless replay of
  reasoning-paired tool calls (#7233), `from_dict` type enforcement (#7256),
  `PropertySchema` nested recursion (#7200), feature-usage User-Agent telemetry
  (#7420), tool-def JSON for observability (#7029), restricting an unknown
  `finish_reason` from the OTel attribute (#7105).
- **Orchestration**: Magentic manager duplicating conversation history (#6297).
- **Providers**: Responses conversation-ID helper (#7234, BREAKING). The rest
  of this cluster was triaged and is not applicable — see below.
- **MCP**: `header_provider` headers on the initialize handshake and ambient
  requests (#7305, #7218), tool-use sampling results (#7189).
- **New package**: `azure-cosmos-memory` context provider (#6719).
- **Verification pass**: upstream added a Mistral *chat* client (#7392); this
  port already has `MistralChatClient`, but it was written before upstream's and
  has not been diffed against it.

## Post-`68136ee` drift (checked against `beb65b21`, 2026-07-13)

Upstream moved 4 Python commits past the `68136ee` baseline; all four are
accounted for:

- **`as_tool` session propagation** (`f3057ef2`, fixing a feature already in
  `68136ee` that the port had not yet carried): `AsToolOptions` gained
  `propagate_session` (plus the previously missing `stream_callback` and
  `approval_mode`). Implemented with upstream's *fixed* child-session
  semantics: the sub-agent runs on an `AgentSession::child` of the parent —
  same `session_id`, **shared** `state` bag, **isolated** (cleared)
  `service_session_id`, so the parent's pending server-side conversation
  pointer never leaks into the sub-agent's own service calls. Plumbing:
  `AgentSession.state` became a `SessionState` handle (shared by reference
  across clones, matching Python's dict-reference semantics), the agent hands
  its session to the function-invocation loop via a non-wire
  `ChatOptions::session` side channel (popped before the provider client sees
  the options, exactly like upstream's client-kwargs `pop("session")`), and
  tools can read it through `FunctionInvocationContext::session` /
  `Tool::invoke_in_context`.
- **Parallel tool-span context** (`7f4cc296`): Python lost the ambient span
  when fanning parallel tool calls out via `asyncio.create_task` without
  copying contextvars. The Rust loop polls all invocations in-task under the
  instrumented future, so the parent span always propagates — no code change
  needed; a regression test now pins the behavior
  (`observability.rs::parallel_tool_call_spans_keep_the_surrounding_span_as_parent`).
- **Harness compaction fix** (`b3d523ee`): `@experimental` harness module —
  out of scope (see "Remaining").
- **OTel Distro sample** (`8e74360d`): Python-only sample — no Rust action.

A subsequent example-gallery audit against upstream's `python/samples` closed
two further gaps that predated the re-baseline:

- **Embeddings** (UPSTREAM_DRIFT §4/§5's "if in scope" item — now in scope):
  `Embedding`/`GeneratedEmbeddings`/`EmbeddingGenerationOptions` types + the
  `EmbeddingClient` trait in core, with provider clients for **OpenAI**
  (`/v1/embeddings`, loopback-tested), **Azure OpenAI** (deployment-scoped,
  api-key/Entra), **Ollama** (OpenAI-compatible surface), and **Mistral**
  (`mistral-embed` default — upstream's mistral package is embeddings-only).
  Bedrock/Foundry/Gemini embedding clients remain open (small, independent
  additions).
- **Progressive tool exposure** (upstream `FunctionInvocationContext.tools`):
  a `LiveToolList` handle on the invocation context with
  `add_tools`/`remove_tools` (duplicate-name rejection, batch-validated);
  the function-calling loop re-snapshots it at the top of every model
  iteration, so mutations take effect on the next iteration, never the
  in-flight batch.

## Done

### Naming / type-system cascade — Theme A + Theme F (complete)

`trait Agent`→`SupportsAgentRun`, `ChatAgent`→`Agent`, `ChatAgentBuilder`→
`AgentBuilder`, `ChatMessage`→`Message`, `AgentRunResponse`/`…Update`→
`AgentResponse`/`AgentResponseUpdate`, `ChatResponse.model_id`/`ChatOptions.model_id`→
`model`, `AiFunction`→`FunctionTool`, `AgentRunContext`→`AgentContext`,
`CitationAnnotation`→`Annotation`.

### Types & tools (§5/§6/§8)

- 12 new hosted tool-call/result `Content` variants; typed `UsageDetails`
  cache/reasoning fields (wired from Anthropic/OpenAI); `ContinuationToken`;
  `Annotation` `type:"citation"` discriminator.
- `hosted_image_generation()` + `ToolKind::HostedImageGeneration`.
- Cache/reasoning/`embeddings`/`prompt.name` OTel attributes.

### Sessions / context (§3) & new modules (§9)

- **ContextProvider → SessionContext reshape**: `Context`→`SessionContext`,
  `invoking`/`invoked`/`thread_created`→`before_run`/`after_run` (in-place
  mutation), `AggregateContextProvider` removed; ported across core + the
  redis/mem0/azure-ai-search provider crates.
- **`settings`** module (`SecretString` + `load_setting`).
- **`compaction`** module (`Tokenizer` + Truncation/SlidingWindow/TokenBudget/
  SelectiveToolResult strategies).

### Workflow engine & orchestrations (§10/§12)

- Per-executor serialization within a superstep; staged shared-state
  (commit-per-superstep).
- `WorkflowEvent::Intermediate` + `output_from`/`intermediate_output_from`
  designation + `OutputValidation`, wired through the Sequential/Concurrent/
  GroupChat/Magentic builders.
- Async edge conditions (`should_route`) with a backward-compatible sync API +
  `EdgeGroup::has_condition`.
- **Handoff mesh topology**: `add_handoff(src).to(targets)` edges are now
  enforced per-source (previously the adjacency map was built but discarded, so
  every agent could reach every other). A source is restricted to its declared
  outgoing edges; a source with no edges (when any edge is declared) is a leaf
  that cannot initiate a handoff; an empty map preserves the full-mesh
  back-compat. Rejected targets reuse the existing unknown-target feedback path.

### Providers & hosting (§13/§14)

- Removed the dead `OpenAIAssistantsClient`; flipped OpenAI client names
  (`OpenAIChatClient`=Responses, `OpenAIChatCompletionClient`=Chat Completions).
- New provider crates: **ollama**, **gemini**, **mistral**, **foundry-local**
  (Microsoft Foundry Local's OpenAI-compatible localhost endpoint; reuses
  `agent_framework_openai::convert`), **bedrock** (AWS Bedrock Converse
  API with a dependency-free **SigV4** signer verified against AWS's published
  `get-vanilla` known-answer test vector), and **github-copilot**
  (OpenAI-compatible chat endpoint behind the GitHub→Copilot short-lived-token
  exchange, with token caching/refresh) — full `ChatClient` impls, wired into
  the umbrella crate + examples.
- **`agent-framework-azure-ai` → `agent-framework-foundry`** (the largest
  provider item): upstream deleted the Azure AI Agents threads/runs data-plane
  and replaced it with the `foundry` package on the Responses API. Renamed the
  crate and rewrote it — `FoundryChatClient` (Responses API,
  `POST {endpoint}/openai/v1/responses`, Entra scope `https://ai.azure.com/.default`)
  delegates to the existing `agent_framework_azure::responses::AzureOpenAIResponsesClient`
  rather than reinventing the transport; added `PromptAgentDefinition`,
  `FoundryAgent` (a `SupportsAgentRun` realizing a Prompt Agent client-side) and
  `to_prompt_agent()`. Env prefix `AZURE_AI_`→`FOUNDRY_`,
  `model_deployment_name`→`model`. Rewired umbrella crate (feature/re-export),
  examples, and docs; the distinct `agent-framework-azure-ai-search` crate is
  untouched. (Binding to a server-hosted agent on the Foundry Agents
  control-plane is a documented extension point, not yet wired.)
- **Anthropic multi-cloud** (rework in place, no new crates): the `anthropic`
  crate is now a superset with `AnthropicBedrockClient` (AWS Bedrock
  `InvokeModel`, `anthropic_version: bedrock-2023-05-31`, reusing the verified
  `agent_framework_bedrock::sigv4` signer), `AnthropicVertexClient` (Vertex
  `:rawPredict`, `vertex-2023-10-16`, pluggable `VertexTokenProvider` for the
  Google OAuth token), and `AnthropicFoundryClient` (Entra via
  `agent_framework_azure::TokenCredential`; route/version overridable as a
  documented extension point). A shared `convert::build_cloud_request` omits the
  top-level `model` (it's URL-encoded) and stamps the per-cloud
  `anthropic_version`. Cloud-transport streaming is a documented single-update
  adaptation (the AWS event-stream / `:streamRawPredict` framing is a marked
  extension point). No dependency cycle (bedrock/azure don't depend back).
- `CosmosCheckpointStorage`; DevUI security middleware (Host-header
  anti-DNS-rebinding guard + optional bearer auth, opt-in).
- **Reusable Responses-conversion module** (`hosting::responses`): extracted the
  OpenAI-Responses wire types + conversion (`responses_to_run` /
  `responses_from_run`, `ResponsesRequest`, `ResponseObject`) out of the DevUI
  internals into a public, framework-agnostic module — mirroring upstream's
  `hosting-responses` package and resolving the crate's self-documented TODO.
  Pure refactor; DevUI `/v1/responses` wire output unchanged.

### Streaming API shape — Theme B (satisfied idiomatically)

Upstream's Python unifies buffered vs. streaming behind `run(stream=…)` /
`get_response(stream=…)`. Rust can't cleanly return either a buffered value or
a stream from one function keyed on a runtime bool, so the port already
expresses this idiomatically as method **pairs** — `run`/`run_stream` and
`ChatClient::get_response`/`get_streaming_response`. No further work: the
capability is present, just spelled the Rust way.

## Remaining

The tractable, verifiable alignment is complete. Everything still open falls
into one of three buckets — large-and-externally-blocked, or a deliberate,
documented divergence, or low-verifiability without an upstream artifact this
repo doesn't have. None is a straightforward port.

**Deliberate / documented divergences (not gaps to "fix"):**
- **Streaming API shape (Theme B)** — expressed as Rust method pairs
  (`run`/`run_stream`); a single `stream=`-keyed function isn't idiomatic Rust.
- **Declarative *workflow* DSL** — upstream's declarative workflow schema is the
  Power Platform / Copilot Studio imperative DSL, which doesn't map onto this
  port's graph engine; the crate defines a documented Rust-native `WorkflowSpec`
  instead. Agents and Rust-native workflows already load **and execute**.
- **Server-hosted control-plane bindings** left as documented extension points:
  the Foundry Agents control plane (`FoundryAgent` realizes a Prompt Agent
  client-side), and true incremental cloud-transport streaming (AWS
  event-stream framing / Vertex `:streamRawPredict`).

**Large, externally-blocked ecosystem packages (each a substantial new crate):**
- **`durabletask`** — durable agent/workflow hosting over Microsoft's Durable
  Task Framework via a gRPC sidecar (replay-safe orchestration + entity model).
  Blocked on the sidecar protocol/SDK; second-largest ecosystem item.
- **`@experimental` harness / security / evaluation** modules — upstream-unstable
  surfaces, low value to pin before they settle.
- **`agent-framework-claude`** — a `BaseAgent` that subprocesses the Claude
  Agent SDK / CLI; there is no Rust Claude Agent SDK, so this is a subprocess
  shim of speculative value (distinct from the `anthropic` chat client, which is
  done).

**Low-verifiability without the upstream frontend contract:**
- **DevUI's remaining ~17 UI routes** (conversations / deployments API) — a
  pre-existing gap that serves the bundled web UI; faithfully porting them needs
  the frontend's request/response contract, which isn't in this repo. The
  security-relevant middleware (Host-header guard + bearer auth) and the core
  entity/responses routes are already in place; the reusable Responses
  conversion (`hosting::responses`) is done.
