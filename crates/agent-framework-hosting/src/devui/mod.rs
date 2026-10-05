//! DevUI-style HTTP API: entity discovery and OpenAI-Responses-flavored
//! execution, mirroring the Python `agent_framework_devui` server.
//!
//! # Routes
//! - `GET /` and `GET /ui` — the embedded single-file debug page (see
//!   the crate's `ui` module).
//! - `GET /health` — liveness + entity count.
//! - `GET /v1/entities` — list entities (`DiscoveryResponse`).
//! - `GET /v1/entities/{id}/info` — entity details (`EntityInfo`).
//! - `POST /v1/responses` — execute an entity. Routes on `metadata.entity_id`
//!   (DevUI's convention), then `extra_body.entity_id`, then `model`.
//!
//! # SSE event mapping (`stream: true`)
//! Agents: `response.created` → `response.in_progress` →
//! `response.output_item.added` → `response.content_part.added` →
//! `response.output_text.delta`* → `response.completed` → `data: [DONE]`.
//!
//! Workflows map each engine event to a DevUI event name:
//! `ExecutorInvoked` → `response.output_item.added` (an `executor_action`
//! item), `ExecutorCompleted`/`ExecutorFailed` → `response.output_item.done`,
//! `Output` → `response.output_item.added` (a message), `RequestInfo` →
//! `response.request_info.requested`, everything else →
//! `response.workflow_event.completed`; then `response.completed` + `[DONE]`.
//!
//! # Divergences from DevUI
//! - **Continuation** follows the OpenAI Responses API rather than DevUI's
//!   own conversation endpoints. An agent request may carry
//!   `previous_response_id` (branch from an immutable snapshot) or
//!   `conversation` (advance a mutable head); every response is stored and
//!   continuable, and an unknown `previous_response_id` is a `400
//!   previous_response_not_found`, as on OpenAI's API. Sessions live in the
//!   host's [`SessionStore`] (in memory unless
//!   [`AgentHost::with_session_store`](crate::AgentHost::with_session_store)
//!   says otherwise). Turns on one conversation are serialized.
//! - **Workflows** always run inside a conversation (one is minted when the
//!   request names none, and returned as `conversation.id`), and each
//!   conversation has its own checkpoint storage, as in DevUI. A paused run
//!   resumes when a later request on that conversation carries DevUI's
//!   `workflow_hil_response` content (`{"type": "workflow_hil_response",
//!   "responses": {request_id: value}}` inside a `message` item): the latest
//!   checkpoint — or `extra_body.checkpoint_id` — is restored and the
//!   responses delivered. DevUI's `/v1/conversations` CRUD endpoints are not
//!   served.
//! - SupportsAgentRun streaming drives the core `SupportsAgentRun::run_stream` and frames each update
//!   as SSE live (one `response.output_text.delta` per update); the terminal
//!   `response.completed` aggregates the run. Event ordering and payloads match
//!   DevUI's names. Non-streaming requests stay on `SupportsAgentRun::run`. (Workflow
//!   streaming still frames the workflow's own event stream after the run.)

pub mod models;

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::{json, Map, Value};

use agent_framework_core::history::ensure_history_provider;
use agent_framework_core::session::AgentSession;
use agent_framework_core::session_store::SessionStore;
use agent_framework_core::types::{AgentResponse, AgentResponseUpdate, Message};
use agent_framework_core::workflow::{CheckpointStorage, WorkflowEvent};

use crate::continuation::Continuations;
use crate::registry::{AgentRecord, EntityRecord, HostState, WorkflowRecord};
use crate::responses::{
    create_conversation_id, create_response_id, openai_error, responses_from_run,
    responses_run_options, responses_session_id, responses_to_run, InputTokensDetails, OutputItem,
    OutputMessage, OutputTokensDetails, ResponseObject, ResponsesContinuation, ResponsesRequest,
    Usage,
};
use crate::sse::{sse_response, sse_response_stream};
use crate::util;
use models::{DiscoveryResponse, EntityInfo, HealthResponse};

/// Build the DevUI router for a registry.
///
/// The stateful API routes are merged with the stateless embedded debug page
/// ([`crate::ui`]) served at `GET /` and `GET /ui`.
pub(crate) fn router(state: Arc<HostState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/entities", get(list_entities))
        .route("/v1/entities/{entity_id}/info", get(entity_info))
        .route("/v1/responses", post(create_response))
        .with_state(state)
        .merge(crate::ui::router())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(state): State<Arc<HostState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy",
        entities_count: state.list().len(),
        framework: "agent_framework",
    })
}

async fn list_entities(State(state): State<Arc<HostState>>) -> Json<DiscoveryResponse> {
    let entities = state.list().iter().map(entity_info_for).collect();
    Json(DiscoveryResponse { entities })
}

async fn entity_info(
    State(state): State<Arc<HostState>>,
    Path(entity_id): Path<String>,
) -> Response {
    match state.get(&entity_id) {
        Some(record) => Json(entity_info_for(record)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(openai_error(
                format!("Entity {entity_id} not found"),
                "invalid_request_error",
                Some("entity_not_found"),
            )),
        )
            .into_response(),
    }
}

async fn create_response(
    State(state): State<Arc<HostState>>,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    let Some(entity_id) = request.entity_id() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(openai_error(
                "Missing entity_id. Provide metadata.entity_id (or extra_body.entity_id, or model).",
                "invalid_request_error",
                Some("missing_entity_id"),
            )),
        )
            .into_response();
    };

    let Some(record) = state.get(&entity_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(openai_error(
                format!("Entity not found: {entity_id}"),
                "invalid_request_error",
                Some("entity_not_found"),
            )),
        )
            .into_response();
    };

    let model = request.model.clone().unwrap_or_else(|| entity_id.clone());

    let continuation = match responses_session_id(&request) {
        Ok(c) => c,
        Err(message) => return invalid_request(message, "invalid_continuation"),
    };

    match record {
        EntityRecord::Agent(agent) => run_agent(&state, agent, &request, continuation, model).await,
        EntityRecord::Workflow(workflow) => {
            run_workflow(&state, workflow, &request, continuation, model).await
        }
    }
}

// ---------------------------------------------------------------------------
// Entity info
// ---------------------------------------------------------------------------

fn entity_info_for(record: &EntityRecord) -> EntityInfo {
    match record {
        EntityRecord::Agent(a) => EntityInfo {
            id: a.id.clone(),
            entity_type: "agent",
            name: a.name.clone(),
            description: a.description.clone(),
            framework: "agent_framework",
            // The core `SupportsAgentRun` trait exposes no tool list, so this is absent.
            tools: None,
            metadata: Map::new(),
            source: "in_memory",
            instructions: a.instructions.clone(),
            // `model` is not accessible through the `SupportsAgentRun` trait.
            model: None,
            executors: None,
            input_schema: None,
            start_executor_id: None,
        },
        EntityRecord::Workflow(w) => EntityInfo {
            id: w.id.clone(),
            entity_type: "workflow",
            name: w.name.clone(),
            description: w.description.clone(),
            framework: "agent_framework",
            tools: None,
            metadata: Map::new(),
            source: "in_memory",
            instructions: None,
            model: None,
            // The full executor set is not enumerable through the public
            // `Workflow` API; the start executor is.
            executors: Some(vec![w.workflow.start_executor_id().to_string()]),
            input_schema: Some(json!({ "type": "string" })),
            start_executor_id: Some(w.workflow.start_executor_id().to_string()),
        },
    }
}

// ---------------------------------------------------------------------------
// SupportsAgentRun execution
// ---------------------------------------------------------------------------

async fn run_agent(
    state: &HostState,
    agent: &AgentRecord,
    request: &ResponsesRequest,
    continuation: Option<ResponsesContinuation>,
    model: String,
) -> Response {
    let messages = responses_to_run(request);
    let options = responses_run_options(request);
    let input_len = approx_input_len(&request.input);
    let response_id = create_response_id();
    let conts = &state.continuations;

    // A conversation is a mutable head: hold it for the whole
    // read-run-write cycle so concurrent turns serialize rather than both
    // starting from the same head and one overwriting the other.
    let head_key = match &continuation {
        Some(ResponsesContinuation::Conversation(id)) => Some(Continuations::key(&agent.id, id)),
        _ => None,
    };
    let head_guard = match &head_key {
        Some(key) => Some(conts.lock_head(key).await),
        None => None,
    };

    let session = match load_agent_session(state, agent, continuation.as_ref(), &response_id).await
    {
        Ok(s) => s,
        Err(resp) => return resp,
    };
    let response_key = Continuations::key(&agent.id, &response_id);

    if request.stream {
        // Live streaming: drive `run_stream` and frame each update as OpenAI
        // Responses SSE events (one `response.output_text.delta` per update)
        // as they arrive, then a final `response.completed`.
        // Bounded channel: a slow client applies backpressure and a
        // disconnected client cancels the run (see `crate::sse`).
        let target = agent.agent.clone();
        let store = conts.sessions().clone();
        let (tx, rx) = crate::sse::bounded_sse_channel();
        let disconnect = tx.clone();
        let mut session = session;
        // `run_stream` takes the session by value and writes history back
        // through its providers once the stream is drained. Attaching the
        // history provider *before* cloning means the clone kept here
        // shares it, so the post-run session can be stored afterwards.
        ensure_history_provider(&mut session);
        let kept = session.clone();
        tokio::spawn(async move {
            let _head_guard = head_guard;
            let produce = async move {
                let mut framing =
                    AgentStreamFraming::new(model, input_len, response_id, continuation.clone());
                for ev in framing.preamble() {
                    if tx.send(ev).await.is_err() {
                        return;
                    }
                }
                match target
                    .run_stream(messages, Some(session), Some(options))
                    .await
                {
                    Ok(mut stream) => {
                        while let Some(item) = stream.next().await {
                            match item {
                                Ok(update) => {
                                    for ev in framing.push_update(&update) {
                                        if tx.send(ev).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(framing.error_event(&e.to_string())).await;
                                    return;
                                }
                            }
                        }
                        // Stored before the terminal event, so a client that
                        // continues as soon as it sees `response.completed`
                        // finds the response already continuable.
                        if let Err(e) =
                            store_agent_session(&*store, &response_key, head_key.as_deref(), &kept)
                                .await
                        {
                            let _ = tx.send(framing.error_event(&e.to_string())).await;
                            return;
                        }
                        // Flushed before the terminal event, so a client
                        // sees each call announced and filled in before the
                        // payload that lists them.
                        for ev in framing.call_events() {
                            if tx.send(ev).await.is_err() {
                                return;
                            }
                        }
                        let _ = tx.send(framing.completed()).await;
                    }
                    Err(e) => {
                        let _ = tx.send(framing.error_event(&e.to_string())).await;
                    }
                }
            };
            tokio::select! {
                _ = disconnect.closed() => {}
                _ = produce => {}
            }
        });
        sse_response_stream(rx)
    } else {
        let mut session = session;
        let response = match agent
            .agent
            .run_with_options(messages, Some(&mut session), options)
            .await
        {
            Ok(r) => r,
            Err(e) => return execution_error(e.to_string()),
        };
        if let Err(e) = store_agent_session(
            &**conts.sessions(),
            &response_key,
            head_key.as_deref(),
            &session,
        )
        .await
        {
            return execution_error(e.to_string());
        }
        drop(head_guard);
        let obj = agent_response_object(&response, &response_id, &model, input_len)
            .with_continuation(continuation.as_ref());
        Json(obj).into_response()
    }
}

/// The session a request runs against: a working copy of the one it
/// continues, or a new one.
///
/// An unknown `previous_response_id` is an error, as it is on OpenAI's API:
/// silently starting a fresh conversation would turn a typo into lost
/// context. An unknown `conversation` starts one under that id, as upstream
/// DevUI does for a conversation it has not seen.
async fn load_agent_session(
    state: &HostState,
    agent: &AgentRecord,
    continuation: Option<&ResponsesContinuation>,
    response_id: &str,
) -> Result<AgentSession, Response> {
    let store = state.continuations.sessions();
    let fresh = |id: &str| agent.agent.create_session().with_session_id(id);
    match continuation {
        None => Ok(fresh(response_id)),
        Some(ResponsesContinuation::PreviousResponse(id)) => {
            match store.get(&Continuations::key(&agent.id, id)).await {
                Ok(Some(s)) => Ok(s),
                Ok(None) => Err(invalid_request(
                    format!("Previous response with id '{id}' not found."),
                    "previous_response_not_found",
                )),
                Err(e) => Err(execution_error(e.to_string())),
            }
        }
        Some(ResponsesContinuation::Conversation(id)) => {
            match store.get(&Continuations::key(&agent.id, id)).await {
                Ok(Some(s)) => Ok(s),
                Ok(None) => Ok(fresh(id)),
                Err(e) => Err(execution_error(e.to_string())),
            }
        }
    }
}

/// Store a completed run's session under its response id — every response
/// is continuable — and advance the conversation head when there is one.
async fn store_agent_session(
    store: &dyn SessionStore,
    response_key: &str,
    head_key: Option<&str>,
    session: &AgentSession,
) -> agent_framework_core::Result<()> {
    store.set(response_key, session).await?;
    if let Some(head) = head_key {
        store.set(head, session).await?;
    }
    Ok(())
}

fn invalid_request(message: impl Into<String>, code: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(openai_error(message, "invalid_request_error", Some(code))),
    )
        .into_response()
}

/// Build the aggregated (non-streaming) response for an agent run.
///
/// Delegates the OpenAI-Responses shape to [`responses_from_run`], then fills
/// in DevUI's `~4-chars-per-token` usage estimate when the run reported none.
fn agent_response_object(
    resp: &AgentResponse,
    response_id: &str,
    model: &str,
    input_len: usize,
) -> ResponseObject {
    let mut obj = responses_from_run(resp, response_id, model);
    if obj.usage.is_none() {
        let output_len = obj.output_text.as_deref().unwrap_or_default().len();
        obj.usage = Some(usage_estimate(input_len, output_len));
    }
    obj
}

/// Incremental OpenAI-Responses SSE framing for a streamed agent run, driven
/// one [`AgentResponseUpdate`] at a time. Emits the fixed preamble, one
/// `response.output_text.delta` per non-empty update, and a final
/// `response.completed` aggregating the run (text + usage).
struct AgentStreamFraming {
    model: String,
    input_len: usize,
    rid: String,
    mid: String,
    seq: u64,
    /// Updates collected so the terminal `response.completed` can aggregate the
    /// full text and usage via [`AgentResponse::from_updates`].
    collected: Vec<AgentResponseUpdate>,
    /// Function calls seen so far, held until the stream ends so a call
    /// the agent answered itself is never advertised.
    pending: util::StreamingCalls,
    /// The item ids minted for the calls actually announced, in order. The
    /// message occupies `output_index` 0, so a call's index is its
    /// position here plus one.
    calls: Vec<String>,
    /// Rendered on every response object this stream emits.
    continuation: Option<ResponsesContinuation>,
}

impl AgentStreamFraming {
    fn new(
        model: String,
        input_len: usize,
        rid: String,
        continuation: Option<ResponsesContinuation>,
    ) -> Self {
        Self {
            model,
            input_len,
            continuation,
            rid,
            mid: util::msg_id(),
            seq: 0,
            collected: Vec::new(),
            pending: util::StreamingCalls::default(),
            calls: Vec::new(),
        }
    }

    fn next(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// The four fixed opening events (`response.created` … `content_part.added`).
    fn preamble(&mut self) -> Vec<Value> {
        let in_progress = serde_json::to_value(
            ResponseObject::in_progress(&self.rid, &self.model)
                .with_continuation(self.continuation.as_ref()),
        )
        .unwrap_or(Value::Null);
        let mid = self.mid.clone();
        vec![
            json!({ "type": "response.created", "sequence_number": self.next(), "response": in_progress.clone() }),
            json!({ "type": "response.in_progress", "sequence_number": self.next(), "response": in_progress }),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "sequence_number": self.next(),
                "item": { "type": "message", "id": mid, "role": "assistant", "content": [], "status": "in_progress" }
            }),
            json!({
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "item_id": mid,
                "sequence_number": self.next(),
                "part": { "type": "output_text", "text": "", "annotations": [] }
            }),
        ]
    }

    /// Frame one streamed update: a `response.output_text.delta` when it carries
    /// text (otherwise nothing). The update is retained for final aggregation.
    fn push_update(&mut self, update: &AgentResponseUpdate) -> Vec<Value> {
        self.collected.push(update.clone());
        let mut events = Vec::new();

        let delta = update.text();
        if !delta.is_empty() {
            let mid = self.mid.clone();
            let seq = self.next();
            events.push(json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "item_id": mid,
                "delta": delta,
                "logprobs": [],
                "sequence_number": seq,
            }));
        }

        // Calls are *not* emitted here. A call the agent answers itself
        // arrives with its result in a later update — with local tools,
        // `FunctionInvokingChatClient` runs the loop and then replays each
        // message as its own update — so announcing one on sight would ask
        // the client to re-run work already done, and no later event can
        // recall it. They are held and flushed by `call_events` once the
        // stream has ended and the outstanding set is known.
        self.pending.push(&update.contents);

        events
    }

    /// The call events held back during the stream, emitted once it has
    /// ended: each outstanding call is announced as its own output item and
    /// then given its (complete) arguments.
    ///
    /// The arguments arrive in one delta rather than forming incrementally,
    /// which is a presentation detail — a client cannot execute a call
    /// before it has the whole argument object. See [`util::StreamingCalls`]
    /// for why that trade is worth making.
    fn call_events(&mut self) -> Vec<Value> {
        let outstanding: Vec<(String, String, String)> = self
            .pending
            .outstanding()
            .into_iter()
            .map(|(id, name, args)| (id.to_string(), name.to_string(), args.to_string()))
            .collect();
        let mut events = Vec::with_capacity(outstanding.len() * 2);
        for (call_id, name, arguments) in outstanding {
            let item_id = util::msg_id();
            self.calls.push(item_id.clone());
            let output_index = self.calls.len();
            let seq = self.next();
            events.push(json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "sequence_number": seq,
                "item": {
                    "type": "function_call",
                    "id": item_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                    "status": "in_progress",
                }
            }));
            let seq = self.next();
            events.push(json!({
                "type": "response.function_call_arguments.delta",
                "output_index": output_index,
                "item_id": item_id,
                "delta": arguments,
                "sequence_number": seq,
            }));
        }
        events
    }

    /// The terminal `response.completed`, aggregating all collected updates.
    fn completed(&mut self) -> Value {
        let response = AgentResponse::from_updates(std::mem::take(&mut self.collected));
        let mut completed = responses_from_run(&response, &self.rid, &self.model)
            .with_continuation(self.continuation.as_ref());
        // The streamed response item id (`mid`) was already announced in the
        // preamble; use it here too instead of `responses_from_run`'s freshly
        // generated one, so the completed event refers to the same item.
        // Only the *message* item is rebuilt; any function-call items
        // `responses_from_run` produced are carried through, or the
        // streaming path would drop the calls the buffered one reports.
        // Rebuilding also has to preserve the status, or it would reset to
        // `completed` and contradict an `incomplete` response on this path
        // alone.
        //
        // The message item is emitted unconditionally here, unlike on the
        // buffered path, which omits it for a turn that is only a call. The
        // preamble already announced it at `output_index` 0 before any
        // content was known, and every call event numbered itself from
        // there; dropping it now would shift each call one index away from
        // the event that announced it.
        let text = completed.output_text.clone().unwrap_or_default();
        let status = completed.status;
        let mut output = Vec::with_capacity(self.calls.len() + 1);
        output.push(OutputItem::Message(
            OutputMessage::assistant_text(self.mid.clone(), text).with_status(status),
        ));
        // Reuse the item ids already announced, for the same reason the
        // message does: a client correlating the terminal payload with the
        // events it saw must find the same ids.
        let mut announced = self.calls.iter().cloned();
        output.extend(completed.output.into_iter().filter_map(|item| match item {
            OutputItem::Message(_) => None,
            OutputItem::FunctionCall(mut call) => {
                if let Some(id) = announced.next() {
                    call.id = id;
                }
                Some(OutputItem::FunctionCall(call))
            }
        }));
        completed.output = output;
        if completed.usage.is_none() {
            let output_len = completed.output_text.as_deref().unwrap_or_default().len();
            completed.usage = Some(usage_estimate(self.input_len, output_len));
        }
        // OpenAI pairs an incomplete response with its own terminal event
        // name, so a client switching on the event type — rather than reading
        // `status` out of the payload — still sees that the turn was cut off.
        //
        // Keyed on `status`, not on `incomplete_details`: those two came
        // apart once an unfamiliar provider reason started producing an
        // `incomplete` response with no schema-nameable detail. Reading the
        // optional field would announce `response.completed` around exactly
        // the payloads this whole path exists to flag.
        let event_type = if completed.status == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        let seq = self.next();
        json!({
            "type": event_type,
            "sequence_number": seq,
            "response": serde_json::to_value(completed).unwrap_or(Value::Null),
        })
    }

    /// An in-band error event for a failure that occurs after streaming began.
    fn error_event(&mut self, message: &str) -> Value {
        let seq = self.next();
        json!({
            "type": "error",
            "sequence_number": seq,
            "message": format!("Request execution failed: {message}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Workflow execution
// ---------------------------------------------------------------------------

async fn run_workflow(
    state: &HostState,
    workflow: &WorkflowRecord,
    request: &ResponsesRequest,
    continuation: Option<ResponsesContinuation>,
    model: String,
) -> Response {
    let conts = &state.continuations;
    let response_id = create_response_id();

    // Every workflow run belongs to a conversation, as in upstream DevUI:
    // the conversation owns the checkpoint storage a paused run resumes
    // from. A `previous_response_id` names the conversation its response
    // belonged to.
    let conversation_id = match &continuation {
        Some(ResponsesContinuation::Conversation(id)) => id.clone(),
        Some(ResponsesContinuation::PreviousResponse(id)) => {
            match conts.workflow_conversation_of(&Continuations::key(&workflow.id, id)) {
                Some(conv) => conv,
                None => {
                    return invalid_request(
                        format!("Previous response with id '{id}' not found."),
                        "previous_response_not_found",
                    )
                }
            }
        }
        None => create_conversation_id(),
    };
    let conversation_key = Continuations::key(&workflow.id, &conversation_id);
    let _head_guard = conts.lock_head(&conversation_key).await;
    let storage = conts.workflow_storage(&conversation_key);

    let hil_responses = match extract_workflow_hil_responses(&request.input) {
        Ok(r) => r,
        Err(message) => return invalid_request(message, "invalid_hil_response"),
    };
    let explicit_checkpoint = request
        .extra_body
        .as_ref()
        .and_then(|b| b.get("checkpoint_id"))
        .and_then(Value::as_str)
        .map(str::to_string);

    let run = if let Some(responses) = hil_responses {
        // Resume the paused run: the named checkpoint, else the latest one
        // this conversation wrote, then deliver the responses.
        let checkpoint_id = match explicit_checkpoint {
            Some(id) => id,
            None => match latest_checkpoint(&*storage, workflow.workflow.id()).await {
                Ok(Some(id)) => id,
                Ok(None) => {
                    return invalid_request(
                        "Cannot process HIL responses without a checkpoint: this conversation \
                         has no paused workflow run to resume.",
                        "checkpoint_not_found",
                    )
                }
                Err(e) => return execution_error(e.to_string()),
            },
        };
        let mut run = match workflow
            .workflow
            .run_from_checkpoint(&checkpoint_id, storage.clone())
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return invalid_request(
                    format!("Cannot resume from checkpoint: {e}"),
                    "checkpoint_error",
                )
            }
        };
        if let Err(e) = run.send_responses(responses).await {
            return invalid_request(
                format!("Failed to send HIL responses: {e}"),
                "invalid_hil_response",
            );
        }
        run
    } else if let Some(checkpoint_id) = explicit_checkpoint {
        match workflow
            .workflow
            .run_from_checkpoint(&checkpoint_id, storage.clone())
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return invalid_request(
                    format!("Cannot resume from checkpoint: {e}"),
                    "checkpoint_error",
                )
            }
        }
    } else {
        let input = input_to_workflow_value(&request.input);
        match workflow
            .workflow
            .run_with_checkpointing(input, storage.clone())
            .await
        {
            Ok(r) => r,
            Err(e) => return execution_error(e.to_string()),
        }
    };
    conts.record_workflow_response(
        Continuations::key(&workflow.id, &response_id),
        conversation_id.clone(),
    );

    let outputs = run.outputs();
    let pending: Vec<Value> = run
        .pending_requests()
        .into_iter()
        .map(|p| {
            json!({
                "request_id": p.request_id,
                "source_executor_id": p.source_executor_id,
                "request_data": p.request_data,
            })
        })
        .collect();

    let mut completed = workflow_response_object(&response_id, &outputs, pending, &model)
        .with_continuation(Some(&ResponsesContinuation::Conversation(conversation_id)));
    if let Some(ResponsesContinuation::PreviousResponse(id)) = &continuation {
        completed.previous_response_id = Some(id.clone());
    }
    if request.stream {
        let events = workflow_stream_events(run.events(), completed);
        sse_response(events)
    } else {
        Json(completed).into_response()
    }
}

/// The most recent checkpoint `storage` holds for `workflow_id`.
async fn latest_checkpoint(
    storage: &dyn CheckpointStorage,
    workflow_id: &str,
) -> agent_framework_core::Result<Option<String>> {
    Ok(storage
        .list(Some(workflow_id))
        .await?
        .into_iter()
        .max_by_key(|cp| (cp.timestamp_millis, cp.iteration_count))
        .map(|cp| cp.checkpoint_id))
}

/// Human-in-the-loop responses carried by a request, as upstream DevUI
/// sends them: a `message` input item whose content holds
/// `{"type": "workflow_hil_response", "responses": {request_id: value}}`.
/// The input may also arrive JSON-encoded as a string. A value wrapped as
/// `{"response": v}` is unwrapped to `v`, as upstream does.
fn extract_workflow_hil_responses(input: &Value) -> Result<Option<HashMap<String, Value>>, String> {
    let parsed;
    let items = match input {
        Value::Array(items) => items,
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Array(items)) => {
                parsed = items;
                &parsed
            }
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    for item in items {
        if item.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(parts) = item.get("content").and_then(Value::as_array) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("workflow_hil_response") {
                continue;
            }
            let Some(responses) = part.get("responses").and_then(Value::as_object) else {
                return Err("`workflow_hil_response` requires a `responses` object".into());
            };
            if responses.is_empty() {
                return Err("`workflow_hil_response` carried no responses".into());
            }
            return Ok(Some(
                responses
                    .iter()
                    .map(|(id, v)| {
                        let v = match v {
                            Value::Object(m) if m.contains_key("response") => m["response"].clone(),
                            other => other.clone(),
                        };
                        (id.clone(), v)
                    })
                    .collect(),
            ));
        }
    }
    Ok(None)
}

/// Build the aggregated response object for a workflow run.
fn workflow_response_object(
    response_id: &str,
    outputs: &[Value],
    pending: Vec<Value>,
    model: &str,
) -> ResponseObject {
    let output: Vec<OutputItem> = outputs
        .iter()
        .map(|o| {
            OutputItem::Message(OutputMessage::assistant_text(
                util::msg_id(),
                value_to_text(o),
            ))
        })
        .collect();
    let text = outputs
        .iter()
        .map(value_to_text)
        .collect::<Vec<_>>()
        .join("\n");
    ResponseObject {
        id: response_id.to_string(),
        object: "response",
        created_at: util::now_ts(),
        model: model.to_string(),
        status: "completed",
        // A workflow run has no single model turn to be cut off in; the
        // per-agent reasons, when there are any, belong to the executors.
        incomplete_details: None,
        x_finish_reason: None,
        output,
        previous_response_id: None,
        conversation: None,
        output_text: Some(text),
        usage: None,
        outputs: Some(outputs.to_vec()),
        pending_requests: pending,
        parallel_tool_calls: false,
        tool_choice: "none",
        tools: Vec::new(),
    }
}

/// State threaded through workflow-event mapping (mirrors DevUI's per-request
/// conversion context: sequence numbers, output index, executor item ids).
struct WfCtx {
    seq: u64,
    output_index: i64,
    item_id: String,
    exec_items: HashMap<String, (String, i64)>,
}

impl WfCtx {
    fn next(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }
}

/// Build the SSE event sequence for a streamed workflow run, ending with
/// `completed` as the terminal `response.completed` payload.
fn workflow_stream_events(wf_events: &[WorkflowEvent], completed: ResponseObject) -> Vec<Value> {
    let mut ctx = WfCtx {
        seq: 0,
        output_index: -1,
        item_id: util::msg_id(),
        exec_items: HashMap::new(),
    };
    let mut skeleton = ResponseObject::in_progress(&completed.id, &completed.model);
    skeleton.conversation = completed.conversation.clone();
    skeleton.previous_response_id = completed.previous_response_id.clone();
    let in_progress = serde_json::to_value(skeleton).unwrap_or(Value::Null);

    let mut events = vec![
        json!({ "type": "response.created", "sequence_number": ctx.next(), "response": in_progress }),
        json!({ "type": "response.in_progress", "sequence_number": ctx.next(), "response": in_progress }),
    ];

    for ev in wf_events {
        map_workflow_event(ev, &mut ctx, &mut events);
    }

    events.push(json!({
        "type": "response.completed",
        "sequence_number": ctx.next(),
        "response": serde_json::to_value(completed).unwrap_or(Value::Null),
    }));
    events
}

/// Map a single engine event onto DevUI SSE event(s), pushing onto `out`.
fn map_workflow_event(ev: &WorkflowEvent, ctx: &mut WfCtx, out: &mut Vec<Value>) {
    match ev {
        WorkflowEvent::ExecutorInvoked { executor_id } => {
            ctx.output_index += 1;
            let item_id = format!("exec_{}_{}", executor_id, util::short_hex());
            ctx.exec_items
                .insert(executor_id.clone(), (item_id.clone(), ctx.output_index));
            out.push(json!({
                "type": "response.output_item.added",
                "output_index": ctx.output_index,
                "sequence_number": ctx.next(),
                "item": {
                    "type": "executor_action",
                    "id": item_id,
                    "executor_id": executor_id,
                    "status": "in_progress",
                }
            }));
        }
        WorkflowEvent::ExecutorCompleted { executor_id } => {
            let (item_id, output_index) = ctx
                .exec_items
                .get(executor_id)
                .cloned()
                .unwrap_or_else(|| (format!("exec_{executor_id}"), ctx.output_index));
            out.push(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "sequence_number": ctx.next(),
                "item": {
                    "type": "executor_action",
                    "id": item_id,
                    "executor_id": executor_id,
                    "status": "completed",
                }
            }));
        }
        WorkflowEvent::ExecutorFailed { executor_id, error } => {
            let (item_id, output_index) = ctx
                .exec_items
                .get(executor_id)
                .cloned()
                .unwrap_or_else(|| (format!("exec_{executor_id}"), ctx.output_index));
            out.push(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "sequence_number": ctx.next(),
                "item": {
                    "type": "executor_action",
                    "id": item_id,
                    "executor_id": executor_id,
                    "status": "failed",
                    "error": { "message": error },
                }
            }));
        }
        WorkflowEvent::Output {
            data,
            source_executor_id,
        } => {
            ctx.output_index += 1;
            out.push(json!({
                "type": "response.output_item.added",
                "output_index": ctx.output_index,
                "sequence_number": ctx.next(),
                "item": {
                    "type": "message",
                    "id": util::msg_id(),
                    "role": "assistant",
                    "content": [ { "type": "output_text", "text": value_to_text(data), "annotations": [] } ],
                    "status": "completed",
                    "metadata": { "source_executor_id": source_executor_id },
                }
            }));
        }
        WorkflowEvent::Intermediate {
            data,
            source_executor_id,
        } => {
            // Non-terminal progress signal, analogous to `Output` but never
            // recorded as the run's final output: surfaced as a workflow
            // debug event rather than an output message item.
            out.push(json!({
                "type": "response.workflow_event.completed",
                "item_id": ctx.item_id,
                "output_index": ctx.output_index.max(0),
                "sequence_number": ctx.next(),
                "data": {
                    "event_type": "WorkflowIntermediateEvent",
                    "source_executor_id": source_executor_id,
                    "data": data,
                },
            }));
        }
        WorkflowEvent::RequestInfo {
            request_id,
            source_executor_id,
            request_data,
        } => {
            out.push(json!({
                "type": "response.request_info.requested",
                "request_id": request_id,
                "source_executor_id": source_executor_id,
                "request_data": request_data,
                "item_id": ctx.item_id,
                "output_index": ctx.output_index.max(0),
                "sequence_number": ctx.next(),
            }));
        }
        other => {
            // Started, Status, SuperStep*, AgentRun*, Custom, Failed → the
            // catch-all workflow debug event (DevUI's
            // `response.workflow_event.completed`).
            out.push(json!({
                "type": "response.workflow_event.completed",
                "item_id": ctx.item_id,
                "output_index": ctx.output_index.max(0),
                "sequence_number": ctx.next(),
                "data": workflow_event_data(other),
            }));
        }
    }
}

/// The `data` payload for a catch-all `response.workflow_event.completed`.
fn workflow_event_data(ev: &WorkflowEvent) -> Value {
    match ev {
        WorkflowEvent::Started => json!({ "event_type": "WorkflowStartedEvent" }),
        WorkflowEvent::Status(state) => json!({
            "event_type": "WorkflowStatusEvent",
            "state": serde_json::to_value(state).unwrap_or(Value::Null),
        }),
        WorkflowEvent::SuperStepStarted(n) => {
            json!({ "event_type": "SuperStepStartedEvent", "step": n })
        }
        WorkflowEvent::SuperStepCompleted(n) => {
            json!({ "event_type": "SuperStepCompletedEvent", "step": n })
        }
        WorkflowEvent::AgentRunUpdate {
            executor_id,
            update,
        } => json!({
            "event_type": "AgentRunUpdateEvent",
            "executor_id": executor_id,
            "data": update,
        }),
        WorkflowEvent::AgentRun {
            executor_id,
            response,
        } => json!({
            "event_type": "AgentRunEvent",
            "executor_id": executor_id,
            "data": response,
        }),
        WorkflowEvent::Custom(v) => json!({ "event_type": "CustomEvent", "data": v }),
        WorkflowEvent::Failed { error } => {
            json!({ "event_type": "WorkflowFailedEvent", "message": error })
        }
        // Handled by the caller; included for completeness.
        WorkflowEvent::ExecutorInvoked { executor_id } => {
            json!({ "event_type": "ExecutorInvokedEvent", "executor_id": executor_id })
        }
        WorkflowEvent::ExecutorCompleted { executor_id } => {
            json!({ "event_type": "ExecutorCompletedEvent", "executor_id": executor_id })
        }
        WorkflowEvent::ExecutorFailed { executor_id, error } => json!({
            "event_type": "ExecutorFailedEvent", "executor_id": executor_id, "error": error,
        }),
        WorkflowEvent::Output {
            data,
            source_executor_id,
        } => json!({
            "event_type": "WorkflowOutputEvent",
            "source_executor_id": source_executor_id,
            "data": data,
        }),
        WorkflowEvent::RequestInfo {
            request_id,
            source_executor_id,
            request_data,
        } => json!({
            "event_type": "RequestInfoEvent",
            "request_id": request_id,
            "source_executor_id": source_executor_id,
            "request_data": request_data,
        }),
        WorkflowEvent::Intermediate {
            data,
            source_executor_id,
        } => json!({
            "event_type": "WorkflowIntermediateEvent",
            "source_executor_id": source_executor_id,
            "data": data,
        }),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn execution_error(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(openai_error(
            format!("Request execution failed: {message}"),
            "server_error",
            None,
        )),
    )
        .into_response()
}

/// DevUI's fallback usage estimate (~4 characters per token) for runs that
/// report no usage details, applied on top of [`responses_from_run`]'s
/// pass-through usage mapping.
fn usage_estimate(input_len: usize, output_len: usize) -> Usage {
    let input = (input_len / 4) as u64;
    let output = (output_len / 4) as u64;
    Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + output,
        input_tokens_details: InputTokensDetails { cached_tokens: 0 },
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 0,
        },
    }
}

/// Approximate the character length of the request input (for usage estimates).
fn approx_input_len(input: &Value) -> usize {
    match input {
        Value::String(s) => s.len(),
        other => other.to_string().len(),
    }
}

/// Convert an arbitrary JSON output value to display text.
fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Reduce the OpenAI-style `input` to a value fed to `Workflow::run`.
///
/// Strings and structured objects pass through; a message array is flattened to
/// its concatenated user text so a string-typed start executor still works.
fn input_to_workflow_value(input: &Value) -> Value {
    match input {
        Value::String(s) => Value::String(s.clone()),
        Value::Null => Value::String(String::new()),
        Value::Array(_) => {
            let text = crate::responses::input_to_messages(input)
                .iter()
                .map(Message::text)
                .collect::<Vec<_>>()
                .join("\n");
            Value::String(text)
        }
        other => other.clone(),
    }
}
