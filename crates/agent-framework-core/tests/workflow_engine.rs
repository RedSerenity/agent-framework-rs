//! Integration tests for the upgraded workflow engine: HITL request/response,
//! shared state, checkpointing (in-memory + file), validation, visualization,
//! sub-workflows, and streaming.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_framework_core::error::Result;
use agent_framework_core::prelude::{AgentResponse, ChatResponse, Message, SupportsAgentRun};
use agent_framework_core::session::AgentSession;
use agent_framework_core::workflow::{
    get_checkpoint_summary, validate_workflow_graph, AgentExecutor, Case, CheckpointStorage,
    Default as SwitchDefault, EdgeGroup, Executor, FileCheckpointStorage, FunctionExecutor,
    InMemoryCheckpointStorage, RequestInfoExecutor, RequestResponse, ValidationType, Workflow,
    WorkflowBuilder, WorkflowCheckpoint, WorkflowContext, WorkflowEvent, WorkflowExecutor,
    WorkflowRunState,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

// ----------------------------------------------------------------------------
// Human-in-the-loop: pause -> pending_requests -> send_responses -> completion
// ----------------------------------------------------------------------------

/// Build a workflow whose start executor asks a question through a
/// `RequestInfoExecutor` and yields the human's answer once it arrives.
fn hitl_workflow() -> Workflow {
    let asker = FunctionExecutor::new("asker", |msg, ctx| async move {
        if let Some(resp) = RequestResponse::from_message(&msg) {
            // The response was routed back to us: emit it as the final answer.
            ctx.yield_output(resp.data).await?;
        } else {
            // Fresh input: forward the question to the request node.
            ctx.send_message(msg).await?;
        }
        Ok(())
    });
    let request_node = RequestInfoExecutor::new("request_node");

    WorkflowBuilder::new()
        .add_executor(Arc::new(asker))
        .add_executor(Arc::new(request_node))
        .set_start("asker")
        .add_edge("asker", "request_node")
        .build()
        .unwrap()
}

#[tokio::test]
async fn hitl_pause_and_resume() {
    let workflow = hitl_workflow();

    let mut run = workflow.run(json!("what is your name?")).await.unwrap();

    // The run pauses awaiting external input.
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].request_data, json!("what is your name?"));
    assert_eq!(pending[0].source_executor_id, "request_node");

    // A RequestInfo event was surfaced.
    assert!(run
        .events()
        .iter()
        .any(|e| matches!(e, WorkflowEvent::RequestInfo { .. })));

    // Supply the answer; the run resumes and completes.
    let request_id = pending[0].request_id.clone();
    run.send_response(request_id, json!("Ada")).await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(run.last_output(), Some(json!("Ada")));
}

#[tokio::test]
async fn hitl_send_responses_map() {
    let workflow = hitl_workflow();
    let mut run = workflow.run(json!("q")).await.unwrap();
    let id = run.pending_requests()[0].request_id.clone();

    let mut responses = HashMap::new();
    responses.insert(id, json!("answer"));
    run.send_responses(responses).await.unwrap();

    assert_eq!(run.last_output(), Some(json!("answer")));
    assert!(run.pending_requests().is_empty());
}

// ----------------------------------------------------------------------------
// Shared state is visible across executors within a run
// ----------------------------------------------------------------------------

#[tokio::test]
async fn shared_state_visible_across_executors() {
    let writer = FunctionExecutor::new("writer", |msg, ctx| async move {
        ctx.shared_state().set("greeting", json!("hello")).await;
        ctx.send_message(msg).await?;
        Ok(())
    });
    let reader = FunctionExecutor::new("reader", |_msg, ctx| async move {
        let g = ctx
            .shared_state()
            .get("greeting")
            .await
            .unwrap_or(json!(null));
        ctx.yield_output(g).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(writer))
        .add_executor(Arc::new(reader))
        .set_start("writer")
        .add_edge("writer", "reader")
        .build()
        .unwrap();

    let run = workflow.run(json!("go")).await.unwrap();
    assert_eq!(run.last_output(), Some(json!("hello")));
    // The run handle exposes the same shared state.
    assert_eq!(
        run.shared_state().get("greeting").await,
        Some(json!("hello"))
    );
}

// ----------------------------------------------------------------------------
// Validation: duplicate edge and unreachable node
// ----------------------------------------------------------------------------

fn noop(id: &str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id.to_string(), |_m, _c| async {
        Ok(())
    }))
}

#[tokio::test]
async fn validation_rejects_duplicate_edge() {
    let err = WorkflowBuilder::new()
        .add_executor(noop("a"))
        .add_executor(noop("b"))
        .set_start("a")
        .add_edge("a", "b")
        .add_edge("a", "b")
        .build()
        .err()
        .expect("expected a build error");
    assert!(
        err.to_string().contains("EDGE_DUPLICATION"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn validation_rejects_unreachable_node() {
    let err = WorkflowBuilder::new()
        .add_executor(noop("a"))
        .add_executor(noop("b"))
        .add_executor(noop("c")) // never connected
        .set_start("a")
        .add_edge("a", "b")
        .build()
        .err()
        .expect("expected a build error");
    let msg = err.to_string();
    assert!(
        msg.contains("GRAPH_CONNECTIVITY"),
        "unexpected error: {msg}"
    );
    assert!(
        msg.contains("\"c\""),
        "should name the unreachable node: {msg}"
    );
}

#[test]
fn validate_workflow_graph_returns_typed_error() {
    let mut execs: HashMap<String, Arc<dyn Executor>> = HashMap::new();
    execs.insert("a".into(), noop("a"));
    execs.insert("b".into(), noop("b"));
    execs.insert("c".into(), noop("c"));
    let groups = vec![EdgeGroup::Single {
        source: "a".into(),
        target: "b".into(),
        condition: None,
    }];
    let err = validate_workflow_graph(&execs, &groups, "a", &[], &[]).unwrap_err();
    assert_eq!(err.validation_type, ValidationType::GraphConnectivity);

    // Duplicate edge is also surfaced with the right category.
    let dup_groups = vec![
        EdgeGroup::Single {
            source: "a".into(),
            target: "b".into(),
            condition: None,
        },
        EdgeGroup::Single {
            source: "a".into(),
            target: "b".into(),
            condition: None,
        },
    ];
    let mut ab: HashMap<String, Arc<dyn Executor>> = HashMap::new();
    ab.insert("a".into(), noop("a"));
    ab.insert("b".into(), noop("b"));
    let err = validate_workflow_graph(&ab, &dup_groups, "a", &[], &[]).unwrap_err();
    assert_eq!(err.validation_type, ValidationType::EdgeDuplication);
}

// ----------------------------------------------------------------------------
// Visualization: Mermaid + Graphviz DOT
// ----------------------------------------------------------------------------

fn viz_workflow() -> Workflow {
    WorkflowBuilder::new()
        .add_executor(noop("a"))
        .add_executor(noop("b"))
        .add_executor(noop("c"))
        .add_executor(noop("d"))
        .add_executor(noop("joiner"))
        .set_start("a")
        .add_conditional_edge("a", "b", |_m| true)
        .add_switch(
            "a",
            vec![Case::labeled(|_m| true, "c", "hot")],
            SwitchDefault::new("d"),
        )
        .add_fan_in(vec!["c".to_string(), "d".to_string()], "joiner")
        .build()
        .unwrap()
}

#[test]
fn viz_mermaid_snapshot() {
    let workflow = viz_workflow();
    let mermaid = workflow.viz().to_mermaid();

    for expected in [
        "flowchart TD",
        "a[\"a (Start)\"]",
        "a -. conditional .-> b",
        "a -- \"hot\" --> c",
        "a -- \"default\" --> d",
        "fan_in_joiner_0((fan-in))",
        "c --> fan_in_joiner_0",
        "d --> fan_in_joiner_0",
        "fan_in_joiner_0 --> joiner",
    ] {
        assert!(
            mermaid.contains(expected),
            "mermaid missing `{expected}`:\n{mermaid}"
        );
    }
}

#[test]
fn viz_dot_snapshot() {
    let workflow = viz_workflow();
    let dot = workflow.viz().to_dot();

    for expected in [
        "digraph Workflow {",
        "\"a\" [fillcolor=lightgreen, label=\"a\\n(Start)\"];",
        "\"a\" -> \"b\" [style=dashed, label=\"conditional\"];",
        "\"a\" -> \"c\" [label=\"hot\"];",
        "\"a\" -> \"d\" [label=\"default\"];",
        "shape=ellipse, fillcolor=lightgoldenrod, label=\"fan-in\"",
        "\"c\" -> \"fan_in_joiner_0\";",
        "\"fan_in_joiner_0\" -> \"joiner\";",
    ] {
        assert!(dot.contains(expected), "dot missing `{expected}`:\n{dot}");
    }
}

// ----------------------------------------------------------------------------
// run_stream: events are streamed in deterministic order
// ----------------------------------------------------------------------------

fn tag(event: &WorkflowEvent) -> String {
    match event {
        WorkflowEvent::Started => "Started".into(),
        WorkflowEvent::Status(s) => format!("Status({s:?})"),
        WorkflowEvent::SuperStepStarted(i) => format!("SuperStepStarted({i})"),
        WorkflowEvent::SuperStepCompleted(i) => format!("SuperStepCompleted({i})"),
        WorkflowEvent::ExecutorInvoked { executor_id } => format!("Invoked({executor_id})"),
        WorkflowEvent::ExecutorCompleted { executor_id } => format!("Completed({executor_id})"),
        WorkflowEvent::ExecutorFailed { executor_id, .. } => format!("Failed({executor_id})"),
        WorkflowEvent::AgentRunUpdate { .. } => "AgentRunUpdate".into(),
        WorkflowEvent::AgentRun { .. } => "AgentRun".into(),
        WorkflowEvent::Output { .. } => "Output".into(),
        WorkflowEvent::Intermediate { .. } => "Intermediate".into(),
        WorkflowEvent::Custom(_) => "Custom".into(),
        WorkflowEvent::RequestInfo { .. } => "RequestInfo".into(),
        WorkflowEvent::Failed { .. } => "Failed".into(),
    }
}

#[tokio::test]
async fn run_stream_event_ordering() {
    let doubler = FunctionExecutor::new("double", |msg, ctx| async move {
        let n = msg.as_i64().unwrap_or(0);
        ctx.send_message(json!(n * 2)).await?;
        Ok(())
    });
    let out = FunctionExecutor::new("out", |msg, ctx| async move {
        ctx.yield_output(msg).await?;
        Ok(())
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(doubler))
        .add_executor(Arc::new(out))
        .set_start("double")
        .add_edge("double", "out")
        .build()
        .unwrap();

    let mut stream = workflow.run_stream(json!(21));
    let mut tags = Vec::new();
    while let Some(event) = stream.next().await {
        tags.push(tag(&event));
    }

    assert_eq!(
        tags,
        vec![
            "Started",
            "Status(InProgress)",
            "SuperStepStarted(1)",
            "Invoked(double)",
            "Completed(double)",
            "SuperStepCompleted(1)",
            "SuperStepStarted(2)",
            "Invoked(out)",
            "Output",
            "Completed(out)",
            "SuperStepCompleted(2)",
            "Status(Idle)",
        ]
    );

    // The final run state is recoverable after the stream ends.
    let run = stream.into_run().await.unwrap();
    assert_eq!(run.last_output(), Some(json!(42)));
    assert_eq!(run.state(), WorkflowRunState::Idle);
}

// ----------------------------------------------------------------------------
// Per-executor serialization within a superstep (upstream PR #6776)
// ----------------------------------------------------------------------------

/// An executor that records the peak number of concurrently-running
/// `execute()` calls it observes. It yields to the runtime while "inside"
/// `execute` so a racing sibling call on the *same* instance would be seen.
struct ConcurrencyProbe {
    id: String,
    in_flight: Arc<std::sync::atomic::AtomicUsize>,
    max_in_flight: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait]
impl Executor for ConcurrencyProbe {
    fn id(&self) -> &str {
        &self.id
    }
    async fn execute(&self, _message: Value, _ctx: WorkflowContext) -> Result<()> {
        use std::sync::atomic::Ordering;
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        // Give any concurrently-scheduled sibling call ample opportunity to
        // interleave before we lower the in-flight count.
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Two messages delivered to the same executor instance in one superstep must
/// have their `execute()` calls serialized, even though distinct executors run
/// concurrently. Without per-executor serialization the two calls overlap and
/// `max_in_flight` reaches 2.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_executor_deliveries_serialize_within_a_superstep() {
    // `src` emits two messages down the single edge to `sink`, so in the next
    // superstep `sink` receives two separate (non-fan-in) deliveries.
    let src = FunctionExecutor::new("src", |_msg, ctx| async move {
        ctx.send_message(json!(1)).await?;
        ctx.send_message(json!(2)).await?;
        Ok(())
    });
    let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sink = ConcurrencyProbe {
        id: "sink".into(),
        in_flight: in_flight.clone(),
        max_in_flight: max_in_flight.clone(),
    };

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(src))
        .add_executor(Arc::new(sink))
        .set_start("src")
        .add_edge("src", "sink")
        .build()
        .unwrap();

    let run = workflow.run(json!(0)).await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(
        max_in_flight.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "two deliveries to the same executor must not run concurrently"
    );
}

// ----------------------------------------------------------------------------
// Checkpointing: save -> restore -> resume (both storages) + executor state
// ----------------------------------------------------------------------------

/// A stateful executor: accumulates the sum of inputs and round-trips it.
struct Counter {
    id: String,
    count: Mutex<i64>,
}

#[async_trait]
impl Executor for Counter {
    fn id(&self) -> &str {
        &self.id
    }
    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        let n = message.as_i64().unwrap_or(0);
        let total = {
            let mut c = self.count.lock().unwrap();
            *c += n;
            *c
        };
        ctx.yield_output(json!(total)).await?;
        Ok(())
    }
    async fn snapshot_state(&self) -> Option<Value> {
        Some(json!({ "count": *self.count.lock().unwrap() }))
    }
    async fn restore_state(&self, state: Value) -> Result<()> {
        if let Some(n) = state.get("count").and_then(|v| v.as_i64()) {
            *self.count.lock().unwrap() = n;
        }
        Ok(())
    }
}

/// A 3-stage pipeline that accumulates into shared state, yielding the total.
fn build_pipeline(storage: Option<Arc<dyn CheckpointStorage>>) -> Workflow {
    let p1 = FunctionExecutor::new("p1", |msg, ctx| async move {
        let n = msg.as_i64().unwrap_or(0);
        ctx.shared_state()
            .update("sum", move |cur| {
                let c = cur.and_then(|v| v.as_i64()).unwrap_or(0);
                json!(c + n)
            })
            .await;
        ctx.send_message(json!(n)).await?;
        Ok(())
    });
    let p2 = FunctionExecutor::new("p2", |msg, ctx| async move {
        ctx.send_message(msg).await?;
        Ok(())
    });
    let p3 = FunctionExecutor::new("p3", |_msg, ctx| async move {
        let sum = ctx.shared_state().get("sum").await.unwrap_or(json!(0));
        ctx.yield_output(sum).await?;
        Ok(())
    });

    let mut builder = WorkflowBuilder::new()
        .add_executor(Arc::new(p1))
        .add_executor(Arc::new(p2))
        .add_executor(Arc::new(p3))
        .set_start("p1")
        .add_edge("p1", "p2")
        .add_edge("p2", "p3");
    if let Some(s) = storage {
        builder = builder.with_checkpointing(s);
    }
    builder.build().unwrap()
}

async fn pipeline_roundtrip(storage: Arc<dyn CheckpointStorage>) {
    let workflow = build_pipeline(Some(storage.clone()));
    let run = workflow.run(json!(10)).await.unwrap();
    assert_eq!(run.last_output(), Some(json!(10)));

    // A mid-run checkpoint has an in-flight message and iteration_count == 1.
    let checkpoints = storage.list(None).await.unwrap();
    let mid = checkpoints
        .iter()
        .find(|c| c.iteration_count == 1)
        .expect("a mid-run checkpoint");
    assert!(!mid.messages.is_empty());

    let summary = get_checkpoint_summary(mid);
    assert_eq!(summary.iteration_count, 1);
    assert_eq!(summary.status, "awaiting next superstep");

    // Restore into a fresh, identical workflow and drive to completion.
    let resumed = build_pipeline(Some(storage.clone()));
    let run2 = resumed
        .run_from_checkpoint(&mid.checkpoint_id, storage.clone())
        .await
        .unwrap();
    assert_eq!(run2.state(), WorkflowRunState::Idle);
    assert_eq!(run2.last_output(), Some(json!(10)));
}

async fn counter_state_roundtrip(storage: Arc<dyn CheckpointStorage>) {
    let counter = Arc::new(Counter {
        id: "counter".into(),
        count: Mutex::new(0),
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(counter.clone() as Arc<dyn Executor>)
        .set_start("counter")
        .with_checkpointing(storage.clone())
        .build()
        .unwrap();

    let run = workflow.run(json!(5)).await.unwrap();
    assert_eq!(run.last_output(), Some(json!(5)));
    assert_eq!(*counter.count.lock().unwrap(), 5);

    let checkpoints = storage.list(None).await.unwrap();
    let cp = checkpoints
        .iter()
        .find(|c| c.executor_states.contains_key("counter"))
        .expect("a checkpoint capturing executor state");
    assert_eq!(cp.executor_states["counter"], json!({ "count": 5 }));

    // A fresh counter starts at 0; restoring must bring it to 5.
    let counter2 = Arc::new(Counter {
        id: "counter".into(),
        count: Mutex::new(0),
    });
    let resumed = WorkflowBuilder::new()
        .add_executor(counter2.clone() as Arc<dyn Executor>)
        .set_start("counter")
        .build()
        .unwrap();
    let run2 = resumed
        .run_from_checkpoint(&cp.checkpoint_id, storage.clone())
        .await
        .unwrap();
    assert_eq!(run2.state(), WorkflowRunState::Idle);
    assert_eq!(*counter2.count.lock().unwrap(), 5);
}

#[tokio::test]
async fn checkpoint_roundtrip_in_memory() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    pipeline_roundtrip(storage.clone()).await;

    let storage2: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    counter_state_roundtrip(storage2).await;
}

#[tokio::test]
async fn checkpoint_roundtrip_file() {
    let dir = std::env::temp_dir().join(format!("af_ckpt_{}", uuid::Uuid::new_v4()));
    let storage: Arc<dyn CheckpointStorage> = Arc::new(FileCheckpointStorage::new(&dir).unwrap());

    pipeline_roundtrip(storage.clone()).await;
    counter_state_roundtrip(storage.clone()).await;

    // Persistence: a brand-new storage handle over the same directory can load.
    let counter_cp = {
        let fresh = FileCheckpointStorage::new(&dir).unwrap();
        let all = fresh.list(None).await.unwrap();
        assert!(!all.is_empty(), "checkpoints should persist on disk");
        all.into_iter()
            .find(|c| c.executor_states.contains_key("counter"))
            .expect("a persisted counter checkpoint")
    };
    assert_eq!(counter_cp.executor_states["counter"], json!({ "count": 5 }));

    // Deleting removes the file.
    assert!(storage.delete(&counter_cp.checkpoint_id).await.unwrap());
    assert!(storage
        .load(&counter_cp.checkpoint_id)
        .await
        .unwrap()
        .is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

// ----------------------------------------------------------------------------
// Sub-workflows: output forwarding and request interception/forwarding
// ----------------------------------------------------------------------------

#[tokio::test]
async fn sub_workflow_forwards_output() {
    let child = WorkflowBuilder::new()
        .add_executor(Arc::new(FunctionExecutor::new(
            "c1",
            |msg, ctx| async move {
                let n = msg.as_i64().unwrap_or(0);
                ctx.yield_output(json!(n + 100)).await?;
                Ok(())
            },
        )))
        .set_start("c1")
        .build()
        .unwrap();

    let sink = FunctionExecutor::new("sink", |msg, ctx| async move {
        ctx.yield_output(msg).await?;
        Ok(())
    });

    let parent = WorkflowBuilder::new()
        .add_executor(Arc::new(WorkflowExecutor::new("wrapper", child)))
        .add_executor(Arc::new(sink))
        .set_start("wrapper")
        .add_edge("wrapper", "sink")
        .build()
        .unwrap();

    let run = parent.run(json!(5)).await.unwrap();
    assert_eq!(run.last_output(), Some(json!(105)));
}

#[tokio::test]
async fn sub_workflow_forwards_and_answers_requests() {
    // Child asks a question via a request node, then yields the answer.
    let child = {
        let casker = FunctionExecutor::new("casker", |msg, ctx| async move {
            if let Some(resp) = RequestResponse::from_message(&msg) {
                ctx.yield_output(resp.data).await?;
            } else {
                ctx.send_message(msg).await?;
            }
            Ok(())
        });
        WorkflowBuilder::new()
            .add_executor(Arc::new(casker))
            .add_executor(Arc::new(RequestInfoExecutor::new("creq")))
            .set_start("casker")
            .add_edge("casker", "creq")
            .build()
            .unwrap()
    };

    let psink = FunctionExecutor::new("psink", |msg, ctx| async move {
        ctx.yield_output(msg).await?;
        Ok(())
    });
    let parent = WorkflowBuilder::new()
        .add_executor(Arc::new(WorkflowExecutor::new("wrapper", child)))
        .add_executor(Arc::new(psink))
        .set_start("wrapper")
        .add_edge("wrapper", "psink")
        .build()
        .unwrap();

    // The child's request is intercepted and re-emitted by the parent.
    let mut run = parent.run(json!("need-info")).await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].request_data, json!("need-info"));
    assert_eq!(pending[0].source_executor_id, "wrapper");

    // Answering via the parent routes the response into the child, whose output
    // is then forwarded back out through the parent.
    let id = pending[0].request_id.clone();
    run.send_response(id, json!("the-answer")).await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(run.last_output(), Some(json!("the-answer")));
}

// ----------------------------------------------------------------------------
// Events parity: AgentExecutor emits AgentRun / AgentRunUpdate events
// ----------------------------------------------------------------------------

/// A trivial agent that echoes a fixed reply, for exercising AgentExecutor.
struct MockAgent {
    id: String,
    reply: String,
}

#[async_trait]
impl SupportsAgentRun for MockAgent {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _thread: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        Ok(AgentResponse::from_chat_response(ChatResponse::from_text(
            &self.reply,
        )))
    }
    fn id(&self) -> &str {
        &self.id
    }
}

#[tokio::test]
async fn agent_executor_emits_agent_events() {
    let agent = Arc::new(MockAgent {
        id: "m".into(),
        reply: "hello".into(),
    }) as Arc<dyn SupportsAgentRun>;
    let exec = AgentExecutor::new("a1", agent).with_output(true);

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(exec))
        .set_start("a1")
        .build()
        .unwrap();

    let run = workflow.run(json!("hi")).await.unwrap();

    assert!(
        run.events()
            .iter()
            .any(|e| matches!(e, WorkflowEvent::AgentRun { .. })),
        "expected an AgentRun event"
    );
    assert!(
        run.events()
            .iter()
            .any(|e| matches!(e, WorkflowEvent::AgentRunUpdate { .. })),
        "expected an AgentRunUpdate event"
    );
}

/// An agent that returns several messages, to exercise incremental per-update
/// `AgentRunUpdate` emission by `run_agent_and_emit`.
struct MultiMessageAgent;

#[async_trait]
impl SupportsAgentRun for MultiMessageAgent {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _thread: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        Ok(AgentResponse {
            messages: vec![
                Message::assistant("one"),
                Message::assistant("two"),
                Message::assistant("three"),
            ],
            ..Default::default()
        })
    }
    fn id(&self) -> &str {
        "multi"
    }
}

#[tokio::test]
async fn agent_executor_emits_incremental_agent_run_updates() {
    // The orchestration layer now drives `run_stream` and emits one
    // `AgentRunUpdate` per streamed update, then a single terminal `AgentRun`.
    let agent = Arc::new(MultiMessageAgent) as Arc<dyn SupportsAgentRun>;
    let exec = AgentExecutor::new("a1", agent).with_output(true);
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(exec))
        .set_start("a1")
        .build()
        .unwrap();

    let run = workflow.run(json!("hi")).await.unwrap();

    let update_count = run
        .events()
        .iter()
        .filter(|e| matches!(e, WorkflowEvent::AgentRunUpdate { .. }))
        .count();
    assert_eq!(update_count, 3, "one AgentRunUpdate per streamed update");
    let run_count = run
        .events()
        .iter()
        .filter(|e| matches!(e, WorkflowEvent::AgentRun { .. }))
        .count();
    assert_eq!(run_count, 1, "exactly one terminal AgentRun");
}

#[tokio::test]
async fn fanin_sink_request_info_response_bypasses_barrier() {
    // Two sources fan into a joiner; the joiner asks a question through
    // ctx.request_info(). The routed response targets the joiner directly and
    // must NOT be swallowed by the fan-in barrier (its source is the request
    // plumbing, not one of the fan-in edges).
    let split = FunctionExecutor::new("split", |msg, ctx| async move {
        ctx.send_message(msg).await?;
        Ok(())
    });
    let a = FunctionExecutor::new("a", |msg, ctx| async move {
        ctx.send_message(json!(format!("a:{}", msg.as_str().unwrap_or(""))))
            .await?;
        Ok(())
    });
    let b = FunctionExecutor::new("b", |msg, ctx| async move {
        ctx.send_message(json!(format!("b:{}", msg.as_str().unwrap_or(""))))
            .await?;
        Ok(())
    });
    let join = FunctionExecutor::new("join", |msg, ctx| async move {
        if let Some(resp) = RequestResponse::from_message(&msg) {
            ctx.yield_output(resp.data).await?;
        } else {
            // Barrier fired with both inputs: ask a human before finishing.
            ctx.request_info(json!({ "joined": msg })).await?;
        }
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(split))
        .add_executor(Arc::new(a))
        .add_executor(Arc::new(b))
        .add_executor(Arc::new(join))
        .set_start("split")
        .add_fan_out("split", vec!["a".to_string(), "b".to_string()])
        .add_fan_in(vec!["a".to_string(), "b".to_string()], "join")
        .build()
        .unwrap();

    let mut run = workflow.run(json!("x")).await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].source_executor_id, "join");

    let id = pending[0].request_id.clone();
    run.send_response(id, json!("approved")).await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(run.last_output(), Some(json!("approved")));
}

// ----------------------------------------------------------------------------
// Checkpointing: a fan-in partially satisfied across supersteps survives a
// resume (BUG: the buffered messages used to be dropped on checkpoint).
// ----------------------------------------------------------------------------

/// `split` fans out to `a` and `hop`; `a` reaches the `join` fan-in one
/// superstep before `b` (which sits behind the extra `hop`). So there is a
/// superstep boundary at which `join`'s barrier holds `a`'s message but not
/// `b`'s — exactly the state a checkpoint must preserve.
fn build_staggered_fanin(storage: Arc<dyn CheckpointStorage>) -> Workflow {
    let split = FunctionExecutor::new("split", |msg, ctx| async move {
        ctx.send_message(msg).await?;
        Ok(())
    });
    let a = FunctionExecutor::new("a", |_msg, ctx| async move {
        ctx.send_message(json!("a-done")).await?;
        Ok(())
    });
    let hop = FunctionExecutor::new("hop", |msg, ctx| async move {
        ctx.send_message(msg).await?;
        Ok(())
    });
    let b = FunctionExecutor::new("b", |_msg, ctx| async move {
        ctx.send_message(json!("b-done")).await?;
        Ok(())
    });
    let join = FunctionExecutor::new("join", |msg, ctx| async move {
        // The barrier fires with an array of both sources' payloads (source order).
        ctx.yield_output(msg).await?;
        Ok(())
    });

    WorkflowBuilder::new()
        .add_executor(Arc::new(split))
        .add_executor(Arc::new(a))
        .add_executor(Arc::new(hop))
        .add_executor(Arc::new(b))
        .add_executor(Arc::new(join))
        .set_start("split")
        .add_fan_out("split", vec!["a".to_string(), "hop".to_string()])
        .add_edge("hop", "b")
        .add_fan_in(vec!["a".to_string(), "b".to_string()], "join")
        .with_checkpointing(storage)
        .build()
        .unwrap()
}

#[tokio::test]
async fn checkpoint_preserves_partial_fanin_across_supersteps() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());

    // Baseline: the full run joins both inputs, in source order.
    let run = build_staggered_fanin(storage.clone())
        .run(json!("go"))
        .await
        .unwrap();
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(run.last_output(), Some(json!(["a-done", "b-done"])));

    // The superstep-3 checkpoint is taken while `a` has delivered to `join` but
    // `b` has not: the partial barrier must be captured in `fanin_state`.
    let cp = storage
        .list(None)
        .await
        .unwrap()
        .into_iter()
        .find(|c| c.iteration_count == 3)
        .expect("a checkpoint taken between the two fan-in deliveries");
    let join_buf = cp
        .fanin_state
        .get("join")
        .expect("join's partial fan-in buffer is captured");
    assert_eq!(
        join_buf.get("a"),
        Some(&json!("a-done")),
        "a's message is buffered"
    );
    assert!(
        !join_buf.contains_key("b"),
        "b has not delivered at this checkpoint"
    );

    // Resume from that mid-barrier checkpoint into a fresh, identical workflow:
    // the barrier still fires with BOTH inputs (it would silently never fire if
    // the buffered `a` message were lost on resume).
    let resumed = build_staggered_fanin(storage.clone());
    let run2 = resumed
        .run_from_checkpoint(&cp.checkpoint_id, storage.clone())
        .await
        .unwrap();
    assert_eq!(run2.state(), WorkflowRunState::Idle);
    assert_eq!(run2.last_output(), Some(json!(["a-done", "b-done"])));
}

#[tokio::test]
async fn legacy_checkpoint_without_fanin_state_loads() {
    // A checkpoint written before `fanin_state` existed omits the field
    // entirely; it must still deserialize (serde default = empty map).
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let _ = build_staggered_fanin(storage.clone())
        .run(json!("go"))
        .await
        .unwrap();
    let cp = storage
        .list(None)
        .await
        .unwrap()
        .into_iter()
        .find(|c| c.iteration_count == 3)
        .expect("a mid-barrier checkpoint");

    let mut value = serde_json::to_value(&cp).unwrap();
    assert!(
        value
            .as_object_mut()
            .unwrap()
            .remove("fanin_state")
            .is_some(),
        "sanity: the field is present before stripping"
    );
    let legacy: WorkflowCheckpoint = serde_json::from_value(value).unwrap();
    assert!(
        legacy.fanin_state.is_empty(),
        "a signatureless/fan-in-less checkpoint deserializes with an empty buffer"
    );
}

// ----------------------------------------------------------------------------
// Within-superstep execution is concurrent (BUG: it used to be sequential),
// while events and outputs stay deterministic.
// ----------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn superstep_executes_targets_concurrently() {
    use std::time::Duration;

    // Two fan-out targets that each sleep 100ms. Run sequentially the fan-out
    // superstep would take 200ms; run concurrently the two sleeps overlap and
    // it takes ~100ms of (paused) virtual time.
    fn slow(id: &str) -> FunctionExecutor {
        FunctionExecutor::new(id.to_string(), |_msg, ctx| async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            ctx.yield_output(json!("done")).await?;
            Ok(())
        })
    }
    let split = FunctionExecutor::new("split", |msg, ctx| async move {
        ctx.send_message(msg).await?;
        Ok(())
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(split))
        .add_executor(Arc::new(slow("a")))
        .add_executor(Arc::new(slow("b")))
        .set_start("split")
        .add_fan_out("split", vec!["a".to_string(), "b".to_string()])
        .build()
        .unwrap();

    let start = tokio::time::Instant::now();
    let run = workflow.run(json!("go")).await.unwrap();
    let elapsed = start.elapsed();

    // Exactly one sleep of virtual time: the two deliveries genuinely overlap.
    assert_eq!(
        elapsed,
        Duration::from_millis(100),
        "fan-out targets must run concurrently, not one-after-another"
    );
    assert_eq!(run.outputs().len(), 2, "both targets produced output");
}

#[tokio::test]
async fn fan_out_event_and_output_order_is_deterministic() {
    // The concurrent superstep must still emit events and outputs in a fixed
    // (sorted-target) order, so two runs of the same fan-out graph agree.
    fn build() -> Workflow {
        let split = FunctionExecutor::new("split", |msg, ctx| async move {
            ctx.send_message(msg).await?;
            Ok(())
        });
        let mk = |id: &'static str| {
            FunctionExecutor::new(id, move |_m, ctx| async move {
                ctx.yield_output(json!(id)).await?;
                Ok(())
            })
        };
        WorkflowBuilder::new()
            .add_executor(Arc::new(split))
            .add_executor(Arc::new(mk("a")))
            .add_executor(Arc::new(mk("b")))
            .add_executor(Arc::new(mk("c")))
            .set_start("split")
            .add_fan_out(
                "split",
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
            )
            .build()
            .unwrap()
    }

    let run1 = build().run(json!("go")).await.unwrap();
    let run2 = build().run(json!("go")).await.unwrap();

    let tags1: Vec<String> = run1.events().iter().map(tag).collect();
    let tags2: Vec<String> = run2.events().iter().map(tag).collect();
    assert_eq!(
        tags1, tags2,
        "the event sequence is identical across runs of the same fan-out graph"
    );

    // Outputs are ordered by (sorted) target, independent of completion order.
    assert_eq!(
        run1.outputs(),
        vec![json!("a"), json!("b"), json!("c")],
        "outputs follow sorted-target order"
    );
    assert_eq!(run2.outputs(), run1.outputs());
}

// ----------------------------------------------------------------------------
// Workflow output designation: output_from / intermediate_output_from
// ----------------------------------------------------------------------------

/// Two-stage pipeline, `first -> second`, both yielding a value. Used to probe
/// the interaction between `output_from`/`intermediate_output_from` and
/// `last_output`/`outputs`/`events`.
fn two_stage_yield_workflow(
    output_from: Option<Vec<&'static str>>,
    intermediate_from: Option<Vec<&'static str>>,
) -> Result<Workflow> {
    let first = FunctionExecutor::new("first", |_msg, ctx| async move {
        ctx.yield_output(json!("from-first")).await?;
        ctx.send_message(json!("go")).await?;
        Ok(())
    });
    let second = FunctionExecutor::new("second", |_msg, ctx| async move {
        ctx.yield_output(json!("from-second")).await?;
        Ok(())
    });

    let mut builder = WorkflowBuilder::new()
        .add_executor(Arc::new(first))
        .add_executor(Arc::new(second))
        .set_start("first")
        .add_edge("first", "second");
    if let Some(ids) = output_from {
        builder = builder.output_from(ids);
    }
    if let Some(ids) = intermediate_from {
        builder = builder.intermediate_output_from(ids);
    }
    builder.build()
}

#[tokio::test]
async fn default_output_designation_is_unchanged() {
    // With neither `output_from` nor `intermediate_output_from` configured,
    // every yield is a terminal `Output` and `last_output` is the last one
    // produced, exactly as before this feature existed.
    let workflow = two_stage_yield_workflow(None, None).unwrap();
    let run = workflow.run(json!("hi")).await.unwrap();

    assert_eq!(
        run.outputs(),
        vec![json!("from-first"), json!("from-second")]
    );
    assert_eq!(run.last_output(), Some(json!("from-second")));
    assert!(
        !run.events()
            .iter()
            .any(|e| matches!(e, WorkflowEvent::Intermediate { .. })),
        "no Intermediate events without a designation"
    );
}

#[tokio::test]
async fn intermediate_output_from_is_non_terminal_output_from_wins() {
    // `first` is marked intermediate-only; `second` is the designated output.
    // `first`'s yield must surface as `Intermediate` (never as the run's final
    // output), while `second`'s yield is the `Output` / `last_output`.
    let workflow = two_stage_yield_workflow(Some(vec!["second"]), Some(vec!["first"])).unwrap();
    let run = workflow.run(json!("hi")).await.unwrap();

    assert_eq!(
        run.outputs(),
        vec![json!("from-second")],
        "only the output_from executor's yield counts as Output"
    );
    assert_eq!(run.last_output(), Some(json!("from-second")));

    let intermediates: Vec<Value> = run
        .events()
        .iter()
        .filter_map(|e| e.as_intermediate().cloned())
        .collect();
    assert_eq!(
        intermediates,
        vec![json!("from-first")],
        "the intermediate_output_from executor's yield is Intermediate, not Output"
    );

    // Sanity: the intermediate value never leaks into last_output/outputs.
    assert_ne!(run.last_output(), Some(json!("from-first")));
    assert!(!run.outputs().contains(&json!("from-first")));
}

#[tokio::test]
async fn output_from_demotes_undesignated_executors_to_intermediate() {
    // Only `second` is designated as an output source; `first` is not listed
    // in either set. Per the documented precedence, its yield is demoted to a
    // non-terminal Intermediate rather than silently dropped.
    let workflow = two_stage_yield_workflow(Some(vec!["second"]), None).unwrap();
    let run = workflow.run(json!("hi")).await.unwrap();

    assert_eq!(run.outputs(), vec![json!("from-second")]);
    assert_eq!(run.last_output(), Some(json!("from-second")));
    assert!(run.events().iter().any(
        |e| matches!(e, WorkflowEvent::Intermediate { data, source_executor_id }
            if data == &json!("from-first") && source_executor_id == "first")
    ));
}

#[test]
fn output_designation_validation_rejects_overlap_and_unknown_ids() {
    let mut execs: HashMap<String, Arc<dyn Executor>> = HashMap::new();
    execs.insert("a".into(), noop("a"));
    execs.insert("b".into(), noop("b"));
    let groups = vec![EdgeGroup::Single {
        source: "a".into(),
        target: "b".into(),
        condition: None,
    }];

    // Overlapping designation.
    let overlap = vec!["a".to_string()];
    let err = validate_workflow_graph(&execs, &groups, "a", &overlap, &overlap).unwrap_err();
    assert_eq!(err.validation_type, ValidationType::OutputValidation);

    // Unknown id in output_from.
    let err = validate_workflow_graph(
        &execs,
        &groups,
        "a",
        &["not-a-real-executor".to_string()],
        &[],
    )
    .unwrap_err();
    assert_eq!(err.validation_type, ValidationType::OutputValidation);

    // Unknown id in intermediate_output_from.
    let err = validate_workflow_graph(
        &execs,
        &groups,
        "a",
        &[],
        &["not-a-real-executor".to_string()],
    )
    .unwrap_err();
    assert_eq!(err.validation_type, ValidationType::OutputValidation);

    // Disjoint, known ids: valid.
    assert!(
        validate_workflow_graph(&execs, &groups, "a", &["a".to_string()], &["b".to_string()],)
            .is_ok()
    );
}

#[test]
fn workflow_builder_rejects_overlapping_output_designation_at_build() {
    let first = FunctionExecutor::new("first", |_msg, ctx| async move {
        ctx.yield_output(json!("x")).await?;
        Ok(())
    });
    let result = WorkflowBuilder::new()
        .add_executor(Arc::new(first))
        .set_start("first")
        .output_from(["first"])
        .intermediate_output_from(["first"])
        .build();
    let err = match result {
        Ok(_) => panic!("expected build() to reject an overlapping output designation"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("OUTPUT_VALIDATION"));
}

// ----------------------------------------------------------------------------
// Runtime checkpoint storage (upstream `run(checkpoint_storage=...)`)
// ----------------------------------------------------------------------------

/// A workflow built with no checkpointing still checkpoints into storage
/// supplied for one run, and a run resumed from that storage keeps writing
/// to it — so a run that pauses twice can be resumed twice, each time from
/// the storage alone.
#[tokio::test]
async fn runtime_checkpoint_storage_survives_two_pauses() {
    let asker = FunctionExecutor::new("asker", |msg, ctx| async move {
        match RequestResponse::from_message(&msg) {
            Some(resp) if resp.data == json!("first") => ctx.send_message(json!("q2")).await?,
            Some(resp) => ctx.yield_output(resp.data).await?,
            None => ctx.send_message(msg).await?,
        }
        Ok(())
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(asker))
        .add_executor(Arc::new(RequestInfoExecutor::new("ask")))
        .set_start("asker")
        .add_edge("asker", "ask")
        .build()
        .unwrap();
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let latest = |storage: Arc<dyn CheckpointStorage>| async move {
        storage
            .list(None)
            .await
            .unwrap()
            .into_iter()
            .max_by_key(|c| (c.timestamp_millis, c.iteration_count))
            .unwrap()
            .checkpoint_id
    };

    let run = workflow
        .run_with_checkpointing(json!("q1"), storage.clone())
        .await
        .unwrap();
    let first = run.pending_requests()[0].request_id.clone();
    drop(run);

    let mut run = workflow
        .run_from_checkpoint(&latest(storage.clone()).await, storage.clone())
        .await
        .unwrap();
    run.send_response(first, json!("first")).await.unwrap();
    let second = run.pending_requests()[0].clone();
    assert_eq!(second.request_data, json!("q2"));
    drop(run);

    let mut run = workflow
        .run_from_checkpoint(&latest(storage.clone()).await, storage)
        .await
        .unwrap();
    run.send_response(second.request_id, json!("Ada"))
        .await
        .unwrap();
    assert_eq!(run.last_output(), Some(json!("Ada")));
}
