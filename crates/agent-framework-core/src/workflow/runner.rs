//! The workflow builder, the [`Workflow`] type, the [`WorkflowRun`] handle, and
//! the superstep runner.

use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use super::checkpoint::{CheckpointStorage, WorkflowCheckpoint};
use super::context::{DrainedEffects, WorkflowContext, WorkflowMessage};
use super::edge::{
    wrap_async_condition, wrap_sync_condition, Case, Default as SwitchDefault, EdgeGroup, Selection,
};
use super::events::{WorkflowEvent, WorkflowRunState};
use super::executor::Executor;
use super::request_info::{PendingRequest, RequestResponse};
use super::shared_state::SharedState;
use super::validation::validate_workflow_graph;
use super::viz::WorkflowViz;
use crate::error::{Error, Result};
use crate::tools::BoxFuture;

const DEFAULT_MAX_ITERATIONS: usize = 100;

/// FNV-1a (64-bit) hash of `bytes`. Implemented inline to avoid a hashing
/// dependency; used only to fingerprint a workflow's topology, so collision
/// resistance beyond "changed graphs almost always differ" is not required.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// A stable, normalized descriptor for one edge group.
///
/// Opaque runtime closures (a [`EdgeGroup::Single`] condition, a
/// [`EdgeGroup::FanOut`] selection) cannot be inspected, so they contribute
/// only their *kind* — whether a condition/selection is present, plus any
/// declared switch-case labels — never their behavior. Target and source lists
/// are sorted so the descriptor is independent of declaration order.
///
/// **JSON-encoded, not string-joined.** Executor ids are only required to be
/// non-empty, so a descriptor built by joining them with `,` and `->` is not
/// injective: a fan-in from sources `["a", "b"]` and one from the single
/// source `["a,b"]` render identically, as do `["x->t", "y"]` and
/// `["x", "t,y"]` into the right target. Two different graphs would then hash
/// to one signature, and a checkpoint taken against either would be accepted
/// for the other — resuming a run onto a topology it was never written for,
/// which is the precise failure the signature exists to prevent. Upstream hit
/// the same collision in its edge-state keys (#7948) and fixed it the same
/// way. JSON quotes and escapes each id, so no id can impersonate a
/// separator.
fn edge_group_descriptor(group: &EdgeGroup) -> String {
    let value = match group {
        EdgeGroup::Single {
            source,
            target,
            condition,
        } => {
            let kind = if condition.is_some() {
                "conditional"
            } else {
                "plain"
            };
            serde_json::json!(["single", source, target, kind])
        }
        EdgeGroup::FanOut {
            source,
            targets,
            selection,
            case_labels,
        } => {
            let mut targets = targets.clone();
            targets.sort_unstable();
            let kind = if selection.is_some() {
                "selection"
            } else {
                "broadcast"
            };
            let labels = case_labels.as_ref().map(|labels| {
                let mut labels = labels.clone();
                labels.sort_unstable();
                labels
            });
            serde_json::json!(["fanout", source, targets, kind, labels])
        }
        EdgeGroup::FanIn { sources, target } => {
            let mut sources = sources.clone();
            sources.sort_unstable();
            serde_json::json!(["fanin", sources, target])
        }
    };
    // Arrays and strings only, so serialization cannot fail on a map whose
    // key order varies.
    value.to_string()
}

/// Compute a deterministic signature of a built workflow's graph.
///
/// The signature is the FNV-1a hash (hex) of a canonical rendering of the
/// start executor, the sorted executor ids, and the sorted, normalized
/// [`edge_group_descriptor`]s. It is stable across processes and independent of
/// executor/edge insertion order, and changes whenever a node or edge is
/// added, removed, retargeted, or has its condition/selection *presence* or
/// switch labels change. Because conditions/selections are opaque closures,
/// changing only a predicate's *body* (same presence, same labels) does not
/// change the signature — documented, and acceptable for a
/// resume-compatibility guard.
/// The signature scheme's version tag.
///
/// Bumped from `v1` when the canonical rendering became JSON: every signature
/// changes, so a checkpoint written by an earlier version cannot match. The
/// tag is what lets [`WorkflowRun::check_graph_signature`] say *that* rather
/// than reporting it as a graph change, which is what the mismatch message
/// would otherwise claim — and would be wrong about.
const SIGNATURE_SCHEME: &str = "v2";

pub(crate) fn compute_graph_signature(
    executors: &HashMap<String, Arc<dyn Executor>>,
    edge_groups: &[EdgeGroup],
    start: &str,
) -> String {
    let mut ids: Vec<&str> = executors.keys().map(String::as_str).collect();
    ids.sort_unstable();

    let mut edges: Vec<String> = edge_groups.iter().map(edge_group_descriptor).collect();
    edges.sort_unstable();

    // JSON throughout, for the reason `edge_group_descriptor` documents: an
    // executor id may contain whatever a caller put in it, including the
    // separators a joined rendering relies on. Node ids joined with `,` had
    // the same collision as the edge descriptors — `{"a", "b"}` and `{"a,b"}`
    // render identically — so both are encoded rather than only the half
    // upstream's own fix touched.
    let canonical = serde_json::json!({
        "start": start,
        "nodes": ids,
        "edges": edges,
    })
    .to_string();

    format!("{SIGNATURE_SCHEME}-{:016x}", fnv1a_64(canonical.as_bytes()))
}

/// Immutable, shared definition of a built workflow graph. Held behind an `Arc`
/// so a [`WorkflowRun`] can be driven on its own (including in a spawned task).
pub(crate) struct WorkflowShared {
    pub executors: HashMap<String, Arc<dyn Executor>>,
    pub edge_groups: Vec<EdgeGroup>,
    pub start: String,
    pub max_iterations: usize,
    pub name: Option<String>,
    pub description: Option<String>,
    pub checkpoint_storage: Option<Arc<dyn CheckpointStorage>>,
    pub id: String,
    /// Deterministic fingerprint of the graph topology, checked on resume.
    pub graph_signature: String,
    /// Executor ids whose yields are workflow-level [`Output`](WorkflowEvent::Output)
    /// events (see [`WorkflowBuilder::output_from`]). Empty means "no
    /// restriction" — see [`WorkflowRun::classify_yield`] for the full
    /// precedence rule.
    pub output_executors: HashSet<String>,
    /// Executor ids whose yields are non-terminal
    /// [`Intermediate`](WorkflowEvent::Intermediate) events (see
    /// [`WorkflowBuilder::intermediate_output_from`]).
    pub intermediate_executors: HashSet<String>,
}

/// Fluent builder for a [`Workflow`]. Rust equivalent of `WorkflowBuilder`.
pub struct WorkflowBuilder {
    executors: HashMap<String, Arc<dyn Executor>>,
    edge_groups: Vec<EdgeGroup>,
    start: Option<String>,
    max_iterations: usize,
    name: Option<String>,
    description: Option<String>,
    checkpoint_storage: Option<Arc<dyn CheckpointStorage>>,
    output_executors: Vec<String>,
    intermediate_executors: Vec<String>,
}

impl std::default::Default for WorkflowBuilder {
    fn default() -> Self {
        Self {
            executors: HashMap::new(),
            edge_groups: Vec::new(),
            start: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            name: None,
            description: None,
            checkpoint_storage: None,
            output_executors: Vec::new(),
            intermediate_executors: Vec::new(),
        }
    }
}

impl WorkflowBuilder {
    /// Create a new, empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an executor. Executors are keyed by their [`Executor::id`].
    pub fn add_executor(mut self, executor: Arc<dyn Executor>) -> Self {
        self.executors.insert(executor.id().to_string(), executor);
        self
    }

    /// Set the entry-point executor (by id). The initial message is delivered
    /// here.
    pub fn set_start(mut self, id: impl Into<String>) -> Self {
        self.start = Some(id.into());
        self
    }

    /// Add a single directed edge.
    pub fn add_edge(mut self, source: impl Into<String>, target: impl Into<String>) -> Self {
        self.edge_groups.push(EdgeGroup::Single {
            source: source.into(),
            target: target.into(),
            condition: None,
        });
        self
    }

    /// Add a single directed edge guarded by a synchronous condition.
    pub fn add_conditional_edge(
        mut self,
        source: impl Into<String>,
        target: impl Into<String>,
        condition: impl Fn(&Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.edge_groups.push(EdgeGroup::Single {
            source: source.into(),
            target: target.into(),
            condition: Some(wrap_sync_condition(condition)),
        });
        self
    }

    /// Add a single directed edge guarded by an async condition.
    ///
    /// Equivalent to [`WorkflowBuilder::add_conditional_edge`] but for a
    /// predicate that itself needs to `.await` (e.g. an I/O check); the
    /// condition is awaited at routing time (see `UPSTREAM_DRIFT.md` §10,
    /// `Edge.should_route` becoming async upstream).
    pub fn add_conditional_edge_async<F, Fut>(
        mut self,
        source: impl Into<String>,
        target: impl Into<String>,
        condition: F,
    ) -> Self
    where
        F: Fn(&Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.edge_groups.push(EdgeGroup::Single {
            source: source.into(),
            target: target.into(),
            condition: Some(wrap_async_condition(condition)),
        });
        self
    }

    /// Broadcast from `source` to all `targets`.
    pub fn add_fan_out(
        mut self,
        source: impl Into<String>,
        targets: impl IntoIterator<Item = String>,
    ) -> Self {
        self.edge_groups.push(EdgeGroup::FanOut {
            source: source.into(),
            targets: targets.into_iter().collect(),
            selection: None,
            case_labels: None,
        });
        self
    }

    /// Fan in from `sources` to `target` (barrier).
    pub fn add_fan_in(
        mut self,
        sources: impl IntoIterator<Item = String>,
        target: impl Into<String>,
    ) -> Self {
        self.edge_groups.push(EdgeGroup::FanIn {
            sources: sources.into_iter().collect(),
            target: target.into(),
        });
        self
    }

    /// Add a switch/case group: evaluate `cases` in order, falling back to
    /// `default`.
    pub fn add_switch(
        mut self,
        source: impl Into<String>,
        cases: Vec<Case>,
        default: SwitchDefault,
    ) -> Self {
        let source = source.into();
        let mut targets: Vec<String> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        for (i, case) in cases.iter().enumerate() {
            targets.push(case.target.clone());
            labels.push(case.label.clone().unwrap_or_else(|| format!("case {i}")));
        }
        targets.push(default.target.clone());
        labels.push("default".to_string());

        // Extract owned `(condition, target)` pairs so the selection closure
        // below doesn't need to borrow `cases` from its own captured
        // environment (the returned future must own everything it touches).
        let case_targets: Vec<(super::edge::Condition, String)> = cases
            .iter()
            .map(|case| (case.condition.clone(), case.target.clone()))
            .collect();
        let default_target = default.target.clone();
        let selection: Selection = Arc::new(move |msg: &Value, _candidates: &[String]| {
            let case_targets = case_targets.clone();
            let default_target = default_target.clone();
            let msg = msg.clone();
            Box::pin(async move {
                for (condition, target) in &case_targets {
                    if condition(&msg).await {
                        return vec![target.clone()];
                    }
                }
                vec![default_target.clone()]
            }) as BoxFuture<Vec<String>>
        });
        self.edge_groups.push(EdgeGroup::FanOut {
            source,
            targets,
            selection: Some(selection),
            case_labels: Some(labels),
        });
        self
    }

    /// Chain executors sequentially (sugar for consecutive edges).
    pub fn add_chain(mut self, ids: impl IntoIterator<Item = String>) -> Self {
        let ids: Vec<String> = ids.into_iter().collect();
        for pair in ids.windows(2) {
            self = self.add_edge(pair[0].clone(), pair[1].clone());
        }
        self
    }

    /// Set the maximum number of supersteps before the run fails.
    pub fn set_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n.max(1);
        self
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the workflow description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Enable automatic checkpointing to `storage` at each superstep boundary.
    pub fn with_checkpointing(mut self, storage: Arc<dyn CheckpointStorage>) -> Self {
        self.checkpoint_storage = Some(storage);
        self
    }

    /// Designate `ids` as workflow-level output sources: their `yield_output`
    /// calls become terminal [`WorkflowEvent::Output`] events and are recorded
    /// in [`WorkflowRun::last_output`]/[`WorkflowRun::outputs`].
    ///
    /// Once any executor is designated with `output_from`, executors *not*
    /// named here (and not named via [`WorkflowBuilder::intermediate_output_from`])
    /// have their yields demoted to non-terminal [`WorkflowEvent::Intermediate`]
    /// events rather than dropped — see [`WorkflowEvent::Intermediate`] for the
    /// full precedence rule. Leaving both designation lists empty preserves the
    /// default behavior: every yield is a terminal `Output`.
    ///
    /// Rejected at [`WorkflowBuilder::build`] if any id is unknown or overlaps
    /// [`WorkflowBuilder::intermediate_output_from`].
    pub fn output_from(mut self, ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.output_executors
            .extend(ids.into_iter().map(Into::into));
        self
    }

    /// Designate `ids` as intermediate-output sources: their `yield_output`
    /// calls become non-terminal [`WorkflowEvent::Intermediate`] events and are
    /// never recorded as the run's final output.
    ///
    /// Rejected at [`WorkflowBuilder::build`] if any id is unknown or overlaps
    /// [`WorkflowBuilder::output_from`].
    pub fn intermediate_output_from(
        mut self,
        ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.intermediate_executors
            .extend(ids.into_iter().map(Into::into));
        self
    }

    /// Validate and build the workflow.
    pub fn build(self) -> Result<Workflow> {
        let start = self
            .start
            .ok_or_else(|| Error::Workflow("no start executor set".into()))?;

        // Full graph validation (start presence, duplicate edges, connectivity,
        // output designation).
        validate_workflow_graph(
            &self.executors,
            &self.edge_groups,
            &start,
            &self.output_executors,
            &self.intermediate_executors,
        )
        .map_err(|e| Error::Workflow(e.to_string()))?;

        let graph_signature = compute_graph_signature(&self.executors, &self.edge_groups, &start);

        Ok(Workflow {
            shared: Arc::new(WorkflowShared {
                executors: self.executors,
                edge_groups: self.edge_groups,
                start,
                max_iterations: self.max_iterations,
                name: self.name,
                description: self.description,
                checkpoint_storage: self.checkpoint_storage,
                id: uuid::Uuid::new_v4().to_string(),
                graph_signature,
                output_executors: self.output_executors.into_iter().collect(),
                intermediate_executors: self.intermediate_executors.into_iter().collect(),
            }),
        })
    }
}

/// A built, runnable workflow graph.
#[derive(Clone)]
pub struct Workflow {
    pub(crate) shared: Arc<WorkflowShared>,
}

impl Workflow {
    /// The workflow name, if set.
    pub fn name(&self) -> Option<&str> {
        self.shared.name.as_deref()
    }
    /// The workflow description, if set.
    pub fn description(&self) -> Option<&str> {
        self.shared.description.as_deref()
    }
    /// The workflow's unique id.
    pub fn id(&self) -> &str {
        &self.shared.id
    }
    /// The start executor id.
    pub fn start_executor_id(&self) -> &str {
        &self.shared.start
    }

    /// This workflow's deterministic graph signature (see
    /// [`WorkflowBuilder::build`]). Checkpoints record the signature of the
    /// graph that produced them; [`Workflow::run_from_checkpoint`] refuses to
    /// resume a checkpoint whose signature does not match this workflow's.
    pub fn graph_signature(&self) -> &str {
        &self.shared.graph_signature
    }

    /// Run the workflow to completion (or until it pauses awaiting external
    /// input), returning the run handle with observed events and final state.
    pub async fn run(&self, input: impl Into<Value>) -> Result<WorkflowRun> {
        let mut run = WorkflowRun::new(self.shared.clone(), None);
        run.start(input.into()).await?;
        Ok(run)
    }

    /// Run the workflow with `storage` as its checkpoint storage for this run
    /// only, overriding any storage set with
    /// [`WorkflowBuilder::with_checkpointing`]. Mirrors upstream's
    /// `workflow.run(..., checkpoint_storage=...)`: a host can give each
    /// conversation its own storage and resume a paused run from it later
    /// with [`Workflow::run_from_checkpoint`].
    pub async fn run_with_checkpointing(
        &self,
        input: impl Into<Value>,
        storage: Arc<dyn CheckpointStorage>,
    ) -> Result<WorkflowRun> {
        let mut run = WorkflowRun::new(self.shared.clone(), None);
        run.checkpoint_storage = Some(storage);
        run.start(input.into()).await?;
        Ok(run)
    }

    /// Run the workflow, streaming events as they happen.
    ///
    /// Returns a [`WorkflowRunStream`] that yields [`WorkflowEvent`]s live; call
    /// [`WorkflowRunStream::into_run`] after the stream ends to obtain the final
    /// [`WorkflowRun`] state.
    pub fn run_stream(&self, input: impl Into<Value>) -> WorkflowRunStream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let shared = self.shared.clone();
        let input = input.into();
        let handle = tokio::spawn(async move {
            let mut run = WorkflowRun::new(shared, Some(tx));
            let result = run.start(input).await;
            // Drop the streaming sender so the receiver observes end-of-stream;
            // otherwise the sender lives on inside the returned run and the
            // stream would never terminate.
            run.event_tx = None;
            result?;
            Ok(run)
        });
        WorkflowRunStream {
            rx,
            handle: Some(handle),
        }
    }

    /// Restore a run from a checkpoint and continue it to completion (or pause).
    ///
    /// Restores the in-flight message queue, iteration count, shared state,
    /// executor states, and any outstanding requests.
    ///
    /// Before restoring, the checkpoint's recorded [graph
    /// signature](Workflow::graph_signature) is compared against this
    /// workflow's: a mismatch is a hard [`Error::Workflow`] (the topology
    /// changed since the checkpoint was saved, so resuming could misroute
    /// messages or drop executor state). A legacy checkpoint carrying no
    /// signature is resumed with a warning. Use
    /// [`Workflow::run_from_checkpoint_unchecked`] to bypass the check.
    pub async fn run_from_checkpoint(
        &self,
        checkpoint_id: &str,
        storage: Arc<dyn CheckpointStorage>,
    ) -> Result<WorkflowRun> {
        self.run_from_checkpoint_inner(checkpoint_id, storage, true)
            .await
    }

    /// Like [`Workflow::run_from_checkpoint`], but skips graph-signature
    /// validation (a mismatch is logged, not rejected). For deliberately
    /// resuming a checkpoint into an intentionally-evolved graph, where the
    /// caller accepts responsibility for compatibility.
    pub async fn run_from_checkpoint_unchecked(
        &self,
        checkpoint_id: &str,
        storage: Arc<dyn CheckpointStorage>,
    ) -> Result<WorkflowRun> {
        self.run_from_checkpoint_inner(checkpoint_id, storage, false)
            .await
    }

    async fn run_from_checkpoint_inner(
        &self,
        checkpoint_id: &str,
        storage: Arc<dyn CheckpointStorage>,
        validate: bool,
    ) -> Result<WorkflowRun> {
        let cp = storage
            .load(checkpoint_id)
            .await?
            .ok_or_else(|| Error::Workflow(format!("checkpoint '{checkpoint_id}' not found")))?;
        self.check_graph_signature(&cp, validate)?;
        let mut run = WorkflowRun::new(self.shared.clone(), None);
        // The storage a run was resumed from is where it keeps
        // checkpointing, overriding any build-time storage — upstream's
        // "runtime storage takes precedence". Otherwise a run resumed from a
        // per-conversation store would write its next checkpoint somewhere
        // else (or nowhere), and a second pause could not be resumed.
        run.checkpoint_storage = Some(storage);
        run.restore(cp).await?;
        Ok(run)
    }

    /// Compare a checkpoint's graph signature against this workflow's.
    ///
    /// * No signature (legacy checkpoint) → warn and continue.
    /// * Matching signature → continue.
    /// * Mismatch with `validate` → [`Error::Workflow`] naming both signatures.
    /// * Mismatch without `validate` → warn and continue.
    fn check_graph_signature(&self, cp: &WorkflowCheckpoint, validate: bool) -> Result<()> {
        let expected = &self.shared.graph_signature;
        if cp.graph_signature.is_empty() {
            tracing::warn!(
                workflow_id = %self.shared.id,
                expected = %expected,
                "resuming a checkpoint with no graph signature (written before signature \
                 validation existed); skipping the graph-compatibility check"
            );
            return Ok(());
        }
        if &cp.graph_signature == expected {
            return Ok(());
        }
        // A checkpoint from an older signature *scheme* has not necessarily
        // been written for a different graph — the encoding changed under it.
        // Saying "your graph changed" there sends the reader looking for a
        // change that is not in their code.
        let scheme_prefix = format!("{SIGNATURE_SCHEME}-");
        if validate {
            if !cp.graph_signature.starts_with(&scheme_prefix) {
                return Err(Error::Workflow(format!(
                    "checkpoint graph signature scheme mismatch: checkpoint '{}' carries \
                     signature '{}', written by an older version of this library whose signature \
                     encoding differed, so it cannot be compared against this workflow's '{}'. \
                     The graph itself may well be identical. Finish the run on the version that \
                     wrote it, or call `run_from_checkpoint_unchecked` to resume it here.",
                    cp.checkpoint_id, cp.graph_signature, expected
                )));
            }
            return Err(Error::Workflow(format!(
                "checkpoint graph signature mismatch: checkpoint '{}' was saved for graph \
                 signature '{}', but this workflow's graph signature is '{}'. The workflow's \
                 executors or edges changed since the checkpoint was written, so resuming could \
                 misroute messages or drop executor state. Rebuild the identical graph, or call \
                 `run_from_checkpoint_unchecked` to override.",
                cp.checkpoint_id, cp.graph_signature, expected
            )));
        }
        tracing::warn!(
            workflow_id = %self.shared.id,
            checkpoint_signature = %cp.graph_signature,
            expected = %expected,
            "resuming a checkpoint whose graph signature does not match this workflow \
             (unchecked); resume may misbehave if the topology is incompatible"
        );
        Ok(())
    }

    /// A visualization helper (Mermaid / Graphviz DOT) for this workflow.
    pub fn viz(&self) -> WorkflowViz<'_> {
        WorkflowViz::new(self)
    }
}

/// One planned executor invocation within a superstep: the target executor, the
/// payload to hand it, and the source executor id(s) that produced the payload.
///
/// A superstep's invocations are planned up front (draining the delivery queue
/// and resolving fan-in barriers, which mutate `WorkflowRun::fanin`) and then
/// run concurrently, so nothing here borrows the runner.
struct Invocation {
    executor: Arc<dyn Executor>,
    /// The target executor's id. Invocations sharing an id are serialized
    /// within a superstep (see [`WorkflowRun::run_loop`]).
    executor_id: String,
    data: Value,
    source_ids: Vec<String>,
}

/// A live workflow run: owns the pending message queue, fan-in buffers,
/// iteration count, shared state, and outstanding requests, so a run can pause
/// (awaiting external input) and later resume.
pub struct WorkflowRun {
    shared: Arc<WorkflowShared>,
    shared_state: SharedState,
    queue: Vec<WorkflowMessage>,
    fanin: HashMap<String, HashMap<String, Value>>,
    iteration: usize,
    pending_requests: BTreeMap<String, PendingRequest>,
    events: Vec<WorkflowEvent>,
    state: WorkflowRunState,
    event_tx: Option<UnboundedSender<WorkflowEvent>>,
    /// Where this run writes its superstep checkpoints: the build-time
    /// storage unless the run was started with a runtime override.
    checkpoint_storage: Option<Arc<dyn CheckpointStorage>>,
}

impl WorkflowRun {
    fn new(shared: Arc<WorkflowShared>, event_tx: Option<UnboundedSender<WorkflowEvent>>) -> Self {
        Self {
            checkpoint_storage: shared.checkpoint_storage.clone(),
            shared,
            shared_state: SharedState::new(),
            queue: Vec::new(),
            fanin: HashMap::new(),
            iteration: 0,
            pending_requests: BTreeMap::new(),
            events: Vec::new(),
            state: WorkflowRunState::Started,
            event_tx,
        }
    }

    fn emit(&mut self, event: WorkflowEvent) {
        if let Some(tx) = &self.event_tx {
            // Ignore send errors: the receiver may have been dropped.
            let _ = tx.send(event.clone());
        }
        self.events.push(event);
    }

    /// Classify a value yielded by executor `id` into the right
    /// [`WorkflowEvent`], per the workflow's output designation
    /// (`output_from`/`intermediate_output_from`).
    ///
    /// Precedence:
    /// 1. `id` is in `intermediate_executors` -> [`WorkflowEvent::Intermediate`]
    ///    (never counted as the run's output, regardless of `output_executors`).
    /// 2. Otherwise, if `output_executors` is empty (no designation configured
    ///    at all) -> [`WorkflowEvent::Output`], preserving the pre-designation
    ///    default where every yield is terminal output.
    /// 3. Otherwise (`output_executors` is non-empty), `id` in it ->
    ///    [`WorkflowEvent::Output`]; `id` not in it -> demoted to
    ///    [`WorkflowEvent::Intermediate`] as a safe non-terminal signal, rather
    ///    than silently dropped.
    fn classify_yield(&self, id: &str, data: Value) -> WorkflowEvent {
        let source_executor_id = id.to_string();
        if self.shared.intermediate_executors.contains(id) {
            return WorkflowEvent::Intermediate {
                data,
                source_executor_id,
            };
        }
        if self.shared.output_executors.is_empty() || self.shared.output_executors.contains(id) {
            WorkflowEvent::Output {
                data,
                source_executor_id,
            }
        } else {
            WorkflowEvent::Intermediate {
                data,
                source_executor_id,
            }
        }
    }

    async fn start(&mut self, input: Value) -> Result<()> {
        self.emit(WorkflowEvent::Started);
        self.state = WorkflowRunState::InProgress;
        self.emit(WorkflowEvent::Status(WorkflowRunState::InProgress));
        self.queue.push(WorkflowMessage {
            data: input,
            source_id: "__start__".to_string(),
            target_id: Some(self.shared.start.clone()),
        });
        self.drive().await
    }

    async fn restore(&mut self, cp: WorkflowCheckpoint) -> Result<()> {
        self.shared_state.import(cp.shared_state).await;
        self.iteration = cp.iteration_count;
        self.queue = cp.messages;
        // Restore partially-satisfied fan-in barriers so a checkpoint taken
        // between source deliveries can still fire the barrier on resume.
        self.fanin = cp.fanin_state;
        self.pending_requests = cp
            .pending_requests
            .into_iter()
            .map(|p| (p.request_id.clone(), p))
            .collect();
        for (id, state) in cp.executor_states {
            if let Some(ex) = self.shared.executors.get(&id) {
                ex.restore_state(state).await?;
            }
        }

        self.emit(WorkflowEvent::Started);
        self.state = WorkflowRunState::InProgress;
        self.emit(WorkflowEvent::Status(WorkflowRunState::InProgress));
        // Re-surface any outstanding requests so consumers observe them.
        let pending_events: Vec<WorkflowEvent> = self
            .pending_requests
            .values()
            .map(|pr| WorkflowEvent::RequestInfo {
                request_id: pr.request_id.clone(),
                source_executor_id: pr.source_executor_id.clone(),
                request_data: pr.request_data.clone(),
            })
            .collect();
        for ev in pending_events {
            self.emit(ev);
        }
        self.drive().await
    }

    /// Drive the superstep loop, emitting the terminal status on success or a
    /// `Failed` status on error.
    async fn drive(&mut self) -> Result<()> {
        match self.run_loop().await {
            Ok(()) => Ok(()),
            Err(e) => {
                self.state = WorkflowRunState::Failed;
                self.emit(WorkflowEvent::Failed {
                    error: e.to_string(),
                });
                self.emit(WorkflowEvent::Status(WorkflowRunState::Failed));
                Err(e)
            }
        }
    }

    async fn run_loop(&mut self) -> Result<()> {
        let mut emitted_pending = false;
        while !self.queue.is_empty() {
            if self.iteration >= self.shared.max_iterations {
                return Err(Error::Workflow(format!(
                    "max_iterations ({}) exceeded",
                    self.shared.max_iterations
                )));
            }
            let step = self.iteration + 1;
            self.emit(WorkflowEvent::SuperStepStarted(step));

            // Group this superstep's deliveries by target executor.
            let deliveries = std::mem::take(&mut self.queue);
            let mut by_target: HashMap<String, Vec<WorkflowMessage>> = HashMap::new();
            for msg in deliveries {
                for target in self.resolve_targets(&msg).await? {
                    by_target.entry(target).or_default().push(msg.clone());
                }
            }

            // Deterministic iteration order within a superstep.
            let mut targets: Vec<String> = by_target.keys().cloned().collect();
            targets.sort();

            // Plan this superstep's executor invocations. Fan-in barriers are
            // resolved here, synchronously, because they mutate `self.fanin`;
            // once planned, each invocation owns everything it needs so the
            // executes below touch no `&mut self` state. Iterating `targets` in
            // sorted order (and, within a target, direct messages before the
            // fan-in/regular messages) fixes the invocation order, which in turn
            // fixes event and next-queue order after the join.
            let mut invocations: Vec<Invocation> = Vec::new();
            for target_id in &targets {
                let msgs = by_target.remove(target_id).unwrap_or_default();
                let executor = match self.shared.executors.get(target_id) {
                    Some(e) => e.clone(),
                    None => continue,
                };

                // Explicitly-targeted messages (request-info responses routed
                // straight back to the requesting executor) bypass the fan-in
                // barrier: their source is the responder, not one of the
                // fan-in edges, so accumulating them would stall the resumed
                // run forever.
                let (direct, msgs): (Vec<_>, Vec<_>) =
                    msgs.into_iter().partition(|m| m.target_id.is_some());
                for m in direct {
                    invocations.push(Invocation {
                        executor: executor.clone(),
                        executor_id: target_id.clone(),
                        data: m.data,
                        source_ids: vec![m.source_id],
                    });
                }
                if msgs.is_empty() {
                    continue;
                }

                if let Some(sources) = self.fanin_group_for(target_id) {
                    // Barrier: accumulate by source, fire once every source has
                    // delivered, collecting in declared order.
                    let buf = self.fanin.entry(target_id.clone()).or_default();
                    for m in &msgs {
                        buf.insert(m.source_id.clone(), m.data.clone());
                    }
                    if buf.len() < sources.len() {
                        continue;
                    }
                    let mut collected = Vec::with_capacity(sources.len());
                    for source in &sources {
                        if let Some(v) = buf.remove(source) {
                            collected.push(v);
                        }
                    }
                    self.fanin.remove(target_id);
                    invocations.push(Invocation {
                        executor,
                        executor_id: target_id.clone(),
                        data: Value::Array(collected),
                        source_ids: sources,
                    });
                } else {
                    for m in msgs {
                        invocations.push(Invocation {
                            executor: executor.clone(),
                            executor_id: target_id.clone(),
                            data: m.data,
                            source_ids: vec![m.source_id],
                        });
                    }
                }
            }

            // Group the planned invocations by target executor. Invocations
            // for one executor are contiguous (each target is visited once, in
            // sorted order), so a linear scan reconstructs the groups while
            // preserving planned order.
            let mut groups: Vec<Vec<Invocation>> = Vec::new();
            for inv in invocations {
                match groups.last_mut() {
                    Some(g) if g[0].executor_id == inv.executor_id => g.push(inv),
                    _ => groups.push(vec![inv]),
                }
            }

            // Run the groups concurrently, but serialize the invocations *within*
            // a group: two deliveries to the same executor instance in one
            // superstep must not have overlapping `execute()` calls, or a
            // stateful executor sees a nondeterministic interleaving. This
            // mirrors upstream's per-executor `_execution_lock` (PR #6776):
            // different executors stay concurrent (`asyncio.gather` /
            // `join_all`), same-executor deliveries run one at a time. Each
            // future builds its own context and drains its own effects, so
            // distinct executors (typically LLM calls) still genuinely overlap.
            // Flattening the group results in group order — which is the sorted
            // planned order, with each group's own invocations kept sequential —
            // yields `results` in exactly the planned order the apply loop below
            // relies on for determinism.
            let shared_state = self.shared_state.clone();
            let group_results = futures::future::join_all(groups.into_iter().map(|group| {
                let shared_state = shared_state.clone();
                async move {
                    let mut out = Vec::with_capacity(group.len());
                    for inv in group {
                        out.push(
                            Self::run_executor_isolated(
                                inv.executor,
                                inv.data,
                                inv.source_ids,
                                shared_state.clone(),
                            )
                            .await,
                        );
                    }
                    out
                }
            }))
            .await;
            let results: Vec<(String, Result<DrainedEffects>)> =
                group_results.into_iter().flatten().collect();

            // Apply results in planned order so the emitted event sequence and
            // the next superstep's queue are identical to sequential execution.
            // Events are buffered per invocation (not emitted live during the
            // executes), which keeps per-executor event pairs coherent and the
            // overall order deterministic. The first error in planned order
            // fails the superstep, exactly as the sequential `?` did — later
            // invocations' *drained* effects (messages, events, outputs,
            // requests) are simply not applied.
            //
            // External side effects are a different matter, by design: sibling
            // invocations that already ran (or were mid-flight) when another
            // failed have still performed their external work (network calls,
            // etc.), and there is no rollback for those. `SharedState` writes,
            // however, are staged: they land in the pending buffer during a
            // superstep and are folded into committed state only by the
            // `commit()` below, which a failed superstep never reaches. So a
            // failed superstep discards its shared-state writes rather than
            // persisting a partial, branch-dependent state — matching upstream's
            // Pregel-style commit-at-boundary model (`_state.py` / `_runner.py`),
            // where `commit()` runs only after a superstep's executors all
            // complete. A resume then replays the whole superstep from the last
            // committed boundary.
            let mut next: Vec<WorkflowMessage> = Vec::new();
            for (id, result) in results {
                self.emit(WorkflowEvent::ExecutorInvoked {
                    executor_id: id.clone(),
                });
                match result {
                    Ok((sent, outputs, custom, requests)) => {
                        for out in outputs {
                            self.emit(self.classify_yield(&id, out));
                        }
                        for ev in custom {
                            self.emit(ev);
                        }
                        for draft in requests {
                            let pending = PendingRequest {
                                request_id: draft.request_id.clone(),
                                source_executor_id: id.clone(),
                                reply_to_executor_id: draft.reply_to,
                                request_data: draft.data.clone(),
                            };
                            self.pending_requests
                                .insert(draft.request_id.clone(), pending);
                            self.emit(WorkflowEvent::RequestInfo {
                                request_id: draft.request_id,
                                source_executor_id: id.clone(),
                                request_data: draft.data,
                            });
                            if !emitted_pending {
                                emitted_pending = true;
                                self.emit(WorkflowEvent::Status(
                                    WorkflowRunState::InProgressPendingRequests,
                                ));
                            }
                        }
                        next.extend(sent);
                        self.emit(WorkflowEvent::ExecutorCompleted { executor_id: id });
                    }
                    Err(e) => {
                        self.emit(WorkflowEvent::ExecutorFailed {
                            executor_id: id,
                            error: e.to_string(),
                        });
                        return Err(e);
                    }
                }
            }

            // Superstep barrier: fold this superstep's staged shared-state
            // writes into committed state before emitting completion or
            // checkpointing, so a checkpoint captures exactly the committed
            // boundary. Only reached when every executor succeeded (an error
            // returns above, leaving the writes discarded).
            self.shared_state.commit().await;

            self.emit(WorkflowEvent::SuperStepCompleted(step));
            self.queue = next;
            self.iteration += 1;
            self.maybe_checkpoint(step).await;
        }

        // Terminal status: distinguish idle from paused-with-requests.
        if self.pending_requests.is_empty() {
            self.state = WorkflowRunState::Idle;
            self.emit(WorkflowEvent::Status(WorkflowRunState::Idle));
        } else {
            self.state = WorkflowRunState::IdleWithPendingRequests;
            self.emit(WorkflowEvent::Status(
                WorkflowRunState::IdleWithPendingRequests,
            ));
        }
        Ok(())
    }

    /// Run one executor in isolation: build its context, execute it, and drain
    /// the effects it produced (messages, outputs, custom events, requests).
    ///
    /// Deliberately takes no `&self` — only owned inputs and the `Arc`-shared
    /// [`SharedState`] — so a superstep's invocations can run concurrently
    /// without contending on the runner. Returns the executor id alongside
    /// either its drained effects or its execution error; the caller emits the
    /// corresponding events (in a deterministic order) after the join.
    async fn run_executor_isolated(
        executor: Arc<dyn Executor>,
        data: Value,
        source_ids: Vec<String>,
        shared_state: SharedState,
    ) -> (String, Result<DrainedEffects>) {
        let id = executor.id().to_string();
        let ctx = WorkflowContext::new(id.clone(), source_ids, shared_state);
        let result = match executor.execute(data, ctx.clone()).await {
            Ok(()) => Ok(ctx.take()),
            Err(e) => Err(e),
        };
        (id, result)
    }

    async fn maybe_checkpoint(&self, step: usize) {
        let Some(storage) = self.checkpoint_storage.clone() else {
            return;
        };
        let mut executor_states: HashMap<String, Value> = HashMap::new();
        for (id, ex) in &self.shared.executors {
            if let Some(state) = ex.snapshot_state().await {
                executor_states.insert(id.clone(), state);
            }
        }
        let shared_state = self.shared_state.export().await;
        let mut metadata: HashMap<String, Value> = HashMap::new();
        metadata.insert("superstep".to_string(), Value::from(self.iteration as u64));
        metadata.insert(
            "checkpoint_type".to_string(),
            Value::from(format!("superstep_{step}")),
        );
        let checkpoint = WorkflowCheckpoint::new(
            self.shared.id.clone(),
            self.shared.name.clone(),
            self.iteration,
            self.queue.clone(),
            executor_states,
            shared_state,
            self.pending_requests.values().cloned().collect(),
            self.fanin.clone(),
            metadata,
            self.shared.graph_signature.clone(),
        );
        if let Err(e) = storage.save(checkpoint).await {
            tracing::warn!("failed to save checkpoint: {e}");
        }
    }

    /// Deliver a single response, resuming execution.
    pub async fn send_response(
        &mut self,
        request_id: impl Into<String>,
        value: Value,
    ) -> Result<()> {
        let mut map = HashMap::new();
        map.insert(request_id.into(), value);
        self.send_responses(map).await
    }

    /// Deliver responses to outstanding requests and resume execution.
    ///
    /// Each response is routed back to the executor that made the corresponding
    /// request as a [`RequestResponse`] message.
    pub async fn send_responses(&mut self, responses: HashMap<String, Value>) -> Result<()> {
        if self.pending_requests.is_empty() {
            return Err(Error::Workflow("no pending requests to respond to".into()));
        }
        for id in responses.keys() {
            if !self.pending_requests.contains_key(id) {
                return Err(Error::Workflow(format!(
                    "response provided for unknown request id '{id}'"
                )));
            }
        }

        self.state = WorkflowRunState::InProgress;
        self.emit(WorkflowEvent::Status(WorkflowRunState::InProgress));

        // Deliver in deterministic order.
        let mut ids: Vec<String> = responses.keys().cloned().collect();
        ids.sort();
        for id in ids {
            if let Some(value) = responses.get(&id) {
                self.deliver_response(&id, value.clone());
            }
        }
        self.drive().await
    }

    fn deliver_response(&mut self, request_id: &str, value: Value) {
        if let Some(pending) = self.pending_requests.remove(request_id) {
            let response = RequestResponse {
                request_id: pending.request_id.clone(),
                data: value,
                original_request: pending.request_data.clone(),
            };
            let data = serde_json::to_value(response).unwrap_or(Value::Null);
            self.queue.push(WorkflowMessage {
                data,
                source_id: pending.source_executor_id.clone(),
                target_id: Some(pending.reply_to_executor_id.clone()),
            });
        }
    }

    /// Resolve which executors a message should be delivered to.
    ///
    /// Awaits each matching group's condition/selection (see
    /// `UPSTREAM_DRIFT.md` §10 — upstream's `Edge.should_route` is async).
    async fn resolve_targets(&self, msg: &WorkflowMessage) -> Result<Vec<String>> {
        // Explicit target wins (direct sends and routed responses). An explicit
        // target that names no registered executor is a routing error, not a
        // silent drop: without this check the message would be quietly discarded
        // (the planning loop's `executors.get(...)` miss) and the run would still
        // report `Idle`, masking a typo'd `send_to` target.
        if let Some(t) = &msg.target_id {
            if !self.shared.executors.contains_key(t) {
                return Err(Error::Workflow(format!(
                    "executor '{}' sent a message to unknown target executor '{}'",
                    msg.source_id, t
                )));
            }
            return Ok(vec![t.clone()]);
        }
        let mut targets = Vec::new();
        for group in &self.shared.edge_groups {
            match group {
                EdgeGroup::Single {
                    source,
                    target,
                    condition,
                } if *source == msg.source_id => {
                    let route = match condition {
                        Some(c) => c(&msg.data).await,
                        None => true,
                    };
                    if route {
                        targets.push(target.clone());
                    }
                }
                EdgeGroup::FanOut {
                    source,
                    targets: outs,
                    selection,
                    ..
                } if *source == msg.source_id => match selection {
                    Some(sel) => targets.extend(sel(&msg.data, outs).await),
                    None => targets.extend(outs.clone()),
                },
                EdgeGroup::FanIn { sources, target } if sources.contains(&msg.source_id) => {
                    targets.push(target.clone());
                }
                _ => {}
            }
        }
        Ok(targets)
    }

    /// If `target` is the sink of a fan-in group, return its source ids.
    fn fanin_group_for(&self, target: &str) -> Option<Vec<String>> {
        for group in &self.shared.edge_groups {
            if let EdgeGroup::FanIn { sources, target: t } = group {
                if t == target {
                    return Some(sources.clone());
                }
            }
        }
        None
    }

    /// All events observed during the run, in order.
    pub fn events(&self) -> &[WorkflowEvent] {
        &self.events
    }

    /// The current run state.
    pub fn state(&self) -> WorkflowRunState {
        self.state
    }

    /// Outstanding requests awaiting responses, ordered by request id.
    pub fn pending_requests(&self) -> Vec<PendingRequest> {
        self.pending_requests.values().cloned().collect()
    }

    /// All workflow-level outputs, in order.
    pub fn outputs(&self) -> Vec<Value> {
        self.events
            .iter()
            .filter_map(|e| e.as_output().cloned())
            .collect()
    }

    /// The last workflow output, if any.
    pub fn last_output(&self) -> Option<Value> {
        self.events
            .iter()
            .rev()
            .find_map(|e| e.as_output().cloned())
    }

    /// A handle to the run-scoped shared state (for inspection in tests/tools).
    pub fn shared_state(&self) -> SharedState {
        self.shared_state.clone()
    }
}

/// A live stream of [`WorkflowEvent`]s from a running workflow, plus a way to
/// obtain the final [`WorkflowRun`] once the stream is exhausted.
pub struct WorkflowRunStream {
    rx: UnboundedReceiver<WorkflowEvent>,
    handle: Option<tokio::task::JoinHandle<Result<WorkflowRun>>>,
}

impl WorkflowRunStream {
    /// Await the driving task and return the final run state.
    ///
    /// Drains any remaining buffered events first so the run completes.
    pub async fn into_run(mut self) -> Result<WorkflowRun> {
        while self.rx.recv().await.is_some() {}
        match self.handle.take() {
            Some(handle) => handle
                .await
                .map_err(|e| Error::Workflow(format!("workflow task failed: {e}")))?,
            None => Err(Error::Workflow("run already taken from stream".into())),
        }
    }
}

impl futures::Stream for WorkflowRunStream {
    type Item = WorkflowEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

#[cfg(test)]
mod async_condition_tests {
    //! Regression + new-behavior coverage for `UPSTREAM_DRIFT.md` §10: edge
    //! routing conditions became async upstream (`Edge.should_route`); these
    //! confirm the sync builder API still routes correctly (regression) and
    //! that the new `*_async` builder API also routes correctly.

    use super::*;
    use crate::workflow::executor::FunctionExecutor;
    use futures::StreamExt;
    use serde_json::json;

    /// An executor that immediately yields whatever message it receives as a
    /// workflow output, so a run's `outputs()` directly reveal which branch(es)
    /// were taken.
    fn echo(id: &str) -> Arc<dyn Executor> {
        Arc::new(FunctionExecutor::new(id, |msg, ctx| async move {
            ctx.yield_output(msg).await
        }))
    }

    #[tokio::test]
    async fn sync_condition_still_routes() {
        // Regression: the pre-existing sync `add_conditional_edge` builder
        // must keep routing correctly now that `Condition` is async
        // internally.
        let workflow = WorkflowBuilder::new()
            .add_executor(echo("start"))
            .add_executor(echo("hit"))
            .add_executor(echo("miss"))
            .set_start("start")
            .add_conditional_edge("start", "hit", |v| v == &json!("go"))
            .add_conditional_edge("start", "miss", |v| v != &json!("go"))
            .build()
            .unwrap();

        let run = workflow.run(json!("go")).await.unwrap();
        assert_eq!(run.outputs(), vec![json!("go")]);
    }

    #[tokio::test]
    async fn async_condition_routes() {
        // New: `add_conditional_edge_async` accepts a condition that actually
        // awaits (yields to the scheduler) before resolving.
        let workflow = WorkflowBuilder::new()
            .add_executor(echo("start"))
            .add_executor(echo("hit"))
            .add_executor(echo("miss"))
            .set_start("start")
            .add_conditional_edge_async("start", "hit", |v| {
                let matched = v == &json!("go");
                async move {
                    tokio::task::yield_now().await;
                    matched
                }
            })
            .add_conditional_edge_async("start", "miss", |v| {
                let matched = v != &json!("go");
                async move {
                    tokio::task::yield_now().await;
                    matched
                }
            })
            .build()
            .unwrap();

        let run = workflow.run(json!("go")).await.unwrap();
        assert_eq!(run.outputs(), vec![json!("go")]);
    }

    #[tokio::test]
    async fn async_switch_case_routes() {
        // New: `Case::new_async`/`labeled_async` let a switch/case group's
        // branches await too, exercised through the existing `add_switch`
        // builder (which now awaits each case's condition in order).
        let workflow = WorkflowBuilder::new()
            .add_executor(echo("start"))
            .add_executor(echo("hot"))
            .add_executor(echo("cold"))
            .set_start("start")
            .add_switch(
                "start",
                vec![Case::new_async(
                    |v| {
                        let matched = v == &json!("go");
                        async move {
                            tokio::task::yield_now().await;
                            matched
                        }
                    },
                    "hot",
                )],
                SwitchDefault::new("cold"),
            )
            .build()
            .unwrap();

        let run = workflow.run(json!("go")).await.unwrap();
        assert_eq!(run.outputs(), vec![json!("go")]);

        let run = workflow.run(json!("stop")).await.unwrap();
        assert_eq!(run.outputs(), vec![json!("stop")]);
    }

    #[tokio::test]
    async fn send_to_unknown_target_fails_the_run() {
        // A dynamic `send_to` naming an executor that does not exist must fail
        // the run loudly rather than silently dropping the message and reporting
        // a clean `Idle` completion.
        let start = Arc::new(FunctionExecutor::new("start", |_msg, ctx| async move {
            ctx.send_to("ghost", json!("payload")).await
        }));
        let workflow = WorkflowBuilder::new()
            .add_executor(start)
            .add_executor(echo("sink"))
            .set_start("start")
            .add_edge("start", "sink")
            .build()
            .unwrap();

        let err = match workflow.run(json!("input")).await {
            Ok(_) => panic!("expected the run to fail on an unknown send_to target"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("ghost"),
            "error should name the unknown target executor: {msg}"
        );
        assert!(
            msg.contains("start"),
            "error should name the source executor: {msg}"
        );
    }

    #[tokio::test]
    async fn send_to_unknown_target_emits_failed_event() {
        // The failing routing must surface as a terminal `Failed` event on the
        // event stream, not just a returned `Err`.
        let start = Arc::new(FunctionExecutor::new("start", |_msg, ctx| async move {
            ctx.send_to("ghost", json!("payload")).await
        }));
        let workflow = WorkflowBuilder::new()
            .add_executor(start)
            .add_executor(echo("sink"))
            .set_start("start")
            .add_edge("start", "sink")
            .build()
            .unwrap();

        let stream = workflow.run_stream(json!("input"));
        let events: Vec<WorkflowEvent> = stream.collect().await;
        assert!(
            events
                .iter()
                .any(|e| matches!(e, WorkflowEvent::Failed { error } if error.contains("ghost"))),
            "expected a Failed event naming the unknown target, got {events:?}"
        );
    }
}
