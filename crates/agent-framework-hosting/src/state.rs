//! Execution state shared by hosting routes.
//!
//! Ports upstream's `agent-framework-hosting` package (`_state.py`): two
//! independent holders, one per target kind, because agents and workflows
//! keep different continuation state.
//!
//! - [`AgentState`] pairs an agent with a
//!   [`SessionStore`]
//!   (`session_id -> AgentSession`).
//! - [`WorkflowState`] resolves a workflow. Workflow continuation uses
//!   [`CheckpointStorage`](agent_framework_core::workflow::CheckpointStorage)
//!   directly; [`AgentHost`](crate::AgentHost) keeps one storage per
//!   conversation on top of it.
//!
//! Neither owns routes, middleware or protocol framing — those belong to the
//! web layer ([`crate::devui`], or an application's own `axum` routes).
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use agent_framework_core::agent::SupportsAgentRun;
//! # use agent_framework_core::types::Message;
//! # use agent_framework_hosting::AgentState;
//! # async fn demo(agent: Arc<dyn SupportsAgentRun>) -> agent_framework_core::Result<()> {
//! let state = AgentState::new(agent);
//! // Continue from an earlier response id: a working copy of its session.
//! let mut session = state.get_or_create_session("resp_previous").await?;
//! let target = state.get_target().await?;
//! target.run(vec![Message::user("Hello")], Some(&mut session)).await?;
//! // Store the post-run session under the id handed back to the caller.
//! state.set_session("resp_next", &session).await?;
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use tokio::sync::OnceCell;

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::session::AgentSession;
use agent_framework_core::session_store::{InMemorySessionStore, SessionStore};
use agent_framework_core::workflow::{Workflow, WorkflowBuilder};
use agent_framework_core::Result;

type Factory<T> = Arc<dyn Fn() -> BoxFuture<'static, Result<T>> + Send + Sync>;

/// Where a target comes from: an instance, or a factory run once (cached)
/// or on every request.
struct Target<T: Clone + Send + Sync + 'static> {
    instance: Option<T>,
    factory: Option<Factory<T>>,
    cache: bool,
    cached: OnceCell<T>,
}

impl<T: Clone + Send + Sync + 'static> Target<T> {
    fn instance(target: T) -> Self {
        Self {
            instance: Some(target),
            factory: None,
            cache: true,
            cached: OnceCell::new(),
        }
    }

    fn factory(factory: Factory<T>, cache: bool) -> Self {
        Self {
            instance: None,
            factory: Some(factory),
            cache,
            cached: OnceCell::new(),
        }
    }

    async fn get(&self) -> Result<T> {
        if let Some(t) = &self.instance {
            return Ok(t.clone());
        }
        let factory = self
            .factory
            .as_ref()
            .expect("a target is an instance or a factory");
        if !self.cache {
            return factory().await;
        }
        // `get_or_try_init` runs the factory once even under concurrent
        // first requests, and retries on a later call if it failed.
        self.cached.get_or_try_init(|| factory()).await.cloned()
    }

    fn get_sync(&self) -> Option<T> {
        self.instance.clone().or_else(|| self.cached.get().cloned())
    }
}

/// Shared execution state for routes hosting one agent. Mirrors upstream's
/// `AgentState`.
///
/// Holds the agent (an instance or a factory) and a [`SessionStore`].
/// [`get_or_create_session`](Self::get_or_create_session) is the one place
/// a session is minted for an id the store has not seen, since only this
/// object has both the store and the agent.
///
/// # Continuation patterns
///
/// A protocol that mints a new id per response (OpenAI Responses'
/// `previous_response_id`) reads the previous id and stores the post-run
/// session under the *new* one: the old snapshot is immutable, so two
/// callers may branch from it. A stable conversation id is a mutable head:
/// the completed session is written back under the same id, and only one
/// caller should advance it at a time — this type serializes session
/// *creation*, not that read-run-write cycle.
pub struct AgentState {
    target: Target<Arc<dyn SupportsAgentRun>>,
    store: Arc<dyn SessionStore>,
    creation_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl AgentState {
    /// State for `agent` with a fresh [`InMemorySessionStore`].
    pub fn new(agent: Arc<dyn SupportsAgentRun>) -> Self {
        Self::with_target(Target::instance(agent))
    }

    /// State whose agent is built by `factory`. With `cache` the factory
    /// runs once, on first use, and its result is reused (expensive setup
    /// happens once); without it, every [`get_target`](Self::get_target)
    /// builds a new agent.
    pub fn from_factory<F, Fut>(factory: F, cache: bool) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Arc<dyn SupportsAgentRun>>> + Send + 'static,
    {
        let factory: Factory<Arc<dyn SupportsAgentRun>> = Arc::new(move || Box::pin(factory()));
        Self::with_target(Target::factory(factory, cache))
    }

    fn with_target(target: Target<Arc<dyn SupportsAgentRun>>) -> Self {
        Self {
            target,
            store: Arc::new(InMemorySessionStore::new()),
            creation_locks: Mutex::new(HashMap::new()),
        }
    }

    /// Use `store` instead of the default in-memory one.
    pub fn with_session_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.store = store;
        self
    }

    /// The resolved agent, building it first if this state holds a factory.
    pub async fn get_target(&self) -> Result<Arc<dyn SupportsAgentRun>> {
        self.target.get().await
    }

    /// The agent, if it is available without awaiting: an instance, or a
    /// cached factory result that has already been built.
    pub fn target(&self) -> Option<Arc<dyn SupportsAgentRun>> {
        self.target.get_sync()
    }

    /// This state's session store.
    pub fn session_store(&self) -> &Arc<dyn SessionStore> {
        &self.store
    }

    /// A working copy of the session stored under `session_id`, creating
    /// (and storing) one with that id when there is none.
    ///
    /// Creation is serialized per id, so two concurrent first requests for
    /// one id observe the same new session rather than racing to store two.
    pub async fn get_or_create_session(&self, session_id: &str) -> Result<AgentSession> {
        let lock = {
            let mut locks = self
                .creation_locks
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            locks.entry(session_id.to_string()).or_default().clone()
        };
        let _guard = lock.lock().await;
        if let Some(session) = self.store.get(session_id).await? {
            return Ok(session);
        }
        let session = self
            .get_target()
            .await?
            .create_session()
            .with_session_id(session_id);
        self.store.set(session_id, &session).await?;
        // Hand back the store's copy, not `session`, so the caller holds an
        // independent working copy exactly as it would on a later read.
        Ok(self.store.get(session_id).await?.unwrap_or(session))
    }

    /// Store `session` under `session_id`.
    pub async fn set_session(&self, session_id: &str, session: &AgentSession) -> Result<()> {
        self.store.set(session_id, session).await
    }
}

/// Shared execution state for routes hosting one workflow. Mirrors
/// upstream's `WorkflowState`.
///
/// Holds the workflow: an instance, a [`WorkflowBuilder`] (built on first
/// use), or a factory. Checkpointing is not owned here — pass a
/// [`CheckpointStorage`](agent_framework_core::workflow::CheckpointStorage)
/// to [`Workflow::run_with_checkpointing`] and resume with
/// [`Workflow::run_from_checkpoint`].
pub struct WorkflowState {
    target: Target<Workflow>,
}

impl WorkflowState {
    /// State for an already-built workflow.
    pub fn new(workflow: Workflow) -> Self {
        Self {
            target: Target::instance(workflow),
        }
    }

    /// State for a workflow `builder`, built once on first use. Upstream
    /// accepts any object with a `build()` (the orchestration builders
    /// included); in Rust each orchestration builder's `build()` yields a
    /// [`Workflow`], so pass it through [`from_factory`](Self::from_factory)
    /// or build it first.
    pub fn from_builder(builder: WorkflowBuilder) -> Self {
        let builder = Arc::new(Mutex::new(Some(builder)));
        Self::from_factory(
            move || {
                let builder = builder.clone();
                async move {
                    let builder = builder
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take()
                        .ok_or_else(|| {
                            agent_framework_core::Error::Workflow(
                                "workflow builder was already consumed by a failed build".into(),
                            )
                        })?;
                    builder.build()
                }
            },
            true,
        )
    }

    /// State whose workflow is built by `factory`; see
    /// [`AgentState::from_factory`] for `cache`.
    pub fn from_factory<F, Fut>(factory: F, cache: bool) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Workflow>> + Send + 'static,
    {
        let factory: Factory<Workflow> = Arc::new(move || Box::pin(factory()));
        Self {
            target: Target::factory(factory, cache),
        }
    }

    /// The resolved workflow, building it first if needed.
    pub async fn get_target(&self) -> Result<Workflow> {
        self.target.get().await
    }

    /// The workflow, if it is available without awaiting.
    pub fn target(&self) -> Option<Workflow> {
        self.target.get_sync()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::agent::AgentRunStream;
    use agent_framework_core::types::{AgentResponse, Message};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Echo;

    #[async_trait]
    impl SupportsAgentRun for Echo {
        async fn run(
            &self,
            messages: Vec<Message>,
            session: Option<&mut AgentSession>,
        ) -> Result<AgentResponse> {
            if let Some(s) = session {
                let n = s.state.get("turns").and_then(|v| v.as_u64()).unwrap_or(0);
                s.state.insert("turns", serde_json::json!(n + 1));
            }
            Ok(AgentResponse {
                messages: vec![Message::assistant(
                    messages.last().map(|m| m.text()).unwrap_or_default(),
                )],
                ..Default::default()
            })
        }
        async fn run_stream(
            &self,
            _messages: Vec<Message>,
            _session: Option<AgentSession>,
            _options: Option<agent_framework_core::agent::AgentRunOptions>,
        ) -> Result<AgentRunStream> {
            unreachable!("not streamed in these tests")
        }
        fn id(&self) -> &str {
            "echo"
        }
    }

    fn turns(s: &AgentSession) -> u64 {
        s.state.get("turns").and_then(|v| v.as_u64()).unwrap_or(0)
    }

    #[tokio::test]
    async fn a_new_id_gets_a_session_carrying_that_id() {
        let state = AgentState::new(Arc::new(Echo));
        let s = state.get_or_create_session("resp_1").await.unwrap();
        assert_eq!(s.session_id(), "resp_1");
        assert!(state.session_store().get("resp_1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn branches_from_one_response_id_do_not_see_each_other() {
        let state = AgentState::new(Arc::new(Echo));
        let target = state.get_target().await.unwrap();

        let mut root = state.get_or_create_session("resp_a").await.unwrap();
        target
            .run(vec![Message::user("1")], Some(&mut root))
            .await
            .unwrap();
        state.set_session("resp_b", &root).await.unwrap();

        // Two continuations of resp_b, each stored under its own id.
        for next in ["resp_c1", "resp_c2"] {
            let mut s = state.get_or_create_session("resp_b").await.unwrap();
            target
                .run(vec![Message::user("2")], Some(&mut s))
                .await
                .unwrap();
            state.set_session(next, &s).await.unwrap();
        }
        let store = state.session_store();
        assert_eq!(turns(&store.get("resp_b").await.unwrap().unwrap()), 1);
        assert_eq!(turns(&store.get("resp_c1").await.unwrap().unwrap()), 2);
        assert_eq!(turns(&store.get("resp_c2").await.unwrap().unwrap()), 2);
    }

    #[tokio::test]
    async fn a_cached_factory_runs_once_and_an_uncached_one_every_time() {
        for (cache, expected) in [(true, 1), (false, 3)] {
            let calls = Arc::new(AtomicUsize::new(0));
            let c = calls.clone();
            let state = AgentState::from_factory(
                move || {
                    c.fetch_add(1, Ordering::SeqCst);
                    async { Ok(Arc::new(Echo) as Arc<dyn SupportsAgentRun>) }
                },
                cache,
            );
            assert!(state.target().is_none());
            for _ in 0..3 {
                state.get_target().await.unwrap();
            }
            assert_eq!(calls.load(Ordering::SeqCst), expected, "cache={cache}");
            assert_eq!(state.target().is_some(), cache);
        }
    }

    #[tokio::test]
    async fn concurrent_first_requests_share_one_session() {
        let state = Arc::new(AgentState::new(Arc::new(Echo)));
        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let state = state.clone();
                tokio::spawn(async move { state.get_or_create_session("conv_x").await.unwrap() })
            })
            .collect();
        let mut ids = std::collections::HashSet::new();
        for t in tasks {
            ids.insert(t.await.unwrap().session_id().to_string());
        }
        assert_eq!(ids.len(), 1);
    }

    #[tokio::test]
    async fn workflow_state_builds_a_builder_once() {
        use agent_framework_core::workflow::FunctionExecutor;
        let builder = WorkflowBuilder::new()
            .add_executor(Arc::new(FunctionExecutor::new(
                "start",
                |v, ctx| async move { ctx.yield_output(v).await },
            )))
            .set_start("start");
        let state = WorkflowState::from_builder(builder);
        assert!(state.target().is_none());
        let a = state.get_target().await.unwrap();
        let b = state.get_target().await.unwrap();
        assert_eq!(a.id(), b.id());
        let run = a.run(serde_json::json!("hi")).await.unwrap();
        assert_eq!(run.outputs(), vec![serde_json::json!("hi")]);
    }
}
