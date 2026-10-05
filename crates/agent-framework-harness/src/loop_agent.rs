//! Re-running an agent in a loop until a criterion is met.
//!
//! Rust equivalent of upstream `agent_framework._harness._loop`
//! (`AgentLoopMiddleware`, `JudgeVerdict`, `todos_remaining`,
//! `todos_remaining_message`, `background_tasks_running`,
//! `background_tasks_running_message`) and of .NET's `LoopAgent`.
//!
//! # Divergences
//!
//! - **A decorator agent, not agent middleware** — for the reason given in
//!   [`tool_approval`](crate::tool_approval): a Rust agent middleware sees
//!   no session and cannot re-run the context providers, so [`LoopAgent`]
//!   wraps any [`SupportsAgentRun`] (as .NET's `LoopAgent` does), each
//!   iteration being a complete inner run.
//! - **Typed callbacks.** Upstream passes callbacks keyword arguments
//!   (`iteration`, `last_result`, `messages`, `original_messages`, `session`,
//!   `agent`, `progress`, `feedback`); here they receive one owned
//!   [`LoopContext`] with the same fields. Upstream's `agent` lets helpers
//!   find the harness providers on `agent.context_providers`; Rust agents do
//!   not expose their providers, so the context carries the relevant ones
//!   directly in [`LoopContext::providers`] (filled in by
//!   [`HarnessAgent`](crate::agent::HarnessAgent)).
//! - **`fresh_context` rollback** restores `session.state` and the service
//!   conversation id from a pre-loop snapshot, exactly as upstream does.
//!   Upstream's default history lives in `session.state`, so that also
//!   rolls back the transcript; the harness's default
//!   [`SessionStateHistoryProvider`](crate::history::SessionStateHistoryProvider)
//!   keeps that property, but a history provider that stores messages
//!   elsewhere (e.g. core's `InMemoryHistoryProvider`) is not rolled back.
//! - Upstream's turn-scoped `after_run_once_per_turn` provider hook has no
//!   Rust counterpart and is not modelled.

use std::sync::Arc;

use agent_framework_core::agent::{AgentRunOptions, AgentRunStream, SupportsAgentRun};
use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::BoxFuture;
use agent_framework_core::types::{
    AgentResponse, AgentResponseUpdate, ChatOptions, ChatResponse, Content, Message,
    ResponseFormat, UsageDetails,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::background_agents::BackgroundAgentsProvider;
use crate::mode::AgentModeProvider;
use crate::todo::TodoProvider;
use crate::util::SessionRef;

/// The default "continue" nudge. Mirrors upstream `DEFAULT_NEXT_MESSAGE`.
pub const DEFAULT_NEXT_MESSAGE: &str = "Continue working on the task. If it is complete, say so.";
/// Placeholder for the rendered criteria in judge instructions. Mirrors
/// upstream `CRITERIA_PLACEHOLDER`.
pub const CRITERIA_PLACEHOLDER: &str = "{{criteria}}";
/// Positive text verdict marker. Mirrors upstream `JUDGE_VERDICT_DONE`.
pub const JUDGE_VERDICT_DONE: &str = "VERDICT: DONE";
/// Negative text verdict marker. Mirrors upstream `JUDGE_VERDICT_MORE`.
pub const JUDGE_VERDICT_MORE: &str = "VERDICT: MORE";
/// Default judge instructions. Mirrors upstream `DEFAULT_JUDGE_INSTRUCTIONS`.
pub const DEFAULT_JUDGE_INSTRUCTIONS: &str = concat!(
    "You are an evaluator. You are given a user's original request and an agent's latest response. ",
    "Decide whether the agent has fully addressed the original request. ",
    "Set 'answered' to true if the request has been fully addressed, or false if more work is still ",
    "required, and use 'reasoning' to briefly justify your decision. ",
    "If you cannot return structured output, end your reply with a line reading exactly ",
    "'VERDICT: DONE' when the request has been fully addressed or 'VERDICT: MORE' ",
    "when more work is still required.",
    "{{criteria}}",
);
/// Default iteration cap. Mirrors upstream `DEFAULT_MAX_ITERATIONS`.
pub const DEFAULT_MAX_ITERATIONS: usize = 10;
/// Default iteration cap of judge loops. Mirrors upstream
/// `DEFAULT_JUDGE_MAX_ITERATIONS`.
pub const DEFAULT_JUDGE_MAX_ITERATIONS: usize = 5;

/// The structured verdict a judge returns. Mirrors upstream `JudgeVerdict`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeVerdict {
    /// Whether the original request has been fully addressed.
    pub answered: bool,
    /// Brief justification.
    #[serde(default)]
    pub reasoning: String,
}

impl JudgeVerdict {
    /// The JSON-schema response format requesting a verdict.
    pub fn response_format() -> ResponseFormat {
        ResponseFormat::json_schema(
            "JudgeVerdict",
            json!({
                "type": "object",
                "properties": {
                    "answered": {"type": "boolean", "description": "True if the agent has fully addressed the original request and it adheres to the other judging standards, otherwise False."},
                    "reasoning": {"type": "string", "default": "", "description": "Brief justification for the verdict."},
                },
                "required": ["answered"],
                "title": "JudgeVerdict",
            }),
        )
    }
}

/// The harness providers a loop callback may need (upstream resolves them
/// from `agent.context_providers`).
#[derive(Clone, Debug, Default)]
pub struct HarnessProviders {
    /// The todo provider, when wired.
    pub todo: Option<TodoProvider>,
    /// The mode provider, when wired.
    pub mode: Option<AgentModeProvider>,
    /// The background-agents provider, when wired.
    pub background_agents: Option<BackgroundAgentsProvider>,
}

/// What a loop callback receives — upstream's loop keyword arguments.
#[derive(Clone, Debug)]
pub struct LoopContext {
    /// Completed runs so far (1-based after the first run).
    pub iteration: usize,
    /// The result of the iteration that just completed.
    pub last_result: AgentResponse,
    /// The messages that iteration ran with.
    pub messages: Vec<Message>,
    /// The first iteration's input.
    pub original_messages: Vec<Message>,
    /// The run's session (a clone sharing its state), when there is one.
    pub session: Option<AgentSession>,
    /// The harness providers wired on the looped agent.
    pub providers: HarnessProviders,
    /// The progress log so far (a copy).
    pub progress: Vec<String>,
    /// The feedback `should_continue` returned this iteration, if any.
    pub feedback: Option<String>,
}

/// A `should_continue` decision: whether to run again, plus optional
/// feedback surfaced to the next-message and record-feedback callbacks.
/// Upstream accepts `bool | (bool, str | None)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoopDecision {
    /// Run the agent again.
    pub continue_loop: bool,
    /// Feedback for this iteration.
    pub feedback: Option<String>,
}

impl From<bool> for LoopDecision {
    fn from(continue_loop: bool) -> Self {
        Self {
            continue_loop,
            feedback: None,
        }
    }
}

impl From<(bool, Option<String>)> for LoopDecision {
    fn from((continue_loop, feedback): (bool, Option<String>)) -> Self {
        Self {
            continue_loop,
            feedback,
        }
    }
}

/// Decides whether to run again. Upstream `ShouldContinueCallable`.
pub type ShouldContinueFn =
    Arc<dyn Fn(LoopContext) -> BoxFuture<Result<LoopDecision>> + Send + Sync>;
/// Produces the next iteration's input; `None` reuses the previous input
/// verbatim. Upstream `NextMessageCallable`.
pub type NextMessageFn =
    Arc<dyn Fn(LoopContext) -> BoxFuture<Result<Option<Vec<Message>>>> + Send + Sync>;
/// Captures one progress-log entry per iteration. Upstream `FeedbackCallable`.
pub type RecordFeedbackFn =
    Arc<dyn Fn(LoopContext) -> BoxFuture<Result<Option<String>>> + Send + Sync>;
/// Converts a judge's full response into a verdict.
pub type VerdictParserFn =
    Arc<dyn Fn(ChatResponse) -> BoxFuture<Result<JudgeVerdict>> + Send + Sync>;

/// Wrap a synchronous predicate as a [`ShouldContinueFn`].
pub fn should_continue_fn<F, D>(f: F) -> ShouldContinueFn
where
    F: Fn(&LoopContext) -> D + Send + Sync + 'static,
    D: Into<LoopDecision>,
{
    Arc::new(move |ctx| {
        let decision = f(&ctx).into();
        Box::pin(async move { Ok(decision) })
    })
}

/// Wrap a synchronous text producer as a [`NextMessageFn`] (the text becomes
/// one user message; `None` reuses the previous input).
pub fn next_message_fn<F>(f: F) -> NextMessageFn
where
    F: Fn(&LoopContext) -> Option<String> + Send + Sync + 'static,
{
    Arc::new(move |ctx| {
        let next = f(&ctx).map(|text| vec![Message::user(text)]);
        Box::pin(async move { Ok(next) })
    })
}

/// Wrap a synchronous entry producer as a [`RecordFeedbackFn`].
pub fn record_feedback_fn<F>(f: F) -> RecordFeedbackFn
where
    F: Fn(&LoopContext) -> Option<String> + Send + Sync + 'static,
{
    Arc::new(move |ctx| {
        let entry = f(&ctx);
        Box::pin(async move { Ok(entry) })
    })
}

/// Options of [`LoopAgent::with_judge`].
#[derive(Clone)]
pub struct JudgeOptions {
    /// Criteria the response must satisfy: injected as an agent system
    /// instruction and rendered into the judge instructions at
    /// [`CRITERIA_PLACEHOLDER`].
    pub criteria: Vec<String>,
    /// Judge instructions (default [`DEFAULT_JUDGE_INSTRUCTIONS`]).
    pub instructions: Option<String>,
    /// Structured format requested from the judge (default
    /// [`JudgeVerdict::response_format`]; `None` sends none).
    pub response_format: Option<ResponseFormat>,
    /// Custom verdict parser; owns interpretation completely (no text
    /// fallback).
    pub verdict_parser: Option<VerdictParserFn>,
    /// Iteration cap (default [`DEFAULT_JUDGE_MAX_ITERATIONS`]).
    pub max_iterations: Option<usize>,
    /// Custom next-message callable (default relays the judge's reasoning).
    pub next_message: Option<NextMessageFn>,
    /// Restart each iteration from the original input.
    pub fresh_context: bool,
}

impl Default for JudgeOptions {
    fn default() -> Self {
        Self {
            criteria: Vec::new(),
            instructions: None,
            response_format: Some(JudgeVerdict::response_format()),
            verdict_parser: None,
            max_iterations: Some(DEFAULT_JUDGE_MAX_ITERATIONS),
            next_message: None,
            fresh_context: false,
        }
    }
}

/// Re-runs an inner agent in a loop until a criterion is met.
///
/// Mirrors upstream `AgentLoopMiddleware` / .NET `LoopAgent` (see the
/// [module docs](self) for divergences). After each iteration:
///
/// 1. an iteration that returned a pending tool-approval request stops the
///    loop and is returned so a human can approve (the escape hatch);
/// 2. [`max_iterations`](Self::max_iterations) (default 10, `None` =
///    unbounded) short-circuits before `should_continue` is consulted;
/// 3. a progress entry is recorded (via `record_feedback`, else the
///    response text);
/// 4. the next input comes from `next_message` (default
///    [`DEFAULT_NEXT_MESSAGE`]), preceded — with `inject_progress` — by a
///    `Progress so far:` user message (only the latest entry when a session
///    retains the earlier ones; the full log without a session or with
///    `fresh_context`).
///
/// A non-streaming run returns every iteration's messages plus the injected
/// nudges, usage summed ([`return_final_only`](Self::return_final_only)
/// returns just the last iteration). Streaming forwards each iteration's
/// updates and emits the nudges as `user` updates between iterations.
#[derive(Clone)]
pub struct LoopAgent {
    inner: Arc<dyn SupportsAgentRun>,
    should_continue: ShouldContinueFn,
    max_iterations: Option<usize>,
    next_message: Option<NextMessageFn>,
    record_feedback: Option<RecordFeedbackFn>,
    inject_progress: bool,
    fresh_context: bool,
    return_final_only: bool,
    additional_instructions: Option<String>,
    providers: HarnessProviders,
}

impl LoopAgent {
    /// Loop `inner` while `should_continue` says so (default cap 10).
    pub fn new(inner: Arc<dyn SupportsAgentRun>, should_continue: ShouldContinueFn) -> Self {
        Self {
            inner,
            should_continue,
            max_iterations: Some(DEFAULT_MAX_ITERATIONS),
            next_message: None,
            record_feedback: None,
            inject_progress: true,
            fresh_context: false,
            return_final_only: false,
            additional_instructions: None,
            providers: HarnessProviders::default(),
        }
    }

    /// Loop until `judge_client` decides the original request was answered.
    ///
    /// **Security:** the judge is sent the original request and the agent's
    /// latest response every iteration, and its reasoning is fed back to the
    /// agent; only point it at a service you trust as much as the primary
    /// model. Mirrors upstream `AgentLoopMiddleware.with_judge`.
    pub fn with_judge(
        inner: Arc<dyn SupportsAgentRun>,
        judge_client: Arc<dyn ChatClient>,
        options: JudgeOptions,
    ) -> Result<Self> {
        let instructions = options
            .instructions
            .clone()
            .unwrap_or_else(|| DEFAULT_JUDGE_INSTRUCTIONS.to_string())
            .replace(
                CRITERIA_PLACEHOLDER,
                &render_criteria_block(&options.criteria),
            );
        let should_continue = judge_condition(
            judge_client,
            instructions,
            options.response_format,
            options.verdict_parser,
        );
        let next = options.next_message.unwrap_or_else(judge_next_message);
        let mut agent = Self::new(inner, should_continue)
            .max_iterations(options.max_iterations)?
            .next_message(next)
            .fresh_context(options.fresh_context);
        if !options.criteria.is_empty() {
            agent.additional_instructions = Some(criteria_agent_instruction(&options.criteria));
        }
        Ok(agent)
    }

    /// Safety cap on agent runs; `None` is unbounded. Errors on `Some(0)`.
    pub fn max_iterations(mut self, max: Option<usize>) -> Result<Self> {
        if max == Some(0) {
            return Err(Error::Configuration(
                "max_iterations must be None or a positive integer (>= 1).".into(),
            ));
        }
        self.max_iterations = max;
        Ok(self)
    }

    /// Produce the next iteration's input.
    pub fn next_message(mut self, f: NextMessageFn) -> Self {
        self.next_message = Some(f);
        self
    }

    /// Capture a progress entry per iteration (default: the response text).
    pub fn record_feedback(mut self, f: RecordFeedbackFn) -> Self {
        self.record_feedback = Some(f);
        self
    }

    /// Inject the progress log into the next input (default `true`).
    pub fn inject_progress(mut self, inject: bool) -> Self {
        self.inject_progress = inject;
        self
    }

    /// Restart each iteration from the original input plus the progress
    /// log, restoring the session to its pre-loop snapshot.
    pub fn fresh_context(mut self, fresh: bool) -> Self {
        self.fresh_context = fresh;
        self
    }

    /// Return only the final iteration's response (non-streaming).
    pub fn return_final_only(mut self, final_only: bool) -> Self {
        self.return_final_only = final_only;
        self
    }

    /// An extra instruction injected as a `system` message ahead of the
    /// input (preserved across `fresh_context` resets).
    pub fn additional_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.additional_instructions = Some(instructions.into());
        self
    }

    /// The harness providers handed to callbacks via [`LoopContext::providers`].
    pub fn providers(mut self, providers: HarnessProviders) -> Self {
        self.providers = providers;
        self
    }

    /// The configured iteration cap.
    pub fn get_max_iterations(&self) -> Option<usize> {
        self.max_iterations
    }

    #[allow(clippy::too_many_arguments)]
    fn context(
        &self,
        iteration: usize,
        last_result: &AgentResponse,
        messages: &[Message],
        original: &[Message],
        session: Option<&AgentSession>,
        progress: &[String],
        feedback: Option<String>,
    ) -> LoopContext {
        LoopContext {
            iteration,
            last_result: last_result.clone(),
            messages: messages.to_vec(),
            original_messages: original.to_vec(),
            session: session.cloned(),
            providers: self.providers.clone(),
            progress: progress.to_vec(),
            feedback,
        }
    }

    async fn evaluate_stop(
        &self,
        ctx: LoopContext,
        work_iterations: usize,
    ) -> Result<(bool, Option<String>)> {
        if self
            .max_iterations
            .is_some_and(|max| work_iterations >= max)
        {
            return Ok((true, None));
        }
        let decision = (self.should_continue)(ctx).await?;
        Ok((!decision.continue_loop, decision.feedback))
    }

    async fn record_progress(&self, ctx: LoopContext, progress: &mut Vec<String>) -> Result<()> {
        let entry = match &self.record_feedback {
            Some(f) => f(ctx).await?,
            None => Some(ctx.last_result.text().trim().to_string()),
        };
        if let Some(entry) = entry.filter(|e| !e.is_empty()) {
            progress.push(entry);
        }
        Ok(())
    }

    async fn resolve_next_message(
        &self,
        ctx: LoopContext,
        has_session: bool,
    ) -> Result<Vec<Message>> {
        let next = match &self.next_message {
            None => vec![Message::user(DEFAULT_NEXT_MESSAGE)],
            Some(f) => match f(ctx.clone()).await? {
                Some(messages) => messages,
                None if !self.fresh_context => return Ok(ctx.messages),
                None => vec![Message::user(DEFAULT_NEXT_MESSAGE)],
            },
        };
        let progress_message = if self.inject_progress && !ctx.progress.is_empty() {
            let entries = if !has_session || self.fresh_context {
                &ctx.progress[..]
            } else {
                &ctx.progress[ctx.progress.len() - 1..]
            };
            Some(render_progress(entries))
        } else {
            None
        };
        let mut out = Vec::new();
        if self.fresh_context {
            out.extend(ctx.original_messages);
        }
        out.extend(progress_message);
        out.extend(next);
        Ok(out)
    }

    /// One iteration's bookkeeping after a run. Returns the next input, or
    /// `None` to stop.
    #[allow(clippy::too_many_arguments)]
    async fn after_iteration(
        &self,
        iteration: usize,
        work_iterations: &mut usize,
        result: &AgentResponse,
        messages: &[Message],
        original: &[Message],
        session: Option<&mut AgentSession>,
        snapshot: Option<&serde_json::Value>,
        progress: &mut Vec<String>,
    ) -> Result<Option<Vec<Message>>> {
        if has_pending_approval_request(result) {
            return Ok(None);
        }
        let session_view = session.as_deref().cloned();
        *work_iterations += 1;
        let ctx = self.context(
            iteration,
            result,
            messages,
            original,
            session_view.as_ref(),
            progress,
            None,
        );
        let (stop, feedback) = self.evaluate_stop(ctx, *work_iterations).await?;
        let ctx = self.context(
            iteration,
            result,
            messages,
            original,
            session_view.as_ref(),
            progress,
            feedback.clone(),
        );
        self.record_progress(ctx, progress).await?;
        if stop {
            return Ok(None);
        }
        let has_session = session.is_some();
        if let (Some(snapshot), Some(session)) = (snapshot, session) {
            restore_session(session, snapshot)?;
        }
        let ctx = self.context(
            iteration,
            result,
            messages,
            original,
            session_view.as_ref(),
            progress,
            feedback,
        );
        Ok(Some(self.resolve_next_message(ctx, has_session).await?))
    }

    fn initial_messages(&self, messages: Vec<Message>) -> Vec<Message> {
        match &self.additional_instructions {
            Some(instructions) => {
                let mut out = vec![Message::system(instructions.clone())];
                out.extend(messages);
                out
            }
            None => messages,
        }
    }
}

/// Whether `result` carries a pending tool-approval request. Mirrors
/// upstream `AgentLoopMiddleware._has_pending_approval_request`.
pub fn has_pending_approval_request(result: &AgentResponse) -> bool {
    result
        .messages
        .iter()
        .flat_map(|m| &m.contents)
        .any(|c| matches!(c, Content::FunctionApprovalRequest(_)))
}

fn render_progress(entries: &[String]) -> Message {
    let body = entries
        .iter()
        .map(|e| format!("- {e}"))
        .collect::<Vec<_>>()
        .join("\n");
    Message::user(format!("Progress so far:\n{body}"))
}

fn render_criteria_block(criteria: &[String]) -> String {
    if criteria.is_empty() {
        return String::new();
    }
    let bullets = criteria
        .iter()
        .map(|c| format!("- {c}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("\n\nThe response must satisfy all of the following criteria:\n{bullets}")
}

fn criteria_agent_instruction(criteria: &[String]) -> String {
    let bullets = criteria
        .iter()
        .map(|c| format!("- {c}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("Your response must satisfy all of the following criteria:\n{bullets}")
}

/// Restore `session` in place to a `to_dict()` snapshot: state and service
/// conversation id reset, id and context providers kept. Mirrors upstream
/// `_restore_session`.
pub fn restore_session(session: &mut AgentSession, snapshot: &serde_json::Value) -> Result<()> {
    let restored_view = AgentSession::from_dict(snapshot)?;
    // Refill the live state handle so every holder observes the reset.
    for key in session.state.snapshot().keys() {
        session.state.remove(key);
    }
    for (key, value) in restored_view.state.snapshot() {
        session.state.insert(key, value);
    }
    let mut restored = AgentSession::from_dict(snapshot)?;
    restored.state = session.state.clone();
    restored.context_providers = std::mem::take(&mut session.context_providers);
    *session = restored;
    Ok(())
}

fn aggregate(
    final_result: AgentResponse,
    messages: Vec<Message>,
    usage: Option<UsageDetails>,
) -> AgentResponse {
    AgentResponse {
        messages,
        usage_details: usage,
        ..final_result
    }
}

fn add_usage(total: &mut Option<UsageDetails>, usage: Option<&UsageDetails>) {
    if let Some(u) = usage {
        match total {
            Some(t) => t.add_assign(u),
            None => *total = Some(u.clone()),
        }
    }
}

fn message_to_update(message: &Message) -> AgentResponseUpdate {
    AgentResponseUpdate {
        contents: message.contents.clone(),
        role: Some(message.role.clone()),
        author_name: message.author_name.clone(),
        message_id: message.message_id.clone(),
        ..Default::default()
    }
}

#[async_trait]
impl SupportsAgentRun for LoopAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        self.run_with_options(messages, session, AgentRunOptions::default())
            .await
    }

    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        mut session: Option<&mut AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        let mut messages = self.initial_messages(messages);
        let original = messages.clone();
        let snapshot = match (&session, self.fresh_context) {
            (Some(s), true) => Some(s.to_dict()),
            _ => None,
        };
        let mut iteration = 0;
        let mut work_iterations = 0;
        let mut progress = Vec::new();
        let mut aggregated: Vec<Message> = Vec::new();
        let mut usage: Option<UsageDetails> = None;
        let final_result = loop {
            let result = self
                .inner
                .run_with_options(messages.clone(), session.as_deref_mut(), options.clone())
                .await?;
            iteration += 1;
            aggregated.extend(result.messages.iter().cloned());
            add_usage(&mut usage, result.usage_details.as_ref());
            let next = self
                .after_iteration(
                    iteration,
                    &mut work_iterations,
                    &result,
                    &messages,
                    &original,
                    session.as_deref_mut(),
                    snapshot.as_ref(),
                    &mut progress,
                )
                .await?;
            match next {
                None => break result,
                Some(next) => {
                    aggregated.extend(next.iter().cloned());
                    messages = next;
                }
            }
        };
        if self.return_final_only {
            return Ok(final_result);
        }
        Ok(aggregate(final_result, aggregated, usage))
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let (tx, rx) = futures::channel::mpsc::unbounded::<Result<AgentResponseUpdate>>();
        let this = self.clone();
        let options = options.unwrap_or_default();
        tokio::spawn(async move {
            let mut session = session;
            let mut messages = this.initial_messages(messages);
            let original = messages.clone();
            let snapshot = match (&session, this.fresh_context) {
                (Some(s), true) => Some(s.to_dict()),
                _ => None,
            };
            let mut iteration = 0;
            let mut work_iterations = 0;
            let mut progress = Vec::new();
            loop {
                let mut stream = match this
                    .inner
                    .run_stream(messages.clone(), session.clone(), Some(options.clone()))
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.unbounded_send(Err(e));
                        return;
                    }
                };
                let mut updates = Vec::new();
                while let Some(update) = stream.next().await {
                    match update {
                        Ok(u) => {
                            updates.push(u.clone());
                            if tx.unbounded_send(Ok(u)).is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            let _ = tx.unbounded_send(Err(e));
                            return;
                        }
                    }
                }
                let result = AgentResponse::from_updates(updates);
                iteration += 1;
                let next = this
                    .after_iteration(
                        iteration,
                        &mut work_iterations,
                        &result,
                        &messages,
                        &original,
                        session.as_mut(),
                        snapshot.as_ref(),
                        &mut progress,
                    )
                    .await;
                match next {
                    Err(e) => {
                        let _ = tx.unbounded_send(Err(e));
                        return;
                    }
                    Ok(None) => return,
                    Ok(Some(next)) => {
                        for message in &next {
                            if tx.unbounded_send(Ok(message_to_update(message))).is_err() {
                                return;
                            }
                        }
                        messages = next;
                    }
                }
            }
        });
        Ok(rx.boxed())
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    fn create_session(&self) -> AgentSession {
        self.inner.create_session()
    }
}

fn judge_condition(
    client: Arc<dyn ChatClient>,
    instructions: String,
    response_format: Option<ResponseFormat>,
    parser: Option<VerdictParserFn>,
) -> ShouldContinueFn {
    Arc::new(move |ctx: LoopContext| {
        let client = client.clone();
        let instructions = instructions.clone();
        let response_format = response_format.clone();
        let parser = parser.clone();
        Box::pin(async move {
            let mut messages = vec![
                Message::system(instructions),
                Message::user("Evaluate the agent's work. The user's original request follows:"),
            ];
            messages.extend(ctx.original_messages.iter().cloned());
            messages.push(Message::user("The agent's latest response was:"));
            messages.extend(ctx.last_result.messages.iter().cloned());
            messages.push(Message::user(
                "Has the original request been fully addressed?",
            ));
            let mut options = ChatOptions::new();
            options.response_format = response_format.clone();
            let mut response = client.get_response(messages, options).await?;
            let (answered, reasoning) = match parser {
                Some(parser) => {
                    let verdict = parser(response).await?;
                    (verdict.answered, verdict.reasoning)
                }
                None => {
                    response.try_parse_value(
                        response_format
                            .as_ref()
                            .or(Some(&ResponseFormat::JsonObject)),
                    );
                    match response
                        .value
                        .clone()
                        .and_then(|v| serde_json::from_value::<JudgeVerdict>(v).ok())
                    {
                        Some(verdict) => (verdict.answered, verdict.reasoning),
                        None => {
                            // Text fallback: `MORE` wins, so an ambiguous or
                            // marker-less reply keeps looping.
                            let text = response.text();
                            let upper = text.to_uppercase();
                            let answered = !upper.contains(JUDGE_VERDICT_MORE)
                                && upper.contains(JUDGE_VERDICT_DONE);
                            (answered, text.trim().to_string())
                        }
                    }
                }
            };
            Ok(LoopDecision {
                continue_loop: !answered,
                feedback: Some(reasoning).filter(|r| !r.is_empty()),
            })
        })
    })
}

fn judge_next_message() -> NextMessageFn {
    Arc::new(|ctx: LoopContext| {
        let text = match ctx.feedback.as_deref().filter(|f| !f.is_empty()) {
            Some(feedback) => format!(
                "An evaluator reviewed your previous response and judged that it does not yet fully address the original request.\n\nEvaluator feedback: {feedback}\n\nRevise and continue so the original request is fully addressed."
            ),
            None => DEFAULT_NEXT_MESSAGE.to_string(),
        };
        Box::pin(async move { Ok(Some(vec![Message::user(text)])) })
    })
}

/// A `should_continue` that loops while the harness [`TodoProvider`] has
/// open items.
///
/// With `looping_modes`, the loop only continues while the current mode
/// (from the wired [`AgentModeProvider`], else the default mode helpers) is
/// one of them (case-insensitive) — e.g. loop only in `execute`. An empty
/// list is rejected. Stops when there is no session or no todo provider.
/// Mirrors upstream `todos_remaining`.
pub fn todos_remaining(looping_modes: Option<Vec<String>>) -> Result<ShouldContinueFn> {
    let allowed: Option<Vec<String>> = match looping_modes {
        Some(modes) => {
            let normalized: Vec<String> = modes.iter().map(|m| m.trim().to_lowercase()).collect();
            if normalized.is_empty() {
                return Err(Error::Configuration(
                    "looping_modes must be None or a non-empty sequence of mode names.".into(),
                ));
            }
            Some(normalized)
        }
        None => None,
    };
    let allowed = Arc::new(allowed);
    Ok(Arc::new(move |ctx: LoopContext| {
        let allowed = allowed.clone();
        Box::pin(async move {
            let Some(session) = &ctx.session else {
                return Ok(false.into());
            };
            if let Some(allowed) = allowed.as_ref() {
                let mode = match &ctx.providers.mode {
                    Some(provider) => provider.current_mode(&session.state)?,
                    None => crate::mode::get_agent_mode(
                        &session.state,
                        crate::mode::DEFAULT_MODE_SOURCE_ID,
                        None,
                        None,
                    )?,
                };
                if !allowed.contains(&mode.trim().to_lowercase()) {
                    return Ok(false.into());
                }
            }
            let Some(todo) = &ctx.providers.todo else {
                return Ok(false.into());
            };
            let items = todo.load_items(&SessionRef::from_session(session)).await?;
            Ok(items.iter().any(|i| !i.is_complete).into())
        })
    }))
}

/// A `next_message` that lists the still-open todos and asks the agent to
/// finish them; `None` when unavailable or nothing is open. Mirrors upstream
/// `todos_remaining_message`.
pub fn todos_remaining_message() -> NextMessageFn {
    Arc::new(|ctx: LoopContext| {
        Box::pin(async move {
            let (Some(session), Some(todo)) = (&ctx.session, &ctx.providers.todo) else {
                return Ok(None);
            };
            let open: Vec<_> = todo
                .load_items(&SessionRef::from_session(session))
                .await?
                .into_iter()
                .filter(|i| !i.is_complete)
                .collect();
            if open.is_empty() {
                return Ok(None);
            }
            let lines = open
                .iter()
                .map(|i| format!("- {}", i.title))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(Some(vec![Message::user(format!(
                "You still have {} open todo item(s) that must be addressed before you can finish:\n{lines}\n\nContinue working through them now. Mark each todo complete as you finish it, and only stop once every todo item is complete.",
                open.len()
            ))]))
        })
    })
}

/// A `should_continue` that loops while the harness
/// [`BackgroundAgentsProvider`] has running tasks. Mirrors upstream
/// `background_tasks_running`.
pub fn background_tasks_running() -> ShouldContinueFn {
    Arc::new(|ctx: LoopContext| {
        Box::pin(async move {
            let (Some(session), Some(provider)) = (&ctx.session, &ctx.providers.background_agents)
            else {
                return Ok(false.into());
            };
            Ok((!provider
                .running_tasks(&SessionRef::from_session(session))
                .is_empty())
            .into())
        })
    })
}

/// A `next_message` listing the still-running background tasks; `None` when
/// unavailable or idle. Mirrors upstream `background_tasks_running_message`.
pub fn background_tasks_running_message() -> NextMessageFn {
    Arc::new(|ctx: LoopContext| {
        Box::pin(async move {
            let (Some(session), Some(provider)) = (&ctx.session, &ctx.providers.background_agents)
            else {
                return Ok(None);
            };
            let running = provider.running_tasks(&SessionRef::from_session(session));
            if running.is_empty() {
                return Ok(None);
            }
            let lines = running
                .iter()
                .map(|t| format!("- #{} ({}): {}", t.id, t.agent_name, t.description))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(Some(vec![Message::user(format!(
                "You still have {} background task(s) running that must finish before you can complete the work:\n{lines}\n\nWait for these tasks to complete, retrieve their results, and incorporate them. Only stop once every background task has finished.",
                running.len()
            ))]))
        })
    })
}
