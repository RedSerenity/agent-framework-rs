//! Offline end-to-end tests for `evaluation`: `EvaluateAgent` /
//! `EvaluateWorkflow` over scripted agents and real workflows.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_framework_core::evaluation::{
    evaluate_agent, evaluate_workflow, evaluator, keyword_check, tool_call_args_match,
    tool_called_check, tool_calls_present, ConversationSplit, EvalFields, EvalItem, EvalResults,
    EvalRunStatus, EvaluateAgent, EvaluateWorkflow, Evaluator, ExpectedToolCall, LocalEvaluator,
    ResultCounts,
};
use agent_framework_core::prelude::*;
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::FunctionTool;
use agent_framework_core::types::{FunctionArguments, FunctionCallContent};
use async_trait::async_trait;
use serde_json::{json, Value};

/// An agent that answers `"<prefix>: <last user text>"`, optionally after a
/// `get_weather` call, and counts its runs.
struct EchoAgent {
    id: String,
    prefix: String,
    with_tool_call: bool,
    tools: Vec<ToolDefinition>,
    runs: AtomicUsize,
    seen: Mutex<Vec<Vec<Message>>>,
}

impl EchoAgent {
    fn new(id: &str, prefix: &str) -> Self {
        Self {
            id: id.into(),
            prefix: prefix.into(),
            with_tool_call: false,
            tools: Vec::new(),
            runs: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn with_tool_call(mut self) -> Self {
        self.with_tool_call = true;
        self.tools = vec![FunctionTool::new(
            "get_weather",
            "Get the weather",
            json!({"type": "object", "properties": {"location": {"type": "string"}}}),
            |_| async { Ok(json!("sunny")) },
        )
        .into_definition()];
        self
    }
}

#[async_trait]
impl SupportsAgentRun for EchoAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        self.seen.lock().unwrap().push(messages.clone());
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role.as_str() == "user")
            .map(Message::text)
            .unwrap_or_default();
        let mut out = Vec::new();
        if self.with_tool_call {
            let mut args = std::collections::HashMap::new();
            args.insert("location".to_string(), json!("NYC"));
            out.push(Message::with_contents(
                "assistant",
                vec![Content::FunctionCall(FunctionCallContent::new(
                    "c1",
                    "get_weather",
                    Some(FunctionArguments::Object(args)),
                ))],
            ));
        }
        out.push(
            Message::assistant(format!("{}: {last_user}", self.prefix))
                .with_author(self.id.clone()),
        );
        Ok(AgentResponse {
            messages: out,
            ..AgentResponse::default()
        })
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        Some(&self.id)
    }

    fn default_tools(&self) -> Vec<ToolDefinition> {
        self.tools.clone()
    }
}

/// An evaluator that records what it was asked to evaluate and passes
/// everything.
#[derive(Default)]
struct RecordingEvaluator {
    name: String,
    calls: Mutex<Vec<(String, Vec<EvalItem>)>>,
}

impl RecordingEvaluator {
    fn named(name: &str) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            ..Self::default()
        })
    }

    fn calls(&self) -> Vec<(String, Vec<EvalItem>)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl Evaluator for RecordingEvaluator {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, items: &[EvalItem], eval_name: &str) -> Result<EvalResults> {
        self.calls
            .lock()
            .unwrap()
            .push((eval_name.to_string(), items.to_vec()));
        Ok(EvalResults::new(&self.name)
            .with_run_id(eval_name)
            .with_result_counts(ResultCounts::new(items.len() as u64, 0, 0)))
    }
}

fn response(text: &str) -> AgentResponse {
    AgentResponse {
        messages: vec![Message::assistant(text)],
        ..AgentResponse::default()
    }
}

// ---------------------------------------------------------------------------
// evaluate_agent
// ---------------------------------------------------------------------------

#[tokio::test]
async fn evaluate_agent_runs_each_query_and_scores_locally() {
    let agent = EchoAgent::new("weather-bot", "Weather");
    let results = EvaluateAgent::new()
        .agent(&agent)
        .queries(["sunny in Seattle?", "rain in Paris?"])
        .evaluator(LocalEvaluator::new().with_check(keyword_check(["weather"])))
        .run()
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert_eq!(r.provider, "Local");
    assert_eq!(r.run_id, "Eval: weather-bot");
    assert_eq!((r.passed(), r.failed()), (2, 0));
    assert!(r.all_passed());
    r.raise_for_status(None).unwrap();
    assert_eq!(r.items[0].input_text.as_deref(), Some("sunny in Seattle?"));
    assert_eq!(
        r.items[1].output_text.as_deref(),
        Some("Weather: rain in Paris?")
    );
    // Each query went to the agent as a single user message.
    let seen = agent.seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].len(), 1);
    assert_eq!(seen[0][0].text(), "sunny in Seattle?");
}

#[tokio::test]
async fn evaluate_agent_repetitions_stamp_expectations_and_tools() {
    let agent = EchoAgent::new("bot", "Answer").with_tool_call();
    let recorder = RecordingEvaluator::named("rec");
    let results = EvaluateAgent::new()
        .agent(&agent)
        .queries(["q1", "q2"])
        .expected_outputs(["e1", "e2"])
        .expected_tool_calls([
            vec![ExpectedToolCall::new("get_weather")],
            vec![ExpectedToolCall::new("other")],
        ])
        .context("doc")
        .conversation_split(ConversationSplit::Full)
        .num_repetitions(3)
        .evaluator_arc(recorder.clone())
        .run()
        .await
        .unwrap();
    assert_eq!(agent.runs.load(Ordering::SeqCst), 6);
    assert_eq!(results[0].passed(), 6);
    let calls = recorder.calls();
    assert_eq!(calls.len(), 1);
    let items = &calls[0].1;
    assert_eq!(items.len(), 6);
    // Repetitions are query-major within each repetition: q1, q2, q1, q2, ...
    let queries: Vec<String> = items.iter().map(EvalItem::query).collect();
    assert_eq!(queries, ["q1", "q2", "q1", "q2", "q1", "q2"]);
    let expected: Vec<&str> = items
        .iter()
        .map(|i| i.expected_output.as_deref().unwrap())
        .collect();
    assert_eq!(expected, ["e1", "e2", "e1", "e2", "e1", "e2"]);
    assert_eq!(
        items[3].expected_tool_calls.as_ref().unwrap()[0].name,
        "other"
    );
    assert_eq!(items[0].context.as_deref(), Some("doc"));
    assert!(items[0].split_strategy.is_some());
    // Tools come from the agent's default tools.
    assert_eq!(items[0].tools.as_ref().unwrap()[0].name, "get_weather");
    // The tool call is part of the conversation.
    assert_eq!(items[0].conversation.len(), 3);
}

#[tokio::test]
async fn evaluate_agent_with_expected_tool_call_checks() {
    let agent = EchoAgent::new("bot", "Done").with_tool_call();
    let results = evaluate_agent(
        EvaluateAgent::new()
            .agent(&agent)
            .query("What's the weather in NYC?")
            .expected_tool_calls_for_query([
                ExpectedToolCall::new("get_weather").with_arguments([("location", json!("NYC"))])
            ])
            .check(tool_calls_present)
            .check(tool_call_args_match)
            .check(tool_called_check(["get_weather"])),
    )
    .await
    .unwrap();
    // Consecutive bare checks collapse into a single LocalEvaluator.
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(r.all_passed(), "{:?}", r.error);
    assert_eq!(r.per_evaluator.len(), 3);
    assert_eq!(r.items[0].scores.len(), 3);
}

#[tokio::test]
async fn evaluate_agent_pre_existing_responses() {
    let agent = EchoAgent::new("bot", "unused").with_tool_call();
    let results = EvaluateAgent::new()
        .agent(&agent)
        .queries(["What's 2+2?", "Capital of France?"])
        .responses([response("4"), response("Paris")])
        .expected_outputs(["4", "Paris"])
        .num_repetitions(5) // ignored for pre-existing responses
        .check(evaluator("exact", |f: EvalFields| {
            f.response == f.expected_output
        }))
        .run()
        .await
        .unwrap();
    assert_eq!(agent.runs.load(Ordering::SeqCst), 0, "agent must not run");
    assert_eq!((results[0].passed(), results[0].failed()), (2, 0));

    // Without an agent the default eval name is generic, and no tools attach.
    let recorder = RecordingEvaluator::named("rec");
    EvaluateAgent::new()
        .query("q")
        .response(response("a"))
        .evaluator_arc(recorder.clone())
        .run()
        .await
        .unwrap();
    let calls = recorder.calls();
    assert_eq!(calls[0].0, "Eval: agent");
    assert!(calls[0].1[0].tools.is_none());
}

#[tokio::test]
async fn evaluate_agent_validation_errors() {
    let agent = EchoAgent::new("bot", "x");
    let msg = |r: Result<Vec<EvalResults>>| match r {
        Err(Error::Configuration(m)) => m,
        other => panic!("expected configuration error, got {other:?}"),
    };

    let m = msg(EvaluateAgent::new()
        .agent(&agent)
        .query("q")
        .num_repetitions(0)
        .run()
        .await);
    assert_eq!(m, "num_repetitions must be >= 1, got 0.");

    let m = msg(EvaluateAgent::new()
        .agent(&agent)
        .queries(["a", "b"])
        .expected_output("x")
        .run()
        .await);
    assert_eq!(m, "Got 2 queries but 1 expected_output values.");

    let m = msg(EvaluateAgent::new()
        .agent(&agent)
        .query("a")
        .expected_tool_calls(Vec::<Vec<ExpectedToolCall>>::new())
        .run()
        .await);
    assert_eq!(m, "Got 1 queries but 0 expected_tool_calls lists.");

    let m = msg(EvaluateAgent::new()
        .queries(["a", "b"])
        .response(response("x"))
        .run()
        .await);
    assert_eq!(m, "Got 2 queries but 1 responses.");

    let m = msg(EvaluateAgent::new().response(response("x")).run().await);
    assert!(m.starts_with("Provide 'queries' alongside 'responses'"));

    let m = msg(EvaluateAgent::new().query("q").run().await);
    assert!(m.starts_with("Provide 'agent' when using 'queries'"));

    let m = msg(EvaluateAgent::new().agent(&agent).run().await);
    assert_eq!(
        m,
        "Provide either 'queries' (with 'agent') or 'responses' (or both)."
    );
}

#[tokio::test]
async fn multiple_providers_get_suffixed_names_and_run_in_order() {
    let agent = EchoAgent::new("bot", "Answer");
    let first = RecordingEvaluator::named("First");
    let second = RecordingEvaluator::named("Second");
    let results = EvaluateAgent::new()
        .agent(&agent)
        .query("q")
        .eval_name("Smoke")
        .evaluator_arc(first.clone())
        .check(keyword_check(["answer"]))
        .evaluator_arc(second.clone())
        .run()
        .await
        .unwrap();
    let providers: Vec<&str> = results.iter().map(|r| r.provider.as_str()).collect();
    assert_eq!(providers, ["First", "Local", "Second"]);
    assert_eq!(first.calls()[0].0, "Smoke (First)");
    assert_eq!(second.calls()[0].0, "Smoke (Second)");
    assert_eq!(results[1].run_id, "Smoke (Local)");

    // A single provider gets no suffix.
    let only = RecordingEvaluator::named("Only");
    EvaluateAgent::new()
        .agent(&agent)
        .query("q")
        .eval_name("Smoke")
        .evaluator_arc(only.clone())
        .run()
        .await
        .unwrap();
    assert_eq!(only.calls()[0].0, "Smoke");
}

#[tokio::test]
async fn failing_results_gate_ci() {
    let agent = EchoAgent::new("bot", "Answer");
    let results = EvaluateAgent::new()
        .agent(&agent)
        .query("q")
        .check(keyword_check(["missing"]))
        .run()
        .await
        .unwrap();
    let err = results[0].raise_for_status(None).unwrap_err();
    assert!(matches!(err, Error::EvalNotPassed(_)));
    assert!(err
        .to_string()
        .contains("Error: keyword_check: Missing keywords: ['missing']"));
    assert!(results[0].assert_no_failed_items(None).is_err());
    assert!(results[0]
        .assert_score_at_least(1.0, Some("keyword_check"), None)
        .is_err());
}

// ---------------------------------------------------------------------------
// evaluate_workflow
// ---------------------------------------------------------------------------

fn sequential() -> (Workflow, Arc<EchoAgent>, Arc<EchoAgent>) {
    let writer = Arc::new(EchoAgent::new("writer", "Draft"));
    let reviewer = Arc::new(EchoAgent::new("reviewer", "Review"));
    let workflow = SequentialBuilder::new()
        .participants([
            writer.clone() as Arc<dyn SupportsAgentRun>,
            reviewer.clone() as Arc<dyn SupportsAgentRun>,
        ])
        .build()
        .unwrap();
    (workflow, writer, reviewer)
}

#[tokio::test]
async fn evaluate_workflow_run_mode_breaks_down_per_agent() {
    let (workflow, writer, reviewer) = sequential();
    let recorder = RecordingEvaluator::named("rec");
    let results = EvaluateWorkflow::new(&workflow)
        .queries(["Plan a trip", "Book a hotel"])
        .expected_outputs(["plan", "hotel"])
        .num_repetitions(2)
        .evaluator_arc(recorder.clone())
        .run()
        .await
        .unwrap();
    assert_eq!(writer.runs.load(Ordering::SeqCst), 4);
    assert_eq!(reviewer.runs.load(Ordering::SeqCst), 4);

    assert_eq!(results.len(), 1);
    let r = &results[0];
    let keys: Vec<&str> = r.sub_results.keys().map(String::as_str).collect();
    assert_eq!(keys, ["reviewer", "writer"]);
    assert_eq!(r.sub_results["writer"].passed(), 4);
    assert!(r.all_passed());

    let calls = recorder.calls();
    let names: Vec<&str> = calls.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "Workflow Eval: Workflow — writer",
            "Workflow Eval: Workflow — reviewer",
            "Workflow Eval: Workflow — overall"
        ]
    );
    // Per-agent items: the run's user query, then that agent's reply.
    let writer_items = &calls[0].1;
    assert_eq!(writer_items[0].query(), "Plan a trip");
    assert_eq!(writer_items[0].response(), "Draft: Plan a trip");
    let reviewer_items = &calls[1].1;
    assert_eq!(reviewer_items[1].query(), "Book a hotel");
    // Overall items: the final output's assistant text, with the expected
    // output of their query.
    let overall = &calls[2].1;
    assert_eq!(overall.len(), 4);
    assert_eq!(overall[0].query(), "Plan a trip");
    assert_eq!(
        overall[0].response(),
        "Draft: Plan a trip Review: Plan a trip"
    );
    assert_eq!(overall[1].expected_output.as_deref(), Some("hotel"));
    assert_eq!(overall[2].expected_output.as_deref(), Some("plan"));
}

#[tokio::test]
async fn evaluate_workflow_post_hoc_and_aggregate() {
    let (workflow, _, _) = sequential();
    let run = workflow.run("Plan a trip").await.unwrap();

    // Overall on: the overall evaluation is the parent result.
    let results = EvaluateWorkflow::new(&workflow)
        .workflow_result(&run, "Plan a trip")
        .check(keyword_check(["plan"]))
        .run()
        .await
        .unwrap();
    let r = &results[0];
    assert_eq!(r.provider, "Local");
    assert_eq!(r.run_id, "Workflow Eval: Workflow — overall");
    assert_eq!(r.sub_results.len(), 2);
    assert!(r.all_passed());

    // Overall off: the parent aggregates the sub-results' counts.
    let results = EvaluateWorkflow::new(&workflow)
        .workflow_result(&run, "Plan a trip")
        .include_overall(false)
        .eval_name("Trip")
        .check(keyword_check(["draft"]))
        .run()
        .await
        .unwrap();
    let r = &results[0];
    assert_eq!(
        (r.eval_id.as_str(), r.run_id.as_str()),
        ("aggregate", "aggregate")
    );
    assert_eq!(r.status, EvalRunStatus::Completed);
    assert_eq!((r.passed(), r.failed()), (1, 1));
    assert_eq!(r.sub_results["writer"].run_id, "Trip — writer");
    let err = r.raise_for_status(None).unwrap_err().to_string();
    assert!(err.contains("Failed: reviewer."), "{err}");
}

#[tokio::test]
async fn evaluate_workflow_post_hoc_with_message_input_and_split() {
    let (workflow, _, _) = sequential();
    let input =
        serde_json::to_value(vec![Message::system("be brief"), Message::user("Go")]).unwrap();
    let run = workflow.run(input.clone()).await.unwrap();
    let recorder = RecordingEvaluator::named("rec");
    EvaluateWorkflow::new(&workflow)
        .workflow_result(&run, input)
        .conversation_split(ConversationSplit::Full)
        .evaluator_arc(recorder.clone())
        .run()
        .await
        .unwrap();
    let calls = recorder.calls();
    // Agent query = the input's user messages only.
    assert_eq!(calls[0].1[0].conversation[0].text(), "Go");
    assert_eq!(calls[0].1[0].conversation.len(), 2);
    // Overall query = the user text of the input.
    assert_eq!(calls[2].1[0].query(), "Go");
    assert!(calls
        .iter()
        .all(|(_, items)| items.iter().all(|i| i.split_strategy.is_some())));
}

#[tokio::test]
async fn evaluate_workflow_per_agent_only_and_validation() {
    let (workflow, _, _) = sequential();
    let recorder = RecordingEvaluator::named("rec");
    let results = EvaluateWorkflow::new(&workflow)
        .query("Go")
        .include_per_agent(false)
        .evaluator_arc(recorder.clone())
        .run()
        .await
        .unwrap();
    assert!(results[0].sub_results.is_empty());
    assert_eq!(recorder.calls().len(), 1);

    let msg = |r: Result<Vec<EvalResults>>| match r {
        Err(Error::Configuration(m)) => m,
        other => panic!("expected configuration error, got {other:?}"),
    };
    assert_eq!(
        msg(EvaluateWorkflow::new(&workflow).run().await),
        "Provide either 'workflow_result' or 'queries'."
    );
    let run = workflow.run("Go").await.unwrap();
    assert!(msg(EvaluateWorkflow::new(&workflow)
        .workflow_result(&run, "Go")
        .expected_output("x")
        .run()
        .await)
    .starts_with("Provide 'queries' when using 'expected_output'"));
    assert_eq!(
        msg(EvaluateWorkflow::new(&workflow)
            .queries(["a", "b"])
            .expected_output("x")
            .run()
            .await),
        "Got 2 queries but 1 expected_output values."
    );
    assert_eq!(
        msg(EvaluateWorkflow::new(&workflow)
            .query("a")
            .num_repetitions(0)
            .run()
            .await),
        "num_repetitions must be >= 1, got 0."
    );
    assert!(msg(EvaluateWorkflow::new(&workflow)
        .query("a")
        .include_overall(false)
        .include_per_agent(false)
        .check(keyword_check(["x"]))
        .run()
        .await)
    .starts_with("No agent executor data found"));
}

#[tokio::test]
async fn evaluate_workflow_without_agents_errors() {
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(
            agent_framework_core::workflow::FunctionExecutor::new(
                "plain",
                |msg: Value, ctx: WorkflowContext| async move {
                    ctx.yield_output(msg).await?;
                    Ok(())
                },
            ),
        ))
        .set_start("plain")
        .build()
        .unwrap();
    // Overall output exists, but nothing else; per-agent only must fail.
    let err = EvaluateWorkflow::new(&workflow)
        .query("hi")
        .include_overall(false)
        .check(keyword_check(["hi"]))
        .run()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("No agent executor data found"));
    // With the overall output it evaluates the plain string output.
    let results = evaluate_workflow(
        EvaluateWorkflow::new(&workflow)
            .query("hi")
            .check(keyword_check(["hi"])),
    )
    .await
    .unwrap();
    assert!(results[0].all_passed());
    assert!(results[0].sub_results.is_empty());
}

#[tokio::test]
async fn group_chat_participants_are_broken_down_by_name() {
    let a = Arc::new(EchoAgent::new("alice", "A"));
    let b = Arc::new(EchoAgent::new("bob", "B"));
    let workflow = GroupChatBuilder::new()
        .participant("alice", a as Arc<dyn SupportsAgentRun>)
        .participant("bob", b as Arc<dyn SupportsAgentRun>)
        .round_robin()
        .max_rounds(2)
        .build()
        .unwrap();
    let results = EvaluateWorkflow::new(&workflow)
        .query("topic")
        .check(keyword_check(["topic"]))
        .run()
        .await
        .unwrap();
    let keys: Vec<&str> = results[0].sub_results.keys().map(String::as_str).collect();
    assert_eq!(keys, ["alice", "bob"]);
}
