//! Upstream sample workflows run offline with scripted agents and functions.
//!
//! Fixtures under `tests/fixtures/upstream/` are verbatim copies (MIT) from
//! the upstream microsoft/agent-framework repository:
//!   * `python/*.yaml` — `python/samples/03-workflows/declarative/<name>/workflow.yaml`
//!     (and `test_http_request.yaml` from `python/packages/declarative/tests/workflows/`)
//!   * `workflow-samples/*.yaml` — `declarative-agents/workflow-samples/`
//!   * `dotnet/*.yaml` — `dotnet/tests/Microsoft.Agents.AI.Workflows.Declarative.UnitTests/Workflows/`

mod common;

use std::sync::Arc;

use agent_framework_core::prelude::*;
use agent_framework_core::workflow::WorkflowRunState;
use agent_framework_declarative::flow::WorkflowFactory;
use common::{fixture, pending, texts, vars, ScriptedAgent};
use serde_json::{json, Value as Json};

fn build(factory: &WorkflowFactory, path: &str) -> Workflow {
    factory
        .create_workflow_from_yaml(&fixture(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
}

// ---------------------------------------------------------------- Python samples

#[tokio::test]
async fn simple_workflow_sample_greets_with_default_and_input() {
    let wf = build(&WorkflowFactory::new(), "python/simple_workflow.yaml");
    let run = wf.run(json!({})).await.unwrap();
    assert_eq!(texts(&run), vec!["Hello, World!"]);
    assert_eq!(
        vars(&run).await["Outputs"]["greeting"],
        json!("Hello, World!")
    );

    let run = wf.run(json!({"name": "Ada"})).await.unwrap();
    assert_eq!(texts(&run), vec!["Hello, Ada!"]);
    assert_eq!(run.state(), WorkflowRunState::Idle);
}

#[tokio::test]
async fn conditional_workflow_sample_routes_nested_ifs() {
    let wf = build(&WorkflowFactory::new(), "python/conditional_workflow.yaml");
    for (age, first, category) in [
        (10, "Welcome, young one!", "child"),
        (15, "Hey there!", "teenager"),
        (30, "Welcome! Here are our professional services.", "adult"),
        (70, "Welcome! Enjoy our senior member benefits.", "senior"),
    ] {
        let run = wf.run(json!({"age": age})).await.unwrap();
        let out = texts(&run);
        assert_eq!(out.len(), 2, "{age}: {out:?}");
        assert!(out[0].starts_with(first), "{age}: {out:?}");
        assert_eq!(out[1], format!("You have been categorized as: {category}"));
        assert_eq!(vars(&run).await["Outputs"]["category"], json!(category));
    }
}

#[tokio::test]
async fn human_in_loop_sample_pauses_and_resumes() {
    let wf = build(&WorkflowFactory::new(), "python/human_in_loop.yaml");
    let mut run = wf.run(json!({})).await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    assert_eq!(texts(&run), vec!["Welcome to the interactive survey!"]);
    let (id, req) = pending(&run);
    assert_eq!(req["type"], "ExternalInputRequest");
    assert_eq!(req["request_type"], "question");
    assert_eq!(req["message"], "What is your name?");
    assert_eq!(req["metadata"]["output_property"], "Local.userName");
    assert_eq!(req["metadata"]["default_value"], "Demo User");

    run.send_response(id, json!("Ada")).await.unwrap();
    assert!(texts(&run).contains(&"Nice to meet you, Ada!".to_string()));
    let (id, req) = pending(&run);
    assert!(req["message"]
        .as_str()
        .unwrap()
        .starts_with("How are you feeling"));

    run.send_response(id, json!({"user_input": "good"}))
        .await
        .unwrap();
    assert!(texts(&run).contains(&"That's wonderful to hear! Let's continue.".to_string()));
    let (id, req) = pending(&run);
    assert_eq!(req["request_type"], "external");
    assert_eq!(req["message"], "Do you have any feedback for us?");
    assert_eq!(req["metadata"]["output_property"], "Local.feedback");

    // A non-null `value` wins over `user_input`.
    run.send_response(id, json!({"user_input": "ignored", "value": "Loved it"}))
        .await
        .unwrap();
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(
        texts(&run).last().unwrap(),
        "Thank you, Ada! Your feedback: Loved it"
    );
    assert_eq!(
        vars(&run).await["Outputs"]["survey"],
        json!({"name": "Ada", "feeling": "good", "feedback": "Loved it"})
    );
}

#[tokio::test]
async fn human_in_loop_else_branch() {
    let wf = build(&WorkflowFactory::new(), "python/human_in_loop.yaml");
    let mut run = wf.run(json!({})).await.unwrap();
    let (id, _) = pending(&run);
    run.send_response(id, json!("Bo")).await.unwrap();
    let (id, _) = pending(&run);
    run.send_response(id, json!("okay")).await.unwrap();
    assert!(texts(&run)
        .iter()
        .any(|t| t.starts_with("I hope things get better!")));
}

fn weather_factory() -> WorkflowFactory {
    WorkflowFactory::new()
        .with_function("get_weather", |args: Json| async move {
            Ok(json!({"location": args["location"], "temp": 72, "unit": args["unit"]}))
        })
        .with_function("format_message", |args: Json| async move {
            let d = &args["data"];
            let text = args["template"]
                .as_str()
                .unwrap_or_default()
                .replace("{location}", d["location"].as_str().unwrap_or_default())
                .replace("{temp}", &d["temp"].to_string())
                .replace("{unit}", d["unit"].as_str().unwrap_or_default());
            Ok(Json::String(text))
        })
}

#[tokio::test]
async fn invoke_function_tool_sample_calls_registered_functions() {
    let wf = build(&weather_factory(), "python/invoke_function_tool.yaml");
    let run = wf
        .run(json!({"location": "Paris", "unit": "C"}))
        .await
        .unwrap();
    let out = texts(&run);
    assert_eq!(out[0], r#"{"location":"Paris","temp":72,"unit":"C"}"#);
    assert_eq!(
        &out[1..],
        [
            "The weather in Paris is 72°C",
            "The weather in Paris is 72°C"
        ]
    );
    let v = vars(&run).await;
    assert_eq!(v["Outputs"]["weather"]["temp"], json!(72));
    let items = v["Local"]["weatherToolCallItems"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["role"], "assistant");
    assert_eq!(items[0]["contents"][0]["type"], "function_call");
    assert_eq!(items[0]["contents"][0]["name"], "get_weather");
    assert_eq!(items[1]["role"], "tool");
    assert_eq!(items[1]["contents"][0]["type"], "function_result");

    // Defaults from the If(IsBlank(...)) expressions.
    let run = wf.run(json!({})).await.unwrap();
    assert_eq!(texts(&run)[1], "The weather in Seattle is 72°F");
}

#[tokio::test]
async fn student_teacher_sample_loops_with_goto_until_congratulated() {
    let student = ScriptedAgent::new("StudentAgent", &["Is it 5?", "It is 4."]);
    let teacher = ScriptedAgent::new("TeacherAgent", &["Not quite.", "CONGRATULATIONS, correct!"]);
    let factory = WorkflowFactory::new()
        .with_agent("StudentAgent", student.clone())
        .with_agent("TeacherAgent", teacher.clone());
    let wf = build(&factory, "python/student_teacher.yaml");
    let run = wf.run("What is 2+2?").await.unwrap();
    let out = texts(&run);
    assert_eq!(out[0], "Starting math coaching session for: What is 2+2?");
    assert_eq!(
        out.iter()
            .filter(|t| t.as_str() == "\n[Student]:\n")
            .count(),
        2,
        "{out:?}"
    );
    assert!(out.last().unwrap().contains("GOLD STAR!"));
    let v = vars(&run).await;
    assert_eq!(v["Local"]["TurnCount"], json!(2));
    // `workflow.outputs.result` (lower-case) is a custom namespace upstream too.
    assert_eq!(
        v["Custom"]["workflow"]["outputs"]["result"],
        json!("success")
    );
    // The student starts from the user's question; the teacher sees the
    // student's reply; both share System.ConversationId's history.
    assert_eq!(student.last_inputs()[0], "What is 2+2?");
    assert_eq!(teacher.last_inputs()[0], "Is it 5?");
    let second_teacher_call = &teacher.calls.lock().unwrap()[1];
    assert!(second_teacher_call.len() >= 6, "history accumulates");
}

#[tokio::test]
async fn student_teacher_sample_times_out_after_four_turns() {
    let factory = WorkflowFactory::new()
        .with_agent(
            "StudentAgent",
            ScriptedAgent::new("StudentAgent", &["Is it 5?"]),
        )
        .with_agent("TeacherAgent", ScriptedAgent::new("TeacherAgent", &["No."]));
    let wf = build(&factory, "python/student_teacher.yaml");
    let run = wf.run("What is 2+2?").await.unwrap();
    let v = vars(&run).await;
    assert_eq!(v["Local"]["TurnCount"], json!(4));
    assert_eq!(
        v["Custom"]["workflow"]["outputs"]["result"],
        json!("timeout")
    );
    assert!(texts(&run).last().unwrap().contains("reached its limit"));
}

#[tokio::test]
async fn marketing_samples_chain_three_agents() {
    for path in ["python/marketing.yaml", "workflow-samples/Marketing.yaml"] {
        let analyst = ScriptedAgent::new("AnalystAgent", &["Analysis"]);
        let writer = ScriptedAgent::new("WriterAgent", &["Copy"]);
        let editor = ScriptedAgent::new("EditorAgent", &["Polished"]);
        let factory = WorkflowFactory::new()
            .with_agent("AnalystAgent", analyst.clone())
            .with_agent("WriterAgent", writer.clone())
            .with_agent("EditorAgent", editor.clone());
        let run = build(&factory, path).run("An eco bottle").await.unwrap();
        assert_eq!(texts(&run), vec!["Analysis", "Copy", "Polished"], "{path}");
        assert_eq!(analyst.last_inputs(), vec!["An eco bottle"]);
        assert_eq!(writer.last_inputs(), vec!["Analysis"]);
        assert_eq!(editor.last_inputs(), vec!["Copy"]);
        // Shared conversation: the editor sees the whole exchange.
        assert_eq!(editor.calls.lock().unwrap()[0].len(), 5);
    }
}

#[tokio::test]
async fn customer_support_sample_runs_with_external_loop() {
    for path in [
        "python/customer_support.yaml",
        "workflow-samples/CustomerSupport.yaml",
    ] {
        let service = ScriptedAgent::new(
            "SelfServiceAgent",
            &[
                r#"{"IsResolved": false, "NeedsTicket": false, "IssueDescription": "PC won't boot", "AttemptedResolutionSteps": "none"}"#,
                "Sure, let me check.\n```json\n{\"IsResolved\": false, \"NeedsTicket\": true, \"IssueDescription\": \"PC won't boot\", \"AttemptedResolutionSteps\": \"restart\"}\n```",
            ],
        );
        let ticketing = ScriptedAgent::new(
            "TicketingAgent",
            &[r#"{"TicketId": "TKT-42", "TicketSummary": "boot"}"#],
        );
        let routing = ScriptedAgent::new(
            "TicketRoutingAgent",
            &[r#"{"TeamName": "Windows Support"}"#],
        );
        let support = ScriptedAgent::new(
            "WindowsSupportAgent",
            &[
                r#"{"IsResolved": true, "NeedsEscalation": false, "ResolutionSummary": "Reinstalled drivers"}"#,
            ],
        );
        let resolution = ScriptedAgent::new("TicketResolutionAgent", &["Ticket closed"]);
        let escalation = ScriptedAgent::new("TicketEscalationAgent", &["{}"]);
        let factory = WorkflowFactory::new()
            .with_agent("SelfServiceAgent", service.clone())
            .with_agent("TicketingAgent", ticketing.clone())
            .with_agent("TicketRoutingAgent", routing.clone())
            .with_agent("WindowsSupportAgent", support.clone())
            .with_agent("TicketResolutionAgent", resolution.clone())
            .with_agent("TicketEscalationAgent", escalation.clone());
        let wf = build(&factory, path);
        let mut run = wf.run("My PC won't boot").await.unwrap();
        let (id, req) = pending(&run);
        assert_eq!(req["type"], "AgentExternalInputRequest");
        assert_eq!(req["agent_name"], "SelfServiceAgent");
        assert_eq!(req["iteration"], 0);

        run.send_response(id, json!("I tried restarting"))
            .await
            .unwrap();
        assert_eq!(run.state(), WorkflowRunState::Idle, "{path}");
        assert_eq!(service.call_count(), 2);
        assert_eq!(service.last_inputs()[1], "I tried restarting");
        let out = texts(&run);
        assert!(
            out.contains(&"Created ticket #TKT-42".to_string()),
            "{out:?}"
        );
        assert!(
            out.contains(&"Routing to Windows Support".to_string()),
            "{out:?}"
        );
        assert_eq!(
            ticketing.last_inputs()[0],
            "IssueDescription: PC won't boot\nAttemptedResolutionSteps: restart"
        );
        assert_eq!(routing.last_inputs()[0], "PC won't boot");
        // The support agent runs in its own conversation.
        assert_eq!(support.calls.lock().unwrap()[0].len(), 1);
        assert_eq!(
            resolution.last_inputs()[0],
            "TicketId: TKT-42\nResolutionSummary: Reinstalled drivers"
        );
        assert_eq!(escalation.call_count(), 0);
        let v = vars(&run).await;
        assert_eq!(v["Local"]["ServiceParameters"]["NeedsTicket"], json!(true));
        assert_eq!(v["Local"]["ResolutionSteps"], json!("Reinstalled drivers"));
    }
}

#[tokio::test]
async fn agent_to_function_tool_sample_parses_response_object() {
    let agent = ScriptedAgent::new(
        "OrderAnalysisAgent",
        &["Here you go:\n```json\n{\"items\": [{\"name\": \"tea\", \"qty\": 2, \"price\": 1.5}]}\n```"],
    );
    let factory = WorkflowFactory::new()
        .with_agent("OrderAnalysisAgent", agent)
        .with_function("calculate_order_total", |args: Json| async move {
            let total: f64 = args["order_data"]["items"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| i["qty"].as_f64().unwrap() * i["price"].as_f64().unwrap())
                .sum();
            Ok(json!({"total": total}))
        })
        .with_function("format_order_confirmation", |args: Json| async move {
            Ok(json!(format!(
                "Order confirmed: {} item(s), total {}",
                args["order_data"]["items"].as_array().unwrap().len(),
                args["order_calculation"]["total"]
            )))
        });
    let run = build(&factory, "python/agent_to_function_tool.yaml")
        .run("2 teas please")
        .await
        .unwrap();
    // autoSend: false everywhere except the final SendActivity.
    assert_eq!(texts(&run), vec!["Order confirmed: 1 item(s), total 3"]);
    let v = vars(&run).await;
    assert_eq!(v["Local"]["orderData"]["items"][0]["name"], json!("tea"));
    assert_eq!(v["Local"]["orderCalculation"], json!({"total": 3.0}));
}

#[tokio::test]
async fn function_tools_sample_external_loop_until_exit() {
    let menu = ScriptedAgent::new("MenuAgent", &["Soup of the day", "$5"]);
    let factory = WorkflowFactory::new().with_agent("MenuAgent", menu.clone());
    let wf = build(&factory, "python/function_tools.yaml");
    let mut run = wf.run("Hi").await.unwrap();
    let (id, req) = pending(&run);
    assert_eq!(req["agent_response"], "Soup of the day");
    run.send_response(id, json!("How much is it?"))
        .await
        .unwrap();
    let (id, req) = pending(&run);
    assert_eq!(req["agent_response"], "$5");
    assert_eq!(req["iteration"], 1);
    run.send_response(id, json!("exit")).await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(menu.call_count(), 2);
    assert_eq!(texts(&run), vec!["Soup of the day", "$5"]);
}

// ---------------------------------------------------- workflow-samples

#[tokio::test]
async fn deep_research_sample_runs_to_completion() {
    let ledger = |done: bool| {
        json!({
            "is_request_satisfied": {"answer": done, "reason": if done { "all done" } else { "not yet" }},
            "is_in_loop": {"answer": false, "reason": ""},
            "is_progress_being_made": {"answer": true, "reason": ""},
            "next_speaker": {"answer": "WeatherAgent", "reason": "needs weather"},
            "instruction_or_question": {"answer": "Get the weather", "reason": ""}
        })
        .to_string()
    };
    let (l1, l2) = (ledger(false), ledger(true));
    let research = ScriptedAgent::new("ResearchAgent", &["FACTS"]);
    let planner = ScriptedAgent::new("PlannerAgent", &["PLAN"]);
    let manager = ScriptedAgent::new("ManagerAgent", &[&l1, &l2]);
    let weather = ScriptedAgent::new("WeatherAgent", &["SUNNY"]);
    let summary = ScriptedAgent::new("SummaryAgent", &["FINAL"]);
    let factory = WorkflowFactory::new()
        .with_agent("ResearchAgent", research.clone())
        .with_agent("PlannerAgent", planner.clone())
        .with_agent("ManagerAgent", manager.clone())
        .with_agent("WeatherAgent", weather.clone())
        .with_agent("SummaryAgent", summary.clone());
    let run = build(&factory, "workflow-samples/DeepResearch.yaml")
        .run("What's the weather in Oslo?")
        .await
        .unwrap();
    assert_eq!(
        texts(&run),
        vec![
            "Analyzing facts...",
            "FACTS",
            "Creating a plan...",
            "PLAN",
            "(needs weather)\n\nWeatherAgent - Get the weather",
            "SUNNY",
            "Completed! all done",
            "FINAL",
        ]
    );
    let v = vars(&run).await;
    assert_eq!(
        v["Local"]["TeamDescription"],
        json!("- WeatherAgent: Able to retrieve weather information\n- CoderAgent: Able to write and execute Python code\n- KnowledgeAgent: Able to perform generic websearches")
    );
    let instructions = v["Local"]["TaskInstructions"].as_str().unwrap();
    assert!(instructions.contains("What's the weather in Oslo?"));
    assert!(instructions.contains("Consider this initial fact sheet:\n\nFACTS"));
    assert!(instructions.contains("Here is the plan to follow as best as possible:\n\nPLAN"));
    assert_eq!(v["Local"]["AgentResponseText"], json!("SUNNY"));
    assert_eq!(weather.last_inputs(), vec!["Get the weather"]);
    assert!(planner.last_inputs()[0].starts_with("team: - WeatherAgent"));
    assert_eq!(manager.call_count(), 2);
}

#[tokio::test]
async fn math_chat_sample_stops_on_congratulations() {
    let factory = WorkflowFactory::new()
        .with_agent(
            "StudentAgent",
            ScriptedAgent::new("StudentAgent", &["guess"]),
        )
        .with_agent(
            "TeacherAgent",
            ScriptedAgent::new("TeacherAgent", &["no", "no", "Congratulations!"]),
        );
    let run = build(&factory, "workflow-samples/MathChat.yaml")
        .run("2+2?")
        .await
        .unwrap();
    assert_eq!(texts(&run).last().unwrap(), "GOLD STAR!");
    assert_eq!(vars(&run).await["Local"]["TurnCount"], json!(3));
}

#[tokio::test]
async fn mcp_docs_research_sample_threads_message_text() {
    let planner = ScriptedAgent::new("QueryPlannerAgent", &["query: rust traits"]);
    let learn = ScriptedAgent::new("MicrosoftLearnAgent", &["draft answer"]);
    let reviewer = ScriptedAgent::new("CitationReviewerAgent", &["reviewed answer"]);
    let factory = WorkflowFactory::new()
        .with_agent("QueryPlannerAgent", planner.clone())
        .with_agent("MicrosoftLearnAgent", learn.clone())
        .with_agent("CitationReviewerAgent", reviewer.clone());
    let run = build(&factory, "workflow-samples/McpDocsResearch.yaml")
        .run("How do traits work?")
        .await
        .unwrap();
    assert_eq!(planner.last_inputs(), vec!["How do traits work?"]);
    assert_eq!(learn.last_inputs(), vec!["query: rust traits"]);
    assert_eq!(reviewer.last_inputs(), vec!["draft answer"]);
    assert_eq!(texts(&run).last().unwrap(), "reviewed answer");
}

#[tokio::test]
async fn every_workflow_sample_builds() {
    let names = [
        "QueryPlannerAgent",
        "MicrosoftLearnAgent",
        "CitationReviewerAgent",
    ];
    let mut factory = WorkflowFactory::new();
    for n in names {
        factory.register_agent(n, ScriptedAgent::new(n, &["x"]));
    }
    for f in std::fs::read_dir(format!(
        "{}/tests/fixtures/upstream/workflow-samples",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
    {
        let path = f.unwrap().path();
        let yaml = std::fs::read_to_string(&path).unwrap();
        if let Err(e) = factory.create_workflow_from_yaml(&yaml) {
            panic!("{}: {e}", path.display());
        }
    }
}

// ------------------------------------------------------- .NET workflows

async fn run_dotnet(path: &str, input: Json) -> WorkflowRun {
    build(&WorkflowFactory::new(), &format!("dotnet/{path}.yaml"))
        .run(input)
        .await
        .unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[tokio::test]
async fn dotnet_goto_and_end_actions_skip_dead_code() {
    for path in ["Goto", "EndWorkflow", "EndConversation"] {
        let run = run_dotnet(path, json!("hi")).await;
        assert!(texts(&run).is_empty(), "{path}: {:?}", texts(&run));
        assert_eq!(run.state(), WorkflowRunState::Idle);
    }
}

#[tokio::test]
async fn dotnet_foreach_shapes() {
    let run = run_dotnet("LoopEach", json!("")).await;
    assert_eq!(
        texts(&run),
        vec!["x1 - 0:a", "x2 - 1:b", "x3 - 2:c", "x4 - 3:d", "x5 - 4:e", "x6 - 5:f"]
    );
    for path in ["LoopBreak", "LoopContinue"] {
        let run = run_dotnet(path, json!("")).await;
        assert!(texts(&run).is_empty(), "{path}");
        assert!(vars(&run).await["Local"].get("Count").is_none(), "{path}");
    }
}

#[tokio::test]
async fn dotnet_condition_groups() {
    let run = run_dotnet("Condition", json!("3")).await;
    assert_eq!(texts(&run), vec!["ODD", "All done!"]);
    let run = run_dotnet("Condition", json!("4")).await;
    assert_eq!(texts(&run), vec!["EVEN", "All done!"]);
    let run = run_dotnet("ConditionElse", json!("8")).await;
    assert_eq!(texts(&run), vec!["EVEN", "All done!"]);
    let run = run_dotnet("ConditionFallThrough", json!("8")).await;
    assert_eq!(texts(&run), vec!["All done!"]);
    let run = run_dotnet("ConditionFallThrough", json!("7")).await;
    assert_eq!(texts(&run), vec!["ODD", "All done!"]);
}

#[tokio::test]
async fn dotnet_variable_actions() {
    let local = |run: &WorkflowRun| {
        let run_state = run.shared_state();
        async move {
            agent_framework_declarative::flow::final_state(&run_state)
                .await
                .unwrap()["Local"]
                .clone()
        }
    };
    assert_eq!(
        local(&run_dotnet("SetVariable", json!("")).await).await["TestVar"],
        json!(3)
    );
    assert_eq!(
        local(&run_dotnet("SetTextVariable", json!("")).await).await["TestVar"],
        json!("Test content")
    );
    assert_eq!(
        local(&run_dotnet("ResetVariable", json!("")).await).await["MyVar"],
        json!(null)
    );
    assert_eq!(
        local(&run_dotnet("ParseValue", json!("")).await).await["MyVar"],
        json!(42)
    );
    assert_eq!(
        local(&run_dotnet("ParseValueList", json!("")).await).await["MyVar"],
        json!(["apple", "banana", "cat"])
    );
    assert_eq!(
        local(&run_dotnet("ClearAllVariables", json!("")).await).await,
        json!({})
    );
}

/// The .NET `EditTable.yaml` / `EditTableV2.yaml` fixtures write
/// `value: ={id: 7}` unquoted, which YAML 1.1/1.2 parsers (PyYAML included)
/// reject; these are the same workflows with the value quoted.
#[tokio::test]
async fn dotnet_edit_table_shapes() {
    let v1 = r#"
kind: Workflow
trigger:
  kind: OnConversationStart
  id: my_workflow
  actions:
    - kind: SetVariable
      id: set_var
      variable: Local.MyTable
      value: "=[{id: 3}]"
    - kind: EditTable
      id: edit_var
      itemsVariable: Local.MyTable
      changeType: Add
      value: "={id: 7}"
    - kind: EditTable
      id: take_first
      itemsVariable: Local.MyTable
      changeType: TakeFirst
      resultVariable: Local.First
"#;
    let run = WorkflowFactory::new()
        .create_workflow_from_yaml(v1)
        .unwrap()
        .run("")
        .await
        .unwrap();
    let v = vars(&run).await;
    assert_eq!(v["Local"]["MyTable"], json!([{"id": 7}]));
    assert_eq!(v["Local"]["First"], json!({"id": 3}));

    let v2 = r#"
kind: Workflow
trigger:
  kind: OnConversationStart
  id: my_workflow
  actions:
    - kind: SetVariable
      id: set_var
      variable: Local.MyTable
      value: "=[{id: 3}]"
    - kind: EditTableV2
      id: edit_var
      itemsVariable: Local.MyTable
      changeType:
        kind: AddItemOperation
        value: "={id: 7}"
    - kind: EditTableV2
      id: remove
      itemsVariable: Local.MyTable
      changeType:
        kind: RemoveItemOperation
        value: "=[{id: 3}]"
    - kind: EditTableV2
      id: take_last
      itemsVariable: Local.MyTable
      changeType:
        kind: TakeLastItemOperation
        resultVariable: Local.Last
"#;
    let run = WorkflowFactory::new()
        .create_workflow_from_yaml(v2)
        .unwrap()
        .run("")
        .await
        .unwrap();
    let v = vars(&run).await;
    assert_eq!(v["Local"]["MyTable"], json!([]));
    assert_eq!(v["Local"]["Last"], json!({"id": 7}));
}

#[tokio::test]
async fn dotnet_send_activity_and_scopes() {
    let run = run_dotnet("SendActivity", json!("hello")).await;
    assert_eq!(texts(&run), vec!["Input: \"hello\""]);
    let run = run_dotnet("MixedScopes", json!("hello")).await;
    assert_eq!(texts(&run), vec!["Input: \"\""]);
    assert_eq!(
        vars(&run).await["Custom"]["Topic"]["TestValue"],
        json!("hello")
    );
}

#[tokio::test]
async fn dotnet_unknown_actions_are_skipped_unless_strict() {
    // `CancelWorkflow` is not an upstream action kind: skipped with a warning.
    let run = run_dotnet("CancelWorkflow", json!("")).await;
    assert_eq!(texts(&run), vec!["NEVER 1!"]);
    let strict = WorkflowFactory::new().strict_actions(true);
    match strict.create_workflow_from_yaml(&fixture("dotnet/CancelWorkflow.yaml")) {
        Ok(_) => panic!("strict mode must reject unknown kinds"),
        Err(e) => assert!(e.to_string().contains("CancelWorkflow"), "{e}"),
    }
}

#[tokio::test]
async fn dotnet_conversation_actions() {
    let run = run_dotnet("CreateConversation", json!("")).await;
    let v = vars(&run).await;
    let id = v["Local"]["PrivateConversationId"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(v["System"]["conversations"][&id]["messages"], json!([]));

    let run = run_dotnet("AddConversationMessage", json!("")).await;
    let v = vars(&run).await;
    let conv = v["System"]["ConversationId"].as_str().unwrap().to_string();
    let stored = &v["System"]["conversations"][&conv]["messages"];
    assert_eq!(stored.as_array().unwrap().len(), 1);
    assert_eq!(stored[0]["role"], "user");
    assert_eq!(v["Local"]["MyMessage1"], stored[0]);
    assert!(stored[0]["message_id"].is_string());

    let run = run_dotnet("CopyConversationMessages", json!("")).await;
    let v = vars(&run).await;
    let conv = v["System"]["ConversationId"].as_str().unwrap().to_string();
    let stored = &v["System"]["conversations"][&conv]["messages"];
    assert_eq!(
        stored[0]["contents"][0]["text"],
        "Hello, how can I assist you today?"
    );

    let run = run_dotnet("RetrieveConversationMessages", json!("")).await;
    assert_eq!(vars(&run).await["Local"]["AllMessages"], json!([]));
}

#[tokio::test]
async fn dotnet_invoke_agent_with_env_name_and_message_table() {
    let student = ScriptedAgent::new("Student", &["Hi!"]);
    let factory = WorkflowFactory::new()
        .with_agent("Student", student.clone())
        .with_configuration([("MY_STUDENT", "Student")]);
    let run = build(&factory, "dotnet/InvokeAgent.yaml")
        .run("hello")
        .await
        .unwrap();
    assert_eq!(texts(&run), vec!["Hi!"]);
    assert_eq!(student.last_inputs(), vec!["hello"]);
}

#[tokio::test]
async fn dotnet_agent_failure_fails_the_run() {
    let factory =
        WorkflowFactory::new().with_agent("TestAgent", ScriptedAgent::failing("TestAgent"));
    let wf = build(&factory, "dotnet/AgentFailureFollowup.yaml");
    let err = match wf.run("hello").await {
        Ok(_) => panic!("expected failure"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("Agent 'TestAgent' invocation failed"), "{err}");
}

#[tokio::test]
async fn invoke_agent_without_registered_agent_fails() {
    let wf = build(&WorkflowFactory::new(), "dotnet/AgentFailureFollowup.yaml");
    let err = match wf.run("hello").await {
        Ok(_) => panic!("expected failure"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("not found in registry"), "{err}");
    let _ = Arc::new(());
}
