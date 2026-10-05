//! Action-level tests for upstream-format declarative workflows: control
//! flow, variables, tools, HTTP, MCP, validation, loader dispatch, and
//! checkpoint resume.

mod common;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::workflow::{
    CheckpointStorage, InMemoryCheckpointStorage, WorkflowRunState,
};
use agent_framework_declarative::flow::{
    HttpRequestError, HttpRequestHandler, HttpRequestInfo, HttpRequestResult, McpToolError,
    McpToolHandler, McpToolInvocation, McpToolResult, WorkflowFactory,
};
use agent_framework_declarative::{AgentRegistry, DeclarativeLoader, ToolRegistry};
use async_trait::async_trait;
use common::{pending, texts, vars, ScriptedAgent};
use serde_json::{json, Value as Json};

fn wf(factory: &WorkflowFactory, actions_yaml: &str) -> Workflow {
    let yaml = format!(
        "kind: Workflow\ntrigger:\n  kind: OnConversationStart\n  id: test\n  actions:\n{}",
        indent(actions_yaml, 4)
    );
    factory
        .create_workflow_from_yaml(&yaml)
        .unwrap_or_else(|e| panic!("{e}\n{yaml}"))
}

fn indent(s: &str, n: usize) -> String {
    let pad = " ".repeat(n);
    s.lines()
        .map(|l| {
            if l.trim().is_empty() {
                String::new()
            } else {
                format!("{pad}{l}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_err(factory: &WorkflowFactory, yaml: &str) -> String {
    match factory.create_workflow_from_yaml(yaml) {
        Ok(_) => panic!("expected a build error for:\n{yaml}"),
        Err(e) => e.to_string(),
    }
}

async fn run_err(workflow: &Workflow, input: Json) -> String {
    match workflow.run(input).await {
        Ok(run) => panic!("expected a run error, got outputs {:?}", texts(&run)),
        Err(e) => e.to_string(),
    }
}

// ------------------------------------------------------------ control flow

#[tokio::test]
async fn foreach_python_shape_with_index_break_and_continue() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: Foreach
  id: loop
  source: =[1, 2, 3, 4, 5]
  itemName: n
  indexName: i
  actions:
    - kind: If
      condition: =Local.n = 2
      then:
        - kind: ContinueLoop
    - kind: If
      condition: =Local.n = 4
      then:
        - kind: BreakLoop
    - kind: SendActivity
      activity: "{Local.i}:{Local.n}"
- kind: SendActivity
  activity: done
"#,
    );
    let run = w.run(json!({})).await.unwrap();
    assert_eq!(texts(&run), vec!["0:1", "2:3", "done"]);
    let v = vars(&run).await;
    assert!(v["_declarative_loop_state"].as_object().unwrap().is_empty());
}

#[tokio::test]
async fn foreach_over_empty_and_records() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: Foreach
  source: =Local.none
  actions:
    - kind: SendActivity
      activity: never
- kind: Foreach
  source: =Filter(Workflow.Inputs.people, age > 30)
  actions:
    - kind: SendActivity
      activity: =Local.item.name
"#,
    );
    let run = w
        .run(json!({"people": [{"name": "A", "age": 20}, {"name": "B", "age": 40}, {"name": "C", "age": 50}]}))
        .await
        .unwrap();
    assert_eq!(texts(&run), vec!["B", "C"]);
}

#[tokio::test]
async fn nested_foreach_loops() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: Foreach
  source: =["a", "b"]
  itemName: outer
  actions:
    - kind: Foreach
      source: =[1, 2]
      itemName: inner
      actions:
        - kind: SendActivity
          activity: "{Local.outer}{Local.inner}"
"#,
    );
    let run = w.run(json!({})).await.unwrap();
    assert_eq!(texts(&run), vec!["a1", "a2", "b1", "b2"]);
}

#[tokio::test]
async fn condition_group_is_first_match() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: ConditionGroup
  conditions:
    - condition: =Workflow.Inputs.x > 1
      actions:
        - kind: SendActivity
          activity: gt1
    - condition: =Workflow.Inputs.x > 0
      actions:
        - kind: SendActivity
          activity: gt0
    - condition: true
      actions: []
  elseActions:
    - kind: SendActivity
      activity: else
- kind: SendActivity
  activity: after
"#,
    );
    assert_eq!(
        texts(&w.run(json!({"x": 5})).await.unwrap()),
        vec!["gt1", "after"]
    );
    assert_eq!(
        texts(&w.run(json!({"x": 1})).await.unwrap()),
        vec!["gt0", "after"]
    );
    // A matching branch with no actions ends the path (upstream semantics).
    assert!(texts(&w.run(json!({"x": -1})).await.unwrap()).is_empty());
}

#[tokio::test]
async fn if_without_else_and_bare_conditions() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: If
  condition: Workflow.Inputs.flag
  then:
    - kind: SendActivity
      activity: yes
- kind: If
  condition: false
  actions:
    - kind: SendActivity
      activity: never
- kind: SendActivity
  activity: end
"#,
    );
    assert_eq!(
        texts(&w.run(json!({"flag": true})).await.unwrap()),
        vec!["yes", "end"]
    );
    assert_eq!(
        texts(&w.run(json!({"flag": false})).await.unwrap()),
        vec!["end"]
    );
}

#[tokio::test]
async fn goto_loops_and_targets_structures() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: SetVariable
  variable: Local.n
  value: 0
- kind: SetVariable
  id: inc
  variable: Local.n
  value: =Local.n + 1
- kind: If
  id: check
  condition: =Local.n < 3
  then:
    - kind: GotoAction
      actionId: inc
- kind: If
  id: second
  condition: =Local.n = 3
  then:
    - kind: SetVariable
      variable: Local.n
      value: =Local.n + 10
    - kind: GotoAction
      target: second
- kind: SendActivity
  activity: =Text(Local.n)
"#,
    );
    let run = w.run(json!({})).await.unwrap();
    assert_eq!(texts(&run), vec!["13"]);
}

#[tokio::test]
async fn max_turns_bounds_infinite_loops() {
    let yaml = r#"
kind: Workflow
maxTurns: 10
actions:
  - kind: SetVariable
    id: a
    variable: Local.n
    value: =Local.n + 1
  - kind: GotoAction
    actionId: a
"#;
    let w = WorkflowFactory::new()
        .create_workflow_from_yaml(yaml)
        .unwrap();
    assert!(run_err(&w, json!({})).await.contains("max_iterations"));
    let w = WorkflowFactory::new()
        .with_max_iterations(4)
        .create_workflow_from_yaml(yaml)
        .unwrap();
    assert!(run_err(&w, json!({})).await.contains("max_iterations (4)"));
}

#[tokio::test]
async fn end_actions_stop_the_path() {
    for kind in [
        "EndWorkflow",
        "EndDialog",
        "EndConversation",
        "CancelDialog",
        "CancelAllDialogs",
    ] {
        let w = wf(
            &WorkflowFactory::new(),
            &format!("- kind: SendActivity\n  activity: before\n- kind: {kind}\n- kind: SendActivity\n  activity: after\n"),
        );
        assert_eq!(
            texts(&w.run(json!({})).await.unwrap()),
            vec!["before"],
            "{kind}"
        );
    }
}

// ------------------------------------------------------------ variables

#[tokio::test]
async fn variable_actions_python_shapes() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: SetValue
  path: Local.a
  value: =1 + 1
- kind: SetValue
  variable: Local.ignored
  path: Local.b
  value:
    nested: =Local.a * 10
    list: [=Local.a, plain]
- kind: SetVariable
  variable:
    path: Local.c
  value: =Concatenate("x", Local.a)
- kind: SetTextVariable
  variable: Local.t
  text: =Local.a
- kind: SetMultipleVariables
  assignments:
    - variable: Local.m1
      value: one
    - variable: { path: Local.m2 }
      value: =Upper("two")
    - path: Local.m3
      value: =3
- kind: ResetVariable
  variable: Local.m1
- kind: SetValue
  path: Workflow.Outputs.result
  value: =Local.c
"#,
    );
    let v = vars(&w.run(json!({})).await.unwrap()).await;
    assert_eq!(v["Local"]["a"], json!(2));
    assert_eq!(v["Local"]["b"], json!({"nested": 20, "list": [2, "plain"]}));
    assert_eq!(v["Local"]["c"], json!("x2"));
    assert_eq!(v["Local"]["t"], json!("2"));
    assert_eq!(v["Local"]["m1"], json!(null));
    assert_eq!(v["Local"]["m2"], json!("TWO"));
    assert_eq!(v["Local"]["m3"], json!(3));
    assert_eq!(v["Outputs"]["result"], json!("x2"));
}

#[tokio::test]
async fn writing_inputs_fails_the_run() {
    let w = wf(
        &WorkflowFactory::new(),
        "- kind: SetVariable\n  variable: Workflow.Inputs.x\n  value: 1\n",
    );
    assert!(run_err(&w, json!({})).await.contains("read-only"));
}

#[tokio::test]
async fn parse_value_conversions() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: ParseValue
  variable: Local.n
  value: "3.5"
  valueType: number
- kind: ParseValue
  variable: Local.b
  value: "yes"
  valueType: Boolean
- kind: ParseValue
  variable: Local.o
  value: '{"k": [1]}'
  valueType: object
- kind: ParseValue
  variable: Local.s
  value: =Workflow.Inputs.x
  valueType: string
"#,
    );
    let v = vars(&w.run(json!({"x": 7})).await.unwrap()).await;
    assert_eq!(v["Local"]["n"], json!(3.5));
    assert_eq!(v["Local"]["b"], json!(true));
    assert_eq!(v["Local"]["o"], json!({"k": [1]}));
    assert_eq!(v["Local"]["s"], json!("7"));
}

#[tokio::test]
async fn edit_table_python_operations() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: EditTable
  table: Local.items
  value: a
- kind: EditTable
  table: Local.items
  operation: add
  value: c
- kind: EditTable
  table: Local.items
  operation: insert
  index: 1
  value: b
- kind: EditTable
  table: Local.items
  operation: set
  index: 0
  value: A
- kind: EditTable
  table: Local.items
  operation: remove
  value: c
- kind: EditTableV2
  table: Local.records
  item: "={id: 1, v: \"x\"}"
- kind: EditTableV2
  table: Local.records
  operation: addOrUpdate
  key: id
  item: "={id: 1, v: \"y\"}"
- kind: EditTableV2
  table: Local.records
  operation: addOrUpdate
  key: id
  item: "={id: 2, v: \"z\"}"
- kind: EditTableV2
  table: Local.records
  operation: remove
  key: id
  item: "={id: 1}"
"#,
    );
    let v = vars(&w.run(json!({})).await.unwrap()).await;
    assert_eq!(v["Local"]["items"], json!(["A", "b"]));
    assert_eq!(v["Local"]["records"], json!([{"id": 2, "v": "z"}]));
}

#[tokio::test]
async fn send_activity_renders_like_upstream() {
    let w = wf(
        &WorkflowFactory::new(),
        r#"
- kind: SetVariable
  variable: Local.rec
  value: "={a: 1}"
- kind: SendActivity
  activity: =Local.rec
- kind: SendActivity
  activity: =0
- kind: SendActivity
  activity:
    text: "flag={Workflow.Inputs.flag} missing=[{Local.nope}]"
- kind: SendActivity
  activity: '="Hello, " & Workflow.Inputs.name & "!"'
"#,
    );
    let run = w.run(json!({"flag": true, "name": "Bo"})).await.unwrap();
    assert_eq!(
        texts(&run),
        vec![r#"{"a":1}"#, "flag=True missing=[]", "Hello, Bo!"]
    );
}

#[tokio::test]
async fn message_list_input_seeds_conversation() {
    let w = wf(
        &WorkflowFactory::new(),
        "- kind: SendActivity\n  activity: =System.LastMessage.Text & \"|\" & Text(CountRows(Conversation.messages))\n",
    );
    let input = json!([
        {"role": "user", "contents": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "contents": [{"type": "text", "text": "reply"}]},
        {"role": "user", "contents": [{"type": "text", "text": "second"}]}
    ]);
    let run = w.run(input).await.unwrap();
    assert_eq!(texts(&run), vec!["second|2"]);
    let v = vars(&run).await;
    assert_eq!(v["Inputs"]["input"], json!("second"));
}

#[tokio::test]
async fn expression_errors_fail_with_clear_messages() {
    let w = wf(
        &WorkflowFactory::new(),
        "- kind: SetVariable\n  variable: Local.x\n  value: =Frobnicate(1)\n",
    );
    let err = run_err(&w, json!({})).await;
    assert!(
        err.contains("'Frobnicate' is an unknown or unsupported function"),
        "{err}"
    );

    let w = wf(
        &WorkflowFactory::new(),
        "- kind: SetVariable\n  variable: Local.x\n  value: =1 / 0\n",
    );
    assert!(run_err(&w, json!({})).await.contains("division by zero"));
}

#[tokio::test]
async fn state_budget_is_enforced() {
    let w = WorkflowFactory::new()
        .with_state_budget(agent_framework_declarative::powerfx::limits::StateBudget {
            max_depth: 64,
            max_nodes: 10_000,
            max_text_size: 1_000,
        })
        .create_workflow_from_yaml(
            "actions:\n  - kind: SetVariable\n    variable: Local.big\n    value: =Workflow.Inputs.text\n",
        )
        .unwrap();
    assert!(w.run(json!({"text": "small"})).await.is_ok());
    let err = run_err(&w, json!({"text": "x".repeat(2_000)})).await;
    assert!(err.contains("text size budget"), "{err}");
}

#[tokio::test]
async fn env_symbol_uses_configuration_only_by_default() {
    let yaml = "- kind: SendActivity\n  activity: =Env.GREETING & \"/\" & Env.FALLBACK\n";
    let factory = WorkflowFactory::new()
        .with_configuration([("GREETING", "hi")])
        .with_env_source(|k: &str| (k == "FALLBACK").then(|| "env".to_string()));
    assert_eq!(
        texts(&wf(&factory, yaml).run(json!({})).await.unwrap()),
        vec!["hi/"]
    );
    let factory = factory.restrict_env_to_configuration(false);
    assert_eq!(
        texts(&wf(&factory, yaml).run(json!({})).await.unwrap()),
        vec!["hi/env"]
    );
}

// ------------------------------------------------------------ validation

#[test]
fn validation_errors_match_upstream() {
    let f = WorkflowFactory::new();
    let cases = [
        ("kind: Workflow\nactions: []\n", "no actions"),
        ("kind: Workflow\n", "'actions' field or 'trigger.actions'"),
        ("actions:\n  - id: x\n", "missing 'kind'"),
        ("actions:\n  - kind: SetValue\n    value: 1\n", "missing required field 'path'"),
        ("actions:\n  - kind: SendActivity\n    id: a\n    activity: x\n  - kind: SendActivity\n    id: a\n    activity: y\n", "Duplicate action ID 'a'"),
        ("actions:\n  - kind: SendActivity\n    id: _workflow_entry\n    activity: x\n", "reserved"),
        ("actions:\n  - kind: GotoAction\n    id: g\n    actionId: g\n", "self-referencing"),
        ("actions:\n  - kind: GotoAction\n    actionId: nowhere\n", "GotoAction target 'nowhere' not found"),
        ("actions:\n  - kind: BreakLoop\n", "only be used inside a Foreach"),
        ("actions:\n  - kind: ConditionGroup\n    conditions: []\n    else: []\n", "use 'elseActions'"),
        ("actions:\n  - kind: HttpRequestAction\n    url: https://x.test\n", "no HTTP request handler"),
        ("actions:\n  - kind: InvokeMcpTool\n    serverUrl: https://x.test\n    toolName: t\n", "no MCP tool handler"),
        ("maxTurns: 0\nactions:\n  - kind: EndWorkflow\n", "maxTurns"),
        ("actions: [\n", "Invalid YAML"),
        ("actions:\n  - kind: Foreach\n    actions: []\n", "missing required field 'source'"),
    ];
    for (yaml, expected) in cases {
        let err = build_err(&f, yaml);
        assert!(err.contains(expected), "{yaml}\n=> {err}");
    }
    // Alternates satisfy requirements.
    assert!(f
        .create_workflow_from_yaml("actions:\n  - kind: GotoAction\n    id: a\n    target: b\n  - kind: EndWorkflow\n    id: b\n")
        .is_ok());
}

// ------------------------------------------------------------ function tools

fn tool_factory(calls: Arc<Mutex<Vec<Json>>>) -> WorkflowFactory {
    WorkflowFactory::new()
        .with_function("add", move |args: Json| {
            let calls = calls.clone();
            async move {
                calls.lock().unwrap().push(args.clone());
                Ok(json!(
                    args["a"].as_f64().unwrap() + args["b"].as_f64().unwrap()
                ))
            }
        })
        .with_function("boom", |_args: Json| async move {
            Err(agent_framework_core::error::Error::Tool("kaput".into()))
        })
}

#[tokio::test]
async fn invoke_function_tool_stores_results_and_errors() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let w = wf(
        &tool_factory(calls.clone()),
        r#"
- kind: InvokeFunctionTool
  functionName: =Workflow.Inputs.fn
  arguments: { a: 2, b: =Workflow.Inputs.b }
  output: { result: sum, autoSend: =Workflow.Inputs.send }
- kind: InvokeFunctionTool
  functionName: boom
  output: { result: Local.err }
- kind: InvokeFunctionTool
  functionName: missing
  output: { result: Local.missing }
"#,
    );
    let run = w
        .run(json!({"fn": "add", "b": 3, "send": true}))
        .await
        .unwrap();
    assert_eq!(texts(&run), vec!["5.0"]);
    let v = vars(&run).await;
    assert_eq!(v["Local"]["sum"], json!(5.0));
    assert_eq!(
        v["Local"]["err"],
        json!({"error": "ToolError: tool error: kaput"})
    );
    assert_eq!(
        v["Local"]["missing"],
        json!({"error": "Function 'missing' not found in registry"})
    );
    assert_eq!(calls.lock().unwrap()[0], json!({"a": 2, "b": 3}));

    let run = w
        .run(json!({"fn": "add", "b": 1, "send": false}))
        .await
        .unwrap();
    assert!(texts(&run).is_empty());
}

#[tokio::test]
async fn invoke_function_tool_approval_flow() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let w = wf(
        &tool_factory(calls.clone()),
        r#"
- kind: InvokeFunctionTool
  functionName: add
  requireApproval: true
  arguments: { a: 1, b: 2 }
  output: { result: Local.r, messages: Local.msgs }
- kind: SendActivity
  activity: after
"#,
    );
    // Approve.
    let mut run = w.run(json!({})).await.unwrap();
    let (id, req) = pending(&run);
    assert_eq!(req["type"], "ToolApprovalRequest");
    assert_eq!(req["function_name"], "add");
    assert_eq!(req["arguments"], json!({"a": 1, "b": 2}));
    assert!(calls.lock().unwrap().is_empty());
    run.send_response(id, json!({"approved": true}))
        .await
        .unwrap();
    assert_eq!(texts(&run), vec!["3.0", "after"]);
    assert_eq!(vars(&run).await["Local"]["r"], json!(3.0));

    // Reject.
    let mut run = w.run(json!({})).await.unwrap();
    let (id, _) = pending(&run);
    run.send_response(id, json!({"approved": false, "reason": "no"}))
        .await
        .unwrap();
    let v = vars(&run).await;
    assert_eq!(
        v["Local"]["r"],
        json!({"approved": false, "rejected": true, "reason": "no"})
    );
    assert_eq!(
        v["Local"]["msgs"][0]["contents"][0]["text"],
        json!("Function 'add' was rejected: no")
    );
    assert_eq!(texts(&run), vec!["after"]);

    // Non-Boolean approval is an error.
    let mut run = w.run(json!({})).await.unwrap();
    let (id, _) = pending(&run);
    let err = run
        .send_response(id, json!({"approved": "yes"}))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("approved must be a bool"), "{err}");
}

// ------------------------------------------------------------ HTTP

#[derive(Default)]
struct FakeHttp {
    seen: Mutex<Vec<HttpRequestInfo>>,
    status: u16,
    body: String,
    timeout: bool,
}

#[async_trait]
impl HttpRequestHandler for FakeHttp {
    async fn send(
        &self,
        info: HttpRequestInfo,
    ) -> std::result::Result<HttpRequestResult, HttpRequestError> {
        self.seen.lock().unwrap().push(info);
        if self.timeout {
            return Err(HttpRequestError::Timeout("slow".into()));
        }
        let mut headers = BTreeMap::new();
        headers.insert(
            "set-cookie".to_string(),
            vec!["a=1".to_string(), "b=2".to_string()],
        );
        Ok(HttpRequestResult {
            status_code: self.status,
            is_success_status_code: (200..300).contains(&self.status),
            body: self.body.clone(),
            headers,
        })
    }
}

#[tokio::test]
async fn http_request_success_assigns_body_headers_and_conversation() {
    let http = Arc::new(FakeHttp {
        status: 200,
        body: r#"{"full_name": "dotnet/runtime", "stars": 5}"#.into(),
        ..Default::default()
    });
    let factory = WorkflowFactory::new().with_http_request_handler(http.clone());
    let w = factory
        .create_workflow_from_yaml(&common::fixture("python/test_http_request.yaml"))
        .unwrap();
    let run = w.run("go").await.unwrap();
    let v = vars(&run).await;
    assert_eq!(v["Local"]["RepoInfo"]["full_name"], json!("dotnet/runtime"));
    assert_eq!(v["Local"]["RepoHeaders"], json!({"set-cookie": "a=1,b=2"}));
    let conv = v["System"]["ConversationId"].as_str().unwrap().to_string();
    let msgs = &v["System"]["conversations"][&conv]["messages"];
    assert_eq!(msgs[0]["role"], "assistant");
    let seen = http.seen.lock().unwrap();
    assert_eq!(seen[0].method, "GET");
    assert_eq!(seen[0].url, "https://api.github.com/repos/dotnet/runtime");
    assert_eq!(
        seen[0].headers,
        vec![
            (
                "Accept".to_string(),
                "application/vnd.github+json".to_string()
            ),
            (
                "User-Agent".to_string(),
                "agent-framework-integration-test".to_string()
            )
        ]
    );
}

#[tokio::test]
async fn http_request_bodies_query_and_timeouts() {
    let http = Arc::new(FakeHttp {
        status: 201,
        body: "created".into(),
        ..Default::default()
    });
    let factory = WorkflowFactory::new().with_http_request_handler(http.clone());
    let w = wf(
        &factory,
        r#"
- kind: HttpRequestAction
  method: =Lower("post")
  url: https://x.test/items
  queryParameters: { z: 1, a: true, skip: =Blank() }
  headers: { X-Empty: "", X-Token: =Workflow.Inputs.token }
  requestTimeoutInMilliseconds: 1500
  connection: { name: conn }
  body: { kind: json, content: "={n: 1}" }
  response: { path: Local.resp }
- kind: HttpRequestAction
  url: https://x.test/raw
  body: { kind: RawRequestContent, content: plain text }
- kind: HttpRequestAction
  url: https://x.test/none
  body: { kind: none }
"#,
    );
    let run = w.run(json!({"token": "t"})).await.unwrap();
    assert_eq!(vars(&run).await["Local"]["resp"], json!("created"));
    let seen = http.seen.lock().unwrap();
    assert_eq!(seen[0].method, "POST");
    assert_eq!(
        seen[0].query_parameters,
        vec![("z".into(), "1".into()), ("a".into(), "true".into())]
    );
    assert_eq!(seen[0].headers, vec![("X-Token".into(), "t".into())]);
    assert_eq!(seen[0].timeout_ms, Some(1500));
    assert_eq!(seen[0].connection_name.as_deref(), Some("conn"));
    assert_eq!(seen[0].body.as_deref(), Some(r#"{"n":1}"#));
    assert_eq!(
        seen[0].body_content_type.as_deref(),
        Some("application/json")
    );
    assert_eq!(seen[1].body.as_deref(), Some("plain text"));
    assert_eq!(seen[1].body_content_type.as_deref(), Some("text/plain"));
    assert_eq!(seen[2].body, None);
}

#[tokio::test]
async fn http_request_failures() {
    let yaml = "- kind: HttpRequestAction\n  url: https://x.test/a\n  responseHeaders: Local.h\n";
    let http = Arc::new(FakeHttp {
        status: 500,
        body: "SECRET backend detail".into(),
        ..Default::default()
    });
    let w = wf(
        &WorkflowFactory::new().with_http_request_handler(http),
        yaml,
    );
    let err = run_err(&w, json!({})).await;
    assert!(
        err.contains("HTTP request to 'https://x.test/a' failed with status code 500"),
        "{err}"
    );
    assert!(!err.contains("SECRET"));

    let http = Arc::new(FakeHttp {
        timeout: true,
        ..Default::default()
    });
    let w = wf(
        &WorkflowFactory::new().with_http_request_handler(http),
        yaml,
    );
    assert!(run_err(&w, json!({})).await.contains("timed out"));

    let http = Arc::new(FakeHttp::default());
    let w = wf(
        &WorkflowFactory::new().with_http_request_handler(http),
        "- kind: HttpRequestAction\n  url: =Local.none\n",
    );
    assert!(run_err(&w, json!({}))
        .await
        .contains("'url' evaluated to an empty value"));
}

// ------------------------------------------------------------ MCP

#[derive(Default)]
struct FakeMcp {
    seen: Mutex<Vec<McpToolInvocation>>,
    fail: Option<&'static str>,
}

#[async_trait]
impl McpToolHandler for FakeMcp {
    async fn invoke_tool(
        &self,
        inv: McpToolInvocation,
    ) -> std::result::Result<McpToolResult, McpToolError> {
        self.seen.lock().unwrap().push(inv.clone());
        match self.fail {
            Some("tool") => Err(McpToolError::Tool("server said no".into())),
            Some(_) => Err(McpToolError::Other("bug".into())),
            None => Ok(McpToolResult {
                outputs: vec![
                    Content::text(format!("{{\"query\": {}}}", inv.arguments["query"])),
                    Content::text("note"),
                ],
                is_error: false,
                error_message: None,
            }),
        }
    }
}

#[tokio::test]
async fn invoke_mcp_tool_sample_with_and_without_approval() {
    let mcp = Arc::new(FakeMcp::default());
    let docs = ScriptedAgent::new("DocsAgent", &["answer"]);
    let factory = WorkflowFactory::new()
        .with_mcp_tool_handler(mcp.clone())
        .with_agent("DocsAgent", docs.clone());
    let w = factory
        .create_workflow_from_yaml(&common::fixture("python/invoke_mcp_tool.yaml"))
        .unwrap();

    let run = w
        .run(json!({"text": "rust", "requireApproval": false}))
        .await
        .unwrap();
    assert_eq!(texts(&run), vec!["answer"]);
    let v = vars(&run).await;
    assert_eq!(
        v["Local"]["SearchResults"],
        json!([{"query": "rust"}, "note"])
    );
    let conv = v["System"]["ConversationId"].as_str().unwrap().to_string();
    // The MCP outputs and the agent exchange share the conversation.
    let msgs = v["System"]["conversations"][&conv]["messages"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(msgs[0]["role"], "assistant");
    assert_eq!(msgs[0]["contents"].as_array().unwrap().len(), 2);
    {
        let seen = mcp.seen.lock().unwrap();
        assert_eq!(seen[0].server_url, "https://learn.microsoft.com/api/mcp");
        assert_eq!(seen[0].server_label.as_deref(), Some("MicrosoftLearnDocs"));
        assert_eq!(seen[0].tool_name, "microsoft_docs_search");
    }

    let mut run = w
        .run(json!({"text": "rust", "requireApproval": true}))
        .await
        .unwrap();
    let (id, req) = pending(&run);
    assert_eq!(req["type"], "MCPToolApprovalRequest");
    assert_eq!(req["tool_name"], "microsoft_docs_search");
    assert_eq!(req["arguments"], json!({"query": "rust"}));
    assert_eq!(req["header_binding"], json!(null));
    assert_eq!(mcp.seen.lock().unwrap().len(), 1);
    run.send_response(id, json!(true)).await.unwrap();
    assert_eq!(mcp.seen.lock().unwrap().len(), 2);
    assert_eq!(texts(&run), vec!["answer"]);

    let mut run = w
        .run(json!({"text": "rust", "requireApproval": true}))
        .await
        .unwrap();
    let (id, _) = pending(&run);
    run.send_response(id, json!({"approved": false}))
        .await
        .unwrap();
    assert_eq!(
        vars(&run).await["Local"]["SearchResults"],
        json!("Error: MCP tool invocation was not approved by user.")
    );
    assert_eq!(mcp.seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn invoke_mcp_tool_errors_and_auto_send() {
    let yaml = r#"
- kind: InvokeMcpTool
  serverUrl: https://m.test
  toolName: t
  arguments: { query: x }
  output: { result: Local.r, messages: { path: Local.m } }
- kind: SendActivity
  activity: next
"#;
    let ok = Arc::new(FakeMcp::default());
    let run = wf(&WorkflowFactory::new().with_mcp_tool_handler(ok), yaml)
        .run(json!({}))
        .await
        .unwrap();
    assert_eq!(texts(&run), vec![r#"[{"query":"x"},"note"]"#, "next"]);
    assert_eq!(vars(&run).await["Local"]["m"]["role"], json!("tool"));

    let tool_err = Arc::new(FakeMcp {
        fail: Some("tool"),
        ..Default::default()
    });
    let run = wf(
        &WorkflowFactory::new().with_mcp_tool_handler(tool_err),
        yaml,
    )
    .run(json!({}))
    .await
    .unwrap();
    assert_eq!(texts(&run), vec!["next"]);
    assert_eq!(
        vars(&run).await["Local"]["r"],
        json!("Error: server said no")
    );

    let other = Arc::new(FakeMcp {
        fail: Some("other"),
        ..Default::default()
    });
    let w = wf(&WorkflowFactory::new().with_mcp_tool_handler(other), yaml);
    assert!(run_err(&w, json!({})).await.contains("bug"));
}

#[tokio::test]
async fn invoke_mcp_tool_header_binding_requires_fresh_approval_when_headers_change() {
    let token = Arc::new(Mutex::new("one".to_string()));
    let env_token = token.clone();
    let mcp = Arc::new(FakeMcp::default());
    let factory = WorkflowFactory::new()
        .with_mcp_tool_handler(mcp.clone())
        .restrict_env_to_configuration(false)
        .with_env_source(move |k: &str| (k == "TOKEN").then(|| env_token.lock().unwrap().clone()));
    let w = wf(
        &factory,
        r#"
- kind: InvokeMcpTool
  serverUrl: https://m.test
  toolName: t
  requireApproval: "yes"
  headers: { Authorization: =Env.TOKEN }
  arguments: { query: q }
  output: { result: Local.r, autoSend: false }
"#,
    );
    let mut run = w.run(json!({})).await.unwrap();
    let (id, req) = pending(&run);
    assert_eq!(req["header_names"], json!(["Authorization"]));
    assert!(
        req.to_string().find("one").is_none(),
        "header values must not leak"
    );
    let binding = req["header_binding"].as_str().unwrap().to_string();
    assert_eq!(binding.len(), 64);

    // Headers changed while paused: re-approval instead of invocation.
    *token.lock().unwrap() = "two".into();
    run.send_response(id, json!(true)).await.unwrap();
    assert!(mcp.seen.lock().unwrap().is_empty());
    let (id, req) = pending(&run);
    assert_ne!(req["header_binding"].as_str().unwrap(), binding);

    // Approving the fresh request (headers unchanged since) invokes the tool.
    run.send_response(id, json!(true)).await.unwrap();
    let seen = mcp.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].headers,
        vec![("Authorization".to_string(), "two".to_string())]
    );
}

// ------------------------------------------------------------ loader & checkpoints

#[tokio::test]
async fn loader_dispatches_on_document_shape() {
    let mut agents = AgentRegistry::new();
    agents.register("Echo", ScriptedAgent::new("Echo", &["echoed"]));
    let tools = ToolRegistry::new().with(
        FunctionTool::new(
            "double",
            "double a number",
            json!({"type": "object"}),
            |args: Json| async move { Ok(json!(args["n"].as_f64().unwrap_or(0.0) * 2.0)) },
        )
        .into_definition(),
    );
    let loader = DeclarativeLoader::new().with_tool_registry(tools);
    let yaml = r#"
kind: Workflow
trigger:
  kind: OnConversationStart
  actions:
    - kind: InvokeAzureAgent
      agent: Echo
      input: =Workflow.Inputs.input
      resultProperty: Local.reply
    - kind: InvokeFunctionTool
      functionName: double
      arguments: { n: 21 }
      output: { result: Local.d, autoSend: false }
"#;
    let run = loader
        .load_workflow(yaml, &agents)
        .unwrap()
        .run("hi")
        .await
        .unwrap();
    assert_eq!(texts(&run), vec!["echoed"]);
    let v = vars(&run).await;
    assert_eq!(v["Local"]["reply"], json!("echoed"));
    assert_eq!(v["Local"]["d"], json!(42.0));

    // The Rust-native schema still loads through the same entry point.
    let native = "kind: Workflow\ntype: sequential\nparticipants: [Echo]\n";
    assert!(loader.load_workflow(native, &agents).is_ok());
}

#[tokio::test]
async fn inline_agent_definitions_use_the_agent_loader() {
    let loader = DeclarativeLoader::new().with_client_factory(
        agent_framework_declarative::ChatClientFactory::new().with_default(|_m| {
            Ok(Arc::new(common::MockClient::always("from inline agent")) as Arc<dyn ChatClient>)
        }),
    );
    let yaml = r#"
kind: Workflow
agents:
  Inline:
    kind: Prompt
    name: Inline
    instructions: Be brief.
trigger:
  actions:
    - kind: InvokeAzureAgent
      agent: { name: Inline }
"#;
    let run = WorkflowFactory::new()
        .with_agent_loader(loader)
        .create_workflow_from_yaml(yaml)
        .unwrap()
        .run("hello")
        .await
        .unwrap();
    assert_eq!(texts(&run), vec!["from inline agent"]);

    let err = build_err(
        &WorkflowFactory::new(),
        "agents:\n  X:\n    connection: {}\nactions:\n  - kind: EndWorkflow\n",
    );
    assert!(err.contains("Failed to create agent 'X'"), "{err}");
}

#[tokio::test]
async fn paused_runs_resume_from_checkpoints() {
    let storage = Arc::new(InMemoryCheckpointStorage::new());
    let w = WorkflowFactory::new()
        .with_checkpoint_storage(storage.clone())
        .create_workflow_from_yaml(&common::fixture("python/human_in_loop.yaml"))
        .unwrap();
    let run = w.run(json!({})).await.unwrap();
    let (_, req) = pending(&run);
    assert_eq!(req["message"], "What is your name?");
    drop(run);

    let latest = storage
        .list(None)
        .await
        .unwrap()
        .into_iter()
        .max_by_key(|c| c.iteration_count)
        .unwrap();
    let mut resumed = w
        .run_from_checkpoint(&latest.checkpoint_id, storage.clone())
        .await
        .unwrap();
    assert_eq!(resumed.state(), WorkflowRunState::IdleWithPendingRequests);
    let (id, _) = pending(&resumed);
    resumed.send_response(id, json!("Cy")).await.unwrap();
    assert!(texts(&resumed).contains(&"Nice to meet you, Cy!".to_string()));
}
