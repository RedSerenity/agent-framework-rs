//! Hermetic loopback tests for [`FoundryEvals`], [`evaluate_traces`] and
//! [`evaluate_foundry_target`]: a fake OpenAI Evals API on a bare
//! `std::net::TcpListener` exercises the real `reqwest` path — routes, auth
//! headers, request bodies, polling, output-item paging and error handling.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure::StaticTokenCredential;
use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::evaluation::{
    EvalItem, EvalRunStatus, EvaluateAgent, Evaluator, ExpectedToolCall,
};
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, Message};
use agent_framework_foundry::{
    evaluate_foundry_target, evaluate_traces, EvaluateFoundryTarget, EvaluateTraces, FoundryEvals,
    GeneratedEvaluatorRef,
};
use async_trait::async_trait;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
struct Recorded {
    start_line: String,
    headers: HashMap<String, String>,
    body: Value,
}

fn read_request(stream: &mut std::net::TcpStream) -> Option<Recorded> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let (mut header_end, mut content_length) = (None, 0usize);
    loop {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if header_end.is_none() {
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                header_end = Some(pos);
                let headers = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
                content_length = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
            }
        }
        if let Some(pos) = header_end {
            if buf.len() >= pos + 4 + content_length {
                break;
            }
        }
    }
    let raw = String::from_utf8_lossy(&buf).to_string();
    let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((raw.as_str(), ""));
    let mut lines = head.lines();
    let start_line = lines.next().unwrap_or_default().to_string();
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    Some(Recorded {
        start_line,
        headers,
        body: serde_json::from_str(body).unwrap_or(Value::Null),
    })
}

/// Serve `responses` in order, one per connection, then keep repeating the
/// last one; record every request.
fn server(responses: Vec<(u16, Value)>) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let writer = seen.clone();
    std::thread::spawn(move || {
        let mut i = 0usize;
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { return };
            let Some(req) = read_request(&mut stream) else {
                continue;
            };
            writer.lock().unwrap().push(req);
            let (status, body) = &responses[i.min(responses.len() - 1)];
            i += 1;
            let body = body.to_string();
            let reason = if *status == 200 { "OK" } else { "ERR" };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (format!("http://{addr}"), seen)
}

fn evals(base: &str) -> FoundryEvals {
    FoundryEvals::new(base, "gpt-4o", "secret-key")
        .with_poll_interval(Duration::from_millis(5))
        .with_timeout(Duration::from_secs(5))
}

fn completed_run() -> Value {
    json!({
        "id": "run_456",
        "status": "completed",
        "result_counts": {"passed": 2, "failed": 0, "errored": 0, "total": 2},
        "report_url": "https://portal.azure.com/eval/run_456",
        "per_testing_criteria_results": [{"testing_criteria": "relevance", "passed": 2, "failed": 0}],
    })
}

fn output_item(id: &str) -> Value {
    json!({
        "id": id,
        "status": "pass",
        "results": [{"name": "relevance", "score": 5, "passed": true,
                     "sample": {"properties": {"dimension_scores": [
                         {"id": "clarity", "score": 4, "applicable": true, "weight": 2, "reason": "ok"}]}}}],
        "sample": {"error": null, "usage": null, "input": [], "output": []},
        "datasource_item": {},
    })
}

fn items() -> Vec<EvalItem> {
    vec![
        EvalItem::new(vec![
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ]),
        EvalItem::new(vec![
            Message::user("Weather?"),
            Message::assistant("Sunny."),
        ])
        .with_context("forecast doc"),
    ]
}

#[tokio::test]
async fn evaluate_creates_eval_runs_polls_and_pages_output_items() {
    let (base, seen) = server(vec![
        (200, json!({"id": "eval_123"})),
        (200, json!({"id": "run_456"})),
        (200, json!({"id": "run_456", "status": "queued"})),
        (200, completed_run()),
        (
            200,
            json!({"data": [output_item("oi_1")], "has_more": true, "last_id": "oi_1"}),
        ),
        (
            200,
            json!({"data": [output_item("oi_2")], "has_more": false}),
        ),
    ]);
    let fe = evals(&base).with_evaluators([FoundryEvals::RELEVANCE, FoundryEvals::GROUNDEDNESS]);
    let results = fe.evaluate(&items(), "Agent Framework Eval").await.unwrap();

    assert_eq!(results.provider, "Microsoft Foundry");
    assert_eq!(results.status, EvalRunStatus::Completed);
    assert_eq!(
        (results.eval_id.as_str(), results.run_id.as_str()),
        ("eval_123", "run_456")
    );
    assert_eq!(
        results.report_url.as_deref(),
        Some("https://portal.azure.com/eval/run_456")
    );
    assert!(results.all_passed());
    assert_eq!(results.passed(), 2);
    assert_eq!(results.result_counts.unwrap().total, Some(2));
    assert_eq!(results.per_evaluator["relevance"].passed, 2);
    assert_eq!(results.items.len(), 2);
    assert_eq!(results.items[1].item_id, "oi_2");
    let score = &results.items[0].scores[0];
    assert_eq!((score.name.as_str(), score.score), ("relevance", 5.0));
    assert_eq!(score.dimensions.as_ref().unwrap()[0].id, "clarity");
    results
        .assert_dimension_score_at_least("clarity", 4.0, Some("relevance"), true, None)
        .unwrap();

    let seen = seen.lock().unwrap();
    let lines: Vec<&str> = seen.iter().map(|r| r.start_line.as_str()).collect();
    assert_eq!(
        lines,
        [
            "POST /openai/v1/evals HTTP/1.1",
            "POST /openai/v1/evals/eval_123/runs HTTP/1.1",
            "GET /openai/v1/evals/eval_123/runs/run_456 HTTP/1.1",
            "GET /openai/v1/evals/eval_123/runs/run_456 HTTP/1.1",
            "GET /openai/v1/evals/eval_123/runs/run_456/output_items?limit=100 HTTP/1.1",
            "GET /openai/v1/evals/eval_123/runs/run_456/output_items?limit=100&after=oi_1 HTTP/1.1",
        ]
    );
    assert!(seen
        .iter()
        .all(|r| r.headers.get("api-key").map(String::as_str) == Some("secret-key")));

    let create = &seen[0].body;
    assert_eq!(create["name"], "Agent Framework Eval");
    assert_eq!(create["data_source_config"]["type"], "custom");
    assert_eq!(create["data_source_config"]["include_sample_schema"], true);
    let props = &create["data_source_config"]["item_schema"]["properties"];
    assert!(props.get("context").is_some());
    assert!(props.get("tool_definitions").is_none());
    let criteria = create["testing_criteria"].as_array().unwrap();
    assert_eq!(criteria[0]["evaluator_name"], "builtin.relevance");
    assert_eq!(criteria[1]["data_mapping"]["context"], "{{item.context}}");

    let run = &seen[1].body;
    assert_eq!(run["name"], "Agent Framework Eval Run");
    assert_eq!(run["data_source"]["type"], "jsonl");
    let content = run["data_source"]["source"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["item"]["query"], "Hello");
    assert_eq!(content[0]["item"]["response"], "Hi there!");
    assert_eq!(content[1]["item"]["context"], "forecast doc");
    assert_eq!(
        content[0]["item"]["query_messages"],
        json!([{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])
    );
}

#[tokio::test]
async fn default_evaluators_and_bearer_auth() {
    let (base, seen) = server(vec![
        (200, json!({"id": "e"})),
        (200, json!({"id": "r"})),
        (
            200,
            json!({"status": "completed", "result_counts": {"passed": 1}}),
        ),
        (200, json!({"data": [], "has_more": false})),
    ]);
    let fe = FoundryEvals::with_token_credential(
        base.clone(),
        "gpt-4o",
        Arc::new(StaticTokenCredential::new("tok")),
    )
    .with_poll_interval(Duration::from_millis(5));
    let item = EvalItem::new(vec![Message::user("Hi"), Message::assistant("Hello")])
        .with_expected_output("Hello");
    fe.evaluate(&[item], "E").await.unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(
        seen[0].headers.get("authorization").map(String::as_str),
        Some("Bearer tok")
    );
    let names: Vec<&str> = seen[0].body["testing_criteria"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["relevance", "coherence", "task_adherence"]);
    let props = &seen[0].body["data_source_config"]["item_schema"]["properties"];
    assert!(props.get("ground_truth").is_some());
    assert_eq!(
        seen[1].body["data_source"]["source"]["content"][0]["item"]["ground_truth"],
        "Hello"
    );
}

#[tokio::test]
async fn polling_times_out() {
    let (base, _seen) = server(vec![
        (200, json!({"id": "eval_1"})),
        (200, json!({"id": "run_1"})),
        (200, json!({"status": "queued"})),
    ]);
    let fe = evals(&base)
        .with_evaluators(["relevance"])
        .with_poll_interval(Duration::from_millis(10))
        .with_timeout(Duration::from_millis(50));
    let results = fe.evaluate(&items(), "E").await.unwrap();
    assert_eq!(results.status, EvalRunStatus::Timeout);
    assert_eq!(
        (results.eval_id.as_str(), results.run_id.as_str()),
        ("eval_1", "run_1")
    );
    assert!(results.raise_for_status(None).is_err());
}

#[tokio::test]
async fn failed_and_canceled_runs_skip_output_items() {
    let (base, seen) = server(vec![
        (200, json!({"id": "e"})),
        (200, json!({"id": "r"})),
        (
            200,
            json!({"status": "failed", "error": {"code": "x", "message": "Model deployment unavailable"}}),
        ),
    ]);
    let results = evals(&base)
        .with_evaluators(["relevance"])
        .evaluate(&items(), "E")
        .await
        .unwrap();
    assert_eq!(results.status, EvalRunStatus::Failed);
    assert_eq!(
        results.error.as_deref(),
        Some("Model deployment unavailable")
    );
    assert!(results.items.is_empty());
    assert_eq!(seen.lock().unwrap().len(), 3);

    let (base, _) = server(vec![
        (200, json!({"id": "e"})),
        (200, json!({"id": "r"})),
        (200, json!({"status": "canceled", "error": "ignored"})),
    ]);
    let results = evals(&base)
        .with_evaluators(["relevance"])
        .evaluate(&items(), "E")
        .await
        .unwrap();
    assert_eq!(results.status, EvalRunStatus::Canceled);
    assert!(results.error.is_none());
}

#[tokio::test]
async fn http_errors_propagate_and_tool_only_evaluators_fail_fast() {
    let (base, _) = server(vec![(500, json!({"error": "boom"}))]);
    let err = evals(&base)
        .with_evaluators(["relevance"])
        .evaluate(&items(), "E")
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(500));

    // No request is made when every evaluator needs tools nobody has.
    let err = evals("http://127.0.0.1:9")
        .with_evaluators([FoundryEvals::TOOL_CALL_ACCURACY])
        .evaluate(&items(), "E")
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Configuration(_)));
}

#[tokio::test]
async fn evaluate_traces_response_ids_and_trace_ids() {
    let (base, seen) = server(vec![
        (200, json!({"id": "eval_tr"})),
        (200, json!({"id": "run_tr"})),
        (
            200,
            json!({"status": "completed", "result_counts": {"passed": 1}}),
        ),
        (200, json!({"data": [output_item("oi_resp")]})),
    ]);
    let results = evaluate_traces(
        &evals(&base),
        EvaluateTraces::new().response_ids(["resp_abc", "resp_def"]),
    )
    .await
    .unwrap();
    assert_eq!(results.provider, "foundry");
    assert_eq!(results.items[0].item_id, "oi_resp");
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].body["name"], "Agent Framework Trace Eval");
        assert_eq!(
            seen[0].body["data_source_config"],
            json!({"type": "azure_ai_source", "scenario": "responses"})
        );
        assert!(seen[0].body["testing_criteria"][0]
            .get("data_mapping")
            .is_none());
        let ds = &seen[1].body["data_source"];
        assert_eq!(ds["type"], "azure_ai_responses");
        assert_eq!(
            ds["item_generation_params"]["source"]["content"][0]["item"]["resp_id"],
            "resp_abc"
        );
    }

    let (base, seen) = server(vec![
        (200, json!({"id": "eval_tid"})),
        (200, json!({"id": "run_tid"})),
        (
            200,
            json!({"status": "completed", "result_counts": {"passed": 1}}),
        ),
        (200, json!({"data": []})),
    ]);
    let results = evaluate_traces(
        &evals(&base),
        EvaluateTraces::new()
            .trace_ids(["trace_1"])
            .agent_id("agent-7")
            .lookback_hours(6)
            .evaluators([GeneratedEvaluatorRef::new("rubric", "3")]),
    )
    .await
    .unwrap();
    assert_eq!(results.provider, "Microsoft Foundry");
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen[0].body["data_source_config"]["scenario"], "traces");
    assert_eq!(
        seen[0].body["testing_criteria"][0]["evaluator_version"],
        "3"
    );
    assert_eq!(
        seen[1].body["data_source"],
        json!({"type": "azure_ai_traces", "lookback_hours": 6, "trace_ids": ["trace_1"], "agent_id": "agent-7"})
    );

    let err = evaluate_traces(&evals("http://127.0.0.1:9"), EvaluateTraces::new())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Provide at least one of"));
}

#[tokio::test]
async fn evaluate_foundry_target_builds_target_data_source() {
    let (base, seen) = server(vec![
        (200, json!({"id": "eval_t"})),
        (200, json!({"id": "run_t"})),
        (
            200,
            json!({"status": "completed", "result_counts": {"passed": 1}}),
        ),
        (200, json!({"data": []})),
    ]);
    let results = evaluate_foundry_target(
        &evals(&base),
        EvaluateFoundryTarget::new(
            json!({"type": "azure_ai_agent", "name": "my-agent"}),
            ["Book a flight to Paris"],
        )
        .evaluators(["relevance"]),
    )
    .await
    .unwrap();
    assert!(results.all_passed());
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen[0].body["name"], "Agent Framework Target Eval");
    assert_eq!(
        seen[0].body["data_source_config"]["scenario"],
        "target_completions"
    );
    assert_eq!(
        seen[1].body["data_source"],
        json!({
            "type": "azure_ai_target_completions",
            "target": {"type": "azure_ai_agent", "name": "my-agent"},
            "source": {"type": "file_content", "content": [{"item": {"query": "Book a flight to Paris"}}]},
        })
    );

    let err = evaluate_foundry_target(
        &evals("http://127.0.0.1:9"),
        EvaluateFoundryTarget::new(json!({"name": "x"}), ["q"]),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("must include a 'type' key"));
}

struct WeatherAgent;

#[async_trait]
impl SupportsAgentRun for WeatherAgent {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        Ok(AgentResponse {
            messages: vec![Message::assistant("It is sunny.")],
            ..AgentResponse::default()
        })
    }
    fn id(&self) -> &str {
        "weather"
    }
}

#[tokio::test]
async fn plugs_into_evaluate_agent() {
    let (base, seen) = server(vec![
        (200, json!({"id": "e"})),
        (200, json!({"id": "r"})),
        (
            200,
            json!({"status": "completed", "result_counts": {"passed": 1, "failed": 0}}),
        ),
        (200, json!({"data": [], "has_more": false})),
    ]);
    let results = EvaluateAgent::new()
        .agent(&WeatherAgent)
        .query("Weather?")
        .expected_tool_calls_for_query([ExpectedToolCall::new("get_weather")])
        .evaluator(evals(&base).with_evaluators(["relevance"]))
        .run()
        .await
        .unwrap();
    assert_eq!(results[0].provider, "Microsoft Foundry");
    assert!(results[0].all_passed());
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].body["name"], "Eval: weather");
    assert_eq!(
        seen[1].body["data_source"]["source"]["content"][0]["item"]["response"],
        "It is sunny."
    );
}
