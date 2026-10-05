//! End to end: this workspace's MCP client against `McpServer` over real
//! streamable HTTP, and raw JSON-RPC over the stdio framing.

use std::sync::Arc;

use agent_framework_core::agent::{AgentRunOptions, SupportsAgentRun};
use agent_framework_core::error::Result;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, Message};
use agent_framework_core::workflow::{FunctionExecutor, WorkflowBuilder};
use agent_framework_hosting_mcp::{AgentMcpTool, McpServer, WorkflowMcpTool};
use agent_framework_mcp::{McpClient, McpStreamableHttpTransport};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Replies `"<task> #<turn>"`, counting turns in the session, and reports
/// the temperature it was given.
struct Counter;

#[async_trait]
impl SupportsAgentRun for Counter {
    async fn run(&self, m: Vec<Message>, s: Option<&mut AgentSession>) -> Result<AgentResponse> {
        self.run_with_options(m, s, AgentRunOptions::default())
            .await
    }
    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        let turn = match session {
            Some(s) => {
                let n = s.state.get("n").and_then(|v| v.as_u64()).unwrap_or(0) + 1;
                s.state.insert("n", json!(n));
                n
            }
            None => 0,
        };
        let temperature = options.chat_options.and_then(|c| c.temperature);
        Ok(AgentResponse {
            messages: vec![Message::assistant(format!(
                "{} #{turn} t={temperature:?}",
                messages[0].text()
            ))],
            ..Default::default()
        })
    }
    fn id(&self) -> &str {
        "counter"
    }
    fn name(&self) -> Option<&str> {
        Some("Counter Agent")
    }
}

fn server() -> McpServer {
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(FunctionExecutor::new(
            "up",
            |v: Value, ctx| async move {
                ctx.yield_output(json!(v.as_str().unwrap_or_default().to_uppercase()))
                    .await
            },
        )))
        .set_start("up")
        .build()
        .unwrap();
    McpServer::new("test-server", "0.1.0")
        .instructions("be nice")
        .tool(
            AgentMcpTool::new(Arc::new(Counter))
                .description("counts")
                .parameter("conversation", json!({ "type": "string" }))
                .session_id_parameter("conversation")
                .chat_option_parameter("temperature", json!({ "type": "number" })),
        )
        .tool(WorkflowMcpTool::new(workflow).name("shout"))
}

async fn serve_http() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, server().into_router("/mcp"))
            .await
            .unwrap();
    });
    format!("http://{addr}/mcp")
}

#[tokio::test]
async fn the_workspace_client_lists_and_calls_hosted_tools() {
    let url = serve_http().await;
    let transport =
        McpStreamableHttpTransport::new(url, reqwest::header::HeaderMap::new(), None).unwrap();
    let client = McpClient::new(Arc::new(transport));
    let init = client.initialize("test-client", "1").await.unwrap();
    assert_eq!(init.server_info.name, "test-server");
    assert_eq!(init.instructions.as_deref(), Some("be nice"));

    let tools = client.list_tools().await.unwrap();
    let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["Counter_Agent", "shout"]);
    let schema = &tools[0].input_schema;
    assert_eq!(schema["required"], json!(["task", "conversation"]));
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(
        tools[1].input_schema["properties"]["input"],
        json!({ "type": "string" })
    );

    // The session parameter carries the conversation across calls.
    for expected in ["hi #1 t=Some(0.5)", "hi #2 t=None"] {
        let args = if expected.ends_with("Some(0.5)") {
            json!({ "task": "hi", "conversation": "c1", "temperature": 0.5 })
        } else {
            json!({ "task": "hi", "conversation": "c1" })
        };
        let result = client.call_tool("Counter_Agent", args).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.to_value(), json!(expected));
    }
    let other = client
        .call_tool(
            "Counter_Agent",
            json!({ "task": "yo", "conversation": "c2" }),
        )
        .await
        .unwrap();
    assert_eq!(other.to_value(), json!("yo #1 t=None"));

    let shout = client
        .call_tool("shout", json!({ "input": "quiet" }))
        .await
        .unwrap();
    assert_eq!(shout.to_value(), json!("QUIET"));

    // A contract violation is a tool error the model can read.
    let bad = client
        .call_tool("Counter_Agent", json!({ "conversation": "c1" }))
        .await
        .unwrap();
    assert!(bad.is_error);
    assert!(bad.error_message().contains("task"));

    // An unknown tool is a protocol error.
    assert!(client.call_tool("nope", json!({})).await.is_err());
    client.close().await.unwrap();
}

#[tokio::test]
async fn http_sessions_are_enforced() {
    let url = serve_http().await;
    let http = reqwest::Client::new();
    let ping = json!({ "jsonrpc": "2.0", "id": 1, "method": "ping" });

    let missing = http.post(&url).json(&ping).send().await.unwrap();
    assert_eq!(missing.status(), 400);
    let unknown = http
        .post(&url)
        .header("mcp-session-id", "nope")
        .json(&ping)
        .send()
        .await
        .unwrap();
    assert_eq!(unknown.status(), 404);

    let init = http
        .post(&url)
        .json(&json!({ "jsonrpc": "2.0", "id": 0, "method": "initialize", "params": { "protocolVersion": "2025-03-26" } }))
        .send()
        .await
        .unwrap();
    let session = init.headers()["mcp-session-id"]
        .to_str()
        .unwrap()
        .to_string();
    let body: Value = init.json().await.unwrap();
    // The client's (supported) version is echoed back.
    assert_eq!(body["result"]["protocolVersion"], "2025-03-26");

    let ok = http
        .post(&url)
        .header("mcp-session-id", &session)
        .json(&ping)
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), 200);
    let note = http
        .post(&url)
        .header("mcp-session-id", &session)
        .json(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))
        .send()
        .await
        .unwrap();
    assert_eq!(note.status(), 202);
    assert_eq!(http.get(&url).send().await.unwrap().status(), 405);

    let deleted = http
        .delete(&url)
        .header("mcp-session-id", &session)
        .send()
        .await
        .unwrap();
    assert_eq!(deleted.status(), 200);
    let after = http
        .post(&url)
        .header("mcp-session-id", &session)
        .json(&ping)
        .send()
        .await
        .unwrap();
    assert_eq!(after.status(), 404);
}

#[tokio::test]
async fn stdio_framing_answers_requests_in_order() {
    let (client, server_side) = tokio::io::duplex(64 * 1024);
    let (server_read, server_write) = tokio::io::split(server_side);
    tokio::spawn(async move { server().serve_io(server_read, server_write).await });

    let (client_read, mut client_write) = tokio::io::split(client);
    let mut lines = BufReader::new(client_read).lines();
    let send = |v: Value| format!("{v}\n");
    client_write
        .write_all(
            send(json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} }))
                .as_bytes(),
        )
        .await
        .unwrap();
    client_write
        .write_all(
            send(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })).as_bytes(),
        )
        .await
        .unwrap();
    client_write.write_all(b"{not json\n").await.unwrap();
    client_write
        .write_all(
            send(json!({ "jsonrpc": "2.0", "id": 2, "method": "resources/list" })).as_bytes(),
        )
        .await
        .unwrap();
    client_write
        .write_all(
            send(json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": { "name": "shout", "arguments": { "input": "a" } } }))
            .as_bytes(),
        )
        .await
        .unwrap();

    macro_rules! next {
        () => {
            serde_json::from_str::<Value>(&lines.next_line().await.unwrap().unwrap()).unwrap()
        };
    }
    let init = next!();
    assert_eq!(init["id"], 1);
    assert_eq!(
        init["result"]["capabilities"]["tools"]["listChanged"],
        false
    );
    // No reply to the notification; the parse error comes next.
    assert_eq!(next!()["error"]["code"], -32700);
    assert_eq!(next!()["error"]["code"], -32601);
    let call = next!();
    assert_eq!(call["id"], 3);
    assert_eq!(call["result"]["content"][0]["text"], "A");
}

#[tokio::test]
async fn misconfigured_agent_tools_are_reported() {
    let tool = AgentMcpTool::new(Arc::new(Counter)).session_id_parameter("missing");
    assert!(tool.validate().is_err());
    let tool = AgentMcpTool::new(Arc::new(Counter)).parameter("task", json!({ "type": "string" }));
    assert!(tool.validate().is_err());
    let tool = AgentMcpTool::new(Arc::new(Counter)).required("ghost");
    assert!(tool.validate().is_err());
}

/// The server ends the session (as a restart would): the client tool sees
/// `404` on its session id, reconnects with a fresh `initialize`, and
/// retries the call once — transparently to the caller.
#[tokio::test]
async fn a_tool_survives_the_server_ending_its_session() {
    use agent_framework_core::tools::ToolSource;
    use agent_framework_mcp::McpStreamableHttpTool;

    let server = server();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let router = server.clone().into_router("/mcp");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let tool = McpStreamableHttpTool::new("remote", url).load_prompts(false);
    let defs = tool.resolve_tools().await.unwrap();
    let shout = defs.iter().find(|d| d.name == "shout").unwrap();
    let call = || {
        shout
            .executor
            .as_ref()
            .unwrap()
            .invoke(json!({ "input": "a" }))
    };
    assert_eq!(call().await.unwrap(), json!("A"));
    let first = tool.client().await.unwrap();

    server.end_all_sessions();
    assert_eq!(call().await.unwrap(), json!("A"));
    let second = tool.client().await.unwrap();
    assert!(
        !Arc::ptr_eq(&first, &second),
        "a new session was established"
    );
    assert!(first.is_closed());
}
