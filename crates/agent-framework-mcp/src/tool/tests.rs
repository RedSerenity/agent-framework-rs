use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use agent_framework_core::middleware::{FunctionInvocationContext, LiveToolList};
use agent_framework_core::tools::ApprovalMode;
use serde_json::json;

use super::*;
use crate::protocol::CallToolResult;

// ---------------------------------------------------------------------------
// A scripted in-memory MCP server
// ---------------------------------------------------------------------------

/// What the fake server advertises and records.
#[derive(Default)]
struct Script {
    tools: Vec<Value>,
    prompts: Vec<Value>,
    capabilities: Value,
    /// `tools/call` params seen, in order.
    calls: Mutex<Vec<Value>>,
    /// Requests seen, by method.
    methods: Mutex<Vec<String>>,
    /// The result returned from every `tools/call`.
    call_result: Mutex<Option<Value>>,
    /// Connections made so far.
    connections: AtomicUsize,
}

struct FakeTransport {
    script: Arc<Script>,
    /// When set, the next `tools/call` fails and the connection reads closed.
    drop_next_call: Arc<AtomicBool>,
    closed: AtomicBool,
}

#[async_trait]
impl McpTransport for FakeTransport {
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        if self.closed.load(Ordering::Acquire) {
            return Err(agent_framework_core::Error::service("closed"));
        }
        self.script.methods.lock().unwrap().push(method.to_string());
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": "2025-06-18",
                "capabilities": self.script.capabilities,
                "serverInfo": { "name": "fake", "version": "1" },
            })),
            "ping" | "logging/setLevel" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": self.script.tools })),
            "prompts/list" => Ok(json!({ "prompts": self.script.prompts })),
            "prompts/get" => Ok(json!({
                "messages": [{
                    "role": "user",
                    "content": { "type": "text", "text": format!("prompt {} {}", params["name"], params["arguments"]) },
                }],
            })),
            "tools/call" => {
                if self.drop_next_call.swap(false, Ordering::AcqRel) {
                    self.closed.store(true, Ordering::Release);
                    return Err(agent_framework_core::Error::service("connection lost"));
                }
                self.script.calls.lock().unwrap().push(params.clone());
                Ok(self
                    .script
                    .call_result
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| json!({ "content": [{ "type": "text", "text": "ok" }] })))
            }
            other => Err(agent_framework_core::Error::service(format!(
                "unexpected {other}"
            ))),
        }
    }
    async fn notify(&self, _m: &str, _p: Value) -> Result<()> {
        Ok(())
    }
    async fn close(&self) -> Result<()> {
        Ok(())
    }
    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

fn tool(name: &str, properties: Value) -> Value {
    json!({
        "name": name,
        "description": format!("{name} tool"),
        "inputSchema": { "type": "object", "properties": properties },
    })
}

fn script(tools: Vec<Value>, prompts: Vec<Value>) -> Arc<Script> {
    let capabilities = if prompts.is_empty() {
        json!({ "tools": {} })
    } else {
        json!({ "tools": {}, "prompts": {} })
    };
    Arc::new(Script {
        tools,
        prompts,
        capabilities,
        ..Default::default()
    })
}

/// A session over `script` configured by `configure`, plus the switch that
/// drops the connection on the next call.
fn session_with(
    script: &Arc<Script>,
    configure: impl FnOnce(&mut Common),
) -> (Arc<McpSession>, Arc<AtomicBool>) {
    let mut common = Common::new("fake".into());
    configure(&mut common);
    let drop_next = Arc::new(AtomicBool::new(false));
    let (s, d) = (script.clone(), drop_next.clone());
    let factory: TransportFactory = Arc::new(move || {
        s.connections.fetch_add(1, Ordering::AcqRel);
        let t = FakeTransport {
            script: s.clone(),
            drop_next_call: d.clone(),
            closed: AtomicBool::new(false),
        };
        Box::pin(async move { Ok(Arc::new(t) as Arc<dyn McpTransport>) })
    });
    (McpSession::new(common, factory), drop_next)
}

fn names(defs: &[ToolDefinition]) -> Vec<&str> {
    defs.iter().map(|d| d.name.as_str()).collect()
}

fn find<'a>(defs: &'a [ToolDefinition], name: &str) -> &'a ToolDefinition {
    defs.iter()
        .find(|d| d.name == name)
        .unwrap_or_else(|| panic!("no {name}"))
}

async fn invoke(def: &ToolDefinition, args: Value) -> Result<Value> {
    def.executor.as_ref().unwrap().invoke(args).await
}

// ---------------------------------------------------------------------------
// Naming, filtering, approval
// ---------------------------------------------------------------------------

#[test]
fn prefixes_follow_upstream() {
    use crate::session::build_prefixed_name;
    assert_eq!(build_prefixed_name("search", None), "search");
    assert_eq!(build_prefixed_name("search", Some("")), "search");
    assert_eq!(
        build_prefixed_name("search", Some("github")),
        "github_search"
    );
    assert_eq!(
        build_prefixed_name("search", Some("my server.-")),
        "my-server_search"
    );
    assert_eq!(build_prefixed_name("_search", Some("gh")), "gh_search");
    assert_eq!(build_prefixed_name("", Some("gh")), "gh");
    assert_eq!(build_prefixed_name("search", Some("._-")), "search");
}

#[tokio::test]
async fn names_are_normalized_and_prefixed() {
    let s = script(vec![tool("weather/get current", json!({}))], vec![]);
    let (session, _) = session_with(&s, |c| c.tool_name_prefix = Some("wx".into()));
    let defs = session.tool_definitions(false).await.unwrap();
    assert_eq!(names(&defs), ["wx_weather-get-current"]);
    assert!(defs[0].is_executable());
}

#[tokio::test]
async fn allowed_tools_match_raw_names_not_normalized_aliases() {
    let s = script(
        vec![
            tool("echo", json!({})),
            tool("weather/get", json!({})),
            tool("delete", json!({})),
        ],
        vec![],
    );
    // "weather-get" is only a normalized alias of "weather/get": no match.
    let (session, _) = session_with(&s, |c| {
        c.allowed_tools = Some(["echo", "weather-get"].map(String::from).into())
    });
    assert_eq!(
        names(&session.tool_definitions(false).await.unwrap()),
        ["echo"]
    );

    // The raw name matches.
    let (session, _) = session_with(&s, |c| {
        c.allowed_tools = Some(["weather/get"].map(String::from).into())
    });
    assert_eq!(
        names(&session.tool_definitions(false).await.unwrap()),
        ["weather-get"]
    );

    // With a prefix, the prefixed name matches when the raw name is already
    // normalized.
    let (session, _) = session_with(&s, |c| {
        c.tool_name_prefix = Some("p".into());
        c.allowed_tools = Some(["p_echo"].map(String::from).into());
    });
    assert_eq!(
        names(&session.tool_definitions(false).await.unwrap()),
        ["p_echo"]
    );
}

#[tokio::test]
async fn approval_resolves_per_tool() {
    let s = script(
        vec![tool("read", json!({})), tool("delete", json!({}))],
        vec![],
    );
    let (session, _) = session_with(&s, |c| {
        c.approval_mode = McpApprovalMode::specific(["delete"], ["read"])
    });
    let defs = session.tool_definitions(false).await.unwrap();
    assert_eq!(
        find(&defs, "delete").approval_mode,
        ApprovalMode::AlwaysRequire
    );
    assert_eq!(
        find(&defs, "read").approval_mode,
        ApprovalMode::NeverRequire
    );

    let (session, _) = session_with(&s, |c| {
        c.approval_mode = McpApprovalMode::always_require_all()
    });
    let defs = session.tool_definitions(false).await.unwrap();
    assert!(defs
        .iter()
        .all(|d| d.approval_mode == ApprovalMode::AlwaysRequire));
    assert_eq!(
        McpApprovalMode::default().resolve_for_test("x"),
        ApprovalMode::NeverRequire
    );
}

#[tokio::test]
async fn two_tools_mapping_to_one_local_name_is_an_error() {
    let s = script(
        vec![
            tool("weather/get", json!({})),
            tool("weather-get", json!({})),
        ],
        vec![],
    );
    let (session, _) = session_with(&s, |_| {});
    let err = session
        .tool_definitions(false)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("same local function name"), "{err}");
}

#[tokio::test]
async fn an_ambiguous_configured_name_is_an_error() {
    // With prefix "a", remote "x" exposes as "a_x"; remote "a_x" is already
    // normalized, so its own name and its prefixed "a_a_x" both identify it.
    // Configuring "a_x" therefore names two different remote tools.
    let s = script(vec![tool("x", json!({})), tool("a_x", json!({}))], vec![]);
    let (session, _) = session_with(&s, |c| {
        c.tool_name_prefix = Some("a".into());
        c.allowed_tools = Some(["a_x"].map(String::from).into());
    });
    // "x" -> candidates ["x"] (local "a_x" only when remote is normalized:
    // "x" is, so ["x", "a_x"]); "a_x" -> ["a_x", "a_a_x"]. Both claim "a_x".
    let err = session
        .tool_definitions(false)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("ambiguous"), "{err}");
}

#[tokio::test]
async fn object_schemas_without_properties_get_an_empty_map() {
    let s = script(
        vec![json!({ "name": "now", "inputSchema": { "type": "object" } })],
        vec![],
    );
    let (session, _) = session_with(&s, |_| {});
    let defs = session.tool_definitions(false).await.unwrap();
    assert_eq!(defs[0].parameters["properties"], json!({}));
}

// ---------------------------------------------------------------------------
// Prompts as tools
// ---------------------------------------------------------------------------

#[tokio::test]
async fn prompts_become_functions_unless_disabled_and_never_shadow_tools() {
    let s = script(
        vec![tool("summarize", json!({}))],
        vec![
            json!({
                "name": "greet",
                "description": "Greets",
                "arguments": [
                    { "name": "who", "description": "Whom", "required": true },
                    { "name": "tone" },
                ],
            }),
            json!({ "name": "summarize" }),
        ],
    );
    let (session, _) = session_with(&s, |_| {});
    let defs = session.tool_definitions(false).await.unwrap();
    assert_eq!(names(&defs), ["summarize", "greet"]);
    let greet = find(&defs, "greet");
    assert_eq!(greet.parameters["required"], json!(["who"]));
    assert_eq!(greet.parameters["properties"]["who"]["description"], "Whom");
    assert!(greet.parameters["properties"]["tone"]
        .get("description")
        .is_none());
    let rendered = invoke(greet, json!({ "who": "Ada" })).await.unwrap();
    assert_eq!(rendered, json!("prompt \"greet\" {\"who\":\"Ada\"}"));

    let (session, _) = session_with(&s, |c| c.load_prompts = false);
    assert_eq!(
        names(&session.tool_definitions(false).await.unwrap()),
        ["summarize"]
    );
    assert!(!s.methods.lock().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Calls: argument filtering, _meta, result policy, retries
// ---------------------------------------------------------------------------

#[tokio::test]
async fn calls_forward_only_declared_and_extra_arguments_with_tool_meta() {
    let mut t = tool("search", json!({ "q": { "type": "string" } }));
    t["_meta"] = json!({ "proxy": "echo-me" });
    let s = script(vec![t, tool("other", json!({}))], vec![]);
    let (session, _) = session_with(&s, |c| {
        c.extra_arguments =
            AdditionalArgumentNames::global(["trace"]).for_tool("search", ["tenant"])
    });
    let defs = session.tool_definitions(false).await.unwrap();
    invoke(
        find(&defs, "search"),
        json!({ "q": "rust", "invented": 1, "trace": "t", "tenant": "x", "_meta": {} }),
    )
    .await
    .unwrap();
    invoke(find(&defs, "other"), json!({ "tenant": "x", "trace": "t" }))
        .await
        .unwrap();
    let calls = s.calls.lock().unwrap();
    assert_eq!(calls[0]["name"], "search");
    assert_eq!(
        calls[0]["arguments"],
        json!({ "q": "rust", "trace": "t", "tenant": "x" })
    );
    assert_eq!(calls[0]["_meta"], json!({ "proxy": "echo-me" }));
    assert_eq!(calls[1]["arguments"], json!({ "trace": "t" }));
    assert!(calls[1].get("_meta").is_none());
}

#[tokio::test]
async fn result_content_modes_and_custom_parser() {
    let s = script(vec![tool("t", json!({}))], vec![]);
    *s.call_result.lock().unwrap() = Some(json!({
        "content": [{ "type": "text", "text": "human text" }],
        "structuredContent": { "n": 1 },
    }));
    let expect = [
        (ToolResultContent::StructuredFirst, json!({ "n": 1 })),
        (ToolResultContent::ContentFirst, json!("human text")),
        (ToolResultContent::ContentOnly, json!("human text")),
        (ToolResultContent::StructuredOnly, json!({ "n": 1 })),
        (
            ToolResultContent::Both,
            json!([{ "type": "text", "text": "human text" }, { "type": "text", "text": "{\"n\":1}" }]),
        ),
    ];
    for (mode, expected) in expect {
        let (session, _) = session_with(&s, |c| c.result_content = mode);
        let defs = session.tool_definitions(false).await.unwrap();
        assert_eq!(
            invoke(&defs[0], json!({})).await.unwrap(),
            expected,
            "{mode:?}"
        );
    }
    let parser: ToolResultParser =
        Arc::new(|r: &CallToolResult| Ok(json!(format!("{} blocks", r.content.len()))));
    let (session, _) = session_with(&s, |c| c.result_parser = Some(parser));
    let defs = session.tool_definitions(false).await.unwrap();
    assert_eq!(
        invoke(&defs[0], json!({})).await.unwrap(),
        json!("1 blocks")
    );
}

#[tokio::test]
async fn an_error_result_is_a_tool_error() {
    let s = script(vec![tool("t", json!({}))], vec![]);
    *s.call_result.lock().unwrap() = Some(json!({
        "content": [{ "type": "text", "text": "boom" }],
        "isError": true,
    }));
    let (session, _) = session_with(&s, |_| {});
    let defs = session.tool_definitions(false).await.unwrap();
    let err = invoke(&defs[0], json!({})).await.unwrap_err().to_string();
    assert!(err.contains("boom"), "{err}");
}

#[tokio::test]
async fn a_lost_connection_reconnects_and_retries_once() {
    let s = script(vec![tool("t", json!({}))], vec![]);
    let (session, drop_next) = session_with(&s, |_| {});
    let defs = session.tool_definitions(false).await.unwrap();
    assert_eq!(s.connections.load(Ordering::Acquire), 1);
    drop_next.store(true, Ordering::Release);
    assert_eq!(invoke(&defs[0], json!({})).await.unwrap(), json!("ok"));
    assert_eq!(s.connections.load(Ordering::Acquire), 2);
    assert_eq!(s.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn calls_are_refused_when_tools_are_not_loaded() {
    let s = script(vec![tool("t", json!({}))], vec![]);
    let (session, _) = session_with(&s, |c| c.load_tools = false);
    let defs = session.tool_definitions(false).await.unwrap();
    let err = invoke(&defs[0], json!({})).await.unwrap_err().to_string();
    assert!(err.contains("load_tools"), "{err}");
}

#[tokio::test]
async fn logging_level_is_requested_only_from_servers_that_log() {
    for (capabilities, expected) in [(json!({ "logging": {} }), true), (json!({}), false)] {
        let s = Arc::new(Script {
            capabilities,
            ..Default::default()
        });
        let (session, _) = session_with(&s, |c| c.logging_level = Some(McpLogLevel::Warning));
        session.client().await.unwrap();
        let asked = s
            .methods
            .lock()
            .unwrap()
            .iter()
            .any(|m| m == "logging/setLevel");
        assert_eq!(asked, expected);
    }
}

// ---------------------------------------------------------------------------
// Progressive disclosure
// ---------------------------------------------------------------------------

fn ctx(live: &LiveToolList) -> FunctionInvocationContext {
    let mut ctx = FunctionInvocationContext::new("loader", Value::Null);
    ctx.tools = Some(live.clone());
    ctx
}

async fn loader(defs: &[ToolDefinition], name: &str, args: Value, live: &LiveToolList) -> Value {
    find(defs, name)
        .executor
        .as_ref()
        .unwrap()
        .invoke_in_context(args, &ctx(live))
        .await
        .unwrap()
}

#[tokio::test]
async fn progressive_disclosure_loads_and_unloads_into_the_run() {
    let s = script(
        vec![
            tool("search", json!({})),
            tool("fetch", json!({})),
            tool("pinned", json!({})),
        ],
        vec![],
    );
    let (session, _) = session_with(&s, |c| {
        c.progressive = true;
        c.always_load = ["pinned"].map(String::from).into();
    });
    let defs = session.tool_definitions(true).await.unwrap();
    assert_eq!(
        names(&defs),
        ["list_mcp_tools", "load_tool", "unload_tool", "pinned"]
    );
    assert!(defs
        .iter()
        .all(|d| d.approval_mode == ApprovalMode::NeverRequire));
    let live = LiveToolList::new(defs.clone());

    let listed = loader(&defs, "list_mcp_tools", json!({}), &live).await;
    let listed: Vec<(String, bool, bool)> = listed
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t["name"].as_str().unwrap().to_string(),
                t["loaded"].as_bool().unwrap(),
                t["always_loaded"].as_bool().unwrap(),
            )
        })
        .collect();
    assert!(listed.contains(&("search".into(), false, false)));
    assert!(listed.contains(&("pinned".into(), true, true)));

    let loaded = loader(
        &defs,
        "load_tool",
        json!({ "tool": ["search", "nope", "search"] }),
        &live,
    )
    .await;
    let loaded = loaded.as_str().unwrap();
    assert!(loaded.contains("Loaded MCP tool 'search'"), "{loaded}");
    assert!(loaded.contains("'nope' is not available"), "{loaded}");
    assert!(loaded.contains("already queued"), "{loaded}");
    assert!(live.contains("search"));
    // Loaded tools stay loaded for later runs of this source.
    assert!(names(&session.tool_definitions(true).await.unwrap()).contains(&"search"));

    let again = loader(&defs, "load_tool", json!({ "tool": "search" }), &live).await;
    assert!(again.as_str().unwrap().contains("already available"));

    let unloaded = loader(
        &defs,
        "unload_tool",
        json!({ "tool": ["search", "pinned", "fetch", "load_tool"] }),
        &live,
    )
    .await;
    let unloaded = unloaded.as_str().unwrap();
    assert!(
        unloaded.contains("Unloaded MCP tool 'search'"),
        "{unloaded}"
    );
    assert!(unloaded.contains("always_load"), "{unloaded}");
    assert!(
        unloaded.contains("'fetch' is not currently loaded"),
        "{unloaded}"
    );
    assert!(unloaded.contains("cannot be unloaded"), "{unloaded}");
    assert!(!live.contains("search"));
    assert!(!names(&session.tool_definitions(true).await.unwrap()).contains(&"search"));
}

#[tokio::test]
async fn progressive_loaders_take_the_prefix_and_need_a_live_run() {
    let s = script(vec![tool("search", json!({}))], vec![]);
    let (session, _) = session_with(&s, |c| {
        c.progressive = true;
        c.tool_name_prefix = Some("gh".into());
    });
    let defs = session.tool_definitions(true).await.unwrap();
    assert_eq!(
        names(&defs),
        ["gh_list_mcp_tools", "gh_load_tool", "gh_unload_tool"]
    );
    let err = find(&defs, "gh_load_tool")
        .executor
        .as_ref()
        .unwrap()
        .invoke(json!({ "tool": "search" }))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("inside an agent function-calling run"),
        "{err}"
    );
}

#[tokio::test]
async fn progressive_disclosure_requires_load_tools() {
    let tool = McpStdioTool::new("s", "cmd")
        .progressive_disclosure(true)
        .load_tools(false);
    assert!(tool.tool_definitions().await.is_err());
}

// ---------------------------------------------------------------------------
// Wrapper configuration (ported)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stdio_tool_builder_stores_configuration() {
    let tool = McpStdioTool::new("fs", "npx")
        .args(["-y", "server-filesystem"])
        .env([("KEY", "value")])
        .description("desc")
        .allowed_tools(["read_file"]);
    assert_eq!(tool.name(), "fs");
    assert_eq!(tool.description_text(), Some("desc"));
    assert_eq!(
        tool.args,
        vec!["-y".to_string(), "server-filesystem".to_string()]
    );
    assert_eq!(tool.env.as_ref().unwrap().get("KEY").unwrap(), "value");
    assert!(tool
        .common
        .allowed_tools
        .as_ref()
        .unwrap()
        .contains("read_file"));
}

#[tokio::test]
async fn http_and_websocket_builders_store_configuration() {
    let http = McpStreamableHttpTool::new("api", "https://example.com/mcp")
        .headers([("Authorization", "Bearer x")])
        .timeout(Duration::from_secs(5))
        .description("desc");
    assert_eq!(http.name(), "api");
    assert_eq!(http.description_text(), Some("desc"));
    assert_eq!(
        http.headers,
        vec![("Authorization".to_string(), "Bearer x".to_string())]
    );
    assert_eq!(http.timeout, Some(Duration::from_secs(5)));

    let ws = McpWebsocketTool::new("realtime", "wss://example.com/mcp")
        .headers([("Authorization", "Bearer x")])
        .request_timeout(Duration::from_secs(7))
        .allowed_tools(["echo"]);
    assert_eq!(ws.request_timeout, Some(Duration::from_secs(7)));
    assert!(ws.common.allowed_tools.as_ref().unwrap().contains("echo"));
}

#[test]
fn all_three_wrappers_default_load_tools_and_load_prompts_to_true() {
    for common in [
        McpStdioTool::new("s", "cmd").common,
        McpStreamableHttpTool::new("h", "https://example.com/mcp").common,
        McpWebsocketTool::new("w", "wss://example.com/mcp").common,
    ] {
        assert!(common.load_tools);
        assert!(common.load_prompts);
        assert_eq!(common.result_content, ToolResultContent::StructuredFirst);
        assert!(!common.progressive);
    }
}

#[tokio::test]
async fn resolve_tools_and_prompts_short_circuit_without_connecting() {
    let stdio = McpStdioTool::new("s", "does-not-exist-binary")
        .load_tools(false)
        .load_prompts(false);
    assert!(ToolSource::resolve_tools(&stdio).await.unwrap().is_empty());
    assert!(stdio.prompts().await.unwrap().is_empty());
    assert!(stdio.session.get().is_none(), "nothing may connect");

    let http = McpStreamableHttpTool::new("h", "http://127.0.0.1:0/unused")
        .load_tools(false)
        .load_prompts(false);
    assert!(ToolSource::resolve_tools(&http).await.unwrap().is_empty());
    assert!(http.prompts().await.unwrap().is_empty());
    assert!(http.session.get().is_none());

    let ws = McpWebsocketTool::new("w", "ws://127.0.0.1:0/unused")
        .load_tools(false)
        .load_prompts(false);
    assert!(ToolSource::resolve_tools(&ws).await.unwrap().is_empty());
    assert!(ws.prompts().await.unwrap().is_empty());
    assert!(ws.session.get().is_none());
}

#[test]
fn tool_source_name_matches_configured_name() {
    assert_eq!(ToolSource::source_name(&McpStdioTool::new("a", "cmd")), "a");
    assert_eq!(
        ToolSource::source_name(&McpStreamableHttpTool::new("b", "https://x/mcp")),
        "b"
    );
    assert_eq!(
        ToolSource::source_name(&McpWebsocketTool::new("c", "wss://x/mcp")),
        "c"
    );
}

#[test]
fn dedup_prompts_by_normalized_name_skips_collision_first_wins() {
    let p = |n: &str| PromptDescriptor {
        name: n.into(),
        description: None,
        arguments: None,
    };
    let deduped =
        dedup_prompts_by_normalized_name(vec![p("greet/user"), p("greet-user"), p("bye")]);
    let names: Vec<&str> = deduped.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["greet/user", "bye"]);
    let _ = HashSet::<String>::new();
}
