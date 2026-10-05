//! `HarnessAgent` — modeled on upstream `test_harness_agent.py`.

mod common;

use std::sync::Arc;

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::compaction::Truncation;
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{ApprovalMode, FunctionTool, ToolKind};
use agent_framework_core::types::{ChatOptions, ChatResponse, Content, Message, Role};
use agent_framework_harness::agent::{
    assemble_instructions, ChatClientHarnessExt, HarnessAgent, DEFAULT_HARNESS_INSTRUCTIONS,
    HARNESS_AGENT_PROVIDER_NAME,
};
use agent_framework_harness::file_access::{
    AgentFileStore, FileAccessProvider, InMemoryAgentFileStore,
};
use agent_framework_harness::history::SessionStateHistoryProvider;
use agent_framework_harness::loop_agent::{todos_remaining, todos_remaining_message};
use agent_framework_harness::tool_approval::approval_rule;
use agent_framework_harness::util::SessionRef;
use common::{call, tool_calls, MockClient, ScriptedAgent};
use futures::StreamExt;
use serde_json::json;

fn builder(client: MockClient) -> agent_framework_harness::agent::HarnessAgentBuilder {
    HarnessAgent::builder(client).file_memory_store(Arc::new(InMemoryAgentFileStore::new()))
}

fn tool_names(options: &ChatOptions) -> Vec<String> {
    options.tools.iter().map(|t| t.name.clone()).collect()
}

#[test]
fn instructions_are_assembled_like_upstream() {
    assert_eq!(
        assemble_instructions(None, None).as_deref(),
        Some(DEFAULT_HARNESS_INSTRUCTIONS.trim())
    );
    assert_eq!(
        assemble_instructions(None, Some("Focus.")).unwrap(),
        format!("{DEFAULT_HARNESS_INSTRUCTIONS}\n\nFocus.")
    );
    assert_eq!(
        assemble_instructions(Some(""), Some("Only agent.")).as_deref(),
        Some("Only agent.")
    );
    assert_eq!(assemble_instructions(Some(""), None), None);
}

#[test]
fn validation_mirrors_upstream() {
    let err =
        |b: agent_framework_harness::agent::HarnessAgentBuilder| b.build().unwrap_err().to_string();
    assert!(
        err(builder(MockClient::default()).max_context_window_tokens(0))
            .contains("max_context_window_tokens must be positive.")
    );
    assert!(err(builder(MockClient::default()).max_output_tokens(0))
        .contains("max_output_tokens must be positive."));
    assert!(err(builder(MockClient::default())
        .max_context_window_tokens(100)
        .max_output_tokens(100))
    .contains("max_output_tokens must be less than max_context_window_tokens."));
    let named = ScriptedAgent {
        name: Some("worker".into()),
        ..Default::default()
    };
    assert!(err(builder(MockClient::default())
        .background_agent(Arc::new(named) as Arc<dyn SupportsAgentRun>)
        .background_agents_wait_timeout_seconds(0))
    .contains("wait_timeout_seconds"));
}

#[tokio::test]
async fn defaults_wire_providers_tools_decorators_and_instructions() {
    let client = MockClient::new(vec![ChatResponse::from_text("hi")]);
    let agent = builder(client.clone())
        .name("helper")
        .agent_instructions("Be brief.")
        .max_context_window_tokens(1000)
        .max_output_tokens(100)
        .build()
        .unwrap();
    assert_eq!(agent.name(), Some("helper"));
    assert!(agent.has_tool_approval());
    assert!(!agent.has_loop());
    assert_eq!(agent.compaction_phases(), (true, true));
    assert_eq!(agent.otel_provider_name(), HARNESS_AGENT_PROVIDER_NAME);
    assert!(agent.todo_provider().is_some());
    assert!(agent.mode_provider().is_some());
    assert!(agent.file_memory_provider().is_some());
    assert!(
        agent.file_access_provider().is_none(),
        "file access is opt-in"
    );
    assert!(agent.background_agents_provider().is_none());
    // compaction, todo, mode, file memory
    assert_eq!(agent.context_providers().len(), 4);

    let mut session = agent.create_session();
    assert!(session.context_providers[0].is_history_provider());
    let out = agent
        .run(vec![Message::user("hello")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(out.text(), "hi");
    let options = client.last_options();
    assert_eq!(options.max_tokens, Some(100));
    let names = tool_names(&options);
    assert_eq!(options.tools[0].kind, ToolKind::HostedWebSearch);
    for expected in ["todos_add", "mode_set", "mode_get", "file_memory_write"] {
        assert!(
            names.contains(&expected.to_string()),
            "{expected} in {names:?}"
        );
    }
    let system = client.last_messages()[0].text();
    assert!(system.starts_with("You are a helpful AI assistant that uses tools to complete tasks."));
    assert!(system.contains("\n\nBe brief."));
    assert!(system.contains("## Todo Items"));
    assert!(system.contains("## Agent Mode"));
    assert!(system.contains("## File Based Memory"));
    // History lives in session state (upstream's InMemoryHistoryProvider shape).
    let stored = SessionStateHistoryProvider::new()
        .messages(&session.state)
        .unwrap();
    assert_eq!(stored.len(), 2);
}

#[tokio::test]
async fn features_can_be_disabled() {
    let client = MockClient::new(vec![]);
    let agent = builder(client.clone())
        .disable_todo(true)
        .disable_mode(true)
        .disable_file_memory(true)
        .disable_web_search(true)
        .disable_tool_auto_approval(true)
        .build()
        .unwrap();
    assert!(agent.context_providers().is_empty());
    assert!(!agent.has_tool_approval());
    assert_eq!(
        agent.compaction_phases(),
        (false, false),
        "no token params, no compaction"
    );
    agent.run(vec![Message::user("x")], None).await.unwrap();
    assert!(client.last_options().tools.is_empty());
    let system = client.last_messages()[0].text();
    assert_eq!(system, DEFAULT_HARNESS_INSTRUCTIONS.trim());

    // Custom strategies enable compaction without token params; disable wins.
    let strategy: Arc<dyn agent_framework_core::compaction::CompactionStrategy> =
        Arc::new(Truncation::new(4));
    let custom = builder(MockClient::default())
        .before_compaction_strategy(strategy.clone())
        .build()
        .unwrap();
    assert_eq!(custom.compaction_phases(), (true, false));
    let custom = builder(MockClient::default())
        .after_compaction_strategy(strategy.clone())
        .build()
        .unwrap();
    assert_eq!(custom.compaction_phases(), (false, true));
    let disabled = builder(MockClient::default())
        .before_compaction_strategy(strategy)
        .disable_compaction(true)
        .build()
        .unwrap();
    assert_eq!(disabled.compaction_phases(), (false, false));
}

#[tokio::test]
async fn end_to_end_todo_tool_call_and_history_across_runs() {
    let client = MockClient::new(vec![
        tool_calls(vec![call(
            "c1",
            "todos_add",
            json!({"todos": [{"title": "Draft outline"}]}),
        )]),
        ChatResponse::from_text("Added a todo."),
        ChatResponse::from_text("Still on it."),
    ]);
    let agent = builder(client.clone()).build().unwrap();
    let mut session = agent.create_session();
    let out = agent
        .run(vec![Message::user("plan my essay")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(out.text(), "Added a todo.");
    let items = agent
        .todo_provider()
        .unwrap()
        .load_items(&SessionRef::from_session(&session))
        .await
        .unwrap();
    assert_eq!(items[0].title, "Draft outline");

    agent
        .run(vec![Message::user("status?")], Some(&mut session))
        .await
        .unwrap();
    let sent = client.last_messages();
    let rendered: Vec<String> = sent.iter().map(|m| m.text()).collect();
    assert!(rendered
        .iter()
        .any(|t| t == "### Current todo list\n- 1 [open] Draft outline"));
    assert!(
        rendered.iter().any(|t| t == "plan my essay"),
        "history replayed: {rendered:?}"
    );
    assert_eq!(rendered.last().unwrap(), "status?");

    // State-backed history survives a session round trip through to_dict.
    let restored_state = session.to_dict();
    let mut restored = AgentSession::from_dict(&restored_state).unwrap();
    client.push(ChatResponse::from_text("restored"));
    agent
        .run(vec![Message::user("again")], Some(&mut restored))
        .await
        .unwrap();
    assert!(client
        .last_messages()
        .iter()
        .any(|m| m.text() == "plan my essay"));
}

#[tokio::test]
async fn file_access_requires_approval_and_bypasses_siblings_that_do_not() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let client = MockClient::new(vec![
        tool_calls(vec![
            call(
                "c1",
                "file_access_write",
                json!({"file_name": "out.md", "content": "report"}),
            ),
            call("c2", "todos_add", json!({"todos": [{"title": "Review"}]})),
        ]),
        ChatResponse::from_text("Wrote the report."),
    ]);
    let agent = builder(client.clone())
        .file_access_store(store.clone())
        .build()
        .unwrap();
    assert!(agent.file_access_provider().is_some());
    let mut session = agent.create_session();
    let out = agent
        .run(vec![Message::user("write it")], Some(&mut session))
        .await
        .unwrap();
    let requests = out.user_input_requests();
    assert_eq!(requests.len(), 1, "only the write needs a human");
    assert_eq!(requests[0].function_call.name, "file_access_write");
    assert!(!store.file_exists("out.md").await.unwrap());

    let approval = requests[0].create_response(true);
    let out = agent
        .run(
            vec![Message::with_contents(
                Role::user(),
                vec![Content::FunctionApprovalResponse(approval)],
            )],
            Some(&mut session),
        )
        .await
        .unwrap();
    assert_eq!(out.text(), "Wrote the report.");
    assert_eq!(
        store.read("out.md").await.unwrap().as_deref(),
        Some("report")
    );
    let items = agent
        .todo_provider()
        .unwrap()
        .load_items(&SessionRef::from_session(&session))
        .await
        .unwrap();
    assert_eq!(items.len(), 1, "the bypassed sibling ran too");
}

#[tokio::test]
async fn auto_approval_rules_and_approval_opt_outs() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    store.write("in.txt", "data", true).await.unwrap();
    let client = MockClient::new(vec![
        tool_calls(vec![call(
            "c1",
            "file_access_read",
            json!({"file_name": "in.txt"}),
        )]),
        ChatResponse::from_text("It says data."),
    ]);
    let agent = builder(client.clone())
        .file_access_store(store.clone())
        .auto_approval_rules([approval_rule(
            FileAccessProvider::read_only_tools_auto_approval_rule,
        )])
        .build()
        .unwrap();
    let out = agent.run(vec![Message::user("read")], None).await.unwrap();
    assert_eq!(out.text(), "It says data.");
    let tool_result = client
        .last_messages()
        .iter()
        .flat_map(|m| m.contents.clone())
        .find_map(|c| match c {
            Content::FunctionResult(r) => r.result,
            _ => None,
        });
    assert_eq!(tool_result, Some(json!("data")));

    let agent = builder(MockClient::default())
        .file_access_store(store)
        .file_access_disable_readonly_tool_approval(true)
        .file_access_disable_write_tools(true)
        .build()
        .unwrap();
    let modes = agent.file_access_provider().unwrap().tool_approval_modes();
    assert_eq!(modes.len(), 4);
    assert!(modes.iter().all(|(_, m)| *m == ApprovalMode::NeverRequire));
}

#[tokio::test]
async fn a_caller_tool_requiring_approval_is_never_bypassed() {
    let tool = FunctionTool::new(
        "todos_add",
        "shadow",
        json!({"type": "object"}),
        |_a| async { Ok(json!("x")) },
    )
    .with_approval_mode(ApprovalMode::AlwaysRequire)
    .into_definition();
    let client = MockClient::new(vec![tool_calls(vec![call("c1", "todos_add", json!({}))])]);
    let agent = builder(client)
        .disable_web_search(true)
        .tool(tool)
        .build()
        .unwrap();
    let out = agent.run(vec![Message::user("x")], None).await.unwrap();
    assert_eq!(out.user_input_requests().len(), 1);
}

#[tokio::test]
async fn loop_wiring_runs_until_todos_are_complete() {
    let client = MockClient::new(vec![
        tool_calls(vec![call(
            "c1",
            "todos_add",
            json!({"todos": [{"title": "Step"}]}),
        )]),
        ChatResponse::from_text("Added."),
        tool_calls(vec![call(
            "c2",
            "todos_complete",
            json!({"items": [{"id": 1, "reason": "did it"}]}),
        )]),
        ChatResponse::from_text("Finished."),
    ]);
    let agent = builder(client.clone())
        .loop_should_continue(todos_remaining(None).unwrap())
        .loop_next_message(todos_remaining_message())
        .build()
        .unwrap();
    assert!(agent.has_loop());
    let mut session = agent.create_session();
    let out = agent
        .run(vec![Message::user("do the step")], Some(&mut session))
        .await
        .unwrap();
    assert!(out.text().ends_with("Finished."));
    let nudges: Vec<String> = out
        .messages
        .iter()
        .filter(|m| m.role == Role::user())
        .map(|m| m.text())
        .collect();
    assert_eq!(nudges[0], "Progress so far:\n- Added.");
    assert!(
        nudges[1].starts_with("You still have 1 open todo item(s)"),
        "{nudges:?}"
    );
    let items = agent
        .todo_provider()
        .unwrap()
        .load_items(&SessionRef::from_session(&session))
        .await
        .unwrap();
    assert!(items[0].is_complete);
}

#[tokio::test]
async fn after_compaction_bounds_the_stored_transcript() {
    let client = MockClient::new(vec![]);
    let agent = builder(client)
        .after_compaction_strategy(Arc::new(Truncation::new(3)))
        .build()
        .unwrap();
    let mut session = agent.create_session();
    for turn in ["one", "two", "three"] {
        agent
            .run(vec![Message::user(turn)], Some(&mut session))
            .await
            .unwrap();
    }
    let stored = SessionStateHistoryProvider::new()
        .messages(&session.state)
        .unwrap();
    assert!(stored.len() <= 4, "{}", stored.len());
    assert_eq!(stored.last().unwrap().role, Role::assistant());
}

#[tokio::test]
async fn background_agents_and_streaming() {
    let worker = agent_framework_core::agent::Agent::builder(MockClient::new(vec![
        ChatResponse::from_text("found it"),
    ]))
    .name("worker")
    .description("Does work")
    .build();
    let client = MockClient::new(vec![ChatResponse::from_text("streamed answer")]);
    let agent = client
        .clone()
        .as_harness_agent()
        .file_memory_store(Arc::new(InMemoryAgentFileStore::new()))
        .background_agent(worker)
        .build()
        .unwrap();
    assert!(agent.background_agents_provider().is_some());
    let stream = agent
        .run_stream(vec![Message::user("go")], None, None)
        .await
        .unwrap();
    let text: String = stream
        .map(|u| u.unwrap().text())
        .collect::<Vec<_>>()
        .await
        .concat();
    assert_eq!(text, "streamed answer");
    let system = client.last_messages()[0].text();
    assert!(system.contains("Available background agents:\n- worker: Does work"));
}
