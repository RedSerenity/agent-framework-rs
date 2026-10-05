//! `TodoProvider` — modeled on upstream `test_harness_todo.py`.

mod common;

use std::sync::Arc;

use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::session::SessionState;
use agent_framework_core::tools::ApprovalMode;
use agent_framework_harness::todo::{TodoFileStore, TodoProvider, DEFAULT_TODO_INSTRUCTIONS};
use agent_framework_harness::util::SessionRef;
use common::{ctx, invoke, invoke_text, TempDir};
use serde_json::json;

async fn run(provider: &TodoProvider, state: &SessionState) -> SessionContext {
    let mut c = ctx("s1", state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    c
}

#[tokio::test]
async fn injects_instructions_tools_and_current_list() {
    let provider = TodoProvider::new();
    let state = SessionState::new();
    let c = run(&provider, &state).await;
    assert_eq!(c.instructions.as_deref(), Some(DEFAULT_TODO_INSTRUCTIONS));
    let names: Vec<&str> = c.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "todos_add",
            "todos_complete",
            "todos_remove",
            "todos_get_remaining",
            "todos_get_all"
        ]
    );
    assert!(c
        .tools
        .iter()
        .all(|t| t.approval_mode == ApprovalMode::NeverRequire));
    assert_eq!(c.messages[0].text(), "### Current todo list\n- none yet");
}

#[tokio::test]
async fn tools_add_complete_remove_and_query() {
    let provider = TodoProvider::new();
    let state = SessionState::new();
    let c = run(&provider, &state).await;
    let t = &c.tools;
    assert_eq!(
        invoke_text(t, "todos_add", json!({"todos": [{"title": " Research "}, {"title": "Write", "description": " draft "}]})).await,
        r#"[{"id": 1, "title": "Research", "description": null, "is_complete": false}, {"id": 2, "title": "Write", "description": "draft", "is_complete": false}]"#
    );
    assert!(invoke(t, "todos_add", json!({"todos": []})).await.is_err());
    assert!(invoke(t, "todos_add", json!({"todos": [{"title": "  "}]}))
        .await
        .is_err());
    assert!(invoke(
        t,
        "todos_complete",
        json!({"items": [{"id": 1, "reason": " "}]})
    )
    .await
    .is_err());
    assert_eq!(
        invoke_text(
            t,
            "todos_complete",
            json!({"items": [{"id": 1, "reason": "done"}, {"id": 99, "reason": "x"}]})
        )
        .await,
        r#"{"completed": 1}"#
    );
    assert_eq!(
        invoke_text(
            t,
            "todos_complete",
            json!({"items": [{"id": 1, "reason": "again"}]})
        )
        .await,
        r#"{"completed": 0}"#
    );
    assert_eq!(
        invoke_text(t, "todos_get_remaining", json!({})).await,
        r#"[{"id": 2, "title": "Write", "description": "draft", "is_complete": false}]"#
    );
    let next = run(&provider, &state).await;
    assert_eq!(
        next.messages[0].text(),
        "### Current todo list\n- 1 [done] Research\n- 2 [open] Write: draft"
    );
    assert_eq!(
        invoke_text(t, "todos_remove", json!({"ids": [1, 5]})).await,
        r#"{"removed": 1}"#
    );
    assert!(invoke(t, "todos_remove", json!({"ids": []})).await.is_err());
    assert_eq!(
        invoke_text(t, "todos_get_all", json!({})).await,
        r#"[{"id": 2, "title": "Write", "description": "draft", "is_complete": false}]"#
    );
    // Ids keep increasing past removed ones.
    assert!(
        invoke_text(t, "todos_add", json!({"todos": [{"title": "Next"}]}))
            .await
            .contains("\"id\": 3")
    );
    assert_eq!(state.get("todo").unwrap()["next_id"], 4);
}

#[tokio::test]
async fn file_store_backs_the_provider_and_isolates_sessions() {
    let dir = TempDir::new("todo-files");
    let provider = TodoProvider::new().store(Arc::new(TodoFileStore::new(dir.path())));
    let state = SessionState::new();
    let c = run(&provider, &state).await;
    invoke_text(
        &c.tools,
        "todos_add",
        json!({"todos": [{"title": "Persisted"}]}),
    )
    .await;
    assert!(dir.path().join("s1").join("todos.todo.json").is_file());
    assert!(
        state.get("todo").is_none(),
        "file store keeps nothing in session state"
    );
    let other = SessionRef {
        session_id: "s2".into(),
        state: SessionState::new(),
    };
    assert!(provider.load_items(&other).await.unwrap().is_empty());
    let same = SessionRef {
        session_id: "s1".into(),
        state: SessionState::new(),
    };
    assert_eq!(
        provider.load_items(&same).await.unwrap()[0].title,
        "Persisted"
    );
}

#[tokio::test]
async fn provider_requires_a_session() {
    let provider = TodoProvider::new();
    let mut c = SessionContext::new(vec![]);
    let err = provider.before_run(&mut c).await.unwrap_err();
    assert!(err.to_string().contains("requires an AgentSession"));
}

#[tokio::test]
async fn custom_source_id_and_instructions() {
    let provider = TodoProvider::new()
        .source_id("tasks")
        .instructions("custom");
    let state = SessionState::new();
    let c = run(&provider, &state).await;
    assert_eq!(c.instructions.as_deref(), Some("custom"));
    invoke_text(&c.tools, "todos_add", json!({"todos": [{"title": "x"}]})).await;
    assert!(state.get("tasks").is_some());
    assert!(state.get("todo").is_none());
}
