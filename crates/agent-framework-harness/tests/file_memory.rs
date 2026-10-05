//! `FileMemoryProvider` — modeled on upstream `test_harness_file_memory.py`.

mod common;

use std::sync::Arc;

use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::session::SessionState;
use agent_framework_core::tools::ApprovalMode;
use agent_framework_harness::file_access::{AgentFileStore, InMemoryAgentFileStore};
use agent_framework_harness::file_memory::{
    description_file_name, is_internal_file, FileMemoryProvider, DEFAULT_FILE_MEMORY_INSTRUCTIONS,
    MAX_INDEX_ENTRIES,
};
use common::{ctx, invoke, invoke_text};
use serde_json::json;

async fn tools_for(provider: &FileMemoryProvider, id: &str) -> SessionContext {
    let state = SessionState::new();
    let mut c = ctx(id, &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    c
}

#[test]
fn helpers_match_upstream() {
    assert_eq!(description_file_name("notes.md"), "notes_description.md");
    assert_eq!(description_file_name("notes"), "notes_description.md");
    assert_eq!(description_file_name(".hidden"), ".hidden_description.md");
    assert!(is_internal_file("x_description.md"));
    assert!(is_internal_file("MEMORIES.md"));
    assert!(!is_internal_file("plan.md"));
}

#[tokio::test]
async fn registers_tools_and_instructions() {
    let provider = FileMemoryProvider::new(Arc::new(InMemoryAgentFileStore::new()));
    let c = tools_for(&provider, "s1").await;
    assert_eq!(
        c.instructions.as_deref(),
        Some(DEFAULT_FILE_MEMORY_INSTRUCTIONS)
    );
    let names: Vec<&str> = c.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "file_memory_write",
            "file_memory_read",
            "file_memory_delete",
            "file_memory_ls",
            "file_memory_grep",
            "file_memory_replace",
            "file_memory_replace_lines"
        ]
    );
    assert!(c
        .tools
        .iter()
        .all(|t| t.approval_mode == ApprovalMode::NeverRequire));
    assert!(c.messages.is_empty(), "no index yet");
    let custom =
        FileMemoryProvider::new(Arc::new(InMemoryAgentFileStore::new())).instructions("custom");
    assert_eq!(
        tools_for(&custom, "s1").await.instructions.as_deref(),
        Some("custom")
    );
}

#[tokio::test]
async fn round_trip_sidecar_index_and_listing() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let provider = FileMemoryProvider::new(store.clone());
    let c = tools_for(&provider, "s1").await;
    let t = &c.tools;
    assert_eq!(
        invoke_text(
            t,
            "file_memory_write",
            json!({"file_name": "plan.md", "content": "step 1", "description": "The plan"})
        )
        .await,
        "File 'plan.md' written with description."
    );
    assert_eq!(
        invoke_text(
            t,
            "file_memory_write",
            json!({"file_name": "b.md", "content": "b"})
        )
        .await,
        "File 'b.md' written."
    );
    assert_eq!(
        invoke_text(t, "file_memory_read", json!({"file_name": "plan.md"})).await,
        "step 1"
    );
    assert_eq!(
        store
            .read("s1/plan_description.md")
            .await
            .unwrap()
            .as_deref(),
        Some("The plan")
    );
    assert_eq!(
        store.read("s1/memories.md").await.unwrap().unwrap(),
        "# Memory Index\n\n- **b.md**\n- **plan.md**: The plan\n"
    );
    assert_eq!(
        invoke(t, "file_memory_ls", json!({})).await.unwrap(),
        json!([
            {"name": "plan.md", "type": "file", "description": "The plan"},
            {"name": "b.md", "type": "file", "description": null},
        ])
    );
    // The index is injected on the next run.
    let next = tools_for(&provider, "s1").await;
    assert!(next.messages[0]
        .text()
        .starts_with("The following is your memory index"));
    assert!(next.messages[0].text().contains("- **plan.md**: The plan"));
    // Deleting removes the sidecar and rebuilds the index.
    assert_eq!(
        invoke_text(t, "file_memory_delete", json!({"file_name": "plan.md"})).await,
        "File 'plan.md' deleted."
    );
    assert_eq!(store.read("s1/plan_description.md").await.unwrap(), None);
    assert_eq!(
        store.read("s1/memories.md").await.unwrap().unwrap(),
        "# Memory Index\n\n- **b.md**\n"
    );
    assert_eq!(
        invoke_text(t, "file_memory_delete", json!({"file_name": "plan.md"})).await,
        "File 'plan.md' not found."
    );
}

#[tokio::test]
async fn search_and_listing_hide_internal_files() {
    let provider = FileMemoryProvider::new(Arc::new(InMemoryAgentFileStore::new()));
    let c = tools_for(&provider, "s1").await;
    let t = &c.tools;
    invoke_text(
        t,
        "file_memory_write",
        json!({"file_name": "a.md", "content": "Memory Index mention", "description": "Memory"}),
    )
    .await;
    let hits = invoke(t, "file_memory_grep", json!({"regex_pattern": "memory"}))
        .await
        .unwrap();
    assert_eq!(hits.as_array().unwrap().len(), 1);
    assert_eq!(hits[0]["file_name"], "a.md");
    assert!(
        invoke_text(t, "file_memory_grep", json!({"regex_pattern": "("}))
            .await
            .starts_with("Could not search memory files:")
    );
    let names: Vec<String> = invoke(t, "file_memory_ls", json!({}))
        .await
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, vec!["a.md"]);
}

#[tokio::test]
async fn rejects_reserved_nested_and_invalid_names() {
    let provider = FileMemoryProvider::new(Arc::new(InMemoryAgentFileStore::new()));
    let c = tools_for(&provider, "s1").await;
    let t = &c.tools;
    assert_eq!(
        invoke_text(t, "file_memory_write", json!({"file_name": "memories.md", "content": "x"})).await,
        "Could not write file 'memories.md': the file name is reserved for internal use. Please choose a different file name."
    );
    assert_eq!(
        invoke_text(t, "file_memory_write", json!({"file_name": "notes/plan.md", "content": "x"})).await,
        "Could not write file 'notes/plan.md': memory files must not be written into a subdirectory. Please choose a flat file name without path separators."
    );
    assert_eq!(
        invoke_text(t, "file_memory_read", json!({"file_name": "notes/plan.md"})).await,
        "File 'notes/plan.md' not found."
    );
    assert!(
        invoke_text(t, "file_memory_read", json!({"file_name": "../x"}))
            .await
            .starts_with("Could not read file '../x': Invalid path")
    );
    assert!(invoke_text(
        t,
        "file_memory_replace",
        json!({"file_name": "x_description.md", "old_string": "a", "new_string": "b"})
    )
    .await
    .contains("reserved for internal use"));
}

#[tokio::test]
async fn replace_and_replace_lines_honour_expected_line() {
    let provider = FileMemoryProvider::new(Arc::new(InMemoryAgentFileStore::new()));
    let c = tools_for(&provider, "s1").await;
    let t = &c.tools;
    invoke_text(
        t,
        "file_memory_write",
        json!({"file_name": "m.md", "content": "a\nb\n"}),
    )
    .await;
    assert_eq!(
        invoke_text(
            t,
            "file_memory_replace",
            json!({"file_name": "m.md", "old_string": "a", "new_string": "A"})
        )
        .await,
        "Replaced 1 occurrence(s) in 'm.md'."
    );
    assert!(invoke_text(t, "file_memory_replace_lines", json!({"file_name": "m.md", "edits": [{"line_number": 2, "new_line": "x\n", "expected_line": "zzz"}]}))
        .await
        .contains("does not match the expected text"));
    assert_eq!(
        invoke_text(t, "file_memory_replace_lines", json!({"file_name": "m.md", "edits": [{"line_number": 2, "new_line": "B\n", "expected_line": "b"}]})).await,
        "Replaced 1 line(s) in 'm.md'."
    );
    assert_eq!(
        invoke_text(t, "file_memory_read", json!({"file_name": "m.md"})).await,
        "A\nB\n"
    );
}

#[tokio::test]
async fn scopes_isolate_and_colliding_ids_get_distinct_folders() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let provider = FileMemoryProvider::new(store.clone());
    let victim = tools_for(&provider, "victim").await;
    invoke_text(
        &victim.tools,
        "file_memory_write",
        json!({"file_name": "secret.md", "content": "s"}),
    )
    .await;
    for attacker in ["Victim", "victim/", "../victim", "~scope-76696374696d"] {
        let c = tools_for(&provider, attacker).await;
        assert_eq!(
            invoke_text(
                &c.tools,
                "file_memory_read",
                json!({"file_name": "secret.md"})
            )
            .await,
            "File 'secret.md' not found.",
            "{attacker}"
        );
        assert_eq!(
            invoke_text(
                &c.tools,
                "file_memory_delete",
                json!({"file_name": "secret.md"})
            )
            .await,
            "File 'secret.md' not found."
        );
    }
    assert_eq!(
        provider.resolve_working_folder(Some("session-1")).unwrap(),
        "session-1"
    );
    assert_ne!(
        provider.resolve_working_folder(Some("")).ok(),
        Some(String::new())
    );
    // A multi-segment scope becomes one folder; an explicit scope is shared.
    let scoped = FileMemoryProvider::new(store.clone()).scope("tenants/alice");
    let a = tools_for(&scoped, "s1").await;
    let b = tools_for(&scoped, "s2").await;
    invoke_text(
        &a.tools,
        "file_memory_write",
        json!({"file_name": "shared.md", "content": "hi"}),
    )
    .await;
    assert_eq!(
        invoke_text(
            &b.tools,
            "file_memory_read",
            json!({"file_name": "shared.md"})
        )
        .await,
        "hi"
    );
    let folder = scoped.resolve_working_folder(None).unwrap();
    assert!(!folder.contains('/'));
    assert!(store
        .file_exists(&format!("{folder}/shared.md"))
        .await
        .unwrap());
}

#[tokio::test]
async fn fails_closed_without_a_scope() {
    let provider = FileMemoryProvider::new(Arc::new(InMemoryAgentFileStore::new()));
    let mut c = SessionContext::new(vec![]);
    assert!(provider.before_run(&mut c).await.is_err());
}

#[tokio::test]
async fn index_caps_entries() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let provider = FileMemoryProvider::new(store.clone());
    let c = tools_for(&provider, "s").await;
    for i in 0..(MAX_INDEX_ENTRIES + 3) {
        invoke_text(
            &c.tools,
            "file_memory_write",
            json!({"file_name": format!("f{i:03}.md"), "content": "x"}),
        )
        .await;
    }
    let index = store.read("s/memories.md").await.unwrap().unwrap();
    assert_eq!(
        index.lines().filter(|l| l.starts_with("- **")).count(),
        MAX_INDEX_ENTRIES
    );
}
