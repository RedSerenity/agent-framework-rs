//! `MemoryContextProvider` / `MemoryFileStore` — modeled on upstream
//! `test_harness_memory.py`.

mod common;

use std::sync::Arc;

use agent_framework_core::agent::{Agent, SupportsAgentRun};
use agent_framework_core::memory::ContextProvider;
use agent_framework_core::session::{AgentSession, SessionState};
use agent_framework_core::types::{ChatResponse, Content, FunctionResultContent, Message, Role};
use agent_framework_harness::memory::{
    extract_keywords, format_timestamp, parse_timestamp, select_recent_turn_messages,
    slugify_topic, MemoryContextProvider, MemoryFileStore, MemoryIndexEntry, MemoryStore,
    MemoryTopicRecord,
};
use agent_framework_harness::util::SessionRef;
use common::{call, ctx, invoke_text, MockClient, TempDir};
use serde_json::json;

fn owner_session(owner: &str) -> SessionRef {
    let state = SessionState::new();
    state.insert("owner", json!(owner));
    SessionRef {
        session_id: "s1".into(),
        state,
    }
}

#[test]
fn helpers_match_upstream() {
    assert_eq!(slugify_topic("  User Preferences!! "), "user-preferences");
    assert_eq!(slugify_topic("???"), "memory-topic");
    assert_eq!(format_timestamp(0), "1970-01-01T00:00:00+00:00");
    assert_eq!(format_timestamp(1_700_000_000), "2023-11-14T22:13:20+00:00");
    assert_eq!(
        parse_timestamp("2023-11-14T22:13:20+00:00"),
        Some(1_700_000_000)
    );
    assert_eq!(
        parse_timestamp("2023-11-14T23:13:20+01:00"),
        Some(1_700_000_000)
    );
    assert_eq!(
        parse_timestamp("2023-11-14T22:13:20.5Z"),
        Some(1_700_000_000)
    );
    assert_eq!(parse_timestamp("garbage"), None);
    let keywords = extract_keywords(&["Привет мир, 東京 is great a_b x".to_string()]);
    for expected in ["привет", "мир", "東京", "is", "great", "a_b"] {
        assert!(keywords.contains(expected), "{expected}: {keywords:?}");
    }
    assert!(!keywords.contains("x"));
}

#[test]
fn topic_record_round_trips_through_markdown_even_with_section_markers() {
    let record = MemoryTopicRecord::new(
        "Deploy Notes",
        None,
        "# not a heading\nsecond",
        &[
            "## Memories looks like a section".into(),
            "plain".into(),
            "PLAIN".into(),
        ],
        "2024-01-01T00:00:00+00:00",
        &["s1".into(), "s2".into(), "s1".into()],
    )
    .unwrap();
    assert_eq!(record.slug, "deploy-notes");
    assert_eq!(record.memories.len(), 2, "deduped case-insensitively");
    assert_eq!(record.session_ids, vec!["s1", "s2"]);
    let markdown = record.to_markdown();
    assert!(markdown.contains("\\# not a heading"));
    let parsed = MemoryTopicRecord::from_markdown(&markdown, None).unwrap();
    assert_eq!(parsed, record);
    assert!(MemoryTopicRecord::from_markdown("no heading", None).is_err());
    let entry = MemoryIndexEntry::from_topic_record(&record);
    assert_eq!(entry.to_pointer_line(20), "- [Deploy Notes](...");
}

#[test]
fn file_store_writes_topics_index_and_state() {
    let dir = TempDir::new("memstore");
    let store = MemoryFileStore::new(dir.path(), "owner");
    let session = owner_session("alice");
    let root = store.memory_root(&session, "memory").unwrap();
    assert_eq!(root, dir.path().join("memory").join("alice").join("memory"));
    assert!(store.list_topics(&session, "memory").unwrap().is_empty());
    assert!(!root.exists(), "pure reads do not create directories");
    let record = MemoryTopicRecord::new(
        "Tea",
        None,
        "",
        &["Likes green tea".into()],
        "2024-01-01T00:00:00+00:00",
        &["s1".into()],
    )
    .unwrap();
    store.write_topic(&session, &record, "memory").unwrap();
    assert!(root.join("topics").join("tea.md").is_file());
    let entries = store.rebuild_index(&session, "memory", 200, 150).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(
        std::fs::read_to_string(root.join("MEMORY.md")).unwrap(),
        "# MEMORY\n\n- [Tea](topics/tea.md): Likes green tea\n"
    );
    let state = store.read_state(&session, "memory").unwrap();
    assert_eq!(state["sessions_since_consolidation"], json!([]));
    assert!(store.delete_topic(&session, "memory", "Tea").unwrap());
    assert!(!store.delete_topic(&session, "memory", "Tea").unwrap());
    assert_eq!(
        store
            .get_index_text(&session, "memory", 200, 150, None)
            .unwrap(),
        "# MEMORY\n\n- none yet"
    );
}

#[test]
fn file_store_isolates_owners_and_rejects_traversal() {
    let dir = TempDir::new("memowners");
    let store = MemoryFileStore::new(dir.path(), "owner");
    for bad in ["../evil", "/abs", "a/../../b"] {
        assert!(
            store.memory_root(&owner_session(bad), "memory").is_err(),
            "{bad}"
        );
    }
    let a = store.memory_root(&owner_session("a/b"), "memory").unwrap();
    let b = store.memory_root(&owner_session("a_b"), "memory").unwrap();
    let c = store.memory_root(&owner_session("A"), "memory").unwrap();
    let d = store.memory_root(&owner_session("é"), "memory").unwrap();
    assert_ne!(a, b);
    assert_ne!(c, store.memory_root(&owner_session("a"), "memory").unwrap());
    for root in [&a, &b, &c, &d] {
        assert!(root.starts_with(dir.path()));
        assert_eq!(
            root.strip_prefix(dir.path()).unwrap().components().count(),
            3
        );
    }
    let other_source = store.memory_root(&owner_session("a_b"), "other").unwrap();
    assert_ne!(b, other_source);
    let missing = SessionRef {
        session_id: "s".into(),
        state: SessionState::new(),
    };
    assert!(store.memory_root(&missing, "memory").is_err());
}

#[test]
fn recent_turns_selection_can_skip_tool_groups() {
    let messages = vec![
        Message::user("q1"),
        Message::assistant("a1"),
        Message::user("q2"),
        Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call("c", "t", json!({})))],
        ),
        Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                "c",
                Some(json!("r")),
            ))],
        ),
        Message::assistant("a2"),
    ];
    assert!(select_recent_turn_messages(&messages, 0, true).is_empty());
    let last = select_recent_turn_messages(&messages, 1, true);
    assert_eq!(last.len(), 4);
    let no_tools = select_recent_turn_messages(&messages, 1, false);
    assert_eq!(
        no_tools.iter().map(|m| m.text()).collect::<Vec<_>>(),
        vec!["q2", "a2"]
    );
    assert_eq!(select_recent_turn_messages(&messages, 5, true).len(), 6);
}

#[tokio::test]
async fn provider_tools_and_injection() {
    let dir = TempDir::new("memprov");
    let store: Arc<dyn MemoryStore> = Arc::new(MemoryFileStore::new(dir.path(), "owner"));
    let provider = MemoryContextProvider::new(store.clone());
    let session = owner_session("alice");
    let mut c = ctx("s1", &session.state, vec![Message::user("hello")]);
    provider.before_run(&mut c).await.unwrap();
    let names: Vec<&str> = c.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "list_memory_topics",
            "read_memory_topic",
            "write_memory",
            "delete_memory_topic",
            "search_memory_transcripts",
            "consolidate_memories"
        ]
    );
    assert!(c
        .instructions
        .as_deref()
        .unwrap()
        .starts_with("Use MEMORY.md as the always-loaded table of contents"));
    let injected = c.messages.last().unwrap().text();
    assert_eq!(
        injected,
        "## Memory\nUse MEMORY.md and the loaded topic files when they are relevant.\n\n### MEMORY.md\n# MEMORY\n\n- none yet\n\n### Auto-loaded topic files\n- none auto-loaded for this turn"
    );
    let written = invoke_text(
        &c.tools,
        "write_memory",
        json!({"topic": "Tea", "memory": " Likes  green tea "}),
    )
    .await;
    assert!(
        written.contains("\"memories\": [\"Likes green tea\"]"),
        "{written}"
    );
    assert!(invoke_text(&c.tools, "list_memory_topics", json!({}))
        .await
        .contains("\"slug\": \"tea\""));
    assert!(
        invoke_text(&c.tools, "read_memory_topic", json!({"topic": "tea"}))
            .await
            .starts_with("# Tea")
    );
    // A matching keyword auto-loads the topic and marks cross-session origins.
    let other = SessionRef {
        session_id: "s2".into(),
        state: session.state.clone(),
    };
    let mut c2 = ctx(
        &other.session_id,
        &other.state,
        vec![Message::user("what tea do I like?")],
    );
    provider.before_run(&mut c2).await.unwrap();
    let memory_message = c2.messages.last().unwrap();
    assert!(memory_message.text().contains("### topics/tea.md\n# Tea"));
    assert_eq!(
        memory_message.additional_properties["origin_session_ids"],
        json!(["s1"])
    );
    assert_eq!(
        invoke_text(&c.tools, "delete_memory_topic", json!({"topic": "Tea"})).await,
        "Deleted memory topic 'Tea'."
    );
    assert_eq!(
        invoke_text(&c.tools, "consolidate_memories", json!({})).await,
        "{\"consolidated_topics\": 0}"
    );
}

#[tokio::test]
async fn end_to_end_archives_extracts_and_reloads_recent_turns() {
    let dir = TempDir::new("meme2e");
    let store: Arc<dyn MemoryStore> = Arc::new(MemoryFileStore::new(dir.path(), "owner"));
    let extractor = MockClient::new(vec![ChatResponse::from_text(
        "```json\n{\"memories\": [{\"topic\": \"Coffee\", \"memory\": \"Prefers espresso\"}, {\"bad\": 1}]}\n```",
    )]);
    let provider = Arc::new(
        MemoryContextProvider::new(store.clone())
            .client(Arc::new(extractor.clone()))
            .recent_turns(1),
    );
    let client = MockClient::new(vec![
        ChatResponse::from_text("Noted!"),
        ChatResponse::from_text("Espresso."),
    ]);
    let agent = Agent::builder(client.clone()).build();
    let mut session = AgentSession::new().with_context_providers(vec![provider.clone()]);
    session.state.insert("owner", json!("bob"));
    agent
        .run(vec![Message::user("I love espresso")], Some(&mut session))
        .await
        .unwrap();
    // The extractor saw the transcript delta.
    assert_eq!(
        extractor.calls()[0][1].text(),
        "USER: I love espresso\n\nASSISTANT: Noted!"
    );
    let sref = SessionRef::from_session(&session);
    let topics = store.list_topics(&sref, "memory").unwrap();
    assert_eq!(topics.len(), 1);
    assert_eq!(topics[0].memories, vec!["Prefers espresso"]);
    let transcript = provider
        .load_transcript(&sref, session.session_id())
        .unwrap();
    assert_eq!(transcript.len(), 2);
    let hits = provider
        .search_transcripts(&sref, "ESPRESSO", None, 20)
        .unwrap();
    assert_eq!(hits[0]["session_id"], json!(session.session_id()));
    assert_eq!(hits[0]["line_number"], 1);

    agent
        .run(vec![Message::user("which coffee?")], Some(&mut session))
        .await
        .unwrap();
    let sent: Vec<String> = client.last_messages().iter().map(|m| m.text()).collect();
    // Recent turn reloaded from the transcript, plus the auto-loaded topic.
    assert!(sent.iter().any(|t| t == "I love espresso"), "{sent:?}");
    assert!(sent.iter().any(|t| t.contains("### topics/coffee.md")));
    assert!(
        !sent.iter().any(|t| t == "Noted!") || sent.iter().filter(|t| *t == "Noted!").count() == 1
    );
}

#[tokio::test]
async fn consolidation_respects_thresholds_and_transient_failures() {
    let dir = TempDir::new("memcons");
    let store: Arc<dyn MemoryStore> = Arc::new(MemoryFileStore::new(dir.path(), "owner"));
    let session = owner_session("carol");
    let record = MemoryTopicRecord::new(
        "Tea",
        None,
        "",
        &["a".into(), "b".into()],
        "2024-01-01T00:00:00+00:00",
        &[],
    )
    .unwrap();
    store.write_topic(&session, &record, "memory").unwrap();
    // Not enough sessions yet: nothing happens.
    let provider = MemoryContextProvider::new(store.clone())
        .consolidation_min_sessions(2)
        .unwrap();
    assert_eq!(
        provider.run_consolidation(&session, false).await.unwrap(),
        0
    );
    let mut state = store.read_state(&session, "memory").unwrap();
    state.insert("sessions_since_consolidation".into(), json!(["s1", "s2"]));
    store.write_state(&session, &state, "memory").unwrap();
    // A failing consolidation client keeps the state for a retry.
    let failing = MockClient::new(vec![ChatResponse::from_text("not json")]);
    let provider = MemoryContextProvider::new(store.clone())
        .consolidation_min_sessions(2)
        .unwrap()
        .consolidation_client(Arc::new(failing));
    assert_eq!(
        provider.run_consolidation(&session, false).await.unwrap(),
        0
    );
    assert_eq!(
        store.read_state(&session, "memory").unwrap()["sessions_since_consolidation"],
        json!(["s1", "s2"])
    );
    // A good consolidation rewrites the topic and resets the window.
    let good = MockClient::new(vec![ChatResponse::from_text(
        r#"{"summary": "Tea facts", "memories": ["a and b"]}"#,
    )]);
    let provider = MemoryContextProvider::new(store.clone())
        .consolidation_min_sessions(2)
        .unwrap()
        .consolidation_client(Arc::new(good.clone()));
    assert_eq!(
        provider.run_consolidation(&session, false).await.unwrap(),
        1
    );
    let topic = store.get_topic(&session, "memory", "tea").unwrap().unwrap();
    assert_eq!(
        (topic.summary.as_str(), topic.memories.clone()),
        ("Tea facts", vec!["a and b".to_string()])
    );
    let state = store.read_state(&session, "memory").unwrap();
    assert_eq!(state["sessions_since_consolidation"], json!([]));
    assert!(state["last_consolidated_at"].is_string());
    // Within the interval, no further automatic consolidation.
    let mut state = state;
    state.insert("sessions_since_consolidation".into(), json!(["x", "y"]));
    store.write_state(&session, &state, "memory").unwrap();
    assert_eq!(
        provider.run_consolidation(&session, false).await.unwrap(),
        0
    );
    assert_eq!(good.calls().len(), 1);
}
