//! Live tests for [`RedisHistoryProvider`] and [`ValkeyChatHistoryProvider`]
//! against a real server.
//!
//! Gated exactly like `redis_integration.rs`: `REDIS_URL` (or `VALKEY_URL`)
//! when set, otherwise a private `redis-server` (or `valkey-server`) spawned
//! on an ephemeral port, otherwise a printed skip. Valkey speaks the Redis
//! protocol, so the Valkey provider runs against whichever server is found.

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::types::Message;
use agent_framework_redis::{
    RedisChatMessageStore, RedisHistoryProvider, RedisKeyFormat, ValkeyChatHistoryProvider,
};
use uuid::Uuid;

enum ServerGuard {
    Spawned(Child, std::path::PathBuf),
    External,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        if let ServerGuard::Spawned(child, dir) = self {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn binary_available(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

async fn ready(url: &str) -> bool {
    for _ in 0..50 {
        if let Ok(store) = RedisChatMessageStore::new(url, None) {
            if store.ping().await {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

async fn test_server() -> Option<(String, ServerGuard)> {
    for var in ["REDIS_URL", "VALKEY_URL"] {
        if let Ok(url) = std::env::var(var) {
            return Some((url, ServerGuard::External));
        }
    }
    let Some(binary) = ["redis-server", "valkey-server"]
        .into_iter()
        .find(|b| binary_available(b))
    else {
        eprintln!("skipping live history test: no REDIS_URL/VALKEY_URL and no server binary");
        return None;
    };
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .expect("ephemeral port");
    let dir = std::env::temp_dir().join(format!("af-redis-history-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("server dir");
    let mut child = Command::new(binary)
        .args(["--port", &port.to_string(), "--bind", "127.0.0.1"])
        .args(["--save", "", "--appendonly", "no", "--dir"])
        .arg(&dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn server");
    let url = format!("redis://127.0.0.1:{port}/0");
    if !ready(&url).await {
        eprintln!("skipping live history test: spawned {binary} did not become ready");
        let _ = child.kill();
        let _ = child.wait();
        return None;
    }
    Some((url, ServerGuard::Spawned(child, dir)))
}

fn unique(prefix: &str) -> String {
    format!("{prefix}-{}", Uuid::new_v4())
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages.iter().map(Message::text).collect()
}

// region: RedisHistoryProvider

#[tokio::test]
async fn scoped_keys_isolate_applications_tenants_and_agents() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let session = unique("session");
    let build = |app: &str, tenant: Option<&str>, agent: Option<&str>| {
        let mut b = RedisHistoryProvider::builder()
            .redis_url(&url)
            .application_id(app)
            .session_id(&session);
        if let Some(t) = tenant {
            b = b.tenant_id(t);
        }
        if let Some(a) = agent {
            b = b.agent_id(a);
        }
        b.build().unwrap()
    };
    let app1 = build("app1", Some("t1"), None);
    let app2 = build("app2", Some("t1"), None);
    let tenant2 = build("app1", Some("t2"), None);
    let agent = build("app1", Some("t1"), Some("agent"));

    app1.save_messages(vec![Message::user("one")])
        .await
        .unwrap();
    assert_eq!(app1.get_messages().await.unwrap().len(), 1);
    for other in [&app2, &tenant2, &agent] {
        assert!(other.get_messages().await.unwrap().is_empty());
    }
    let client = redis::Client::open(url.as_str()).unwrap();
    let mut conn = client.get_multiplexed_async_connection().await.unwrap();
    let exists: bool = redis::cmd("EXISTS")
        .arg(app1.redis_key())
        .query_async(&mut conn)
        .await
        .unwrap();
    assert!(exists, "the scoped key is what is on the server");
    assert!(app1.redis_key().contains("|v2|t1|app1|~none|redis_memory|"));
    app1.clear().await.unwrap();
    assert_eq!(app1.message_count().await.unwrap(), 0);
}

#[tokio::test]
async fn borrowed_connection_trims_and_respects_store_flags() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let client = redis::Client::open(url.as_str()).unwrap();
    let conn = client.get_multiplexed_async_connection().await.unwrap();
    let provider = RedisHistoryProvider::builder()
        .connection(conn.clone())
        .application_id("app")
        .session_id(unique("s"))
        .max_messages(3)
        .store_inputs(false)
        .build()
        .unwrap();
    assert!(provider.uses_borrowed_connection());

    for i in 0..3 {
        provider
            .after_run(
                &[Message::user(format!("q{i}"))],
                &[Message::assistant(format!("a{i}"))],
                None,
            )
            .await
            .unwrap();
    }
    // Inputs were not stored.
    assert_eq!(
        texts(&provider.get_messages().await.unwrap()),
        ["a0", "a1", "a2"]
    );
    provider
        .after_run(&[], &[Message::assistant("a3")], None)
        .await
        .unwrap();
    // Trimmed to the last three.
    assert_eq!(
        texts(&provider.get_messages().await.unwrap()),
        ["a1", "a2", "a3"]
    );

    // A failed run stores nothing.
    let err = agent_framework_core::error::Error::service("boom");
    provider
        .after_run(
            &[Message::user("x")],
            &[Message::assistant("y")],
            Some(&err),
        )
        .await
        .unwrap();
    assert_eq!(provider.message_count().await.unwrap(), 3);

    let mut ctx = SessionContext::new(vec![Message::user("next")]);
    provider.before_run(&mut ctx).await.unwrap();
    assert_eq!(ctx.messages.len(), 3);

    provider.clear().await.unwrap();
    // The provider never closes a borrowed connection.
    drop(provider);
    let mut conn = conn;
    let pong: String = redis::cmd("PING").query_async(&mut conn).await.unwrap();
    assert_eq!(pong, "PONG");
}

#[tokio::test]
async fn save_messages_skips_a_replayed_prefix_and_legacy_keys_match_the_store() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = RedisHistoryProvider::builder()
        .redis_url(&url)
        .key_format(RedisKeyFormat::Legacy)
        .session_id(unique("legacy"))
        .build()
        .unwrap();
    let q1 = Message::user("q1");
    let a1 = Message::assistant("a1");
    provider
        .save_messages(vec![q1.clone(), a1.clone()])
        .await
        .unwrap();
    provider
        .save_messages(vec![q1, a1, Message::user("q2")])
        .await
        .unwrap();
    assert_eq!(
        texts(&provider.get_messages().await.unwrap()),
        ["q1", "a1", "q2"]
    );
    let store = RedisChatMessageStore::new(&url, Some(provider.session_id().to_string())).unwrap();
    assert_eq!(store.list_messages().await.unwrap().len(), 3);
    provider.clear().await.unwrap();
}

#[tokio::test]
async fn zero_retention_leaves_stored_history_alone() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let session = unique("zero");
    let writer = RedisHistoryProvider::builder()
        .redis_url(&url)
        .application_id("app")
        .session_id(&session)
        .build()
        .unwrap();
    writer
        .save_messages(vec![Message::user("kept")])
        .await
        .unwrap();
    let zero = RedisHistoryProvider::builder()
        .redis_url(&url)
        .application_id("app")
        .session_id(&session)
        .max_messages(0)
        .build()
        .unwrap();
    zero.save_messages(vec![Message::user("dropped")])
        .await
        .unwrap();
    assert_eq!(texts(&writer.get_messages().await.unwrap()), ["kept"]);
    writer.clear().await.unwrap();
}

// endregion

// region: ValkeyChatHistoryProvider

#[tokio::test]
async fn valkey_stores_trims_and_counts() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = ValkeyChatHistoryProvider::builder(unique("conv"))
        .url(&url)
        .max_messages(4)
        .build()
        .unwrap();
    for i in 0..3 {
        provider
            .after_run(
                &[Message::user(format!("q{i}"))],
                &[Message::assistant(format!("a{i}"))],
                None,
            )
            .await
            .unwrap();
    }
    assert_eq!(provider.message_count().await.unwrap(), 4);
    assert_eq!(
        texts(&provider.get_messages().await.unwrap()),
        ["q1", "a1", "q2", "a2"]
    );
    provider.clear_messages().await.unwrap();
    assert_eq!(provider.message_count().await.unwrap(), 0);
}

#[tokio::test]
async fn valkey_retrieves_the_tail_and_skips_malformed_entries() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let conversation = unique("conv");
    let provider = ValkeyChatHistoryProvider::builder(conversation.clone())
        .url(&url)
        .max_messages_to_retrieve(2)
        .build()
        .unwrap();
    provider
        .after_run(&[Message::user("q1")], &[Message::assistant("a1")], None)
        .await
        .unwrap();
    // Entries written by something else, two of them unreadable.
    let client = redis::Client::open(url.as_str()).unwrap();
    let mut conn = client.get_multiplexed_async_connection().await.unwrap();
    let _: () = redis::cmd("RPUSH")
        .arg(provider.key())
        .arg("not json")
        .arg("")
        .arg(serde_json::to_string(&Message::user("q2")).unwrap())
        .query_async(&mut conn)
        .await
        .unwrap();

    // The tail of two is `""` (skipped) and q2.
    assert_eq!(texts(&provider.get_messages().await.unwrap()), ["q2"]);

    let all = ValkeyChatHistoryProvider::builder(conversation.clone())
        .connection(conn.clone())
        .build()
        .unwrap();
    assert_eq!(
        texts(&all.get_messages().await.unwrap()),
        ["q1", "a1", "q2"]
    );
    assert_eq!(all.message_count().await.unwrap(), 5);

    let none = ValkeyChatHistoryProvider::builder(conversation)
        .url(&url)
        .max_messages_to_retrieve(0)
        .build()
        .unwrap();
    let mut ctx = SessionContext::new(vec![Message::user("hi")]);
    none.before_run(&mut ctx).await.unwrap();
    assert!(ctx.messages.is_empty());
    all.clear_messages().await.unwrap();
}

#[tokio::test]
async fn valkey_does_not_duplicate_a_replayed_transcript() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = ValkeyChatHistoryProvider::builder(unique("conv"))
        .url(&url)
        .build()
        .unwrap();
    let q1 = Message::user("q1");
    let a1 = Message::assistant("a1");
    provider
        .after_run(std::slice::from_ref(&q1), std::slice::from_ref(&a1), None)
        .await
        .unwrap();
    provider
        .after_run(
            &[q1, a1, Message::user("q2")],
            &[Message::assistant("a2")],
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        texts(&provider.get_messages().await.unwrap()),
        ["q1", "a1", "q2", "a2"]
    );
    provider.clear_messages().await.unwrap();
}

// endregion
