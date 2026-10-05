//! Hermetic loopback tests for [`QdrantStore`] / [`QdrantCollection`]
//! against a hand-rolled fake Qdrant REST server on a bare
//! `std::net::TcpListener` — no external process, no real network.
//!
//! The unit tests next to the connector pin what it *builds* (filters,
//! points, request bodies); these pin what goes on the wire and what it
//! makes of the replies: routes and methods, the `api-key` header,
//! `wait=true`, batching, the create-conflict readiness retry, out-of-order
//! retrieve results, and error statuses.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::vectors::{
    DistanceFunction, Filter, VectorCollection, VectorSearchOptions, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_qdrant::QdrantStore;
use serde_json::{json, Value};

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// `(request line, header block, body)`.
type Recorded = (String, String, Vec<u8>);

fn read_http_request(stream: &mut TcpStream) -> Recorded {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read headers");
        if n == 0 {
            break buf.len();
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
    };
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let line = headers.lines().next().unwrap_or_default().to_string();
    let length: usize = headers
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    let start = header_end + 4;
    while buf.len() < start + length {
        let n = stream.read(&mut chunk).expect("read body");
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = buf.get(start..start + length).unwrap_or(&[]).to_vec();
    (line, headers, body)
}

fn write_response(stream: &mut TcpStream, status: u16, body: &str) {
    let response = format!(
        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).expect("write");
    stream.flush().expect("flush");
}

type Route = dyn Fn(&str, &Value) -> (u16, Value) + Send + Sync;

struct FakeQdrant {
    url: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FakeQdrant {
    fn start(route: impl Fn(&str, &Value) -> (u16, Value) + Send + Sync + 'static) -> Self {
        let route: Arc<Route> = Arc::new(route);
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (requests_bg, stop_bg) = (requests.clone(), stop.clone());
        let handle = std::thread::spawn(move || {
            while !stop_bg.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).expect("blocking");
                        let request = read_http_request(&mut stream);
                        let body: Value = serde_json::from_slice(&request.2).unwrap_or(Value::Null);
                        let (status, reply) = route(&request.0, &body);
                        requests_bg.lock().unwrap().push(request);
                        write_response(&mut stream, status, &reply.to_string());
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(e) => panic!("accept: {e}"),
                }
            }
        });
        Self {
            url,
            requests,
            stop,
            handle: Some(handle),
        }
    }

    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    fn lines(&self) -> Vec<String> {
        self.requests().into_iter().map(|r| r.0).collect()
    }
}

impl Drop for FakeQdrant {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn ok(result: Value) -> (u16, Value) {
    (200, json!({"result": result, "status": "ok", "time": 0.0}))
}

fn completed() -> (u16, Value) {
    ok(json!({"operation_id": 1, "status": "completed"}))
}

fn definition() -> VectorStoreCollectionDefinition {
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("text")
            .with_type("str")
            .with_storage_name("body")
            .indexed(),
        VectorStoreField::vector("embedding", 3)
            .with_storage_name("dense")
            .with_distance_function(DistanceFunction::new(DistanceFunction::DOT_PROD)),
    ])
    .unwrap()
}

fn collection_info(index: Option<&str>) -> Value {
    let mut schema = json!({});
    if let Some(index) = index {
        schema = json!({"body": {"data_type": index, "points": 0}});
    }
    json!({
        "status": "green",
        "config": {
            "params": {"vectors": {"dense": {"size": 3, "distance": "Dot"}}},
            "hnsw_config": {"m": 16},
        },
        "payload_schema": schema,
    })
}

#[tokio::test]
async fn ensure_creates_the_collection_and_payload_index_with_the_api_key() {
    let server = FakeQdrant::start(|line, _| {
        if line.starts_with("GET /collections/docs/exists") {
            ok(json!({"exists": false}))
        } else if line.starts_with("PUT /collections/docs ") {
            ok(json!(true))
        } else if line.starts_with("GET /collections/docs ") {
            ok(collection_info(None))
        } else if line.starts_with("PUT /collections/docs/index?wait=true") {
            completed()
        } else {
            (404, json!({"status": {"error": "unexpected"}}))
        }
    });
    let store = QdrantStore::new(&server.url)
        .unwrap()
        .with_api_key("secret");
    let docs = store.collection("docs", definition()).unwrap();
    docs.ensure_collection_exists().await.unwrap();

    let requests = server.requests();
    let lines: Vec<&str> = requests.iter().map(|r| r.0.as_str()).collect();
    assert_eq!(lines.len(), 4, "{lines:?}");
    for request in &requests {
        assert!(
            request.1.to_ascii_lowercase().contains("api-key: secret"),
            "{}",
            request.1
        );
    }
    let create: Value = serde_json::from_slice(&requests[1].2).unwrap();
    assert_eq!(
        create,
        json!({"vectors": {"dense": {"size": 3, "distance": "Dot"}}})
    );
    let index: Value = serde_json::from_slice(&requests[3].2).unwrap();
    assert_eq!(
        index,
        json!({"field_name": "body", "field_schema": "keyword"})
    );
}

#[tokio::test]
async fn a_create_conflict_waits_for_a_readable_schema() {
    let reads = Arc::new(AtomicUsize::new(0));
    let reads_bg = reads.clone();
    let server = FakeQdrant::start(move |line, _| {
        if line.starts_with("GET /collections/docs/exists") {
            ok(json!({"exists": false}))
        } else if line.starts_with("PUT /collections/docs ") {
            (
                409,
                json!({"status": {"error": "Collection `docs` already exists!"}}),
            )
        } else if line.starts_with("GET /collections/docs ") {
            if reads_bg.fetch_add(1, Ordering::SeqCst) < 2 {
                (
                    500,
                    json!({"status": {"error": "Service internal error: 0 of 0 read operations failed"}}),
                )
            } else {
                ok(collection_info(Some("keyword")))
            }
        } else {
            (404, json!({}))
        }
    });
    let docs = QdrantStore::new(&server.url)
        .unwrap()
        .collection("docs", definition())
        .unwrap();
    docs.ensure_collection_exists().await.unwrap();
    assert_eq!(reads.load(Ordering::SeqCst), 3);
    // The existing index matched, so none was created.
    assert!(!server.lines().iter().any(|l| l.contains("/index")));
}

#[tokio::test]
async fn readiness_errors_are_not_retried_without_a_conflict() {
    let server = FakeQdrant::start(|line, _| {
        if line.starts_with("GET /collections/docs/exists") {
            ok(json!({"exists": true}))
        } else {
            (
                500,
                json!({"status": {"error": "0 of 0 read operations failed"}}),
            )
        }
    });
    let docs = QdrantStore::new(&server.url)
        .unwrap()
        .collection("docs", definition())
        .unwrap();
    let err = docs.ensure_collection_exists().await.unwrap_err();
    assert_eq!(err.status(), Some(500));
    assert_eq!(server.lines().len(), 2);
}

#[tokio::test]
async fn existing_schema_mismatches_are_refused() {
    for info in [
        json!({"config": {"params": {"vectors": {"size": 3, "distance": "Dot"}}}}),
        json!({"config": {"params": {"vectors": {"dense": {"size": 4, "distance": "Dot"}}}}}),
        json!({"config": {"params": {"vectors": {"dense": {"size": 3, "distance": "Cosine"}}}}}),
        json!({"config": {"params": {"vectors": {"dense": {"size": 3, "distance": "Dot", "datatype": "uint8"}}}}}),
        json!({"config": {"params": {"vectors": {"dense": {"size": 3, "distance": "Dot",
               "multivector_config": {"comparator": "max_sim"}}}}}}),
        json!({"config": {"params": {"vectors": {"dense": {"size": 3, "distance": "Dot"}}}},
               "payload_schema": {"body": {"data_type": "integer"}}}),
    ] {
        let reply = info.clone();
        let server = FakeQdrant::start(move |line, _| {
            if line.starts_with("GET /collections/docs/exists") {
                ok(json!({"exists": true}))
            } else {
                ok(reply.clone())
            }
        });
        let docs = QdrantStore::new(&server.url)
            .unwrap()
            .collection("docs", definition())
            .unwrap();
        assert!(docs.ensure_collection_exists().await.is_err(), "{info}");
    }
}

#[tokio::test]
async fn upsert_batches_by_256_and_reports_partial_failure() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_bg = calls.clone();
    let server = FakeQdrant::start(move |line, _| {
        assert!(
            line.starts_with("PUT /collections/docs/points?wait=true"),
            "{line}"
        );
        if calls_bg.fetch_add(1, Ordering::SeqCst) == 1 {
            (400, json!({"status": {"error": "Wrong input: bad vector"}}))
        } else {
            completed()
        }
    });
    let docs = QdrantStore::new(&server.url)
        .unwrap()
        .collection("docs", definition())
        .unwrap();
    let records: Vec<Value> = (0..600u64)
        .map(|i| json!({"id": i, "text": "t", "embedding": [1.0, 0.0, 0.0]}))
        .collect();
    let err = docs.upsert(records).await.unwrap_err();
    assert!(err.to_string().contains("wrote 256/600"), "{err}");
    assert!(err.to_string().contains("bad vector"), "{err}");
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    let first: Value = serde_json::from_slice(&requests[0].2).unwrap();
    assert_eq!(first["points"].as_array().unwrap().len(), 256);
    assert_eq!(
        first["points"][0],
        json!({"id": 0, "vector": {"dense": [1.0, 0.0, 0.0]}, "payload": {"body": "t"}})
    );
}

#[tokio::test]
async fn an_invalid_record_stops_the_batch_before_any_request() {
    let server = FakeQdrant::start(|_, _| completed());
    let docs = QdrantStore::new(&server.url)
        .unwrap()
        .collection("docs", definition())
        .unwrap();
    let err = docs
        .upsert(vec![
            json!({"id": 1, "embedding": [1.0, 0.0, 0.0]}),
            json!({"id": "not-a-uuid", "embedding": [1.0, 0.0, 0.0]}),
        ])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("index 1"), "{err}");
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn an_incomplete_write_is_an_error() {
    let server = FakeQdrant::start(|_, _| ok(json!({"operation_id": 1, "status": "acknowledged"})));
    let docs = QdrantStore::new(&server.url)
        .unwrap()
        .collection("docs", definition())
        .unwrap();
    assert!(docs.delete(vec![json!(1)]).await.is_err());
}

#[tokio::test]
async fn get_aligns_out_of_order_results_with_the_keys() {
    let server = FakeQdrant::start(|line, body| {
        assert!(line.starts_with("POST /collections/docs/points "), "{line}");
        assert_eq!(body["ids"], json!([1, 2, 3]));
        assert_eq!(body["with_vector"], json!(false));
        ok(json!([
            {"id": 3, "payload": {"body": "three"}},
            {"id": 1, "payload": {"body": "one"}},
        ]))
    });
    let docs = QdrantStore::new(&server.url)
        .unwrap()
        .collection("docs", definition())
        .unwrap();
    let got = docs
        .get(vec![json!(1), json!(2), json!(3)], false)
        .await
        .unwrap();
    assert_eq!(
        got,
        vec![
            Some(json!({"id": 1, "text": "one"})),
            None,
            Some(json!({"id": 3, "text": "three"})),
        ]
    );
}

#[tokio::test]
async fn search_posts_a_query_and_maps_scored_points() {
    let server = FakeQdrant::start(|line, body| {
        assert!(
            line.starts_with("POST /collections/docs/points/query"),
            "{line}"
        );
        assert_eq!(body["using"], "dense");
        assert_eq!(body["limit"], 2);
        assert_eq!(body["offset"], 1);
        assert_eq!(
            body["filter"],
            json!({"must": [{"key": "body", "match": {"value": "x"}}]})
        );
        ok(json!({"points": [
            {"id": 7, "score": 0.9, "payload": {"body": "x"}, "vector": {"dense": [1.0, 0.0, 0.0]}},
        ]}))
    });
    let docs = QdrantStore::new(&server.url)
        .unwrap()
        .collection("docs", definition())
        .unwrap();
    let hits = docs
        .search(
            vec![1.0, 0.0, 0.0],
            &VectorSearchOptions::new(2)
                .with_skip(1)
                .with_include_vectors(true)
                .with_filter(Filter::eq("text", "x").unwrap()),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].record,
        json!({"id": 7, "text": "x", "embedding": [1.0, 0.0, 0.0]})
    );
    assert_eq!(hits[0].score, Some(0.9));
    assert_eq!(hits[0].score_kind, None);
}

#[tokio::test]
async fn store_lists_checks_and_deletes_collections() {
    let server = FakeQdrant::start(|line, _| {
        if line.starts_with("GET /collections ") {
            ok(json!({"collections": [{"name": "a"}, {"name": "b"}]}))
        } else if line.starts_with("GET /collections/a/exists") {
            ok(json!({"exists": true}))
        } else if line.starts_with("DELETE /collections/a ") {
            ok(json!(true))
        } else {
            (
                404,
                json!({"status": {"error": "Not found: Collection `zzz` doesn't exist!"}}),
            )
        }
    });
    let store = QdrantStore::new(&server.url).unwrap();
    assert_eq!(store.list_collection_names().await.unwrap(), ["a", "b"]);
    assert!(VectorStore::collection_exists(&store, "a").await.unwrap());
    store.ensure_collection_deleted("a").await.unwrap();
    let err = VectorStore::collection_exists(&store, "zzz")
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(404));
    assert!(err.to_string().contains("doesn't exist"), "{err}");
}
