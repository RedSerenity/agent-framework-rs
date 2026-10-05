//! Live tests against a disposable Qdrant server, gated on `QDRANT_TEST_URL`
//! (upstream's variable). Without it every test prints a skip and passes.
//!
//! ```text
//! ./qdrant &   # from https://github.com/qdrant/qdrant/releases
//! QDRANT_TEST_URL=http://127.0.0.1:6333 cargo test -p agent-framework-qdrant --test qdrant_live
//! ```
//!
//! The filter table mirrors upstream's `test_server_native_portable_semantics`.

use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterExpression, FilterGroup, IndexKind, VectorCollection,
    VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_qdrant::{QdrantCollection, QdrantStore};
use serde_json::{json, Value};

fn test_url() -> Option<String> {
    let url = std::env::var("QDRANT_TEST_URL").ok();
    if url.is_none() {
        eprintln!("skipping live Qdrant test: set QDRANT_TEST_URL to a disposable Qdrant server");
    }
    url
}

fn definition() -> VectorStoreCollectionDefinition {
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id")
            .with_type("int")
            .with_storage_name("point_key"),
        VectorStoreField::data("text")
            .with_type("str")
            .with_storage_name("body"),
        VectorStoreField::data("number")
            .with_type("float")
            .with_storage_name("price")
            .indexed(),
        VectorStoreField::data("integer").with_type("int"),
        VectorStoreField::data("flag").with_type("bool"),
        VectorStoreField::data("tags").with_type("list"),
        VectorStoreField::vector("embedding", 3)
            .with_storage_name("dense_text")
            .with_distance_function(DistanceFunction::new(DistanceFunction::DOT_PROD)),
        VectorStoreField::vector("image", 3)
            .with_storage_name("dense_image")
            .with_distance_function(DistanceFunction::new(DistanceFunction::DOT_PROD)),
    ])
    .unwrap()
}

fn unique_name() -> String {
    format!("af_qdrant_test_{}", uuid::Uuid::new_v4().simple())
}

async fn server_collection(url: &str) -> (QdrantStore, QdrantCollection) {
    let store = QdrantStore::new(url).unwrap();
    let collection = store.collection(unique_name(), definition()).unwrap();
    collection.ensure_collection_exists().await.unwrap();
    (store, collection)
}

/// Write raw payloads straight through the REST API, as upstream's fixture
/// does through the SDK, so the table can include shapes the connector's own
/// validation would refuse to produce.
async fn raw_upsert(url: &str, collection: &str, payloads: &[Value]) {
    let points: Vec<Value> = payloads
        .iter()
        .enumerate()
        .map(|(i, p)| json!({"id": i, "vector": {}, "payload": p}))
        .collect();
    let response = reqwest::Client::new()
        .put(format!("{url}/collections/{collection}/points?wait=true"))
        .json(&json!({ "points": points }))
        .send()
        .await
        .unwrap();
    assert!(
        response.status().is_success(),
        "{:?}",
        response.text().await
    );
}

#[tokio::test]
async fn server_native_portable_semantics() {
    let Some(url) = test_url() else {
        return;
    };
    let (_store, collection) = server_collection(&url).await;
    raw_upsert(
        &url,
        collection.name(),
        &[
            json!({}),
            json!({"body": null, "tags": null}),
            json!({"body": "", "tags": []}),
            json!({"body": "x' OR true %_*", "tags": ["one", "two"], "integer": 1, "price": 1.0, "flag": true}),
            json!({"body": "other", "tags": ["two", true, 1.0], "integer": 2, "price": 2.0, "flag": false}),
            json!({"body": "one", "tags": [null], "integer": i64::MAX}),
            json!({"body": "two", "integer": i64::MIN}),
        ],
    )
    .await;

    let f = |filter: Filter| FilterExpression::from(filter);
    let cases: Vec<(FilterExpression, Vec<u64>)> = vec![
        (f(Filter::exists("text").unwrap()), vec![1, 2, 3, 4, 5, 6]),
        (f(Filter::is_null("text").unwrap()), vec![1]),
        (f(Filter::is_not_null("text").unwrap()), vec![2, 3, 4, 5, 6]),
        (f(Filter::ne("text", "other").unwrap()), vec![1, 2, 3, 5, 6]),
        (f(Filter::eq("text", "x' OR true %_*").unwrap()), vec![3]),
        (
            f(Filter::any_of("text", vec![json!("other"), Value::Null]).unwrap()),
            vec![4],
        ),
        (
            f(Filter::none_of("text", vec![json!("other")]).unwrap()),
            vec![2, 3, 5, 6],
        ),
        (
            f(Filter::none_of("text", Vec::<Value>::new()).unwrap()),
            vec![2, 3, 4, 5, 6],
        ),
        (
            f(Filter::any_of("text", Vec::<Value>::new()).unwrap()),
            vec![],
        ),
        (f(Filter::exists("tags").unwrap()), vec![1, 2, 3, 4, 5]),
        (f(Filter::is_null("tags").unwrap()), vec![1]),
        (f(Filter::is_not_null("tags").unwrap()), vec![2, 3, 4, 5]),
        (f(Filter::contains("tags", "two").unwrap()), vec![3, 4]),
        (f(Filter::contains("tags", true).unwrap()), vec![4]),
        (f(Filter::contains("tags", 1).unwrap()), vec![4]),
        (
            f(Filter::contains_any("tags", vec![json!("one"), json!("two")]).unwrap()),
            vec![3, 4],
        ),
        (
            f(Filter::contains_any("tags", Vec::<Value>::new()).unwrap()),
            vec![],
        ),
        (
            f(Filter::contains_all("tags", vec![json!("one"), json!("two")]).unwrap()),
            vec![3],
        ),
        (
            f(Filter::contains_all("tags", Vec::<Value>::new()).unwrap()),
            vec![2, 3, 4, 5],
        ),
        (f(Filter::eq("integer", i64::MAX).unwrap()), vec![5]),
        (f(Filter::ne("integer", i64::MAX).unwrap()), vec![3, 4, 6]),
        (f(Filter::eq("integer", i64::MIN).unwrap()), vec![6]),
        (f(Filter::eq("integer", 1.0).unwrap()), vec![3]),
        (f(Filter::eq("integer", true).unwrap()), vec![]),
        (f(Filter::eq("flag", 1).unwrap()), vec![]),
        (f(Filter::eq("flag", true).unwrap()), vec![3]),
        (f(Filter::eq("number", 1).unwrap()), vec![3]),
        (f(Filter::gt("number", 1).unwrap()), vec![4]),
        (f(Filter::gte("number", 1).unwrap()), vec![3, 4]),
        (f(Filter::lt("number", 2).unwrap()), vec![3]),
        (f(Filter::lte("number", 2).unwrap()), vec![3, 4]),
        (f(Filter::between("number", 1, 2).unwrap()), vec![3, 4]),
        (
            f(Filter::gt("integer", 9_007_199_254_740_991i64).unwrap()),
            vec![5],
        ),
        (
            f(Filter::any_of("id", vec![json!(0), json!(3)]).unwrap()),
            vec![0, 3],
        ),
        (
            f(Filter::none_of("id", vec![json!(0), json!(3)]).unwrap()),
            vec![1, 2, 4, 5, 6],
        ),
        (f(Filter::is_null("id").unwrap()), vec![]),
        (
            FilterGroup::not(f(Filter::eq("text", "other").unwrap())).unwrap(),
            vec![0, 1, 2, 3, 5, 6],
        ),
        (
            FilterGroup::and(vec![
                f(Filter::exists("text").unwrap()),
                FilterGroup::or(vec![
                    f(Filter::is_null("text").unwrap()),
                    f(Filter::eq("text", "other").unwrap()),
                ])
                .unwrap(),
            ])
            .unwrap(),
            vec![1, 4],
        ),
        (
            FilterGroup::not(
                FilterGroup::and(vec![
                    f(Filter::gte("integer", 1).unwrap()),
                    f(Filter::eq("flag", true).unwrap()),
                ])
                .unwrap(),
            )
            .unwrap(),
            vec![0, 1, 2, 4, 5, 6],
        ),
    ];
    for (expression, expected) in cases {
        let records = collection
            .get_filtered(Some(&expression), 100, 0, false)
            .await
            .unwrap();
        let mut ids: Vec<u64> = records.iter().map(|r| r["id"].as_u64().unwrap()).collect();
        ids.sort_unstable();
        assert_eq!(ids, expected, "{expression:?}");
    }

    // A missing payload is not fabricated on read.
    let empty = collection.get(vec![json!(0)], false).await.unwrap();
    assert_eq!(empty[0], Some(json!({"id": 0})));

    // Paging through scroll.
    let page = collection.get_filtered(None, 2, 3, false).await.unwrap();
    assert_eq!(page.len(), 2);

    collection.ensure_collection_deleted().await.unwrap();
}

#[tokio::test]
async fn lifecycle_crud_vectors_and_search() {
    let Some(url) = test_url() else {
        return;
    };
    let (store, collection) = server_collection(&url).await;
    assert!(collection.collection_exists().await.unwrap());
    assert!(store
        .list_collection_names()
        .await
        .unwrap()
        .contains(&collection.name().to_string()));
    assert!(VectorStore::collection_exists(&store, collection.name())
        .await
        .unwrap());
    // Idempotent; validates the existing schema and payload index.
    collection.ensure_collection_exists().await.unwrap();

    let keys = collection
        .upsert(vec![
            json!({"id": 1, "text": "a", "number": 1.0, "tags": ["t", ["nested", 2]],
                   "embedding": [1.0, 0.0, 0.0], "image": [0.0, 1.0, 0.0]}),
            json!({"id": 2, "text": "b", "number": 2.0, "embedding": [0.5, 0.5, 0.0]}),
            json!({"id": 3, "text": "c", "number": 3.0, "embedding": [0.0, 0.0, 1.0]}),
        ])
        .await
        .unwrap();
    assert_eq!(keys, vec![json!(1), json!(2), json!(3)]);

    let got = collection
        .get(vec![json!(3), json!(99), json!(1)], true)
        .await
        .unwrap();
    assert_eq!(got[0].as_ref().unwrap()["text"], "c");
    assert!(got[1].is_none());
    let one = got[2].as_ref().unwrap();
    assert_eq!(one["tags"], json!(["t", ["nested", 2]]));
    assert_eq!(one["embedding"], json!([1.0, 0.0, 0.0]));
    assert_eq!(one["image"], json!([0.0, 1.0, 0.0]));
    // Point 2 has no image vector; it is reported as null, not fabricated.
    let two = collection.get(vec![json!(2)], true).await.unwrap()[0]
        .clone()
        .unwrap();
    assert_eq!(two["image"], Value::Null);

    // Dot product: higher is closer, in native units.
    let options = VectorSearchOptions::new(2).with_vector_field_name("embedding");
    let hits = collection
        .search(vec![1.0, 0.0, 0.0], &options)
        .await
        .unwrap();
    assert_eq!(hits[0].record["id"], 1);
    assert_eq!(hits[1].record["id"], 2);
    assert!((hits[0].score.unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(hits[0].score_kind, None);
    assert!(hits[0].record.get("embedding").is_none());

    let skipped = collection
        .search(vec![1.0, 0.0, 0.0], &options.clone().with_skip(1))
        .await
        .unwrap();
    assert_eq!(skipped[0].record["id"], 2);

    let filtered = collection
        .search(
            vec![1.0, 0.0, 0.0],
            &options
                .clone()
                .with_filter(Filter::gte("number", 2).unwrap())
                .with_include_vectors(true),
        )
        .await
        .unwrap();
    assert_eq!(filtered[0].record["id"], 2);
    assert_eq!(filtered[0].record["embedding"], json!([0.5, 0.5, 0.0]));

    let thresholded = collection
        .search_with_score_threshold(vec![1.0, 0.0, 0.0], 0.75, &options)
        .await
        .unwrap();
    assert_eq!(thresholded.len(), 1);

    let image_hits = collection
        .search(
            vec![0.0, 1.0, 0.0],
            &VectorSearchOptions::new(5).with_vector_field_name("image"),
        )
        .await
        .unwrap();
    assert_eq!(image_hits.len(), 1, "only point 1 has an image vector");

    collection.delete(vec![json!(1), json!(42)]).await.unwrap();
    assert!(collection.get(vec![json!(1)], false).await.unwrap()[0].is_none());

    // A mismatched handle refuses the existing collection.
    let mismatched = store
        .collection(
            collection.name(),
            VectorStoreCollectionDefinition::new(vec![
                VectorStoreField::key("id").with_type("int"),
                VectorStoreField::vector("embedding", 4).with_storage_name("dense_text"),
            ])
            .unwrap(),
        )
        .unwrap();
    assert!(mismatched.ensure_collection_exists().await.is_err());
    let flat = store
        .collection(
            collection.name(),
            VectorStoreCollectionDefinition::new(vec![
                VectorStoreField::key("id").with_type("int"),
                VectorStoreField::vector("embedding", 3)
                    .with_storage_name("dense_text")
                    .with_index_kind(IndexKind::new(IndexKind::FLAT))
                    .with_distance_function(DistanceFunction::new(DistanceFunction::DOT_PROD)),
            ])
            .unwrap(),
        )
        .unwrap();
    let err = flat.ensure_collection_exists().await.unwrap_err();
    assert!(err.to_string().contains("flat"), "{err}");

    store
        .ensure_collection_deleted(collection.name())
        .await
        .unwrap();
    assert!(!collection.collection_exists().await.unwrap());
    collection.ensure_collection_deleted().await.unwrap();
}

#[tokio::test]
async fn uuid_keys_and_large_batches() {
    let Some(url) = test_url() else {
        return;
    };
    let store = QdrantStore::new(&url).unwrap();
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("UUID"),
        VectorStoreField::data("n").with_type("int").indexed(),
        VectorStoreField::vector("v", 8),
    ])
    .unwrap();
    let collection = store.collection(unique_name(), definition).unwrap();
    collection.ensure_collection_exists().await.unwrap();

    // More than one 256-point batch.
    let ids: Vec<String> = (0..600)
        .map(|_| uuid::Uuid::new_v4().to_string().to_uppercase())
        .collect();
    let records: Vec<Value> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let mut v = vec![0.1f32; 8];
            v[i % 8] = 1.0;
            json!({"id": id, "n": i, "v": v})
        })
        .collect();
    let keys = collection.upsert(records).await.unwrap();
    // Keys come back canonical (lowercase).
    assert_eq!(keys[0], json!(ids[0].to_lowercase()));
    let got = collection
        .get(ids.iter().map(|i| json!(i)).collect(), false)
        .await
        .unwrap();
    assert!(got.iter().all(Option::is_some));
    assert_eq!(got[599].as_ref().unwrap()["n"], 599);

    let by_key = collection
        .get_filtered(
            Some(&Filter::eq("id", ids[5].as_str()).unwrap().into()),
            10,
            0,
            false,
        )
        .await
        .unwrap();
    assert_eq!(by_key.len(), 1);
    assert_eq!(by_key[0]["n"], 5);

    // Cosine default: a relevance score, higher is closer.
    let hits = collection
        .search(
            vec![1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            &VectorSearchOptions::new(3),
        )
        .await
        .unwrap();
    assert_eq!(
        hits[0].score_kind.as_deref(),
        Some(agent_framework_core::vectors::VectorSearchResult::SCORE_KIND_RELEVANCE)
    );
    assert!(hits[0].score >= hits[1].score);

    collection
        .delete(ids.iter().map(|i| json!(i)).collect())
        .await
        .unwrap();
    assert!(collection
        .get_filtered(None, 10, 0, false)
        .await
        .unwrap()
        .is_empty());
    collection.ensure_collection_deleted().await.unwrap();
}
