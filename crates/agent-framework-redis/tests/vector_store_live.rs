//! Live tests for [`RedisVectorStore`] against a Redis server with Redis
//! Search (and RedisJSON) loaded — Redis Stack 7.4+ or Redis 8.
//!
//! Gated on `REDIS_STACK_URL`: when it is unset, or the server it names has
//! no RediSearch module, every test prints a skip notice and passes, so CI
//! without Redis Stack stays green. Run locally with e.g.
//!
//! ```text
//! redis-stack-server --port 6391 &
//! REDIS_STACK_URL=redis://127.0.0.1:6391 cargo test -p agent-framework-redis --test vector_store_live
//! ```
//!
//! Each test uses a fresh UUID namespace and drops its collections, so tests
//! can share one server.

use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterExpression, FilterGroup, VectorCollection, VectorSearchOptions,
    VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_redis::{RedisStorageType, RedisVectorStore};
use serde_json::{json, Value};

async fn stack_url() -> Option<String> {
    let Ok(url) = std::env::var("REDIS_STACK_URL") else {
        eprintln!("skipping live Redis vector test: REDIS_STACK_URL is not set");
        return None;
    };
    let client = redis::Client::open(url.as_str()).ok()?;
    let mut conn = client.get_multiplexed_async_connection().await.ok()?;
    if redis::cmd("FT._LIST")
        .query_async::<redis::Value>(&mut conn)
        .await
        .is_err()
    {
        eprintln!("skipping live Redis vector test: {url} has no RediSearch module");
        return None;
    }
    Some(url)
}

fn store(url: &str, storage: RedisStorageType) -> RedisVectorStore {
    RedisVectorStore::new(url)
        .unwrap()
        .with_namespace(format!("test-{}", uuid::Uuid::new_v4()))
        .unwrap()
        .with_storage_type(storage)
}

fn definition() -> VectorStoreCollectionDefinition {
    definition_with(3, DistanceFunction::COSINE_DISTANCE)
}

fn definition_with(dimensions: usize, distance: &str) -> VectorStoreCollectionDefinition {
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id"),
        VectorStoreField::data("title").indexed(),
        VectorStoreField::data("year").with_type("int").indexed(),
        VectorStoreField::data("price").with_type("float").indexed(),
        VectorStoreField::data("flag").with_type("bool").indexed(),
        VectorStoreField::data("tags").with_type("list").indexed(),
        VectorStoreField::data("meta").with_type("dict"),
        VectorStoreField::vector("embedding", dimensions)
            .with_storage_name("vec")
            .with_distance_function(DistanceFunction::new(distance)),
    ])
    .unwrap()
}

fn records() -> Vec<Value> {
    vec![
        json!({"id": "a", "title": "alpha", "year": 2020, "price": 1.5, "flag": true,
               "tags": ["x", "y"], "meta": {"n": 1}, "embedding": [1.0, 0.0, 0.0]}),
        json!({"id": "b", "title": "beta gamma", "year": 2021, "price": 2.5, "flag": false,
               "tags": ["y"], "embedding": [0.9, 0.1, 0.0]}),
        json!({"id": "c", "title": "x' OR true %_*", "year": 2022, "flag": true,
               "tags": ["z"], "embedding": [0.0, 1.0, 0.0]}),
        json!({"id": "d", "embedding": [0.0, 0.0, 1.0]}),
    ]
}

async fn ids(collection: &dyn VectorCollection, filter: Option<FilterExpression>) -> Vec<String> {
    let mut options = VectorSearchOptions::new(10);
    options.filter = filter;
    let mut out: Vec<String> = collection
        .search(vec![1.0, 1.0, 1.0], &options)
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.record["id"].as_str().unwrap().to_string())
        .collect();
    out.sort();
    out
}

async fn lifecycle_crud_search_and_filters(storage: RedisStorageType) {
    let Some(url) = stack_url().await else {
        return;
    };
    let store = store(&url, storage);
    let docs = store.collection("docs", definition()).unwrap();

    assert!(!docs.collection_exists().await.unwrap());
    assert!(docs.upsert(records()).await.is_err(), "no index yet");
    docs.ensure_collection_exists().await.unwrap();
    // Idempotent, and validates the existing schema.
    docs.ensure_collection_exists().await.unwrap();
    assert!(docs.collection_exists().await.unwrap());
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        vec!["docs".to_string()]
    );

    let keys = docs.upsert(records()).await.unwrap();
    assert_eq!(keys, vec![json!("a"), json!("b"), json!("c"), json!("d")]);

    let got = docs
        .get(vec![json!("b"), json!("missing"), json!("a")], true)
        .await
        .unwrap();
    assert!(got[1].is_none());
    let a = got[2].as_ref().unwrap();
    assert_eq!(a["title"], "alpha");
    assert_eq!(a["year"], 2020);
    assert_eq!(a["price"], 1.5);
    assert_eq!(a["flag"], true);
    assert_eq!(a["tags"], json!(["x", "y"]));
    assert_eq!(a["meta"], json!({"n": 1}));
    assert_eq!(a["embedding"], json!([1.0, 0.0, 0.0]));
    let without_vectors = docs.get(vec![json!("a")], false).await.unwrap();
    assert!(without_vectors[0]
        .as_ref()
        .unwrap()
        .get("embedding")
        .is_none());

    // Nearest first, native cosine distance (lower is closer).
    let hits = docs
        .search(vec![1.0, 0.0, 0.0], &VectorSearchOptions::new(2))
        .await
        .unwrap();
    assert_eq!(hits[0].record["id"], "a");
    assert_eq!(hits[1].record["id"], "b");
    assert!(hits[0].score.unwrap() < 1e-6);
    assert!(hits[0].score < hits[1].score);
    assert_eq!(hits[0].score_kind, None);
    let paged = docs
        .search(
            vec![1.0, 0.0, 0.0],
            &VectorSearchOptions::new(1).with_skip(1),
        )
        .await
        .unwrap();
    assert_eq!(paged[0].record["id"], "b");

    // Range search.
    let near = docs
        .search_within_distance(vec![1.0, 0.0, 0.0], 0.1, &VectorSearchOptions::new(10))
        .await
        .unwrap();
    let near_ids: Vec<_> = near.iter().map(|r| r.record["id"].clone()).collect();
    assert_eq!(near_ids, vec![json!("a"), json!("b")]);

    // Portable filters, executed natively.
    let f = |filter: Filter| Some(FilterExpression::from(filter));
    let c = &docs as &dyn VectorCollection;
    assert_eq!(
        ids(c, f(Filter::eq("title", "alpha").unwrap())).await,
        ["a"]
    );
    assert_eq!(
        ids(c, f(Filter::eq("title", "x' OR true %_*").unwrap())).await,
        ["c"]
    );
    assert_eq!(
        ids(c, f(Filter::eq("title", "beta").unwrap())).await,
        Vec::<String>::new()
    );
    assert_eq!(
        ids(c, f(Filter::ne("title", "alpha").unwrap())).await,
        ["b", "c"]
    );
    assert_eq!(
        ids(c, f(Filter::exists("price").unwrap())).await,
        ["a", "b"]
    );
    assert_eq!(
        ids(c, f(Filter::gt("year", 2020).unwrap())).await,
        ["b", "c"]
    );
    assert_eq!(
        ids(c, f(Filter::gte("year", 2021).unwrap())).await,
        ["b", "c"]
    );
    assert_eq!(ids(c, f(Filter::lt("price", 2).unwrap())).await, ["a"]);
    assert_eq!(
        ids(c, f(Filter::between("year", 2020, 2021).unwrap())).await,
        ["a", "b"]
    );
    assert_eq!(ids(c, f(Filter::eq("year", 2021.0).unwrap())).await, ["b"]);
    assert_eq!(
        ids(c, f(Filter::eq("flag", true).unwrap())).await,
        ["a", "c"]
    );
    assert_eq!(
        ids(c, f(Filter::eq("flag", 1).unwrap())).await,
        Vec::<String>::new()
    );
    assert_eq!(
        ids(
            c,
            f(Filter::any_of(
                "title",
                vec![json!("alpha"), json!(null), json!("beta gamma")]
            )
            .unwrap())
        )
        .await,
        ["a", "b"]
    );
    assert_eq!(
        ids(c, f(Filter::none_of("year", vec![json!(2020)]).unwrap())).await,
        ["b", "c"]
    );
    assert_eq!(
        ids(c, f(Filter::contains("tags", "y").unwrap())).await,
        ["a", "b"]
    );
    assert_eq!(
        ids(
            c,
            f(Filter::contains_all("tags", vec![json!("x"), json!("y")]).unwrap())
        )
        .await,
        ["a"]
    );
    assert_eq!(
        ids(
            c,
            f(Filter::contains_any("tags", vec![json!("x"), json!("z")]).unwrap())
        )
        .await,
        ["a", "c"]
    );
    assert_eq!(ids(c, f(Filter::eq("id", "d").unwrap())).await, ["d"]);
    let group = FilterGroup::and(vec![
        Filter::eq("flag", true).unwrap().into(),
        FilterGroup::not(Filter::eq("title", "alpha").unwrap().into()).unwrap(),
    ])
    .unwrap();
    assert_eq!(ids(c, Some(group)).await, ["c"]);
    let either = FilterGroup::or(vec![
        Filter::eq("id", "a").unwrap().into(),
        Filter::eq("year", 2022).unwrap().into(),
    ])
    .unwrap();
    assert_eq!(ids(c, Some(either)).await, ["a", "c"]);
    // A raw RediSearch clause is conjoined.
    let raw = docs
        .search(
            vec![1.0, 1.0, 1.0],
            &VectorSearchOptions::new(10)
                .with_filter(Filter::eq("flag", true).unwrap())
                .with_provider_filter("@year:[2022 2022]"),
        )
        .await
        .unwrap();
    assert_eq!(raw.len(), 1);
    assert_eq!(raw[0].record["id"], "c");

    // Upsert replaces the whole record.
    docs.upsert(vec![
        json!({"id": "a", "title": "renamed", "embedding": [1.0, 0.0, 0.0]}),
    ])
    .await
    .unwrap();
    let a = docs.get(vec![json!("a")], false).await.unwrap()[0]
        .clone()
        .unwrap();
    assert_eq!(a, json!({"id": "a", "title": "renamed"}));

    docs.delete(vec![json!("a"), json!("never-existed")])
        .await
        .unwrap();
    assert!(docs.get(vec![json!("a")], false).await.unwrap()[0].is_none());

    // A handle whose definition differs only in the vector's width or metric
    // refuses the existing index, as does one missing a field.
    for other in [
        definition_with(4, DistanceFunction::COSINE_DISTANCE),
        definition_with(3, DistanceFunction::EUCLIDEAN_SQUARED_DISTANCE),
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("embedding", 3).with_storage_name("vec"),
        ])
        .unwrap(),
    ] {
        let other = store.collection("docs", other).unwrap();
        let err = other.ensure_collection_exists().await.unwrap_err();
        assert!(err.to_string().contains("incompatible"), "{err}");
        assert!(other.get(vec![json!("b")], false).await.is_err());
    }

    docs.ensure_collection_deleted().await.unwrap();
    assert!(!docs.collection_exists().await.unwrap());
    // The documents went with the index.
    assert!(docs.get(vec![json!("b")], false).await.is_err());
    docs.ensure_collection_deleted().await.unwrap();
    assert!(store.list_collection_names().await.unwrap().is_empty());
}

#[tokio::test]
async fn hash_lifecycle_crud_search_and_filters() {
    lifecycle_crud_search_and_filters(RedisStorageType::Hash).await;
}

#[tokio::test]
async fn json_lifecycle_crud_search_and_filters() {
    lifecycle_crud_search_and_filters(RedisStorageType::Json).await;
}

#[tokio::test]
async fn json_storage_round_trips_nulls_and_empty_arrays() {
    let Some(url) = stack_url().await else {
        return;
    };
    let store = store(&url, RedisStorageType::Json);
    let docs = store.collection("nulls", definition()).unwrap();
    docs.ensure_collection_exists().await.unwrap();
    docs.upsert(vec![
        json!({"id": "n", "title": null, "year": null, "flag": null, "tags": [], "embedding": [1.0, 0.0, 0.0]}),
        json!({"id": "v", "title": "t", "year": 1, "flag": false, "tags": ["q"], "embedding": [0.0, 1.0, 0.0]}),
        json!({"id": "m", "embedding": [0.0, 0.0, 1.0]}),
    ])
    .await
    .unwrap();
    let n = docs.get(vec![json!("n")], false).await.unwrap()[0]
        .clone()
        .unwrap();
    assert_eq!(n["title"], Value::Null);
    assert_eq!(n["tags"], json!([]));

    let c = &docs as &dyn VectorCollection;
    let f = |filter: Filter| Some(FilterExpression::from(filter));
    assert_eq!(ids(c, f(Filter::is_null("year").unwrap())).await, ["n"]);
    assert_eq!(ids(c, f(Filter::is_not_null("year").unwrap())).await, ["v"]);
    assert_eq!(ids(c, f(Filter::is_null("flag").unwrap())).await, ["n"]);
    assert_eq!(ids(c, f(Filter::exists("year").unwrap())).await, ["n", "v"]);
    docs.ensure_collection_deleted().await.unwrap();
}

#[tokio::test]
async fn store_level_deletion_and_namespaces_are_isolated() {
    let Some(url) = stack_url().await else {
        return;
    };
    let one = store(&url, RedisStorageType::Hash);
    let two = store(&url, RedisStorageType::Hash);
    let a = one.collection("shared", definition()).unwrap();
    let b = two.collection("shared", definition()).unwrap();
    a.ensure_collection_exists().await.unwrap();
    b.ensure_collection_exists().await.unwrap();
    a.upsert(vec![
        json!({"id": "k", "title": "one", "embedding": [1.0, 0.0, 0.0]}),
    ])
    .await
    .unwrap();
    b.upsert(vec![
        json!({"id": "k", "title": "two", "embedding": [1.0, 0.0, 0.0]}),
    ])
    .await
    .unwrap();
    assert_eq!(
        a.get(vec![json!("k")], false).await.unwrap()[0]
            .as_ref()
            .unwrap()["title"],
        "one"
    );
    assert!(VectorStore::collection_exists(&one, "shared")
        .await
        .unwrap());
    one.ensure_collection_deleted("shared").await.unwrap();
    assert!(!VectorStore::collection_exists(&one, "shared")
        .await
        .unwrap());
    assert_eq!(
        b.get(vec![json!("k")], false).await.unwrap()[0]
            .as_ref()
            .unwrap()["title"],
        "two"
    );
    two.ensure_collection_deleted("shared").await.unwrap();
}
