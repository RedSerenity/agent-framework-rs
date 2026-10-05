//! Live tests against a vector-enabled SQL Server / Azure SQL database.
//!
//! Modeled on upstream's `tests/sql_server/test_integration.py`. They run
//! only when `SQL_SERVER_TEST_CONNECTION_STRING` (upstream's variable)
//! names a deliberately designated test database with table-creation
//! permissions, and return immediately otherwise. Each test creates
//! uniquely named tables and drops only those.

use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterExpression, FilterGroup, VectorCollection, VectorSearchOptions,
    VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_sql_server::{is_committed_cleanup, SqlServerQuery, SqlServerStore};
use serde_json::{json, Value};

fn store() -> Option<SqlServerStore> {
    std::env::var("SQL_SERVER_TEST_CONNECTION_STRING")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|cs| SqlServerStore::new(cs).expect("valid SQL_SERVER_TEST_CONNECTION_STRING"))
}

fn unique(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

fn ids(records: &[Value]) -> Vec<String> {
    let mut out: Vec<String> = records.iter().map(|r| r["id"].to_string()).collect();
    out.sort();
    out
}

/// Runs without a server: the real TDS connector reports a refused
/// connection as an ordinary (not committed-cleanup) error.
#[tokio::test]
async fn an_unreachable_server_is_a_connection_error() {
    let store =
        SqlServerStore::new("Server=tcp:127.0.0.1,1;User ID=u;Password=p;Encrypt=no").unwrap();
    let error = store.list_collection_names().await.unwrap_err();
    assert!(error.to_string().contains("connection failed"), "{error}");
    assert!(!is_committed_cleanup(&error));
}

#[tokio::test]
async fn typed_lifecycle_crud_search_and_portable_filters() {
    let Some(store) = store() else { return };
    let name = unique("af_sql");
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("text").with_type("str").indexed(),
        VectorStoreField::data("when").with_type("datetime"),
        VectorStoreField::data("meta").with_type("dict"),
        VectorStoreField::vector("embedding", 3),
    ])
    .unwrap();
    let collection = store.collection(&name, definition).unwrap();
    collection.ensure_collection_exists().await.unwrap();
    collection.ensure_collection_exists().await.unwrap();
    assert!(store.collection_exists(&name).await.unwrap());
    assert!(store.list_collection_names().await.unwrap().contains(&name));
    let keys = collection
        .upsert(vec![
            json!({"id": "a", "text": "SQL vectors", "when": "2024-01-02T05:00:00+02:00", "meta": {"k": 1}, "embedding": [1, 0, 0]}),
            json!({"id": "b", "text": "travel", "embedding": [0, 1, 0]}),
            json!({"id": "a", "text": "SQL vectors!", "when": "2024-01-02T05:00:00+02:00", "meta": {"k": 2}, "embedding": [1, 0, 0]}),
        ])
        .await
        .unwrap();
    assert_eq!(keys, vec![json!("a"), json!("b"), json!("a")]);
    let got = collection
        .get(vec![json!("a"), json!("zzz")], true)
        .await
        .unwrap();
    let a = got[0].clone().unwrap();
    assert_eq!(a["text"], json!("SQL vectors!"));
    assert_eq!(a["when"], json!("2024-01-02T03:00:00Z"));
    assert_eq!(a["meta"], json!({"k": 2}));
    assert_eq!(a["embedding"], json!([1.0, 0.0, 0.0]));
    assert!(got[1].is_none());
    let hits = collection
        .search_with_threshold(
            vec![1.0, 0.0, 0.0],
            &VectorSearchOptions::new(3).with_filter(Filter::starts_with("text", "SQL").unwrap()),
            Some(0.5),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].score.unwrap().abs() < 1e-6);
    collection.delete(vec![json!("a")]).await.unwrap();
    assert_eq!(
        collection.get(vec![json!("a")], false).await.unwrap(),
        vec![None]
    );
    collection.ensure_collection_deleted().await.unwrap();
    assert!(!collection.collection_exists().await.unwrap());
    store.close().await;
}

#[tokio::test]
async fn generated_key_and_vector_metrics() {
    let Some(store) = store() else { return };
    for (distance, expected) in [
        ("cosine_distance", 0.0),
        ("cosine_similarity", 1.0),
        ("euclidean_distance", 0.0),
        ("dot_prod", 1.0),
        ("negative_dot_prod", -1.0),
    ] {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            VectorStoreField::vector("v", 2)
                .with_distance_function(DistanceFunction::new(distance)),
        ])
        .unwrap();
        let collection = store
            .collection_builder(unique("af_sql_metric"), definition)
            .auto_generated_key(true)
            .build()
            .unwrap();
        collection.ensure_collection_exists().await.unwrap();
        let keys = collection
            .upsert(vec![json!({"v": [1, 0]}), json!({"v": [0, 1]})])
            .await
            .unwrap();
        assert_eq!(keys, vec![json!(1), json!(2)]);
        let hits = collection
            .search(vec![1.0, 0.0], &VectorSearchOptions::new(2))
            .await
            .unwrap();
        assert_eq!(hits[0].record["id"], json!(1), "{distance}");
        assert!(
            (hits[0].score.unwrap() - expected).abs() < 1e-6,
            "{distance}"
        );
        collection.ensure_collection_deleted().await.unwrap();
    }
    store.close().await;
}

#[tokio::test]
async fn portable_scalar_filters_match_in_memory_semantics() {
    let Some(store) = store() else { return };
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("text").with_type("str"),
        VectorStoreField::data("number").with_type("int"),
        VectorStoreField::data("flag").with_type("bool"),
    ])
    .unwrap();
    let collection = store
        .collection(unique("af_sql_filters"), definition)
        .unwrap();
    collection.ensure_collection_exists().await.unwrap();
    collection
        .upsert(vec![
            json!({"id": 1, "text": "Apple", "number": 1, "flag": true}),
            json!({"id": 2, "text": "apple ", "number": 2, "flag": false}),
            json!({"id": 3, "text": "50% [off]", "number": 3, "flag": true}),
            json!({"id": 4, "text": null, "number": null, "flag": null}),
        ])
        .await
        .unwrap();
    let all = collection.query(&SqlServerQuery::new(100)).await.unwrap();
    let filters: Vec<FilterExpression> = vec![
        Filter::eq("text", "Apple").unwrap().into(),
        Filter::eq("text", "apple").unwrap().into(),
        Filter::ne("text", "Apple").unwrap().into(),
        Filter::eq("number", "1").unwrap().into(),
        Filter::gt("number", 1).unwrap().into(),
        Filter::between("number", 2, 3).unwrap().into(),
        Filter::any_of("number", vec![json!(1), json!(3)])
            .unwrap()
            .into(),
        Filter::none_of("number", vec![json!(1)]).unwrap().into(),
        Filter::eq("flag", false).unwrap().into(),
        Filter::contains_text("text", "% [").unwrap().into(),
        Filter::starts_with("text", "app").unwrap().into(),
        Filter::ends_with("text", "e ").unwrap().into(),
        Filter::is_null("number").unwrap().into(),
        FilterGroup::not(Filter::gt("number", 1).unwrap().into()).unwrap(),
    ];
    let resolve = |name: &str| Some(name.to_string());
    for filter in filters {
        let expected: Vec<Value> = all
            .iter()
            .filter(|r| filter.matches(r, &resolve).unwrap())
            .cloned()
            .collect();
        let actual = collection
            .query(&SqlServerQuery::new(100).with_filter(filter.clone()))
            .await
            .unwrap();
        assert_eq!(ids(&actual), ids(&expected), "{filter:?}");
    }
    collection.ensure_collection_deleted().await.unwrap();
    store.close().await;
}

#[tokio::test]
async fn a_failed_batch_rolls_back_without_losing_a_prior_committed_write() {
    let Some(store) = store() else { return };
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("text").with_type("str"),
    ])
    .unwrap();
    let collection = store
        .collection_builder(unique("af_sql_atomic"), definition)
        .auto_generated_key(true)
        .build()
        .unwrap();
    collection.ensure_collection_exists().await.unwrap();
    collection
        .upsert(vec![json!({"text": "prior"})])
        .await
        .unwrap();
    // The second record names an IDENTITY key that does not exist: the
    // whole batch, including the first insert, rolls back.
    assert!(collection
        .upsert(vec![
            json!({"text": "new"}),
            json!({"id": 99, "text": "bad"})
        ])
        .await
        .is_err());
    let all = collection.query(&SqlServerQuery::new(10)).await.unwrap();
    assert_eq!(all, vec![json!({"id": 1, "text": "prior"})]);
    collection.ensure_collection_deleted().await.unwrap();
    store.close().await;
}
