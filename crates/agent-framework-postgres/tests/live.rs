//! Live tests against a real PostgreSQL server with pgvector >= 0.8.
//!
//! Modeled on upstream's `tests/test_postgres_integration.py`. Every test
//! returns immediately unless `POSTGRES_TEST_URL` names a deliberately
//! designated test database (a conninfo or `postgresql://` URI) where the
//! `vector` extension is already installed. Each test creates uniquely
//! named tables and drops only those.

use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterExpression, FilterGroup, IndexKind, VectorCollection,
    VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_postgres::{
    deadpool_postgres, PostgresCollection, PostgresQuery, PostgresSearchOptions, PostgresStore,
    PostgresVectorOptions, PostgresVectorType,
};
use serde_json::{json, Value};

fn url() -> Option<String> {
    std::env::var("POSTGRES_TEST_URL")
        .ok()
        .filter(|v| !v.trim().is_empty())
}

fn unique(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

fn store() -> Option<PostgresStore> {
    url().map(|url| PostgresStore::new(url).expect("valid POSTGRES_TEST_URL"))
}

fn ids(records: &[Value]) -> Vec<String> {
    let mut out: Vec<String> = records.iter().map(|r| r["id"].to_string()).collect();
    out.sort();
    out
}

#[tokio::test]
async fn lifecycle_crud_multiple_vectors_and_aliases() {
    let Some(store) = store() else { return };
    let name = unique("af_lifecycle");
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id")
            .with_type("str")
            .with_storage_name("doc id"),
        VectorStoreField::data("text").with_type("str").indexed(),
        VectorStoreField::vector("first", 3).with_storage_name("first \"vec\""),
        VectorStoreField::vector("second", 2)
            .with_distance_function(DistanceFunction::new(DistanceFunction::EUCLIDEAN_DISTANCE)),
    ])
    .unwrap();
    let collection = store.collection(&name, definition).unwrap();
    assert!(!collection.collection_exists().await.unwrap());
    collection.ensure_collection_exists().await.unwrap();
    // Idempotent.
    collection.ensure_collection_exists().await.unwrap();
    assert!(collection.collection_exists().await.unwrap());
    assert!(store.collection_exists(&name).await.unwrap());
    assert!(store.list_collection_names().await.unwrap().contains(&name));

    let keys = collection
        .upsert(vec![
            json!({"id": "a", "text": "alpha", "first": [1, 0, 0], "second": [0, 1]}),
            json!({"id": "b", "text": "beta", "first": [0, 1, 0], "second": null}),
            json!({"id": "a", "text": "alpha again", "first": [1, 0, 0], "second": [1, 1]}),
        ])
        .await
        .unwrap();
    assert_eq!(keys, vec![json!("a"), json!("b"), json!("a")]);

    let got = collection
        .get(vec![json!("b"), json!("missing"), json!("a")], false)
        .await
        .unwrap();
    assert_eq!(got[0], Some(json!({"id": "b", "text": "beta"})));
    assert_eq!(got[1], None);
    assert_eq!(got[2], Some(json!({"id": "a", "text": "alpha again"})));

    let with_vectors = collection.get(vec![json!("a")], true).await.unwrap();
    assert_eq!(
        with_vectors[0],
        Some(
            json!({"id": "a", "text": "alpha again", "first": [1.0, 0.0, 0.0], "second": [1.0, 1.0]})
        )
    );

    let hits = collection
        .search(
            vec![0.0, 1.0],
            &VectorSearchOptions::new(5).with_vector_field_name("second"),
        )
        .await
        .unwrap();
    // `b` has no `second` vector and is never ranked.
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record["id"], json!("a"));
    assert!((hits[0].score.unwrap() - 1.0).abs() < 1e-6);

    collection
        .delete(vec![json!("a"), json!("zzz")])
        .await
        .unwrap();
    assert_eq!(
        collection.get(vec![json!("a")], false).await.unwrap(),
        vec![None]
    );

    collection.ensure_collection_deleted().await.unwrap();
    collection.ensure_collection_deleted().await.unwrap();
    assert!(!collection.collection_exists().await.unwrap());
    store.close().await;
}

fn filter_definition() -> VectorStoreCollectionDefinition {
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("text").with_type("str"),
        VectorStoreField::data("number").with_type("int"),
        VectorStoreField::data("ratio").with_type("float"),
        VectorStoreField::data("flag").with_type("bool"),
        VectorStoreField::data("tags").with_type("list"),
        VectorStoreField::vector("embedding", 2),
    ])
    .unwrap()
}

#[tokio::test]
async fn filters_match_portable_semantics_in_database() {
    let Some(store) = store() else { return };
    let collection = store
        .collection(unique("af_filters"), filter_definition())
        .unwrap();
    collection.ensure_collection_exists().await.unwrap();
    let records = vec![
        json!({"id": 1, "text": "Apple pie", "number": 1, "ratio": 0.5, "flag": true, "tags": ["a", "b"], "embedding": [1, 0]}),
        json!({"id": 2, "text": "apple_tart", "number": 2, "ratio": 1.5, "flag": false, "tags": ["b", 2], "embedding": [0, 1]}),
        json!({"id": 3, "text": "100% juice", "number": 3, "ratio": 2.0, "flag": true, "tags": [], "embedding": [1, 1]}),
        json!({"id": 4, "text": null, "number": null, "ratio": null, "flag": null, "tags": null, "embedding": null}),
        json!({"id": 5, "text": "Zebra", "number": -7, "ratio": -1.0, "flag": false, "tags": [null], "embedding": [0.5, 0.5]}),
    ];
    collection.upsert(records).await.unwrap();
    let all = collection
        .query(&PostgresQuery::new(100).with_include_vectors(false))
        .await
        .unwrap();
    assert_eq!(all.len(), 5);

    let filters: Vec<FilterExpression> = vec![
        Filter::eq("text", "Apple pie").unwrap().into(),
        Filter::eq("text", "apple pie").unwrap().into(),
        Filter::ne("text", "Apple pie").unwrap().into(),
        Filter::eq("number", 2).unwrap().into(),
        Filter::eq("number", 2.0).unwrap().into(),
        Filter::eq("number", "2").unwrap().into(),
        Filter::eq("ratio", 2).unwrap().into(),
        Filter::eq("flag", true).unwrap().into(),
        Filter::ne("flag", true).unwrap().into(),
        Filter::eq("flag", 1).unwrap().into(),
        Filter::gt("number", 1).unwrap().into(),
        Filter::gte("ratio", 1.5).unwrap().into(),
        Filter::lt("number", 3).unwrap().into(),
        Filter::lte("text", "Zebra").unwrap().into(),
        Filter::between("number", -7, 2).unwrap().into(),
        Filter::any_of("number", vec![json!(1), json!(3), json!(99)])
            .unwrap()
            .into(),
        Filter::none_of("number", vec![json!(1), json!(3)])
            .unwrap()
            .into(),
        Filter::is_null("text").unwrap().into(),
        Filter::is_not_null("tags").unwrap().into(),
        Filter::contains("tags", "b").unwrap().into(),
        Filter::contains("tags", 2).unwrap().into(),
        Filter::contains_any("tags", vec![json!("a"), json!(2)])
            .unwrap()
            .into(),
        Filter::contains_all("tags", vec![json!("a"), json!("b")])
            .unwrap()
            .into(),
        Filter::starts_with("text", "apple").unwrap().into(),
        Filter::starts_with("text", "apple_").unwrap().into(),
        Filter::ends_with("text", "pie").unwrap().into(),
        Filter::contains_text("text", "0%").unwrap().into(),
        Filter::contains_text("text", "e_t").unwrap().into(),
        Filter::exists("text").unwrap().into(),
        FilterGroup::not(Filter::gt("number", 1).unwrap().into()).unwrap(),
        FilterGroup::not(Filter::eq("text", "Zebra").unwrap().into()).unwrap(),
        FilterGroup::or(vec![
            Filter::eq("flag", true).unwrap().into(),
            Filter::is_null("number").unwrap().into(),
        ])
        .unwrap(),
        FilterGroup::and(vec![
            Filter::gt("ratio", 0).unwrap().into(),
            FilterGroup::not(Filter::contains("tags", "a").unwrap().into()).unwrap(),
        ])
        .unwrap(),
    ];
    let resolve = |name: &str| Some(name.to_string());
    for filter in filters {
        let expected: Vec<Value> = all
            .iter()
            .filter(|r| filter.matches(r, &resolve).unwrap())
            .cloned()
            .collect();
        let actual = collection
            .query(&PostgresQuery::new(100).with_filter(filter.clone()))
            .await
            .unwrap();
        assert_eq!(ids(&actual), ids(&expected), "{filter:?}");
    }

    // Ordering, nulls last, key tie-break, paging.
    let page = collection
        .query(&PostgresQuery::new(2).with_skip(1).order_by("flag", true))
        .await
        .unwrap();
    assert_eq!(
        page.iter().map(|r| r["id"].clone()).collect::<Vec<_>>(),
        vec![json!(5), json!(1)]
    );
    let page = collection
        .query(&PostgresQuery::new(5).order_by("ratio", false))
        .await
        .unwrap();
    assert_eq!(
        page.iter().map(|r| r["id"].clone()).collect::<Vec<_>>(),
        vec![json!(3), json!(2), json!(1), json!(5), json!(4)]
    );

    // A search shares the filter translation.
    let hits = collection
        .search(
            vec![1.0, 0.0],
            &VectorSearchOptions::new(5).with_filter(Filter::eq("flag", true).unwrap()),
        )
        .await
        .unwrap();
    assert_eq!(
        hits.iter()
            .map(|h| h.record["id"].clone())
            .collect::<Vec<_>>(),
        vec![json!(1), json!(3)]
    );
    collection.ensure_collection_deleted().await.unwrap();
    store.close().await;
}

#[tokio::test]
async fn metric_units_threshold_ranking_and_offset() {
    let Some(store) = store() else { return };
    for (distance, query, expected_order, expected_first, threshold, kept) in [
        ("cosine_distance", [1.0, 0.0], vec![1, 3, 2], 0.0, 0.5, 2),
        ("cosine_similarity", [1.0, 0.0], vec![1, 3, 2], 1.0, 0.5, 2),
        ("dot_prod", [2.0, 0.0], vec![3, 1, 2], 6.0, 2.5, 1),
        (
            "negative_dot_prod",
            [2.0, 0.0],
            vec![3, 1, 2],
            -6.0,
            -2.5,
            1,
        ),
        ("euclidean_distance", [1.0, 0.0], vec![1, 2, 3], 0.0, 1.5, 2),
        ("manhattan", [1.0, 0.0], vec![1, 2, 3], 0.0, 1.5, 1),
    ] {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            VectorStoreField::vector("v", 2)
                .with_distance_function(DistanceFunction::new(distance)),
        ])
        .unwrap();
        let collection = store.collection(unique("af_metric"), definition).unwrap();
        collection.ensure_collection_exists().await.unwrap();
        collection
            .upsert(vec![
                json!({"id": 1, "v": [1, 0]}),
                json!({"id": 2, "v": [0, 1]}),
                json!({"id": 3, "v": [3, 1]}),
            ])
            .await
            .unwrap();
        let hits = collection
            .search(query.to_vec(), &VectorSearchOptions::new(3))
            .await
            .unwrap();
        assert_eq!(
            hits.iter()
                .map(|h| h.record["id"].clone())
                .collect::<Vec<_>>(),
            expected_order.iter().map(|i| json!(i)).collect::<Vec<_>>(),
            "{distance}"
        );
        assert!(
            (hits[0].score.unwrap() - expected_first).abs() < 1e-6,
            "{distance}"
        );
        let thresholded = collection
            .search_with(
                query.to_vec(),
                &VectorSearchOptions::new(3),
                &PostgresSearchOptions::new().with_score_threshold(threshold),
            )
            .await
            .unwrap();
        assert_eq!(thresholded.len(), kept, "{distance}: {thresholded:?}");
        // Threshold applies before paging.
        let paged = collection
            .search_with(
                query.to_vec(),
                &VectorSearchOptions::new(3).with_skip(1),
                &PostgresSearchOptions::new().with_score_threshold(threshold),
            )
            .await
            .unwrap();
        assert_eq!(paged.len(), kept - 1, "{distance}");
        collection.ensure_collection_deleted().await.unwrap();
    }
    store.close().await;
}

#[tokio::test]
async fn generated_and_preserved_typed_keys() {
    let Some(store) = store() else { return };
    for (kind, explicit) in [
        ("int", json!(41)),
        ("UUID", json!("6F9619FF-8B86-D011-B42D-00C04FC964FF")),
        ("str", json!("explicit")),
    ] {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type(kind),
            VectorStoreField::data("text").with_type("str"),
        ])
        .unwrap();
        let collection = store
            .collection_builder(unique("af_keys"), definition)
            .auto_generated_key(true)
            .build()
            .unwrap();
        collection.ensure_collection_exists().await.unwrap();
        let keys = collection
            .upsert(vec![
                json!({"text": "generated"}),
                json!({"id": explicit, "text": "explicit"}),
            ])
            .await
            .unwrap();
        assert!(!keys[0].is_null(), "{kind}");
        let expected = match kind {
            "UUID" => json!("6f9619ff-8b86-d011-b42d-00c04fc964ff"),
            _ => explicit.clone(),
        };
        assert_eq!(keys[1], expected, "{kind}");
        let got = collection
            .get(vec![keys[0].clone(), explicit.clone()], false)
            .await
            .unwrap();
        assert_eq!(got[0].as_ref().unwrap()["text"], json!("generated"));
        assert_eq!(got[1].as_ref().unwrap()["text"], json!("explicit"));
        collection.ensure_collection_deleted().await.unwrap();
    }
    store.close().await;
}

#[tokio::test]
async fn identity_collision_fails_instead_of_overwriting() {
    let Some(store) = store() else { return };
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("text").with_type("str"),
    ])
    .unwrap();
    let collection = store
        .collection_builder(unique("af_identity"), definition)
        .auto_generated_key(true)
        .build()
        .unwrap();
    collection.ensure_collection_exists().await.unwrap();
    collection
        .upsert(vec![json!({"id": 1, "text": "explicit"})])
        .await
        .unwrap();
    // The identity sequence still starts at 1, so a generated insert
    // collides: it must fail rather than overwrite row 1.
    assert!(collection
        .upsert(vec![json!({"text": "generated"})])
        .await
        .is_err());
    let got = collection.get(vec![json!(1)], false).await.unwrap();
    assert_eq!(got[0].as_ref().unwrap()["text"], json!("explicit"));
    collection.ensure_collection_deleted().await.unwrap();
    store.close().await;
}

#[tokio::test]
async fn a_failed_batch_rolls_back_entirely() {
    let Some(store) = store() else { return };
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::vector("v", 2).with_type("float16"),
    ])
    .unwrap();
    let collection = store.collection(unique("af_atomic"), definition).unwrap();
    collection.ensure_collection_exists().await.unwrap();
    collection
        .upsert(vec![json!({"id": "prior", "v": [1, 0]})])
        .await
        .unwrap();
    // 1e6 overflows halfvec on the server: the whole batch rolls back,
    // and the prior committed write survives.
    let error = collection
        .upsert(vec![
            json!({"id": "new", "v": [1, 0]}),
            json!({"id": "bad", "v": [1_000_000, 0]}),
        ])
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("PostgreSQL operation failed"),
        "{error}"
    );
    let got = collection
        .get(vec![json!("prior"), json!("new"), json!("bad")], false)
        .await
        .unwrap();
    assert!(got[0].is_some());
    assert!(got[1].is_none() && got[2].is_none());
    collection.ensure_collection_deleted().await.unwrap();
    store.close().await;
}

#[tokio::test]
async fn owned_and_borrowed_pool_lifetimes() {
    let Some(url) = url() else { return };
    let pg_config: deadpool_postgres::tokio_postgres::Config = url.parse().unwrap();
    let manager =
        deadpool_postgres::Manager::new(pg_config, deadpool_postgres::tokio_postgres::NoTls);
    let pool = deadpool_postgres::Pool::builder(manager)
        .max_size(2)
        .build()
        .unwrap();
    let definition =
        VectorStoreCollectionDefinition::new(vec![VectorStoreField::key("id").with_type("str")])
            .unwrap();
    let name = unique("af_pool");

    let borrowed = PostgresStore::from_pool(pool.clone());
    let collection = borrowed.collection(&name, definition.clone()).unwrap();
    collection.ensure_collection_exists().await.unwrap();
    borrowed.close().await;
    // The caller's pool is still usable after the store closed.
    assert!(!pool.is_closed());
    drop(pool.get().await.unwrap());
    assert!(collection.upsert(vec![json!({"id": "x"})]).await.is_err());

    // A standalone collection owns its pool; closing it closes only that.
    let standalone = PostgresCollection::builder(&name, definition)
        .pool(pool.clone())
        .build()
        .unwrap();
    standalone.upsert(vec![json!({"id": "x"})]).await.unwrap();
    standalone.ensure_collection_deleted().await.unwrap();
    standalone.close().await;
    assert!(!pool.is_closed());
}

#[tokio::test]
async fn ann_indexes_filtered_query_and_exact_override() {
    let Some(store) = store() else { return };
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("group").with_type("str").indexed(),
        VectorStoreField::vector("h", 3)
            .with_index_kind(IndexKind::new(IndexKind::HNSW))
            .with_distance_function(DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY)),
        VectorStoreField::vector("i", 3)
            .with_index_kind(IndexKind::new(IndexKind::IVF_FLAT))
            .with_distance_function(DistanceFunction::new(DistanceFunction::EUCLIDEAN_DISTANCE)),
    ])
    .unwrap();
    let collection = store
        .collection_builder(unique("af_ann"), definition)
        .vector_options(
            "h",
            PostgresVectorOptions::new()
                .with_m(8)
                .with_ef_construction(32),
        )
        .vector_options("i", PostgresVectorOptions::new().with_lists(2))
        .build()
        .unwrap();
    // IVFFlat on an empty table is refused, after creating nothing.
    let error = collection.ensure_collection_exists().await.unwrap_err();
    assert!(
        error.to_string().contains("IVFFlat requires training data"),
        "{error}"
    );
    collection
        .ensure_collection_exists_with(false)
        .await
        .unwrap();
    let records: Vec<Value> = (0..40)
        .map(|n| {
            let angle = f64::from(n) / 10.0;
            json!({
                "id": n,
                "group": if n % 2 == 0 { "even" } else { "odd" },
                "h": [angle.cos(), angle.sin(), 0.1],
                "i": [angle.cos(), angle.sin(), 0.1],
            })
        })
        .collect();
    collection.upsert(records).await.unwrap();
    collection.ensure_collection_exists().await.unwrap();

    let filter = Filter::eq("group", "odd").unwrap();
    for (field, options) in [
        ("h", PostgresSearchOptions::new().with_hnsw_ef_search(40)),
        ("i", PostgresSearchOptions::new().with_ivfflat_probes(2)),
        ("h", PostgresSearchOptions::new().with_exact(true)),
        ("i", PostgresSearchOptions::new().with_exact(true)),
    ] {
        let hits = collection
            .search_with(
                vec![1.0, 0.0, 0.1],
                &VectorSearchOptions::new(5)
                    .with_vector_field_name(field)
                    .with_filter(filter.clone()),
                &options,
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 5, "{field} {options:?}");
        assert!(hits.iter().all(|h| h.record["group"] == json!("odd")));
        assert_eq!(hits[0].record["id"], json!(1), "{field} {options:?}");
    }
    collection.ensure_collection_deleted().await.unwrap();
    store.close().await;
}

#[tokio::test]
async fn typed_scalars_and_vector_storage_round_trip() {
    let Some(store) = store() else { return };
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("UUID"),
        VectorStoreField::data("ratio").with_type("float"),
        VectorStoreField::data("big").with_type("int"),
        VectorStoreField::data("blob").with_type("bytes"),
        VectorStoreField::data("day").with_type("date"),
        VectorStoreField::data("when").with_type("datetime"),
        VectorStoreField::data("meta").with_type("dict"),
        VectorStoreField::vector("half", 2).with_type("float16"),
        VectorStoreField::vector("full", 2),
        VectorStoreField::vector("forced", 2),
    ])
    .unwrap();
    let collection = store
        .collection_builder(unique("af_types"), definition)
        .vector_options(
            "forced",
            PostgresVectorOptions::new().with_vector_type(PostgresVectorType::Halfvec),
        )
        .build()
        .unwrap();
    collection.ensure_collection_exists().await.unwrap();
    let id = "00000000-0000-0000-0000-000000000001";
    collection
        .upsert(vec![json!({
            "id": id,
            "ratio": 0.1,
            "big": i64::MAX,
            "blob": [0, 255, 7],
            "day": "2024-02-29",
            "when": "2024-01-02T05:04:05.5+02:00",
            "meta": {"nested": [1, "two", null]},
            "half": [0.1, 1],
            "full": [0.1, 1],
            "forced": [0.1, 1],
        })])
        .await
        .unwrap();
    let record = collection.get(vec![json!(id)], true).await.unwrap()[0]
        .clone()
        .unwrap();
    assert_eq!(record["id"], json!(id));
    assert_eq!(record["ratio"], json!(0.1));
    assert_eq!(record["big"], json!(i64::MAX));
    assert_eq!(record["blob"], json!([0, 255, 7]));
    assert_eq!(record["day"], json!("2024-02-29"));
    assert_eq!(record["when"], json!("2024-01-02T03:04:05.5Z"));
    assert_eq!(record["meta"], json!({"nested": [1, "two", null]}));
    // halfvec rounds to 16-bit precision; vector keeps 32-bit.
    let half = record["half"][0].as_f64().unwrap();
    assert!((half - 0.1).abs() < 1e-3 && (half - 0.1).abs() > 1e-6);
    assert!((record["full"][0].as_f64().unwrap() - 0.1).abs() < 1e-7);
    assert_eq!(record["forced"], record["half"]);
    // Datetime ordering compares instants, not text.
    let found = collection
        .query(&PostgresQuery::new(5).with_filter(
            Filter::between("when", "2024-01-02T03:00:00Z", "2024-01-02T05:00:00+01:00").unwrap(),
        ))
        .await
        .unwrap();
    assert_eq!(found.len(), 1);
    collection.ensure_collection_deleted().await.unwrap();
    store.close().await;
}

#[tokio::test]
async fn a_missing_schema_is_not_created() {
    let Some(store) = store() else { return };
    let schema = unique("af_missing_schema");
    let store = store.with_schema(&schema).unwrap();
    let collection = store
        .collection(
            "t",
            VectorStoreCollectionDefinition::new(
                vec![VectorStoreField::key("id").with_type("str")],
            )
            .unwrap(),
        )
        .unwrap();
    assert!(collection.ensure_collection_exists().await.is_err());
    assert!(store.list_collection_names().await.unwrap().is_empty());
    store.close().await;
}

#[tokio::test]
async fn a_key_only_table_and_zero_vectors() {
    let Some(store) = store() else { return };
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::vector("v", 2),
    ])
    .unwrap();
    let collection = store
        .collection_builder(unique("af_zero"), definition)
        .auto_generated_key(true)
        .build()
        .unwrap();
    collection.ensure_collection_exists().await.unwrap();
    let keys = collection
        .upsert(vec![json!({}), json!({"v": [0, 0]}), json!({"v": [1, 0]})])
        .await
        .unwrap();
    assert_eq!(keys, vec![json!(1), json!(2), json!(3)]);
    // A zero vector has an undefined (NaN) cosine distance and is not ranked.
    let hits = collection
        .search(vec![1.0, 0.0], &VectorSearchOptions::new(5))
        .await
        .unwrap();
    assert_eq!(
        hits.iter()
            .map(|h| h.record["id"].clone())
            .collect::<Vec<_>>(),
        vec![json!(3)]
    );
    store.delete_collection(collection.name()).await.unwrap();
    assert!(!collection.collection_exists().await.unwrap());
    store.close().await;
}
