//! # agent-framework-postgres
//!
//! PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) as a
//! [`VectorStore`](agent_framework_core::vectors::VectorStore), porting
//! upstream's `agent-framework-postgres` package
//! (`python/packages/postgres`).
//!
//! - [`PostgresStore`] — mirrors `PostgresStore`: hands out collections that
//!   share one connection pool.
//! - [`PostgresCollection`] — mirrors `PostgresCollection`: one table, with
//!   batch upsert / get / delete, filtered and ordered paging
//!   ([`PostgresCollection::query`]), and dense vector search with
//!   pgvector's distance operators.
//! - [`PostgresSettings`] — mirrors `PostgresSettings`: the connection
//!   string, resolved from an explicit value, `./.env`, or
//!   `POSTGRES_CONNECTION_STRING`.
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorCollection, VectorSearchOptions, VectorStoreCollectionDefinition,
//!     VectorStoreField,
//! };
//! use agent_framework_postgres::{PostgresSearchOptions, PostgresStore};
//! use serde_json::json;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("str"),
//!     VectorStoreField::data("text").with_type("str"),
//!     VectorStoreField::vector("embedding", 3),
//! ])?;
//!
//! // Reads POSTGRES_CONNECTION_STRING (explicit > ./.env > environment).
//! let store = PostgresStore::from_env()?;
//! let articles = store.collection("articles", definition)?;
//! articles.ensure_collection_exists().await?;
//! articles
//!     .upsert(vec![
//!         json!({"id": "1", "text": "PostgreSQL supports vectors", "embedding": [1, 0, 0]}),
//!         json!({"id": "2", "text": "A travel journal", "embedding": [0, 1, 0]}),
//!     ])
//!     .await?;
//! let hits = articles
//!     .search_with(
//!         vec![1.0, 0.0, 0.0],
//!         &VectorSearchOptions::new(3)
//!             .with_filter(Filter::contains_text("text", "PostgreSQL")?),
//!         &PostgresSearchOptions::new().with_score_threshold(0.25),
//!     )
//!     .await?;
//! for hit in hits {
//!     println!("{} {:?}", hit.record["text"], hit.score);
//! }
//! store.close().await;
//! # Ok(())
//! # }
//! ```
//!
//! # Provisioning
//!
//! As upstream, the connector never enables extensions, creates schemas, or
//! migrates tables. The `vector` extension must be installed and visible on
//! the connection's `search_path`, and the schema (default `public`) must
//! exist. [`VectorCollection::ensure_collection_exists`](agent_framework_core::vectors::VectorCollection::ensure_collection_exists) creates the table
//! and requested indexes; it never alters an existing table.
//!
//! # Records and types
//!
//! Records are `serde_json::Value` objects keyed by logical field name, as
//! everywhere in this workspace. Every non-vector field needs an explicit
//! [`VectorStoreField::type_`](agent_framework_core::vectors::VectorStoreField)
//! because it decides the column type — upstream reads it from the Python
//! annotation:
//!
//! | `type_` | Column | JSON value |
//! | --- | --- | --- |
//! | `str` | `text COLLATE "C"` | string |
//! | `int` | `bigint` | integer |
//! | `float` | `double precision` | number |
//! | `bool` | `boolean` | boolean |
//! | `UUID` | `uuid` | string |
//! | `bytes` | `bytea` | array of 0–255 integers (serde's `Vec<u8>`) |
//! | `date` | `date` | `"YYYY-MM-DD"` |
//! | `datetime` | `timestamp with time zone` | RFC 3339 string with an offset; read back in UTC |
//! | `list` / `dict` | `jsonb` | array / object |
//!
//! Keys are `str`, `int`, or `UUID`. Vector fields declare `float`,
//! `float32` (stored as `vector`), or `float16` (stored as `halfvec`); see
//! [`PostgresVectorOptions`] to choose explicitly.
//!
//! # Scores
//!
//! Scores are in the declared metric's units, not probabilities. An
//! undeclared distance function is upstream's default, **cosine distance**
//! (lower is closer). `cosine_similarity` and `dot_prod` are
//! higher-is-closer and their `score_threshold` is a minimum; every other
//! metric's is a maximum. IVFFlat does not support `manhattan`.
//!
//! # Divergences from upstream
//!
//! - **No `provider_annotations` / `operation_options` bags.** This port's
//!   [`VectorStoreField`](agent_framework_core::vectors::VectorStoreField)
//!   and [`VectorSearchOptions`](agent_framework_core::vectors::VectorSearchOptions)
//!   carry neither, so upstream's `postgres.vector_type` / `postgres.m` /
//!   `postgres.ef_construction` / `postgres.lists` annotations are typed
//!   [`PostgresVectorOptions`] on the collection builder, the
//!   `exact` / `hnsw_ef_search` / `ivfflat_probes` options and
//!   `score_threshold` are [`PostgresSearchOptions`], and
//!   `create_indexes=False` is
//!   [`PostgresCollection::ensure_collection_exists_with`].
//! - **Filtered reads are a separate method.** The core
//!   [`VectorCollection::get`](agent_framework_core::vectors::VectorCollection::get) reads by key only; upstream's
//!   `get(filter=..., top=..., skip=..., order_by=...)` is
//!   [`PostgresCollection::query`].
//! - **Generated keys** (`is_auto_generated` upstream) are declared with
//!   [`PostgresCollectionBuilder::auto_generated_key`], because the core
//!   field type has no such flag.
//! - **A borrowed client is a pool.** Upstream accepts a psycopg
//!   `AsyncConnection` or `AsyncConnectionPool`; here a caller-owned
//!   [`deadpool_postgres::Pool`] can be injected with
//!   [`PostgresStore::from_pool`], and is never closed by this crate.
//! - **No `.env` path argument.** Settings resolve through the core
//!   [`load_setting`](agent_framework_core::settings::load_setting), which
//!   reads `./.env` only.
//! - **A missing non-key field is written as `NULL`** rather than raising
//!   `KeyError`: an upsert replaces the whole row, so omitting a field
//!   reads exactly like the `NULL` a typed model would carry.
//! - **No result metadata.** Upstream reports `approximate` /
//!   `distance_function` beside the results; the core result type has no
//!   metadata slot, and both values are known to the caller who chose them.
//! - **`provider_filter` is refused.** Verbatim SQL would bypass the
//!   parameterization this connector guarantees.
//! - **TLS** is rustls with the WebPKI roots; `sslmode=require` therefore
//!   verifies the server certificate, which is stricter than libpq.

mod sql;
mod vector_store;

pub use deadpool_postgres;
pub use vector_store::{
    PostgresCollection, PostgresCollectionBuilder, PostgresQuery, PostgresSearchOptions,
    PostgresSettings, PostgresStore, PostgresVectorOptions, PostgresVectorType,
};
