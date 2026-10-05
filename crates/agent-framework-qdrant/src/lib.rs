//! # agent-framework-qdrant
//!
//! [Qdrant](https://qdrant.tech) as a
//! [`VectorStore`](agent_framework_core::vectors::VectorStore) for
//! `agent-framework-rs`, porting the Python `agent_framework_qdrant`
//! package (`QdrantStore` / `QdrantCollection`).
//!
//! The connector speaks Qdrant's REST API directly through `reqwest`
//! instead of wrapping an SDK, so it adds no gRPC stack to the build. See
//! [`vector_store`] for the full behavior and the divergences from upstream.
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorCollection, VectorSearchOptions, VectorStoreCollectionDefinition,
//!     VectorStoreField,
//! };
//! use agent_framework_qdrant::QdrantStore;
//! use serde_json::json;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("int"),
//!     VectorStoreField::data("category").with_type("str").indexed(),
//!     VectorStoreField::vector("embedding", 3),
//! ])?;
//! // QDRANT_URL / QDRANT_API_KEY, defaulting to http://localhost:6333.
//! let store = QdrantStore::from_env()?;
//! let docs = store.collection("docs", definition)?;
//! docs.ensure_collection_exists().await?;
//! docs.upsert(vec![json!({"id": 1, "category": "news", "embedding": [0.1, 0.2, 0.3]})])
//!     .await?;
//! let hits = docs
//!     .search(
//!         vec![0.1, 0.2, 0.3],
//!         &VectorSearchOptions::new(5).with_filter(Filter::eq("category", "news")?),
//!     )
//!     .await?;
//! # let _ = hits;
//! # Ok(())
//! # }
//! ```

pub mod vector_store;

pub use vector_store::{
    QdrantCollection, QdrantSettings, QdrantStore, DEFAULT_QDRANT_URL, QDRANT_API_KEY_ENV,
    QDRANT_URL_ENV,
};
