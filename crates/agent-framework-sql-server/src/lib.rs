//! # agent-framework-sql-server
//!
//! SQL Server 2025 / Azure SQL native `VECTOR` columns as a
//! [`VectorStore`](agent_framework_core::vectors::VectorStore), porting
//! upstream's `agent-framework-sql-server` package
//! (`python/packages/sql-server`), over the pure-Rust
//! [`tiberius`](https://docs.rs/tiberius) TDS client.
//!
//! - [`SqlServerStore`] — mirrors `SqlServerStore`: hands out collections
//!   sharing one client.
//! - [`SqlServerCollection`] — mirrors `SqlServerCollection`: one table,
//!   batch upsert / get / delete, filtered and ordered paging
//!   ([`SqlServerCollection::query`]), and exact `VECTOR_DISTANCE` search.
//! - [`SqlServerSettings`] — mirrors `SqlServerSettings`: the ADO.NET
//!   connection string from an explicit value, `./.env`, or
//!   `SQL_SERVER_CONNECTION_STRING`.
//! - [`is_committed_cleanup`] — mirrors
//!   `SqlServerCommittedCleanupException`: the transaction **committed**,
//!   but closing its connection failed; do not retry automatically.
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorCollection, VectorSearchOptions, VectorStoreCollectionDefinition,
//!     VectorStoreField,
//! };
//! use agent_framework_sql_server::SqlServerStore;
//! use serde_json::json;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! // e.g. "Server=tcp:my-server.database.windows.net,1433;Database=db;
//! //       Authentication=ActiveDirectoryDefault;Encrypt=yes"
//! let store = SqlServerStore::from_env()?;
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("str"),
//!     VectorStoreField::data("text").with_type("str"),
//!     VectorStoreField::vector("embedding", 3),
//! ])?;
//! let articles = store.collection("articles", definition)?;
//! articles.ensure_collection_exists().await?;
//! articles
//!     .upsert(vec![json!({"id": "1", "text": "SQL Server vectors", "embedding": [1, 0, 0]})])
//!     .await?;
//! let hits = articles
//!     .search_with_threshold(
//!         vec![1.0, 0.0, 0.0],
//!         &VectorSearchOptions::new(3).with_filter(Filter::starts_with("text", "SQL")?),
//!         Some(0.25),
//!     )
//!     .await?;
//! # let _ = hits;
//! store.close().await;
//! # Ok(())
//! # }
//! ```
//!
//! # Provisioning
//!
//! Requires a vector-enabled database (SQL Server 2025, Azure SQL Database,
//! Azure SQL Managed Instance on the 2025 / always-up-to-date policy, or SQL
//! database in Microsoft Fabric) and an existing schema (default `dbo`).
//! [`VectorCollection::ensure_collection_exists`](agent_framework_core::vectors::VectorCollection::ensure_collection_exists)
//! creates the table and scalar indexes, never a schema, and never alters an
//! existing table.
//!
//! # Authentication
//!
//! The ADO.NET connection string's own SQL login works as-is. For
//! passwordless Microsoft Entra authentication, as upstream's README
//! documents, `Authentication=ActiveDirectoryDefault` uses
//! [`agent_framework_azure::DefaultAzureCredential`] and
//! `Authentication=ActiveDirectoryMSI` uses
//! [`agent_framework_azure::ManagedIdentityCredential`] (`UID=<client-id>`
//! for a user-assigned identity). Any other
//! [`TokenCredential`](agent_framework_azure::TokenCredential) can be supplied
//! with [`SqlServerStoreBuilder::token_credential`]. Tokens are requested
//! for [`SQL_SERVER_SCOPE`].
//!
//! # Records, filters, and scores
//!
//! Records are `serde_json::Value` objects keyed by logical field name.
//! Every non-vector field needs an explicit `type_`:
//!
//! | `type_` | Column | JSON value |
//! | --- | --- | --- |
//! | `str` | `NVARCHAR(450)` (key or indexed) / `NVARCHAR(MAX)`, binary collation | string |
//! | `int` | `BIGINT` | integer |
//! | `float` | `FLOAT(53)` | number |
//! | `bool` | `BIT` | boolean |
//! | `UUID` | `UNIQUEIDENTIFIER` | string |
//! | `bytes` | `VARBINARY(MAX)` | array of 0–255 integers |
//! | `date` | `DATE` | `"YYYY-MM-DD"` |
//! | `datetime` | `DATETIME2(7)`, normalized to UTC | RFC 3339 string with an offset |
//! | `list` / `dict` | `NVARCHAR(MAX)` JSON | array / object |
//!
//! Vectors are `float32` `VECTOR(n)` columns of 1–1998 dimensions. String
//! keys cannot end in a space (SQL Server ignores trailing spaces in key
//! comparisons). Filters support scalar `eq` / `ne` / `in` / `not_in` /
//! `is_null` / `is_not_null` / `exists`, ordered comparisons and `between`
//! on numbers, dates, and datetimes, and `starts_with` / `ends_with` /
//! `contains_text` on strings; JSON fields support only the null and
//! existence tests. Scores are raw metric units: cosine **distance** by
//! default (lower is closer); `cosine_similarity` and `dot_prod` are
//! higher-is-closer.
//!
//! # Divergences from upstream
//!
//! - **Driver.** Upstream drives `mssql-python` on a dedicated worker
//!   thread; this crate is natively async. Each operation still opens,
//!   commits or rolls back, and closes its own connection, and
//!   [`SqlServerStore::close`] still waits for in-flight operations. Dropping
//!   (cancelling) an in-flight operation drops its connection, which makes
//!   the server roll back an uncommitted transaction; upstream instead
//!   waits for the worker to finish.
//! - **Committed-cleanup is a predicate, not a type.** The core error enum
//!   is closed, so [`is_committed_cleanup`] recognizes the
//!   `Error::Service` this crate returns in exactly that case.
//! - **`query_timeout`** bounds each statement with a client-side timer
//!   rather than the driver's statement timeout.
//! - **Vectors are read through `CAST(... AS NVARCHAR(MAX))`.** Upstream
//!   projects the native column and lets the driver return JSON text; the
//!   explicit cast keeps the TDS client from ever meeting the `VECTOR` type.
//!   Vectors are still bound as JSON text, as upstream binds them.
//! - **No `provider_annotations` / `operation_options` / `.env` path**, as
//!   in the other connectors: generated keys are
//!   [`SqlServerCollectionBuilder::auto_generated_key`], the score threshold
//!   is [`SqlServerCollection::search_with_threshold`], filtered reads are
//!   [`SqlServerCollection::query`], and `provider_filter` is refused
//!   (verbatim T-SQL would bypass parameterization).
//! - **A missing non-key field is written as `NULL`**, since an upsert
//!   replaces every column.

mod connection;
mod sql;
mod vector_store;

pub use connection::{is_committed_cleanup, SQL_SERVER_SCOPE};
pub use sql::MAX_PARAMETERS;
pub use vector_store::{
    SqlServerClientOptions, SqlServerCollection, SqlServerCollectionBuilder, SqlServerQuery,
    SqlServerSettings, SqlServerStore, SqlServerStoreBuilder,
};
