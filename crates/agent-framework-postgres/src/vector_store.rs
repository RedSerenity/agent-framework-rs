//! The store, the collection, and the pool they share.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::settings::{load_setting, SecretString};
use agent_framework_core::vectors::{
    FilterExpression, VectorCollection, VectorSearchOptions, VectorSearchResult, VectorStore,
    VectorStoreCollectionDefinition,
};
use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use serde_json::{Map, Value};
use time::format_description::well_known::Rfc3339;
use tokio_postgres::Row;

use crate::sql::{
    key_array, key_identity, prepare_key, prepare_value, quote_identifier, Column, FilterCompiler,
    Kind, Params, PgValue, Schema, ScoreForm, VectorIndex,
};

fn config(message: impl Into<String>) -> Error {
    Error::Configuration(message.into())
}

/// Wrap a driver error the way upstream wraps psycopg's: one integration
/// error that keeps the server's own message.
fn driver_error(error: tokio_postgres::Error) -> Error {
    match error.as_db_error() {
        Some(db) => Error::service(format!(
            "PostgreSQL operation failed: {} {}: {}{}",
            db.severity(),
            db.code().code(),
            db.message(),
            db.detail()
                .map(|detail| format!(" ({detail})"))
                .unwrap_or_default()
        )),
        None => Error::service(format!("PostgreSQL operation failed: {error}")),
    }
}

fn pool_error(error: deadpool_postgres::PoolError) -> Error {
    match error {
        deadpool_postgres::PoolError::Backend(e) => driver_error(e),
        other => Error::service(format!(
            "PostgreSQL operation failed: could not acquire a connection: {other}"
        )),
    }
}

// region: settings and options

/// Connection settings — mirrors upstream's `PostgresSettings`.
///
/// Resolved by [`PostgresSettings::load`] with upstream's precedence:
/// explicit value, then `./.env`, then the `POSTGRES_CONNECTION_STRING`
/// environment variable. The value is a libpq-style conninfo
/// (`host=... user=...`) or a `postgresql://` URI, kept in a
/// [`SecretString`] so it never prints.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PostgresSettings {
    /// The conninfo string or URI.
    pub connection_string: Option<SecretString>,
}

impl PostgresSettings {
    /// The environment variable upstream's `env_prefix="POSTGRES_"` resolves.
    pub const CONNECTION_STRING_ENV: &'static str = "POSTGRES_CONNECTION_STRING";

    /// Resolve settings: `connection_string` if given, else `./.env`, else
    /// the environment.
    pub fn load(connection_string: Option<SecretString>) -> Self {
        let connection_string = connection_string.or_else(|| {
            load_setting(Self::CONNECTION_STRING_ENV, None, None).map(SecretString::new)
        });
        Self { connection_string }
    }

    fn require(&self) -> Result<&SecretString> {
        match &self.connection_string {
            Some(value) if !value.expose_secret().trim().is_empty() => Ok(value),
            Some(_) => Err(config("connection_string must not be empty.")),
            None => Err(config(format!(
                "{} or an explicit connection_string is required.",
                Self::CONNECTION_STRING_ENV
            ))),
        }
    }
}

/// pgvector's two dense storage types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PostgresVectorType {
    /// `vector`: 32-bit floats. The default for `float` / `float32` fields.
    Vector,
    /// `halfvec`: 16-bit floats. The default for `float16` fields.
    Halfvec,
}

/// Per-vector-field storage and index tuning — upstream's `postgres.*`
/// provider annotations, typed.
///
/// | Upstream annotation | Here | Range | Default |
/// | --- | --- | --- | --- |
/// | `postgres.vector_type` | [`Self::with_vector_type`] | `vector` / `halfvec` | by element type |
/// | `postgres.m` (HNSW) | [`Self::with_m`] | 2–100 | 16 |
/// | `postgres.ef_construction` (HNSW) | [`Self::with_ef_construction`] | 2·m–1000 | 64 |
/// | `postgres.lists` (IVFFlat) | [`Self::with_lists`] | 1–32768 | 100 |
///
/// As upstream, tuning for an index kind the field does not declare is an
/// error, not something silently ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PostgresVectorOptions {
    /// Storage type override.
    pub vector_type: Option<PostgresVectorType>,
    /// HNSW `m`.
    pub m: Option<u32>,
    /// HNSW `ef_construction`.
    pub ef_construction: Option<u32>,
    /// IVFFlat `lists`.
    pub lists: Option<u32>,
}

impl PostgresVectorOptions {
    /// No overrides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Choose `vector` or `halfvec` storage explicitly.
    pub fn with_vector_type(mut self, vector_type: PostgresVectorType) -> Self {
        self.vector_type = Some(vector_type);
        self
    }

    /// HNSW `m`.
    pub fn with_m(mut self, m: u32) -> Self {
        self.m = Some(m);
        self
    }

    /// HNSW `ef_construction`.
    pub fn with_ef_construction(mut self, ef_construction: u32) -> Self {
        self.ef_construction = Some(ef_construction);
        self
    }

    /// IVFFlat `lists`.
    pub fn with_lists(mut self, lists: u32) -> Self {
        self.lists = Some(lists);
        self
    }
}

/// Postgres-specific search options — upstream's `exact` /
/// `hnsw_ef_search` / `ivfflat_probes` `operation_options` and the
/// `score_threshold` argument.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PostgresSearchOptions {
    /// Force exact (`true`) or permit approximate (`false`) search. Defaults
    /// to exact for `flat` / `default` fields and approximate for HNSW /
    /// IVFFlat fields. Approximate search on a field with no ANN index is
    /// an error. Selective filters can reduce ANN recall; set `true` when
    /// complete recall matters.
    pub exact: Option<bool>,
    /// `hnsw.ef_search` for this query (1–1000); HNSW approximate only.
    pub hnsw_ef_search: Option<u32>,
    /// `ivfflat.probes` for this query (1–32768); IVFFlat approximate only.
    pub ivfflat_probes: Option<u32>,
    /// Keep only hits at least this close, in the metric's own units: a
    /// minimum for `cosine_similarity` / `dot_prod`, a maximum otherwise.
    /// Applied before paging.
    pub score_threshold: Option<f64>,
}

impl PostgresSearchOptions {
    /// No overrides.
    pub fn new() -> Self {
        Self::default()
    }

    /// See [`Self::exact`].
    pub fn with_exact(mut self, exact: bool) -> Self {
        self.exact = Some(exact);
        self
    }

    /// See [`Self::hnsw_ef_search`].
    pub fn with_hnsw_ef_search(mut self, value: u32) -> Self {
        self.hnsw_ef_search = Some(value);
        self
    }

    /// See [`Self::ivfflat_probes`].
    pub fn with_ivfflat_probes(mut self, value: u32) -> Self {
        self.ivfflat_probes = Some(value);
        self
    }

    /// See [`Self::score_threshold`].
    pub fn with_score_threshold(mut self, threshold: f64) -> Self {
        self.score_threshold = Some(threshold);
        self
    }
}

/// A filtered, ordered, paged read — upstream's
/// `get(filter=..., top=..., skip=..., order_by=..., include_vectors=...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct PostgresQuery {
    /// A portable filter; `None` reads every row.
    pub filter: Option<FilterExpression>,
    /// Page size (upstream's default 10). Must be greater than zero.
    pub top: usize,
    /// Rows to skip.
    pub skip: usize,
    /// `(logical field name, ascending)` pairs. Nulls sort last; the key
    /// breaks ties ascending unless it is listed.
    pub order_by: Vec<(String, bool)>,
    /// Whether vector fields are returned.
    pub include_vectors: bool,
}

impl Default for PostgresQuery {
    fn default() -> Self {
        Self {
            filter: None,
            top: 10,
            skip: 0,
            order_by: Vec::new(),
            include_vectors: false,
        }
    }
}

impl PostgresQuery {
    /// A page of `top` rows.
    pub fn new(top: usize) -> Self {
        Self {
            top,
            ..Self::default()
        }
    }

    /// Filter the rows.
    pub fn with_filter(mut self, filter: impl Into<FilterExpression>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Skip leading rows.
    pub fn with_skip(mut self, skip: usize) -> Self {
        self.skip = skip;
        self
    }

    /// Append an ordering column.
    pub fn order_by(mut self, field: impl Into<String>, ascending: bool) -> Self {
        self.order_by.push((field.into(), ascending));
        self
    }

    /// Return vector fields too.
    pub fn with_include_vectors(mut self, include: bool) -> Self {
        self.include_vectors = include;
        self
    }
}

// endregion

// region: client

/// The pool plus its ownership — upstream's `_Client`.
struct ClientInner {
    pool: Pool,
    /// Created here (closed by [`PostgresStore::close`]) rather than handed
    /// in by the caller.
    owned: bool,
    closed: AtomicBool,
}

#[derive(Clone)]
struct Client(Arc<ClientInner>);

impl Client {
    fn owned(connection_string: &SecretString) -> Result<Self> {
        let pg_config: tokio_postgres::Config =
            connection_string.expose_secret().parse().map_err(|e| {
                config(format!(
                    "connection_string is not a valid PostgreSQL conninfo or URI: {e}"
                ))
            })?;
        let manager = Manager::from_config(
            pg_config,
            make_tls()?,
            ManagerConfig {
                recycling_method: RecyclingMethod::Fast,
            },
        );
        // Upstream: `AsyncConnectionPool(min_size=1, max_size=10)`, opened
        // lazily on first use. deadpool opens connections on demand.
        let pool = Pool::builder(manager)
            .max_size(10)
            .build()
            .map_err(|e| config(format!("could not build the PostgreSQL pool: {e}")))?;
        Ok(Self(Arc::new(ClientInner {
            pool,
            owned: true,
            closed: AtomicBool::new(false),
        })))
    }

    fn borrowed(pool: Pool) -> Self {
        Self(Arc::new(ClientInner {
            pool,
            owned: false,
            closed: AtomicBool::new(false),
        }))
    }

    fn ensure_open(&self) -> Result<()> {
        if self.0.closed.load(Ordering::SeqCst) {
            return Err(Error::other("The Postgres client is closed."));
        }
        Ok(())
    }

    async fn connection(&self) -> Result<deadpool_postgres::Object> {
        self.ensure_open()?;
        self.0.pool.get().await.map_err(pool_error)
    }

    fn close(&self) {
        if !self.0.closed.swap(true, Ordering::SeqCst) && self.0.owned {
            self.0.pool.close();
        }
    }
}

fn make_tls() -> Result<tokio_postgres_rustls::MakeRustlsConnect> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| config(format!("could not configure TLS: {e}")))?
    .with_root_certificates(roots)
    .with_no_client_auth();
    Ok(tokio_postgres_rustls::MakeRustlsConnect::new(config))
}

// endregion

// region: the store

/// A PostgreSQL database schema as a [`VectorStore`] — mirrors upstream's
/// `PostgresStore`.
///
/// Construction does no I/O; the owned pool connects on first use. Every
/// collection from [`Self::collection`] borrows this store's pool, so keep
/// the store open while using them, and [`Self::close`] it when done.
#[derive(Clone)]
pub struct PostgresStore {
    client: Client,
    schema: String,
}

impl std::fmt::Debug for PostgresStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresStore")
            .field("schema", &self.schema)
            .field("owned_pool", &self.client.0.owned)
            .finish_non_exhaustive()
    }
}

impl PostgresStore {
    /// Own a pool for `connection_string` (a conninfo or `postgresql://`
    /// URI). The schema defaults to `public`.
    pub fn new(connection_string: impl Into<SecretString>) -> Result<Self> {
        Self::from_settings(PostgresSettings {
            connection_string: Some(connection_string.into()),
        })
    }

    /// Resolve `POSTGRES_CONNECTION_STRING` from `./.env` or the environment.
    pub fn from_env() -> Result<Self> {
        Self::from_settings(PostgresSettings::load(None))
    }

    /// Own a pool for already-resolved settings.
    pub fn from_settings(settings: PostgresSettings) -> Result<Self> {
        Ok(Self {
            client: Client::owned(settings.require()?)?,
            schema: "public".into(),
        })
    }

    /// Borrow a caller-owned pool. [`Self::close`] never closes it.
    pub fn from_pool(pool: Pool) -> Self {
        Self {
            client: Client::borrowed(pool),
            schema: "public".into(),
        }
    }

    /// Use an existing schema other than `public`. The connector never
    /// creates schemas.
    pub fn with_schema(mut self, schema: impl Into<String>) -> Result<Self> {
        let schema = schema.into();
        quote_identifier(&schema)?;
        self.schema = schema;
        Ok(self)
    }

    /// The schema every collection lives in.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Describe one table. No I/O.
    pub fn collection(
        &self,
        name: impl Into<String>,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<PostgresCollection> {
        self.collection_builder(name, definition).build()
    }

    /// Describe one table with Postgres-specific options. No I/O.
    pub fn collection_builder(
        &self,
        name: impl Into<String>,
        definition: VectorStoreCollectionDefinition,
    ) -> PostgresCollectionBuilder {
        let mut builder = PostgresCollection::builder(name, definition);
        builder.schema = self.schema.clone();
        builder.shared = Some(self.client.clone());
        builder
    }

    /// Drop one table in this schema, without `CASCADE` — upstream's
    /// `ensure_collection_deleted(collection_name)`. Absent tables are fine.
    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        let table = format!(
            "{}.{}",
            quote_identifier(&self.schema)?,
            quote_identifier(name)?
        );
        let client = self.client.connection().await?;
        client
            .execute(&format!("DROP TABLE IF EXISTS {table}"), &[])
            .await
            .map_err(driver_error)?;
        Ok(())
    }

    /// Close an owned pool; a pool from [`Self::from_pool`] stays open.
    /// Every collection sharing it stops working.
    pub async fn close(&self) {
        self.client.close();
    }
}

#[async_trait::async_trait]
impl VectorStore for PostgresStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    /// Base tables in this schema, including ones this crate did not create.
    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let client = self.client.connection().await?;
        let rows = client
            .query(
                "SELECT table_name::text FROM information_schema.tables WHERE table_schema = \
                 $1::text AND table_type = 'BASE TABLE' ORDER BY table_name",
                &[&self.schema],
            )
            .await
            .map_err(driver_error)?;
        rows.iter()
            .map(|row| row.try_get::<_, String>(0).map_err(driver_error))
            .collect()
    }

    async fn collection_exists(&self, name: &str) -> Result<bool> {
        table_exists(&self.client, &self.schema, name).await
    }
}

async fn table_exists(client: &Client, schema: &str, table: &str) -> Result<bool> {
    let connection = client.connection().await?;
    let row = connection
        .query_opt(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = $1::text AND \
             table_name = $2::text AND table_type = 'BASE TABLE'",
            &[&schema, &table],
        )
        .await
        .map_err(driver_error)?;
    Ok(row.is_some())
}

// endregion

// region: the collection

/// Builds a [`PostgresCollection`].
///
/// From [`PostgresStore::collection_builder`] it shares the store's pool;
/// from [`PostgresCollection::builder`] it owns one, created from
/// [`Self::connection_string`], [`Self::pool`], or the environment.
pub struct PostgresCollectionBuilder {
    name: String,
    definition: VectorStoreCollectionDefinition,
    schema: String,
    shared: Option<Client>,
    connection_string: Option<SecretString>,
    pool: Option<Pool>,
    vector_options: HashMap<String, PostgresVectorOptions>,
    auto_generated_key: bool,
}

impl std::fmt::Debug for PostgresCollectionBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresCollectionBuilder")
            .field("name", &self.name)
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl PostgresCollectionBuilder {
    /// The existing schema holding the table (default `public`).
    pub fn schema(mut self, schema: impl Into<String>) -> Self {
        self.schema = schema.into();
        self
    }

    /// Own a pool for this conninfo / URI. Ignored for a store's collection.
    pub fn connection_string(mut self, connection_string: impl Into<SecretString>) -> Self {
        self.connection_string = Some(connection_string.into());
        self
    }

    /// Borrow a caller-owned pool. Ignored for a store's collection.
    pub fn pool(mut self, pool: Pool) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Storage and index tuning for one vector field (by logical name).
    pub fn vector_options(
        mut self,
        field: impl Into<String>,
        options: PostgresVectorOptions,
    ) -> Self {
        self.vector_options.insert(field.into(), options);
        self
    }

    /// Let the server generate the key when a record omits it (or sets it
    /// to `null`): `GENERATED BY DEFAULT AS IDENTITY` for `int`,
    /// `gen_random_uuid()` for `UUID` and `str`. Mirrors upstream's
    /// `is_auto_generated` key fields. An explicit key still upserts.
    pub fn auto_generated_key(mut self, generated: bool) -> Self {
        self.auto_generated_key = generated;
        self
    }

    /// Validate the definition and build. No I/O.
    pub fn build(self) -> Result<PostgresCollection> {
        let schema = Schema::new(
            &self.schema,
            &self.name,
            self.definition,
            &self.vector_options,
            self.auto_generated_key,
        )?;
        let (client, shared) = match (self.shared, self.pool, self.connection_string) {
            (Some(client), _, _) => (client, true),
            (None, Some(_), Some(_)) => {
                return Err(config("Supply exactly one of connection_string or pool."))
            }
            (None, Some(pool), None) => (Client::borrowed(pool), false),
            (None, None, connection_string) => {
                let settings = PostgresSettings::load(connection_string);
                (Client::owned(settings.require()?)?, false)
            }
        };
        Ok(PostgresCollection {
            schema: Arc::new(schema),
            client,
            shared,
        })
    }
}

/// One PostgreSQL table as a [`VectorCollection`] — mirrors upstream's
/// `PostgresCollection`.
///
/// Every operation runs in one transaction, so a batch upsert writes all
/// records or none. Search is exact by default; HNSW and IVFFlat fields
/// permit approximate search (see [`PostgresSearchOptions`]).
#[derive(Clone)]
pub struct PostgresCollection {
    schema: Arc<Schema>,
    client: Client,
    /// Borrowed from a store: [`Self::close`] leaves the pool alone.
    shared: bool,
}

impl std::fmt::Debug for PostgresCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresCollection")
            .field("schema", &self.schema.schema)
            .field("table", &self.schema.table_name)
            .finish_non_exhaustive()
    }
}

impl PostgresCollection {
    /// A standalone collection builder.
    pub fn builder(
        name: impl Into<String>,
        definition: VectorStoreCollectionDefinition,
    ) -> PostgresCollectionBuilder {
        PostgresCollectionBuilder {
            name: name.into(),
            definition,
            schema: "public".into(),
            shared: None,
            connection_string: None,
            pool: None,
            vector_options: HashMap::new(),
            auto_generated_key: false,
        }
    }

    /// The schema holding the table.
    pub fn schema_name(&self) -> &str {
        &self.schema.schema
    }

    /// Close an owned pool — never a store's or a caller's.
    pub async fn close(&self) {
        if !self.shared {
            self.client.close();
        }
    }

    /// Create the table, and its indexes when `create_indexes` is `true`.
    ///
    /// Upstream's `ensure_collection_exists(operation_options={"create_indexes":
    /// False})`. IVFFlat needs training data, so its index cannot be built
    /// on an empty table: create with `false`, upsert, then call
    /// [`VectorCollection::ensure_collection_exists`] (which passes `true`).
    pub async fn ensure_collection_exists_with(&self, create_indexes: bool) -> Result<()> {
        let schema = &self.schema;
        let mut connection = self.client.connection().await?;
        let transaction = connection.transaction().await.map_err(driver_error)?;
        transaction
            .execute(&schema.create_table_sql(), &[])
            .await
            .map_err(driver_error)?;
        if create_indexes {
            for column in &schema.columns {
                let Some(statement) = schema.index_sql(column) else {
                    continue;
                };
                if matches!(
                    column.vector.as_ref().map(|v| v.index),
                    Some(VectorIndex::IvfFlat { .. })
                ) {
                    let probe = format!(
                        "SELECT 1 FROM {} WHERE {} IS NOT NULL LIMIT 1",
                        schema.table,
                        column.quoted()
                    );
                    if transaction
                        .query_opt(&probe, &[])
                        .await
                        .map_err(driver_error)?
                        .is_none()
                    {
                        return Err(config(
                            "IVFFlat requires training data. Create with create_indexes=false, \
                             upsert vectors, then call ensure_collection_exists() again.",
                        ));
                    }
                }
                transaction
                    .execute(&statement, &[])
                    .await
                    .map_err(driver_error)?;
            }
        }
        transaction.commit().await.map_err(driver_error)
    }

    /// A filtered, ordered, paged read — upstream's filtered `get`.
    pub async fn query(&self, query: &PostgresQuery) -> Result<Vec<Value>> {
        self.client.ensure_open()?;
        let (statement, params, columns) = build_query(&self.schema, query)?;
        let mut connection = self.client.connection().await?;
        let transaction = connection.transaction().await.map_err(driver_error)?;
        let rows = transaction
            .query(&statement, &params.as_refs())
            .await
            .map_err(driver_error)?;
        transaction.commit().await.map_err(driver_error)?;
        rows.iter()
            .map(|row| self.record(row, &columns, 0))
            .collect()
    }

    /// [`VectorCollection::search`] with Postgres-specific options.
    pub async fn search_with(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
        postgres: &PostgresSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        options.validate()?;
        self.client.ensure_open()?;
        let plan = build_search(&self.schema, &vector, options, postgres)?;
        let mut connection = self.client.connection().await?;
        let transaction = connection.transaction().await.map_err(driver_error)?;
        for (name, value) in &plan.settings {
            // `is_local = true`: the setting ends with this transaction,
            // which is always rolled back below — upstream's
            // `force_rollback=True` savepoint.
            transaction
                .execute(
                    "SELECT set_config($1::text, $2::text, true)",
                    &[name, value],
                )
                .await
                .map_err(driver_error)?;
        }
        let rows = transaction
            .query(&plan.statement, &plan.params.as_refs())
            .await
            .map_err(driver_error)?;
        transaction.rollback().await.map_err(driver_error)?;
        let columns: Vec<&Column> = plan
            .columns
            .iter()
            .map(|&index| &self.schema.columns[index])
            .collect();
        rows.iter()
            .map(|row| {
                let score: Option<f64> = row.try_get(columns.len()).map_err(driver_error)?;
                Ok(VectorSearchResult {
                    record: self.record(row, &columns, 0)?,
                    score,
                    score_kind: None,
                })
            })
            .collect()
    }

    /// Decode one row (starting at `offset`) into a record keyed by logical
    /// field name.
    fn record(&self, row: &Row, columns: &[&Column], offset: usize) -> Result<Value> {
        let mut out = Map::new();
        for (index, column) in columns.iter().enumerate() {
            out.insert(column.name.clone(), decode(row, offset + index, column)?);
        }
        Ok(Value::Object(out))
    }
}

/// `SELECT ... WHERE filter ORDER BY ... LIMIT ... OFFSET ...`.
fn build_query<'a>(
    schema: &'a Schema,
    query: &PostgresQuery,
) -> Result<(String, Params, Vec<&'a Column>)> {
    if query.top == 0 {
        return Err(config("query `top` must be greater than zero"));
    }
    let columns = schema.selected(query.include_vectors);
    let mut params = Params::default();
    let filter = match &query.filter {
        Some(filter) => FilterCompiler::new(schema, &mut params).compile(filter)?,
        None => "TRUE".into(),
    };
    let order = schema.order_by_sql(&query.order_by)?;
    let top = params.bind(PgValue::Int(to_i64(query.top)));
    let skip = params.bind(PgValue::Int(to_i64(query.skip)));
    let statement = format!(
        "SELECT {} FROM {} WHERE {filter} ORDER BY {order} LIMIT {top} OFFSET {skip}",
        projection(&columns, ""),
        schema.table
    );
    Ok((statement, params, columns))
}

fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn projection(columns: &[&Column], alias: &str) -> String {
    columns
        .iter()
        .map(|c| c.projection(alias))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Everything a search sends.
#[derive(Debug)]
struct SearchPlan {
    /// `set_config` pairs, applied transaction-locally.
    settings: Vec<(String, String)>,
    statement: String,
    params: Params,
    /// Indexes into the schema's columns, in projection order.
    columns: Vec<usize>,
}

/// Upstream's `_inner_search`, minus the I/O.
fn build_search(
    schema: &Schema,
    vector: &[f32],
    options: &VectorSearchOptions,
    postgres: &PostgresSearchOptions,
) -> Result<SearchPlan> {
    if options.provider_filter.is_some() {
        return Err(config(
            "Postgres does not accept a provider_filter: verbatim SQL would bypass \
             parameterization. Use a portable filter.",
        ));
    }
    let field = schema
        .definition
        .try_get_vector_field(options.vector_field_name.as_deref())
        .ok_or_else(|| config("Select a vector_field_name from the collection definition."))?;
    let column = schema
        .columns
        .iter()
        .find(|c| c.name == field.name)
        .expect("definition fields are schema columns");
    let spec = column.vector.as_ref().expect("a vector column");
    let ann = !matches!(spec.index, VectorIndex::Exact);
    let exact = postgres.exact.unwrap_or(!ann);
    if !exact && !ann {
        return Err(config(
            "Approximate search requires an HNSW or IVFFlat vector field.",
        ));
    }
    let mut settings = Vec::new();
    if let Some(ef) = postgres.hnsw_ef_search {
        if exact || !matches!(spec.index, VectorIndex::Hnsw { .. }) {
            return Err(config(
                "hnsw_ef_search requires an HNSW approximate search.",
            ));
        }
        if !(1..=1000).contains(&ef) {
            return Err(config(
                "hnsw_ef_search must be an integer between 1 and 1000.",
            ));
        }
        settings.push(("hnsw.ef_search".to_string(), ef.to_string()));
    }
    if let Some(probes) = postgres.ivfflat_probes {
        if exact || !matches!(spec.index, VectorIndex::IvfFlat { .. }) {
            return Err(config(
                "ivfflat_probes requires an IVFFlat approximate search.",
            ));
        }
        if !(1..=32_768).contains(&probes) {
            return Err(config(
                "ivfflat_probes must be an integer between 1 and 32768.",
            ));
        }
        settings.push(("ivfflat.probes".to_string(), probes.to_string()));
    }
    if !exact {
        // pgvector >= 0.8: keep scanning until enough rows pass the filter,
        // in strict distance order.
        settings.push((
            "hnsw.iterative_scan".to_string(),
            "strict_order".to_string(),
        ));
        settings.push(("ivfflat.iterative_scan".to_string(), "off".to_string()));
    }
    let query_value = Value::Array(
        vector
            .iter()
            .map(|v| serde_json::Number::from_f64(f64::from(*v)).map(Value::Number))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| config("query vector elements must be finite"))?,
    );
    let mut params = Params::default();
    let query_vector = params.bind(prepare_value(column, &query_value)?);
    let distance = format!(
        "{} {} {query_vector}",
        column.quoted(),
        spec.metric.operator
    );
    let score = match spec.metric.score {
        ScoreForm::Distance => distance.clone(),
        ScoreForm::OneMinus => format!("1 - ({distance})"),
        ScoreForm::Negated => format!("-({distance})"),
    };
    let filter = match &options.filter {
        Some(filter) => FilterCompiler::new(schema, &mut params).compile(filter)?,
        None => "TRUE".into(),
    };
    let columns: Vec<usize> = schema
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| options.include_vectors || c.vector.is_none())
        .map(|(i, _)| i)
        .collect();
    let selected: Vec<&Column> = columns.iter().map(|&i| &schema.columns[i]).collect();
    let mut statement = format!(
        "SELECT {}, {score} FROM {} WHERE ({filter}) AND ({distance}) < 'Infinity'::double \
         precision",
        projection(&selected, ""),
        schema.table
    );
    if let Some(threshold) = postgres.score_threshold {
        if !threshold.is_finite() {
            return Err(config("score_threshold must be a finite number."));
        }
        let cutoff = match spec.metric.score {
            ScoreForm::Distance => threshold,
            ScoreForm::OneMinus => 1.0 - threshold,
            ScoreForm::Negated => -threshold,
        };
        let cutoff = params.bind(PgValue::Float(cutoff));
        statement.push_str(&format!(" AND ({distance}) <= {cutoff}"));
    }
    // Adding zero keeps an exact query off the ANN index without disabling
    // ordinary data indexes.
    let ranking = if exact {
        format!("({distance}) + 0")
    } else {
        distance.clone()
    };
    let top = params.bind(PgValue::Int(to_i64(options.top)));
    let skip = params.bind(PgValue::Int(to_i64(options.skip)));
    statement.push_str(&format!(
        " ORDER BY {ranking} ASC LIMIT {top} OFFSET {skip}"
    ));
    Ok(SearchPlan {
        settings,
        statement,
        params,
        columns,
    })
}

/// Decode one projected column into its JSON form (see the crate docs'
/// type table).
fn decode(row: &Row, index: usize, column: &Column) -> Result<Value> {
    let invalid = |what: &str| {
        Error::service(format!(
            "PostgreSQL returned an invalid {what} for '{}'.",
            column.name
        ))
    };
    if column.vector.is_some() {
        let text: Option<String> = row.try_get(index).map_err(driver_error)?;
        return match text {
            None => Ok(Value::Null),
            // pgvector's text form is a JSON array; components are floats
            // whatever they look like (`[1,0]` is `[1.0, 0.0]`).
            Some(text) => serde_json::from_str::<Vec<f64>>(&text)
                .map_err(|_| invalid("vector"))?
                .into_iter()
                .map(|f| {
                    serde_json::Number::from_f64(f)
                        .map(Value::Number)
                        .ok_or_else(|| invalid("vector"))
                })
                .collect::<Result<Vec<_>>>()
                .map(Value::Array),
        };
    }
    let kind = column.kind.expect("a scalar column");
    let get_err = driver_error;
    Ok(match kind {
        Kind::Str => row
            .try_get::<_, Option<String>>(index)
            .map_err(get_err)?
            .map(Value::String)
            .unwrap_or(Value::Null),
        Kind::Int => row
            .try_get::<_, Option<i64>>(index)
            .map_err(get_err)?
            .map(Value::from)
            .unwrap_or(Value::Null),
        Kind::Float => match row.try_get::<_, Option<f64>>(index).map_err(get_err)? {
            None => Value::Null,
            Some(f) => serde_json::Number::from_f64(f)
                .map(Value::Number)
                .ok_or_else(|| invalid("number"))?,
        },
        Kind::Bool => row
            .try_get::<_, Option<bool>>(index)
            .map_err(get_err)?
            .map(Value::Bool)
            .unwrap_or(Value::Null),
        Kind::Uuid => row
            .try_get::<_, Option<uuid::Uuid>>(index)
            .map_err(get_err)?
            .map(|u| Value::String(u.hyphenated().to_string()))
            .unwrap_or(Value::Null),
        Kind::Bytes => row
            .try_get::<_, Option<Vec<u8>>>(index)
            .map_err(get_err)?
            .map(|b| Value::Array(b.into_iter().map(Value::from).collect()))
            .unwrap_or(Value::Null),
        Kind::Date => match row
            .try_get::<_, Option<time::Date>>(index)
            .map_err(get_err)?
        {
            None => Value::Null,
            Some(date) => Value::String(
                date.format(time::macros::format_description!("[year]-[month]-[day]"))
                    .map_err(|_| invalid("date"))?,
            ),
        },
        Kind::DateTime => match row
            .try_get::<_, Option<time::OffsetDateTime>>(index)
            .map_err(get_err)?
        {
            None => Value::Null,
            Some(when) => Value::String(
                when.to_offset(time::UtcOffset::UTC)
                    .format(&Rfc3339)
                    .map_err(|_| invalid("datetime"))?,
            ),
        },
        Kind::List | Kind::Dict => row
            .try_get::<_, Option<Value>>(index)
            .map_err(get_err)?
            .unwrap_or(Value::Null),
    })
}

#[async_trait::async_trait]
impl VectorCollection for PostgresCollection {
    fn name(&self) -> &str {
        &self.schema.table_name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.schema.definition
    }

    /// Create the table and every requested index. Never alters an existing
    /// table, creates a schema, or enables an extension.
    async fn ensure_collection_exists(&self) -> Result<()> {
        self.ensure_collection_exists_with(true).await
    }

    async fn collection_exists(&self) -> Result<bool> {
        table_exists(&self.client, &self.schema.schema, &self.schema.table_name).await
    }

    /// Drop only this table, without `CASCADE`.
    async fn ensure_collection_deleted(&self) -> Result<()> {
        let connection = self.client.connection().await?;
        connection
            .execute(&format!("DROP TABLE IF EXISTS {}", self.schema.table), &[])
            .await
            .map_err(driver_error)?;
        Ok(())
    }

    /// Insert or update every record in one transaction, returning keys in
    /// input order. A repeated key is written twice, in order — the last
    /// write wins, exactly as sequential upserts would.
    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        self.client.ensure_open()?;
        let rows = prepare_rows(&self.schema, records)?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let schema = &self.schema;
        let key = schema.key_column();
        let mut connection = self.client.connection().await?;
        let transaction = connection.transaction().await.map_err(driver_error)?;
        let mut keys = Vec::with_capacity(rows.len());
        for row in rows {
            let (statement, params) = upsert_statement(schema, row);
            let prepared = transaction
                .prepare_cached(&statement)
                .await
                .map_err(driver_error)?;
            let returned = transaction
                .query_opt(&prepared, &params.as_refs())
                .await
                .map_err(driver_error)?
                .ok_or_else(|| Error::service("PostgreSQL did not return an upserted key."))?;
            keys.push(decode(&returned, 0, key)?);
        }
        transaction.commit().await.map_err(driver_error)?;
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        self.client.ensure_open()?;
        let schema = &self.schema;
        let adapted = keys
            .iter()
            .map(|key| prepare_key(schema, key))
            .collect::<Result<Vec<_>>>()?;
        if adapted.is_empty() {
            return Ok(Vec::new());
        }
        let columns = schema.selected(include_vectors);
        let key = schema.key_column();
        let key_position = columns
            .iter()
            .position(|c| c.name == key.name)
            .expect("the key is always selected");
        let mut params = Params::default();
        let array = params.bind(key_array(&adapted));
        let statement = format!(
            "SELECT {} FROM {} WHERE {} = ANY({array})",
            projection(&columns, ""),
            schema.table,
            key.quoted()
        );
        let connection = self.client.connection().await?;
        let rows = connection
            .query(&statement, &params.as_refs())
            .await
            .map_err(driver_error)?;
        let mut by_key: HashMap<String, Value> = HashMap::with_capacity(rows.len());
        for row in &rows {
            let record = self.record(row, &columns, 0)?;
            let identity = decode(row, key_position, key)?;
            by_key.insert(identity.to_string(), record);
        }
        Ok(adapted
            .iter()
            .map(|k| by_key.get(&key_identity(k).to_string()).cloned())
            .collect())
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        self.client.ensure_open()?;
        let schema = &self.schema;
        let adapted = keys
            .iter()
            .map(|key| prepare_key(schema, key))
            .collect::<Result<Vec<_>>>()?;
        if adapted.is_empty() {
            return Ok(());
        }
        let mut params = Params::default();
        let array = params.bind(key_array(&adapted));
        let connection = self.client.connection().await?;
        connection
            .execute(
                &format!(
                    "DELETE FROM {} WHERE {} = ANY({array})",
                    schema.table,
                    schema.key_column().quoted()
                ),
                &params.as_refs(),
            )
            .await
            .map_err(driver_error)?;
        Ok(())
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_with(vector, options, &PostgresSearchOptions::default())
            .await
    }
}

/// One record, adapted: `(column index, value)` in definition order, with a
/// generated key omitted.
type PreparedRow = Vec<(usize, PgValue)>;

/// Adapt every record before any I/O, so a bad record fails the batch
/// without a round trip.
fn prepare_rows(schema: &Schema, records: Vec<Value>) -> Result<Vec<PreparedRow>> {
    let mut rows = Vec::with_capacity(records.len());
    for (record_index, record) in records.into_iter().enumerate() {
        let stored = schema.definition.to_storage(&record)?;
        let object = stored
            .as_object()
            .ok_or_else(|| config("a vector store record must be a JSON object"))?;
        let mut row = Vec::with_capacity(schema.columns.len());
        for (index, column) in schema.columns.iter().enumerate() {
            let value = object.get(&column.storage);
            if index == schema.key {
                match value {
                    None | Some(Value::Null) if schema.auto_generated_key => continue,
                    None => {
                        return Err(config(format!(
                            "record at index {record_index} is missing its key field '{}'",
                            column.name
                        )))
                    }
                    Some(key) => row.push((index, prepare_key(schema, key)?)),
                }
            } else {
                row.push((index, prepare_value(column, value.unwrap_or(&Value::Null))?));
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

/// `INSERT ... ON CONFLICT (key) DO UPDATE ... RETURNING key` — upstream's
/// upsert statement. Without a key (generated) there is no conflict
/// target, so an identity collision fails instead of overwriting.
fn upsert_statement(schema: &Schema, row: PreparedRow) -> (String, Params) {
    let key = schema.key_column();
    let mut params = Params::default();
    let has_key = row.iter().any(|(index, _)| *index == schema.key);
    let names: Vec<String> = row
        .iter()
        .map(|(index, _)| schema.columns[*index].quoted())
        .collect();
    let mut statement = if row.is_empty() {
        format!("INSERT INTO {} DEFAULT VALUES", schema.table)
    } else {
        let placeholders: Vec<String> = row.into_iter().map(|(_, v)| params.bind(v)).collect();
        format!(
            "INSERT INTO {} ({}) VALUES ({})",
            schema.table,
            names.join(", "),
            placeholders.join(", ")
        )
    };
    if has_key {
        let key_name = key.quoted();
        let mut updates: Vec<String> = names
            .iter()
            .filter(|n| **n != key_name)
            .map(|n| format!("{n} = EXCLUDED.{n}"))
            .collect();
        if updates.is_empty() {
            updates.push(format!("{key_name} = EXCLUDED.{key_name}"));
        }
        statement.push_str(&format!(
            " ON CONFLICT ({key_name}) DO UPDATE SET {}",
            updates.join(", ")
        ));
    }
    statement.push_str(&format!(" RETURNING {}", key.projection("")));
    (statement, params)
}

// endregion

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::vectors::{DistanceFunction, Filter, IndexKind, VectorStoreField};
    use serde_json::json;

    fn definition(distance: &str, index: &str) -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str"),
            VectorStoreField::vector("embedding", 3)
                .with_storage_name("dense vector")
                .with_distance_function(DistanceFunction::new(distance))
                .with_index_kind(IndexKind::new(index)),
        ])
        .unwrap()
    }

    fn schema(distance: &str, index: &str) -> Schema {
        Schema::new(
            "public",
            "documents",
            definition(distance, index),
            &HashMap::new(),
            false,
        )
        .unwrap()
    }

    #[test]
    fn search_applies_threshold_before_paging() {
        let schema = schema("cosine_similarity", "flat");
        let plan = build_search(
            &schema,
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(2)
                .with_skip(1)
                .with_filter(Filter::eq("text", "a").unwrap()),
            &PostgresSearchOptions::new().with_score_threshold(0.75),
        )
        .unwrap();
        assert_eq!(
            plan.statement,
            "SELECT \"id\", \"text\", 1 - (\"dense vector\" <=> $1::text::vector) FROM \
             \"public\".\"documents\" WHERE (\"text\" COLLATE \"C\" IS NOT DISTINCT FROM \
             $2::text) AND (\"dense vector\" <=> $1::text::vector) < 'Infinity'::double \
             precision AND (\"dense vector\" <=> $1::text::vector) <= $3::double precision \
             ORDER BY (\"dense vector\" <=> $1::text::vector) + 0 ASC LIMIT $4::bigint OFFSET \
             $5::bigint"
        );
        assert_eq!(
            plan.params.values,
            vec![
                PgValue::Vector("[1,0,0]".into(), crate::PostgresVectorType::Vector),
                PgValue::Text("a".into()),
                PgValue::Float(0.25),
                PgValue::Int(2),
                PgValue::Int(1),
            ]
        );
        assert!(plan.settings.is_empty());
    }

    #[test]
    fn score_forms_and_cutoffs_follow_the_metric() {
        for (distance, score, cutoff) in [
            ("dot_prod", "-(\"dense vector\" <#> $1::text::vector)", -0.5),
            (
                "negative_dot_prod",
                "\"dense vector\" <#> $1::text::vector",
                0.5,
            ),
            (
                "euclidean_distance",
                "\"dense vector\" <-> $1::text::vector",
                0.5,
            ),
            ("manhattan", "\"dense vector\" <+> $1::text::vector", 0.5),
            (
                "cosine_distance",
                "\"dense vector\" <=> $1::text::vector",
                0.5,
            ),
        ] {
            let plan = build_search(
                &schema(distance, "flat"),
                &[1.0, 0.0, 0.0],
                &VectorSearchOptions::new(3).with_include_vectors(true),
                &PostgresSearchOptions::new().with_score_threshold(0.5),
            )
            .unwrap();
            assert!(
                plan.statement.starts_with(&format!(
                    "SELECT \"id\", \"text\", \"dense vector\"::text, {score} FROM"
                )),
                "{}",
                plan.statement
            );
            assert_eq!(plan.params.values[1], PgValue::Float(cutoff));
        }
    }

    #[test]
    fn ann_search_options_are_validated_and_scoped() {
        let hnsw = schema("cosine_distance", "hnsw");
        let plan = build_search(
            &hnsw,
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(3),
            &PostgresSearchOptions::new().with_hnsw_ef_search(80),
        )
        .unwrap();
        assert_eq!(
            plan.settings,
            vec![
                ("hnsw.ef_search".into(), "80".into()),
                ("hnsw.iterative_scan".into(), "strict_order".into()),
                ("ivfflat.iterative_scan".into(), "off".into()),
            ]
        );
        assert!(plan
            .statement
            .contains("ORDER BY \"dense vector\" <=> $1::text::vector ASC"));
        // Exact on an HNSW field ranks by `+ 0` and sets nothing.
        let plan = build_search(
            &hnsw,
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(3),
            &PostgresSearchOptions::new().with_exact(true),
        )
        .unwrap();
        assert!(plan.settings.is_empty());
        assert!(plan.statement.contains(") + 0 ASC"));

        let flat = schema("cosine_distance", "flat");
        let search = |schema: &Schema, options: PostgresSearchOptions| {
            build_search(
                schema,
                &[1.0, 0.0, 0.0],
                &VectorSearchOptions::new(3),
                &options,
            )
        };
        assert!(search(&flat, PostgresSearchOptions::new().with_exact(false)).is_err());
        assert!(search(&flat, PostgresSearchOptions::new().with_hnsw_ef_search(10)).is_err());
        assert!(search(
            &hnsw,
            PostgresSearchOptions::new()
                .with_exact(true)
                .with_hnsw_ef_search(10)
        )
        .is_err());
        assert!(search(&hnsw, PostgresSearchOptions::new().with_ivfflat_probes(10)).is_err());
        assert!(search(&hnsw, PostgresSearchOptions::new().with_hnsw_ef_search(0)).is_err());
        assert!(search(
            &hnsw,
            PostgresSearchOptions::new().with_hnsw_ef_search(1001)
        )
        .is_err());
        assert!(search(
            &flat,
            PostgresSearchOptions::new().with_score_threshold(f64::NAN)
        )
        .is_err());
        let ivf = schema("cosine_distance", "ivf_flat");
        assert_eq!(
            search(&ivf, PostgresSearchOptions::new().with_ivfflat_probes(5))
                .unwrap()
                .settings[0],
            ("ivfflat.probes".to_string(), "5".to_string())
        );
    }

    #[test]
    fn search_rejects_bad_input_before_io() {
        let flat = schema("cosine_distance", "flat");
        assert!(build_search(
            &flat,
            &[1.0, 0.0],
            &VectorSearchOptions::new(3),
            &PostgresSearchOptions::new()
        )
        .is_err());
        assert!(build_search(
            &flat,
            &[f32::NAN, 0.0, 0.0],
            &VectorSearchOptions::new(3),
            &PostgresSearchOptions::new()
        )
        .is_err());
        assert!(build_search(
            &flat,
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(3).with_provider_filter("1 = 1; DROP TABLE x"),
            &PostgresSearchOptions::new()
        )
        .is_err());
        assert!(build_search(
            &flat,
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(3).with_vector_field_name("text"),
            &PostgresSearchOptions::new()
        )
        .is_err());
    }

    #[test]
    fn halfvec_queries_cast_to_halfvec() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            VectorStoreField::vector("v", 2).with_type("float16"),
        ])
        .unwrap();
        let schema = Schema::new("public", "t", definition, &HashMap::new(), false).unwrap();
        let plan = build_search(
            &schema,
            &[1.0, 0.5],
            &VectorSearchOptions::new(1),
            &PostgresSearchOptions::new(),
        )
        .unwrap();
        assert!(plan.statement.contains("\"v\" <=> $1::text::halfvec"));
    }

    #[test]
    fn filtered_reads_order_and_page_on_the_server() {
        let schema = schema("cosine_distance", "flat");
        let (statement, params, columns) = build_query(
            &schema,
            &PostgresQuery::new(5)
                .with_skip(2)
                .order_by("text", false)
                .with_filter(Filter::starts_with("text", "a").unwrap()),
        )
        .unwrap();
        assert_eq!(
            statement,
            "SELECT \"id\", \"text\" FROM \"public\".\"documents\" WHERE (\"text\" COLLATE \"C\" \
             LIKE $1::text ESCAPE '!') IS TRUE ORDER BY \"text\" DESC NULLS LAST, \"id\" ASC \
             LIMIT $2::bigint OFFSET $3::bigint"
        );
        assert_eq!(params.values[1..], [PgValue::Int(5), PgValue::Int(2)]);
        assert_eq!(columns.len(), 2);
        assert!(build_query(&schema, &PostgresQuery::new(0)).is_err());
        let (statement, _, _) =
            build_query(&schema, &PostgresQuery::new(1).with_include_vectors(true)).unwrap();
        assert!(statement.starts_with("SELECT \"id\", \"text\", \"dense vector\"::text FROM"));
    }

    #[test]
    fn upsert_statements_conflict_on_the_key() {
        let schema = schema("cosine_distance", "flat");
        let rows = prepare_rows(
            &schema,
            vec![json!({"id": "same", "text": "a", "embedding": [1, 2, 3], "extra": 1})],
        )
        .unwrap();
        let (statement, params) = upsert_statement(&schema, rows.into_iter().next().unwrap());
        assert_eq!(
            statement,
            "INSERT INTO \"public\".\"documents\" (\"id\", \"text\", \"dense vector\") VALUES \
             ($1::text, $2::text, $3::text::vector) ON CONFLICT (\"id\") DO UPDATE SET \"text\" \
             = EXCLUDED.\"text\", \"dense vector\" = EXCLUDED.\"dense vector\" RETURNING \"id\""
        );
        assert_eq!(params.values.len(), 3);
        // A missing non-key field is NULL; a missing key is an error.
        let rows = prepare_rows(&schema, vec![json!({"id": "x"})]).unwrap();
        assert_eq!(rows[0][1].1, PgValue::Null("text"));
        assert!(prepare_rows(&schema, vec![json!({"text": "x"})]).is_err());
        assert!(prepare_rows(&schema, vec![json!({"id": null})]).is_err());
        assert!(prepare_rows(&schema, vec![json!("not an object")]).is_err());
    }

    #[test]
    fn generated_keys_insert_without_a_conflict_target() {
        let definition =
            VectorStoreCollectionDefinition::new(
                vec![VectorStoreField::key("id").with_type("int")],
            )
            .unwrap();
        let schema = Schema::new("public", "t", definition, &HashMap::new(), true).unwrap();
        let rows = prepare_rows(
            &schema,
            vec![json!({}), json!({"id": null}), json!({"id": 7})],
        )
        .unwrap();
        let statements: Vec<String> = rows
            .into_iter()
            .map(|row| upsert_statement(&schema, row).0)
            .collect();
        assert_eq!(
            statements[0],
            "INSERT INTO \"public\".\"t\" DEFAULT VALUES RETURNING \"id\""
        );
        assert_eq!(statements[1], statements[0]);
        assert_eq!(
            statements[2],
            "INSERT INTO \"public\".\"t\" (\"id\") VALUES ($1::bigint) ON CONFLICT (\"id\") DO \
             UPDATE SET \"id\" = EXCLUDED.\"id\" RETURNING \"id\""
        );
    }

    #[test]
    fn settings_require_a_non_empty_connection_string() {
        assert!(PostgresSettings::default().require().is_err());
        assert!(PostgresSettings {
            connection_string: Some(SecretString::new("  "))
        }
        .require()
        .is_err());
        let explicit = PostgresSettings::load(Some(SecretString::new("host=x")));
        assert_eq!(
            explicit.connection_string.unwrap().expose_secret(),
            "host=x"
        );
        assert!(!format!(
            "{:?}",
            PostgresSettings::load(Some("password=hunter2".into()))
        )
        .contains("hunter2"));
    }

    #[tokio::test]
    async fn construction_does_no_io_and_close_is_enforced() {
        let store = PostgresStore::new("host=127.0.0.1 port=1 user=nobody").unwrap();
        let collection = store
            .collection("documents", definition("cosine_distance", "flat"))
            .unwrap();
        assert!(PostgresStore::new("not a = valid = conninfo '").is_err());
        assert!(store.clone().with_schema("").is_err());
        store.close().await;
        // A closed client fails fast, even for empty batches.
        let error = collection.upsert(vec![]).await.unwrap_err();
        assert!(error.to_string().contains("closed"), "{error}");
        assert!(collection.get(vec![], false).await.is_err());
        assert!(collection.delete(vec![]).await.is_err());
    }

    #[tokio::test]
    async fn a_store_collection_close_leaves_the_pool_open() {
        let store = PostgresStore::new("host=127.0.0.1 port=1 user=nobody").unwrap();
        let collection = store
            .collection("documents", definition("cosine_distance", "flat"))
            .unwrap();
        collection.close().await;
        // Still open: an empty batch needs no connection.
        assert_eq!(
            collection.upsert(vec![]).await.unwrap(),
            Vec::<Value>::new()
        );
        let standalone =
            PostgresCollection::builder("documents", definition("cosine_distance", "flat"))
                .connection_string("host=127.0.0.1 port=1 user=nobody")
                .build()
                .unwrap();
        standalone.close().await;
        assert!(standalone.upsert(vec![]).await.is_err());
    }
}
