//! The store and the collection.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use agent_framework_azure::TokenCredential;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::settings::{load_setting, SecretString};
use agent_framework_core::vectors::{
    FilterExpression, VectorCollection, VectorSearchOptions, VectorSearchResult, VectorStore,
    VectorStoreCollectionDefinition,
};
use serde_json::{Map, Value};

use crate::connection::{parse_connection_string, ClientHandle, Connector, TiberiusConnector};
use crate::sql::{
    key_identity, parse_value, prepare_key, prepare_value, quote_identifier, Cell, Column,
    FilterCompiler, Kind, Params, ResultKind, Schema, SqlParam, KEY_BATCH_SIZE, MAX_PARAMETERS,
};

fn config(message: impl Into<String>) -> Error {
    Error::Configuration(message.into())
}

fn invalid_response(message: impl Into<String>) -> Error {
    Error::service(message.into())
}

fn check_parameter_count(params: &[SqlParam]) -> Result<()> {
    if params.len() > MAX_PARAMETERS {
        return Err(config(format!(
            "SQL Server queries support at most {MAX_PARAMETERS} bound parameters."
        )));
    }
    Ok(())
}

/// Upstream's `_rows`: every row must have the projected width.
fn rows(rows: Vec<Vec<Cell>>, width: usize) -> Result<Vec<Vec<Cell>>> {
    if rows.iter().any(|row| row.len() != width) {
        return Err(invalid_response(
            "SQL Server returned a row with an unexpected column count.",
        ));
    }
    Ok(rows)
}

// region: settings

/// Connection settings — mirrors upstream's `SqlServerSettings`.
///
/// Resolved by [`SqlServerSettings::load`] with upstream's precedence:
/// explicit value, then `./.env`, then `SQL_SERVER_CONNECTION_STRING`. The
/// value is an ADO.NET connection string
/// (`Server=tcp:host,1433;Database=db;...`), kept in a [`SecretString`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SqlServerSettings {
    /// The ADO.NET connection string.
    pub connection_string: Option<SecretString>,
}

impl SqlServerSettings {
    /// The environment variable upstream's `env_prefix="SQL_SERVER_"`
    /// resolves.
    pub const CONNECTION_STRING_ENV: &'static str = "SQL_SERVER_CONNECTION_STRING";

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

/// Builds a [`SqlServerStore`] or a standalone [`SqlServerCollection`]'s
/// client.
#[derive(Clone, Default)]
pub struct SqlServerClientOptions {
    connection_string: Option<SecretString>,
    credential: Option<Arc<dyn TokenCredential>>,
    query_timeout: Option<Duration>,
}

impl std::fmt::Debug for SqlServerClientOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqlServerClientOptions")
            .field("connection_string", &self.connection_string)
            .field("credential", &self.credential.is_some())
            .field("query_timeout", &self.query_timeout)
            .finish()
    }
}

impl SqlServerClientOptions {
    fn connector(&self) -> Result<Arc<dyn Connector>> {
        let settings = SqlServerSettings::load(self.connection_string.clone());
        let (config, auth) = parse_connection_string(settings.require()?, self.credential.clone())?;
        Ok(Arc::new(TiberiusConnector {
            config,
            auth,
            query_timeout: self.query_timeout,
        }))
    }
}

// endregion

// region: the store

/// A SQL Server / Azure SQL database schema as a [`VectorStore`] — mirrors
/// upstream's `SqlServerStore`.
///
/// Construction does no I/O. The store owns its connections: each
/// operation opens one, runs in one transaction, and closes it. Collections
/// from [`Self::collection`] share the store's client; [`Self::close`]
/// waits for in-flight operations and then refuses new ones, for the store
/// and every collection it handed out.
#[derive(Clone)]
pub struct SqlServerStore {
    client: Arc<ClientHandle>,
    schema: String,
}

impl std::fmt::Debug for SqlServerStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqlServerStore")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

/// Builds a [`SqlServerStore`].
#[derive(Debug, Clone, Default)]
pub struct SqlServerStoreBuilder {
    options: SqlServerClientOptions,
    schema: Option<String>,
}

impl SqlServerStoreBuilder {
    /// The ADO.NET connection string. Without one, settings resolve from
    /// `./.env` or `SQL_SERVER_CONNECTION_STRING`.
    pub fn connection_string(mut self, connection_string: impl Into<SecretString>) -> Self {
        self.options.connection_string = Some(connection_string.into());
        self
    }

    /// Authenticate with a Microsoft Entra token from `credential`
    /// (audience [`SQL_SERVER_SCOPE`](crate::SQL_SERVER_SCOPE)), overriding
    /// the connection string's own login and `Authentication=` key.
    pub fn token_credential(mut self, credential: Arc<dyn TokenCredential>) -> Self {
        self.options.credential = Some(credential);
        self
    }

    /// Bound each statement; `Duration::ZERO` disables the bound, as
    /// upstream's `query_timeout=0` does. Unset uses no bound.
    pub fn query_timeout(mut self, timeout: Duration) -> Self {
        self.options.query_timeout = Some(timeout);
        self
    }

    /// The existing schema every collection lives in (default `dbo`).
    pub fn schema(mut self, schema: impl Into<String>) -> Self {
        self.schema = Some(schema.into());
        self
    }

    /// Validate and build. No I/O.
    pub fn build(self) -> Result<SqlServerStore> {
        let schema = self.schema.unwrap_or_else(|| "dbo".into());
        quote_identifier(&schema)?;
        Ok(SqlServerStore {
            client: Arc::new(ClientHandle::new(self.options.connector()?)),
            schema,
        })
    }
}

impl SqlServerStore {
    /// A store for `connection_string` in schema `dbo`.
    pub fn new(connection_string: impl Into<SecretString>) -> Result<Self> {
        Self::builder().connection_string(connection_string).build()
    }

    /// A store from `./.env` or `SQL_SERVER_CONNECTION_STRING`.
    pub fn from_env() -> Result<Self> {
        Self::builder().build()
    }

    /// The full set of options.
    pub fn builder() -> SqlServerStoreBuilder {
        SqlServerStoreBuilder::default()
    }

    #[cfg(test)]
    pub(crate) fn with_connector(connector: Arc<dyn Connector>, schema: &str) -> Self {
        Self {
            client: Arc::new(ClientHandle::new(connector)),
            schema: schema.into(),
        }
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
    ) -> Result<SqlServerCollection> {
        self.collection_builder(name, definition).build()
    }

    /// Describe one table with SQL Server-specific options. No I/O.
    pub fn collection_builder(
        &self,
        name: impl Into<String>,
        definition: VectorStoreCollectionDefinition,
    ) -> SqlServerCollectionBuilder {
        let mut builder = SqlServerCollection::builder(name, definition);
        builder.schema = self.schema.clone();
        builder.shared = Some(Arc::clone(&self.client));
        builder
    }

    /// Drop one table in this schema — upstream's
    /// `ensure_collection_deleted(collection_name)`. Absent tables are fine.
    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        let statement = format!(
            "DROP TABLE IF EXISTS {}.{}",
            quote_identifier(&self.schema)?,
            quote_identifier(name)?
        );
        self.client
            .run(move |session| {
                Box::pin(async move {
                    session.query(&statement, &[]).await?;
                    Ok(())
                })
            })
            .await
    }

    /// Wait for in-flight operations, then refuse new ones.
    pub async fn close(&self) {
        self.client.close().await;
    }
}

#[async_trait::async_trait]
impl VectorStore for SqlServerStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    /// Base tables in this schema.
    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let schema = self.schema.clone();
        let found = self
            .client
            .run(move |session| {
                Box::pin(async move {
                    session
                        .query(
                            "SELECT t.name FROM sys.tables AS t JOIN sys.schemas AS s ON \
                             s.schema_id = t.schema_id WHERE s.name = @P1 ORDER BY t.name",
                            &[SqlParam::Str(schema)],
                        )
                        .await
                })
            })
            .await?;
        rows(found, 1)?
            .into_iter()
            .map(|mut row| match row.pop() {
                Some(Cell::Str(name)) => Ok(name),
                _ => Err(invalid_response(
                    "SQL Server returned a non-string table name.",
                )),
            })
            .collect()
    }

    async fn collection_exists(&self, name: &str) -> Result<bool> {
        exists(&self.client, &self.schema, name).await
    }
}

async fn exists(client: &ClientHandle, schema: &str, table: &str) -> Result<bool> {
    let params = vec![SqlParam::Str(schema.into()), SqlParam::Str(table.into())];
    let found = client
        .run(move |session| {
            Box::pin(async move {
                session
                    .query(
                        "SELECT 1 FROM sys.tables AS t JOIN sys.schemas AS s ON s.schema_id = \
                         t.schema_id WHERE s.name = @P1 AND t.name = @P2",
                        &params,
                    )
                    .await
            })
        })
        .await?;
    Ok(!found.is_empty())
}

// endregion

// region: the collection

/// Builds a [`SqlServerCollection`].
pub struct SqlServerCollectionBuilder {
    name: String,
    definition: VectorStoreCollectionDefinition,
    schema: String,
    shared: Option<Arc<ClientHandle>>,
    options: SqlServerClientOptions,
    auto_generated_key: bool,
}

impl std::fmt::Debug for SqlServerCollectionBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqlServerCollectionBuilder")
            .field("name", &self.name)
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl SqlServerCollectionBuilder {
    /// The existing schema holding the table (default `dbo`).
    pub fn schema(mut self, schema: impl Into<String>) -> Self {
        self.schema = schema.into();
        self
    }

    /// The connection string for a standalone collection. Ignored for a
    /// store's collection.
    pub fn connection_string(mut self, connection_string: impl Into<SecretString>) -> Self {
        self.options.connection_string = Some(connection_string.into());
        self
    }

    /// A Microsoft Entra credential for a standalone collection. Ignored for
    /// a store's collection.
    pub fn token_credential(mut self, credential: Arc<dyn TokenCredential>) -> Self {
        self.options.credential = Some(credential);
        self
    }

    /// A per-statement timeout for a standalone collection. Ignored for a
    /// store's collection.
    pub fn query_timeout(mut self, timeout: Duration) -> Self {
        self.options.query_timeout = Some(timeout);
        self
    }

    /// Let the server generate the key when a record omits it (or sets it
    /// to `null`): `IDENTITY(1,1)` for `int`, `NEWID()` for `UUID` and
    /// `str`. Mirrors upstream's `is_auto_generated` key fields. An explicit
    /// key updates an existing row; for an `IDENTITY` key it cannot create a
    /// new one.
    pub fn auto_generated_key(mut self, generated: bool) -> Self {
        self.auto_generated_key = generated;
        self
    }

    /// Validate and build. No I/O.
    pub fn build(self) -> Result<SqlServerCollection> {
        let schema = Schema::new(
            &self.schema,
            &self.name,
            self.definition,
            self.auto_generated_key,
        )?;
        let (client, owned) = match self.shared {
            Some(client) => (client, false),
            None => (Arc::new(ClientHandle::new(self.options.connector()?)), true),
        };
        Ok(SqlServerCollection {
            schema: Arc::new(schema),
            client,
            owned,
        })
    }
}

/// A filtered, ordered, paged read — upstream's
/// `get(filter=..., top=..., skip=..., order_by=..., include_vectors=...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct SqlServerQuery {
    /// A portable filter; `None` reads every row.
    pub filter: Option<FilterExpression>,
    /// Page size (default 10). Must be greater than zero.
    pub top: usize,
    /// Rows to skip.
    pub skip: usize,
    /// `(logical field name, ascending)` pairs. Nulls sort last; the key
    /// breaks ties ascending unless it is listed.
    pub order_by: Vec<(String, bool)>,
    /// Whether vector fields are returned.
    pub include_vectors: bool,
}

impl Default for SqlServerQuery {
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

impl SqlServerQuery {
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

/// One table with native `VECTOR` columns as a [`VectorCollection`] —
/// mirrors upstream's `SqlServerCollection`.
///
/// Search is exact (`VECTOR_DISTANCE`); filters and the score threshold are
/// applied before paging. Every operation is one transaction: a batch
/// upsert writes all records or none.
#[derive(Clone)]
pub struct SqlServerCollection {
    schema: Arc<Schema>,
    client: Arc<ClientHandle>,
    /// Standalone (true) or borrowed from a store (false).
    owned: bool,
}

impl std::fmt::Debug for SqlServerCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqlServerCollection")
            .field("schema", &self.schema.schema)
            .field("table", &self.schema.table_name)
            .finish_non_exhaustive()
    }
}

impl SqlServerCollection {
    /// A standalone collection builder; the collection owns its client.
    pub fn builder(
        name: impl Into<String>,
        definition: VectorStoreCollectionDefinition,
    ) -> SqlServerCollectionBuilder {
        SqlServerCollectionBuilder {
            name: name.into(),
            definition,
            schema: "dbo".into(),
            shared: None,
            options: SqlServerClientOptions::default(),
            auto_generated_key: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_connector(
        connector: Arc<dyn Connector>,
        name: &str,
        definition: VectorStoreCollectionDefinition,
        auto_generated_key: bool,
    ) -> Result<Self> {
        Ok(Self {
            schema: Arc::new(Schema::new("dbo", name, definition, auto_generated_key)?),
            client: Arc::new(ClientHandle::new(connector)),
            owned: true,
        })
    }

    /// The schema holding the table.
    pub fn schema_name(&self) -> &str {
        &self.schema.schema
    }

    /// Close a standalone collection's client; a store's collection leaves
    /// the store's client open.
    pub async fn close(&self) {
        if self.owned {
            self.client.close().await;
        }
    }

    /// A filtered, ordered, paged read — upstream's filtered `get`.
    pub async fn query(&self, query: &SqlServerQuery) -> Result<Vec<Value>> {
        self.client.ensure_open().await?;
        let (statement, params, width) = build_query(&self.schema, query)?;
        let found = self
            .client
            .run(move |session| Box::pin(async move { session.query(&statement, &params).await }))
            .await?;
        let columns = self.schema.selected(query.include_vectors);
        rows(found, width)?
            .into_iter()
            .map(|row| record(&columns, row))
            .collect()
    }

    /// [`VectorCollection::search`] with a score threshold: a minimum for
    /// `cosine_similarity` / `dot_prod`, a maximum otherwise, in the
    /// metric's raw units. Applied before paging.
    pub async fn search_with_threshold(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
        score_threshold: Option<f64>,
    ) -> Result<Vec<VectorSearchResult>> {
        options.validate()?;
        self.client.ensure_open().await?;
        let plan = build_search(&self.schema, &vector, options, score_threshold)?;
        let statement = plan.statement.clone();
        let params = plan.params.values.clone();
        let found = self
            .client
            .run(move |session| Box::pin(async move { session.query(&statement, &params).await }))
            .await?;
        let columns: Vec<&Column> = plan
            .columns
            .iter()
            .map(|&i| &self.schema.columns[i])
            .collect();
        rows(found, columns.len() + 1)?
            .into_iter()
            .map(|mut row| {
                let distance = match row.pop() {
                    Some(Cell::F64(d)) if d.is_finite() => d,
                    Some(Cell::I64(d)) => d as f64,
                    _ => {
                        return Err(invalid_response(
                            "SQL Server returned a non-finite vector distance.",
                        ))
                    }
                };
                let score = match plan.result {
                    ResultKind::Distance => distance,
                    ResultKind::Similarity => 1.0 - distance,
                    ResultKind::Negative => -distance,
                };
                Ok(VectorSearchResult {
                    record: record(&columns, row)?,
                    score: Some(score),
                    score_kind: None,
                })
            })
            .collect()
    }
}

/// Decode one row into a record keyed by logical field name.
fn record(columns: &[&Column], row: Vec<Cell>) -> Result<Value> {
    let mut out = Map::new();
    for (column, cell) in columns.iter().zip(row) {
        out.insert(column.name.clone(), parse_value(column, cell)?);
    }
    Ok(Value::Object(out))
}

fn projection(columns: &[&Column], alias: &str) -> String {
    columns
        .iter()
        .map(|c| c.projection(alias))
        .collect::<Vec<_>>()
        .join(", ")
}

fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Upstream's filtered `_inner_get`, minus the I/O. Returns the statement,
/// its parameters, and the projected width.
fn build_query(schema: &Schema, query: &SqlServerQuery) -> Result<(String, Vec<SqlParam>, usize)> {
    if query.top == 0 {
        return Err(config("query `top` must be greater than zero"));
    }
    let columns = schema.selected(query.include_vectors);
    let mut params = Params::default();
    let filter = match &query.filter {
        Some(filter) => FilterCompiler::new(schema, "", &mut params).compile(filter)?,
        None => "1 = 1".into(),
    };
    let order = schema.order_by_sql(&query.order_by)?;
    let skip = params.bind(SqlParam::I64(to_i64(query.skip)))?;
    let top = params.bind(SqlParam::I64(to_i64(query.top)))?;
    let statement = format!(
        "SELECT {} FROM {} WHERE {filter} ORDER BY {order} OFFSET {skip} ROWS FETCH NEXT {top} \
         ROWS ONLY",
        projection(&columns, ""),
        schema.table
    );
    Ok((statement, params.values, columns.len()))
}

#[derive(Debug)]
struct SearchPlan {
    statement: String,
    params: Params,
    columns: Vec<usize>,
    result: ResultKind,
}

/// Upstream's `_inner_search`, minus the I/O.
fn build_search(
    schema: &Schema,
    vector: &[f32],
    options: &VectorSearchOptions,
    score_threshold: Option<f64>,
) -> Result<SearchPlan> {
    if options.provider_filter.is_some() {
        return Err(config(
            "SQL Server does not accept a provider_filter: verbatim T-SQL would bypass \
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
    let (dimensions, metric, result) = column.vector.expect("a vector column");
    let query_value = Value::Array(
        vector
            .iter()
            .map(|v| serde_json::Number::from_f64(f64::from(*v)).map(Value::Number))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| config("query vector elements must be finite"))?,
    );
    let mut params = Params::default();
    let query_vector = params.bind(prepare_value(column, &query_value)?)?;
    let filter = match &options.filter {
        Some(filter) => FilterCompiler::new(schema, "t.", &mut params).compile(filter)?,
        None => "1 = 1".into(),
    };
    let columns: Vec<usize> = schema
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| options.include_vectors || c.vector.is_none())
        .map(|(i, _)| i)
        .collect();
    let selected: Vec<&Column> = columns.iter().map(|&i| &schema.columns[i]).collect();
    let vector_column = format!("t.{}", column.quoted());
    let key_column = format!("t.{}", schema.key_column().quoted());
    let mut statement = format!(
        "SELECT {}, d.[distance] FROM {} AS t CROSS APPLY (VALUES (VECTOR_DISTANCE('{metric}', \
         CAST({query_vector} AS VECTOR({dimensions})), {vector_column}))) AS d([distance]) WHERE \
         {vector_column} IS NOT NULL AND ({filter}) AND d.[distance] IS NOT NULL",
        projection(&selected, "t."),
        schema.table
    );
    if let Some(threshold) = score_threshold {
        if !threshold.is_finite() {
            return Err(config("score_threshold must be a finite number."));
        }
        let cutoff = match result {
            ResultKind::Distance => threshold,
            ResultKind::Similarity => 1.0 - threshold,
            ResultKind::Negative => -threshold,
        };
        let cutoff = params.bind(SqlParam::F64(cutoff))?;
        statement.push_str(&format!(" AND d.[distance] <= {cutoff}"));
    }
    let skip = params.bind(SqlParam::I64(to_i64(options.skip)))?;
    let top = params.bind(SqlParam::I64(to_i64(options.top)))?;
    statement.push_str(&format!(
        " ORDER BY d.[distance] ASC, {key_column} ASC OFFSET {skip} ROWS FETCH NEXT {top} ROWS \
         ONLY"
    ));
    Ok(SearchPlan {
        statement,
        params,
        columns,
        result,
    })
}

/// One record, adapted: `(column index, value)` in definition order, with
/// a generated key omitted.
type PreparedRow = Vec<(usize, SqlParam)>;

fn prepare_rows(schema: &Schema, records: Vec<Value>) -> Result<Vec<PreparedRow>> {
    let mut prepared = Vec::with_capacity(records.len());
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
        prepared.push(row);
    }
    Ok(prepared)
}

/// Upstream's `_inner_upsert` loop body: lock-and-check the key, then
/// `UPDATE ... OUTPUT` or `INSERT ... OUTPUT`.
async fn upsert_rows(
    session: &mut dyn crate::connection::Session,
    schema: &Schema,
    rows_in: Vec<PreparedRow>,
) -> Result<Vec<Value>> {
    let key = schema.key_column();
    let key_column = key.quoted();
    let identity = schema.auto_generated_key && key.kind == Some(Kind::Int);
    let mut keys = Vec::with_capacity(rows_in.len());
    for row in rows_in {
        let key_value = row
            .iter()
            .find(|(index, _)| *index == schema.key)
            .map(|(_, value)| value.clone());
        if let Some(key_value) = &key_value {
            let existing = session
                .query(
                    &format!(
                        "SELECT {key_column} FROM {} WITH (UPDLOCK, HOLDLOCK) WHERE {key_column} \
                         = @P1",
                        schema.table
                    ),
                    std::slice::from_ref(key_value),
                )
                .await?;
            if !existing.is_empty() {
                let mut params = Params::default();
                let mut assignments = Vec::new();
                for (index, value) in &row {
                    if *index == schema.key {
                        continue;
                    }
                    let placeholder = params.bind(value.clone())?;
                    assignments.push(format!(
                        "{} = {placeholder}",
                        schema.columns[*index].quoted()
                    ));
                }
                if assignments.is_empty() {
                    assignments.push(format!("{key_column} = {key_column}"));
                }
                let key_placeholder = params.bind(key_value.clone())?;
                check_parameter_count(&params.values)?;
                let returned = session
                    .query(
                        &format!(
                            "UPDATE {} SET {} OUTPUT INSERTED.{key_column} WHERE {key_column} = \
                             {key_placeholder}",
                            schema.table,
                            assignments.join(", ")
                        ),
                        &params.values,
                    )
                    .await?;
                let cell = returned
                    .into_iter()
                    .next()
                    .and_then(|row| row.into_iter().next())
                    .ok_or_else(|| invalid_response("SQL Server did not return an updated key."))?;
                keys.push(parse_value(key, cell)?);
                continue;
            }
            if identity {
                return Err(config(
                    "SQL Server IDENTITY columns cannot insert an explicit new key; omit it to \
                     generate a key.",
                ));
            }
        }
        let returned = if row.is_empty() {
            session
                .query(
                    &format!(
                        "INSERT INTO {} OUTPUT INSERTED.{key_column} DEFAULT VALUES",
                        schema.table
                    ),
                    &[],
                )
                .await?
        } else {
            let mut params = Params::default();
            let mut names = Vec::with_capacity(row.len());
            let mut markers = Vec::with_capacity(row.len());
            for (index, value) in row {
                names.push(schema.columns[index].quoted());
                markers.push(params.bind(value)?);
            }
            check_parameter_count(&params.values)?;
            session
                .query(
                    &format!(
                        "INSERT INTO {} ({}) OUTPUT INSERTED.{key_column} VALUES ({})",
                        schema.table,
                        names.join(", "),
                        markers.join(", ")
                    ),
                    &params.values,
                )
                .await?
        };
        let cell = returned
            .into_iter()
            .next()
            .and_then(|row| row.into_iter().next())
            .ok_or_else(|| invalid_response("SQL Server did not return an inserted key."))?;
        keys.push(parse_value(key, cell)?);
    }
    Ok(keys)
}

#[async_trait::async_trait]
impl VectorCollection for SqlServerCollection {
    fn name(&self) -> &str {
        &self.schema.table_name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.schema.definition
    }

    /// Create the table and requested scalar indexes in the existing
    /// schema, never altering an existing table.
    async fn ensure_collection_exists(&self) -> Result<()> {
        self.client.ensure_open().await?;
        let schema = Arc::clone(&self.schema);
        self.client
            .run(move |session| {
                Box::pin(async move {
                    let create = format!(
                        "IF NOT EXISTS (SELECT 1 FROM sys.tables AS t JOIN sys.schemas AS s ON \
                         s.schema_id = t.schema_id WHERE s.name = @P1 AND t.name = @P2) BEGIN \
                         CREATE TABLE {} ({}) END",
                        schema.table,
                        schema.column_definitions()
                    );
                    session
                        .query(
                            &create,
                            &[
                                SqlParam::Str(schema.schema.clone()),
                                SqlParam::Str(schema.table_name.clone()),
                            ],
                        )
                        .await?;
                    for column in &schema.columns {
                        if column.role != agent_framework_core::vectors::FieldType::Data
                            || !column.indexed
                        {
                            continue;
                        }
                        let index_name = schema.index_name(column);
                        let statement = format!(
                            "IF NOT EXISTS (SELECT 1 FROM sys.indexes AS i JOIN sys.tables AS t \
                             ON t.object_id = i.object_id JOIN sys.schemas AS s ON s.schema_id = \
                             t.schema_id WHERE s.name = @P1 AND t.name = @P2 AND i.name = @P3) \
                             CREATE INDEX {} ON {} ({})",
                            quote_identifier(&index_name)?,
                            schema.table,
                            column.quoted()
                        );
                        session
                            .query(
                                &statement,
                                &[
                                    SqlParam::Str(schema.schema.clone()),
                                    SqlParam::Str(schema.table_name.clone()),
                                    SqlParam::Str(index_name),
                                ],
                            )
                            .await?;
                    }
                    Ok(())
                })
            })
            .await
    }

    async fn collection_exists(&self) -> Result<bool> {
        exists(&self.client, &self.schema.schema, &self.schema.table_name).await
    }

    /// Drop only this table; never the schema or other tables.
    async fn ensure_collection_deleted(&self) -> Result<()> {
        let statement = format!("DROP TABLE IF EXISTS {}", self.schema.table);
        self.client
            .run(move |session| {
                Box::pin(async move {
                    session.query(&statement, &[]).await?;
                    Ok(())
                })
            })
            .await
    }

    /// Insert or update every record in one transaction, in input order;
    /// a repeated key is written twice, the last write winning.
    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        self.client.ensure_open().await?;
        let prepared = prepare_rows(&self.schema, records)?;
        if prepared.is_empty() {
            return Ok(Vec::new());
        }
        let schema = Arc::clone(&self.schema);
        self.client
            .run(move |session| {
                Box::pin(async move { upsert_rows(session, &schema, prepared).await })
            })
            .await
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        self.client.ensure_open().await?;
        let schema = Arc::clone(&self.schema);
        let prepared = keys
            .iter()
            .map(|key| prepare_key(&schema, key))
            .collect::<Result<Vec<_>>>()?;
        if prepared.is_empty() {
            return Ok(Vec::new());
        }
        let columns: Vec<&Column> = schema.selected(include_vectors);
        let select = format!("SELECT {} FROM {}", projection(&columns, ""), schema.table);
        let key = schema.key_column();
        let key_position = columns
            .iter()
            .position(|c| c.name == key.name)
            .expect("the key is always selected");
        let width = columns.len();
        let batches: Vec<(String, Vec<SqlParam>)> = prepared
            .chunks(KEY_BATCH_SIZE)
            .map(|batch| {
                let markers: Vec<String> = (1..=batch.len()).map(|n| format!("@P{n}")).collect();
                (
                    format!(
                        "{select} WHERE {} IN ({})",
                        key.quoted(),
                        markers.join(", ")
                    ),
                    batch.to_vec(),
                )
            })
            .collect();
        let found = self
            .client
            .run(move |session| {
                Box::pin(async move {
                    let mut all = Vec::new();
                    for (statement, params) in batches {
                        all.extend(session.query(&statement, &params).await?);
                    }
                    Ok(all)
                })
            })
            .await?;
        let mut by_key: HashMap<String, Value> = HashMap::new();
        for row in rows(found, width)? {
            let identity = parse_value(key, row[key_position].clone())?;
            by_key.insert(identity.to_string(), record(&columns, row)?);
        }
        Ok(prepared
            .iter()
            .map(|k| by_key.get(&key_identity(k).to_string()).cloned())
            .collect())
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        self.client.ensure_open().await?;
        let prepared = keys
            .iter()
            .map(|key| prepare_key(&self.schema, key))
            .collect::<Result<Vec<_>>>()?;
        if prepared.is_empty() {
            return Ok(());
        }
        let key = self.schema.key_column().quoted();
        let table = self.schema.table.clone();
        let batches: Vec<(String, Vec<SqlParam>)> = prepared
            .chunks(KEY_BATCH_SIZE)
            .map(|batch| {
                let markers: Vec<String> = (1..=batch.len()).map(|n| format!("@P{n}")).collect();
                (
                    format!(
                        "DELETE FROM {table} WHERE {key} IN ({})",
                        markers.join(", ")
                    ),
                    batch.to_vec(),
                )
            })
            .collect();
        self.client
            .run(move |session| {
                Box::pin(async move {
                    for (statement, params) in batches {
                        session.query(&statement, &params).await?;
                    }
                    Ok(())
                })
            })
            .await
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_with_threshold(vector, options, None).await
    }
}

// endregion

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::fake::{Event, FakeConnector};
    use agent_framework_core::vectors::{DistanceFunction, Filter, VectorStoreField};
    use serde_json::json;

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str").indexed(),
            VectorStoreField::data("tags").with_type("list"),
            VectorStoreField::vector("embedding", 3).with_storage_name("dense vector"),
        ])
        .unwrap()
    }

    fn collection(fake: &FakeConnector) -> SqlServerCollection {
        SqlServerCollection::with_connector(
            Arc::new(fake.clone()),
            "documents",
            definition(),
            false,
        )
        .unwrap()
    }

    fn s(text: &str) -> Cell {
        Cell::Str(text.into())
    }

    #[tokio::test]
    async fn create_check_and_drop_use_quoted_identifiers_and_bound_catalog_values() {
        let fake = FakeConnector::default();
        let collection = collection(&fake);
        collection.ensure_collection_exists().await.unwrap();
        let queries = fake.queries();
        assert_eq!(queries.len(), 2);
        assert!(queries[0].0.starts_with(
            "IF NOT EXISTS (SELECT 1 FROM sys.tables AS t JOIN sys.schemas AS s ON s.schema_id = \
             t.schema_id WHERE s.name = @P1 AND t.name = @P2) BEGIN CREATE TABLE \
             [dbo].[documents] ([id] NVARCHAR(450)"
        ));
        assert!(queries[0].0.ends_with("[dense vector] VECTOR(3) NULL) END"));
        assert_eq!(
            queries[0].1,
            vec![
                SqlParam::Str("dbo".into()),
                SqlParam::Str("documents".into())
            ]
        );
        assert!(queries[1].0.contains("CREATE INDEX [af_sql_"));
        assert!(queries[1].0.ends_with("ON [dbo].[documents] ([text])"));
        assert_eq!(queries[1].1.len(), 3);

        fake.respond(vec![vec![Cell::I64(1)]]);
        assert!(collection.collection_exists().await.unwrap());
        assert!(!collection.collection_exists().await.unwrap());
        collection.ensure_collection_deleted().await.unwrap();
        assert_eq!(
            fake.queries().last().unwrap().0,
            "DROP TABLE IF EXISTS [dbo].[documents]"
        );
    }

    #[tokio::test]
    async fn batch_upsert_preserves_duplicate_key_order_and_binds_json() {
        let fake = FakeConnector::default();
        let collection = collection(&fake);
        // First "a": not found → insert. "b": found → update. Second "a":
        // found → update.
        fake.respond(vec![]);
        fake.respond(vec![vec![s("a")]]);
        fake.respond(vec![vec![s("b")]]);
        fake.respond(vec![vec![s("b")]]);
        fake.respond(vec![vec![s("a")]]);
        fake.respond(vec![vec![s("a")]]);
        let keys = collection
            .upsert(vec![
                json!({"id": "a", "text": "one", "tags": ["x", 1], "embedding": [1, 0, 0]}),
                json!({"id": "b", "text": "two"}),
                json!({"id": "a", "text": "three"}),
            ])
            .await
            .unwrap();
        assert_eq!(keys, vec![json!("a"), json!("b"), json!("a")]);
        let queries = fake.queries();
        assert_eq!(
            queries[0].0,
            "SELECT [id] FROM [dbo].[documents] WITH (UPDLOCK, HOLDLOCK) WHERE [id] = @P1"
        );
        assert_eq!(
            queries[1].0,
            "INSERT INTO [dbo].[documents] ([id], [text], [tags], [dense vector]) OUTPUT \
             INSERTED.[id] VALUES (@P1, @P2, @P3, @P4)"
        );
        assert_eq!(
            queries[1].1,
            vec![
                SqlParam::Str("a".into()),
                SqlParam::Str("one".into()),
                SqlParam::Str("[\"x\",1]".into()),
                SqlParam::Str("[1.0,0.0,0.0]".into()),
            ]
        );
        assert_eq!(
            queries[3].0,
            "UPDATE [dbo].[documents] SET [text] = @P1, [tags] = @P2, [dense vector] = @P3 \
             OUTPUT INSERTED.[id] WHERE [id] = @P4"
        );
        assert_eq!(queries[3].1[1], SqlParam::Null(crate::sql::NullKind::Str));
        // One transaction for the whole batch.
        let events = fake.events();
        assert_eq!(events.iter().filter(|e| **e == Event::Connect).count(), 1);
    }

    #[tokio::test]
    async fn generated_keys_insert_without_a_key_or_lookup() {
        let fake = FakeConnector::default();
        let definition =
            VectorStoreCollectionDefinition::new(
                vec![VectorStoreField::key("id").with_type("int")],
            )
            .unwrap();
        let collection =
            SqlServerCollection::with_connector(Arc::new(fake.clone()), "t", definition, true)
                .unwrap();
        fake.respond(vec![vec![Cell::I64(7)]]);
        assert_eq!(
            collection.upsert(vec![json!({})]).await.unwrap(),
            vec![json!(7)]
        );
        assert_eq!(
            fake.queries(),
            vec![(
                "INSERT INTO [dbo].[t] OUTPUT INSERTED.[id] DEFAULT VALUES".to_string(),
                vec![]
            )]
        );
        // An explicit key that does not exist cannot become a new IDENTITY row.
        let error = collection
            .upsert(vec![json!({"id": 99})])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("IDENTITY"), "{error}");
        assert!(fake.events().contains(&Event::Batch(
            "IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION".into()
        )));
    }

    #[tokio::test]
    async fn a_missing_returned_key_is_an_invalid_response() {
        let fake = FakeConnector::default();
        let collection = collection(&fake);
        let error = collection
            .upsert(vec![json!({"id": "a"})])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("did not return an inserted key"));
    }

    #[tokio::test]
    async fn get_preserves_input_order_and_optional_vectors() {
        let fake = FakeConnector::default();
        let collection = collection(&fake);
        fake.respond(vec![
            vec![s("b"), s("two"), Cell::Null, s("[0.0,1.0,0.0]")],
            vec![s("a"), s("one"), s("[1]"), Cell::Null],
        ]);
        let got = collection
            .get(vec![json!("a"), json!("missing"), json!("b")], true)
            .await
            .unwrap();
        assert_eq!(
            got,
            vec![
                Some(json!({"id": "a", "text": "one", "tags": [1], "embedding": null})),
                None,
                Some(json!({"id": "b", "text": "two", "tags": null, "embedding": [0.0, 1.0, 0.0]})),
            ]
        );
        let (statement, params) = &fake.queries()[0];
        assert_eq!(
            statement,
            "SELECT [id], [text], [tags], CAST([dense vector] AS NVARCHAR(MAX)) FROM \
             [dbo].[documents] WHERE [id] IN (@P1, @P2, @P3)"
        );
        assert_eq!(params.len(), 3);
        // A row of the wrong width is refused.
        fake.respond(vec![vec![s("a")]]);
        assert!(collection.get(vec![json!("a")], false).await.is_err());
    }

    #[tokio::test]
    async fn large_key_batches_stay_under_the_parameter_budget() {
        let fake = FakeConnector::default();
        let collection = collection(&fake);
        let keys: Vec<Value> = (0..2500).map(|n| json!(format!("k{n}"))).collect();
        collection.get(keys.clone(), false).await.unwrap();
        let sizes: Vec<usize> = fake.queries().iter().map(|(_, p)| p.len()).collect();
        assert_eq!(sizes, vec![1000, 1000, 500]);
        collection.delete(keys).await.unwrap();
        let deletes: Vec<(String, Vec<SqlParam>)> = fake.queries().into_iter().skip(3).collect();
        assert_eq!(deletes.len(), 3);
        assert!(deletes[2]
            .0
            .starts_with("DELETE FROM [dbo].[documents] WHERE [id] IN (@P1, "));
        // Each operation is one transaction.
        assert_eq!(
            fake.events()
                .iter()
                .filter(|e| **e == Event::Connect)
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn filtered_listing_orders_and_pages_on_the_server() {
        let fake = FakeConnector::default();
        let collection = collection(&fake);
        fake.respond(vec![vec![s("a"), s("one"), Cell::Null]]);
        let found = collection
            .query(
                &SqlServerQuery::new(5)
                    .with_skip(10)
                    .order_by("text", false)
                    .with_filter(Filter::starts_with("text", "o").unwrap()),
            )
            .await
            .unwrap();
        assert_eq!(found, vec![json!({"id": "a", "text": "one", "tags": null})]);
        let (statement, params) = &fake.queries()[0];
        assert_eq!(
            statement,
            "SELECT [id], [text], [tags] FROM [dbo].[documents] WHERE ([text] IS NOT NULL AND \
             [text] COLLATE Latin1_General_100_BIN2 LIKE @P1 ESCAPE '!') ORDER BY CASE WHEN \
             [text] IS NULL THEN 1 ELSE 0 END, [text] DESC, [id] ASC OFFSET @P2 ROWS FETCH NEXT \
             @P3 ROWS ONLY"
        );
        assert_eq!(params[1..], [SqlParam::I64(10), SqlParam::I64(5)]);
        assert!(build_query(&collection.schema, &SqlServerQuery::new(0)).is_err());
    }

    #[tokio::test]
    async fn exact_search_filters_and_thresholds_before_paging() {
        let fake = FakeConnector::default();
        let collection = collection(&fake);
        fake.respond(vec![vec![s("a"), s("one"), Cell::Null, Cell::F64(0.25)]]);
        let hits = collection
            .search_with_threshold(
                vec![1.0, 0.0, 0.0],
                &VectorSearchOptions::new(2)
                    .with_skip(1)
                    .with_filter(Filter::eq("text", "one").unwrap()),
                Some(0.5),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].score, Some(0.25));
        let (statement, params) = &fake.queries()[0];
        assert_eq!(
            statement,
            "SELECT t.[id], t.[text], t.[tags], d.[distance] FROM [dbo].[documents] AS t CROSS \
             APPLY (VALUES (VECTOR_DISTANCE('cosine', CAST(@P1 AS VECTOR(3)), t.[dense \
             vector]))) AS d([distance]) WHERE t.[dense vector] IS NOT NULL AND ((t.[text] IS \
             NOT NULL AND CONVERT(VARBINARY(MAX), t.[text]) = CONVERT(VARBINARY(MAX), \
             CONVERT(NVARCHAR(MAX), @P2)))) AND d.[distance] IS NOT NULL AND d.[distance] <= @P3 \
             ORDER BY d.[distance] ASC, t.[id] ASC OFFSET @P4 ROWS FETCH NEXT @P5 ROWS ONLY"
        );
        assert_eq!(
            params,
            &vec![
                SqlParam::Str("[1.0,0.0,0.0]".into()),
                SqlParam::Str("one".into()),
                SqlParam::F64(0.5),
                SqlParam::I64(1),
                SqlParam::I64(2),
            ]
        );
        // A non-finite distance is an invalid response.
        fake.respond(vec![vec![
            s("a"),
            s("one"),
            Cell::Null,
            Cell::F64(f64::NAN),
        ]]);
        assert!(collection
            .search(vec![1.0, 0.0, 0.0], &VectorSearchOptions::new(2))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn metric_score_direction_and_threshold() {
        for (distance, raw, expected) in [
            ("cosine_distance", 0.25, 0.25),
            ("cosine_similarity", 0.25, 0.75),
            ("euclidean_distance", 2.0, 2.0),
            ("dot_prod", -4.0, 4.0),
            ("negative_dot_prod", -4.0, -4.0),
        ] {
            let fake = FakeConnector::default();
            let definition = VectorStoreCollectionDefinition::new(vec![
                VectorStoreField::key("id").with_type("int"),
                VectorStoreField::vector("v", 2)
                    .with_distance_function(DistanceFunction::new(distance)),
            ])
            .unwrap();
            let collection =
                SqlServerCollection::with_connector(Arc::new(fake.clone()), "t", definition, false)
                    .unwrap();
            fake.respond(vec![vec![Cell::I64(1), Cell::F64(raw)]]);
            let hits = collection
                .search_with_threshold(vec![1.0, 0.0], &VectorSearchOptions::new(1), Some(3.0))
                .await
                .unwrap();
            assert_eq!(hits[0].score, Some(expected), "{distance}");
            assert_eq!(
                fake.queries()[0].1[1],
                SqlParam::F64(cutoff_for(distance, 3.0))
            );
        }
    }

    fn cutoff_for(distance: &str, threshold: f64) -> f64 {
        match distance {
            "cosine_similarity" => 1.0 - threshold,
            "dot_prod" => -threshold,
            _ => threshold,
        }
    }

    #[tokio::test]
    async fn invalid_options_and_dimensions_fail_before_io() {
        let fake = FakeConnector::default();
        let collection = collection(&fake);
        assert!(collection
            .search(vec![1.0, 0.0], &VectorSearchOptions::new(1))
            .await
            .is_err());
        assert!(collection
            .search(vec![1.0, 0.0, 0.0], &VectorSearchOptions::new(0))
            .await
            .is_err());
        assert!(collection
            .search(
                vec![1.0, 0.0, 0.0],
                &VectorSearchOptions::new(1).with_provider_filter("1 = 1")
            )
            .await
            .is_err());
        assert!(collection
            .search_with_threshold(
                vec![1.0, 0.0, 0.0],
                &VectorSearchOptions::new(1),
                Some(f64::INFINITY)
            )
            .await
            .is_err());
        assert!(collection
            .upsert(vec![json!({"id": "a", "embedding": [1, 2]})])
            .await
            .is_err());
        assert!(fake.events().is_empty());
    }

    #[tokio::test]
    async fn a_closed_collection_rejects_even_empty_operations() {
        let fake = FakeConnector::default();
        let collection = collection(&fake);
        collection.close().await;
        assert!(collection.upsert(vec![]).await.is_err());
        assert!(collection.get(vec![], false).await.is_err());
        assert!(collection.delete(vec![]).await.is_err());
        assert!(collection.query(&SqlServerQuery::new(1)).await.is_err());
    }

    #[tokio::test]
    async fn store_children_share_the_client_without_closing_it() {
        let fake = FakeConnector::default();
        let store = SqlServerStore::with_connector(Arc::new(fake.clone()), "app");
        let collection = store.collection("t", definition()).unwrap();
        assert_eq!(collection.schema_name(), "app");
        collection.close().await;
        assert_eq!(
            collection.upsert(vec![]).await.unwrap(),
            Vec::<Value>::new()
        );
        fake.respond(vec![vec![s("a")], vec![s("b")]]);
        assert_eq!(store.list_collection_names().await.unwrap(), vec!["a", "b"]);
        assert_eq!(fake.queries()[0].1, vec![SqlParam::Str("app".into())]);
        store.delete_collection("x]y").await.unwrap();
        assert_eq!(fake.queries()[1].0, "DROP TABLE IF EXISTS [app].[x]]y]");
        store.close().await;
        assert!(collection.upsert(vec![]).await.is_err());
        assert!(store.list_collection_names().await.is_err());
    }

    #[tokio::test]
    async fn a_committed_generated_key_is_distinguishable_from_a_failed_upsert() {
        let fake = FakeConnector::default();
        let definition =
            VectorStoreCollectionDefinition::new(
                vec![VectorStoreField::key("id").with_type("int")],
            )
            .unwrap();
        let collection =
            SqlServerCollection::with_connector(Arc::new(fake.clone()), "t", definition, true)
                .unwrap();
        fake.set(|s| s.fail_close = true);
        fake.respond(vec![vec![Cell::I64(1)]]);
        let error = collection.upsert(vec![json!({})]).await.unwrap_err();
        assert!(crate::is_committed_cleanup(&error));
        fake.set(|s| s.fail_close = false);
        fake.fail_next("SQL Server operation failed: insert failed");
        let error = collection.upsert(vec![json!({})]).await.unwrap_err();
        assert!(!crate::is_committed_cleanup(&error));
    }

    #[test]
    fn settings_priority_and_secret_masking() {
        let explicit =
            SqlServerSettings::load(Some(SecretString::new("Server=x;Password=hunter2")));
        assert!(!format!("{explicit:?}").contains("hunter2"));
        assert!(SqlServerSettings::default().require().is_err());
        assert!(SqlServerSettings {
            connection_string: Some(SecretString::new(" "))
        }
        .require()
        .is_err());
        let options = SqlServerStore::builder().connection_string("Server=x;Password=hunter2");
        assert!(!format!("{options:?}").contains("hunter2"));
        assert!(SqlServerStore::builder()
            .connection_string("Server=tcp:x,1433")
            .schema("")
            .build()
            .is_err());
        let store = SqlServerStore::builder()
            .connection_string("Server=tcp:x,1433;User ID=u;Password=p")
            .query_timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        assert_eq!(store.schema(), "dbo");
    }
}
