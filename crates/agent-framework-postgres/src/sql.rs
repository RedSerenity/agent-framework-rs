//! Identifiers, column types, value adaptation, and the portable-filter
//! compiler — everything that turns a collection definition into SQL text
//! plus bound parameters, with no I/O.
//!
//! Mirrors the module-level helpers of upstream's
//! `agent_framework_postgres/_vector_store.py` (`_prepare_identifier`,
//! `_prepare_column_type`, `_prepare_value`, `_FilterCompiler`, …). Kept
//! separate from the connection code so every statement this crate sends
//! can be asserted on offline.

use std::error::Error as StdError;
use std::fmt::Write as _;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use bytes::BytesMut;
use serde_json::Value;
use sha2::{Digest, Sha256};
use time::format_description::well_known::Rfc3339;
use time::{Date, OffsetDateTime, PrimitiveDateTime};
use tokio_postgres::types::{IsNull, ToSql, Type};
use uuid::Uuid;

use crate::{PostgresVectorOptions, PostgresVectorType};

/// PostgreSQL's identifier limit (`NAMEDATALEN - 1`) in bytes. A longer name
/// is silently truncated by the server, so two long names could collide —
/// upstream refuses rather than let that happen.
pub(crate) const MAX_IDENTIFIER_BYTES: usize = 63;

/// pgvector's storage ceiling for a `vector` / `halfvec` column.
pub(crate) const MAX_DIMENSIONS: usize = 16_000;

fn config(message: impl Into<String>) -> Error {
    Error::Configuration(message.into())
}

// region: identifiers

/// Quote one identifier with double quotes, doubling embedded quotes.
///
/// Mirrors upstream's `_prepare_identifier` (`psycopg.sql.Identifier`):
/// 1–63 UTF-8 bytes, no NUL. A dot is part of the name, never a separator —
/// `"public.documents"` is one table name, not a schema-qualified one.
pub(crate) fn quote_identifier(name: &str) -> Result<String> {
    if name.is_empty() || name.contains('\0') || name.len() > MAX_IDENTIFIER_BYTES {
        return Err(config(
            "Postgres identifiers must contain 1-63 UTF-8 bytes and no NUL.",
        ));
    }
    Ok(format!("\"{}\"", name.replace('"', "\"\"")))
}

// endregion

// region: field kinds

/// A non-vector field's declared scalar type — upstream's `_TYPES` keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Str,
    Int,
    Float,
    Bool,
    Uuid,
    Bytes,
    Date,
    DateTime,
    List,
    Dict,
}

impl Kind {
    /// Resolve the free-form [`VectorStoreField::type_`] hint.
    ///
    /// Upstream reads the Python annotation, so every field is typed; here the
    /// hint is optional and a field without one is refused, because the
    /// column type of `CREATE TABLE` has to come from somewhere. Upstream's
    /// spellings (`str`, `int`, `float`, `bool`, `UUID`, `bytes`, `date`,
    /// `datetime`, `list`, `dict`) are accepted case-insensitively, plus the
    /// JSON-ish aliases `string`, `integer`, `boolean`, `array`, `object`.
    pub(crate) fn of(field: &VectorStoreField) -> Result<Self> {
        let hint = field.type_.as_deref().unwrap_or("");
        Ok(match hint.trim().to_ascii_lowercase().as_str() {
            "str" | "string" => Self::Str,
            "int" | "integer" => Self::Int,
            "float" => Self::Float,
            "bool" | "boolean" => Self::Bool,
            "uuid" => Self::Uuid,
            "bytes" => Self::Bytes,
            "date" => Self::Date,
            "datetime" => Self::DateTime,
            "list" | "array" => Self::List,
            "dict" | "object" => Self::Dict,
            _ => {
                return Err(config(format!(
                    "Field '{}' needs a supported explicit type; got '{hint}'. Declare one of \
                     str, int, float, bool, UUID, bytes, date, datetime, list, dict with \
                     `VectorStoreField::with_type`.",
                    field.name
                )))
            }
        })
    }

    /// The upstream spelling, for error messages.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Str => "str",
            Self::Int => "int",
            Self::Float => "float",
            Self::Bool => "bool",
            Self::Uuid => "UUID",
            Self::Bytes => "bytes",
            Self::Date => "date",
            Self::DateTime => "datetime",
            Self::List => "list",
            Self::Dict => "dict",
        }
    }

    /// The column type — upstream's `_TYPES`.
    pub(crate) fn column_type(self) -> &'static str {
        match self {
            Self::Str => "text COLLATE \"C\"",
            Self::Int => "bigint",
            Self::Float => "double precision",
            Self::Bool => "boolean",
            Self::Uuid => "uuid",
            Self::Bytes => "bytea",
            Self::Date => "date",
            Self::DateTime => "timestamp with time zone",
            Self::List | Self::Dict => "jsonb",
        }
    }

    /// The cast a bound parameter of this kind carries.
    fn cast(self) -> &'static str {
        match self {
            Self::Str => "text",
            Self::Int => "bigint",
            Self::Float => "double precision",
            Self::Bool => "boolean",
            Self::Uuid => "uuid",
            Self::Bytes => "bytea",
            Self::Date => "date",
            Self::DateTime => "timestamptz",
            Self::List | Self::Dict => "jsonb",
        }
    }
}

// endregion

// region: vector metrics and indexes

/// How a raw pgvector operator result becomes the reported score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScoreForm {
    /// The operator result as-is (a distance; lower is closer).
    Distance,
    /// `1 - distance` (cosine similarity; higher is closer).
    OneMinus,
    /// `-distance` (`<#>` is the *negative* inner product, so negating it
    /// gives the dot product; higher is closer).
    Negated,
}

/// A resolved distance function — one row of upstream's `_METRICS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Metric {
    /// The pgvector operator.
    pub(crate) operator: &'static str,
    /// The operator-class suffix (`vector_<ops>_ops`).
    pub(crate) ops: &'static str,
    pub(crate) score: ScoreForm,
}

/// Mirrors upstream's `_prepare_metric`. An undeclared function is
/// upstream's `"DEFAULT"`: cosine **distance**.
pub(crate) fn metric_for(field: &VectorStoreField) -> Result<Metric> {
    let name = field
        .distance_function
        .as_ref()
        .map(|d| d.as_str())
        .unwrap_or("DEFAULT");
    let (operator, ops, score) = match name {
        "DEFAULT" | "cosine_distance" => ("<=>", "cosine", ScoreForm::Distance),
        "cosine_similarity" => ("<=>", "cosine", ScoreForm::OneMinus),
        "dot_prod" => ("<#>", "ip", ScoreForm::Negated),
        "negative_dot_prod" => ("<#>", "ip", ScoreForm::Distance),
        "euclidean_distance" => ("<->", "l2", ScoreForm::Distance),
        "manhattan" => ("<+>", "l1", ScoreForm::Distance),
        other => {
            return Err(config(format!(
                "Unsupported Postgres distance function '{other}'."
            )))
        }
    };
    Ok(Metric {
        operator,
        ops,
        score,
    })
}

/// A vector field's index, resolved from its `index_kind` and the
/// collection's [`PostgresVectorOptions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorIndex {
    /// `default` or `flat`: no ANN index; search is exact.
    Exact,
    Hnsw {
        m: u32,
        ef_construction: u32,
    },
    IvfFlat {
        lists: u32,
    },
}

/// Everything about one vector column the SQL needs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VectorSpec {
    pub(crate) dimensions: usize,
    pub(crate) storage: PostgresVectorType,
    pub(crate) metric: Metric,
    pub(crate) index: VectorIndex,
}

impl PostgresVectorType {
    pub(crate) fn sql_name(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Halfvec => "halfvec",
        }
    }
}

fn check_range(value: u32, name: &str, minimum: u32, maximum: u32) -> Result<u32> {
    if !(minimum..=maximum).contains(&value) {
        return Err(config(format!(
            "{name} must be an integer between {minimum} and {maximum}."
        )));
    }
    Ok(value)
}

/// Mirrors upstream's `_prepare_vector_type` plus
/// `PostgresCollection._validate_vector_field`.
pub(crate) fn vector_spec(
    field: &VectorStoreField,
    options: Option<&PostgresVectorOptions>,
) -> Result<VectorSpec> {
    let default_storage = match field.type_.as_deref().map(str::trim) {
        None | Some("float") | Some("float32") => PostgresVectorType::Vector,
        Some("float16") => PostgresVectorType::Halfvec,
        Some(other) => {
            return Err(config(format!(
                "Vector field '{}' declares element type '{other}'; Postgres supports float, \
                 float32, and float16 vectors only.",
                field.name
            )))
        }
    };
    let options = options.cloned().unwrap_or_default();
    let storage = options.vector_type.unwrap_or(default_storage);
    let dimensions = field.dimensions.unwrap_or(0);
    if !(1..=MAX_DIMENSIONS).contains(&dimensions) {
        return Err(config(format!(
            "dimensions must be an integer between 1 and {MAX_DIMENSIONS}."
        )));
    }
    let metric = metric_for(field)?;
    let kind = field
        .index_kind
        .as_ref()
        .map(|k| k.as_str())
        .unwrap_or("default");
    let index = match kind {
        "default" | "flat" => VectorIndex::Exact,
        "hnsw" => {
            let m = check_range(options.m.unwrap_or(16), "postgres.m", 2, 100)?;
            let ef_construction = check_range(
                options.ef_construction.unwrap_or(64),
                "postgres.ef_construction",
                2 * m,
                1000,
            )?;
            VectorIndex::Hnsw { m, ef_construction }
        }
        "ivf_flat" => {
            let lists = check_range(options.lists.unwrap_or(100), "postgres.lists", 1, 32_768)?;
            if metric.ops == "l1" {
                return Err(config("IVFFlat does not support Manhattan distance."));
            }
            VectorIndex::IvfFlat { lists }
        }
        other => {
            return Err(config(format!(
                "Unsupported Postgres index kind '{other}'."
            )))
        }
    };
    // Upstream's `allowed` set: tuning for one index kind on another is a
    // mistake, not something to silently ignore.
    if !matches!(index, VectorIndex::Hnsw { .. })
        && (options.m.is_some() || options.ef_construction.is_some())
    {
        return Err(config(format!(
            "Unsupported Postgres provider annotation(s) on '{}': m and ef_construction apply \
             only to an HNSW index.",
            field.name
        )));
    }
    if !matches!(index, VectorIndex::IvfFlat { .. }) && options.lists.is_some() {
        return Err(config(format!(
            "Unsupported Postgres provider annotation(s) on '{}': lists applies only to an \
             IVFFlat index.",
            field.name
        )));
    }
    if !matches!(index, VectorIndex::Exact) {
        let limit = match storage {
            PostgresVectorType::Halfvec => 4000,
            PostgresVectorType::Vector => 2000,
        };
        if dimensions > limit {
            return Err(config(format!(
                "indexed vector dimensions must be an integer between 1 and {limit}."
            )));
        }
    }
    Ok(VectorSpec {
        dimensions,
        storage,
        metric,
        index,
    })
}

// endregion

// region: schema

/// One column, resolved from the definition.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Column {
    /// The logical field name.
    pub(crate) name: String,
    /// The storage (column) name.
    pub(crate) storage: String,
    pub(crate) role: FieldType,
    /// The scalar kind; `None` for a vector column.
    pub(crate) kind: Option<Kind>,
    pub(crate) vector: Option<VectorSpec>,
    pub(crate) indexed: bool,
}

impl Column {
    /// The quoted column name.
    pub(crate) fn quoted(&self) -> String {
        // Validated when the schema was built.
        format!("\"{}\"", self.storage.replace('"', "\"\""))
    }

    /// The column type in `CREATE TABLE` — upstream's `_prepare_column_type`.
    pub(crate) fn column_type(&self) -> String {
        match (&self.vector, self.kind) {
            (Some(spec), _) => format!("\"{}\"({})", spec.storage.sql_name(), spec.dimensions),
            (None, Some(kind)) => kind.column_type().to_string(),
            (None, None) => unreachable!("every column has a kind or a vector spec"),
        }
    }

    /// The expression a `SELECT` projects for this column. Vectors come back
    /// through their text form (`[1,2,3]`, which is JSON), so this crate needs
    /// no binary codec for pgvector's types.
    pub(crate) fn projection(&self, alias: &str) -> String {
        if self.vector.is_some() {
            format!("{alias}{}::text", self.quoted())
        } else {
            format!("{alias}{}", self.quoted())
        }
    }
}

/// A validated collection: the qualified table and its columns, in
/// definition order.
#[derive(Debug, Clone)]
pub(crate) struct Schema {
    pub(crate) schema: String,
    pub(crate) table_name: String,
    /// `"schema"."table"`.
    pub(crate) table: String,
    pub(crate) columns: Vec<Column>,
    pub(crate) key: usize,
    pub(crate) auto_generated_key: bool,
    pub(crate) definition: VectorStoreCollectionDefinition,
}

impl Schema {
    /// Validate a definition against what this connector can store — the
    /// checks upstream's `PostgresCollection.__init__` runs before any I/O.
    pub(crate) fn new(
        schema: &str,
        table_name: &str,
        definition: VectorStoreCollectionDefinition,
        vector_options: &std::collections::HashMap<String, PostgresVectorOptions>,
        auto_generated_key: bool,
    ) -> Result<Self> {
        let table = format!(
            "{}.{}",
            quote_identifier(schema)?,
            quote_identifier(table_name)?
        );
        for name in vector_options.keys() {
            match definition.try_get_field(name) {
                Some(field) if field.field_type == FieldType::Vector => {}
                Some(_) => {
                    return Err(config(format!(
                        "Postgres vector options are supported only on vector fields; '{name}' \
                         is not one."
                    )))
                }
                None => {
                    return Err(config(format!(
                        "Postgres vector options name unknown field '{name}'."
                    )))
                }
            }
        }
        let mut columns = Vec::with_capacity(definition.fields().len());
        let mut key = 0;
        for (index, field) in definition.fields().iter().enumerate() {
            let storage = field.effective_storage_name().to_string();
            quote_identifier(&storage)?;
            if field.is_full_text_indexed == Some(true) {
                return Err(config(
                    "Postgres full-text indexes and keyword-hybrid search are not supported.",
                ));
            }
            let (kind, vector) = match field.field_type {
                FieldType::Vector => (
                    None,
                    Some(vector_spec(field, vector_options.get(&field.name))?),
                ),
                _ => (Some(Kind::of(field)?), None),
            };
            if field.field_type == FieldType::Key {
                key = index;
                if !matches!(kind, Some(Kind::Str | Kind::Int | Kind::Uuid)) {
                    return Err(config(format!(
                        "Postgres keys must be str, int, or UUID; '{}' is '{}'.",
                        field.name,
                        kind.map(Kind::name).unwrap_or("")
                    )));
                }
            }
            columns.push(Column {
                name: field.name.clone(),
                storage,
                role: field.field_type,
                kind,
                vector,
                indexed: field.is_indexed == Some(true),
            });
        }
        Ok(Self {
            schema: schema.to_string(),
            table_name: table_name.to_string(),
            table,
            columns,
            key,
            auto_generated_key,
            definition,
        })
    }

    pub(crate) fn key_column(&self) -> &Column {
        &self.columns[self.key]
    }

    #[cfg(test)]
    pub(crate) fn vector_columns(&self) -> impl Iterator<Item = &Column> {
        self.columns.iter().filter(|c| c.vector.is_some())
    }

    /// Columns a read returns — upstream's `_prepare_columns`.
    pub(crate) fn selected(&self, include_vectors: bool) -> Vec<&Column> {
        self.columns
            .iter()
            .filter(|c| include_vectors || c.vector.is_none())
            .collect()
    }

    /// `CREATE TABLE IF NOT EXISTS` — upstream's
    /// `ensure_collection_exists` column list.
    pub(crate) fn create_table_sql(&self) -> String {
        let columns: Vec<String> = self
            .columns
            .iter()
            .map(|column| {
                let mut suffix = "";
                if column.role == FieldType::Key {
                    suffix = " PRIMARY KEY";
                    if self.auto_generated_key {
                        suffix = match column.kind {
                            Some(Kind::Int) => " GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY",
                            Some(Kind::Uuid) => " DEFAULT pg_catalog.gen_random_uuid() PRIMARY KEY",
                            _ => " DEFAULT pg_catalog.gen_random_uuid()::text PRIMARY KEY",
                        };
                    }
                }
                format!("{} {}{suffix}", column.quoted(), column.column_type())
            })
            .collect();
        format!(
            "CREATE TABLE IF NOT EXISTS {} ({})",
            self.table,
            columns.join(", ")
        )
    }

    /// The index name upstream's `_prepare_index` derives:
    /// `af_vector_` + the first 32 hex digits of
    /// SHA-256(`schema \0 table \0 column`). Used for data indexes too.
    pub(crate) fn index_name(&self, column: &Column) -> String {
        let digest = Sha256::digest(
            format!("{}\0{}\0{}", self.schema, self.table_name, column.storage).as_bytes(),
        );
        let mut hex = String::with_capacity(32);
        for byte in &digest[..16] {
            let _ = write!(hex, "{byte:02x}");
        }
        format!("af_vector_{hex}")
    }

    /// `CREATE INDEX IF NOT EXISTS` for a data or ANN-indexed vector column —
    /// upstream's `_prepare_index`. `None` for a column that wants none.
    pub(crate) fn index_sql(&self, column: &Column) -> Option<String> {
        let name = format!("\"{}\"", self.index_name(column));
        match &column.vector {
            None if column.role == FieldType::Data && column.indexed => Some(format!(
                "CREATE INDEX IF NOT EXISTS {name} ON {} ({})",
                self.table,
                column.quoted()
            )),
            None => None,
            Some(spec) => {
                let (method, with) = match spec.index {
                    VectorIndex::Exact => return None,
                    VectorIndex::Hnsw { m, ef_construction } => (
                        "hnsw",
                        format!("m = {m}, ef_construction = {ef_construction}"),
                    ),
                    VectorIndex::IvfFlat { lists } => ("ivfflat", format!("lists = {lists}")),
                };
                Some(format!(
                    "CREATE INDEX IF NOT EXISTS {name} ON {} USING {method} ({} \"{}_{}_ops\") \
                     WITH ({with})",
                    self.table,
                    column.quoted(),
                    spec.storage.sql_name(),
                    spec.metric.ops
                ))
            }
        }
    }

    /// `ORDER BY` for a filtered read — upstream's `_prepare_order_by`.
    pub(crate) fn order_by_sql(&self, order_by: &[(String, bool)]) -> Result<String> {
        let mut parts = Vec::with_capacity(order_by.len() + 1);
        for (name, ascending) in order_by {
            let column = self.filter_column(name)?;
            if matches!(column.kind, Some(Kind::List | Kind::Dict)) {
                return Err(config("Ordering JSON columns is not supported."));
            }
            parts.push(format!(
                "{} {} NULLS LAST",
                column.quoted(),
                if *ascending { "ASC" } else { "DESC" }
            ));
        }
        let key = self.key_column();
        if !order_by.iter().any(|(name, _)| *name == key.name) {
            parts.push(format!("{} ASC", key.quoted()));
        }
        Ok(parts.join(", "))
    }

    /// Resolve a filter or ordering field — upstream's
    /// `_prepare_filter_field`.
    pub(crate) fn filter_column(&self, name: &str) -> Result<&Column> {
        if name.contains('.') {
            return Err(config(
                "Postgres filters and ordering do not support nested field paths.",
            ));
        }
        let column = self
            .columns
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| config(format!("Unknown Postgres field '{name}'.")))?;
        if column.vector.is_some() {
            return Err(config(
                "Filtering and ordering vector columns is not supported.",
            ));
        }
        Ok(column)
    }
}

// endregion

// region: values

/// One bound parameter. Every placeholder this crate emits carries an
/// explicit cast ([`PgValue::cast`]), so the server never infers a type from
/// context and a string can never be coerced into a number or a boolean —
/// upstream gets the same guarantee from psycopg's typed adaptation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PgValue {
    /// A typed SQL `NULL`.
    Null(&'static str),
    Text(String),
    Int(i64),
    Float(f64),
    /// An integer beyond `bigint`, compared as `numeric` so no precision is
    /// lost.
    Numeric(String),
    Bool(bool),
    Uuid(Uuid),
    Bytes(Vec<u8>),
    Date(Date),
    Timestamp(OffsetDateTime),
    Json(Value),
    /// pgvector's text form, `[1,2,3]`, cast to the storage type.
    Vector(String, PostgresVectorType),
    TextArray(Vec<String>),
    IntArray(Vec<i64>),
    UuidArray(Vec<Uuid>),
}

impl PgValue {
    /// The cast the placeholder carries.
    pub(crate) fn cast(&self) -> &'static str {
        match self {
            Self::Null(cast) => cast,
            Self::Text(_) => "text",
            Self::Int(_) => "bigint",
            Self::Float(_) => "double precision",
            Self::Numeric(_) => "text::numeric",
            Self::Bool(_) => "boolean",
            Self::Uuid(_) => "uuid",
            Self::Bytes(_) => "bytea",
            Self::Date(_) => "date",
            Self::Timestamp(_) => "timestamptz",
            Self::Json(_) => "jsonb",
            Self::Vector(_, PostgresVectorType::Vector) => "text::vector",
            Self::Vector(_, PostgresVectorType::Halfvec) => "text::halfvec",
            Self::TextArray(_) => "text[]",
            Self::IntArray(_) => "bigint[]",
            Self::UuidArray(_) => "uuid[]",
        }
    }
}

impl ToSql for PgValue {
    fn to_sql(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> std::result::Result<IsNull, Box<dyn StdError + Sync + Send>> {
        self.to_sql_checked(ty, out)
    }

    fn accepts(_ty: &Type) -> bool {
        // Checked per variant in `to_sql_checked`.
        true
    }

    fn to_sql_checked(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> std::result::Result<IsNull, Box<dyn StdError + Sync + Send>> {
        match self {
            Self::Null(_) => Ok(IsNull::Yes),
            Self::Text(v) | Self::Numeric(v) | Self::Vector(v, _) => v.to_sql_checked(ty, out),
            Self::Int(v) => v.to_sql_checked(ty, out),
            Self::Float(v) => v.to_sql_checked(ty, out),
            Self::Bool(v) => v.to_sql_checked(ty, out),
            Self::Uuid(v) => v.to_sql_checked(ty, out),
            Self::Bytes(v) => v.to_sql_checked(ty, out),
            Self::Date(v) => v.to_sql_checked(ty, out),
            Self::Timestamp(v) => v.to_sql_checked(ty, out),
            Self::Json(v) => v.to_sql_checked(ty, out),
            Self::TextArray(v) => v.to_sql_checked(ty, out),
            Self::IntArray(v) => v.to_sql_checked(ty, out),
            Self::UuidArray(v) => v.to_sql_checked(ty, out),
        }
    }
}

/// An ordered parameter list that hands out `$n::cast` placeholders.
#[derive(Debug, Default)]
pub(crate) struct Params {
    pub(crate) values: Vec<PgValue>,
}

impl Params {
    pub(crate) fn bind(&mut self, value: PgValue) -> String {
        let cast = value.cast();
        self.values.push(value);
        format!("${}::{cast}", self.values.len())
    }

    pub(crate) fn as_refs(&self) -> Vec<&(dyn ToSql + Sync)> {
        self.values
            .iter()
            .map(|v| v as &(dyn ToSql + Sync))
            .collect()
    }
}

fn type_error(column: &Column) -> Error {
    config(format!(
        "Field '{}' requires a value of type '{}'.",
        column.name,
        column.kind.map(Kind::name).unwrap_or("vector")
    ))
}

/// Parse an ISO-8601 calendar date (`YYYY-MM-DD`).
pub(crate) fn parse_date(text: &str) -> Option<Date> {
    let format = time::macros::format_description!("[year]-[month]-[day]");
    Date::parse(text, &format).ok()
}

/// Parse a timezone-aware timestamp: RFC 3339, also with a space instead
/// of `T` (Python's `fromisoformat` accepts both). `Err(true)` when the
/// value is a valid timestamp *without* an offset, so the error can say so.
pub(crate) fn parse_timestamp(text: &str) -> std::result::Result<OffsetDateTime, bool> {
    let normalized = if text.len() > 10 && text.as_bytes()[10] == b' ' {
        format!("{}T{}", &text[..10], &text[11..])
    } else {
        text.to_string()
    };
    if let Ok(parsed) = OffsetDateTime::parse(&normalized, &Rfc3339) {
        return Ok(parsed);
    }
    let naive = time::macros::format_description!(
        version = 2,
        "[year]-[month]-[day]T[hour]:[minute][optional [:[second][optional [.[subsecond]]]]]"
    );
    Err(PrimitiveDateTime::parse(&normalized, &naive).is_ok())
}

/// Render a dense vector in pgvector's text form after validating it.
pub(crate) fn vector_text(column: &Column, spec: &VectorSpec, value: &Value) -> Result<String> {
    let items = value.as_array().ok_or_else(|| {
        config(format!(
            "Vector field '{}' requires a dense numeric vector, not text or bytes.",
            column.name
        ))
    })?;
    if items.len() != spec.dimensions {
        return Err(config(format!(
            "Vector field '{}' requires {} dimensions; got {}.",
            column.name,
            spec.dimensions,
            items.len()
        )));
    }
    let mut out = String::with_capacity(items.len() * 8 + 2);
    out.push('[');
    for (index, item) in items.iter().enumerate() {
        let number = item.as_f64().filter(|n| n.is_finite()).ok_or_else(|| {
            config(format!(
                "Vector field '{}' requires finite numeric elements.",
                column.name
            ))
        })?;
        if index > 0 {
            out.push(',');
        }
        let _ = write!(out, "{number}");
    }
    out.push(']');
    Ok(out)
}

/// Adapt one JSON value for storage — upstream's `_prepare_value`. No
/// silent coercion: a string is never a number, a number never a boolean.
pub(crate) fn prepare_value(column: &Column, value: &Value) -> Result<PgValue> {
    if let Some(spec) = &column.vector {
        if value.is_null() {
            return Ok(PgValue::Null(match spec.storage {
                PostgresVectorType::Vector => "text::vector",
                PostgresVectorType::Halfvec => "text::halfvec",
            }));
        }
        return Ok(PgValue::Vector(
            vector_text(column, spec, value)?,
            spec.storage,
        ));
    }
    let kind = column.kind.expect("a scalar column has a kind");
    if value.is_null() {
        return Ok(PgValue::Null(kind.cast()));
    }
    let adapted = match (kind, value) {
        (Kind::Str, Value::String(s)) => PgValue::Text(s.clone()),
        (Kind::Bool, Value::Bool(b)) => PgValue::Bool(*b),
        (Kind::Int, Value::Number(n)) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => PgValue::Int(i),
            (None, Some(_)) => {
                return Err(config(format!(
                    "Field '{}' exceeds the PostgreSQL bigint range.",
                    column.name
                )))
            }
            _ => return Err(type_error(column)),
        },
        (Kind::Float, Value::Number(n)) => {
            let f = n.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
                config(format!("Field '{}' requires a finite number.", column.name))
            })?;
            PgValue::Float(f)
        }
        (Kind::Uuid, Value::String(s)) => PgValue::Uuid(
            Uuid::parse_str(s)
                .map_err(|_| config(format!("Field '{}' requires a valid UUID.", column.name)))?,
        ),
        (Kind::Date, Value::String(s)) => PgValue::Date(parse_date(s).ok_or_else(|| {
            config(format!(
                "Field '{}' requires an ISO-8601 date (YYYY-MM-DD).",
                column.name
            ))
        })?),
        (Kind::DateTime, Value::String(s)) => match parse_timestamp(s) {
            Ok(parsed) => PgValue::Timestamp(parsed),
            Err(true) => {
                return Err(config(format!(
                    "Datetime field '{}' requires a timezone.",
                    column.name
                )))
            }
            Err(false) => {
                return Err(config(format!(
                    "Field '{}' requires an RFC 3339 timestamp.",
                    column.name
                )))
            }
        },
        (Kind::Bytes, Value::Array(items)) => PgValue::Bytes(bytes_of(items).ok_or_else(|| {
            config(format!(
                "Field '{}' requires bytes, encoded as an array of integers 0-255.",
                column.name
            ))
        })?),
        (Kind::List, Value::Array(_)) | (Kind::Dict, Value::Object(_)) => {
            PgValue::Json(value.clone())
        }
        _ => return Err(type_error(column)),
    };
    Ok(adapted)
}

/// A `bytes` value is serde's encoding of `Vec<u8>`: an array of integers.
pub(crate) fn bytes_of(items: &[Value]) -> Option<Vec<u8>> {
    items
        .iter()
        .map(|item| item.as_u64().and_then(|n| u8::try_from(n).ok()))
        .collect()
}

/// Adapt a key — upstream's `_prepare_key`: never null.
pub(crate) fn prepare_key(schema: &Schema, value: &Value) -> Result<PgValue> {
    if value.is_null() {
        return Err(config("Postgres keys cannot be null."));
    }
    prepare_value(schema.key_column(), value)
}

/// The canonical JSON form of an adapted key, matching what a read decodes,
/// so input keys and returned rows can be paired.
pub(crate) fn key_identity(value: &PgValue) -> Value {
    match value {
        PgValue::Text(s) => Value::String(s.clone()),
        PgValue::Int(i) => Value::from(*i),
        PgValue::Uuid(u) => Value::String(u.hyphenated().to_string()),
        _ => Value::Null,
    }
}

/// Bind a list of adapted keys as one array parameter for `= ANY(...)`.
pub(crate) fn key_array(keys: &[PgValue]) -> PgValue {
    if keys.iter().all(|k| matches!(k, PgValue::Int(_))) {
        return PgValue::IntArray(
            keys.iter()
                .filter_map(|k| match k {
                    PgValue::Int(i) => Some(*i),
                    _ => None,
                })
                .collect(),
        );
    }
    if keys.iter().all(|k| matches!(k, PgValue::Uuid(_))) {
        return PgValue::UuidArray(
            keys.iter()
                .filter_map(|k| match k {
                    PgValue::Uuid(u) => Some(*u),
                    _ => None,
                })
                .collect(),
        );
    }
    PgValue::TextArray(
        keys.iter()
            .filter_map(|k| match k {
                PgValue::Text(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
    )
}

// endregion

// region: filters

/// Translate a portable filter into total-boolean, parameterized SQL —
/// upstream's `_FilterCompiler`.
///
/// "Total-boolean" means every condition evaluates to `TRUE` or `FALSE`,
/// never `NULL`, so `NOT` behaves as the portable semantics say: equality
/// uses `IS NOT DISTINCT FROM`, ordered and text comparisons are wrapped in
/// `(...) IS TRUE`, and membership requires a non-null column.
pub(crate) struct FilterCompiler<'a> {
    schema: &'a Schema,
    params: &'a mut Params,
}

impl<'a> FilterCompiler<'a> {
    pub(crate) fn new(schema: &'a Schema, params: &'a mut Params) -> Self {
        Self { schema, params }
    }

    /// Compile one expression, appending its parameters in placeholder order.
    pub(crate) fn compile(&mut self, expression: &FilterExpression) -> Result<String> {
        expression.validate()?;
        self.condition(expression)
    }

    fn condition(&mut self, expression: &FilterExpression) -> Result<String> {
        match expression {
            FilterExpression::Group(group) => {
                let parts = group
                    .filters
                    .iter()
                    .map(|child| self.condition(child))
                    .collect::<Result<Vec<_>>>()?;
                Ok(match group.operator {
                    FilterGroupOperator::Not => format!("(NOT ({}))", parts[0]),
                    FilterGroupOperator::And => format!("({})", parts.join(" AND ")),
                    FilterGroupOperator::Or => format!("({})", parts.join(" OR ")),
                })
            }
            FilterExpression::Condition(filter) => self.leaf(filter),
        }
    }

    fn equality(&mut self, column: &Column, sql_column: &str, value: &Value) -> Result<String> {
        if value.is_null() {
            return Ok(format!("{sql_column} IS NULL"));
        }
        let kind = column.kind.expect("filter columns are scalar");
        if (kind == Kind::Bool) != value.is_boolean() {
            return Ok("FALSE".into());
        }
        let adapted = match (kind, value) {
            (Kind::Int | Kind::Float, Value::Number(n)) => numeric_operand(kind, n)?,
            // Do not let Postgres coerce, e.g., string '1' into an integer.
            (Kind::Str, v) if !v.is_string() => return Ok("FALSE".into()),
            (Kind::Int | Kind::Float, _) => return Ok("FALSE".into()),
            (Kind::Bytes, Value::Array(items)) if bytes_of(items).is_none() => {
                return Ok("FALSE".into())
            }
            (Kind::Bytes, v) if !v.is_array() => return Ok("FALSE".into()),
            (Kind::List, v) if !v.is_array() => return Ok("FALSE".into()),
            (Kind::Dict, v) if !v.is_object() => return Ok("FALSE".into()),
            _ => prepare_value(column, value)?,
        };
        let placeholder = self.params.bind(adapted);
        Ok(format!("{sql_column} IS NOT DISTINCT FROM {placeholder}"))
    }

    fn leaf(&mut self, filter: &Filter) -> Result<String> {
        let schema = self.schema;
        let column = schema.filter_column(&filter.field_name)?;
        let kind = column.kind.expect("filter columns are scalar");
        let mut sql_column = column.quoted();
        if kind == Kind::Str {
            sql_column.push_str(" COLLATE \"C\"");
        }
        let null = Value::Null;
        let value = filter.value.as_ref().unwrap_or(&null);
        let items = || value.as_array().map(Vec::as_slice).unwrap_or(&[]);
        Ok(match &filter.operator {
            // Declared SQL columns are present in every row, even when NULL.
            FilterOperator::Exists => "TRUE".into(),
            FilterOperator::IsNull => format!("{sql_column} IS NULL"),
            FilterOperator::IsNotNull => format!("{sql_column} IS NOT NULL"),
            FilterOperator::Eq => self.equality(column, &sql_column, value)?,
            FilterOperator::Ne => {
                format!("(NOT ({}))", self.equality(column, &sql_column, value)?)
            }
            op @ (FilterOperator::In | FilterOperator::NotIn) => {
                let parts = items()
                    .iter()
                    .map(|item| self.equality(column, &sql_column, item))
                    .collect::<Result<Vec<_>>>()?;
                let choices = if parts.is_empty() {
                    "FALSE".to_string()
                } else {
                    parts.join(" OR ")
                };
                let mut membership = format!("({choices})");
                if *op == FilterOperator::NotIn {
                    membership = format!("(NOT {membership})");
                }
                // Portable membership, unlike eq/ne, never matches a null field.
                format!("({sql_column} IS NOT NULL AND {membership})")
            }
            op @ (FilterOperator::Gt
            | FilterOperator::Gte
            | FilterOperator::Lt
            | FilterOperator::Lte
            | FilterOperator::Between) => {
                if matches!(kind, Kind::List | Kind::Dict | Kind::Bool | Kind::Bytes) {
                    return Err(config(format!(
                        "Ordered filtering is not supported for '{}'.",
                        kind.name()
                    )));
                }
                let operands: Vec<&Value> = if *op == FilterOperator::Between {
                    items().iter().collect()
                } else {
                    vec![value]
                };
                if operands.iter().any(|v| v.is_null() || v.is_boolean()) {
                    return Err(config(
                        "Ordered filter operands must be non-null scalars of the column's type.",
                    ));
                }
                let mut placeholders = Vec::with_capacity(operands.len());
                for operand in operands {
                    let adapted = match (kind, operand) {
                        (Kind::Int | Kind::Float, Value::Number(n)) => numeric_operand(kind, n)?,
                        _ => prepare_value(column, operand)?,
                    };
                    placeholders.push(self.params.bind(adapted));
                }
                let comparison = match op {
                    FilterOperator::Between => format!(
                        "{sql_column} BETWEEN {} AND {}",
                        placeholders[0], placeholders[1]
                    ),
                    _ => format!(
                        "{sql_column} {} {}",
                        match op {
                            FilterOperator::Gt => ">",
                            FilterOperator::Gte => ">=",
                            FilterOperator::Lt => "<",
                            _ => "<=",
                        },
                        placeholders[0]
                    ),
                };
                format!("({comparison}) IS TRUE")
            }
            op @ (FilterOperator::ContainsText
            | FilterOperator::StartsWith
            | FilterOperator::EndsWith) => {
                if kind != Kind::Str {
                    return Err(config(format!(
                        "Text filtering requires a string column, not '{}'.",
                        kind.name()
                    )));
                }
                let text = value
                    .as_str()
                    .ok_or_else(|| config("Text filtering requires a string operand."))?;
                // `!` is the explicit escape character; % and _ stay literal.
                let escaped = text
                    .replace('!', "!!")
                    .replace('%', "!%")
                    .replace('_', "!_");
                let pattern = format!(
                    "{}{escaped}{}",
                    if *op != FilterOperator::StartsWith {
                        "%"
                    } else {
                        ""
                    },
                    if *op != FilterOperator::EndsWith {
                        "%"
                    } else {
                        ""
                    }
                );
                let placeholder = self.params.bind(PgValue::Text(pattern));
                format!("({sql_column} LIKE {placeholder} ESCAPE '!') IS TRUE")
            }
            op @ (FilterOperator::Contains
            | FilterOperator::ContainsAny
            | FilterOperator::ContainsAll) => {
                if kind != Kind::List {
                    return Err(config("Collection membership requires a list column."));
                }
                let operands: Vec<&Value> = if *op == FilterOperator::Contains {
                    vec![value]
                } else {
                    items().iter().collect()
                };
                let parts: Vec<String> = operands
                    .into_iter()
                    .map(|item| {
                        let placeholder = self.params.bind(PgValue::Json(item.clone()));
                        format!(
                            "EXISTS (SELECT 1 FROM jsonb_array_elements({sql_column}) AS \
                             item(value) WHERE item.value = {placeholder})"
                        )
                    })
                    .collect();
                let all = *op == FilterOperator::ContainsAll;
                let combined = if parts.is_empty() {
                    if all { "TRUE" } else { "FALSE" }.to_string()
                } else {
                    parts.join(if all { " AND " } else { " OR " })
                };
                format!("({sql_column} IS NOT NULL AND ({combined}))")
            }
            FilterOperator::Provider(name) => {
                return Err(config(format!(
                    "Unsupported Postgres filter operator '{name}'."
                )))
            }
        })
    }
}

/// A numeric operand for an `int` / `float` column. Integers keep full
/// precision (`numeric` past `bigint`), and a fractional operand against an
/// `int` column compares as `double precision` instead of being rounded.
fn numeric_operand(kind: Kind, n: &serde_json::Number) -> Result<PgValue> {
    if let Some(i) = n.as_i64() {
        return Ok(PgValue::Int(i));
    }
    if let Some(u) = n.as_u64() {
        return Ok(if kind == Kind::Int {
            PgValue::Numeric(u.to_string())
        } else {
            PgValue::Float(u as f64)
        });
    }
    let f = n
        .as_f64()
        .filter(|f| f.is_finite())
        .ok_or_else(|| config("Numeric filter values must be finite."))?;
    Ok(PgValue::Float(f))
}

// endregion

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::vectors::{DistanceFunction, FilterGroup, IndexKind};
    use serde_json::json;
    use std::collections::HashMap;

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str"),
            VectorStoreField::data("number").with_type("int"),
            VectorStoreField::data("ratio").with_type("float"),
            VectorStoreField::data("flag").with_type("bool"),
            VectorStoreField::data("tags").with_type("list"),
            VectorStoreField::data("meta").with_type("dict"),
            VectorStoreField::data("when").with_type("datetime"),
            VectorStoreField::data("day").with_type("date"),
            VectorStoreField::data("ref").with_type("UUID"),
            VectorStoreField::data("blob").with_type("bytes"),
            VectorStoreField::vector("embedding", 3).with_storage_name("dense vector"),
        ])
        .unwrap()
    }

    fn schema() -> Schema {
        Schema::new("public", "documents", definition(), &HashMap::new(), false).unwrap()
    }

    fn compile(expression: impl Into<FilterExpression>) -> Result<(String, Vec<PgValue>)> {
        let schema = schema();
        let mut params = Params::default();
        let sql = FilterCompiler::new(&schema, &mut params).compile(&expression.into())?;
        Ok((sql, params.values))
    }

    #[test]
    fn identifiers_are_quoted_not_executed() {
        assert_eq!(
            quote_identifier("a\"; DROP TABLE users; --").unwrap(),
            "\"a\"\"; DROP TABLE users; --\""
        );
        assert_eq!(
            quote_identifier("public.documents").unwrap(),
            "\"public.documents\""
        );
    }

    #[test]
    fn identifiers_reject_truncation_and_nul() {
        assert!(quote_identifier("").is_err());
        assert!(quote_identifier("a\0b").is_err());
        assert!(quote_identifier(&"x".repeat(64)).is_err());
        assert!(quote_identifier(&"x".repeat(63)).is_ok());
        // 32 two-byte characters are 64 bytes.
        assert!(quote_identifier(&"é".repeat(32)).is_err());
    }

    #[test]
    fn filter_values_are_bound() {
        let (sql, params) =
            compile(Filter::contains_text("text", "%'; DROP TABLE documents; --!%_").unwrap())
                .unwrap();
        assert!(!sql.contains("DROP"));
        assert_eq!(
            sql,
            "(\"text\" COLLATE \"C\" LIKE $1::text ESCAPE '!') IS TRUE"
        );
        assert_eq!(
            params,
            vec![PgValue::Text(
                "%!%'; DROP TABLE documents; --!!!%!_%".into()
            )]
        );
    }

    #[test]
    fn starts_and_ends_with_anchor_the_pattern() {
        let (_, params) = compile(Filter::starts_with("text", "ab").unwrap()).unwrap();
        assert_eq!(params, vec![PgValue::Text("ab%".into())]);
        let (_, params) = compile(Filter::ends_with("text", "ab").unwrap()).unwrap();
        assert_eq!(params, vec![PgValue::Text("%ab".into())]);
    }

    #[test]
    fn scalar_operators_are_parameterized() {
        for (filter, expected_sql, expected) in [
            (
                Filter::eq("number", 3).unwrap(),
                "\"number\" IS NOT DISTINCT FROM $1::bigint",
                vec![PgValue::Int(3)],
            ),
            (
                Filter::gt("number", 3).unwrap(),
                "(\"number\" > $1::bigint) IS TRUE",
                vec![PgValue::Int(3)],
            ),
            (
                Filter::lte("ratio", 1.5).unwrap(),
                "(\"ratio\" <= $1::double precision) IS TRUE",
                vec![PgValue::Float(1.5)],
            ),
            (
                Filter::between("number", 1, 5).unwrap(),
                "(\"number\" BETWEEN $1::bigint AND $2::bigint) IS TRUE",
                vec![PgValue::Int(1), PgValue::Int(5)],
            ),
            (
                Filter::gte("text", "m").unwrap(),
                "(\"text\" COLLATE \"C\" >= $1::text) IS TRUE",
                vec![PgValue::Text("m".into())],
            ),
        ] {
            let (sql, params) = compile(filter).unwrap();
            assert_eq!(sql, expected_sql);
            assert_eq!(params, expected);
        }
    }

    #[test]
    fn type_mismatched_equality_is_not_coerced() {
        for (field, value) in [
            ("number", json!("1")),
            ("number", json!(true)),
            ("flag", json!(1)),
            ("text", json!(1)),
            ("tags", json!({"a": 1})),
            ("meta", json!([1])),
            ("blob", json!("abc")),
        ] {
            let (sql, params) = compile(Filter::eq(field, value.clone()).unwrap()).unwrap();
            assert_eq!(sql, "FALSE", "{field} = {value}");
            assert!(params.is_empty());
            let (sql, params) = compile(Filter::ne(field, value).unwrap()).unwrap();
            assert_eq!(sql, "(NOT (FALSE))");
            assert!(params.is_empty());
        }
    }

    #[test]
    fn null_and_negative_predicates_are_two_valued() {
        assert!(compile(Filter::is_null("text").unwrap())
            .unwrap()
            .0
            .ends_with("IS NULL"));
        assert_eq!(compile(Filter::exists("text").unwrap()).unwrap().0, "TRUE");
        assert!(compile(Filter::is_not_null("text").unwrap())
            .unwrap()
            .0
            .ends_with("IS NOT NULL"));
        let (sql, _) =
            compile(FilterGroup::not(Filter::gt("number", 3).unwrap().into()).unwrap()).unwrap();
        assert_eq!(sql, "(NOT ((\"number\" > $1::bigint) IS TRUE))");
        let (sql, params) =
            compile(Filter::none_of("number", vec![json!(1), Value::Null]).unwrap()).unwrap();
        assert_eq!(
            sql,
            "(\"number\" IS NOT NULL AND (NOT (\"number\" IS NOT DISTINCT FROM $1::bigint OR \
             \"number\" IS NULL)))"
        );
        assert_eq!(params, vec![PgValue::Int(1)]);
        let (sql, _) = compile(Filter::eq("text", Value::Null).unwrap()).unwrap();
        assert_eq!(sql, "\"text\" COLLATE \"C\" IS NULL");
    }

    #[test]
    fn empty_membership_decides_without_parameters() {
        let (sql, params) =
            compile(Filter::any_of("number", Vec::<Value>::new()).unwrap()).unwrap();
        assert_eq!(sql, "(\"number\" IS NOT NULL AND (FALSE))");
        assert!(params.is_empty());
        let (sql, params) =
            compile(Filter::none_of("number", Vec::<Value>::new()).unwrap()).unwrap();
        assert_eq!(sql, "(\"number\" IS NOT NULL AND (NOT (FALSE)))");
        assert!(params.is_empty());
        let (sql, _) = compile(Filter::contains_all("tags", Vec::<Value>::new()).unwrap()).unwrap();
        assert_eq!(sql, "(\"tags\" IS NOT NULL AND (TRUE))");
        let (sql, _) = compile(Filter::contains_any("tags", Vec::<Value>::new()).unwrap()).unwrap();
        assert_eq!(sql, "(\"tags\" IS NOT NULL AND (FALSE))");
    }

    #[test]
    fn nested_groups_and_array_membership() {
        let expression = FilterGroup::and(vec![
            FilterGroup::or(vec![
                Filter::eq("flag", true).unwrap().into(),
                Filter::is_null("text").unwrap().into(),
            ])
            .unwrap(),
            Filter::contains_all("tags", vec![json!(null), json!(2)])
                .unwrap()
                .into(),
        ])
        .unwrap();
        let (sql, params) = compile(expression).unwrap();
        assert_eq!(
            sql,
            "((\"flag\" IS NOT DISTINCT FROM $1::boolean OR \"text\" COLLATE \"C\" IS NULL) AND \
             (\"tags\" IS NOT NULL AND (EXISTS (SELECT 1 FROM jsonb_array_elements(\"tags\") AS \
             item(value) WHERE item.value = $2::jsonb) AND EXISTS (SELECT 1 FROM \
             jsonb_array_elements(\"tags\") AS item(value) WHERE item.value = $3::jsonb))))"
        );
        assert_eq!(
            params,
            vec![
                PgValue::Bool(true),
                PgValue::Json(Value::Null),
                PgValue::Json(json!(2))
            ]
        );
    }

    #[test]
    fn parameter_order_follows_placeholder_order() {
        let expression = FilterGroup::or(vec![
            Filter::eq("text", "one").unwrap().into(),
            Filter::between("number", 2, 3).unwrap().into(),
            Filter::eq("text", "last").unwrap().into(),
        ])
        .unwrap();
        let (sql, params) = compile(expression).unwrap();
        for n in 1..=4 {
            assert!(sql.contains(&format!("${n}::")));
        }
        assert!(!sql.contains("$5"));
        assert_eq!(
            params,
            vec![
                PgValue::Text("one".into()),
                PgValue::Int(2),
                PgValue::Int(3),
                PgValue::Text("last".into())
            ]
        );
    }

    #[test]
    fn typed_operands_carry_their_cast() {
        let (sql, params) =
            compile(Filter::eq("ref", "00000000-0000-0000-0000-000000000001").unwrap()).unwrap();
        assert_eq!(sql, "\"ref\" IS NOT DISTINCT FROM $1::uuid");
        assert_eq!(params, vec![PgValue::Uuid(Uuid::from_u128(1))]);
        let (sql, _) = compile(Filter::gt("when", "2024-01-02T03:04:05Z").unwrap()).unwrap();
        assert_eq!(sql, "(\"when\" > $1::timestamptz) IS TRUE");
        let (sql, _) = compile(Filter::lt("day", "2024-01-02").unwrap()).unwrap();
        assert_eq!(sql, "(\"day\" < $1::date) IS TRUE");
        let (sql, params) = compile(Filter::eq("meta", json!({"a": [1]})).unwrap()).unwrap();
        assert_eq!(sql, "\"meta\" IS NOT DISTINCT FROM $1::jsonb");
        assert_eq!(params, vec![PgValue::Json(json!({"a": [1]}))]);
        let (sql, params) = compile(Filter::eq("blob", json!([1, 2])).unwrap()).unwrap();
        assert_eq!(sql, "\"blob\" IS NOT DISTINCT FROM $1::bytea");
        assert_eq!(params, vec![PgValue::Bytes(vec![1, 2])]);
    }

    #[test]
    fn numeric_operands_keep_their_precision() {
        let (sql, params) = compile(Filter::eq("number", 1.5).unwrap()).unwrap();
        assert_eq!(sql, "\"number\" IS NOT DISTINCT FROM $1::double precision");
        assert_eq!(params, vec![PgValue::Float(1.5)]);
        let (sql, params) = compile(Filter::gt("number", u64::MAX).unwrap()).unwrap();
        assert_eq!(sql, "(\"number\" > $1::text::numeric) IS TRUE");
        assert_eq!(params, vec![PgValue::Numeric(u64::MAX.to_string())]);
    }

    #[test]
    fn unsupported_filters_fail() {
        for filter in [
            Filter::eq("embedding", json!([1, 2, 3])).unwrap(),
            Filter::eq("missing", 1).unwrap(),
            Filter::eq("meta.nested", 1).unwrap(),
            Filter::gt("flag", 1).unwrap(),
            Filter::gt("tags", 1).unwrap(),
            Filter::gt("number", Value::Null).unwrap(),
            Filter::gt("number", true).unwrap(),
            Filter::gt("number", "1").unwrap(),
            Filter::contains_text("number", "1").unwrap(),
            Filter::contains("text", "a").unwrap(),
            Filter::eq("ref", 5).unwrap(),
            Filter::gt("when", "2024-01-02T03:04:05").unwrap(),
            Filter::new(
                "text",
                FilterOperator::provider("postgres.match").unwrap(),
                Some(json!("x")),
            )
            .unwrap(),
        ] {
            assert!(compile(filter.clone()).is_err(), "{filter:?} should fail");
        }
    }

    #[test]
    fn values_are_adapted_without_coercion() {
        let schema = schema();
        let column = |name: &str| schema.columns.iter().find(|c| c.name == name).unwrap();
        assert_eq!(
            prepare_value(
                column("ref"),
                &json!("00000000-0000-0000-0000-000000000001")
            )
            .unwrap(),
            PgValue::Uuid(Uuid::from_u128(1))
        );
        assert!(prepare_value(column("number"), &json!(1.0)).is_err());
        assert!(prepare_value(column("number"), &json!(u64::MAX)).is_err());
        assert!(prepare_value(column("number"), &json!("1")).is_err());
        assert!(prepare_value(column("flag"), &json!(1)).is_err());
        assert!(prepare_value(column("text"), &json!(1)).is_err());
        assert_eq!(
            prepare_value(column("ratio"), &json!(2)).unwrap(),
            PgValue::Float(2.0)
        );
        assert!(prepare_value(column("when"), &json!("2024-01-02T03:04:05")).is_err());
        assert!(matches!(
            prepare_value(column("when"), &json!("2024-01-02 03:04:05+02:00")).unwrap(),
            PgValue::Timestamp(_)
        ));
        assert!(prepare_value(column("day"), &json!("01/02/2024")).is_err());
        assert!(prepare_value(column("blob"), &json!([256])).is_err());
        assert_eq!(
            prepare_value(column("embedding"), &json!([0, 1, 2.5])).unwrap(),
            PgValue::Vector("[0,1,2.5]".into(), PostgresVectorType::Vector)
        );
        assert!(prepare_value(column("embedding"), &json!("[0,1,2]")).is_err());
        assert!(prepare_value(column("embedding"), &json!([0, 1])).is_err());
        assert!(prepare_value(column("embedding"), &json!([0, true, 1])).is_err());
        assert_eq!(
            prepare_value(column("embedding"), &Value::Null).unwrap(),
            PgValue::Null("text::vector")
        );
        assert!(prepare_key(&schema, &Value::Null).is_err());
    }

    #[test]
    fn schema_sql_matches_upstream() {
        let schema = schema();
        assert_eq!(schema.table, "\"public\".\"documents\"");
        let create = schema.create_table_sql();
        assert!(create.starts_with(
            "CREATE TABLE IF NOT EXISTS \"public\".\"documents\" (\"id\" text COLLATE \"C\" \
             PRIMARY KEY, \"text\" text COLLATE \"C\", \"number\" bigint"
        ));
        assert!(create.ends_with("\"dense vector\" \"vector\"(3))"));
        assert_eq!(
            schema.order_by_sql(&[("text".into(), false)]).unwrap(),
            "\"text\" DESC NULLS LAST, \"id\" ASC"
        );
        assert_eq!(
            schema.order_by_sql(&[("id".into(), false)]).unwrap(),
            "\"id\" DESC NULLS LAST"
        );
        assert!(schema.order_by_sql(&[("meta".into(), true)]).is_err());
        assert!(schema.order_by_sql(&[("embedding".into(), true)]).is_err());
    }

    #[test]
    fn index_names_are_stable_digests() {
        let schema = schema();
        let column = schema.vector_columns().next().unwrap();
        let name = schema.index_name(column);
        assert!(name.starts_with("af_vector_"));
        assert_eq!(name.len(), "af_vector_".len() + 32);
        // sha256("public\0documents\0dense vector")[:32]
        let digest = Sha256::digest(b"public\0documents\0dense vector");
        let expected: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(name, format!("af_vector_{expected}"));
        // An exact field has no index.
        assert!(schema.index_sql(column).is_none());
    }

    fn vector_field(kind: &str, distance: &str) -> VectorStoreField {
        VectorStoreField::vector("v", 3)
            .with_index_kind(IndexKind::new(kind))
            .with_distance_function(DistanceFunction::new(distance))
    }

    #[test]
    fn ann_indexes_use_the_metric_opclass_and_options() {
        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            vector_field("hnsw", "cosine_similarity"),
        ])
        .unwrap();
        let mut options = HashMap::new();
        options.insert(
            "v".to_string(),
            PostgresVectorOptions::new()
                .with_m(12)
                .with_ef_construction(40),
        );
        let schema = Schema::new("s", "t", definition, &options, false).unwrap();
        let column = schema.vector_columns().next().unwrap();
        assert_eq!(
            schema.index_sql(column).unwrap(),
            format!(
                "CREATE INDEX IF NOT EXISTS \"{}\" ON \"s\".\"t\" USING hnsw (\"v\" \
                 \"vector_cosine_ops\") WITH (m = 12, ef_construction = 40)",
                schema.index_name(column)
            )
        );

        let definition = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            vector_field("ivf_flat", "euclidean_distance").with_type("float16"),
        ])
        .unwrap();
        let schema = Schema::new("s", "t", definition, &HashMap::new(), false).unwrap();
        let column = schema.vector_columns().next().unwrap();
        assert!(schema
            .index_sql(column)
            .unwrap()
            .ends_with("USING ivfflat (\"v\" \"halfvec_l2_ops\") WITH (lists = 100)"));
        assert_eq!(column.column_type(), "\"halfvec\"(3)");
    }

    #[test]
    fn vector_capabilities_are_rejected_before_io() {
        let build = |field: VectorStoreField, options: Option<PostgresVectorOptions>| {
            let definition = VectorStoreCollectionDefinition::new(vec![
                VectorStoreField::key("id").with_type("str"),
                field,
            ])
            .unwrap();
            let mut map = HashMap::new();
            if let Some(options) = options {
                map.insert("v".to_string(), options);
            }
            Schema::new("public", "t", definition, &map, false)
        };
        assert!(build(vector_field("ivf_flat", "manhattan"), None).is_err());
        assert!(build(vector_field("disk_ann", "cosine_distance"), None).is_err());
        assert!(build(vector_field("flat", "hamming"), None).is_err());
        assert!(build(
            vector_field("flat", "cosine_distance"),
            Some(PostgresVectorOptions::new().with_m(8))
        )
        .is_err());
        assert!(build(
            vector_field("hnsw", "cosine_distance"),
            Some(PostgresVectorOptions::new().with_lists(8))
        )
        .is_err());
        assert!(build(
            vector_field("hnsw", "cosine_distance"),
            Some(
                PostgresVectorOptions::new()
                    .with_m(40)
                    .with_ef_construction(64)
            )
        )
        .is_err());
        assert!(build(
            vector_field("flat", "cosine_distance").with_type("int"),
            None
        )
        .is_err());
        assert!(build(
            VectorStoreField::vector("v", 2001).with_index_kind(IndexKind::new("hnsw")),
            None
        )
        .is_err());
        assert!(build(
            VectorStoreField::vector("v", 2001).with_index_kind(IndexKind::new("hnsw")),
            Some(PostgresVectorOptions::new().with_vector_type(PostgresVectorType::Halfvec))
        )
        .is_ok());
        assert!(build(VectorStoreField::vector("v", 16_001), None).is_err());
        assert!(build(VectorStoreField::vector("v", 16_000), None).is_ok());
    }

    #[test]
    fn unsupported_fields_are_rejected_before_io() {
        let build = |fields: Vec<VectorStoreField>| {
            Schema::new(
                "public",
                "t",
                VectorStoreCollectionDefinition::new(fields).unwrap(),
                &HashMap::new(),
                false,
            )
        };
        // Untyped data field.
        assert!(build(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("x"),
        ])
        .is_err());
        // Full-text indexed.
        assert!(build(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("x")
                .with_type("str")
                .full_text_indexed(),
        ])
        .is_err());
        // Float key.
        assert!(build(vec![VectorStoreField::key("id").with_type("float")]).is_err());
        // Too-long storage name.
        assert!(build(vec![VectorStoreField::key("id")
            .with_type("str")
            .with_storage_name("x".repeat(64))])
        .is_err());
        assert!(Schema::new(
            "public",
            &"t".repeat(64),
            VectorStoreCollectionDefinition::new(
                vec![VectorStoreField::key("id").with_type("str")]
            )
            .unwrap(),
            &HashMap::new(),
            false,
        )
        .is_err());
    }

    #[test]
    fn generated_keys_get_server_defaults() {
        for (kind, suffix) in [
            ("int", "bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY"),
            (
                "UUID",
                "uuid DEFAULT pg_catalog.gen_random_uuid() PRIMARY KEY",
            ),
            (
                "str",
                "text COLLATE \"C\" DEFAULT pg_catalog.gen_random_uuid()::text PRIMARY KEY",
            ),
        ] {
            let schema = Schema::new(
                "public",
                "t",
                VectorStoreCollectionDefinition::new(vec![
                    VectorStoreField::key("id").with_type(kind)
                ])
                .unwrap(),
                &HashMap::new(),
                true,
            )
            .unwrap();
            assert_eq!(
                schema.create_table_sql(),
                format!("CREATE TABLE IF NOT EXISTS \"public\".\"t\" (\"id\" {suffix})")
            );
        }
    }

    #[test]
    fn metrics_map_to_pgvector_operators() {
        for (name, operator, ops, score) in [
            ("cosine_distance", "<=>", "cosine", ScoreForm::Distance),
            ("cosine_similarity", "<=>", "cosine", ScoreForm::OneMinus),
            ("dot_prod", "<#>", "ip", ScoreForm::Negated),
            ("negative_dot_prod", "<#>", "ip", ScoreForm::Distance),
            ("euclidean_distance", "<->", "l2", ScoreForm::Distance),
            ("manhattan", "<+>", "l1", ScoreForm::Distance),
        ] {
            let metric = metric_for(&vector_field("flat", name)).unwrap();
            assert_eq!(
                (metric.operator, metric.ops, metric.score),
                (operator, ops, score)
            );
        }
        let default = metric_for(&VectorStoreField::vector("v", 3)).unwrap();
        assert_eq!(default.operator, "<=>");
        assert_eq!(default.score, ScoreForm::Distance);
    }

    #[test]
    fn key_arrays_use_the_key_type() {
        assert_eq!(
            key_array(&[PgValue::Int(1), PgValue::Int(2)]),
            PgValue::IntArray(vec![1, 2])
        );
        assert_eq!(
            key_array(&[PgValue::Text("a".into())]),
            PgValue::TextArray(vec!["a".into()])
        );
        assert_eq!(key_array(&[PgValue::Uuid(Uuid::nil())]).cast(), "uuid[]");
        assert_eq!(
            key_identity(&PgValue::Uuid(Uuid::from_u128(1))),
            json!("00000000-0000-0000-0000-000000000001")
        );
    }
}
