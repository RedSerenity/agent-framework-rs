//! SQL Server identifiers, values, and portable filter translation.
//!
//! Mirrors upstream's `agent_framework_sql_server/_sql.py`: everything that
//! turns a definition and a filter into T-SQL text plus bound parameters,
//! with no I/O, so every statement can be asserted on offline.

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use time::format_description::well_known::Rfc3339;
use time::{Date, OffsetDateTime, PrimitiveDateTime, UtcOffset};
use uuid::Uuid;

/// SQL Server's limit is 2100 parameters per request; leave room for paging
/// and search arguments — upstream's `MAX_PARAMETERS`.
pub const MAX_PARAMETERS: usize = 2000;

/// Keys per `IN (...)` batch for reads and deletes — upstream's
/// `_KEY_BATCH_SIZE`.
pub(crate) const KEY_BATCH_SIZE: usize = 1000;

/// The largest finite `float32`.
const FLOAT32_MAX: f64 = f32::MAX as f64;

/// A binary collation: case- and accent-sensitive, ordinal comparison —
/// upstream's `_COLLATION`.
pub(crate) const COLLATION: &str = "Latin1_General_100_BIN2";

/// The native `VECTOR` type's dimension ceiling.
pub(crate) const MAX_DIMENSIONS: usize = 1998;

fn config(message: impl Into<String>) -> Error {
    Error::Configuration(message.into())
}

fn invalid_response(message: impl Into<String>) -> Error {
    Error::service(message.into())
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

/// Quote one identifier with brackets, doubling `]` — upstream's
/// `_quote_identifier`. 1–128 UTF-16 code units, no NUL. Data values are
/// never quoted into SQL; they are always bound.
pub(crate) fn quote_identifier(name: &str) -> Result<String> {
    if name.is_empty() || name.contains('\0') || utf16_len(name) > 128 {
        return Err(config(
            "SQL Server identifiers must contain 1-128 UTF-16 code units and no NUL.",
        ));
    }
    Ok(format!("[{}]", name.replace(']', "]]")))
}

// region: kinds and metrics

/// A non-vector field's declared scalar type.
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
    /// Resolve the free-form type hint; upstream's spellings
    /// case-insensitively plus `string` / `integer` / `boolean` / `array` /
    /// `object`. A field without one is refused: the column type has to come
    /// from somewhere.
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
}

/// What the raw `VECTOR_DISTANCE` result is, relative to the reported
/// score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResultKind {
    /// Reported as-is (lower is closer).
    Distance,
    /// Reported as `1 - distance` (higher is closer).
    Similarity,
    /// Reported as `-distance` (higher is closer).
    Negative,
}

/// Upstream's `_METRICS`: `(VECTOR_DISTANCE metric, result kind)`. An
/// undeclared function is cosine **distance**.
pub(crate) fn metric_for(field: &VectorStoreField) -> Result<(&'static str, ResultKind)> {
    let name = field
        .distance_function
        .as_ref()
        .map(|d| d.as_str())
        .unwrap_or("DEFAULT");
    Ok(match name {
        "DEFAULT" | "cosine_distance" => ("cosine", ResultKind::Distance),
        "cosine_similarity" => ("cosine", ResultKind::Similarity),
        "euclidean_distance" => ("euclidean", ResultKind::Distance),
        "dot_prod" => ("dot", ResultKind::Negative),
        "negative_dot_prod" => ("dot", ResultKind::Distance),
        other => {
            return Err(config(format!(
                "Unsupported SQL Server distance function '{other}'."
            )))
        }
    })
}

// endregion

// region: schema

/// One resolved column.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Column {
    pub(crate) name: String,
    pub(crate) storage: String,
    pub(crate) role: FieldType,
    /// `None` for a vector column.
    pub(crate) kind: Option<Kind>,
    /// Dimensions and metric of a vector column.
    pub(crate) vector: Option<(usize, &'static str, ResultKind)>,
    pub(crate) indexed: bool,
}

impl Column {
    pub(crate) fn quoted(&self) -> String {
        format!("[{}]", self.storage.replace(']', "]]"))
    }

    /// The column type — upstream's `_column_type`.
    pub(crate) fn column_type(&self) -> String {
        if let Some((dimensions, _, _)) = self.vector {
            return format!("VECTOR({dimensions})");
        }
        match self.kind.expect("a scalar column") {
            Kind::Str => {
                let width = if self.role == FieldType::Key || self.indexed {
                    "450"
                } else {
                    "MAX"
                };
                format!("NVARCHAR({width}) COLLATE {COLLATION}")
            }
            Kind::Int => "BIGINT".into(),
            Kind::Float => "FLOAT(53)".into(),
            Kind::Bool => "BIT".into(),
            Kind::Uuid => "UNIQUEIDENTIFIER".into(),
            Kind::Bytes => "VARBINARY(MAX)".into(),
            Kind::Date => "DATE".into(),
            Kind::DateTime => "DATETIME2(7)".into(),
            Kind::List | Kind::Dict => "NVARCHAR(MAX)".into(),
        }
    }

    /// A string column limited to `NVARCHAR(450)` (a key or an indexed
    /// field).
    fn is_narrow(&self) -> bool {
        self.role == FieldType::Key || self.indexed
    }

    /// The projected expression. A `VECTOR` comes back through an explicit
    /// `NVARCHAR(MAX)` cast — its JSON form — so the driver never meets the
    /// native type.
    pub(crate) fn projection(&self, alias: &str) -> String {
        if self.vector.is_some() {
            format!("CAST({alias}{} AS NVARCHAR(MAX))", self.quoted())
        } else {
            format!("{alias}{}", self.quoted())
        }
    }
}

/// A validated collection.
#[derive(Debug, Clone)]
pub(crate) struct Schema {
    pub(crate) schema: String,
    pub(crate) table_name: String,
    /// `[schema].[table]`.
    pub(crate) table: String,
    pub(crate) columns: Vec<Column>,
    pub(crate) key: usize,
    pub(crate) auto_generated_key: bool,
    pub(crate) definition: VectorStoreCollectionDefinition,
}

impl Schema {
    /// The checks upstream's `SqlServerCollection.__init__` runs before any
    /// I/O.
    pub(crate) fn new(
        schema: &str,
        table_name: &str,
        definition: VectorStoreCollectionDefinition,
        auto_generated_key: bool,
    ) -> Result<Self> {
        let table = format!(
            "{}.{}",
            quote_identifier(schema)?,
            quote_identifier(table_name)?
        );
        let mut columns = Vec::with_capacity(definition.fields().len());
        let mut key = 0;
        for (index, field) in definition.fields().iter().enumerate() {
            let storage = field.effective_storage_name().to_string();
            quote_identifier(&storage)?;
            if field.is_full_text_indexed == Some(true) {
                return Err(config(
                    "SQL Server full-text and keyword-hybrid search are not supported.",
                ));
            }
            let indexed = field.is_indexed == Some(true);
            let (kind, vector) = if field.field_type == FieldType::Vector {
                let dimensions = field.dimensions.unwrap_or(0);
                if !(1..=MAX_DIMENSIONS).contains(&dimensions) {
                    return Err(config(format!(
                        "SQL Server vector dimensions must be an integer between 1 and \
                         {MAX_DIMENSIONS}."
                    )));
                }
                match field.type_.as_deref().map(str::trim) {
                    None | Some("float") | Some("float32") => {}
                    Some(_) => {
                        return Err(config(
                            "SQL Server VECTOR columns support float32 vectors only.",
                        ))
                    }
                }
                let (metric, result) = metric_for(field)?;
                match field.index_kind.as_ref().map(|k| k.as_str()) {
                    None | Some("default") | Some("flat") => {}
                    Some(_) => {
                        return Err(config(
                            "SQL Server approximate vector indexes are not supported.",
                        ))
                    }
                }
                (None, Some((dimensions, metric, result)))
            } else {
                let kind = Kind::of(field)?;
                if field.field_type == FieldType::Data
                    && indexed
                    && matches!(kind, Kind::Bytes | Kind::List | Kind::Dict)
                {
                    return Err(config(format!(
                        "SQL Server cannot index data fields of type '{}'.",
                        kind.name()
                    )));
                }
                (Some(kind), None)
            };
            if field.field_type == FieldType::Key {
                key = index;
                if !matches!(kind, Some(Kind::Str | Kind::Int | Kind::Uuid)) {
                    return Err(config(format!(
                        "SQL Server keys must be str, int, or UUID; '{}' is '{}'.",
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
                indexed,
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

    pub(crate) fn selected(&self, include_vectors: bool) -> Vec<&Column> {
        self.columns
            .iter()
            .filter(|c| include_vectors || c.vector.is_none())
            .collect()
    }

    /// Upstream's `_column_definitions`.
    pub(crate) fn column_definitions(&self) -> String {
        self.columns
            .iter()
            .map(|column| {
                if column.role == FieldType::Key {
                    let generated = if self.auto_generated_key {
                        match column.kind {
                            Some(Kind::Int) => " IDENTITY(1,1)",
                            Some(Kind::Uuid) => " DEFAULT NEWID()",
                            _ => " DEFAULT CONVERT(NVARCHAR(36), NEWID())",
                        }
                    } else {
                        ""
                    };
                    format!(
                        "{} {}{generated} NOT NULL PRIMARY KEY",
                        column.quoted(),
                        column.column_type()
                    )
                } else {
                    format!("{} {} NULL", column.quoted(), column.column_type())
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// `af_sql_` + the first 32 hex digits of SHA-256(`schema \0 table \0
    /// column`) — upstream's `_index_name`.
    pub(crate) fn index_name(&self, column: &Column) -> String {
        let digest = Sha256::digest(
            format!("{}\0{}\0{}", self.schema, self.table_name, column.storage).as_bytes(),
        );
        let hex: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
        format!("af_sql_{hex}")
    }

    /// `ORDER BY` with nulls last and the key as tie-break — upstream's
    /// `_order_by`.
    pub(crate) fn order_by_sql(&self, order_by: &[(String, bool)]) -> Result<String> {
        let mut parts = Vec::new();
        for (name, ascending) in order_by {
            let column = self.filter_column(name)?;
            if matches!(column.kind, Some(Kind::List | Kind::Dict | Kind::Bytes)) {
                return Err(config(format!(
                    "SQL Server ordering is not supported for '{}'.",
                    column.kind.map(Kind::name).unwrap_or("")
                )));
            }
            let quoted = column.quoted();
            parts.push(format!("CASE WHEN {quoted} IS NULL THEN 1 ELSE 0 END"));
            parts.push(format!(
                "{quoted} {}",
                if *ascending { "ASC" } else { "DESC" }
            ));
        }
        let key = self.key_column();
        if !order_by.iter().any(|(name, _)| *name == key.name) {
            parts.push(format!("{} ASC", key.quoted()));
        }
        Ok(parts.join(", "))
    }

    /// Upstream's `_filter_field`.
    pub(crate) fn filter_column(&self, name: &str) -> Result<&Column> {
        if name.contains('.') {
            return Err(config(
                "SQL Server filters and ordering do not support nested field paths.",
            ));
        }
        let column = self
            .columns
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| config(format!("Unknown SQL Server field '{name}'.")))?;
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

/// The declared type of a typed `NULL` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NullKind {
    Str,
    I64,
    F64,
    Bool,
    Uuid,
    Bytes,
    Date,
    DateTime,
}

/// One bound parameter (`@Pn`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SqlParam {
    Null(NullKind),
    Str(String),
    I64(i64),
    F64(f64),
    Bool(bool),
    Uuid(Uuid),
    Bytes(Vec<u8>),
    Date(Date),
    /// A UTC timestamp for `DATETIME2(7)`.
    DateTime(PrimitiveDateTime),
}

/// One value the server returned.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Cell {
    Null,
    Str(String),
    I64(i64),
    F64(f64),
    Bool(bool),
    Uuid(Uuid),
    Bytes(Vec<u8>),
    Date(Date),
    DateTime(PrimitiveDateTime),
}

/// An ordered parameter list handing out `@P1`, `@P2`, … placeholders,
/// capped at [`MAX_PARAMETERS`].
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct Params {
    pub(crate) values: Vec<SqlParam>,
}

impl Params {
    pub(crate) fn bind(&mut self, value: SqlParam) -> Result<String> {
        if self.values.len() >= MAX_PARAMETERS {
            return Err(config(format!(
                "SQL Server queries support at most {MAX_PARAMETERS} bound parameters."
            )));
        }
        self.values.push(value);
        Ok(format!("@P{}", self.values.len()))
    }
}

fn null_kind(column: &Column) -> NullKind {
    match column.kind {
        None | Some(Kind::Str | Kind::List | Kind::Dict) => NullKind::Str,
        Some(Kind::Int) => NullKind::I64,
        Some(Kind::Float) => NullKind::F64,
        Some(Kind::Bool) => NullKind::Bool,
        Some(Kind::Uuid) => NullKind::Uuid,
        Some(Kind::Bytes) => NullKind::Bytes,
        Some(Kind::Date) => NullKind::Date,
        Some(Kind::DateTime) => NullKind::DateTime,
    }
}

fn type_error(column: &Column) -> Error {
    config(format!(
        "Field '{}' requires a value of type '{}'.",
        column.name,
        column.kind.map(Kind::name).unwrap_or("vector")
    ))
}

/// Upstream's `_prepare_vector`: a JSON array of finite float32 values with
/// exactly the declared dimensions, bound as text that SQL Server converts
/// to `VECTOR`.
pub(crate) fn prepare_vector(column: &Column, value: &Value) -> Result<String> {
    let (dimensions, _, _) = column.vector.expect("a vector column");
    let items = value.as_array().ok_or_else(|| {
        config(format!(
            "Vector field '{}' requires a dense numeric sequence.",
            column.name
        ))
    })?;
    if items.len() != dimensions {
        return Err(config(format!(
            "Vector field '{}' requires {dimensions} dimensions.",
            column.name
        )));
    }
    let mut components = Vec::with_capacity(items.len());
    for item in items {
        let number = item.as_f64().ok_or_else(|| {
            config(format!(
                "Vector field '{}' requires numeric elements, not booleans or strings.",
                column.name
            ))
        })?;
        if !number.is_finite() || number.abs() > FLOAT32_MAX {
            return Err(config(format!(
                "Vector field '{}' requires finite float32 elements.",
                column.name
            )));
        }
        components.push(number);
    }
    Ok(serde_json::to_string(&components)?)
}

pub(crate) fn parse_date(text: &str) -> Option<Date> {
    let format = time::macros::format_description!("[year]-[month]-[day]");
    Date::parse(text, &format).ok()
}

/// Parse a timezone-aware timestamp and normalize it to UTC. `Err(true)`
/// when the text is a valid timestamp with no offset.
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

pub(crate) fn bytes_of(items: &[Value]) -> Option<Vec<u8>> {
    items
        .iter()
        .map(|item| item.as_u64().and_then(|n| u8::try_from(n).ok()))
        .collect()
}

/// Adapt one JSON value — upstream's `_prepare_value`.
pub(crate) fn prepare_value(column: &Column, value: &Value) -> Result<SqlParam> {
    if value.is_null() {
        return Ok(SqlParam::Null(null_kind(column)));
    }
    if column.vector.is_some() {
        return Ok(SqlParam::Str(prepare_vector(column, value)?));
    }
    let kind = column.kind.expect("a scalar column");
    Ok(match (kind, value) {
        (Kind::Uuid, Value::String(s)) => SqlParam::Uuid(
            Uuid::parse_str(s)
                .map_err(|_| config(format!("Field '{}' requires a valid UUID.", column.name)))?,
        ),
        (Kind::Int, Value::Number(n)) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => SqlParam::I64(i),
            (None, Some(_)) => {
                return Err(config(format!(
                    "Field '{}' exceeds the SQL Server bigint range.",
                    column.name
                )))
            }
            _ => return Err(type_error(column)),
        },
        (Kind::Float, Value::Number(n)) => {
            SqlParam::F64(n.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
                config(format!("Field '{}' requires a finite number.", column.name))
            })?)
        }
        (Kind::Bool, Value::Bool(b)) => SqlParam::Bool(*b),
        (Kind::Str, Value::String(s)) => {
            if column.is_narrow() && utf16_len(s) > 450 {
                return Err(config(format!(
                    "Field '{}' exceeds the indexed NVARCHAR(450) limit.",
                    column.name
                )));
            }
            if column.role == FieldType::Key && s.ends_with(' ') {
                return Err(config(
                    "SQL Server string keys cannot end in a space; SQL Server ignores trailing \
                     spaces in keys.",
                ));
            }
            SqlParam::Str(s.clone())
        }
        (Kind::Bytes, Value::Array(items)) => {
            SqlParam::Bytes(bytes_of(items).ok_or_else(|| {
                config(format!(
                    "Field '{}' requires bytes, encoded as an array of integers 0-255.",
                    column.name
                ))
            })?)
        }
        (Kind::Date, Value::String(s)) => SqlParam::Date(parse_date(s).ok_or_else(|| {
            config(format!(
                "Field '{}' requires an ISO-8601 date (YYYY-MM-DD).",
                column.name
            ))
        })?),
        (Kind::DateTime, Value::String(s)) => match parse_timestamp(s) {
            Ok(parsed) => {
                let utc = parsed.to_offset(UtcOffset::UTC);
                SqlParam::DateTime(PrimitiveDateTime::new(utc.date(), utc.time()))
            }
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
        (Kind::List, Value::Array(_)) | (Kind::Dict, Value::Object(_)) => {
            SqlParam::Str(serde_json::to_string(value)?)
        }
        _ => return Err(type_error(column)),
    })
}

pub(crate) fn prepare_key(schema: &Schema, value: &Value) -> Result<SqlParam> {
    if value.is_null() {
        return Err(config("SQL Server keys cannot be null."));
    }
    prepare_value(schema.key_column(), value)
}

/// The canonical JSON form of an adapted key, matching what
/// [`parse_value`] returns for the same key.
pub(crate) fn key_identity(value: &SqlParam) -> Value {
    match value {
        SqlParam::Str(s) => Value::String(s.clone()),
        SqlParam::I64(i) => Value::from(*i),
        SqlParam::Uuid(u) => Value::String(u.hyphenated().to_string()),
        _ => Value::Null,
    }
}

/// Decode a returned cell — upstream's `_parse_value`. A value of the wrong
/// shape is an invalid-response error, never passed through.
pub(crate) fn parse_value(column: &Column, cell: Cell) -> Result<Value> {
    let wrong = |what: &str| {
        invalid_response(format!(
            "SQL Server returned an invalid {what} for '{}'.",
            column.name
        ))
    };
    if cell == Cell::Null {
        return Ok(Value::Null);
    }
    if let Some((dimensions, _, _)) = column.vector {
        let Cell::Str(text) = cell else {
            return Err(wrong("vector"));
        };
        let components: Vec<f64> = serde_json::from_str(&text).map_err(|_| wrong("vector"))?;
        if components.len() != dimensions
            || components
                .iter()
                .any(|c| !c.is_finite() || c.abs() > FLOAT32_MAX)
        {
            return Err(wrong("vector"));
        }
        return Ok(Value::Array(
            components
                .into_iter()
                .map(|c| serde_json::Number::from_f64(c).map(Value::Number))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| wrong("vector"))?,
        ));
    }
    let kind = column.kind.expect("a scalar column");
    Ok(match (kind, cell) {
        (Kind::List | Kind::Dict, Cell::Str(text)) => {
            let parsed: Value = serde_json::from_str(&text).map_err(|_| wrong("JSON value"))?;
            let ok = if kind == Kind::List {
                parsed.is_array()
            } else {
                parsed.is_object()
            };
            if !ok {
                return Err(wrong("JSON type"));
            }
            parsed
        }
        (Kind::Uuid, Cell::Uuid(u)) => Value::String(u.hyphenated().to_string()),
        (Kind::Uuid, Cell::Str(s)) => Value::String(
            Uuid::parse_str(&s)
                .map_err(|_| wrong("UUID"))?
                .hyphenated()
                .to_string(),
        ),
        (Kind::DateTime, Cell::DateTime(dt)) => Value::String(
            dt.assume_utc()
                .format(&Rfc3339)
                .map_err(|_| wrong("datetime"))?,
        ),
        (Kind::Date, Cell::Date(d)) => Value::String(
            d.format(time::macros::format_description!("[year]-[month]-[day]"))
                .map_err(|_| wrong("date"))?,
        ),
        (Kind::Bool, Cell::Bool(b)) => Value::Bool(b),
        (Kind::Bool, Cell::I64(i)) if i == 0 || i == 1 => Value::Bool(i == 1),
        (Kind::Int, Cell::I64(i)) => Value::from(i),
        (Kind::Float, Cell::F64(f)) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| wrong("number"))?,
        (Kind::Float, Cell::I64(i)) => Value::from(i as f64),
        (Kind::Str, Cell::Str(s)) => Value::String(s),
        (Kind::Bytes, Cell::Bytes(b)) => Value::Array(b.into_iter().map(Value::from).collect()),
        _ => return Err(wrong(kind.name())),
    })
}

// endregion

// region: filters

/// Compile bounded, data-only filters into parameterized two-valued T-SQL —
/// upstream's `_FilterCompiler`.
///
/// SQL Server has no `IS NOT DISTINCT FROM` before 2022, so two-valued
/// logic is spelled out: every comparison is guarded by `col IS NOT NULL`,
/// which makes `NOT (...)` behave as the portable semantics require.
pub(crate) struct FilterCompiler<'a> {
    schema: &'a Schema,
    alias: &'a str,
    params: &'a mut Params,
}

impl<'a> FilterCompiler<'a> {
    pub(crate) fn new(schema: &'a Schema, alias: &'a str, params: &'a mut Params) -> Self {
        Self {
            schema,
            alias,
            params,
        }
    }

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
        let kind = column.kind.expect("a scalar column");
        if matches!(kind, Kind::List | Kind::Dict) {
            return Err(config("Equality filtering JSON fields is not supported."));
        }
        if (kind == Kind::Bool) != value.is_boolean() {
            return Ok("1 = 0".into());
        }
        let adapted = match (kind, value) {
            (Kind::Int | Kind::Float, Value::Number(n)) => numeric_operand(column, n)?,
            (Kind::Str, Value::String(_))
            | (Kind::Uuid, Value::String(_))
            | (Kind::Date, Value::String(_))
            | (Kind::DateTime, Value::String(_))
            | (Kind::Bool, Value::Bool(_)) => prepare_value(column, value)?,
            (Kind::Bytes, Value::Array(items)) if bytes_of(items).is_some() => {
                prepare_value(column, value)?
            }
            _ => return Ok("1 = 0".into()),
        };
        let placeholder = self.params.bind(adapted)?;
        if kind == Kind::Str {
            // SQL Server pads trailing spaces even with binary collations;
            // compare bytes instead.
            return Ok(format!(
                "({sql_column} IS NOT NULL AND CONVERT(VARBINARY(MAX), {sql_column}) = \
                 CONVERT(VARBINARY(MAX), CONVERT(NVARCHAR(MAX), {placeholder})))"
            ));
        }
        Ok(format!(
            "({sql_column} IS NOT NULL AND {sql_column} = {placeholder})"
        ))
    }

    fn leaf(&mut self, filter: &Filter) -> Result<String> {
        let schema = self.schema;
        let column = schema.filter_column(&filter.field_name)?;
        let kind = column.kind.expect("a scalar column");
        let sql_column = format!("{}{}", self.alias, column.quoted());
        let null = Value::Null;
        let value = filter.value.as_ref().unwrap_or(&null);
        let items = || value.as_array().map(Vec::as_slice).unwrap_or(&[]);
        Ok(match &filter.operator {
            // SQL columns exist even when their values are NULL.
            FilterOperator::Exists => "1 = 1".into(),
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
                    "1 = 0".to_string()
                } else {
                    parts.join(" OR ")
                };
                let membership = if *op == FilterOperator::In {
                    format!("({choices})")
                } else {
                    format!("(NOT ({choices}))")
                };
                format!("({sql_column} IS NOT NULL AND {membership})")
            }
            op @ (FilterOperator::Gt
            | FilterOperator::Gte
            | FilterOperator::Lt
            | FilterOperator::Lte
            | FilterOperator::Between) => {
                if !matches!(kind, Kind::Int | Kind::Float | Kind::Date | Kind::DateTime) {
                    return Err(config(format!(
                        "Ordered SQL Server filtering is not supported for '{}'.",
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
                let mut adapted = Vec::with_capacity(operands.len());
                for operand in operands {
                    adapted.push(match (kind, operand) {
                        (Kind::Int | Kind::Float, Value::Number(n)) => numeric_operand(column, n)?,
                        _ => prepare_value(column, operand)?,
                    });
                }
                let mut placeholders = Vec::with_capacity(adapted.len());
                for value in adapted {
                    placeholders.push(self.params.bind(value)?);
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
                format!("({sql_column} IS NOT NULL AND {comparison})")
            }
            op @ (FilterOperator::ContainsText
            | FilterOperator::StartsWith
            | FilterOperator::EndsWith) => {
                if kind != Kind::Str {
                    return Err(config("Text filtering requires a string column."));
                }
                let text = value
                    .as_str()
                    .ok_or_else(|| config("Text filtering requires a string operand."))?;
                if text.contains('\0') {
                    return Err(config("SQL Server LIKE does not support NUL characters."));
                }
                let escaped = text
                    .replace('!', "!!")
                    .replace('%', "!%")
                    .replace('_', "!_")
                    .replace('[', "[[]");
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
                if utf16_len(&pattern) * 2 > 8000 {
                    return Err(config("SQL Server LIKE patterns cannot exceed 8000 bytes."));
                }
                let placeholder = self.params.bind(SqlParam::Str(pattern))?;
                format!(
                    "({sql_column} IS NOT NULL AND {sql_column} COLLATE {COLLATION} LIKE \
                     {placeholder} ESCAPE '!')"
                )
            }
            other => {
                return Err(config(format!(
                    "Unsupported SQL Server filter operator '{}'.",
                    other.as_str()
                )))
            }
        })
    }
}

/// Upstream's `_prepare_numeric_filter_value`: an `int` column takes any
/// finite number but integers must fit `bigint`; a `float` column takes any
/// finite number, including integers beyond `bigint`.
fn numeric_operand(column: &Column, n: &serde_json::Number) -> Result<SqlParam> {
    if column.kind == Some(Kind::Int) {
        if let Some(i) = n.as_i64() {
            return Ok(SqlParam::I64(i));
        }
        if n.as_u64().is_some() {
            return Err(config(
                "Integer filter values must fit in the SQL Server bigint range.",
            ));
        }
    }
    n.as_f64()
        .filter(|f| f.is_finite())
        .map(SqlParam::F64)
        .ok_or_else(|| config("Numeric filter values must be finite."))
}

// endregion

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::vectors::{DistanceFunction, FilterGroup, IndexKind};
    use serde_json::json;

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("str"),
            VectorStoreField::data("text").with_type("str"),
            VectorStoreField::data("label").with_type("str").indexed(),
            VectorStoreField::data("number").with_type("int"),
            VectorStoreField::data("ratio").with_type("float"),
            VectorStoreField::data("flag").with_type("bool"),
            VectorStoreField::data("tags").with_type("list"),
            VectorStoreField::data("when").with_type("datetime"),
            VectorStoreField::data("day").with_type("date"),
            VectorStoreField::data("ref").with_type("UUID"),
            VectorStoreField::data("blob").with_type("bytes"),
            VectorStoreField::vector("embedding", 3).with_storage_name("dense]vector"),
        ])
        .unwrap()
    }

    fn schema() -> Schema {
        Schema::new("dbo", "documents", definition(), false).unwrap()
    }

    fn compile(expression: impl Into<FilterExpression>) -> Result<(String, Vec<SqlParam>)> {
        let schema = schema();
        let mut params = Params::default();
        let sql = FilterCompiler::new(&schema, "", &mut params).compile(&expression.into())?;
        Ok((sql, params.values))
    }

    fn column<'a>(schema: &'a Schema, name: &str) -> &'a Column {
        schema.columns.iter().find(|c| c.name == name).unwrap()
    }

    #[test]
    fn identifiers_reject_invalid_or_truncated_names() {
        assert!(quote_identifier("").is_err());
        assert!(quote_identifier("a\0").is_err());
        assert!(quote_identifier(&"x".repeat(129)).is_err());
        assert!(quote_identifier(&"x".repeat(128)).is_ok());
        // One astral-plane character is two UTF-16 code units.
        assert!(quote_identifier(&"😀".repeat(65)).is_err());
        assert!(quote_identifier(&"😀".repeat(64)).is_ok());
    }

    #[test]
    fn identifiers_quote_embedded_brackets() {
        assert_eq!(quote_identifier("a]b").unwrap(), "[a]]b]");
        assert_eq!(
            quote_identifier("x]; DROP TABLE t; --").unwrap(),
            "[x]]; DROP TABLE t; --]"
        );
        assert_eq!(schema().table, "[dbo].[documents]");
        assert_eq!(column(&schema(), "embedding").quoted(), "[dense]]vector]");
    }

    #[test]
    fn vector_uses_native_type_and_json_values() {
        let schema = schema();
        let embedding = column(&schema, "embedding");
        assert_eq!(embedding.column_type(), "VECTOR(3)");
        assert_eq!(
            prepare_value(embedding, &json!([1, 0.5, -2])).unwrap(),
            SqlParam::Str("[1.0,0.5,-2.0]".into())
        );
        assert_eq!(
            parse_value(embedding, Cell::Str("[1.0,0.5,-2.0]".into())).unwrap(),
            json!([1.0, 0.5, -2.0])
        );
        assert_eq!(
            embedding.projection("t."),
            "CAST(t.[dense]]vector] AS NVARCHAR(MAX))"
        );
        for bad in [
            json!([1, 2]),
            json!("[1,2,3]"),
            json!([1, true, 2]),
            json!([1e39, 0, 0]),
        ] {
            assert!(prepare_value(embedding, &bad).is_err(), "{bad}");
        }
        for bad in [
            Cell::Str("[1,2]".into()),
            Cell::Str("not json".into()),
            Cell::I64(1),
            Cell::Str("[1,2,1e39]".into()),
        ] {
            assert!(parse_value(embedding, bad).is_err());
        }
    }

    #[test]
    fn column_types_follow_upstream() {
        let schema = schema();
        assert_eq!(
            column(&schema, "id").column_type(),
            format!("NVARCHAR(450) COLLATE {COLLATION}")
        );
        assert_eq!(
            column(&schema, "label").column_type(),
            format!("NVARCHAR(450) COLLATE {COLLATION}")
        );
        assert_eq!(
            column(&schema, "text").column_type(),
            format!("NVARCHAR(MAX) COLLATE {COLLATION}")
        );
        for (name, expected) in [
            ("number", "BIGINT"),
            ("ratio", "FLOAT(53)"),
            ("flag", "BIT"),
            ("tags", "NVARCHAR(MAX)"),
            ("when", "DATETIME2(7)"),
            ("day", "DATE"),
            ("ref", "UNIQUEIDENTIFIER"),
            ("blob", "VARBINARY(MAX)"),
        ] {
            assert_eq!(column(&schema, name).column_type(), expected);
        }
    }

    #[test]
    fn vector_types_metrics_and_dimensions_are_restricted() {
        let build = |field: VectorStoreField| {
            Schema::new(
                "dbo",
                "t",
                VectorStoreCollectionDefinition::new(vec![
                    VectorStoreField::key("id").with_type("int"),
                    field,
                ])
                .unwrap(),
                false,
            )
        };
        assert!(build(VectorStoreField::vector("v", 1999)).is_err());
        assert!(build(VectorStoreField::vector("v", 1998)).is_ok());
        assert!(build(VectorStoreField::vector("v", 3).with_type("float16")).is_err());
        assert!(build(
            VectorStoreField::vector("v", 3)
                .with_distance_function(DistanceFunction::new(DistanceFunction::MANHATTAN))
        )
        .is_err());
        assert!(build(
            VectorStoreField::vector("v", 3).with_index_kind(IndexKind::new(IndexKind::DISK_ANN))
        )
        .is_err());
        assert!(build(VectorStoreField::data("x").with_type("dict").indexed()).is_err());
        assert!(build(
            VectorStoreField::data("x")
                .with_type("str")
                .full_text_indexed()
        )
        .is_err());
        assert!(build(VectorStoreField::data("x")).is_err());
        for (name, metric, kind) in [
            ("cosine_distance", "cosine", ResultKind::Distance),
            ("cosine_similarity", "cosine", ResultKind::Similarity),
            ("euclidean_distance", "euclidean", ResultKind::Distance),
            ("dot_prod", "dot", ResultKind::Negative),
            ("negative_dot_prod", "dot", ResultKind::Distance),
        ] {
            let field = VectorStoreField::vector("v", 3)
                .with_distance_function(DistanceFunction::new(name));
            assert_eq!(metric_for(&field).unwrap(), (metric, kind));
        }
    }

    #[test]
    fn typed_values_and_json_do_not_coerce_invalid_data() {
        let schema = schema();
        let c = |name| column(&schema, name);
        assert_eq!(
            prepare_value(c("ref"), &json!("6F9619FF-8B86-D011-B42D-00C04FC964FF")).unwrap(),
            SqlParam::Uuid(Uuid::parse_str("6f9619ff-8b86-d011-b42d-00c04fc964ff").unwrap())
        );
        assert!(prepare_value(c("number"), &json!(1.5)).is_err());
        assert!(prepare_value(c("number"), &json!(u64::MAX)).is_err());
        assert!(prepare_value(c("flag"), &json!(1)).is_err());
        assert!(prepare_value(c("text"), &json!(1)).is_err());
        assert_eq!(
            prepare_value(c("ratio"), &json!(2)).unwrap(),
            SqlParam::F64(2.0)
        );
        assert_eq!(
            prepare_value(c("tags"), &json!(["é", 1])).unwrap(),
            SqlParam::Str("[\"é\",1]".into())
        );
        assert!(prepare_value(c("tags"), &json!({"a": 1})).is_err());
        // Timestamps are normalized to naive UTC for DATETIME2.
        assert_eq!(
            prepare_value(c("when"), &json!("2024-01-02T05:04:05+02:00")).unwrap(),
            SqlParam::DateTime(PrimitiveDateTime::new(
                Date::from_calendar_date(2024, time::Month::January, 2).unwrap(),
                time::Time::from_hms(3, 4, 5).unwrap()
            ))
        );
        assert!(prepare_value(c("when"), &json!("2024-01-02T05:04:05")).is_err());
        assert!(prepare_value(c("day"), &json!("2024-13-01")).is_err());
        // Narrow strings and key trailing spaces.
        assert!(prepare_value(c("label"), &json!("x".repeat(451))).is_err());
        assert!(prepare_value(c("text"), &json!("x".repeat(451))).is_ok());
        assert!(prepare_key(&schema, &json!("key ")).is_err());
        assert!(prepare_key(&schema, &Value::Null).is_err());
        assert_eq!(
            prepare_value(c("number"), &Value::Null).unwrap(),
            SqlParam::Null(NullKind::I64)
        );
        // Parsing refuses the wrong shapes.
        assert!(parse_value(c("tags"), Cell::Str("{}".into())).is_err());
        assert!(parse_value(c("tags"), Cell::Str("[".into())).is_err());
        assert_eq!(parse_value(c("flag"), Cell::I64(1)).unwrap(), json!(true));
        assert!(parse_value(c("flag"), Cell::I64(2)).is_err());
        assert!(parse_value(c("number"), Cell::Str("1".into())).is_err());
        assert_eq!(
            parse_value(
                c("when"),
                Cell::DateTime(PrimitiveDateTime::new(
                    Date::from_calendar_date(2024, time::Month::January, 2).unwrap(),
                    time::Time::from_hms_milli(3, 4, 5, 500).unwrap()
                ))
            )
            .unwrap(),
            json!("2024-01-02T03:04:05.5Z")
        );
        assert_eq!(
            parse_value(c("ref"), Cell::Uuid(Uuid::from_u128(1))).unwrap(),
            json!("00000000-0000-0000-0000-000000000001")
        );
    }

    #[test]
    fn filter_values_are_bound_and_wildcards_escaped() {
        let (sql, params) =
            compile(Filter::contains_text("text", "50%_[x]!'; DROP TABLE t; --").unwrap()).unwrap();
        assert!(!sql.contains("DROP"));
        assert_eq!(
            sql,
            format!("([text] IS NOT NULL AND [text] COLLATE {COLLATION} LIKE @P1 ESCAPE '!')")
        );
        assert_eq!(
            params,
            vec![SqlParam::Str("%50!%!_[[]x]!!'; DROP TABLE t; --%".into())]
        );
        let (_, params) = compile(Filter::starts_with("text", "a").unwrap()).unwrap();
        assert_eq!(params, vec![SqlParam::Str("a%".into())]);
        let (_, params) = compile(Filter::ends_with("text", "a").unwrap()).unwrap();
        assert_eq!(params, vec![SqlParam::Str("%a".into())]);
        assert!(compile(Filter::contains_text("text", "a\0b").unwrap()).is_err());
        assert!(compile(Filter::contains_text("text", "x".repeat(4000)).unwrap()).is_err());
    }

    #[test]
    fn scalar_filters_bind_values_in_order() {
        let (sql, params) = compile(
            FilterGroup::and(vec![
                Filter::eq("number", 1).unwrap().into(),
                Filter::between("ratio", 0.5, 2).unwrap().into(),
                Filter::gt("when", "2024-01-02T03:04:05Z").unwrap().into(),
                Filter::eq("text", "a ").unwrap().into(),
            ])
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            sql,
            "(([number] IS NOT NULL AND [number] = @P1) AND ([ratio] IS NOT NULL AND [ratio] \
             BETWEEN @P2 AND @P3) AND ([when] IS NOT NULL AND [when] > @P4) AND ([text] IS NOT \
             NULL AND CONVERT(VARBINARY(MAX), [text]) = CONVERT(VARBINARY(MAX), \
             CONVERT(NVARCHAR(MAX), @P5))))"
        );
        assert_eq!(params.len(), 5);
        assert_eq!(params[0], SqlParam::I64(1));
        assert_eq!(params[1], SqlParam::F64(0.5));
        assert_eq!(params[2], SqlParam::F64(2.0));
        assert_eq!(params[4], SqlParam::Str("a ".into()));
    }

    #[test]
    fn groups_and_nullable_membership_are_two_valued() {
        let (sql, params) = compile(
            FilterGroup::not(
                Filter::none_of("number", vec![json!(1), Value::Null])
                    .unwrap()
                    .into(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            sql,
            "(NOT (([number] IS NOT NULL AND (NOT (([number] IS NOT NULL AND [number] = @P1) OR \
             [number] IS NULL)))))"
        );
        assert_eq!(params, vec![SqlParam::I64(1)]);
        let (sql, params) =
            compile(Filter::any_of("number", Vec::<Value>::new()).unwrap()).unwrap();
        assert_eq!(sql, "([number] IS NOT NULL AND (1 = 0))");
        assert!(params.is_empty());
        assert_eq!(compile(Filter::exists("tags").unwrap()).unwrap().0, "1 = 1");
        assert_eq!(
            compile(Filter::is_null("tags").unwrap()).unwrap().0,
            "[tags] IS NULL"
        );
        let (sql, _) = compile(Filter::ne("flag", true).unwrap()).unwrap();
        assert_eq!(sql, "(NOT (([flag] IS NOT NULL AND [flag] = @P1)))");
    }

    #[test]
    fn equality_does_not_coerce_mismatched_types() {
        for (field, value) in [
            ("number", json!("1")),
            ("number", json!(true)),
            ("flag", json!(1)),
            ("text", json!(1)),
            ("ref", json!(1)),
            ("blob", json!("abc")),
            ("blob", json!([300])),
            ("day", json!(5)),
        ] {
            let (sql, params) = compile(Filter::eq(field, value.clone()).unwrap()).unwrap();
            assert_eq!(sql, "1 = 0", "{field} = {value}");
            assert!(params.is_empty());
        }
    }

    #[test]
    fn numeric_ranges_follow_the_column_type() {
        // Float columns accept integral values beyond bigint.
        for filter in [
            Filter::eq("ratio", u64::MAX).unwrap(),
            Filter::gt("ratio", u64::MAX).unwrap(),
            Filter::any_of("ratio", vec![json!(u64::MAX)]).unwrap(),
            Filter::between("ratio", 0, u64::MAX).unwrap(),
        ] {
            let (_, params) = compile(filter).unwrap();
            assert!(params.contains(&SqlParam::F64(u64::MAX as f64)));
        }
        // Integer columns reject them.
        for filter in [
            Filter::eq("number", u64::MAX).unwrap(),
            Filter::gt("number", u64::MAX).unwrap(),
            Filter::any_of("number", vec![json!(u64::MAX)]).unwrap(),
        ] {
            assert!(compile(filter).is_err());
        }
        // A fractional operand against an int column is compared, not rounded.
        let (_, params) = compile(Filter::lt("number", 1.5).unwrap()).unwrap();
        assert_eq!(params, vec![SqlParam::F64(1.5)]);
    }

    #[test]
    fn unsupported_filters_fail_explicitly() {
        for filter in [
            Filter::eq("tags", json!([1])).unwrap(),
            Filter::contains("tags", 1).unwrap(),
            Filter::contains_any("tags", vec![json!(1)]).unwrap(),
            Filter::gt("text", "a").unwrap(),
            Filter::gt("flag", true).unwrap(),
            Filter::gt("number", Value::Null).unwrap(),
            Filter::gt("number", "1").unwrap(),
            Filter::contains_text("number", "1").unwrap(),
            Filter::eq("embedding", json!([1, 2, 3])).unwrap(),
            Filter::eq("missing", 1).unwrap(),
            Filter::eq("tags.nested", 1).unwrap(),
            Filter::new(
                "text",
                FilterOperator::provider("sql_server.match").unwrap(),
                Some(json!("x")),
            )
            .unwrap(),
        ] {
            assert!(compile(filter.clone()).is_err(), "{filter:?}");
        }
    }

    #[test]
    fn the_parameter_budget_is_enforced() {
        let mut params = Params::default();
        for _ in 0..MAX_PARAMETERS {
            params.bind(SqlParam::I64(1)).unwrap();
        }
        assert!(params.bind(SqlParam::I64(1)).is_err());
        let many: Vec<Value> = (0..MAX_PARAMETERS as i64 + 1).map(Value::from).collect();
        assert!(compile(Filter::any_of("number", many).unwrap()).is_err());
    }

    #[test]
    fn the_alias_prefixes_every_column() {
        let schema = schema();
        let mut params = Params::default();
        let sql = FilterCompiler::new(&schema, "t.", &mut params)
            .compile(&Filter::is_null("text").unwrap().into())
            .unwrap();
        assert_eq!(sql, "t.[text] IS NULL");
    }

    #[test]
    fn ddl_matches_upstream() {
        let schema = schema();
        let definitions = schema.column_definitions();
        assert!(definitions.starts_with(&format!(
            "[id] NVARCHAR(450) COLLATE {COLLATION} NOT NULL PRIMARY KEY, [text] NVARCHAR(MAX) \
             COLLATE {COLLATION} NULL"
        )));
        assert!(definitions.ends_with("[dense]]vector] VECTOR(3) NULL"));
        let label = column(&schema, "label");
        let name = schema.index_name(label);
        let digest = Sha256::digest(b"dbo\0documents\0label");
        let hex: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(name, format!("af_sql_{hex}"));
        for (kind, generated) in [
            ("int", "BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY"),
            (
                "UUID",
                "UNIQUEIDENTIFIER DEFAULT NEWID() NOT NULL PRIMARY KEY",
            ),
            (
                "str",
                "NVARCHAR(450) COLLATE Latin1_General_100_BIN2 DEFAULT CONVERT(NVARCHAR(36), \
                 NEWID()) NOT NULL PRIMARY KEY",
            ),
        ] {
            let schema = Schema::new(
                "dbo",
                "t",
                VectorStoreCollectionDefinition::new(vec![
                    VectorStoreField::key("id").with_type(kind)
                ])
                .unwrap(),
                true,
            )
            .unwrap();
            assert_eq!(schema.column_definitions(), format!("[id] {generated}"));
        }
    }

    #[test]
    fn ordering_puts_nulls_last_and_breaks_ties_by_key() {
        let schema = schema();
        assert_eq!(
            schema.order_by_sql(&[("number".into(), false)]).unwrap(),
            "CASE WHEN [number] IS NULL THEN 1 ELSE 0 END, [number] DESC, [id] ASC"
        );
        assert_eq!(
            schema.order_by_sql(&[("id".into(), true)]).unwrap(),
            "CASE WHEN [id] IS NULL THEN 1 ELSE 0 END, [id] ASC"
        );
        assert!(schema.order_by_sql(&[("blob".into(), true)]).is_err());
        assert!(schema.order_by_sql(&[("embedding".into(), true)]).is_err());
    }
}
