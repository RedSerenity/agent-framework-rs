//! Redis (RediSearch / Redis Stack) as a [`VectorStore`].
//!
//! Ports `agent_framework_redis._vector_store` (`RedisStore` /
//! `RedisCollection`). A collection is one RediSearch index over the keys
//! under one prefix; each record is one Redis `HASH` (binary vectors, the
//! default) or one RedisJSON document (vector arrays). Search is a native
//! `KNN` (or `VECTOR_RANGE`) query, and the portable
//! [`FilterExpression`] compiles to RediSearch query syntax over `TAG` and
//! `NUMERIC` indexes.
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorCollection, VectorSearchOptions, VectorStoreCollectionDefinition,
//!     VectorStoreField,
//! };
//! use agent_framework_redis::RedisVectorStore;
//! use serde_json::json;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id"),
//!     VectorStoreField::data("category").indexed(),
//!     VectorStoreField::vector("embedding", 3),
//! ])?;
//! let store = RedisVectorStore::new("redis://127.0.0.1:6379")?;
//! let docs = store.collection("docs", definition)?;
//! docs.ensure_collection_exists().await?;
//! docs.upsert(vec![json!({"id": "a", "category": "news", "embedding": [0.1, 0.2, 0.3]})])
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
//!
//! # Server requirements
//!
//! Redis Search with `INDEXMISSING` / `INDEXEMPTY` (RediSearch 2.10+, i.e.
//! Redis Stack 7.4+ or Redis 8), plus RedisJSON for
//! [`RedisStorageType::Json`]. The index lives in database 0 — Redis Search
//! cannot index any other — so a URL selecting another database is refused.
//! Connections must speak RESP2 (the client default).
//!
//! # Naming
//!
//! Index and document prefix are derived from a *namespace* (default
//! `"default"`) and the collection name, each hex-encoded:
//! `af:vector:{hex(namespace)}:index:{hex(collection)}` and
//! `af:vector:{hex(namespace)}:data:{hex(collection)}:`. A record with key
//! `k` lives at `{prefix}k`. [`RedisVectorStore::list_collection_names`]
//! lists only canonical index names within its namespace.
//!
//! # Field rules (mirroring upstream)
//!
//! - Storage names are ASCII identifiers of at most 128 characters and must
//!   not start with `_af_`.
//! - The key is a non-empty string (`type_` unset or `"str"`), and is always
//!   indexed as a `TAG`.
//! - Data `type_` is one of `str` (also the meaning of an unset type),
//!   `bool`, `int`, `float`, `list`, `tuple`, `dict`. Indexed fields map to a
//!   case-sensitive `TAG` (str/bool/list/tuple, separator `U+001F`) or a
//!   `NUMERIC` (int/float) index, both with `INDEXMISSING`.
//! - Vector `type_` is unset/`float`/`float32` (FLOAT32) or `float64`;
//!   distance is unset or `cosine_distance` (`COSINE`),
//!   `euclidean_squared_distance` (`L2`) or `redis.ip` (`IP`); index kind is
//!   unset/`default`/`hnsw` (`HNSW`) or `flat` (`FLAT`).
//! - Full-text indexing is refused: RediSearch `TEXT` is tokenized and is not
//!   the literal matching the portable operators promise.
//! - `HASH` storage has no null; a `null` value is refused (use JSON).
//!
//! # Scores
//!
//! Scores are native Redis distances — **lower is closer** — for every
//! metric: `COSINE` is cosine distance, `L2` squared Euclidean distance and
//! `IP` is `1 - dot product`. When the declared distance function already
//! names that quantity (`cosine_distance`, `euclidean_squared_distance`) the
//! result's `score_kind` is `None`; otherwise it is
//! [`SCORE_KIND_DISTANCE`], so a caller does not read an unset or `redis.ip`
//! function the wrong way round.
//!
//! # Divergences
//!
//! - Records are `serde_json::Value` objects (see the core module docs), so
//!   the upstream `bytes` data and vector types have no JSON representation
//!   here and are refused.
//! - Filtered/ordered retrieval without keys (`get(filter=…, order_by=…)`)
//!   and score thresholds are not part of the core [`VectorCollection`]
//!   trait; the threshold is available as
//!   [`RedisVectorCollection::search_within_distance`].
//! - [`VectorSearchOptions::provider_filter`] is a raw RediSearch query
//!   clause, conjoined with the portable filter.
//! - A borrowed connection is not checked for database 0 / RESP2; that is
//!   the caller's responsibility, as with any borrowed client.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use redis::aio::MultiplexedConnection;
use redis::Value as RValue;
use serde_json::{Map, Value};

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    DistanceFunction, FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator,
    IndexKind, VectorCollection, VectorSearchOptions, VectorSearchResult, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};

use crate::internal::{map_redis_err, LazyConnection};

/// `score_kind` for a native Redis distance that the declared distance
/// function does not name: lower is closer.
pub const SCORE_KIND_DISTANCE: &str = "distance";

/// Default namespace, as upstream.
pub const DEFAULT_NAMESPACE: &str = "default";

/// Default URL when neither an explicit URL nor `REDIS_URL` is given.
pub const DEFAULT_REDIS_URL: &str = "redis://localhost:6379";

/// Upstream's `_BATCH_SIZE`: keys per pipeline.
const BATCH_SIZE: usize = 100;
/// Upstream's `_TAG_SEPARATOR`.
const TAG_SEPARATOR: char = '\u{1f}';
/// The alias the distance is yielded as.
const DISTANCE_ALIAS: &str = "_af_distance";
/// The largest integer an indexed numeric value round-trips exactly.
const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0; // 2^53
/// RediSearch's term-length limit for a TAG value, in bytes.
const MAX_TAG_BYTES: usize = 4096;

/// How records are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedisStorageType {
    /// One `HASH` per record; vectors as little-endian binary. Upstream's
    /// default.
    #[default]
    Hash,
    /// One RedisJSON document per record; vectors as number arrays.
    Json,
}

impl RedisStorageType {
    fn on(self) -> &'static str {
        match self {
            Self::Hash => "HASH",
            Self::Json => "JSON",
        }
    }
}

// region: naming

fn prepare_namespace_component(value: &str) -> Result<String> {
    if value.is_empty() || value.len() > 256 {
        return Err(Error::Configuration(
            "Redis namespace and collection names must contain 1 to 256 UTF-8 bytes".into(),
        ));
    }
    Ok(hex_encode(value.as_bytes()))
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

/// Upstream's `_RedisNamespaceNames`.
struct NamespaceNames {
    index_prefix: String,
    data_prefix: String,
}

impl NamespaceNames {
    fn new(namespace: &str) -> Result<Self> {
        let base = format!("af:vector:{}:", prepare_namespace_component(namespace)?);
        Ok(Self {
            index_prefix: format!("{base}index:"),
            data_prefix: format!("{base}data:"),
        })
    }

    /// `(index name, document prefix)` for a collection.
    fn for_collection(&self, collection: &str) -> Result<(String, String)> {
        let encoded = prepare_namespace_component(collection)?;
        Ok((
            format!("{}{encoded}", self.index_prefix),
            format!("{}{encoded}:", self.data_prefix),
        ))
    }

    /// The canonical collection name an index name encodes, or `None` for a
    /// foreign or malformed one.
    fn try_parse_index_name(&self, index_name: &str) -> Option<String> {
        let encoded = index_name.strip_prefix(&self.index_prefix)?;
        if !(2..=512).contains(&encoded.len()) {
            return None;
        }
        let name = String::from_utf8(hex_decode(encoded)?).ok()?;
        let (canonical, _) = self.for_collection(&name).ok()?;
        (canonical == index_name).then_some(name)
    }
}

// endregion

// region: field specs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Str,
    Bool,
    Int,
    Float,
    List,
    Dict,
}

impl Kind {
    fn of(field: &VectorStoreField) -> Result<Self> {
        Ok(match field.type_.as_deref() {
            None | Some("str") => Self::Str,
            Some("bool") => Self::Bool,
            Some("int") => Self::Int,
            Some("float") => Self::Float,
            Some("list") | Some("tuple") => Self::List,
            Some("dict") => Self::Dict,
            Some(other) => {
                return Err(Error::Configuration(format!(
                    "Redis does not support data type '{other}' (field '{}')",
                    field.name
                )))
            }
        })
    }

    fn numeric(self) -> bool {
        matches!(self, Self::Int | Self::Float)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorDtype {
    F32,
    F64,
}

impl VectorDtype {
    fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    fn redis(self) -> &'static str {
        match self {
            Self::F32 => "FLOAT32",
            Self::F64 => "FLOAT64",
        }
    }
}

#[derive(Debug, Clone)]
struct DataSpec {
    storage: String,
    kind: Kind,
    indexed: bool,
}

#[derive(Debug, Clone)]
struct VectorSpec {
    storage: String,
    dimensions: usize,
    dtype: VectorDtype,
    metric: &'static str,
    algorithm: &'static str,
    declared: Option<String>,
}

fn validate_storage_name(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let first_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let rest_ok = chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !first_ok || !rest_ok || name.len() > 128 || name.starts_with("_af_") {
        return Err(Error::Configuration(format!(
            "Redis storage names must be ASCII identifiers (up to 128 chars), excluding '_af_'; \
             got '{name}'"
        )));
    }
    Ok(())
}

fn metric_of(field: &VectorStoreField) -> Result<&'static str> {
    Ok(
        match field
            .distance_function
            .as_ref()
            .map(DistanceFunction::as_str)
        {
            None | Some(DistanceFunction::COSINE_DISTANCE) => "COSINE",
            Some(DistanceFunction::EUCLIDEAN_SQUARED_DISTANCE) => "L2",
            Some("redis.ip") => "IP",
            Some(other) => {
                return Err(Error::Configuration(format!(
                    "Redis does not support distance function '{other}'"
                )))
            }
        },
    )
}

fn algorithm_of(field: &VectorStoreField) -> Result<&'static str> {
    Ok(match field.index_kind.as_ref().map(IndexKind::as_str) {
        None | Some(IndexKind::DEFAULT) | Some(IndexKind::HNSW) => "HNSW",
        Some(IndexKind::FLAT) => "FLAT",
        Some(other) => {
            return Err(Error::Configuration(format!(
                "Redis does not support index kind '{other}'"
            )))
        }
    })
}

fn dtype_of(field: &VectorStoreField) -> Result<VectorDtype> {
    Ok(match field.type_.as_deref() {
        None | Some("float") | Some("float32") => VectorDtype::F32,
        Some("float64") => VectorDtype::F64,
        Some(other) => {
            return Err(Error::Configuration(format!(
                "Redis vector field '{}' does not support type '{other}' (bytes vectors have no \
                 JSON record representation in this port)",
                field.name
            )))
        }
    })
}

// endregion

// region: value preparation

fn prepare_numeric_value(value: &Value) -> Result<f64> {
    let number = value
        .as_f64()
        .filter(|_| value.is_number())
        .ok_or_else(|| {
            Error::Configuration(
                "Redis numeric filters and indexed values require numbers, not booleans".into(),
            )
        })?;
    let exact_int = value.is_i64() || value.is_u64();
    if !number.is_finite() || (exact_int && number.abs() > MAX_EXACT_INTEGER) {
        return Err(Error::Configuration(
            "Redis indexed numbers must be finite; integers must be within [-2**53, 2**53]".into(),
        ));
    }
    if exact_int {
        // `as_f64` on a u64/i64 beyond 2^53 rounds, so recheck exactly.
        let in_range = value
            .as_i64()
            .map(|i| i.unsigned_abs() <= 1 << 53)
            .or_else(|| value.as_u64().map(|u| u <= 1 << 53))
            .unwrap_or(false);
        if !in_range {
            return Err(Error::Configuration(
                "Redis indexed integers must be within [-2**53, 2**53]".into(),
            ));
        }
    }
    Ok(number)
}

fn prepare_tag_value(value: &Value) -> Result<&str> {
    let text = value
        .as_str()
        .ok_or_else(|| Error::Configuration("Redis string TAG values must be strings".into()))?;
    if text != text.trim() || text.contains(TAG_SEPARATOR) || text.contains('\0') {
        return Err(Error::Configuration(
            "indexed Redis strings cannot have surrounding whitespace, NUL, or the TAG separator \
             U+001F"
                .into(),
        ));
    }
    if text.len() > MAX_TAG_BYTES {
        return Err(Error::Configuration(
            "Redis TAG values cannot exceed the native 4096-byte term limit".into(),
        ));
    }
    Ok(text)
}

/// Prepare a query or record vector: the declared width, numbers only,
/// finite in the field's datatype, non-zero for cosine. Returns the values
/// cast to the datatype's precision.
fn prepare_vector(values: &[f64], spec: &VectorSpec, name: &str) -> Result<Vec<f64>> {
    if values.len() != spec.dimensions {
        return Err(Error::Configuration(format!(
            "vector '{name}' must contain exactly {} dimensions, got {}",
            spec.dimensions,
            values.len()
        )));
    }
    let cast: Vec<f64> = values
        .iter()
        .map(|v| match spec.dtype {
            VectorDtype::F32 => f64::from(*v as f32),
            VectorDtype::F64 => *v,
        })
        .collect();
    if cast.iter().any(|v| !v.is_finite()) {
        return Err(Error::Configuration(format!(
            "vector '{name}' must contain only finite values representable in its datatype"
        )));
    }
    if spec.metric == "COSINE" && cast.iter().all(|v| *v == 0.0) {
        return Err(Error::Configuration(format!(
            "cosine vector '{name}' must have a nonzero magnitude"
        )));
    }
    Ok(cast)
}

fn json_vector(value: &Value, name: &str) -> Result<Vec<f64>> {
    let items = value.as_array().ok_or_else(|| {
        Error::Configuration(format!("vector '{name}' must be an array of numbers"))
    })?;
    items
        .iter()
        .map(|item| {
            if item.is_number() {
                item.as_f64().ok_or_else(|| {
                    Error::Configuration(format!("vector '{name}' holds a non-finite number"))
                })
            } else {
                Err(Error::Configuration(format!(
                    "vector '{name}' must contain numbers, not booleans or strings"
                )))
            }
        })
        .collect()
}

fn vector_bytes(values: &[f64], dtype: VectorDtype) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * dtype.width());
    for v in values {
        match dtype {
            VectorDtype::F32 => out.extend_from_slice(&(*v as f32).to_le_bytes()),
            VectorDtype::F64 => out.extend_from_slice(&v.to_le_bytes()),
        }
    }
    out
}

fn vector_from_bytes(bytes: &[u8], spec: &VectorSpec, name: &str) -> Result<Vec<f64>> {
    if bytes.len() != spec.dimensions * spec.dtype.width() {
        return Err(Error::service(format!(
            "binary vector '{name}' has an invalid byte length"
        )));
    }
    Ok(match spec.dtype {
        VectorDtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        VectorDtype::F64 => bytes
            .chunks_exact(8)
            .map(|c| {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                f64::from_le_bytes(a)
            })
            .collect(),
    })
}

fn vector_json(values: Vec<f64>) -> Value {
    Value::Array(
        values
            .into_iter()
            .map(|v| serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number))
            .collect(),
    )
}

// endregion

// region: filter compilation

/// Upstream's `TokenEscaper(re.compile(r"[^\w]", re.UNICODE))`.
fn escape_tag(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        if !(c.is_alphanumeric() || c == '_') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn fmt_number(n: f64) -> String {
    // Debug keeps a decimal point or exponent (`1.0`, `1e300`), both of
    // which RediSearch's numeric parser accepts.
    format!("{n:?}")
}

fn tag_filter(name: &str, value: &str) -> String {
    // RedisVL turns an empty TAG into `*`; INDEXEMPTY needs an explicit empty.
    if value.is_empty() {
        format!("@{name}:{{\"\"}}")
    } else {
        format!("@{name}:{{{}}}", escape_tag(value))
    }
}

fn filter_and(parts: &[String]) -> String {
    if parts.is_empty() {
        "*".to_string()
    } else {
        format!("({})", parts.join(" "))
    }
}

fn filter_or(parts: &[String]) -> Result<String> {
    if parts.is_empty() {
        return Err(Error::Configuration(
            "Redis OR expressions require at least one operand".into(),
        ));
    }
    Ok(format!("({})", parts.join(" | ")))
}

fn filter_not(part: &str) -> String {
    format!("(-({part}))")
}

fn filter_false(name: &str) -> String {
    format!("(ismissing(@{name}) -ismissing(@{name}))")
}

fn non_null_filter(name: &str, kind: Kind, storage: RedisStorageType) -> Result<String> {
    if storage == RedisStorageType::Hash {
        return Ok(filter_not(&format!("ismissing(@{name})")));
    }
    match kind {
        Kind::Int | Kind::Float => Ok(format!("@{name}:[-inf +inf]")),
        Kind::Bool => Ok(format!("@{name}:{{true|false}}")),
        _ => Err(Error::Configuration(
            "Redis JSON TAG indexes cannot distinguish null from all non-null string/collection \
             values without enumerating terms; this operator is unsupported"
                .into(),
        )),
    }
}

fn equality_filter(name: &str, kind: Kind, value: &Value) -> Result<String> {
    if value.is_null() {
        return Err(Error::Configuration(
            "use is_null/is_not_null instead of equality with null".into(),
        ));
    }
    match kind {
        Kind::List | Kind::Dict => Err(Error::Configuration(
            "Redis supports string collection membership, not whole-collection equality".into(),
        )),
        Kind::Int | Kind::Float => {
            if !value.is_number() {
                return Ok(filter_false(name));
            }
            let n = fmt_number(prepare_numeric_value(value)?);
            Ok(format!("@{name}:[{n} {n}]"))
        }
        Kind::Bool => Ok(match value.as_bool() {
            Some(b) => tag_filter(name, if b { "true" } else { "false" }),
            None => filter_false(name),
        }),
        Kind::Str => Ok(match value {
            Value::String(_) => tag_filter(name, prepare_tag_value(value)?),
            _ => filter_false(name),
        }),
    }
}

fn operand_array(filter: &Filter) -> Result<&[Value]> {
    filter
        .value
        .as_ref()
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| {
            Error::Configuration(format!(
                "filter operator '{}' requires an array operand",
                filter.operator
            ))
        })
}

fn condition_filter(filter: &Filter, spec: &DataSpec, storage: RedisStorageType) -> Result<String> {
    let name = spec.storage.as_str();
    let kind = spec.kind;
    let null = Value::Null;
    let value = filter.value.as_ref().unwrap_or(&null);
    let present = filter_not(&format!("ismissing(@{name})"));
    match &filter.operator {
        FilterOperator::Exists => Ok(present),
        FilterOperator::IsNull | FilterOperator::IsNotNull => {
            let non_null = non_null_filter(name, kind, storage)?;
            Ok(if filter.operator == FilterOperator::IsNotNull {
                non_null
            } else {
                filter_and(&[present, filter_not(&non_null)])
            })
        }
        FilterOperator::Eq => equality_filter(name, kind, value),
        FilterOperator::Ne => Ok(filter_and(&[
            present,
            filter_not(&equality_filter(name, kind, value)?),
        ])),
        FilterOperator::In | FilterOperator::NotIn => {
            // Portable membership never matches an actual null, even when
            // null is among the operands.
            let parts = operand_array(filter)?
                .iter()
                .filter(|v| !v.is_null())
                .map(|v| equality_filter(name, kind, v))
                .collect::<Result<Vec<_>>>()?;
            let matches = if parts.is_empty() {
                filter_false(name)
            } else {
                filter_or(&parts)?
            };
            Ok(if filter.operator == FilterOperator::In {
                matches
            } else {
                filter_and(&[non_null_filter(name, kind, storage)?, filter_not(&matches)])
            })
        }
        FilterOperator::Gt
        | FilterOperator::Gte
        | FilterOperator::Lt
        | FilterOperator::Lte
        | FilterOperator::Between => {
            if !kind.numeric() {
                return Err(Error::Configuration(
                    "Redis ordered comparisons require a numeric indexed field".into(),
                ));
            }
            if filter.operator == FilterOperator::Between {
                let bounds = operand_array(filter)?;
                let [lower, upper] = bounds else {
                    return Err(Error::Configuration(
                        "between requires exactly [lower, upper]".into(),
                    ));
                };
                return Ok(format!(
                    "@{name}:[{} {}]",
                    fmt_number(prepare_numeric_value(lower)?),
                    fmt_number(prepare_numeric_value(upper)?)
                ));
            }
            let n = fmt_number(prepare_numeric_value(value)?);
            Ok(match filter.operator {
                FilterOperator::Gt => format!("@{name}:[({n} +inf]"),
                FilterOperator::Gte => format!("@{name}:[{n} +inf]"),
                FilterOperator::Lt => format!("@{name}:[-inf ({n}]"),
                _ => format!("@{name}:[-inf {n}]"),
            })
        }
        FilterOperator::Contains | FilterOperator::ContainsAny | FilterOperator::ContainsAll => {
            if kind != Kind::List {
                return Err(Error::Configuration(
                    "Redis collection membership requires an indexed list or tuple of strings"
                        .into(),
                ));
            }
            let values: Vec<&Value> = if filter.operator == FilterOperator::Contains {
                vec![value]
            } else {
                operand_array(filter)?.iter().collect()
            };
            if values.is_empty() && filter.operator == FilterOperator::ContainsAll {
                return Err(Error::Configuration(
                    "Redis cannot distinguish null from an empty TAG array for contains_all=[]"
                        .into(),
                ));
            }
            let parts = values
                .into_iter()
                .map(|v| Ok(tag_filter(name, prepare_tag_value(v)?)))
                .collect::<Result<Vec<_>>>()?;
            if parts.is_empty() {
                return Ok(filter_false(name));
            }
            if filter.operator == FilterOperator::ContainsAll {
                Ok(filter_and(&parts))
            } else {
                filter_or(&parts)
            }
        }
        other => Err(Error::Configuration(format!(
            "Redis does not support portable operator '{other}'; TEXT tokenization is not \
             literal string matching"
        ))),
    }
}

// endregion

// region: response helpers

fn bytes_of(value: &RValue) -> Option<&[u8]> {
    match value {
        RValue::BulkString(b) => Some(b),
        RValue::SimpleString(s) => Some(s.as_bytes()),
        RValue::VerbatimString { text, .. } => Some(text.as_bytes()),
        RValue::Okay => Some(b"OK"),
        _ => None,
    }
}

fn text_of(value: &RValue) -> Option<String> {
    match value {
        RValue::Int(i) => Some(i.to_string()),
        RValue::Double(d) => Some(d.to_string()),
        other => bytes_of(other).map(|b| String::from_utf8_lossy(b).into_owned()),
    }
}

fn array_of(value: &RValue) -> Option<Vec<&RValue>> {
    match value {
        RValue::Array(items) | RValue::Set(items) => Some(items.iter().collect()),
        RValue::Map(pairs) => Some(pairs.iter().flat_map(|(k, v)| [k, v]).collect()),
        _ => None,
    }
}

/// Look up `key` (case-insensitively) in a flat `[k, v, k, v, …]` reply.
fn lookup<'a>(items: &[&'a RValue], key: &str) -> Option<&'a RValue> {
    items.iter().enumerate().find_map(|(i, item)| {
        text_of(item)
            .filter(|t| t.eq_ignore_ascii_case(key))
            .and_then(|_| items.get(i + 1).copied())
    })
}

/// One index attribute as `FT.INFO` describes it, reduced to what this
/// connector creates.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AttributeSignature {
    attribute: String,
    identifier: String,
    kind: String,
    details: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SchemaSignature {
    key_type: String,
    prefixes: Vec<String>,
    attributes: Vec<AttributeSignature>,
}

const VECTOR_DETAIL_KEYS: [&str; 4] = ["algorithm", "data_type", "dim", "distance_metric"];

fn parse_ft_info(reply: &RValue) -> Result<SchemaSignature> {
    let malformed = || Error::service("Redis returned a malformed FT.INFO reply");
    let top = array_of(reply).ok_or_else(malformed)?;
    let definition = lookup(&top, "index_definition")
        .and_then(array_of)
        .ok_or_else(malformed)?;
    let key_type = lookup(&definition, "key_type")
        .and_then(text_of)
        .ok_or_else(malformed)?
        .to_ascii_uppercase();
    let prefixes = lookup(&definition, "prefixes")
        .and_then(array_of)
        .map(|p| p.into_iter().filter_map(text_of).collect())
        .unwrap_or_default();
    let mut attributes = Vec::new();
    for attribute in lookup(&top, "attributes")
        .and_then(array_of)
        .unwrap_or_default()
    {
        let items = array_of(attribute).ok_or_else(malformed)?;
        let get = |k: &str| lookup(&items, k).and_then(text_of);
        let kind = get("type").ok_or_else(malformed)?.to_ascii_uppercase();
        let mut details = BTreeMap::new();
        if kind == "VECTOR" {
            for key in VECTOR_DETAIL_KEYS {
                if let Some(v) = get(key) {
                    details.insert(key.to_string(), v.to_ascii_uppercase());
                }
            }
        } else if kind == "TAG" {
            if let Some(sep) = get("SEPARATOR") {
                details.insert("separator".into(), sep);
            }
            let flag = items
                .iter()
                .any(|i| text_of(i).is_some_and(|t| t.eq_ignore_ascii_case("CASESENSITIVE")));
            details.insert("casesensitive".into(), flag.to_string());
        }
        attributes.push(AttributeSignature {
            attribute: get("attribute").ok_or_else(malformed)?,
            identifier: get("identifier").ok_or_else(malformed)?,
            kind,
            details,
        });
    }
    attributes.sort();
    Ok(SchemaSignature {
        key_type,
        prefixes,
        attributes,
    })
}

// endregion

/// A Redis vector store: a namespace plus a shared connection. Mirrors
/// upstream's `RedisStore`.
#[derive(Clone)]
pub struct RedisVectorStore {
    conn: Arc<LazyConnection>,
    storage_type: RedisStorageType,
    namespace: String,
}

impl std::fmt::Debug for RedisVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisVectorStore")
            .field("namespace", &self.namespace)
            .field("storage_type", &self.storage_type)
            .finish_non_exhaustive()
    }
}

impl RedisVectorStore {
    /// Connect (lazily) to `redis_url`. The URL must select database 0 and
    /// RESP2.
    pub fn new(redis_url: &str) -> Result<Self> {
        let conn = LazyConnection::open(redis_url)?;
        if conn.db() != Some(0) {
            return Err(Error::Configuration(
                "Redis vector stores require database 0; Redis Search cannot index other \
                 databases"
                    .into(),
            ));
        }
        if conn.resp3() {
            return Err(Error::Configuration(
                "Redis vector stores require RESP protocol 2".into(),
            ));
        }
        Ok(Self::with_conn(conn))
    }

    /// Connect using `REDIS_URL`, falling back to [`DEFAULT_REDIS_URL`] —
    /// upstream's settings resolution (without `.env` file support).
    pub fn from_env() -> Result<Self> {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| DEFAULT_REDIS_URL.to_string());
        Self::new(&url)
    }

    /// Use a caller-owned connection (upstream's borrowed `redis_client`). It
    /// must be on database 0 and speak RESP2.
    pub fn with_connection(connection: MultiplexedConnection) -> Self {
        Self::with_conn(LazyConnection::borrowed(connection))
    }

    fn with_conn(conn: LazyConnection) -> Self {
        Self {
            conn: Arc::new(conn),
            storage_type: RedisStorageType::default(),
            namespace: DEFAULT_NAMESPACE.to_string(),
        }
    }

    /// Default storage format for collections opened from this store.
    pub fn with_storage_type(mut self, storage_type: RedisStorageType) -> Self {
        self.storage_type = storage_type;
        self
    }

    /// Isolate this store's indexes and documents under `namespace` (1–256
    /// UTF-8 bytes).
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Result<Self> {
        let namespace = namespace.into();
        prepare_namespace_component(&namespace)?;
        self.namespace = namespace;
        Ok(self)
    }

    /// The namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// The default storage type.
    pub fn storage_type(&self) -> RedisStorageType {
        self.storage_type
    }

    /// Open a collection with the store's storage type. No I/O.
    pub fn collection(
        &self,
        name: impl Into<String>,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<RedisVectorCollection> {
        self.collection_with_storage_type(name, definition, self.storage_type)
    }

    /// Open a collection overriding the storage type. No I/O.
    pub fn collection_with_storage_type(
        &self,
        name: impl Into<String>,
        definition: VectorStoreCollectionDefinition,
        storage_type: RedisStorageType,
    ) -> Result<RedisVectorCollection> {
        RedisVectorCollection::new(
            self.conn.clone(),
            &self.namespace,
            name.into(),
            definition,
            storage_type,
        )
    }

    /// Drop a collection's index and documents by name, refusing an index
    /// whose document prefix is not this namespace's. Upstream's
    /// `_inner_ensure_collection_deleted`; a missing index is not an error.
    pub async fn ensure_collection_deleted(&self, name: &str) -> Result<()> {
        let (index, prefix) = NamespaceNames::new(&self.namespace)?.for_collection(name)?;
        if !index_exists(&self.conn, &index).await? {
            return Ok(());
        }
        let signature = ft_info(&self.conn, &index).await?;
        if signature.prefixes != [prefix] {
            return Err(Error::Configuration(
                "refusing to delete a Redis index with a different document prefix".into(),
            ));
        }
        drop_index(&self.conn, &index).await
    }
}

async fn index_names(conn: &LazyConnection) -> Result<Vec<String>> {
    let mut c = conn.get().await?;
    let reply: RValue = redis::cmd("FT._LIST")
        .query_async(&mut c)
        .await
        .map_err(map_redis_err)?;
    Ok(array_of(&reply)
        .unwrap_or_default()
        .into_iter()
        .filter_map(text_of)
        .collect())
}

async fn index_exists(conn: &LazyConnection, index: &str) -> Result<bool> {
    Ok(index_names(conn).await?.iter().any(|n| n == index))
}

async fn ft_info(conn: &LazyConnection, index: &str) -> Result<SchemaSignature> {
    let mut c = conn.get().await?;
    let reply: RValue = redis::cmd("FT.INFO")
        .arg(index)
        .query_async(&mut c)
        .await
        .map_err(map_redis_err)?;
    parse_ft_info(&reply)
}

async fn drop_index(conn: &LazyConnection, index: &str) -> Result<()> {
    let mut c = conn.get().await?;
    let _: RValue = redis::cmd("FT.DROPINDEX")
        .arg(index)
        .arg("DD")
        .query_async(&mut c)
        .await
        .map_err(map_redis_err)?;
    Ok(())
}

#[async_trait]
impl VectorStore for RedisVectorStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    /// Canonical collection names in this namespace, sorted; foreign index
    /// names are ignored.
    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let names = NamespaceNames::new(&self.namespace)?;
        let mut out: Vec<String> = index_names(&self.conn)
            .await?
            .iter()
            .filter_map(|i| names.try_parse_index_name(i))
            .collect();
        out.sort();
        Ok(out)
    }

    async fn collection_exists(&self, name: &str) -> Result<bool> {
        let (index, _) = NamespaceNames::new(&self.namespace)?.for_collection(name)?;
        index_exists(&self.conn, &index).await
    }
}

/// One Redis vector collection (a RediSearch index plus its documents).
/// Mirrors upstream's `RedisCollection`.
pub struct RedisVectorCollection {
    conn: Arc<LazyConnection>,
    name: String,
    definition: VectorStoreCollectionDefinition,
    storage_type: RedisStorageType,
    index_name: String,
    key_prefix: String,
    /// Indexed non-vector fields (the key included), by logical name.
    indexed: HashMap<String, DataSpec>,
    /// Every non-vector field, by logical name.
    data: HashMap<String, DataSpec>,
    /// Vector fields, by logical name.
    vectors: HashMap<String, VectorSpec>,
}

impl std::fmt::Debug for RedisVectorCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisVectorCollection")
            .field("name", &self.name)
            .field("index_name", &self.index_name)
            .field("storage_type", &self.storage_type)
            .finish_non_exhaustive()
    }
}

impl RedisVectorCollection {
    fn new(
        conn: Arc<LazyConnection>,
        namespace: &str,
        name: String,
        definition: VectorStoreCollectionDefinition,
        storage_type: RedisStorageType,
    ) -> Result<Self> {
        let (index_name, key_prefix) = NamespaceNames::new(namespace)?.for_collection(&name)?;
        let mut indexed = HashMap::new();
        let mut data = HashMap::new();
        let mut vectors = HashMap::new();
        for field in definition.fields() {
            let storage = field.effective_storage_name().to_string();
            validate_storage_name(&storage)?;
            if field.is_full_text_indexed == Some(true) {
                return Err(Error::Configuration(
                    "RedisVectorCollection does not expose tokenized full-text or hybrid search"
                        .into(),
                ));
            }
            match field.field_type {
                FieldType::Vector => {
                    let dimensions = field.dimensions.ok_or_else(|| {
                        Error::Configuration("Redis vector fields require dimensions".into())
                    })?;
                    vectors.insert(
                        field.name.clone(),
                        VectorSpec {
                            storage,
                            dimensions,
                            dtype: dtype_of(field)?,
                            metric: metric_of(field)?,
                            algorithm: algorithm_of(field)?,
                            declared: field
                                .distance_function
                                .as_ref()
                                .map(|d| d.as_str().to_string()),
                        },
                    );
                }
                FieldType::Key | FieldType::Data => {
                    let kind = Kind::of(field)?;
                    let is_key = field.field_type == FieldType::Key;
                    if is_key && kind != Kind::Str {
                        return Err(Error::Configuration(
                            "Redis record keys must be strings (key type 'str')".into(),
                        ));
                    }
                    let is_indexed = is_key || field.is_indexed == Some(true);
                    if is_indexed && kind == Kind::Dict {
                        return Err(Error::Configuration(format!(
                            "Redis cannot index field '{}' of type 'dict'",
                            field.name
                        )));
                    }
                    let spec = DataSpec {
                        storage,
                        kind,
                        indexed: is_indexed,
                    };
                    if is_indexed {
                        indexed.insert(field.name.clone(), spec.clone());
                    }
                    data.insert(field.name.clone(), spec);
                }
            }
        }
        Ok(Self {
            conn,
            name,
            definition,
            storage_type,
            index_name,
            key_prefix,
            indexed,
            data,
            vectors,
        })
    }

    /// The RediSearch index name.
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// The document key prefix.
    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    /// The storage format.
    pub fn storage_type(&self) -> RedisStorageType {
        self.storage_type
    }

    fn fields_in_order(&self) -> impl Iterator<Item = &VectorStoreField> {
        self.definition.fields().iter()
    }

    /// The `FT.CREATE` arguments (after the command name) this collection
    /// issues. Public for inspection and tests.
    pub fn create_index_args(&self) -> Vec<String> {
        let mut args = vec![
            self.index_name.clone(),
            "ON".into(),
            self.storage_type.on().into(),
            "PREFIX".into(),
            "1".into(),
            self.key_prefix.clone(),
            "SCHEMA".into(),
        ];
        let json = self.storage_type == RedisStorageType::Json;
        let push_name = |args: &mut Vec<String>, storage: &str| {
            if json {
                args.extend([format!("$.{storage}"), "AS".into(), storage.to_string()]);
            } else {
                args.push(storage.to_string());
            }
        };
        for field in self.fields_in_order() {
            if let Some(spec) = self.indexed.get(&field.name) {
                push_name(&mut args, &spec.storage);
                if spec.kind.numeric() {
                    args.extend(["NUMERIC", "INDEXMISSING", "SORTABLE"].map(String::from));
                } else {
                    args.extend(
                        [
                            "TAG",
                            "SEPARATOR",
                            "\u{1f}",
                            "CASESENSITIVE",
                            "INDEXEMPTY",
                            "INDEXMISSING",
                        ]
                        .map(String::from),
                    );
                    if spec.kind != Kind::List {
                        args.push("SORTABLE".into());
                    }
                }
            }
        }
        for field in self.fields_in_order() {
            if let Some(spec) = self.vectors.get(&field.name) {
                push_name(&mut args, &spec.storage);
                args.extend([
                    "VECTOR".into(),
                    spec.algorithm.into(),
                    "6".into(),
                    "TYPE".into(),
                    spec.dtype.redis().into(),
                    "DIM".into(),
                    spec.dimensions.to_string(),
                    "DISTANCE_METRIC".into(),
                    spec.metric.into(),
                ]);
            }
        }
        args
    }

    fn expected_signature(&self) -> SchemaSignature {
        let json = self.storage_type == RedisStorageType::Json;
        let identifier = |storage: &str| {
            if json {
                format!("$.{storage}")
            } else {
                storage.to_string()
            }
        };
        let mut attributes = Vec::new();
        for spec in self.indexed.values() {
            let mut details = BTreeMap::new();
            let kind = if spec.kind.numeric() {
                "NUMERIC"
            } else {
                details.insert("separator".into(), TAG_SEPARATOR.to_string());
                details.insert("casesensitive".into(), "true".into());
                "TAG"
            };
            attributes.push(AttributeSignature {
                attribute: spec.storage.clone(),
                identifier: identifier(&spec.storage),
                kind: kind.into(),
                details,
            });
        }
        for spec in self.vectors.values() {
            let details = BTreeMap::from([
                ("algorithm".to_string(), spec.algorithm.to_string()),
                ("data_type".to_string(), spec.dtype.redis().to_string()),
                ("dim".to_string(), spec.dimensions.to_string()),
                ("distance_metric".to_string(), spec.metric.to_string()),
            ]);
            attributes.push(AttributeSignature {
                attribute: spec.storage.clone(),
                identifier: identifier(&spec.storage),
                kind: "VECTOR".into(),
                details,
            });
        }
        attributes.sort();
        SchemaSignature {
            key_type: self.storage_type.on().into(),
            prefixes: vec![self.key_prefix.clone()],
            attributes,
        }
    }

    fn signature_matches(&self, actual: &SchemaSignature) -> bool {
        let expected = self.expected_signature();
        if actual.key_type != expected.key_type
            || actual.prefixes != expected.prefixes
            || actual.attributes.len() != expected.attributes.len()
        {
            return false;
        }
        // A detail the server does not report is not a mismatch (older
        // RediSearch versions omit some), but one it reports differently is.
        actual
            .attributes
            .iter()
            .zip(&expected.attributes)
            .all(|(a, e)| {
                a.attribute == e.attribute
                    && a.identifier == e.identifier
                    && a.kind == e.kind
                    && e.details
                        .iter()
                        .all(|(k, v)| a.details.get(k).is_none_or(|got| got == v))
            })
    }

    async fn validate_schema(&self) -> Result<()> {
        let actual = ft_info(&self.conn, &self.index_name).await?;
        if !self.signature_matches(&actual) {
            return Err(Error::Configuration(format!(
                "Redis index '{}' has an incompatible schema or document prefix",
                self.index_name
            )));
        }
        Ok(())
    }

    async fn require_index(&self) -> Result<()> {
        if !index_exists(&self.conn, &self.index_name).await? {
            return Err(Error::service(
                "Redis collection does not exist; call ensure_collection_exists() first",
            ));
        }
        self.validate_schema().await
    }

    fn prepare_key(key: &Value) -> Result<&str> {
        match key.as_str() {
            Some(k) if !k.is_empty() => Ok(k),
            _ => Err(Error::Configuration(
                "Redis record keys must be nonempty strings".into(),
            )),
        }
    }

    /// Translate a portable filter into a RediSearch query clause. `None`
    /// yields `*`. Public so a caller can see exactly what will run.
    pub fn prepare_filter(&self, filter: Option<&FilterExpression>) -> Result<String> {
        let Some(filter) = filter else {
            return Ok("*".into());
        };
        filter.validate()?;
        self.translate(filter)
    }

    fn translate(&self, expression: &FilterExpression) -> Result<String> {
        match expression {
            FilterExpression::Group(group) => {
                let parts = group
                    .filters
                    .iter()
                    .map(|f| self.translate(f))
                    .collect::<Result<Vec<_>>>()?;
                match group.operator {
                    FilterGroupOperator::And => Ok(filter_and(&parts)),
                    FilterGroupOperator::Or => filter_or(&parts),
                    FilterGroupOperator::Not => Ok(filter_not(parts.first().ok_or_else(|| {
                        Error::Configuration("a 'not' group requires one filter".into())
                    })?)),
                }
            }
            FilterExpression::Condition(filter) => {
                let spec = self.indexed.get(&filter.field_name).ok_or_else(|| {
                    Error::Configuration(format!(
                        "Redis filters require a declared, indexed, non-vector field; got '{}'",
                        filter.field_name
                    ))
                })?;
                condition_filter(filter, spec, self.storage_type)
            }
        }
    }

    fn query_clause(&self, options: &VectorSearchOptions) -> Result<String> {
        let portable = self.prepare_filter(options.filter.as_ref())?;
        Ok(match options.provider_filter.as_deref() {
            Some(raw) if !raw.trim().is_empty() => {
                if portable == "*" {
                    format!("({raw})")
                } else {
                    format!("({portable} ({raw}))")
                }
            }
            _ => portable,
        })
    }

    // region: document encoding

    fn prepare_data_value(&self, value: &Value, spec: &DataSpec, name: &str) -> Result<Stored> {
        let hash = self.storage_type == RedisStorageType::Hash;
        if value.is_null() {
            if hash {
                return Err(Error::Configuration(
                    "Redis HASH has no native null value; use JSON storage for nullable fields"
                        .into(),
                ));
            }
            return Ok(Stored::Json(Value::Null));
        }
        let type_ok = match spec.kind {
            Kind::Str => value.is_string(),
            Kind::Bool => value.is_boolean(),
            Kind::Int => value.is_i64() || value.is_u64(),
            Kind::Float => value.is_number(),
            Kind::List => value.is_array(),
            Kind::Dict => value.is_object(),
        };
        if !type_ok {
            return Err(Error::Configuration(format!(
                "Redis field '{name}' requires {:?} values",
                spec.kind
            )));
        }
        if spec.indexed {
            match spec.kind {
                Kind::Int | Kind::Float => {
                    prepare_numeric_value(value)?;
                }
                Kind::Str => {
                    prepare_tag_value(value)?;
                }
                Kind::List => {
                    let items = value
                        .as_array()
                        .map(|a| a.iter().map(prepare_tag_value).collect::<Result<Vec<_>>>())
                        .unwrap_or_else(|| Ok(Vec::new()))?;
                    if hash {
                        if items.is_empty() || items.iter().any(|i| i.is_empty()) {
                            return Err(Error::Configuration(
                                "Redis HASH TAG arrays cannot preserve empty arrays or empty \
                                 elements"
                                    .into(),
                            ));
                        }
                        return Ok(Stored::Hash(
                            items.join(&TAG_SEPARATOR.to_string()).into_bytes(),
                        ));
                    }
                }
                _ => {}
            }
        }
        if !hash {
            return Ok(Stored::Json(value.clone()));
        }
        Ok(Stored::Hash(match value {
            Value::Bool(b) => b.to_string().into_bytes(),
            Value::String(s) => s.clone().into_bytes(),
            Value::Number(n) => n.to_string().into_bytes(),
            other => serde_json::to_vec(other)?,
        }))
    }

    /// Encode one logical record into a key plus a document.
    fn encode_record(&self, record: &Value) -> Result<(String, Document)> {
        let stored = self.definition.to_storage(record)?;
        let object = stored.as_object().ok_or_else(|| {
            Error::Configuration("a vector store record must be a JSON object".into())
        })?;
        let key_field = self.definition.key_field();
        let key = Self::prepare_key(
            object
                .get(key_field.effective_storage_name())
                .unwrap_or(&Value::Null),
        )?
        .to_string();
        let mut hash_fields = Vec::new();
        let mut json = Map::new();
        for field in self.fields_in_order() {
            let storage = field.effective_storage_name();
            let Some(value) = object.get(storage) else {
                continue;
            };
            let encoded = if let Some(spec) = self.vectors.get(&field.name) {
                if value.is_null() {
                    if self.storage_type == RedisStorageType::Hash {
                        return Err(Error::Configuration(
                            "Redis HASH has no native null value; use JSON storage for nullable \
                             fields"
                                .into(),
                        ));
                    }
                    Stored::Json(Value::Null)
                } else {
                    let values =
                        prepare_vector(&json_vector(value, &field.name)?, spec, &field.name)?;
                    match self.storage_type {
                        RedisStorageType::Hash => Stored::Hash(vector_bytes(&values, spec.dtype)),
                        RedisStorageType::Json => Stored::Json(vector_json(values)),
                    }
                }
            } else {
                let spec = self.data.get(&field.name).ok_or_else(|| {
                    Error::Configuration(format!("unknown field '{}'", field.name))
                })?;
                self.prepare_data_value(value, spec, &field.name)?
            };
            match encoded {
                Stored::Hash(bytes) => hash_fields.push((storage.to_string(), bytes)),
                Stored::Json(v) => {
                    json.insert(storage.to_string(), v);
                }
            }
        }
        Ok((
            key,
            match self.storage_type {
                RedisStorageType::Hash => Document::Hash(hash_fields),
                RedisStorageType::Json => Document::Json(Value::Object(json)),
            },
        ))
    }

    fn decode_hash_value(&self, field: &VectorStoreField, raw: &[u8]) -> Result<Value> {
        if let Some(spec) = self.vectors.get(&field.name) {
            return Ok(vector_json(vector_from_bytes(raw, spec, &field.name)?));
        }
        let spec = self
            .data
            .get(&field.name)
            .ok_or_else(|| Error::service(format!("unknown field '{}'", field.name)))?;
        let text = String::from_utf8(raw.to_vec())
            .map_err(|_| Error::service(format!("Redis field '{}' is not UTF-8", field.name)))?;
        let invalid = |what: &str| {
            Error::service(format!(
                "Redis HASH field '{}' does not hold a valid {what}",
                field.name
            ))
        };
        Ok(match spec.kind {
            Kind::Str => Value::String(text),
            Kind::Int => Value::from(text.parse::<i64>().map_err(|_| invalid("integer"))?),
            Kind::Float => {
                serde_json::Number::from_f64(text.parse::<f64>().map_err(|_| invalid("number"))?)
                    .map(Value::Number)
                    .ok_or_else(|| invalid("finite number"))?
            }
            Kind::Bool => match text.as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => {
                    return Err(Error::service(
                        "Redis boolean HASH fields must contain 'true' or 'false'",
                    ))
                }
            },
            Kind::List if spec.indexed => Value::Array(
                text.split(TAG_SEPARATOR)
                    .map(|s| Value::String(s.to_string()))
                    .collect(),
            ),
            Kind::List | Kind::Dict => serde_json::from_str(&text).map_err(|_| invalid("JSON"))?,
        })
    }

    // endregion

    /// Fetch documents by Redis key, in order; `None` for a missing one.
    async fn fetch_records(
        &self,
        keys: &[String],
        include_vectors: bool,
    ) -> Result<Vec<Option<Value>>> {
        let fields: Vec<&VectorStoreField> = self
            .fields_in_order()
            .filter(|f| include_vectors || f.field_type != FieldType::Vector)
            .collect();
        let names: Vec<&str> = fields.iter().map(|f| f.effective_storage_name()).collect();
        let mut out = Vec::with_capacity(keys.len());
        let mut c = self.conn.get().await?;
        for chunk in keys.chunks(BATCH_SIZE) {
            let mut pipe = redis::pipe();
            for key in chunk {
                if !key.starts_with(&self.key_prefix) {
                    return Err(Error::service(
                        "Redis returned a key outside the collection prefix",
                    ));
                }
                match self.storage_type {
                    RedisStorageType::Hash => {
                        pipe.cmd("EXISTS").arg(key);
                        pipe.cmd("HMGET").arg(key).arg(&names);
                    }
                    RedisStorageType::Json => {
                        let paths: Vec<String> = names.iter().map(|n| format!("$.{n}")).collect();
                        pipe.cmd("JSON.GET").arg(key).arg(&paths);
                    }
                }
            }
            let replies: Vec<RValue> = pipe.query_async(&mut c).await.map_err(map_redis_err)?;
            match self.storage_type {
                RedisStorageType::Hash => {
                    for pair in replies.chunks(2) {
                        let [exists, values] = pair else {
                            return Err(Error::service("Redis returned a truncated pipeline"));
                        };
                        if !matches!(exists, RValue::Int(n) if *n > 0) {
                            out.push(None);
                            continue;
                        }
                        let values = array_of(values).unwrap_or_default();
                        let mut record = Map::new();
                        for (field, value) in fields.iter().zip(values) {
                            if let Some(bytes) = bytes_of(value) {
                                record.insert(
                                    field.effective_storage_name().to_string(),
                                    self.decode_hash_value(field, bytes)?,
                                );
                            }
                        }
                        out.push(Some(Value::Object(record)));
                    }
                }
                RedisStorageType::Json => {
                    for reply in replies {
                        let Some(bytes) = bytes_of(&reply) else {
                            out.push(None);
                            continue;
                        };
                        let mut parsed: Value = serde_json::from_slice(bytes)?;
                        if names.len() == 1 {
                            parsed = serde_json::json!({ format!("$.{}", names[0]): parsed });
                        }
                        let mut record = Map::new();
                        for name in &names {
                            if let Some(first) = parsed
                                .get(format!("$.{name}"))
                                .and_then(Value::as_array)
                                .and_then(|m| m.first())
                            {
                                record.insert((*name).to_string(), first.clone());
                            }
                        }
                        out.push(Some(Value::Object(record)));
                    }
                }
            }
        }
        out.into_iter()
            .map(|r| {
                r.map(|v| self.definition.from_storage(&v, include_vectors))
                    .transpose()
            })
            .collect()
    }

    fn vector_spec(
        &self,
        options: &VectorSearchOptions,
    ) -> Result<(&VectorStoreField, &VectorSpec)> {
        let field = self
            .definition
            .try_get_vector_field(options.vector_field_name.as_deref())
            .ok_or_else(|| {
                Error::Configuration("Redis search requires a declared vector field".into())
            })?;
        let spec = self
            .vectors
            .get(&field.name)
            .ok_or_else(|| Error::Configuration(format!("'{}' is not a vector", field.name)))?;
        Ok((field, spec))
    }

    /// Build the `FT.SEARCH` arguments (after the command name) for a search;
    /// the query vector is returned separately because it is binary.
    fn search_args(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
        max_distance: Option<f64>,
    ) -> Result<(Vec<String>, Vec<u8>)> {
        options.validate()?;
        let (field, spec) = self.vector_spec(options)?;
        let values: Vec<f64> = vector.iter().map(|v| f64::from(*v)).collect();
        let query_vector = vector_bytes(&prepare_vector(&values, spec, &field.name)?, spec.dtype);
        let predicate = self.query_clause(options)?;
        let name = &spec.storage;
        let mut args = vec![self.index_name.clone()];
        let mut params = vec![];
        match max_distance {
            None => args.push(format!(
                "({predicate})=>[KNN {} @{name} $vector AS {DISTANCE_ALIAS}]",
                options.top.saturating_add(options.skip)
            )),
            Some(radius) => {
                if !radius.is_finite() {
                    return Err(Error::Configuration(
                        "Redis distance threshold must be finite".into(),
                    ));
                }
                if radius < 0.0 {
                    return Err(Error::Configuration(
                        "Redis VECTOR_RANGE requires a non-negative distance threshold".into(),
                    ));
                }
                let range = format!(
                    "@{name}:[VECTOR_RANGE $radius $vector]=>{{$yield_distance_as: {DISTANCE_ALIAS}}}"
                );
                args.push(if predicate == "*" {
                    range
                } else {
                    format!("({range} {predicate})")
                });
                params = vec!["radius".to_string(), fmt_number(radius)];
            }
        }
        args.extend([
            "RETURN".into(),
            "1".into(),
            DISTANCE_ALIAS.into(),
            "SORTBY".into(),
            DISTANCE_ALIAS.into(),
            "ASC".into(),
            "LIMIT".into(),
            options.skip.to_string(),
            options.top.to_string(),
            "PARAMS".into(),
            (2 + params.len()).to_string(),
        ]);
        args.extend(params);
        // `vector` value follows as a binary argument; DIALECT 2 is appended
        // by the caller after it.
        args.push("vector".into());
        Ok((args, query_vector))
    }

    async fn run_search(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
        max_distance: Option<f64>,
    ) -> Result<Vec<VectorSearchResult>> {
        let (args, query_vector) = self.search_args(vector, options, max_distance)?;
        let (_, spec) = self.vector_spec(options)?;
        let score_kind = match spec.declared.as_deref() {
            Some(DistanceFunction::COSINE_DISTANCE)
            | Some(DistanceFunction::EUCLIDEAN_SQUARED_DISTANCE) => None,
            _ => Some(SCORE_KIND_DISTANCE.to_string()),
        };
        self.require_index().await?;
        let mut c = self.conn.get().await?;
        let reply: RValue = redis::cmd("FT.SEARCH")
            .arg(&args)
            .arg(query_vector.as_slice())
            .arg("DIALECT")
            .arg("2")
            .query_async(&mut c)
            .await
            .map_err(map_redis_err)?;
        let items = array_of(&reply)
            .ok_or_else(|| Error::service("Redis returned a malformed FT.SEARCH reply"))?;
        let mut keys = Vec::new();
        let mut scores = Vec::new();
        let mut i = 1;
        while i + 1 < items.len() {
            let key = text_of(items[i])
                .ok_or_else(|| Error::service("Redis returned a non-string document key"))?;
            let attrs = array_of(items[i + 1]).unwrap_or_default();
            let score: f64 = lookup(&attrs, DISTANCE_ALIAS)
                .and_then(text_of)
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| Error::service("Redis search result is missing its distance"))?;
            if !score.is_finite() {
                return Err(Error::service(
                    "Redis returned a non-finite vector distance",
                ));
            }
            keys.push(key);
            scores.push(score);
            i += 2;
        }
        let records = self.fetch_records(&keys, options.include_vectors).await?;
        Ok(records
            .into_iter()
            .zip(scores)
            .filter_map(|(record, score)| {
                record.map(|record| VectorSearchResult {
                    record,
                    score: Some(score),
                    score_kind: score_kind.clone(),
                })
            })
            .collect())
    }

    /// A range search: every record within `max_distance` (a native Redis
    /// distance, lower is closer) of `vector`, nearest first, paged by
    /// `options`. Upstream's `score_threshold`, executed as `VECTOR_RANGE`.
    pub async fn search_within_distance(
        &self,
        vector: Vec<f32>,
        max_distance: f64,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        self.run_search(&vector, options, Some(max_distance)).await
    }
}

/// One encoded field value.
enum Stored {
    Hash(Vec<u8>),
    Json(Value),
}

/// One encoded record.
enum Document {
    Hash(Vec<(String, Vec<u8>)>),
    Json(Value),
}

#[async_trait]
impl VectorCollection for RedisVectorCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    /// Create the index if absent (never overwriting one), then validate that
    /// the index present matches this definition.
    async fn ensure_collection_exists(&self) -> Result<()> {
        let init_error = |e: Error| {
            Error::service(format!(
                "Redis vector index initialization failed ({e}). Requires Redis Search with \
                 INDEXMISSING/INDEXEMPTY (RediSearch 2.10+) and RedisJSON for JSON storage"
            ))
        };
        let mut c = self.conn.get().await?;
        if self.storage_type == RedisStorageType::Json {
            // Read-only capability probe; PING alone does not establish
            // RedisJSON support.
            let _: RValue = redis::cmd("JSON.TYPE")
                .arg(format!("{}_af_capability_probe", self.key_prefix))
                .arg("$")
                .query_async(&mut c)
                .await
                .map_err(|e| init_error(map_redis_err(e)))?;
        }
        if !index_exists(&self.conn, &self.index_name)
            .await
            .map_err(init_error)?
        {
            let created: redis::RedisResult<RValue> = redis::cmd("FT.CREATE")
                .arg(self.create_index_args())
                .query_async(&mut c)
                .await;
            if let Err(e) = created {
                if !crate::context_provider::is_index_exists_error(&e.to_string()) {
                    return Err(init_error(map_redis_err(e)));
                }
            }
        }
        self.validate_schema().await
    }

    async fn collection_exists(&self) -> Result<bool> {
        index_exists(&self.conn, &self.index_name).await
    }

    /// Drop the index and its documents (`FT.DROPINDEX … DD`) after
    /// checking it is this collection's; keys under other prefixes are never
    /// touched.
    async fn ensure_collection_deleted(&self) -> Result<()> {
        if index_exists(&self.conn, &self.index_name).await? {
            self.validate_schema().await?;
            drop_index(&self.conn, &self.index_name).await?;
        }
        Ok(())
    }

    /// Replace each record whole (`DEL` + `HSET`, or `JSON.SET $`) in
    /// transactional batches of 100. Every record is validated before
    /// anything is written.
    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let mut encoded = Vec::with_capacity(records.len());
        for (index, record) in records.iter().enumerate() {
            encoded.push(self.encode_record(record).map_err(|e| {
                Error::Configuration(format!("record at index {index} is invalid: {e}"))
            })?);
        }
        self.require_index().await?;
        let mut c = self.conn.get().await?;
        for chunk in encoded.chunks(BATCH_SIZE) {
            let mut pipe = redis::pipe();
            pipe.atomic();
            for (key, document) in chunk {
                let redis_key = format!("{}{key}", self.key_prefix);
                match document {
                    Document::Hash(fields) => {
                        pipe.cmd("DEL").arg(&redis_key).ignore();
                        let mut cmd = redis::cmd("HSET");
                        cmd.arg(&redis_key);
                        for (name, value) in fields {
                            cmd.arg(name).arg(value.as_slice());
                        }
                        pipe.add_command(cmd).ignore();
                    }
                    Document::Json(value) => {
                        pipe.cmd("JSON.SET")
                            .arg(&redis_key)
                            .arg("$")
                            .arg(serde_json::to_string(value)?)
                            .ignore();
                    }
                }
            }
            let _: () = pipe.query_async(&mut c).await.map_err(map_redis_err)?;
        }
        Ok(encoded
            .into_iter()
            .map(|(key, _)| Value::String(key))
            .collect())
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let redis_keys = keys
            .iter()
            .map(|k| Ok(format!("{}{}", self.key_prefix, Self::prepare_key(k)?)))
            .collect::<Result<Vec<_>>>()?;
        self.require_index().await?;
        self.fetch_records(&redis_keys, include_vectors).await
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        let redis_keys = keys
            .iter()
            .map(|k| Ok(format!("{}{}", self.key_prefix, Self::prepare_key(k)?)))
            .collect::<Result<Vec<_>>>()?;
        if redis_keys.is_empty() {
            return Ok(());
        }
        self.require_index().await?;
        let mut c = self.conn.get().await?;
        for chunk in redis_keys.chunks(BATCH_SIZE) {
            let _: () = redis::cmd("DEL")
                .arg(chunk)
                .query_async(&mut c)
                .await
                .map_err(map_redis_err)?;
        }
        Ok(())
    }

    /// Dense `KNN` search over the chosen vector field, filtered by the
    /// portable filter (and any raw `provider_filter` clause), nearest first.
    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        self.run_search(&vector, options, None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::vectors::FilterGroup;
    use serde_json::json;

    fn store() -> RedisVectorStore {
        RedisVectorStore::new("redis://127.0.0.1:6379").unwrap()
    }

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::data("title").indexed(),
            VectorStoreField::data("year").with_type("int").indexed(),
            VectorStoreField::data("price").with_type("float").indexed(),
            VectorStoreField::data("flag").with_type("bool").indexed(),
            VectorStoreField::data("tags").with_type("list").indexed(),
            VectorStoreField::data("meta").with_type("dict"),
            VectorStoreField::data("note"),
            VectorStoreField::vector("embedding", 3).with_storage_name("vec"),
        ])
        .unwrap()
    }

    fn hash() -> RedisVectorCollection {
        store().collection("docs", definition()).unwrap()
    }

    fn json_coll() -> RedisVectorCollection {
        store()
            .collection_with_storage_type("docs", definition(), RedisStorageType::Json)
            .unwrap()
    }

    #[test]
    fn names_are_namespaced_and_hex_encoded() {
        let c = hash();
        assert_eq!(c.index_name(), "af:vector:64656661756c74:index:646f6373");
        assert_eq!(c.key_prefix(), "af:vector:64656661756c74:data:646f6373:");
        let names = NamespaceNames::new("default").unwrap();
        assert_eq!(
            names.try_parse_index_name(c.index_name()).as_deref(),
            Some("docs")
        );
        // Foreign, malformed and non-canonical (uppercase hex) names are ignored.
        assert_eq!(names.try_parse_index_name("idx:other"), None);
        assert_eq!(
            names.try_parse_index_name("af:vector:64656661756c74:index:zz"),
            None
        );
        assert_eq!(
            names.try_parse_index_name("af:vector:64656661756c74:index:646F6373"),
            None
        );
        let other = NamespaceNames::new("other").unwrap();
        assert_eq!(other.try_parse_index_name(c.index_name()), None);
    }

    #[test]
    fn namespace_and_collection_names_are_bounded() {
        assert!(store().with_namespace("").is_err());
        assert!(store().with_namespace("x".repeat(257)).is_err());
        assert!(store().collection("", definition()).is_err());
    }

    #[test]
    fn a_non_zero_database_is_refused() {
        let err = RedisVectorStore::new("redis://127.0.0.1:6379/2").unwrap_err();
        assert!(err.to_string().contains("database 0"), "{err}");
    }

    #[test]
    fn unsupported_definitions_are_refused() {
        let cases = vec![
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::data("t").full_text_indexed(),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::vector("v", 3).with_distance_function(DistanceFunction::new(
                    DistanceFunction::COSINE_SIMILARITY,
                )),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::vector("v", 3)
                    .with_index_kind(IndexKind::new(IndexKind::DISK_ANN)),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::vector("v", 3).with_type("bytes"),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::data("d").with_type("dict").indexed(),
            ],
            vec![VectorStoreField::key("id").with_type("int")],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::data("x").with_storage_name("_af_x"),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::data("x").with_storage_name("has-dash"),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::data("x").with_type("set"),
            ],
        ];
        for fields in cases {
            let def = VectorStoreCollectionDefinition::new(fields.clone()).unwrap();
            assert!(
                store().collection("c", def).is_err(),
                "should refuse {fields:?}"
            );
        }
    }

    #[test]
    fn hash_index_schema_matches_upstream() {
        let args = hash().create_index_args();
        let joined = args.join(" ");
        assert!(joined.starts_with(&format!(
            "{} ON HASH PREFIX 1 {} SCHEMA ",
            hash().index_name(),
            hash().key_prefix()
        )));
        assert!(joined
            .contains("id TAG SEPARATOR \u{1f} CASESENSITIVE INDEXEMPTY INDEXMISSING SORTABLE"));
        assert!(joined.contains("year NUMERIC INDEXMISSING SORTABLE"));
        // A TAG array is not sortable.
        assert!(
            joined.contains("tags TAG SEPARATOR \u{1f} CASESENSITIVE INDEXEMPTY INDEXMISSING vec")
        );
        assert!(joined.ends_with("vec VECTOR HNSW 6 TYPE FLOAT32 DIM 3 DISTANCE_METRIC COSINE"));
        // Unindexed fields are not in the schema.
        assert!(!joined.contains("meta") && !joined.contains("note"));
    }

    #[test]
    fn json_index_schema_uses_paths() {
        let joined = json_coll().create_index_args().join(" ");
        assert!(joined.contains(" ON JSON "));
        assert!(joined.contains("$.title AS title TAG"));
        assert!(joined.contains("$.vec AS vec VECTOR HNSW"));
    }

    #[test]
    fn vector_options_map_to_redis() {
        let def = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("a", 2)
                .with_type("float64")
                .with_index_kind(IndexKind::new(IndexKind::FLAT))
                .with_distance_function(DistanceFunction::new(
                    DistanceFunction::EUCLIDEAN_SQUARED_DISTANCE,
                )),
            VectorStoreField::vector("b", 2)
                .with_distance_function(DistanceFunction::new("redis.ip")),
        ])
        .unwrap();
        let joined = store()
            .collection("c", def)
            .unwrap()
            .create_index_args()
            .join(" ");
        assert!(joined.contains("a VECTOR FLAT 6 TYPE FLOAT64 DIM 2 DISTANCE_METRIC L2"));
        assert!(joined.contains("b VECTOR HNSW 6 TYPE FLOAT32 DIM 2 DISTANCE_METRIC IP"));
    }

    fn f(filter: Filter) -> Option<FilterExpression> {
        Some(filter.into())
    }

    #[test]
    fn filters_translate_like_upstream() {
        let c = hash();
        let t = |e: Option<FilterExpression>| c.prepare_filter(e.as_ref()).unwrap();
        assert_eq!(t(None), "*");
        assert_eq!(t(f(Filter::eq("title", "a b").unwrap())), "@title:{a\\ b}");
        assert_eq!(t(f(Filter::eq("title", "").unwrap())), "@title:{\"\"}");
        assert_eq!(
            t(f(Filter::eq("title", 3).unwrap())),
            "(ismissing(@title) -ismissing(@title))"
        );
        assert_eq!(
            t(f(Filter::eq("year", 2020).unwrap())),
            "@year:[2020.0 2020.0]"
        );
        assert_eq!(
            t(f(Filter::eq("year", true).unwrap())),
            "(ismissing(@year) -ismissing(@year))"
        );
        assert_eq!(t(f(Filter::eq("flag", true).unwrap())), "@flag:{true}");
        assert_eq!(
            t(f(Filter::ne("title", "x").unwrap())),
            "((-(ismissing(@title))) (-(@title:{x})))"
        );
        assert_eq!(
            t(f(Filter::gt("price", 1.5).unwrap())),
            "@price:[(1.5 +inf]"
        );
        assert_eq!(t(f(Filter::gte("price", 1).unwrap())), "@price:[1.0 +inf]");
        assert_eq!(t(f(Filter::lt("price", 2).unwrap())), "@price:[-inf (2.0]");
        assert_eq!(t(f(Filter::lte("price", 2).unwrap())), "@price:[-inf 2.0]");
        assert_eq!(
            t(f(Filter::between("year", 1, 2).unwrap())),
            "@year:[1.0 2.0]"
        );
        assert_eq!(
            t(f(Filter::exists("title").unwrap())),
            "(-(ismissing(@title)))"
        );
        assert_eq!(
            t(f(Filter::is_null("title").unwrap())),
            "((-(ismissing(@title))) (-((-(ismissing(@title))))))"
        );
        assert_eq!(
            t(f(Filter::any_of(
                "title",
                vec![json!("a"), json!(null), json!("b")]
            )
            .unwrap())),
            "(@title:{a} | @title:{b})"
        );
        assert_eq!(
            t(f(Filter::any_of("title", Vec::<Value>::new()).unwrap())),
            "(ismissing(@title) -ismissing(@title))"
        );
        assert_eq!(
            t(f(Filter::none_of("title", vec![json!("a")]).unwrap())),
            "((-(ismissing(@title))) (-((@title:{a}))))"
        );
        assert_eq!(t(f(Filter::contains("tags", "x").unwrap())), "(@tags:{x})");
        assert_eq!(
            t(f(Filter::contains_all(
                "tags",
                vec![json!("x"), json!("y")]
            )
            .unwrap())),
            "(@tags:{x} @tags:{y})"
        );
        assert_eq!(
            t(f(Filter::contains_any("tags", Vec::<Value>::new()).unwrap())),
            "(ismissing(@tags) -ismissing(@tags))"
        );
        let group = FilterGroup::and(vec![
            Filter::eq("title", "a").unwrap().into(),
            FilterGroup::not(Filter::eq("flag", false).unwrap().into()).unwrap(),
        ])
        .unwrap();
        assert_eq!(t(Some(group)), "(@title:{a} (-(@flag:{false})))");
        let or = FilterGroup::or(vec![
            Filter::eq("id", "k1").unwrap().into(),
            Filter::eq("id", "k-2").unwrap().into(),
        ])
        .unwrap();
        assert_eq!(t(Some(or)), "(@id:{k1} | @id:{k\\-2})");
    }

    #[test]
    fn json_null_semantics_differ_from_hash() {
        let c = json_coll();
        assert_eq!(
            c.prepare_filter(Some(&Filter::is_not_null("year").unwrap().into()))
                .unwrap(),
            "@year:[-inf +inf]"
        );
        assert_eq!(
            c.prepare_filter(Some(&Filter::is_not_null("flag").unwrap().into()))
                .unwrap(),
            "@flag:{true|false}"
        );
        assert!(c
            .prepare_filter(Some(&Filter::is_not_null("title").unwrap().into()))
            .is_err());
    }

    #[test]
    fn unsupported_filters_are_refused_not_dropped() {
        let c = hash();
        for expr in [
            Filter::eq("note", "x").unwrap(),           // not indexed
            Filter::eq("embedding", "x").unwrap(),      // vector
            Filter::starts_with("title", "x").unwrap(), // literal text
            Filter::gt("title", 1).unwrap(),            // ordered on a TAG
            Filter::contains("title", "x").unwrap(),    // membership on a scalar
            Filter::eq("tags", "x").unwrap(),           // whole-collection equality
            Filter::gt("year", 1u64 << 60).unwrap(),    // beyond 2^53
            Filter::contains("tags", 1).unwrap(),       // non-string TAG element
        ] {
            assert!(
                c.prepare_filter(Some(&expr.clone().into())).is_err(),
                "{expr:?}"
            );
        }
        assert!(c
            .prepare_filter(Some(
                &Filter::contains_all("tags", Vec::<Value>::new())
                    .unwrap()
                    .into()
            ))
            .is_err());
    }

    #[test]
    fn hash_records_encode_natively() {
        let c = hash();
        let (key, doc) = c
            .encode_record(&json!({
                "id": "k1", "title": "T", "year": 2020, "price": 1.5, "flag": true,
                "tags": ["a", "b"], "meta": {"x": [1, 2]}, "note": "n",
                "embedding": [1.0, 0.0, 0.5], "undeclared": 1
            }))
            .unwrap();
        assert_eq!(key, "k1");
        let Document::Hash(fields) = doc else {
            panic!("hash")
        };
        let get = |n: &str| fields.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
        assert_eq!(get("flag").unwrap(), b"true");
        assert_eq!(get("tags").unwrap(), "a\u{1f}b".as_bytes());
        assert_eq!(get("meta").unwrap(), br#"{"x":[1,2]}"#);
        assert_eq!(get("year").unwrap(), b"2020");
        assert_eq!(get("vec").unwrap().len(), 12);
        assert!(get("undeclared").is_none());
        // And decode back.
        let field = definition().try_get_field("tags").unwrap().clone();
        assert_eq!(
            c.decode_hash_value(&field, &get("tags").unwrap()).unwrap(),
            json!(["a", "b"])
        );
        let field = definition().try_get_field("embedding").unwrap().clone();
        assert_eq!(
            c.decode_hash_value(&field, &get("vec").unwrap()).unwrap(),
            json!([1.0, 0.0, 0.5])
        );
    }

    #[test]
    fn invalid_records_are_refused_before_io() {
        let c = hash();
        for record in [
            json!({"title": "no key"}),
            json!({"id": ""}),
            json!({"id": 5}),
            json!({"id": "k", "year": 1.5}),
            json!({"id": "k", "year": true}),
            json!({"id": "k", "flag": "yes"}),
            json!({"id": "k", "title": " padded "}),
            json!({"id": "k", "title": null}),
            json!({"id": "k", "tags": []}),
            json!({"id": "k", "tags": ["a", ""]}),
            json!({"id": "k", "tags": ["a", 1]}),
            json!({"id": "k", "embedding": [1, 2]}),
            json!({"id": "k", "embedding": [0, 0, 0]}),
            json!({"id": "k", "embedding": [1e39, 0, 0]}),
            json!({"id": "k", "embedding": [true, 0, 1]}),
            json!({"id": "k", "year": 9007199254740993u64}),
        ] {
            assert!(c.encode_record(&record).is_err(), "{record}");
        }
        // JSON accepts null and empty arrays.
        let j = json_coll();
        assert!(j
            .encode_record(&json!({"id": "k", "title": null, "tags": []}))
            .is_ok());
    }

    #[test]
    fn search_builds_a_knn_query() {
        let c = hash();
        let options = VectorSearchOptions::new(2)
            .with_skip(1)
            .with_filter(Filter::eq("title", "a").unwrap());
        let (args, vector) = c.search_args(&[1.0, 0.0, 0.0], &options, None).unwrap();
        assert_eq!(
            args[1],
            "(@title:{a})=>[KNN 3 @vec $vector AS _af_distance]"
        );
        assert_eq!(
            args[2..].join(" "),
            "RETURN 1 _af_distance SORTBY _af_distance ASC LIMIT 1 2 PARAMS 2 vector"
        );
        assert_eq!(vector.len(), 12);

        let range = c
            .search_args(&[1.0, 0.0, 0.0], &VectorSearchOptions::new(2), Some(0.5))
            .unwrap()
            .0;
        assert_eq!(
            range[1],
            "@vec:[VECTOR_RANGE $radius $vector]=>{$yield_distance_as: _af_distance}"
        );
        assert!(range.join(" ").contains("PARAMS 4 radius 0.5 vector"));
        assert!(c
            .search_args(&[1.0, 0.0, 0.0], &VectorSearchOptions::new(2), Some(-1.0))
            .is_err());
        // Zero cosine query and wrong width are refused.
        assert!(c
            .search_args(&[0.0, 0.0, 0.0], &VectorSearchOptions::new(2), None)
            .is_err());
        assert!(c
            .search_args(&[1.0], &VectorSearchOptions::new(2), None)
            .is_err());
    }

    #[test]
    fn provider_filters_are_conjoined() {
        let c = hash();
        let options = VectorSearchOptions::new(1)
            .with_filter(Filter::eq("title", "a").unwrap())
            .with_provider_filter("@year:[1 2]");
        assert_eq!(
            c.query_clause(&options).unwrap(),
            "(@title:{a} (@year:[1 2]))"
        );
        let raw_only = VectorSearchOptions::new(1).with_provider_filter("@year:[1 2]");
        assert_eq!(c.query_clause(&raw_only).unwrap(), "(@year:[1 2])");
    }

    #[test]
    fn ft_info_signature_parsing() {
        let bulk = |s: &str| RValue::BulkString(s.as_bytes().to_vec());
        let reply = RValue::Array(vec![
            bulk("index_definition"),
            RValue::Array(vec![
                bulk("key_type"),
                bulk("HASH"),
                bulk("prefixes"),
                RValue::Array(vec![bulk("p:")]),
            ]),
            bulk("attributes"),
            RValue::Array(vec![RValue::Array(vec![
                bulk("identifier"),
                bulk("vec"),
                bulk("attribute"),
                bulk("vec"),
                bulk("type"),
                bulk("VECTOR"),
                bulk("algorithm"),
                bulk("HNSW"),
                bulk("dim"),
                RValue::Int(3),
            ])]),
        ]);
        let sig = parse_ft_info(&reply).unwrap();
        assert_eq!(sig.key_type, "HASH");
        assert_eq!(sig.prefixes, vec!["p:".to_string()]);
        assert_eq!(sig.attributes[0].details["dim"], "3");
    }

    #[test]
    fn the_store_is_object_safe() {
        let s: Box<dyn VectorStore> = Box::new(store());
        let c = s.get_collection("docs", definition()).unwrap();
        assert_eq!(c.name(), "docs");
    }
}
