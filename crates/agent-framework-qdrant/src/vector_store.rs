//! [`QdrantStore`] / [`QdrantCollection`]: dense-vector collections on a
//! Qdrant server, over its REST API.
//!
//! Ports `agent_framework_qdrant._vector_store`. Every vector field is a
//! *named* dense vector; data fields are the point payload; the key is the
//! point id.
//!
//! # Settings
//!
//! [`QdrantSettings::load`] mirrors upstream's `QdrantSettings`: an explicit
//! value wins, then the `QDRANT_URL` / `QDRANT_API_KEY` environment
//! variables, then the default URL [`DEFAULT_QDRANT_URL`]. The API key is
//! sent as Qdrant's `api-key` header and never appears in `Debug` output.
//!
//! # Keys
//!
//! A point id is an unsigned 64-bit integer or a UUID. String keys must
//! parse as a UUID and are canonicalized (lowercase, hyphenated); arbitrary
//! strings and booleans are refused, never hashed. The key field's `type_`
//! may be unset, `"int"`, `"str"` or `"UUID"`, and a record's key must match
//! a declared type.
//!
//! # Schema
//!
//! - Vector `type_`: unset, `float`, `float16`, `float32`, `float64` (sent
//!   as float32). Distance: unset or `cosine_similarity` (`Cosine`),
//!   `dot_prod` (`Dot`), `euclidean_distance` (`Euclid`), `manhattan`
//!   (`Manhattan`). Index kind: unset/`default`/`hnsw`, or `flat` (HNSW
//!   with `m = 0`).
//! - Data `type_`: unset (any JSON), `str`, `int`, `float`, `bool`, `list`,
//!   `tuple`, `set`, `Sequence`, `dict`. Tuples, sets and sequences are
//!   JSON arrays on the wire — upstream's "tuple payloads round-trip as
//!   lists" rule is simply what a `serde_json::Value` already is. Integers
//!   anywhere in a payload must fit a signed 64-bit integer.
//! - An indexed data field gets a payload index (`keyword`, `integer`,
//!   `float`, `bool` for `str`/`int`/`float`/`bool`); other indexed types are
//!   refused. Payload storage names cannot contain JSON-path punctuation
//!   (`.`, `[`, `]`, `"`, `\`). Full-text indexing is refused.
//!
//! [`VectorCollection::ensure_collection_exists`] creates the collection and
//! its payload indexes, or validates an existing one: named vectors only,
//! matching size, distance, float32 datatype, no multivector, and the index
//! kind (HNSW `m`); an existing payload index of another type is an error.
//! A creation conflict (another writer won the race) is tolerated, and the
//! winner's schema is read back — retrying, with bounded backoff, only the
//! specific "0 of 0 read operations failed" readiness error.
//!
//! # Scores
//!
//! Scores are Qdrant's native values: similarity (higher is closer) for
//! `Cosine` / `Dot`, distance (lower is closer) for `Euclid` /
//! `Manhattan`. When the field declares a distance function, that function
//! describes the score and `score_kind` is `None`; when it declares none
//! (the cosine default), `score_kind` is
//! [`VectorSearchResult::SCORE_KIND_RELEVANCE`].
//!
//! # Batching
//!
//! Upsert, get and delete run in batches of 256 points, each with
//! `wait=true`. Every record is validated before the first request; a batch
//! that fails part-way reports how many records were already written.
//!
//! # Divergences
//!
//! - Records are `serde_json::Value` objects keyed by logical name (see the
//!   core module); typed models and custom codecs are the caller's serde.
//! - `provider_annotations` (`qdrant.payload_index`) and `operation_options`
//!   (consistency, shard keys, timeouts, creation options) have no carrier
//!   on the core types and are not ported; a payload index comes from
//!   `is_indexed` plus a scalar `type_`.
//! - Local (in-process) Qdrant mode does not exist outside the Python SDK.
//! - Filtered retrieval without keys is the inherent
//!   [`QdrantCollection::get_filtered`]; a score threshold is
//!   [`QdrantCollection::search_with_score_threshold`].
//! - [`VectorSearchOptions::provider_filter`] is a Qdrant filter object as
//!   JSON text, conjoined (`must`) with the portable filter.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{json, Map, Value};

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    DistanceFunction, FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator,
    IndexKind, VectorCollection, VectorSearchOptions, VectorSearchResult, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};

/// Default server URL when neither an explicit URL nor `QDRANT_URL` is set
/// (the Python SDK's localhost default).
pub const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";
/// Environment variable for the server URL.
pub const QDRANT_URL_ENV: &str = "QDRANT_URL";
/// Environment variable for the API key.
pub const QDRANT_API_KEY_ENV: &str = "QDRANT_API_KEY";

/// Upstream's `_BATCH_SIZE`.
const BATCH_SIZE: usize = 256;
/// Upstream's `_MAX_EXACT_INTEGER` (2^53 − 1).
const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_991.0;
/// Readiness retries after a creation conflict (upstream: 5).
const MAX_READINESS_RETRIES: u32 = 5;

// region: settings

/// Connection settings. Mirrors upstream's `QdrantSettings`.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct QdrantSettings {
    /// Server URL.
    pub url: Option<String>,
    /// API key, sent as the `api-key` header.
    pub api_key: Option<String>,
}

impl std::fmt::Debug for QdrantSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantSettings")
            .field("url", &self.url)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .finish()
    }
}

impl QdrantSettings {
    /// Resolve settings: each explicit value wins over its environment
    /// variable (`QDRANT_URL`, `QDRANT_API_KEY`). An empty variable counts as
    /// unset.
    pub fn load(url: Option<String>, api_key: Option<String>) -> Self {
        let env = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        Self {
            url: url.or_else(|| env(QDRANT_URL_ENV)),
            api_key: api_key.or_else(|| env(QDRANT_API_KEY_ENV)),
        }
    }
}

// endregion

// region: HTTP

#[derive(Clone)]
struct Http {
    client: reqwest::Client,
    base: String,
    api_key: Option<String>,
}

/// One non-2xx reply.
struct HttpFailure {
    status: u16,
    body: String,
}

impl Http {
    fn new(url: &str, api_key: Option<String>, client: reqwest::Client) -> Result<Self> {
        let base = url.trim_end_matches('/').to_string();
        if !(base.starts_with("http://") || base.starts_with("https://")) {
            return Err(Error::Configuration(format!(
                "Qdrant URL must be http(s)://…; got '{base}'"
            )));
        }
        Ok(Self {
            client,
            base,
            api_key,
        })
    }

    /// Send a request; `Ok(Err(..))` is a non-2xx reply, so callers can
    /// tolerate specific statuses.
    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<std::result::Result<Value, HttpFailure>> {
        let mut request = self.client.request(method, format!("{}{path}", self.base));
        if let Some(key) = &self.api_key {
            request = request.header("api-key", key);
        }
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request
            .send()
            .await
            .map_err(|e| Error::service(format!("Qdrant request failed: {e}")))?;
        let status = response.status().as_u16();
        let text = response
            .text()
            .await
            .map_err(|e| Error::service(format!("Qdrant response could not be read: {e}")))?;
        if !(200..300).contains(&status) {
            return Ok(Err(HttpFailure { status, body: text }));
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| Error::service(format!("Qdrant returned a non-JSON response: {e}")))?;
        Ok(Ok(value.get("result").cloned().unwrap_or(Value::Null)))
    }

    async fn call(&self, method: Method, path: &str, body: Option<&Value>) -> Result<Value> {
        self.send(method, path, body).await?.map_err(failure_error)
    }
}

fn failure_error(failure: HttpFailure) -> Error {
    let detail = serde_json::from_str::<Value>(&failure.body)
        .ok()
        .and_then(|v| {
            v.pointer("/status/error")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or(failure.body);
    Error::service_status(
        failure.status,
        format!("Qdrant returned HTTP {}: {detail}", failure.status),
        None,
    )
}

/// Percent-encode one path segment.
fn segment(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn require_completed(result: &Value) -> Result<()> {
    match result.get("status").and_then(Value::as_str) {
        Some("completed") => Ok(()),
        other => Err(Error::service(format!(
            "Qdrant write did not complete: {}",
            other.unwrap_or("missing status")
        ))),
    }
}

// endregion

// region: value preparation

/// Validate and canonicalize a point id: a `u64` or a UUID string.
fn prepare_point_id(value: &Value) -> Result<Value> {
    match value {
        Value::Number(n) => n.as_u64().map(Value::from).ok_or_else(|| {
            Error::Configuration("Qdrant integer keys must be unsigned 64-bit integers".into())
        }),
        Value::String(s) => uuid::Uuid::parse_str(s)
            .map(|u| Value::String(u.hyphenated().to_string()))
            .map_err(|_| {
                Error::Configuration(
                    "Qdrant string keys must be UUIDs; arbitrary strings are not supported".into(),
                )
            }),
        _ => Err(Error::Configuration(
            "Qdrant keys must be unsigned 64-bit integers or UUIDs, not booleans".into(),
        )),
    }
}

fn id_token(id: &Value) -> String {
    match id {
        Value::String(s) => uuid::Uuid::parse_str(s)
            .map(|u| format!("s:{}", u.hyphenated()))
            .unwrap_or_else(|_| format!("s:{s}")),
        other => format!("n:{other}"),
    }
}

fn prepare_dense_vector(value: &Value, name: &str) -> Result<Vec<f64>> {
    let items = value.as_array().ok_or_else(|| {
        Error::Configuration(format!(
            "Qdrant vector '{name}' must be a dense numeric sequence; sparse and binary vectors \
             are unsupported"
        ))
    })?;
    items
        .iter()
        .map(|item| {
            let number = item.as_f64().filter(|_| item.is_number()).ok_or_else(|| {
                Error::Configuration(format!(
                    "Qdrant vector '{name}' must contain numbers, not booleans"
                ))
            })?;
            if !number.is_finite() || number.abs() > f64::from(f32::MAX) {
                return Err(Error::Configuration(format!(
                    "Qdrant vector '{name}' must contain finite float32 values"
                )));
            }
            Ok(number)
        })
        .collect()
}

/// Upstream's `_prepare_payload`: every integer must be a signed 64-bit
/// integer. (`serde_json` cannot hold a non-finite float.)
fn validate_payload(value: &Value) -> Result<()> {
    match value {
        Value::Number(n) if n.is_u64() && n.as_i64().is_none() => Err(Error::Configuration(
            "Qdrant integer payloads must be signed 64-bit integers".into(),
        )),
        Value::Array(items) => items.iter().try_for_each(validate_payload),
        Value::Object(map) => map.values().try_for_each(validate_payload),
        _ => Ok(()),
    }
}

fn type_matches(type_: &str, value: &Value) -> Option<bool> {
    Some(match type_ {
        "str" => value.is_string(),
        "int" => value.is_i64(),
        "float" => value.is_number(),
        "bool" => value.is_boolean(),
        "list" | "tuple" | "set" | "Sequence" => value.is_array(),
        "dict" => value.is_object(),
        _ => return None,
    })
}

fn prepare_field_payload(field: &VectorStoreField, value: &Value) -> Result<Value> {
    validate_payload(value)?;
    if value.is_null() {
        return Ok(Value::Null);
    }
    if let Some(type_) = field.type_.as_deref() {
        match type_matches(type_, value) {
            None => {
                return Err(Error::Configuration(format!(
                    "Qdrant payload field type '{type_}' is not supported"
                )))
            }
            Some(false) => {
                return Err(Error::Configuration(format!(
                    "Qdrant payload field '{}' must have declared type '{type_}'",
                    field.name
                )))
            }
            Some(true) => {}
        }
    }
    Ok(value.clone())
}

fn payload_index(field: &VectorStoreField) -> Result<Option<&'static str>> {
    if field.is_indexed != Some(true) {
        return Ok(None);
    }
    match field.type_.as_deref() {
        Some("str") => Ok(Some("keyword")),
        Some("int") => Ok(Some("integer")),
        Some("float") => Ok(Some("float")),
        Some("bool") => Ok(Some("bool")),
        _ => Err(Error::Configuration(format!(
            "indexed field '{}' requires a scalar type (str, int, float or bool)",
            field.name
        ))),
    }
}

fn distance_of(field: &VectorStoreField) -> Result<&'static str> {
    Ok(
        match field
            .distance_function
            .as_ref()
            .map(DistanceFunction::as_str)
        {
            None | Some(DistanceFunction::COSINE_SIMILARITY) => "Cosine",
            Some(DistanceFunction::DOT_PROD) => "Dot",
            Some(DistanceFunction::EUCLIDEAN_DISTANCE) => "Euclid",
            Some(DistanceFunction::MANHATTAN) => "Manhattan",
            Some(other) => {
                return Err(Error::Configuration(format!(
                    "Qdrant distance function '{other}' is not supported"
                )))
            }
        },
    )
}

fn is_flat(field: &VectorStoreField) -> Result<bool> {
    match field.index_kind.as_ref().map(IndexKind::as_str) {
        None | Some(IndexKind::DEFAULT) | Some(IndexKind::HNSW) => Ok(false),
        Some(IndexKind::FLAT) => Ok(true),
        Some(other) => Err(Error::Configuration(format!(
            "Qdrant index kind '{other}' is not supported"
        ))),
    }
}

// endregion

// region: filters

fn false_condition() -> Value {
    json!({"has_id": []})
}

fn true_condition() -> Value {
    json!({"must_not": [false_condition()]})
}

fn presence_condition(name: &str) -> Value {
    // Unlike is_empty, values_count distinguishes missing from null and [].
    json!({"key": name, "values_count": {"gte": 0}})
}

fn null_condition(name: &str) -> Value {
    json!({"is_null": {"key": name}})
}

fn non_null_condition(name: &str) -> Value {
    json!({"must": [presence_condition(name)], "must_not": [null_condition(name)]})
}

fn range_operand(value: &Value) -> Result<f64> {
    let number = value
        .as_f64()
        .filter(|_| value.is_number())
        .ok_or_else(|| {
            Error::Configuration(
                "Qdrant ordered filters require numeric operands, not booleans".into(),
            )
        })?;
    let exact = value
        .as_i64()
        .map(|i| i.unsigned_abs() <= 9_007_199_254_740_991)
        .or_else(|| value.as_u64().map(|u| u <= 9_007_199_254_740_991))
        .unwrap_or(number.abs() <= MAX_EXACT_INTEGER);
    if !number.is_finite() || !exact {
        return Err(Error::Configuration(
            "Qdrant numeric range operands must be within +/- (2**53-1) to compare integers \
             exactly"
                .into(),
        ));
    }
    Ok(number)
}

fn equality_condition(
    name: &str,
    value: &Value,
    field: &VectorStoreField,
    element: bool,
) -> Result<Value> {
    if value.is_null() {
        if element {
            return Err(Error::Configuration(
                "Qdrant collection membership with a null operand is not supported".into(),
            ));
        }
        return Ok(null_condition(name));
    }
    if value.is_array() || value.is_object() {
        return Err(Error::Configuration(
            "Qdrant equality and membership operands must be JSON scalars".into(),
        ));
    }
    let kind = field.type_.as_deref();
    if !element {
        if !matches!(kind, Some("str" | "bool" | "int" | "float")) {
            return Err(Error::Configuration(format!(
                "Qdrant scalar equality requires a str, bool, int or float field type; '{}' has \
                 {kind:?}",
                field.name
            )));
        }
        let compatible = match value {
            Value::Bool(_) => kind == Some("bool"),
            Value::String(_) => kind == Some("str"),
            _ => matches!(kind, Some("int" | "float")),
        };
        if !compatible {
            return Ok(false_condition());
        }
    }
    if value.is_string() || value.is_boolean() {
        return Ok(json!({"key": name, "match": {"value": value}}));
    }
    if kind == Some("int") && !element {
        let integer = if let Some(i) = value.as_i64() {
            i
        } else if value.is_u64() {
            return Ok(false_condition()); // beyond i64
        } else {
            let f = value.as_f64().unwrap_or(f64::NAN);
            // [-2^63, 2^63): the floats that convert to an i64 exactly.
            let i64_range = -9.223_372_036_854_776e18..9.223_372_036_854_776e18;
            if !f.is_finite() || f.fract() != 0.0 || !i64_range.contains(&f) {
                return Ok(false_condition());
            }
            f as i64
        };
        return Ok(json!({"key": name, "match": {"value": integer}}));
    }
    let number = range_operand(value)?;
    // Range also matches equal integer/float values, but never booleans.
    Ok(json!({"key": name, "range": {"gte": number, "lte": number}}))
}

fn key_filter_id(value: &Value, kind: Option<&str>) -> Option<Value> {
    match value {
        Value::Bool(_) => None,
        Value::Number(n) if matches!(kind, None | Some("int")) => {
            if let Some(u) = n.as_u64() {
                Some(Value::from(u))
            } else {
                let f = n.as_f64()?;
                (f.is_finite() && f.fract() == 0.0 && f >= 0.0 && f <= u64::MAX as f64)
                    .then(|| Value::from(f as u64))
            }
        }
        Value::String(_) if matches!(kind, None | Some("str" | "UUID")) => {
            prepare_point_id(value).ok()
        }
        _ => None,
    }
}

fn key_filter_condition(filter: &Filter, field: &VectorStoreField) -> Result<Value> {
    let null = Value::Null;
    let value = filter.value.as_ref().unwrap_or(&null);
    let kind = field.type_.as_deref();
    let condition = match &filter.operator {
        FilterOperator::Exists | FilterOperator::IsNotNull => return Ok(true_condition()),
        FilterOperator::IsNull => return Ok(false_condition()),
        FilterOperator::Eq if value.is_null() => return Ok(false_condition()),
        FilterOperator::Ne if value.is_null() => return Ok(true_condition()),
        FilterOperator::Eq | FilterOperator::Ne => {
            json!({"has_id": key_filter_id(value, kind).into_iter().collect::<Vec<_>>()})
        }
        FilterOperator::In | FilterOperator::NotIn => {
            let ids: Vec<Value> = value
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|i| key_filter_id(i, kind))
                        .collect()
                })
                .unwrap_or_default();
            json!({"has_id": ids})
        }
        other => {
            return Err(Error::Configuration(format!(
                "Qdrant key filters do not support '{other}'"
            )))
        }
    };
    Ok(
        if matches!(filter.operator, FilterOperator::Ne | FilterOperator::NotIn) {
            json!({"must_not": [condition]})
        } else {
            condition
        },
    )
}

fn operands(filter: &Filter) -> Result<&[Value]> {
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

fn data_filter_condition(filter: &Filter, field: &VectorStoreField) -> Result<Value> {
    let name = field.effective_storage_name();
    let null = Value::Null;
    let value = filter.value.as_ref().unwrap_or(&null);
    match &filter.operator {
        FilterOperator::Exists => Ok(presence_condition(name)),
        FilterOperator::IsNull => Ok(null_condition(name)),
        FilterOperator::IsNotNull => Ok(non_null_condition(name)),
        FilterOperator::Eq => equality_condition(name, value, field, false),
        FilterOperator::Ne => Ok(json!({
            "must": [presence_condition(name)],
            "must_not": [equality_condition(name, value, field, false)?],
        })),
        FilterOperator::Gt
        | FilterOperator::Gte
        | FilterOperator::Lt
        | FilterOperator::Lte
        | FilterOperator::Between => {
            if !matches!(field.type_.as_deref(), Some("int" | "float")) {
                return Err(Error::Configuration(
                    "Qdrant ordered comparisons require a declared int or float field".into(),
                ));
            }
            let range = if filter.operator == FilterOperator::Between {
                let [lower, upper] = operands(filter)? else {
                    return Err(Error::Configuration(
                        "between requires exactly [lower, upper]".into(),
                    ));
                };
                json!({"gte": range_operand(lower)?, "lte": range_operand(upper)?})
            } else {
                let mut range = Map::new();
                range.insert(
                    filter.operator.as_str().into(),
                    json!(range_operand(value)?),
                );
                Value::Object(range)
            };
            Ok(json!({"key": name, "range": range}))
        }
        FilterOperator::In | FilterOperator::NotIn => {
            let items = operands(filter)?;
            let condition = if items.is_empty() {
                false_condition()
            } else {
                json!({"should": items
                    .iter()
                    .map(|item| equality_condition(name, item, field, false))
                    .collect::<Result<Vec<_>>>()?})
            };
            Ok(if filter.operator == FilterOperator::NotIn {
                json!({"must": [non_null_condition(name)], "must_not": [condition]})
            } else {
                json!({"must": [non_null_condition(name), condition]})
            })
        }
        FilterOperator::Contains | FilterOperator::ContainsAny | FilterOperator::ContainsAll => {
            if !matches!(
                field.type_.as_deref(),
                Some("list" | "tuple" | "set" | "Sequence")
            ) {
                return Err(Error::Configuration(
                    "Qdrant collection membership requires a declared collection field type".into(),
                ));
            }
            let items: Vec<&Value> = if filter.operator == FilterOperator::Contains {
                vec![value]
            } else {
                operands(filter)?.iter().collect()
            };
            let conditions = items
                .into_iter()
                .map(|item| equality_condition(name, item, field, true))
                .collect::<Result<Vec<_>>>()?;
            if filter.operator == FilterOperator::ContainsAll {
                let mut must = vec![non_null_condition(name)];
                must.extend(conditions);
                return Ok(json!({ "must": must }));
            }
            Ok(if conditions.is_empty() {
                false_condition()
            } else {
                json!({ "should": conditions })
            })
        }
        other => Err(Error::Configuration(format!(
            "Qdrant does not support portable filter operator '{other}'; literal text operations \
             are not equivalent to tokenized full-text search"
        ))),
    }
}

// endregion

/// A Qdrant server connection that hands out collections. Mirrors upstream's
/// `QdrantStore`; collections share its HTTP client.
#[derive(Clone)]
pub struct QdrantStore {
    http: Http,
}

impl std::fmt::Debug for QdrantStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantStore")
            .field("url", &self.http.base)
            .field("api_key", &self.http.api_key.as_ref().map(|_| "***"))
            .finish()
    }
}

impl QdrantStore {
    /// Connect to `url` without an API key (no I/O).
    pub fn new(url: impl AsRef<str>) -> Result<Self> {
        Self::from_settings(QdrantSettings {
            url: Some(url.as_ref().to_string()),
            api_key: None,
        })
    }

    /// Connect using `QDRANT_URL` / `QDRANT_API_KEY`, defaulting the URL to
    /// [`DEFAULT_QDRANT_URL`].
    pub fn from_env() -> Result<Self> {
        Self::from_settings(QdrantSettings::load(None, None))
    }

    /// Connect with resolved settings (see [`QdrantSettings::load`]).
    pub fn from_settings(settings: QdrantSettings) -> Result<Self> {
        let url = settings
            .url
            .unwrap_or_else(|| DEFAULT_QDRANT_URL.to_string());
        Ok(Self {
            http: Http::new(&url, settings.api_key, reqwest::Client::new())?,
        })
    }

    /// Send the `api-key` header.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.http.api_key = Some(api_key.into());
        self
    }

    /// Use a caller-configured HTTP client (proxies, timeouts, TLS roots) —
    /// the counterpart of upstream's borrowed `async_client`.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http.client = client;
        self
    }

    /// The server URL.
    pub fn url(&self) -> &str {
        &self.http.base
    }

    /// Open a collection (no I/O).
    pub fn collection(
        &self,
        name: impl Into<String>,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<QdrantCollection> {
        QdrantCollection::new(self.http.clone(), name.into(), definition)
    }

    /// Delete a collection by name if it exists. Upstream's store-level
    /// `ensure_collection_deleted`.
    pub async fn ensure_collection_deleted(&self, name: &str) -> Result<()> {
        if collection_exists(&self.http, name).await? {
            delete_collection(&self.http, name).await?;
        }
        Ok(())
    }
}

async fn collection_exists(http: &Http, name: &str) -> Result<bool> {
    let result = http
        .call(
            Method::GET,
            &format!("/collections/{}/exists", segment(name)),
            None,
        )
        .await?;
    result
        .get("exists")
        .and_then(Value::as_bool)
        .ok_or_else(|| Error::service("Qdrant returned a malformed exists response"))
}

async fn delete_collection(http: &Http, name: &str) -> Result<()> {
    let result = http
        .call(
            Method::DELETE,
            &format!("/collections/{}", segment(name)),
            None,
        )
        .await?;
    if result != Value::Bool(true) {
        return Err(Error::service(format!(
            "Qdrant did not delete collection '{name}'"
        )));
    }
    Ok(())
}

#[async_trait]
impl VectorStore for QdrantStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let result = self.http.call(Method::GET, "/collections", None).await?;
        Ok(result
            .get("collections")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|c| c.get("name").and_then(Value::as_str).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default())
    }

    /// A direct existence check, without listing every collection.
    async fn collection_exists(&self, name: &str) -> Result<bool> {
        collection_exists(&self.http, name).await
    }
}

/// One Qdrant collection. Mirrors upstream's `QdrantCollection`.
pub struct QdrantCollection {
    http: Http,
    name: String,
    definition: VectorStoreCollectionDefinition,
}

impl std::fmt::Debug for QdrantCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantCollection")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl QdrantCollection {
    fn new(http: Http, name: String, definition: VectorStoreCollectionDefinition) -> Result<Self> {
        if name.is_empty() {
            return Err(Error::Configuration(
                "Qdrant collection names must be non-empty".into(),
            ));
        }
        for field in definition.fields() {
            if field.is_full_text_indexed == Some(true) {
                return Err(Error::Configuration(
                    "Qdrant full-text indexes are not part of this dense-vector connector".into(),
                ));
            }
            match field.field_type {
                FieldType::Key => {
                    if !matches!(field.type_.as_deref(), None | Some("int" | "str" | "UUID")) {
                        return Err(Error::Configuration(format!(
                            "Qdrant keys must be int, str or UUID; got '{}'",
                            field.type_.as_deref().unwrap_or_default()
                        )));
                    }
                }
                FieldType::Vector => {
                    if !matches!(
                        field.type_.as_deref(),
                        None | Some("float" | "float16" | "float32" | "float64")
                    ) {
                        return Err(Error::Configuration(format!(
                            "Qdrant vector '{}' must be a float type; got '{}'",
                            field.name,
                            field.type_.as_deref().unwrap_or_default()
                        )));
                    }
                    distance_of(field)?;
                    is_flat(field)?;
                }
                FieldType::Data => {
                    let name = field.effective_storage_name();
                    if name.is_empty() || name.contains(['.', '[', ']', '"', '\\']) {
                        return Err(Error::Configuration(
                            "Qdrant payload storage names cannot contain JSON-path punctuation"
                                .into(),
                        ));
                    }
                    if let Some(type_) = field.type_.as_deref() {
                        if type_matches(type_, &Value::Null).is_none() {
                            return Err(Error::Configuration(format!(
                                "Qdrant payload field type '{type_}' is not supported"
                            )));
                        }
                    }
                    payload_index(field)?;
                }
            }
        }
        Ok(Self {
            http,
            name,
            definition,
        })
    }

    fn path(&self, suffix: &str) -> String {
        format!("/collections/{}{suffix}", segment(&self.name))
    }

    /// The body `PUT /collections/{name}` is sent when creating.
    pub fn create_collection_body(&self) -> Result<Value> {
        let mut vectors = Map::new();
        for field in self.definition.vector_fields() {
            let size = field.dimensions.ok_or_else(|| {
                Error::Configuration(format!("vector '{}' requires dimensions", field.name))
            })?;
            let mut params = json!({"size": size, "distance": distance_of(field)?});
            if is_flat(field)? {
                params["hnsw_config"] = json!({"m": 0});
            }
            vectors.insert(field.effective_storage_name().to_string(), params);
        }
        Ok(json!({ "vectors": vectors }))
    }

    /// Translate a portable filter into a Qdrant filter object. Public so a
    /// caller can see exactly what will be sent.
    pub fn prepare_filter(&self, filter: Option<&FilterExpression>) -> Result<Option<Value>> {
        let Some(filter) = filter else {
            return Ok(None);
        };
        filter.validate()?;
        Ok(Some(json!({"must": [self.translate(filter)?]})))
    }

    fn translate(&self, expression: &FilterExpression) -> Result<Value> {
        match expression {
            FilterExpression::Group(group) => {
                let children = group
                    .filters
                    .iter()
                    .map(|c| self.translate(c))
                    .collect::<Result<Vec<_>>>()?;
                Ok(match group.operator {
                    FilterGroupOperator::And => json!({ "must": children }),
                    FilterGroupOperator::Or => json!({ "should": children }),
                    // NOT applies to the whole child, not to its leaves.
                    FilterGroupOperator::Not => json!({ "must_not": children }),
                })
            }
            FilterExpression::Condition(filter) => {
                let field = (!filter.field_name.contains('.'))
                    .then(|| self.definition.try_get_field(&filter.field_name))
                    .flatten()
                    .ok_or_else(|| {
                        Error::Configuration(format!(
                            "Qdrant filters support declared top-level logical fields only; got \
                             '{}'",
                            filter.field_name
                        ))
                    })?;
                match field.field_type {
                    FieldType::Key => key_filter_condition(filter, field),
                    FieldType::Vector => Err(Error::Configuration(
                        "portable Qdrant filters do not operate on vector fields".into(),
                    )),
                    FieldType::Data => data_filter_condition(filter, field),
                }
            }
        }
    }

    fn search_filter(&self, options: &VectorSearchOptions) -> Result<Option<Value>> {
        let portable = self.prepare_filter(options.filter.as_ref())?;
        let provider = match options.provider_filter.as_deref() {
            Some(raw) if !raw.trim().is_empty() => {
                let parsed: Value = serde_json::from_str(raw).map_err(|e| {
                    Error::Configuration(format!(
                        "Qdrant provider_filter must be a JSON filter object: {e}"
                    ))
                })?;
                if !parsed.is_object() {
                    return Err(Error::Configuration(
                        "Qdrant provider_filter must be a JSON filter object".into(),
                    ));
                }
                Some(parsed)
            }
            _ => None,
        };
        Ok(match (portable, provider) {
            (None, None) => None,
            (Some(f), None) | (None, Some(f)) => Some(f),
            (Some(a), Some(b)) => Some(json!({"must": [a, b]})),
        })
    }

    /// Validate and encode one record into a Qdrant point.
    fn encode_point(&self, record: &Value) -> Result<(Value, Value)> {
        let stored = self.definition.to_storage(record)?;
        let object = stored.as_object().ok_or_else(|| {
            Error::Configuration("a vector store record must be a JSON object".into())
        })?;
        let key_field = self.definition.key_field();
        let raw_key = object
            .get(key_field.effective_storage_name())
            .ok_or_else(|| {
                Error::Configuration(format!("record is missing its key '{}'", key_field.name))
            })?;
        let key = prepare_point_id(raw_key)?;
        match key_field.type_.as_deref() {
            Some("int") if !key.is_u64() => {
                return Err(Error::Configuration(
                    "Qdrant key must match declared type 'int'".into(),
                ))
            }
            Some("str" | "UUID") if !key.is_string() => {
                return Err(Error::Configuration(format!(
                    "Qdrant key must match declared type '{}'",
                    key_field.type_.as_deref().unwrap_or_default()
                )))
            }
            _ => {}
        }
        let mut payload = Map::new();
        for field in self.definition.data_fields() {
            let name = field.effective_storage_name();
            if let Some(value) = object.get(name) {
                payload.insert(name.to_string(), prepare_field_payload(field, value)?);
            }
        }
        let mut vectors = Map::new();
        for field in self.definition.vector_fields() {
            let name = field.effective_storage_name();
            match object.get(name) {
                None | Some(Value::Null) => {}
                Some(value) => {
                    vectors.insert(
                        name.to_string(),
                        json!(prepare_dense_vector(value, &field.name)?),
                    );
                }
            }
        }
        Ok((
            key.clone(),
            json!({"id": key, "vector": vectors, "payload": payload}),
        ))
    }

    /// Turn a Qdrant `Record` / `ScoredPoint` into a logical record.
    fn decode_point(&self, point: &Value, include_vectors: bool) -> Result<Value> {
        let mut record = match point.get("payload") {
            Some(Value::Object(map)) => map.clone(),
            _ => Map::new(),
        };
        let id = point
            .get("id")
            .cloned()
            .ok_or_else(|| Error::service("Qdrant returned a point without an id"))?;
        record.insert(self.definition.key_field_storage_name().to_string(), id);
        match point.get("vector") {
            None | Some(Value::Null) => {}
            Some(Value::Object(vectors)) => {
                for field in self.definition.vector_fields() {
                    let name = field.effective_storage_name();
                    record.insert(
                        name.to_string(),
                        vectors.get(name).cloned().unwrap_or(Value::Null),
                    );
                }
            }
            Some(_) => {
                return Err(Error::service(
                    "expected named vectors in the Qdrant response",
                ))
            }
        }
        self.definition
            .from_storage(&Value::Object(record), include_vectors)
    }

    fn score_kind(&self, field: &VectorStoreField) -> Option<String> {
        field
            .distance_function
            .is_none()
            .then(|| VectorSearchResult::SCORE_KIND_RELEVANCE.to_string())
    }

    /// Build the `points/query` body for a search.
    pub fn search_body(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
        score_threshold: Option<f64>,
    ) -> Result<Value> {
        options.validate()?;
        let field = self
            .definition
            .try_get_vector_field(options.vector_field_name.as_deref())
            .ok_or_else(|| {
                Error::Configuration("Qdrant search requires a declared vector field".into())
            })?;
        if let Some(dimensions) = field.dimensions {
            if vector.len() != dimensions {
                return Err(Error::Configuration(format!(
                    "query vector has {} dimensions but field '{}' declares {dimensions}",
                    vector.len(),
                    field.name
                )));
            }
        }
        let query = prepare_dense_vector(&json!(vector), &field.name)?;
        let mut body = json!({
            "query": query,
            "using": field.effective_storage_name(),
            "limit": options.top,
            "offset": options.skip,
            "with_payload": true,
            "with_vector": options.include_vectors,
        });
        if let Some(filter) = self.search_filter(options)? {
            body["filter"] = filter;
        }
        if let Some(threshold) = score_threshold {
            if !threshold.is_finite() {
                return Err(Error::Configuration(
                    "Qdrant score_threshold must be a finite number".into(),
                ));
            }
            body["score_threshold"] = json!(threshold);
        }
        Ok(body)
    }

    async fn run_search(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
        score_threshold: Option<f64>,
    ) -> Result<Vec<VectorSearchResult>> {
        let body = self.search_body(vector, options, score_threshold)?;
        let field = self
            .definition
            .try_get_vector_field(options.vector_field_name.as_deref())
            .ok_or_else(|| Error::Configuration("no vector field".into()))?;
        let score_kind = self.score_kind(field);
        let result = self
            .http
            .call(Method::POST, &self.path("/points/query"), Some(&body))
            .await?;
        let points = result
            .get("points")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::service("Qdrant returned a malformed query response"))?;
        points
            .iter()
            .map(|point| {
                Ok(VectorSearchResult {
                    record: self.decode_point(point, options.include_vectors)?,
                    score: point.get("score").and_then(Value::as_f64),
                    score_kind: score_kind.clone(),
                })
            })
            .collect()
    }

    /// Dense search keeping only scores past `score_threshold`, in Qdrant's
    /// native units (a minimum similarity for Cosine/Dot, a maximum distance
    /// for Euclid/Manhattan). Upstream's `score_threshold`.
    pub async fn search_with_score_threshold(
        &self,
        vector: Vec<f32>,
        score_threshold: f64,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        self.run_search(&vector, options, Some(score_threshold))
            .await
    }

    /// Unordered filtered retrieval via `points/scroll`: up to `top` records
    /// matching `filter` after skipping `skip`. Upstream's keyless `get`
    /// (ordering is unsupported there too).
    pub async fn get_filtered(
        &self,
        filter: Option<&FilterExpression>,
        top: usize,
        skip: usize,
        include_vectors: bool,
    ) -> Result<Vec<Value>> {
        let native = self.prepare_filter(filter)?;
        let mut records = Vec::new();
        let mut offset = Value::Null;
        let mut remaining_skip = skip;
        while records.len() < top {
            let limit = BATCH_SIZE.min(remaining_skip + top - records.len());
            let mut body = json!({
                "limit": limit,
                "with_payload": true,
                "with_vector": include_vectors,
            });
            if let Some(filter) = &native {
                body["filter"] = filter.clone();
            }
            if !offset.is_null() {
                body["offset"] = offset.clone();
            }
            let result = self
                .http
                .call(Method::POST, &self.path("/points/scroll"), Some(&body))
                .await?;
            let page = result
                .get("points")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let discarded = remaining_skip.min(page.len());
            remaining_skip -= discarded;
            for point in &page[discarded..] {
                if records.len() < top {
                    records.push(self.decode_point(point, include_vectors)?);
                }
            }
            offset = result
                .get("next_page_offset")
                .cloned()
                .unwrap_or(Value::Null);
            if offset.is_null() {
                break;
            }
        }
        Ok(records)
    }

    async fn read_collection_info(&self, creation_conflict: bool) -> Result<Value> {
        let mut attempt = 0;
        loop {
            match self.http.send(Method::GET, &self.path(""), None).await? {
                Ok(info) => return Ok(info),
                Err(failure) => {
                    // A creation conflict can be reported before the winning
                    // creator has initialized its shards; retry only that.
                    let not_ready = failure.status == 500
                        && failure.body.contains("0 of 0 read operations failed");
                    if !creation_conflict || !not_ready || attempt >= MAX_READINESS_RETRIES {
                        return Err(failure_error(failure));
                    }
                    tokio::time::sleep(Duration::from_millis(50 * 2u64.pow(attempt))).await;
                    attempt += 1;
                }
            }
        }
    }

    fn validate_info(&self, info: &Value) -> Result<Vec<(String, &'static str)>> {
        let configured = info
            .pointer("/config/params/vectors")
            .and_then(Value::as_object)
            .filter(|v| !v.get("size").is_some_and(Value::is_number))
            .ok_or_else(|| {
                Error::Configuration(
                    "QdrantCollection requires named vectors, not an unnamed-vector collection"
                        .into(),
                )
            })?;
        let default_m = info
            .pointer("/config/hnsw_config/m")
            .and_then(Value::as_u64);
        for field in self.definition.vector_fields() {
            let mismatch = || {
                Error::Configuration(format!(
                    "existing Qdrant vector '{}' does not match the collection definition",
                    field.name
                ))
            };
            let vector = configured
                .get(field.effective_storage_name())
                .ok_or_else(mismatch)?;
            let size_ok =
                vector.get("size").and_then(Value::as_u64) == field.dimensions.map(|d| d as u64);
            let distance_ok =
                vector.get("distance").and_then(Value::as_str) == Some(distance_of(field)?);
            let multivector_ok = vector.get("multivector_config").is_none_or(Value::is_null);
            let datatype_ok = match vector.get("datatype") {
                None | Some(Value::Null) => true,
                Some(v) => v
                    .as_str()
                    .is_some_and(|s| s.eq_ignore_ascii_case("float32")),
            };
            if !(size_ok && distance_ok && multivector_ok && datatype_ok) {
                return Err(mismatch());
            }
            let m = vector
                .pointer("/hnsw_config/m")
                .and_then(Value::as_u64)
                .or(default_m);
            let flat = is_flat(field)?;
            let explicit_hnsw =
                field.index_kind.as_ref().map(IndexKind::as_str) == Some(IndexKind::HNSW);
            if flat && m != Some(0) {
                return Err(Error::Configuration(format!(
                    "existing Qdrant vector '{}' does not use a flat index",
                    field.name
                )));
            }
            if explicit_hnsw && m == Some(0) {
                return Err(Error::Configuration(format!(
                    "existing Qdrant vector '{}' does not use an HNSW index",
                    field.name
                )));
            }
        }
        let schema = info.get("payload_schema").and_then(Value::as_object);
        let mut missing = Vec::new();
        for field in self.definition.data_fields() {
            let Some(index) = payload_index(field)? else {
                continue;
            };
            let name = field.effective_storage_name();
            match schema
                .and_then(|s| s.get(name))
                .and_then(|e| e.get("data_type"))
                .and_then(Value::as_str)
            {
                Some(existing) if existing != index => {
                    return Err(Error::Configuration(format!(
                        "existing Qdrant payload index '{name}' has a different type"
                    )))
                }
                Some(_) => {}
                None => missing.push((name.to_string(), index)),
            }
        }
        Ok(missing)
    }

    async fn upsert_batch(&self, points: &[Value]) -> Result<()> {
        let result = self
            .http
            .call(
                Method::PUT,
                &self.path("/points?wait=true"),
                Some(&json!({ "points": points })),
            )
            .await?;
        require_completed(&result)
    }
}

#[async_trait]
impl VectorCollection for QdrantCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        let mut creation_conflict = false;
        if !collection_exists(&self.http, &self.name).await? {
            let body = self.create_collection_body()?;
            match self
                .http
                .send(Method::PUT, &self.path(""), Some(&body))
                .await?
            {
                Ok(created) => {
                    if created != Value::Bool(true)
                        && !collection_exists(&self.http, &self.name).await?
                    {
                        return Err(Error::service(format!(
                            "Qdrant did not create collection '{}'",
                            self.name
                        )));
                    }
                }
                Err(failure) if failure.status == 409 => creation_conflict = true,
                Err(failure) => return Err(failure_error(failure)),
            }
        }
        let info = self.read_collection_info(creation_conflict).await?;
        for (name, schema) in self.validate_info(&info)? {
            let result = self
                .http
                .call(
                    Method::PUT,
                    &self.path("/index?wait=true"),
                    Some(&json!({"field_name": name, "field_schema": schema})),
                )
                .await?;
            require_completed(&result)?;
        }
        Ok(())
    }

    async fn collection_exists(&self) -> Result<bool> {
        collection_exists(&self.http, &self.name).await
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        if collection_exists(&self.http, &self.name).await? {
            delete_collection(&self.http, &self.name).await?;
        }
        Ok(())
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        let mut keys = Vec::with_capacity(records.len());
        let mut points = Vec::with_capacity(records.len());
        for (index, record) in records.iter().enumerate() {
            let (key, point) = self.encode_point(record).map_err(|e| {
                Error::Configuration(format!("record at index {index} is invalid: {e}"))
            })?;
            keys.push(key);
            points.push(point);
        }
        let total = points.len();
        for (batch, chunk) in points.chunks(BATCH_SIZE).enumerate() {
            let written = batch * BATCH_SIZE;
            self.upsert_batch(chunk).await.map_err(|e| {
                Error::service(format!(
                    "Qdrant upsert wrote {written}/{total} records before failing: {e}"
                ))
            })?;
        }
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        let ids = keys
            .iter()
            .map(prepare_point_id)
            .collect::<Result<Vec<_>>>()?;
        let mut found: HashMap<String, Value> = HashMap::new();
        for chunk in ids.chunks(BATCH_SIZE) {
            let result = self
                .http
                .call(
                    Method::POST,
                    &self.path("/points"),
                    Some(&json!({
                        "ids": chunk,
                        "with_payload": true,
                        "with_vector": include_vectors,
                    })),
                )
                .await?;
            for point in result.as_array().cloned().unwrap_or_default() {
                let token = id_token(point.get("id").unwrap_or(&Value::Null));
                found.insert(token, self.decode_point(&point, include_vectors)?);
            }
        }
        Ok(ids
            .iter()
            .map(|id| found.get(&id_token(id)).cloned())
            .collect())
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        let ids = keys
            .iter()
            .map(prepare_point_id)
            .collect::<Result<Vec<_>>>()?;
        for chunk in ids.chunks(BATCH_SIZE) {
            let result = self
                .http
                .call(
                    Method::POST,
                    &self.path("/points/delete?wait=true"),
                    Some(&json!({ "points": chunk })),
                )
                .await?;
            require_completed(&result)?;
        }
        Ok(())
    }

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

    fn store() -> QdrantStore {
        QdrantStore::new("http://127.0.0.1:6333").unwrap()
    }

    fn definition() -> VectorStoreCollectionDefinition {
        VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id").with_type("int"),
            VectorStoreField::data("text")
                .with_type("str")
                .with_storage_name("body"),
            VectorStoreField::data("tags").with_type("list"),
            VectorStoreField::data("integer").with_type("int").indexed(),
            VectorStoreField::data("number")
                .with_type("float")
                .with_storage_name("price"),
            VectorStoreField::data("flag").with_type("bool"),
            VectorStoreField::data("any"),
            VectorStoreField::vector("embedding", 3)
                .with_distance_function(DistanceFunction::new(DistanceFunction::DOT_PROD)),
        ])
        .unwrap()
    }

    fn collection() -> QdrantCollection {
        store().collection("docs", definition()).unwrap()
    }

    fn t(filter: Filter) -> Value {
        collection()
            .prepare_filter(Some(&filter.into()))
            .unwrap()
            .unwrap()["must"][0]
            .clone()
    }

    #[test]
    fn settings_prefer_explicit_values() {
        let s = QdrantSettings::load(Some("http://x:1".into()), Some("k".into()));
        assert_eq!(s.url.as_deref(), Some("http://x:1"));
        assert_eq!(s.api_key.as_deref(), Some("k"));
        assert!(!format!("{s:?}").contains('k') || format!("{s:?}").contains("***"));
        let store = QdrantStore::from_settings(s).unwrap();
        assert!(!format!("{store:?}").contains("\"k\""));
        assert!(QdrantStore::new("ftp://nope").is_err());
    }

    #[test]
    fn point_ids_are_u64_or_canonical_uuids() {
        assert_eq!(prepare_point_id(&json!(7)).unwrap(), json!(7));
        assert_eq!(
            prepare_point_id(&json!("6BA7B810-9DAD-11D1-80B4-00C04FD430C8")).unwrap(),
            json!("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        );
        for bad in [
            json!(-1),
            json!(1.5),
            json!(true),
            json!("abc"),
            json!(null),
        ] {
            assert!(prepare_point_id(&bad).is_err(), "{bad}");
        }
        assert_eq!(prepare_point_id(&json!(u64::MAX)).unwrap(), json!(u64::MAX));
    }

    #[test]
    fn unsupported_definitions_are_refused() {
        let cases: Vec<Vec<VectorStoreField>> = vec![
            vec![VectorStoreField::key("id").with_type("float")],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::vector("v", 3).with_type("int"),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::vector("v", 3)
                    .with_distance_function(DistanceFunction::new(DistanceFunction::HAMMING)),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::vector("v", 3)
                    .with_index_kind(IndexKind::new(IndexKind::DISK_ANN)),
            ],
            vec![VectorStoreField::key("id"), VectorStoreField::data("a.b")],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::data("t").full_text_indexed(),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::data("t").with_type("list").indexed(),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::data("t").indexed(),
            ],
            vec![
                VectorStoreField::key("id"),
                VectorStoreField::data("t").with_type("bytes"),
            ],
        ];
        for fields in cases {
            let def = VectorStoreCollectionDefinition::new(fields.clone()).unwrap();
            assert!(store().collection("c", def).is_err(), "{fields:?}");
        }
    }

    #[test]
    fn collection_body_uses_named_vectors() {
        let def = VectorStoreCollectionDefinition::new(vec![
            VectorStoreField::key("id"),
            VectorStoreField::vector("a", 2),
            VectorStoreField::vector("b", 4)
                .with_storage_name("bee")
                .with_index_kind(IndexKind::new(IndexKind::FLAT))
                .with_distance_function(DistanceFunction::new(DistanceFunction::MANHATTAN)),
        ])
        .unwrap();
        let body = store()
            .collection("c", def)
            .unwrap()
            .create_collection_body()
            .unwrap();
        assert_eq!(
            body,
            json!({"vectors": {
                "a": {"size": 2, "distance": "Cosine"},
                "bee": {"size": 4, "distance": "Manhattan", "hnsw_config": {"m": 0}},
            }})
        );
    }

    #[test]
    fn data_filters_translate_like_upstream() {
        assert_eq!(
            t(Filter::exists("text").unwrap()),
            json!({"key": "body", "values_count": {"gte": 0}})
        );
        assert_eq!(
            t(Filter::is_null("text").unwrap()),
            json!({"is_null": {"key": "body"}})
        );
        assert_eq!(
            t(Filter::eq("text", "x").unwrap()),
            json!({"key": "body", "match": {"value": "x"}})
        );
        // Type-incompatible equality is a constant false, not an error.
        assert_eq!(t(Filter::eq("text", 1).unwrap()), json!({"has_id": []}));
        assert_eq!(t(Filter::eq("flag", 1).unwrap()), json!({"has_id": []}));
        assert_eq!(
            t(Filter::eq("integer", true).unwrap()),
            json!({"has_id": []})
        );
        // Integer fields match exactly; an integral float is the same value.
        assert_eq!(
            t(Filter::eq("integer", 1.0).unwrap()),
            json!({"key": "integer", "match": {"value": 1}})
        );
        assert_eq!(
            t(Filter::eq("integer", 1.5).unwrap()),
            json!({"has_id": []})
        );
        assert_eq!(
            t(Filter::eq("integer", i64::MAX).unwrap()),
            json!({"key": "integer", "match": {"value": i64::MAX}})
        );
        // Float fields compare through a degenerate range.
        assert_eq!(
            t(Filter::eq("number", 1).unwrap()),
            json!({"key": "price", "range": {"gte": 1.0, "lte": 1.0}})
        );
        assert_eq!(
            t(Filter::ne("text", "other").unwrap()),
            json!({"must": [{"key": "body", "values_count": {"gte": 0}}],
                   "must_not": [{"key": "body", "match": {"value": "other"}}]})
        );
        assert_eq!(
            t(Filter::gt("number", 1).unwrap()),
            json!({"key": "price", "range": {"gt": 1.0}})
        );
        assert_eq!(
            t(Filter::between("number", 1, 2).unwrap()),
            json!({"key": "price", "range": {"gte": 1.0, "lte": 2.0}})
        );
        assert_eq!(
            t(Filter::any_of("text", Vec::<Value>::new()).unwrap()),
            json!({"must": [{"must": [{"key": "body", "values_count": {"gte": 0}}],
                             "must_not": [{"is_null": {"key": "body"}}]},
                            {"has_id": []}]})
        );
        assert_eq!(
            t(Filter::contains("tags", "two").unwrap()),
            json!({"should": [{"key": "tags", "match": {"value": "two"}}]})
        );
        assert_eq!(
            t(Filter::contains("tags", 1).unwrap()),
            json!({"should": [{"key": "tags", "range": {"gte": 1.0, "lte": 1.0}}]})
        );
        assert_eq!(
            t(Filter::contains_any("tags", Vec::<Value>::new()).unwrap()),
            json!({"has_id": []})
        );
        let all = t(Filter::contains_all("tags", Vec::<Value>::new()).unwrap());
        assert_eq!(all["must"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn key_filters_use_has_id() {
        assert_eq!(t(Filter::eq("id", 3).unwrap()), json!({"has_id": [3]}));
        assert_eq!(t(Filter::eq("id", 3.0).unwrap()), json!({"has_id": [3]}));
        assert_eq!(t(Filter::eq("id", 3.5).unwrap()), json!({"has_id": []}));
        assert_eq!(t(Filter::eq("id", true).unwrap()), json!({"has_id": []}));
        assert_eq!(t(Filter::eq("id", -1).unwrap()), json!({"has_id": []}));
        // An int key never matches a UUID operand.
        assert_eq!(
            t(Filter::eq("id", "6ba7b810-9dad-11d1-80b4-00c04fd430c8").unwrap()),
            json!({"has_id": []})
        );
        assert_eq!(
            t(Filter::none_of("id", vec![json!(0), json!(3)]).unwrap()),
            json!({"must_not": [{"has_id": [0, 3]}]})
        );
        assert_eq!(t(Filter::is_null("id").unwrap()), json!({"has_id": []}));
        assert_eq!(
            t(Filter::exists("id").unwrap()),
            json!({"must_not": [{"has_id": []}]})
        );
        assert!(collection()
            .prepare_filter(Some(&Filter::gt("id", 1).unwrap().into()))
            .is_err());
    }

    #[test]
    fn groups_translate_structurally() {
        let expr = FilterGroup::not(
            FilterGroup::and(vec![
                Filter::gte("integer", 1).unwrap().into(),
                Filter::eq("flag", true).unwrap().into(),
            ])
            .unwrap(),
        )
        .unwrap();
        let native = collection().prepare_filter(Some(&expr)).unwrap().unwrap();
        assert_eq!(
            native,
            json!({"must": [{"must_not": [{"must": [
                {"key": "integer", "range": {"gte": 1.0}},
                {"key": "flag", "match": {"value": true}},
            ]}]}]})
        );
    }

    #[test]
    fn unsupported_filters_are_refused() {
        let c = collection();
        for filter in [
            Filter::starts_with("text", "x").unwrap(),
            Filter::contains_text("text", "x").unwrap(),
            Filter::gt("text", 1).unwrap(),
            Filter::contains("text", "x").unwrap(),
            Filter::eq("embedding", 1).unwrap(),
            Filter::eq("missing", 1).unwrap(),
            Filter::eq("any", 1).unwrap(),
            Filter::gt("number", 1u64 << 60).unwrap(),
            Filter::contains("tags", Value::Null).unwrap(),
        ] {
            assert!(
                c.prepare_filter(Some(&filter.clone().into())).is_err(),
                "{filter:?}"
            );
        }
    }

    #[test]
    fn points_are_validated_and_shaped() {
        let c = collection();
        let (key, point) = c
            .encode_point(&json!({"id": 1, "text": "t", "tags": ["a", [2, null]],
                                  "embedding": [1, 2.5, 3], "extra": true}))
            .unwrap();
        assert_eq!(key, json!(1));
        assert_eq!(
            point,
            json!({"id": 1, "vector": {"embedding": [1.0, 2.5, 3.0]},
                   "payload": {"body": "t", "tags": ["a", [2, null]]}})
        );
        // A missing vector is omitted, not sent as null.
        let (_, point) = c
            .encode_point(&json!({"id": 2, "embedding": null}))
            .unwrap();
        assert_eq!(point["vector"], json!({}));
        for bad in [
            json!({"text": "no key"}),
            json!({"id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8"}), // int key declared
            json!({"id": 1, "integer": 1.5}),
            json!({"id": 1, "flag": 1}),
            json!({"id": 1, "tags": "x"}),
            json!({"id": 1, "any": {"deep": [u64::MAX]}}),
            json!({"id": 1, "embedding": [true, 1, 2]}),
            json!({"id": 1, "embedding": [1e39, 1, 2]}),
            json!({"id": 1, "embedding": "x"}),
        ] {
            assert!(c.encode_point(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn decode_restores_logical_names_and_key() {
        let c = collection();
        let record = c
            .decode_point(
                &json!({"id": 5, "payload": {"body": "t"}, "vector": {"embedding": [1.0, 0.0, 0.0]}}),
                true,
            )
            .unwrap();
        assert_eq!(
            record,
            json!({"id": 5, "text": "t", "embedding": [1.0, 0.0, 0.0]})
        );
        assert!(c
            .decode_point(&json!({"id": 5, "vector": [1.0]}), true)
            .is_err());
    }

    #[test]
    fn search_body_shape() {
        let c = collection();
        let body = c
            .search_body(
                &[1.0, 0.0, 0.0],
                &VectorSearchOptions::new(2)
                    .with_skip(1)
                    .with_include_vectors(true)
                    .with_filter(Filter::eq("flag", true).unwrap())
                    .with_provider_filter(r#"{"must": [{"key": "x", "match": {"value": 1}}]}"#),
                Some(0.5),
            )
            .unwrap();
        assert_eq!(body["using"], "embedding");
        assert_eq!(body["limit"], 2);
        assert_eq!(body["offset"], 1);
        assert_eq!(body["with_vector"], true);
        assert_eq!(body["score_threshold"], 0.5);
        assert_eq!(body["filter"]["must"].as_array().unwrap().len(), 2);
        assert!(c
            .search_body(&[1.0], &VectorSearchOptions::new(2), None)
            .is_err());
        assert!(c
            .search_body(
                &[1.0, 0.0, 0.0],
                &VectorSearchOptions::new(2).with_provider_filter("[1]"),
                None
            )
            .is_err());
        assert!(c
            .search_body(
                &[1.0, 0.0, 0.0],
                &VectorSearchOptions::new(2),
                Some(f64::NAN)
            )
            .is_err());
    }

    #[test]
    fn path_segments_are_encoded() {
        assert_eq!(segment("a b/c"), "a%20b%2Fc");
        assert_eq!(collection().path("/points"), "/collections/docs/points");
    }
}
