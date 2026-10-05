//! The PowerFx runtime value model and its JSON mapping.

use serde_json::{Map, Number, Value as Json};

use super::{PowerFxError, PowerFxErrorKind};

/// A PowerFx record: an ordered list of named fields.
///
/// Field order is preserved (records render in declaration order); lookups
/// are linear, which is fine for the small records workflows use.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Record {
    fields: Vec<(String, Value)>,
}

impl Record {
    /// An empty record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a field by (case-sensitive) name.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    /// Insert or replace a field, keeping the original position on replace.
    pub fn insert(&mut self, name: impl Into<String>, value: Value) {
        let name = name.into();
        if let Some(slot) = self.fields.iter_mut().find(|(k, _)| *k == name) {
            slot.1 = value;
        } else {
            self.fields.push((name, value));
        }
    }

    /// Remove a field, returning its value.
    pub fn remove(&mut self, name: &str) -> Option<Value> {
        let idx = self.fields.iter().position(|(k, _)| k == name)?;
        Some(self.fields.remove(idx).1)
    }

    /// Whether the record has a field named `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Iterate over `(name, value)` pairs in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the record has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

impl FromIterator<(String, Value)> for Record {
    fn from_iter<I: IntoIterator<Item = (String, Value)>>(iter: I) -> Self {
        let mut r = Record::new();
        for (k, v) in iter {
            r.insert(k, v);
        }
        r
    }
}

/// A PowerFx runtime value.
///
/// Numbers are IEEE-754 doubles. PowerFx V1 defaults to a decimal type; the
/// difference is hidden when values are rendered (see [`format_number`]),
/// which rounds to 15 significant digits so `0.1 + 0.2` renders as `0.3`.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Value {
    /// `Blank()` — the PowerFx null.
    #[default]
    Blank,
    /// A Boolean.
    Boolean(bool),
    /// A number.
    Number(f64),
    /// Text.
    Text(String),
    /// A record.
    Record(Record),
    /// A table. Rows are always records; scalar tables use a single `Value`
    /// column, exactly like PowerFx's `[1, 2, 3]`.
    Table(Vec<Record>),
}

impl Value {
    /// A short type name for diagnostics.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Blank => "Blank",
            Value::Boolean(_) => "Boolean",
            Value::Number(_) => "Number",
            Value::Text(_) => "Text",
            Value::Record(_) => "Record",
            Value::Table(_) => "Table",
        }
    }

    /// Whether the value is `Blank()` or the empty string (PowerFx `IsBlank`).
    pub fn is_blank(&self) -> bool {
        match self {
            Value::Blank => true,
            Value::Text(s) => s.is_empty(),
            _ => false,
        }
    }

    /// Wrap a scalar in a single-column `{Value: …}` row; records pass
    /// through unchanged.
    pub fn into_row(self) -> Record {
        match self {
            Value::Record(r) => r,
            other => {
                let mut r = Record::new();
                r.insert("Value", other);
                r
            }
        }
    }

    /// Convert a JSON value into a PowerFx value.
    ///
    /// Arrays become tables; non-object elements are wrapped in a `Value`
    /// column so `["a", "b"]` behaves like the PowerFx literal `["a", "b"]`.
    pub fn from_json(json: &Json) -> Value {
        match json {
            Json::Null => Value::Blank,
            Json::Bool(b) => Value::Boolean(*b),
            Json::Number(n) => Value::Number(n.as_f64().unwrap_or(0.0)),
            Json::String(s) => Value::Text(s.clone()),
            Json::Object(map) => Value::Record(
                map.iter()
                    .map(|(k, v)| (k.clone(), Value::from_json(v)))
                    .collect(),
            ),
            Json::Array(items) => Value::Table(
                items
                    .iter()
                    .map(|item| Value::from_json(item).into_row())
                    .collect(),
            ),
        }
    }

    /// Convert to JSON.
    ///
    /// Tables become arrays; a row whose only field is `Value` is unwrapped to
    /// its scalar so `["a", "b"]` round-trips. (A JSON object whose sole key is
    /// `Value` is therefore also unwrapped — a deliberate, documented
    /// simplification matching .NET's `ToLoopValue`.) Integral numbers become
    /// JSON integers; others are rounded to 15 significant digits.
    pub fn to_json(&self) -> Json {
        match self {
            Value::Blank => Json::Null,
            Value::Boolean(b) => Json::Bool(*b),
            Value::Number(n) => number_to_json(*n),
            Value::Text(s) => Json::String(s.clone()),
            Value::Record(r) => record_to_json(r),
            Value::Table(rows) => Json::Array(
                rows.iter()
                    .map(|row| {
                        if row.len() == 1 {
                            if let Some(v) = row.get("Value") {
                                return v.to_json();
                            }
                        }
                        record_to_json(row)
                    })
                    .collect(),
            ),
        }
    }
}

fn record_to_json(r: &Record) -> Json {
    let mut m = Map::new();
    for (k, v) in r.iter() {
        m.insert(k.to_string(), v.to_json());
    }
    Json::Object(m)
}

/// Render a number to JSON: integers as integers, others rounded to 15
/// significant digits; non-finite values become `null`.
pub(crate) fn number_to_json(n: f64) -> Json {
    if !n.is_finite() {
        return Json::Null;
    }
    let rounded = round_significant(n);
    if rounded.fract() == 0.0 && rounded.abs() < 9.0e15 {
        return Json::Number(Number::from(rounded as i64));
    }
    Number::from_f64(rounded)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

/// Round to 15 significant digits (the precision PowerFx's decimal/float
/// rendering guarantees), removing binary-float noise such as
/// `0.30000000000000004`.
pub(crate) fn round_significant(n: f64) -> f64 {
    if n == 0.0 || !n.is_finite() {
        return n;
    }
    let s = format!("{n:.14e}");
    s.parse::<f64>().unwrap_or(n)
}

/// Format a number the way PowerFx's `Text()` (without a format string) does
/// in the en-US locale: integers without a decimal point, fractions with up
/// to 15 significant digits and no trailing zeros.
pub fn format_number(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_string();
    }
    if n.is_infinite() {
        return if n > 0.0 { "∞" } else { "-∞" }.to_string();
    }
    let r = round_significant(n);
    if r.fract() == 0.0 && r.abs() < 1e15 {
        return format!("{}", r as i64);
    }
    let s = format!("{r}");
    if s.contains('e') {
        // Very large/small magnitudes: fall back to plain decimal expansion.
        let plain = format!("{r:.20}");
        return trim_zeros(&plain);
    }
    s
}

fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

pub(crate) fn type_error(msg: impl Into<String>) -> PowerFxError {
    PowerFxError::new(PowerFxErrorKind::Type, msg)
}

/// Coerce to a number (`Blank` → 0, Booleans → 1/0, numeric text parses).
pub(crate) fn to_number(v: &Value) -> Result<f64, PowerFxError> {
    match v {
        Value::Blank => Ok(0.0),
        Value::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => Ok(*n),
        Value::Text(s) => parse_number_text(s).ok_or_else(|| {
            PowerFxError::new(
                PowerFxErrorKind::Runtime,
                format!("the value {s:?} cannot be converted to a number"),
            )
        }),
        other => Err(type_error(format!(
            "expected a number, found a {}",
            other.type_name()
        ))),
    }
}

/// Parse numeric text the way PowerFx's `Value()` does in en-US: optional
/// surrounding whitespace, thousands separators, a trailing `%`.
pub(crate) fn parse_number_text(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let (body, percent) = match t.strip_suffix('%') {
        Some(b) => (b.trim_end(), true),
        None => (t, false),
    };
    let cleaned: String = body.chars().filter(|c| *c != ',').collect();
    // Reject things f64::from_str accepts but PowerFx does not.
    let lower = cleaned.to_ascii_lowercase();
    if lower.contains("inf") || lower.contains("nan") {
        return None;
    }
    let n: f64 = cleaned.parse().ok()?;
    Some(if percent { n / 100.0 } else { n })
}

/// Coerce to text (`Blank` → `""`, numbers formatted, Booleans as
/// `true`/`false`). Records and tables are type errors.
pub(crate) fn to_text(v: &Value) -> Result<String, PowerFxError> {
    match v {
        Value::Blank => Ok(String::new()),
        Value::Boolean(b) => Ok(b.to_string()),
        Value::Number(n) => Ok(format_number(*n)),
        Value::Text(s) => Ok(s.clone()),
        other => Err(type_error(format!(
            "expected text, found a {}",
            other.type_name()
        ))),
    }
}

/// Coerce to a Boolean (`Blank` → false, numbers non-zero, text
/// `"true"`/`"false"` case-insensitively).
pub(crate) fn to_bool(v: &Value) -> Result<bool, PowerFxError> {
    match v {
        Value::Blank => Ok(false),
        Value::Boolean(b) => Ok(*b),
        Value::Number(n) => Ok(*n != 0.0),
        Value::Text(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            "" => Ok(false),
            _ => Err(PowerFxError::new(
                PowerFxErrorKind::Runtime,
                format!("the value {s:?} cannot be converted to a Boolean"),
            )),
        },
        other => Err(type_error(format!(
            "expected a Boolean, found a {}",
            other.type_name()
        ))),
    }
}

/// The scalar of a single-column `Value` row, else the row as a record.
pub(crate) fn row_scalar(row: &Record) -> Value {
    if row.len() == 1 {
        if let Some(v) = row.get("Value") {
            return v.clone();
        }
    }
    Value::Record(row.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_round_trip_preserves_shapes() {
        let j = json!({"a": [1, "x", {"b": true}], "n": null, "f": 1.5});
        assert_eq!(Value::from_json(&j).to_json(), j);
    }

    #[test]
    fn numbers_render_without_float_noise() {
        assert_eq!(format_number(0.1 + 0.2), "0.3");
        assert_eq!(format_number(3.0), "3");
        assert_eq!(format_number(-2.5), "-2.5");
        assert_eq!(number_to_json(0.1 + 0.2), json!(0.3));
        assert_eq!(number_to_json(4.0), json!(4));
    }

    #[test]
    fn coercions() {
        assert_eq!(to_number(&Value::Text(" 1,234.5 ".into())).unwrap(), 1234.5);
        assert_eq!(to_number(&Value::Text("50%".into())).unwrap(), 0.5);
        assert!(to_number(&Value::Text("abc".into())).is_err());
        assert!(to_number(&Value::Text("inf".into())).is_err());
        assert_eq!(to_text(&Value::Boolean(true)).unwrap(), "true");
        assert!(to_bool(&Value::Text("TRUE".into())).unwrap());
        assert!(to_bool(&Value::Text("yes".into())).is_err());
        assert!(to_text(&Value::Record(Record::new())).is_err());
    }
}
