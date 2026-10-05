//! Python-`repr` renderings used in check reasons.
//!
//! Upstream builds human-readable reasons with f-strings over Python lists
//! and dicts (`f"Missing keywords: {missing}"`), so the text reads
//! `['weather', 'temp']`. These helpers reproduce that formatting so reason
//! strings match upstream's byte for byte where the values allow.

use serde_json::Value;

/// `repr(str)`: single quotes unless the string contains a single quote and
/// no double quote, with the usual escapes.
pub(crate) fn string(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr(list[str])`.
pub(crate) fn string_list<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items.iter().map(|s| string(s.as_ref())).collect();
    format!("[{}]", inner.join(", "))
}

/// `repr(float)`: always shows a fractional part (`1.0`, `0.5`).
pub(crate) fn float(f: f64) -> String {
    if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// `repr` of a JSON value as the equivalent Python object (`None`, `True`,
/// `{'k': 'v'}`). Object keys render in the map's iteration order.
pub(crate) fn value(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => i.to_string(),
            (_, Some(u), _) => u.to_string(),
            (_, _, Some(f)) => float(f),
            _ => n.to_string(),
        },
        Value::String(s) => string(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(value).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", string(k), value(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Python `==` between two JSON values: numbers compare by value across the
/// int/float split (`1 == 1.0`), and `True == 1` / `False == 0`, recursing
/// through arrays and objects. `serde_json`'s own equality treats `1` and
/// `1.0` as different.
pub(crate) fn py_eq(a: &Value, b: &Value) -> bool {
    fn num(v: &Value) -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        (Value::Number(_) | Value::Bool(_), Value::Number(_) | Value::Bool(_)) => match (a, b) {
            (Value::Bool(x), Value::Bool(y)) => x == y,
            _ => num(a) == num(b),
        },
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strings_and_lists_render_like_python() {
        assert_eq!(string_list(&["a", "b"]), "['a', 'b']");
        assert_eq!(string("it's"), "\"it's\"");
        assert_eq!(string("a'\"b"), "'a\\'\"b'");
        assert_eq!(string_list::<&str>(&[]), "[]");
    }

    #[test]
    fn values_render_like_python() {
        assert_eq!(
            value(&json!({"a": 1, "b": [true, null, 1.5, "x"]})),
            "{'a': 1, 'b': [True, None, 1.5, 'x']}"
        );
        assert_eq!(float(0.8), "0.8");
        assert_eq!(float(1.0), "1.0");
    }

    #[test]
    fn python_equality_spans_int_and_float() {
        assert!(py_eq(&json!(1), &json!(1.0)));
        assert!(py_eq(&json!(true), &json!(1)));
        assert!(!py_eq(&json!("1"), &json!(1)));
        assert!(py_eq(&json!({"a": [1, 2.0]}), &json!({"a": [1.0, 2]})));
        assert!(!py_eq(&json!({"a": 1}), &json!({"a": 1, "b": 2})));
    }
}
