//! The PowerFx function library (see the module docs of
//! [`powerfx`](super) for the supported list).

use std::cmp::Ordering;

use super::eval::{values_equal, Evaluator};
use super::parser::Expr;
use super::value::{
    parse_number_text, row_scalar, to_bool, to_number, to_text, type_error, Record, Value,
};
use super::{PowerFxError, PowerFxErrorKind};

/// Every function name the evaluator understands, for documentation and
/// diagnostics.
pub const SUPPORTED_FUNCTIONS: &[&str] = &[
    // logical
    "If",
    "Switch",
    "And",
    "Or",
    "Not",
    "IsBlank",
    "IsEmpty",
    "Coalesce",
    "Blank",
    "IsError",
    "IfError",
    "Boolean",
    // text
    "Text",
    "Value",
    "Concatenate",
    "Concat",
    "Len",
    "Lower",
    "Upper",
    "Proper",
    "Trim",
    "TrimEnds",
    "Left",
    "Right",
    "Mid",
    "Find",
    "StartsWith",
    "EndsWith",
    "Substitute",
    "Replace",
    "Split",
    "Char",
    "UniChar",
    "GUID",
    // tables
    "Table",
    "CountRows",
    "CountIf",
    "CountA",
    "Count",
    "First",
    "Last",
    "FirstN",
    "LastN",
    "Index",
    "Filter",
    "LookUp",
    "Search",
    "Sum",
    "Max",
    "Min",
    "Average",
    "Sort",
    "Distinct",
    "ForAll",
    "AddColumns",
    "ShowColumns",
    "DropColumns",
    "RenameColumns",
    "Sequence",
    // math
    "Round",
    "RoundUp",
    "RoundDown",
    "Int",
    "Trunc",
    "Abs",
    "Mod",
    "Power",
    "Sqrt",
    // JSON
    "ParseJSON",
    "JSON",
    // agent-framework custom functions
    "MessageText",
    "UserMessage",
    "AgentMessage",
    "AssistantMessage",
    "SystemMessage",
];

fn arity(name: &str, args: &[Expr], min: usize, max: usize) -> Result<(), PowerFxError> {
    if args.len() < min || args.len() > max {
        let expected = if min == max {
            format!("{min}")
        } else if max == usize::MAX {
            format!("at least {min}")
        } else {
            format!("{min} to {max}")
        };
        return Err(PowerFxError::new(
            PowerFxErrorKind::InvalidArguments,
            format!(
                "{name} expects {expected} argument(s), but {} were supplied",
                args.len()
            ),
        ));
    }
    Ok(())
}

fn runtime(msg: impl Into<String>) -> PowerFxError {
    PowerFxError::new(PowerFxErrorKind::Runtime, msg)
}

/// A column-name argument: a bare identifier or a text literal.
fn column_name(name: &str, e: &Expr) -> Result<String, PowerFxError> {
    match e {
        Expr::Name(n) => Ok(n.clone()),
        Expr::Text(t) => Ok(t.clone()),
        _ => Err(PowerFxError::new(
            PowerFxErrorKind::InvalidArguments,
            format!("{name} expects a column name"),
        )),
    }
}

/// Extract the text of a message record: `text`/`Text`, else the text items
/// of `contents`, else a string `content`.
pub(crate) fn message_record_text(r: &Record) -> String {
    for key in ["text", "Text"] {
        if let Some(v) = r.get(key) {
            return match v {
                Value::Blank => String::new(),
                other => to_text(other).unwrap_or_default(),
            };
        }
    }
    if let Some(Value::Table(contents)) = r.get("contents") {
        let parts: Vec<String> = contents
            .iter()
            .filter(|c| {
                matches!(c.get("type"), Some(Value::Text(t)) if t == "text") || c.contains("text")
            })
            .map(|c| match c.get("text") {
                Some(Value::Blank) | None => String::new(),
                Some(v) => to_text(v).unwrap_or_default(),
            })
            .collect();
        if !parts.is_empty() {
            return parts.join(" ");
        }
        return String::new();
    }
    if let Some(Value::Text(s)) = r.get("content") {
        return s.clone();
    }
    String::new()
}

fn round_to(n: f64, digits: f64, mode: Rounding) -> f64 {
    let factor = 10f64.powi(digits as i32);
    let scaled = n * factor;
    // Nudge to absorb binary representation error (2.675 → 267.49999…).
    let nudged = super::value::round_significant(scaled);
    let r = match mode {
        Rounding::HalfAwayFromZero => nudged.abs().round() * nudged.signum(),
        Rounding::Up => nudged.abs().ceil() * nudged.signum(),
        Rounding::Down => nudged.abs().floor() * nudged.signum(),
    };
    r / factor
}

#[derive(Clone, Copy)]
enum Rounding {
    HalfAwayFromZero,
    Up,
    Down,
}

fn compare_sort(a: &Value, b: &Value) -> Result<Ordering, PowerFxError> {
    match (a, b) {
        (Value::Blank, Value::Blank) => Ok(Ordering::Equal),
        (Value::Blank, _) => Ok(Ordering::Less),
        (_, Value::Blank) => Ok(Ordering::Greater),
        (Value::Text(x), Value::Text(y)) => Ok(x.cmp(y)),
        (Value::Boolean(x), Value::Boolean(y)) => Ok(x.cmp(y)),
        (x, y) => {
            let a = to_number(x)?;
            let b = to_number(y)?;
            Ok(a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        }
    }
}

/// Format a number with an Excel/PowerFx-style numeric format string such as
/// `"0"`, `"0.00"`, `"#,##0.0"` (optionally with literal prefix/suffix text).
fn format_with_pattern(n: f64, format: &str) -> Result<String, PowerFxError> {
    let start = format.find(['0', '#']);
    let Some(start) = start else {
        return Err(runtime(format!(
            "unsupported Text format string {format:?}"
        )));
    };
    let end = format[start..]
        .find(|c: char| !matches!(c, '0' | '#' | ',' | '.'))
        .map(|i| start + i)
        .unwrap_or(format.len());
    let (prefix, pattern, suffix) = (&format[..start], &format[start..end], &format[end..]);
    let literal = |s: &str| -> Result<String, PowerFxError> {
        let unquoted: String = s.chars().filter(|c| *c != '"').collect();
        if unquoted.chars().any(|c| c.is_alphabetic()) {
            return Err(runtime(format!(
                "unsupported Text format string {format:?} (date/time and named formats are not supported)"
            )));
        }
        Ok(unquoted)
    };
    let prefix = literal(prefix)?;
    let suffix = literal(suffix)?;
    let (int_pat, frac_pat) = match pattern.split_once('.') {
        Some((i, f)) => (i, f),
        None => (pattern, ""),
    };
    let grouping = int_pat.contains(',');
    let min_int = int_pat.chars().filter(|c| *c == '0').count();
    let max_frac = frac_pat.chars().filter(|c| matches!(c, '0' | '#')).count();
    let min_frac = frac_pat.chars().take_while(|c| *c == '0').count();
    let rounded = round_to(n, max_frac as f64, Rounding::HalfAwayFromZero);
    let s = format!("{:.*}", max_frac, rounded.abs());
    let (int_part, frac_part) = match s.split_once('.') {
        Some((i, f)) => (i.to_string(), f.to_string()),
        None => (s.clone(), String::new()),
    };
    let mut frac = frac_part;
    while frac.len() > min_frac && frac.ends_with('0') {
        frac.pop();
    }
    let mut int_digits = int_part.trim_start_matches('0').to_string();
    while int_digits.len() < min_int {
        int_digits.insert(0, '0');
    }
    if grouping {
        let chars: Vec<char> = int_digits.chars().collect();
        let mut grouped = String::new();
        for (i, c) in chars.iter().enumerate() {
            if i > 0 && (chars.len() - i).is_multiple_of(3) {
                grouped.push(',');
            }
            grouped.push(*c);
        }
        int_digits = grouped;
    }
    let mut out = String::new();
    if rounded < 0.0 {
        out.push('-');
    }
    out.push_str(&prefix);
    out.push_str(&int_digits);
    if !frac.is_empty() {
        out.push('.');
        out.push_str(&frac);
    }
    out.push_str(&suffix);
    Ok(out)
}

impl Evaluator<'_> {
    fn arg(&mut self, args: &[Expr], i: usize) -> Result<Value, PowerFxError> {
        match args.get(i) {
            Some(e) => self.eval(e),
            None => Ok(Value::Blank),
        }
    }

    fn text_arg(&mut self, args: &[Expr], i: usize) -> Result<String, PowerFxError> {
        let v = self.arg(args, i)?;
        to_text(&v)
    }

    fn num_arg(&mut self, args: &[Expr], i: usize) -> Result<f64, PowerFxError> {
        let v = self.arg(args, i)?;
        to_number(&v)
    }

    fn table_arg(
        &mut self,
        name: &str,
        args: &[Expr],
        i: usize,
    ) -> Result<Vec<Record>, PowerFxError> {
        let v = self.arg(args, i)?;
        match v {
            Value::Table(rows) => Ok(rows),
            Value::Blank => Ok(Vec::new()),
            other => Err(type_error(format!(
                "{name} expects a table, found a {}",
                other.type_name()
            ))),
        }
    }

    /// Evaluate numeric aggregate input: `F(table, expr)` or `F(n1, n2, …)`.
    fn aggregate_inputs(&mut self, name: &str, args: &[Expr]) -> Result<Vec<f64>, PowerFxError> {
        arity(name, args, 1, usize::MAX)?;
        let first = self.arg(args, 0)?;
        let mut out = Vec::new();
        if let Value::Table(rows) = first {
            arity(name, args, 2, 2)?;
            for row in &rows {
                let v = self.with_row(row, &args[1])?;
                if !matches!(v, Value::Blank) {
                    out.push(to_number(&v)?);
                }
            }
        } else {
            if !matches!(first, Value::Blank) {
                out.push(to_number(&first)?);
            }
            for i in 1..args.len() {
                let v = self.arg(args, i)?;
                if !matches!(v, Value::Blank) {
                    out.push(to_number(&v)?);
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn call_function(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<Value, PowerFxError> {
        match name {
            // ---------------------------------------------------- logical
            "If" => {
                arity(name, args, 2, usize::MAX)?;
                let mut i = 0;
                while i + 1 < args.len() {
                    let c = self.eval(&args[i])?;
                    if to_bool(&c)? {
                        return self.eval(&args[i + 1]);
                    }
                    i += 2;
                }
                if i < args.len() {
                    self.eval(&args[i])
                } else {
                    Ok(Value::Blank)
                }
            }
            "Switch" => {
                arity(name, args, 3, usize::MAX)?;
                let subject = self.eval(&args[0])?;
                let mut i = 1;
                while i + 1 < args.len() {
                    let candidate = self.eval(&args[i])?;
                    if values_equal(&subject, &candidate)? {
                        return self.eval(&args[i + 1]);
                    }
                    i += 2;
                }
                if i < args.len() {
                    self.eval(&args[i])
                } else {
                    Ok(Value::Blank)
                }
            }
            "And" => {
                arity(name, args, 1, usize::MAX)?;
                for a in args {
                    let v = self.eval(a)?;
                    if !to_bool(&v)? {
                        return Ok(Value::Boolean(false));
                    }
                }
                Ok(Value::Boolean(true))
            }
            "Or" => {
                arity(name, args, 1, usize::MAX)?;
                for a in args {
                    let v = self.eval(a)?;
                    if to_bool(&v)? {
                        return Ok(Value::Boolean(true));
                    }
                }
                Ok(Value::Boolean(false))
            }
            "Not" => {
                arity(name, args, 1, 1)?;
                let v = self.eval(&args[0])?;
                Ok(Value::Boolean(!to_bool(&v)?))
            }
            "IsBlank" => {
                arity(name, args, 1, 1)?;
                Ok(Value::Boolean(self.eval(&args[0])?.is_blank()))
            }
            "IsEmpty" => {
                arity(name, args, 1, 1)?;
                let v = self.eval(&args[0])?;
                Ok(Value::Boolean(match v {
                    Value::Blank => true,
                    Value::Table(rows) => rows.is_empty(),
                    other => {
                        return Err(type_error(format!(
                            "IsEmpty expects a table, found a {}",
                            other.type_name()
                        )))
                    }
                }))
            }
            "Coalesce" => {
                arity(name, args, 1, usize::MAX)?;
                for a in args {
                    let v = self.eval(a)?;
                    if !v.is_blank() {
                        return Ok(v);
                    }
                }
                Ok(Value::Blank)
            }
            "Blank" => {
                arity(name, args, 0, 0)?;
                Ok(Value::Blank)
            }
            "IsError" => {
                arity(name, args, 1, 1)?;
                match self.eval(&args[0]) {
                    Ok(_) => Ok(Value::Boolean(false)),
                    Err(e) if e.is_catchable() => Ok(Value::Boolean(true)),
                    Err(e) => Err(e),
                }
            }
            "IfError" => {
                arity(name, args, 2, usize::MAX)?;
                let mut i = 0;
                let mut last = Value::Blank;
                while i + 1 < args.len() {
                    match self.eval(&args[i]) {
                        Ok(v) => last = v,
                        Err(e) if e.is_catchable() => return self.eval(&args[i + 1]),
                        Err(e) => return Err(e),
                    }
                    i += 2;
                }
                if i < args.len() {
                    self.eval(&args[i])
                } else {
                    Ok(last)
                }
            }
            "Boolean" => {
                arity(name, args, 1, 1)?;
                let v = self.eval(&args[0])?;
                if v.is_blank() {
                    return Ok(Value::Blank);
                }
                Ok(Value::Boolean(to_bool(&v)?))
            }

            // ---------------------------------------------------- text
            "Text" => {
                arity(name, args, 1, 3)?;
                let v = self.eval(&args[0])?;
                if args.len() == 1 {
                    return Ok(Value::Text(to_text(&v)?));
                }
                let format = self.text_arg(args, 1)?;
                if v.is_blank() {
                    return Ok(Value::Text(String::new()));
                }
                let n = to_number(&v)?;
                Ok(Value::Text(format_with_pattern(n, &format)?))
            }
            "Value" => {
                arity(name, args, 1, 2)?;
                let v = self.eval(&args[0])?;
                match v {
                    Value::Blank => Ok(Value::Blank),
                    Value::Number(n) => Ok(Value::Number(n)),
                    Value::Boolean(b) => Ok(Value::Number(if b { 1.0 } else { 0.0 })),
                    Value::Text(s) if s.trim().is_empty() => Ok(Value::Blank),
                    Value::Text(s) => parse_number_text(&s).map(Value::Number).ok_or_else(|| {
                        runtime(format!("the value {s:?} cannot be converted to a number"))
                    }),
                    other => Err(type_error(format!(
                        "Value expects text or a number, found a {}",
                        other.type_name()
                    ))),
                }
            }
            "Concatenate" => {
                let mut out = String::new();
                for i in 0..args.len() {
                    out.push_str(&self.text_arg(args, i)?);
                }
                Ok(Value::Text(out))
            }
            "Concat" => {
                arity(name, args, 1, usize::MAX)?;
                let first = self.eval(&args[0])?;
                if let Value::Table(rows) = first {
                    // Standard PowerFx: Concat(table, expression [, separator]).
                    arity(name, args, 2, 3)?;
                    let sep = if args.len() == 3 {
                        self.text_arg(args, 2)?
                    } else {
                        String::new()
                    };
                    let mut parts = Vec::with_capacity(rows.len());
                    for row in &rows {
                        let v = self.with_row(row, &args[1])?;
                        parts.push(to_text(&v)?);
                    }
                    return Ok(Value::Text(parts.join(&sep)));
                }
                // Copilot Studio dialect (as accepted upstream): string concatenation.
                let mut out = to_text(&first)?;
                for i in 1..args.len() {
                    out.push_str(&self.text_arg(args, i)?);
                }
                Ok(Value::Text(out))
            }
            "Len" => {
                arity(name, args, 1, 1)?;
                Ok(Value::Number(self.text_arg(args, 0)?.chars().count() as f64))
            }
            "Lower" => {
                arity(name, args, 1, 1)?;
                Ok(Value::Text(self.text_arg(args, 0)?.to_lowercase()))
            }
            "Upper" => {
                arity(name, args, 1, 1)?;
                Ok(Value::Text(self.text_arg(args, 0)?.to_uppercase()))
            }
            "Proper" => {
                arity(name, args, 1, 1)?;
                let s = self.text_arg(args, 0)?;
                let mut out = String::with_capacity(s.len());
                let mut start = true;
                for c in s.chars() {
                    if c.is_alphabetic() {
                        if start {
                            out.extend(c.to_uppercase());
                        } else {
                            out.extend(c.to_lowercase());
                        }
                        start = false;
                    } else {
                        out.push(c);
                        start = true;
                    }
                }
                Ok(Value::Text(out))
            }
            "Trim" => {
                arity(name, args, 1, 1)?;
                let s = self.text_arg(args, 0)?;
                Ok(Value::Text(
                    s.split(' ')
                        .filter(|p| !p.is_empty())
                        .collect::<Vec<_>>()
                        .join(" "),
                ))
            }
            "TrimEnds" => {
                arity(name, args, 1, 1)?;
                Ok(Value::Text(
                    self.text_arg(args, 0)?.trim_matches(' ').to_string(),
                ))
            }
            "Left" | "Right" => {
                arity(name, args, 2, 2)?;
                let s: Vec<char> = self.text_arg(args, 0)?.chars().collect();
                let n = self.num_arg(args, 1)?;
                if n < 0.0 {
                    return Err(runtime(format!("{name} requires a non-negative count")));
                }
                let n = (n as usize).min(s.len());
                let out: String = if name == "Left" {
                    s[..n].iter().collect()
                } else {
                    s[s.len() - n..].iter().collect()
                };
                Ok(Value::Text(out))
            }
            "Mid" => {
                arity(name, args, 2, 3)?;
                let s: Vec<char> = self.text_arg(args, 0)?.chars().collect();
                let start = self.num_arg(args, 1)?;
                if start < 1.0 {
                    return Err(runtime("Mid requires a start position of at least 1"));
                }
                let start = (start as usize - 1).min(s.len());
                let count = if args.len() == 3 {
                    let c = self.num_arg(args, 2)?;
                    if c < 0.0 {
                        return Err(runtime("Mid requires a non-negative count"));
                    }
                    c as usize
                } else {
                    s.len()
                };
                let end = start.saturating_add(count).min(s.len());
                Ok(Value::Text(s[start..end].iter().collect()))
            }
            "Find" => {
                arity(name, args, 2, 3)?;
                let needle = self.text_arg(args, 0)?;
                let hay = self.text_arg(args, 1)?;
                let start = if args.len() == 3 {
                    let s = self.num_arg(args, 2)?;
                    if s < 1.0 {
                        return Err(runtime("Find requires a start position of at least 1"));
                    }
                    s as usize - 1
                } else {
                    0
                };
                let chars: Vec<char> = hay.chars().collect();
                if start > chars.len() {
                    return Ok(Value::Blank);
                }
                let tail: String = chars[start..].iter().collect();
                Ok(match tail.find(&needle) {
                    Some(byte) => Value::Number((start + tail[..byte].chars().count() + 1) as f64),
                    None => Value::Blank,
                })
            }
            "StartsWith" | "EndsWith" => {
                arity(name, args, 2, 2)?;
                let s = self.text_arg(args, 0)?.to_lowercase();
                let p = self.text_arg(args, 1)?.to_lowercase();
                Ok(Value::Boolean(if name == "StartsWith" {
                    s.starts_with(&p)
                } else {
                    s.ends_with(&p)
                }))
            }
            "Substitute" => {
                arity(name, args, 3, 4)?;
                let s = self.text_arg(args, 0)?;
                let old = self.text_arg(args, 1)?;
                let new = self.text_arg(args, 2)?;
                if old.is_empty() {
                    return Ok(Value::Text(s));
                }
                if args.len() == 4 {
                    let instance = self.num_arg(args, 3)?;
                    if instance < 1.0 {
                        return Err(runtime("Substitute requires an instance of at least 1"));
                    }
                    let instance = instance as usize;
                    if let Some((byte, _)) = s.match_indices(&old).nth(instance - 1) {
                        let mut out = String::with_capacity(s.len());
                        out.push_str(&s[..byte]);
                        out.push_str(&new);
                        out.push_str(&s[byte + old.len()..]);
                        return Ok(Value::Text(out));
                    }
                    return Ok(Value::Text(s));
                }
                Ok(Value::Text(s.replace(&old, &new)))
            }
            "Replace" => {
                arity(name, args, 4, 4)?;
                let s: Vec<char> = self.text_arg(args, 0)?.chars().collect();
                let start = self.num_arg(args, 1)?;
                let count = self.num_arg(args, 2)?;
                let new = self.text_arg(args, 3)?;
                if start < 1.0 || count < 0.0 {
                    return Err(runtime("Replace requires start >= 1 and count >= 0"));
                }
                let start = (start as usize - 1).min(s.len());
                let end = start.saturating_add(count as usize).min(s.len());
                let mut out: String = s[..start].iter().collect();
                out.push_str(&new);
                out.extend(s[end..].iter());
                Ok(Value::Text(out))
            }
            "Split" => {
                arity(name, args, 2, 2)?;
                let s = self.text_arg(args, 0)?;
                let sep = self.text_arg(args, 1)?;
                let parts: Vec<String> = if sep.is_empty() {
                    s.chars().map(String::from).collect()
                } else {
                    s.split(sep.as_str()).map(str::to_string).collect()
                };
                Ok(Value::Table(
                    parts
                        .into_iter()
                        .map(|p| Value::Text(p).into_row())
                        .collect(),
                ))
            }
            "Char" | "UniChar" => {
                arity(name, args, 1, 1)?;
                let n = self.num_arg(args, 0)?;
                let c = char::from_u32(n as u32)
                    .filter(|_| n >= 0.0 && (name == "UniChar" || n <= 255.0))
                    .ok_or_else(|| runtime(format!("{name}({n}) is not a valid character code")))?;
                Ok(Value::Text(c.to_string()))
            }
            "GUID" => {
                arity(name, args, 0, 1)?;
                if args.is_empty() {
                    return Ok(Value::Text(uuid::Uuid::new_v4().to_string()));
                }
                let s = self.text_arg(args, 0)?;
                uuid::Uuid::parse_str(s.trim())
                    .map(|u| Value::Text(u.to_string()))
                    .map_err(|_| runtime(format!("{s:?} is not a valid GUID")))
            }

            // ---------------------------------------------------- tables
            "Table" => {
                let mut rows = Vec::new();
                for i in 0..args.len() {
                    match self.arg(args, i)? {
                        Value::Record(r) => rows.push(r),
                        Value::Table(t) => rows.extend(t),
                        Value::Blank => {}
                        other => {
                            return Err(type_error(format!(
                                "Table expects records, found a {}",
                                other.type_name()
                            )))
                        }
                    }
                }
                Ok(Value::Table(rows))
            }
            "CountRows" => {
                arity(name, args, 1, 1)?;
                Ok(Value::Number(self.table_arg(name, args, 0)?.len() as f64))
            }
            "CountIf" => {
                arity(name, args, 2, usize::MAX)?;
                let rows = self.table_arg(name, args, 0)?;
                let mut n = 0;
                for row in &rows {
                    if self.row_matches(row, &args[1..])? {
                        n += 1;
                    }
                }
                Ok(Value::Number(n as f64))
            }
            "CountA" | "Count" => {
                arity(name, args, 1, 1)?;
                let rows = self.table_arg(name, args, 0)?;
                let n = rows
                    .iter()
                    .map(row_scalar)
                    .filter(|v| {
                        if name == "Count" {
                            matches!(v, Value::Number(_))
                        } else {
                            !v.is_blank()
                        }
                    })
                    .count();
                Ok(Value::Number(n as f64))
            }
            "First" | "Last" => {
                arity(name, args, 1, 1)?;
                let rows = self.table_arg(name, args, 0)?;
                let row = if name == "First" {
                    rows.into_iter().next()
                } else {
                    rows.into_iter().next_back()
                };
                Ok(row.map(Value::Record).unwrap_or(Value::Blank))
            }
            "FirstN" | "LastN" => {
                arity(name, args, 1, 2)?;
                let rows = self.table_arg(name, args, 0)?;
                let n = if args.len() == 2 {
                    let n = self.num_arg(args, 1)?;
                    if n < 0.0 {
                        return Err(runtime(format!("{name} requires a non-negative count")));
                    }
                    n as usize
                } else {
                    1
                };
                let n = n.min(rows.len());
                Ok(Value::Table(if name == "FirstN" {
                    rows[..n].to_vec()
                } else {
                    rows[rows.len() - n..].to_vec()
                }))
            }
            "Index" => {
                arity(name, args, 2, 2)?;
                let rows = self.table_arg(name, args, 0)?;
                let i = self.num_arg(args, 1)?;
                if i < 1.0 || i as usize > rows.len() {
                    return Err(runtime(format!(
                        "Index {i} is out of range for a table of {} row(s)",
                        rows.len()
                    )));
                }
                Ok(Value::Record(rows[i as usize - 1].clone()))
            }
            "Filter" => {
                arity(name, args, 2, usize::MAX)?;
                let rows = self.table_arg(name, args, 0)?;
                let mut out = Vec::new();
                for row in rows {
                    if self.row_matches(&row, &args[1..])? {
                        out.push(row);
                    }
                }
                Ok(Value::Table(out))
            }
            "LookUp" => {
                arity(name, args, 2, 3)?;
                let rows = self.table_arg(name, args, 0)?;
                for row in &rows {
                    if self.row_matches(row, &args[1..2])? {
                        return if args.len() == 3 {
                            self.with_row(row, &args[2])
                        } else {
                            Ok(Value::Record(row.clone()))
                        };
                    }
                }
                Ok(Value::Blank)
            }
            "Search" => {
                arity(name, args, 3, usize::MAX)?;
                let rows = self.table_arg(name, args, 0)?;
                let needle = self.text_arg(args, 1)?.to_lowercase();
                let columns: Vec<String> = args[2..]
                    .iter()
                    .map(|e| column_name(name, e))
                    .collect::<Result<_, _>>()?;
                if needle.is_empty() {
                    return Ok(Value::Table(rows));
                }
                let mut out = Vec::new();
                for row in rows {
                    let hit = columns.iter().any(|c| {
                        row.get(c)
                            .and_then(|v| to_text(v).ok())
                            .is_some_and(|t| t.to_lowercase().contains(&needle))
                    });
                    if hit {
                        out.push(row);
                    }
                }
                Ok(Value::Table(out))
            }
            "Sum" => {
                let xs = self.aggregate_inputs(name, args)?;
                Ok(Value::Number(xs.iter().sum()))
            }
            "Max" | "Min" => {
                let xs = self.aggregate_inputs(name, args)?;
                let r = xs
                    .into_iter()
                    .reduce(|a, b| if (name == "Max") == (b > a) { b } else { a });
                Ok(r.map(Value::Number).unwrap_or(Value::Blank))
            }
            "Average" => {
                let xs = self.aggregate_inputs(name, args)?;
                if xs.is_empty() {
                    return Err(runtime("Average of an empty set (division by zero)"));
                }
                Ok(Value::Number(xs.iter().sum::<f64>() / xs.len() as f64))
            }
            "Sort" => {
                arity(name, args, 2, 3)?;
                let rows = self.table_arg(name, args, 0)?;
                let descending = if args.len() == 3 {
                    match self.text_arg(args, 2)?.to_ascii_lowercase().as_str() {
                        "ascending" => false,
                        "descending" => true,
                        other => {
                            return Err(runtime(format!(
                                "Sort order must be Ascending or Descending, found {other:?}"
                            )))
                        }
                    }
                } else {
                    false
                };
                let mut keyed = Vec::with_capacity(rows.len());
                for row in rows {
                    let key = self.with_row(&row, &args[1])?;
                    keyed.push((key, row));
                }
                let mut err = None;
                keyed.sort_by(|a, b| match compare_sort(&a.0, &b.0) {
                    Ok(o) => {
                        if descending {
                            o.reverse()
                        } else {
                            o
                        }
                    }
                    Err(e) => {
                        err.get_or_insert(e);
                        Ordering::Equal
                    }
                });
                if let Some(e) = err {
                    return Err(e);
                }
                Ok(Value::Table(keyed.into_iter().map(|(_, r)| r).collect()))
            }
            "Distinct" => {
                arity(name, args, 2, 2)?;
                let rows = self.table_arg(name, args, 0)?;
                let mut seen: Vec<Value> = Vec::new();
                for row in &rows {
                    let v = self.with_row(row, &args[1])?;
                    if !seen.contains(&v) {
                        seen.push(v);
                    }
                }
                Ok(Value::Table(
                    seen.into_iter().map(Value::into_row).collect(),
                ))
            }
            "ForAll" => {
                arity(name, args, 2, 2)?;
                let rows = self.table_arg(name, args, 0)?;
                let mut out = Vec::with_capacity(rows.len());
                for row in &rows {
                    out.push(self.with_row(row, &args[1])?.into_row());
                }
                Ok(Value::Table(out))
            }
            "AddColumns" => {
                arity(name, args, 3, usize::MAX)?;
                if args.len().is_multiple_of(2) {
                    return Err(PowerFxError::new(
                        PowerFxErrorKind::InvalidArguments,
                        "AddColumns expects column name / expression pairs",
                    ));
                }
                let rows = self.table_arg(name, args, 0)?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let mut new_row = row.clone();
                    for pair in args[1..].chunks(2) {
                        let col = column_name(name, &pair[0])?;
                        let v = self.with_row(&row, &pair[1])?;
                        new_row.insert(col, v);
                    }
                    out.push(new_row);
                }
                Ok(Value::Table(out))
            }
            "ShowColumns" | "DropColumns" => {
                arity(name, args, 2, usize::MAX)?;
                let rows = self.table_arg(name, args, 0)?;
                let cols: Vec<String> = args[1..]
                    .iter()
                    .map(|e| column_name(name, e))
                    .collect::<Result<_, _>>()?;
                let out = rows
                    .into_iter()
                    .map(|row| {
                        if name == "ShowColumns" {
                            cols.iter()
                                .map(|c| (c.clone(), row.get(c).cloned().unwrap_or(Value::Blank)))
                                .collect()
                        } else {
                            let mut r = row;
                            for c in &cols {
                                r.remove(c);
                            }
                            r
                        }
                    })
                    .collect();
                Ok(Value::Table(out))
            }
            "RenameColumns" => {
                arity(name, args, 3, usize::MAX)?;
                if args.len().is_multiple_of(2) {
                    return Err(PowerFxError::new(
                        PowerFxErrorKind::InvalidArguments,
                        "RenameColumns expects old/new column name pairs",
                    ));
                }
                let rows = self.table_arg(name, args, 0)?;
                let mut pairs = Vec::new();
                for pair in args[1..].chunks(2) {
                    pairs.push((column_name(name, &pair[0])?, column_name(name, &pair[1])?));
                }
                let out = rows
                    .into_iter()
                    .map(|row| {
                        row.iter()
                            .map(|(k, v)| {
                                let k = pairs
                                    .iter()
                                    .find(|(old, _)| old == k)
                                    .map(|(_, new)| new.clone())
                                    .unwrap_or_else(|| k.to_string());
                                (k, v.clone())
                            })
                            .collect()
                    })
                    .collect();
                Ok(Value::Table(out))
            }
            "Sequence" => {
                arity(name, args, 1, 3)?;
                let n = self.num_arg(args, 0)?;
                if !(0.0..=50_000.0).contains(&n) {
                    return Err(runtime("Sequence count must be between 0 and 50000"));
                }
                let start = if args.len() > 1 {
                    self.num_arg(args, 1)?
                } else {
                    1.0
                };
                let step = if args.len() > 2 {
                    self.num_arg(args, 2)?
                } else {
                    1.0
                };
                Ok(Value::Table(
                    (0..n as usize)
                        .map(|i| Value::Number(start + step * i as f64).into_row())
                        .collect(),
                ))
            }

            // ---------------------------------------------------- math
            "Round" | "RoundUp" | "RoundDown" => {
                arity(name, args, 2, 2)?;
                let n = self.num_arg(args, 0)?;
                let d = self.num_arg(args, 1)?.trunc();
                let mode = match name {
                    "Round" => Rounding::HalfAwayFromZero,
                    "RoundUp" => Rounding::Up,
                    _ => Rounding::Down,
                };
                Ok(Value::Number(round_to(n, d, mode)))
            }
            "Int" => {
                arity(name, args, 1, 1)?;
                Ok(Value::Number(self.num_arg(args, 0)?.floor()))
            }
            "Trunc" => {
                arity(name, args, 1, 2)?;
                let n = self.num_arg(args, 0)?;
                let d = if args.len() == 2 {
                    self.num_arg(args, 1)?.trunc()
                } else {
                    0.0
                };
                Ok(Value::Number(round_to(n, d, Rounding::Down)))
            }
            "Abs" => {
                arity(name, args, 1, 1)?;
                Ok(Value::Number(self.num_arg(args, 0)?.abs()))
            }
            "Mod" => {
                arity(name, args, 2, 2)?;
                let n = self.num_arg(args, 0)?;
                let d = self.num_arg(args, 1)?;
                if d == 0.0 {
                    return Err(runtime("Mod: division by zero"));
                }
                Ok(Value::Number(n - d * (n / d).floor()))
            }
            "Power" => {
                arity(name, args, 2, 2)?;
                let v = self.num_arg(args, 0)?.powf(self.num_arg(args, 1)?);
                if !v.is_finite() {
                    return Err(runtime("Power: the result is not a finite number"));
                }
                Ok(Value::Number(v))
            }
            "Sqrt" => {
                arity(name, args, 1, 1)?;
                let n = self.num_arg(args, 0)?;
                if n < 0.0 {
                    return Err(runtime("Sqrt of a negative number"));
                }
                Ok(Value::Number(n.sqrt()))
            }

            // ---------------------------------------------------- JSON
            "ParseJSON" => {
                arity(name, args, 1, 1)?;
                let s = self.text_arg(args, 0)?;
                if s.trim().is_empty() {
                    return Ok(Value::Blank);
                }
                let json: serde_json::Value = serde_json::from_str(&s)
                    .map_err(|e| runtime(format!("ParseJSON: invalid JSON: {e}")))?;
                Ok(Value::from_json(&json))
            }
            "JSON" => {
                arity(name, args, 1, 2)?;
                let v = self.arg(args, 0)?;
                let indent = if args.len() == 2 {
                    let f = match &args[1] {
                        Expr::Member(_, field) => field.clone(),
                        _ => self.text_arg(args, 1)?,
                    };
                    f.contains("IndentFour")
                } else {
                    false
                };
                let json = v.to_json();
                let text = if indent {
                    serde_json::to_string_pretty(&json)
                } else {
                    serde_json::to_string(&json)
                }
                .map_err(|e| runtime(format!("JSON: {e}")))?;
                Ok(Value::Text(text))
            }

            // ---------------------------------------------------- custom
            "MessageText" => {
                arity(name, args, 1, 1)?;
                let v = self.arg(args, 0)?;
                Ok(Value::Text(match v {
                    Value::Blank => String::new(),
                    Value::Text(s) => s,
                    Value::Record(r) => message_record_text(&r),
                    Value::Table(rows) => match rows.last() {
                        None => String::new(),
                        Some(row) => match row_scalar(row) {
                            Value::Record(r) => message_record_text(&r),
                            other => to_text(&other).unwrap_or_default(),
                        },
                    },
                    other => to_text(&other)?,
                }))
            }
            "UserMessage" | "AgentMessage" | "AssistantMessage" | "SystemMessage" => {
                arity(name, args, 1, 1)?;
                let text = self.text_arg(args, 0)?;
                let role = match name {
                    "UserMessage" => "user",
                    "SystemMessage" => "system",
                    _ => "assistant",
                };
                let mut r = Record::new();
                r.insert("role", Value::Text(role.into()));
                r.insert("text", Value::Text(text));
                Ok(Value::Record(r))
            }

            _ => Err(PowerFxError::new(
                PowerFxErrorKind::UnknownFunction,
                format!("'{name}' is an unknown or unsupported function."),
            )),
        }
    }

    fn row_matches(&mut self, row: &Record, conditions: &[Expr]) -> Result<bool, PowerFxError> {
        for c in conditions {
            let v = self.with_row(row, c)?;
            if !to_bool(&v)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
