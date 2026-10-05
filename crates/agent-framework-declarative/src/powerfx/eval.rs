//! Tree-walking evaluator.

use super::parser::{BinaryOp, Expr, InterpPart, UnaryOp};
use super::value::{row_scalar, to_bool, to_number, to_text, type_error, Record, Value};
use super::{PowerFxError, PowerFxErrorKind, Symbols};

/// Evaluates an [`Expr`] against a symbol table, with a stack of record
/// scopes for row-scoped functions (`Filter`, `ForAll`, `LookUp`, …).
pub(crate) struct Evaluator<'a> {
    symbols: &'a dyn Symbols,
    scopes: Vec<Record>,
    depth: usize,
    max_depth: usize,
}

impl<'a> Evaluator<'a> {
    pub(crate) fn new(symbols: &'a dyn Symbols, max_depth: usize) -> Self {
        Self {
            symbols,
            scopes: Vec::new(),
            depth: 0,
            max_depth,
        }
    }

    /// Evaluate `expr` with `row` pushed as the innermost record scope.
    pub(crate) fn with_row(&mut self, row: &Record, expr: &Expr) -> Result<Value, PowerFxError> {
        self.scopes.push(row.clone());
        let result = self.eval(expr);
        self.scopes.pop();
        result
    }

    pub(crate) fn eval(&mut self, expr: &Expr) -> Result<Value, PowerFxError> {
        self.depth += 1;
        if self.depth > self.max_depth.saturating_mul(4) {
            self.depth -= 1;
            return Err(PowerFxError::new(
                PowerFxErrorKind::Limit,
                format!(
                    "evaluation exceeds the maximum call depth of {}",
                    self.max_depth
                ),
            ));
        }
        let result = self.eval_inner(expr);
        self.depth -= 1;
        result
    }

    fn eval_inner(&mut self, expr: &Expr) -> Result<Value, PowerFxError> {
        match expr {
            Expr::Number(n) => Ok(Value::Number(*n)),
            Expr::Text(s) => Ok(Value::Text(s.clone())),
            Expr::Bool(b) => Ok(Value::Boolean(*b)),
            Expr::Interp(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        InterpPart::Text(t) => out.push_str(t),
                        InterpPart::Expr(e) => {
                            let v = self.eval(e)?;
                            out.push_str(&to_text(&v)?);
                        }
                    }
                }
                Ok(Value::Text(out))
            }
            Expr::Name(name) => self.resolve(name),
            Expr::ThisRecord => match self.scopes.last() {
                Some(r) => Ok(Value::Record(r.clone())),
                None => Err(unknown_name("ThisRecord")),
            },
            Expr::Member(base, field) => {
                let base = self.eval(base)?;
                member(&base, field)
            }
            Expr::Call(name, args) => self.call_function(name, args),
            Expr::Unary(op, operand) => {
                let v = self.eval(operand)?;
                match op {
                    UnaryOp::Neg => Ok(Value::Number(-to_number(&v)?)),
                    UnaryOp::Not => Ok(Value::Boolean(!to_bool(&v)?)),
                    UnaryOp::Percent => Ok(Value::Number(to_number(&v)? / 100.0)),
                }
            }
            Expr::Binary(op, left, right) => self.binary(*op, left, right),
            Expr::Record(fields) => {
                let mut r = Record::new();
                for (name, e) in fields {
                    let v = self.eval(e)?;
                    r.insert(name.clone(), v);
                }
                Ok(Value::Record(r))
            }
            Expr::Table(items) => {
                let mut rows = Vec::with_capacity(items.len());
                for e in items {
                    rows.push(self.eval(e)?.into_row());
                }
                Ok(Value::Table(rows))
            }
        }
    }

    fn resolve(&self, name: &str) -> Result<Value, PowerFxError> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Ok(v.clone());
            }
        }
        if let Some(v) = self.symbols.lookup(name) {
            return Ok(v);
        }
        if name == "SortOrder" {
            let mut r = Record::new();
            r.insert("Ascending", Value::Text("Ascending".into()));
            r.insert("Descending", Value::Text("Descending".into()));
            return Ok(Value::Record(r));
        }
        Err(unknown_name(name))
    }

    fn binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr) -> Result<Value, PowerFxError> {
        match op {
            BinaryOp::Or => {
                let l = self.eval(left)?;
                if to_bool(&l)? {
                    return Ok(Value::Boolean(true));
                }
                let r = self.eval(right)?;
                Ok(Value::Boolean(to_bool(&r)?))
            }
            BinaryOp::And => {
                let l = self.eval(left)?;
                if !to_bool(&l)? {
                    return Ok(Value::Boolean(false));
                }
                let r = self.eval(right)?;
                Ok(Value::Boolean(to_bool(&r)?))
            }
            _ => {
                let l = self.eval(left)?;
                let r = self.eval(right)?;
                apply_binary(op, &l, &r)
            }
        }
    }
}

pub(crate) fn unknown_name(name: &str) -> PowerFxError {
    PowerFxError::new(
        PowerFxErrorKind::UnknownName,
        format!("Name isn't valid. '{name}' isn't recognized."),
    )
}

/// Field access. A missing field (or a field of `Blank`) is `Blank`; a field
/// of a table projects that column into a single-column table.
pub(crate) fn member(base: &Value, field: &str) -> Result<Value, PowerFxError> {
    match base {
        Value::Record(r) => Ok(r.get(field).cloned().unwrap_or(Value::Blank)),
        Value::Blank => Ok(Value::Blank),
        Value::Table(rows) => Ok(Value::Table(
            rows.iter()
                .map(|row| row.get(field).cloned().unwrap_or(Value::Blank).into_row())
                .collect(),
        )),
        other => Err(type_error(format!(
            "cannot access field '{field}' of a {}",
            other.type_name()
        ))),
    }
}

fn apply_binary(op: BinaryOp, l: &Value, r: &Value) -> Result<Value, PowerFxError> {
    Ok(match op {
        BinaryOp::Eq => Value::Boolean(values_equal(l, r)?),
        BinaryOp::Ne => Value::Boolean(!values_equal(l, r)?),
        BinaryOp::Lt => Value::Boolean(compare_order(l, r)? == std::cmp::Ordering::Less),
        BinaryOp::Le => Value::Boolean(compare_order(l, r)? != std::cmp::Ordering::Greater),
        BinaryOp::Gt => Value::Boolean(compare_order(l, r)? == std::cmp::Ordering::Greater),
        BinaryOp::Ge => Value::Boolean(compare_order(l, r)? != std::cmp::Ordering::Less),
        BinaryOp::Concat => Value::Text(format!("{}{}", to_text(l)?, to_text(r)?)),
        BinaryOp::Add => Value::Number(to_number(l)? + to_number(r)?),
        BinaryOp::Sub => Value::Number(to_number(l)? - to_number(r)?),
        BinaryOp::Mul => Value::Number(to_number(l)? * to_number(r)?),
        BinaryOp::Div => {
            let d = to_number(r)?;
            if d == 0.0 {
                return Err(PowerFxError::new(
                    PowerFxErrorKind::Runtime,
                    "division by zero",
                ));
            }
            Value::Number(to_number(l)? / d)
        }
        BinaryOp::Pow => {
            let v = to_number(l)?.powf(to_number(r)?);
            if !v.is_finite() {
                return Err(PowerFxError::new(
                    PowerFxErrorKind::Runtime,
                    "the result of '^' is not a finite number",
                ));
            }
            Value::Number(v)
        }
        BinaryOp::In => Value::Boolean(contains(l, r, false)?),
        BinaryOp::ExactIn => Value::Boolean(contains(l, r, true)?),
        BinaryOp::Or | BinaryOp::And => unreachable!("short-circuit operators handled above"),
    })
}

/// PowerFx `=` semantics: same-typed scalars compare by value; `Blank()`
/// equals `Blank()` and the empty string; comparing other mismatched types
/// (e.g. a number with text) is an error, as it is in PowerFx.
pub(crate) fn values_equal(l: &Value, r: &Value) -> Result<bool, PowerFxError> {
    Ok(match (l, r) {
        (Value::Blank, Value::Blank) => true,
        (Value::Blank, Value::Text(s)) | (Value::Text(s), Value::Blank) => s.is_empty(),
        (Value::Blank, _) | (_, Value::Blank) => false,
        (Value::Number(a), Value::Number(b)) => a == b,
        (Value::Text(a), Value::Text(b)) => a == b,
        (Value::Boolean(a), Value::Boolean(b)) => a == b,
        (a, b) => {
            return Err(type_error(format!(
                "incompatible types for comparison: {} and {}",
                a.type_name(),
                b.type_name()
            )))
        }
    })
}

/// Ordering for `<`, `<=`, `>`, `>=`: numeric (Blank counts as 0, numeric
/// text is coerced).
fn compare_order(l: &Value, r: &Value) -> Result<std::cmp::Ordering, PowerFxError> {
    let a = to_number(l)?;
    let b = to_number(r)?;
    a.partial_cmp(&b)
        .ok_or_else(|| PowerFxError::new(PowerFxErrorKind::Runtime, "cannot compare NaN"))
}

/// `in` / `exactin`: substring test for text, membership test for tables.
fn contains(needle: &Value, haystack: &Value, exact: bool) -> Result<bool, PowerFxError> {
    match haystack {
        Value::Table(rows) => {
            for row in rows {
                let candidate = row_scalar(row);
                let hit = match (needle, &candidate) {
                    (Value::Text(a), Value::Text(b)) if !exact => {
                        a.to_lowercase() == b.to_lowercase()
                    }
                    (Value::Record(a), Value::Record(b)) => a == b,
                    (a, b) => values_equal(a, b).unwrap_or(false),
                };
                if hit {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Value::Blank | Value::Text(_) | Value::Number(_) | Value::Boolean(_) => {
            let n = to_text(needle)?;
            let h = to_text(haystack)?;
            Ok(if exact {
                h.contains(&n)
            } else {
                h.to_lowercase().contains(&n.to_lowercase())
            })
        }
        Value::Record(_) => Err(type_error(
            "the right operand of 'in' must be text or a table",
        )),
    }
}
