//! A PowerFx expression interpreter for declarative workflows.
//!
//! Upstream evaluates `=`-prefixed workflow expressions with Microsoft's
//! PowerFx engine (the .NET `RecalcEngine`, reached from Python through the
//! `powerfx` package). There is no Rust PowerFx implementation, so this module
//! provides an interpreter for the subset declarative workflows, upstream
//! samples, and upstream tests use. Formulas are passed **without** the
//! leading `=`.
//!
//! # Supported language
//!
//! * **Literals** — numbers (`42`, `3.14`, `1e3`), text (`"say ""hi"""`; the
//!   only escape is a doubled quote, as in PowerFx), `true`/`false`, records
//!   (`{name: "a", n: 1}`), tables (`[1, 2]`, `[{a: 1}]`), interpolated text
//!   (`$"Hello {Local.name}"`, `{{`/`}}` for literal braces).
//! * **Names** — dotted paths (`Local.x`, `System.LastMessage.Text`,
//!   `Workflow.Inputs.q`), single-quoted identifiers (`Local.'my var'`),
//!   `ThisRecord`, the `SortOrder` enum, and row-scope column names inside
//!   `Filter`/`ForAll`/`LookUp`/….
//! * **Operators** — `+ - * / ^`, postfix `%`, unary `-`, `&` (text concat),
//!   `= <> < <= > >=`, `And`/`&&`, `Or`/`||`, `Not`/`!`, `in` (case-insensitive)
//!   and `exactin`, with PowerFx precedence.
//! * **Comments** — `//` and `/* */`.
//! * **Functions** — see [`SUPPORTED_FUNCTIONS`]: logical (`If`, `Switch`,
//!   `And`, `Or`, `Not`, `IsBlank`, `IsEmpty`, `Coalesce`, `Blank`, `IsError`,
//!   `IfError`, `Boolean`), text (`Text` with numeric format strings, `Value`,
//!   `Concatenate`, `Concat`, `Len`, `Lower`, `Upper`, `Proper`, `Trim`,
//!   `TrimEnds`, `Left`, `Right`, `Mid`, `Find`, `StartsWith`, `EndsWith`,
//!   `Substitute`, `Replace`, `Split`, `Char`, `UniChar`, `GUID`), tables
//!   (`Table`, `CountRows`, `CountIf`, `CountA`, `Count`, `First`, `Last`,
//!   `FirstN`, `LastN`, `Index`, `Filter`, `LookUp`, `Search`, `Sum`, `Max`,
//!   `Min`, `Average`, `Sort`, `Distinct`, `ForAll`, `AddColumns`,
//!   `ShowColumns`, `DropColumns`, `RenameColumns`, `Sequence`), math
//!   (`Round`, `RoundUp`, `RoundDown`, `Int`, `Trunc`, `Abs`, `Mod`, `Power`,
//!   `Sqrt`), JSON (`ParseJSON`, `JSON`), and the agent-framework custom
//!   functions `MessageText`, `UserMessage`, `AgentMessage`,
//!   `AssistantMessage`, `SystemMessage`.
//!
//! Calling any other function is a [`PowerFxErrorKind::UnknownFunction`]
//! error — never a silent fallback.
//!
//! # Not supported
//!
//! Date/time (`Now`, `Today`, `DateAdd`, `DateValue`, …; nothing upstream
//! uses them), regular expressions (`IsMatch`, `Match`), `With`, `Patch`,
//! `Set`/`UpdateContext` (behaviour functions), `As`, `Parent`, `;` chaining,
//! untyped-object casts, and locale-specific parsing (en-US only, as upstream
//! pins).
//!
//! # Divergences from PowerFx
//!
//! * PowerFx is statically typed; this interpreter is dynamic. Reading a
//!   **missing record field** yields `Blank()` instead of a compile error
//!   (upstream Python turns that compile error into `None` for the *whole*
//!   expression, so `If(IsBlank(inputs.name), "World", inputs.name)` with no
//!   `name` input yields `None` there but `"World"` here — the documented
//!   intent of the upstream samples). Unknown *root* names still raise
//!   [`PowerFxErrorKind::UnknownName`], which the workflow state maps to
//!   `null` exactly like upstream.
//! * Numbers are `f64` rather than decimal; rendering rounds to 15
//!   significant digits so results print like PowerFx's decimals.
//! * `Concat` whose first argument is not a table concatenates all its
//!   arguments as text — the Copilot Studio dialect upstream Python accepts.
//! * `MessageText` accepts text (returned as-is), a message record, or a table
//!   of messages (the **last** message's text, as upstream Python does).
//! * `UserMessage`/`AgentMessage`/… return `{role, text}` records, the shape
//!   upstream Python's evaluator produces (an empty argument yields
//!   `text: ""`, not `Blank()` as in .NET).

mod eval;
mod functions;
mod lexer;
pub mod limits;
mod parser;
mod value;

use std::collections::HashMap;

pub use functions::SUPPORTED_FUNCTIONS;
pub use value::{format_number, Record, Value};

/// The category of a [`PowerFxError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerFxErrorKind {
    /// The formula could not be tokenized or parsed.
    Syntax,
    /// A root name is not defined (`Name isn't valid. 'X' isn't recognized.`).
    UnknownName,
    /// The function does not exist in the supported library.
    UnknownFunction,
    /// A function was called with the wrong number or shape of arguments.
    InvalidArguments,
    /// A value had the wrong type for an operation.
    Type,
    /// A runtime failure (division by zero, invalid conversion, …).
    Runtime,
    /// An expression length, nesting, or call-depth limit was exceeded.
    Limit,
}

/// An error raised while parsing or evaluating a PowerFx formula.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("PowerFx error: {message}")]
pub struct PowerFxError {
    kind: PowerFxErrorKind,
    message: String,
}

impl PowerFxError {
    pub(crate) fn new(kind: PowerFxErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// The error category.
    pub fn kind(&self) -> PowerFxErrorKind {
        self.kind
    }

    /// The human-readable message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether `IfError`/`IsError` may intercept this error (runtime and type
    /// errors; PowerFx compile-time errors are never catchable).
    pub(crate) fn is_catchable(&self) -> bool {
        matches!(
            self.kind,
            PowerFxErrorKind::Runtime | PowerFxErrorKind::Type
        )
    }
}

/// Limits on formula size and nesting.
///
/// `max_expression_length` defaults to 10 000 characters — the
/// `DefaultMaximumExpressionLength` .NET's declarative runtime passes to
/// PowerFx. `max_depth` bounds parse nesting and evaluation call depth
/// (PowerFx's `MaxCallDepth`); 64 is this port's default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpressionLimits {
    /// Maximum formula length in characters.
    pub max_expression_length: usize,
    /// Maximum syntactic nesting / call depth.
    pub max_depth: usize,
}

impl Default for ExpressionLimits {
    fn default() -> Self {
        Self {
            max_expression_length: 10_000,
            max_depth: 64,
        }
    }
}

/// Resolves root names (`Local`, `System`, `Workflow`, …) to values.
pub trait Symbols {
    /// Return the value bound to `name`, or `None` if it is unknown.
    fn lookup(&self, name: &str) -> Option<Value>;
}

impl Symbols for HashMap<String, Value> {
    fn lookup(&self, name: &str) -> Option<Value> {
        self.get(name).cloned()
    }
}

/// A reusable PowerFx engine configured with [`ExpressionLimits`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Engine {
    limits: ExpressionLimits,
}

impl Engine {
    /// An engine with default limits.
    pub fn new() -> Self {
        Self::default()
    }

    /// An engine with custom limits.
    pub fn with_limits(limits: ExpressionLimits) -> Self {
        Self { limits }
    }

    /// The configured limits.
    pub fn limits(&self) -> ExpressionLimits {
        self.limits
    }

    /// Check that `formula` parses, without evaluating it.
    pub fn check(&self, formula: &str) -> Result<(), PowerFxError> {
        parser::parse(formula, &self.limits).map(|_| ())
    }

    /// Parse and evaluate `formula` (without the leading `=`).
    pub fn eval(&self, formula: &str, symbols: &dyn Symbols) -> Result<Value, PowerFxError> {
        let expr = parser::parse(formula, &self.limits)?;
        eval::Evaluator::new(symbols, self.limits.max_depth).eval(&expr)
    }

    /// Evaluate and convert the result to JSON (see [`Value::to_json`]).
    pub fn eval_json(
        &self,
        formula: &str,
        symbols: &dyn Symbols,
    ) -> Result<serde_json::Value, PowerFxError> {
        self.eval(formula, symbols).map(|v| v.to_json())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn symbols(local: serde_json::Value) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("Local".to_string(), Value::from_json(&local));
        m
    }

    fn ev(formula: &str, local: serde_json::Value) -> serde_json::Value {
        Engine::new()
            .eval_json(formula, &symbols(local))
            .unwrap_or_else(|e| panic!("{formula}: {e}"))
    }

    fn err(formula: &str) -> PowerFxErrorKind {
        Engine::new()
            .eval(formula, &symbols(json!({})))
            .unwrap_err()
            .kind()
    }

    #[test]
    fn arithmetic_and_precedence() {
        assert_eq!(ev("1 + 2 * 3", json!({})), json!(7));
        assert_eq!(ev("(1 + 2) * 3", json!({})), json!(9));
        assert_eq!(ev("10 / 4", json!({})), json!(2.5));
        assert_eq!(ev("2 ^ 3 ^ 2", json!({})), json!(64));
        assert_eq!(ev("-2 ^ 2", json!({})), json!(-4));
        assert_eq!(ev("50%", json!({})), json!(0.5));
        assert_eq!(ev("0.1 + 0.2", json!({})), json!(0.3));
        assert_eq!(ev("Local.x + 1", json!({"x": 3})), json!(4));
        assert_eq!(ev("Local.missing + 1", json!({})), json!(1));
        assert_eq!(ev("\"5\" + 1", json!({})), json!(6));
    }

    #[test]
    fn comparisons_and_logic() {
        assert_eq!(ev("Local.x < 4", json!({"x": 2})), json!(true));
        assert_eq!(ev("Local.x >= 4", json!({"x": 2})), json!(false));
        assert_eq!(ev("Local.s = \"a\"", json!({"s": "a"})), json!(true));
        assert_eq!(ev("Local.s <> \"a\"", json!({"s": "a"})), json!(false));
        assert_eq!(ev("true And false", json!({})), json!(false));
        assert_eq!(ev("true && Not false", json!({})), json!(true));
        assert_eq!(ev("false Or true", json!({})), json!(true));
        assert_eq!(ev("false || false", json!({})), json!(false));
        assert_eq!(ev("!(\"a\" = \"b\")", json!({})), json!(true));
        assert_eq!(ev("Blank() = \"\"", json!({})), json!(true));
        assert_eq!(ev("Blank() = 0", json!({})), json!(false));
        // Short-circuit: the right side would fail.
        assert_eq!(ev("false And (1/0 = 1)", json!({})), json!(false));
        assert_eq!(err("1 = \"1\""), PowerFxErrorKind::Type);
    }

    #[test]
    fn in_and_exactin() {
        assert_eq!(ev("\"THE\" in \"The keyboard\"", json!({})), json!(true));
        assert_eq!(
            ev(
                "\"WINDOWS\" exactin \"To display windows in the Windows operating system\"",
                json!({})
            ),
            json!(false)
        );
        assert_eq!(
            ev(
                "\"Windows\" exactin \"To display windows in the Windows operating system\"",
                json!({})
            ),
            json!(true)
        );
        assert_eq!(ev("\"b\" in [\"a\", \"B\"]", json!({})), json!(true));
        assert_eq!(ev("\"b\" exactin [\"a\", \"B\"]", json!({})), json!(false));
    }

    #[test]
    fn text_functions() {
        assert_eq!(
            ev("Concat(\"a\", Local.n, \"!\")", json!({"n": "b"})),
            json!("ab!")
        );
        assert_eq!(
            ev("Concatenate(\"a\", 1, true)", json!({})),
            json!("a1true")
        );
        assert_eq!(ev("\"x\" & 1.5", json!({})), json!("x1.5"));
        assert_eq!(
            ev("Upper(\"abc\") & Lower(\"DEF\")", json!({})),
            json!("ABCdef")
        );
        assert_eq!(ev("Len(\"héllo\")", json!({})), json!(5));
        assert_eq!(ev("Trim(\"  a   b  \")", json!({})), json!("a b"));
        assert_eq!(ev("TrimEnds(\"  a   b  \")", json!({})), json!("a   b"));
        assert_eq!(
            ev("Left(\"abcdef\", 2) & Right(\"abcdef\", 2)", json!({})),
            json!("abef")
        );
        assert_eq!(ev("Mid(\"abcdef\", 2, 3)", json!({})), json!("bcd"));
        assert_eq!(ev("Find(\"c\", \"abcabc\")", json!({})), json!(3));
        assert_eq!(ev("Find(\"c\", \"abcabc\", 4)", json!({})), json!(6));
        assert_eq!(ev("Find(\"z\", \"abc\")", json!({})), json!(null));
        assert_eq!(ev("StartsWith(\"Hello\", \"he\")", json!({})), json!(true));
        assert_eq!(ev("EndsWith(\"Hello\", \"LO\")", json!({})), json!(true));
        assert_eq!(
            ev("Substitute(\"a-b-c\", \"-\", \"+\")", json!({})),
            json!("a+b+c")
        );
        assert_eq!(
            ev("Substitute(\"a-b-c\", \"-\", \"+\", 2)", json!({})),
            json!("a-b+c")
        );
        assert_eq!(
            ev("Replace(\"abcdef\", 2, 3, \"X\")", json!({})),
            json!("aXef")
        );
        assert_eq!(ev("Split(\"a,b\", \",\")", json!({})), json!(["a", "b"]));
        assert_eq!(
            ev("Proper(\"hello wORLD\")", json!({})),
            json!("Hello World")
        );
        assert_eq!(
            ev("Text(1234.567, \"#,##0.00\")", json!({})),
            json!("1,234.57")
        );
        assert_eq!(ev("Text(5, \"000\")", json!({})), json!("005"));
        assert_eq!(ev("Text(2.5)", json!({})), json!("2.5"));
        assert_eq!(ev("Value(\" 42 \")", json!({})), json!(42));
        assert_eq!(ev("Value(\"\")", json!({})), json!(null));
        assert_eq!(err("Value(\"abc\")"), PowerFxErrorKind::Runtime);
        assert_eq!(ev("Char(65) & UniChar(9731)", json!({})), json!("A☃"));
        assert_eq!(
            ev("$\"Hi {Local.n}, {{x}}\"", json!({"n": "Bo"})),
            json!("Hi Bo, {x}")
        );
    }

    #[test]
    fn logical_functions() {
        assert_eq!(
            ev("If(IsBlank(Local.n), \"W\", Local.n)", json!({})),
            json!("W")
        );
        assert_eq!(
            ev("If(IsBlank(Local.n), \"W\", Local.n)", json!({"n": "A"})),
            json!("A")
        );
        assert_eq!(ev("If(false, 1, false, 2, 3)", json!({})), json!(3));
        assert_eq!(ev("If(false, 1)", json!({})), json!(null));
        assert_eq!(
            ev("Switch(2, 1, \"a\", 2, \"b\", \"c\")", json!({})),
            json!("b")
        );
        assert_eq!(ev("Switch(9, 1, \"a\", \"c\")", json!({})), json!("c"));
        assert_eq!(ev("Coalesce(Blank(), \"\", \"x\")", json!({})), json!("x"));
        assert_eq!(ev("IsBlank(\"\")", json!({})), json!(true));
        assert_eq!(ev("IsEmpty([])", json!({})), json!(true));
        assert_eq!(ev("And(true, 1, \"true\")", json!({})), json!(true));
        assert_eq!(ev("Or(false, 0)", json!({})), json!(false));
        assert_eq!(ev("Not(Local.flag)", json!({"flag": false})), json!(true));
        assert_eq!(ev("IfError(1/0, -1)", json!({})), json!(-1));
        assert_eq!(ev("IfError(5, -1)", json!({})), json!(5));
        assert_eq!(ev("IsError(Value(\"x\"))", json!({})), json!(true));
        assert_eq!(ev("Boolean(\"TRUE\")", json!({})), json!(true));
    }

    #[test]
    fn table_functions() {
        let agents = json!({"agents": [
            {"name": "WeatherAgent", "description": "weather"},
            {"name": "CoderAgent", "description": "code"}
        ]});
        assert_eq!(
            ev(
                "Concat(ForAll(Local.agents, $\"- \" & name & $\": \" & description), Value, \"\n\")",
                agents.clone()
            ),
            json!("- WeatherAgent: weather\n- CoderAgent: code")
        );
        assert_eq!(
            ev("Search(Local.agents, \"coder\", name)", agents.clone()),
            json!([{"name": "CoderAgent", "description": "code"}])
        );
        assert_eq!(
            ev(
                "CountRows(Search(Local.agents, \"zzz\", name))",
                agents.clone()
            ),
            json!(0)
        );
        assert_eq!(
            ev("First(Local.agents).name", agents.clone()),
            json!("WeatherAgent")
        );
        assert_eq!(
            ev("Last(Local.agents).name", agents.clone()),
            json!("CoderAgent")
        );
        assert_eq!(
            ev("Index(Local.agents, 2).description", agents.clone()),
            json!("code")
        );
        assert_eq!(
            ev("Local.agents.name", agents.clone()),
            json!(["WeatherAgent", "CoderAgent"])
        );
        assert_eq!(
            ev(
                "LookUp(Local.agents, name = \"CoderAgent\", description)",
                agents.clone()
            ),
            json!("code")
        );
        assert_eq!(
            ev("Filter([1, 2, 3, 4], Value > 2)", json!({})),
            json!([3, 4])
        );
        assert_eq!(ev("Sum([1, 2, 3], Value)", json!({})), json!(6));
        assert_eq!(ev("Sum(1, 2, 3)", json!({})), json!(6));
        assert_eq!(ev("Max([4, 9, 2], Value)", json!({})), json!(9));
        assert_eq!(ev("Min(4, 9, 2)", json!({})), json!(2));
        assert_eq!(ev("Average([1, 2], Value)", json!({})), json!(1.5));
        assert_eq!(
            ev("Sort([3, 1, 2], Value, SortOrder.Descending)", json!({})),
            json!([3, 2, 1])
        );
        assert_eq!(ev("Distinct([1, 2, 1], Value)", json!({})), json!([1, 2]));
        assert_eq!(ev("FirstN([1, 2, 3], 2)", json!({})), json!([1, 2]));
        assert_eq!(ev("LastN([1, 2, 3], 2)", json!({})), json!([2, 3]));
        assert_eq!(ev("CountIf([1, 2, 3], Value > 1)", json!({})), json!(2));
        assert_eq!(ev("First([1, 2]).Value", json!({})), json!(1));
        assert_eq!(
            ev("AddColumns([{a: 1}], b, a * 2)", json!({})),
            json!([{"a": 1, "b": 2}])
        );
        assert_eq!(
            ev("ShowColumns([{a: 1, b: 2}], a)", json!({})),
            json!([{"a": 1}])
        );
        assert_eq!(
            ev("DropColumns([{a: 1, b: 2}], a)", json!({})),
            json!([{"b": 2}])
        );
        assert_eq!(
            ev("RenameColumns([{a: 1}], a, z)", json!({})),
            json!([{"z": 1}])
        );
        assert_eq!(ev("Sequence(3)", json!({})), json!([1, 2, 3]));
        assert_eq!(
            ev("Table({a: 1}, {a: 2})", json!({})),
            json!([{"a": 1}, {"a": 2}])
        );
        assert_eq!(
            ev("ForAll([1, 2], ThisRecord.Value * 10)", json!({})),
            json!([10, 20])
        );
        assert_eq!(err("Index([1], 5)"), PowerFxErrorKind::Runtime);
    }

    #[test]
    fn math_functions() {
        assert_eq!(ev("Mod(7, 2)", json!({})), json!(1));
        assert_eq!(ev("Mod(-7, 2)", json!({})), json!(1));
        assert_eq!(ev("Round(2.675, 2)", json!({})), json!(2.68));
        assert_eq!(ev("Round(-2.5, 0)", json!({})), json!(-3));
        assert_eq!(ev("RoundUp(2.01, 1)", json!({})), json!(2.1));
        assert_eq!(ev("RoundDown(2.99, 1)", json!({})), json!(2.9));
        assert_eq!(ev("Int(-1.5)", json!({})), json!(-2));
        assert_eq!(ev("Trunc(-1.5)", json!({})), json!(-1));
        assert_eq!(ev("Abs(-3) + Power(2, 3) + Sqrt(16)", json!({})), json!(15));
        assert_eq!(err("1 / 0"), PowerFxErrorKind::Runtime);
        assert_eq!(err("Mod(1, 0)"), PowerFxErrorKind::Runtime);
    }

    #[test]
    fn json_and_custom_functions() {
        assert_eq!(
            ev("ParseJSON(\"{\"\"a\"\": [1]}\").a", json!({})),
            json!([1])
        );
        assert_eq!(
            ev("JSON({a: 1, b: \"x\"})", json!({})),
            json!("{\"a\":1,\"b\":\"x\"}")
        );
        assert_eq!(
            ev("UserMessage(\"hi\")", json!({})),
            json!({"role": "user", "text": "hi"})
        );
        assert_eq!(
            ev("AgentMessage(Local.r)", json!({"r": ""})),
            json!({"role": "assistant", "text": ""})
        );
        assert_eq!(
            ev(
                "MessageText(Local.m)",
                json!({"m": [{"role": "user", "text": "Hello"}, {"role": "assistant", "text": "Hi there!"}]})
            ),
            json!("Hi there!")
        );
        assert_eq!(
            ev(
                "MessageText(Local.m)",
                json!({"m": [{"role": "assistant", "contents": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]})
            ),
            json!("a b")
        );
        assert_eq!(ev("MessageText(Local.m)", json!({"m": []})), json!(""));
        assert_eq!(ev("MessageText(\"plain\")", json!({})), json!("plain"));
        assert_eq!(
            ev(
                "!IsBlank(Find(\"CONGRATULATIONS\", Upper(MessageText(Local.m))))",
                json!({"m": [{"role": "assistant", "text": "congratulations!"}]})
            ),
            json!(true)
        );
    }

    #[test]
    fn records_quoted_names_and_comments() {
        assert_eq!(ev("{id: 7}", json!({})), json!({"id": 7}));
        assert_eq!(ev("[{id: 3}]", json!({})), json!([{"id": 3}]));
        assert_eq!(
            ev("Local.'my var' // trailing", json!({"my var": 5})),
            json!(5)
        );
        assert_eq!(
            ev("/* lead */ Local.a.b.c", json!({"a": {"b": {"c": "d"}}})),
            json!("d")
        );
        assert_eq!(ev("Local.a.missing.deep", json!({"a": {}})), json!(null));
    }

    #[test]
    fn errors_are_classified() {
        assert_eq!(err("Nope.x"), PowerFxErrorKind::UnknownName);
        assert_eq!(err("Frobnicate(1)"), PowerFxErrorKind::UnknownFunction);
        assert_eq!(err("Left(\"a\")"), PowerFxErrorKind::InvalidArguments);
        assert_eq!(err("1 +"), PowerFxErrorKind::Syntax);
        assert_eq!(err("{a: 1} & \"x\""), PowerFxErrorKind::Type);
        assert_eq!(err("Upper(\"a\").x"), PowerFxErrorKind::Type);
    }

    #[test]
    fn evaluation_depth_is_bounded() {
        let engine = Engine::with_limits(ExpressionLimits {
            max_expression_length: 10_000,
            max_depth: 8,
        });
        let nested = format!("{}1{}", "Abs(".repeat(30), ")".repeat(30));
        assert_eq!(
            engine.eval(&nested, &HashMap::new()).unwrap_err().kind(),
            PowerFxErrorKind::Limit
        );
    }
}
