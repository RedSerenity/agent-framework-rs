//! Small shared helpers: session handles, Python-compatible JSON rendering,
//! and closure-backed tool construction.

use std::future::Future;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::SessionContext;
use agent_framework_core::session::{AgentSession, SessionState};
use agent_framework_core::tools::{ApprovalMode, FunctionTool, ToolDefinition};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{Map, Value};

/// The part of an [`AgentSession`] the harness providers use: its id and a
/// by-reference handle onto its [`state`](AgentSession::state) bag.
///
/// Upstream harness stores and helpers take the whole `AgentSession`; in
/// Rust a provider's `before_run` sees only a [`SessionContext`] (which
/// carries the id and, since this crate's core hook, the state handle), so
/// stores and helpers take this lighter handle instead. Build one from either
/// side with [`SessionRef::from_session`] / [`SessionRef::from_context`].
#[derive(Clone, Debug)]
pub struct SessionRef {
    /// The session's local id.
    pub session_id: String,
    /// The session's state bag (shared by reference with the session).
    pub state: SessionState,
}

impl SessionRef {
    /// A handle onto `session`'s id and state.
    pub fn from_session(session: &AgentSession) -> Self {
        Self {
            session_id: session.session_id().to_string(),
            state: session.state.clone(),
        }
    }

    /// A handle onto the session of a provider invocation. Errors when the
    /// context carries no session state — upstream harness providers always
    /// receive an `AgentSession`, and so do providers run by
    /// [`Agent`](agent_framework_core::agent::Agent).
    pub fn from_context(ctx: &SessionContext, provider: &str) -> Result<Self> {
        let state = ctx.session_state.clone().ok_or_else(|| {
            Error::AgentExecution(format!(
                "{provider} requires an AgentSession: run the agent with a session so the provider can keep its state in session.state."
            ))
        })?;
        Ok(Self {
            session_id: ctx.session_id.clone().unwrap_or_default(),
            state,
        })
    }
}

/// The Python type name of a JSON value, for error messages.
pub(crate) fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_i64() || n.is_u64() => "int",
        Value::Number(_) => "float",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// A `serde_json` formatter reproducing Python's `json.dumps` defaults:
/// `", "` / `": "` separators and, optionally, `ensure_ascii` escaping.
struct PyFormatter {
    ensure_ascii: bool,
}

impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
    fn write_string_fragment<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        if !self.ensure_ascii || fragment.is_ascii() {
            return writer.write_all(fragment.as_bytes());
        }
        for c in fragment.chars() {
            if c.is_ascii() {
                writer.write_all(&[c as u8])?;
            } else {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
}

/// Render `value` exactly as Python's `json.dumps(value, ensure_ascii=…)`
/// would (default separators). Struct fields keep declaration order, which
/// is how the harness reproduces upstream's dict insertion order in tool
/// results.
pub fn py_json_dumps<T: Serialize + ?Sized>(value: &T, ensure_ascii: bool) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFormatter { ensure_ascii });
    // Serializing plain data into a Vec cannot fail.
    let _ = value.serialize(&mut ser);
    String::from_utf8(out).unwrap_or_default()
}

/// Parse a tool's JSON arguments into `T`, with the same error shape as
/// [`FunctionTool::typed`].
pub(crate) fn parse_args<T: DeserializeOwned>(tool: &str, args: Value) -> Result<T> {
    let args = if args.is_null() {
        Value::Object(Map::new())
    } else {
        args
    };
    serde_json::from_value(args)
        .map_err(|e| Error::tool(format!("invalid arguments for tool '{tool}': {e}")))
}

/// Build a closure-backed function tool with a hand-written schema.
pub(crate) fn function_tool<F, Fut>(
    name: &str,
    description: &str,
    parameters: Value,
    approval_mode: ApprovalMode,
    f: F,
) -> ToolDefinition
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Value>> + Send + 'static,
{
    FunctionTool::new(name, description, parameters, f)
        .with_approval_mode(approval_mode)
        .into_definition()
}

/// The `{"type": "object", "properties": {}}` schema of a no-argument tool.
pub(crate) fn empty_object_schema() -> Value {
    serde_json::json!({"type": "object", "properties": {}})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Item {
        id: i64,
        title: String,
        description: Option<String>,
        is_complete: bool,
    }

    #[test]
    fn py_json_dumps_matches_python_defaults() {
        const BS: char = '\\';
        let item = Item {
            id: 1,
            title: "café".into(),
            description: None,
            is_complete: false,
        };
        assert_eq!(
            py_json_dumps(&vec![item], true),
            format!(
                r#"[{{"id": 1, "title": "caf{bs}u00e9", "description": null, "is_complete": false}}]"#,
                bs = BS
            )
        );
        assert_eq!(
            py_json_dumps(&serde_json::json!({"a": "é"}), false),
            r#"{"a": "é"}"#
        );
        assert_eq!(
            py_json_dumps(&"😀", true),
            format!(r#""{bs}ud83d{bs}ude00""#, bs = BS)
        );
    }
}
