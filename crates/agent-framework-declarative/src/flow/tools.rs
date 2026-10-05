//! `InvokeFunctionTool` (port of `_executors_tools.py`).
//!
//! Functions are resolved by `functionName` (which may be an `=` expression)
//! from the tools registered on the
//! [`WorkflowFactory`](super::WorkflowFactory) — any core
//! [`Tool`](agent_framework_core::tools::Tool) — and invoked with the
//! evaluated `arguments` object.
//!
//! ```yaml
//! - kind: InvokeFunctionTool
//!   functionName: get_weather
//!   requireApproval: true          # optional
//!   arguments: { location: =Local.location, unit: F }
//!   output:
//!     messages: Local.toolCallItems   # [assistant function_call, tool function_result]
//!     result: Local.weather
//!     autoSend: true                  # default true; may be an = expression
//! ```
//!
//! With `requireApproval`, the run pauses with a
//! `{"type": "ToolApprovalRequest", request_id, function_name, arguments}`
//! request; answer with `{"approved": true|false, "reason": "..."}` (or a
//! bare Boolean). `approved` must be a Boolean — anything else fails the
//! action, as upstream's `ToolApprovalResponse` rejects it. The approved call
//! uses the reviewed `function_name`/`arguments` from the request.
//!
//! Errors from the tool (and a missing function) are stored as
//! `{"error": "..."}` at `output.result` rather than failing the workflow;
//! rejections store `{"approved": false, "rejected": true, "reason": ...}`.
//!
//! Divergence: auto-sent results render with Python `str()` semantics for
//! scalars but as compact JSON for objects/arrays (upstream prints Python
//! `repr` of dicts).

use agent_framework_core::error::Error as CoreError;
use agent_framework_core::types::{
    Content, FunctionArguments, FunctionCallContent, FunctionResultContent, Message, Role,
};
use agent_framework_core::workflow::{RequestResponse, WorkflowContext};
use serde_json::{json, Map, Value as Json};

use super::agents::normalize_variable_path;
use super::executor::{field, ActionResult, DeclarativeExecutor};
use super::messages::{action_complete, message_to_json};
use super::state::{py_str, py_truthy, DeclarativeState, StateError};

struct ToolOutcome {
    success: bool,
    result: Json,
    error: Option<String>,
    messages: Vec<Json>,
    rejected: bool,
    rejection_reason: Option<Json>,
}

impl DeclarativeExecutor {
    fn tool_output_config(&self) -> (Option<String>, Option<String>, Json) {
        let Some(Json::Object(output)) = field(&self.def, "output") else {
            return (None, None, json!(true));
        };
        let s = |k: &str| {
            output
                .get(k)
                .filter(|v| py_truthy(v))
                .map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| py_str(v)))
        };
        (
            s("messages"),
            s("result"),
            output.get("autoSend").cloned().unwrap_or(json!(true)),
        )
    }

    fn store_tool_result(
        &self,
        state: &mut DeclarativeState,
        outcome: &ToolOutcome,
        messages_var: Option<&str>,
        result_var: Option<&str>,
    ) -> Result<(), StateError> {
        if let Some(var) = messages_var {
            state.set(
                &normalize_variable_path(var),
                Json::Array(outcome.messages.clone()),
            )?;
        }
        if let Some(var) = result_var {
            let path = normalize_variable_path(var);
            let value = if outcome.rejected {
                json!({"approved": false, "rejected": true, "reason": outcome.rejection_reason})
            } else if outcome.success {
                outcome.result.clone()
            } else {
                json!({"error": outcome.error})
            };
            state.set(&path, value)?;
        }
        Ok(())
    }

    async fn execute_tool(
        &self,
        function_name: &str,
        arguments: &Map<String, Json>,
    ) -> ToolOutcome {
        let Some(tool) = self.rt.tools.get(function_name).cloned() else {
            let error = format!("Function '{function_name}' not found in registry");
            tracing::error!("InvokeFunctionTool: {error}");
            return ToolOutcome {
                success: false,
                result: Json::Null,
                error: Some(error),
                messages: Vec::new(),
                rejected: false,
                rejection_reason: None,
            };
        };
        match tool.invoke(Json::Object(arguments.clone())).await {
            Ok(result) => {
                let call_id = uuid::Uuid::new_v4().to_string();
                let arguments_str = Json::Object(arguments.clone()).to_string();
                let call = Content::FunctionCall(FunctionCallContent::new(
                    call_id.clone(),
                    function_name,
                    Some(FunctionArguments::Raw(arguments_str)),
                ));
                let result_str = match &result {
                    Json::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let result_content = Content::FunctionResult(FunctionResultContent::new(
                    call_id,
                    Some(Json::String(result_str)),
                ));
                let messages = vec![
                    message_to_json(&Message::with_contents(
                        Role::new(Role::ASSISTANT),
                        vec![call],
                    )),
                    message_to_json(&Message::with_contents(
                        Role::new(Role::TOOL),
                        vec![result_content],
                    )),
                ];
                ToolOutcome {
                    success: true,
                    result,
                    error: None,
                    messages,
                    rejected: false,
                    rejection_reason: None,
                }
            }
            Err(e) => {
                tracing::error!(
                    "InvokeFunctionTool: error invoking function '{function_name}': {e}"
                );
                ToolOutcome {
                    success: false,
                    result: Json::Null,
                    error: Some(format!("{}: {e}", error_type_name(&e))),
                    messages: Vec::new(),
                    rejected: false,
                    rejection_reason: None,
                }
            }
        }
    }

    async fn finish_tool_call(
        &self,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
        function_name: &str,
        arguments: &Map<String, Json>,
    ) -> ActionResult {
        let (messages_var, result_var, auto_send_expr) = self.tool_output_config();
        let auto_send = py_truthy(&state.eval_if_expression(&auto_send_expr)?);
        let outcome = self.execute_tool(function_name, arguments).await;
        self.store_tool_result(
            state,
            &outcome,
            messages_var.as_deref(),
            result_var.as_deref(),
        )?;
        if auto_send && outcome.success && !outcome.result.is_null() {
            ctx.yield_output(Json::String(py_str(&outcome.result)))
                .await?;
        }
        ctx.send_message(action_complete()).await
    }

    pub(crate) async fn invoke_function_tool(
        &self,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let (_, result_var, _) = self.tool_output_config();
        let store_error = |state: &mut DeclarativeState, msg: String| -> Result<(), StateError> {
            tracing::error!("InvokeFunctionTool: {msg}");
            if let Some(var) = &result_var {
                state.set(&normalize_variable_path(var), json!({"error": msg}))?;
            }
            Ok(())
        };
        let Some(name_expr) = field(&self.def, "functionName").filter(|v| py_truthy(v)) else {
            store_error(
                state,
                format!(
                    "Action '{}' is missing required 'functionName' field",
                    self.id
                ),
            )?;
            return ctx.send_message(action_complete()).await;
        };
        let function_name = state.eval_if_expression(name_expr)?;
        if !py_truthy(&function_name) {
            store_error(
                state,
                format!(
                    "Action '{}': functionName expression evaluated to empty",
                    self.id
                ),
            )?;
            return ctx.send_message(action_complete()).await;
        }
        let function_name = py_str(&function_name);
        let mut arguments = Map::new();
        match field(&self.def, "arguments") {
            Some(Json::Object(m)) => {
                for (k, v) in m {
                    arguments.insert(k.clone(), state.eval_if_expression(v)?);
                }
            }
            None | Some(Json::Null) => {}
            Some(_) => {
                tracing::warn!("InvokeFunctionTool: 'arguments' must be a dictionary - ignoring")
            }
        }
        let require_approval = field(&self.def, "requireApproval").is_some_and(py_truthy);
        if require_approval {
            return ctx
                .request_info(json!({
                    "type": "ToolApprovalRequest",
                    "request_id": uuid::Uuid::new_v4().to_string(),
                    "function_name": function_name,
                    "arguments": arguments,
                }))
                .await;
        }
        self.finish_tool_call(state, ctx, &function_name, &arguments)
            .await
    }

    pub(crate) async fn tool_approval_response(
        &self,
        state: &mut DeclarativeState,
        response: RequestResponse,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let (approved, reason) = parse_approval(&response.data)?;
        let original = &response.original_request;
        let function_name = original
            .get("function_name")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_string();
        let arguments = original
            .get("arguments")
            .and_then(Json::as_object)
            .cloned()
            .unwrap_or_default();
        if !approved {
            let (messages_var, result_var, _) = self.tool_output_config();
            let reason_text = reason
                .as_ref()
                .filter(|r| !r.is_null())
                .map(py_str)
                .unwrap_or_else(|| "No reason provided".to_string());
            let message = Message::new(
                Role::new(Role::ASSISTANT),
                format!("Function '{function_name}' was rejected: {reason_text}"),
            );
            let outcome = ToolOutcome {
                success: false,
                result: Json::Null,
                error: None,
                messages: vec![message_to_json(&message)],
                rejected: true,
                rejection_reason: Some(reason.unwrap_or(Json::Null)),
            };
            self.store_tool_result(
                state,
                &outcome,
                messages_var.as_deref(),
                result_var.as_deref(),
            )?;
            return ctx.send_message(action_complete()).await;
        }
        self.finish_tool_call(state, ctx, &function_name, &arguments)
            .await
    }
}

/// Parse a `ToolApprovalResponse`: a bare Boolean or
/// `{"approved": bool, "reason": ...}`. A non-Boolean `approved` is an error.
pub(crate) fn parse_approval(data: &Json) -> Result<(bool, Option<Json>), CoreError> {
    match data {
        Json::Bool(b) => Ok((*b, None)),
        Json::Object(m) => match m.get("approved") {
            Some(Json::Bool(b)) => Ok((*b, m.get("reason").cloned())),
            _ => Err(CoreError::Workflow(
                "declarative action error: approved must be a bool.".into(),
            )),
        },
        _ => Err(CoreError::Workflow(
            "declarative action error: approval response must be a Boolean or {\"approved\": bool}"
                .into(),
        )),
    }
}

fn error_type_name(e: &CoreError) -> &'static str {
    match e {
        CoreError::Tool(_) => "ToolError",
        CoreError::Workflow(_) => "WorkflowError",
        CoreError::Serialization(_) | CoreError::Json(_) => "SerializationError",
        _ => "Error",
    }
}
