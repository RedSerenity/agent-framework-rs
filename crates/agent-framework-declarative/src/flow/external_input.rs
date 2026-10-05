//! `Question` and `RequestExternalInput` (port of
//! `_executors_external_input.py`), mapped onto the core engine's
//! request/response (human-in-the-loop) mechanism.
//!
//! The executor pauses the run with a request whose payload is
//!
//! ```json
//! {"type": "ExternalInputRequest", "request_id": "...", "message": "What is your name?",
//!  "request_type": "question", "metadata": {"output_property": "Local.userName", ...}}
//! ```
//!
//! and resumes when the caller answers with
//! [`WorkflowRun::send_response`](agent_framework_core::workflow::WorkflowRun::send_response).
//! The answer may be the user's text, or `{"user_input": "...", "value": ...}`
//! (a non-null `value` wins, as upstream). It is stored at the configured
//! output path (`variable`, then `output.property`, then `property`, else
//! `Local.answer` / `Local.externalInput`).
//!
//! `RequestHumanInput` and `WaitForHumanInput` — which upstream validates
//! (`variable` required) but has no executor for, so they are silently
//! skipped there — run as `RequestExternalInput` here; this is a documented
//! extension rather than dropping the action.

use agent_framework_core::workflow::{RequestResponse, WorkflowContext};
use serde_json::{json, Map, Value as Json};

use super::executor::{field, Action, ActionResult, DeclarativeExecutor};
use super::messages::action_complete;
use super::state::{py_str, DeclarativeState};

/// Prompt text: `{key: {text: ...}}`, `{key: "..."}`, else `fallback`.
fn prompt_text(def: &Json, primary: &str, fallback: &str) -> Json {
    match field(def, primary) {
        Some(Json::Object(m)) if m.contains_key("text") => m["text"].clone(),
        Some(Json::String(s)) => Json::String(s.clone()),
        _ => field(def, fallback).cloned().unwrap_or(json!("")),
    }
}

fn output_path(def: &Json, default: &str) -> String {
    let nonempty = |v: Option<&Json>| {
        v.and_then(Json::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    nonempty(field(def, "variable"))
        .or_else(|| nonempty(field(def, "output").and_then(|o| o.get("property"))))
        .or_else(|| nonempty(field(def, "property")))
        .unwrap_or_else(|| default.to_string())
}

impl DeclarativeExecutor {
    pub(crate) async fn request_external_input(
        &self,
        action: Action,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let def = &self.def;
        let request = if action == Action::Question {
            let question = prompt_text(def, "question", "text");
            let output_property = output_path(def, "Local.answer");
            let default_value = field(def, "default")
                .or_else(|| field(def, "defaultValue"))
                .cloned()
                .unwrap_or(Json::Null);
            let allow_free_text = field(def, "allowFreeText").cloned().unwrap_or(json!(true));
            let choices: Option<Vec<Json>> = field(def, "choices")
                .and_then(Json::as_array)
                .filter(|c| !c.is_empty())
                .map(|choices| {
                    choices
                        .iter()
                        .map(|c| match c {
                            Json::Object(m) => {
                                let value = m.get("value").cloned().unwrap_or(json!(""));
                                let label = m
                                    .get("label")
                                    .filter(|l| super::state::py_truthy(l))
                                    .cloned()
                                    .unwrap_or_else(|| value.clone());
                                json!({"value": value, "label": label})
                            }
                            other => json!({"value": py_str(other), "label": py_str(other)}),
                        })
                        .collect()
                });
            let message = state.eval_if_expression(&question)?;
            json!({
                "type": "ExternalInputRequest",
                "request_id": uuid::Uuid::new_v4().to_string(),
                "message": py_str(&message),
                "request_type": "question",
                "metadata": {
                    "output_property": output_property,
                    "choices": choices,
                    "allow_free_text": allow_free_text,
                    "default_value": default_value,
                },
            })
        } else {
            let message = prompt_text(def, "prompt", "message");
            // RequestHumanInput / WaitForHumanInput may phrase it as a question.
            let message = match (&message, field(def, "question")) {
                (Json::String(s), Some(q)) if s.is_empty() => match q {
                    Json::Object(m) => m.get("text").cloned().unwrap_or(json!("")),
                    other => other.clone(),
                },
                _ => message,
            };
            let output_property = output_path(def, "Local.externalInput");
            let mut metadata: Map<String, Json> = field(def, "metadata")
                .and_then(Json::as_object)
                .cloned()
                .unwrap_or_default();
            metadata.insert("output_property".into(), json!(output_property));
            metadata.insert(
                "required_fields".into(),
                field(def, "requiredFields").cloned().unwrap_or(json!([])),
            );
            metadata.insert(
                "default_value".into(),
                field(def, "default").cloned().unwrap_or(Json::Null),
            );
            if let Some(t) = field(def, "timeout").filter(|t| super::state::py_truthy(t)) {
                metadata.insert("timeout_seconds".into(), t.clone());
            }
            let evaluated = state.eval_if_expression(&message)?;
            json!({
                "type": "ExternalInputRequest",
                "request_id": uuid::Uuid::new_v4().to_string(),
                "message": py_str(&evaluated),
                "request_type": field(def, "requestType").cloned().unwrap_or(json!("external")),
                "metadata": metadata,
            })
        };
        ctx.request_info(request).await
    }

    pub(crate) async fn external_input_response(
        &self,
        state: &mut DeclarativeState,
        response: RequestResponse,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let default_path = if matches!(self.node, super::executor::Node::Action(Action::Question)) {
            "Local.answer"
        } else {
            "Local.externalInput"
        };
        let output_property = response
            .original_request
            .get("metadata")
            .and_then(|m| m.get("output_property"))
            .and_then(Json::as_str)
            .unwrap_or(default_path)
            .to_string();
        let answer = match &response.data {
            Json::Object(m) if m.contains_key("user_input") || m.contains_key("value") => {
                match m.get("value").filter(|v| !v.is_null()) {
                    Some(v) => v.clone(),
                    None => m.get("user_input").cloned().unwrap_or(Json::Null),
                }
            }
            other => other.clone(),
        };
        if !output_property.is_empty() {
            state.set(&output_property, answer)?;
        }
        ctx.send_message(action_complete()).await
    }
}
