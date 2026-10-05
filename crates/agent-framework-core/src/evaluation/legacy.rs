//! The deprecated `AgentEvalConverter` compatibility surface.
//!
//! Upstream keeps `AgentEvalConverter` (with a `DeprecationWarning` on every
//! call) so code written against earlier releases keeps working, preserving
//! the legacy Foundry evaluator wire format it emitted. Rust expresses the
//! runtime warning as a compile-time `#[deprecated]`.

#![allow(deprecated)]

use std::collections::HashSet;

use serde_json::{json, Map, Value};

use super::{to_eval_item, EvalItem, EvalQuery};
use crate::agent::SupportsAgentRun;
use crate::tools::{ToolDefinition, ToolKind};
use crate::types::{AgentResponse, Content, FunctionArguments, Message};

const DEPRECATION: &str = "`AgentEvalConverter` is deprecated and will be removed in a future \
     version. Construct `EvalItem` directly or use `evaluate_agent()` / `evaluate_workflow()`; \
     Foundry wire conversion is internal to `agent-framework-foundry`.";

/// Deprecated compatibility surface for earlier Agent Framework releases.
/// Mirrors upstream's `AgentEvalConverter`.
///
/// New code should construct [`EvalItem`] directly or use
/// [`evaluate_agent`](super::evaluate_agent) /
/// [`evaluate_workflow`](super::evaluate_workflow). Foundry wire
/// serialization is owned by `agent-framework-foundry`.
#[deprecated(
    note = "construct `EvalItem` directly or use `evaluate_agent` / `evaluate_workflow`; Foundry wire conversion is internal to `agent-framework-foundry`"
)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AgentEvalConverter;

impl AgentEvalConverter {
    /// Convert one message to the legacy Foundry evaluator wire format.
    ///
    /// Text becomes `{"type": "text"}`, data/URI content `{"type":
    /// "input_image"}` (with `"detail": "auto"` when a media type is known),
    /// function calls `{"type": "tool_call"}` (string arguments are parsed;
    /// unparseable ones become `{"_raw_arguments": "[unparseable]"}`), and
    /// function results become one `role: "tool"` message each. A message
    /// with nothing convertible becomes a single empty text item.
    pub fn convert_message(message: &Message) -> Vec<Value> {
        tracing::warn!("{DEPRECATION}");
        convert_legacy_message(message)
    }

    /// Convert messages to the legacy Foundry evaluator wire format.
    pub fn convert_messages(messages: &[Message]) -> Vec<Value> {
        tracing::warn!("{DEPRECATION}");
        messages.iter().flat_map(convert_legacy_message).collect()
    }

    /// Extract legacy evaluator tool-definition objects
    /// (`{"name", "description", "parameters"}`) from an agent's function
    /// tools, de-duplicated by name.
    pub fn extract_tools(agent: &dyn SupportsAgentRun) -> Vec<Value> {
        tracing::warn!("{DEPRECATION}");
        let mut seen: HashSet<String> = HashSet::new();
        agent
            .default_tools()
            .into_iter()
            .filter(|t| t.kind == ToolKind::Function && seen.insert(t.name.clone()))
            .map(|t| tool_definition_json(&t))
            .collect()
    }

    /// Build an [`EvalItem`] through the provider-neutral path.
    pub fn to_eval_item(
        query: impl Into<EvalQuery>,
        response: &AgentResponse,
        agent: Option<&dyn SupportsAgentRun>,
        tools: Option<Vec<ToolDefinition>>,
        context: Option<String>,
    ) -> EvalItem {
        tracing::warn!("{DEPRECATION}");
        to_eval_item(query.into(), response, agent, tools, context)
    }
}

fn tool_definition_json(tool: &ToolDefinition) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    })
}

fn arguments_json(arguments: &Option<FunctionArguments>) -> Value {
    match arguments {
        None => json!({}),
        Some(FunctionArguments::Object(map)) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Map<String, Value>>(),
            )
        }
        Some(FunctionArguments::Raw(raw)) => {
            serde_json::from_str(raw).unwrap_or_else(|_| json!({"_raw_arguments": "[unparseable]"}))
        }
    }
}

/// Upstream's `_convert_legacy_foundry_message`.
fn convert_legacy_message(message: &Message) -> Vec<Value> {
    let mut content_items: Vec<Value> = Vec::new();
    let mut tool_results: Vec<(String, Value)> = Vec::new();

    for content in &message.contents {
        match content {
            Content::Text(t) if !t.text.is_empty() => {
                content_items.push(json!({"type": "text", "text": t.text}));
            }
            Content::Data(d) if !d.uri.is_empty() => {
                let mut image = json!({"type": "input_image", "image_url": d.uri});
                if d.media_type.as_deref().is_some_and(|m| !m.is_empty()) {
                    image["detail"] = json!("auto");
                }
                content_items.push(image);
            }
            Content::Uri(u) if !u.uri.is_empty() => {
                let mut image = json!({"type": "input_image", "image_url": u.uri});
                if !u.media_type.is_empty() {
                    image["detail"] = json!("auto");
                }
                content_items.push(image);
            }
            Content::FunctionCall(fc) => {
                content_items.push(json!({
                    "type": "tool_call",
                    "tool_call_id": fc.call_id,
                    "name": fc.name,
                    "arguments": arguments_json(&fc.arguments),
                }));
            }
            Content::FunctionResult(fr) => {
                let result = match &fr.result {
                    Some(Value::String(s)) => {
                        serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
                    }
                    Some(other) => other.clone(),
                    None => Value::Null,
                };
                tool_results.push((fr.call_id.clone(), result));
            }
            _ => {}
        }
    }

    if !tool_results.is_empty() {
        return tool_results
            .into_iter()
            .map(|(call_id, result)| {
                json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": [{"type": "tool_result", "tool_result": result}],
                })
            })
            .collect();
    }
    if !content_items.is_empty() {
        return vec![json!({"role": message.role.as_str(), "content": content_items})];
    }
    vec![json!({"role": message.role.as_str(), "content": [{"type": "text", "text": ""}]})]
}
