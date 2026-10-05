//! Interception-point codecs: framework values <-> AGENT-HOOKS wire JSON.
//!
//! Mirrors upstream's codec region (`_InputCodec`, `_ModelRequestCodec`,
//! `_ModelResponseCodec`, `_ToolArgumentsCodec`, `_ToolResultCodec`,
//! `_OutputCodec` in `_agent_hooks.py`; `Codecs/` in .NET). One codec per
//! interception point owns both directions: `to_wire` projects the native
//! value into the spec's payload and `write_back` converts the (possibly
//! transformed) wire target back. Every `write_back` applies the same rule:
//! a wire value the interceptors left untouched maps back to the untouched
//! native value — only genuine transforms modify native state — and an
//! untranslatable transform fails closed with [`Error::MiddlewareFailure`]
//! (upstream's `_AgentHooksWriteBackError`) rather than being dropped.
//!
//! Rich (non-text) content is projected as the content's serde form
//! (`{"type": ..., ...}`, the same shape upstream's `Content.to_dict()`
//! produces) and decoded back into [`Content`], never flattened to text.

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::types::{
    AgentResponse, ChatResponse, Content, FinishReason, FunctionArguments, FunctionCallContent,
    Message, Role, UsageDetails,
};

/// A write-back failure (upstream `_AgentHooksWriteBackError`): fails the
/// guarded action closed.
pub(crate) fn write_back_error(message: impl Into<String>) -> Error {
    Error::MiddlewareFailure(message.into())
}

/// Type-aware equality for wire values (upstream `_wire_equal`): bools never
/// equal numbers, and numbers compare by value (`1 == 1.0`).
pub(crate) fn wire_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => match (a.as_i64(), b.as_i64()) {
            (Some(x), Some(y)) => x == y,
            _ => match (a.as_u64(), b.as_u64()) {
                (Some(x), Some(y)) => x == y,
                _ => a.as_f64() == b.as_f64(),
            },
        },
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|other| wire_equal(v, other)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| wire_equal(x, y))
        }
        _ => left == right,
    }
}

fn role_str(role: &Role) -> String {
    if role.as_str().is_empty() {
        "user".into()
    } else {
        role.as_str().to_string()
    }
}

/// Map a framework role onto the spec's input role enum
/// (`user | system | external`).
pub(crate) fn input_role(role: &Role) -> String {
    match role.as_str() {
        "" | "user" => "user".into(),
        "system" => "system".into(),
        _ => "external".into(),
    }
}

/// Project contents faithfully: plain text as a string, rich content as
/// content objects.
pub(crate) fn contents_to_wire(contents: &[Content]) -> Value {
    if let [Content::Text(t)] = contents {
        return Value::String(t.text.clone());
    }
    Value::Array(
        contents
            .iter()
            .map(|c| serde_json::to_value(c).unwrap_or(Value::Null))
            .collect(),
    )
}

pub(crate) fn message_to_wire(message: &Message) -> Value {
    json!({ "role": role_str(&message.role), "content": contents_to_wire(&message.contents) })
}

pub(crate) fn messages_to_wire(messages: &[Message]) -> Vec<Value> {
    messages.iter().map(message_to_wire).collect()
}

/// Decode a transformed wire content value back into framework contents.
pub(crate) fn wire_to_contents(value: &Value, point: &str) -> Result<Vec<Content>> {
    let items: Vec<&Value> = match value {
        Value::Null => return Ok(Vec::new()),
        Value::String(s) => return Ok(vec![Content::text(s.clone())]),
        Value::Object(_) => vec![value],
        Value::Array(items) => items.iter().collect(),
        _ => {
            return Err(write_back_error(format!(
                "agent-hooks {point} transform produced an unsupported content value type."
            )))
        }
    };
    items
        .into_iter()
        .map(|item| match item {
            Value::String(s) => Ok(Content::text(s.clone())),
            Value::Object(obj) if obj.contains_key("type") => {
                match serde_json::from_value::<Content>(item.clone()) {
                    // `Content` deserializes unknown tags leniently; a
                    // transform must not smuggle an inert placeholder in.
                    Ok(Content::Unknown) if obj.get("type") != Some(&json!("unknown")) => {
                        Err(undecodable(point))
                    }
                    Ok(c) => Ok(c),
                    Err(_) => Err(undecodable(point)),
                }
            }
            _ => Err(write_back_error(format!(
                "agent-hooks {point} transform produced an unsupported content item."
            ))),
        })
        .collect()
}

fn undecodable(point: &str) -> Error {
    write_back_error(format!(
        "agent-hooks {point} transform produced an undecodable content item."
    ))
}

fn looks_like_message_objects(value: &Value) -> bool {
    matches!(value, Value::Array(items)
        if !items.is_empty() && items.iter().all(|i| i.get("content").is_some() && i.is_object()))
}

fn wire_role(item: &Value) -> String {
    item.get("role")
        .and_then(Value::as_str)
        .filter(|r| !r.is_empty())
        .unwrap_or("user")
        .to_string()
}

/// Convert a transformed wire message list back into framework messages
/// (upstream `_write_back_message_list`).
///
/// The transformed list is authoritative; entries are matched to originals by
/// projection identity rather than position:
///
/// - an entry equal to an unconsumed original's projection reuses that
///   original untouched (originals skipped over were removed);
/// - a changed entry rewrites the next unconsumed original's contents only
///   when that original's projection is not preserved later in the list and
///   its role is unchanged (keeping its id, author and properties);
/// - anything else (insertions, role changes) becomes a new message.
pub(crate) fn write_back_message_list(
    originals: Vec<Message>,
    before: &[Value],
    after: &Value,
    point: &str,
) -> Result<Vec<Message>> {
    let Value::Array(after_items) = after else {
        return Err(write_back_error(format!(
            "agent-hooks {point} transform must produce a list of messages."
        )));
    };
    if after_items
        .iter()
        .any(|i| !i.is_object() || i.get("content").is_none())
    {
        return Err(write_back_error(format!(
            "agent-hooks {point} transform produced a message without role/content."
        )));
    }
    let mut originals: Vec<Option<Message>> = originals.into_iter().map(Some).collect();
    let mut result = Vec::with_capacity(after_items.len());
    let mut cursor = 0usize;
    for (index, item) in after_items.iter().enumerate() {
        let matched = (cursor..originals.len()).find(|p| wire_equal(&before[*p], item));
        if let Some(position) = matched {
            if let Some(m) = originals[position].take() {
                result.push(m);
            }
            cursor = position + 1;
            continue;
        }
        if cursor < originals.len() {
            let candidate = &before[cursor];
            let preserved_later = after_items[index + 1..]
                .iter()
                .any(|later| wire_equal(later, candidate));
            let role = wire_role(item);
            if !preserved_later && candidate.get("role").and_then(Value::as_str) == Some(&role) {
                if let Some(mut message) = originals[cursor].take() {
                    cursor += 1;
                    message.contents = wire_to_contents(&item["content"], point)?;
                    result.push(message);
                    continue;
                }
            }
        }
        result.push(Message::with_contents(
            Role::new(wire_role(item)),
            wire_to_contents(&item["content"], point)?,
        ));
    }
    Ok(result)
}

/// Project tool-call arguments as the spec's `args` object (upstream
/// `_arguments_to_wire`).
pub(crate) fn arguments_to_wire(arguments: &Value) -> Map<String, Value> {
    match arguments {
        Value::Null => Map::new(),
        Value::Object(map) => map.clone(),
        Value::String(raw) => match serde_json::from_str::<Value>(raw) {
            Ok(Value::Object(map)) => map,
            _ => Map::from_iter([("raw_arguments".to_string(), Value::String(raw.clone()))]),
        },
        other => Map::from_iter([(
            "raw_arguments".to_string(),
            Value::String(other.to_string()),
        )]),
    }
}

fn function_arguments_to_wire(arguments: &Option<FunctionArguments>) -> Map<String, Value> {
    match arguments {
        None => Map::new(),
        Some(FunctionArguments::Object(map)) => {
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        }
        Some(FunctionArguments::Raw(raw)) => arguments_to_wire(&Value::String(raw.clone())),
    }
}

/// `usage` projection: the integer counts of the usage details, `None` when
/// there are none.
pub(crate) fn usage_to_wire(usage: Option<&UsageDetails>) -> Option<Value> {
    let Value::Object(map) = serde_json::to_value(usage?).ok()? else {
        return None;
    };
    let counts: Map<String, Value> = map.into_iter().filter(|(_, v)| v.is_u64()).collect();
    (!counts.is_empty()).then_some(Value::Object(counts))
}

pub(crate) fn finish_reason_str(reason: Option<&FinishReason>) -> String {
    reason
        .map(FinishReason::as_str)
        .filter(|r| !r.is_empty())
        .unwrap_or("stop")
        .to_string()
}

/// `input`: the run's input messages <-> the spec's input payload.
pub(crate) struct InputCodec;

impl InputCodec {
    fn message_to_wire(message: &Message) -> Value {
        json!({ "role": input_role(&message.role), "content": contents_to_wire(&message.contents) })
    }

    /// A single message projects as its content (a plain string for plain
    /// text, so string-matching perimeter guards fire) and role; multiple
    /// messages project as a list of per-message objects under role `user`.
    /// Returns the `(content, role)` payload pair.
    pub(crate) fn to_wire(messages: &[Message]) -> (Value, String) {
        if let [only] = messages {
            return (contents_to_wire(&only.contents), input_role(&only.role));
        }
        (
            Value::Array(messages.iter().map(Self::message_to_wire).collect()),
            "user".into(),
        )
    }

    /// Write a transformed `input` target back; returns whether it changed.
    pub(crate) fn write_back(
        messages: &mut Vec<Message>,
        before: &Value,
        after: &Value,
    ) -> Result<bool> {
        if after.is_null() || wire_equal(after, before) {
            return Ok(false);
        }
        let Value::Object(after_map) = after else {
            return Err(write_back_error(
                "agent-hooks input transform must produce an input object target.",
            ));
        };
        let mut changed = false;
        let after_role = after_map.get("role").unwrap_or(&Value::Null);
        if !wire_equal(after_role, &before["role"]) {
            // The role is per-message only for single-message input.
            match (messages.len(), after_role.as_str()) {
                (1, Some(role)) => {
                    messages[0].role = Role::new(role);
                    changed = true;
                }
                _ => {
                    return Err(write_back_error(
                        "agent-hooks input transform changed the input role in a way that cannot \
                         be written back.",
                    ))
                }
            }
        }
        let after_content = after_map.get("content").unwrap_or(&Value::Null);
        if wire_equal(after_content, &before["content"]) {
            return Ok(changed);
        }
        if messages.len() == 1 && !looks_like_message_objects(after_content) {
            messages[0].contents = wire_to_contents(after_content, "input")?;
            return Ok(true);
        }
        let before_list: Vec<Value> = messages.iter().map(Self::message_to_wire).collect();
        let originals = std::mem::take(messages);
        *messages = write_back_message_list(originals, &before_list, after_content, "input")?;
        Ok(true)
    }
}

/// `pre_model_call`: the outgoing request messages <-> the spec's messages.
pub(crate) struct ModelRequestCodec;

impl ModelRequestCodec {
    pub(crate) fn to_wire(messages: &[Message]) -> Vec<Value> {
        messages_to_wire(messages)
    }

    /// The transformed message list, or the original messages unchanged
    /// when the target is untouched.
    pub(crate) fn write_back(
        messages: Vec<Message>,
        before: &[Value],
        after: &Value,
    ) -> Result<Vec<Message>> {
        if wire_equal(after, &Value::Array(before.to_vec())) {
            return Ok(messages);
        }
        write_back_message_list(messages, before, after, "pre_model_call")
    }
}

/// `post_model_call`: the assembled chat response <-> the spec's response.
///
/// Host-executed tool calls ([`Content::FunctionCall`]) ride `tool_calls`
/// (they drive the function seam); service-executed tool activity (hosted
/// MCP / search / code-interpreter content) is part of the model response
/// itself and is surfaced in `content`, so it is interceptable here even
/// though the function seam never sees it.
pub(crate) struct ModelResponseCodec;

impl ModelResponseCodec {
    fn content_to_wire(messages: &[Message]) -> Value {
        let parts: Vec<Value> = messages
            .iter()
            .filter_map(|m| {
                let visible: Vec<Content> = m
                    .contents
                    .iter()
                    .filter(|c| !matches!(c, Content::FunctionCall(_)))
                    .cloned()
                    .collect();
                (!visible.is_empty()).then(
                    || json!({ "role": role_str(&m.role), "content": contents_to_wire(&visible) }),
                )
            })
            .collect();
        match parts.as_slice() {
            [] => Value::Null,
            [only] if only["content"].is_string() => only["content"].clone(),
            _ => Value::Array(parts),
        }
    }

    fn tool_calls_to_wire(messages: &[Message]) -> Value {
        Value::Array(
            messages
                .iter()
                .flat_map(|m| &m.contents)
                .filter_map(|c| match c {
                    Content::FunctionCall(call) => Some(json!({
                        "id": call.call_id,
                        "name": call.name,
                        "args": function_arguments_to_wire(&call.arguments),
                    })),
                    _ => None,
                })
                .collect(),
        )
    }

    pub(crate) fn to_wire(response: &ChatResponse) -> Value {
        json!({
            "content": Self::content_to_wire(&response.messages),
            "tool_calls": Self::tool_calls_to_wire(&response.messages),
            "finish_reason": finish_reason_str(response.finish_reason.as_ref()),
        })
    }

    /// Write a transformed target back; returns whether the response changed.
    pub(crate) fn write_back(
        response: &mut ChatResponse,
        before: &Value,
        after: &Value,
    ) -> Result<bool> {
        if after.is_null() || wire_equal(after, before) {
            return Ok(false);
        }
        let Value::Object(after_map) = after else {
            return Err(write_back_error(
                "agent-hooks post_model_call transform must produce a response object.",
            ));
        };
        let mut changed = false;
        let after_finish = after_map.get("finish_reason").unwrap_or(&Value::Null);
        if !wire_equal(after_finish, &before["finish_reason"]) {
            let Some(reason) = after_finish.as_str() else {
                return Err(write_back_error(
                    "agent-hooks post_model_call transform must keep finish_reason a string.",
                ));
            };
            response.finish_reason = Some(FinishReason::new(reason));
            changed = true;
        }
        let after_calls = after_map.get("tool_calls").unwrap_or(&Value::Null);
        if !wire_equal(after_calls, &before["tool_calls"]) {
            changed = Self::write_back_tool_calls(response, after_calls)? || changed;
        }
        let after_content = after_map.get("content").unwrap_or(&Value::Null);
        if !wire_equal(after_content, &before["content"]) {
            Self::write_back_content(response, after_content)?;
            changed = true;
        }
        Ok(changed)
    }

    fn write_back_tool_calls(response: &mut ChatResponse, after_calls: &Value) -> Result<bool> {
        let Value::Array(items) = after_calls else {
            return Err(write_back_error(
                "agent-hooks post_model_call transform must keep tool_calls a list.",
            ));
        };
        let mut wire_calls: Vec<&Map<String, Value>> = Vec::new();
        for item in items {
            match item.as_object() {
                Some(obj) if obj.contains_key("id") && obj.contains_key("name") => {
                    wire_calls.push(obj)
                }
                _ => {
                    return Err(write_back_error(
                        "agent-hooks post_model_call transform produced a tool call without \
                         id/name.",
                    ))
                }
            }
        }
        let id_of = |call: &Map<String, Value>| match &call["id"] {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let by_id: HashMap<String, &Map<String, Value>> =
            wire_calls.iter().map(|c| (id_of(c), *c)).collect();
        let mut consumed = std::collections::HashSet::new();
        let mut changed = false;
        for message in &mut response.messages {
            let mut kept = Vec::with_capacity(message.contents.len());
            for content in std::mem::take(&mut message.contents) {
                let Content::FunctionCall(mut call) = content else {
                    kept.push(content);
                    continue;
                };
                let Some(wire) = by_id.get(&call.call_id) else {
                    changed = true; // the transform dropped this tool call
                    continue;
                };
                consumed.insert(call.call_id.clone());
                let name = wire
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|n| !n.is_empty())
                    .ok_or_else(|| {
                        write_back_error(
                            "agent-hooks post_model_call transform must keep each tool call's \
                             name a non-empty string.",
                        )
                    })?;
                if name != call.name {
                    call.name = name.to_string();
                    changed = true;
                }
                let Some(Value::Object(args)) = wire.get("args") else {
                    return Err(write_back_error(
                        "agent-hooks post_model_call transform must keep each tool call's args \
                         an object.",
                    ));
                };
                let current = Value::Object(function_arguments_to_wire(&call.arguments));
                if !wire_equal(&current, &Value::Object(args.clone())) {
                    call.arguments = Some(FunctionArguments::Object(
                        args.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                    ));
                    changed = true;
                }
                kept.push(Content::FunctionCall(call));
            }
            message.contents = kept;
        }
        let added: Vec<Content> = wire_calls
            .iter()
            .filter(|c| !consumed.contains(&id_of(c)))
            .map(|c| {
                let args = c
                    .get("args")
                    .and_then(Value::as_object)
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                Content::FunctionCall(FunctionCallContent::new(
                    id_of(c),
                    c.get("name").and_then(Value::as_str).unwrap_or_default(),
                    Some(FunctionArguments::Object(args)),
                ))
            })
            .collect();
        if !added.is_empty() {
            match response
                .messages
                .iter_mut()
                .rev()
                .find(|m| m.role == Role::assistant())
            {
                Some(target) => target.contents.extend(added),
                None => response
                    .messages
                    .push(Message::with_contents(Role::assistant(), added)),
            }
            changed = true;
        }
        Ok(changed)
    }

    fn write_back_content(response: &mut ChatResponse, after_content: &Value) -> Result<()> {
        let calls: Vec<Content> = response
            .messages
            .iter()
            .flat_map(|m| &m.contents)
            .filter(|c| matches!(c, Content::FunctionCall(_)))
            .cloned()
            .collect();
        let mut base: Vec<Message> = match after_content {
            Value::Null => Vec::new(),
            Value::String(s) => vec![Message::assistant(s.clone())],
            Value::Array(items) => items
                .iter()
                .map(|item| {
                    if !item.is_object() || item.get("content").is_none() {
                        return Err(write_back_error(
                            "agent-hooks post_model_call transform produced content without \
                             role/content.",
                        ));
                    }
                    let role = item
                        .get("role")
                        .and_then(Value::as_str)
                        .filter(|r| !r.is_empty())
                        .unwrap_or("assistant");
                    Ok(Message::with_contents(
                        Role::new(role),
                        wire_to_contents(&item["content"], "post_model_call")?,
                    ))
                })
                .collect::<Result<_>>()?,
            _ => {
                return Err(write_back_error(
                    "agent-hooks post_model_call transform produced unsupported content.",
                ))
            }
        };
        if !calls.is_empty() {
            match base.last_mut() {
                Some(last) if last.role == Role::assistant() => last.contents.extend(calls),
                _ => base.push(Message::with_contents(Role::assistant(), calls)),
            }
        }
        response.messages = base;
        Ok(())
    }
}

/// `pre_tool_call`: the native tool arguments <-> the spec's `args` object.
pub(crate) struct ToolArgumentsCodec;

impl ToolArgumentsCodec {
    pub(crate) fn to_wire(arguments: &Value) -> Map<String, Value> {
        arguments_to_wire(arguments)
    }

    /// Merge a transformed `args` target back onto the native arguments.
    /// Returns `(native_arguments, effective_wire_args)`, or `None` for the
    /// native arguments when the target is untouched. Only keys the transform
    /// changed (or added) are taken from the wire value; removed keys are
    /// dropped and untouched keys keep their native values.
    pub(crate) fn write_back(
        arguments: &Value,
        before: &Map<String, Value>,
        after: &Value,
    ) -> Result<(Option<Value>, Map<String, Value>)> {
        let Value::Object(effective) = after else {
            return Err(write_back_error(
                "agent-hooks pre_tool_call transform must produce an arguments object.",
            ));
        };
        if wire_equal(after, &Value::Object(before.clone())) {
            return Ok((None, effective.clone()));
        }
        let native = arguments.as_object().cloned().unwrap_or_default();
        let mut merged: Map<String, Value> = native
            .into_iter()
            .filter(|(k, _)| effective.contains_key(k))
            .collect();
        for (key, value) in effective {
            if before.get(key).is_none_or(|b| !wire_equal(b, value)) {
                merged.insert(key.clone(), value.clone());
            }
        }
        Ok((Some(Value::Object(merged)), effective.clone()))
    }
}

/// `post_tool_call`: the native tool result <-> the spec's result value.
///
/// Tool results in this port are already JSON values (upstream unwraps its
/// `Content` result containers first), so the projection is the identity.
pub(crate) struct ToolResultCodec;

impl ToolResultCodec {
    pub(crate) fn to_wire(result: Option<&Value>) -> Value {
        result.cloned().unwrap_or(Value::Null)
    }

    /// The untouched native result, or the transformed value.
    pub(crate) fn write_back(
        original: Option<Value>,
        before: &Value,
        after: &Value,
    ) -> Option<Value> {
        if wire_equal(after, before) {
            original
        } else {
            Some(after.clone())
        }
    }
}

/// `output`: the final agent response <-> the spec's output payload.
pub(crate) struct OutputCodec;

impl OutputCodec {
    /// A single plain-text message projects as a string, else a list of
    /// per-message objects.
    pub(crate) fn to_wire(response: &AgentResponse) -> Value {
        let parts = messages_to_wire(&response.messages);
        match parts.as_slice() {
            [only] if only["content"].is_string() => only["content"].clone(),
            _ => Value::Array(parts),
        }
    }

    /// Write a transformed `output` target back; returns whether it changed.
    pub(crate) fn write_back(
        response: &mut AgentResponse,
        before_content: &Value,
        after: &Value,
    ) -> Result<bool> {
        if after.is_null() {
            return Ok(false);
        }
        let Value::Object(after_map) = after else {
            return Err(write_back_error(
                "agent-hooks output transform must produce an output object target.",
            ));
        };
        let after_content = after_map.get("content").unwrap_or(&Value::Null);
        if wire_equal(after_content, before_content) {
            return Ok(false);
        }
        match after_content {
            Value::String(s) => {
                if response.messages.len() == 1 {
                    response.messages[0].contents = wire_to_contents(after_content, "output")?;
                } else {
                    response.messages = vec![Message::assistant(s.clone())];
                }
            }
            Value::Null => response.messages.clear(),
            _ => {
                let before_list = messages_to_wire(&response.messages);
                let originals = std::mem::take(&mut response.messages);
                response.messages =
                    write_back_message_list(originals, &before_list, after_content, "output")?;
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DataContent;

    #[test]
    fn input_codec_maps_roles_onto_the_spec_enum() {
        assert_eq!(input_role(&Role::user()), "user");
        assert_eq!(input_role(&Role::system()), "system");
        assert_eq!(input_role(&Role::assistant()), "external");
        assert_eq!(input_role(&Role::tool()), "external");
        let (content, role) = InputCodec::to_wire(&[Message::user("hi")]);
        assert_eq!((content, role.as_str()), (json!("hi"), "user"));
        let (content, role) = InputCodec::to_wire(&[Message::system("s"), Message::user("u")]);
        assert_eq!(role, "user");
        assert_eq!(
            content,
            json!([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
        );
    }

    #[test]
    fn rich_content_round_trips_as_content_objects() {
        let data = Content::Data(DataContent::from_bytes(b"png", "image/png"));
        let message =
            Message::with_contents(Role::user(), vec![Content::text("look"), data.clone()]);
        let wire = contents_to_wire(&message.contents);
        assert_eq!(wire[0]["type"], "text");
        assert_eq!(wire[1]["type"], "data");
        let back = wire_to_contents(&wire, "input").unwrap();
        assert_eq!(back[1], data);
        assert!(wire_to_contents(&json!([{"type": "nope"}]), "input").is_err());
        assert!(wire_to_contents(&json!(5), "input").is_err());
        assert!(wire_to_contents(&json!([5]), "input").is_err());
    }

    #[test]
    fn tool_arguments_codec_merges_only_changed_keys() {
        let native = json!({"a": 1, "b": "x", "c": [1]});
        let before = ToolArgumentsCodec::to_wire(&native);
        let (untouched, _) = ToolArgumentsCodec::write_back(
            &native,
            &before,
            &json!({"a": 1.0, "b": "x", "c": [1]}),
        )
        .unwrap();
        assert!(untouched.is_none(), "1 == 1.0 is untouched");
        let (merged, effective) =
            ToolArgumentsCodec::write_back(&native, &before, &json!({"a": 2, "b": "x", "d": true}))
                .unwrap();
        assert_eq!(merged.unwrap(), json!({"a": 2, "b": "x", "d": true}));
        assert_eq!(effective.len(), 3);
        assert!(ToolArgumentsCodec::write_back(&native, &before, &json!("x")).is_err());
        assert_eq!(
            arguments_to_wire(&json!("not json")),
            Map::from_iter([("raw_arguments".into(), json!("not json"))])
        );
        assert_eq!(
            arguments_to_wire(&json!("{\"k\": 1}")),
            Map::from_iter([("k".into(), json!(1))])
        );
    }

    #[test]
    fn message_list_write_back_matches_by_identity_not_position() {
        let mut first = Message::user("one");
        first.message_id = Some("m1".into());
        let mut third = Message::user("three");
        third.message_id = Some("m3".into());
        let originals = vec![first, Message::user("two"), third];
        let before = messages_to_wire(&originals);
        // Remove the middle message: "three" must keep its identity.
        let after = json!([before[0], before[2]]);
        let out = write_back_message_list(originals.clone(), &before, &after, "input").unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].message_id.as_deref(), Some("m3"));
        // Modify the first in place: identity is kept, contents change.
        let after = json!([{"role": "user", "content": "ONE"}, before[1], before[2]]);
        let out = write_back_message_list(originals.clone(), &before, &after, "input").unwrap();
        assert_eq!(out[0].message_id.as_deref(), Some("m1"));
        assert_eq!(out[0].text(), "ONE");
        // A role change becomes a new message.
        let after = json!([{"role": "system", "content": "one"}, before[1], before[2]]);
        let out = write_back_message_list(originals.clone(), &before, &after, "input").unwrap();
        // The role-changed original is replaced, not kept.
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].message_id, None);
        assert_eq!(out[0].role, Role::system());
        assert!(
            write_back_message_list(originals, &before, &json!([{"role": "user"}]), "input")
                .is_err()
        );
    }

    #[test]
    fn model_response_codec_projects_tool_calls_separately_from_hosted_content() {
        let mut resp = ChatResponse::default();
        resp.messages.push(Message::with_contents(
            Role::assistant(),
            vec![
                Content::text("calling"),
                Content::FunctionCall(FunctionCallContent::new(
                    "c1",
                    "weather",
                    Some(FunctionArguments::Raw("{\"city\":\"x\"}".into())),
                )),
                Content::HostedFile(crate::types::HostedFileContent {
                    file_id: "file-1".into(),
                }),
            ],
        ));
        let wire = ModelResponseCodec::to_wire(&resp);
        assert_eq!(
            wire["tool_calls"],
            json!([{"id": "c1", "name": "weather", "args": {"city": "x"}}])
        );
        assert_eq!(wire["content"][0]["content"][1]["type"], "hosted_file");
        assert_eq!(wire["finish_reason"], "stop");

        // Rename the call, rewrite its args, add one, change finish_reason.
        let mut after = wire.clone();
        after["tool_calls"] = json!([
            {"id": "c1", "name": "forecast", "args": {"city": "y"}},
            {"id": "c2", "name": "added", "args": {}}
        ]);
        after["finish_reason"] = json!("tool_calls");
        assert!(ModelResponseCodec::write_back(&mut resp, &wire, &after).unwrap());
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "forecast");
        assert_eq!(calls[0].parse_arguments().unwrap()["city"], "y");
        assert_eq!(resp.finish_reason.as_ref().unwrap().as_str(), "tool_calls");

        let before = ModelResponseCodec::to_wire(&resp);
        let mut bad = before.clone();
        bad["tool_calls"][0]["args"] = json!("x");
        assert!(ModelResponseCodec::write_back(&mut resp.clone(), &before, &bad).is_err());
        let mut bad = before.clone();
        bad["finish_reason"] = json!(1);
        assert!(ModelResponseCodec::write_back(&mut resp.clone(), &before, &bad).is_err());

        // Content rewrite keeps the host-executed calls.
        let mut after = before.clone();
        after["content"] = json!("redacted");
        assert!(ModelResponseCodec::write_back(&mut resp, &before, &after).unwrap());
        assert_eq!(resp.messages.len(), 1);
        assert_eq!(resp.text(), "redacted");
        assert_eq!(resp.function_calls().len(), 2);
    }

    #[test]
    fn wire_equality_distinguishes_bool_from_number() {
        assert!(!wire_equal(&json!(1), &json!(true)));
        assert!(!wire_equal(&json!({"a": 0}), &json!({"a": false})));
        assert!(wire_equal(&json!([1, {"b": 2.0}]), &json!([1.0, {"b": 2}])));
    }

    #[test]
    fn output_and_tool_result_codecs_treat_untouched_targets_as_no_ops() {
        let mut resp = AgentResponse {
            messages: vec![Message::assistant("hello")],
            ..Default::default()
        };
        let before = OutputCodec::to_wire(&resp);
        assert_eq!(before, json!("hello"));
        assert!(
            !OutputCodec::write_back(&mut resp, &before, &json!({"content": "hello"})).unwrap()
        );
        assert!(!OutputCodec::write_back(&mut resp, &before, &Value::Null).unwrap());
        assert!(OutputCodec::write_back(&mut resp, &before, &json!({"content": "bye"})).unwrap());
        assert_eq!(resp.text(), "bye");
        assert!(OutputCodec::write_back(&mut resp, &before, &json!("x")).is_err());

        let original = Some(json!({"n": 1}));
        assert_eq!(
            ToolResultCodec::write_back(original.clone(), &json!({"n": 1}), &json!({"n": 1.0})),
            original
        );
        assert_eq!(
            ToolResultCodec::write_back(original, &json!({"n": 1}), &json!("redacted")),
            Some(json!("redacted"))
        );
    }

    #[test]
    fn usage_projects_integer_counts_only() {
        assert_eq!(usage_to_wire(None), None);
        assert_eq!(usage_to_wire(Some(&UsageDetails::default())), None);
        let usage = UsageDetails {
            input_token_count: Some(3),
            output_token_count: Some(4),
            ..Default::default()
        };
        assert_eq!(
            usage_to_wire(Some(&usage)),
            Some(json!({"input_token_count": 3, "output_token_count": 4}))
        );
    }
}
