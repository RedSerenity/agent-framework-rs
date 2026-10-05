//! Messages exchanged between declarative executors, chat-message
//! conversion, and JSON extraction from agent replies.
//!
//! Upstream routes typed Python objects (`ActionComplete`, `ConditionResult`,
//! `LoopIterationResult`, `LoopControl`) between executors and dispatches on
//! their class. The core Rust engine carries `serde_json::Value`s, so this
//! port tags internal messages with a `"$declarative"` discriminator; any
//! untagged value reaching an executor is treated as raw workflow input.

use std::collections::BTreeSet;

use agent_framework_core::types::{Content, Message, Role};
use agent_framework_core::workflow::RequestResponse;
use serde_json::{json, Value as Json};

/// The discriminator key of internal declarative messages.
pub const MESSAGE_TAG: &str = "$declarative";

/// `ActionComplete` — continue to the next action.
pub(crate) fn action_complete() -> Json {
    json!({ MESSAGE_TAG: "ActionComplete" })
}

/// Index of the else/default branch in a [`condition_result`].
pub(crate) const ELSE_BRANCH_INDEX: i64 = -1;

/// `ConditionResult` — which If/ConditionGroup branch matched.
pub(crate) fn condition_result(matched: bool, branch_index: i64) -> Json {
    json!({ MESSAGE_TAG: "ConditionResult", "matched": matched, "branch_index": branch_index })
}

/// `LoopIterationResult` — whether a Foreach has another item.
pub(crate) fn loop_iteration(has_next: bool, current_index: usize) -> Json {
    json!({ MESSAGE_TAG: "LoopIterationResult", "has_next": has_next, "current_index": current_index })
}

/// `LoopControl` — `break` / `continue` from a loop body.
pub(crate) fn loop_control(action: &str) -> Json {
    json!({ MESSAGE_TAG: "LoopControl", "action": action })
}

/// The tag of an internal message, if `v` is one.
pub(crate) fn tag(v: &Json) -> Option<&str> {
    v.get(MESSAGE_TAG).and_then(Json::as_str)
}

/// Edge predicate: a `ConditionResult` for `branch`.
pub(crate) fn is_branch(v: &Json, branch: i64) -> bool {
    tag(v) == Some("ConditionResult")
        && v.get("branch_index").and_then(Json::as_i64) == Some(branch)
}

/// Edge predicate: a `LoopIterationResult` with `has_next == expected`.
pub(crate) fn is_loop_result(v: &Json, expected: bool) -> bool {
    tag(v) == Some("LoopIterationResult")
        && v.get("has_next").and_then(Json::as_bool) == Some(expected)
}

/// What an executor received.
#[derive(Debug)]
pub(crate) enum Incoming {
    /// An internal declarative message (`ActionComplete`, …).
    Internal(Json),
    /// A response to a request this executor issued.
    Response(RequestResponse),
    /// Raw workflow input.
    Input(Json),
}

pub(crate) fn classify(message: Json) -> Incoming {
    if tag(&message).is_some() {
        return Incoming::Internal(message);
    }
    if let Json::Object(map) = &message {
        let keys: BTreeSet<&str> = map.keys().map(String::as_str).collect();
        if keys == BTreeSet::from(["request_id", "data", "original_request"]) {
            if let Some(resp) = RequestResponse::from_message(&message) {
                return Incoming::Response(resp);
            }
        }
    }
    Incoming::Input(message)
}

/// Whether `v` is a serialized core [`Message`] (`role` + `contents`).
pub(crate) fn is_chat_message(v: &Json) -> bool {
    v.as_object().is_some_and(|m| {
        m.get("role").is_some_and(Json::is_string) && m.get("contents").is_some_and(Json::is_array)
    }) && serde_json::from_value::<Message>(v.clone()).is_ok()
}

/// Serialize a message for storage in workflow state.
pub(crate) fn message_to_json(m: &Message) -> Json {
    serde_json::to_value(m).unwrap_or(Json::Null)
}

/// Convert a stored value back to a chat message.
///
/// Accepts serialized core messages, `{role, text}` records (what
/// `UserMessage()` produces), `{role, content}` records, and bare strings
/// (as user messages). Anything else yields `None`.
pub(crate) fn json_to_message(v: &Json) -> Option<Message> {
    match v {
        Json::String(s) => Some(Message::new(Role::new(Role::USER), s.clone())),
        Json::Object(map) => {
            if map.contains_key("contents") {
                if let Ok(m) = serde_json::from_value::<Message>(v.clone()) {
                    return Some(m);
                }
            }
            let role = map
                .get("role")
                .and_then(Json::as_str)
                .map(|r| r.to_ascii_lowercase())
                .unwrap_or_else(|| Role::USER.to_string());
            let text = ["text", "Text", "content"]
                .iter()
                .find_map(|k| map.get(*k))
                .map(|t| match t {
                    Json::String(s) => s.clone(),
                    Json::Null => String::new(),
                    other => other.to_string(),
                })?;
            Some(Message::new(Role::new(role), text))
        }
        _ => None,
    }
}

/// The text of a stored message value (record or serialized message).
pub(crate) fn json_message_text(v: &Json) -> String {
    match v {
        Json::String(s) => s.clone(),
        _ => json_to_message(v).map(|m| m.text()).unwrap_or_default(),
    }
}

/// Function-call contents of a list of messages, serialized.
pub(crate) fn function_calls(messages: &[Message]) -> Vec<Json> {
    messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter(|c| matches!(c, Content::FunctionCall(_)))
        .map(|c| serde_json::to_value(c).unwrap_or(Json::Null))
        .collect()
}

// ---------------------------------------------------------------------------
// JSON extraction from agent text (port of `_extract_json_from_response`)
// ---------------------------------------------------------------------------

const CODE_FENCE: &str = "```";
const JSON_QUALIFIER: &str = "json";
const MAX_JSON_DECODE_BUDGET_MULTIPLIER: usize = 4;

/// Extract JSON from an agent reply: the whole text, else the last decodable
/// ```` ```json ```` fence, else the last decodable plain fence, else the last
/// decodable object/array embedded in prose. Empty text yields `Ok(None)`;
/// text with no JSON is an error.
pub(crate) fn extract_json_from_response(text: &str) -> Result<Option<Json>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    if let Ok(v) = serde_json::from_str::<Json>(text) {
        return Ok(Some(v));
    }
    for require_qualifier in [true, false] {
        let mut last = None;
        for block in fenced_blocks(text, require_qualifier) {
            if let Ok(v) = serde_json::from_str::<Json>(block) {
                last = Some(v);
            }
        }
        if last.is_some() {
            return Ok(last);
        }
    }
    find_last_decodable_json(text)
        .map(Some)
        .ok_or_else(|| "No valid JSON found in response".to_string())
}

fn fenced_blocks(text: &str, require_json_qualifier: bool) -> Vec<&str> {
    let mut out = Vec::new();
    let mut search_start = 0;
    loop {
        let Some(rel) = text[search_start..].find(CODE_FENCE) else {
            return out;
        };
        let opening = search_start + rel;
        let mut content_start = opening + CODE_FENCE.len();
        if require_json_qualifier {
            if !text[content_start..].starts_with(JSON_QUALIFIER) {
                search_start = content_start;
                continue;
            }
            let qualifier_end = content_start + JSON_QUALIFIER.len();
            if let Some(next) = text[qualifier_end..].chars().next() {
                if !next.is_whitespace()
                    && next != '{'
                    && next != '['
                    && !text[qualifier_end..].starts_with(CODE_FENCE)
                {
                    search_start = content_start;
                    continue;
                }
            }
            content_start = qualifier_end;
        }
        while let Some(c) = text[content_start..].chars().next() {
            if c.is_whitespace() {
                content_start += c.len_utf8();
            } else {
                break;
            }
        }
        let Some(rel_close) = text[content_start..].find(CODE_FENCE) else {
            return out;
        };
        let closing = content_start + rel_close;
        out.push(text[content_start..closing].trim());
        search_start = closing + CODE_FENCE.len();
    }
}

fn escaped_quotes(bytes: &[u8]) -> Vec<bool> {
    let mut escaped = vec![false; bytes.len()];
    let mut backslashes = 0usize;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\\' {
            backslashes += 1;
            continue;
        }
        if *b == b'"' && backslashes % 2 == 1 {
            escaped[i] = true;
        }
        backslashes = 0;
    }
    escaped
}

fn candidates_forward(bytes: &[u8], escaped: &[bool]) -> BTreeSet<(usize, usize)> {
    let mut out = BTreeSet::new();
    let (mut objects, mut arrays): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut in_string = false;
    for (i, b) in bytes.iter().enumerate() {
        if objects.is_empty() && arrays.is_empty() {
            match b {
                b'{' => objects.push(i),
                b'[' => arrays.push(i),
                _ => {}
            }
            continue;
        }
        if *b == b'"' && !escaped[i] {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match b {
            b'{' => objects.push(i),
            b'[' => arrays.push(i),
            b'}' => {
                if let Some(s) = objects.pop() {
                    out.insert((s, i));
                }
            }
            b']' => {
                if let Some(s) = arrays.pop() {
                    out.insert((s, i));
                }
            }
            _ => {}
        }
    }
    out
}

fn candidates_reverse(bytes: &[u8], escaped: &[bool]) -> BTreeSet<(usize, usize)> {
    let mut out = BTreeSet::new();
    let (mut objects, mut arrays): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut in_string = false;
    for i in (0..bytes.len()).rev() {
        let b = bytes[i];
        if objects.is_empty() && arrays.is_empty() {
            match b {
                b'}' => objects.push(i),
                b']' => arrays.push(i),
                _ => {}
            }
            continue;
        }
        if b == b'"' && !escaped[i] {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match b {
            b'}' => objects.push(i),
            b']' => arrays.push(i),
            b'{' => {
                if let Some(e) = objects.pop() {
                    out.insert((i, e));
                }
            }
            b'[' => {
                if let Some(e) = arrays.pop() {
                    out.insert((i, e));
                }
            }
            _ => {}
        }
    }
    out
}

fn decode(text: &str, start: usize, end: usize) -> Option<Json> {
    serde_json::from_str::<Json>(text.get(start..=end)?).ok()
}

/// A group of overlapping candidate ranges: `(start, end, members)`.
type CandidateGroup = (usize, usize, Vec<(usize, usize)>);

fn find_last_decodable_json(text: &str) -> Option<Json> {
    let bytes = text.as_bytes();
    let escaped = escaped_quotes(bytes);
    let mut candidates = candidates_forward(bytes, &escaped);
    candidates.extend(candidates_reverse(bytes, &escaped));

    let mut groups: Vec<CandidateGroup> = Vec::new();
    for c in candidates {
        match groups.last_mut() {
            Some(g) if c.0 <= g.1 => {
                g.2.push(c);
                g.1 = g.1.max(c.1);
            }
            _ => groups.push((c.0, c.1, vec![c])),
        }
    }

    for (group_start, group_end, group) in groups.iter().rev() {
        let span = group_end - group_start + 1;
        let mut primary_budget = span * (MAX_JSON_DECODE_BUDGET_MULTIPLIER / 2);
        let mut recovery_budget =
            span * (MAX_JSON_DECODE_BUDGET_MULTIPLIER - MAX_JSON_DECODE_BUDGET_MULTIPLIER / 2);
        let mut attempted: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut last_json: Option<Json> = None;
        let mut consumed_end: Option<usize> = None;
        let mut idx = 0;
        while idx < group.len() && primary_budget > 0 {
            let (s, e) = group[idx];
            idx += 1;
            if consumed_end.is_some_and(|c| s <= c) {
                continue;
            }
            let len = e - s + 1;
            if len > primary_budget {
                continue;
            }
            primary_budget -= len;
            attempted.insert((s, e));
            let Some(v) = decode(text, s, e) else {
                continue;
            };
            last_json = Some(v);
            consumed_end = Some(e);
            while idx < group.len() && group[idx].0 <= e {
                idx += 1;
            }
        }

        let mut recovery: Vec<(usize, usize)> = group.clone();
        recovery.sort_by(|a, b| {
            (a.1 - a.0, std::cmp::Reverse(a.0)).cmp(&(b.1 - b.0, std::cmp::Reverse(b.0)))
        });
        let mut recovered: Option<(Json, (usize, usize))> = None;
        for (s, e) in recovery {
            if recovery_budget == 0 {
                break;
            }
            if attempted.contains(&(s, e)) || consumed_end.is_some_and(|c| s <= c) {
                continue;
            }
            let len = e - s + 1;
            if len > recovery_budget {
                continue;
            }
            recovery_budget -= len;
            let Some(v) = decode(text, s, e) else {
                continue;
            };
            let take = match &recovered {
                None => true,
                Some((_, (rs, re))) => {
                    let candidate_contains = s <= *rs && e >= *re;
                    let recovered_contains = *rs <= s && *re >= e;
                    candidate_contains || (!recovered_contains && s > *rs)
                }
            };
            if take {
                recovered = Some((v, (s, e)));
            }
        }
        if let Some((v, _)) = recovered {
            return Some(v);
        }
        if last_json.is_some() {
            return last_json;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_messages_round_trip_through_classify() {
        assert!(matches!(classify(action_complete()), Incoming::Internal(_)));
        assert!(matches!(classify(json!("hi")), Incoming::Input(_)));
        assert!(matches!(
            classify(json!({"request_id": "r", "data": 1, "original_request": {}})),
            Incoming::Response(_)
        ));
        assert!(is_branch(&condition_result(true, 2), 2));
        assert!(!is_branch(&condition_result(true, 2), ELSE_BRANCH_INDEX));
        assert!(is_loop_result(&loop_iteration(false, 0), false));
    }

    #[test]
    fn message_conversions() {
        let m = json_to_message(&json!({"role": "user", "text": "hi"})).unwrap();
        assert_eq!(m.role.0, "user");
        assert_eq!(m.text(), "hi");
        let stored = message_to_json(&Message::new(Role::new("assistant"), "yo"));
        assert!(is_chat_message(&stored));
        assert_eq!(json_message_text(&stored), "yo");
        assert_eq!(json_to_message(&json!("plain")).unwrap().role.0, "user");
        assert!(json_to_message(&json!(5)).is_none());
    }

    #[test]
    fn json_extraction_formats() {
        let cases = [
            (r#"{"a": 1}"#, json!({"a": 1})),
            ("```json\n{\"a\": 2}\n```", json!({"a": 2})),
            ("```\n[1, 2]\n```", json!([1, 2])),
            (
                r#"Here is the result: {"a": 3} hope it helps"#,
                json!({"a": 3}),
            ),
            (r#"first {"a": 1} then {"a": 4}"#, json!({"a": 4})),
            (
                "```json\n{\"a\": 5}\n``` and ```\n{\"a\": 6}\n```",
                json!({"a": 5}),
            ),
            (
                r#"text {"s": "with } brace and \" quote"} end"#,
                json!({"s": "with } brace and \" quote"}),
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(
                extract_json_from_response(text).unwrap(),
                Some(expected),
                "{text}"
            );
        }
        assert_eq!(extract_json_from_response("   ").unwrap(), None);
        assert!(extract_json_from_response("no json here").is_err());
        assert!(extract_json_from_response("{broken").is_err());
    }
}
