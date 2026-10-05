//! Small shared helpers: timestamps and id generation.

use agent_framework_core::types::{Content, FunctionArguments, FunctionCallContent, Message};

use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

/// Unix time in fractional seconds (OpenAI `created_at` convention).
pub(crate) fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A short hex fragment for synthesized ids.
pub(crate) fn short_hex() -> String {
    Uuid::new_v4().simple().to_string()[..8].to_string()
}

/// A `msg_…` id (OpenAI message-item id convention).
pub(crate) fn msg_id() -> String {
    format!("msg_{}", short_hex())
}

/// The **unresolved** function calls of a response, in order.
///
/// A declaration-only call is one the *caller* is expected to execute: core
/// deliberately leaves [`FunctionCallContent`] intact rather than resolving
/// it, so a host that serializes only text drops the id, name and arguments
/// the client needs and leaves it nothing to act on.
///
/// A call that already carries a `FunctionResultContent` is *not* one of
/// those, and must not be re-advertised. Core keeps both in the response —
/// deliberately, and `FunctionInvokingChatClient` filters the same way for
/// the same reason: a provider that ran a hosted tool itself (an Anthropic
/// server-side web search, a hosted MCP call) returns the call *and* its
/// result together, and an agent that ran a local tool leaves the pair
/// behind too. Serializing those to the client would ask it to re-execute
/// work that is already done — duplicating whatever side effects it had.
///
/// A call in one message is resolved by a result in any other, which is
/// the usual shape once a tool round trip has been folded back into the
/// conversation.
pub(crate) fn function_calls_of(messages: &[Message]) -> Vec<&FunctionCallContent> {
    let resolved: HashSet<&str> = messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(Content::as_function_result)
        .map(|r| r.call_id.as_str())
        .collect();
    messages
        .iter()
        .flat_map(|m| unresolved(&m.contents, |id| resolved.contains(id)))
        .collect()
}

/// Function calls accumulated across a stream, emitted only once it ends.
///
/// The streaming surfaces cannot filter a resolved call the way the
/// buffered ones do, because the result arrives *after* the call: with
/// local tools, `FunctionInvokingChatClient::get_streaming_response` runs
/// the whole loop and then replays each message as its own update, so the
/// call update always precedes the tool-result update that answers it. A
/// per-update filter would have already put the call on the wire, and no
/// later event can recall it — leaving the client to re-execute work the
/// agent already did.
///
/// So calls are held here until the stream ends and it is known which are
/// genuinely outstanding. The cost is that a call's arguments reach the
/// client at the end of the stream rather than forming incrementally,
/// which is a presentation detail: a client cannot execute a call until it
/// has the whole argument object anyway. Duplicating a tool's side effects
/// is not a presentation detail, so the trade goes this way.
#[derive(Default)]
pub(crate) struct StreamingCalls {
    /// `(call_id, name, arguments)`, in the order the calls first appeared.
    calls: Vec<(String, String, String)>,
    resolved: HashSet<String>,
}

impl StreamingCalls {
    /// Fold one update's contents in: results mark their call resolved,
    /// and a call either starts a new entry or appends its argument
    /// fragment to the entry it shares a `call_id` with.
    pub(crate) fn push(&mut self, contents: &[Content]) {
        for result in contents.iter().filter_map(Content::as_function_result) {
            self.resolved.insert(result.call_id.clone());
        }
        for call in contents.iter().filter_map(Content::as_function_call) {
            let fragment = arguments_delta(call).unwrap_or_default();
            match self.calls.iter_mut().find(|(id, _, _)| *id == call.call_id) {
                Some((_, name, arguments)) => {
                    // A fragment update carries no name — this repository's
                    // own Responses parser sends `""` — so the one from the
                    // announcement is kept.
                    if name.is_empty() {
                        name.clone_from(&call.name);
                    }
                    arguments.push_str(&fragment);
                }
                None => self
                    .calls
                    .push((call.call_id.clone(), call.name.clone(), fragment)),
            }
        }
    }

    /// The calls still unanswered when the stream ended, in first-seen
    /// order, each with its arguments reassembled.
    pub(crate) fn outstanding(&self) -> Vec<(&str, &str, &str)> {
        self.calls
            .iter()
            .filter(|(id, _, _)| !self.resolved.contains(id))
            .map(|(id, name, arguments)| (id.as_str(), name.as_str(), arguments.as_str()))
            .collect()
    }
}

fn unresolved(
    contents: &[Content],
    is_resolved: impl Fn(&str) -> bool,
) -> Vec<&FunctionCallContent> {
    contents
        .iter()
        .filter_map(Content::as_function_call)
        .filter(|call| !is_resolved(&call.call_id))
        .collect()
}

/// A call's arguments as the **JSON string** both OpenAI surfaces expect.
///
/// `FunctionArguments` is either a raw string — which is what a provider
/// streams, possibly a fragment — or a parsed object, which has to be
/// re-serialized. Absent arguments become `{}` rather than `null`, since the
/// wire field is typed as a string and clients `JSON.parse` it.
pub(crate) fn arguments_string(call: &FunctionCallContent) -> String {
    arguments_delta(call).unwrap_or_else(|| "{}".to_string())
}

/// The arguments to send as a streaming **delta**, or `None` when the call
/// carries none yet.
///
/// A streamed call is often announced before any arguments exist — this
/// repository's own Responses parser builds exactly that from
/// `response.output_item.added`, a `FunctionCallContent` with
/// `arguments: None`. Sending [`arguments_string`]'s `"{}"` for it would
/// put a literal `{}` at the head of the fragment sequence, and a client
/// concatenating deltas would parse `{}{"city":"Oslo"}` — invalid JSON.
/// `"{}"` is right for the *buffered* surfaces, where it is the whole value
/// rather than the first of several, so only the delta case distinguishes
/// the two.
pub(crate) fn arguments_delta(call: &FunctionCallContent) -> Option<String> {
    match &call.arguments {
        Some(FunctionArguments::Raw(raw)) => Some(raw.clone()),
        Some(FunctionArguments::Object(map)) => {
            Some(serde_json::to_string(map).unwrap_or_else(|_| "{}".to_string()))
        }
        None => None,
    }
}
