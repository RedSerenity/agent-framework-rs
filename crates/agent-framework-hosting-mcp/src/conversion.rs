//! Conversion between native MCP values and Agent Framework run values.
//!
//! Ports upstream's `mcp_to_run` / `mcp_from_run`
//! (`agent_framework_hosting_mcp/_conversion.py`). MCP content blocks are
//! plain JSON values here, in the wire shape a `tools/call` result carries.

use agent_framework_core::agent::AgentRunOptions;
use agent_framework_core::types::{AgentResponse, ChatOptions, Content, Message, Role};
use base64::Engine as _;
use serde_json::{json, Map, Value};

use crate::McpHostError;

/// Arguments for an agent run, prepared from MCP tool arguments. Mirrors
/// upstream's `AgentRunArgs` (`stream` is always `false` for MCP).
#[derive(Debug, Clone, Default)]
pub struct McpRunArgs {
    /// The request as a single user message.
    pub messages: Vec<Message>,
    /// Chat options copied from the arguments named as option arguments.
    pub options: AgentRunOptions,
}

/// Convert MCP `tools/call` arguments into run arguments.
///
/// `argument_name` names the required string argument holding the request.
/// Only arguments listed in `chat_option_arguments` are copied into chat
/// options: a name matching a [`ChatOptions`] field sets it (`temperature`,
/// `top_p`, `max_tokens`, `seed`, `stop`, `user`, `instructions`, …) and any
/// other becomes a provider-specific `additional_properties` entry (a
/// `reasoning_effort`, say). Every other argument stays on the message's
/// `additional_properties["mcp_arguments"]` — visible to middleware, not
/// sent to the model — as upstream keeps it in `raw_representation`.
///
/// # Errors
/// [`McpHostError::InvalidArguments`] when the request argument is missing
/// or not a string, or an option argument has the wrong type for its field.
pub fn mcp_to_run(
    arguments: Option<&Map<String, Value>>,
    argument_name: &str,
    chat_option_arguments: &[&str],
) -> Result<McpRunArgs, McpHostError> {
    let invalid = |m: String| McpHostError::InvalidArguments(m);
    let arguments = arguments
        .filter(|a| a.contains_key(argument_name))
        .ok_or_else(|| {
            invalid(format!(
                "MCP tool arguments must include a '{argument_name}' string."
            ))
        })?;
    let text = arguments[argument_name].as_str().ok_or_else(|| {
        invalid(format!(
            "MCP tool argument '{argument_name}' must be a string."
        ))
    })?;

    let mut message = Message::user(text);
    message
        .additional_properties
        .insert("mcp_arguments".into(), Value::Object(arguments.clone()));

    let mut chat = ChatOptions::default();
    let mut any = false;
    for name in chat_option_arguments {
        if let Some(value) = arguments.get(*name) {
            apply_chat_option(&mut chat, name, value).map_err(invalid)?;
            any = true;
        }
    }
    let options = if any {
        AgentRunOptions::default().with_chat_options(chat)
    } else {
        AgentRunOptions::default()
    };
    Ok(McpRunArgs {
        messages: vec![message],
        options,
    })
}

/// Set one chat option from an MCP argument value.
fn apply_chat_option(chat: &mut ChatOptions, name: &str, value: &Value) -> Result<(), String> {
    let wrong = |ty: &str| format!("MCP tool argument '{name}' must be {ty}.");
    let f32_of = |v: &Value| {
        v.as_f64()
            .map(|f| f as f32)
            .ok_or_else(|| wrong("a number"))
    };
    let string_of = |v: &Value| {
        v.as_str()
            .map(str::to_string)
            .ok_or_else(|| wrong("a string"))
    };
    match name {
        "temperature" => chat.temperature = Some(f32_of(value)?),
        "top_p" => chat.top_p = Some(f32_of(value)?),
        "frequency_penalty" => chat.frequency_penalty = Some(f32_of(value)?),
        "presence_penalty" => chat.presence_penalty = Some(f32_of(value)?),
        "max_tokens" => {
            chat.max_tokens = Some(
                value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| wrong("a non-negative integer"))?,
            )
        }
        "seed" => chat.seed = Some(value.as_i64().ok_or_else(|| wrong("an integer"))?),
        "allow_multiple_tool_calls" => {
            chat.allow_multiple_tool_calls =
                Some(value.as_bool().ok_or_else(|| wrong("a boolean"))?)
        }
        "store" => chat.store = Some(value.as_bool().ok_or_else(|| wrong("a boolean"))?),
        "user" => chat.user = Some(string_of(value)?),
        "instructions" => chat.instructions = Some(string_of(value)?),
        "model" => chat.model = Some(string_of(value)?),
        "stop" => {
            chat.stop = Some(match value {
                Value::String(s) => vec![s.clone()],
                Value::Array(items) => items
                    .iter()
                    .map(|i| i.as_str().map(str::to_string))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| wrong("a string or an array of strings"))?,
                _ => return Err(wrong("a string or an array of strings")),
            })
        }
        other => {
            chat.additional_properties
                .insert(other.to_string(), value.clone());
        }
    }
    Ok(())
}

/// Convert a completed agent response into MCP content blocks.
///
/// Messages are flattened in order and user-role messages are skipped.
/// Text becomes `text`, a URI becomes a `resource_link`, and inline data
/// becomes `image`, `audio`, or an embedded `resource` blob by media type.
/// Content with no MCP counterpart (function calls, reasoning, usage, …) is
/// omitted with a warning, as upstream omits it.
///
/// # Errors
/// [`McpHostError::InvalidContent`] when data content is not a valid base64
/// data URI.
pub fn mcp_from_run(response: &AgentResponse) -> Result<Vec<Value>, McpHostError> {
    mcp_from_messages(&response.messages)
}

/// [`mcp_from_run`] for bare messages.
pub fn mcp_from_messages(messages: &[Message]) -> Result<Vec<Value>, McpHostError> {
    let mut blocks = Vec::new();
    for message in messages {
        if message.role == Role::user() {
            continue;
        }
        for content in &message.contents {
            match content {
                Content::Text(t) => blocks.push(json!({ "type": "text", "text": t.text })),
                Content::Uri(u) => {
                    let name = u
                        .uri
                        .rsplit('/')
                        .next()
                        .filter(|n| !n.is_empty())
                        .unwrap_or(&u.uri);
                    blocks.push(json!({
                        "type": "resource_link",
                        "name": name,
                        "uri": u.uri,
                        "mimeType": u.media_type,
                    }));
                }
                Content::Data(d) => blocks.push(data_block(&d.uri, d.media_type.as_deref())?),
                other => tracing::warn!(
                    content = ?std::mem::discriminant(other),
                    "content type is not supported in MCP tool results and was omitted"
                ),
            }
        }
    }
    Ok(blocks)
}

fn data_block(uri: &str, media_type: Option<&str>) -> Result<Value, McpHostError> {
    let invalid = || {
        McpHostError::InvalidContent(
            "Agent Framework data content must contain a base64 data URI.".into(),
        )
    };
    let (prefix, encoded) = uri.split_once(',').ok_or_else(invalid)?;
    if !prefix.starts_with("data:") || !prefix.contains(";base64") {
        return Err(invalid());
    }
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| {
            McpHostError::InvalidContent(
                "Agent Framework data content contains invalid base64 data.".into(),
            )
        })?;
    // The URI's own media type is the fallback when the content has none.
    let mime = media_type.map(str::to_string).or_else(|| {
        prefix
            .trim_start_matches("data:")
            .split(';')
            .next()
            .filter(|m| !m.is_empty())
            .map(str::to_string)
    });
    Ok(match mime.as_deref() {
        Some(m) if m.starts_with("image/") => {
            json!({ "type": "image", "data": encoded, "mimeType": m })
        }
        Some(m) if m.starts_with("audio/") => {
            json!({ "type": "audio", "data": encoded, "mimeType": m })
        }
        _ => {
            let mut resource = json!({ "uri": "af://binary", "blob": encoded });
            if let Some(m) = mime {
                resource["mimeType"] = json!(m);
            }
            json!({ "type": "resource", "resource": resource })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::{DataContent, UriContent};

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn the_request_argument_becomes_one_user_message() {
        let a = args(json!({ "task": "hello", "audience": "kids" }));
        let run = mcp_to_run(Some(&a), "task", &[]).unwrap();
        assert_eq!(run.messages.len(), 1);
        assert_eq!(run.messages[0].role, Role::user());
        assert_eq!(run.messages[0].text(), "hello");
        assert_eq!(
            run.messages[0].additional_properties["mcp_arguments"]["audience"],
            "kids"
        );
        assert!(run.options.chat_options.is_none());
    }

    #[test]
    fn missing_or_non_string_request_is_rejected() {
        assert!(mcp_to_run(None, "task", &[]).is_err());
        assert!(mcp_to_run(Some(&args(json!({}))), "task", &[]).is_err());
        assert!(mcp_to_run(Some(&args(json!({ "task": 3 }))), "task", &[]).is_err());
    }

    #[test]
    fn only_named_option_arguments_become_options() {
        let a = args(json!({
            "task": "t",
            "temperature": 0.25,
            "max_tokens": 9,
            "stop": "END",
            "reasoning_effort": "low",
            "secret": "not an option",
        }));
        let run = mcp_to_run(
            Some(&a),
            "task",
            &[
                "temperature",
                "max_tokens",
                "stop",
                "reasoning_effort",
                "absent",
            ],
        )
        .unwrap();
        let chat = run.options.chat_options.unwrap();
        assert_eq!(chat.temperature, Some(0.25));
        assert_eq!(chat.max_tokens, Some(9));
        assert_eq!(chat.stop, Some(vec!["END".to_string()]));
        assert_eq!(chat.additional_properties["reasoning_effort"], "low");
        assert!(!chat.additional_properties.contains_key("secret"));
    }

    #[test]
    fn a_mistyped_option_is_rejected() {
        let a = args(json!({ "task": "t", "temperature": "hot" }));
        assert!(mcp_from_run(&AgentResponse::default()).unwrap().is_empty());
        assert!(mcp_to_run(Some(&a), "task", &["temperature"]).is_err());
    }

    #[test]
    fn response_contents_map_to_blocks_and_user_turns_are_skipped() {
        let png = base64::engine::general_purpose::STANDARD.encode(b"png");
        let response = AgentResponse {
            messages: vec![
                Message::user("ignored"),
                Message::with_contents(
                    Role::assistant(),
                    vec![
                        Content::text("hi"),
                        Content::Uri(UriContent {
                            uri: "https://x.test/a/report.pdf".into(),
                            media_type: "application/pdf".into(),
                        }),
                        Content::Data(DataContent {
                            uri: format!("data:image/png;base64,{png}"),
                            media_type: Some("image/png".into()),
                        }),
                        Content::Data(DataContent {
                            uri: format!("data:audio/wav;base64,{png}"),
                            media_type: None,
                        }),
                        Content::Data(DataContent {
                            uri: format!("data:application/zip;base64,{png}"),
                            media_type: Some("application/zip".into()),
                        }),
                    ],
                ),
            ],
            ..Default::default()
        };
        let blocks = mcp_from_run(&response).unwrap();
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[0], json!({ "type": "text", "text": "hi" }));
        assert_eq!(blocks[1]["type"], "resource_link");
        assert_eq!(blocks[1]["name"], "report.pdf");
        assert_eq!(
            blocks[2],
            json!({ "type": "image", "data": png, "mimeType": "image/png" })
        );
        assert_eq!(blocks[3]["type"], "audio");
        assert_eq!(blocks[3]["mimeType"], "audio/wav");
        assert_eq!(blocks[4]["type"], "resource");
        assert_eq!(blocks[4]["resource"]["mimeType"], "application/zip");
    }

    #[test]
    fn invalid_data_uris_are_errors() {
        for uri in [
            "not-a-data-uri",
            "data:image/png,plain",
            "data:image/png;base64,!!!",
        ] {
            let m = Message::with_contents(
                Role::assistant(),
                vec![Content::Data(DataContent {
                    uri: uri.into(),
                    media_type: Some("image/png".into()),
                })],
            );
            assert!(mcp_from_messages(&[m]).is_err(), "{uri}");
        }
    }
}
