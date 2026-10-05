//! The [`AgentSpec`] type and its nested spec model.
//!
//! Field names mirror the official Microsoft Agent Framework agent schema
//! (the Python `PromptAgent` vocabulary): `kind`, `name`, `description`,
//! `instructions`, `model` (`id`/`provider`/`apiType`/`connection`/`options`),
//! `tools`, `outputSchema`, etc. Keys are camelCase, matching the YAML samples
//! under `agent-samples/` in the upstream repository.

use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::collections::BTreeMap;

use crate::error::{DeclarativeError, Result};

/// A declarative agent specification (`kind: Prompt` / `kind: SupportsAgentRun`).
///
/// Round-trips losslessly through [`AgentSpec::from_yaml`] / [`AgentSpec::to_yaml`].
/// Environment-variable interpolation is applied by the
/// [`DeclarativeLoader`](crate::DeclarativeLoader), not by `from_yaml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AgentSpec {
    /// The agent kind. Supported: `Prompt`, `SupportsAgentRun` (case-insensitive).
    pub kind: String,
    /// The agent name (exposed to callers and used as the tool name in `as_tool`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// An optional human-friendly display name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// A description of what the agent does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The system prompt / instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Instructions appended after [`AgentSpec::instructions`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_instructions: Option<String>,
    /// Free-form metadata carried alongside the agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, JsonValue>>,
    /// The model / chat-client configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelSpec>,
    /// The tools available to the agent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpec>,
    /// An input schema (parsed but not currently enforced at runtime).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<PropertySchema>,
    /// A structured-output schema, mapped to the chat `response_format`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<PropertySchema>,
    /// The prompt template (`format`/`parser`), parsed for schema parity
    /// with upstream's `PromptAgent.template` but not interpreted (upstream's
    /// loader does not use it either).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<TemplateSpec>,
}

/// The `template` block of a prompt agent (upstream `Template`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TemplateSpec {
    /// The template format (`kind`, `strict`, `options`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<TemplateFormatSpec>,
    /// The template parser (`kind`, `options`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parser: Option<TemplateParserSpec>,
}

/// A template `format` (upstream `Format`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TemplateFormatSpec {
    /// The format kind (e.g. `mustache`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Whether the format is strict.
    #[serde(default, skip_serializing_if = "is_false")]
    pub strict: bool,
    /// Format-specific options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<BTreeMap<String, JsonValue>>,
}

/// A template `parser` (upstream `Parser`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TemplateParserSpec {
    /// The parser kind (e.g. `prompty`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Parser-specific options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<BTreeMap<String, JsonValue>>,
}

/// Accept `properties` either as a map (`{name: {...}}`) or as upstream's
/// list form (`[{name: ..., kind: ...}]`).
fn properties_map_or_list<'de, D>(
    d: D,
) -> std::result::Result<BTreeMap<String, PropertySpec>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let raw = JsonValue::deserialize(d)?;
    properties_from_json(raw).map_err(D::Error::custom)
}

fn optional_properties_map_or_list<'de, D>(
    d: D,
) -> std::result::Result<Option<BTreeMap<String, PropertySpec>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let raw = Option::<JsonValue>::deserialize(d)?;
    match raw {
        None | Some(JsonValue::Null) => Ok(None),
        Some(v) => properties_from_json(v).map(Some).map_err(D::Error::custom),
    }
}

fn properties_from_json(
    raw: JsonValue,
) -> std::result::Result<BTreeMap<String, PropertySpec>, String> {
    match raw {
        JsonValue::Null => Ok(BTreeMap::new()),
        JsonValue::Object(map) => map
            .into_iter()
            .map(|(k, v)| {
                serde_json::from_value::<PropertySpec>(v)
                    .map(|p| (k.clone(), p))
                    .map_err(|e| format!("property {k:?}: {e}"))
            })
            .collect(),
        JsonValue::Array(items) => items
            .into_iter()
            .map(|item| {
                let JsonValue::Object(mut m) = item else {
                    return Err("list-form properties must be mappings with a 'name'".to_string());
                };
                let name = match m.remove("name") {
                    Some(JsonValue::String(n)) => n,
                    _ => return Err("list-form property is missing 'name'".to_string()),
                };
                serde_json::from_value::<PropertySpec>(JsonValue::Object(m))
                    .map(|p| (name.clone(), p))
                    .map_err(|e| format!("property {name:?}: {e}"))
            })
            .collect(),
        other => Err(format!(
            "'properties' must be a mapping or a list, found {other}"
        )),
    }
}

impl AgentSpec {
    /// Parse an [`AgentSpec`] from YAML (or JSON — JSON is valid YAML) without
    /// environment interpolation.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).map_err(|e| DeclarativeError::Parse(e.to_string()))
    }

    /// Serialize this spec back to YAML.
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self).map_err(|e| DeclarativeError::Serialize(e.to_string()))
    }

    /// The effective display name: `name`, else `displayName`.
    pub fn effective_name(&self) -> Option<&str> {
        self.name.as_deref().or(self.display_name.as_deref())
    }

    /// The combined instructions (`instructions` then `additionalInstructions`).
    pub fn combined_instructions(&self) -> Option<String> {
        match (&self.instructions, &self.additional_instructions) {
            (Some(a), Some(b)) => Some(format!("{a}\n{b}")),
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        }
    }

    /// Validate that `kind` names a supported agent type, returning it normalized.
    pub(crate) fn validate_kind(&self) -> Result<()> {
        match self.kind.to_ascii_lowercase().as_str() {
            "prompt" | "agent" => Ok(()),
            _ => Err(DeclarativeError::UnsupportedKind {
                what: "agent",
                kind: self.kind.clone(),
                expected: vec!["Prompt", "Agent"],
            }),
        }
    }
}

/// The `model` block: identifier, provider routing, connection, and options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelSpec {
    /// The model / deployment id (e.g. `gpt-4.1-mini`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The provider name (e.g. `OpenAI`, `AzureOpenAI`, `Anthropic`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// The API surface (e.g. `Chat`, `Responses`, `Assistants`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_type: Option<String>,
    /// Connection details handed to the client factory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection: Option<ConnectionSpec>,
    /// Generation options mapped onto `ChatOptions`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<ModelOptions>,
}

impl ModelSpec {
    /// The provider-routing key: `provider.apiType`, else `provider`, else `None`.
    pub fn provider_key(&self) -> Option<String> {
        match (&self.provider, &self.api_type) {
            (Some(p), Some(a)) => Some(format!("{p}.{a}")),
            (Some(p), None) => Some(p.clone()),
            _ => None,
        }
    }
}

/// Generation options. Field names are camelCase per the schema
/// (`maxOutputTokens`, `topP`, `chatToolMode`, …). Unknown inline keys are
/// rejected; provider-specific extras belong under `additionalProperties`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelOptions {
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling (carried in `additional_properties` on `ChatOptions`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Frequency penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Maximum output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Deterministic sampling seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Whether multiple tool calls per turn are allowed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_multiple_tool_calls: Option<bool>,
    /// Tool-choice mode: `auto`, `required`, or `none`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_tool_mode: Option<String>,
    /// Provider-specific extra options, passed through to `ChatOptions`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub additional_properties: BTreeMap<String, JsonValue>,
}

/// A connection block. A flat union of every connection kind's fields (matching
/// the Python base `Connection` + subclasses); which fields are meaningful
/// depends on `kind` (`reference` / `remote` / `key` / `anonymous`). The
/// [`ChatClientFactory`](crate::ChatClientFactory) interprets it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConnectionSpec {
    /// The connection kind (case-insensitive): `reference`, `remote`, `key`,
    /// `apikey`, or `anonymous`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Authentication mode hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authentication_mode: Option<String>,
    /// Human description of how the connection is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_description: Option<String>,
    /// Named reference (for `reference`/`remote` connections).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Target of a `reference` connection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Endpoint URL (for `remote`/`key`/`anonymous`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// API key (`apiKey`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// API key (`key`; takes precedence over `apiKey` when both are present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

impl ConnectionSpec {
    /// The effective API key: `key` if present, else `apiKey`.
    pub fn resolved_key(&self) -> Option<&str> {
        self.key.as_deref().or(self.api_key.as_deref())
    }
}

/// A tool declaration. A flat union over every tool `kind` (matching the Python
/// base `Tool` + subclasses). Required fields are validated per-kind at
/// build time by the loader.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ToolSpec {
    /// The tool kind: `function`, `web_search`, `file_search`,
    /// `code_interpreter`, `mcp`, `custom`, or `openapi`.
    pub kind: String,
    /// The tool name exposed to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// A human/model-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Bindings mapping a logical name to a native tool key in the
    /// [`ToolRegistry`](crate::ToolRegistry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bindings: Option<BTreeMap<String, JsonValue>>,

    // --- function ---
    /// Parameter schema (function tools).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<PropertySchema>,
    /// Whether the function schema is strict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,

    // --- file_search ---
    /// Vector-store ids to search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_store_ids: Option<Vec<String>>,
    /// Maximum number of results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_result_count: Option<u32>,
    /// The ranker to use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ranker: Option<String>,
    /// Minimum score threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
    /// Search filters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<BTreeMap<String, JsonValue>>,

    // --- mcp ---
    /// MCP server name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_name: Option<String>,
    /// MCP server description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_description: Option<String>,
    /// MCP approval mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_mode: Option<ApprovalModeSpec>,
    /// MCP tools that are permitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    /// MCP server URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    // --- code_interpreter ---
    /// File ids available to the interpreter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_ids: Option<Vec<String>>,

    // --- custom / web_search / openapi ---
    /// Connection for provider-hosted tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection: Option<ConnectionSpec>,
    /// Free-form tool options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<BTreeMap<String, JsonValue>>,
    /// OpenAPI specification (url or inline document).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub specification: Option<String>,
}

/// An MCP approval mode: a bare string (`always` / `never` / `specify`) or a
/// detailed object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ApprovalModeSpec {
    /// A bare mode name.
    Named(String),
    /// A detailed approval mode with per-tool overrides.
    Detailed(ApprovalModeDetail),
}

impl ApprovalModeSpec {
    /// The mode kind (`always` / `never` / `specify`), if determinable.
    pub fn kind(&self) -> Option<&str> {
        match self {
            ApprovalModeSpec::Named(s) => Some(s.as_str()),
            ApprovalModeSpec::Detailed(d) => d.kind.as_deref(),
        }
    }
}

/// The detailed form of [`ApprovalModeSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovalModeDetail {
    /// The approval-mode kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Tools that always require approval (`specify` mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub always_require_approval_tools: Option<Vec<String>>,
    /// Tools that never require approval (`specify` mode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub never_require_approval_tools: Option<Vec<String>>,
}

/// A schema describing a set of named properties. Used for tool `parameters`
/// and agent `outputSchema`. Lenient about extra keys, matching the upstream
/// behavior of filtering stray `type`/`name`/`description` at this level.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PropertySchema {
    /// Whether the schema is strict (`additionalProperties: false`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub strict: bool,
    /// Illustrative examples.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<JsonValue>,
    /// Named properties (a map of property-name to its definition). Upstream's
    /// list form (`[{name: x, kind: string}]`) is accepted on input.
    #[serde(
        default,
        skip_serializing_if = "BTreeMap::is_empty",
        deserialize_with = "properties_map_or_list"
    )]
    pub properties: BTreeMap<String, PropertySpec>,
}

impl PropertySchema {
    /// Convert to a standard JSON Schema object:
    /// `{"type":"object","properties":{…},"required":[…]}`.
    ///
    /// Diverges slightly from the upstream `to_json_schema` (which keeps
    /// `required` inline on each property): here `required` is hoisted into the
    /// standard top-level array so the result is a conventional JSON Schema
    /// consumable by the core `ResponseFormat` and tool parameter machinery.
    pub fn to_json_schema(&self) -> JsonValue {
        let mut props = JsonMap::new();
        let mut required: Vec<JsonValue> = Vec::new();
        for (name, prop) in &self.properties {
            if prop.required == Some(true) {
                required.push(JsonValue::String(name.clone()));
            }
            props.insert(name.clone(), prop.to_json_schema());
        }
        let mut schema = JsonMap::new();
        schema.insert("type".into(), JsonValue::String("object".into()));
        schema.insert("properties".into(), JsonValue::Object(props));
        if !required.is_empty() {
            schema.insert("required".into(), JsonValue::Array(required));
        }
        if self.strict {
            schema.insert("additionalProperties".into(), JsonValue::Bool(false));
        }
        JsonValue::Object(schema)
    }
}

/// A single property definition within a [`PropertySchema`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PropertySpec {
    /// The JSON type: `string`, `number`, `integer`, `boolean`, `array`,
    /// `object`. Accepts both `kind:` and `type:` (the latter as an alias).
    #[serde(default, alias = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// A description of the property.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether the property is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    /// A default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<JsonValue>,
    /// An example value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub example: Option<JsonValue>,
    /// The set of permitted values.
    #[serde(rename = "enum", default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<JsonValue>>,
    /// The element schema for `array` properties.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<PropertySpec>>,
    /// Nested properties for `object` properties (map or upstream list form).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optional_properties_map_or_list"
    )]
    pub properties: Option<BTreeMap<String, PropertySpec>>,
}

impl PropertySpec {
    /// Convert this property to a JSON Schema fragment.
    pub fn to_json_schema(&self) -> JsonValue {
        let mut m = JsonMap::new();
        if let Some(kind) = &self.kind {
            m.insert("type".into(), JsonValue::String(kind.clone()));
        }
        if let Some(desc) = &self.description {
            m.insert("description".into(), JsonValue::String(desc.clone()));
        }
        if let Some(default) = &self.default {
            m.insert("default".into(), default.clone());
        }
        // Upstream drops empty `enum` placeholders.
        if let Some(en) = self.enum_values.as_ref().filter(|e| !e.is_empty()) {
            m.insert("enum".into(), JsonValue::Array(en.clone()));
        }
        if let Some(items) = &self.items {
            m.insert("items".into(), items.to_json_schema());
        }
        if let Some(properties) = &self.properties {
            let mut nested = JsonMap::new();
            let mut required: Vec<JsonValue> = Vec::new();
            for (name, prop) in properties {
                if prop.required == Some(true) {
                    required.push(JsonValue::String(name.clone()));
                }
                nested.insert(name.clone(), prop.to_json_schema());
            }
            m.insert("properties".into(), JsonValue::Object(nested));
            if !required.is_empty() {
                m.insert("required".into(), JsonValue::Array(required));
            }
        }
        // Upstream's `_normalize_schema_node`: strict structured outputs need
        // `additionalProperties: false` on every nested object node (chat
        // clients only add it at the root).
        if self.kind.as_deref() == Some("object") {
            m.entry("additionalProperties")
                .or_insert(JsonValue::Bool(false));
        }
        JsonValue::Object(m)
    }
}

fn is_false(b: &bool) -> bool {
    !*b
}
