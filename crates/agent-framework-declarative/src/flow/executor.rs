//! The executor node used for every declarative action, plus the basic,
//! control-flow, and conversation actions.
//!
//! Ports `_executors_basic.py` and `_executors_control_flow.py`; the agent,
//! external-input, tool, HTTP and MCP actions live in sibling modules as
//! further `impl DeclarativeExecutor` blocks.

use std::collections::HashMap;
use std::sync::Arc;

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::error::{Error as CoreError, Result as CoreResult};
use agent_framework_core::tools::Tool;
use agent_framework_core::types::{Content, Message, Role};
use agent_framework_core::workflow::{Executor, RequestResponse, WorkflowContext};
use async_trait::async_trait;
use serde_json::{json, Map, Value as Json};

use super::http::HttpRequestHandler;
use super::mcp::McpToolHandler;
use super::messages::{
    action_complete, classify, condition_result, is_chat_message, json_to_message, loop_control,
    loop_iteration, message_to_json, tag, Incoming, ELSE_BRANCH_INDEX,
};
use super::state::{py_str, py_truthy, DeclarativeState, StateConfig, StateError, LOOP_STATE_KEY};

/// Everything executors need at run time, shared by all nodes of a workflow.
pub(crate) struct Runtime {
    pub agents: HashMap<String, Arc<dyn SupportsAgentRun>>,
    pub tools: HashMap<String, Arc<dyn Tool>>,
    pub http: Option<Arc<dyn HttpRequestHandler>>,
    pub mcp: Option<Arc<dyn McpToolHandler>>,
    pub state: StateConfig,
}

/// The role a node plays in the graph.
#[derive(Debug, Clone)]
pub(crate) enum Node {
    /// Pass-through: the entry node, else/default passthroughs, loop exits,
    /// and `GotoAction` sources (upstream `JoinExecutor`).
    Join,
    /// Evaluates an `If` condition (upstream `IfConditionEvaluatorExecutor`).
    IfEval { condition: Json },
    /// Evaluates `ConditionGroup` conditions, first match wins.
    GroupEval { conditions: Vec<Json> },
    /// Starts a `Foreach` (upstream `ForeachInitExecutor`).
    ForeachInit,
    /// Advances a `Foreach` (upstream `ForeachNextExecutor`).
    ForeachNext { init_id: String },
    /// `BreakLoop` (`"break"`) / `ContinueLoop` (`"continue"`).
    LoopSignal(&'static str),
    /// An ordinary action.
    Action(Action),
}

/// Ordinary (non-structural) action kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    SetValue,
    SetVariable,
    SetTextVariable,
    SetMultipleVariables,
    ResetVariable,
    ClearAllVariables,
    SendActivity,
    ParseValue,
    EditTable,
    EditTableV2,
    CreateConversation,
    AddConversationMessage,
    CopyConversationMessages,
    RetrieveConversationMessage,
    RetrieveConversationMessages,
    /// `EndWorkflow`, `EndDialog`, `EndConversation`, `CancelDialog`,
    /// `CancelAllDialogs`: stop without continuing.
    Terminate,
    InvokeAgent,
    Question,
    RequestExternalInput,
    InvokeFunctionTool,
    HttpRequest,
    InvokeMcpTool,
}

impl Action {
    /// Map an action `kind` to its executor, if supported.
    pub(crate) fn from_kind(kind: &str) -> Option<Action> {
        Some(match kind {
            "SetValue" => Action::SetValue,
            "SetVariable" => Action::SetVariable,
            "SetTextVariable" => Action::SetTextVariable,
            "SetMultipleVariables" => Action::SetMultipleVariables,
            "ResetVariable" => Action::ResetVariable,
            "ClearAllVariables" => Action::ClearAllVariables,
            "SendActivity" => Action::SendActivity,
            "ParseValue" => Action::ParseValue,
            "EditTable" => Action::EditTable,
            "EditTableV2" => Action::EditTableV2,
            "CreateConversation" => Action::CreateConversation,
            "AddConversationMessage" => Action::AddConversationMessage,
            "CopyConversationMessages" => Action::CopyConversationMessages,
            "RetrieveConversationMessage" => Action::RetrieveConversationMessage,
            "RetrieveConversationMessages" => Action::RetrieveConversationMessages,
            "EndWorkflow" | "EndDialog" | "EndConversation" | "CancelDialog"
            | "CancelAllDialogs" => Action::Terminate,
            "InvokeAzureAgent" => Action::InvokeAgent,
            "Question" => Action::Question,
            "RequestExternalInput" | "RequestHumanInput" | "WaitForHumanInput" => {
                Action::RequestExternalInput
            }
            "InvokeFunctionTool" => Action::InvokeFunctionTool,
            "HttpRequestAction" => Action::HttpRequest,
            "InvokeMcpTool" => Action::InvokeMcpTool,
            _ => return None,
        })
    }
}

/// The single executor type backing every node of a declarative workflow.
pub(crate) struct DeclarativeExecutor {
    pub id: String,
    pub node: Node,
    pub def: Json,
    /// The action as parsed YAML, kept for mapping key order (JSON maps are
    /// sorted; upstream's Python dicts keep authoring order).
    pub raw: Option<serde_yaml::Value>,
    pub rt: Arc<Runtime>,
}

pub(crate) type ActionResult = Result<(), CoreError>;

/// `def[key]` when `def` is an object.
pub(crate) fn field<'a>(def: &'a Json, key: &str) -> Option<&'a Json> {
    def.as_object().and_then(|m| m.get(key))
}

/// `def[key]` as a non-empty string.
pub(crate) fn str_field<'a>(def: &'a Json, key: &str) -> Option<&'a str> {
    field(def, key)
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
}

/// A state path given as `key: Local.X` or `key: {path: Local.X}`, falling
/// back to a top-level `path` (upstream `_get_variable_path`).
pub(crate) fn variable_path(def: &Json, key: &str) -> Option<String> {
    match field(def, key) {
        Some(Json::String(s)) => Some(s.clone()),
        Some(Json::Object(m)) => m.get("path").and_then(Json::as_str).map(str::to_string),
        _ => field(def, "path")
            .and_then(Json::as_str)
            .map(str::to_string),
    }
}

/// Like [`variable_path`] without the `path` fallback; empty → `None`.
pub(crate) fn path_ref(v: Option<&Json>) -> Option<String> {
    match v {
        Some(Json::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Json::Object(m)) => m
            .get("path")
            .and_then(Json::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        _ => None,
    }
}

/// Python-style equality for JSON (numbers compare numerically).
pub(crate) fn json_eq(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Number(x), Json::Number(y)) => x.as_f64() == y.as_f64(),
        (Json::Array(x), Json::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| json_eq(a, b))
        }
        (Json::Object(x), Json::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => a == b,
    }
}

/// Python `int(value)` for a JSON value.
pub(crate) fn json_to_int(v: &Json) -> Option<i64> {
    match v {
        Json::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Json::Bool(b) => Some(*b as i64),
        Json::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Normalize a `condition` value the way upstream does: Booleans become
/// `=true`/`=false`, bare strings are treated as expressions.
pub(crate) fn normalize_condition(c: &Json) -> Json {
    match c {
        Json::Bool(true) => json!("=true"),
        Json::Bool(false) => json!("=false"),
        Json::String(s) if !s.starts_with('=') => Json::String(format!("={s}")),
        other => other.clone(),
    }
}

/// `System.conversations.{id}.messages` for an evaluated `conversationId`
/// expression, or `None` when unset/empty.
pub(crate) fn conversation_messages_path(
    state: &DeclarativeState,
    expr: Option<&Json>,
) -> Result<Option<String>, StateError> {
    let Some(expr) = expr else {
        return Ok(None);
    };
    if matches!(expr, Json::String(s) if s.is_empty()) || expr.is_null() {
        return Ok(None);
    }
    let evaluated = state.eval_if_expression(expr)?;
    if !py_truthy(&evaluated) {
        return Ok(None);
    }
    Ok(Some(format!(
        "System.conversations.{}.messages",
        py_str(&evaluated)
    )))
}

#[async_trait]
impl Executor for DeclarativeExecutor {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, message: Json, ctx: WorkflowContext) -> CoreResult<()> {
        let shared = ctx.shared_state();
        let was_initialized = DeclarativeState::is_initialized_in(&shared).await;
        let mut state = DeclarativeState::load(&shared, self.rt.state.clone()).await?;
        match classify(message) {
            Incoming::Response(resp) => self.on_response(&mut state, resp, &ctx).await?,
            incoming => {
                self.ensure_initialized(&mut state, &incoming, was_initialized)?;
                self.on_message(&mut state, incoming, &ctx).await?;
            }
        }
        state.save(&shared).await;
        Ok(())
    }
}

impl DeclarativeExecutor {
    /// The entries of the mapping at `path` in the action definition, in
    /// YAML authoring order (keys absent from the YAML follow, sorted).
    pub(crate) fn ordered_entries<'a>(&'a self, path: &[&str]) -> Vec<(&'a String, &'a Json)> {
        let mut json = Some(&self.def);
        let mut raw = self.raw.as_ref();
        for p in path {
            json = json.and_then(|j| j.get(*p));
            raw = raw.and_then(|r| r.get(*p));
        }
        let Some(Json::Object(map)) = json else {
            return Vec::new();
        };
        let mut out: Vec<(&String, &Json)> = Vec::with_capacity(map.len());
        if let Some(serde_yaml::Value::Mapping(m)) = raw {
            for k in m.keys() {
                if let Some(entry) = map.get_key_value(&super::yaml_key_string(k)) {
                    out.push(entry);
                }
            }
        }
        for entry in map {
            if !out.iter().any(|(k, _)| *k == entry.0) {
                out.push(entry);
            }
        }
        out
    }

    /// Port of upstream `_ensure_state_initialized`: initialize the state
    /// from raw workflow input (internal messages leave it untouched).
    ///
    /// * object → `Workflow.Inputs`;
    /// * a serialized chat message, or an array of them → the last user
    ///   message becomes `Inputs.input` / `System.LastMessage*` and the rest
    ///   becomes `Conversation.messages` (a list continues an initialized
    ///   state rather than resetting it);
    /// * text → `{"input": text}` plus `System.LastMessage`;
    /// * any other value → rendered with Python `str()` semantics, then as text.
    fn ensure_initialized(
        &self,
        state: &mut DeclarativeState,
        incoming: &Incoming,
        was_initialized: bool,
    ) -> Result<(), StateError> {
        let Incoming::Input(input) = incoming else {
            return Ok(());
        };
        let message_list: Option<(Vec<Json>, bool)> = match input {
            Json::Array(items) if !items.is_empty() && items.iter().all(is_chat_message) => {
                Some((items.clone(), true))
            }
            v if is_chat_message(v) => Some((vec![v.clone()], false)),
            _ => None,
        };
        if let Some((messages, is_list)) = message_list {
            let parsed: Vec<Message> = messages.iter().filter_map(json_to_message).collect();
            let last_user = parsed
                .iter()
                .rposition(|m| m.role.0.eq_ignore_ascii_case(Role::USER));
            let (text, id, history): (String, String, Vec<&Message>) = match last_user {
                Some(i) => (
                    parsed[i].text(),
                    parsed[i].message_id.clone().unwrap_or_default(),
                    parsed
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, m)| m)
                        .collect(),
                ),
                None => {
                    let tail = parsed.last();
                    (
                        tail.map(Message::text).unwrap_or_default(),
                        tail.and_then(|m| m.message_id.clone()).unwrap_or_default(),
                        parsed.iter().collect(),
                    )
                }
            };
            if is_list && was_initialized {
                let mut data = state.data().clone();
                let inputs = data
                    .entry("Inputs")
                    .or_insert_with(|| Json::Object(Map::new()));
                if !inputs.is_object() {
                    *inputs = Json::Object(Map::new());
                }
                inputs
                    .as_object_mut()
                    .expect("object")
                    .insert("input".into(), Json::String(text.clone()));
                state.set_data(data)?;
            } else {
                let mut inputs = Map::new();
                inputs.insert("input".into(), Json::String(text.clone()));
                state.initialize(Some(inputs))?;
            }
            let conv_path = state
                .get("System.ConversationId")
                .filter(|v| py_truthy(v))
                .map(|id| format!("System.conversations.{}.messages", py_str(id)));
            for m in history {
                let j = message_to_json(m);
                state.append("Conversation.messages", j.clone())?;
                state.append("Conversation.history", j.clone())?;
                if let Some(p) = &conv_path {
                    state.append(p, j)?;
                }
            }
            state.set("System.LastMessage", json!({"Text": text, "Id": id}))?;
            state.set("System.LastMessageText", Json::String(text))?;
            state.set("System.LastMessageId", Json::String(id))?;
            return Ok(());
        }
        match input {
            Json::Object(map) => state.initialize(Some(map.clone())),
            other => {
                let text = py_str(other);
                let mut inputs = Map::new();
                inputs.insert("input".into(), Json::String(text.clone()));
                state.initialize(Some(inputs))?;
                state.set("System.LastMessage", json!({"Text": text, "Id": ""}))?;
                state.set("System.LastMessageText", Json::String(text))
            }
        }
    }

    async fn on_message(
        &self,
        state: &mut DeclarativeState,
        incoming: Incoming,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        match &self.node {
            Node::Join => ctx.send_message(action_complete()).await,
            Node::IfEval { condition } => {
                let result = state.eval_if_expression(condition)?;
                let msg = if py_truthy(&result) {
                    condition_result(true, 0)
                } else {
                    condition_result(false, ELSE_BRANCH_INDEX)
                };
                ctx.send_message(msg).await
            }
            Node::GroupEval { conditions } => {
                for (i, item) in conditions.iter().enumerate() {
                    let Some(cond) = field(item, "condition").filter(|c| !c.is_null()) else {
                        continue;
                    };
                    let result = state.eval_if_expression(&normalize_condition(cond))?;
                    if py_truthy(&result) {
                        return ctx.send_message(condition_result(true, i as i64)).await;
                    }
                }
                ctx.send_message(condition_result(false, ELSE_BRANCH_INDEX))
                    .await
            }
            Node::ForeachInit => self.foreach_init(state, ctx).await,
            Node::ForeachNext { init_id } => {
                let control = match &incoming {
                    Incoming::Internal(m) if tag(m) == Some("LoopControl") => {
                        m.get("action").and_then(Json::as_str).map(str::to_string)
                    }
                    _ => None,
                };
                if control.as_deref() == Some("break") {
                    remove_loop_state(state, init_id)?;
                    return ctx.send_message(loop_iteration(false, 0)).await;
                }
                self.foreach_next(state, init_id, ctx).await
            }
            Node::LoopSignal(action) => ctx.send_message(loop_control(action)).await,
            Node::Action(action) => self.run_action(*action, state, ctx).await,
        }
    }

    async fn on_response(
        &self,
        state: &mut DeclarativeState,
        response: RequestResponse,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        match &self.node {
            Node::Action(Action::InvokeAgent) => self.agent_response(state, response, ctx).await,
            Node::Action(Action::Question) | Node::Action(Action::RequestExternalInput) => {
                self.external_input_response(state, response, ctx).await
            }
            Node::Action(Action::InvokeFunctionTool) => {
                self.tool_approval_response(state, response, ctx).await
            }
            Node::Action(Action::InvokeMcpTool) => {
                self.mcp_approval_response(state, response, ctx).await
            }
            _ => Err(CoreError::Workflow(format!(
                "executor '{}' received a response but never issues requests",
                self.id
            ))),
        }
    }

    async fn run_action(
        &self,
        action: Action,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let def = &self.def;
        match action {
            Action::SetValue => {
                if let Some(path) = field(def, "path").and_then(Json::as_str) {
                    if !path.is_empty() {
                        let value = field(def, "value").cloned().unwrap_or(Json::Null);
                        let v = state.eval_if_expression(&value)?;
                        state.set(path, v)?;
                    }
                }
            }
            Action::SetVariable => {
                if let Some(path) = variable_path(def, "variable").filter(|p| !p.is_empty()) {
                    let value = field(def, "value").cloned().unwrap_or(Json::Null);
                    let v = state.eval_if_expression(&value)?;
                    state.set(&path, v)?;
                }
            }
            Action::SetTextVariable => {
                if let Some(path) = variable_path(def, "variable").filter(|p| !p.is_empty()) {
                    let text = match (field(def, "text"), field(def, "value")) {
                        (Some(t), _) => {
                            let v = state.eval_if_expression(t)?;
                            if v.is_null() {
                                String::new()
                            } else {
                                py_str(&v)
                            }
                        }
                        // .NET shape: `value` is a template whose `{…}` holes
                        // are PowerFx expressions.
                        (None, Some(Json::String(template))) => state.format_template(template)?,
                        (None, Some(other)) if !other.is_null() => py_str(other),
                        _ => String::new(),
                    };
                    state.set(&path, Json::String(text))?;
                }
            }
            Action::SetMultipleVariables => {
                let assignments = field(def, "assignments")
                    .and_then(Json::as_array)
                    .cloned()
                    .unwrap_or_default();
                for a in assignments.iter().filter(|a| a.is_object()) {
                    let path = match field(a, "variable") {
                        Some(Json::String(s)) => Some(s.clone()),
                        Some(Json::Object(m)) => {
                            m.get("path").and_then(Json::as_str).map(str::to_string)
                        }
                        _ => field(a, "path").and_then(Json::as_str).map(str::to_string),
                    };
                    if let Some(path) = path.filter(|p| !p.is_empty()) {
                        let value = field(a, "value").cloned().unwrap_or(Json::Null);
                        let v = state.eval_if_expression(&value)?;
                        state.set(&path, v)?;
                    }
                }
            }
            Action::ResetVariable => {
                if let Some(path) = variable_path(def, "variable").filter(|p| !p.is_empty()) {
                    state.set(&path, Json::Null)?;
                }
            }
            Action::ClearAllVariables => {
                let mut data = state.data().clone();
                data.insert("Local".into(), Json::Object(Map::new()));
                state.set_data(data)?;
            }
            Action::SendActivity => {
                let activity = field(def, "activity").cloned().unwrap_or(json!(""));
                let text = match &activity {
                    Json::Object(m) => m.get("text").cloned().unwrap_or(json!("")),
                    other => other.clone(),
                };
                let text = match &text {
                    Json::String(s) if s.starts_with('=') => state.eval_if_expression(&text)?,
                    Json::String(s) => Json::String(state.interpolate_string(s)),
                    other => other.clone(),
                };
                if py_truthy(&text) {
                    ctx.yield_output(Json::String(py_str(&text))).await?;
                }
            }
            Action::ParseValue => {
                let path = variable_path(def, "variable").filter(|p| !p.is_empty());
                let value = field(def, "value").filter(|v| !v.is_null());
                if let (Some(path), Some(value)) = (path, value) {
                    let mut v = state.eval_if_expression(value)?;
                    if let Some(t) = str_field(def, "valueType") {
                        v = convert_to_type(v, t);
                    }
                    state.set(&path, v)?;
                }
            }
            Action::EditTable => self.edit_table(state, false)?,
            Action::EditTableV2 => self.edit_table(state, true)?,
            Action::CreateConversation => {
                let id = uuid::Uuid::new_v4().to_string();
                if let Some(path) = path_ref(field(def, "conversationId")) {
                    state.set(&path, Json::String(id.clone()))?;
                }
                let mut conversations = match state.get("System.conversations") {
                    Some(Json::Object(m)) => m.clone(),
                    _ => Map::new(),
                };
                conversations.insert(id.clone(), json!({"id": id, "messages": []}));
                state.set("System.conversations", Json::Object(conversations))?;
            }
            Action::AddConversationMessage => self.add_conversation_message(state)?,
            Action::CopyConversationMessages => self.copy_conversation_messages(state)?,
            Action::RetrieveConversationMessage => self.retrieve_conversation_message(state)?,
            Action::RetrieveConversationMessages => self.retrieve_conversation_messages(state)?,
            Action::Terminate => return Ok(()),
            Action::InvokeAgent => return self.invoke_agent(state, ctx).await,
            Action::Question | Action::RequestExternalInput => {
                return self.request_external_input(action, state, ctx).await
            }
            Action::InvokeFunctionTool => return self.invoke_function_tool(state, ctx).await,
            Action::HttpRequest => return self.http_request(state, ctx).await,
            Action::InvokeMcpTool => return self.invoke_mcp_tool(state, ctx).await,
        }
        ctx.send_message(action_complete()).await
    }

    // ------------------------------------------------------------ Foreach

    fn loop_vars(&self) -> (String, Option<String>) {
        // .NET shape: `value` / `index` are full paths.
        if field(&self.def, "items").is_some() && field(&self.def, "source").is_none() {
            let item = path_ref(field(&self.def, "value")).unwrap_or_else(|| "Local.item".into());
            return (item, path_ref(field(&self.def, "index")));
        }
        let item = format!(
            "Local.{}",
            str_field(&self.def, "itemName").unwrap_or("item")
        );
        let index = field(&self.def, "indexName").map(|n| {
            format!(
                "Local.{}",
                n.as_str().map(str::to_string).unwrap_or_else(|| py_str(n))
            )
        });
        (item, index)
    }

    async fn foreach_init(
        &self,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let dotnet_shape = field(&self.def, "source").is_none();
        let expr = field(&self.def, "source")
            .or_else(|| field(&self.def, "items"))
            .cloned()
            .unwrap_or(Json::Null);
        let raw = state.eval_if_expression(&expr)?;
        let items: Vec<Json> = match raw {
            Json::Array(items) => items,
            other if !py_truthy(&other) => Vec::new(),
            // .NET treats a scalar as a one-item loop.
            other if dotnet_shape => vec![other],
            // Upstream Python: `list(value)` — characters of a string, keys
            // of a mapping.
            Json::String(s) => s.chars().map(|c| Json::String(c.to_string())).collect(),
            Json::Object(m) => m.keys().map(|k| Json::String(k.clone())).collect(),
            other => {
                return Err(CoreError::Workflow(format!(
                    "declarative action error: Foreach source must be a list, found {other}"
                )))
            }
        };
        let mut data = state.data().clone();
        let loops = data
            .entry(LOOP_STATE_KEY)
            .or_insert_with(|| Json::Object(Map::new()));
        if !loops.is_object() {
            *loops = Json::Object(Map::new());
        }
        loops.as_object_mut().expect("object").insert(
            self.id.clone(),
            json!({"items": items, "index": 0, "length": items.len()}),
        );
        state.set_data(data)?;
        if let Some(first) = items.first() {
            let (item_var, index_var) = self.loop_vars();
            state.set(&item_var, first.clone())?;
            if let Some(index_var) = index_var {
                state.set(&index_var, json!(0))?;
            }
            ctx.send_message(loop_iteration(true, 0)).await
        } else {
            ctx.send_message(loop_iteration(false, 0)).await
        }
    }

    async fn foreach_next(
        &self,
        state: &mut DeclarativeState,
        init_id: &str,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let loop_state = state
            .data()
            .get(LOOP_STATE_KEY)
            .and_then(|l| l.get(init_id))
            .cloned();
        let Some(loop_state) = loop_state else {
            return ctx.send_message(loop_iteration(false, 0)).await;
        };
        let items = loop_state
            .get("items")
            .and_then(Json::as_array)
            .cloned()
            .unwrap_or_default();
        let index = loop_state.get("index").and_then(Json::as_u64).unwrap_or(0) as usize + 1;
        if index < items.len() {
            let mut data = state.data().clone();
            if let Some(ls) = data
                .get_mut(LOOP_STATE_KEY)
                .and_then(|l| l.get_mut(init_id))
                .and_then(Json::as_object_mut)
            {
                ls.insert("index".into(), json!(index));
            }
            state.set_data(data)?;
            let (item_var, index_var) = self.loop_vars();
            state.set(&item_var, items[index].clone())?;
            if let Some(index_var) = index_var {
                state.set(&index_var, json!(index))?;
            }
            ctx.send_message(loop_iteration(true, index)).await
        } else {
            remove_loop_state(state, init_id)?;
            ctx.send_message(loop_iteration(false, 0)).await
        }
    }

    // ------------------------------------------------------------ EditTable

    fn edit_table(&self, state: &mut DeclarativeState, v2: bool) -> Result<(), StateError> {
        let def = &self.def;
        if let Some(items_var) = path_ref(field(def, "itemsVariable")) {
            return self.edit_table_dotnet(state, &items_var, v2);
        }
        let Some(table_path) = str_field(def, "table")
            .map(str::to_string)
            .or_else(|| variable_path(def, "variable").filter(|p| !p.is_empty()))
        else {
            return Ok(());
        };
        let operation = str_field(def, "operation").unwrap_or("add").to_lowercase();
        let value = if v2 {
            field(def, "item")
                .filter(|v| !v.is_null())
                .or_else(|| field(def, "value"))
        } else {
            field(def, "value")
        }
        .filter(|v| !v.is_null())
        .cloned();
        let key_field = if v2 { str_field(def, "key") } else { None };
        let index = field(def, "index").filter(|v| !v.is_null()).cloned();

        let mut table: Vec<Json> = match state.get(&table_path) {
            None | Some(Json::Null) => Vec::new(),
            Some(Json::Array(a)) => a.clone(),
            Some(other) => vec![other.clone()],
        };
        let eval_index = |state: &DeclarativeState, default: i64| -> Result<i64, StateError> {
            match &index {
                Some(i) => Ok(json_to_int(&state.eval_if_expression(i)?).unwrap_or(default)),
                None => Ok(default),
            }
        };
        let key_of = |row: &Json, key: &str| row.as_object().and_then(|m| m.get(key)).cloned();
        match operation.as_str() {
            "add" | "insert" if !v2 || operation == "add" => {
                let v = match &value {
                    Some(v) => state.eval_if_expression(v)?,
                    None => Json::Null,
                };
                if index.is_some() {
                    let idx = eval_index(state, table.len() as i64)?;
                    let idx = python_insert_index(idx, table.len());
                    table.insert(idx, v);
                } else {
                    table.push(v);
                }
            }
            "remove" => {
                if let Some(value) = &value {
                    let v = state.eval_if_expression(value)?;
                    match (key_field, &v) {
                        (Some(key), Json::Object(m)) => {
                            let target = m.get(key).cloned();
                            table.retain(|r| {
                                !(r.is_object() && json_opt_eq(&key_of(r, key), &target))
                            });
                        }
                        _ => {
                            if let Some(pos) = table.iter().position(|r| json_eq(r, &v)) {
                                table.remove(pos);
                            }
                        }
                    }
                } else if index.is_some() {
                    let idx = eval_index(state, -1)?;
                    if idx >= 0 && (idx as usize) < table.len() {
                        table.remove(idx as usize);
                    }
                }
            }
            "clear" => table.clear(),
            "addorupdate" if v2 => {
                let v = match &value {
                    Some(v) => state.eval_if_expression(v)?,
                    None => Json::Null,
                };
                match (key_field, &v) {
                    (Some(key), Json::Object(m)) => {
                        let target = m.get(key).cloned();
                        match table
                            .iter()
                            .position(|r| r.is_object() && json_opt_eq(&key_of(r, key), &target))
                        {
                            Some(i) => table[i] = v,
                            None => table.push(v),
                        }
                    }
                    _ => table.push(v),
                }
            }
            "set" | "update" if !v2 || operation == "update" => {
                if index.is_some() {
                    let v = match &value {
                        Some(v) => state.eval_if_expression(v)?,
                        None => Json::Null,
                    };
                    let idx = eval_index(state, 0)?;
                    if idx >= 0 && (idx as usize) < table.len() {
                        table[idx as usize] = v;
                    }
                } else if v2 {
                    let v = match &value {
                        Some(v) => state.eval_if_expression(v)?,
                        None => Json::Null,
                    };
                    if let (Some(key), Json::Object(m)) = (key_field, &v) {
                        let target = m.get(key).cloned();
                        if let Some(i) = table
                            .iter()
                            .position(|r| r.is_object() && json_opt_eq(&key_of(r, key), &target))
                        {
                            table[i] = v;
                        }
                    }
                }
            }
            _ => {}
        }
        state.set(&table_path, Json::Array(table))
    }

    /// The .NET `EditTable`/`EditTableV2` shape: `itemsVariable` plus a
    /// `changeType` (`Add`/`Remove`/`Clear`/`TakeFirst`/`TakeLast`, or for V2
    /// a `{kind: AddItemOperation|RemoveItemOperation|ClearItemsOperation|
    /// TakeFirstItemOperation|TakeLastItemOperation, value, resultVariable}`
    /// mapping). Records are appended as-is (scalars as `{Value: x}` on an
    /// empty table); .NET's schema-driven field projection is not reproduced.
    fn edit_table_dotnet(
        &self,
        state: &mut DeclarativeState,
        items_var: &str,
        v2: bool,
    ) -> Result<(), StateError> {
        let def = &self.def;
        let (change, op_def): (String, &Json) = if v2 {
            let ct = field(def, "changeType").unwrap_or(&Json::Null);
            let kind = str_field(ct, "kind").unwrap_or_default();
            let change = match kind {
                "AddItemOperation" => "add",
                "RemoveItemOperation" => "remove",
                "ClearItemsOperation" => "clear",
                "TakeFirstItemOperation" => "takefirst",
                "TakeLastItemOperation" => "takelast",
                _ => "",
            };
            (change.to_string(), ct)
        } else {
            let change = match field(def, "changeType") {
                Some(Json::String(s)) => s.to_lowercase(),
                Some(Json::Object(m)) => m
                    .get("value")
                    .and_then(Json::as_str)
                    .unwrap_or_default()
                    .to_lowercase(),
                _ => String::new(),
            };
            (change, def)
        };
        let mut table = match state.get(items_var) {
            None | Some(Json::Null) => Vec::new(),
            Some(Json::Array(a)) => a.clone(),
            Some(other) => {
                return Err(StateError::InvalidPath(format!(
                    "Require '{items_var}' to be a table, not: {other}"
                )))
            }
        };
        let result_var = path_ref(field(op_def, "resultVariable"));
        let mut result: Option<Json> = None;
        match change.as_str() {
            "add" => {
                let v = match field(op_def, "value") {
                    Some(v) => state.eval_if_expression(v)?,
                    None => Json::Null,
                };
                let row = if table.is_empty() && !v.is_object() {
                    json!({"Value": v})
                } else {
                    v
                };
                table.push(row.clone());
                result = Some(row);
            }
            "remove" => {
                if let Some(v) = field(op_def, "value") {
                    let v = state.eval_if_expression(v)?;
                    let remove: Vec<Json> = match v {
                        Json::Array(a) => a,
                        other => vec![other],
                    };
                    table.retain(|r| !remove.iter().any(|x| json_eq(r, x)));
                    result = Some(json!({}));
                }
            }
            "clear" => {
                table.clear();
                result = Some(Json::Null);
            }
            "takefirst" | "takelast" => {
                let row = if table.is_empty() {
                    None
                } else if change == "takefirst" {
                    Some(table.remove(0))
                } else {
                    table.pop()
                };
                result = Some(row.unwrap_or(Json::Null));
            }
            _ => {}
        }
        state.set(items_var, Json::Array(table))?;
        if let (Some(path), Some(r)) = (result_var, result) {
            state.set(&path, r)?;
        }
        Ok(())
    }

    // ------------------------------------------------- conversation actions

    fn conversation_path(&self, state: &DeclarativeState) -> Result<String, StateError> {
        conversation_messages_path(state, field(&self.def, "conversationId"))?.ok_or_else(|| {
            StateError::InvalidPath(format!(
                "action '{}' requires a non-empty conversationId",
                self.id
            ))
        })
    }

    /// .NET `AddConversationMessage`: build a message from `role`, `content`
    /// (`[{type: Text, value: template}]`) and `metadata`, append it to the
    /// conversation, and store it at `message`.
    fn add_conversation_message(&self, state: &mut DeclarativeState) -> Result<(), StateError> {
        let def = &self.def;
        let path = self.conversation_path(state)?;
        let role = str_field(def, "role").unwrap_or("User").to_lowercase();
        let role = if role == "agent" {
            "assistant".to_string()
        } else {
            role
        };
        let mut contents = Vec::new();
        for c in field(def, "content")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
        {
            let text = match field(c, "value") {
                Some(Json::String(s)) => state.format_template(s)?,
                Some(Json::Object(m)) if m.len() == 1 && m.values().all(Json::is_null) => {
                    // YAML parsed `{Local.X}` as a flow mapping: treat the key
                    // as a template hole, as .NET's template syntax intends.
                    let key = m.keys().next().cloned().unwrap_or_default();
                    state.format_template(&format!("{{{key}}}"))?
                }
                Some(other) => py_str(other),
                None => String::new(),
            };
            contents.push(Content::text(text));
        }
        let mut message = Message::with_contents(Role::new(role), contents);
        message.message_id = Some(uuid::Uuid::new_v4().simple().to_string());
        if let Some(meta) = field(def, "metadata") {
            if let Json::Object(m) = state.eval_if_expression(meta)? {
                message.additional_properties = m.into_iter().collect();
            }
        }
        let j = message_to_json(&message);
        state.append(&path, j.clone())?;
        if let Some(target) = path_ref(field(def, "message")) {
            state.set(&target, j)?;
        }
        Ok(())
    }

    /// .NET `CopyConversationMessages`: append the evaluated `messages` to the
    /// conversation.
    fn copy_conversation_messages(&self, state: &mut DeclarativeState) -> Result<(), StateError> {
        let path = self.conversation_path(state)?;
        let messages = match field(&self.def, "messages") {
            Some(m) => state.eval_if_expression(m)?,
            None => Json::Null,
        };
        let items = match messages {
            Json::Array(a) => a,
            Json::Null => Vec::new(),
            other => vec![other],
        };
        for item in items {
            if let Some(m) = json_to_message(&item) {
                state.append(&path, message_to_json(&m))?;
            }
        }
        Ok(())
    }

    /// .NET `RetrieveConversationMessage`: look a message up by `messageId`.
    fn retrieve_conversation_message(
        &self,
        state: &mut DeclarativeState,
    ) -> Result<(), StateError> {
        let path = self.conversation_path(state)?;
        let id = match field(&self.def, "messageId") {
            Some(e) => py_str(&state.eval_if_expression(e)?),
            None => String::new(),
        };
        let found = state
            .get(&path)
            .and_then(Json::as_array)
            .and_then(|msgs| {
                msgs.iter()
                    .find(|m| m.get("message_id").and_then(Json::as_str) == Some(id.as_str()))
            })
            .cloned()
            .unwrap_or(Json::Null);
        if let Some(target) = path_ref(field(&self.def, "message")) {
            state.set(&target, found)?;
        }
        Ok(())
    }

    /// .NET `RetrieveConversationMessages`: copy (optionally `limit`ed, at
    /// most 100, `sortOrder: NewestFirst` aware) messages into `messages`.
    fn retrieve_conversation_messages(
        &self,
        state: &mut DeclarativeState,
    ) -> Result<(), StateError> {
        let path = self.conversation_path(state)?;
        let mut msgs = state
            .get(&path)
            .and_then(Json::as_array)
            .cloned()
            .unwrap_or_default();
        let newest_first = match field(&self.def, "sortOrder") {
            Some(v) => py_str(&state.eval_if_expression(v)?).eq_ignore_ascii_case("NewestFirst"),
            None => false,
        };
        if newest_first {
            msgs.reverse();
        }
        let limit = match field(&self.def, "limit") {
            Some(v) => json_to_int(&state.eval_if_expression(v)?).unwrap_or(100),
            None => 100,
        }
        .clamp(0, 100) as usize;
        msgs.truncate(limit);
        if let Some(target) = path_ref(field(&self.def, "messages")) {
            state.set(&target, Json::Array(msgs))?;
        }
        Ok(())
    }
}

fn json_opt_eq(a: &Option<Json>, b: &Option<Json>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => json_eq(x, y),
        (None, None) => true,
        // Python's dict.get returns None for a missing key.
        (Some(Json::Null), None) | (None, Some(Json::Null)) => true,
        _ => false,
    }
}

/// Python `list.insert` index semantics (negative counts from the end, then
/// clamps).
fn python_insert_index(idx: i64, len: usize) -> usize {
    let len = len as i64;
    let i = if idx < 0 {
        (len + idx).max(0)
    } else {
        idx.min(len)
    };
    i as usize
}

fn remove_loop_state(state: &mut DeclarativeState, init_id: &str) -> Result<(), StateError> {
    let mut data = state.data().clone();
    if let Some(Json::Object(loops)) = data.get_mut(LOOP_STATE_KEY) {
        if loops.remove(init_id).is_some() {
            state.set_data(data)?;
        }
    }
    Ok(())
}

/// Port of upstream `ParseValueExecutor._convert_to_type`.
pub(crate) fn convert_to_type(value: Json, target: &str) -> Json {
    match target.to_lowercase().as_str() {
        "string" => match value {
            Json::Null => json!(""),
            other => Json::String(py_str(&other)),
        },
        "number" | "int" | "integer" | "float" | "decimal" => match value {
            Json::Null => json!(0),
            Json::String(s) => {
                if s.contains('.') {
                    s.trim()
                        .parse::<f64>()
                        .map(|f| json!(f))
                        .unwrap_or(json!(0))
                } else {
                    s.trim()
                        .parse::<i64>()
                        .map(|i| json!(i))
                        .unwrap_or(json!(0))
                }
            }
            Json::Number(n) => json!(n.as_f64().unwrap_or(0.0)),
            Json::Bool(b) => json!(if b { 1.0 } else { 0.0 }),
            _ => json!(0),
        },
        "boolean" | "bool" => match value {
            Json::Null => json!(false),
            Json::String(s) => json!(matches!(
                s.to_lowercase().as_str(),
                "true" | "yes" | "1" | "on"
            )),
            other => json!(py_truthy(&other)),
        },
        "object" | "record" => match value {
            Json::Null => json!({}),
            Json::Object(m) => Json::Object(m),
            Json::String(s) => match serde_json::from_str::<Json>(&s) {
                Ok(Json::Object(m)) => Json::Object(m),
                Ok(other) => json!({"value": other}),
                Err(_) => json!({"value": s}),
            },
            other => json!({"value": other}),
        },
        "array" | "table" | "list" => match value {
            Json::Null => json!([]),
            Json::Array(a) => Json::Array(a),
            Json::String(s) => match serde_json::from_str::<Json>(&s) {
                Ok(Json::Array(a)) => Json::Array(a),
                Ok(other) => json!([other]),
                Err(_) => json!([s]),
            },
            other => json!([other]),
        },
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_to_type_matches_upstream() {
        assert_eq!(convert_to_type(json!("42"), "Number"), json!(42));
        assert_eq!(convert_to_type(json!("4.5"), "number"), json!(4.5));
        assert_eq!(convert_to_type(json!("x"), "number"), json!(0));
        assert_eq!(convert_to_type(json!(7), "number"), json!(7.0));
        assert_eq!(convert_to_type(json!("Yes"), "boolean"), json!(true));
        assert_eq!(convert_to_type(json!(null), "string"), json!(""));
        assert_eq!(convert_to_type(json!(true), "string"), json!("True"));
        assert_eq!(
            convert_to_type(json!("{\"a\":1}"), "object"),
            json!({"a": 1})
        );
        assert_eq!(convert_to_type(json!("[1]"), "array"), json!([1]));
        assert_eq!(convert_to_type(json!("x"), "array"), json!(["x"]));
        assert_eq!(convert_to_type(json!(5), "unknown"), json!(5));
    }

    #[test]
    fn helpers() {
        assert_eq!(
            variable_path(&json!({"variable": "Local.a"}), "variable").unwrap(),
            "Local.a"
        );
        assert_eq!(
            variable_path(&json!({"variable": {"path": "Local.b"}}), "variable").unwrap(),
            "Local.b"
        );
        assert_eq!(
            variable_path(&json!({"path": "Local.c"}), "variable").unwrap(),
            "Local.c"
        );
        assert!(json_eq(&json!(1), &json!(1.0)));
        assert_eq!(normalize_condition(&json!(true)), json!("=true"));
        assert_eq!(normalize_condition(&json!("Local.x")), json!("=Local.x"));
        assert_eq!(python_insert_index(-1, 3), 2);
        assert_eq!(python_insert_index(9, 3), 3);
        assert_eq!(
            Action::from_kind("WaitForHumanInput"),
            Some(Action::RequestExternalInput)
        );
        assert_eq!(Action::from_kind("Nope"), None);
    }
}
