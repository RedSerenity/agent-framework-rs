//! `InvokeAzureAgent` (port of `_executors_agents.py`).
//!
//! Agents are resolved by name from the registry supplied to the
//! [`WorkflowFactory`](super::WorkflowFactory); any
//! [`SupportsAgentRun`](agent_framework_core::agent::SupportsAgentRun) works,
//! not only Azure agents.
//!
//! Both YAML shapes upstream accepts are supported:
//!
//! ```yaml
//! - kind: InvokeAzureAgent          # Python style
//!   agent: MenuAgent
//!   input: =Local.userInput
//!   resultProperty: Local.agentResponse
//!
//! - kind: InvokeAzureAgent          # .NET style
//!   agent: { name: AgentName }
//!   conversationId: =System.ConversationId
//!   input:
//!     arguments: { param1: =Local.value1 }
//!     messages: =Conversation.messages
//!     externalLoop: { when: =Local.needsMoreInput, maxIterations: 10 }
//!   output:
//!     messages: Local.ResponseMessages
//!     responseObject: Local.StructuredResponse
//!     autoSend: true
//! ```
//!
//! # Human in the loop
//!
//! When `input.externalLoop.when` is true after the agent replies, the
//! executor pauses the run with a request whose payload is
//! `{"type": "AgentExternalInputRequest", request_id, agent_name,
//! agent_response, iteration, messages, function_calls}`. Answer it with
//! [`WorkflowRun::send_response`](agent_framework_core::workflow::WorkflowRun::send_response)
//! passing either the user's text or `{"user_input": "..."}`. As upstream,
//! the condition is re-checked *before* (against the new input) and *after*
//! the next agent turn, and the loop stops after `maxIterations` (default 100).
//!
//! # Divergences
//!
//! * `input.messages` evaluating to a message record (e.g.
//!   `=UserMessage(...)`) contributes that message's **text**; upstream
//!   Python stringifies the record's dict representation, which is a bug the
//!   .NET reference does not share.
//! * The agent is invoked with the stored conversation as core
//!   [`Message`](agent_framework_core::types::Message)s and no session; run
//!   kwargs/options propagation (`WORKFLOW_RUN_KWARGS_KEY`) has no Rust
//!   equivalent.
//! * External-loop bookkeeping is keyed per executor id (upstream keeps one
//!   global slot).

use std::sync::Arc;

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::error::Error as CoreError;
use agent_framework_core::types::{Message, Role};
use agent_framework_core::workflow::{RequestResponse, WorkflowContext};
use serde_json::{json, Map, Value as Json};

use super::executor::{field, ActionResult, DeclarativeExecutor};
use super::messages::{
    action_complete, extract_json_from_response, function_calls, json_message_text,
    json_to_message, message_to_json,
};
use super::state::{py_str, py_truthy, DeclarativeState, StateError};

/// [`SharedState`](agent_framework_core::workflow::SharedState) key for
/// external-loop resumption state.
pub const EXTERNAL_LOOP_STATE_KEY: &str = "_external_loop_state";

/// Prefix-less variable names default to the `Local` scope.
pub(crate) fn normalize_variable_path(variable: &str) -> String {
    if variable.contains('.') {
        variable.to_string()
    } else {
        format!("Local.{variable}")
    }
}

/// `(arguments, messages expression, externalLoop.when, maxIterations)`.
type InputConfig = (Vec<(String, Json)>, Option<Json>, Option<String>, i64);

struct OutputConfig {
    messages_var: Option<String>,
    response_obj_var: Option<String>,
    result_property: Option<String>,
    auto_send: bool,
}

fn opt_string(v: Option<&Json>) -> Option<String> {
    match v {
        None | Some(Json::Null) => None,
        Some(Json::String(s)) => Some(s.clone()),
        Some(other) => Some(py_str(other)),
    }
}

impl DeclarativeExecutor {
    fn agent_name(&self, state: &DeclarativeState) -> Result<Option<String>, StateError> {
        let eval_name = |s: &str| -> Result<Option<String>, StateError> {
            if s.starts_with('=') {
                let v = state.eval(s)?;
                Ok((!v.is_null()).then(|| py_str(&v)))
            } else {
                Ok(Some(s.to_string()))
            }
        };
        match field(&self.def, "agent") {
            Some(Json::String(s)) => return eval_name(s),
            Some(Json::Object(m)) => {
                if let Some(Json::String(name)) = m.get("name") {
                    return eval_name(name);
                }
            }
            _ => {}
        }
        match field(&self.def, "agentName") {
            Some(Json::String(s)) => eval_name(s),
            _ => Ok(None),
        }
    }

    fn input_config(&self) -> InputConfig {
        let input = field(&self.def, "input");
        let Some(Json::Object(input)) = input else {
            // Non-mapping input is treated as the messages expression.
            return (Vec::new(), input.cloned(), None, 100);
        };
        let arguments: Vec<(String, Json)> = self
            .ordered_entries(&["input", "arguments"])
            .into_iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let messages = input.get("messages").cloned();
        let mut when = None;
        let mut max_iterations = 100;
        if let Some(Json::Object(lp)) = input.get("externalLoop") {
            when = opt_string(lp.get("when"));
            if let Some(m) = lp.get("maxIterations") {
                max_iterations = super::executor::json_to_int(m).unwrap_or(100);
            }
        }
        (arguments, messages, when, max_iterations)
    }

    fn output_config(&self, state: &DeclarativeState) -> Result<OutputConfig, StateError> {
        let result_property = opt_string(field(&self.def, "resultProperty"));
        let Some(Json::Object(output)) = field(&self.def, "output") else {
            return Ok(OutputConfig {
                messages_var: None,
                response_obj_var: None,
                result_property,
                auto_send: true,
            });
        };
        let auto_send = match output.get("autoSend") {
            None => true,
            Some(v) => py_truthy(&state.eval_if_expression(v)?),
        };
        Ok(OutputConfig {
            messages_var: opt_string(output.get("messages")),
            response_obj_var: opt_string(output.get("responseObject")),
            result_property: opt_string(output.get("property")).or(result_property),
            auto_send,
        })
    }

    fn agent_messages_path(&self, state: &DeclarativeState) -> Result<String, StateError> {
        Ok(
            super::executor::conversation_messages_path(state, field(&self.def, "conversationId"))?
                .unwrap_or_else(|| "Conversation.messages".to_string()),
        )
    }

    fn build_input_text(
        &self,
        state: &DeclarativeState,
        arguments: &[(String, Json)],
        messages_expr: Option<&Json>,
    ) -> Result<String, StateError> {
        let mut evaluated_args = Vec::new();
        for (k, v) in arguments {
            evaluated_args.push(format!("{k}: {}", py_str(&state.eval_if_expression(v)?)));
        }
        let args_text = evaluated_args.join("\n");

        let mut messages_text = String::new();
        if let Some(expr) = messages_expr.filter(|e| py_truthy(e)) {
            let evaluated = state.eval_if_expression(expr)?;
            messages_text = match &evaluated {
                Json::String(s) => s.clone(),
                Json::Array(items) => match items.last() {
                    Some(Json::String(s)) => s.clone(),
                    Some(Json::Null) | None => String::new(),
                    Some(last) => json_message_text(last),
                },
                Json::Object(_) => json_message_text(&evaluated),
                v if py_truthy(v) => py_str(v),
                _ => String::new(),
            };
        } else if arguments.is_empty() {
            // Implicit input: Local.input / Local.userInput, then the last
            // message, then the workflow inputs.
            let local = [state.get("Local.input"), state.get("Local.userInput")]
                .into_iter()
                .flatten()
                .find(|v| py_truthy(v));
            messages_text = local.map(py_str).unwrap_or_default();
            if messages_text.is_empty() {
                if let Some(t) = state
                    .get("System.LastMessage.Text")
                    .filter(|v| py_truthy(v))
                {
                    messages_text = py_str(t);
                }
            }
            if messages_text.is_empty() {
                if let Some(Json::Object(inputs)) = state.get("Workflow.Inputs") {
                    messages_text = if inputs.len() == 1 {
                        py_str(inputs.values().next().expect("one input"))
                    } else {
                        inputs
                            .iter()
                            .map(|(k, v)| format!("{k}: {}", py_str(v)))
                            .collect::<Vec<_>>()
                            .join("\n")
                    };
                }
            }
        }
        Ok(match (args_text.is_empty(), messages_text.is_empty()) {
            (false, false) => format!("{args_text}\n{messages_text}"),
            (false, true) => args_text,
            _ => messages_text,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn invoke_and_store(
        &self,
        agent: Arc<dyn SupportsAgentRun>,
        agent_name: &str,
        input_text: &str,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
        output: &OutputConfig,
        messages_path: &str,
    ) -> Result<(String, Vec<Message>, Vec<Json>), CoreError> {
        if !input_text.is_empty() {
            let user = Message::new(Role::new(Role::USER), input_text.to_string());
            state.append(messages_path, message_to_json(&user))?;
        }
        let history: Vec<Message> = match state.get(messages_path) {
            Some(Json::Array(items)) => items.iter().filter_map(json_to_message).collect(),
            _ => Vec::new(),
        };
        let response = agent.run(history, None).await.map_err(|e| {
            CoreError::AgentExecution(format!("Agent '{agent_name}' invocation failed: {e}"))
        })?;
        let text = response.text();
        if !text.is_empty() && output.auto_send {
            ctx.yield_output(Json::String(text.clone())).await?;
        }
        let all_messages = response.messages.clone();
        let tool_calls = function_calls(&all_messages);
        if !all_messages.is_empty() {
            for m in &all_messages {
                state.append(messages_path, message_to_json(m))?;
            }
        } else if !text.is_empty() {
            let assistant = Message::new(Role::new(Role::ASSISTANT), text.clone());
            state.append(messages_path, message_to_json(&assistant))?;
        }
        let messages_json: Vec<Json> = all_messages.iter().map(message_to_json).collect();
        state.set("Agent.response", Json::String(text.clone()))?;
        state.set("Agent.name", Json::String(agent_name.to_string()))?;
        state.set("Agent.text", Json::String(text.clone()))?;
        state.set("Agent.messages", Json::Array(messages_json.clone()))?;
        state.set("Agent.toolCalls", Json::Array(tool_calls.clone()))?;
        state.set("System.LastMessage", json!({"Text": text}))?;
        if let Some(var) = &output.messages_var {
            let value = if messages_json.is_empty() {
                Json::String(text.clone())
            } else {
                Json::Array(messages_json)
            };
            state.set(&normalize_variable_path(var), value)?;
        }
        if let Some(var) = &output.response_obj_var {
            let path = normalize_variable_path(var);
            match extract_json_from_response(&text) {
                Ok(parsed) => state.set(&path, parsed.unwrap_or(Json::Null))?,
                Err(e) => {
                    tracing::warn!(
                        "InvokeAzureAgent: failed to parse JSON for '{path}': {e}, storing as string"
                    );
                    state.set(&path, Json::String(text.clone()))?;
                }
            }
        }
        if let Some(prop) = &output.result_property {
            state.set(prop, Json::String(text.clone()))?;
        }
        Ok((text, all_messages, tool_calls))
    }

    fn resolve_agent(&self, name: &str) -> Option<Arc<dyn SupportsAgentRun>> {
        self.rt.agents.get(name).cloned()
    }

    pub(crate) async fn invoke_agent(
        &self,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let Some(agent_name) = self.agent_name(state)? else {
            tracing::warn!("InvokeAzureAgent action missing 'agent' or 'agent.name' property");
            return ctx.send_message(action_complete()).await;
        };
        let (arguments, messages_expr, when, max_iterations) = self.input_config();
        let output = self.output_config(state)?;
        let messages_path = self.agent_messages_path(state)?;
        let input_text = self.build_input_text(state, &arguments, messages_expr.as_ref())?;
        let Some(agent) = self.resolve_agent(&agent_name) else {
            return Err(CoreError::AgentExecution(format!(
                "Agent '{agent_name}' invocation failed: not found in registry"
            )));
        };
        let (text, messages, calls) = self
            .invoke_and_store(
                agent,
                &agent_name,
                &input_text,
                state,
                ctx,
                &output,
                &messages_path,
            )
            .await?;

        if let Some(when) = &when {
            if py_truthy(&state.eval(when)?) {
                let loop_state = json!({
                    "agent_name": agent_name,
                    "iteration": 1,
                    "external_loop_when": when,
                    "messages_var": output.messages_var,
                    "response_obj_var": output.response_obj_var,
                    "result_property": output.result_property,
                    "auto_send": output.auto_send,
                    "messages_path": messages_path,
                    "max_iterations": max_iterations,
                });
                self.save_loop_state(ctx, Some(loop_state)).await;
                return self
                    .request_agent_input(ctx, &agent_name, &text, 0, &messages, calls)
                    .await;
            }
        }
        ctx.send_message(action_complete()).await
    }

    async fn request_agent_input(
        &self,
        ctx: &WorkflowContext,
        agent_name: &str,
        response: &str,
        iteration: i64,
        messages: &[Message],
        calls: Vec<Json>,
    ) -> ActionResult {
        ctx.request_info(json!({
            "type": "AgentExternalInputRequest",
            "request_id": uuid::Uuid::new_v4().to_string(),
            "agent_name": agent_name,
            "agent_response": response,
            "iteration": iteration,
            "messages": messages.iter().map(message_to_json).collect::<Vec<_>>(),
            "function_calls": calls,
        }))
        .await
    }

    async fn save_loop_state(&self, ctx: &WorkflowContext, value: Option<Json>) {
        let shared = ctx.shared_state();
        let mut all = match shared.get(EXTERNAL_LOOP_STATE_KEY).await {
            Some(Json::Object(m)) => m,
            _ => Map::new(),
        };
        match value {
            Some(v) => {
                all.insert(self.id.clone(), v);
            }
            None => {
                all.remove(&self.id);
            }
        }
        shared.set(EXTERNAL_LOOP_STATE_KEY, Json::Object(all)).await;
    }

    pub(crate) async fn agent_response(
        &self,
        state: &mut DeclarativeState,
        response: RequestResponse,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let loop_state = ctx
            .shared_state()
            .get(EXTERNAL_LOOP_STATE_KEY)
            .await
            .and_then(|m| m.get(&self.id).cloned());
        let Some(mut loop_state) = loop_state else {
            tracing::error!("InvokeAzureAgent: external loop state not found, cannot resume");
            return ctx.send_message(action_complete()).await;
        };
        let input = match &response.data {
            Json::String(s) => s.clone(),
            Json::Object(m) => m
                .get("user_input")
                .or_else(|| m.get("userInput"))
                .map(|v| match v {
                    Json::String(s) => s.clone(),
                    other => py_str(other),
                })
                .unwrap_or_default(),
            Json::Null => String::new(),
            other => py_str(other),
        };
        let text_field = |k: &str| loop_state.get(k).and_then(Json::as_str).map(str::to_string);
        let agent_name = text_field("agent_name").unwrap_or_default();
        let when = text_field("external_loop_when").unwrap_or_default();
        let messages_path =
            text_field("messages_path").unwrap_or_else(|| "Conversation.messages".into());
        let iteration = loop_state
            .get("iteration")
            .and_then(Json::as_i64)
            .unwrap_or(0);
        let max_iterations = loop_state
            .get("max_iterations")
            .and_then(Json::as_i64)
            .unwrap_or(100);

        state.set("Local.userInput", Json::String(input.clone()))?;
        state.set("System.LastMessage", json!({"Text": input}))?;

        if !py_truthy(&state.eval(&when)?) {
            self.save_loop_state(ctx, None).await;
            return ctx.send_message(action_complete()).await;
        }
        let Some(agent) = self.resolve_agent(&agent_name) else {
            return Err(CoreError::AgentExecution(format!(
                "Agent '{agent_name}' invocation failed: not found during loop resumption"
            )));
        };
        let auto_send = self.output_config(state)?.auto_send;
        let output = OutputConfig {
            messages_var: text_field("messages_var"),
            response_obj_var: text_field("response_obj_var"),
            result_property: text_field("result_property"),
            auto_send,
        };
        let (text, messages, calls) = self
            .invoke_and_store(
                agent,
                &agent_name,
                &input,
                state,
                ctx,
                &output,
                &messages_path,
            )
            .await?;
        if !py_truthy(&state.eval(&when)?) {
            self.save_loop_state(ctx, None).await;
            return ctx.send_message(action_complete()).await;
        }
        if iteration < max_iterations {
            loop_state["iteration"] = json!(iteration + 1);
            loop_state["auto_send"] = json!(auto_send);
            self.save_loop_state(ctx, Some(loop_state)).await;
            return self
                .request_agent_input(ctx, &agent_name, &text, iteration, &messages, calls)
                .await;
        }
        tracing::warn!(
            "InvokeAzureAgent: external loop exceeded max iterations ({max_iterations})"
        );
        self.save_loop_state(ctx, None).await;
        ctx.send_message(action_complete()).await
    }
}
