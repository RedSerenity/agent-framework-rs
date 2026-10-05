//! Operating-mode tracking ("plan" / "execute") for the harness agent.
//!
//! Rust equivalent of upstream `agent_framework._harness._mode`
//! (`AgentModeProvider`, `get_agent_mode`, `set_agent_mode`).

use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::session::SessionState;
use agent_framework_core::tools::{ApprovalMode, ToolDefinition};
use agent_framework_core::types::Message;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::paths::py_repr;
use crate::util::{
    empty_object_schema, function_tool, json_type_name, parse_args, py_json_dumps, SessionRef,
};

/// Default source id (session-state key) of [`AgentModeProvider`]. Mirrors
/// upstream `DEFAULT_MODE_SOURCE_ID`.
pub const DEFAULT_MODE_SOURCE_ID: &str = "agent_mode";

const MODE_GET_INSTRUCTIONS: &str = "Use the mode_get tool to check your current operating mode.\n";
const MODE_SET_INSTRUCTIONS: &str = "Use the mode_set tool to switch between modes as your work progresses. Only use mode_set if the user explicitly instructs/allows you to change modes.\n\n";
const APPLICATION_MODE_SET_INSTRUCTIONS: &str = "Mode changes are controlled by the application. Use its configured mode-change mechanism only when the user explicitly instructs/allows a mode change.\n\n";
const PLAN_MODE_TRANSITION: &str = "7. When approval is granted, always switch to execute mode (using the `mode_set` tool), and follow the steps for *Execute mode*.";
const APPLICATION_PLAN_MODE_TRANSITION: &str = "7. When approval is granted, use the application's configured mode-change mechanism to transition to execute mode. Follow the steps for *Execute mode* only after the mode has changed.";
const PREVIOUS_MODE_STATE_KEY: &str = "previous_mode_for_notification";

/// Default mode-provider instructions template. `{current_mode}` and
/// `{available_modes}` are substituted. Mirrors upstream
/// `DEFAULT_MODE_INSTRUCTIONS` verbatim.
pub const DEFAULT_MODE_INSTRUCTIONS: &str = concat!(
    "## Agent Mode\n\n",
    "- You can operate in different modes. Depending on the mode you are in, you will be required to follow different processes.\n\n",
    "Use the mode_get tool to check your current operating mode.\n",
    "Use the mode_set tool to switch between modes as your work progresses. Only use mode_set if the user explicitly instructs/allows you to change modes.\n\n",
    "You are currently operating in the {current_mode} mode.\n\n",
    "### Mandatory Mode based Workflow\n\n",
    "For every new substantive user request, including short factual questions, your behavior is determined by the mode you are in.\n\n",
    "{available_modes}\n",
);

/// The user message injected after an external mode change. Mirrors upstream
/// `DEFAULT_MODE_CHANGE_NOTIFICATION`.
pub const DEFAULT_MODE_CHANGE_NOTIFICATION: &str = "[Mode changed: The operating mode has been switched from \"{previous_mode}\" to \"{current_mode}\". You must now adjust your behavior to match the \"{current_mode}\" mode.]";

/// Upstream `DEFAULT_MODE_MAP["plan"]`.
pub const DEFAULT_PLAN_MODE_INSTRUCTIONS: &str = concat!(
    "Use this mode when analyzing requirements, breaking down tasks, and creating plans. This is the interactive mode — ask clarifying questions, discuss options, and get user approval before proceeding.\n\n",
    "Process to follow when in plan mode:\n",
    "1. Analyze the request with the purpose of building a research plan.\n",
    "2. Create a list of todo items.\n",
    "3. If needed, use the provided tools to do some exploratory checks to help build a plan and determine what clarifying questions you may need from the user.\n",
    "4. Ask for clarifications from the user where needed.\n",
    "   1. Ask each clarification one by one.\n",
    "   2. When asking for clarification and you have specific options in mind, present them to the user, so they can choose the option instead of having to retype the entire response.\n",
    "   3. Do not proceed until you have received all the needed clarifications.\n",
    "   4. Do short exploratory research if it helps with being able to ask sensible clarifications from the user.\n",
    "5. Write the plan to a memory file, so that it is retained even if compaction happens. Make sure to update the plan file if the user requests changes.\n",
    "6. Present the plan to the user and ask for approval to switch to execute mode and process the plan.\n",
    "7. When approval is granted, always switch to execute mode (using the `mode_set` tool), and follow the steps for *Execute mode*.",
);

/// Upstream `DEFAULT_MODE_MAP["execute"]`.
pub const DEFAULT_EXECUTE_MODE_INSTRUCTIONS: &str = concat!(
    "Determine the type of ask:\n",
    "1. Simple question that doesn't require any further work to answer.\n",
    "2. Any other work, including complex user request that requires a multi-step process to satisfy.\n\n",
    "If 1. just answer the question directly.\n",
    "If 2. Work autonomously using your best judgment — do not ask the user questions or wait for feedback and follow the following process:\n",
    "1. If you don't have a plan or tasks yet, analyze the user request and create tasks and a plan. (**Skip this step if you came from plan mode**)\n",
    "2. Work autonomously — use your best judgment to make decisions and keep progressing without asking the user questions. The goal is to have a complete, useful result ready when the user returns.\n",
    "3. If you encounter ambiguity or an unexpected situation during execution, choose the most reasonable option, note your choice, and keep going.\n",
    "4. Mark tasks as completed as you finish them.\n",
    "5. Continue working, thinking and calling tools until you have the research result for the user.",
);

/// The built-in modes, in order: `plan` then `execute`. Mirrors upstream
/// `DEFAULT_MODE_MAP`.
pub fn default_mode_map() -> Vec<(String, String)> {
    vec![
        ("plan".into(), DEFAULT_PLAN_MODE_INSTRUCTIONS.into()),
        ("execute".into(), DEFAULT_EXECUTE_MODE_INSTRUCTIONS.into()),
    ]
}

/// Normalized (`trim().lower()`) mode → display name, in order. Rejects
/// duplicates and an empty set.
fn normalize_available_modes(modes: &[String]) -> Result<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = Vec::new();
    for mode in modes {
        let display = mode.trim().to_string();
        let normalized = display.to_lowercase();
        if out.iter().any(|(n, _)| *n == normalized) {
            return Err(Error::Configuration(format!(
                "Duplicate mode configured: {mode}."
            )));
        }
        out.push((normalized, display));
    }
    if out.is_empty() {
        return Err(Error::Configuration(
            "available_modes must contain at least one mode.".into(),
        ));
    }
    Ok(out)
}

fn resolve_available_modes(modes: Option<&[String]>) -> Result<Vec<(String, String)>> {
    match modes {
        Some(modes) => normalize_available_modes(modes),
        None => normalize_available_modes(&["plan".to_string(), "execute".to_string()]),
    }
}

fn normalize_mode(mode: &str, available: &[(String, String)]) -> Result<String> {
    let normalized = mode.trim().to_lowercase();
    if available.iter().any(|(n, _)| *n == normalized) {
        return Ok(normalized);
    }
    let supported = available
        .iter()
        .map(|(_, d)| py_repr(d))
        .collect::<Vec<_>>()
        .join(", ");
    Err(Error::tool(format!(
        "Invalid mode: {mode}. Supported modes are {supported}."
    )))
}

fn resolve_default_mode(
    default_mode: Option<&str>,
    available: &[(String, String)],
) -> Result<String> {
    match default_mode {
        None => Ok(available[0].0.clone()),
        Some(mode) => {
            normalize_mode(mode, available).map_err(|e| Error::Configuration(e.to_string()))
        }
    }
}

fn mode_state(state: &SessionState, source_id: &str) -> Result<Map<String, Value>> {
    match state.get(source_id) {
        None | Some(Value::Null) => {
            state.insert(source_id, Value::Object(Map::new()));
            Ok(Map::new())
        }
        Some(Value::Object(map)) => Ok(map),
        Some(other) => Err(Error::AgentExecution(format!(
            "Session state for source_id {} must be a dict, got {}.",
            py_repr(source_id),
            json_type_name(&other)
        ))),
    }
}

/// Get the current operating mode from session state.
///
/// Returns the stored mode when it is one of `available_modes` (default: the
/// built-in `plan`/`execute`); otherwise (nothing stored, or a stored mode
/// no longer configured) stores and returns `default_mode` — the first
/// available mode when `None`. Mirrors upstream `get_agent_mode`.
pub fn get_agent_mode(
    state: &SessionState,
    source_id: &str,
    default_mode: Option<&str>,
    available_modes: Option<&[String]>,
) -> Result<String> {
    let available = resolve_available_modes(available_modes)?;
    let default = resolve_default_mode(default_mode, &available)?;
    let mut provider_state = mode_state(state, source_id)?;
    if let Some(Value::String(current)) = provider_state.get("current_mode") {
        if let Ok(normalized) = normalize_mode(current, &available) {
            return Ok(normalized);
        }
    }
    provider_state.insert("current_mode".into(), Value::String(default.clone()));
    state.insert(source_id, Value::Object(provider_state));
    Ok(default)
}

/// Set the current operating mode in session state, returning the
/// normalized mode.
///
/// External callers (e.g. a slash-command handler) should use this rather
/// than mutating state directly. With `notify`, an actual change records the
/// previous mode so the provider's next `before_run` injects a user message
/// announcing the switch; without it, any pending notification is cleared
/// (the agent's own `mode_set` tool uses `notify = false`). Mirrors upstream
/// `set_agent_mode`.
pub fn set_agent_mode(
    state: &SessionState,
    mode: &str,
    source_id: &str,
    available_modes: Option<&[String]>,
    notify: bool,
) -> Result<String> {
    let available = resolve_available_modes(available_modes)?;
    let normalized = normalize_mode(mode, &available)?;
    let mut provider_state = mode_state(state, source_id)?;
    let previous = provider_state.get("current_mode").cloned();
    provider_state.insert("current_mode".into(), Value::String(normalized.clone()));
    if notify {
        if let Some(Value::String(previous)) = previous {
            if previous != normalized {
                provider_state.insert(PREVIOUS_MODE_STATE_KEY.into(), Value::String(previous));
            }
        }
    } else {
        provider_state.remove(PREVIOUS_MODE_STATE_KEY);
    }
    state.insert(source_id, Value::Object(provider_state));
    Ok(normalized)
}

#[derive(Deserialize)]
struct ModeSetArgs {
    mode: String,
}

#[derive(Serialize)]
struct ModeSetResult<'a> {
    mode: &'a str,
    message: String,
}

/// Context provider tracking the agent's operating mode in session state.
///
/// Mirrors upstream `AgentModeProvider`. The configured modes (default
/// `plan` and `execute`, see [`default_mode_map`]) and the current mode are
/// rendered into the injected instructions each run. Tools (both
/// `never_require`):
///
/// - `mode_set` — switch mode (`{"mode": ...}`); returns
///   `{"mode": m, "message": "Mode changed to 'm'."}`.
/// - `mode_get` — returns `{"mode": m}`.
///
/// Either tool can be hidden ([`expose_mode_set`](AgentModeProviderBuilder::expose_mode_set) /
/// [`expose_mode_get`](AgentModeProviderBuilder::expose_mode_get)); the default instructions are
/// then rewritten accordingly. After an external change via
/// [`set_agent_mode`] with `notify`, the next run injects a
/// [`DEFAULT_MODE_CHANGE_NOTIFICATION`] user message once.
#[derive(Debug, Clone)]
pub struct AgentModeProvider {
    source_id: String,
    /// `(normalized, display, instructions)` in configured order.
    modes: Vec<(String, String, String)>,
    default_mode: String,
    instructions: Option<String>,
    expose_mode_set: bool,
    expose_mode_get: bool,
}

impl Default for AgentModeProvider {
    fn default() -> Self {
        Self::builder().build().expect("default modes are valid")
    }
}

/// Builder for [`AgentModeProvider`].
#[derive(Debug, Clone)]
pub struct AgentModeProviderBuilder {
    source_id: String,
    default_mode: Option<String>,
    mode_instructions: Option<Vec<(String, String)>>,
    instructions: Option<String>,
    expose_mode_set: bool,
    expose_mode_get: bool,
}

impl AgentModeProviderBuilder {
    /// Override the source id (session-state key).
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }
    /// The initial mode when none is stored (default: the first mode).
    pub fn default_mode(mut self, mode: impl Into<String>) -> Self {
        self.default_mode = Some(mode.into());
        self
    }
    /// The supported modes, in order, each with its instructions. Custom text
    /// is not rewritten when tools are hidden.
    pub fn mode_instructions(mut self, modes: Vec<(String, String)>) -> Self {
        self.mode_instructions = Some(modes);
        self
    }
    /// Custom instructions template (`{available_modes}` / `{current_mode}`
    /// placeholders). Not rewritten when tools are hidden.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
    /// Whether to contribute the `mode_set` tool (default `true`).
    pub fn expose_mode_set(mut self, expose: bool) -> Self {
        self.expose_mode_set = expose;
        self
    }
    /// Whether to contribute the `mode_get` tool (default `true`).
    pub fn expose_mode_get(mut self, expose: bool) -> Self {
        self.expose_mode_get = expose;
        self
    }
    /// Validate and build. Errors when no modes are configured, a mode is
    /// duplicated, or the default mode is not configured.
    pub fn build(self) -> Result<AgentModeProvider> {
        let mode_instructions = match self.mode_instructions {
            Some(m) => m,
            None => {
                let mut m = default_mode_map();
                if !self.expose_mode_set {
                    m[0].1 = m[0]
                        .1
                        .replace(PLAN_MODE_TRANSITION, APPLICATION_PLAN_MODE_TRANSITION);
                }
                m
            }
        };
        let names: Vec<String> = mode_instructions.iter().map(|(n, _)| n.clone()).collect();
        let normalized = normalize_available_modes(&names).map_err(|e| match e {
            Error::Configuration(m) if m.starts_with("available_modes") => {
                Error::Configuration("mode_instructions must contain at least one mode.".into())
            }
            other => other,
        })?;
        let default_mode = resolve_default_mode(self.default_mode.as_deref(), &normalized)?;
        let modes = normalized
            .into_iter()
            .zip(mode_instructions)
            .map(|((n, d), (_, i))| (n, d, i))
            .collect();
        Ok(AgentModeProvider {
            source_id: self.source_id,
            modes,
            default_mode,
            instructions: self.instructions,
            expose_mode_set: self.expose_mode_set,
            expose_mode_get: self.expose_mode_get,
        })
    }
}

impl AgentModeProvider {
    /// A provider with upstream's defaults (`plan` / `execute`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Start configuring a provider.
    pub fn builder() -> AgentModeProviderBuilder {
        AgentModeProviderBuilder {
            source_id: DEFAULT_MODE_SOURCE_ID.into(),
            default_mode: None,
            mode_instructions: None,
            instructions: None,
            expose_mode_set: true,
            expose_mode_get: true,
        }
    }

    /// The source id (session-state key).
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// The normalized available modes, in order.
    pub fn available_modes(&self) -> Vec<String> {
        self.modes.iter().map(|(n, _, _)| n.clone()).collect()
    }

    /// The normalized default mode.
    pub fn default_mode(&self) -> &str {
        &self.default_mode
    }

    /// The current mode of `state` under this provider's configuration.
    pub fn current_mode(&self, state: &SessionState) -> Result<String> {
        get_agent_mode(
            state,
            &self.source_id,
            Some(&self.default_mode),
            Some(&self.available_modes()),
        )
    }

    /// Set the mode of `state` under this provider's configuration (see
    /// [`set_agent_mode`]).
    pub fn set_mode(&self, state: &SessionState, mode: &str, notify: bool) -> Result<String> {
        set_agent_mode(
            state,
            mode,
            &self.source_id,
            Some(&self.available_modes()),
            notify,
        )
    }

    /// Render the instructions for `current_mode`. Mirrors upstream
    /// `_build_instructions`.
    pub fn build_instructions(&self, current_mode: &str) -> String {
        let mode_lines: String = self
            .modes
            .iter()
            .map(|(_, display, text)| format!("#### {display}\n\n{text}\n\n"))
            .collect();
        let mut instructions = match self.instructions.as_deref().filter(|i| !i.is_empty()) {
            Some(custom) => custom.to_string(),
            None => {
                let mut text = DEFAULT_MODE_INSTRUCTIONS.to_string();
                if !self.expose_mode_get {
                    text = text.replace(MODE_GET_INSTRUCTIONS, "");
                }
                if !self.expose_mode_set {
                    text = text.replace(MODE_SET_INSTRUCTIONS, APPLICATION_MODE_SET_INSTRUCTIONS);
                }
                text
            }
        };
        instructions = instructions
            .replace("{available_modes}", &mode_lines)
            .replace("{current_mode}", current_mode);
        instructions
    }

    fn display_name<'a>(&'a self, mode: &'a str) -> &'a str {
        self.modes
            .iter()
            .find(|(n, _, _)| n == mode)
            .map(|(_, d, _)| d.as_str())
            .unwrap_or(mode)
    }

    fn tools(&self, state: SessionState) -> Vec<ToolDefinition> {
        let mut tools = Vec::new();
        if self.expose_mode_set {
            let p = self.clone();
            let s = state.clone();
            tools.push(function_tool(
                "mode_set",
                "Switch the agent's operating mode.",
                json!({
                    "type": "object",
                    "properties": {"mode": {"type": "string"}},
                    "required": ["mode"],
                }),
                ApprovalMode::NeverRequire,
                move |args| {
                    let p = p.clone();
                    let s = s.clone();
                    async move {
                        let args: ModeSetArgs = parse_args("mode_set", args)?;
                        let mode = p.set_mode(&s, &args.mode, false)?;
                        let result = ModeSetResult {
                            mode: &mode,
                            message: format!("Mode changed to '{mode}'."),
                        };
                        Ok(Value::String(py_json_dumps(&result, true)))
                    }
                },
            ));
        }
        if self.expose_mode_get {
            let p = self.clone();
            tools.push(function_tool(
                "mode_get",
                "Get the agent's current operating mode.",
                empty_object_schema(),
                ApprovalMode::NeverRequire,
                move |_args| {
                    let p = p.clone();
                    let s = state.clone();
                    async move {
                        let mode = p.current_mode(&s)?;
                        Ok(Value::String(py_json_dumps(&json!({"mode": mode}), true)))
                    }
                },
            ));
        }
        tools
    }
}

#[async_trait]
impl ContextProvider for AgentModeProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session = SessionRef::from_context(ctx, "AgentModeProvider")?;
        let current = self.current_mode(&session.state)?;
        // Pop the external-change marker before injecting, so the agent sees
        // the notification exactly once.
        let mut provider_state = mode_state(&session.state, &self.source_id)?;
        let previous = provider_state.remove(PREVIOUS_MODE_STATE_KEY);
        session
            .state
            .insert(self.source_id.clone(), Value::Object(provider_state));

        ctx.add_instructions(self.build_instructions(&current));
        ctx.tools.extend(self.tools(session.state.clone()));
        if let Some(Value::String(previous)) = previous {
            if previous != current {
                let notification = DEFAULT_MODE_CHANGE_NOTIFICATION
                    .replace("{previous_mode}", self.display_name(&previous))
                    .replace("{current_mode}", self.display_name(&current));
                ctx.messages.push(Message::user(notification));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(state: &SessionState) -> SessionContext {
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some("s".into());
        ctx.session_state = Some(state.clone());
        ctx
    }

    #[test]
    fn get_and_set_manage_session_state() {
        let state = SessionState::new();
        assert_eq!(
            get_agent_mode(&state, DEFAULT_MODE_SOURCE_ID, None, None).unwrap(),
            "plan"
        );
        assert_eq!(
            set_agent_mode(&state, " Execute ", DEFAULT_MODE_SOURCE_ID, None, true).unwrap(),
            "execute"
        );
        assert_eq!(
            state.get(DEFAULT_MODE_SOURCE_ID).unwrap(),
            json!({"current_mode": "execute", "previous_mode_for_notification": "plan"})
        );
        let err = set_agent_mode(&state, "bogus", DEFAULT_MODE_SOURCE_ID, None, true).unwrap_err();
        assert!(err
            .to_string()
            .contains("Invalid mode: bogus. Supported modes are 'plan', 'execute'."));
        // A no-op change records nothing new; notify=false clears pending.
        set_agent_mode(&state, "execute", DEFAULT_MODE_SOURCE_ID, None, false).unwrap();
        assert!(state
            .get(DEFAULT_MODE_SOURCE_ID)
            .unwrap()
            .get(PREVIOUS_MODE_STATE_KEY)
            .is_none());
        assert!(get_agent_mode(&state, DEFAULT_MODE_SOURCE_ID, None, Some(&[])).is_err());
        state.insert("bad", json!([]));
        assert!(get_agent_mode(&state, "bad", None, None).is_err());
    }

    #[test]
    fn stored_mode_outside_available_resets_to_default() {
        let state = SessionState::new();
        set_agent_mode(&state, "plan", "m", None, false).unwrap();
        let modes = vec!["Draft".to_string(), "Review".to_string()];
        assert_eq!(
            get_agent_mode(&state, "m", None, Some(&modes)).unwrap(),
            "draft"
        );
    }

    #[test]
    fn builder_validates_configuration() {
        assert!(AgentModeProvider::builder()
            .mode_instructions(vec![])
            .build()
            .is_err());
        assert!(AgentModeProvider::builder()
            .mode_instructions(vec![("a".into(), "x".into()), ("A ".into(), "y".into())])
            .build()
            .is_err());
        assert!(AgentModeProvider::builder()
            .default_mode("nope")
            .build()
            .is_err());
        let p = AgentModeProvider::builder()
            .mode_instructions(vec![
                ("Draft".into(), "d".into()),
                ("Ship".into(), "s".into()),
            ])
            .build()
            .unwrap();
        assert_eq!(p.default_mode(), "draft");
        assert_eq!(p.available_modes(), vec!["draft", "ship"]);
    }

    #[tokio::test]
    async fn before_run_injects_instructions_tools_and_notification_once() {
        let state = SessionState::new();
        let provider = AgentModeProvider::new();
        let mut c = ctx(&state);
        provider.before_run(&mut c).await.unwrap();
        let instructions = c.instructions.unwrap();
        assert!(instructions.contains("You are currently operating in the plan mode."));
        assert!(instructions.contains("#### plan\n\n"));
        assert!(instructions.contains("#### execute\n\n"));
        assert_eq!(
            c.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["mode_set", "mode_get"]
        );
        assert!(c.messages.is_empty());

        // The agent's own tool changes mode without a notification.
        let set = c.tools[0].executor.clone().unwrap();
        let out = set.invoke(json!({"mode": "EXECUTE"})).await.unwrap();
        assert_eq!(
            out,
            json!("{\"mode\": \"execute\", \"message\": \"Mode changed to 'execute'.\"}")
        );
        let get = c.tools[1].executor.clone().unwrap();
        assert_eq!(
            get.invoke(json!({})).await.unwrap(),
            json!("{\"mode\": \"execute\"}")
        );
        let mut c = ctx(&state);
        provider.before_run(&mut c).await.unwrap();
        assert!(c.messages.is_empty());

        // An external change is announced exactly once.
        set_agent_mode(&state, "plan", DEFAULT_MODE_SOURCE_ID, None, true).unwrap();
        let mut c = ctx(&state);
        provider.before_run(&mut c).await.unwrap();
        assert_eq!(
            c.messages[0].text(),
            "[Mode changed: The operating mode has been switched from \"execute\" to \"plan\". You must now adjust your behavior to match the \"plan\" mode.]"
        );
        let mut c = ctx(&state);
        provider.before_run(&mut c).await.unwrap();
        assert!(c.messages.is_empty());
    }

    #[tokio::test]
    async fn hidden_tools_rewrite_default_instructions() {
        let provider = AgentModeProvider::builder()
            .expose_mode_set(false)
            .expose_mode_get(false)
            .build()
            .unwrap();
        let state = SessionState::new();
        let mut c = ctx(&state);
        provider.before_run(&mut c).await.unwrap();
        let text = c.instructions.unwrap();
        assert!(c.tools.is_empty());
        assert!(!text.contains("mode_get"));
        assert!(text.contains("Mode changes are controlled by the application."));
        assert!(
            text.contains("use the application's configured mode-change mechanism to transition")
        );
        assert!(!text.contains("(using the `mode_set` tool)"));
    }

    #[test]
    fn default_constant_matches_rendered_template_pieces() {
        assert!(DEFAULT_MODE_INSTRUCTIONS.contains(MODE_GET_INSTRUCTIONS));
        assert!(DEFAULT_MODE_INSTRUCTIONS.contains(MODE_SET_INSTRUCTIONS));
        assert!(DEFAULT_PLAN_MODE_INSTRUCTIONS.ends_with(PLAN_MODE_TRANSITION));
    }
}
