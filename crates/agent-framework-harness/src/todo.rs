//! Todo-list management for the harness agent.
//!
//! Rust equivalent of upstream `agent_framework._harness._todo`
//! (`TodoProvider`, `TodoItem`, `TodoInput`, `TodoCompleteInput`,
//! `TodoStore`, `TodoSessionStore`, `TodoFileStore`).

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::tools::ApprovalMode;
use agent_framework_core::types::Message;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::paths::{py_repr, storage_path_segment};
use crate::util::{
    empty_object_schema, function_tool, json_type_name, parse_args, py_json_dumps, SessionRef,
};

/// Default source id (session-state key) of [`TodoProvider`]. Mirrors
/// upstream `DEFAULT_TODO_SOURCE_ID`.
pub const DEFAULT_TODO_SOURCE_ID: &str = "todo";

/// Default instructions injected by [`TodoProvider`]. Mirrors upstream
/// `DEFAULT_TODO_INSTRUCTIONS` verbatim.
pub const DEFAULT_TODO_INSTRUCTIONS: &str = "## Todo Items\n\n\
You have access to a todo list for tracking work items.\n\
When a user asks you to perform a task, follow these steps to manage your work:\n\
1. Determine whether the ask requires multiple steps to complete (complex) or can be completed using a single step (simple).\n\
2. If complex, turn the task into manageable todo items and add them to the list.\n\
3. If simple, don't add a todo item, but rather just complete the task directly.\n\n\
### General TODO Guidelines\n\
Ask questions from the user where clarification is needed to create effective todos.\n\
If the user provides feedback on your plan, adjust your todos accordingly by adding new items or removing irrelevant ones.\n\
During execution, use the todo list to keep track of what needs to be done, mark items as complete when finished, and remove any items that are no longer needed.\n\
When a user changes the topic, changes their mind or switches to a new request, ensure that you update the todo list accordingly by removing irrelevant/old items, clearing the list, or adding new ones as needed.\n\n\
Use these tools to manage your tasks:\n\
- Use todos_add to break down complex work into trackable items (supports adding one or many at once).\n\
- Use todos_complete to mark items as done when finished (supports one or many at once). Include a reason describing how the items were completed.\n\
- Use todos_get_remaining to check what work is still pending.\n\
- Use todos_get_all to review the full list including completed items.\n\
- Use todos_remove to remove items that are no longer needed (supports one or many at once).";

/// One todo item tracked for a session. Mirrors upstream `TodoItem`; the
/// serialized shape (`{"id", "title", "description", "is_complete"}`, with
/// `description` always present, possibly `null`) is upstream's persisted
/// and tool-result format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    /// The item's id, unique within the session.
    pub id: i64,
    /// The item's title.
    pub title: String,
    /// Optional free-text description.
    pub description: Option<String>,
    /// Whether the item has been completed.
    pub is_complete: bool,
}

impl TodoItem {
    /// A new, open item.
    pub fn new(id: i64, title: impl Into<String>, description: Option<String>) -> Self {
        Self {
            id,
            title: title.into(),
            description,
            is_complete: false,
        }
    }

    /// Parse one persisted item, validating it the way upstream's
    /// `TodoItem.from_dict` does.
    pub fn from_value(value: &Value) -> Result<Self> {
        let map = value
            .as_object()
            .ok_or_else(|| Error::Serialization("Todo item must be a mapping.".into()))?;
        let id = map
            .get("id")
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::Serialization("Todo item id must be an integer.".into()))?;
        let title = match map.get("title") {
            Some(Value::String(t)) if !t.trim().is_empty() => t.clone(),
            _ => {
                return Err(Error::Serialization(
                    "Todo item title must be a non-empty string.".into(),
                ))
            }
        };
        let description = match map.get("description") {
            None | Some(Value::Null) => None,
            Some(Value::String(d)) => Some(d.clone()),
            Some(_) => {
                return Err(Error::Serialization(
                    "Todo item description must be a string or null.".into(),
                ))
            }
        };
        let is_complete = match map.get("is_complete") {
            None => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => {
                return Err(Error::Serialization(
                    "Todo item is_complete must be a boolean.".into(),
                ))
            }
        };
        Ok(Self {
            id,
            title,
            description,
            is_complete,
        })
    }

    /// The persisted / tool-result JSON of this item.
    pub fn to_value(&self) -> Value {
        json!({
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "is_complete": self.is_complete,
        })
    }
}

/// One todo item to create. Mirrors upstream `TodoInput`: the title is
/// trimmed and must be non-empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoInput {
    /// The trimmed title.
    pub title: String,
    /// Optional description.
    pub description: Option<String>,
}

impl TodoInput {
    /// Validate and normalize one todo input.
    pub fn new(title: &str, description: Option<String>) -> Result<Self> {
        let title = title.trim();
        if title.is_empty() {
            return Err(Error::tool("Todo input title must be a non-empty string."));
        }
        Ok(Self {
            title: title.to_string(),
            description,
        })
    }
}

/// One todo item to mark complete. Mirrors upstream `TodoCompleteInput`: the
/// reason is trimmed and must be non-empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoCompleteInput {
    /// The item id.
    pub id: i64,
    /// The trimmed completion reason.
    pub reason: String,
}

impl TodoCompleteInput {
    /// Validate and normalize one completion input.
    pub fn new(id: i64, reason: &str) -> Result<Self> {
        let reason = reason.trim();
        if reason.is_empty() {
            return Err(Error::tool(
                "Todo complete input reason must be a non-empty string.",
            ));
        }
        Ok(Self {
            id,
            reason: reason.to_string(),
        })
    }
}

fn parse_todo_items(items: &[Value], source: &str) -> Result<Vec<TodoItem>> {
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            if !item.is_object() {
                return Err(Error::Serialization(format!(
                    "Todo item at index {index} in {source} must be a mapping; got {}.",
                    json_type_name(item)
                )));
            }
            TodoItem::from_value(item)
        })
        .collect()
}

/// Clamp `next_id` so it cannot collide with any persisted id. Mirrors
/// upstream `_safe_next_id`.
fn safe_next_id(items: &[TodoItem], next_id: i64) -> i64 {
    next_id.max(items.iter().map(|i| i.id).max().unwrap_or(0) + 1)
}

/// Backing store for session todo items. Mirrors upstream `TodoStore`.
#[async_trait]
pub trait TodoStore: Send + Sync {
    /// Load the persisted items and the next available id.
    async fn load_state(
        &self,
        session: &SessionRef,
        source_id: &str,
    ) -> Result<(Vec<TodoItem>, i64)>;

    /// Persist the items and the next available id.
    async fn save_state(
        &self,
        session: &SessionRef,
        items: &[TodoItem],
        next_id: i64,
        source_id: &str,
    ) -> Result<()>;

    /// Load only the items.
    async fn load_items(&self, session: &SessionRef, source_id: &str) -> Result<Vec<TodoItem>> {
        Ok(self.load_state(session, source_id).await?.0)
    }
}

/// Store todo state inside [`AgentSession::state`](agent_framework_core::session::AgentSession::state)
/// under the provider's source id, as `{"items": [...], "next_id": n}`.
/// The default store. Mirrors upstream `TodoSessionStore`.
#[derive(Debug, Clone, Default)]
pub struct TodoSessionStore;

#[async_trait]
impl TodoStore for TodoSessionStore {
    async fn load_state(
        &self,
        session: &SessionRef,
        source_id: &str,
    ) -> Result<(Vec<TodoItem>, i64)> {
        let provider_state = match session.state.get(source_id) {
            None | Some(Value::Null) => {
                session.state.insert(source_id, Value::Object(Map::new()));
                Map::new()
            }
            Some(Value::Object(map)) => map,
            Some(other) => {
                return Err(Error::Serialization(format!(
                    "Session state for source_id {} must be a dict; got {}.",
                    py_repr(source_id),
                    json_type_name(&other)
                )))
            }
        };
        let items = match provider_state.get("items") {
            None => Vec::new(),
            Some(Value::Array(items)) => parse_todo_items(items, "session todo state")?,
            Some(other) => {
                return Err(Error::Serialization(format!(
                    "Session state for source_id {} has a non-list 'items' field; got {}.",
                    py_repr(source_id),
                    json_type_name(other)
                )))
            }
        };
        let next_id = match provider_state.get("next_id") {
            None => 1,
            Some(v) => v.as_i64().ok_or_else(|| {
                Error::Serialization(format!(
                    "Session state for source_id {} has a non-integer 'next_id' field; got {}.",
                    py_repr(source_id),
                    json_type_name(v)
                ))
            })?,
        };
        let next = safe_next_id(&items, next_id);
        Ok((items, next))
    }

    async fn save_state(
        &self,
        session: &SessionRef,
        items: &[TodoItem],
        next_id: i64,
        source_id: &str,
    ) -> Result<()> {
        let mut provider_state = match session.state.get(source_id) {
            Some(Value::Object(map)) => map,
            _ => Map::new(),
        };
        provider_state.insert(
            "items".into(),
            Value::Array(items.iter().map(TodoItem::to_value).collect()),
        );
        provider_state.insert("next_id".into(), json!(safe_next_id(items, next_id)));
        session
            .state
            .insert(source_id, Value::Object(provider_state));
        Ok(())
    }
}

/// Store todo state in one JSON file per session and source id. Mirrors
/// upstream `TodoFileStore` (experimental upstream).
///
/// Layout: `{base_path}[/{owner_prefix}{owner}/{kind}]/{session}/{stem}.{source}{ext}`,
/// where the owner (from `session.state[owner_state_key]`, when configured),
/// the session id and the source id are each mapped onto exactly one path
/// segment by [`storage_path_segment`] (prefix `~todo-`). A session id
/// containing a path separator is rejected, as upstream does.
#[derive(Debug, Clone)]
pub struct TodoFileStore {
    base_path: PathBuf,
    kind: String,
    owner_prefix: String,
    owner_state_key: Option<String>,
    state_filename: String,
}

const TODO_ENCODED_SEGMENT_PREFIX: &str = "~todo-";

impl TodoFileStore {
    /// A store rooted at `base_path` with upstream's defaults
    /// (`kind="todos"`, no owner, `state_filename="todos.json"`).
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
            kind: "todos".into(),
            owner_prefix: String::new(),
            owner_state_key: None,
            state_filename: "todos.json".into(),
        }
    }

    /// Storage bucket name under each owner directory.
    pub fn kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = kind.into();
        self
    }

    /// Prefix applied to the resolved owner id.
    pub fn owner_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.owner_prefix = prefix.into();
        self
    }

    /// Session-state key holding the logical owner id.
    pub fn owner_state_key(mut self, key: impl Into<String>) -> Self {
        self.owner_state_key = Some(key.into());
        self
    }

    /// File name used for the persisted state.
    pub fn state_filename(mut self, name: impl Into<String>) -> Self {
        self.state_filename = name.into();
        self
    }

    fn path_segment(value: &str, label: &str, reject_separators: bool) -> Result<String> {
        if reject_separators && (value.contains('/') || value.contains('\\')) {
            return Err(Error::Configuration(format!(
                "TodoFileStore {label} must not contain path separators: {}",
                py_repr(value)
            )));
        }
        Ok(storage_path_segment(value, TODO_ENCODED_SEGMENT_PREFIX))
    }

    fn state_file_name(&self, source_id: &str) -> Result<String> {
        let source = Self::path_segment(source_id, "source_id", false)?;
        let path = Path::new(&self.state_filename);
        match (path.file_stem(), path.extension()) {
            (Some(stem), Some(ext)) => Ok(format!(
                "{}.{source}.{}",
                stem.to_string_lossy(),
                ext.to_string_lossy()
            )),
            _ => Ok(format!("{}.{source}.json", self.state_filename)),
        }
    }

    /// The JSON file holding `source_id`'s state for `session`.
    pub fn state_path(&self, session: &SessionRef, source_id: &str) -> Result<PathBuf> {
        let mut dir = self.base_path.clone();
        if let Some(key) = &self.owner_state_key {
            let owner = match session.state.get(key) {
                None | Some(Value::Null) => {
                    return Err(Error::Configuration(format!(
                    "TodoFileStore requires session.state[{}] to be set for file-backed storage.",
                    py_repr(key)
                )))
                }
                Some(Value::String(s)) => s,
                Some(other) => other.to_string(),
            };
            let owner = Self::path_segment(&owner, "owner", false)?;
            dir = dir
                .join(format!("{}{owner}", self.owner_prefix))
                .join(&self.kind);
        }
        dir = dir.join(Self::path_segment(&session.session_id, "session_id", true)?);
        let path = dir.join(self.state_file_name(source_id)?);
        // Containment check (upstream: `resolve().is_relative_to(base)`):
        // every derived segment is a single, non-traversing component.
        let relative = path.strip_prefix(&self.base_path).map_err(|_| {
            Error::Configuration(format!(
                "Todo file path escaped base directory for session_id {}.",
                py_repr(&session.session_id)
            ))
        })?;
        if relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(Error::Configuration(format!(
                "Todo file path escaped base directory for session_id {}.",
                py_repr(&session.session_id)
            )));
        }
        Ok(path)
    }
}

#[async_trait]
impl TodoStore for TodoFileStore {
    async fn load_state(
        &self,
        session: &SessionRef,
        source_id: &str,
    ) -> Result<(Vec<TodoItem>, i64)> {
        let path = self.state_path(session, source_id)?;
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 1)),
            Err(e) => {
                return Err(Error::other(format!(
                    "failed to read todo file {path:?}: {e}"
                )))
            }
        };
        let display = path.display().to_string();
        let payload: Value = serde_json::from_str(&text).map_err(|e| {
            Error::Serialization(format!("Todo file {display} is not valid JSON: {e}"))
        })?;
        let map = payload.as_object().ok_or_else(|| {
            Error::Serialization(format!("Todo file {display} must contain a JSON object."))
        })?;
        let items = match map.get("items") {
            None => Vec::new(),
            Some(Value::Array(items)) => parse_todo_items(items, &format!("todo file {display}"))?,
            Some(_) => {
                return Err(Error::Serialization(format!(
                    "Todo file {display} has a non-list 'items' field."
                )))
            }
        };
        let next_id = match map.get("next_id") {
            None => 1,
            Some(v) => v.as_i64().ok_or_else(|| {
                Error::Serialization(format!(
                    "Todo file {display} has a non-integer 'next_id' field."
                ))
            })?,
        };
        let next = safe_next_id(&items, next_id);
        Ok((items, next))
    }

    async fn save_state(
        &self,
        session: &SessionRef,
        items: &[TodoItem],
        next_id: i64,
        source_id: &str,
    ) -> Result<()> {
        let path = self.state_path(session, source_id)?;
        let payload = TodoFilePayload {
            items,
            next_id: safe_next_id(items, next_id),
        };
        let text = format!("{}\n", py_json_dumps(&payload, true));
        atomic_write(&path, &text).await
    }
}

/// Field-ordered view of the todo file payload (`items` before `next_id`,
/// each item `id, title, description, is_complete`) — upstream's
/// `json.dumps` insertion order.
#[derive(Serialize)]
struct TodoFilePayload<'a> {
    items: &'a [TodoItem],
    next_id: i64,
}

/// Write `text` to `path` via a uniquely named sibling temp file plus an
/// atomic rename, so a crash mid-write cannot leave a truncated file.
pub(crate) async fn atomic_write(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| Error::other(format!("failed to create {parent:?}: {e}")))?;
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "state".into());
    let tmp = path.with_file_name(format!("{name}.tmp.{}", uuid::Uuid::new_v4().simple()));
    if let Err(e) = tokio::fs::write(&tmp, text).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(Error::other(format!("failed to write {tmp:?}: {e}")));
    }
    if let Err(e) = tokio::fs::rename(&tmp, path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(Error::other(format!("failed to replace {path:?}: {e}")));
    }
    Ok(())
}

#[derive(Deserialize)]
struct AddItemArgs {
    title: String,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Deserialize)]
struct AddArgs {
    todos: Vec<AddItemArgs>,
}

#[derive(Deserialize)]
struct CompleteItemArgs {
    id: i64,
    reason: String,
}

#[derive(Deserialize)]
struct CompleteArgs {
    items: Vec<CompleteItemArgs>,
}

#[derive(Deserialize)]
struct RemoveArgs {
    ids: Vec<i64>,
}

/// Context provider giving an agent a session-scoped todo list.
///
/// Mirrors upstream `TodoProvider`: on every run it injects
/// [`DEFAULT_TODO_INSTRUCTIONS`] (or an override), a `### Current todo list`
/// user message, and five tools, all `never_require` approval:
///
/// - `todos_add` — add one or more items (`{"todos": [{"title", "description"?}]}`);
///   returns the created items as JSON.
/// - `todos_complete` — mark items complete (`{"items": [{"id", "reason"}]}`);
///   returns `{"completed": n}`.
/// - `todos_remove` — remove items by id (`{"ids": [...]}`); returns `{"removed": n}`.
/// - `todos_get_remaining` — the incomplete items as JSON.
/// - `todos_get_all` — every item as JSON.
///
/// State lives in the configured [`TodoStore`] (default
/// [`TodoSessionStore`]). Read-modify-write tool calls are serialized by a
/// provider-wide async lock (upstream keys one lock per session in a
/// weak map; one lock is simpler and cannot leak, at the cost of
/// serializing mutations across sessions sharing one provider).
#[derive(Clone)]
pub struct TodoProvider {
    source_id: String,
    instructions: String,
    store: Arc<dyn TodoStore>,
    lock: Arc<tokio::sync::Mutex<()>>,
}

impl std::fmt::Debug for TodoProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TodoProvider")
            .field("source_id", &self.source_id)
            .finish_non_exhaustive()
    }
}

impl Default for TodoProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl TodoProvider {
    /// A provider with upstream's defaults.
    pub fn new() -> Self {
        Self {
            source_id: DEFAULT_TODO_SOURCE_ID.into(),
            instructions: DEFAULT_TODO_INSTRUCTIONS.into(),
            store: Arc::new(TodoSessionStore),
            lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Override the source id (session-state key / store namespace).
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// Override the injected instructions. An empty string keeps the
    /// default, as upstream's `instructions or DEFAULT_TODO_INSTRUCTIONS` does.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        let instructions = instructions.into();
        if !instructions.is_empty() {
            self.instructions = instructions;
        }
        self
    }

    /// Use a custom [`TodoStore`] (e.g. [`TodoFileStore`]).
    pub fn store(mut self, store: Arc<dyn TodoStore>) -> Self {
        self.store = store;
        self
    }

    /// The configured source id.
    pub fn get_source_id(&self) -> &str {
        &self.source_id
    }

    /// The configured store.
    pub fn get_store(&self) -> &Arc<dyn TodoStore> {
        &self.store
    }

    /// The configured instructions.
    pub fn get_instructions(&self) -> &str {
        &self.instructions
    }

    /// Load `session`'s items from the store (used by the loop helpers).
    pub async fn load_items(&self, session: &SessionRef) -> Result<Vec<TodoItem>> {
        self.store.load_items(session, &self.source_id).await
    }

    fn tools(&self, session: SessionRef) -> Vec<agent_framework_core::tools::ToolDefinition> {
        let p = self.clone();
        let s = session.clone();
        let add = function_tool(
            "todos_add",
            "Add one or more todo items for the current session.",
            json!({
                "type": "object",
                "properties": {"todos": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
                    "required": ["title"],
                }}},
                "required": ["todos"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let s = s.clone();
                async move {
                    let args: AddArgs = parse_args("todos_add", args)?;
                    if args.todos.is_empty() {
                        return Err(Error::tool("todos must contain at least one item."));
                    }
                    let inputs = args
                        .todos
                        .iter()
                        .map(|t| TodoInput::new(&t.title, t.description.clone()))
                        .collect::<Result<Vec<_>>>()?;
                    let _guard = p.lock.lock().await;
                    let (mut items, mut next_id) = p.store.load_state(&s, &p.source_id).await?;
                    let mut created = Vec::new();
                    for input in inputs {
                        let item = TodoItem::new(
                            next_id,
                            input.title,
                            input.description.map(|d| d.trim().to_string()),
                        );
                        items.push(item.clone());
                        created.push(item);
                        next_id += 1;
                    }
                    p.store
                        .save_state(&s, &items, next_id, &p.source_id)
                        .await?;
                    Ok(Value::String(py_json_dumps(&created, true)))
                }
            },
        );

        let p = self.clone();
        let s = session.clone();
        let complete = function_tool(
            "todos_complete",
            "Mark one or more todo items as complete.\n\nEach entry has an id (int) and a reason (string) describing how/why the item was completed.",
            json!({
                "type": "object",
                "properties": {"items": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "reason": {"type": "string"}},
                    "required": ["id", "reason"],
                }}},
                "required": ["items"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let s = s.clone();
                async move {
                    let args: CompleteArgs = parse_args("todos_complete", args)?;
                    if args.items.is_empty() {
                        return Err(Error::tool("items must contain at least one entry."));
                    }
                    let parsed = args
                        .items
                        .iter()
                        .map(|i| TodoCompleteInput::new(i.id, &i.reason))
                        .collect::<Result<Vec<_>>>()?;
                    let ids: std::collections::HashSet<i64> = parsed.iter().map(|i| i.id).collect();
                    let _guard = p.lock.lock().await;
                    let (mut items, next_id) = p.store.load_state(&s, &p.source_id).await?;
                    let mut completed = 0;
                    for item in &mut items {
                        if !item.is_complete && ids.contains(&item.id) {
                            item.is_complete = true;
                            completed += 1;
                        }
                    }
                    if completed > 0 {
                        p.store.save_state(&s, &items, next_id, &p.source_id).await?;
                    }
                    Ok(Value::String(format!("{{\"completed\": {completed}}}")))
                }
            },
        );

        let p = self.clone();
        let s = session.clone();
        let remove = function_tool(
            "todos_remove",
            "Remove one or more todo items by ID.",
            json!({
                "type": "object",
                "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
                "required": ["ids"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let s = s.clone();
                async move {
                    let args: RemoveArgs = parse_args("todos_remove", args)?;
                    if args.ids.is_empty() {
                        return Err(Error::tool("ids must contain at least one todo ID."));
                    }
                    let ids: std::collections::HashSet<i64> = args.ids.into_iter().collect();
                    let _guard = p.lock.lock().await;
                    let (items, next_id) = p.store.load_state(&s, &p.source_id).await?;
                    let before = items.len();
                    let remaining: Vec<TodoItem> =
                        items.into_iter().filter(|i| !ids.contains(&i.id)).collect();
                    let removed = before - remaining.len();
                    if removed > 0 {
                        p.store
                            .save_state(&s, &remaining, next_id, &p.source_id)
                            .await?;
                    }
                    Ok(Value::String(format!("{{\"removed\": {removed}}}")))
                }
            },
        );

        let p = self.clone();
        let s = session.clone();
        let remaining = function_tool(
            "todos_get_remaining",
            "Retrieve only incomplete todo items for the current session.",
            empty_object_schema(),
            ApprovalMode::NeverRequire,
            move |_args| {
                let p = p.clone();
                let s = s.clone();
                async move {
                    let items: Vec<TodoItem> = p
                        .store
                        .load_items(&s, &p.source_id)
                        .await?
                        .into_iter()
                        .filter(|i| !i.is_complete)
                        .collect();
                    Ok(Value::String(py_json_dumps(&items, true)))
                }
            },
        );

        let p = self.clone();
        let s = session;
        let all = function_tool(
            "todos_get_all",
            "Retrieve all todo items for the current session.",
            empty_object_schema(),
            ApprovalMode::NeverRequire,
            move |_args| {
                let p = p.clone();
                let s = s.clone();
                async move {
                    let items = p.store.load_items(&s, &p.source_id).await?;
                    Ok(Value::String(py_json_dumps(&items, true)))
                }
            },
        );
        vec![add, complete, remove, remaining, all]
    }
}

/// Render the `### Current todo list` context message body.
pub(crate) fn render_todo_list(items: &[TodoItem]) -> String {
    let lines: Vec<String> = items
        .iter()
        .map(|item| {
            let status = if item.is_complete { "done" } else { "open" };
            let mut line = format!("- {} [{status}] {}", item.id, item.title);
            if let Some(d) = item.description.as_deref().filter(|d| !d.is_empty()) {
                line.push_str(": ");
                line.push_str(d);
            }
            line
        })
        .collect();
    let body = if lines.is_empty() {
        "- none yet".to_string()
    } else {
        lines.join("\n")
    };
    format!("### Current todo list\n{body}")
}

#[async_trait]
impl ContextProvider for TodoProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session = SessionRef::from_context(ctx, "TodoProvider")?;
        ctx.add_instructions(self.instructions.clone());
        ctx.tools.extend(self.tools(session.clone()));
        let items = self.store.load_items(&session, &self.source_id).await?;
        ctx.messages.push(Message::user(render_todo_list(&items)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::session::SessionState;

    fn session() -> SessionRef {
        SessionRef {
            session_id: "s1".into(),
            state: SessionState::new(),
        }
    }

    #[test]
    fn todo_item_round_trips_and_validates() {
        let item = TodoItem::new(1, "t", Some("d".into()));
        assert_eq!(TodoItem::from_value(&item.to_value()).unwrap(), item);
        assert!(TodoItem::from_value(&json!({"id": "x", "title": "t"})).is_err());
        assert!(TodoItem::from_value(&json!({"id": 1, "title": "  "})).is_err());
        assert!(TodoItem::from_value(&json!({"id": 1, "title": "t", "is_complete": 1})).is_err());
        assert!(TodoInput::new("  ", None).is_err());
        assert_eq!(TodoInput::new(" a ", None).unwrap().title, "a");
        assert!(TodoCompleteInput::new(1, " ").is_err());
    }

    #[tokio::test]
    async fn session_store_initializes_and_round_trips() {
        let s = session();
        let store = TodoSessionStore;
        assert_eq!(store.load_state(&s, "todo").await.unwrap(), (vec![], 1));
        assert_eq!(s.state.get("todo"), Some(json!({})));
        let items = vec![TodoItem::new(5, "a", None)];
        store.save_state(&s, &items, 2, "todo").await.unwrap();
        // next_id is clamped above the max persisted id.
        assert_eq!(store.load_state(&s, "todo").await.unwrap(), (items, 6));
        s.state.insert("bad", json!({"items": [1]}));
        assert!(store.load_state(&s, "bad").await.is_err());
        s.state.insert("bad2", json!({"items": {}}));
        assert!(store.load_state(&s, "bad2").await.is_err());
        s.state.insert("bad3", json!("x"));
        assert!(store.load_state(&s, "bad3").await.is_err());
    }

    #[tokio::test]
    async fn file_store_round_trips_and_rejects_separators() {
        let dir = std::env::temp_dir().join(format!("todo-test-{}", uuid::Uuid::new_v4()));
        let store = TodoFileStore::new(&dir);
        let s = session();
        assert_eq!(store.load_state(&s, "todo").await.unwrap(), (vec![], 1));
        assert!(!dir.exists(), "load must not create directories");
        let items = vec![TodoItem::new(1, "a", None)];
        store.save_state(&s, &items, 2, "todo").await.unwrap();
        let path = store.state_path(&s, "todo").unwrap();
        assert_eq!(path, dir.join("s1").join("todos.todo.json"));
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text,
            "{\"items\": [{\"id\": 1, \"title\": \"a\", \"description\": null, \"is_complete\": false}], \"next_id\": 2}\n"
        );
        assert_eq!(store.load_state(&s, "todo").await.unwrap(), (items, 2));
        let bad = SessionRef {
            session_id: "../x".into(),
            state: SessionState::new(),
        };
        assert!(store.state_path(&bad, "todo").is_err());
        let dots = SessionRef {
            session_id: "..".into(),
            state: SessionState::new(),
        };
        let p = store.state_path(&dots, "todo").unwrap();
        assert!(p.starts_with(&dir) && !p.components().any(|c| c == Component::ParentDir));
        let owned = TodoFileStore::new(&dir).owner_state_key("owner");
        assert!(owned.state_path(&s, "todo").is_err());
        s.state.insert("owner", json!("alice"));
        assert_eq!(
            owned.state_path(&s, "todo").unwrap(),
            dir.join("alice")
                .join("todos")
                .join("s1")
                .join("todos.todo.json")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn renders_current_todo_list() {
        assert_eq!(render_todo_list(&[]), "### Current todo list\n- none yet");
        let mut done = TodoItem::new(2, "b", Some("desc".into()));
        done.is_complete = true;
        assert_eq!(
            render_todo_list(&[TodoItem::new(1, "a", Some(String::new())), done]),
            "### Current todo list\n- 1 [open] a\n- 2 [done] b: desc"
        );
    }
}
