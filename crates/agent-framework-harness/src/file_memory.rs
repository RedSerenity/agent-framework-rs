//! Session-scoped, file-based memory.
//!
//! Rust equivalent of upstream `agent_framework._harness._file_memory`
//! (`FileMemoryProvider`). Each memory is a file under a working folder
//! derived from the session id (or an explicit scope); large files can carry
//! a `<stem>_description.md` sidecar, and a `memories.md` index is rebuilt on
//! every write/delete and injected into the agent's context.

use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::tools::{ApprovalMode, ToolDefinition};
use agent_framework_core::types::Message;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::file_access::{line_edit_schema, AgentFileStore, ReplaceArgs, ReplaceLinesArgs};
use crate::paths::{
    apply_replace, apply_replace_lines, combine_paths, matches_glob, normalize_relative_path,
    storage_path_segment, FileStoreError,
};
use crate::util::{function_tool, parse_args};

/// Default source id of [`FileMemoryProvider`]. Mirrors upstream
/// `DEFAULT_FILE_MEMORY_SOURCE_ID`.
pub const DEFAULT_FILE_MEMORY_SOURCE_ID: &str = "file_memory";

/// Default instructions of [`FileMemoryProvider`]. Mirrors upstream
/// `DEFAULT_FILE_MEMORY_INSTRUCTIONS` verbatim.
pub const DEFAULT_FILE_MEMORY_INSTRUCTIONS: &str = concat!(
    "## File Based Memory\n",
    "You have access to a session-scoped, file-based memory system via the `file_memory_*` tools for storing and retrieving information across interactions. ",
    "These files act as your working memory for the current session and are isolated from other sessions. ",
    "Use these tools to store plans, memories, processing results, or downloaded data.\n\n",
    "- Use descriptive file names (e.g., \"projectarchitecture.md\", \"userpreferences.md\").\n",
    "- Include a description when writing a file to help with future discovery.\n",
    "- Before starting new tasks, use file_memory_ls and file_memory_grep to check for relevant existing memories to avoid duplicate work.\n",
    "- Keep memories up-to-date by overwriting files when information changes.\n",
    "- When you receive large amounts of data (e.g., downloaded web pages, API responses, research results), write them to files if they will be required later, so that they are not lost when older context is compacted or truncated. This ensures important data remains accessible across long-running sessions.",
);

const DESCRIPTION_SUFFIX: &str = "_description.md";
/// The automatically maintained index file. Mirrors upstream
/// `_MEMORY_INDEX_FILE_NAME`.
pub const MEMORY_INDEX_FILE_NAME: &str = "memories.md";
/// Maximum entries listed in the index. Mirrors upstream `_MAX_INDEX_ENTRIES`.
pub const MAX_INDEX_ENTRIES: usize = 50;
const ENCODED_SCOPE_PREFIX: &str = "~scope-";

/// The companion description file name of `file_name`: the suffix replaces
/// the extension when present (`notes.md` → `notes_description.md`).
/// Mirrors upstream `_description_file_name`.
pub fn description_file_name(file_name: &str) -> String {
    match file_name.rfind('.') {
        Some(index) if index > 0 => format!("{}{DESCRIPTION_SUFFIX}", &file_name[..index]),
        _ => format!("{file_name}{DESCRIPTION_SUFFIX}"),
    }
}

/// Whether `file_name` is internal bookkeeping hidden from the agent (a
/// description sidecar or the index). Mirrors upstream `_is_internal_file`.
pub fn is_internal_file(file_name: &str) -> bool {
    let lowered = file_name.to_lowercase();
    lowered.ends_with(DESCRIPTION_SUFFIX) || lowered == MEMORY_INDEX_FILE_NAME
}

#[derive(Deserialize)]
struct WriteArgs {
    file_name: String,
    content: String,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Deserialize)]
struct FileNameArgs {
    file_name: String,
}

#[derive(Deserialize)]
struct ListArgs {
    #[serde(default)]
    glob_pattern: Option<String>,
}

#[derive(Deserialize)]
struct GrepArgs {
    regex_pattern: String,
    #[serde(default)]
    glob_pattern: Option<String>,
}

/// Context provider giving an agent session-scoped, file-based memory.
///
/// Mirrors upstream `FileMemoryProvider`. Tools (all `never_require`):
/// `file_memory_write`, `file_memory_read`, `file_memory_delete`,
/// `file_memory_ls`, `file_memory_grep`, `file_memory_replace`,
/// `file_memory_replace_lines`. Memory is flat (nested names are refused, as
/// they would never surface again), and the sidecars plus `memories.md` are
/// reserved.
///
/// The working folder is the [`scope`](Self::scope) (default: the session
/// id) mapped onto exactly one folder by [`storage_path_segment`] (prefix
/// `~scope-`), so two byte-distinct sessions or scopes never share memories;
/// a missing scope fails closed.
#[derive(Clone)]
pub struct FileMemoryProvider {
    store: Arc<dyn AgentFileStore>,
    source_id: String,
    scope: Option<String>,
    instructions: String,
    write_lock: Arc<tokio::sync::Mutex<()>>,
}

impl std::fmt::Debug for FileMemoryProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMemoryProvider")
            .field("source_id", &self.source_id)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl FileMemoryProvider {
    /// A provider over `store` with upstream's defaults.
    pub fn new(store: Arc<dyn AgentFileStore>) -> Self {
        Self {
            store,
            source_id: DEFAULT_FILE_MEMORY_SOURCE_ID.into(),
            scope: None,
            instructions: DEFAULT_FILE_MEMORY_INSTRUCTIONS.into(),
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Override the source id.
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// Group memories under an explicit namespace (e.g. a user id) instead
    /// of the session id. Treated as an opaque key, not a path.
    pub fn scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// Override the instructions (empty keeps the default).
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        let instructions = instructions.into();
        if !instructions.is_empty() {
            self.instructions = instructions;
        }
        self
    }

    /// The underlying store.
    pub fn store(&self) -> &Arc<dyn AgentFileStore> {
        &self.store
    }

    /// The working folder for a run with `session_id`. Mirrors upstream
    /// `_resolve_working_folder`.
    pub fn resolve_working_folder(&self, session_id: Option<&str>) -> Result<String> {
        let raw = self
            .scope
            .as_deref()
            .filter(|s| !s.is_empty())
            .or(session_id)
            .unwrap_or_default();
        if raw.is_empty() {
            return Err(Error::Configuration(
                "FileMemoryProvider requires a memory scope: pass an explicit 'scope' or run with a session that has a 'session_id'. Without one, memories cannot be isolated from other scopes.".into(),
            ));
        }
        Ok(storage_path_segment(raw, ENCODED_SCOPE_PREFIX))
    }

    /// Rebuild `memories.md` for `working_folder`. Mirrors upstream
    /// `_rebuild_index`.
    async fn rebuild_index(&self, working_folder: &str) -> std::result::Result<(), FileStoreError> {
        let entries = self.store.list_children(working_folder).await?;
        let mut files: Vec<String> = entries
            .into_iter()
            .filter(|e| !e.is_directory() && !is_internal_file(&e.name))
            .map(|e| e.name)
            .collect();
        files.sort_by_key(|n| n.to_lowercase());
        let mut lines = vec!["# Memory Index".to_string(), String::new()];
        for file_name in files.iter().take(MAX_INDEX_ENTRIES) {
            let description = self
                .store
                .read(&combine_paths(
                    working_folder,
                    &description_file_name(file_name),
                ))
                .await?;
            match description
                .as_deref()
                .map(str::trim)
                .filter(|d| !d.is_empty())
            {
                Some(d) => lines.push(format!("- **{file_name}**: {d}")),
                None => lines.push(format!("- **{file_name}**")),
            }
        }
        let index_path = combine_paths(working_folder, MEMORY_INDEX_FILE_NAME);
        self.store
            .write(&index_path, &format!("{}\n", lines.join("\n")), true)
            .await
    }

    fn tools(&self, folder: String) -> Vec<ToolDefinition> {
        let folder = Arc::new(folder);
        let reserved = |name: &str, verb: &str| {
            format!(
                "Could not {verb} file '{name}': the file name is reserved for internal use. Please choose a different file name."
            )
        };
        let mut tools = Vec::new();

        let p = self.clone();
        let f = folder.clone();
        tools.push(function_tool(
            "file_memory_write",
            "Write a memory file with the given name and content. Overwrites the file if it already exists. Include a description for large files to provide a summary that helps with future discovery.",
            json!({
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Flat file name to write under; must not contain path separators."},
                    "content": {"type": "string", "description": "Full text content to write to the file."},
                    "description": {"type": "string", "description": "Optional summary used to aid future discovery; recommended for large files."},
                },
                "required": ["file_name", "content"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let f = f.clone();
                async move {
                    let args: WriteArgs = parse_args("file_memory_write", args)?;
                    let name = &args.file_name;
                    let normalized = match normalize_relative_path(name, false) {
                        Ok(n) => n,
                        Err(e) => return Ok(Value::String(format!("Could not write file '{name}': {e}"))),
                    };
                    if normalized.contains('/') {
                        return Ok(Value::String(format!(
                            "Could not write file '{name}': memory files must not be written into a subdirectory. Please choose a flat file name without path separators."
                        )));
                    }
                    if is_internal_file(&normalized) {
                        return Ok(Value::String(reserved(name, "write")));
                    }
                    let path = combine_paths(&f, &normalized);
                    let desc_path = combine_paths(&f, &description_file_name(&normalized));
                    let has_description = args.description.as_deref().is_some_and(|d| !d.trim().is_empty());
                    let _guard = p.write_lock.lock().await;
                    let result: std::result::Result<(), FileStoreError> = async {
                        p.store.write(&path, &args.content, true).await?;
                        if has_description {
                            p.store
                                .write(&desc_path, args.description.as_deref().unwrap_or_default(), true)
                                .await?;
                        } else {
                            p.store.delete(&desc_path).await?;
                        }
                        p.rebuild_index(&f).await
                    }
                    .await;
                    Ok(Value::String(match result {
                        Err(e) => format!("Could not write file '{name}': {e}"),
                        Ok(()) if has_description => format!("File '{name}' written with description."),
                        Ok(()) => format!("File '{name}' written."),
                    }))
                }
            },
        ));

        let p = self.clone();
        let f = folder.clone();
        tools.push(function_tool(
            "file_memory_read",
            "Read the content of a memory file by name. Returns the file content or a message indicating the file was not found. Line numbers count lines split on \\n only: a lone \\r never starts a new line, each line keeps its own terminator, and content ending in a newline has a final empty line.",
            json!({
                "type": "object",
                "properties": {"file_name": {"type": "string", "description": "Name of the memory file to read."}},
                "required": ["file_name"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let f = f.clone();
                async move {
                    let args: FileNameArgs = parse_args("file_memory_read", args)?;
                    let name = &args.file_name;
                    let normalized = match normalize_relative_path(name, false) {
                        Ok(n) => n,
                        Err(e) => return Ok(Value::String(format!("Could not read file '{name}': {e}"))),
                    };
                    if normalized.contains('/') {
                        return Ok(Value::String(format!("File '{name}' not found.")));
                    }
                    Ok(Value::String(match p.store.read(&combine_paths(&f, &normalized)).await {
                        Ok(Some(content)) => content,
                        Ok(None) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not read file '{name}': {e}"),
                    }))
                }
            },
        ));

        let p = self.clone();
        let f = folder.clone();
        tools.push(function_tool(
            "file_memory_delete",
            "Delete a memory file by name. Also removes its companion description file if one exists.",
            json!({
                "type": "object",
                "properties": {"file_name": {"type": "string", "description": "Name of the memory file to delete."}},
                "required": ["file_name"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let f = f.clone();
                async move {
                    let args: FileNameArgs = parse_args("file_memory_delete", args)?;
                    let name = &args.file_name;
                    let normalized = match normalize_relative_path(name, false) {
                        Ok(n) => n,
                        Err(e) => return Ok(Value::String(format!("Could not delete file '{name}': {e}"))),
                    };
                    if normalized.contains('/') {
                        return Ok(Value::String(format!("File '{name}' not found.")));
                    }
                    let path = combine_paths(&f, &normalized);
                    let desc_path = combine_paths(&f, &description_file_name(&normalized));
                    let _guard = p.write_lock.lock().await;
                    let result: std::result::Result<bool, FileStoreError> = async {
                        let deleted = p.store.delete(&path).await?;
                        p.store.delete(&desc_path).await?;
                        p.rebuild_index(&f).await?;
                        Ok(deleted)
                    }
                    .await;
                    Ok(Value::String(match result {
                        Ok(true) => format!("File '{name}' deleted."),
                        Ok(false) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not delete file '{name}': {e}"),
                    }))
                }
            },
        ));

        let p = self.clone();
        let f = folder.clone();
        tools.push(function_tool(
            "file_memory_ls",
            "List all memory files with their descriptions (if available). Optionally filter file names with a glob_pattern (e.g. \"*.md\"). Internal files (description sidecars and the memory index) are not shown. Each entry is {\"name\": <name>, \"type\": \"file\", \"description\": <desc-or-null>}.",
            json!({
                "type": "object",
                "properties": {"glob_pattern": {"type": "string", "description": "Optional glob (e.g. \"*.md\") matched against file names to filter the listing."}},
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let f = f.clone();
                async move {
                    let args: ListArgs = parse_args("file_memory_ls", args)?;
                    let entries = match p.store.list_children(&f).await {
                        Ok(e) => e,
                        Err(e) => return Ok(Value::String(format!("Could not list memory files: {e}"))),
                    };
                    let names: Vec<String> = entries.into_iter().filter(|e| !e.is_directory()).map(|e| e.name).collect();
                    let mut out = Vec::new();
                    for name in &names {
                        if is_internal_file(name) || !matches_glob(name, args.glob_pattern.as_deref()) {
                            continue;
                        }
                        let sidecar = description_file_name(name);
                        let description = if names.contains(&sidecar) {
                            p.store.read(&combine_paths(&f, &sidecar)).await.ok().flatten()
                        } else {
                            None
                        };
                        out.push(json!({"name": name, "type": "file", "description": description}));
                    }
                    Ok(Value::Array(out))
                }
            },
        ));

        let p = self.clone();
        let f = folder.clone();
        tools.push(function_tool(
            "file_memory_grep",
            "Search memory file contents using a case-insensitive regular expression. Optionally filter which files to search using a glob pattern (e.g., \"*.md\", \"research*\"). Returns matching file names, content snippets, and matching lines with line numbers. The regex_pattern must be 256 characters or fewer.",
            json!({
                "type": "object",
                "properties": {
                    "regex_pattern": {"type": "string", "description": "Case-insensitive regex matched against file contents; 256 characters or fewer."},
                    "glob_pattern": {"type": "string", "description": "Optional glob to filter which files are searched (e.g. \"*.md\", \"research*\")."},
                },
                "required": ["regex_pattern"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let f = f.clone();
                async move {
                    let args: GrepArgs = parse_args("file_memory_grep", args)?;
                    let glob = args.glob_pattern.filter(|g| !g.trim().is_empty());
                    match p.store.search(&f, &args.regex_pattern, glob.as_deref(), false).await {
                        Err(e) => Ok(Value::String(format!("Could not search memory files: {e}"))),
                        Ok(results) => Ok(Value::Array(
                            results
                                .into_iter()
                                .filter(|r| !is_internal_file(&r.file_name))
                                .map(|r| serde_json::to_value(r).unwrap_or(Value::Null))
                                .collect(),
                        )),
                    }
                }
            },
        ));

        let p = self.clone();
        let f = folder.clone();
        tools.push(function_tool(
            "file_memory_replace",
            "Replace occurrences of old_string with new_string in a memory file. Fails if old_string is not found, or if it occurs more than once and replace_all is false. Returns the number of occurrences replaced.",
            json!({
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of the memory file to modify."},
                    "old_string": {"type": "string", "description": "Substring to find and replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "default": false, "description": "When true, replace every occurrence; when false, fail unless exactly one occurrence exists."},
                },
                "required": ["file_name", "old_string", "new_string"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let f = f.clone();
                async move {
                    let args: ReplaceArgs = parse_args("file_memory_replace", args)?;
                    let name = args.file_name.clone();
                    let normalized = match normalize_relative_path(&name, false) {
                        Ok(n) => n,
                        Err(e) => return Ok(Value::String(format!("Could not replace in file '{name}': {e}"))),
                    };
                    if normalized.contains('/') {
                        return Ok(Value::String(format!("File '{name}' not found.")));
                    }
                    if is_internal_file(&normalized) {
                        return Ok(Value::String(reserved(&name, "replace in")));
                    }
                    let path = combine_paths(&f, &normalized);
                    let _guard = p.write_lock.lock().await;
                    let result: std::result::Result<Option<usize>, FileStoreError> = async {
                        let Some(content) = p.store.read(&path).await? else {
                            return Ok(None);
                        };
                        let (new_content, count) =
                            apply_replace(&content, &args.old_string, &args.new_string, args.replace_all)?;
                        p.store.write(&path, &new_content, true).await?;
                        Ok(Some(count))
                    }
                    .await;
                    Ok(Value::String(match result {
                        Ok(Some(count)) => format!("Replaced {count} occurrence(s) in '{name}'."),
                        Ok(None) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not replace in file '{name}': {e}"),
                    }))
                }
            },
        ));

        let p = self.clone();
        let f = folder;
        tools.push(function_tool(
            "file_memory_replace_lines",
            "Replace lines in a memory file. Provide a list of edits, each with a 1-based line_number and a literal new_line (include your own trailing newline); an empty new_line deletes the line, including its line break. Fails on out-of-range or duplicate line numbers. Line numbers count lines split on \\n only: a lone \\r never starts a new line, each line keeps its own terminator, and content ending in a newline has a final empty line.",
            json!({
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of the memory file to modify."},
                    "edits": {"type": "array", "description": "List of 1-based line numbers and their literal replacement text.", "items": line_edit_schema("file_memory_grep", false)},
                },
                "required": ["file_name", "edits"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let p = p.clone();
                let f = f.clone();
                async move {
                    let args: ReplaceLinesArgs = parse_args("file_memory_replace_lines", args)?;
                    let name = args.file_name.clone();
                    let normalized = match normalize_relative_path(&name, false) {
                        Ok(n) => n,
                        Err(e) => return Ok(Value::String(format!("Could not edit file '{name}': {e}"))),
                    };
                    if normalized.contains('/') {
                        return Ok(Value::String(format!("File '{name}' not found.")));
                    }
                    if is_internal_file(&normalized) {
                        return Ok(Value::String(reserved(&name, "edit")));
                    }
                    let path = combine_paths(&f, &normalized);
                    let _guard = p.write_lock.lock().await;
                    let result: std::result::Result<bool, FileStoreError> = async {
                        let Some(content) = p.store.read(&path).await? else {
                            return Ok(false);
                        };
                        let new_content = apply_replace_lines(&content, &args.edits)?;
                        p.store.write(&path, &new_content, true).await?;
                        Ok(true)
                    }
                    .await;
                    Ok(Value::String(match result {
                        Ok(true) => format!("Replaced {} line(s) in '{name}'.", args.edits.len()),
                        Ok(false) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not edit file '{name}': {e}"),
                    }))
                }
            },
        ));
        tools
    }
}

/// The name of every tool [`FileMemoryProvider`] contributes.
pub const FILE_MEMORY_TOOL_NAMES: [&str; 7] = [
    "file_memory_write",
    "file_memory_read",
    "file_memory_delete",
    "file_memory_ls",
    "file_memory_grep",
    "file_memory_replace",
    "file_memory_replace_lines",
];

#[async_trait]
impl ContextProvider for FileMemoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let folder = self.resolve_working_folder(ctx.session_id.as_deref())?;
        self.store.create_directory(&folder).await?;
        ctx.add_instructions(self.instructions.clone());
        ctx.tools.extend(self.tools(folder.clone()));
        // A corrupt or unavailable index must not block the run; it
        // self-heals on the next write/delete.
        let index = match self
            .store
            .read(&combine_paths(&folder, MEMORY_INDEX_FILE_NAME))
            .await
        {
            Ok(index) => index,
            Err(e) => {
                tracing::warn!(error = %e, "Could not read memory index; skipping index injection");
                None
            }
        };
        if let Some(index) = index.filter(|i| !i.trim().is_empty()) {
            ctx.messages.push(Message::user(format!(
                "The following is your memory index — a list of files you have previously written. You can read any of these files using the file_memory_read tool.\n\n{index}"
            )));
        }
        Ok(())
    }
}
