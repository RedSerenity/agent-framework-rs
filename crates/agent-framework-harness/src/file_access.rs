//! File stores and the file-access provider.
//!
//! Rust equivalent of upstream `agent_framework._harness._file_access`:
//! the [`AgentFileStore`] abstraction (shared with
//! [`FileMemoryProvider`](crate::file_memory::FileMemoryProvider)), its
//! [`InMemoryAgentFileStore`] and disk-backed [`FileSystemAgentFileStore`]
//! implementations, and [`FileAccessProvider`], which exposes CRUD/search
//! tools over a *shared* store.
//!
//! # Sandboxing
//!
//! [`FileSystemAgentFileStore`] confines every operation to its root exactly
//! as upstream does:
//!
//! - tool-supplied paths are normalized by
//!   [`normalize_relative_path`],
//!   which rejects absolute paths, drive letters and `.`/`..` segments;
//! - every existing segment between the root and the target is probed with
//!   `lstat` and a symbolic link (or Windows reparse point) anywhere along it
//!   is refused; a probe that cannot be completed fails closed;
//! - the root itself is refused when it is (or has been swapped for) a link;
//! - opens pass `O_NOFOLLOW` on Unix, so a leaf swapped for a symlink between
//!   the probe and the open is refused by the kernel;
//! - listings and searches skip symlinked entries instead of following them.
//!
//! Like upstream, this is designed for single-tenant or co-operating-tenant
//! use: an intermediate directory swapped for a link *between* the probe and
//! the open cannot be closed from user space, so it is not a sandbox against
//! a hostile process sharing the root.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_framework_core::error::Result;
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::tools::{ApprovalMode, ToolDefinition};
use agent_framework_core::types::FunctionCallContent;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::paths::{
    apply_replace, apply_replace_lines, combine_paths, combine_search_path, compile_search_regex,
    is_link_or_reparse_point, matches_glob, normalize_relative_path, slice_lines, split_lines,
    storage_path_segment, FileStoreError, LineEdit, SearchPattern, SEARCH_SNIPPET_RADIUS,
    SEARCH_TIMEOUT,
};
use crate::util::{function_tool, parse_args};

/// Default source id of [`FileAccessProvider`]. Mirrors upstream
/// `DEFAULT_FILE_ACCESS_SOURCE_ID`.
pub const DEFAULT_FILE_ACCESS_SOURCE_ID: &str = "file_access";

/// Default instructions of [`FileAccessProvider`]. Mirrors upstream
/// `DEFAULT_FILE_ACCESS_INSTRUCTIONS` verbatim.
pub const DEFAULT_FILE_ACCESS_INSTRUCTIONS: &str = concat!(
    "## File Access\n",
    "You have access to a shared file storage area via the `file_access_*` tools for reading, writing, and managing files.\n",
    "These files persist beyond the current session and may be shared across sessions or agents.\n",
    "Use these tools to read input data provided by the user, write output artifacts, and manage any files the user has asked you to work with.\n\n",
    "- Never delete or overwrite existing files unless the user has explicitly asked you to do so.\n",
    "- Files may be organized into subdirectories. Use `file_access_ls` to explore the tree level by level, or `file_access_grep` to search file contents recursively across the whole store.\n",
    "- To change part of a file, find the line numbers with `file_access_grep`, read the range around them with `file_access_read_lines`, then edit with `file_access_replace_lines`. Reading the whole file first is rarely necessary.",
);

/// Prefix for session-derived working folders (upstream
/// `_ENCODED_FILE_ACCESS_SESSION_PREFIX`).
const ENCODED_FILE_ACCESS_SESSION_PREFIX: &str = "~access-";
/// Fixed namespace directory wrapping every session-scoped working folder, so
/// a session-scoped file-access folder never collides with a file-memory
/// folder sharing the same store root (upstream
/// `_FILE_ACCESS_WORKSPACE_NAMESPACE`).
const FILE_ACCESS_WORKSPACE_NAMESPACE: &str = "~access-";
/// Upstream `_SESSION_SCOPED_INSTRUCTIONS_SUFFIX`.
const SESSION_SCOPED_INSTRUCTIONS_SUFFIX: &str = "\n- Your file workspace is isolated to the current session or configured scope: files written here are not visible outside that workspace.";

const SYMLINK_MESSAGE: &str =
    "Invalid path: the resolved path contains a symbolic link or reparse point.";

/// One line within a file that matched a search pattern. Mirrors upstream
/// `FileSearchMatch`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSearchMatch {
    /// The 1-based line number, a coordinate into
    /// [`split_lines`].
    pub line_number: i64,
    /// The matching line, verbatim (terminator included).
    pub line: String,
}

/// The search result for one file. Mirrors upstream `FileSearchResult`
/// (serialized as `{"file_name", "snippet", "matching_lines"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSearchResult {
    /// The file's path relative to the searched directory.
    pub file_name: String,
    /// Up to ±50 characters of context around the first match.
    pub snippet: String,
    /// Every matching line.
    pub matching_lines: Vec<FileSearchMatch>,
}

/// One directory-listing entry: a file or a subdirectory. Mirrors upstream
/// `FileStoreEntry` (serialized as `{"name", "type"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileStoreEntry {
    /// The entry's name (not a full path).
    pub name: String,
    /// [`FileStoreEntry::FILE`] or [`FileStoreEntry::DIRECTORY`].
    #[serde(rename = "type")]
    pub entry_type: String,
}

impl FileStoreEntry {
    /// `type` value of a file entry.
    pub const FILE: &'static str = "file";
    /// `type` value of a subdirectory entry.
    pub const DIRECTORY: &'static str = "directory";

    /// A file entry.
    pub fn file(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entry_type: Self::FILE.into(),
        }
    }

    /// A directory entry.
    pub fn directory(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entry_type: Self::DIRECTORY.into(),
        }
    }

    /// Whether this is a directory entry.
    pub fn is_directory(&self) -> bool {
        self.entry_type == Self::DIRECTORY
    }
}

/// Find every line of `content` matching `pattern`, numbered by
/// [`split_lines`].
///
/// Lines are reported verbatim, terminator included; the pattern is matched
/// against the line *without* its `\r\n`/`\n`, so `$` anchors to the end of
/// the text even on a CRLF file. The snippet holds up to
/// [`SEARCH_SNIPPET_RADIUS`] characters either side of the first match.
/// Published (upstream `AgentFileStore.scan_content`) so a store that
/// overrides [`AgentFileStore::search`] can stay aligned with the line
/// editor.
pub fn scan_content(
    file_name: &str,
    content: &str,
    pattern: &SearchPattern,
) -> std::result::Result<Option<FileSearchResult>, FileStoreError> {
    let mut matching_lines = Vec::new();
    let mut snippet: Option<String> = None;
    let mut line_start_chars = 0usize;
    for (index, line) in split_lines(content).into_iter().enumerate() {
        let scanned = crate::paths::strip_line_terminator(line);
        if let Some((start, end)) = pattern.find(scanned)? {
            matching_lines.push(FileSearchMatch {
                line_number: index as i64 + 1,
                line: line.to_string(),
            });
            if snippet.is_none() {
                let match_start = line_start_chars + scanned[..start].chars().count();
                let match_len = scanned[start..end].chars().count();
                let snippet_start = match_start.saturating_sub(SEARCH_SNIPPET_RADIUS);
                let snippet_end = match_start + match_len + SEARCH_SNIPPET_RADIUS;
                snippet = Some(
                    content
                        .chars()
                        .skip(snippet_start)
                        .take(snippet_end - snippet_start)
                        .collect(),
                );
            }
        }
        line_start_chars += line.chars().count();
    }
    if matching_lines.is_empty() {
        return Ok(None);
    }
    Ok(Some(FileSearchResult {
        file_name: file_name.to_string(),
        snippet: snippet.unwrap_or_default(),
        matching_lines,
    }))
}

/// Await `work` under [`SEARCH_TIMEOUT`], mapping expiry to the documented
/// [`FileStoreError::Invalid`]. Mirrors upstream `_run_search_with_timeout`.
async fn run_search_with_timeout<F>(
    work: F,
) -> std::result::Result<Vec<FileSearchResult>, FileStoreError>
where
    F: std::future::Future<Output = std::result::Result<Vec<FileSearchResult>, FileStoreError>>,
{
    match tokio::time::timeout(SEARCH_TIMEOUT, work).await {
        Ok(result) => result,
        Err(_) => Err(FileStoreError::Invalid(
            crate::paths::search_timeout_message(),
        )),
    }
}

/// File storage used by [`FileAccessProvider`] and
/// [`FileMemoryProvider`](crate::file_memory::FileMemoryProvider).
///
/// All paths are relative, forward-slash separated, and must not escape the
/// implementation-defined root; implementations enforce that. Mirrors
/// upstream `AgentFileStore`.
#[async_trait]
pub trait AgentFileStore: Send + Sync {
    /// Write `content` to `path`. With `overwrite = false` this must be an
    /// atomic exclusive create failing with [`FileStoreError::AlreadyExists`].
    async fn write(
        &self,
        path: &str,
        content: &str,
        overwrite: bool,
    ) -> std::result::Result<(), FileStoreError>;

    /// The file's content, or `None` when it does not exist.
    async fn read(&self, path: &str) -> std::result::Result<Option<String>, FileStoreError>;

    /// Delete the file; `true` when something was removed.
    async fn delete(&self, path: &str) -> std::result::Result<bool, FileStoreError>;

    /// The direct children of `directory` (`""` for the root), directories
    /// before files.
    async fn list_children(
        &self,
        directory: &str,
    ) -> std::result::Result<Vec<FileStoreEntry>, FileStoreError>;

    /// Whether a file exists at `path`.
    async fn file_exists(&self, path: &str) -> std::result::Result<bool, FileStoreError>;

    /// Ensure `path` exists as a directory.
    async fn create_directory(&self, path: &str) -> std::result::Result<(), FileStoreError>;

    /// The names (relative to `directory`) of the files that **may** satisfy
    /// a search — a superset is harmless, an omission loses matches.
    ///
    /// The default walks [`list_children`](Self::list_children) and returns
    /// every file in scope that matches `glob_pattern`; a store with a native
    /// index should override this to narrow the candidates (remembering the
    /// search is case-insensitive). Mirrors upstream `find_matching_files`.
    async fn find_matching_files(
        &self,
        directory: &str,
        regex_pattern: &str,
        glob_pattern: Option<&str>,
        recursive: bool,
    ) -> std::result::Result<Vec<String>, FileStoreError> {
        let _ = regex_pattern;
        let mut names = Vec::new();
        let mut pending = vec![String::new()];
        while let Some(relative_dir) = pending.pop() {
            for entry in self
                .list_children(&combine_search_path(directory, &relative_dir))
                .await?
            {
                let child = format!("{relative_dir}/{}", entry.name)
                    .trim_start_matches('/')
                    .to_string();
                if entry.is_directory() {
                    if recursive {
                        pending.push(child);
                    }
                } else if matches_glob(&child, glob_pattern) {
                    names.push(child);
                }
            }
        }
        Ok(names)
    }

    /// Search files under `directory` for lines matching the
    /// case-insensitive `regex_pattern`, optionally filtered by a glob
    /// matched against each file's path relative to `directory`.
    ///
    /// The default asks [`find_matching_files`](Self::find_matching_files)
    /// for candidates, re-applies the glob and recursion rule (so an
    /// over-returning store cannot widen the scope), reads each candidate —
    /// skipping unreadable ones — and scans it with [`scan_content`], so
    /// every `line_number` is aligned with the line editor by construction.
    /// An override owns line numbering. Mirrors upstream `search`.
    async fn search(
        &self,
        directory: &str,
        regex_pattern: &str,
        glob_pattern: Option<&str>,
        recursive: bool,
    ) -> std::result::Result<Vec<FileSearchResult>, FileStoreError> {
        let pattern = compile_search_regex(regex_pattern)?;
        run_search_with_timeout(async {
            let names = self
                .find_matching_files(directory, pattern.pattern(), glob_pattern, recursive)
                .await?;
            let mut results = Vec::new();
            for name in names {
                if !matches_glob(&name, glob_pattern) || (!recursive && name.contains('/')) {
                    continue;
                }
                pattern.check_deadline()?;
                let content = match self.read(&combine_search_path(directory, &name)).await {
                    Ok(Some(content)) => content,
                    Ok(None) => continue,
                    Err(FileStoreError::Invalid(_)) | Err(FileStoreError::Io(_)) => {
                        tracing::warn!(file = %name, "Skipping unreadable file during search");
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                if let Some(result) = scan_content(&name, &content, &pattern)? {
                    results.push(result);
                }
            }
            Ok(results)
        })
        .await
    }
}

/// An in-memory [`AgentFileStore`] — suitable for tests and lightweight use.
///
/// Keys are case-insensitive (normalized + lowercased) while listings and
/// search results report the original casing; directories are implicit in
/// file paths; insertion order is preserved. Mirrors upstream
/// `InMemoryAgentFileStore`.
#[derive(Debug, Default, Clone)]
pub struct InMemoryAgentFileStore {
    /// `(lowercased key, display path, content)` in insertion order.
    files: Arc<Mutex<Vec<(String, String, String)>>>,
}

impl InMemoryAgentFileStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    fn key(path: &str) -> std::result::Result<String, FileStoreError> {
        Ok(normalize_relative_path(path, false)?.to_lowercase())
    }

    fn directory_prefix(directory: &str) -> std::result::Result<(String, usize), FileStoreError> {
        let mut prefix = normalize_relative_path(directory, true)?.to_lowercase();
        if !prefix.is_empty() && !prefix.ends_with('/') {
            prefix.push('/');
        }
        let depth = prefix.matches('/').count();
        Ok((prefix, depth))
    }
}

/// `display` with its first `depth` segments removed (Python
/// `display.split("/", depth)[-1]`).
fn strip_segments(display: &str, depth: usize) -> &str {
    display.splitn(depth + 1, '/').last().unwrap_or(display)
}

#[async_trait]
impl AgentFileStore for InMemoryAgentFileStore {
    async fn write(
        &self,
        path: &str,
        content: &str,
        overwrite: bool,
    ) -> std::result::Result<(), FileStoreError> {
        let display = normalize_relative_path(path, false)?;
        let key = display.to_lowercase();
        let mut files = self.files.lock().unwrap();
        match files.iter_mut().find(|(k, _, _)| *k == key) {
            Some(_) if !overwrite => Err(FileStoreError::AlreadyExists(format!(
                "File already exists: {}",
                crate::paths::py_repr(path)
            ))),
            Some(entry) => {
                entry.1 = display;
                entry.2 = content.to_string();
                Ok(())
            }
            None => {
                files.push((key, display, content.to_string()));
                Ok(())
            }
        }
    }

    async fn read(&self, path: &str) -> std::result::Result<Option<String>, FileStoreError> {
        let key = Self::key(path)?;
        Ok(self
            .files
            .lock()
            .unwrap()
            .iter()
            .find(|(k, _, _)| *k == key)
            .map(|(_, _, c)| c.clone()))
    }

    async fn delete(&self, path: &str) -> std::result::Result<bool, FileStoreError> {
        let key = Self::key(path)?;
        let mut files = self.files.lock().unwrap();
        let before = files.len();
        files.retain(|(k, _, _)| *k != key);
        Ok(files.len() != before)
    }

    async fn list_children(
        &self,
        directory: &str,
    ) -> std::result::Result<Vec<FileStoreEntry>, FileStoreError> {
        let (prefix, depth) = Self::directory_prefix(directory)?;
        let entries: Vec<(String, String)> = self
            .files
            .lock()
            .unwrap()
            .iter()
            .map(|(k, d, _)| (k.clone(), d.clone()))
            .collect();
        let mut files = Vec::new();
        let mut directories: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (key, display) in entries {
            if !key.starts_with(&prefix) {
                continue;
            }
            let remainder = strip_segments(&display, depth);
            match remainder.split_once('/') {
                None => files.push(remainder.to_string()),
                Some((segment, _)) if !segment.is_empty() => {
                    if seen.insert(segment.to_lowercase()) {
                        directories.push(segment.to_string());
                    }
                }
                Some(_) => {}
            }
        }
        let mut out: Vec<FileStoreEntry> = directories
            .into_iter()
            .map(FileStoreEntry::directory)
            .collect();
        out.extend(files.into_iter().map(FileStoreEntry::file));
        Ok(out)
    }

    async fn file_exists(&self, path: &str) -> std::result::Result<bool, FileStoreError> {
        let key = Self::key(path)?;
        Ok(self.files.lock().unwrap().iter().any(|(k, _, _)| *k == key))
    }

    async fn create_directory(&self, _path: &str) -> std::result::Result<(), FileStoreError> {
        Ok(())
    }

    async fn search(
        &self,
        directory: &str,
        regex_pattern: &str,
        glob_pattern: Option<&str>,
        recursive: bool,
    ) -> std::result::Result<Vec<FileSearchResult>, FileStoreError> {
        let (prefix, depth) = Self::directory_prefix(directory)?;
        let pattern = compile_search_regex(regex_pattern)?;
        let entries = self.files.lock().unwrap().clone();
        let glob = glob_pattern.map(str::to_string);
        let scan = tokio::task::spawn_blocking(move || {
            let mut results = Vec::new();
            for (key, display, content) in entries {
                if !key.starts_with(&prefix) {
                    continue;
                }
                let relative_key = &key[prefix.len()..];
                if !recursive && relative_key.contains('/') {
                    continue;
                }
                let relative_display = strip_segments(&display, depth);
                if !matches_glob(relative_display, glob.as_deref()) {
                    continue;
                }
                if let Some(result) = scan_content(relative_display, &content, &pattern)? {
                    results.push(result);
                }
            }
            Ok(results)
        });
        run_search_with_timeout(async move {
            scan.await
                .map_err(|e| FileStoreError::Io(format!("search task failed: {e}")))?
        })
        .await
    }
}

/// Serializes deletes across every [`FileSystemAgentFileStore`] (case
/// aliases may name one file). Mirrors upstream's class-level `_DELETE_LOCK`.
static DELETE_LOCK: Mutex<()> = Mutex::new(());

/// A disk-backed [`AgentFileStore`] rooted under a directory, with traversal
/// and symlink protections (see the [module docs](self)).
///
/// The root is resolved at construction but created lazily on the first
/// write (or [`create_directory`](AgentFileStore::create_directory)), so
/// constructing a store never touches the filesystem. Mirrors upstream
/// `FileSystemAgentFileStore`.
#[derive(Debug, Clone)]
pub struct FileSystemAgentFileStore {
    root_path: PathBuf,
}

/// Resolve `path` like Python's non-strict `Path.resolve()`: absolute, with
/// the longest existing ancestor canonicalized (symlinks resolved) and the
/// remaining components appended.
fn resolve_non_strict(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut existing = absolute.clone();
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(canonical) = std::fs::canonicalize(&existing) {
            let mut out = canonical;
            for component in rest.iter().rev() {
                if component == ".." {
                    out.pop();
                } else if component != "." {
                    out.push(component);
                }
            }
            return out;
        }
        match (
            existing.file_name().map(|n| n.to_os_string()),
            existing.parent(),
        ) {
            (Some(name), Some(parent)) => {
                rest.push(name);
                existing = parent.to_path_buf();
            }
            _ => return absolute,
        }
    }
}

impl FileSystemAgentFileStore {
    /// A store rooted at `root_directory`. Errors when the root is empty or
    /// whitespace-only.
    pub fn new(root_directory: impl AsRef<Path>) -> std::result::Result<Self, FileStoreError> {
        let raw = root_directory.as_ref();
        if raw.as_os_str().to_string_lossy().trim().is_empty() {
            return Err(FileStoreError::Invalid(
                "root_directory must not be empty or whitespace-only.".into(),
            ));
        }
        Ok(Self {
            root_path: resolve_non_strict(raw),
        })
    }

    /// The resolved root directory.
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }

    /// Refuse a root that is (or was swapped for) a link, or that no longer
    /// resolves to itself. A missing root is normal (it is created lazily).
    fn check_root(&self) -> std::result::Result<(), FileStoreError> {
        match std::fs::symlink_metadata(&self.root_path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(_) => Err(FileStoreError::Invalid(
                "Invalid path: unable to verify whether the root directory is a symbolic link or reparse point.".into(),
            )),
            Ok(_) => {
                match is_link_or_reparse_point(&self.root_path) {
                    Ok(false) => {}
                    _ => return Err(FileStoreError::Invalid(SYMLINK_MESSAGE.into())),
                }
                if resolve_non_strict(&self.root_path) != self.root_path {
                    return Err(FileStoreError::Invalid(
                        "Invalid path: the resolved path escapes the root directory.".into(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Reject any existing segment between the root and `candidate` that is
    /// a link; stop at the first missing segment (writes stay allowed); fail
    /// closed when a segment cannot be probed. Mirrors upstream
    /// `_throw_if_contains_symlink`.
    fn throw_if_contains_symlink(
        &self,
        normalized: &str,
    ) -> std::result::Result<(), FileStoreError> {
        let mut current = self.root_path.clone();
        for segment in normalized.split('/') {
            current.push(segment);
            match is_link_or_reparse_point(&current) {
                Ok(true) => return Err(FileStoreError::Invalid(SYMLINK_MESSAGE.into())),
                Ok(false) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
                Err(e) if e.raw_os_error() == Some(libc_enotdir()) => {
                    return Err(FileStoreError::NotADirectory(format!(
                        "Parent path is not a directory: {}",
                        current.display()
                    )))
                }
                Err(_) => {
                    let probed = current
                        .strip_prefix(&self.root_path)
                        .map(|p| p.to_string_lossy().replace('\\', "/"))
                        .unwrap_or_default();
                    return Err(FileStoreError::Invalid(format!(
                        "Invalid path: unable to verify whether {} is a symbolic link or reparse point.",
                        crate::paths::py_repr(&probed)
                    )));
                }
            }
        }
        Ok(())
    }

    /// Resolve a relative *file* path safely under the root. Mirrors
    /// upstream `_resolve_safe_path` (plus the root screen, which upstream
    /// applies only to the root directory itself — applying it to every
    /// operation is strictly safer).
    fn resolve_safe_path(
        &self,
        relative_path: &str,
    ) -> std::result::Result<PathBuf, FileStoreError> {
        let normalized = normalize_relative_path(relative_path, false)?;
        self.resolve_normalized(relative_path, &normalized)
    }

    fn resolve_normalized(
        &self,
        original: &str,
        normalized: &str,
    ) -> std::result::Result<PathBuf, FileStoreError> {
        self.check_root()?;
        self.throw_if_contains_symlink(normalized)?;
        let candidate = self.root_path.join(normalized);
        let resolved = resolve_non_strict(&candidate);
        if !resolved.starts_with(&self.root_path) {
            return Err(FileStoreError::Invalid(format!(
                "Invalid path: {}. The resolved path escapes the root directory.",
                crate::paths::py_repr(original)
            )));
        }
        Ok(resolved)
    }

    /// Resolve a relative *directory* path (blank = the root). Mirrors
    /// upstream `_resolve_safe_directory_path`.
    fn resolve_safe_directory_path(
        &self,
        relative_directory: &str,
    ) -> std::result::Result<PathBuf, FileStoreError> {
        let normalized = normalize_relative_path(relative_directory, true)?;
        if normalized.is_empty() {
            self.check_root()?;
            return Ok(self.root_path.clone());
        }
        self.resolve_normalized(relative_directory, &normalized)
    }

    /// Read a file through a no-follow open; `None` when it is not a file.
    fn read_file_sync(path: &Path) -> std::result::Result<Option<String>, FileStoreError> {
        if !path.is_file() {
            return Ok(None);
        }
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        set_nofollow(&mut options);
        let mut file = options.open(path).map_err(|e| {
            if is_eloop(&e) {
                FileStoreError::Invalid(SYMLINK_MESSAGE.into())
            } else {
                FileStoreError::io(&e)
            }
        })?;
        let mut raw = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut raw).map_err(|e| FileStoreError::io(&e))?;
        String::from_utf8(raw).map(Some).map_err(|_| {
            FileStoreError::Invalid(format!(
                "File '{}' is not UTF-8 text and cannot be read.",
                path.file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default()
            ))
        })
    }

    fn write_file_sync(
        path: &Path,
        content: &str,
        overwrite: bool,
    ) -> std::result::Result<(), FileStoreError> {
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                let blocking = parent
                    .ancestors()
                    .find(|p| p.exists() && !p.is_dir())
                    .map(Path::to_path_buf);
                return Err(match blocking {
                    Some(file) => FileStoreError::NotADirectory(format!(
                        "Parent path is not a directory: {}",
                        file.display()
                    )),
                    None => FileStoreError::io(&e),
                });
            }
        }
        let mut options = std::fs::OpenOptions::new();
        options.write(true);
        if overwrite {
            options.create(true).truncate(true);
        } else {
            options.create_new(true);
        }
        set_nofollow(&mut options);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o644);
        }
        let mut file = match options.open(path) {
            Ok(f) => f,
            Err(e) => {
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::AlreadyExists
                        | std::io::ErrorKind::PermissionDenied
                        | std::io::ErrorKind::IsADirectory
                ) && path.is_dir()
                {
                    return Err(FileStoreError::IsADirectory(format!(
                        "Path is a directory: {}",
                        path.display()
                    )));
                }
                if !overwrite && e.kind() == std::io::ErrorKind::AlreadyExists {
                    return Err(FileStoreError::AlreadyExists(format!(
                        "File already exists: {}",
                        path.display()
                    )));
                }
                if is_eloop(&e) {
                    return Err(FileStoreError::Invalid(SYMLINK_MESSAGE.into()));
                }
                return Err(FileStoreError::io(&e));
            }
        };
        std::io::Write::write_all(&mut file, content.as_bytes()).map_err(|e| FileStoreError::io(&e))
    }

    /// `(relative_name, path)` for every non-link file under `dir`.
    fn enumerate_search_files(dir: &Path, recursive: bool) -> Vec<(String, PathBuf)> {
        let mut found = Vec::new();
        let mut pending = vec![dir.to_path_buf()];
        while let Some(current) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(&current) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                match is_link_or_reparse_point(&path) {
                    Ok(false) => {}
                    _ => continue,
                }
                if path.is_dir() {
                    if recursive {
                        pending.push(path);
                    }
                } else if path.is_file() {
                    if let Ok(relative) = path.strip_prefix(dir) {
                        let name = relative
                            .components()
                            .map(|c| c.as_os_str().to_string_lossy().into_owned())
                            .collect::<Vec<_>>()
                            .join("/");
                        found.push((name, path));
                    }
                }
            }
        }
        found.sort();
        found
    }
}

#[cfg(unix)]
fn set_nofollow(options: &mut std::fs::OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    options.custom_flags(libc::O_NOFOLLOW);
}

#[cfg(not(unix))]
fn set_nofollow(_options: &mut std::fs::OpenOptions) {}

#[cfg(unix)]
fn is_eloop(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(libc::ELOOP)
}

#[cfg(not(unix))]
fn is_eloop(_e: &std::io::Error) -> bool {
    false
}

#[cfg(unix)]
fn libc_enotdir() -> i32 {
    libc::ENOTDIR
}

#[cfg(not(unix))]
fn libc_enotdir() -> i32 {
    267 // ERROR_DIRECTORY
}

async fn blocking<T, F>(f: F) -> std::result::Result<T, FileStoreError>
where
    F: FnOnce() -> std::result::Result<T, FileStoreError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| FileStoreError::Io(format!("file task failed: {e}")))?
}

#[async_trait]
impl AgentFileStore for FileSystemAgentFileStore {
    async fn write(
        &self,
        path: &str,
        content: &str,
        overwrite: bool,
    ) -> std::result::Result<(), FileStoreError> {
        let full = self.resolve_safe_path(path)?;
        let content = content.to_string();
        blocking(move || Self::write_file_sync(&full, &content, overwrite)).await
    }

    async fn read(&self, path: &str) -> std::result::Result<Option<String>, FileStoreError> {
        let full = self.resolve_safe_path(path)?;
        blocking(move || Self::read_file_sync(&full)).await
    }

    async fn delete(&self, path: &str) -> std::result::Result<bool, FileStoreError> {
        let full = self.resolve_safe_path(path)?;
        blocking(move || {
            let _guard = DELETE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            if !full.is_file() {
                return Ok(false);
            }
            match std::fs::remove_file(&full) {
                Ok(()) => Ok(true),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
                Err(e) => Err(FileStoreError::io(&e)),
            }
        })
        .await
    }

    async fn list_children(
        &self,
        directory: &str,
    ) -> std::result::Result<Vec<FileStoreEntry>, FileStoreError> {
        let dir = self.resolve_safe_directory_path(directory)?;
        blocking(move || {
            if !dir.is_dir() {
                return Ok(Vec::new());
            }
            let mut directories = Vec::new();
            let mut files = Vec::new();
            let entries = std::fs::read_dir(&dir).map_err(|e| FileStoreError::io(&e))?;
            for entry in entries.flatten() {
                let path = entry.path();
                match is_link_or_reparse_point(&path) {
                    Ok(false) => {}
                    _ => continue, // links are hidden; un-probeable entries fail closed
                }
                let name = entry.file_name().to_string_lossy().into_owned();
                if path.is_dir() {
                    directories.push(name);
                } else if path.is_file() {
                    files.push(name);
                }
            }
            directories.sort();
            files.sort();
            let mut out: Vec<FileStoreEntry> = directories
                .into_iter()
                .map(FileStoreEntry::directory)
                .collect();
            out.extend(files.into_iter().map(FileStoreEntry::file));
            Ok(out)
        })
        .await
    }

    async fn file_exists(&self, path: &str) -> std::result::Result<bool, FileStoreError> {
        let full = self.resolve_safe_path(path)?;
        Ok(full.is_file())
    }

    async fn create_directory(&self, path: &str) -> std::result::Result<(), FileStoreError> {
        let full = self.resolve_safe_directory_path(path)?;
        blocking(move || std::fs::create_dir_all(&full).map_err(|e| FileStoreError::io(&e))).await
    }

    async fn search(
        &self,
        directory: &str,
        regex_pattern: &str,
        glob_pattern: Option<&str>,
        recursive: bool,
    ) -> std::result::Result<Vec<FileSearchResult>, FileStoreError> {
        let dir = self.resolve_safe_directory_path(directory)?;
        let pattern = compile_search_regex(regex_pattern)?;
        let glob = glob_pattern.map(str::to_string);
        let work = tokio::task::spawn_blocking(move || {
            if !dir.is_dir() {
                return Ok(Vec::new());
            }
            let mut results = Vec::new();
            let mut skipped = 0usize;
            for (name, path) in Self::enumerate_search_files(&dir, recursive) {
                if !matches_glob(&name, glob.as_deref()) {
                    continue;
                }
                pattern.check_deadline()?;
                // Re-checked: a candidate can be swapped for a link after
                // enumeration; the no-follow read closes the leaf race.
                if !matches!(is_link_or_reparse_point(&path), Ok(false)) {
                    tracing::warn!(path = %path.display(), "Skipping symlinked file during search");
                    skipped += 1;
                    continue;
                }
                let content = match Self::read_file_sync(&path) {
                    Ok(Some(c)) => c,
                    Ok(None) => continue,
                    Err(_) => {
                        tracing::warn!(path = %path.display(), "Skipping unreadable file during search");
                        skipped += 1;
                        continue;
                    }
                };
                if let Some(result) = scan_content(&name, &content, &pattern)? {
                    results.push(result);
                }
            }
            if skipped > 0 {
                tracing::info!(
                    dir = %dir.display(),
                    skipped,
                    matched = results.len(),
                    "Search skipped unreadable file(s)"
                );
            }
            Ok(results)
        });
        run_search_with_timeout(async move {
            work.await
                .map_err(|e| FileStoreError::Io(format!("search task failed: {e}")))?
        })
        .await
    }
}

/// The names of the read-only file-access tools.
pub const FILE_ACCESS_READ_ONLY_TOOL_NAMES: [&str; 4] = [
    FileAccessProvider::READ_TOOL_NAME,
    FileAccessProvider::READ_LINES_TOOL_NAME,
    FileAccessProvider::LS_TOOL_NAME,
    FileAccessProvider::GREP_TOOL_NAME,
];

/// The names of the file-access tools that modify the store.
pub const FILE_ACCESS_WRITE_TOOL_NAMES: [&str; 4] = [
    FileAccessProvider::WRITE_TOOL_NAME,
    FileAccessProvider::DELETE_TOOL_NAME,
    FileAccessProvider::REPLACE_TOOL_NAME,
    FileAccessProvider::REPLACE_LINES_TOOL_NAME,
];

/// Context provider giving an agent CRUD/search tools over a *shared*
/// [`AgentFileStore`].
///
/// Mirrors upstream `FileAccessProvider` (experimental upstream). Tools:
/// `file_access_read`, `file_access_read_lines`, `file_access_ls`,
/// `file_access_grep` (read-only), and `file_access_write`,
/// `file_access_delete`, `file_access_replace`, `file_access_replace_lines`
/// (hidden with [`disable_write_tools`](Self::disable_write_tools)).
///
/// **Every tool requires approval by default** (`ApprovalMode::AlwaysRequire`):
/// the model's calls surface as approval requests and run only once
/// approved. Drive that handshake with
/// [`ToolApprovalAgent`](crate::tool_approval::ToolApprovalAgent) (wired by
/// [`HarnessAgent`](crate::agent::HarnessAgent) by default), opt out per
/// group with [`disable_readonly_tool_approval`](Self::disable_readonly_tool_approval)
/// / [`disable_write_tool_approval`](Self::disable_write_tool_approval), or
/// auto-approve via [`read_only_tools_auto_approval_rule`](Self::read_only_tools_auto_approval_rule)
/// / [`all_tools_auto_approval_rule`](Self::all_tools_auto_approval_rule).
///
/// With [`session_scoped`](Self::session_scoped) (or a non-empty
/// [`scope`](Self::scope)), operations are confined to
/// `~access-/<segment>` where the segment is the scope (or session id)
/// mapped injectively by [`storage_path_segment`]; a missing scope fails
/// closed rather than falling back to the shared root.
#[derive(Clone)]
pub struct FileAccessProvider {
    store: Arc<dyn AgentFileStore>,
    source_id: String,
    instructions: String,
    disable_write_tools: bool,
    disable_readonly_tool_approval: bool,
    disable_write_tool_approval: bool,
    session_scoped: bool,
    scope: Option<String>,
    write_lock: Arc<tokio::sync::Mutex<()>>,
}

impl std::fmt::Debug for FileAccessProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileAccessProvider")
            .field("source_id", &self.source_id)
            .field("disable_write_tools", &self.disable_write_tools)
            .field(
                "disable_readonly_tool_approval",
                &self.disable_readonly_tool_approval,
            )
            .field(
                "disable_write_tool_approval",
                &self.disable_write_tool_approval,
            )
            .field("session_scoped", &self.session_scoped)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
struct WriteArgs {
    file_name: String,
    content: String,
    #[serde(default)]
    overwrite: bool,
}

#[derive(Deserialize)]
struct FileNameArgs {
    file_name: String,
}

#[derive(Deserialize)]
struct ReadLinesArgs {
    file_name: String,
    start_line: i64,
    #[serde(default)]
    end_line: Option<i64>,
}

#[derive(Deserialize)]
struct ListArgs {
    #[serde(default)]
    directory: Option<String>,
    #[serde(default)]
    glob_pattern: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct ReplaceArgs {
    pub(crate) file_name: String,
    pub(crate) old_string: String,
    pub(crate) new_string: String,
    #[serde(default)]
    pub(crate) replace_all: bool,
}

#[derive(Deserialize)]
pub(crate) struct ReplaceLinesArgs {
    pub(crate) file_name: String,
    pub(crate) edits: Vec<LineEdit>,
}

#[derive(Deserialize)]
struct GrepArgs {
    regex_pattern: String,
    #[serde(default)]
    glob_pattern: Option<String>,
    #[serde(default)]
    directory: Option<String>,
}

/// The `new_line` / `expected_line` edit item schema shared with file memory.
pub(crate) fn line_edit_schema(grep_tool: &str, mention_read_lines: bool) -> Value {
    let expected = if mention_read_lines {
        format!(
            "Optional: the text you believe is currently on that line, as reported by {grep_tool}. Give the line's own text only: file_access_read_lines prefixes each line with its number and a tab, and that prefix is not part of the line. When supplied, the edit is rejected unless it matches, which catches an out-of-date line number or a file that changed since you looked. The trailing newline is ignored in the comparison."
        )
    } else {
        format!(
            "Optional: the text you believe is currently on that line, as reported by {grep_tool}. When supplied, the edit is rejected unless it matches, which catches an out-of-date line number or a file that changed since you looked. The trailing newline is ignored in the comparison."
        )
    };
    json!({
        "type": "object",
        "properties": {
            "line_number": {"type": "integer", "description": "1-based line number to replace."},
            "new_line": {"type": "string", "description": "Literal replacement text for the line, including any trailing newline you want to keep (the editor does not add one). Set to an empty string to delete the line entirely, including its line break."},
            "expected_line": {"type": "string", "description": expected},
        },
        "required": ["line_number", "new_line"],
    })
}

/// Python's `exc.strerror or exc` rendering of a store error.
pub(crate) fn store_error_text(e: &FileStoreError) -> String {
    e.to_string()
}

fn approval(disabled: bool) -> ApprovalMode {
    if disabled {
        ApprovalMode::NeverRequire
    } else {
        ApprovalMode::AlwaysRequire
    }
}

impl FileAccessProvider {
    /// Name of the tool that writes a file.
    pub const WRITE_TOOL_NAME: &'static str = "file_access_write";
    /// Name of the tool that reads a file.
    pub const READ_TOOL_NAME: &'static str = "file_access_read";
    /// Name of the tool that reads a range of lines.
    pub const READ_LINES_TOOL_NAME: &'static str = "file_access_read_lines";
    /// Name of the tool that deletes a file.
    pub const DELETE_TOOL_NAME: &'static str = "file_access_delete";
    /// Name of the tool that lists a directory.
    pub const LS_TOOL_NAME: &'static str = "file_access_ls";
    /// Name of the tool that searches file contents.
    pub const GREP_TOOL_NAME: &'static str = "file_access_grep";
    /// Name of the tool that replaces a substring.
    pub const REPLACE_TOOL_NAME: &'static str = "file_access_replace";
    /// Name of the tool that replaces whole lines.
    pub const REPLACE_LINES_TOOL_NAME: &'static str = "file_access_replace_lines";

    /// A provider over `store` with upstream's defaults (all tools,
    /// all requiring approval, shared-store semantics).
    pub fn new(store: Arc<dyn AgentFileStore>) -> Self {
        Self {
            store,
            source_id: DEFAULT_FILE_ACCESS_SOURCE_ID.into(),
            instructions: DEFAULT_FILE_ACCESS_INSTRUCTIONS.into(),
            disable_write_tools: false,
            disable_readonly_tool_approval: false,
            disable_write_tool_approval: false,
            session_scoped: false,
            scope: None,
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Override the source id.
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
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

    /// Advertise only the read-only tools.
    pub fn disable_write_tools(mut self, disable: bool) -> Self {
        self.disable_write_tools = disable;
        self
    }

    /// Register the read-only tools with `never_require` approval.
    pub fn disable_readonly_tool_approval(mut self, disable: bool) -> Self {
        self.disable_readonly_tool_approval = disable;
        self
    }

    /// Register the write tools with `never_require` approval.
    pub fn disable_write_tool_approval(mut self, disable: bool) -> Self {
        self.disable_write_tool_approval = disable;
        self
    }

    /// Confine operations to a per-session (or per-scope) working folder.
    pub fn session_scoped(mut self, scoped: bool) -> Self {
        self.session_scoped = scoped;
        self
    }

    /// An explicit isolation namespace (e.g. a user id). A non-empty scope
    /// enables scoped mode by itself.
    pub fn scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// The underlying store.
    pub fn store(&self) -> &Arc<dyn AgentFileStore> {
        &self.store
    }

    /// The names of the tools this provider currently advertises, with
    /// whether each requires approval.
    pub fn tool_approval_modes(&self) -> Vec<(&'static str, ApprovalMode)> {
        let read = approval(self.disable_readonly_tool_approval);
        let write = approval(self.disable_write_tool_approval);
        let mut out: Vec<(&'static str, ApprovalMode)> = FILE_ACCESS_READ_ONLY_TOOL_NAMES
            .iter()
            .map(|n| (*n, read))
            .collect();
        if !self.disable_write_tools {
            out.extend(FILE_ACCESS_WRITE_TOOL_NAMES.iter().map(|n| (*n, write)));
        }
        out
    }

    /// Auto-approval rule approving only the read-only file-access tools.
    ///
    /// **Security:** it matches by tool name only, so any other local tool
    /// registered under one of these names would also be auto-approved —
    /// ensure no other tool collides with them. (Upstream additionally
    /// refuses hosted-tool calls carrying a `server_label`; Rust function
    /// calls never carry one, so every call here is local.)
    pub fn read_only_tools_auto_approval_rule(call: &FunctionCallContent) -> bool {
        FILE_ACCESS_READ_ONLY_TOOL_NAMES.contains(&call.name.as_str())
    }

    /// Auto-approval rule approving every file-access tool, including the
    /// write tools. The same name-collision caveat applies.
    pub fn all_tools_auto_approval_rule(call: &FunctionCallContent) -> bool {
        FILE_ACCESS_READ_ONLY_TOOL_NAMES.contains(&call.name.as_str())
            || FILE_ACCESS_WRITE_TOOL_NAMES.contains(&call.name.as_str())
    }

    fn resolve_session_key(&self, ctx: &SessionContext) -> Result<String> {
        let raw = self
            .scope
            .clone()
            .filter(|s| !s.is_empty())
            .or_else(|| ctx.session_id.clone())
            .unwrap_or_default();
        if raw.is_empty() {
            return Err(agent_framework_core::Error::Configuration(
                "FileAccessProvider session-scoped mode requires a scope: pass an explicit 'scope' or run with a session that has a 'session_id'. Without one, files cannot be isolated from other sessions.".into(),
            ));
        }
        let segment = storage_path_segment(&raw, ENCODED_FILE_ACCESS_SESSION_PREFIX);
        Ok(combine_paths(FILE_ACCESS_WORKSPACE_NAMESPACE, &segment))
    }

    fn tools(&self, session_key: String) -> Vec<ToolDefinition> {
        let readonly = approval(self.disable_readonly_tool_approval);
        let write_mode = approval(self.disable_write_tool_approval);
        let key = Arc::new(session_key);

        let p = self.clone();
        let k = key.clone();
        let read = function_tool(
            Self::READ_TOOL_NAME,
            "Read the content of a file by name. Returns the file content or a message indicating the file could not be read. Line numbers count lines split on \\n only: a lone \\r never starts a new line, each line keeps its own terminator, and content ending in a newline has a final empty line.",
            json!({
                "type": "object",
                "properties": {"file_name": {"type": "string", "description": "Name (relative path) of the file to read."}},
                "required": ["file_name"],
            }),
            readonly,
            move |args| {
                let p = p.clone();
                let k = k.clone();
                async move {
                    let args: FileNameArgs = parse_args(Self::READ_TOOL_NAME, args)?;
                    let name = &args.file_name;
                    let out = match normalize_relative_path(name, false) {
                        Err(e) => format!("Could not read file '{name}': {e}"),
                        Ok(normalized) => match p.store.read(&combine_paths(&k, &normalized)).await {
                            Ok(Some(content)) => content,
                            Ok(None) => format!("File '{name}' not found."),
                            Err(e) => format!("Could not read file '{name}': {}", store_error_text(&e)),
                        },
                    };
                    Ok(Value::String(out))
                }
            },
        );

        let p = self.clone();
        let k = key.clone();
        let read_lines = function_tool(
            Self::READ_LINES_TOOL_NAME,
            "Read part of a file by 1-based inclusive line number; omit end_line to read to the end of the file, and an end_line past the last line is clamped. Each line is prefixed with its number and a tab; everything after that tab is verbatim, including the line's own terminator, so it can be reused as a file_access_replace_lines new_line. Line numbers count lines split on \\n only: a lone \\r never starts a new line, each line keeps its own terminator, and content ending in a newline has a final empty line.",
            json!({
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name (relative path) of the file to read."},
                    "start_line": {"type": "integer", "description": "1-based line number to read from, inclusive."},
                    "end_line": {"type": "integer", "description": "1-based line number to read to, inclusive. Omit to read to the end of the file; a value past the last line is clamped to it."},
                },
                "required": ["file_name", "start_line"],
            }),
            readonly,
            move |args| {
                let p = p.clone();
                let k = k.clone();
                async move {
                    let args: ReadLinesArgs = parse_args(Self::READ_LINES_TOOL_NAME, args)?;
                    let name = &args.file_name;
                    let fail = |e: &FileStoreError| format!("Could not read lines from file '{name}': {e}");
                    let normalized = match normalize_relative_path(name, false) {
                        Ok(n) => n,
                        Err(e) => return Ok(Value::String(fail(&e))),
                    };
                    let content = match p.store.read(&combine_paths(&k, &normalized)).await {
                        Ok(Some(c)) => c,
                        Ok(None) => return Ok(Value::String(format!("File '{name}' not found."))),
                        Err(e) => return Ok(Value::String(fail(&e))),
                    };
                    let out = match slice_lines(&content, args.start_line, args.end_line) {
                        Ok(lines) => lines
                            .iter()
                            .enumerate()
                            .map(|(i, line)| format!("{}\t{line}", args.start_line + i as i64))
                            .collect::<String>(),
                        Err(e) => fail(&e),
                    };
                    Ok(Value::String(out))
                }
            },
        );

        let p = self.clone();
        let k = key.clone();
        let ls = function_tool(
            Self::LS_TOOL_NAME,
            "List the direct child files and subdirectories of a directory. Omit ``directory`` (or pass an empty string) to list the root. To enumerate a subdirectory, pass its relative path, for example ``\"reports\"`` or ``\"reports/2024\"``. Optionally filter entries with a ``glob_pattern`` (e.g. ``\"*.md\"``). Subdirectories are listed before files, and each entry is ``{\"name\": <name>, \"type\": \"file\"|\"directory\"}``.",
            json!({
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Relative directory to list; omit or pass empty to list the root."},
                    "glob_pattern": {"type": "string", "description": "Optional glob (e.g. \"*.md\") matched against entry names to filter the listing."},
                },
            }),
            readonly,
            move |args| {
                let p = p.clone();
                let k = k.clone();
                async move {
                    let args: ListArgs = parse_args(Self::LS_TOOL_NAME, args)?;
                    let raw = args.directory.clone().unwrap_or_default();
                    let target = if raw.trim().is_empty() { String::new() } else { raw.clone() };
                    let store_target = if !k.is_empty() {
                        let normalized = if target.is_empty() {
                            Ok(String::new())
                        } else {
                            normalize_relative_path(&target, true)
                        };
                        match normalized {
                            Ok(n) if n.is_empty() => Ok(k.to_string()),
                            Ok(n) => Ok(combine_paths(&k, &n)),
                            Err(e) => Err(e),
                        }
                    } else {
                        Ok(target)
                    };
                    let listed = match store_target {
                        Ok(t) => p.store.list_children(&t).await,
                        Err(e) => Err(e),
                    };
                    match listed {
                        Err(e) => Ok(Value::String(format!("Could not list directory '{raw}': {e}"))),
                        Ok(entries) => Ok(Value::Array(
                            entries
                                .into_iter()
                                .filter(|e| matches_glob(&e.name, args.glob_pattern.as_deref()))
                                .map(|e| json!({"name": e.name, "type": e.entry_type}))
                                .collect(),
                        )),
                    }
                }
            },
        );

        let p = self.clone();
        let k = key.clone();
        let grep = function_tool(
            Self::GREP_TOOL_NAME,
            "Search the contents of files in the store using a case-insensitive regular expression.\n\nThe search runs recursively across all subdirectories. Optionally restrict the search to a\n``directory`` (relative path), and filter which files to search using a glob ``glob_pattern``\nmatched against each file's path relative to that directory.\nThe glob uses fnmatch semantics where ``*`` matches any characters including ``/``: use\n``\"*.md\"`` to match markdown files at any depth,\nor ``\"reports/*\"`` to restrict the search to the ``reports`` subtree.\nLeave empty or omit to search all files.\nReturns matching results whose file_name values are paths relative to the store root\n(directly usable with file_access_read), along with snippets and matching lines with line numbers.\nStores are expected to report each matching line verbatim, including its own line\nterminator, so it can normally be reused as a file_access_replace_lines new_line; a\ncustom store may not, so prefer file_access_read_lines when the exact text matters.\nLine numbers count lines split on \\n only, with a final empty line when content ends in a\nnewline.\nThe regex_pattern must be 256 characters or fewer.",
            json!({
                "type": "object",
                "properties": {
                    "regex_pattern": {"type": "string", "description": "Case-insensitive regex matched against file contents; 256 characters or fewer."},
                    "glob_pattern": {"type": "string", "description": "Optional glob to filter which files are searched (e.g. \"*.md\", \"reports/*\")."},
                    "directory": {"type": "string", "description": "Optional relative directory to search; omit or pass empty for the root."},
                },
                "required": ["regex_pattern"],
            }),
            readonly,
            move |args| {
                let p = p.clone();
                let k = k.clone();
                async move {
                    let args: GrepArgs = parse_args(Self::GREP_TOOL_NAME, args)?;
                    let glob = args.glob_pattern.filter(|g| !g.trim().is_empty());
                    let target = args.directory.filter(|d| !d.trim().is_empty()).unwrap_or_default();
                    let resolved = if !k.is_empty() {
                        let normalized = if target.is_empty() {
                            Ok(String::new())
                        } else {
                            normalize_relative_path(&target, true)
                        };
                        normalized.map(|n| {
                            let store_target = if n.is_empty() { k.to_string() } else { combine_paths(&k, &n) };
                            (n, store_target)
                        })
                    } else {
                        Ok((target.clone(), target.clone()))
                    };
                    let (normalized_target, store_target) = match resolved {
                        Ok(r) => r,
                        Err(e) => return Ok(Value::String(format!("Could not search files: {e}"))),
                    };
                    let results = match p.store.search(&store_target, &args.regex_pattern, glob.as_deref(), true).await {
                        Ok(r) => r,
                        Err(e) => return Ok(Value::String(format!("Could not search files: {e}"))),
                    };
                    let prefix = normalized_target.trim_matches('/').to_string();
                    let output = results
                        .into_iter()
                        .map(|mut r| {
                            if !prefix.is_empty() {
                                r.file_name = format!("{prefix}/{}", r.file_name);
                            }
                            serde_json::to_value(r).unwrap_or(Value::Null)
                        })
                        .collect();
                    Ok(Value::Array(output))
                }
            },
        );

        let mut tools = vec![read, read_lines, ls, grep];
        if self.disable_write_tools {
            return tools;
        }

        let p = self.clone();
        let k = key.clone();
        tools.push(function_tool(
            Self::WRITE_TOOL_NAME,
            "Write a file with the given name and content. By default, does not overwrite an existing file unless overwrite is set to true.",
            json!({
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name (relative path) of the file to write."},
                    "content": {"type": "string", "description": "Full text content to write to the file."},
                    "overwrite": {"type": "boolean", "default": false, "description": "When true, replace an existing file; otherwise writing fails if it exists."},
                },
                "required": ["file_name", "content"],
            }),
            write_mode,
            move |args| {
                let p = p.clone();
                let k = k.clone();
                async move {
                    let args: WriteArgs = parse_args(Self::WRITE_TOOL_NAME, args)?;
                    let name = &args.file_name;
                    let result = match normalize_relative_path(name, false) {
                        Err(e) => Err(e),
                        Ok(normalized) => {
                            let _guard = p.write_lock.lock().await;
                            p.store
                                .write(&combine_paths(&k, &normalized), &args.content, args.overwrite)
                                .await
                        }
                    };
                    let out = match result {
                        Ok(()) => format!("File '{name}' written."),
                        Err(FileStoreError::NotADirectory(_)) => format!(
                            "Could not write file '{name}': a parent path is already a file. Choose a different path."
                        ),
                        Err(FileStoreError::IsADirectory(_)) => format!(
                            "Could not write file '{name}': this path is already a directory. Choose a different file name."
                        ),
                        Err(FileStoreError::AlreadyExists(_)) => format!(
                            "File '{name}' already exists. To replace it, write again with overwrite set to true."
                        ),
                        Err(e) => format!("Could not write file '{name}': {e}"),
                    };
                    Ok(Value::String(out))
                }
            },
        ));

        let p = self.clone();
        let k = key.clone();
        tools.push(function_tool(
            Self::DELETE_TOOL_NAME,
            "Delete a file by name.",
            json!({
                "type": "object",
                "properties": {"file_name": {"type": "string", "description": "Name (relative path) of the file to delete."}},
                "required": ["file_name"],
            }),
            write_mode,
            move |args| {
                let p = p.clone();
                let k = k.clone();
                async move {
                    let args: FileNameArgs = parse_args(Self::DELETE_TOOL_NAME, args)?;
                    let name = &args.file_name;
                    let result = match normalize_relative_path(name, false) {
                        Err(e) => Err(e),
                        Ok(normalized) => {
                            let _guard = p.write_lock.lock().await;
                            p.store.delete(&combine_paths(&k, &normalized)).await
                        }
                    };
                    Ok(Value::String(match result {
                        Ok(true) => format!("File '{name}' deleted."),
                        Ok(false) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not delete file '{name}': {e}"),
                    }))
                }
            },
        ));

        let p = self.clone();
        let k = key.clone();
        tools.push(function_tool(
            Self::REPLACE_TOOL_NAME,
            "Replace occurrences of old_string with new_string in a file. Fails if old_string is not found, or if it occurs more than once and replace_all is false. Returns the number of occurrences replaced.",
            json!({
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name (relative path) of the file to modify."},
                    "old_string": {"type": "string", "description": "Substring to find and replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "default": false, "description": "When true, replace every occurrence; when false, fail unless exactly one occurrence exists."},
                },
                "required": ["file_name", "old_string", "new_string"],
            }),
            write_mode,
            move |args| {
                let p = p.clone();
                let k = k.clone();
                async move {
                    let args: ReplaceArgs = parse_args(Self::REPLACE_TOOL_NAME, args)?;
                    let name = args.file_name.clone();
                    let result: std::result::Result<Option<usize>, FileStoreError> = async {
                        let normalized = normalize_relative_path(&name, false)?;
                        let path = combine_paths(&k, &normalized);
                        let _guard = p.write_lock.lock().await;
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
        let k = key;
        tools.push(function_tool(
            Self::REPLACE_LINES_TOOL_NAME,
            "Replace lines in a file. Provide a list of edits, each with a 1-based line_number and a literal new_line (include your own trailing newline); an empty new_line deletes the line, including its line break. Fails on out-of-range or duplicate line numbers. Line numbers count lines split on \\n only: a lone \\r never starts a new line, each line keeps its own terminator, and content ending in a newline has a final empty line.",
            json!({
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name (relative path) of the file to modify."},
                    "edits": {"type": "array", "description": "List of 1-based line numbers and their literal replacement text.", "items": line_edit_schema("file_access_grep", true)},
                },
                "required": ["file_name", "edits"],
            }),
            write_mode,
            move |args| {
                let p = p.clone();
                let k = k.clone();
                async move {
                    let args: ReplaceLinesArgs = parse_args(Self::REPLACE_LINES_TOOL_NAME, args)?;
                    let name = args.file_name.clone();
                    let result: std::result::Result<bool, FileStoreError> = async {
                        let normalized = normalize_relative_path(&name, false)?;
                        let path = combine_paths(&k, &normalized);
                        let _guard = p.write_lock.lock().await;
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

#[async_trait]
impl ContextProvider for FileAccessProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let scoped = self.session_scoped || self.scope.as_deref().is_some_and(|s| !s.is_empty());
        let session_key = if scoped {
            let key = self.resolve_session_key(ctx)?;
            tracing::debug!(working_folder = %key, "Session-scoped file access");
            self.store.create_directory(&key).await?;
            key
        } else {
            String::new()
        };
        let mut instructions = self.instructions.clone();
        if !session_key.is_empty() {
            instructions.push_str(SESSION_SCOPED_INSTRUCTIONS_SUFFIX);
        }
        ctx.add_instructions(instructions);
        ctx.tools.extend(self.tools(session_key));
        Ok(())
    }
}
