//! Topic-file durable memory with transcript archiving and LLM extraction.
//!
//! Rust equivalent of upstream `agent_framework._harness._memory`
//! (`MemoryContextProvider`, `MemoryStore`, `MemoryFileStore`,
//! `MemoryIndexEntry`, `MemoryTopicRecord`; experimental upstream).
//!
//! Layout under a [`MemoryFileStore`] (one root per source id and owner):
//!
//! ```text
//! {base}/{source}/{owner}/{kind}/MEMORY.md          one-line pointer per topic
//! {base}/{source}/{owner}/{kind}/topics/<slug>.md   one markdown file per topic
//! {base}/{source}/{owner}/{kind}/transcripts/<session>.jsonl  raw transcripts
//! {base}/{source}/{owner}/{kind}/state.json         consolidation bookkeeping
//! ```
//!
//! # Divergences
//!
//! - **Chat client.** Upstream extracts with `agent.client`; a Rust provider
//!   cannot reach its agent's client, so extraction uses the client given to
//!   [`MemoryContextProvider::client`] (none ⇒ extraction is skipped and
//!   consolidation only de-duplicates, upstream's no-client behavior).
//! - **Errors from the extraction/consolidation client** are all treated as
//!   transient (logged and skipped); upstream lets programmer errors raise.
//! - **Transcript stems** use this crate's [`storage_path_segment`] (hex
//!   encoding) rather than upstream's base32, so archives are not
//!   interchangeable with Python's.
//! - **Session hook.** Archiving needs the session in `after_run`, so this
//!   provider relies on core's `after_run_in_session` hook, which
//!   [`Agent`](agent_framework_core::agent::Agent) calls.
//! - It is a history provider (as upstream's subclass of `HistoryProvider`
//!   is): attach it to the session's `context_providers` so core does not
//!   also auto-attach an in-memory history.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{ApprovalMode, ToolDefinition};
use agent_framework_core::types::{ChatOptions, Content, Message, Role};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::paths::{collapse_whitespace, py_repr, storage_path_segment};
use crate::util::{empty_object_schema, function_tool, parse_args, py_json_dumps, SessionRef};

/// Default source id. Upstream `DEFAULT_MEMORY_SOURCE_ID`.
pub const DEFAULT_MEMORY_SOURCE_ID: &str = "memory";
/// Default context prompt. Upstream `DEFAULT_MEMORY_CONTEXT_PROMPT`.
pub const DEFAULT_MEMORY_CONTEXT_PROMPT: &str =
    "## Memory\nUse MEMORY.md and the loaded topic files when they are relevant.";
/// Upstream `DEFAULT_MEMORY_INDEX_FILE_NAME`.
pub const DEFAULT_MEMORY_INDEX_FILE_NAME: &str = "MEMORY.md";
/// Upstream `DEFAULT_MEMORY_TOPICS_DIRECTORY_NAME`.
pub const DEFAULT_MEMORY_TOPICS_DIRECTORY_NAME: &str = "topics";
/// Upstream `DEFAULT_MEMORY_TRANSCRIPTS_DIRECTORY_NAME`.
pub const DEFAULT_MEMORY_TRANSCRIPTS_DIRECTORY_NAME: &str = "transcripts";
/// Upstream `DEFAULT_MEMORY_STATE_FILE_NAME`.
pub const DEFAULT_MEMORY_STATE_FILE_NAME: &str = "state.json";
/// Upstream `DEFAULT_MEMORY_INDEX_LINE_LIMIT`.
pub const DEFAULT_MEMORY_INDEX_LINE_LIMIT: usize = 200;
/// Upstream `DEFAULT_MEMORY_INDEX_LINE_LENGTH`.
pub const DEFAULT_MEMORY_INDEX_LINE_LENGTH: usize = 150;
/// Upstream `DEFAULT_MEMORY_SELECTION_LIMIT`.
pub const DEFAULT_MEMORY_SELECTION_LIMIT: usize = 3;
/// Upstream `DEFAULT_MEMORY_CONSOLIDATION_MIN_SESSIONS`.
pub const DEFAULT_MEMORY_CONSOLIDATION_MIN_SESSIONS: usize = 5;
/// Upstream `DEFAULT_MEMORY_MAX_EXTRACTIONS`.
pub const DEFAULT_MEMORY_MAX_EXTRACTIONS: usize = 5;
/// Upstream `DEFAULT_MEMORY_CONSOLIDATION_INTERVAL` (24 hours).
pub const DEFAULT_MEMORY_CONSOLIDATION_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
/// Upstream `DEFAULT_MEMORY_INDEX_HEADER`.
pub const DEFAULT_MEMORY_INDEX_HEADER: &str = "# MEMORY";
/// Upstream `DEFAULT_MEMORY_NO_TOPICS_TEXT`.
pub const DEFAULT_MEMORY_NO_TOPICS_TEXT: &str = "- none yet";
/// Upstream `DEFAULT_MEMORY_EXTRACTION_PROMPT`.
pub const DEFAULT_MEMORY_EXTRACTION_PROMPT: &str = r#"You extract durable memory candidates from an agent transcript delta.

Return only JSON with this exact shape:
{"memories":[{"topic":"short topic name","memory":"durable fact"}]}

Rules:
- include only durable facts, preferences, decisions, or patterns worth remembering later
- do not include transient tasks, temporary reminders, one-off outputs, or tool chatter
- keep topic names short and stable
- keep each memory item to one sentence
- return at most 5 memory items
- return {"memories": []} when nothing should be remembered
"#;
/// Upstream `DEFAULT_MEMORY_CONSOLIDATION_PROMPT`.
pub const DEFAULT_MEMORY_CONSOLIDATION_PROMPT: &str = r#"You consolidate one topic memory file into a tighter durable form.

Return only JSON with this exact shape:
{"summary":"short summary","memories":["memory 1","memory 2"]}

Rules:
- preserve durable facts, preferences, and decisions
- remove duplicates and overlaps
- drop stale or obviously transient items
- keep the summary concise
- keep the memory list short and high-signal
"#;

const TRANSCRIPT_SESSION_PREFIX: &str = "~session-";
const DEFAULT_SESSION_FILE_STEM: &str = "default";
const MEMORY_SEGMENT_PREFIX: &str = "~memory-";

/// The instructions the provider injects each run (upstream's list).
const MEMORY_INSTRUCTIONS: [&str; 8] = [
    "Use MEMORY.md as the always-loaded table of contents for durable memory.",
    "Use the loaded topic files when they are relevant, but do not assume every topic file was loaded.",
    "Use loaded recent transcript turns for short-term continuity when they are present, and use topic files for durable memory.",
    "Recent transcript loading can omit grouped tool-call turns, so use transcript search only when exact raw tool chatter is needed.",
    "Use write_memory to save durable facts, decisions, or preferences into a topic file.",
    "Use read_memory_topic to inspect or correct a specific topic file before editing it.",
    "Use search_memory_transcripts only when raw historical detail is necessary, because it searches the raw archive.",
    "Use consolidate_memories only when the user asks to rebuild or clean up memory explicitly.",
];

fn normalize_topic(topic: &str) -> Result<String> {
    let normalized = collapse_whitespace(topic);
    if normalized.is_empty() {
        return Err(Error::tool("topic must not be empty."));
    }
    Ok(normalized)
}

fn normalize_memory_text(memory: &str) -> Result<String> {
    let normalized = collapse_whitespace(memory);
    if normalized.is_empty() {
        return Err(Error::tool("memory must not be empty."));
    }
    Ok(normalized)
}

/// Escape a line that would read back as a heading. Upstream
/// `_escape_markdown_line`.
fn escape_markdown_line(line: &str) -> String {
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') || trimmed.starts_with("\\#") {
        format!("\\{line}")
    } else {
        line.to_string()
    }
}

fn unescape_markdown_line(line: &str) -> String {
    let body = line.trim_start();
    let indent = &line[..line.len() - body.len()];
    match body.strip_prefix('\\') {
        Some(rest) if rest.starts_with('#') => format!("{indent}{rest}"),
        _ => line.to_string(),
    }
}

/// The stable file stem of a topic: lowercase ASCII alphanumerics joined by
/// `-` (`"memory-topic"` when nothing remains). Upstream `_slugify_topic`.
pub fn slugify_topic(topic: &str) -> String {
    let normalized = collapse_whitespace(topic).to_lowercase();
    let mut slug = String::new();
    let mut pending_dash = false;
    for c in normalized.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            if pending_dash && !slug.is_empty() {
                slug.push('-');
            }
            pending_dash = false;
            slug.push(c);
        } else {
            pending_dash = true;
        }
    }
    if slug.is_empty() {
        "memory-topic".into()
    } else {
        slug
    }
}

/// Civil date from days since the Unix epoch (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `secs` since the epoch as upstream's `_timestamp` renders it:
/// `YYYY-MM-DDTHH:MM:SS+00:00`.
pub fn format_timestamp(secs: i64) -> String {
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Parse an ISO-8601 timestamp (`…[.fff][Z|±HH:MM]`, no offset = UTC) to
/// seconds since the epoch.
pub fn parse_timestamp(value: &str) -> Option<i64> {
    let value = value.trim();
    let (date, time) = value.split_once(['T', ' '])?;
    let mut date_parts = date.split('-');
    let y: i64 = date_parts.next()?.parse().ok()?;
    let m: u32 = date_parts.next()?.parse().ok()?;
    let d: u32 = date_parts.next()?.parse().ok()?;
    let (clock, offset_secs) = if let Some(clock) = time.strip_suffix('Z') {
        (clock, 0)
    } else if let Some(index) = time.rfind(['+', '-']) {
        let (clock, offset) = time.split_at(index);
        let sign = if offset.starts_with('-') { -1 } else { 1 };
        let (h, mi) = offset[1..].split_once(':').unwrap_or((&offset[1..], "0"));
        (
            clock,
            sign * (h.parse::<i64>().ok()? * 3600 + mi.parse::<i64>().ok()? * 60),
        )
    } else {
        (time, 0)
    };
    let mut clock_parts = clock.split(':');
    let h: i64 = clock_parts.next()?.parse().ok()?;
    let mi: i64 = clock_parts.next()?.parse().ok()?;
    let s: f64 = clock_parts.next().unwrap_or("0").parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some(days_from_civil(y, m, d) * 86_400 + h * 3600 + mi * 60 + s as i64 - offset_secs)
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn dedupe_strings(values: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    values
        .iter()
        .map(|v| v.trim())
        .filter(|v| !v.is_empty() && seen.insert(v.to_lowercase()))
        .map(str::to_string)
        .collect()
}

fn trim_pointer_line(line: &str, max_length: usize) -> String {
    let count = line.chars().count();
    if count <= max_length {
        return line.to_string();
    }
    if max_length <= 3 {
        return line.chars().take(max_length).collect();
    }
    let head: String = line.chars().take(max_length - 3).collect();
    format!("{}...", head.trim_end())
}

fn coerce_summary(summary: &str, memories: &[String]) -> String {
    let normalized = collapse_whitespace(summary);
    if !normalized.is_empty() {
        return normalized;
    }
    match memories {
        [] => "No summary yet.".into(),
        [one] => one.clone(),
        [a, b, ..] => format!("{a} {b}").trim().to_string(),
    }
}

/// Strip a surrounding Markdown code fence. Upstream `_extract_json_text`.
fn extract_json_text(text: &str) -> String {
    let stripped = text.trim();
    if stripped.starts_with("```") && stripped.ends_with("```") {
        let lines: Vec<&str> = stripped.lines().collect();
        if lines.len() >= 3 {
            return lines[1..lines.len() - 1].join("\n").trim().to_string();
        }
    }
    stripped.to_string()
}

/// Word-like tokens of two or more characters, lowercased (Unicode-aware,
/// upstream's `[^\W_][\w-]+`).
pub fn extract_keywords(texts: &[String]) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for text in texts {
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].is_alphanumeric() {
                let mut j = i + 1;
                while j < chars.len()
                    && (chars[j].is_alphanumeric() || chars[j] == '_' || chars[j] == '-')
                {
                    j += 1;
                }
                if j - i >= 2 {
                    out.insert(chars[i..j].iter().collect::<String>().to_lowercase());
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
    }
    out
}

/// One pointer line of `MEMORY.md`. Mirrors upstream `MemoryIndexEntry`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryIndexEntry {
    /// Human-readable topic.
    pub topic: String,
    /// Stable file stem.
    pub slug: String,
    /// Short summary.
    pub summary: String,
    /// Last update timestamp.
    pub updated_at: String,
}

impl MemoryIndexEntry {
    /// The entry summarizing `record`.
    pub fn from_topic_record(record: &MemoryTopicRecord) -> Self {
        Self {
            topic: record.topic.clone(),
            slug: record.slug.clone(),
            summary: coerce_summary(&record.summary, &record.memories),
            updated_at: record.updated_at.clone(),
        }
    }

    /// `- [topic](topics/slug.md): summary`, trimmed to `max_length`.
    pub fn to_pointer_line(&self, max_length: usize) -> String {
        trim_pointer_line(
            &format!(
                "- [{}](topics/{}.md): {}",
                self.topic, self.slug, self.summary
            ),
            max_length,
        )
    }
}

/// One topic memory file. Mirrors upstream `MemoryTopicRecord`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryTopicRecord {
    /// Human-readable topic.
    pub topic: String,
    /// Stable file stem.
    pub slug: String,
    /// Short summary.
    pub summary: String,
    /// Durable memory bullets.
    pub memories: Vec<String>,
    /// Last update timestamp.
    pub updated_at: String,
    /// Sessions that contributed.
    pub session_ids: Vec<String>,
}

impl MemoryTopicRecord {
    /// A normalized record (topic collapsed, slug derived, memories and
    /// session ids de-duplicated, summary coerced).
    pub fn new(
        topic: &str,
        slug: Option<&str>,
        summary: &str,
        memories: &[String],
        updated_at: impl Into<String>,
        session_ids: &[String],
    ) -> Result<Self> {
        let topic = normalize_topic(topic)?;
        let slug = slugify_topic(slug.unwrap_or(&topic));
        let memories = dedupe_strings(memories);
        let summary = coerce_summary(summary, &memories);
        Ok(Self {
            topic,
            slug,
            summary,
            memories,
            updated_at: updated_at.into(),
            session_ids: dedupe_strings(session_ids),
        })
    }

    /// The canonical on-disk markdown. Mirrors upstream `to_markdown`.
    pub fn to_markdown(&self) -> String {
        let sessions = if self.session_ids.is_empty() {
            "-".to_string()
        } else {
            self.session_ids.join(", ")
        };
        let summary = self
            .summary
            .lines()
            .map(escape_markdown_line)
            .collect::<Vec<_>>()
            .join("\n");
        let memory_lines: Vec<String> = if self.memories.is_empty() {
            vec![format!("- {}", &DEFAULT_MEMORY_NO_TOPICS_TEXT[2..])]
        } else {
            self.memories
                .iter()
                .map(|m| format!("- {}", escape_markdown_line(m)))
                .collect()
        };
        let mut lines = vec![
            format!("# {}", self.topic),
            String::new(),
            format!("Updated: {}", self.updated_at),
            format!("Sessions: {sessions}"),
            String::new(),
            "## Summary".into(),
            summary,
            String::new(),
            "## Memories".into(),
        ];
        lines.extend(memory_lines);
        lines.join("\n").trim_end().to_string()
    }

    /// Parse the canonical markdown. Mirrors upstream `from_markdown`.
    pub fn from_markdown(markdown: &str, fallback_topic: Option<&str>) -> Result<Self> {
        let mut topic = fallback_topic.map(str::to_string);
        let mut updated_at = format_timestamp(now_secs());
        let mut session_ids = Vec::new();
        let mut summary_lines = Vec::new();
        let mut memories = Vec::new();
        let mut section: Option<&str> = None;
        for raw in markdown.lines() {
            let stripped = raw.trim_end().trim();
            if let Some(t) = stripped.strip_prefix("# ") {
                topic = Some(t.trim().to_string());
                section = None;
            } else if let Some(u) = stripped.strip_prefix("Updated: ") {
                updated_at = u.trim().to_string();
            } else if let Some(s) = stripped.strip_prefix("Sessions: ") {
                let s = s.trim();
                session_ids = if s.is_empty() || s == "-" {
                    Vec::new()
                } else {
                    s.split(',')
                        .map(str::trim)
                        .filter(|x| !x.is_empty())
                        .map(str::to_string)
                        .collect()
                };
            } else if stripped == "## Summary" {
                section = Some("summary");
            } else if stripped == "## Memories" {
                section = Some("memories");
            } else if section == Some("summary") {
                if !stripped.is_empty() {
                    summary_lines.push(unescape_markdown_line(stripped));
                }
            } else if section == Some("memories") {
                if let Some(item) = stripped.strip_prefix("- ") {
                    let text = unescape_markdown_line(item.trim());
                    if !text.is_empty() && text != DEFAULT_MEMORY_NO_TOPICS_TEXT[2..] {
                        memories.push(text);
                    }
                }
            }
        }
        let topic = topic.ok_or_else(|| {
            Error::Serialization("Memory topic markdown is missing a '# <topic>' heading.".into())
        })?;
        Self::new(
            &topic,
            None,
            summary_lines.join("\n").trim(),
            &memories,
            updated_at,
            &session_ids,
        )
    }
}

/// Backing store of [`MemoryContextProvider`]. Mirrors upstream
/// `MemoryStore` (methods take a [`SessionRef`] instead of the session).
pub trait MemoryStore: Send + Sync {
    /// The logical owner of `session`, if the store uses one.
    fn owner_id(&self, session: &SessionRef) -> Result<Option<String>> {
        let _ = session;
        Ok(None)
    }
    /// Every topic visible to the owner.
    fn list_topics(&self, session: &SessionRef, source_id: &str) -> Result<Vec<MemoryTopicRecord>>;
    /// One topic by name or slug; `None` when missing.
    fn get_topic(
        &self,
        session: &SessionRef,
        source_id: &str,
        topic: &str,
    ) -> Result<Option<MemoryTopicRecord>>;
    /// Persist one topic.
    fn write_topic(
        &self,
        session: &SessionRef,
        record: &MemoryTopicRecord,
        source_id: &str,
    ) -> Result<()>;
    /// Delete one topic; `false` when missing.
    fn delete_topic(&self, session: &SessionRef, source_id: &str, topic: &str) -> Result<bool>;
    /// Rebuild `MEMORY.md` and return its entries.
    fn rebuild_index(
        &self,
        session: &SessionRef,
        source_id: &str,
        line_limit: usize,
        line_length: usize,
    ) -> Result<Vec<MemoryIndexEntry>>;
    /// The current `MEMORY.md` text.
    fn get_index_text(
        &self,
        session: &SessionRef,
        source_id: &str,
        line_limit: usize,
        line_length: usize,
        entries: Option<&[MemoryIndexEntry]>,
    ) -> Result<String>;
    /// The maintenance state (`last_consolidated_at`,
    /// `sessions_since_consolidation`).
    fn read_state(&self, session: &SessionRef, source_id: &str) -> Result<Map<String, Value>>;
    /// Persist the maintenance state.
    fn write_state(
        &self,
        session: &SessionRef,
        state: &Map<String, Value>,
        source_id: &str,
    ) -> Result<()>;
    /// The transcript archive directory.
    fn transcripts_directory(&self, session: &SessionRef, source_id: &str) -> Result<PathBuf>;
}

fn default_state() -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("last_consolidated_at".into(), Value::Null);
    map.insert("sessions_since_consolidation".into(), json!([]));
    map
}

fn atomic_write_sync(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::other(format!("failed to create {parent:?}: {e}")))?;
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!("{name}.tmp.{}", uuid::Uuid::new_v4().simple()));
    if let Err(e) = std::fs::write(&tmp, text) {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::other(format!("failed to write {tmp:?}: {e}")));
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        Error::other(format!("failed to replace {path:?}: {e}"))
    })
}

/// The transcript file stem for `session_id` (upstream
/// `_transcript_file_stem`).
fn transcript_file_stem(session_id: &str) -> String {
    let raw = if session_id.is_empty() {
        DEFAULT_SESSION_FILE_STEM
    } else {
        session_id
    };
    storage_path_segment(raw, TRANSCRIPT_SESSION_PREFIX)
}

/// The session id a transcript stem encodes, if it can be recovered.
fn decode_transcript_session_id(stem: &str) -> Option<String> {
    if stem == DEFAULT_SESSION_FILE_STEM {
        return None;
    }
    let Some(encoded) = stem.strip_prefix(TRANSCRIPT_SESSION_PREFIX) else {
        return Some(stem.to_string());
    };
    if encoded.len() % 2 != 0 || !encoded.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None; // a digest stem (or an unrelated file)
    }
    let bytes: Option<Vec<u8>> = (0..encoded.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&encoded[i..i + 2], 16).ok())
        .collect();
    String::from_utf8(bytes?).ok()
}

/// File-backed [`MemoryStore`]. Mirrors upstream `MemoryFileStore`.
#[derive(Debug, Clone)]
pub struct MemoryFileStore {
    base_path: PathBuf,
    kind: String,
    owner_prefix: String,
    owner_state_key: String,
    index_file_name: String,
    topics_directory_name: String,
    transcripts_directory_name: String,
    state_file_name: String,
}

impl MemoryFileStore {
    /// A store under `base_path`, reading the owner id from
    /// `session.state[owner_state_key]`.
    pub fn new(base_path: impl Into<PathBuf>, owner_state_key: impl Into<String>) -> Self {
        Self {
            base_path: base_path.into(),
            kind: "memory".into(),
            owner_prefix: String::new(),
            owner_state_key: owner_state_key.into(),
            index_file_name: DEFAULT_MEMORY_INDEX_FILE_NAME.into(),
            topics_directory_name: DEFAULT_MEMORY_TOPICS_DIRECTORY_NAME.into(),
            transcripts_directory_name: DEFAULT_MEMORY_TRANSCRIPTS_DIRECTORY_NAME.into(),
            state_file_name: DEFAULT_MEMORY_STATE_FILE_NAME.into(),
        }
    }

    /// Storage bucket under each owner.
    pub fn kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = kind.into();
        self
    }

    /// Prefix applied to the owner id.
    pub fn owner_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.owner_prefix = prefix.into();
        self
    }

    fn owner(&self, session: &SessionRef) -> Result<String> {
        let owner = match session.state.get(&self.owner_state_key) {
            None | Some(Value::Null) => {
                return Err(Error::Configuration(format!(
                    "MemoryFileStore requires session.state[{}] to be set for file-backed storage.",
                    py_repr(&self.owner_state_key)
                )))
            }
            Some(Value::String(s)) => s,
            Some(other) => other.to_string(),
        };
        let path = Path::new(&owner);
        if path.is_absolute()
            || owner.starts_with('/')
            || path
                .components()
                .any(|c| c == std::path::Component::ParentDir)
        {
            return Err(Error::Configuration(
                "Memory owner ID must not contain path traversal segments.".into(),
            ));
        }
        Ok(owner)
    }

    /// The memory root for `session` and `source_id`.
    pub fn memory_root(&self, session: &SessionRef, source_id: &str) -> Result<PathBuf> {
        let owner = storage_path_segment(
            &format!("{}{}", self.owner_prefix, self.owner(session)?),
            MEMORY_SEGMENT_PREFIX,
        );
        let source = storage_path_segment(source_id, MEMORY_SEGMENT_PREFIX);
        let kind = storage_path_segment(&self.kind, MEMORY_SEGMENT_PREFIX);
        Ok(self.base_path.join(source).join(owner).join(kind))
    }

    fn topic_path(&self, session: &SessionRef, source_id: &str, topic: &str) -> Result<PathBuf> {
        Ok(self
            .memory_root(session, source_id)?
            .join(&self.topics_directory_name)
            .join(format!("{}.md", slugify_topic(topic))))
    }

    fn index_path(&self, session: &SessionRef, source_id: &str) -> Result<PathBuf> {
        Ok(self
            .memory_root(session, source_id)?
            .join(&self.index_file_name))
    }

    fn render_index(entries: &[MemoryIndexEntry], line_limit: usize, line_length: usize) -> String {
        let mut lines = vec![DEFAULT_MEMORY_INDEX_HEADER.to_string(), String::new()];
        let pointers: Vec<String> = entries
            .iter()
            .take(line_limit)
            .map(|e| e.to_pointer_line(line_length))
            .collect();
        if pointers.is_empty() {
            lines.push(DEFAULT_MEMORY_NO_TOPICS_TEXT.into());
        } else {
            lines.extend(pointers);
        }
        format!("{}\n", lines.join("\n").trim_end())
    }

    fn write_index_if_changed(path: &Path, text: &str) -> Result<()> {
        if std::fs::read_to_string(path).ok().as_deref() != Some(text) {
            atomic_write_sync(path, text)?;
        }
        Ok(())
    }
}

impl MemoryStore for MemoryFileStore {
    fn owner_id(&self, session: &SessionRef) -> Result<Option<String>> {
        self.owner(session).map(Some)
    }

    fn list_topics(&self, session: &SessionRef, source_id: &str) -> Result<Vec<MemoryTopicRecord>> {
        let dir = self
            .memory_root(session, source_id)?
            .join(&self.topics_directory_name);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Ok(Vec::new());
        };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "md"))
            .collect();
        paths.sort();
        let mut topics = Vec::new();
        for path in paths {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| Error::other(format!("failed to read {path:?}: {e}")))?;
            let fallback = path
                .file_stem()
                .map(|s| s.to_string_lossy().replace('-', " "));
            topics.push(MemoryTopicRecord::from_markdown(
                &text,
                fallback.as_deref(),
            )?);
        }
        topics.sort_by(|a, b| {
            (a.topic.to_lowercase(), &a.updated_at).cmp(&(b.topic.to_lowercase(), &b.updated_at))
        });
        Ok(topics)
    }

    fn get_topic(
        &self,
        session: &SessionRef,
        source_id: &str,
        topic: &str,
    ) -> Result<Option<MemoryTopicRecord>> {
        let path = self.topic_path(session, source_id, topic)?;
        match std::fs::read_to_string(&path) {
            Ok(text) => MemoryTopicRecord::from_markdown(&text, Some(topic)).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::other(format!("failed to read {path:?}: {e}"))),
        }
    }

    fn write_topic(
        &self,
        session: &SessionRef,
        record: &MemoryTopicRecord,
        source_id: &str,
    ) -> Result<()> {
        let path = self.topic_path(session, source_id, &record.slug)?;
        atomic_write_sync(&path, &format!("{}\n", record.to_markdown()))
    }

    fn delete_topic(&self, session: &SessionRef, source_id: &str, topic: &str) -> Result<bool> {
        let path = self.topic_path(session, source_id, topic)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(Error::other(format!("failed to delete {path:?}: {e}"))),
        }
    }

    fn rebuild_index(
        &self,
        session: &SessionRef,
        source_id: &str,
        line_limit: usize,
        line_length: usize,
    ) -> Result<Vec<MemoryIndexEntry>> {
        let entries: Vec<MemoryIndexEntry> = self
            .list_topics(session, source_id)?
            .iter()
            .map(MemoryIndexEntry::from_topic_record)
            .collect();
        let text = Self::render_index(&entries, line_limit, line_length);
        Self::write_index_if_changed(&self.index_path(session, source_id)?, &text)?;
        Ok(entries.into_iter().take(line_limit).collect())
    }

    fn get_index_text(
        &self,
        session: &SessionRef,
        source_id: &str,
        line_limit: usize,
        line_length: usize,
        entries: Option<&[MemoryIndexEntry]>,
    ) -> Result<String> {
        match entries {
            None => {
                self.rebuild_index(session, source_id, line_limit, line_length)?;
            }
            Some(entries) => {
                let text = Self::render_index(entries, line_limit, line_length);
                Self::write_index_if_changed(&self.index_path(session, source_id)?, &text)?;
            }
        }
        let path = self.index_path(session, source_id)?;
        Ok(std::fs::read_to_string(&path)
            .map_err(|e| Error::other(format!("failed to read {path:?}: {e}")))?
            .trim()
            .to_string())
    }

    fn read_state(&self, session: &SessionRef, source_id: &str) -> Result<Map<String, Value>> {
        let path = self
            .memory_root(session, source_id)?
            .join(&self.state_file_name);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(default_state()),
            Err(e) => return Err(Error::other(format!("failed to read {path:?}: {e}"))),
        };
        let raw: Value =
            serde_json::from_str(&text).map_err(|e| Error::Serialization(e.to_string()))?;
        let Value::Object(raw) = raw else {
            return Err(Error::Serialization(
                "Memory state file must contain a JSON object.".into(),
            ));
        };
        let mut state = default_state();
        state.extend(raw);
        if !state
            .get("sessions_since_consolidation")
            .is_some_and(Value::is_array)
        {
            state.insert("sessions_since_consolidation".into(), json!([]));
        }
        if !matches!(
            state.get("last_consolidated_at"),
            Some(Value::String(_)) | Some(Value::Null)
        ) {
            state.insert("last_consolidated_at".into(), Value::Null);
        }
        Ok(state)
    }

    fn write_state(
        &self,
        session: &SessionRef,
        state: &Map<String, Value>,
        source_id: &str,
    ) -> Result<()> {
        let path = self
            .memory_root(session, source_id)?
            .join(&self.state_file_name);
        atomic_write_sync(&path, &format!("{}\n", py_json_dumps(state, true)))
    }

    fn transcripts_directory(&self, session: &SessionRef, source_id: &str) -> Result<PathBuf> {
        Ok(self
            .memory_root(session, source_id)?
            .join(&self.transcripts_directory_name))
    }
}

/// Rewrites or drops a message before it is archived. Upstream
/// `HistoryMessageFilter`.
pub type HistoryMessageFilter = Arc<dyn Fn(&Message) -> Option<Message> + Send + Sync>;

#[derive(Deserialize)]
struct TopicArgs {
    topic: String,
}

#[derive(Deserialize)]
struct WriteMemoryArgs {
    topic: String,
    memory: String,
}

#[derive(Deserialize)]
struct SearchArgs {
    query: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    20
}

#[derive(Serialize)]
struct TranscriptHit {
    session_id: Option<String>,
    line_number: usize,
    role: String,
    text: String,
}

/// Group spans used for the recent-turn window (upstream compaction's
/// `group_messages` kinds: system, user, tool_call, assistant_text).
fn group_kinds(messages: &[Message]) -> Vec<(&'static str, usize, usize)> {
    let is_tool_call = |m: &Message| {
        m.role == Role::assistant()
            && m.contents
                .iter()
                .any(|c| matches!(c, Content::FunctionCall(_)))
    };
    let is_reasoning_only = |m: &Message| {
        m.role == Role::assistant()
            && !m.contents.is_empty()
            && m.contents
                .iter()
                .all(|c| matches!(c, Content::TextReasoning(_)))
    };
    let mut spans = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let current = &messages[i];
        if current.role == Role::system() {
            spans.push(("system", i, i));
            i += 1;
            continue;
        }
        if current.role == Role::user() {
            spans.push(("user", i, i));
            i += 1;
            continue;
        }
        let mut start = i;
        let mut j = i;
        while j < messages.len() && is_reasoning_only(&messages[j]) {
            j += 1;
        }
        if j > i && !(j < messages.len() && is_tool_call(&messages[j])) {
            j = i; // a reasoning prefix that does not lead into a call
        }
        if is_tool_call(&messages[j]) || messages[j].role == Role::tool() {
            if j == i {
                start = i;
            }
            let mut k = j + 1;
            while k < messages.len()
                && (is_reasoning_only(&messages[k]) || messages[k].role == Role::tool())
            {
                k += 1;
            }
            spans.push(("tool_call", start, k - 1));
            i = k;
            continue;
        }
        spans.push(("assistant_text", i, i));
        i += 1;
    }
    spans
}

/// The last `turn_count` user-started turns of `messages`, optionally
/// without tool-call groups. Upstream `_select_recent_turn_messages`.
pub fn select_recent_turn_messages(
    messages: &[Message],
    turn_count: usize,
    load_tool_turns: bool,
) -> Vec<Message> {
    if turn_count == 0 || messages.is_empty() {
        return Vec::new();
    }
    let spans = group_kinds(messages);
    let user_spans: Vec<usize> = spans
        .iter()
        .enumerate()
        .filter(|(_, s)| s.0 == "user")
        .map(|(i, _)| i)
        .collect();
    let start = if user_spans.is_empty() {
        0
    } else {
        user_spans[user_spans.len().saturating_sub(turn_count)]
    };
    spans[start..]
        .iter()
        .filter(|s| load_tool_turns || s.0 != "tool_call")
        .flat_map(|s| messages[s.1..=s.2].iter().cloned())
        .collect()
}

fn format_messages_for_memory_model(messages: &[Message]) -> String {
    messages
        .iter()
        .filter_map(|m| {
            let text = m.text();
            let text = text.trim();
            (!text.is_empty()).then(|| format!("{}: {text}", m.role.as_str().to_uppercase()))
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Context provider injecting `MEMORY.md` and relevant topic files, with
/// topic-memory tools, transcript archiving, LLM extraction and periodic
/// consolidation. Mirrors upstream `MemoryContextProvider` (see the
/// [module docs](self) for divergences).
///
/// Tools (all `never_require`): `list_memory_topics`, `read_memory_topic`,
/// `write_memory`, `delete_memory_topic`, `search_memory_transcripts`,
/// `consolidate_memories`.
#[derive(Clone)]
pub struct MemoryContextProvider {
    store: Arc<dyn MemoryStore>,
    source_id: String,
    context_prompt: String,
    index_line_limit: usize,
    index_line_length: usize,
    selection_limit: usize,
    recent_turns: usize,
    load_tool_turns: bool,
    max_extractions: usize,
    consolidation_interval: Duration,
    consolidation_min_sessions: usize,
    extraction_prompt: String,
    consolidation_prompt: String,
    client: Option<Arc<dyn ChatClient>>,
    consolidation_client: Option<Arc<dyn ChatClient>>,
    history_message_filter: Option<HistoryMessageFilter>,
    locks: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl std::fmt::Debug for MemoryContextProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryContextProvider")
            .field("source_id", &self.source_id)
            .field("recent_turns", &self.recent_turns)
            .finish_non_exhaustive()
    }
}

impl MemoryContextProvider {
    /// A provider over `store` with upstream's defaults.
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self {
            store,
            source_id: DEFAULT_MEMORY_SOURCE_ID.into(),
            context_prompt: DEFAULT_MEMORY_CONTEXT_PROMPT.into(),
            index_line_limit: DEFAULT_MEMORY_INDEX_LINE_LIMIT,
            index_line_length: DEFAULT_MEMORY_INDEX_LINE_LENGTH,
            selection_limit: DEFAULT_MEMORY_SELECTION_LIMIT,
            recent_turns: 0,
            load_tool_turns: true,
            max_extractions: DEFAULT_MEMORY_MAX_EXTRACTIONS,
            consolidation_interval: DEFAULT_MEMORY_CONSOLIDATION_INTERVAL,
            consolidation_min_sessions: DEFAULT_MEMORY_CONSOLIDATION_MIN_SESSIONS,
            extraction_prompt: DEFAULT_MEMORY_EXTRACTION_PROMPT.into(),
            consolidation_prompt: DEFAULT_MEMORY_CONSOLIDATION_PROMPT.into(),
            client: None,
            consolidation_client: None,
            history_message_filter: None,
            locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Override the source id.
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }
    /// Override the context prompt.
    pub fn context_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.context_prompt = prompt.into();
        self
    }
    /// Maximum pointers in `MEMORY.md` (> 0).
    pub fn index_line_limit(mut self, limit: usize) -> Result<Self> {
        if limit == 0 {
            return Err(Error::Configuration(
                "index_line_limit must be greater than 0.".into(),
            ));
        }
        self.index_line_limit = limit;
        Ok(self)
    }
    /// Maximum pointer-line length (> 0).
    pub fn index_line_length(mut self, length: usize) -> Result<Self> {
        if length == 0 {
            return Err(Error::Configuration(
                "index_line_length must be greater than 0.".into(),
            ));
        }
        self.index_line_length = length;
        Ok(self)
    }
    /// Maximum topic files auto-loaded per turn.
    pub fn selection_limit(mut self, limit: usize) -> Self {
        self.selection_limit = limit;
        self
    }
    /// Recent transcript turns injected alongside durable memory.
    pub fn recent_turns(mut self, turns: usize) -> Self {
        self.recent_turns = turns;
        self
    }
    /// Whether the recent-turn window includes tool-call groups.
    pub fn load_tool_turns(mut self, load: bool) -> Self {
        self.load_tool_turns = load;
        self
    }
    /// Maximum extracted memories per turn.
    pub fn max_extractions(mut self, max: usize) -> Self {
        self.max_extractions = max;
        self
    }
    /// Minimum time between automatic consolidations.
    pub fn consolidation_interval(mut self, interval: Duration) -> Self {
        self.consolidation_interval = interval;
        self
    }
    /// Sessions required before consolidation runs (> 0).
    pub fn consolidation_min_sessions(mut self, sessions: usize) -> Result<Self> {
        if sessions == 0 {
            return Err(Error::Configuration(
                "consolidation_min_sessions must be greater than 0.".into(),
            ));
        }
        self.consolidation_min_sessions = sessions;
        Ok(self)
    }
    /// Override the extraction prompt.
    pub fn extraction_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.extraction_prompt = prompt.into();
        self
    }
    /// Override the consolidation prompt.
    pub fn consolidation_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.consolidation_prompt = prompt.into();
        self
    }
    /// The chat client used for extraction (and consolidation unless
    /// overridden) — upstream uses the agent's own client.
    pub fn client(mut self, client: Arc<dyn ChatClient>) -> Self {
        self.client = Some(client);
        self
    }
    /// A separate (e.g. cheaper) client for consolidation only.
    pub fn consolidation_client(mut self, client: Arc<dyn ChatClient>) -> Self {
        self.consolidation_client = Some(client);
        self
    }
    /// Rewrite or drop messages before they are archived.
    pub fn history_message_filter(mut self, filter: HistoryMessageFilter) -> Self {
        self.history_message_filter = Some(filter);
        self
    }

    fn lock(&self, session: &SessionRef, kind: &str, key: &str) -> Arc<tokio::sync::Mutex<()>> {
        let owner = self
            .store
            .owner_id(session)
            .ok()
            .flatten()
            .unwrap_or_default();
        let lock_key = format!("{}:{owner}:{kind}:{key}", self.source_id);
        self.locks
            .lock()
            .unwrap()
            .entry(lock_key)
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    fn transcript_path(&self, session: &SessionRef, session_id: &str) -> Result<PathBuf> {
        Ok(self
            .store
            .transcripts_directory(session, &self.source_id)?
            .join(format!("{}.jsonl", transcript_file_stem(session_id))))
    }

    /// The archived transcript of `session_id` (empty when none).
    pub fn load_transcript(&self, session: &SessionRef, session_id: &str) -> Result<Vec<Message>> {
        let path = self.transcript_path(session, session_id)?;
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Ok(Vec::new());
        };
        Ok(text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<Message>(l).ok())
            .collect())
    }

    fn append_transcript(&self, session: &SessionRef, messages: &[Message]) -> Result<()> {
        let filtered: Vec<Message> = messages
            .iter()
            .filter_map(|m| match &self.history_message_filter {
                Some(f) => f(m),
                None => Some(m.clone()),
            })
            .collect();
        if filtered.is_empty() {
            return Ok(());
        }
        let path = self.transcript_path(session, &session.session_id)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::other(format!("failed to create {parent:?}: {e}")))?;
        }
        let mut lines = String::new();
        for m in &filtered {
            lines.push_str(
                &serde_json::to_string(m).map_err(|e| Error::Serialization(e.to_string()))?,
            );
            lines.push('\n');
        }
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| Error::other(format!("failed to open {path:?}: {e}")))?;
        file.write_all(lines.as_bytes())
            .map_err(|e| Error::other(format!("failed to append {path:?}: {e}")))
    }

    /// Search the transcript archive for `query` (case-insensitive).
    /// Mirrors upstream `MemoryFileStore.search_transcripts`.
    pub fn search_transcripts(
        &self,
        session: &SessionRef,
        query: &str,
        session_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Value>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(Error::tool("query must not be empty."));
        }
        let needle = query.to_lowercase();
        let dir = self.store.transcripts_directory(session, &self.source_id)?;
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Ok(Vec::new());
        };
        let mut files: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
            .collect();
        files.sort();
        let expected = session_id.map(transcript_file_stem);
        let mut results = Vec::new();
        for file in files {
            let stem = file
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let matched = match (&expected, session_id) {
                (Some(expected), Some(id)) => {
                    if stem != *expected {
                        continue;
                    }
                    Some(id.to_string())
                }
                _ => decode_transcript_session_id(&stem),
            };
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            for (index, line) in text.lines().enumerate() {
                let Ok(message) = serde_json::from_str::<Message>(line.trim()) else {
                    continue;
                };
                let body = message.text();
                let body = body.trim();
                if body.is_empty() || !body.to_lowercase().contains(&needle) {
                    continue;
                }
                results.push(
                    serde_json::to_value(TranscriptHit {
                        session_id: matched.clone(),
                        line_number: index + 1,
                        role: message.role.as_str().to_string(),
                        text: body.to_string(),
                    })
                    .unwrap_or(Value::Null),
                );
                if results.len() >= limit {
                    return Ok(results);
                }
            }
        }
        Ok(results)
    }

    fn select_topics(
        &self,
        entries: &[MemoryIndexEntry],
        input: &[Message],
    ) -> Vec<MemoryIndexEntry> {
        if self.selection_limit == 0 || entries.is_empty() {
            return Vec::new();
        }
        let keywords = extract_keywords(&input.iter().map(Message::text).collect::<Vec<_>>());
        if keywords.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, &MemoryIndexEntry)> = entries
            .iter()
            .map(|e| {
                let topic_keywords = extract_keywords(&[format!("{} {}", e.topic, e.summary)]);
                (topic_keywords.intersection(&keywords).count(), e)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| a.1.topic.to_lowercase().cmp(&b.1.topic.to_lowercase()))
        });
        scored
            .into_iter()
            .take(self.selection_limit)
            .filter(|(score, _)| *score > 0)
            .map(|(_, e)| e.clone())
            .collect()
    }

    /// Append one memory to a topic file (creating it). Upstream
    /// `_merge_memory`.
    async fn merge_memory(
        &self,
        session: &SessionRef,
        topic: &str,
        memory: &str,
        now: i64,
    ) -> Result<MemoryTopicRecord> {
        let topic = normalize_topic(topic)?;
        let memory = normalize_memory_text(memory)?;
        let lock = self.lock(session, "topic", &slugify_topic(&topic));
        let _guard = lock.lock().await;
        let record = match self.store.get_topic(session, &self.source_id, &topic)? {
            None => MemoryTopicRecord::new(
                &topic,
                None,
                &memory,
                std::slice::from_ref(&memory),
                format_timestamp(now),
                std::slice::from_ref(&session.session_id),
            )?,
            Some(existing) => {
                let mut memories = existing.memories.clone();
                memories.push(memory);
                let mut sessions = existing.session_ids.clone();
                sessions.push(session.session_id.clone());
                MemoryTopicRecord::new(
                    &existing.topic,
                    Some(&existing.slug),
                    &existing.summary,
                    &memories,
                    format_timestamp(now),
                    &sessions,
                )?
            }
        };
        self.store.write_topic(session, &record, &self.source_id)?;
        Ok(record)
    }

    fn rebuild(&self, session: &SessionRef) -> Result<Vec<MemoryIndexEntry>> {
        self.store.rebuild_index(
            session,
            &self.source_id,
            self.index_line_limit,
            self.index_line_length,
        )
    }

    async fn extract_memories(
        &self,
        session: &SessionRef,
        input: &[Message],
        response: &[Message],
        now: i64,
    ) -> Result<usize> {
        let Some(client) = &self.client else {
            return Ok(0);
        };
        if self.max_extractions == 0 || response.is_empty() {
            return Ok(0);
        }
        let all: Vec<Message> = input.iter().chain(response).cloned().collect();
        let delta = format_messages_for_memory_model(&all);
        if delta.is_empty() {
            return Ok(0);
        }
        let reply = match client
            .get_response(
                vec![
                    Message::system(self.extraction_prompt.clone()),
                    Message::user(delta),
                ],
                ChatOptions::new(),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "Skipping memory extraction: extractor call failed");
                return Ok(0);
            }
        };
        let text = reply.text();
        let text = text.trim();
        if text.is_empty() {
            return Ok(0);
        }
        let payload: Value = match serde_json::from_str(&extract_json_text(text)) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "Skipping memory extraction: extractor returned invalid JSON");
                return Ok(0);
            }
        };
        let items = match &payload {
            Value::Object(map) => map.get("memories").cloned().unwrap_or(Value::Null),
            other => other.clone(),
        };
        let Value::Array(items) = items else {
            tracing::warn!("Skipping memory extraction: 'memories' is not a list");
            return Ok(0);
        };
        let mut count = 0;
        for item in items.into_iter().take(self.max_extractions) {
            let (Some(topic), Some(memory)) = (
                item.get("topic").and_then(Value::as_str),
                item.get("memory").and_then(Value::as_str),
            ) else {
                tracing::warn!("Skipping memory item: missing 'topic' or 'memory' string");
                continue;
            };
            if self.merge_memory(session, topic, memory, now).await.is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    fn should_consolidate(&self, state: &Map<String, Value>, now: i64) -> bool {
        let sessions = state
            .get("sessions_since_consolidation")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        if sessions < self.consolidation_min_sessions {
            return false;
        }
        match state.get("last_consolidated_at") {
            None | Some(Value::Null) => true,
            Some(Value::String(s)) => match parse_timestamp(s) {
                None => true,
                Some(last) => now - last >= self.consolidation_interval.as_secs() as i64,
            },
            Some(_) => false,
        }
    }

    /// Consolidate one topic; `(record, succeeded)`. Upstream
    /// `_consolidate_topic`.
    async fn consolidate_topic(
        &self,
        record: &MemoryTopicRecord,
        now: i64,
    ) -> Result<(MemoryTopicRecord, bool)> {
        let client = self.consolidation_client.as_ref().or(self.client.as_ref());
        let Some(client) = client else {
            let deduped = MemoryTopicRecord::new(
                &record.topic,
                Some(&record.slug),
                &coerce_summary(&record.summary, &record.memories),
                &record.memories,
                format_timestamp(now),
                &record.session_ids,
            )?;
            return Ok((deduped, true));
        };
        let reply = match client
            .get_response(
                vec![
                    Message::system(self.consolidation_prompt.clone()),
                    Message::user(py_json_dumps(record, false)),
                ],
                ChatOptions::new(),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(topic = %record.topic, error = %e, "Skipping memory consolidation");
                return Ok((record.clone(), false));
            }
        };
        let text = reply.text();
        let parsed: Option<Value> = serde_json::from_str(&extract_json_text(text.trim())).ok();
        let (Some(summary), Some(Value::Array(memories))) = (
            parsed
                .as_ref()
                .and_then(|p| p.get("summary"))
                .and_then(Value::as_str),
            parsed.as_ref().and_then(|p| p.get("memories")),
        ) else {
            tracing::warn!(topic = %record.topic, "Skipping consolidation: invalid payload");
            return Ok((record.clone(), false));
        };
        let memories: Vec<String> = memories
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        let consolidated = MemoryTopicRecord::new(
            &record.topic,
            Some(&record.slug),
            summary,
            &memories,
            format_timestamp(now),
            &record.session_ids,
        )?;
        Ok((consolidated, true))
    }

    /// Run a consolidation pass; returns the number of topics consolidated.
    /// Upstream `_run_consolidation`.
    pub async fn run_consolidation(&self, session: &SessionRef, force: bool) -> Result<usize> {
        let now = now_secs();
        let topics = {
            let lock = self.lock(session, "state", "maintenance");
            let _guard = lock.lock().await;
            let mut state = self.store.read_state(session, &self.source_id)?;
            if !force && !self.should_consolidate(&state, now) {
                return Ok(0);
            }
            let topics = self.store.list_topics(session, &self.source_id)?;
            if topics.is_empty() {
                state.insert("last_consolidated_at".into(), json!(format_timestamp(now)));
                state.insert("sessions_since_consolidation".into(), json!([]));
                self.store.write_state(session, &state, &self.source_id)?;
                return Ok(0);
            }
            topics
        };
        let mut successes = 0;
        for record in topics {
            let lock = self.lock(session, "topic", &record.slug);
            let _guard = lock.lock().await;
            let Some(current) = self
                .store
                .get_topic(session, &self.source_id, &record.slug)?
            else {
                continue;
            };
            let (consolidated, ok) = self.consolidate_topic(&current, now).await?;
            if ok {
                self.store
                    .write_topic(session, &consolidated, &self.source_id)?;
                successes += 1;
            }
        }
        if successes > 0 {
            let lock = self.lock(session, "state", "maintenance");
            let _guard = lock.lock().await;
            let mut state = self.store.read_state(session, &self.source_id)?;
            state.insert("last_consolidated_at".into(), json!(format_timestamp(now)));
            state.insert("sessions_since_consolidation".into(), json!([]));
            self.store.write_state(session, &state, &self.source_id)?;
            self.rebuild(session)?;
        }
        Ok(successes)
    }

    fn tools(&self, session: SessionRef) -> Vec<ToolDefinition> {
        let mut tools = Vec::new();
        let p = self.clone();
        let s = session.clone();
        tools.push(function_tool(
            "list_memory_topics",
            "List the current topic pointers recorded in ``MEMORY.md``.",
            empty_object_schema(),
            ApprovalMode::NeverRequire,
            move |_args| {
                let (p, s) = (p.clone(), s.clone());
                async move {
                    let entries = p.rebuild(&s)?;
                    Ok(Value::String(py_json_dumps(&entries, false)))
                }
            },
        ));
        let p = self.clone();
        let s = session.clone();
        tools.push(function_tool(
            "read_memory_topic",
            "Read one topic memory file by topic name or slug.",
            json!({"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}),
            ApprovalMode::NeverRequire,
            move |args| {
                let (p, s) = (p.clone(), s.clone());
                async move {
                    let args: TopicArgs = parse_args("read_memory_topic", args)?;
                    let topic = normalize_topic(&args.topic)?;
                    match p.store.get_topic(&s, &p.source_id, &topic)? {
                        Some(record) => Ok(Value::String(record.to_markdown())),
                        None => Err(Error::tool(format!(
                            "No memory topic named '{topic}' was found for this owner."
                        ))),
                    }
                }
            },
        ));
        let p = self.clone();
        let s = session.clone();
        tools.push(function_tool(
            "write_memory",
            "Add one durable memory line to a topic file.",
            json!({"type": "object", "properties": {"topic": {"type": "string"}, "memory": {"type": "string"}}, "required": ["topic", "memory"]}),
            ApprovalMode::NeverRequire,
            move |args| {
                let (p, s) = (p.clone(), s.clone());
                async move {
                    let args: WriteMemoryArgs = parse_args("write_memory", args)?;
                    let record = p.merge_memory(&s, &args.topic, &args.memory, now_secs()).await?;
                    p.rebuild(&s)?;
                    Ok(Value::String(py_json_dumps(&record, false)))
                }
            },
        ));
        let p = self.clone();
        let s = session.clone();
        tools.push(function_tool(
            "delete_memory_topic",
            "Delete one topic memory file by topic name or slug.",
            json!({"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}),
            ApprovalMode::NeverRequire,
            move |args| {
                let (p, s) = (p.clone(), s.clone());
                async move {
                    let args: TopicArgs = parse_args("delete_memory_topic", args)?;
                    let topic = normalize_topic(&args.topic)?;
                    let lock = p.lock(&s, "topic", &slugify_topic(&topic));
                    let _guard = lock.lock().await;
                    if !p.store.delete_topic(&s, &p.source_id, &topic)? {
                        return Err(Error::tool(format!(
                            "No memory topic named '{topic}' was found for this owner."
                        )));
                    }
                    p.rebuild(&s)?;
                    Ok(Value::String(format!("Deleted memory topic '{topic}'.")))
                }
            },
        ));
        let p = self.clone();
        let s = session.clone();
        tools.push(function_tool(
            "search_memory_transcripts",
            "Search the raw transcript archive for matching text snippets.",
            json!({
                "type": "object",
                "properties": {"query": {"type": "string"}, "session_id": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                "required": ["query"],
            }),
            ApprovalMode::NeverRequire,
            move |args| {
                let (p, s) = (p.clone(), s.clone());
                async move {
                    let args: SearchArgs = parse_args("search_memory_transcripts", args)?;
                    let hits = p.search_transcripts(&s, &args.query, args.session_id.as_deref(), args.limit)?;
                    Ok(Value::String(py_json_dumps(&hits, false)))
                }
            },
        ));
        let p = self.clone();
        let s = session;
        tools.push(function_tool(
            "consolidate_memories",
            "Force an immediate consolidation pass across all topic files.",
            empty_object_schema(),
            ApprovalMode::NeverRequire,
            move |_args| {
                let (p, s) = (p.clone(), s.clone());
                async move {
                    let count = p.run_consolidation(&s, true).await?;
                    Ok(Value::String(format!(
                        "{{\"consolidated_topics\": {count}}}"
                    )))
                }
            },
        ));
        tools
    }
}

#[async_trait]
impl ContextProvider for MemoryContextProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session = SessionRef::from_context(ctx, "MemoryContextProvider")?;
        let entries = self.rebuild(&session)?;
        let index_text = self.store.get_index_text(
            &session,
            &self.source_id,
            self.index_line_limit,
            self.index_line_length,
            Some(&entries),
        )?;
        let recent = select_recent_turn_messages(
            &self.load_transcript(&session, &session.session_id)?,
            self.recent_turns,
            self.load_tool_turns,
        );
        let mut selected = Vec::new();
        for entry in self.select_topics(&entries, &ctx.input_messages) {
            if let Some(record) = self
                .store
                .get_topic(&session, &self.source_id, &entry.slug)?
            {
                selected.push(record);
            }
        }
        ctx.tools.extend(self.tools(session.clone()));
        ctx.add_instructions(MEMORY_INSTRUCTIONS.join("\n"));
        ctx.messages.extend(recent);
        let blocks: Vec<String> = selected
            .iter()
            .map(|r| format!("### topics/{}.md\n{}", r.slug, r.to_markdown()))
            .collect();
        let loaded = if blocks.is_empty() {
            "- none auto-loaded for this turn".to_string()
        } else {
            blocks.join("\n\n")
        };
        let mut message = Message::user(format!(
            "{}\n\n### MEMORY.md\n{index_text}\n\n### Auto-loaded topic files\n{loaded}",
            self.context_prompt
        ));
        // Surface cross-session origins so observers can tell injected
        // memory from content native to this session.
        let origins: Vec<String> = selected
            .iter()
            .flat_map(|r| r.session_ids.iter())
            .filter(|id| !id.is_empty() && **id != session.session_id)
            .cloned()
            .collect();
        if !origins.is_empty() {
            message
                .additional_properties
                .insert("origin_session_ids".into(), json!(origins));
        }
        ctx.messages.push(message);
        Ok(())
    }

    async fn after_run_in_session(
        &self,
        session: &AgentSession,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_some() {
            return Ok(());
        }
        let session = SessionRef::from_session(session);
        let mut to_store: Vec<Message> = request_messages.to_vec();
        to_store.extend(response_messages.iter().cloned());
        self.append_transcript(&session, &to_store)?;
        let now = now_secs();
        {
            let lock = self.lock(&session, "state", "maintenance");
            let _guard = lock.lock().await;
            let mut state = self.store.read_state(&session, &self.source_id)?;
            let mut ids: Vec<Value> = state
                .get("sessions_since_consolidation")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if !ids
                .iter()
                .any(|v| v.as_str() == Some(session.session_id.as_str()))
            {
                ids.push(json!(session.session_id));
            }
            state.insert("sessions_since_consolidation".into(), Value::Array(ids));
            self.store.write_state(&session, &state, &self.source_id)?;
        }
        if self
            .extract_memories(&session, request_messages, response_messages, now)
            .await?
            > 0
        {
            self.rebuild(&session)?;
        }
        self.run_consolidation(&session, false).await?;
        Ok(())
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl agent_framework_core::history::HistoryProvider for MemoryContextProvider {}
