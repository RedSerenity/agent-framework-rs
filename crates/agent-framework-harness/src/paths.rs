//! Path, line, glob and search primitives shared by the file-backed harness
//! providers.
//!
//! Rust equivalent of the private helpers in upstream's
//! `_harness/_file_access.py` (`_normalize_relative_path`, `_matches_glob`,
//! `_split_lines_keepends`, `_apply_replace_lines`, `_compile_search_regex`,
//! …) and `_filesystem.py` (`_storage_key_segment`,
//! `_is_link_or_reparse_point`). They are `pub` because a custom
//! [`AgentFileStore`](crate::file_access::AgentFileStore) needs the same
//! definitions of a path and of a line to stay aligned with the built-in
//! stores — upstream publishes the line rule for the same reason
//! (`AgentFileStore.split_lines` / `scan_content`).

use std::fmt;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use regex_automata::meta::Regex;
use regex_automata::util::syntax;
use sha2::{Digest, Sha256};

/// Maximum number of characters of context included on either side of the
/// first regex match in a [`FileSearchResult`](crate::file_access::FileSearchResult)
/// snippet. Mirrors upstream `_SEARCH_SNIPPET_RADIUS`.
pub const SEARCH_SNIPPET_RADIUS: usize = 50;

/// Hard cap on the length (in characters) of a model-supplied search regex.
/// Mirrors upstream `_MAX_SEARCH_PATTERN_LENGTH`.
pub const MAX_SEARCH_PATTERN_LENGTH: usize = 256;

/// Wall-clock budget for one search, covering every file read and every line
/// matched. Mirrors upstream `_SEARCH_TIMEOUT_SECONDS`.
pub const SEARCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Longest storage path segment produced by [`storage_path_segment`] before
/// it falls back to a digest. Filesystems cap a name at 255 bytes; the margin
/// leaves room for the suffixes callers append (`.json`, `.tmp.<uuid>`).
const MAX_PATH_SEGMENT_LEN: usize = 160;

/// The error type of [`AgentFileStore`](crate::file_access::AgentFileStore)
/// operations.
///
/// Upstream distinguishes these failures by Python exception class
/// (`ValueError`, `FileExistsError`, `NotADirectoryError`,
/// `IsADirectoryError`, `OSError`), and the file tools turn each into a
/// different message for the model; this enum carries the same distinction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStoreError {
    /// Invalid input or content: a rejected path, a symlink, non-UTF-8 bytes,
    /// an invalid or over-long regex, a search timeout. Upstream `ValueError`.
    Invalid(String),
    /// An exclusive create found an existing file. Upstream `FileExistsError`.
    AlreadyExists(String),
    /// A parent path segment is a file. Upstream `NotADirectoryError`.
    NotADirectory(String),
    /// The target path is a directory. Upstream `IsADirectoryError`.
    IsADirectory(String),
    /// Any other I/O failure; the payload is the OS error text without the
    /// `(os error N)` suffix (upstream reports `exc.strerror`).
    Io(String),
}

impl fmt::Display for FileStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileStoreError::Invalid(m)
            | FileStoreError::AlreadyExists(m)
            | FileStoreError::NotADirectory(m)
            | FileStoreError::IsADirectory(m)
            | FileStoreError::Io(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for FileStoreError {}

impl From<FileStoreError> for agent_framework_core::Error {
    fn from(e: FileStoreError) -> Self {
        agent_framework_core::Error::other(e)
    }
}

impl FileStoreError {
    /// Map an [`io::Error`] to [`FileStoreError::Io`] with upstream's
    /// `strerror`-style text.
    pub fn io(e: &io::Error) -> Self {
        FileStoreError::Io(io_message(e))
    }
}

/// The OS error text of `e` without Rust's ` (os error N)` suffix — the
/// equivalent of Python's `OSError.strerror`.
pub(crate) fn io_message(e: &io::Error) -> String {
    let text = e.to_string();
    match text.rfind(" (os error ") {
        Some(index) if text.ends_with(')') => text[..index].to_string(),
        _ => text,
    }
}

/// Render `value` as exactly one storage *path* segment.
///
/// Delegates to [`agent_framework_core::storage_keys::storage_key_segment`]
/// (literal-safe identifiers pass through; anything else is hex-encoded
/// behind a `~`-prefixed marker, which is injective), then applies the
/// length cap upstream's `_storage_key_segment` applies for filesystem use:
/// an encoding longer than a filesystem name may be is replaced by a
/// SHA-256 digest under a distinct marker. The digest is collision-resistant
/// rather than injective — the same trade-off upstream makes past its cap.
pub fn storage_path_segment(value: &str, encoded_prefix: &str) -> String {
    let marker = encoded_prefix.trim_start_matches('~');
    // `.` and `..` are literal-safe *key* segments but path-traversal *path*
    // segments, so they are always encoded here. The hex form is the same
    // injective encoding the core derivation uses for unsafe values, and no
    // literal-safe value starts with `~`, so this cannot collide.
    if value == "." || value == ".." {
        let hex: String = value.bytes().map(|b| format!("{b:02x}")).collect();
        return format!("~{marker}{hex}");
    }
    let segment = agent_framework_core::storage_keys::storage_key_segment(value, encoded_prefix);
    if segment.len() <= MAX_PATH_SEGMENT_LEN {
        return segment;
    }
    let digest = Sha256::digest(value.as_bytes());
    let mut out = String::from("~");
    out.push_str(encoded_prefix.trim_start_matches('~'));
    // `s` is not a hex digit, so a digest segment can never equal a hex one.
    out.push_str("sha256-");
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Join a working-folder path with a relative path using forward slashes.
/// Mirrors upstream `_combine_paths`.
pub fn combine_paths(base_path: &str, relative_path: &str) -> String {
    if base_path.is_empty() {
        return relative_path.to_string();
    }
    if relative_path.is_empty() {
        return base_path.to_string();
    }
    format!(
        "{}/{}",
        base_path.trim_end_matches('/'),
        relative_path.trim_start_matches('/')
    )
}

/// Join a search `directory` and a path relative to it into one store path.
/// Mirrors upstream `_combine_search_path`.
pub fn combine_search_path(directory: &str, relative: &str) -> String {
    let base = directory.trim_matches('/');
    let tail = relative.trim_matches('/');
    if base.is_empty() {
        return tail.to_string();
    }
    if tail.is_empty() {
        base.to_string()
    } else {
        format!("{base}/{tail}")
    }
}

/// Python's `repr()` of a string, used to quote paths in error messages the
/// way upstream does (`f"Invalid path: {path!r}"`).
pub(crate) fn py_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Normalize and validate a relative store path.
///
/// Trims surrounding whitespace, converts backslashes to forward slashes,
/// collapses repeated separators, and rejects rooted paths, drive letters,
/// and `.`/`..` segments. With `is_directory`, an empty result (the root) and
/// trailing separators are allowed; for a file path both are rejected, so
/// `"foo/"` never silently becomes the file `"foo"`.
///
/// This is for paths supplied as *tool arguments*. It is deliberately lossy
/// (`"a/"` and `"a//"` both become `"a"`), so never derive an isolation
/// namespace from an identifier with it — use [`storage_path_segment`].
/// Mirrors upstream `_normalize_relative_path`, error texts included.
pub fn normalize_relative_path(path: &str, is_directory: bool) -> Result<String, FileStoreError> {
    if path.trim().is_empty() {
        if !is_directory {
            return Err(FileStoreError::Invalid(
                "A file path must not be empty or whitespace-only.".into(),
            ));
        }
        return Ok(String::new());
    }
    let trimmed = path.trim();
    let converted = trimmed.replace('\\', "/");
    if !is_directory && converted.ends_with('/') {
        return Err(FileStoreError::Invalid(format!(
            "Invalid path: {}. A file path must not end with a path separator.",
            py_repr(trimmed)
        )));
    }
    let normalized = converted.trim_matches('/');
    let mut chars = normalized.chars();
    let drive_letter = matches!(
        (chars.next(), chars.next()),
        (Some(first), Some(':')) if first.is_alphabetic()
    );
    if Path::new(trimmed).is_absolute()
        || trimmed.starts_with('/')
        || trimmed.starts_with('\\')
        || drive_letter
    {
        return Err(FileStoreError::Invalid(format!(
            "Invalid path: {}. Paths must be relative and must not start with '/', '\\', or a drive root.",
            py_repr(trimmed)
        )));
    }
    let mut clean: Vec<&str> = Vec::new();
    for segment in normalized.split('/') {
        if segment.is_empty() {
            continue;
        }
        if segment == "." || segment == ".." {
            return Err(FileStoreError::Invalid(format!(
                "Invalid path: {}. Paths must not contain '.' or '..' segments.",
                py_repr(trimmed)
            )));
        }
        clean.push(segment);
    }
    let result = clean.join("/");
    if !is_directory && result.is_empty() {
        return Err(FileStoreError::Invalid(format!(
            "Invalid path: {}. A file path must not be empty.",
            py_repr(trimmed)
        )));
    }
    Ok(result)
}

/// Whether `file_name` matches the optional glob pattern, case-insensitively.
///
/// `None` or a blank pattern matches everything. Uses Python `fnmatch`
/// semantics, so `*` matches any characters **including** `/` (`"*.md"`
/// matches markdown files at any depth), `?` one character, and `[seq]` /
/// `[!seq]` a character class. Mirrors upstream `_matches_glob`
/// (`fnmatch.fnmatchcase` over lowercased operands).
pub fn matches_glob(file_name: &str, glob_pattern: Option<&str>) -> bool {
    match glob_pattern {
        None => true,
        Some(p) if p.trim().is_empty() => true,
        Some(p) => {
            let name: Vec<char> = file_name.to_lowercase().chars().collect();
            let pattern: Vec<char> = p.to_lowercase().chars().collect();
            fnmatch(&name, &pattern)
        }
    }
}

/// Iterative wildcard matcher with single-star backtracking (linear in
/// practice, no recursion depth issues).
fn fnmatch(name: &[char], pattern: &[char]) -> bool {
    let (mut n, mut p) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None;
    while n < name.len() {
        if p < pattern.len() {
            match pattern[p] {
                '*' => {
                    star = Some((p, n));
                    p += 1;
                    continue;
                }
                '?' => {
                    n += 1;
                    p += 1;
                    continue;
                }
                '[' => {
                    if let Some((matched, next_p)) = match_class(&pattern[p..], name[n]) {
                        if matched {
                            n += 1;
                            p += next_p;
                            continue;
                        }
                    } else if name[n] == '[' {
                        // An unterminated class is a literal `[` (fnmatch).
                        n += 1;
                        p += 1;
                        continue;
                    }
                }
                c if c == name[n] => {
                    n += 1;
                    p += 1;
                    continue;
                }
                _ => {}
            }
        }
        match star {
            Some((star_p, star_n)) => {
                p = star_p + 1;
                n = star_n + 1;
                star = Some((star_p, star_n + 1));
            }
            None => return false,
        }
    }
    pattern[p..].iter().all(|c| *c == '*')
}

/// Match `c` against the class starting at `pattern[0] == '['`. Returns
/// `(matched, class_len)`, or `None` when the class is unterminated.
fn match_class(pattern: &[char], c: char) -> Option<(bool, usize)> {
    let mut i = 1;
    let negate = matches!(pattern.get(i), Some('!'));
    if negate {
        i += 1;
    }
    let start = i;
    let mut matched = false;
    while i < pattern.len() {
        if pattern[i] == ']' && i > start {
            return Some((matched != negate, i + 1));
        }
        if i + 2 < pattern.len() && pattern[i + 1] == '-' && pattern[i + 2] != ']' {
            if pattern[i] <= c && c <= pattern[i + 2] {
                matched = true;
            }
            i += 3;
        } else {
            if pattern[i] == c {
                matched = true;
            }
            i += 1;
        }
    }
    None
}

/// Split `content` into lines on `\n` only, keeping each terminator attached.
///
/// This is the published definition of a line for the whole file-access
/// surface: `read_lines`, `replace_lines` and every reported `line_number`
/// are coordinates into this list. A trailing `\r` stays on its line, a
/// trailing `\n` yields a final empty (editable) line, empty content yields
/// one empty line, and concatenating the result reproduces `content`.
/// Mirrors upstream `AgentFileStore.split_lines` / `_split_lines_keepends`.
pub fn split_lines(content: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    for (index, byte) in content.bytes().enumerate() {
        if byte == b'\n' {
            lines.push(&content[start..=index]);
            start = index + 1;
        }
    }
    lines.push(&content[start..]);
    lines
}

/// `line` without its trailing `\r\n` or `\n`. A lone `\r` is content and
/// stays. Mirrors upstream `_strip_line_terminator`.
pub fn strip_line_terminator(line: &str) -> &str {
    line.strip_suffix("\r\n")
        .or_else(|| line.strip_suffix('\n'))
        .unwrap_or(line)
}

/// Replace `old_string` with `new_string` in `content`, returning the new
/// content and the number of replacements. Mirrors upstream `_apply_replace`.
pub fn apply_replace(
    content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<(String, usize), FileStoreError> {
    if old_string.is_empty() {
        return Err(FileStoreError::Invalid(
            "old_string must not be empty.".into(),
        ));
    }
    let count = content.matches(old_string).count();
    if count == 0 {
        return Err(FileStoreError::Invalid(format!(
            "old_string not found: {}.",
            py_repr(old_string)
        )));
    }
    if count > 1 && !replace_all {
        return Err(FileStoreError::Invalid(format!(
            "old_string occurs {count} times; pass replace_all=true to replace all, or provide a more specific old_string."
        )));
    }
    Ok((content.replace(old_string, new_string), count))
}

/// One literal line replacement for the `*_replace_lines` tools.
/// Mirrors upstream `_LineEdit`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct LineEdit {
    /// 1-based line number to replace.
    pub line_number: i64,
    /// Literal replacement text, terminator included; empty deletes the line.
    pub new_line: String,
    /// Optional current text of the line; when supplied the edit is refused
    /// unless it matches (terminators ignored).
    #[serde(default)]
    pub expected_line: Option<String>,
}

/// Apply literal 1-based line replacements to `content`.
///
/// Each `new_line` is written verbatim in place of its target line (the
/// editor never adds a separator); an empty `new_line` deletes the line and
/// its terminator. Rejects an empty edit list, out-of-range or duplicate line
/// numbers, and an `expected_line` that does not match the current line once
/// both terminators are stripped. Mirrors upstream `_apply_replace_lines`.
pub fn apply_replace_lines(content: &str, edits: &[LineEdit]) -> Result<String, FileStoreError> {
    if edits.is_empty() {
        return Err(FileStoreError::Invalid(
            "At least one line edit must be provided.".into(),
        ));
    }
    let mut lines: Vec<String> = split_lines(content)
        .into_iter()
        .map(str::to_string)
        .collect();
    let total = lines.len() as i64;
    let mut seen = std::collections::HashSet::new();
    for edit in edits {
        if !seen.insert(edit.line_number) {
            return Err(FileStoreError::Invalid(format!(
                "Duplicate line_number {} in edits.",
                edit.line_number
            )));
        }
        if edit.line_number < 1 || edit.line_number > total {
            return Err(FileStoreError::Invalid(format!(
                "line_number {} is out of range (file has {total} lines).",
                edit.line_number
            )));
        }
        if let Some(expected) = &edit.expected_line {
            let actual = strip_line_terminator(&lines[(edit.line_number - 1) as usize]);
            if actual != strip_line_terminator(expected) {
                return Err(FileStoreError::Invalid(format!(
                    "line_number {} does not match the expected text. Re-read the file to get current line numbers.",
                    edit.line_number
                )));
            }
        }
    }
    for edit in edits {
        lines[(edit.line_number - 1) as usize] = edit.new_line.clone();
    }
    Ok(lines.concat())
}

/// The 1-based inclusive `[start_line, end_line]` slice of `content`,
/// terminators kept. `end_line` of `None` reads to the end; past the end is
/// clamped. Mirrors upstream `_slice_lines`.
pub fn slice_lines(
    content: &str,
    start_line: i64,
    end_line: Option<i64>,
) -> Result<Vec<&str>, FileStoreError> {
    let lines = split_lines(content);
    let total = lines.len() as i64;
    if start_line < 1 {
        return Err(FileStoreError::Invalid(format!(
            "start_line must be a positive integer, got {start_line}."
        )));
    }
    if let Some(end) = end_line {
        if end < 1 {
            return Err(FileStoreError::Invalid(format!(
                "end_line must be a positive integer, got {end}."
            )));
        }
        if end < start_line {
            return Err(FileStoreError::Invalid(format!(
                "end_line ({end}) must not be less than start_line ({start_line})."
            )));
        }
    }
    if start_line > total {
        return Err(FileStoreError::Invalid(format!(
            "start_line {start_line} is out of range (file has {total} lines)."
        )));
    }
    let end = end_line.map_or(total, |e| e.min(total));
    Ok(lines[(start_line - 1) as usize..end as usize].to_vec())
}

/// A compiled, case-insensitive search pattern carrying one wall-clock
/// deadline shared by every match it performs.
///
/// Rust equivalent of upstream `_BoundedSearchPattern`. The engine is
/// `regex-automata`'s meta regex, which guarantees linear-time matching, so
/// a catastrophically backtracking pattern cannot stall a match the way it
/// can under Python's `re` (upstream's reason for switching to the `regex`
/// package with a mid-match timeout). The shared deadline still bounds the
/// *whole* search — every file read plus every line matched — exactly as
/// upstream's does.
#[derive(Debug, Clone)]
pub struct SearchPattern {
    regex: Regex,
    pattern: String,
    deadline: Instant,
}

impl SearchPattern {
    /// The pattern string this was compiled from.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// The byte span of the first match in `text`, charging elapsed time
    /// against the shared deadline.
    pub fn find(&self, text: &str) -> Result<Option<(usize, usize)>, FileStoreError> {
        self.check_deadline()?;
        Ok(self.regex.find(text).map(|m| (m.start(), m.end())))
    }

    /// Error once the shared deadline has passed.
    pub fn check_deadline(&self) -> Result<(), FileStoreError> {
        if Instant::now() >= self.deadline {
            return Err(FileStoreError::Invalid(search_timeout_message()));
        }
        Ok(())
    }
}

/// Compile a model-supplied search regex: case-insensitive, at most
/// [`MAX_SEARCH_PATTERN_LENGTH`] characters, bounded by [`SEARCH_TIMEOUT`].
/// Mirrors upstream `_compile_search_regex`.
pub fn compile_search_regex(pattern: &str) -> Result<SearchPattern, FileStoreError> {
    compile_search_regex_with_timeout(pattern, SEARCH_TIMEOUT)
}

/// [`compile_search_regex`] with an explicit budget (tests shorten it).
pub fn compile_search_regex_with_timeout(
    pattern: &str,
    timeout: Duration,
) -> Result<SearchPattern, FileStoreError> {
    let length = pattern.chars().count();
    if length > MAX_SEARCH_PATTERN_LENGTH {
        return Err(FileStoreError::Invalid(format!(
            "Regex pattern is too long ({length} characters). Maximum supported length is {MAX_SEARCH_PATTERN_LENGTH} characters."
        )));
    }
    let regex = Regex::builder()
        .syntax(syntax::Config::new().case_insensitive(true))
        .build(pattern)
        .map_err(|e| {
            FileStoreError::Invalid(format!("Invalid regex pattern {}: {e}", py_repr(pattern)))
        })?;
    Ok(SearchPattern {
        regex,
        pattern: pattern.to_string(),
        deadline: Instant::now() + timeout,
    })
}

/// The message for a search that ran out of budget. Mirrors upstream
/// `_search_timeout_message`.
pub fn search_timeout_message() -> String {
    format!(
        "Search did not complete within {} seconds. The bound covers the whole search, so this is either a pathological pattern (avoid nested quantifiers such as '(a+)+') or a store too slow to read this many files in time. Narrow the pattern, or search a smaller directory.",
        SEARCH_TIMEOUT.as_secs()
    )
}

/// Whether `path` is a symbolic link (or, on Windows, a reparse point such
/// as a junction). Errors — including `NotFound` — propagate so callers can
/// fail closed. Mirrors upstream `_is_link_or_reparse_point`.
pub fn is_link_or_reparse_point(path: &Path) -> io::Result<bool> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Ok(true);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Collapse internal whitespace runs to single spaces and trim.
pub(crate) fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_relative_path_collapses_and_validates() {
        assert_eq!(
            normalize_relative_path(" a\\b//c.md ", false).unwrap(),
            "a/b/c.md"
        );
        assert_eq!(normalize_relative_path("", true).unwrap(), "");
        assert_eq!(normalize_relative_path("  ", true).unwrap(), "");
        assert_eq!(normalize_relative_path("dir/", true).unwrap(), "dir");
        for bad in [
            "../x",
            "a/../b",
            "./a",
            "/etc/passwd",
            "\\x",
            "C:/x",
            "c:x",
            "a/.",
        ] {
            assert!(normalize_relative_path(bad, false).is_err(), "{bad}");
            assert!(normalize_relative_path(bad, true).is_err(), "{bad}");
        }
        assert!(normalize_relative_path("", false).is_err());
        let err = normalize_relative_path("foo/", false).unwrap_err();
        assert!(err
            .to_string()
            .contains("must not end with a path separator"));
        let err = normalize_relative_path("../x", false).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid path: '../x'. Paths must not contain '.' or '..' segments."
        );
    }

    #[test]
    fn glob_is_case_insensitive_optional_and_star_crosses_slashes() {
        assert!(matches_glob("x.md", None));
        assert!(matches_glob("x.md", Some("  ")));
        assert!(matches_glob("Notes/Plan.MD", Some("*.md")));
        assert!(matches_glob("reports/2024/a.txt", Some("reports/*")));
        assert!(!matches_glob("a.txt", Some("*.md")));
        assert!(matches_glob("a1.txt", Some("a?.txt")));
        assert!(matches_glob("ab.txt", Some("a[a-c].txt")));
        assert!(!matches_glob("ad.txt", Some("a[a-c].txt")));
        assert!(matches_glob("ad.txt", Some("a[!a-c].txt")));
        assert!(matches_glob("a[.txt", Some("a[.txt")));
    }

    #[test]
    fn split_lines_is_the_published_rule() {
        assert_eq!(split_lines(""), vec![""]);
        assert_eq!(split_lines("a\nb"), vec!["a\n", "b"]);
        assert_eq!(split_lines("a\r\nb\n"), vec!["a\r\n", "b\n", ""]);
        assert_eq!(split_lines("a\rb"), vec!["a\rb"]);
        assert_eq!(split_lines("x\ny\n").concat(), "x\ny\n");
    }

    #[test]
    fn replace_and_replace_lines() {
        assert_eq!(
            apply_replace("a b a", "a", "c", true).unwrap(),
            ("c b c".into(), 2)
        );
        assert!(apply_replace("a b a", "a", "c", false)
            .unwrap_err()
            .to_string()
            .contains("occurs 2 times"));
        assert!(apply_replace("a", "z", "c", false).is_err());
        assert!(apply_replace("a", "", "c", false).is_err());

        let edit = |n, s: &str, e: Option<&str>| LineEdit {
            line_number: n,
            new_line: s.into(),
            expected_line: e.map(str::to_string),
        };
        assert_eq!(
            apply_replace_lines("a\nb\nc", &[edit(2, "B\n", None)]).unwrap(),
            "a\nB\nc"
        );
        assert_eq!(
            apply_replace_lines("a\nb\nc", &[edit(2, "", None)]).unwrap(),
            "a\nc"
        );
        assert!(apply_replace_lines("a", &[]).is_err());
        assert!(apply_replace_lines("a", &[edit(2, "x", None)]).is_err());
        assert!(apply_replace_lines("a\nb", &[edit(1, "x", None), edit(1, "y", None)]).is_err());
        assert!(apply_replace_lines("a\r\nb", &[edit(1, "x\n", Some("a"))]).is_ok());
        assert!(apply_replace_lines("a\nb", &[edit(1, "x\n", Some("b"))])
            .unwrap_err()
            .to_string()
            .contains("does not match the expected text"));
        // A lone `\r` is content, not a terminator.
        assert!(apply_replace_lines("a\r", &[edit(1, "x", Some("a"))]).is_err());
    }

    #[test]
    fn slice_lines_validates_and_clamps() {
        assert_eq!(slice_lines("a\nb\nc", 2, None).unwrap(), vec!["b\n", "c"]);
        assert_eq!(slice_lines("a\nb\nc", 1, Some(99)).unwrap().len(), 3);
        assert!(slice_lines("a", 0, None).is_err());
        assert!(slice_lines("a", 1, Some(0)).is_err());
        assert!(slice_lines("a\nb", 2, Some(1)).is_err());
        assert!(slice_lines("a", 2, None).is_err());
    }

    #[test]
    fn search_regex_rejects_invalid_and_oversize_and_is_case_insensitive() {
        assert!(compile_search_regex("(").is_err());
        let long = "a".repeat(MAX_SEARCH_PATTERN_LENGTH + 1);
        assert!(compile_search_regex(&long)
            .unwrap_err()
            .to_string()
            .contains("too long"));
        let p = compile_search_regex("hello").unwrap();
        assert_eq!(p.find("say HeLLo").unwrap(), Some((4, 9)));
        // A pathological pattern is linear-time here.
        let p = compile_search_regex("(a|a)*$").unwrap();
        let text = format!("{}!", "a".repeat(10_000));
        assert!(p.find(&text).is_ok());
        let expired = compile_search_regex_with_timeout("a", Duration::ZERO).unwrap();
        assert!(expired
            .find("a")
            .unwrap_err()
            .to_string()
            .contains("did not complete"));
    }

    #[test]
    fn storage_path_segment_caps_length_with_a_digest() {
        assert_eq!(storage_path_segment("session-1", "~s-"), "session-1");
        assert_eq!(storage_path_segment("..", "~s-"), "~s-2e2e");
        assert_eq!(storage_path_segment(".", "~s-"), "~s-2e");
        assert_eq!(storage_path_segment("a/b", "~s-"), "~s-612f62");
        let long = "X".repeat(500);
        let seg = storage_path_segment(&long, "~s-");
        assert!(seg.len() <= MAX_PATH_SEGMENT_LEN);
        assert!(seg.starts_with("~s-sha256-"));
        assert_ne!(seg, storage_path_segment(&"X".repeat(501), "~s-"));
    }

    #[test]
    fn io_message_strips_os_error_suffix() {
        let e = io::Error::from_raw_os_error(2);
        assert!(!io_message(&e).contains("os error"));
    }
}
