//! File stores and `FileAccessProvider` — modeled on upstream
//! `test_harness_file_access.py`.

mod common;

use std::sync::Arc;

use agent_framework_core::memory::ContextProvider;
use agent_framework_core::session::SessionState;
use agent_framework_core::tools::ApprovalMode;
use agent_framework_harness::file_access::{
    scan_content, AgentFileStore, FileAccessProvider, FileStoreEntry, FileSystemAgentFileStore,
    InMemoryAgentFileStore, DEFAULT_FILE_ACCESS_INSTRUCTIONS,
};
use agent_framework_harness::file_memory::FileMemoryProvider;
use agent_framework_harness::paths::{compile_search_regex, FileStoreError};
use common::{call, ctx, invoke, invoke_text, TempDir};
use serde_json::{json, Value};

fn fs_store(dir: &TempDir) -> FileSystemAgentFileStore {
    FileSystemAgentFileStore::new(dir.path().join("root")).unwrap()
}

async fn round_trip(store: &dyn AgentFileStore) {
    store
        .write("notes/a.md", "hello\nworld", false)
        .await
        .unwrap();
    assert_eq!(
        store.read("notes/a.md").await.unwrap().as_deref(),
        Some("hello\nworld")
    );
    assert!(store.file_exists("notes/a.md").await.unwrap());
    assert_eq!(store.read("missing.md").await.unwrap(), None);
    assert!(matches!(
        store.write("notes/a.md", "x", false).await,
        Err(FileStoreError::AlreadyExists(_))
    ));
    store.write("notes/a.md", "replaced", true).await.unwrap();
    assert_eq!(
        store.read("notes/a.md").await.unwrap().as_deref(),
        Some("replaced")
    );
    store.write("top.txt", "t", false).await.unwrap();
    let root = store.list_children("").await.unwrap();
    assert_eq!(
        root,
        vec![
            FileStoreEntry::directory("notes"),
            FileStoreEntry::file("top.txt")
        ]
    );
    assert_eq!(
        store.list_children("notes").await.unwrap(),
        vec![FileStoreEntry::file("a.md")]
    );
    assert!(store.delete("notes/a.md").await.unwrap());
    assert!(!store.delete("notes/a.md").await.unwrap());
    assert!(!store.file_exists("notes/a.md").await.unwrap());
}

#[tokio::test]
async fn in_memory_store_round_trips_files() {
    round_trip(&InMemoryAgentFileStore::new()).await;
}

#[tokio::test]
async fn filesystem_store_round_trips_files() {
    let dir = TempDir::new("fs-rt");
    round_trip(&fs_store(&dir)).await;
}

#[tokio::test]
async fn filesystem_store_does_not_create_root_until_write() {
    let dir = TempDir::new("lazy-root");
    let store = fs_store(&dir);
    assert!(!dir.path().join("root").exists());
    assert!(store.list_children("").await.unwrap().is_empty());
    assert_eq!(store.read("x").await.unwrap(), None);
    assert!(!dir.path().join("root").exists());
    store.write("x", "1", true).await.unwrap();
    assert!(dir.path().join("root").join("x").is_file());
    assert!(FileSystemAgentFileStore::new("  ").is_err());
}

async fn rejects_traversal(store: &dyn AgentFileStore) {
    for bad in [
        "../escape.txt",
        "a/../../b",
        "/etc/passwd",
        "\\abs",
        "C:/x",
        "./a",
    ] {
        assert!(
            matches!(
                store.write(bad, "x", true).await,
                Err(FileStoreError::Invalid(_))
            ),
            "{bad}"
        );
        assert!(store.read(bad).await.is_err(), "{bad}");
        assert!(store.delete(bad).await.is_err(), "{bad}");
    }
    assert!(store.list_children("../").await.is_err());
    assert!(store.search("..", "x", None, true).await.is_err());
}

#[tokio::test]
async fn stores_reject_traversal_and_rooted_paths() {
    rejects_traversal(&InMemoryAgentFileStore::new()).await;
    let dir = TempDir::new("traversal");
    rejects_traversal(&fs_store(&dir)).await;
    // Nothing escaped next to the root.
    assert!(!dir.path().join("escape.txt").exists());
}

#[cfg(unix)]
mod symlinks {
    use super::*;
    use std::os::unix::fs::symlink;

    #[tokio::test]
    async fn filesystem_store_rejects_symlinks_out_of_and_into_root() {
        let dir = TempDir::new("symlink");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        let store = fs_store(&dir);
        store.write("inside.txt", "in", true).await.unwrap();
        let root = store.root_path().to_path_buf();

        // A leaf link to a file outside the root.
        symlink(outside.join("secret.txt"), root.join("leak.txt")).unwrap();
        // A leaf link to a file inside the root.
        symlink(root.join("inside.txt"), root.join("alias.txt")).unwrap();
        // An intermediate directory link.
        symlink(&outside, root.join("dirlink")).unwrap();

        for path in ["leak.txt", "alias.txt", "dirlink/secret.txt"] {
            let err = store.read(path).await.unwrap_err();
            assert!(err.to_string().contains("symbolic link"), "{path}: {err}");
            assert!(store.write(path, "pwned", true).await.is_err(), "{path}");
            assert!(store.delete(path).await.is_err(), "{path}");
            assert!(store.file_exists(path).await.is_err(), "{path}");
        }
        assert!(store.list_children("dirlink").await.is_err());
        assert!(store.search("dirlink", "secret", None, true).await.is_err());
        assert_eq!(
            std::fs::read_to_string(outside.join("secret.txt")).unwrap(),
            "secret"
        );

        // Listings and searches skip links rather than following them.
        let names: Vec<String> = store
            .list_children("")
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, vec!["inside.txt"]);
        let hits = store.search("", "secret|in", None, true).await.unwrap();
        assert_eq!(
            hits.iter()
                .map(|h| h.file_name.as_str())
                .collect::<Vec<_>>(),
            vec!["inside.txt"]
        );
    }

    #[tokio::test]
    async fn filesystem_store_rejects_a_root_that_is_a_link_before_first_use() {
        let dir = TempDir::new("root-link");
        let target = dir.path().join("target");
        std::fs::create_dir_all(&target).unwrap();
        let store = FileSystemAgentFileStore::new(dir.path().join("root")).unwrap();
        // Planted after construction, before first use.
        symlink(&target, dir.path().join("root")).unwrap();
        assert!(store.list_children("").await.is_err());
        assert!(store.write("x.txt", "x", true).await.is_err());
        assert!(store.create_directory("").await.is_err());
        assert!(!target.join("x.txt").exists());
    }

    #[tokio::test]
    async fn provider_surfaces_symlink_refusals_as_messages() {
        let dir = TempDir::new("prov-link");
        let outside = dir.path().join("outside.txt");
        std::fs::write(&outside, "secret").unwrap();
        let store = Arc::new(fs_store(&dir));
        store.create_directory("").await.unwrap();
        symlink(&outside, store.root_path().join("leak.txt")).unwrap();
        let provider = FileAccessProvider::new(store);
        let state = SessionState::new();
        let mut c = ctx("s", &state, vec![]);
        provider.before_run(&mut c).await.unwrap();
        let out = invoke_text(
            &c.tools,
            "file_access_read",
            json!({"file_name": "leak.txt"}),
        )
        .await;
        assert!(
            out.starts_with("Could not read file 'leak.txt': Invalid path:"),
            "{out}"
        );
    }
}

#[tokio::test]
async fn filesystem_store_skips_non_utf8_files_in_search_and_errors_on_read() {
    let dir = TempDir::new("utf8");
    let store = fs_store(&dir);
    store.write("good.txt", "needle", true).await.unwrap();
    std::fs::write(store.root_path().join("bad.bin"), [0xff, 0xfe, b'n']).unwrap();
    let err = store.read("bad.bin").await.unwrap_err();
    assert_eq!(
        err.to_string(),
        "File 'bad.bin' is not UTF-8 text and cannot be read."
    );
    let hits = store.search("", "needle|n", None, true).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].file_name, "good.txt");
}

#[tokio::test]
async fn filesystem_store_write_reports_path_collisions() {
    let dir = TempDir::new("collide");
    let store = fs_store(&dir);
    store.write("file", "x", true).await.unwrap();
    assert!(matches!(
        store.write("file/child.txt", "x", true).await,
        Err(FileStoreError::NotADirectory(_))
    ));
    store.write("dir/a.txt", "x", true).await.unwrap();
    assert!(matches!(
        store.write("dir", "x", true).await,
        Err(FileStoreError::IsADirectory(_))
    ));
}

#[tokio::test]
async fn in_memory_store_preserves_original_case_and_is_case_insensitive() {
    let store = InMemoryAgentFileStore::new();
    store.write("Docs/Plan.MD", "Hello", true).await.unwrap();
    assert_eq!(
        store.read("docs/plan.md").await.unwrap().as_deref(),
        Some("Hello")
    );
    assert_eq!(
        store.list_children("DOCS").await.unwrap(),
        vec![FileStoreEntry::file("Plan.MD")]
    );
    let hits = store
        .search("docs", "hello", Some("*.md"), false)
        .await
        .unwrap();
    assert_eq!(hits[0].file_name, "Plan.MD");
    // Unicode names are not truncated.
    store.write("ünï/çødé.txt", "x", true).await.unwrap();
    assert_eq!(
        store.list_children("ünï").await.unwrap(),
        vec![FileStoreEntry::file("çødé.txt")]
    );
}

#[tokio::test]
async fn search_matches_lines_snippets_globs_and_recursion() {
    let store = InMemoryAgentFileStore::new();
    store
        .write("a.md", "first\nsecond ERROR here\nthird", true)
        .await
        .unwrap();
    store.write("b.txt", "error", true).await.unwrap();
    store.write("sub/c.md", "error deep", true).await.unwrap();
    let flat = store
        .search("", "error", Some("*.md"), false)
        .await
        .unwrap();
    assert_eq!(flat.len(), 1);
    assert_eq!(flat[0].file_name, "a.md");
    assert_eq!(flat[0].matching_lines[0].line_number, 2);
    assert_eq!(flat[0].matching_lines[0].line, "second ERROR here\n");
    assert!(flat[0].snippet.contains("ERROR"));
    let deep = store.search("", "error", Some("*.md"), true).await.unwrap();
    let mut names: Vec<_> = deep.iter().map(|r| r.file_name.clone()).collect();
    names.sort();
    assert_eq!(names, vec!["a.md", "sub/c.md"]);
    assert!(store.search("", "(", None, true).await.is_err());
    assert!(store
        .search("", &"a".repeat(257), None, true)
        .await
        .is_err());
}

#[tokio::test]
async fn search_bounds_catastrophic_backtracking() {
    let store = InMemoryAgentFileStore::new();
    store
        .write("x.txt", &format!("{}!", "a".repeat(5000)), true)
        .await
        .unwrap();
    let started = std::time::Instant::now();
    let out = store.search("", "(a|a)*$", None, true).await.unwrap();
    assert_eq!(out.len(), 1);
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}

#[tokio::test]
async fn filesystem_search_reports_crlf_lines_verbatim_and_numbers_align_with_editor() {
    let dir = TempDir::new("crlf");
    let store = Arc::new(fs_store(&dir));
    store
        .write("w.txt", "alpha\r\nbeta\r\ngamma\n", true)
        .await
        .unwrap();
    let hits = store.search("", "beta$", None, true).await.unwrap();
    assert_eq!(hits[0].matching_lines[0].line_number, 2);
    assert_eq!(hits[0].matching_lines[0].line, "beta\r\n");
    // The reported line feeds straight back into replace_lines.
    let provider = FileAccessProvider::new(store.clone()).disable_write_tool_approval(true);
    let state = SessionState::new();
    let mut c = ctx("s", &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    let out = invoke_text(
        &c.tools,
        "file_access_replace_lines",
        json!({"file_name": "w.txt", "edits": [{"line_number": 2, "new_line": "BETA\r\n", "expected_line": "beta"}]}),
    )
    .await;
    assert_eq!(out, "Replaced 1 line(s) in 'w.txt'.");
    assert_eq!(
        store.read("w.txt").await.unwrap().unwrap(),
        "alpha\r\nBETA\r\ngamma\n"
    );
}

#[test]
fn scan_content_numbers_by_split_lines_and_treats_lone_cr_as_content() {
    let pattern = compile_search_regex("x\\r").unwrap();
    let hit = scan_content("f", "ax\rb\nc", &pattern).unwrap().unwrap();
    assert_eq!(hit.matching_lines[0].line_number, 1);
    let pattern = compile_search_regex("^$").unwrap();
    let hit = scan_content("f", "a\n", &pattern).unwrap().unwrap();
    assert_eq!(
        hit.matching_lines[0].line_number, 2,
        "trailing newline yields an empty last line"
    );
}

/// A store implementing only the required methods, exercising the default
/// `find_matching_files` + `search` pipeline.
struct MinimalStore(InMemoryAgentFileStore);

#[async_trait::async_trait]
impl AgentFileStore for MinimalStore {
    async fn write(&self, p: &str, c: &str, o: bool) -> Result<(), FileStoreError> {
        self.0.write(p, c, o).await
    }
    async fn read(&self, p: &str) -> Result<Option<String>, FileStoreError> {
        self.0.read(p).await
    }
    async fn delete(&self, p: &str) -> Result<bool, FileStoreError> {
        self.0.delete(p).await
    }
    async fn list_children(&self, d: &str) -> Result<Vec<FileStoreEntry>, FileStoreError> {
        self.0.list_children(d).await
    }
    async fn file_exists(&self, p: &str) -> Result<bool, FileStoreError> {
        self.0.file_exists(p).await
    }
    async fn create_directory(&self, _p: &str) -> Result<(), FileStoreError> {
        Ok(())
    }
}

#[tokio::test]
async fn base_search_pipeline_walks_and_stays_aligned() {
    let store = MinimalStore(InMemoryAgentFileStore::new());
    store
        .write("r/a.md", "one\ntwo needle", true)
        .await
        .unwrap();
    store.write("r/deep/b.md", "needle", true).await.unwrap();
    store.write("r/c.txt", "needle", true).await.unwrap();
    let hits = store
        .search("r", "NEEDLE", Some("*.md"), true)
        .await
        .unwrap();
    let mut names: Vec<_> = hits
        .iter()
        .map(|h| (h.file_name.clone(), h.matching_lines[0].line_number))
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec![("a.md".to_string(), 2), ("deep/b.md".to_string(), 1)]
    );
    let flat = store.search("r", "needle", None, false).await.unwrap();
    assert_eq!(flat.len(), 2);
}

#[tokio::test]
async fn provider_registers_tools_and_instructions_all_requiring_approval() {
    let provider = FileAccessProvider::new(Arc::new(InMemoryAgentFileStore::new()));
    let state = SessionState::new();
    let mut c = ctx("s", &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    assert_eq!(
        c.instructions.as_deref(),
        Some(DEFAULT_FILE_ACCESS_INSTRUCTIONS)
    );
    let names: Vec<&str> = c.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "file_access_read",
            "file_access_read_lines",
            "file_access_ls",
            "file_access_grep",
            "file_access_write",
            "file_access_delete",
            "file_access_replace",
            "file_access_replace_lines",
        ]
    );
    assert!(c
        .tools
        .iter()
        .all(|t| t.approval_mode == ApprovalMode::AlwaysRequire));
}

#[tokio::test]
async fn provider_approval_opt_outs_and_disable_write_tools() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let state = SessionState::new();
    let provider = FileAccessProvider::new(store.clone()).disable_readonly_tool_approval(true);
    let mut c = ctx("s", &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    for t in &c.tools {
        let expected = if t.name.contains("write")
            || t.name.contains("delete")
            || t.name.contains("replace")
        {
            ApprovalMode::AlwaysRequire
        } else {
            ApprovalMode::NeverRequire
        };
        assert_eq!(t.approval_mode, expected, "{}", t.name);
    }
    let provider = FileAccessProvider::new(store.clone())
        .disable_write_tool_approval(true)
        .disable_write_tools(true);
    let mut c = ctx("s", &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    assert_eq!(c.tools.len(), 4);
    assert!(c
        .tools
        .iter()
        .all(|t| t.approval_mode == ApprovalMode::AlwaysRequire));
}

#[test]
fn auto_approval_rules_scope_by_tool_name() {
    for name in [
        "file_access_read",
        "file_access_read_lines",
        "file_access_ls",
        "file_access_grep",
    ] {
        assert!(FileAccessProvider::read_only_tools_auto_approval_rule(
            &call("c", name, json!({}))
        ));
        assert!(FileAccessProvider::all_tools_auto_approval_rule(&call(
            "c",
            name,
            json!({})
        )));
    }
    for name in [
        "file_access_write",
        "file_access_delete",
        "file_access_replace",
        "file_access_replace_lines",
    ] {
        assert!(!FileAccessProvider::read_only_tools_auto_approval_rule(
            &call("c", name, json!({}))
        ));
        assert!(FileAccessProvider::all_tools_auto_approval_rule(&call(
            "c",
            name,
            json!({})
        )));
    }
    assert!(!FileAccessProvider::all_tools_auto_approval_rule(&call(
        "c",
        "shell",
        json!({})
    )));
}

#[tokio::test]
async fn provider_tools_round_trip_files_with_upstream_messages() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let provider = FileAccessProvider::new(store.clone());
    let state = SessionState::new();
    let mut c = ctx("s", &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    let t = &c.tools;
    assert_eq!(
        invoke_text(
            t,
            "file_access_write",
            json!({"file_name": "a.txt", "content": "x\ny"})
        )
        .await,
        "File 'a.txt' written."
    );
    assert_eq!(
        invoke_text(
            t,
            "file_access_write",
            json!({"file_name": "a.txt", "content": "z"})
        )
        .await,
        "File 'a.txt' already exists. To replace it, write again with overwrite set to true."
    );
    assert_eq!(
        invoke_text(
            t,
            "file_access_write",
            json!({"file_name": "a.txt", "content": "one two one", "overwrite": true})
        )
        .await,
        "File 'a.txt' written."
    );
    assert_eq!(
        invoke_text(t, "file_access_read", json!({"file_name": "a.txt"})).await,
        "one two one"
    );
    assert_eq!(
        invoke_text(t, "file_access_read", json!({"file_name": "nope"})).await,
        "File 'nope' not found."
    );
    assert_eq!(
        invoke_text(t, "file_access_read", json!({"file_name": "../x"})).await,
        "Could not read file '../x': Invalid path: '../x'. Paths must not contain '.' or '..' segments."
    );
    assert_eq!(
        invoke_text(t, "file_access_replace", json!({"file_name": "a.txt", "old_string": "one", "new_string": "1"})).await,
        "Could not replace in file 'a.txt': old_string occurs 2 times; pass replace_all=true to replace all, or provide a more specific old_string."
    );
    assert_eq!(
        invoke_text(t, "file_access_replace", json!({"file_name": "a.txt", "old_string": "one", "new_string": "1", "replace_all": true})).await,
        "Replaced 2 occurrence(s) in 'a.txt'."
    );
    assert_eq!(store.read("a.txt").await.unwrap().unwrap(), "1 two 1");
    assert_eq!(
        invoke_text(t, "file_access_delete", json!({"file_name": "a.txt"})).await,
        "File 'a.txt' deleted."
    );
    assert_eq!(
        invoke_text(t, "file_access_delete", json!({"file_name": "a.txt"})).await,
        "File 'a.txt' not found."
    );
}

#[tokio::test]
async fn read_lines_prefixes_numbers_and_round_trips_into_replace_lines() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    store.write("f.txt", "a\nb\nc\n", true).await.unwrap();
    let provider = FileAccessProvider::new(store.clone());
    let state = SessionState::new();
    let mut c = ctx("s", &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    let t = &c.tools;
    assert_eq!(
        invoke_text(
            t,
            "file_access_read_lines",
            json!({"file_name": "f.txt", "start_line": 2})
        )
        .await,
        "2\tb\n3\tc\n4\t"
    );
    assert_eq!(
        invoke_text(
            t,
            "file_access_read_lines",
            json!({"file_name": "f.txt", "start_line": 2, "end_line": 2})
        )
        .await,
        "2\tb\n"
    );
    assert_eq!(
        invoke_text(
            t,
            "file_access_read_lines",
            json!({"file_name": "f.txt", "start_line": 9})
        )
        .await,
        "Could not read lines from file 'f.txt': start_line 9 is out of range (file has 4 lines)."
    );
    assert_eq!(
        invoke_text(
            t,
            "file_access_replace_lines",
            json!({"file_name": "f.txt", "edits": [{"line_number": 2, "new_line": "B\n", "expected_line": "x"}]})
        )
        .await,
        "Could not edit file 'f.txt': line_number 2 does not match the expected text. Re-read the file to get current line numbers."
    );
    assert_eq!(
        invoke_text(
            t,
            "file_access_replace_lines",
            json!({"file_name": "f.txt", "edits": [{"line_number": 2, "new_line": ""}]})
        )
        .await,
        "Replaced 1 line(s) in 'f.txt'."
    );
    assert_eq!(store.read("f.txt").await.unwrap().unwrap(), "a\nc\n");
}

#[tokio::test]
async fn ls_and_grep_tools() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    store
        .write("reports/2024/q1.md", "revenue up", true)
        .await
        .unwrap();
    store
        .write("reports/notes.txt", "revenue", true)
        .await
        .unwrap();
    let provider = FileAccessProvider::new(store.clone());
    let state = SessionState::new();
    let mut c = ctx("s", &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    let t = &c.tools;
    assert_eq!(
        invoke(t, "file_access_ls", json!({"directory": "reports"}))
            .await
            .unwrap(),
        json!([{"name": "2024", "type": "directory"}, {"name": "notes.txt", "type": "file"}])
    );
    assert_eq!(
        invoke(
            t,
            "file_access_ls",
            json!({"directory": "reports", "glob_pattern": "*.txt"})
        )
        .await
        .unwrap(),
        json!([{"name": "notes.txt", "type": "file"}])
    );
    let hits = invoke(
        t,
        "file_access_grep",
        json!({"regex_pattern": "REVENUE", "directory": "reports", "glob_pattern": "*.md"}),
    )
    .await
    .unwrap();
    // Re-rooted to the store root so it composes with file_access_read.
    assert_eq!(hits[0]["file_name"], "reports/2024/q1.md");
    assert_eq!(hits[0]["matching_lines"][0]["line_number"], 1);
    let err = invoke_text(t, "file_access_grep", json!({"regex_pattern": "("})).await;
    assert!(err.starts_with("Could not search files: "), "{err}");
}

#[tokio::test]
async fn session_scoped_provider_isolates_sessions_and_colliding_ids() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let provider = FileAccessProvider::new(store.clone()).session_scoped(true);
    let mut tools = Vec::new();
    for id in ["alice", "a/../alice", "ALICE"] {
        let state = SessionState::new();
        let mut c = ctx(id, &state, vec![]);
        provider.before_run(&mut c).await.unwrap();
        assert!(c
            .instructions
            .as_deref()
            .unwrap()
            .contains("Your file workspace is isolated"));
        tools.push(c.tools);
    }
    assert_eq!(
        invoke_text(
            &tools[0],
            "file_access_write",
            json!({"file_name": "f.txt", "content": "A"})
        )
        .await,
        "File 'f.txt' written."
    );
    for other in &tools[1..] {
        assert_eq!(
            invoke_text(other, "file_access_read", json!({"file_name": "f.txt"})).await,
            "File 'f.txt' not found."
        );
    }
    // Stored under the fixed namespace directory.
    assert!(store.file_exists("~access-/alice/f.txt").await.unwrap());
    // Parent traversal out of the workspace is rejected.
    let out = invoke(&tools[0], "file_access_ls", json!({"directory": ".."}))
        .await
        .unwrap();
    assert!(out
        .as_str()
        .unwrap()
        .starts_with("Could not list directory '..':"));
    let out = invoke_text(
        &tools[0],
        "file_access_grep",
        json!({"regex_pattern": "A", "directory": "../x"}),
    )
    .await;
    assert!(out.starts_with("Could not search files:"));
    // Grep names are session-relative.
    let hits = invoke(&tools[0], "file_access_grep", json!({"regex_pattern": "A"}))
        .await
        .unwrap();
    assert_eq!(hits[0]["file_name"], "f.txt");
}

#[tokio::test]
async fn session_scoped_fails_closed_without_scope_and_scope_alone_enables_it() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let provider = FileAccessProvider::new(store.clone()).session_scoped(true);
    let mut c = agent_framework_core::memory::SessionContext::new(vec![]);
    assert!(provider.before_run(&mut c).await.is_err());
    let provider = FileAccessProvider::new(store.clone()).scope("tenant-1");
    for id in ["s1", "s2"] {
        let state = SessionState::new();
        let mut c = ctx(id, &state, vec![]);
        provider.before_run(&mut c).await.unwrap();
        invoke_text(
            &c.tools,
            "file_access_write",
            json!({"file_name": format!("{id}.txt"), "content": "x"}),
        )
        .await;
    }
    assert!(store.file_exists("~access-/tenant-1/s1.txt").await.unwrap());
    assert!(store.file_exists("~access-/tenant-1/s2.txt").await.unwrap());
    // Unscoped keeps shared-store semantics.
    let shared = FileAccessProvider::new(store.clone());
    let state = SessionState::new();
    let mut c = ctx("s3", &state, vec![]);
    shared.before_run(&mut c).await.unwrap();
    assert_eq!(
        c.instructions.as_deref(),
        Some(DEFAULT_FILE_ACCESS_INSTRUCTIONS)
    );
}

#[tokio::test]
async fn session_scoped_namespace_does_not_collide_with_file_memory() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let access = FileAccessProvider::new(store.clone()).session_scoped(true);
    let memory = FileMemoryProvider::new(store.clone());
    let state = SessionState::new();
    let mut a = ctx("session-1", &state, vec![]);
    access.before_run(&mut a).await.unwrap();
    let mut m = ctx("session-1", &state, vec![]);
    memory.before_run(&mut m).await.unwrap();
    invoke_text(
        &a.tools,
        "file_access_write",
        json!({"file_name": "plan.md", "content": "access"}),
    )
    .await;
    invoke_text(
        &m.tools,
        "file_memory_write",
        json!({"file_name": "plan.md", "content": "memory"}),
    )
    .await;
    assert_eq!(
        invoke_text(
            &a.tools,
            "file_access_read",
            json!({"file_name": "plan.md"})
        )
        .await,
        "access"
    );
    assert_eq!(
        invoke_text(
            &m.tools,
            "file_memory_read",
            json!({"file_name": "plan.md"})
        )
        .await,
        "memory"
    );
}

#[tokio::test]
async fn write_tool_reports_actionable_collision_errors() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let provider = FileAccessProvider::new(store);
    let state = SessionState::new();
    let mut c = ctx("s", &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    let out = invoke_text(
        &c.tools,
        "file_access_write",
        json!({"file_name": "dir/", "content": "x"}),
    )
    .await;
    assert_eq!(
        out,
        "Could not write file 'dir/': Invalid path: 'dir/'. A file path must not end with a path separator."
    );
    let dir = TempDir::new("prov-collide");
    let fs = Arc::new(fs_store(&dir));
    fs.write("blocker", "x", true).await.unwrap();
    fs.write("folder/x", "x", true).await.unwrap();
    let provider = FileAccessProvider::new(fs);
    let mut c = ctx("s", &state, vec![]);
    provider.before_run(&mut c).await.unwrap();
    assert_eq!(
        invoke_text(&c.tools, "file_access_write", json!({"file_name": "blocker/x", "content": "x"})).await,
        "Could not write file 'blocker/x': a parent path is already a file. Choose a different path."
    );
    assert_eq!(
        invoke_text(&c.tools, "file_access_write", json!({"file_name": "folder", "content": "x"})).await,
        "Could not write file 'folder': this path is already a directory. Choose a different file name."
    );
    let _ = Value::Null;
}
