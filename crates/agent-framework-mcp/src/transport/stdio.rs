//! Stdio transport: spawns an MCP server as a child process and speaks
//! newline-delimited JSON-RPC over its stdin/stdout.
//!
//! Note: MCP's stdio framing is one JSON object per line — unlike LSP, there
//! are no `Content-Length` headers.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;
use std::process::Stdio;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use tokio::task::JoinHandle;

use agent_framework_core::error::{Error, Result};

use crate::protocol::{self, IdGenerator, IncomingMessage, RpcError};
use crate::sampling::{BoxedNotificationHandler, BoxedServerRequestHandler};
use crate::transport::McpTransport;

type PendingMap = StdMutex<HashMap<i64, oneshot::Sender<std::result::Result<Value, RpcError>>>>;

/// The minimal baseline of parent environment variables passed through to a
/// spawned MCP stdio server when *not* inheriting the full parent environment
/// (the default). This is deliberately small: just what a child typically needs
/// to start and behave (executable lookup, temp dir, locale) — not application
/// secrets. Names cover both Unix and Windows so spawning works on either.
const BASELINE_ENV_NAMES: &[&str] = &[
    // Executable/library lookup and user home.
    "PATH",
    "HOME",
    // Temporary directory (POSIX + Windows).
    "TMPDIR",
    "TMP",
    "TEMP",
    // Locale.
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    // Windows essentials so a child process can even be created / find its DLLs.
    "SystemRoot",
    "SYSTEMROOT",
    "windir",
    "USERPROFILE",
    "PATHEXT",
    "ComSpec",
    "APPDATA",
    "LOCALAPPDATA",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
];

/// The environment policy for a spawned MCP stdio server.
///
/// **Secure by default:** unless [`StdioEnv::inherit_parent_environment`] is
/// enabled, the child does **not** inherit the parent process's environment.
/// It starts from a minimal baseline allowlist (`BASELINE_ENV_NAMES` — PATH,
/// temp-dir, and locale variables) plus any explicitly configured variables.
/// This follows least privilege: an MCP package launched through `npx`,
/// `python`, or another package runner does not automatically receive unrelated
/// secrets (OpenAI/Anthropic keys, AWS/Azure credentials, GitHub tokens,
/// database URLs, tracing-exporter secrets) that happen to be present in the
/// host environment.
///
/// Escape hatches:
/// - [`StdioEnv::inherit_parent_environment(true)`](StdioEnv::inherit_parent_environment)
///   restores full inheritance (the old behavior).
/// - [`StdioEnv::inherit_var`] / [`StdioEnv::inherit_vars`] pass through
///   selected named variables by value without inheriting everything.
#[derive(Debug, Clone, Default)]
pub struct StdioEnv {
    /// Explicit key/value pairs, applied last (highest precedence).
    vars: HashMap<String, String>,
    /// Inherit the full parent environment (legacy behavior). Off by default.
    inherit_all: bool,
    /// Additional parent variable names to pass through by value when not
    /// inheriting the whole parent environment.
    inherit_names: Vec<String>,
}

impl StdioEnv {
    /// A secure-by-default policy: minimal baseline only, no parent secrets.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add/override an explicit environment variable on the child.
    pub fn var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.vars.insert(key.into(), value.into());
        self
    }

    /// Add/override several explicit environment variables on the child.
    pub fn vars<I, K, V>(mut self, vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (k, v) in vars {
            self.vars.insert(k.into(), v.into());
        }
        self
    }

    /// Inherit the full parent process environment (default `false`).
    ///
    /// Enabling this reverts to inheriting every parent variable — convenient,
    /// but it re-exposes all host secrets to the child. Prefer
    /// [`Self::inherit_var`] for the specific variables the server needs.
    pub fn inherit_parent_environment(mut self, inherit: bool) -> Self {
        self.inherit_all = inherit;
        self
    }

    /// Pass through a single named parent variable by value (no effect when
    /// [`Self::inherit_parent_environment`] is enabled, which already inherits
    /// everything).
    pub fn inherit_var(mut self, name: impl Into<String>) -> Self {
        self.inherit_names.push(name.into());
        self
    }

    /// Pass through several named parent variables by value.
    pub fn inherit_vars<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inherit_names.extend(names.into_iter().map(Into::into));
        self
    }

    /// Compute the explicit variables to set on the child in the (default)
    /// non-inheriting case: the baseline allowlist and any explicitly-named
    /// inherited variables resolved through `lookup`, then the configured
    /// `vars` layered on top (highest precedence). Pure, so it is unit-testable
    /// without touching the real process environment.
    fn baseline_resolved(
        &self,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> HashMap<String, String> {
        let mut out = HashMap::new();
        for name in BASELINE_ENV_NAMES {
            if let Some(v) = lookup(name) {
                out.insert((*name).to_string(), v);
            }
        }
        for name in &self.inherit_names {
            if let Some(v) = lookup(name) {
                out.insert(name.clone(), v);
            }
        }
        for (k, v) in &self.vars {
            out.insert(k.clone(), v.clone());
        }
        out
    }

    /// Apply this policy to `cmd`.
    fn apply(&self, cmd: &mut Command) {
        if self.inherit_all {
            // Keep the inherited parent environment; layer explicit overrides.
            cmd.envs(&self.vars);
        } else {
            // Drop the parent environment entirely, then set only the resolved
            // baseline/allowlisted/explicit variables.
            cmd.env_clear();
            let resolved = self.baseline_resolved(|n| std::env::var(n).ok());
            cmd.envs(&resolved);
        }
    }
}

impl From<HashMap<String, String>> for StdioEnv {
    fn from(vars: HashMap<String, String>) -> Self {
        Self {
            vars,
            ..Self::default()
        }
    }
}

/// An MCP transport backed by a child process's stdin/stdout.
///
/// The child is spawned with `kill_on_drop`, and is additionally killed
/// explicitly by [`McpStdioTransport::close`] and by `Drop`, so the process
/// never outlives this transport.
pub struct McpStdioTransport {
    inner: std::sync::Arc<StdioInner>,
    /// Applied to every [`McpTransport::call`] while awaiting its response.
    /// Unset (the default) waits indefinitely. See [`Self::with_request_timeout`].
    request_timeout: Option<Duration>,
}

struct StdioInner {
    stdin: AsyncMutex<ChildStdin>,
    child: StdMutex<Child>,
    pending: PendingMap,
    next_id: IdGenerator,
    reader_task: StdMutex<Option<JoinHandle<()>>>,
    stderr_task: StdMutex<Option<JoinHandle<()>>>,
    /// Handler for server-initiated requests (`ping`, `sampling/createMessage`,
    /// `roots/list`), installed via [`McpTransport::set_server_request_handler`].
    server_request_handler: StdMutex<Option<BoxedServerRequestHandler>>,
    /// Handler for server notifications (e.g. `notifications/tools/list_changed`),
    /// installed via [`McpTransport::set_notification_handler`].
    notification_handler: StdMutex<Option<BoxedNotificationHandler>>,
    /// Set once the server's stdout has closed (see [`McpTransport::is_closed`]).
    closed: std::sync::atomic::AtomicBool,
}

impl McpStdioTransport {
    /// Spawn `command` with `args`, wiring up stdio pipes and background
    /// tasks that route responses back to their callers and drain
    /// notifications/stderr to `tracing`.
    ///
    /// `env` controls the child's environment. By default ([`StdioEnv::new`])
    /// the child does **not** inherit the parent process environment — it sees
    /// only a minimal baseline allowlist plus any explicitly configured
    /// variables, so host secrets are not automatically disclosed to the
    /// server. See [`StdioEnv`] for the escape hatches. `cwd`, if provided,
    /// sets the child's working directory.
    pub async fn spawn(
        command: impl AsRef<OsStr>,
        args: &[String],
        env: &StdioEnv,
        cwd: Option<&Path>,
    ) -> Result<Self> {
        let command_ref = command.as_ref();
        let mut cmd = Command::new(command_ref);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        env.apply(&mut cmd);
        if let Some(cwd) = cwd {
            cmd.current_dir(cwd);
        }

        let mut child = cmd.spawn().map_err(|e| {
            Error::service(format!(
                "failed to start MCP server '{}': {e}",
                command_ref.to_string_lossy()
            ))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::service("MCP child process has no stdin pipe"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::service("MCP child process has no stdout pipe"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| Error::service("MCP child process has no stderr pipe"))?;

        let pending: PendingMap = StdMutex::new(HashMap::new());
        let inner = std::sync::Arc::new(StdioInner {
            stdin: AsyncMutex::new(stdin),
            child: StdMutex::new(child),
            pending,
            next_id: IdGenerator::new(),
            reader_task: StdMutex::new(None),
            stderr_task: StdMutex::new(None),
            server_request_handler: StdMutex::new(None),
            notification_handler: StdMutex::new(None),
            closed: std::sync::atomic::AtomicBool::new(false),
        });

        let reader_task = spawn_reader(stdout, inner.clone());
        let stderr_task = spawn_stderr_drain(stderr);
        *inner.reader_task.lock().unwrap() = Some(reader_task);
        *inner.stderr_task.lock().unwrap() = Some(stderr_task);

        Ok(Self {
            inner,
            request_timeout: None,
        })
    }

    /// Set a per-request timeout applied while awaiting a response to any
    /// JSON-RPC request sent over this transport. Mirrors
    /// [`crate::McpStreamableHttpTransport`]'s `timeout` option; unset (the
    /// default) waits indefinitely. A request that times out is removed from
    /// the pending-response table, so a late reply from the server is
    /// discarded rather than mis-delivered to a later call reusing the id
    /// space.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }
}

/// Encode `message` as one newline-terminated JSON line and write it to the
/// child's stdin. A free function (rather than an `McpStdioTransport`
/// method) so the background reader task — which only holds `Arc<StdioInner>`,
/// not the outer transport handle — can use it too, to write responses to
/// server-initiated requests.
async fn write_line(inner: &StdioInner, message: &Value) -> Result<()> {
    let mut line = serde_json::to_string(message)
        .map_err(|e| Error::service(format!("failed to encode MCP message: {e}")))?;
    line.push('\n');
    let mut stdin = inner.stdin.lock().await;
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|e| Error::service(format!("failed to write to MCP server stdin: {e}")))?;
    stdin
        .flush()
        .await
        .map_err(|e| Error::service(format!("failed to flush MCP server stdin: {e}")))?;
    Ok(())
}

#[async_trait]
impl McpTransport for McpStdioTransport {
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.inner.next_id.next();
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().unwrap().insert(id, tx);

        let request = protocol::build_request(id, method, params);
        if let Err(e) = write_line(&self.inner, &request).await {
            self.inner.pending.lock().unwrap().remove(&id);
            return Err(e);
        }

        let await_response = async {
            match rx.await {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(rpc_err)) => Err(Error::service(rpc_err.to_string())),
                Err(_) => Err(Error::service(
                    "MCP server closed the connection before responding",
                )),
            }
        };
        match self.request_timeout {
            None => await_response.await,
            Some(timeout) => match tokio::time::timeout(timeout, await_response).await {
                Ok(result) => result,
                Err(_) => {
                    self.inner.pending.lock().unwrap().remove(&id);
                    Err(Error::service(format!(
                        "MCP request '{method}' timed out after {timeout:?}"
                    )))
                }
            },
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let notification = protocol::build_notification(method, params);
        write_line(&self.inner, &notification).await
    }

    async fn close(&self) -> Result<()> {
        if let Ok(mut child) = self.inner.child.lock() {
            let _ = child.start_kill();
        }
        Ok(())
    }

    fn set_server_request_handler(&self, handler: BoxedServerRequestHandler) {
        *self.inner.server_request_handler.lock().unwrap() = Some(handler);
    }

    fn set_notification_handler(&self, handler: BoxedNotificationHandler) {
        *self.inner.notification_handler.lock().unwrap() = Some(handler);
    }

    fn is_closed(&self) -> bool {
        self.inner.closed.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl Drop for StdioInner {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.start_kill();
        }
        if let Ok(mut task) = self.reader_task.lock() {
            if let Some(task) = task.take() {
                task.abort();
            }
        }
        if let Ok(mut task) = self.stderr_task.lock() {
            if let Some(task) = task.take() {
                task.abort();
            }
        }
        // Fail any requests still waiting on a response so callers don't hang.
        if let Ok(mut pending) = self.pending.lock() {
            for (_, tx) in pending.drain() {
                let _ = tx.send(Err(RpcError {
                    code: -1,
                    message: "MCP stdio transport dropped".to_string(),
                    data: None,
                }));
            }
        }
    }
}

/// Spawn the background task that reads newline-delimited JSON-RPC messages
/// from the server's stdout and routes them: responses go to their waiting
/// caller by id, notifications are logged, and server-initiated requests are
/// answered via whatever handler [`McpTransport::set_server_request_handler`]
/// installed (see [`spawn_server_request_response`]).
fn spawn_reader(
    stdout: tokio::process::ChildStdout,
    inner: std::sync::Arc<StdioInner>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(value) => route_incoming(&inner, value),
                        Err(e) => {
                            tracing::warn!(error = %e, line = %trimmed, "MCP: non-JSON line from server stdout");
                        }
                    }
                }
                Ok(None) => {
                    tracing::debug!("MCP server stdout closed");
                    break;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "MCP: error reading server stdout");
                    break;
                }
            }
        }
        // The server is gone; unblock anyone still waiting on a response.
        // Marked closed first, so a caller woken below already sees it.
        inner
            .closed
            .store(true, std::sync::atomic::Ordering::Release);
        let mut pending = inner.pending.lock().unwrap();
        for (_, tx) in pending.drain() {
            let _ = tx.send(Err(RpcError {
                code: -1,
                message: "MCP server connection closed".to_string(),
                data: None,
            }));
        }
    })
}

fn route_incoming(inner: &std::sync::Arc<StdioInner>, value: Value) {
    match protocol::parse_incoming(value) {
        IncomingMessage::Response { id, result } => {
            if let Some(tx) = inner.pending.lock().unwrap().remove(&id) {
                let _ = tx.send(result);
            } else {
                tracing::debug!(id, "MCP: response for unknown/already-resolved request id");
            }
        }
        IncomingMessage::Notification { method, params } => {
            tracing::debug!(method = %method, params = %params, "MCP server notification");
            dispatch_notification(inner.clone(), method, params);
        }
        IncomingMessage::ServerRequest { id, method, params } => {
            spawn_server_request_response(inner.clone(), id, method, params);
        }
        IncomingMessage::Malformed(v) => {
            tracing::warn!(raw = %v, "MCP: unrecognized JSON-RPC message shape");
        }
    }
}

/// Dispatch a server notification (e.g. `notifications/tools/list_changed`)
/// to whatever handler [`McpTransport::set_notification_handler`] installed,
/// in its own task so a slow handler doesn't block the reader loop from
/// noticing other incoming messages in the meantime. No response is expected
/// or sent; a notification with no handler registered is simply dropped
/// (already logged by the caller).
fn dispatch_notification(inner: std::sync::Arc<StdioInner>, method: String, params: Value) {
    tokio::spawn(async move {
        let handler = inner.notification_handler.lock().unwrap().clone();
        if let Some(handler) = handler {
            handler(method, params).await;
        }
    });
}

/// Compute the response to a server-initiated request (via whatever handler
/// is registered — see [`McpTransport::set_server_request_handler`]) and
/// write it back over stdin, in its own task so a slow handler (e.g. a
/// sampling handler calling out to an LLM) doesn't block the reader loop
/// from noticing other incoming messages in the meantime.
fn spawn_server_request_response(
    inner: std::sync::Arc<StdioInner>,
    id: Value,
    method: String,
    params: Value,
) {
    tokio::spawn(async move {
        let handler = inner.server_request_handler.lock().unwrap().clone();
        let result = match handler {
            Some(h) => h(method.clone(), params).await,
            None => Err(RpcError {
                code: -32601,
                message: format!("Method not found: {method}"),
                data: None,
            }),
        };
        let envelope = match result {
            Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
            Err(err) => json!({ "jsonrpc": "2.0", "id": id, "error": err }),
        };
        if let Err(e) = write_line(&inner, &envelope).await {
            tracing::warn!(
                error = %e,
                "MCP: failed to write response for a server-initiated request"
            );
        }
    });
}

/// Spawn the background task that drains the server's stderr to `tracing`.
fn spawn_stderr_drain(stderr: tokio::process::ChildStderr) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if !line.trim().is_empty() {
                tracing::debug!(target: "mcp::server_stderr", "{line}");
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake parent environment for exercising the (pure) resolution logic
    /// without depending on the real process environment.
    fn fake_parent(name: &str) -> Option<String> {
        match name {
            "PATH" => Some("/usr/bin:/bin".to_string()),
            "LANG" => Some("en_US.UTF-8".to_string()),
            // Secrets that must NOT leak into the child by default.
            "OPENAI_API_KEY" => Some("sk-secret".to_string()),
            "AWS_SECRET_ACCESS_KEY" => Some("aws-secret".to_string()),
            "GITHUB_TOKEN" => Some("ghp_secret".to_string()),
            _ => None,
        }
    }

    #[test]
    fn default_policy_keeps_baseline_and_drops_secrets() {
        let env = StdioEnv::new();
        let resolved = env.baseline_resolved(fake_parent);
        // Baseline present variables pass through.
        assert_eq!(
            resolved.get("PATH").map(String::as_str),
            Some("/usr/bin:/bin")
        );
        assert_eq!(
            resolved.get("LANG").map(String::as_str),
            Some("en_US.UTF-8")
        );
        // Secrets are not inherited.
        assert!(!resolved.contains_key("OPENAI_API_KEY"));
        assert!(!resolved.contains_key("AWS_SECRET_ACCESS_KEY"));
        assert!(!resolved.contains_key("GITHUB_TOKEN"));
    }

    #[test]
    fn explicit_vars_are_applied_and_override() {
        let env = StdioEnv::new()
            .var("MCP_CONFIG", "value")
            .var("PATH", "/custom/bin");
        let resolved = env.baseline_resolved(fake_parent);
        assert_eq!(
            resolved.get("MCP_CONFIG").map(String::as_str),
            Some("value")
        );
        // Explicit vars win over the inherited baseline.
        assert_eq!(
            resolved.get("PATH").map(String::as_str),
            Some("/custom/bin")
        );
    }

    #[test]
    fn inherit_named_var_passes_one_secret_through_opt_in() {
        let env = StdioEnv::new().inherit_var("GITHUB_TOKEN");
        let resolved = env.baseline_resolved(fake_parent);
        // Only the explicitly named variable is inherited; others still dropped.
        assert_eq!(
            resolved.get("GITHUB_TOKEN").map(String::as_str),
            Some("ghp_secret")
        );
        assert!(!resolved.contains_key("OPENAI_API_KEY"));
    }

    #[test]
    fn from_hashmap_sets_explicit_vars_and_keeps_inherit_all_false() {
        // `From<HashMap>` only seeds the explicit `vars`; it does not enable
        // full parent inheritance, and the baseline allowlist still applies
        // when resolved against a real parent environment.
        let mut map = HashMap::new();
        map.insert("A".to_string(), "1".to_string());
        let env = StdioEnv::from(map);
        assert!(!env.inherit_all);
        let resolved = env.baseline_resolved(|_| None);
        assert_eq!(resolved.get("A").map(String::as_str), Some("1"));
    }
}
