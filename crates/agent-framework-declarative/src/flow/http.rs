//! `HttpRequestAction` and the HTTP handler abstraction (port of
//! `_executors_http.py` and `_http_handler.py`, mirroring .NET's
//! `IHttpRequestHandler`).
//!
//! A workflow containing an `HttpRequestAction` only builds when an
//! [`HttpRequestHandler`] is supplied to the
//! [`WorkflowFactory`](super::WorkflowFactory). The executor:
//!
//! * evaluates `method` (default `GET`, upper-cased), `url` (must be
//!   non-empty), `headers` (empty values dropped), `queryParameters`
//!   (Booleans as `true`/`false`, `null` dropped), `body`
//!   (`kind: json | raw | none` or the .NET `JsonRequestContent` /
//!   `RawRequestContent` / `NoRequestContent`; raw bodies default to
//!   `text/plain`), `requestTimeoutInMilliseconds` (positive integers only)
//!   and `connection.name`;
//! * on a 2xx status stores the body (JSON-parsed when possible, else the raw
//!   text, `null` when empty) at `response`, the comma-folded headers at
//!   `responseHeaders`, and — with a `conversationId` — appends an assistant
//!   message with the body to that conversation;
//! * on any other status still publishes `responseHeaders`, then fails with
//!   the URL and status code but **never** the body (it may hold private
//!   backend data).
//!
//! Transport errors become action errors naming the URL and error kind.
//!
//! Divergence: JSON bodies are serialized compactly (`{"a":1}`) where
//! upstream's `json.dumps` inserts spaces after separators.
//!
//! Security: the optional [`DefaultHttpRequestHandler`] (feature `http`)
//! performs **no** URL allow-listing or SSRF protection, exactly like
//! upstream; production deployments should supply their own handler.

use std::collections::BTreeMap;

use agent_framework_core::error::Error as CoreError;
use agent_framework_core::types::{Message, Role};
use agent_framework_core::workflow::WorkflowContext;
use async_trait::async_trait;
use serde_json::{Map, Value as Json};

use super::executor::{field, path_ref, ActionResult, DeclarativeExecutor};
use super::messages::{action_complete, message_to_json};
use super::state::{py_str, DeclarativeState, StateError};

/// A request for an [`HttpRequestHandler`] (upstream `HttpRequestInfo`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HttpRequestInfo {
    /// Upper-cased HTTP method.
    pub method: String,
    /// Absolute URL (already evaluated).
    pub url: String,
    /// Single-valued request headers.
    pub headers: Vec<(String, String)>,
    /// Query parameters to append to `url`.
    pub query_parameters: Vec<(String, String)>,
    /// The request body, if any.
    pub body: Option<String>,
    /// The body content type (ignored without a body).
    pub body_content_type: Option<String>,
    /// Per-request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Optional connection name for handlers that resolve credentials.
    pub connection_name: Option<String>,
}

/// A response from an [`HttpRequestHandler`] (upstream `HttpRequestResult`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HttpRequestResult {
    /// The HTTP status code.
    pub status_code: u16,
    /// Whether the status is 2xx.
    pub is_success_status_code: bool,
    /// The response body text.
    pub body: String,
    /// Response headers (names lower-cased), preserving repeated values.
    pub headers: BTreeMap<String, Vec<String>>,
}

/// Transport-level failures an [`HttpRequestHandler`] reports. Non-2xx
/// statuses are **not** errors; return them in [`HttpRequestResult`].
#[derive(Debug, thiserror::Error)]
pub enum HttpRequestError {
    /// The request timed out.
    #[error("timed out: {0}")]
    Timeout(String),
    /// A connection/protocol failure.
    #[error("transport error: {0}")]
    Transport(String),
    /// The request was invalid (bad URL, method, …).
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

/// Dispatches HTTP requests for `HttpRequestAction`. Implementations own any
/// allow-listing, SSRF guards, retries, and authentication.
#[async_trait]
pub trait HttpRequestHandler: Send + Sync {
    /// Send `info` and return the response (any status).
    async fn send(&self, info: HttpRequestInfo) -> Result<HttpRequestResult, HttpRequestError>;
}

/// Parse a response body: JSON first, else the raw text; empty → `null`.
pub(crate) fn parse_response_body(body: &str) -> Json {
    if body.is_empty() {
        return Json::Null;
    }
    serde_json::from_str(body).unwrap_or_else(|_| Json::String(body.to_string()))
}

fn action_error(msg: impl Into<String>) -> CoreError {
    CoreError::Workflow(format!("declarative action error: {}", msg.into()))
}

impl DeclarativeExecutor {
    fn eval_opt_text(
        &self,
        state: &DeclarativeState,
        v: Option<&Json>,
    ) -> Result<Option<String>, StateError> {
        let Some(v) = v.filter(|v| !v.is_null()) else {
            return Ok(None);
        };
        let e = state.eval_if_expression(v)?;
        Ok(match e {
            Json::Null => None,
            other => Some(py_str(&other)).filter(|s| !s.is_empty()),
        })
    }

    /// Evaluate the `headers` mapping in authoring order, dropping empty
    /// names and empty/`null` values.
    pub(crate) fn eval_headers(
        &self,
        state: &DeclarativeState,
    ) -> Result<Vec<(String, String)>, StateError> {
        let mut out = Vec::new();
        for (k, v) in self.ordered_entries(&["headers"]) {
            if k.is_empty() {
                continue;
            }
            if let Some(text) = self.eval_opt_text(state, Some(v))? {
                out.push((k.clone(), text));
            }
        }
        Ok(out)
    }

    pub(crate) fn connection_name(
        &self,
        state: &DeclarativeState,
    ) -> Result<Option<String>, StateError> {
        match field(&self.def, "connection") {
            Some(Json::Object(m)) => self.eval_opt_text(state, m.get("name")),
            _ => Ok(None),
        }
    }

    fn http_body(
        &self,
        state: &DeclarativeState,
    ) -> Result<(Option<String>, Option<String>), CoreError> {
        let Some(raw) = field(&self.def, "body").filter(|b| !b.is_null()) else {
            return Ok((None, None));
        };
        let Json::Object(body) = raw else {
            return Err(action_error(
                "HttpRequestAction 'body' must be a mapping with a 'kind' field (json, raw) or omitted entirely.",
            ));
        };
        let kind = body
            .get("kind")
            .filter(|k| super::state::py_truthy(k))
            .or_else(|| body.get("$kind"));
        let kind = match kind {
            None | Some(Json::Null) => {
                return Err(action_error(
                    "HttpRequestAction 'body' is missing 'kind'. Use 'json', 'raw', or omit 'body' for no request body.",
                ))
            }
            Some(Json::String(s)) => s.as_str(),
            Some(other) => {
                return Err(action_error(format!(
                    "HttpRequestAction 'body.kind' must be a string, got {other}."
                )))
            }
        };
        match kind {
            "none" | "NoRequestContent" => Ok((None, None)),
            "json" | "JsonRequestContent" => {
                let Some(content) = body.get("content").filter(|c| !c.is_null()) else {
                    return Ok((None, None));
                };
                let v = state.eval_if_expression(content)?;
                Ok((Some(v.to_string()), Some("application/json".into())))
            }
            "raw" | "RawRequestContent" => {
                let content = self.eval_opt_raw(state, body.get("content"))?;
                let content_type = self.eval_opt_text(state, body.get("contentType"))?;
                let content_type = match (&content, content_type) {
                    (Some(_), None) => Some("text/plain".to_string()),
                    (_, ct) => ct,
                };
                Ok((content, content_type))
            }
            other => Err(action_error(format!(
                "HttpRequestAction 'body.kind' has unsupported value '{other}'. Expected one of: json, raw, JsonRequestContent, RawRequestContent, NoRequestContent."
            ))),
        }
    }

    fn eval_opt_raw(
        &self,
        state: &DeclarativeState,
        v: Option<&Json>,
    ) -> Result<Option<String>, StateError> {
        let Some(v) = v.filter(|v| !v.is_null()) else {
            return Ok(None);
        };
        Ok(match state.eval_if_expression(v)? {
            Json::Null => None,
            other => Some(py_str(&other)),
        })
    }

    pub(crate) async fn http_request(
        &self,
        state: &mut DeclarativeState,
        ctx: &WorkflowContext,
    ) -> ActionResult {
        let handler = self
            .rt
            .http
            .clone()
            .ok_or_else(|| action_error("no HttpRequestHandler is configured"))?;
        let def = &self.def;
        let method = self
            .eval_opt_text(state, field(def, "method"))?
            .map(|m| m.to_uppercase())
            .unwrap_or_else(|| "GET".into());
        let url = match field(def, "url") {
            None => return Err(action_error("HttpRequestAction requires a 'url' field.")),
            Some(u) => match state.eval_if_expression(u)? {
                Json::String(s) if !s.is_empty() => s,
                _ => {
                    return Err(action_error(
                        "HttpRequestAction 'url' evaluated to an empty value.",
                    ))
                }
            },
        };
        let headers = self.eval_headers(state)?;
        let mut query_parameters = Vec::new();
        for (k, v) in self.ordered_entries(&["queryParameters"]) {
            if k.is_empty() || v.is_null() {
                continue;
            }
            match state.eval_if_expression(v)? {
                Json::Null => {}
                Json::Bool(b) => query_parameters.push((k.clone(), b.to_string())),
                other => query_parameters.push((k.clone(), py_str(&other))),
            }
        }
        let (body, body_content_type) = self.http_body(state)?;
        let timeout_ms = match field(def, "requestTimeoutInMilliseconds").filter(|v| !v.is_null()) {
            Some(v) => super::executor::json_to_int(&state.eval_if_expression(v)?)
                .filter(|n| *n > 0)
                .map(|n| n as u64),
            None => None,
        };
        let info = HttpRequestInfo {
            method,
            url: url.clone(),
            headers,
            query_parameters,
            body,
            body_content_type,
            timeout_ms,
            connection_name: self.connection_name(state)?,
        };
        let result = handler.send(info).await.map_err(|e| match e {
            HttpRequestError::Timeout(_) => {
                action_error(format!("HTTP request to '{url}' timed out."))
            }
            HttpRequestError::Transport(_) => {
                action_error(format!("HTTP request to '{url}' failed: TransportError"))
            }
            HttpRequestError::InvalidRequest(msg) => action_error(format!(
                "HTTP request to '{url}' failed: InvalidRequest: {msg}"
            )),
        })?;
        if result.is_success_status_code {
            if let Some(path) = path_ref(field(def, "response")) {
                state.set(&path, parse_response_body(&result.body))?;
            }
            self.assign_response_headers(state, &result)?;
            if !result.body.is_empty() {
                if let Some(path) = super::executor::conversation_messages_path(
                    state,
                    field(def, "conversationId"),
                )? {
                    let m = Message::new(Role::new(Role::ASSISTANT), result.body.clone());
                    state.append(&path, message_to_json(&m))?;
                }
            }
            return ctx.send_message(action_complete()).await;
        }
        // Upstream publishes the headers before raising; the core engine
        // discards a failed superstep's state writes, so they are only
        // observable to code inspecting `state` within this action.
        self.assign_response_headers(state, &result)?;
        Err(action_error(format!(
            "HTTP request to '{url}' failed with status code {}.",
            result.status_code
        )))
    }

    fn assign_response_headers(
        &self,
        state: &mut DeclarativeState,
        result: &HttpRequestResult,
    ) -> Result<(), StateError> {
        let Some(path) = path_ref(field(&self.def, "responseHeaders")) else {
            return Ok(());
        };
        if result.headers.is_empty() {
            return state.set(&path, Json::Null);
        }
        let folded: Map<String, Json> = result
            .headers
            .iter()
            .map(|(k, v)| (k.clone(), Json::String(v.join(","))))
            .collect();
        state.set(&path, Json::Object(folded))
    }
}

#[cfg(feature = "http")]
pub use default_handler::DefaultHttpRequestHandler;

#[cfg(feature = "http")]
mod default_handler {
    use super::*;
    use std::time::Duration;

    /// Characters a query string may carry literally (upstream `_QUERY_SAFE`),
    /// in addition to RFC 3986 unreserved characters.
    const QUERY_SAFE: &str = "!$&'()*+,;=:@/?%[]~";

    fn encode(s: &str, safe: &str) -> String {
        let mut out = String::new();
        for b in s.bytes() {
            let c = b as char;
            if c.is_ascii_alphanumeric() || "-._~".contains(c) || (c.is_ascii() && safe.contains(c))
            {
                out.push(c);
            } else {
                out.push_str(&format!("%{b:02X}"));
            }
        }
        out
    }

    /// The default [`HttpRequestHandler`], backed by `reqwest`.
    ///
    /// Like upstream's `httpx`-based default it does not follow redirects,
    /// keeps no cookies, appends `query_parameters` after any query already
    /// in the URL (preserving the authored query bytes), defaults a body's
    /// `Content-Type` to the request's content type or `text/plain`, and
    /// lower-cases response header names. It performs **no** URL filtering
    /// or SSRF protection.
    #[derive(Debug, Clone)]
    pub struct DefaultHttpRequestHandler {
        client: reqwest::Client,
    }

    impl Default for DefaultHttpRequestHandler {
        fn default() -> Self {
            Self::new()
        }
    }

    impl DefaultHttpRequestHandler {
        /// A handler with its own client (no redirects, no cookie store).
        pub fn new() -> Self {
            let client = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default();
            Self { client }
        }

        /// A handler using a caller-owned client (its redirect and cookie
        /// behaviour is retained).
        pub fn with_client(client: reqwest::Client) -> Self {
            Self { client }
        }

        /// Compose the final URL (exposed for tests).
        pub fn compose_url(info: &HttpRequestInfo) -> Result<String, HttpRequestError> {
            if info.url.is_empty() {
                return Err(HttpRequestError::InvalidRequest(
                    "HttpRequestInfo.url must be a non-empty string.".into(),
                ));
            }
            let parsed = reqwest::Url::parse(&info.url).map_err(|_| {
                HttpRequestError::InvalidRequest(
                    "HttpRequestInfo.url must be an absolute HTTP or HTTPS URL.".into(),
                )
            })?;
            if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                return Err(HttpRequestError::InvalidRequest(
                    "HttpRequestInfo.url must be an absolute HTTP or HTTPS URL.".into(),
                ));
            }
            let (base, fragment) = match info.url.split_once('#') {
                Some((b, f)) => (b, Some(f)),
                None => (info.url.as_str(), None),
            };
            let (path, mut query) = match base.split_once('?') {
                Some((p, q)) => (p.to_string(), q.to_string()),
                None => (base.to_string(), String::new()),
            };
            let explicit: Vec<String> = info
                .query_parameters
                .iter()
                .filter(|(k, _)| !k.is_empty())
                .map(|(k, v)| format!("{}={}", encode(k, "/"), encode(v, "/")))
                .collect();
            if !explicit.is_empty() {
                if !query.is_empty() {
                    query.push('&');
                }
                query.push_str(&explicit.join("&"));
            }
            let mut url = path;
            if !query.is_empty() {
                url.push('?');
                url.push_str(&encode(&query, QUERY_SAFE));
            }
            if let Some(f) = fragment {
                url.push('#');
                url.push_str(f);
            }
            Ok(url)
        }
    }

    #[async_trait]
    impl HttpRequestHandler for DefaultHttpRequestHandler {
        async fn send(&self, info: HttpRequestInfo) -> Result<HttpRequestResult, HttpRequestError> {
            if info.method.is_empty() {
                return Err(HttpRequestError::InvalidRequest(
                    "HttpRequestInfo.method must be a non-empty string.".into(),
                ));
            }
            let url = Self::compose_url(&info)?;
            let method = reqwest::Method::from_bytes(info.method.as_bytes())
                .map_err(|e| HttpRequestError::InvalidRequest(e.to_string()))?;
            let mut req = self.client.request(method, &url);
            for (k, v) in &info.headers {
                req = req.header(k.as_str(), v.as_str());
            }
            if let Some(body) = &info.body {
                let has_ct = info
                    .headers
                    .iter()
                    .any(|(k, _)| k.eq_ignore_ascii_case("content-type"));
                if !has_ct {
                    let ct = info.body_content_type.as_deref().unwrap_or("text/plain");
                    req = req.header("Content-Type", ct);
                }
                req = req.body(body.clone());
            }
            if let Some(ms) = info.timeout_ms.filter(|ms| *ms > 0) {
                req = req.timeout(Duration::from_millis(ms));
            }
            let response = req.send().await.map_err(|e| {
                if e.is_timeout() {
                    HttpRequestError::Timeout(e.to_string())
                } else {
                    HttpRequestError::Transport(e.to_string())
                }
            })?;
            let status = response.status().as_u16();
            let mut headers: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for (k, v) in response.headers() {
                headers
                    .entry(k.as_str().to_ascii_lowercase())
                    .or_default()
                    .push(String::from_utf8_lossy(v.as_bytes()).into_owned());
            }
            let body = response.text().await.map_err(|e| {
                if e.is_timeout() {
                    HttpRequestError::Timeout(e.to_string())
                } else {
                    HttpRequestError::Transport(e.to_string())
                }
            })?;
            Ok(HttpRequestResult {
                status_code: status,
                is_success_status_code: (200..300).contains(&status),
                body,
                headers,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn info(url: &str, params: &[(&str, &str)]) -> HttpRequestInfo {
            HttpRequestInfo {
                method: "GET".into(),
                url: url.into(),
                query_parameters: params
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
                ..Default::default()
            }
        }

        #[test]
        fn composes_query_preserving_authored_bytes() {
            let u = DefaultHttpRequestHandler::compose_url(&info(
                "https://x.test/a?flag&k=%2F&k=2#frag",
                &[("k", "a b"), ("q", "café")],
            ))
            .unwrap();
            assert_eq!(
                u,
                "https://x.test/a?flag&k=%2F&k=2&k=a%20b&q=caf%C3%A9#frag"
            );
        }

        #[test]
        fn rejects_non_http_urls() {
            for bad in ["", "ftp://x.test/", "/relative", "not a url"] {
                assert!(
                    DefaultHttpRequestHandler::compose_url(&info(bad, &[])).is_err(),
                    "{bad}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn body_parsing_is_json_first() {
        assert_eq!(parse_response_body(""), Json::Null);
        assert_eq!(parse_response_body("{\"a\":1}"), json!({"a": 1}));
        assert_eq!(parse_response_body("plain"), json!("plain"));
    }
}
