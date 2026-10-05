//! Error types for the agent framework.

use std::fmt;

/// The result type used throughout the framework.
pub type Result<T> = std::result::Result<T, Error>;

/// The primary error type for the agent framework.
///
/// This mirrors the exception hierarchy used by the Python
/// `agent_framework.exceptions` module while remaining idiomatic Rust.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error occurred while initializing an agent.
    #[error("agent initialization error: {0}")]
    AgentInitialization(String),

    /// An error occurred while executing an agent run.
    #[error("agent execution error: {0}")]
    AgentExecution(String),

    /// An error occurred while (de)serializing a value.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// A content item could not be parsed or was of an unknown type.
    #[error("content error: {0}")]
    Content(String),

    /// A tool/function invocation failed.
    #[error("tool error: {0}")]
    Tool(String),

    /// A chat client / service returned an error.
    ///
    /// Used for non-HTTP service failures (transport errors, stream-decode
    /// errors, in-body error payloads on an otherwise-successful response). For
    /// a non-success HTTP status, prefer [`Error::ServiceStatus`], which also
    /// carries the status code and any `Retry-After`.
    #[error("service error: {0}")]
    Service(String),

    /// A chat client / service returned a non-success HTTP status.
    ///
    /// Distinct from [`Error::Service`] so a retry layer can inspect the
    /// numeric status code and any server-advised `Retry-After` delay (in
    /// seconds). Displays like [`Error::Service`] (a `service error: ...`
    /// message), with the status code folded into the message.
    ///
    /// This is the fallback classification for a non-success status that
    /// isn't one of the more specific variants below (notably `408`/`429`/
    /// `5xx`, which a retry layer treats as transient) — see
    /// [`Error::ServiceInvalidAuth`], [`Error::ServiceInvalidRequest`], and
    /// [`Error::ServiceContentFilter`] for statuses a provider client can
    /// classify more precisely.
    #[error("service error: {message}")]
    ServiceStatus {
        /// The HTTP status code returned by the service.
        status: u16,
        /// A human-readable message (typically the response body).
        message: String,
        /// The server-advised retry delay in seconds, parsed from the
        /// `Retry-After` header when present.
        retry_after: Option<f64>,
    },

    /// The service rejected the request due to missing or invalid
    /// credentials (typically HTTP `401`/`403`).
    ///
    /// Mirrors upstream's `ServiceInvalidAuthError`. Like [`Error::Service`],
    /// this carries no status code of its own (the numeric status, when
    /// known, is folded into the message by the provider client) — it exists
    /// so callers, and the default retry policy, can treat authentication /
    /// authorization failures as definitively non-transient without
    /// inspecting a status code themselves. Never retried by
    /// [`RetryOn::Default`](crate::client::RetryOn::Default).
    #[error("service error: {message}")]
    ServiceInvalidAuth {
        /// A human-readable message (typically the response body).
        message: String,
    },

    /// The service rejected the request as malformed or otherwise invalid
    /// (typically HTTP `400`/`404`/`422`) for a reason other than content
    /// filtering.
    ///
    /// Mirrors upstream's `ServiceInvalidRequestError`. See
    /// [`Error::ServiceContentFilter`] for the content-filter-specific case.
    /// Never retried by [`RetryOn::Default`](crate::client::RetryOn::Default)
    /// — a request that was rejected as invalid will be rejected again
    /// unchanged.
    #[error("service error: {message}")]
    ServiceInvalidRequest {
        /// A human-readable message (typically the response body).
        message: String,
    },

    /// The service refused the request (or part of a response) because it
    /// tripped a content filter / moderation policy.
    ///
    /// Mirrors upstream's `ServiceContentFilterException`
    /// (`OpenAIContentFilterException` for OpenAI/Azure OpenAI specifically).
    /// Never retried by [`RetryOn::Default`](crate::client::RetryOn::Default)
    /// — the content, not the service, is the problem.
    #[error("service error: {message}")]
    ServiceContentFilter {
        /// A human-readable message (typically the response body).
        message: String,
        /// The provider's structured breakdown of *why*, when the body
        /// carried one. See [`ContentFilterDetail`].
        ///
        /// Boxed so the common `None` costs one pointer rather than the
        /// whole map — this variant sits in an enum every fallible call
        /// returns.
        detail: Option<Box<ContentFilterDetail>>,
    },

    /// Function middleware signalled an unrecoverable failure: the run must
    /// stop rather than continue with a tool-error result.
    ///
    /// The function-invocation loop absorbs every other error a tool or its
    /// middleware produces into a `FunctionResultContent { exception, .. }`,
    /// hands it back to the model, and keeps looping — the right default for
    /// an ordinary tool failure the model can recover from or route around.
    /// An enforcement layer (a guardrail, a policy check, an authorization
    /// gate) needs the opposite: when it refuses a call, the run must fail
    /// closed, not hand the model an error string and let it try again.
    ///
    /// Middleware returning this variant gets that fail-closed escape. It is
    /// the only error the loop propagates instead of absorbing; when one of a
    /// parallel batch of calls raises it, the batch's in-flight siblings are
    /// dropped (cancelled) and the failure surfaces from the run. Mirrors
    /// upstream's `MiddlewareFailure` exception (Python #7562).
    ///
    /// The signal is carried by the error *type*, not by who produced it, so a
    /// tool executor that returns this variant is propagated the same way.
    /// Middleware that wants the ordinary absorb-and-continue contract should
    /// keep returning any other variant — [`Error::Tool`] is the usual choice.
    #[error("middleware failure: {0}")]
    MiddlewareFailure(String),

    /// An agent-hooks interception point blocked the guarded action.
    ///
    /// Raised by the [`agent_hooks`](crate::agent_hooks) enforcement when a
    /// verdict denies at `input`, `pre_model_call`, `post_model_call` or
    /// `output` (and, for `host_error:*` failures of the enforcement layer
    /// itself, at the tool seam). Carries the interception record. Mirrors
    /// upstream `agent_hooks.InterceptionBlocked`.
    #[error("{0}")]
    InterceptionBlocked(Box<crate::agent_hooks::InterceptionBlocked>),

    /// A workflow validation or execution error.
    #[error("workflow error: {0}")]
    Workflow(String),

    /// Two streamed content items could not be merged (mismatched ids).
    #[error("addition item mismatch: {0}")]
    AdditionItemMismatch(String),

    /// A required configuration value was missing or invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// Evaluation results did not pass: a failed or errored item, a score
    /// below a threshold, or a run that did not complete.
    ///
    /// Returned by the CI-gate assertions on
    /// [`EvalResults`](crate::evaluation::EvalResults) (`raise_for_status`,
    /// `assert_score_at_least`, ...). Mirrors upstream's `EvalNotPassedError`.
    #[error("evaluation not passed: {0}")]
    EvalNotPassed(String),

    /// An underlying JSON error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Any other error, wrapping a boxed source.
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Create an [`Error::Other`] from anything displayable.
    pub fn other(msg: impl fmt::Display) -> Self {
        Error::Other(msg.to_string())
    }

    /// Create an [`Error::Service`] from anything displayable.
    pub fn service(msg: impl fmt::Display) -> Self {
        Error::Service(msg.to_string())
    }

    /// Create an [`Error::ServiceStatus`] from an HTTP status code, a message,
    /// and an optional `Retry-After` delay (in seconds).
    pub fn service_status(status: u16, msg: impl fmt::Display, retry_after: Option<f64>) -> Self {
        Error::ServiceStatus {
            status,
            message: msg.to_string(),
            retry_after,
        }
    }

    /// Create an [`Error::ServiceInvalidAuth`] from anything displayable.
    pub fn service_invalid_auth(msg: impl fmt::Display) -> Self {
        Error::ServiceInvalidAuth {
            message: msg.to_string(),
        }
    }

    /// Create an [`Error::ServiceInvalidRequest`] from anything displayable.
    pub fn service_invalid_request(msg: impl fmt::Display) -> Self {
        Error::ServiceInvalidRequest {
            message: msg.to_string(),
        }
    }

    /// Create an [`Error::ServiceContentFilter`] from anything displayable.
    pub fn service_content_filter(msg: impl fmt::Display) -> Self {
        Error::ServiceContentFilter {
            message: msg.to_string(),
            detail: None,
        }
    }

    /// As [`Self::service_content_filter`], carrying the provider's
    /// structured breakdown of which categories tripped.
    pub fn service_content_filter_with_detail(
        msg: impl fmt::Display,
        detail: ContentFilterDetail,
    ) -> Self {
        Error::ServiceContentFilter {
            message: msg.to_string(),
            detail: Some(Box::new(detail)),
        }
    }

    /// The content-filter breakdown, when this is an
    /// [`Error::ServiceContentFilter`] that carried one.
    pub fn content_filter_detail(&self) -> Option<&ContentFilterDetail> {
        match self {
            Error::ServiceContentFilter { detail, .. } => detail.as_deref(),
            _ => None,
        }
    }

    /// The HTTP status code carried by this error, if it is an
    /// [`Error::ServiceStatus`].
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::ServiceStatus { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// The server-advised retry delay in seconds, if this is an
    /// [`Error::ServiceStatus`] that carried a `Retry-After` header.
    pub fn retry_after(&self) -> Option<f64> {
        match self {
            Error::ServiceStatus { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Create an [`Error::Tool`] from anything displayable.
    pub fn tool(msg: impl fmt::Display) -> Self {
        Error::Tool(msg.to_string())
    }

    /// Create an [`Error::MiddlewareFailure`] from anything displayable: the
    /// fail-closed signal function middleware returns to stop a run outright
    /// instead of having its error absorbed into a tool-error result.
    pub fn middleware_failure(msg: impl fmt::Display) -> Self {
        Error::MiddlewareFailure(msg.to_string())
    }

    /// Whether this error is the [`Error::MiddlewareFailure`] fail-closed
    /// signal, which the function-invocation loop propagates rather than
    /// absorbing into a tool-error result.
    pub fn is_middleware_failure(&self) -> bool {
        matches!(self, Error::MiddlewareFailure(_))
    }

    /// The agent-hooks block this error carries, if it is an
    /// [`Error::InterceptionBlocked`].
    pub fn interception_blocked(&self) -> Option<&crate::agent_hooks::InterceptionBlocked> {
        match self {
            Error::InterceptionBlocked(blocked) => Some(blocked),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_invalid_auth_constructor_and_display() {
        let err = Error::service_invalid_auth("OpenAI API error 401: unauthorized");
        assert!(matches!(err, Error::ServiceInvalidAuth { .. }));
        assert_eq!(
            err.to_string(),
            "service error: OpenAI API error 401: unauthorized"
        );
        // Not a `ServiceStatus`, so it carries no status/retry_after of its
        // own (the numeric status lives in the message text instead).
        assert_eq!(err.status(), None);
        assert_eq!(err.retry_after(), None);
    }

    #[test]
    fn service_invalid_request_constructor_and_display() {
        let err = Error::service_invalid_request("OpenAI API error 400: bad request");
        assert!(matches!(err, Error::ServiceInvalidRequest { .. }));
        assert_eq!(
            err.to_string(),
            "service error: OpenAI API error 400: bad request"
        );
    }

    #[test]
    fn service_content_filter_constructor_and_display() {
        let err = Error::service_content_filter("OpenAI API error 400: content filtered");
        assert!(matches!(err, Error::ServiceContentFilter { .. }));
        assert_eq!(
            err.to_string(),
            "service error: OpenAI API error 400: content filtered"
        );
    }

    /// The new variants must not silently become retryable-looking:
    /// `status()`/`retry_after()` only ever return `Some` for
    /// [`Error::ServiceStatus`].
    #[test]
    fn new_variants_are_not_service_status() {
        for err in [
            Error::service_invalid_auth("x"),
            Error::service_invalid_request("x"),
            Error::service_content_filter("x"),
        ] {
            assert_eq!(err.status(), None, "{err:?}");
            assert_eq!(err.retry_after(), None, "{err:?}");
        }
    }
}

/// Why a content filter refused a request, as the provider reported it.
///
/// Azure OpenAI answers a filtered request with a nested `innererror` naming
/// the policy that fired and a per-category breakdown; without it a caller
/// holds only a message string and cannot tell a self-harm block from a
/// jailbreak detection, or a prompt block from a completion block — which is
/// the whole of what an application does with this error (re-prompt, escalate,
/// log the category).
///
/// Mirrors upstream's `OpenAIContentFilterException` fields. Every field here
/// is an **open** value rather than an enum: upstream parses the code and the
/// severity into `Enum(...)`, which raises on a value Azure has not shipped
/// yet — the bug its #8393 fixed for the code and still has for the severity.
/// A string cannot have that failure mode, so a new Azure category or
/// severity arrives as itself rather than as an error while classifying an
/// error.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContentFilterDetail {
    /// The provider's inner error code, verbatim — e.g.
    /// [`Self::RESPONSIBLE_AI_POLICY_VIOLATION`] or
    /// [`Self::CONTENT_FILTERED`].
    pub code: Option<String>,
    /// Which request parameter tripped the filter, when the provider says —
    /// `"prompt"` for input, absent for a filtered completion. The one bit
    /// that tells a caller whether to change the request or the expectation.
    pub param: Option<String>,
    /// Per-category results, keyed by the provider's category name
    /// (`hate`, `self_harm`, `sexual`, `violence`, `jailbreak`,
    /// `profanity`, ...). A `BTreeMap` so a rendered error is stable across
    /// runs.
    pub categories: std::collections::BTreeMap<String, ContentFilterCategory>,
}

impl ContentFilterDetail {
    /// Azure's code for a Responsible AI policy block.
    pub const RESPONSIBLE_AI_POLICY_VIOLATION: &'static str = "ResponsibleAIPolicyViolation";
    /// Azure's code for content removed by the filter.
    pub const CONTENT_FILTERED: &'static str = "ContentFiltered";

    /// The names of the categories that actually fired, in name order.
    ///
    /// A filtered request reports every category it evaluated, most of them
    /// `filtered: false`; this is the subset a caller is asking about when
    /// they ask "why".
    pub fn filtered_categories(&self) -> Vec<&str> {
        self.categories
            .iter()
            .filter(|(_, r)| r.filtered || r.detected == Some(true))
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

/// One content-filter category's verdict.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContentFilterCategory {
    /// Whether this category caused content to be filtered.
    pub filtered: bool,
    /// Whether this category was *detected* without necessarily filtering —
    /// reported for the detection-only categories (jailbreak, protected
    /// material) rather than the severity-scored ones.
    pub detected: Option<bool>,
    /// The severity the provider assigned (`safe`, `low`, `medium`, `high`),
    /// verbatim. `None` for a detection-only category, which carries none.
    pub severity: Option<String>,
}
