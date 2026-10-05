//! The AGENT-HOOKS-0.1 control contract: wire types, the per-session
//! [`AgentContextBuilder`], and the host-side [`InterceptionEmitter`].
//!
//! Rust equivalent of the `agent-hooks-sdk` package (`agent_hooks`) that the
//! upstream Python feature imports and of the `ResponsibleAI.AgentHooks`
//! NuGet package the .NET feature references. Upstream treats that SDK as an
//! external dependency (its canonical core is a Rust crate that is not
//! published to crates.io), so this port carries a self-contained
//! implementation of the slice of it the enforcement needs:
//!
//! - the closed enums ([`InterceptionPoint`], [`Decision`],
//!   [`EnforcementMode`], [`HostError`]) and the verdict wire types
//!   ([`Verdict`], [`Transform`], [`Evidence`], [`VerdictWarning`]),
//! - the four composition profiles ([`CompositionConfig`]) with
//!   severity-max aggregation and the §7.3 metadata unions,
//! - the approval seam ([`ApprovalResolver`]) with the §9 echo rule,
//! - the identity seam ([`IdentityProvider`]: `jcs-sha256` over RFC 8785
//!   canonical JSON, a custom function, or identity-unbound),
//! - `$target` transform paths with L1 write-back (§4.3, §5.2), and
//! - the payload-free [`InterceptionRecord`] (§10.3).
//!
//! Behaviour was matched against `agent-hooks-sdk` 0.1.0b1 (the Python
//! wrapper over the canonical core) case by case: verdict-gate messages,
//! aggregation winners, record projection (128/256-character message caps,
//! dropped `transform.value`), envelope validation, and post-fold I-JSON
//! domain rejection.
//!
//! Deliberate divergences from the SDK:
//!
//! - **Interceptors and resolvers are async traits returning
//!   [`Result`]**: an `Err` (or a panic) maps to
//!   `host_error:interceptor_failed` / `host_error:approval_resolver_failed`
//!   exactly like a raised exception upstream, with the error's type tag as
//!   the message. The returned [`Verdict`] crosses the same §5 gate the SDK
//!   applies to wire-shaped returns ([`Verdict::validate`]); wire-shaped
//!   verdicts can be decoded with [`Verdict::from_wire`].
//! - **Every interceptor future is preemptible** by the timeout (upstream
//!   can only preempt awaitable returns).
//! - **Cancellation is drop-based**: dropping an in-flight emission records
//!   nothing (upstream appends a fail-closed `CancelledError` record before
//!   re-raising). The guarded action never proceeds either way.
//! - `InterceptionSuspended` (deferred out-of-band approval) is not ported:
//!   resolvers resolve synchronously from the emitter's point of view.

use std::collections::BTreeMap;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// Spec version this implementation speaks (§4.1 `spec` field). Mirrors
/// `agent_hooks.SPEC_VERSION`.
pub const SPEC_VERSION: &str = "agent-hooks/0.1";

/// Name of the default identity provider (§10.1, §10.2). Mirrors
/// `agent_hooks.JCS_SHA256`.
pub const JCS_SHA256: &str = "jcs-sha256";

/// The §7 RECOMMENDED interceptor/resolver timeout (5 s). Mirrors the SDK's
/// `DEFAULT_TIMEOUT` and upstream's `_DEFAULT_TIMEOUT`.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// The reserved prefix of host-synthesized verdict reasons (§11).
pub const HOST_ERROR_PREFIX: &str = "host_error:";

/// The §5.3 evidence size cap, in canonical-JSON bytes.
const EVIDENCE_CAP_BYTES: usize = 10_240;
/// The §10.3 record projection caps the verdict message to this many
/// characters (plus an ellipsis).
const RECORD_MESSAGE_CAP: usize = 128;
/// The §10.3 record projection caps warning messages to this many characters.
const RECORD_WARNING_MESSAGE_CAP: usize = 256;
/// Largest integer magnitude inside the I-JSON domain (§4.4): 2^53 − 1.
const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

// region Enums

/// The closed set of agent lifecycle interception points (§3). Mirrors
/// `agent_hooks.InterceptionPoint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterceptionPoint {
    /// The agent session starts (`agent_init` L1 block).
    AgentStartup,
    /// Run input arrives (`input` L1 block).
    Input,
    /// Before a model service call (`messages` L1 block).
    PreModelCall,
    /// After a model service call (`response` L1 block).
    PostModelCall,
    /// Before a tool executes (`tool_call` L1 block).
    PreToolCall,
    /// After a tool executed (`tool_call` + `tool_result` L1 blocks).
    PostToolCall,
    /// The run's output egresses (`output` L1 block).
    Output,
    /// The agent session ends (`summary` L1 block).
    AgentShutdown,
}

impl InterceptionPoint {
    /// All eight points, in lifecycle order.
    pub const ALL: [InterceptionPoint; 8] = [
        InterceptionPoint::AgentStartup,
        InterceptionPoint::Input,
        InterceptionPoint::PreModelCall,
        InterceptionPoint::PostModelCall,
        InterceptionPoint::PreToolCall,
        InterceptionPoint::PostToolCall,
        InterceptionPoint::Output,
        InterceptionPoint::AgentShutdown,
    ];

    /// The wire value (e.g. `"pre_tool_call"`).
    pub fn as_str(self) -> &'static str {
        match self {
            InterceptionPoint::AgentStartup => "agent_startup",
            InterceptionPoint::Input => "input",
            InterceptionPoint::PreModelCall => "pre_model_call",
            InterceptionPoint::PostModelCall => "post_model_call",
            InterceptionPoint::PreToolCall => "pre_tool_call",
            InterceptionPoint::PostToolCall => "post_tool_call",
            InterceptionPoint::Output => "output",
            InterceptionPoint::AgentShutdown => "agent_shutdown",
        }
    }

    /// Parse a wire value; `None` for anything outside the closed set.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.as_str() == value)
    }

    /// Whether a `transform` verdict is permitted at this point (§3, §4.3):
    /// everywhere except the session boundaries.
    pub fn transform_permitted(self) -> bool {
        !matches!(
            self,
            InterceptionPoint::AgentStartup | InterceptionPoint::AgentShutdown
        )
    }

    /// Whether this is a pre-action point (`pre_model_call`, `pre_tool_call`).
    pub fn is_pre(self) -> bool {
        matches!(
            self,
            InterceptionPoint::PreModelCall | InterceptionPoint::PreToolCall
        )
    }

    /// Whether this is a post-action point (`post_model_call`, `post_tool_call`).
    pub fn is_post(self) -> bool {
        matches!(
            self,
            InterceptionPoint::PostModelCall | InterceptionPoint::PostToolCall
        )
    }

    /// The L1 location that aliases `target` at this point (§4.3): the
    /// write-back destination of an applied transform.
    fn l1_alias(self) -> &'static [&'static str] {
        match self {
            InterceptionPoint::AgentStartup => &["agent_init"],
            InterceptionPoint::Input => &["input"],
            InterceptionPoint::PreModelCall => &["messages"],
            InterceptionPoint::PostModelCall => &["response"],
            InterceptionPoint::PreToolCall => &["tool_call", "args"],
            InterceptionPoint::PostToolCall => &["tool_result", "value"],
            InterceptionPoint::Output => &["output"],
            InterceptionPoint::AgentShutdown => &["summary"],
        }
    }
}

impl fmt::Display for InterceptionPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Verdict decision values (§5.1). Three, closed: `warn` is `allow` +
/// `warnings[]`, `escalate` is `deny` + an `approval` block. Mirrors
/// `agent_hooks.Decision`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Decision {
    /// The action proceeds unchanged.
    Allow,
    /// The action is blocked.
    Deny,
    /// The action proceeds with a rewritten target.
    Transform,
}

impl Decision {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            Decision::Allow => "allow",
            Decision::Deny => "deny",
            Decision::Transform => "transform",
        }
    }

    /// Parse a wire value.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "allow" => Some(Decision::Allow),
            "deny" => Some(Decision::Deny),
            "transform" => Some(Decision::Transform),
            _ => None,
        }
    }

    /// Whether the action proceeds under this decision (§2).
    pub fn permits(self) -> bool {
        matches!(self, Decision::Allow | Decision::Transform)
    }

    /// Whether the action is blocked under this decision.
    pub fn blocks(self) -> bool {
        !self.permits()
    }
}

/// Whether the host acts on verdicts (§8). Mirrors
/// `agent_hooks.EnforcementMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EnforcementMode {
    /// Verdicts are honoured (the default).
    #[default]
    Enforce,
    /// Verdicts are recorded but never act: every emission proceeds and
    /// transforms are validated, not applied.
    EvaluateOnly,
}

impl EnforcementMode {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            EnforcementMode::Enforce => "enforce",
            EnforcementMode::EvaluateOnly => "evaluate_only",
        }
    }
}

impl FromStr for EnforcementMode {
    type Err = Error;

    /// Parse `"enforce"` / `"evaluate_only"` (upstream accepts the string
    /// form for its `mode` argument).
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "enforce" => Ok(EnforcementMode::Enforce),
            "evaluate_only" => Ok(EnforcementMode::EvaluateOnly),
            other => Err(Error::Configuration(format!(
                "'{other}' is not a valid EnforcementMode (expected 'enforce' or 'evaluate_only')"
            ))),
        }
    }
}

/// Reserved `host_error:*` reasons a host synthesizes (§11). Mirrors
/// `agent_hooks.HostError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostError {
    /// The context failed envelope validation or left the I-JSON domain.
    ContextInvalid,
    /// An interceptor failed (returned an error or panicked).
    InterceptorFailed,
    /// An interceptor exceeded the timeout.
    InterceptorTimeout,
    /// A verdict failed the §5 gate.
    VerdictInvalid,
    /// A transform could not be applied.
    TransformInvalid,
    /// A transform targeted a forbidden root or point.
    TransformTargetForbidden,
    /// Two or more transforms against one snapshot (parallel profiles).
    TransformConflict,
    /// A non-unanimous outcome under `parallel/unanimous`.
    CompositionDisagreement,
    /// The approval resolver failed (error, panic, timeout, redactor fault).
    ApprovalResolverFailed,
    /// The resolver returned `unresolved`.
    ApprovalUnresolved,
    /// The resolution did not echo the request's context identity.
    ApprovalIdentityMismatch,
    /// The adapter cannot support the requested behaviour.
    AdapterUnsupported,
    /// An enforce-mode emission found no registered interceptor.
    NoInterceptor,
    /// Streaming cannot be enforced.
    StreamingUnsupported,
}

impl HostError {
    const ALL: [HostError; 14] = [
        HostError::ContextInvalid,
        HostError::InterceptorFailed,
        HostError::InterceptorTimeout,
        HostError::VerdictInvalid,
        HostError::TransformInvalid,
        HostError::TransformTargetForbidden,
        HostError::TransformConflict,
        HostError::CompositionDisagreement,
        HostError::ApprovalResolverFailed,
        HostError::ApprovalUnresolved,
        HostError::ApprovalIdentityMismatch,
        HostError::AdapterUnsupported,
        HostError::NoInterceptor,
        HostError::StreamingUnsupported,
    ];

    /// The wire value (e.g. `"host_error:interceptor_failed"`).
    pub fn as_str(self) -> &'static str {
        match self {
            HostError::ContextInvalid => "host_error:context_invalid",
            HostError::InterceptorFailed => "host_error:interceptor_failed",
            HostError::InterceptorTimeout => "host_error:interceptor_timeout",
            HostError::VerdictInvalid => "host_error:verdict_invalid",
            HostError::TransformInvalid => "host_error:transform_invalid",
            HostError::TransformTargetForbidden => "host_error:transform_target_forbidden",
            HostError::TransformConflict => "host_error:transform_conflict",
            HostError::CompositionDisagreement => "host_error:composition_disagreement",
            HostError::ApprovalResolverFailed => "host_error:approval_resolver_failed",
            HostError::ApprovalUnresolved => "host_error:approval_unresolved",
            HostError::ApprovalIdentityMismatch => "host_error:approval_identity_mismatch",
            HostError::AdapterUnsupported => "host_error:adapter_unsupported",
            HostError::NoInterceptor => "host_error:no_interceptor",
            HostError::StreamingUnsupported => "host_error:streaming_unsupported",
        }
    }

    /// Parse a wire value.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|e| e.as_str() == value)
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A gate failure: the reserved reason plus its detail message.
type GateError = (HostError, String);

fn gate(err: HostError, detail: impl fmt::Display) -> GateError {
    (err, format!("{}: {detail}", err.as_str()))
}

// endregion

// region Verdict

/// A single `$target`-rooted replacement (§5.2). Mirrors
/// `agent_hooks.Transform`.
///
/// `value` of [`Value::Null`] means "no value" on the wire (the SDK
/// serializes `value` only when non-null). The path root is validated by the
/// §5 gate ([`Verdict::validate`]) rather than at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    /// The `$target`-rooted path (`$policy_target` is accepted as a
    /// deprecated alias).
    pub path: String,
    /// The replacement value.
    pub value: Value,
}

impl Transform {
    /// A transform replacing the value at `path` with `value`.
    pub fn new(path: impl Into<String>, value: Value) -> Self {
        Self {
            path: path.into(),
            value,
        }
    }

    /// Wire shape (`value` omitted when null).
    pub fn to_wire(&self) -> Value {
        if self.value.is_null() {
            json!({ "path": self.path })
        } else {
            json!({ "path": self.path, "value": self.value })
        }
    }
}

/// Opaque pointer to an offline-verifiable artefact (§5.3). Mirrors
/// `agent_hooks.Evidence`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Evidence {
    /// The artefact reference.
    pub artefact: Option<String>,
    /// Named verification pointers.
    pub verification_pointers: BTreeMap<String, String>,
}

impl Evidence {
    /// Wire shape.
    pub fn to_wire(&self) -> Value {
        let mut out = Map::new();
        if let Some(a) = &self.artefact {
            out.insert("artefact".into(), Value::String(a.clone()));
        }
        if !self.verification_pointers.is_empty() {
            out.insert(
                "verification_pointers".into(),
                json!(self.verification_pointers),
            );
        }
        Value::Object(out)
    }
}

/// A recorded concern that does not affect control flow (§5.1). Mirrors
/// `agent_hooks.Warning` (renamed to avoid shadowing the common word).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VerdictWarning {
    /// A machine-readable reason (must not use the reserved prefix).
    pub reason: Option<String>,
    /// A human-readable message.
    pub message: Option<String>,
}

impl VerdictWarning {
    /// Wire shape.
    pub fn to_wire(&self) -> Value {
        let mut out = Map::new();
        if let Some(r) = &self.reason {
            out.insert("reason".into(), Value::String(r.clone()));
        }
        if let Some(m) = &self.message {
            out.insert("message".into(), Value::String(m.clone()));
        }
        Value::Object(out)
    }
}

/// Interceptor return value (§5). Mirrors `agent_hooks.Verdict`.
///
/// Fields are public so any shape can be constructed; the emitter runs every
/// verdict through the §5 gate ([`Verdict::validate`]) and maps a violation
/// to `host_error:verdict_invalid` (or `transform_target_forbidden` for a
/// bad transform root), exactly as the SDK does for wire-shaped returns.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// The decision.
    pub decision: Decision,
    /// A machine-readable reason.
    pub reason: Option<String>,
    /// A human-readable message.
    pub message: Option<String>,
    /// Recorded concerns; permitted on any decision.
    pub warnings: Vec<VerdictWarning>,
    /// Present only on `deny`: marks the deny as liftable by the approval
    /// seam (§9). May be empty.
    pub approval: Option<Map<String, Value>>,
    /// Required iff `decision == Transform`.
    pub transform: Option<Transform>,
    /// Optional evidence pointer.
    pub evidence: Option<Evidence>,
    /// Labels attached to a proceeding result (§5.4).
    pub result_labels: Vec<String>,
}

impl Verdict {
    fn bare(decision: Decision) -> Self {
        Self {
            decision,
            reason: None,
            message: None,
            warnings: Vec::new(),
            approval: None,
            transform: None,
            evidence: None,
            result_labels: Vec::new(),
        }
    }

    /// The trivial permit verdict (`agent_hooks.ALLOW`).
    pub fn allow() -> Self {
        Self::bare(Decision::Allow)
    }

    /// An `allow` carrying one warning (`Verdict.warn`).
    pub fn warn(reason: impl Into<String>) -> Self {
        let mut v = Self::allow();
        v.warnings.push(VerdictWarning {
            reason: Some(reason.into()),
            message: None,
        });
        v
    }

    /// A plain, final deny (`Verdict.deny`): no approval block, so the
    /// approval seam cannot lift it.
    pub fn deny(reason: impl Into<String>) -> Self {
        let mut v = Self::bare(Decision::Deny);
        v.reason = Some(reason.into());
        v
    }

    /// A liftable deny (`Verdict.escalate`): denied as-is unless the approval
    /// seam lifts it (§5.1, §9).
    pub fn escalate(reason: impl Into<String>) -> Self {
        let mut v = Self::deny(reason);
        v.approval = Some(Map::new());
        v
    }

    /// A `transform` verdict replacing the value at `path`.
    pub fn transform(path: impl Into<String>, value: Value) -> Self {
        let mut v = Self::bare(Decision::Transform);
        v.transform = Some(Transform::new(path, value));
        v
    }

    /// Host-synthesized deny for a §11 failure (`Verdict.host_error`). The
    /// reserved reason bypasses the interceptor-side gate; `liftable` is the
    /// §7.5 `"approval"` knob value.
    pub fn host_error(err: HostError, message: Option<String>, liftable: bool) -> Self {
        let mut v = Self::bare(Decision::Deny);
        v.reason = Some(err.as_str().to_string());
        v.message = message;
        if liftable {
            v.approval = Some(Map::new());
        }
        v
    }

    /// Builder: set the message.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Builder: set the reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Builder: add a warning.
    pub fn with_warning(mut self, reason: Option<String>, message: Option<String>) -> Self {
        self.warnings.push(VerdictWarning { reason, message });
        self
    }

    /// Builder: add a result label.
    pub fn with_result_label(mut self, label: impl Into<String>) -> Self {
        self.result_labels.push(label.into());
        self
    }

    /// Builder: attach evidence.
    pub fn with_evidence(mut self, evidence: Evidence) -> Self {
        self.evidence = Some(evidence);
        self
    }

    /// A deny carrying an approval block (§5.1).
    pub fn is_liftable(&self) -> bool {
        self.decision == Decision::Deny && self.approval.is_some()
    }

    /// Whether the reason carries the reserved `host_error:` prefix (the
    /// verdict was synthesized by the host).
    pub fn is_host_error(&self) -> bool {
        self.reason
            .as_deref()
            .is_some_and(|r| r.starts_with(HOST_ERROR_PREFIX))
    }

    /// Wire shape (optional members omitted when absent/empty).
    pub fn to_wire(&self) -> Value {
        let mut out = Map::new();
        out.insert("decision".into(), json!(self.decision.as_str()));
        if let Some(r) = &self.reason {
            out.insert("reason".into(), json!(r));
        }
        if let Some(m) = &self.message {
            out.insert("message".into(), json!(m));
        }
        if !self.warnings.is_empty() {
            out.insert(
                "warnings".into(),
                Value::Array(self.warnings.iter().map(VerdictWarning::to_wire).collect()),
            );
        }
        if let Some(a) = &self.approval {
            out.insert("approval".into(), Value::Object(a.clone()));
        }
        if let Some(t) = &self.transform {
            out.insert("transform".into(), t.to_wire());
        }
        if let Some(e) = &self.evidence {
            out.insert("evidence".into(), e.to_wire());
        }
        if !self.result_labels.is_empty() {
            out.insert("result_labels".into(), json!(self.result_labels));
        }
        Value::Object(out)
    }

    /// Decode and validate a wire-shaped verdict (`Verdict.from_wire` + the
    /// core's §5 gate). The error message carries the `host_error:*` code the
    /// emitter would record.
    pub fn from_wire(value: &Value) -> Result<Self> {
        Self::decode(value)
            .and_then(|v| v.validate().map(|()| v))
            .map_err(|(_, msg)| Error::Content(msg))
    }

    fn decode(value: &Value) -> std::result::Result<Self, GateError> {
        let invalid = |detail: &str| gate(HostError::VerdictInvalid, detail);
        let obj = value
            .as_object()
            .ok_or_else(|| invalid("verdict must be a JSON object"))?;
        let raw_decision = obj.get("decision");
        let decision = raw_decision
            .and_then(Value::as_str)
            .and_then(Decision::parse)
            .ok_or_else(|| {
                invalid(&format!(
                    "verdict.decision invalid: {} (§5.1: allow|deny|transform)",
                    raw_decision.map(Value::to_string).unwrap_or("None".into())
                ))
            })?;
        let opt_str = |key: &str| -> std::result::Result<Option<String>, GateError> {
            match obj.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(s)) => Ok(Some(s.clone())),
                Some(_) => Err(invalid(&format!("verdict.{key} must be string or null"))),
            }
        };
        let reason = opt_str("reason")?;
        let message = opt_str("message")?;
        let warnings = match obj.get("warnings") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|w| {
                    let w = w
                        .as_object()
                        .ok_or_else(|| invalid("warnings must be an array of objects (§5)"))?;
                    let field = |k: &str| match w.get(k) {
                        None | Some(Value::Null) => Ok(None),
                        Some(Value::String(s)) => Ok(Some(s.clone())),
                        Some(_) => Err(invalid(&format!("warnings[].{k} must be string or null"))),
                    };
                    Ok(VerdictWarning {
                        reason: field("reason")?,
                        message: field("message")?,
                    })
                })
                .collect::<std::result::Result<_, GateError>>()?,
            Some(_) => return Err(invalid("warnings must be an array of objects (§5)")),
        };
        let approval = match obj.get("approval") {
            None | Some(Value::Null) => None,
            Some(Value::Object(a)) => Some(a.clone()),
            Some(_) => return Err(invalid("verdict.approval must be an object (§5)")),
        };
        let transform = match obj.get("transform") {
            None | Some(Value::Null) => None,
            Some(Value::Object(t)) => {
                let path = t
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("verdict.transform must be {path, value}"))?;
                Some(Transform::new(
                    path,
                    t.get("value").cloned().unwrap_or(Value::Null),
                ))
            }
            Some(_) => return Err(invalid("verdict.transform must be {path, value}")),
        };
        let evidence = match obj.get("evidence") {
            None | Some(Value::Null) => None,
            Some(Value::Object(e)) => Some(Evidence {
                artefact: e
                    .get("artefact")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                verification_pointers: e
                    .get("verification_pointers")
                    .and_then(Value::as_object)
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                            .collect()
                    })
                    .unwrap_or_default(),
            }),
            Some(_) => return Err(invalid("verdict.evidence must be an object")),
        };
        let result_labels = match obj.get("result_labels") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) if items.iter().all(Value::is_string) => items
                .iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect(),
            Some(_) => return Err(invalid("verdict.result_labels must be an array of strings")),
        };
        Ok(Self {
            decision,
            reason,
            message,
            warnings,
            approval,
            transform,
            evidence,
            result_labels,
        })
    }

    /// The §5 gate an interceptor's (or resolver's) verdict must pass.
    ///
    /// Rejects a reserved `host_error:` reason, an approval block on a
    /// non-deny, a missing/forbidden transform body, a reserved warning
    /// reason, a transform path not rooted at `$target`, and evidence beyond
    /// 10240 canonical bytes. Mirrors the core's `validate_verdict`.
    pub fn validate(&self) -> std::result::Result<(), (HostError, String)> {
        let invalid = |detail: &str| gate(HostError::VerdictInvalid, detail);
        if self.is_host_error() {
            return Err(invalid(
                "verdict.reason MUST NOT start with 'host_error:' (§5)",
            ));
        }
        if self.approval.is_some() && self.decision != Decision::Deny {
            return Err(invalid("approval block permitted only on deny (§5.1)"));
        }
        match (&self.transform, self.decision) {
            (None, Decision::Transform) => {
                return Err(invalid(
                    "transform body REQUIRED when decision=='transform' (§5)",
                ))
            }
            (Some(_), d) if d != Decision::Transform => {
                return Err(invalid(
                    "transform body FORBIDDEN when decision!='transform' (§5)",
                ))
            }
            _ => {}
        }
        if self.warnings.iter().any(|w| {
            w.reason
                .as_deref()
                .is_some_and(|r| r.starts_with(HOST_ERROR_PREFIX))
        }) {
            return Err(invalid("warnings[].reason must be a non-reserved string"));
        }
        if let Some(t) = &self.transform {
            if parse_path(&t.path).is_err_and(|(e, _)| e == HostError::TransformTargetForbidden) {
                return Err(gate(
                    HostError::TransformTargetForbidden,
                    format!(
                        "transform.path must be rooted at $target (got {:?})",
                        t.path
                    ),
                ));
            }
        }
        if let Some(e) = &self.evidence {
            let size = canonical_json(&e.to_wire()).len();
            if size > EVIDENCE_CAP_BYTES {
                return Err(invalid(&format!(
                    "evidence canonical size {size} exceeds {EVIDENCE_CAP_BYTES} bytes (§5.3)"
                )));
            }
        }
        Ok(())
    }

    /// The §10.3 record projection: `transform.value` dropped, the message
    /// capped at 128 characters and warning messages at 256 (ellipsized).
    fn record_projection(&self) -> Self {
        let mut v = self.clone();
        if let Some(t) = &mut v.transform {
            t.value = Value::Null;
        }
        v.message = v.message.map(|m| cap_chars(m, RECORD_MESSAGE_CAP));
        for w in &mut v.warnings {
            w.message = w
                .message
                .take()
                .map(|m| cap_chars(m, RECORD_WARNING_MESSAGE_CAP));
        }
        v
    }
}

fn cap_chars(text: String, cap: usize) -> String {
    if text.chars().count() <= cap {
        text
    } else {
        let mut out: String = text.chars().take(cap).collect();
        out.push('\u{2026}');
        out
    }
}

// endregion

// region Composition

/// The closed profile set (§7.2). Mirrors `agent_hooks.CompositionProfile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompositionProfile {
    /// Fold-through; the first deny short-circuits.
    SequentialFirstDeny,
    /// Fold-through; every interceptor runs; severity-max aggregate.
    SequentialRunAll,
    /// Isolated snapshots; severity-max aggregate; conflicting transforms
    /// are a host error.
    ParallelStrictest,
    /// Isolated snapshots; any non-allow is a disagreement.
    ParallelUnanimous,
}

impl CompositionProfile {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            CompositionProfile::SequentialFirstDeny => "sequential/first_deny",
            CompositionProfile::SequentialRunAll => "sequential/run_all",
            CompositionProfile::ParallelStrictest => "parallel/strictest",
            CompositionProfile::ParallelUnanimous => "parallel/unanimous",
        }
    }

    /// Whether interceptors observe predecessors' transforms (§7.4) rather
    /// than isolated snapshots (§7.5).
    pub fn is_sequential(self) -> bool {
        matches!(
            self,
            CompositionProfile::SequentialFirstDeny | CompositionProfile::SequentialRunAll
        )
    }
}

/// `sequential/first_deny` knob (§7.4): what a permit resolution does to the
/// rest of the fold. Mirrors `agent_hooks.OnApproval`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnApproval {
    /// The resolution becomes the combined verdict; the emission ends.
    Stop,
    /// The resolution substitutes and the fold continues.
    Resume,
}

impl OnApproval {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            OnApproval::Stop => "stop",
            OnApproval::Resume => "resume",
        }
    }
}

/// `"deny" | "approval"` knob value (§7.5): synthesize a plain deny, or a
/// liftable one and consult the seam. Mirrors `agent_hooks.SynthesisPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthesisPolicy {
    /// Synthesize a plain deny.
    Deny,
    /// Synthesize a liftable deny and consult the approval seam.
    Approval,
}

impl SynthesisPolicy {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            SynthesisPolicy::Deny => "deny",
            SynthesisPolicy::Approval => "approval",
        }
    }
}

/// The composition profile and knobs in effect for one emission (§7.1,
/// §10.3). Mirrors `agent_hooks.CompositionConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompositionConfig {
    /// The profile.
    pub profile: CompositionProfile,
    /// `sequential/first_deny` only.
    pub on_approval: Option<OnApproval>,
    /// `parallel/unanimous` only.
    pub on_disagreement: Option<SynthesisPolicy>,
    /// Parallel profiles only.
    pub on_transform_conflict: Option<SynthesisPolicy>,
}

impl Default for CompositionConfig {
    /// `sequential/first_deny` with `on_approval: stop` (the SDK default).
    fn default() -> Self {
        Self::first_deny(OnApproval::Stop)
    }
}

impl CompositionConfig {
    /// `sequential/first_deny` with the given approval knob.
    pub fn first_deny(on_approval: OnApproval) -> Self {
        Self {
            profile: CompositionProfile::SequentialFirstDeny,
            on_approval: Some(on_approval),
            on_disagreement: None,
            on_transform_conflict: None,
        }
    }

    /// `sequential/run_all`.
    pub fn run_all() -> Self {
        Self {
            profile: CompositionProfile::SequentialRunAll,
            on_approval: None,
            on_disagreement: None,
            on_transform_conflict: None,
        }
    }

    /// `parallel/strictest` with the given transform-conflict policy.
    pub fn strictest(on_transform_conflict: SynthesisPolicy) -> Self {
        Self {
            profile: CompositionProfile::ParallelStrictest,
            on_approval: None,
            on_disagreement: None,
            on_transform_conflict: Some(on_transform_conflict),
        }
    }

    /// `parallel/unanimous` with the given policies.
    pub fn unanimous(
        on_disagreement: SynthesisPolicy,
        on_transform_conflict: SynthesisPolicy,
    ) -> Self {
        Self {
            profile: CompositionProfile::ParallelUnanimous,
            on_approval: None,
            on_disagreement: Some(on_disagreement),
            on_transform_conflict: Some(on_transform_conflict),
        }
    }

    /// Wire shape (the record's `composition` block). A `first_deny` config
    /// without an explicit knob records the effective `on_approval: stop`.
    pub fn to_wire(&self) -> Value {
        let mut out = Map::new();
        out.insert("profile".into(), json!(self.profile.as_str()));
        let on_approval = match (self.profile, self.on_approval) {
            (CompositionProfile::SequentialFirstDeny, None) => Some(OnApproval::Stop),
            (_, knob) => knob,
        };
        if let Some(k) = on_approval {
            out.insert("on_approval".into(), json!(k.as_str()));
        }
        if let Some(k) = self.on_disagreement {
            out.insert("on_disagreement".into(), json!(k.as_str()));
        }
        if let Some(k) = self.on_transform_conflict {
            out.insert("on_transform_conflict".into(), json!(k.as_str()));
        }
        Value::Object(out)
    }
}

// endregion

// region Record

/// Payload-free per-interceptor summary on the record (§10.3). Mirrors
/// `agent_hooks.VerdictSummary`.
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictSummary {
    /// Registration index.
    pub index: usize,
    /// The interceptor's decision.
    pub decision: Decision,
    /// The interceptor's reason.
    pub reason: Option<String>,
    /// The host-chosen registration name.
    pub name: Option<String>,
}

impl VerdictSummary {
    /// Wire shape.
    pub fn to_wire(&self) -> Value {
        let mut out = Map::new();
        out.insert("index".into(), json!(self.index));
        out.insert("decision".into(), json!(self.decision.as_str()));
        if let Some(r) = &self.reason {
            out.insert("reason".into(), json!(r));
        }
        if let Some(n) = &self.name {
            out.insert("name".into(), json!(n));
        }
        Value::Object(out)
    }
}

/// Host-side record of one emission (§10.3). Mirrors
/// `agent_hooks.InterceptionRecord`.
///
/// Payload-free by design: the identities bind the record to the exact
/// pre/post-composition context without duplicating the (possibly sensitive)
/// payload into audit storage.
#[derive(Debug, Clone, PartialEq)]
pub struct InterceptionRecord {
    /// The point emitted.
    pub interception_point: InterceptionPoint,
    /// The enforcement mode in effect.
    pub mode: EnforcementMode,
    /// The combined verdict (record projection: `transform.value` dropped,
    /// messages capped).
    pub verdict: Verdict,
    /// Provider output before dispatch; `None` when identity-unbound or the
    /// context was rejected.
    pub input_identity: Option<String>,
    /// Provider output after composition.
    pub enforced_identity: Option<String>,
    /// The declared identity provider; `None` = unbound.
    pub identity_provider: Option<String>,
    /// `ctx.session.id` (empty when unknown).
    pub session_id: String,
    /// `ctx.sequence` (`-1` when unknown).
    pub sequence: i64,
    /// `ctx.timestamp`, when present.
    pub timestamp: Option<String>,
    /// W3C trace correlation echoed from the context's `trace` block.
    pub trace: Option<Map<String, Value>>,
    /// Registration index of the deciding interceptor (§7.6).
    pub decided_by: Option<usize>,
    /// The composition in effect.
    pub composition: CompositionConfig,
    /// Per-interceptor summaries.
    pub verdicts: Vec<VerdictSummary>,
    /// Whether registered interceptors were skipped (sequential profiles).
    pub fold_truncated: Option<bool>,
    /// `"approval"` / `"rejection"` when the approval seam was consulted.
    pub resolved_by: Option<String>,
    /// Interceptors registered at emission time.
    pub interceptors_registered: usize,
}

impl InterceptionRecord {
    /// Whether the guarded action executes (§6, §8): always in
    /// `evaluate_only`, otherwise iff the combined decision permits.
    pub fn proceeds(&self) -> bool {
        self.mode == EnforcementMode::EvaluateOnly || self.verdict.decision.permits()
    }

    /// Wire shape per the interception-record schema.
    pub fn to_wire(&self) -> Value {
        let mut out = Map::new();
        out.insert(
            "interception_point".into(),
            json!(self.interception_point.as_str()),
        );
        out.insert("mode".into(), json!(self.mode.as_str()));
        out.insert("verdict".into(), self.verdict.to_wire());
        out.insert("input_identity".into(), json!(self.input_identity));
        out.insert("enforced_identity".into(), json!(self.enforced_identity));
        out.insert("identity_provider".into(), json!(self.identity_provider));
        out.insert("session_id".into(), json!(self.session_id));
        out.insert("sequence".into(), json!(self.sequence));
        out.insert("decided_by".into(), json!(self.decided_by));
        out.insert("composition".into(), self.composition.to_wire());
        if let Some(ts) = &self.timestamp {
            out.insert("timestamp".into(), json!(ts));
        }
        if let Some(trace) = &self.trace {
            out.insert("trace".into(), Value::Object(trace.clone()));
        }
        if !self.verdicts.is_empty() {
            out.insert(
                "verdicts".into(),
                Value::Array(self.verdicts.iter().map(VerdictSummary::to_wire).collect()),
            );
        }
        if let Some(t) = self.fold_truncated {
            out.insert("fold_truncated".into(), json!(t));
        }
        if let Some(r) = &self.resolved_by {
            out.insert("resolved_by".into(), json!(r));
        }
        out.insert(
            "interceptors_registered".into(),
            json!(self.interceptors_registered),
        );
        Value::Object(out)
    }
}

/// Raised to the host when a verdict blocks the guarded action (§6).
/// Mirrors `agent_hooks.InterceptionBlocked`; surfaces to callers as
/// [`Error::InterceptionBlocked`].
#[derive(Debug, Clone, PartialEq)]
pub struct InterceptionBlocked {
    /// The record of the blocking emission.
    pub record: InterceptionRecord,
}

impl InterceptionBlocked {
    /// The point that blocked.
    pub fn interception_point(&self) -> InterceptionPoint {
        self.record.interception_point
    }
}

impl fmt::Display for InterceptionBlocked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} blocked: {} ({})",
            self.record.interception_point,
            self.record.verdict.decision.as_str(),
            self.record.verdict.reason.as_deref().unwrap_or("no reason")
        )
    }
}

impl std::error::Error for InterceptionBlocked {}

/// Returned by [`InterceptionEmitter::emit`] on a proceeding emission: the
/// record plus the **effective** (post-composition) target the guarded
/// action must consume (§4.3). Mirrors `agent_hooks.EmitOutcome`.
#[derive(Debug, Clone)]
pub struct EmitOutcome {
    /// The emission's record.
    pub record: InterceptionRecord,
    /// The post-composition target.
    pub target: Value,
}

// endregion

// region Protocols

/// A policy hook that receives an agent context and returns a verdict
/// (§7). Mirrors the `agent_hooks.Interceptor` protocol.
///
/// The context is the interceptor's own deep copy: mutating it cannot alter
/// enforcement. An `Err` or a panic maps to `host_error:interceptor_failed`
/// and the configured timeout to `host_error:interceptor_timeout`.
#[async_trait]
pub trait Interceptor: Send + Sync {
    /// Evaluate one interception.
    async fn intercept(&self, context: Value) -> Result<Verdict>;
}

/// [`Interceptor`] adapter for an async closure; see [`interceptor_fn`].
pub struct FnInterceptor<F>(F);

#[async_trait]
impl<F, Fut> Interceptor for FnInterceptor<F>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Verdict>> + Send,
{
    async fn intercept(&self, context: Value) -> Result<Verdict> {
        (self.0)(context).await
    }
}

/// Build an [`Interceptor`] from an async closure.
pub fn interceptor_fn<F, Fut>(f: F) -> FnInterceptor<F>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Verdict>> + Send,
{
    FnInterceptor(f)
}

/// Approval outcome (§9). Mirrors `agent_hooks.ApprovalOutcome`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApprovalOutcome {
    /// Lift the deny with a permit verdict.
    Approve,
    /// Keep the deny (with a deny verdict).
    Reject,
    /// No decision; the deny stands as `host_error:approval_unresolved`.
    Unresolved,
}

/// What the host hands the resolver (§9). Mirrors
/// `agent_hooks.ApprovalRequest`.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// Identity of the presented context; `None` when identity-unbound.
    pub context_identity: Option<String>,
    /// The point under consultation.
    pub interception_point: InterceptionPoint,
    /// The liftable deny being consulted.
    pub verdict: Verdict,
    /// The context as presented (after any approval redaction).
    pub context: Value,
}

/// What the resolver returns (§9). Mirrors `agent_hooks.ApprovalResolution`.
/// `context_identity` must echo the request's byte for byte.
#[derive(Debug, Clone)]
pub struct ApprovalResolution {
    /// The outcome.
    pub outcome: ApprovalOutcome,
    /// The echoed identity.
    pub context_identity: Option<String>,
    /// The substituting verdict (absent for `unresolved`).
    pub verdict: Option<Verdict>,
}

impl ApprovalResolution {
    /// Approve with a permit verdict, echoing `request`'s identity.
    pub fn approve(request: &ApprovalRequest, verdict: Verdict) -> Self {
        Self {
            outcome: ApprovalOutcome::Approve,
            context_identity: request.context_identity.clone(),
            verdict: Some(verdict),
        }
    }

    /// Reject with a deny verdict, echoing `request`'s identity.
    pub fn reject(request: &ApprovalRequest, verdict: Verdict) -> Self {
        Self {
            outcome: ApprovalOutcome::Reject,
            context_identity: request.context_identity.clone(),
            verdict: Some(verdict),
        }
    }

    /// No decision, echoing `request`'s identity.
    pub fn unresolved(request: &ApprovalRequest) -> Self {
        Self {
            outcome: ApprovalOutcome::Unresolved,
            context_identity: request.context_identity.clone(),
            verdict: None,
        }
    }
}

/// Host-registered resolver for liftable denies (§9). Mirrors the
/// `agent_hooks.ApprovalResolver` protocol.
#[async_trait]
pub trait ApprovalResolver: Send + Sync {
    /// Resolve one liftable deny.
    async fn resolve(&self, request: ApprovalRequest) -> Result<ApprovalResolution>;
}

/// [`ApprovalResolver`] adapter for an async closure; see [`resolver_fn`].
pub struct FnApprovalResolver<F>(F);

#[async_trait]
impl<F, Fut> ApprovalResolver for FnApprovalResolver<F>
where
    F: Fn(ApprovalRequest) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<ApprovalResolution>> + Send,
{
    async fn resolve(&self, request: ApprovalRequest) -> Result<ApprovalResolution> {
        (self.0)(request).await
    }
}

/// Build an [`ApprovalResolver`] from an async closure.
pub fn resolver_fn<F, Fut>(f: F) -> FnApprovalResolver<F>
where
    F: Fn(ApprovalRequest) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<ApprovalResolution>> + Send,
{
    FnApprovalResolver(f)
}

/// A pure function computing a context identity.
pub type IdentityFn = Arc<dyn Fn(&Value) -> Result<String> + Send + Sync>;

/// The identity seam (§10.1). Mirrors the SDK's
/// `str | IdentityProvider | None` argument: use `Some(JcsSha256)` (the
/// default), `Some(custom(..))`, or `None` for identity-unbound records.
#[derive(Clone)]
pub enum IdentityProvider {
    /// `"sha256:" + hex(SHA-256(canonical_json(ctx)))` (§10.2).
    JcsSha256,
    /// A host-supplied provider: a name and a pure function.
    Custom {
        /// The provider name recorded on records.
        name: String,
        /// The identity function.
        function: IdentityFn,
    },
}

impl fmt::Debug for IdentityProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl IdentityProvider {
    /// A custom provider. The name must match `^[a-z][a-z0-9_-]*$` and must
    /// not begin with `jcs` (reserved, §10.1).
    pub fn custom<F>(name: impl Into<String>, function: F) -> Result<Self>
    where
        F: Fn(&Value) -> Result<String> + Send + Sync + 'static,
    {
        let name = name.into();
        let mut chars = name.chars();
        let valid = chars.next().is_some_and(|c| c.is_ascii_lowercase())
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
            && !name.starts_with("jcs");
        if !valid {
            return Err(Error::Configuration(
                "identity provider name must match ^[a-z][a-z0-9_-]*$ and must not begin with \
                 'jcs' (§10.1)"
                    .into(),
            ));
        }
        Ok(IdentityProvider::Custom {
            name,
            function: Arc::new(function),
        })
    }

    /// The declared provider name.
    pub fn name(&self) -> &str {
        match self {
            IdentityProvider::JcsSha256 => JCS_SHA256,
            IdentityProvider::Custom { name, .. } => name,
        }
    }

    fn compute(&self, ctx: &Value) -> std::result::Result<String, GateError> {
        match self {
            IdentityProvider::JcsSha256 => {
                check_domain(ctx).map_err(|m| (HostError::ContextInvalid, m))?;
                Ok(context_identity(ctx))
            }
            IdentityProvider::Custom { function, .. } => function(ctx).map_err(|e| {
                (
                    HostError::ContextInvalid,
                    crate::observability::error_type(&e),
                )
            }),
        }
    }
}

/// Per-emission record callback.
pub type RecordSink = Arc<dyn Fn(&InterceptionRecord) + Send + Sync>;

/// The §9/§14 approval redactor: the context placed in every
/// [`ApprovalRequest`].
pub type ApprovalRedactor = Arc<dyn Fn(&Value) -> Result<Value> + Send + Sync>;

// endregion

// region Canonical JSON, identity, paths

/// Serialize per §10.1 / RFC 8785 (JCS): keys sorted by UTF-16 code units,
/// no whitespace, ECMA-262 number formatting, minimal string escapes.
/// Mirrors `agent_hooks.canonical_json`.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&ecma_number(n)),
        Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                write_canonical(&map[key], out);
            }
            out.push('}');
        }
    }
}

/// ECMA-262 `Number::toString` for a JSON number.
fn ecma_number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    let f = n.as_f64().unwrap_or(0.0);
    if f == 0.0 {
        return "0".into();
    }
    let sign = if f < 0.0 { "-" } else { "" };
    // Shortest round-trip digits and exponent, e.g. "1.5e-7".
    let sci = format!("{:e}", f.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i64;
    let n_exp = exp.parse::<i64>().unwrap_or(0) + 1;
    let body = if k <= n_exp && n_exp <= 21 {
        format!("{digits}{}", "0".repeat((n_exp - k) as usize))
    } else if 0 < n_exp && n_exp <= 21 {
        let (a, b) = digits.split_at(n_exp as usize);
        format!("{a}.{b}")
    } else if -6 < n_exp && n_exp <= 0 {
        format!("0.{}{digits}", "0".repeat((-n_exp) as usize))
    } else {
        let e = n_exp - 1;
        let exp_sign = if e < 0 { "-" } else { "+" };
        let (first, rest) = digits.split_at(1);
        if rest.is_empty() {
            format!("{first}e{exp_sign}{}", e.abs())
        } else {
            format!("{first}.{rest}e{exp_sign}{}", e.abs())
        }
    };
    format!("{sign}{body}")
}

/// `"sha256:" + hex(SHA-256(canonical_json(ctx)))` (§10.2). Mirrors
/// `agent_hooks.context_identity` (the I-JSON domain check is applied by
/// the emitter before identities are computed).
pub fn context_identity(ctx: &Value) -> String {
    let digest = Sha256::digest(canonical_json(ctx).as_bytes());
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// §4.4: integers must stay within ±(2^53 − 1).
fn check_domain(value: &Value) -> std::result::Result<(), String> {
    match value {
        Value::Number(n) => {
            let out_of_domain = match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => i.unsigned_abs() > MAX_SAFE_INTEGER,
                (None, Some(u)) => u > MAX_SAFE_INTEGER,
                (None, None) => false,
            };
            if out_of_domain {
                Err(format!(
                    "{}: integer token exceeds 2^53; string-encode 64-bit identifiers, see \
                     spec §4.4",
                    HostError::ContextInvalid.as_str()
                ))
            } else {
                Ok(())
            }
        }
        Value::Array(items) => items.iter().try_for_each(check_domain),
        Value::Object(map) => map.values().try_for_each(check_domain),
        _ => Ok(()),
    }
}

/// §4 envelope validation (the core's `validate_envelope`).
fn validate_envelope(ctx: &Value) -> std::result::Result<(), String> {
    let fail = |detail: &str| Err(format!("{}: {detail}", HostError::ContextInvalid.as_str()));
    let Some(obj) = ctx.as_object() else {
        return fail("$: context must be an object (see spec §4)");
    };
    let spec_ok = obj.get("spec").and_then(Value::as_str).is_some_and(|s| {
        s.strip_prefix("agent-hooks/")
            .and_then(|v| v.split_once('.'))
            .is_some_and(|(maj, min)| {
                !maj.is_empty()
                    && !min.is_empty()
                    && maj.chars().all(|c| c.is_ascii_digit())
                    && min.chars().all(|c| c.is_ascii_digit())
            })
    });
    if !spec_ok {
        return fail("$.spec: missing or not an agent-hooks/<maj>.<min> string (see spec §4)");
    }
    let Some(point) = obj.get("interception_point").and_then(Value::as_str) else {
        return fail("$.interception_point: missing or not a string (see spec §4.1)");
    };
    let Some(point) = InterceptionPoint::parse(point) else {
        return fail("$.interception_point: not one of the eight closed values (see spec §3)");
    };
    if !obj.get("timestamp").is_some_and(Value::is_string) {
        return fail("$.timestamp: missing or not a string (see spec §4)");
    }
    if obj.get("sequence").and_then(Value::as_u64).is_none() {
        return fail("$.sequence: missing or not an integer >= 0 (see spec §4)");
    }
    if !obj.get("agent").is_some_and(Value::is_object) {
        return fail("$.agent: missing or not an object (see spec §4)");
    }
    if !obj
        .get("session")
        .and_then(|s| s.get("id"))
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        return fail("$.session.id: missing or not a non-empty string (see spec §4)");
    }
    if !obj.contains_key("target") {
        return fail("$.target: missing or not present (see spec §4)");
    }
    let l1 = point.l1_alias()[0];
    if !obj.contains_key(l1) {
        return fail(&format!(
            "$.{l1}: missing or not present at this interception point (see spec §4)"
        ));
    }
    check_domain(ctx)
}

#[derive(Debug, Clone, PartialEq)]
enum PathSegment {
    Key(String),
    Index(usize),
}

/// Parse a §5.2 `$target` path into segments.
fn parse_path(path: &str) -> std::result::Result<Vec<PathSegment>, GateError> {
    let rest = path
        .strip_prefix("$policy_target")
        .or_else(|| path.strip_prefix("$target"))
        .ok_or_else(|| gate(HostError::TransformTargetForbidden, path))?;
    let is_member = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    let mut segments = Vec::new();
    let mut rest = rest;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('.') {
            let len = after.find(|c: char| !is_member(c)).unwrap_or(after.len());
            if len == 0 {
                return Err(gate(HostError::TransformInvalid, path));
            }
            segments.push(PathSegment::Key(after[..len].to_string()));
            rest = &after[len..];
        } else if let Some(after) = rest.strip_prefix("[\"") {
            let end = after
                .find("\"]")
                .ok_or_else(|| gate(HostError::TransformInvalid, path))?;
            let key = &after[..end];
            if key.is_empty() || !key.chars().all(is_member) {
                return Err(gate(HostError::TransformInvalid, path));
            }
            segments.push(PathSegment::Key(key.to_string()));
            rest = &after[end + 2..];
        } else if let Some(after) = rest.strip_prefix('[') {
            let end = after
                .find(']')
                .ok_or_else(|| gate(HostError::TransformInvalid, path))?;
            let index = after[..end]
                .parse::<usize>()
                .ok()
                .filter(|_| after[..end].chars().all(|c| c.is_ascii_digit()))
                .ok_or_else(|| gate(HostError::TransformInvalid, path))?;
            segments.push(PathSegment::Index(index));
            rest = &after[end + 1..];
        } else {
            return Err(gate(HostError::TransformInvalid, path));
        }
    }
    Ok(segments)
}

/// Return `target` with the existing value at `path` replaced (§5.2): a path
/// may only replace a slot that already exists.
fn apply_path(target: &Value, path: &str, value: Value) -> std::result::Result<Value, GateError> {
    let segments = parse_path(path)?;
    let mut out = target.clone();
    let mut slot = &mut out;
    for seg in &segments {
        slot = match (seg, slot) {
            (PathSegment::Key(k), Value::Object(map)) => map
                .get_mut(k)
                .ok_or_else(|| gate(HostError::TransformInvalid, path))?,
            (PathSegment::Index(i), Value::Array(items)) => items
                .get_mut(*i)
                .ok_or_else(|| gate(HostError::TransformInvalid, path))?,
            _ => return Err(gate(HostError::TransformInvalid, path)),
        };
    }
    *slot = value;
    Ok(out)
}

/// Apply a transform to a context's `target` and its L1 alias (§4.3) — the
/// core's `apply_transform_ctx`. Returns the new context.
fn apply_transform_ctx(
    ctx: &Value,
    path: &str,
    value: &Value,
) -> std::result::Result<Value, GateError> {
    let point = ctx
        .get("interception_point")
        .and_then(Value::as_str)
        .and_then(InterceptionPoint::parse)
        .ok_or_else(|| gate(HostError::ContextInvalid, "$.interception_point"))?;
    if !point.transform_permitted() {
        return Err(gate(HostError::TransformTargetForbidden, path));
    }
    let target = ctx.get("target").cloned().unwrap_or(Value::Null);
    let new_target = apply_path(&target, path, value.clone())?;
    let mut out = ctx.clone();
    out["target"] = new_target.clone();
    let alias = point.l1_alias();
    let mut slot = &mut out;
    for key in alias {
        if !slot.get(*key).is_some() {
            // No L1 block to mirror into (a hand-built context): target only.
            return Ok(out);
        }
        slot = &mut slot[*key];
    }
    *slot = new_target;
    Ok(out)
}

// endregion

// region Context builder

/// Stateful per-session builder for agent contexts (§4). Mirrors
/// `agent_hooks.AgentContextBuilder`.
///
/// Owns `sequence` (assigned atomically, §12.2.3) and the L0
/// `agent`/`session` envelope. One instance per session; share it via
/// [`Arc`]. Contexts are wire-shaped [`Value`] objects.
#[derive(Debug)]
pub struct AgentContextBuilder {
    agent: Map<String, Value>,
    session: Map<String, Value>,
    sequence: AtomicU64,
    l2: Mutex<Map<String, Value>>,
}

impl AgentContextBuilder {
    /// A builder for one session.
    pub fn new(
        agent_id: impl Into<String>,
        framework: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Self {
        let mut agent = Map::new();
        agent.insert("id".into(), Value::String(agent_id.into()));
        agent.insert("framework".into(), Value::String(framework.into()));
        let mut session = Map::new();
        session.insert("id".into(), Value::String(session_id.into()));
        Self {
            agent,
            session,
            sequence: AtomicU64::new(0),
            l2: Mutex::new(Map::new()),
        }
    }

    /// Builder: the agent's display name (omitted when empty).
    pub fn with_agent_name(mut self, name: Option<String>) -> Self {
        if let Some(n) = name.filter(|n| !n.is_empty()) {
            self.agent.insert("name".into(), Value::String(n));
        }
        self
    }

    /// Builder: the agent's version.
    pub fn with_agent_version(mut self, version: impl Into<String>) -> Self {
        self.agent
            .insert("version".into(), Value::String(version.into()));
        self
    }

    /// Builder: the session start instant.
    pub fn with_session_started_at(mut self, started_at: impl Into<String>) -> Self {
        self.session
            .insert("started_at".into(), Value::String(started_at.into()));
        self
    }

    /// Attach an L2 field (`trace`, `tenant`, `budgets`, `actor`, …) to every
    /// subsequent context; a null value is ignored. Mirrors `with_l2`.
    pub fn with_l2(&self, key: impl Into<String>, value: Value) -> &Self {
        if !value.is_null() {
            self.l2.lock().unwrap().insert(key.into(), value);
        }
        self
    }

    /// The session id.
    pub fn session_id(&self) -> &str {
        self.session.get("id").and_then(Value::as_str).unwrap_or("")
    }

    fn envelope(&self, point: InterceptionPoint, target: Value) -> Map<String, Value> {
        let mut ctx = Map::new();
        ctx.insert("spec".into(), json!(SPEC_VERSION));
        ctx.insert("interception_point".into(), json!(point.as_str()));
        ctx.insert("timestamp".into(), json!(now_rfc3339()));
        ctx.insert(
            "sequence".into(),
            json!(self.sequence.fetch_add(1, Ordering::SeqCst)),
        );
        ctx.insert("agent".into(), Value::Object(self.agent.clone()));
        ctx.insert("session".into(), Value::Object(self.session.clone()));
        ctx.insert("target".into(), target);
        for (k, v) in self.l2.lock().unwrap().iter() {
            ctx.insert(k.clone(), v.clone());
        }
        ctx
    }

    /// `agent_startup` (target = `agent_init`).
    pub fn agent_startup(&self, tools_registered: Vec<String>) -> Value {
        let init = json!({ "tools_registered": tools_registered });
        let mut ctx = self.envelope(InterceptionPoint::AgentStartup, init.clone());
        ctx.insert("agent_init".into(), init);
        Value::Object(ctx)
    }

    /// `input` (target = `input`).
    pub fn input(&self, content: Value, role: &str) -> Value {
        let inp = json!({ "content": content, "role": role });
        let mut ctx = self.envelope(InterceptionPoint::Input, inp.clone());
        ctx.insert("input".into(), inp);
        Value::Object(ctx)
    }

    /// `pre_model_call` (target = `messages`).
    pub fn pre_model_call(
        &self,
        model_id: &str,
        messages: Value,
        tools: Option<Value>,
        request_id: Option<String>,
    ) -> Value {
        let mut ctx = self.envelope(InterceptionPoint::PreModelCall, messages.clone());
        ctx.insert("model".into(), json!({ "id": model_id }));
        ctx.insert("messages".into(), messages);
        if let Some(t) = tools {
            ctx.insert("tools".into(), t);
        }
        if let Some(r) = request_id {
            ctx.insert("request_id".into(), json!(r));
        }
        Value::Object(ctx)
    }

    /// `post_model_call` (target = `response`).
    #[allow(clippy::too_many_arguments)]
    pub fn post_model_call(
        &self,
        model_id: &str,
        content: Value,
        tool_calls: Value,
        finish_reason: &str,
        usage: Option<Value>,
        request_id: Option<String>,
    ) -> Value {
        let response = json!({
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        });
        let mut ctx = self.envelope(InterceptionPoint::PostModelCall, response.clone());
        ctx.insert("model".into(), json!({ "id": model_id }));
        ctx.insert("response".into(), response);
        if let Some(u) = usage {
            ctx.insert("usage".into(), u);
        }
        if let Some(r) = request_id {
            ctx.insert("request_id".into(), json!(r));
        }
        Value::Object(ctx)
    }

    /// `pre_tool_call` (target = the `args` object).
    pub fn pre_tool_call(&self, call_id: &str, name: &str, args: Value) -> Value {
        let mut ctx = self.envelope(InterceptionPoint::PreToolCall, args.clone());
        ctx.insert(
            "tool_call".into(),
            json!({ "id": call_id, "name": name, "args": args }),
        );
        Value::Object(ctx)
    }

    /// `post_tool_call` (target = the result value).
    pub fn post_tool_call(
        &self,
        call_id: &str,
        name: &str,
        args: Value,
        value: Value,
        is_error: bool,
        duration_ms: Option<f64>,
    ) -> Value {
        let mut result = json!({ "value": value, "is_error": is_error });
        if let Some(d) = duration_ms {
            result["duration_ms"] = json!(d);
        }
        let mut ctx = self.envelope(InterceptionPoint::PostToolCall, value);
        ctx.insert(
            "tool_call".into(),
            json!({ "id": call_id, "name": name, "args": args }),
        );
        ctx.insert("tool_result".into(), result);
        Value::Object(ctx)
    }

    /// `output` (target = `output`).
    pub fn output(&self, content: Value) -> Value {
        let out = json!({ "content": content });
        let mut ctx = self.envelope(InterceptionPoint::Output, out.clone());
        ctx.insert("output".into(), out);
        Value::Object(ctx)
    }

    /// `agent_shutdown` (target = `summary`).
    pub fn agent_shutdown(&self, reason: &str) -> Value {
        let summary = json!({ "reason": reason });
        let mut ctx = self.envelope(InterceptionPoint::AgentShutdown, summary.clone());
        ctx.insert("summary".into(), summary);
        Value::Object(ctx)
    }
}

/// The current UTC instant as RFC 3339 with microseconds
/// (`YYYY-MM-DDTHH:MM:SS.ffffffZ`).
fn now_rfc3339() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let micros = now.subsec_micros();
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{micros:06}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

// endregion

// region Emitter

/// Internal result of one profile dispatch.
struct Outcome {
    combined: Verdict,
    decided_by: Option<usize>,
    verdicts: Vec<VerdictSummary>,
    fold_truncated: Option<bool>,
    resolved_by: Option<String>,
}

impl Outcome {
    fn bare(combined: Verdict) -> Self {
        Self {
            combined,
            decided_by: None,
            verdicts: Vec::new(),
            fold_truncated: None,
            resolved_by: None,
        }
    }
}

type Registered = (Option<String>, Arc<dyn Interceptor>);

/// Host-side helper implementing §6–§10: dispatch the context to the
/// interceptors per the declared composition profile, apply the combined
/// verdict, and record the emission. Mirrors
/// `agent_hooks.InterceptionEmitter`.
///
/// Configure it with the builder methods before sharing it (via [`Arc`]).
/// "Parallel" profiles dispatch serially over isolated snapshots (§7.2:
/// parallel names isolation semantics, not scheduling), exactly like the SDK.
pub struct InterceptionEmitter {
    interceptors: Vec<Registered>,
    resolver: Option<Arc<dyn ApprovalResolver>>,
    mode: EnforcementMode,
    timeout: Option<Duration>,
    composition: CompositionConfig,
    identity: Option<IdentityProvider>,
    approval_redactor: Option<ApprovalRedactor>,
    record_sink: Option<RecordSink>,
    max_records: Option<usize>,
    records: Mutex<Vec<InterceptionRecord>>,
    records_dropped: AtomicU64,
}

impl Default for InterceptionEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for InterceptionEmitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterceptionEmitter")
            .field("interceptors", &self.interceptors.len())
            .field("mode", &self.mode)
            .field("composition", &self.composition)
            .field("identity", &self.identity)
            .finish()
    }
}

impl InterceptionEmitter {
    /// An emitter with the SDK defaults: `enforce`, no resolver, 5 s timeout,
    /// `sequential/first_deny` + `on_approval: stop`, `jcs-sha256` identity.
    pub fn new() -> Self {
        Self {
            interceptors: Vec::new(),
            resolver: None,
            mode: EnforcementMode::Enforce,
            timeout: Some(DEFAULT_TIMEOUT),
            composition: CompositionConfig::default(),
            identity: Some(IdentityProvider::JcsSha256),
            approval_redactor: None,
            record_sink: None,
            max_records: None,
            records: Mutex::new(Vec::new()),
            records_dropped: AtomicU64::new(0),
        }
    }

    /// Register an interceptor (`register(interceptor)`).
    pub fn register(self, interceptor: impl Interceptor + 'static) -> Self {
        self.register_arc(None, Arc::new(interceptor))
    }

    /// Register an interceptor with a payload-free name recorded on
    /// `verdicts[].name` (`register(interceptor, name)`).
    pub fn register_named(
        self,
        name: impl Into<String>,
        interceptor: impl Interceptor + 'static,
    ) -> Self {
        self.register_arc(Some(name.into()), Arc::new(interceptor))
    }

    /// Register a shared interceptor with an optional name.
    pub fn register_arc(mut self, name: Option<String>, interceptor: Arc<dyn Interceptor>) -> Self {
        self.interceptors.push((name, interceptor));
        self
    }

    /// Set the enforcement mode.
    pub fn with_mode(mut self, mode: EnforcementMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the approval resolver consulted for liftable denies.
    pub fn with_resolver(mut self, resolver: Arc<dyn ApprovalResolver>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Set the per-interceptor/resolver timeout; `None` disables it.
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Declare the composition profile (`set_composition`).
    pub fn with_composition(mut self, composition: CompositionConfig) -> Self {
        self.composition = composition;
        self
    }

    /// Declare the identity provider (`set_identity_provider`); `None` for
    /// identity-unbound records.
    pub fn with_identity_provider(mut self, provider: Option<IdentityProvider>) -> Self {
        self.identity = provider;
        self
    }

    /// Register the approval redactor (`set_approval_redactor`).
    pub fn with_approval_redactor(mut self, redactor: ApprovalRedactor) -> Self {
        self.approval_redactor = Some(redactor);
        self
    }

    /// Register the per-emission record callback (`set_record_sink`). It runs
    /// synchronously after every emission; a panic in it is swallowed.
    pub fn with_record_sink(mut self, sink: RecordSink) -> Self {
        self.record_sink = Some(sink);
        self
    }

    /// Bound the in-memory record buffer, dropping the oldest when full
    /// (`set_max_records`).
    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = Some(max_records);
        self
    }

    /// The enforcement mode.
    pub fn mode(&self) -> EnforcementMode {
        self.mode
    }

    /// The composition in effect.
    pub fn composition(&self) -> CompositionConfig {
        self.composition
    }

    /// The number of registered interceptors.
    pub fn interceptor_count(&self) -> usize {
        self.interceptors.len()
    }

    /// All records emitted so far, in order (`results`).
    pub fn results(&self) -> Vec<InterceptionRecord> {
        self.records.lock().unwrap().clone()
    }

    /// Drain the record buffer (`take_records`).
    pub fn take_records(&self) -> Vec<InterceptionRecord> {
        std::mem::take(&mut *self.records.lock().unwrap())
    }

    /// Records evicted by the [`with_max_records`](Self::with_max_records)
    /// bound.
    pub fn records_dropped(&self) -> u64 {
        self.records_dropped.load(Ordering::SeqCst)
    }

    /// Run the emission and fail with [`Error::InterceptionBlocked`] when the
    /// guarded action must not proceed (§6). Returns the record plus the
    /// effective (post-composition) target.
    pub async fn emit(&self, ctx: Value) -> Result<EmitOutcome> {
        let (record, ctx) = self.emit_inner(ctx).await;
        if !record.proceeds() {
            return Err(Error::InterceptionBlocked(Box::new(InterceptionBlocked {
                record,
            })));
        }
        let target = ctx.get("target").cloned().unwrap_or(Value::Null);
        Ok(EmitOutcome { record, target })
    }

    /// Run the emission and return the record without failing; the caller
    /// must inspect [`InterceptionRecord::proceeds`] (`emit_unchecked`).
    pub async fn emit_unchecked(&self, ctx: Value) -> InterceptionRecord {
        self.emit_inner(ctx).await.0
    }

    async fn emit_inner(&self, mut ctx: Value) -> (InterceptionRecord, Value) {
        // §10.3: input identity binds to the context BEFORE dispatch.
        let mut input_identity = None;
        let pre = validate_envelope(&ctx)
            .map_err(|m| (HostError::ContextInvalid, m))
            .and_then(|()| match &self.identity {
                Some(p) => p.compute(&ctx).map(Some),
                None => Ok(None),
            });
        let outcome = match pre {
            Ok(identity) => {
                input_identity = identity;
                self.dispatch(&mut ctx).await
            }
            Err((err, msg)) => Outcome::bare(Verdict::host_error(err, Some(msg), false)),
        };
        let record = self.finalize(&ctx, outcome, input_identity);
        (self.append(record), ctx)
    }

    /// §10.3/§11 host projection failure: record the fail-closed
    /// `deny host_error:context_invalid` for an emission whose context the
    /// host could not construct (`record_host_failure`).
    pub fn record_host_failure(
        &self,
        point: InterceptionPoint,
        detail: Option<String>,
        session_id: Option<String>,
        sequence: Option<i64>,
        timestamp: Option<String>,
    ) -> InterceptionRecord {
        let record = InterceptionRecord {
            interception_point: point,
            mode: self.mode,
            verdict: Verdict::host_error(HostError::ContextInvalid, detail, false)
                .record_projection(),
            input_identity: None,
            enforced_identity: None,
            identity_provider: self.identity.as_ref().map(|p| p.name().to_string()),
            session_id: session_id.unwrap_or_default(),
            sequence: sequence.unwrap_or(-1),
            timestamp,
            trace: None,
            decided_by: None,
            composition: self.composition,
            verdicts: Vec::new(),
            fold_truncated: None,
            resolved_by: None,
            interceptors_registered: self.interceptors.len(),
        };
        self.append(record)
    }

    fn append(&self, record: InterceptionRecord) -> InterceptionRecord {
        if let Some(sink) = &self.record_sink {
            // Audit delivery must not take down the control plane (§10.3).
            let _ = std::panic::catch_unwind(AssertUnwindSafe(|| sink(&record)));
        }
        let mut records = self.records.lock().unwrap();
        if let Some(max) = self.max_records {
            while records.len() >= max.max(1) {
                records.remove(0);
                self.records_dropped.fetch_add(1, Ordering::SeqCst);
            }
        }
        records.push(record.clone());
        record
    }

    /// Build the §10.3 record (the core's `finalize`).
    fn finalize(
        &self,
        ctx: &Value,
        mut outcome: Outcome,
        input_identity: Option<String>,
    ) -> InterceptionRecord {
        let mut enforced_identity = None;
        if let (Some(provider), Some(_)) = (&self.identity, &input_identity) {
            match provider {
                IdentityProvider::JcsSha256 => {
                    if check_domain(ctx).is_err() {
                        // A fold left the I-JSON domain: fail closed.
                        outcome.combined = Verdict::host_error(
                            HostError::ContextInvalid,
                            Some(
                                "post-fold context left the I-JSON domain; string-encode 64-bit \
                                 identifiers, see spec §4.4"
                                    .into(),
                            ),
                            false,
                        );
                        outcome.decided_by = None;
                    } else {
                        enforced_identity = Some(context_identity(ctx));
                    }
                }
                IdentityProvider::Custom { function, .. } => {
                    enforced_identity = function(ctx).ok();
                }
            }
        }
        let point = ctx
            .get("interception_point")
            .and_then(Value::as_str)
            .and_then(InterceptionPoint::parse)
            .unwrap_or(InterceptionPoint::AgentStartup);
        InterceptionRecord {
            interception_point: point,
            mode: self.mode,
            verdict: outcome.combined.record_projection(),
            input_identity: input_identity.clone(),
            enforced_identity,
            identity_provider: self.identity.as_ref().map(|p| p.name().to_string()),
            session_id: ctx
                .get("session")
                .and_then(|s| s.get("id"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            sequence: ctx.get("sequence").and_then(Value::as_i64).unwrap_or(-1),
            timestamp: ctx
                .get("timestamp")
                .and_then(Value::as_str)
                .map(str::to_string),
            trace: ctx.get("trace").and_then(Value::as_object).map(|t| {
                t.iter()
                    .filter(|(k, _)| *k == "trace_id" || *k == "span_id")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            }),
            decided_by: outcome.decided_by,
            composition: self.composition,
            verdicts: outcome.verdicts,
            fold_truncated: outcome.fold_truncated,
            resolved_by: outcome.resolved_by,
            interceptors_registered: self.interceptors.len(),
        }
    }

    async fn dispatch(&self, ctx: &mut Value) -> Outcome {
        if self.interceptors.is_empty() {
            // §7: zero interceptors fails closed, profile-independent.
            return Outcome::bare(Verdict::host_error(HostError::NoInterceptor, None, false));
        }
        match self.composition.profile {
            CompositionProfile::SequentialFirstDeny => self.dispatch_first_deny(ctx).await,
            CompositionProfile::SequentialRunAll => self.dispatch_run_all(ctx).await,
            _ => self.dispatch_parallel(ctx).await,
        }
    }

    /// Invoke one interceptor on its own copy of the context and run the §5
    /// gate; every failure maps to a host-synthesized deny (§6.3).
    async fn invoke(&self, interceptor: &Arc<dyn Interceptor>, ctx: &Value) -> Verdict {
        let fut = AssertUnwindSafe(interceptor.intercept(ctx.clone())).catch_unwind();
        let raw = match self.timeout {
            Some(t) => match tokio::time::timeout(t, fut).await {
                Ok(r) => r,
                Err(_) => return Verdict::host_error(HostError::InterceptorTimeout, None, false),
            },
            None => fut.await,
        };
        match raw {
            Err(_panic) => {
                Verdict::host_error(HostError::InterceptorFailed, Some("panic".into()), false)
            }
            Ok(Err(e)) => Verdict::host_error(
                HostError::InterceptorFailed,
                Some(crate::observability::error_type(&e)),
                false,
            ),
            Ok(Ok(v)) => match v.validate() {
                Ok(()) => v,
                Err((err, msg)) => Verdict::host_error(err, Some(msg), false),
            },
        }
    }

    fn names(&self) -> impl Iterator<Item = &Option<String>> {
        self.interceptors.iter().map(|(n, _)| n)
    }

    fn summaries(&self, verdicts: &[Verdict]) -> Vec<VerdictSummary> {
        verdicts
            .iter()
            .zip(self.names())
            .enumerate()
            .map(|(index, (v, name))| VerdictSummary {
                index,
                decision: v.decision,
                reason: v.reason.clone(),
                name: name.clone(),
            })
            .collect()
    }

    /// `sequential/first_deny` (§7.4).
    async fn dispatch_first_deny(&self, ctx: &mut Value) -> Outcome {
        let n = self.interceptors.len();
        let on_approval = self.composition.on_approval.unwrap_or(OnApproval::Stop);
        let mut per: Vec<Verdict> = Vec::new();
        let mut pool: Vec<Verdict> = Vec::new();
        let mut last_transform: Option<(usize, Verdict)> = None;
        let mut resolved_by: Option<String> = None;
        let truncated = |i: usize| Some(i + 1 < n);

        for (i, (_, interceptor)) in self.interceptors.iter().enumerate() {
            let v = self.invoke(interceptor, ctx).await;
            per.push(v.clone());
            pool.push(v.clone());
            if v.is_host_error() {
                return Outcome {
                    combined: with_unions(v, &pool),
                    decided_by: Some(i),
                    verdicts: self.summaries(&per),
                    fold_truncated: truncated(i),
                    resolved_by,
                };
            }
            match v.decision {
                Decision::Deny => {
                    let Some((rv, permitted)) = self.consult(ctx, &v).await else {
                        return Outcome {
                            combined: with_unions(v, &pool),
                            decided_by: Some(i),
                            verdicts: self.summaries(&per),
                            fold_truncated: truncated(i),
                            resolved_by,
                        };
                    };
                    if !permitted {
                        let synthesized = rv.is_host_error();
                        return Outcome {
                            combined: with_unions(rv, &pool),
                            decided_by: if synthesized { None } else { Some(i) },
                            verdicts: self.summaries(&per),
                            fold_truncated: truncated(i),
                            resolved_by: Some("rejection".into()),
                        };
                    }
                    resolved_by = Some("approval".into());
                    let sub = if rv.decision == Decision::Transform {
                        self.fold_transform(ctx, rv)
                    } else {
                        rv
                    };
                    if !sub.decision.permits() {
                        return Outcome {
                            combined: sub,
                            decided_by: None,
                            verdicts: self.summaries(&per),
                            fold_truncated: truncated(i),
                            resolved_by,
                        };
                    }
                    pool.push(sub.clone());
                    if on_approval == OnApproval::Stop {
                        return Outcome {
                            combined: with_unions(sub, &pool),
                            decided_by: Some(i),
                            verdicts: self.summaries(&per),
                            fold_truncated: truncated(i),
                            resolved_by,
                        };
                    }
                    if sub.decision == Decision::Transform {
                        last_transform = Some((i, sub));
                    }
                }
                Decision::Transform => {
                    let folded = self.fold_transform(ctx, v);
                    if !folded.decision.permits() {
                        return Outcome {
                            combined: folded,
                            decided_by: None,
                            verdicts: self.summaries(&per),
                            fold_truncated: truncated(i),
                            resolved_by,
                        };
                    }
                    last_transform = Some((i, folded));
                }
                Decision::Allow => {}
            }
        }
        let (decided_by, combined) = match last_transform {
            Some((i, v)) => (Some(i), v),
            None => (None, Verdict::allow()),
        };
        Outcome {
            combined: with_unions(combined, &pool),
            decided_by,
            verdicts: self.summaries(&per),
            fold_truncated: Some(false),
            resolved_by,
        }
    }

    /// `sequential/run_all` (§7.4).
    async fn dispatch_run_all(&self, ctx: &mut Value) -> Outcome {
        let mut all: Vec<Verdict> = Vec::new();
        for (_, interceptor) in &self.interceptors {
            let v = self.invoke(interceptor, ctx).await;
            if v.decision == Decision::Transform {
                let folded = self.fold_transform(ctx, v);
                all.push(folded.clone());
                if !folded.decision.permits() {
                    // §7.4: a transform that fails to apply short-circuits.
                    return Outcome {
                        combined: folded,
                        decided_by: None,
                        verdicts: self.summaries(&all),
                        fold_truncated: None,
                        resolved_by: None,
                    };
                }
            } else {
                all.push(v);
            }
        }
        self.aggregate_and_consult(ctx, all).await
    }

    /// Parallel profiles (§7.5): isolated snapshots of the same untransformed
    /// context, no fold.
    async fn dispatch_parallel(&self, ctx: &mut Value) -> Outcome {
        let snapshot = ctx.clone();
        let mut all = Vec::new();
        for (_, interceptor) in &self.interceptors {
            all.push(self.invoke(interceptor, &snapshot).await);
        }
        self.aggregate_and_consult(ctx, all).await
    }

    /// Severity-max aggregation (§7.3) + winner handling, shared by
    /// `sequential/run_all` and the parallel profiles.
    async fn aggregate_and_consult(&self, ctx: &mut Value, all: Vec<Verdict>) -> Outcome {
        let agg = compose_aggregate(&self.composition, &all);
        let verdicts = self.summaries(&all);
        let mut combined = agg.combined;
        let mut decided_by = agg.decided_by;
        let mut resolved_by = None;
        if agg.apply_transform {
            let folded = self.fold_transform(ctx, combined);
            if !folded.decision.permits() {
                return Outcome {
                    combined: folded,
                    decided_by: None,
                    verdicts,
                    fold_truncated: None,
                    resolved_by: None,
                };
            }
            combined = folded;
        } else if agg.consult {
            if let Some((rv, permitted)) = self.consult(ctx, &combined).await {
                if permitted {
                    resolved_by = Some("approval".to_string());
                    let sub = if rv.decision == Decision::Transform {
                        self.fold_transform(ctx, rv)
                    } else {
                        rv
                    };
                    combined = if sub.decision.permits() {
                        let mut pool = all.clone();
                        pool.push(sub.clone());
                        with_unions(sub, &pool)
                    } else {
                        sub
                    };
                } else {
                    resolved_by = Some("rejection".to_string());
                    if rv.is_host_error() {
                        decided_by = None;
                    }
                    combined = with_unions(rv, &all);
                }
            }
        }
        Outcome {
            combined,
            decided_by,
            verdicts,
            fold_truncated: None,
            resolved_by,
        }
    }

    /// Apply (enforce) or validate (evaluate_only) one transform (§7.4, §8).
    fn fold_transform(&self, ctx: &mut Value, v: Verdict) -> Verdict {
        let Some(t) = &v.transform else {
            return Verdict::host_error(HostError::TransformInvalid, None, false);
        };
        match apply_transform_ctx(ctx, &t.path, &t.value) {
            Ok(new_ctx) => {
                if self.mode == EnforcementMode::Enforce {
                    *ctx = new_ctx;
                }
                v
            }
            Err((err, msg)) => Verdict::host_error(err, Some(msg), false),
        }
    }

    /// Consult the approval seam for a liftable deny (§9). `None` when the
    /// seam was not consulted (the deny stands as-is); otherwise the
    /// substituting verdict and whether it permits.
    async fn consult(&self, ctx: &Value, verdict: &Verdict) -> Option<(Verdict, bool)> {
        if !verdict.is_liftable() || self.mode != EnforcementMode::Enforce {
            return None;
        }
        let point = ctx
            .get("interception_point")
            .and_then(Value::as_str)
            .and_then(InterceptionPoint::parse)
            .unwrap_or(InterceptionPoint::AgentStartup);
        // §6.1a: nothing to approve at agent_shutdown.
        if point == InterceptionPoint::AgentShutdown {
            return None;
        }
        let resolver = self.resolver.as_ref()?;
        let failed = |detail: String| {
            Some((
                Verdict::host_error(HostError::ApprovalResolverFailed, Some(detail), false),
                false,
            ))
        };
        let presented = match &self.approval_redactor {
            Some(redactor) => match std::panic::catch_unwind(AssertUnwindSafe(|| redactor(ctx))) {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => return failed(crate::observability::error_type(&e)),
                Err(_) => return failed("panic".into()),
            },
            None => ctx.clone(),
        };
        let identity = match &self.identity {
            Some(p) => match p.compute(&presented) {
                Ok(id) => Some(id),
                Err((err, msg)) => {
                    return Some((Verdict::host_error(err, Some(msg), false), false))
                }
            },
            None => None,
        };
        let request = ApprovalRequest {
            context_identity: identity.clone(),
            interception_point: point,
            verdict: verdict.clone(),
            context: presented,
        };
        let fut = AssertUnwindSafe(resolver.resolve(request)).catch_unwind();
        let raw = match self.timeout {
            Some(t) => match tokio::time::timeout(t, fut).await {
                Ok(r) => r,
                Err(_) => return failed("timeout".into()),
            },
            None => fut.await,
        };
        let res = match raw {
            Err(_) => return failed("panic".into()),
            Ok(Err(e)) => return failed(crate::observability::error_type(&e)),
            Ok(Ok(res)) => res,
        };
        // §9 echo rule.
        if res.context_identity != identity {
            return Some((
                Verdict::host_error(HostError::ApprovalIdentityMismatch, None, false),
                false,
            ));
        }
        let rv = match (res.outcome, res.verdict) {
            (ApprovalOutcome::Unresolved, _) | (_, None) => {
                return Some((
                    Verdict::host_error(HostError::ApprovalUnresolved, None, false),
                    false,
                ))
            }
            (_, Some(rv)) => rv,
        };
        if let Err((_, msg)) = rv.validate() {
            return Some((
                Verdict::host_error(HostError::VerdictInvalid, Some(msg), false),
                false,
            ));
        }
        if res.outcome == ApprovalOutcome::Approve {
            if !rv.decision.permits() {
                return Some((
                    Verdict::host_error(
                        HostError::VerdictInvalid,
                        Some("approve MUST carry a permit verdict (§9)".into()),
                        false,
                    ),
                    false,
                ));
            }
            return Some((rv, true));
        }
        if rv.decision != Decision::Deny {
            return Some((
                Verdict::host_error(
                    HostError::VerdictInvalid,
                    Some("reject MUST carry a deny verdict (§9)".into()),
                    false,
                ),
                false,
            ));
        }
        Some((rv, false))
    }
}

/// First-seen-ordered union of warnings from every verdict, and of result
/// labels from every **permit** verdict when the combined verdict permits
/// (§7.3).
fn with_unions(mut combined: Verdict, pool: &[Verdict]) -> Verdict {
    let mut warnings: Vec<VerdictWarning> = Vec::new();
    for w in pool.iter().flat_map(|v| &v.warnings) {
        if !warnings.contains(w) {
            warnings.push(w.clone());
        }
    }
    if !warnings.is_empty() {
        combined.warnings = warnings;
    }
    if combined.decision.permits() {
        let mut labels: Vec<String> = Vec::new();
        for l in pool
            .iter()
            .filter(|v| v.decision.permits())
            .flat_map(|v| &v.result_labels)
        {
            if !labels.contains(l) {
                labels.push(l.clone());
            }
        }
        if !labels.is_empty() {
            combined.result_labels = labels;
        }
    }
    combined
}

struct Aggregate {
    combined: Verdict,
    decided_by: Option<usize>,
    consult: bool,
    apply_transform: bool,
}

/// Severity-max aggregation (the core's `compose_aggregate`): plain deny >
/// liftable deny > transform > allow, ties to the lowest index (the last
/// one for folded `run_all` transforms); conflicting parallel transforms
/// and unanimous disagreements synthesize per their knobs.
fn compose_aggregate(config: &CompositionConfig, all: &[Verdict]) -> Aggregate {
    let synthesize = |err: HostError, message: String, policy: Option<SynthesisPolicy>| {
        let liftable = policy == Some(SynthesisPolicy::Approval);
        let combined = with_unions(Verdict::host_error(err, Some(message), liftable), all);
        Aggregate {
            consult: liftable,
            combined,
            decided_by: None,
            apply_transform: false,
        }
    };
    if config.profile == CompositionProfile::ParallelUnanimous
        && all.iter().any(|v| v.decision != Decision::Allow)
    {
        return synthesize(
            HostError::CompositionDisagreement,
            "non-unanimous outcome under parallel/unanimous".into(),
            config.on_disagreement,
        );
    }
    let severity = |v: &Verdict| match v.decision {
        Decision::Deny if !v.is_liftable() => 3,
        Decision::Deny => 2,
        Decision::Transform => 1,
        Decision::Allow => 0,
    };
    let max = all.iter().map(severity).max().unwrap_or(0);
    let winners: Vec<usize> = (0..all.len())
        .filter(|i| severity(&all[*i]) == max)
        .collect();
    match max {
        0 => Aggregate {
            combined: with_unions(Verdict::allow(), all),
            decided_by: None,
            consult: false,
            apply_transform: false,
        },
        1 if config.profile.is_sequential() => {
            // Already folded: the last transform is the effective one.
            let i = *winners.last().unwrap_or(&0);
            Aggregate {
                combined: with_unions(all[i].clone(), all),
                decided_by: Some(i),
                consult: false,
                apply_transform: false,
            }
        }
        1 if winners.len() > 1 => synthesize(
            HostError::TransformConflict,
            format!(
                "conflicting transforms from interceptors [{}]",
                winners
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            config.on_transform_conflict,
        ),
        _ => {
            let i = winners[0];
            Aggregate {
                combined: with_unions(all[i].clone(), all),
                decided_by: Some(i),
                consult: max == 2,
                apply_transform: max == 1,
            }
        }
    }
}

// endregion

#[cfg(test)]
mod tests {
    use super::*;

    fn builder() -> AgentContextBuilder {
        AgentContextBuilder::new("a", "f", "s")
    }

    fn fixed(
        v: Verdict,
    ) -> FnInterceptor<impl Fn(Value) -> futures::future::Ready<Result<Verdict>>> {
        interceptor_fn(move |_| futures::future::ready(Ok(v.clone())))
    }

    #[test]
    fn canonical_json_matches_jcs() {
        let v = json!({"b": 1.0, "a": [1e21, 1.5e-7, 0.1, 100.0, -0.0, 123456789.125], "c": "é\n"});
        assert_eq!(
            canonical_json(&v),
            "{\"a\":[1e+21,1.5e-7,0.1,100,0,123456789.125],\"b\":1,\"c\":\"é\\n\"}"
        );
    }

    #[test]
    fn identity_is_sha256_of_the_canonical_context() {
        let ctx = json!({"z": 1, "a": "x"});
        let id = context_identity(&ctx);
        assert!(id.starts_with("sha256:"));
        assert_eq!(id.len(), 7 + 64);
        assert_eq!(id, context_identity(&json!({"a": "x", "z": 1})));
    }

    #[test]
    fn verdict_gate_rejects_spec_violations() {
        let cases = [
            (
                Verdict::allow().with_reason("host_error:x"),
                "MUST NOT start",
            ),
            (
                {
                    let mut v = Verdict::allow();
                    v.approval = Some(Map::new());
                    v
                },
                "approval block permitted only on deny",
            ),
            (
                {
                    let mut v = Verdict::allow();
                    v.decision = Decision::Transform;
                    v
                },
                "transform body REQUIRED",
            ),
            (
                Verdict::deny("d").with_warning(Some("host_error:x".into()), None),
                "non-reserved",
            ),
            (
                Verdict::allow().with_evidence(Evidence {
                    artefact: Some("x".repeat(20_000)),
                    ..Default::default()
                }),
                "exceeds 10240 bytes",
            ),
        ];
        for (v, needle) in cases {
            let (code, msg) = v.validate().unwrap_err();
            assert_eq!(code, HostError::VerdictInvalid);
            assert!(msg.contains(needle), "{msg}");
        }
        let (code, _) = Verdict::transform("foo", json!(1)).validate().unwrap_err();
        assert_eq!(code, HostError::TransformTargetForbidden);
        assert!(Verdict::from_wire(&json!({"decision": "warn"})).is_err());
        let v = Verdict::from_wire(
            &json!({"decision": "transform", "transform": {"path": "$target.a"}}),
        )
        .unwrap();
        assert_eq!(v.transform.unwrap().value, Value::Null);
    }

    #[test]
    fn transform_paths_replace_existing_slots_and_mirror_l1() {
        let b = builder();
        let ctx = b.pre_tool_call("c", "n", json!({"url": "x", "k": 1}));
        let out = apply_transform_ctx(&ctx, "$target.url", &json!("y")).unwrap();
        assert_eq!(out["target"]["url"], "y");
        assert_eq!(out["tool_call"]["args"]["url"], "y");
        let out = apply_transform_ctx(&ctx, "$policy_target", &json!({"z": 1})).unwrap();
        assert_eq!(out["tool_call"]["args"], json!({"z": 1}));
        for (path, code) in [
            ("$target.missing", HostError::TransformInvalid),
            ("$target[0]", HostError::TransformInvalid),
            ("$tgt", HostError::TransformTargetForbidden),
            ("$target.url.x", HostError::TransformInvalid),
        ] {
            assert_eq!(
                apply_transform_ctx(&ctx, path, &json!(1)).unwrap_err().0,
                code
            );
        }
        let msgs = b.pre_model_call("m", json!([{"role": "user", "content": "a"}]), None, None);
        let out = apply_transform_ctx(&msgs, "$target[0][\"content\"]", &json!("yo")).unwrap();
        assert_eq!(out["messages"][0]["content"], "yo");
        let startup = b.agent_startup(vec![]);
        assert_eq!(
            apply_transform_ctx(&startup, "$target", &json!(1))
                .unwrap_err()
                .0,
            HostError::TransformTargetForbidden
        );
    }

    #[test]
    fn envelope_validation_names_the_failing_member() {
        let b = builder();
        let ctx = b.input(json!("x"), "user");
        assert!(validate_envelope(&ctx).is_ok());
        let mut bad = ctx.clone();
        bad.as_object_mut().unwrap().remove("input");
        assert!(validate_envelope(&bad).unwrap_err().contains("$.input"));
        let mut bad = ctx.clone();
        bad["sequence"] = json!(-1);
        assert!(validate_envelope(&bad).unwrap_err().contains("$.sequence"));
        let mut bad = ctx;
        bad["input"]["content"] = json!(1u64 << 60);
        assert!(validate_envelope(&bad).unwrap_err().contains("2^53"));
    }

    fn agg(config: CompositionConfig, vs: &[Verdict]) -> Aggregate {
        compose_aggregate(&config, vs)
    }

    #[test]
    fn aggregation_follows_severity_max_and_the_profile_knobs() {
        let d = Verdict::deny("d1");
        let e = Verdict::escalate("e");
        let t1 = Verdict::transform("$target.x", json!(1)).with_result_label("L2");
        let t2 = Verdict::transform("$target.x", json!(2));
        let w = Verdict::warn("w").with_result_label("L1");
        let run_all = CompositionConfig::run_all();

        let a = agg(run_all, &[t1.clone(), d.clone(), w.clone()]);
        assert_eq!(a.combined.reason.as_deref(), Some("d1"));
        assert_eq!(a.decided_by, Some(1));
        assert_eq!(a.combined.warnings.len(), 1);
        assert!(a.combined.result_labels.is_empty());

        let a = agg(run_all, &[d.clone(), e.clone()]);
        assert_eq!((a.decided_by, a.consult), (Some(0), false));
        let a = agg(run_all, &[w.clone(), e.clone()]);
        assert_eq!((a.decided_by, a.consult), (Some(1), true));
        let a = agg(run_all, &[t1.clone(), t2.clone()]);
        assert_eq!(a.decided_by, Some(1));
        assert_eq!(a.combined.result_labels, vec!["L2".to_string()]);

        let strict = CompositionConfig::strictest(SynthesisPolicy::Deny);
        let a = agg(strict, &[Verdict::allow(), t1.clone()]);
        assert!(a.apply_transform);
        let a = agg(strict, &[t1.clone(), t2.clone()]);
        assert_eq!(
            a.combined.reason.as_deref(),
            Some("host_error:transform_conflict")
        );
        assert_eq!(
            a.combined.message.as_deref(),
            Some("conflicting transforms from interceptors [0, 1]")
        );
        assert!(!a.consult);
        let a = agg(
            CompositionConfig::strictest(SynthesisPolicy::Approval),
            &[t1.clone(), t2.clone(), w.clone()],
        );
        assert!(a.consult && a.combined.is_liftable());
        assert_eq!(a.combined.warnings.len(), 1);
        let a = agg(strict, &[t1.clone(), t2.clone(), e.clone()]);
        assert_eq!((a.decided_by, a.consult), (Some(2), true));

        let unanimous = CompositionConfig::unanimous(SynthesisPolicy::Deny, SynthesisPolicy::Deny);
        let a = agg(unanimous, &[Verdict::allow(), w.clone()]);
        assert_eq!(a.combined.decision, Decision::Allow);
        assert_eq!(a.combined.result_labels, vec!["L1".to_string()]);
        let a = agg(unanimous, &[Verdict::allow(), d]);
        assert_eq!(
            a.combined.reason.as_deref(),
            Some("host_error:composition_disagreement")
        );
        assert_eq!(a.decided_by, None);
    }

    #[tokio::test]
    async fn emitter_records_and_blocks() {
        let b = builder();
        let emitter = InterceptionEmitter::new().register_named(
            "g",
            fixed(Verdict::deny("nope").with_message("M".repeat(300))),
        );
        let err = emitter.emit(b.input(json!("x"), "user")).await.unwrap_err();
        let blocked = err.interception_blocked().unwrap();
        assert_eq!(blocked.to_string(), "input blocked: deny (nope)");
        let record = &blocked.record;
        assert_eq!(
            record.verdict.message.as_ref().unwrap().chars().count(),
            129
        );
        assert_eq!(record.verdicts[0].name.as_deref(), Some("g"));
        assert_eq!(record.fold_truncated, Some(false));
        assert_eq!(record.decided_by, Some(0));
        assert!(record.input_identity.is_some());
        assert_eq!(record.input_identity, record.enforced_identity);
        assert_eq!(emitter.results().len(), 1);
    }

    #[tokio::test]
    async fn zero_interceptors_fail_closed_unless_evaluating() {
        let b = builder();
        let r = InterceptionEmitter::new()
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert_eq!(
            r.verdict.reason.as_deref(),
            Some("host_error:no_interceptor")
        );
        assert!(!r.proceeds());
        let r = InterceptionEmitter::new()
            .with_mode(EnforcementMode::EvaluateOnly)
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert!(r.proceeds());
    }

    #[tokio::test]
    async fn transforms_fold_and_records_drop_the_value() {
        let b = builder();
        let emitter = InterceptionEmitter::new()
            .register(fixed(Verdict::transform("$target.url", json!("y"))))
            .register(interceptor_fn(|ctx: Value| async move {
                // Fold-through: the second interceptor sees the first's transform.
                assert_eq!(ctx["target"]["url"], "y");
                Ok(Verdict::allow())
            }));
        let out = emitter
            .emit(b.pre_tool_call("c", "n", json!({"url": "x"})))
            .await
            .unwrap();
        assert_eq!(out.target, json!({"url": "y"}));
        assert_eq!(
            out.record.verdict.transform.as_ref().unwrap().value,
            Value::Null
        );
        assert_ne!(out.record.input_identity, out.record.enforced_identity);

        // evaluate_only validates but never applies.
        let emitter = InterceptionEmitter::new()
            .with_mode(EnforcementMode::EvaluateOnly)
            .register(fixed(Verdict::transform("$target.content", json!("y"))));
        let out = emitter.emit(b.input(json!("x"), "user")).await.unwrap();
        assert_eq!(out.target["content"], "x");
    }

    #[tokio::test]
    async fn failures_synthesize_host_errors() {
        let b = builder();
        let failing = interceptor_fn(|_| async { Err::<Verdict, _>(Error::other("boom")) });
        let r = InterceptionEmitter::new()
            .register(failing)
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert_eq!(
            r.verdict.reason.as_deref(),
            Some("host_error:interceptor_failed")
        );
        assert_eq!(r.verdict.message.as_deref(), Some("other"));

        let slow = interceptor_fn(|_| async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(Verdict::allow())
        });
        let r = InterceptionEmitter::new()
            .with_timeout(Some(Duration::from_millis(10)))
            .register(slow)
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert_eq!(
            r.verdict.reason.as_deref(),
            Some("host_error:interceptor_timeout")
        );

        let r = InterceptionEmitter::new()
            .register(fixed(Verdict::transform(
                "$target.content",
                json!(1u64 << 60),
            )))
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert_eq!(
            r.verdict.reason.as_deref(),
            Some("host_error:context_invalid")
        );
        assert!(r.verdict.message.unwrap().contains("post-fold"));
        assert!(r.enforced_identity.is_none());
    }

    #[tokio::test]
    async fn approval_seam_lifts_rejects_and_enforces_the_echo_rule() {
        let b = builder();
        let approve = resolver_fn(|req: ApprovalRequest| async move {
            Ok(ApprovalResolution::approve(&req, Verdict::allow()))
        });
        let emitter = InterceptionEmitter::new()
            .with_resolver(Arc::new(approve))
            .register(fixed(Verdict::escalate("needs-approval")))
            .register(fixed(Verdict::deny("never-reached")));
        let r = emitter.emit_unchecked(b.input(json!("x"), "user")).await;
        assert!(r.proceeds());
        assert_eq!(r.resolved_by.as_deref(), Some("approval"));
        assert_eq!(r.fold_truncated, Some(true));

        let mismatched = resolver_fn(|_req: ApprovalRequest| async move {
            Ok(ApprovalResolution {
                outcome: ApprovalOutcome::Approve,
                context_identity: Some("sha256:wrong".into()),
                verdict: Some(Verdict::allow()),
            })
        });
        let r = InterceptionEmitter::new()
            .with_resolver(Arc::new(mismatched))
            .register(fixed(Verdict::escalate("e")))
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert_eq!(
            r.verdict.reason.as_deref(),
            Some("host_error:approval_identity_mismatch")
        );
        assert_eq!(r.resolved_by.as_deref(), Some("rejection"));
        assert_eq!(r.decided_by, None);

        let approve_with_deny = resolver_fn(|req: ApprovalRequest| async move {
            Ok(ApprovalResolution::approve(&req, Verdict::deny("x")))
        });
        let r = InterceptionEmitter::new()
            .with_resolver(Arc::new(approve_with_deny))
            .register(fixed(Verdict::escalate("e")))
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert_eq!(
            r.verdict.reason.as_deref(),
            Some("host_error:verdict_invalid")
        );

        // No resolver: the liftable deny stands as-is.
        let r = InterceptionEmitter::new()
            .register(fixed(Verdict::escalate("e")))
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert_eq!(r.verdict.reason.as_deref(), Some("e"));
        assert_eq!(r.resolved_by, None);
    }

    #[tokio::test]
    async fn first_deny_resume_continues_the_fold() {
        let b = builder();
        let approve = resolver_fn(|req: ApprovalRequest| async move {
            Ok(ApprovalResolution::approve(&req, Verdict::allow()))
        });
        let r = InterceptionEmitter::new()
            .with_composition(CompositionConfig::first_deny(OnApproval::Resume))
            .with_resolver(Arc::new(approve))
            .register(fixed(Verdict::escalate("e")))
            .register(fixed(Verdict::deny("second")))
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert_eq!(r.verdict.reason.as_deref(), Some("second"));
        assert_eq!(r.decided_by, Some(1));
    }

    #[tokio::test]
    async fn identity_providers_and_sinks() {
        let b = builder();
        let seen = Arc::new(Mutex::new(0usize));
        let seen2 = seen.clone();
        let emitter = InterceptionEmitter::new()
            .with_identity_provider(None)
            .with_record_sink(Arc::new(move |_r: &InterceptionRecord| {
                *seen2.lock().unwrap() += 1;
                panic!("sink failures are swallowed");
            }))
            .with_max_records(1)
            .register(fixed(Verdict::allow()));
        emitter.emit(b.input(json!("x"), "user")).await.unwrap();
        let r = emitter
            .emit(b.input(json!("y"), "user"))
            .await
            .unwrap()
            .record;
        assert_eq!((r.input_identity, r.identity_provider), (None, None));
        assert_eq!(*seen.lock().unwrap(), 2);
        assert_eq!(emitter.results().len(), 1);
        assert_eq!(emitter.records_dropped(), 1);

        assert!(IdentityProvider::custom("jcs-x", |_| Ok(String::new())).is_err());
        let custom = IdentityProvider::custom("mine", |_| Ok("id".into())).unwrap();
        let r = InterceptionEmitter::new()
            .with_identity_provider(Some(custom))
            .register(fixed(Verdict::allow()))
            .emit_unchecked(b.input(json!("x"), "user"))
            .await;
        assert_eq!(r.identity_provider.as_deref(), Some("mine"));
        assert_eq!(r.enforced_identity.as_deref(), Some("id"));
    }

    #[test]
    fn record_wire_shape_and_host_failure_records() {
        let emitter = InterceptionEmitter::new();
        let r = emitter.record_host_failure(
            InterceptionPoint::PreToolCall,
            Some("ValueError".into()),
            Some("s".into()),
            Some(3),
            None,
        );
        let wire = r.to_wire();
        assert_eq!(wire["verdict"]["reason"], "host_error:context_invalid");
        assert_eq!(wire["sequence"], 3);
        assert_eq!(wire["composition"]["on_approval"], "stop");
        assert!(wire.get("verdicts").is_none());
        assert!(!r.proceeds());
    }

    #[test]
    fn timestamps_are_rfc3339_utc() {
        let ts = now_rfc3339();
        assert_eq!(ts.len(), 27, "{ts}");
        assert!(ts.ends_with('Z') && ts.as_bytes()[10] == b'T');
    }
}
