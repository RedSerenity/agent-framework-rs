//! The [`WorkflowSpec`] type: a declarative description of a multi-agent
//! workflow.
//!
//! ## Relationship to the official schema
//!
//! The upstream declarative *workflow* schema (Python
//! `agent_framework_declarative._workflows`, .NET
//! `Microsoft.Agents.AI.Workflows.Declarative`) — a `trigger` with ordered
//! `actions` (`SetVariable`, `ConditionGroup`, `GotoAction`,
//! `InvokeAzureAgent`, `SendActivity`, …) evaluated with PowerFx — is
//! implemented by the [`flow`](crate::flow) module, and
//! [`DeclarativeLoader::load_workflow`](crate::DeclarativeLoader::load_workflow)
//! routes such documents there automatically.
//!
//! This [`WorkflowSpec`] is a separate, **documented Rust-native extension**
//! kept for backward compatibility. It drives the existing
//! [`WorkflowBuilder`](agent_framework_core::workflow::WorkflowBuilder) and the
//! orchestration builders (`SequentialBuilder`, `ConcurrentBuilder`,
//! `GroupChatBuilder`, `HandoffBuilder`). It keeps the official top-level
//! `kind: Workflow` key but defines its own body. Two forms are supported:
//!
//! * **Orchestration shorthand** — set `type:` to one of `sequential`,
//!   `concurrent`, `group_chat`, `handoff` and list `participants:` (agent ids).
//! * **Explicit graph** — declare `nodes:` (each referencing an agent by id),
//!   `edges:`, `fanOut:`/`fanIn:`, and `switch:`, with a `start:` node.

use serde::{Deserialize, Serialize};

use crate::error::{DeclarativeError, Result};

/// A declarative workflow specification (`kind: Workflow`).
///
/// Either the `type` field (orchestration shorthand) or
/// [`WorkflowSpec::nodes`] (explicit graph) must be provided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkflowSpec {
    /// The kind discriminator; must be `Workflow`.
    pub kind: String,
    /// The workflow name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// A description of the workflow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    // --- orchestration shorthand ---
    /// The orchestration pattern, when using shorthand.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<OrchestrationType>,
    /// Participant agent ids (resolved from the agent registry).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub participants: Vec<String>,
    /// Whether a `group_chat` uses a round-robin manager (default `true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub round_robin: Option<bool>,
    /// Maximum rounds for a `group_chat`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rounds: Option<usize>,
    /// Whether a `handoff` runs autonomously (no human input; default `true`
    /// so shorthand workflows are runnable offline).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autonomous: Option<bool>,
    /// Handoff edges (for `type: handoff`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub handoffs: Vec<HandoffEdgeSpec>,

    // --- explicit graph ---
    /// The entry node id (explicit graph) or initial agent (handoff shorthand).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
    /// The maximum number of supersteps (explicit graph).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<usize>,
    /// The workflow nodes, each wrapping an agent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<NodeSpec>,
    /// Plain and conditional edges.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<EdgeSpec>,
    /// Fan-out (broadcast) groups.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fan_out: Vec<FanOutSpec>,
    /// Fan-in (barrier) groups.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fan_in: Vec<FanInSpec>,
    /// Switch/case groups.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub switch: Vec<SwitchSpec>,
}

/// The orchestration pattern for shorthand workflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrchestrationType {
    /// A pipeline of agents (each sees and extends the conversation).
    Sequential,
    /// Fan an input to every participant and aggregate the replies.
    Concurrent,
    /// A manager coordinates a multi-agent conversation.
    GroupChat,
    /// Agents transfer control via handoff edges.
    Handoff,
}

/// A workflow node wrapping an agent as an executor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NodeSpec {
    /// The executor id (must be unique within the workflow).
    pub id: String,
    /// The agent id, resolved from the agent registry.
    pub agent: String,
    /// When `true`, the node yields its conversation as workflow output rather
    /// than forwarding it downstream.
    #[serde(default, skip_serializing_if = "is_false")]
    pub output: bool,
}

/// A directed edge, optionally guarded by a condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EdgeSpec {
    /// The source node id.
    pub from: String,
    /// The target node id.
    pub to: String,
    /// A condition mini-expression (`path OP literal`). See
    /// [`crate::condition`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    /// A named predicate registered in the
    /// [`PredicateRegistry`](crate::PredicateRegistry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
}

/// A fan-out (broadcast) group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FanOutSpec {
    /// The source node id.
    pub from: String,
    /// The target node ids.
    pub to: Vec<String>,
}

/// A fan-in (barrier) group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FanInSpec {
    /// The source node ids that must all deliver.
    pub from: Vec<String>,
    /// The target node id.
    pub to: String,
}

/// A switch/case group: evaluate `cases` in order, else route to `default`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SwitchSpec {
    /// The source node id.
    pub from: String,
    /// The ordered cases.
    pub cases: Vec<CaseSpec>,
    /// The fallback target node id.
    pub default: String,
}

/// A single branch of a [`SwitchSpec`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CaseSpec {
    /// A condition mini-expression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    /// A named predicate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    /// The target node id when the case matches.
    pub to: String,
    /// An optional visualization label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// A handoff edge (`type: handoff`): `from` may transfer control to any `to`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HandoffEdgeSpec {
    /// The source participant id.
    pub from: String,
    /// The participant ids that `from` may hand off to.
    pub to: Vec<String>,
}

impl WorkflowSpec {
    /// Parse a [`WorkflowSpec`] from YAML without environment interpolation.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).map_err(|e| DeclarativeError::Parse(e.to_string()))
    }

    /// Serialize this spec back to YAML.
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self).map_err(|e| DeclarativeError::Serialize(e.to_string()))
    }

    /// Validate `kind == "Workflow"` (case-insensitive).
    pub(crate) fn validate_kind(&self) -> Result<()> {
        if self.kind.eq_ignore_ascii_case("workflow") {
            Ok(())
        } else {
            Err(DeclarativeError::UnsupportedKind {
                what: "workflow",
                kind: self.kind.clone(),
                expected: vec!["Workflow"],
            })
        }
    }
}

fn is_false(b: &bool) -> bool {
    !*b
}
