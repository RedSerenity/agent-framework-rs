//! Turns an upstream-format action list into a graph on the core
//! [`WorkflowBuilder`] (port of `_declarative_builder.py`, itself modelled on
//! .NET's `WorkflowActionVisitor`).
//!
//! * Every action becomes an executor; consecutive actions are joined by
//!   plain edges, except after a terminator (`GotoAction`, `BreakLoop`,
//!   `ContinueLoop`, `EndWorkflow`, `EndDialog`, `EndConversation`,
//!   `CancelDialog`, `CancelAllDialogs`).
//! * `If` / `ConditionGroup` become an evaluator node emitting the index of
//!   the first matching branch, with conditional edges into each branch
//!   (first-match semantics) and a pass-through for the missing else; every
//!   branch exit is wired to the successor.
//! * `Foreach` becomes `<id>_init` → body → `<id>_next` → body (loop back)
//!   with `<id>_exit` reached when the items run out; `BreakLoop` /
//!   `ContinueLoop` signal the enclosing `<id>_next`.
//! * `GotoAction` is a pass-through node with an edge to its target,
//!   resolved after all nodes exist (back edges create loops).
//! * A fixed `_workflow_entry` node receives the run input and initializes
//!   the state.
//!
//! Validation (as upstream): duplicate or reserved (`_workflow_entry`) ids,
//! missing required fields (with the accepted alternates), `ConditionGroup`
//! `else`/`default` (use `elseActions`), self-targeting gotos, unknown goto
//! targets, `BreakLoop`/`ContinueLoop` outside a loop, and HTTP/MCP actions
//! without a configured handler.
//!
//! Divergences, all strictly more permissive:
//!
//! * Executors that can never run (e.g. actions after a terminator) are
//!   pruned before the graph is built; upstream's graph validation rejects
//!   them as unreachable, while .NET accepts such workflows.
//! * A `GotoAction` may target the id of an `If`/`ConditionGroup`/`Foreach`
//!   (resolved to its entry node); upstream only finds plain action ids.
//! * `Foreach.items` satisfies the `source` requirement (the .NET field name).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use agent_framework_core::workflow::{CheckpointStorage, Workflow, WorkflowBuilder};
use serde_json::Value as Json;
use serde_yaml::Value as Yaml;

use super::executor::{field, str_field, Action, DeclarativeExecutor, Node, Runtime};
use super::messages::{is_branch, is_loop_result, ELSE_BRANCH_INDEX};
use crate::error::DeclarativeError;

/// The fixed id of the entry node.
pub const ENTRY_ID: &str = "_workflow_entry";

const TERMINATORS: &[&str] = &[
    "GotoAction",
    "BreakLoop",
    "ContinueLoop",
    "EndWorkflow",
    "EndDialog",
    "EndConversation",
    "CancelDialog",
    "CancelAllDialogs",
];

/// Required fields per action kind (upstream `ACTION_REQUIRED_FIELDS`).
const REQUIRED_FIELDS: &[(&str, &[&str])] = &[
    ("SetValue", &["path"]),
    ("SetVariable", &["variable"]),
    ("SendActivity", &["activity"]),
    ("InvokeAzureAgent", &["agent"]),
    ("GotoAction", &["actionId"]),
    ("Foreach", &["source", "actions"]),
    ("If", &["condition"]),
    ("ConditionGroup", &["conditions"]),
    ("Question", &["question", "variable"]),
    ("RequestExternalInput", &["prompt", "variable"]),
    ("RequestHumanInput", &["variable"]),
    ("WaitForHumanInput", &["variable"]),
    ("InvokeFunctionTool", &["functionName"]),
    ("HttpRequestAction", &["url"]),
    ("InvokeMcpTool", &["serverUrl", "toolName"]),
];

/// Alternate field names (upstream `ACTION_ALTERNATE_FIELDS`, plus
/// `Foreach.items`).
const ALTERNATE_FIELDS: &[(&str, &[&str])] = &[
    ("SetValue.path", &["variable"]),
    ("GotoAction.actionId", &["target"]),
    ("InvokeAzureAgent.agent", &["agentName"]),
    ("Question.question", &["text"]),
    ("Question.variable", &["property"]),
    ("RequestExternalInput.prompt", &["message"]),
    ("RequestExternalInput.variable", &["property"]),
    ("Foreach.source", &["items"]),
];

#[derive(Debug, Clone, Copy)]
enum EdgeCond {
    Always,
    Branch(i64),
    HasNext(bool),
}

/// A built element: a plain executor (with an optional distinct exit node, as
/// for `Foreach`) or an `If`/`ConditionGroup` structure.
#[derive(Debug, Clone)]
enum Item {
    Exec {
        id: String,
        kind: String,
        exit: Option<String>,
    },
    Struct {
        evaluator: String,
        branch_exits: Vec<Item>,
    },
}

#[derive(Debug, Clone, Default)]
struct Context {
    parent_id: Option<String>,
    loop_next: Option<String>,
}

/// Options for [`build_graph`].
pub(crate) struct BuildOptions {
    pub name: String,
    pub description: Option<String>,
    pub max_iterations: Option<usize>,
    pub checkpoint_storage: Option<Arc<dyn CheckpointStorage>>,
    pub strict_actions: bool,
}

struct GraphBuilder {
    rt: Arc<Runtime>,
    nodes: Vec<(String, Node, Json, Option<Yaml>)>,
    ids: HashMap<String, usize>,
    aliases: HashMap<String, String>,
    edges: Vec<(String, String, EdgeCond)>,
    pending_gotos: Vec<(String, String)>,
    index: usize,
    strict: bool,
}

fn invalid(msg: impl Into<String>) -> DeclarativeError {
    DeclarativeError::Invalid(msg.into())
}

/// Validate and build the workflow graph for `actions`.
pub(crate) fn build_graph(
    actions: &[Json],
    raw_actions: Option<&Yaml>,
    rt: Arc<Runtime>,
    options: BuildOptions,
) -> Result<Workflow, DeclarativeError> {
    if actions.is_empty() {
        return Err(invalid(
            "Cannot build workflow with no actions. At least one action is required.",
        ));
    }
    validate(actions)?;
    let mut g = GraphBuilder {
        rt,
        nodes: Vec::new(),
        ids: HashMap::new(),
        aliases: HashMap::new(),
        edges: Vec::new(),
        pending_gotos: Vec::new(),
        index: 0,
        strict: options.strict_actions,
    };
    g.add_node(
        ENTRY_ID.to_string(),
        Node::Join,
        serde_json::json!({"kind": "Entry"}),
        None,
    );
    let chain = g.create_chain(actions, raw_actions, &Context::default())?;
    let Some(chain) = chain else {
        return Err(invalid("Failed to create any executors from actions."));
    };
    let entry = Item::Exec {
        id: ENTRY_ID.into(),
        kind: "Entry".into(),
        exit: None,
    };
    g.sequential_edge(&entry, &chain[0]);
    g.resolve_gotos()?;
    g.finish(options)
}

fn validate(actions: &[Json]) -> Result<(), DeclarativeError> {
    // Top-level shape checks (upstream `WorkflowFactory._validate_workflow_def`).
    for (i, def) in actions.iter().enumerate() {
        if !def.is_object() {
            return Err(invalid(format!("Action at index {i} must be a dictionary")));
        }
        if field(def, "kind").is_none() {
            return Err(invalid(format!("Action at index {i} missing 'kind' field")));
        }
    }
    let mut seen = HashSet::new();
    validate_recursive(actions, &mut seen)
}

fn has_field(def: &Json, kind: &str, name: &str) -> bool {
    if field(def, name).is_some() {
        return true;
    }
    let key = format!("{kind}.{name}");
    ALTERNATE_FIELDS
        .iter()
        .find(|(k, _)| *k == key)
        .is_some_and(|(_, alts)| alts.iter().any(|a| field(def, a).is_some()))
}

fn validate_recursive(
    actions: &[Json],
    seen: &mut HashSet<String>,
) -> Result<(), DeclarativeError> {
    for (i, def) in actions.iter().enumerate() {
        if !def.is_object() {
            return Err(invalid(format!("Action at index {i} must be a dictionary")));
        }
        let kind = str_field(def, "kind").unwrap_or_default();
        let explicit_id = str_field(def, "id");
        if let Some(id) = explicit_id {
            if id == ENTRY_ID {
                return Err(invalid(format!(
                    "Action ID '{id}' is reserved for internal use. Choose a different ID."
                )));
            }
            if !seen.insert(id.to_string()) {
                return Err(invalid(format!(
                    "Duplicate action ID '{id}'. Action IDs must be unique."
                )));
            }
        }
        if let Some((_, required)) = REQUIRED_FIELDS.iter().find(|(k, _)| *k == kind) {
            for name in *required {
                if !has_field(def, kind, name) {
                    return Err(invalid(format!(
                        "Action '{kind}' is missing required field '{name}'. Action: {def}"
                    )));
                }
            }
        }
        match kind {
            "GotoAction" => {
                let target = str_field(def, "target").or_else(|| str_field(def, "actionId"));
                if let (Some(t), Some(id)) = (target, explicit_id) {
                    if t == id {
                        return Err(invalid(format!(
                            "Action '{id}' has a direct self-referencing GotoAction, which would cause an infinite loop."
                        )));
                    }
                }
            }
            "If" => {
                let then = field(def, "then").or_else(|| field(def, "actions"));
                validate_nested(then, seen)?;
                validate_nested(field(def, "else"), seen)?;
            }
            "ConditionGroup" => {
                for forbidden in ["else", "default"] {
                    if field(def, forbidden).is_some() {
                        return Err(invalid(format!(
                            "Action 'ConditionGroup' field '{forbidden}' is not supported; use 'elseActions' instead."
                        )));
                    }
                }
                for c in field(def, "conditions")
                    .and_then(Json::as_array)
                    .into_iter()
                    .flatten()
                {
                    validate_nested(field(c, "actions"), seen)?;
                }
                validate_nested(field(def, "elseActions"), seen)?;
            }
            "Foreach" => validate_nested(field(def, "actions"), seen)?,
            _ => {}
        }
    }
    Ok(())
}

fn validate_nested(v: Option<&Json>, seen: &mut HashSet<String>) -> Result<(), DeclarativeError> {
    match v {
        Some(Json::Array(a)) if !a.is_empty() => validate_recursive(a, seen),
        _ => Ok(()),
    }
}

fn actions_of(v: Option<&Json>) -> Vec<Json> {
    v.and_then(Json::as_array).cloned().unwrap_or_default()
}

impl GraphBuilder {
    fn add_node(&mut self, id: String, node: Node, def: Json, raw: Option<&Yaml>) {
        let raw = raw.cloned();
        if let Some(&i) = self.ids.get(&id) {
            self.nodes[i] = (id, node, def, raw);
        } else {
            self.ids.insert(id.clone(), self.nodes.len());
            self.nodes.push((id, node, def, raw));
        }
    }

    fn edge(&mut self, src: &str, tgt: &str, cond: EdgeCond) {
        self.edges.push((src.to_string(), tgt.to_string(), cond));
    }

    fn next_index(&mut self) -> usize {
        let i = self.index;
        self.index += 1;
        i
    }

    /// Build a chain of actions; returns the built items in order.
    fn create_chain(
        &mut self,
        actions: &[Json],
        raw: Option<&Yaml>,
        ctx: &Context,
    ) -> Result<Option<Vec<Item>>, DeclarativeError> {
        let mut items = Vec::new();
        let mut prev: Option<Item> = None;
        for (i, def) in actions.iter().enumerate() {
            let raw_def = raw.and_then(|r| r.get(i));
            let Some(item) = self.create_item(def, raw_def, ctx)? else {
                continue;
            };
            if let Some(p) = &prev {
                self.sequential_edge(p, &item);
            }
            let kind = str_field(def, "kind").unwrap_or_default();
            prev = if TERMINATORS.contains(&kind) {
                None
            } else {
                Some(item.clone())
            };
            items.push(item);
        }
        Ok((!items.is_empty()).then_some(items))
    }

    fn create_item(
        &mut self,
        def: &Json,
        raw: Option<&Yaml>,
        ctx: &Context,
    ) -> Result<Option<Item>, DeclarativeError> {
        let kind = str_field(def, "kind").unwrap_or_default().to_string();
        match kind.as_str() {
            "If" => return self.create_if(def, raw, ctx).map(Some),
            "ConditionGroup" => return self.create_condition_group(def, raw, ctx).map(Some),
            "Foreach" => return self.create_foreach(def, raw, ctx).map(Some),
            "GotoAction" => return Ok(self.create_goto(def, raw)),
            "BreakLoop" | "ContinueLoop" => {
                return self.create_loop_signal(def, raw, &kind, ctx).map(Some)
            }
            _ => {}
        }
        let Some(action) = Action::from_kind(&kind) else {
            if self.strict {
                return Err(invalid(format!("Unknown action kind '{kind}'")));
            }
            tracing::warn!(
                "Unknown action kind '{kind}' encountered at index {} - action will be skipped",
                self.index
            );
            return Ok(None);
        };
        let id = match str_field(def, "id") {
            Some(id) => id.to_string(),
            None => match &ctx.parent_id {
                Some(p) => format!("{p}_{kind}_{}", self.index),
                None => format!("{kind}_{}", self.index),
            },
        };
        self.index += 1;
        if action == Action::HttpRequest && self.rt.http.is_none() {
            return Err(invalid(format!(
                "Workflow defines HttpRequestAction '{id}' but no HTTP request handler was supplied \
                 to the WorkflowFactory (use with_http_request_handler)."
            )));
        }
        if action == Action::InvokeMcpTool && self.rt.mcp.is_none() {
            return Err(invalid(format!(
                "Workflow defines InvokeMcpTool '{id}' but no MCP tool handler was supplied \
                 to the WorkflowFactory (use with_mcp_tool_handler)."
            )));
        }
        self.add_node(id.clone(), Node::Action(action), def.clone(), raw);
        Ok(Some(Item::Exec {
            id,
            kind,
            exit: None,
        }))
    }

    fn create_if(
        &mut self,
        def: &Json,
        raw: Option<&Yaml>,
        ctx: &Context,
    ) -> Result<Item, DeclarativeError> {
        let i = self.next_index();
        let id = str_field(def, "id")
            .map(str::to_string)
            .unwrap_or_else(|| format!("If_{i}"));
        let condition = match field(def, "condition") {
            None => Json::String("true".into()),
            Some(c) => c.clone(),
        };
        let condition = super::executor::normalize_condition(&condition);
        let branch_ctx = Context {
            parent_id: Some(id.clone()),
            ..ctx.clone()
        };
        let evaluator = format!("{id}_eval");
        self.add_node(
            evaluator.clone(),
            Node::IfEval { condition },
            def.clone(),
            raw,
        );
        self.aliases.insert(id.clone(), evaluator.clone());

        let then_actions = actions_of(field(def, "then").or_else(|| field(def, "actions")));
        let then_raw = raw.and_then(|r| r.get("then").or_else(|| r.get("actions")));
        let then = self.create_chain(&then_actions, then_raw, &branch_ctx)?;
        let else_actions = actions_of(field(def, "else"));
        let els = if else_actions.is_empty() {
            None
        } else {
            self.create_chain(&else_actions, raw.and_then(|r| r.get("else")), &branch_ctx)?
        };
        let mut branch_exits = Vec::new();
        if let Some(then) = &then {
            let target = entry_of(&then[0]);
            self.edge(&evaluator, &target, EdgeCond::Branch(0));
            if let Some(exit) = branch_exit(then) {
                branch_exits.push(exit);
            }
        }
        match &els {
            Some(els) => {
                let target = entry_of(&els[0]);
                self.edge(&evaluator, &target, EdgeCond::Branch(ELSE_BRANCH_INDEX));
                if let Some(exit) = branch_exit(els) {
                    branch_exits.push(exit);
                }
            }
            None => {
                let pass = format!("{id}_else_pass");
                self.add_node(
                    pass.clone(),
                    Node::Join,
                    serde_json::json!({"kind": "ElsePassthrough"}),
                    None,
                );
                self.edge(&evaluator, &pass, EdgeCond::Branch(ELSE_BRANCH_INDEX));
                branch_exits.push(Item::Exec {
                    id: pass,
                    kind: "ElsePassthrough".into(),
                    exit: None,
                });
            }
        }
        Ok(Item::Struct {
            evaluator,
            branch_exits,
        })
    }

    fn create_condition_group(
        &mut self,
        def: &Json,
        raw: Option<&Yaml>,
        ctx: &Context,
    ) -> Result<Item, DeclarativeError> {
        let i = self.next_index();
        let id = str_field(def, "id")
            .map(str::to_string)
            .unwrap_or_else(|| format!("ConditionGroup_{i}"));
        let conditions = actions_of(field(def, "conditions"));
        let evaluator = format!("{id}_eval");
        self.add_node(
            evaluator.clone(),
            Node::GroupEval {
                conditions: conditions.clone(),
            },
            def.clone(),
            raw,
        );
        self.aliases.insert(id.clone(), evaluator.clone());
        let mut branch_exits = Vec::new();
        let mut entries = Vec::new();
        for (ci, cond) in conditions.iter().enumerate() {
            let case_ctx = Context {
                parent_id: Some(format!("{id}_case{ci}")),
                ..ctx.clone()
            };
            let case_raw = raw
                .and_then(|r| r.get("conditions"))
                .and_then(|c| c.get(ci))
                .and_then(|c| c.get("actions"));
            if let Some(chain) =
                self.create_chain(&actions_of(field(cond, "actions")), case_raw, &case_ctx)?
            {
                entries.push((ci as i64, entry_of(&chain[0])));
                if let Some(exit) = branch_exit(&chain) {
                    branch_exits.push(exit);
                }
            }
        }
        let else_actions = actions_of(field(def, "elseActions"));
        let mut default_target = None;
        if !else_actions.is_empty() {
            let else_ctx = Context {
                parent_id: Some(format!("{id}_else")),
                ..ctx.clone()
            };
            if let Some(chain) = self.create_chain(
                &else_actions,
                raw.and_then(|r| r.get("elseActions")),
                &else_ctx,
            )? {
                default_target = Some(entry_of(&chain[0]));
                if let Some(exit) = branch_exit(&chain) {
                    branch_exits.push(exit);
                }
            }
        } else {
            let pass = format!("{id}_default");
            self.add_node(
                pass.clone(),
                Node::Join,
                serde_json::json!({"kind": "DefaultPassthrough"}),
                None,
            );
            branch_exits.push(Item::Exec {
                id: pass.clone(),
                kind: "DefaultPassthrough".into(),
                exit: None,
            });
            default_target = Some(pass);
        }
        for (ci, target) in entries {
            self.edge(&evaluator, &target, EdgeCond::Branch(ci));
        }
        if let Some(t) = default_target {
            self.edge(&evaluator, &t, EdgeCond::Branch(ELSE_BRANCH_INDEX));
        }
        Ok(Item::Struct {
            evaluator,
            branch_exits,
        })
    }

    fn create_foreach(
        &mut self,
        def: &Json,
        raw: Option<&Yaml>,
        ctx: &Context,
    ) -> Result<Item, DeclarativeError> {
        let i = self.next_index();
        let id = str_field(def, "id")
            .map(str::to_string)
            .unwrap_or_else(|| format!("Foreach_{i}"));
        let init = format!("{id}_init");
        let next = format!("{id}_next");
        let exit = format!("{id}_exit");
        self.add_node(init.clone(), Node::ForeachInit, def.clone(), raw);
        self.add_node(
            next.clone(),
            Node::ForeachNext {
                init_id: init.clone(),
            },
            def.clone(),
            raw,
        );
        self.add_node(
            exit.clone(),
            Node::Join,
            serde_json::json!({"kind": "Join"}),
            None,
        );
        self.aliases.insert(id.clone(), init.clone());
        let loop_ctx = Context {
            parent_id: ctx.parent_id.clone(),
            loop_next: Some(next.clone()),
        };
        let body = self.create_chain(
            &actions_of(field(def, "actions")),
            raw.and_then(|r| r.get("actions")),
            &loop_ctx,
        )?;
        if let Some(body) = body {
            let target = entry_of(&body[0]);
            self.edge(&init, &target, EdgeCond::HasNext(true));
            if let Some(body_exit) = branch_exit(&body) {
                for src in source_exits(&body_exit) {
                    self.edge(&src, &next, EdgeCond::Always);
                }
            }
            self.edge(&next, &target, EdgeCond::HasNext(true));
        }
        self.edge(&init, &exit, EdgeCond::HasNext(false));
        self.edge(&next, &exit, EdgeCond::HasNext(false));
        Ok(Item::Exec {
            id: init,
            kind: "Foreach".into(),
            exit: Some(exit),
        })
    }

    fn create_goto(&mut self, def: &Json, raw: Option<&Yaml>) -> Option<Item> {
        let target = str_field(def, "target").or_else(|| str_field(def, "actionId"))?;
        let i = self.next_index();
        let id = str_field(def, "id")
            .map(str::to_string)
            .unwrap_or_else(|| format!("goto_{target}_{i}"));
        self.add_node(id.clone(), Node::Join, def.clone(), raw);
        self.pending_gotos.push((id.clone(), target.to_string()));
        Some(Item::Exec {
            id,
            kind: "GotoAction".into(),
            exit: None,
        })
    }

    fn create_loop_signal(
        &mut self,
        def: &Json,
        raw: Option<&Yaml>,
        kind: &str,
        ctx: &Context,
    ) -> Result<Item, DeclarativeError> {
        let Some(next) = ctx.loop_next.clone() else {
            return Err(invalid(format!(
                "{kind} action can only be used inside a Foreach loop"
            )));
        };
        let i = self.next_index();
        let prefix = if kind == "BreakLoop" {
            "Break"
        } else {
            "Continue"
        };
        let id = str_field(def, "id")
            .map(str::to_string)
            .unwrap_or_else(|| format!("{prefix}_{i}"));
        let signal = if kind == "BreakLoop" {
            "break"
        } else {
            "continue"
        };
        self.add_node(id.clone(), Node::LoopSignal(signal), def.clone(), raw);
        self.edge(&id, &next, EdgeCond::Always);
        Ok(Item::Exec {
            id,
            kind: kind.to_string(),
            exit: None,
        })
    }

    fn sequential_edge(&mut self, source: &Item, target: &Item) {
        let target = entry_of(target);
        for src in source_exits(source) {
            self.edge(&src, &target, EdgeCond::Always);
        }
    }

    fn resolve_gotos(&mut self) -> Result<(), DeclarativeError> {
        for (goto, target) in std::mem::take(&mut self.pending_gotos) {
            let resolved = if self.ids.contains_key(&target) {
                target.clone()
            } else if let Some(alias) = self.aliases.get(&target) {
                alias.clone()
            } else {
                let mut available: Vec<&String> = self.ids.keys().collect();
                available.sort();
                return Err(invalid(format!(
                    "GotoAction target '{target}' not found. Available action IDs: {available:?}"
                )));
            };
            self.edge(&goto, &resolved, EdgeCond::Always);
        }
        Ok(())
    }

    fn finish(self, options: BuildOptions) -> Result<Workflow, DeclarativeError> {
        // Prune nodes unreachable from the entry (dead code after terminators).
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
        for (s, t, _) in &self.edges {
            adjacency.entry(s.as_str()).or_default().push(t.as_str());
        }
        let mut reachable: HashSet<&str> = HashSet::new();
        let mut stack = vec![ENTRY_ID];
        while let Some(n) = stack.pop() {
            if reachable.insert(n) {
                if let Some(next) = adjacency.get(n) {
                    stack.extend(next.iter().copied());
                }
            }
        }
        let mut builder = WorkflowBuilder::new();
        for (id, node, def, raw) in &self.nodes {
            if !reachable.contains(id.as_str()) {
                tracing::debug!("declarative workflow: pruning unreachable action '{id}'");
                continue;
            }
            builder = builder.add_executor(Arc::new(DeclarativeExecutor {
                id: id.clone(),
                node: node.clone(),
                def: def.clone(),
                raw: raw.clone(),
                rt: self.rt.clone(),
            }));
        }
        for (s, t, cond) in &self.edges {
            if !reachable.contains(s.as_str()) {
                continue;
            }
            builder = match *cond {
                EdgeCond::Always => builder.add_edge(s.clone(), t.clone()),
                EdgeCond::Branch(b) => {
                    builder
                        .add_conditional_edge(s.clone(), t.clone(), move |v: &Json| is_branch(v, b))
                }
                EdgeCond::HasNext(h) => {
                    builder.add_conditional_edge(s.clone(), t.clone(), move |v: &Json| {
                        is_loop_result(v, h)
                    })
                }
            };
        }
        builder = builder.set_start(ENTRY_ID).name(options.name);
        if let Some(d) = options.description {
            builder = builder.description(d);
        }
        if let Some(m) = options.max_iterations {
            builder = builder.set_max_iterations(m);
        }
        if let Some(storage) = options.checkpoint_storage {
            builder = builder.with_checkpointing(storage);
        }
        Ok(builder.build()?)
    }
}

/// The node a predecessor wires into (an evaluator for structures).
fn entry_of(item: &Item) -> String {
    match item {
        Item::Exec { id, .. } => id.clone(),
        Item::Struct { evaluator, .. } => evaluator.clone(),
    }
}

/// The exit of a branch: `None` when it ends in a terminator.
fn branch_exit(chain: &[Item]) -> Option<Item> {
    let last = chain.last()?;
    if let Item::Exec { kind, .. } = last {
        if TERMINATORS.contains(&kind.as_str()) {
            return None;
        }
    }
    Some(last.clone())
}

/// Every node that continues to the successor of `item`.
fn source_exits(item: &Item) -> Vec<String> {
    match item {
        Item::Exec { id, exit, .. } => vec![exit.clone().unwrap_or_else(|| id.clone())],
        Item::Struct { branch_exits, .. } => branch_exits.iter().flat_map(source_exits).collect(),
    }
}
