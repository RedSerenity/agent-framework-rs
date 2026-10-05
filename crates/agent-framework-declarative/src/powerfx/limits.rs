//! Structural budgets for workflow state, ported from upstream
//! `_powerfx_limits.py`.
//!
//! Every traversal has its own budget and rejects excess with an error rather
//! than truncating values. Like upstream, the budget counts **values,
//! containers, and mapping keys** (each key is one node, at the depth of its
//! value), depth starts at zero, and text size counts string *characters*
//! (not encoded bytes). Upstream also rejects cyclic Python structures; JSON
//! values cannot be cyclic, so that check has no Rust counterpart.
//!
//! These limits bound the *state* that expressions read; expression length and
//! nesting are bounded separately by [`ExpressionLimits`](super::ExpressionLimits).

use serde_json::Value as Json;

/// Upstream `_MAX_POWERFX_STATE_DEPTH`.
pub const MAX_STATE_DEPTH: usize = 64;
/// Upstream `_MAX_POWERFX_STATE_NODES`.
pub const MAX_STATE_NODES: usize = 10_000;
/// Upstream `_MAX_POWERFX_STATE_TEXT_SIZE` (characters).
pub const MAX_STATE_TEXT_SIZE: usize = 1_048_576;

/// The budget applied to declarative workflow state before it is copied,
/// written, or exposed to expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateBudget {
    /// Maximum nesting depth (the root is depth 0).
    pub max_depth: usize,
    /// Maximum number of values, containers and object keys.
    pub max_nodes: usize,
    /// Maximum total characters across all strings (including object keys).
    pub max_text_size: usize,
}

impl Default for StateBudget {
    fn default() -> Self {
        Self {
            max_depth: MAX_STATE_DEPTH,
            max_nodes: MAX_STATE_NODES,
            max_text_size: MAX_STATE_TEXT_SIZE,
        }
    }
}

/// A state value exceeded the [`StateBudget`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("PowerFx state exceeds the {0} budget")]
pub struct StateLimitError(pub &'static str);

/// A running tally, so several values can share one budget (upstream's
/// `_PowerFxStateBudget`).
#[derive(Debug, Clone, Copy)]
pub struct BudgetTally {
    budget: StateBudget,
    nodes: usize,
    text_size: usize,
}

impl BudgetTally {
    /// Start a fresh tally against `budget`.
    pub fn new(budget: StateBudget) -> Self {
        Self {
            budget,
            nodes: 0,
            text_size: 0,
        }
    }

    fn consume(&mut self, text_len: Option<usize>, depth: usize) -> Result<(), StateLimitError> {
        self.nodes += 1;
        if self.nodes > self.budget.max_nodes {
            return Err(StateLimitError("node"));
        }
        if depth > self.budget.max_depth {
            return Err(StateLimitError("depth"));
        }
        if let Some(len) = text_len {
            self.text_size += len;
            if self.text_size > self.budget.max_text_size {
                return Err(StateLimitError("text size"));
            }
        }
        Ok(())
    }

    /// Count `value` (and everything inside it) against the tally.
    pub fn visit(&mut self, value: &Json) -> Result<(), StateLimitError> {
        self.visit_at(value, 0)
    }

    fn visit_at(&mut self, value: &Json, depth: usize) -> Result<(), StateLimitError> {
        let text = match value {
            Json::String(s) => Some(s.chars().count()),
            _ => None,
        };
        self.consume(text, depth)?;
        match value {
            Json::Object(map) => {
                for (k, v) in map {
                    self.consume(Some(k.chars().count()), depth + 1)?;
                    self.visit_at(v, depth + 1)?;
                }
            }
            Json::Array(items) => {
                for item in items {
                    self.visit_at(item, depth + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Validate a single value against `budget` (upstream `_validate_powerfx_state`).
pub fn validate_state(value: &Json, budget: StateBudget) -> Result<(), StateLimitError> {
    BudgetTally::new(budget).visit(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn budget(depth: usize, nodes: usize, text: usize) -> StateBudget {
        StateBudget {
            max_depth: depth,
            max_nodes: nodes,
            max_text_size: text,
        }
    }

    #[test]
    fn boundaries_match_upstream() {
        // Mirrors upstream test_budget_boundaries.
        let b = budget(2, 10_000, 1_000_000);
        assert!(validate_state(&json!([[0]]), b).is_ok());
        assert_eq!(
            validate_state(&json!([[[0]]]), b).unwrap_err(),
            StateLimitError("depth")
        );
        let b = budget(64, 3, 1_000_000);
        assert!(validate_state(&json!([0, 1]), b).is_ok());
        assert_eq!(
            validate_state(&json!([0, 1, 2]), b).unwrap_err(),
            StateLimitError("node")
        );
        let b = budget(64, 10_000, 4);
        assert!(validate_state(&json!({"ab": "cd"}), b).is_ok());
        assert_eq!(
            validate_state(&json!({"ab": "cde"}), b).unwrap_err(),
            StateLimitError("text size")
        );
    }

    #[test]
    fn default_depth_limit_rejects_65_levels() {
        let mut v = json!("leaf");
        for _ in 0..65 {
            v = json!([v]);
        }
        assert!(validate_state(&v, StateBudget::default()).is_err());
    }

    #[test]
    fn shared_tally_accumulates() {
        let mut tally = BudgetTally::new(budget(64, 5, 100));
        tally.visit(&json!([1])).unwrap();
        tally.visit(&json!([1])).unwrap();
        assert!(tally.visit(&json!([1])).is_err());
    }
}
