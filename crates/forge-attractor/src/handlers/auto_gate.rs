use crate::{
    AttractorError, Graph, Node, NodeOutcome, NodeStatus, RuntimeContext, handlers::NodeHandler,
};
use async_trait::async_trait;
use serde_json::Value;

/// Handler for automatic gate nodes (hexagon shape with `auto_policy` attribute).
///
/// Supports:
/// - `max_iterations` — hard cap on how many times the gate's loop can cycle.
///   When exceeded, the gate forces the "pass" edge.
/// - `auto_policy="findings_empty"` — routes to the "pass" edge when no findings
///   are detected in the runtime context, otherwise routes to the "findings" edge.
///
/// Iteration count is tracked via a context variable `gate.<node_id>.iterations`.
///
/// Edge detection uses the `[P]` / `[F]` accelerator-key convention from the DOT
/// edge labels (e.g. `[P] Pass`, `[F] Findings`).
#[derive(Debug, Default)]
pub struct AutoGateHandler;

#[async_trait]
impl NodeHandler for AutoGateHandler {
    async fn execute(
        &self,
        node: &Node,
        context: &RuntimeContext,
        graph: &Graph,
    ) -> Result<NodeOutcome, AttractorError> {
        let max_iterations = node
            .attrs
            .get("max_iterations")
            .and_then(|v| v.as_i64())
            .unwrap_or(5) as usize;

        let iteration_key = format!("gate.{}.iterations", node.id);
        let current = context
            .get(&iteration_key)
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let next = current + 1;

        let edges: Vec<_> = graph.outgoing_edges(&node.id).collect();
        let pass_edge = edges.iter().find(|e| {
            let label = e.attrs.get_str("label").unwrap_or("");
            label_is_pass(label)
        });
        let findings_edge = edges.iter().find(|e| {
            let label = e.attrs.get_str("label").unwrap_or("");
            label_is_findings(label)
        });

        let force_pass = next > max_iterations;
        let policy_pass = if !force_pass {
            evaluate_policy(node, context)
        } else {
            false // doesn't matter, force_pass overrides
        };

        let should_pass = force_pass || policy_pass;

        let mut updates = RuntimeContext::new();
        updates.insert(iteration_key, Value::Number(next.into()));

        if should_pass {
            let reason = if force_pass {
                format!(
                    "auto gate: max_iterations reached ({}/{}), forcing pass",
                    next, max_iterations
                )
            } else {
                format!(
                    "auto gate: policy satisfied (iteration {}/{})",
                    next, max_iterations
                )
            };

            if let Some(edge) = pass_edge {
                return Ok(NodeOutcome {
                    status: NodeStatus::Success,
                    preferred_label: Some(
                        edge.attrs.get_str("label").unwrap_or("").to_string(),
                    ),
                    suggested_next_ids: vec![edge.to.clone()],
                    context_updates: updates,
                    notes: Some(reason),
                    ..Default::default()
                });
            }
            // No pass edge found — fall through to first edge
            return Ok(NodeOutcome {
                status: NodeStatus::Success,
                context_updates: updates,
                notes: Some(format!("{reason} (no pass edge found)")),
                ..Default::default()
            });
        }

        // Policy says loop: take the findings edge
        let reason = format!(
            "auto gate: findings detected (iteration {}/{})",
            next, max_iterations
        );
        if let Some(edge) = findings_edge {
            Ok(NodeOutcome {
                status: NodeStatus::Success,
                preferred_label: Some(
                    edge.attrs.get_str("label").unwrap_or("").to_string(),
                ),
                suggested_next_ids: vec![edge.to.clone()],
                context_updates: updates,
                notes: Some(reason),
                ..Default::default()
            })
        } else {
            // No findings edge — pass by default
            Ok(NodeOutcome {
                status: NodeStatus::Success,
                context_updates: updates,
                notes: Some("auto gate: no findings edge, defaulting to pass".to_string()),
                ..Default::default()
            })
        }
    }
}

/// Evaluate the `auto_policy` attribute on the node.
/// Returns `true` if the gate should pass (route to the pass edge).
fn evaluate_policy(node: &Node, context: &RuntimeContext) -> bool {
    let policy = match node.attrs.get_str("auto_policy") {
        Some(p) => p,
        None => return false, // no policy → default to loop
    };

    match policy {
        "findings_empty" => {
            // Check for a context variable that signals findings.
            // Convention: the immediately preceding stage can set
            // `<stage>.findings_count` or `findings_count` to 0.
            let findings_count = context
                .get("findings_count")
                .and_then(|v| v.as_u64());
            match findings_count {
                Some(0) => true,  // explicitly zero findings → pass
                Some(_) => false, // non-zero findings → loop
                None => false,    // no signal → assume findings exist, loop
            }
        }
        "always_pass" => true,
        "always_loop" => false,
        _ => false, // unknown policy → default to loop
    }
}

fn label_is_pass(label: &str) -> bool {
    let lower = label.to_ascii_lowercase();
    lower.contains("[p]") || lower.contains("pass")
}

fn label_is_findings(label: &str) -> bool {
    let lower = label.to_ascii_lowercase();
    lower.contains("[f]") || lower.contains("findings") || lower.contains("finding")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_dot;

    #[tokio::test(flavor = "current_thread")]
    async fn auto_gate_first_iteration_with_findings_expected_findings_edge() {
        let graph = parse_dot(
            r#"
            digraph G {
                gate [shape=hexagon, auto_policy="findings_empty", max_iterations=5]
                fix
                pass
                gate -> fix [label="[F] Findings"]
                gate -> pass [label="[P] Pass"]
            }
            "#,
        )
        .expect("graph should parse");
        let node = graph.nodes.get("gate").expect("gate should exist");
        let context = RuntimeContext::new();

        let outcome = AutoGateHandler
            .execute(node, &context, &graph)
            .await
            .expect("execution should succeed");

        assert_eq!(outcome.status, NodeStatus::Success);
        assert_eq!(outcome.suggested_next_ids, vec!["fix".to_string()]);
        assert!(outcome.notes.unwrap().contains("findings detected"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn auto_gate_max_iterations_exceeded_expected_pass_edge() {
        let graph = parse_dot(
            r#"
            digraph G {
                gate [shape=hexagon, auto_policy="findings_empty", max_iterations=3]
                fix
                pass
                gate -> fix [label="[F] Findings"]
                gate -> pass [label="[P] Pass"]
            }
            "#,
        )
        .expect("graph should parse");
        let node = graph.nodes.get("gate").expect("gate should exist");
        let mut context = RuntimeContext::new();
        // Simulate 3 prior iterations
        context.insert(
            "gate.gate.iterations".to_string(),
            Value::Number(3.into()),
        );

        let outcome = AutoGateHandler
            .execute(node, &context, &graph)
            .await
            .expect("execution should succeed");

        assert_eq!(outcome.status, NodeStatus::Success);
        assert_eq!(outcome.suggested_next_ids, vec!["pass".to_string()]);
        assert!(outcome.notes.unwrap().contains("max_iterations reached"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn auto_gate_findings_empty_with_zero_count_expected_pass() {
        let graph = parse_dot(
            r#"
            digraph G {
                gate [shape=hexagon, auto_policy="findings_empty", max_iterations=5]
                fix
                pass
                gate -> fix [label="[F] Findings"]
                gate -> pass [label="[P] Pass"]
            }
            "#,
        )
        .expect("graph should parse");
        let node = graph.nodes.get("gate").expect("gate should exist");
        let mut context = RuntimeContext::new();
        context.insert("findings_count".to_string(), Value::Number(0.into()));

        let outcome = AutoGateHandler
            .execute(node, &context, &graph)
            .await
            .expect("execution should succeed");

        assert_eq!(outcome.status, NodeStatus::Success);
        assert_eq!(outcome.suggested_next_ids, vec!["pass".to_string()]);
        assert!(outcome.notes.unwrap().contains("policy satisfied"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn auto_gate_increments_iteration_count() {
        let graph = parse_dot(
            r#"
            digraph G {
                gate [shape=hexagon, auto_policy="findings_empty", max_iterations=5]
                fix
                pass
                gate -> fix [label="[F] Findings"]
                gate -> pass [label="[P] Pass"]
            }
            "#,
        )
        .expect("graph should parse");
        let node = graph.nodes.get("gate").expect("gate should exist");
        let mut context = RuntimeContext::new();
        context.insert(
            "gate.gate.iterations".to_string(),
            Value::Number(1.into()),
        );

        let outcome = AutoGateHandler
            .execute(node, &context, &graph)
            .await
            .expect("execution should succeed");

        let updated = outcome
            .context_updates
            .get("gate.gate.iterations")
            .and_then(|v| v.as_u64())
            .expect("iteration count should exist");
        assert_eq!(updated, 2);
    }
}
