use crate::{
    AttractorError, Graph, Node, NodeOutcome, NodeStatus, RuntimeContext, handlers::NodeHandler,
};
use async_trait::async_trait;
use serde_json::Value;

/// Handler for automatic gate nodes (hexagon shape with `auto_policy` attribute).
///
/// Supports:
/// - `max_iterations` — total cap on how many times this gate may evaluate.
///   `max_iterations=N` means "at most N evaluations" — the Nth evaluation
///   force-passes if the policy is still unsatisfied. So `max_iterations=3`
///   permits up to 2 findings → refine cycles, with the 3rd gate visit always
///   advancing the pipeline.
/// - `auto_policy="findings_empty"` — routes to the "pass" edge when no findings
///   are detected in the runtime context, otherwise routes to the "findings" edge.
///
/// Iteration count is tracked via a context variable `gate.<node_id>.iterations`.
///
/// Edge detection uses the `[P]` / `[F]` accelerator-key convention from the DOT
/// edge labels (e.g. `[P] Pass`, `[F] Findings`).
///
/// **Known limitation:** the `findings_empty` policy reads `findings_count` from
/// the runtime context, but agent-driven box stages have no built-in path to
/// write context variables — so for those stages the policy is effectively
/// "always_loop_until_max_iterations". The structural fix (parse agent stdout
/// for a `findings_count=N` line or surface it via output_schema) is tracked
/// separately; until then, set `max_iterations` low to bound wasted refines.
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
            .map(|v| v.to_string_value().parse::<usize>().unwrap_or(5))
            .unwrap_or(5);

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

        // Force-pass on the Nth evaluation when max_iterations=N, so the gate
        // total visit count never exceeds max_iterations. Previously this used
        // `>`, which silently permitted one extra refine cycle per gate.
        let force_pass = next >= max_iterations;
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

            let edge = pass_edge.copied().or_else(|| edges.first().copied()).ok_or_else(|| {
                AttractorError::Runtime(format!(
                    "auto gate '{}': no outgoing edges",
                    node.id
                ))
            })?;
            let notes = if pass_edge.is_some() {
                reason
            } else {
                format!("{reason} (no pass edge found, falling back to first edge)")
            };
            return Ok(NodeOutcome {
                status: NodeStatus::Success,
                preferred_label: Some(edge.attrs.get_str("label").unwrap_or("").to_string()),
                suggested_next_ids: vec![edge.to.clone()],
                context_updates: updates,
                notes: Some(notes),
                ..Default::default()
            });
        }

        // Policy says loop: take the findings edge
        let reason = format!(
            "auto gate: findings detected (iteration {}/{})",
            next, max_iterations
        );
        let edge = findings_edge
            .copied()
            .or_else(|| pass_edge.copied())
            .or_else(|| edges.first().copied())
            .ok_or_else(|| {
                AttractorError::Runtime(format!(
                    "auto gate '{}': no outgoing edges",
                    node.id
                ))
            })?;
        let notes = if findings_edge.is_some() {
            reason
        } else {
            format!("{reason} (no findings edge, defaulting to pass)")
        };
        Ok(NodeOutcome {
            status: NodeStatus::Success,
            preferred_label: Some(edge.attrs.get_str("label").unwrap_or("").to_string()),
            suggested_next_ids: vec![edge.to.clone()],
            context_updates: updates,
            notes: Some(notes),
            ..Default::default()
        })
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
    let lower = label.trim().to_ascii_lowercase();
    lower.contains("[p]") || lower == "pass" || lower.starts_with("pass ")
}

fn label_is_findings(label: &str) -> bool {
    let lower = label.trim().to_ascii_lowercase();
    lower.contains("[f]") || lower == "findings" || lower.starts_with("findings ") || lower == "finding" || lower.starts_with("finding ")
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
    async fn auto_gate_at_nth_visit_with_findings_force_passes() {
        // Regression: previously `force_pass = next > max_iterations` permitted
        // an extra refine cycle past the cap. The 3rd visit to a max=3 gate
        // with findings still present must now force-pass instead of looping.
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
        // 2 prior iterations → this is the 3rd visit, equal to the cap.
        context.insert(
            "gate.gate.iterations".to_string(),
            Value::Number(2.into()),
        );

        let outcome = AutoGateHandler
            .execute(node, &context, &graph)
            .await
            .expect("execution should succeed");

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

    #[tokio::test(flavor = "current_thread")]
    async fn auto_gate_string_max_iterations_parses_correctly() {
        // DOT parsers often surface numeric attributes as strings when quoted.
        let graph = parse_dot(
            r#"
            digraph G {
                gate [shape=hexagon, auto_policy="findings_empty", max_iterations="2"]
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
        // Simulate 2 prior iterations — next=3 > 2, should force pass.
        context.insert(
            "gate.gate.iterations".to_string(),
            Value::Number(2.into()),
        );

        let outcome = AutoGateHandler
            .execute(node, &context, &graph)
            .await
            .expect("execution should succeed");

        assert_eq!(outcome.suggested_next_ids, vec!["pass".to_string()]);
        assert!(outcome.notes.unwrap().contains("max_iterations reached"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn auto_gate_no_pass_edge_falls_back_to_first_edge() {
        let graph = parse_dot(
            r#"
            digraph G {
                gate [shape=hexagon, auto_policy="findings_empty", max_iterations=5]
                fix
                exit_node
                gate -> fix [label="[F] Findings"]
                gate -> exit_node [label="done"]
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

        // No pass edge found — should fall back to the first edge (fix) instead
        // of returning an empty suggested_next_ids that would stall the graph.
        assert_eq!(outcome.status, NodeStatus::Success);
        assert_eq!(outcome.suggested_next_ids.len(), 1);
        assert!(outcome.notes.unwrap().contains("falling back"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn auto_gate_no_findings_edge_falls_back_to_pass_edge() {
        let graph = parse_dot(
            r#"
            digraph G {
                gate [shape=hexagon, auto_policy="findings_empty", max_iterations=5]
                pass
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

        // Policy says loop (no findings_count signal), but no findings edge
        // exists — should fall back to the pass edge rather than stalling.
        assert_eq!(outcome.status, NodeStatus::Success);
        assert_eq!(outcome.suggested_next_ids, vec!["pass".to_string()]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn auto_gate_no_outgoing_edges_returns_error() {
        let graph = parse_dot(
            r#"
            digraph G {
                gate [shape=hexagon, auto_policy="findings_empty", max_iterations=5]
            }
            "#,
        )
        .expect("graph should parse");
        let node = graph.nodes.get("gate").expect("gate should exist");
        let context = RuntimeContext::new();

        let result = AutoGateHandler.execute(node, &context, &graph).await;
        assert!(result.is_err(), "expected error when gate has no outgoing edges");
    }

    #[test]
    fn label_is_pass_does_not_match_bypass() {
        assert!(!label_is_pass("bypass"));
        assert!(!label_is_pass("Bypass Review"));
        assert!(label_is_pass("[P] Pass"));
        assert!(label_is_pass("Pass"));
        assert!(label_is_pass("pass to next"));
    }

    #[test]
    fn label_is_findings_does_not_match_no_findings() {
        assert!(!label_is_findings("no findings"));
        assert!(label_is_findings("[F] Findings"));
        assert!(label_is_findings("Findings"));
        assert!(label_is_findings("findings detected"));
        assert!(label_is_findings("Finding"));
    }
}
