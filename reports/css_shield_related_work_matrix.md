# CSS-Shield Related Work Matrix

This concise matrix is a placeholder for scoped related-work tracking. It should not become a broad survey before rollout diagnostics produce signal.

| Area | Relevant Comparison | CSS-Shield Difference |
| --- | --- | --- |
| Action clipping | Clips action magnitude only | CSS-Shield must beat clipping-only on semantic or safety metrics |
| Runtime safety filters | Reject unsafe controls | CSS-Shield adds language/target counterfactual checks |
| VLA grounding diagnostics | Measure wrong-target actions | CSS-Shield intervenes at runtime and reports utility preservation |
| Recovery policies | Recover after failure | CSS-Shield first tries prevention via accept/damp/redirect/safe-stop |
| Full policy retraining | Learns a safer policy | CSS-Shield remains lightweight and runtime-only |

No SOTA or paper-grade claim is allowed until rollout metrics exist.

