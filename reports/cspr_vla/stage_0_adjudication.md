# CSPR-VLA Stage 0 Adjudication

Final decision: `CSPR_STAGE_0_IMPLEMENTATION_FAILURE`

Raw runner decision: `CSPR_STAGE_0_DESIGN_FAILURE`.

The raw decision is corrected without rerunning rows because the persisted decision inputs show an objective-scale defect: `weighted_gradient_norm_ratio_max = 129.38210738906673`, which exceeds the frozen `100.0` limit. The frozen protocol classifies objective-scale and gradient defects as implementation failures.

Completed/planned rows are `5760 / 5760`; exception count is `0`; duplicate manifest keys, duplicate partial keys, missing keys, extra keys, and split-overlap keys are all `0`; key sets are equal.

This is a development-only implementation failure, not a closed-loop scientific kill. CSPR rescue by changing thresholds, label construction, task/demo identities, proxy definition, action-validity semantics, or criticality features is disallowed.

Valid scientific result: `false`
Closed-loop scientific kill: `false`
