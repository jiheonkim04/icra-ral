# AFID-VLA Stage 0 Adjudication

Decision: `AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`

This is a valid frozen Stage 0 development stop, not a closed-loop scientific
result and not a paper claim.

Integrity checks:

- completed/planned rows: `5120 / 5120`;
- exception count: `0`;
- duplicate manifest keys: `0`;
- duplicate partial keys: `0`;
- missing manifest keys: `0`;
- extra partial keys: `0`;
- split-overlap keys: `0`;
- PID `375` is no longer alive and exit code is `0`.

Decision precedence:

- implementation/objective-scale failure fires because
  `action_deltas_bounded = false`;
- action validity and clean retention are otherwise true;
- factor labels and factor mask are noncollapsed, with usable factor count
  `2`;
- factor-conditioned oracle reduction is positive (`0.14932795099984086`);
- FineVLA proxy residual headroom is positive (`0.16468672613404625`);
- factor prediction does not beat the required trivial baselines
  (`factor_predictor_beats_majority = false`,
  `factor_predictor_beats_task_phase = false`), which would independently
  prevent progression even without the objective-scale failure.

AFID may not proceed to bounded validation search in this formulation. Do not
rescue this result by changing thresholds, proxy definitions, tasks, reset
identities, masks, or action-validity semantics after seeing the result.

No confirmatory-test records, simulator rollouts, reward rows, success flags, or done flags were read.
