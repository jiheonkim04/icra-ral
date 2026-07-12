# Autonomous Until Paper Final Decision

2026-07-12 KST continuity update:

Current campaign decision: `CYCLE_2_KILLED_PIVOT_TO_CYCLE_3`

This is not a terminal decision.

Cycle 1 `DICD-VLA` is closed with valid kill `SIMPLE_BASELINE_EXPLAINS_METHOD`. Cycle 2 `FEDO-VLA` is closed with valid kill `CLEAN_RETENTION_FAILURE`: full FEDO reached `1 / 10` under faults, the strongest faulted baselines reached `2 / 10`, and clean FEDO dropped from clean frozen `4 / 10` to `0 / 10`.

Next required action: Cycle 3, the final permitted distinct method cycle under the governance correction.

Current campaign decision: `DICD_REAL_TRACE_TRAINING_PASSED_STAGE_A_PENDING`

This is not a terminal decision.

Allowed terminal decisions:

- `READY_TO_DRAFT_RAL_PAPER_PACKAGE`
- `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`
- `HARD_EXTERNAL_BLOCKER`
- `SAFETY_RESOURCE_STOP`

The campaign has opened epoch 1, completed candidate discovery, selected `DICD-VLA`, frozen proposal/review/preregistration, implemented the core adapter, and passed synthetic smoke, real SmolVLA action-chunk smoke, and real trace training. It must continue to Stage A closed-loop evaluation.
