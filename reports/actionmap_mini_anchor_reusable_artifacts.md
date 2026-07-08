# ActionMap Mini-Anchor Reusable Artifacts

Date: 2026-07-08

## Reuse Boundary

These artifacts are reusable for diagnostics and archive evidence. They are not approval to implement Target-Grounded ActionMap or another local proxy method.

## Reusable Code

- `scripts/220_actionmap_mini_anchor_diagnostic.ps1`: gated CPU-only runner for the mini-anchor diagnostic.
- `tca_map/actionmap_anchor/diagnostic.py`: local HDF5 action-loader, tiny baseline heads, candidate grid metrics, oracle upper bound, and report writer.
- `tests/test_actionmap_mini_anchor_diagnostic.py`: focused tests for final decision labels and runner output paths.

## Reusable Reports

- `reports/actionmap_mini_anchor_task_definition.md`: scope and decision labels.
- `reports/actionmap_mini_anchor_experiment_plan.md`: bounded diagnostic plan.
- `reports/actionmap_mini_anchor_kill_criteria.md`: simple-baseline and collapse gates.
- `reports/actionmap_mini_anchor_state1_result.md`: human-readable STATE 1 result.
- `reports/actionmap_mini_anchor_state1_result.json`: machine-readable metrics.
- `reports/actionmap_mini_anchor_decision_log.md`: gate chronology.
- `reports/actionmap_mini_anchor_risk_register.md`: residual risks and mitigations.

## Reusable Evidence

- Real LIBERO/HDF5-backed train/eval split: `8` demos, `1008 / 432` records.
- Baseline table: mean-action, linear/L1, simple MLP, ActionMap-style candidate head, and oracle nearest-candidate upper bound.
- Candidate diagnostics: top-k accuracy, heatmap NLL, and candidate-collapse/diversity checks.
- Execution boundary: no downloads, no GPU, no OpenVLA-OFT, no rollout, no full official ActionMap reproduction, no target-grounded method implementation.

## What Must Not Be Reused As A Claim

- The oracle candidate upper bound must not be reported as learned method evidence.
- The local mini-head must not be described as official ActionMap reproduction.
- The local proxy failure must not be used to claim that the official ActionMap paper is false.
- The failed local mini-anchor must not be used as a foundation for Target-Grounded ActionMap.
