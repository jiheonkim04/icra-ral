# ExecSpec-Repair Reusable Artifacts

ExecSpec-Repair is killed as a broad RA-L route, but its tooling should be preserved. The useful residue is the baseline-first discipline, exact-init replay infrastructure, and negative-evidence audit pattern.

## Reusable Code And Scripts

- `tca_map.execspec.mismatch_diagnostic`
- `tca_map.execspec.exact_init_mismatch_replay`
- `tca_map.execspec.repair`
- `tca_map.execspec.replay_validation`
- `tca_map.execspec.baseline_dominance_audit`
- `scripts\163_execspec_mismatch_diagnostic.ps1`
- `scripts\164_execspec_exact_init_mismatch_replay.ps1`
- `scripts\165_execspec_calibrated_repair.ps1`
- `scripts\166_execspec_replay_validation.ps1`
- `scripts\167_execspec_baseline_dominance_audit.ps1`

## Reusable Reports

- `reports/execspec_mismatch_diagnostic_report.md`
- `reports/execspec_exact_init_mismatch_replay_report.md`
- `reports/execspec_state2_calibrated_repair.md`
- `reports/execspec_state3_replay_validation.md`
- `reports/execspec_state3_5_baseline_dominance_audit.md`

## Reusable Tests

- `tests/test_execspec_mismatch_diagnostic.py`
- `tests/test_execspec_exact_init_mismatch_replay.py`
- `tests/test_execspec_repair.py`
- `tests/test_execspec_replay_validation.py`
- `tests/test_execspec_baseline_dominance_audit.py`

## Reusable Method Lessons

- Exact-init expert replay is a useful first simulator/data compatibility gate.
- Wrong-spec replay is a useful way to create controlled execution failures.
- Every new route should include strong simple baselines before scaling.
- A method that only beats identity or no-method is not yet interesting.
- The best single simple baseline must be reported, not hidden inside ablations.
- Default-reset and deployment evidence must be separated from exact-init replay evidence.

## Not Reusable As Claims

Do not reuse these as positive paper claims:

- "ExecSpec-Repair beats simple baselines" as a broad statement.
- "Mismatch-aware routing is meaningful" under STATE 3.5 evidence.
- "Exact-init recovery implies rollout success."
- "Calibration success implies executable-spec novelty."

## Recommended Reuse In The Next Topic

Use the artifacts to build early gates:

- 48-hour direct control/replay metric gate,
- 72-hour simple-baseline dominance gate,
- report-only baseline audit before any scaling,
- route-kill summary if the simple baseline catches up.

