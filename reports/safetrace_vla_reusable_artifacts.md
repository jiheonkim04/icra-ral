# SafeTrace-VLA Reusable Artifacts

Keep these artifacts as archive and diagnostic infrastructure, not as evidence that SafeTrace-VLA is RA-L-stable.

## Code

- `tca_map/safetrace_vla/diagnostic.py`
  - official/local source availability audit,
  - local LIBERO HDF5 temporal safety proxy reader,
  - temporal violation rate, risk exposure time, cumulative safety cost, monitor coverage, and safe/unsafe success fields,
  - oracle diagnostic preference-pair generation without eval success labels,
  - safety-only, stop-on-risk, clipping-only, generic DPO proxy, and SafeTrace proxy comparison,
  - exact `KILL` / `SOURCE_BLOCKED` / `CONTINUE_TO_STATE_2` decision output.
- `scripts/220_safetrace_vla_diagnostic.ps1`
  - bounded runner,
  - refusal of download, GPU, rollout, simulator, heavy import, runtime install, OpenVLA, and OpenVLA-OFT gates.
- `tests/test_safetrace_vla_diagnostic.py`
  - synthetic HDF5 temporal-safety fixture,
  - required kill behavior when safety-only matches,
  - runner JSON smoke.

## Reports

- `reports/safetrace_vla_state1_result.md`
- `reports/safetrace_vla_state1_result.json`
- `reports/safetrace_vla_task_definition.md`
- `reports/safetrace_vla_experiment_plan.md`
- `reports/safetrace_vla_kill_criteria.md`
- `reports/safetrace_vla_related_work_matrix.md`
- `reports/safetrace_vla_risk_register.md`
- `reports/safetrace_vla_decision_log.md`
- `reports/safetrace_vla_autopilot_state.md`

## Reuse Guidance

- Reuse the source-audit table before any future safety benchmark work.
- Reuse temporal violation rate, risk exposure time, cumulative safety cost, safe success, unsafe success, and monitor coverage as metric names.
- Keep safety-only/risk-only, stop-on-risk, clipping-only, reward-penalty, generic DPO/preference, and no-training baselines mandatory.
- Do not reuse local standard LIBERO proxy traces as paper-grade safety evidence.
- Do not start a custom SafeTrace-like method until an official safety benchmark/source reproduction is first working.

