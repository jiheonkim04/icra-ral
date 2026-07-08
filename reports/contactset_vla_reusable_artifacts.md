# ContactSet-VLA Reusable Artifacts

Keep these as diagnostic infrastructure, not as evidence that ContactSet-VLA is RA-L-stable.

## Code

- `tca_map/contactset_vla/diagnostic.py`
  - local LIBERO HDF5 action/EFF reader,
  - embedded MuJoCo XML free-joint and static body/site geometry extraction,
  - qpos-offset audit against HDF5 joint observations,
  - instruction-only source/destination phrase selector,
  - role-tagged point-set encoder,
  - tiny CPU NumPy ridge action-head variants,
  - held-out action metrics and contact/placement consistency proxies.
- `scripts/200_contactset_vla_diagnostic.ps1`
  - task-local `ALLOW_TINY_TRAINING=1` gate,
  - refusal of download, GPU, rollout, simulator, heavy import, runtime install, OpenVLA, and OpenVLA-OFT gates.
- `tests/test_contactset_vla_diagnostic.py`
  - synthetic HDF5 geometry/action fixtures,
  - required variant coverage,
  - runner gate behavior,
  - JSON report smoke.

## Reports

- `reports/contactset_vla_diagnostic_report.md`
- `reports/contactset_vla_diagnostic_report.json`
- `reports/contactset_vla_task_definition.md`
- `reports/contactset_vla_experiment_plan.md`
- `reports/contactset_vla_kill_criteria.md`
- `reports/contactset_vla_autopilot_state.md`

## Reuse Guidance

- Reuse the HDF5/XML geometry audit before any future action-head geometry method.
- Keep active single-point injection, source-only, destination-only, source+destination, and no-geometry as mandatory early baselines.
- Reuse the qpos-offset and leakage audit to avoid accidental simulator/eval-label target leakage.
- Do not reuse the full ContactSet-VLA encoder as a main method unless a new predeclared variant first beats the simple point/no-geometry baselines on a real held-out action metric.

