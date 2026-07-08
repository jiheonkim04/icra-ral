# ContactTube-Aug Reusable Artifacts

Keep these as diagnostic infrastructure, not as evidence that ContactTube-Aug is RA-L-stable.

## Code

- `tca_map/contacttube_aug/state1_smoke.py`
  - contact-tube extraction from EEF, gripper, and object traces,
  - runtime object trace collection during bounded replay,
  - exact/default-reset replay variant evaluation,
  - contact-tube preservation metrics,
  - controller-valid action rate and clip-rate diagnostics,
  - random action jitter, random pose jitter, raw replay, simple object-relative retargeting, and ContactTube-Aug variants.
- `scripts/180_contacttube_aug_state1_smoke.ps1`
  - ungated extraction/report path,
  - gated WSL bounded replay path under `ALLOW_CONTACTTUBE_AUG_STATE1=1`.
- `tests/test_contacttube_aug_state1_smoke.py`
  - synthetic HDF5 tube extraction,
  - case construction,
  - simple-baseline kill behavior,
  - task-local gate behavior.

## Reports

- `reports/contacttube_aug_state1_result.md`
- `reports/contacttube_aug_state1_result.json`
- `reports/contacttube_aug_task_definition.md`
- `reports/contacttube_aug_experiment_plan.md`
- `reports/contacttube_aug_kill_criteria.md`
- `reports/contacttube_aug_autopilot_state.md`

## Reuse Guidance

- Reuse contact-tube extraction to audit future data-augmentation ideas.
- Reuse runtime object trace collection when HDF5 object trajectories are missing.
- Reuse augmentation-validity diagnostics before any training.
- Keep random jitter and simple object-relative retargeting as mandatory early baselines for any future contact-preserving augmentation topic.
- Do not reuse the current ContactTube-Aug action generator as a training-data producer unless a new predeclared version first passes controller-validity and simple-retarget gates.

