# Task75 Second-Prior Result

Decision: `TASK75_SECOND_PRIOR_INFRASTRUCTURE_BLOCKED`

`libero_90/task_75`, reset identity `20260725`, remains a valid local diagnostic thread, but no valid second-prior policy result was produced. This is not a policy failure and does not authorize Ours design or training.

## Matched Task75 Evidence

- X-VLA first prior failed cleanly: `runs/xvla_prior/failure_scan_libero90_identity20260725_tasks70_89_post_noheadroom_20260718T003659KST/task_75/result.json`
- SmolVLA Base failed cleanly: `runs/xvla_prior/diagnostic_smolvla_base_libero90_task75_id20260725_officialenv_20260718T004412KST/result.json`
- Task-level expert headroom is positive, but same-reset HDF5 headroom is unavailable: `runs/xvla_prior/diagnostic_libero90_task75_expert_headroom_20260725_20260718T004553KST/result.json`
- Preservation manifest: `reports/task75_local_evidence_manifest.json`

## Second-Prior Preflight

Preferred prior, Quantized OpenVLA-OFT INT4, is unsupported for this exact gate. Its checkpoint statistics contain only:

- `libero_spatial_no_noops`
- `libero_object_no_noops`
- `libero_goal_no_noops`
- `libero_10_no_noops`

There is no `libero_90` or `libero_90_no_noops` key, so the official action unnormalization check would reject `libero_90`.

The next preregistered local prior, LightVLA-libero-10-4bit, is also unsupported for `libero_90`; its dataset statistics contain only `libero_10_no_noops`. RIPT-VLA and VLA-GSE were already resource/comparability blocked, and VLA-0/VLA-JEPA were unselected large-asset fallbacks with no local executable task75 checkpoint.

Preflight logs:

- `runs/task75_second_prior/infra_preflight_20260718T0115KST/stdout.log`
- `runs/task75_second_prior/infra_preflight_20260718T0115KST/stderr.log`
- `runs/task75_second_prior/infra_preflight_20260718T0115KST/exit_code.txt`
- `runs/task75_second_prior/infra_preflight_20260718T0115KST/heartbeat.txt`

No model was loaded, no rollout ran, no training happened, no optimizer step happened, no checkpoint was written, and no Ours design happened.

## Decision Boundary

Published aggregate references are not used as the kill threshold. The local matched task75 decision separates:

- `PUBLISHED_REFERENCE`: not used as threshold.
- `LOCAL_OFFICIAL_OR_QUANTIZED_PRIOR_RESULT`: no valid task75 second-prior rollout exists.
- `LOCAL_MATCHED_TASK75_RESULT`: Base and X-VLA fail; task-level headroom positive; same-reset headroom unavailable.

Action: task75 is not authorized for method generation. Resume official-prior-first search by selecting a new preregistered residual source, reset identity, or prior ecosystem with valid local official-prior support.
