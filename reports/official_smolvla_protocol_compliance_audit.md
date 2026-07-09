# Official SmolVLA Protocol Compliance Audit

Date: 2026-07-10 KST

Audit boundary: repository audit only; no experiments, training, GPU, downloads, rollout, OpenVLA-OFT, FCAR revival, or LoRA rerun.

## Compliance Summary

| area | status | evidence |
| --- | --- | --- |
| official model path | pass | final runners use `C:\assets\checkpoints\smolvla_libero` |
| official dataset path | pass | final runners use `C:\assets\datasets\lerobot_libero` |
| fixed manifest | pass | `reports/official_smolvla_split_manifest.json`, 2800 frames, episode-disjoint |
| metric protocol | pass | `reports/official_smolvla_metric_protocol.md` |
| CUDA training checks in final LoRA runs | pass | seed repro JSON records CUDA use and no CPU fallback |
| no downloads in final stable/seed runs | pass | final JSON policies record `downloads_performed: false` |
| no OpenVLA-OFT | pass | final JSON policies record `openvla_oft_executed: false` |
| no old custom route in final official runs | pass | final JSON policies record `old_custom_route_used: false` |
| static alpha validation-only | pass | selection split is validation; test tuning forbidden |
| artifact persistence | partial | prediction artifacts exist; adapter checkpoints are not persisted |
| HF revision pinning | fail for future reproducibility | local paths and hashes exist, source revisions missing |
| baseline naming | partial | current final result mostly says static mix / MoIRA-style, but older FCAR and some summaries use ambiguous names |
| official closed-loop evaluation | not done | explicitly deferred |
| official success-rate evaluation | not done | no simulator success rates |

## Scientific Status

Genuinely established:

- Official SmolVLA-LIBERO model, processor, pre/postprocessor, and LeRobot dataset can load locally.
- Rank-4 LoRA CUDA training runs on the official path without CPU fallback.
- A fixed episode-disjoint manifest and metric protocol exist.
- The stable prediction artifact and seed-specific prediction artifacts exist and match expected record counts.
- Under the fixed offline action-L2 protocol, validation-selected action-space static mix is the strongest realistic offline baseline across LoRA seeds `11`, `22`, and `33`.

Established only offline:

- action L2, translation L2, rotation L2, gripper error/sign, task-balanced metrics, win counts, and oracle headroom
- frame oracle headroom after static mix
- weakness of the local task/instruction router proxy

Still missing for RA-L evidence:

- official closed-loop LIBERO rollout
- official task success-rate evaluation
- full benchmark after rollout readiness
- pinned HF model and dataset revisions
- environment/package lock for exact reproduction
- corrected baseline names in future reports
- a policy on whether seed-specific LoRA adapter checkpoints must be saved

## Static Mixture Status

The current strongest realistic offline baseline is:

`validation_selected_action_space_static_mix`

Evidence:

- seed win count: `3` / `3`
- action L2 mean/std: `0.080616431` / `0.002595356`
- task win counts summed over seeds: static mix `93`, frozen/base `20`, rank-4 LoRA `7`

## Oracle Status

Frame oracle remains an upper bound only. It uses label knowledge and must not be presented as a deployable method.

Frame oracle headroom after static mix:

- mean: `0.011499227`
- min: `0.009974197`
- max: `0.013403030`

This is useful as a ceiling for future method planning, not as evidence of current method performance.

## Method Design Status

Method design should remain paused. FCAR remains killed. A future method gate must first beat `validation_selected_action_space_static_mix` under the fixed manifest and metric protocol, and must do so without oracle label access.

## Rollout Status

Official simulator rollout is the next major scientific milestone after protocol gaps are fixed. The audit does not authorize a rollout run by itself.

## Protocol Decision

The final audit decision is:

`AUDIT_FOUND_PROTOCOL_GAPS_FIX_BEFORE_ROLLOUT`

Reason: no result-invalidating leakage or artifact inconsistency was found, but revision pinning, naming correction, and checkpoint persistence policy should be fixed before official rollout or paper-facing claims.
