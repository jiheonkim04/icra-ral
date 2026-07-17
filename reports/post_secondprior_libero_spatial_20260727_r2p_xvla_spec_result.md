# R2P-XVLA Frozen No-Training Spec

Decision: `R2P_XVLA_FROZEN_NO_TRAINING_SPEC_CREATED`

This step freezes the first `R2P-XVLA` configuration for `libero_spatial/task_5`, reset identity `20260727`. It does not train, run an optimizer step, write checkpoints, load a VLA model, launch a simulator, or evaluate Ours.

## Frozen spec artifact

- Builder: `tca_map/xvla_spatial_task5/training_spec.py`
- Focused tests: `tests/test_xvla_spatial_task5_training_spec.py`
- Ignored runtime snapshot: `runs/xvla_prior/epoch5_r2p_xvla_task5_training_spec_v1.json`
- Snapshot SHA-256: `d795dc72373f32d36cacd4b5b6a695607154d6f65c588d56e6bd010ef4312f78`

## Locked arms

| Arm | Role | Phase weights | Purpose |
| --- | --- | --- | --- |
| `r2p_xvla_rank8_phase_weights_lr1e4_steps64` | primary selected method | source `1.0`, transit `2.0`, target `1.5` | Test whether phase-balanced source-to-target supervision helps the bowl-on-ramekin to plate residual. |
| `uniform_task5_xvla_rank8_lambda0_lr1e4_steps64` | uniform-weight ablation | source `1.0`, transit `1.0`, target `1.0` | Test whether generic task-5 adaptation explains any gain. |

No third task-5 configuration is allowed after residual rollout. Retuning on identity `20260727` is forbidden.

## Locked safeguards

- Inference inputs: RGB, wrist RGB, proprioception, instruction.
- Privileged object positions at inference: `false`.
- Phase labels at inference: `false`.
- Closed-loop residual reset used for model selection: `false`.
- Paper claim from one identity: `false`.
- Same-reset expert headroom: unavailable; any future positive target result is diagnostic until independent condition evidence exists.

## Gates still closed

| Gate | Current value |
| --- | --- |
| X-VLA-format data adapter materialized | `false` |
| X-VLA-format data adapter smoke passed | `false` |
| One-batch gradient smoke passed | `false` |
| Optimizer step allowed before all gates | `false` |
| Training authorized now | `false` |
| Checkpoint write authorized now | `false` |
| Closed-loop Ours rollout authorized now | `false` |

## Validation

- `py_compile`: passed for the builder and focused test.
- Focused pytest: `3 passed`.
- Spec write command: `C:/Users/jiheo/miniconda3/envs/tca_map/python.exe -m tca_map.xvla_spatial_task5.training_spec`

## Next action

Materialize and validate a tiny X-VLA-format data-adapter smoke for `R2P-XVLA` without optimizer steps, checkpoint writes, downloads, simulator rollouts, or closed-loop Ours evaluation.
