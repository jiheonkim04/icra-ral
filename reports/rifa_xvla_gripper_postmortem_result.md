# RIFA-XVLA Gripper Action-Semantics Postmortem

- Diagnostic decision: `RIFA_GRIPPER_POSTPROCESS_DISCONTINUITY_CONFIRMED`
- Execution valid: `True`
- FROZEN_PROTOCOL_DECISION: `RIFA_XVLA_STAGE0_DESIGN_FAILURE`
- CALIBRATED_SCIENTIFIC_INTERPRETATION: RIFA v1 is not Stage-A-ready because one binary gripper flip violated the frozen action-delta gate and the full-versus-no-reliability action difference was practically negligible despite technically exceeding the preregistered minimum.
- Chunk / action index: `0 / 12`

## Gripper signal at the flip

| policy | raw score | threshold margin | discrete action |
|---|---:|---:|---:|
| `BASE` | `0.5001511573791504` | `0.00015115737915039062` | `1.0` |
| `RIFA_XVLA` | `0.49706852436065674` | `-0.0029314756393432617` | `-1.0` |
| `RIFA_XVLA_NO_RELIABILITY` | `0.49706581234931946` | `-0.002934187650680542` | `-1.0` |

The `2.0` delta is a sign/threshold discontinuity: `True`.
Full and ablation make the same gripper decision: `True`.

No training, backward pass, optimizer step, checkpoint write, Stage 0 rerun, or closed-loop rollout occurred.
RIFA v1 remains closed and is not Stage-A-ready; this does not rule out the broader reliability-conditioned family.
