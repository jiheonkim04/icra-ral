# RIFA-XVLA v1 Archive Decision

- Archive decision: `RIFA_XVLA_V1_ARCHIVED_NOT_STAGE_A_READY`
- FROZEN_PROTOCOL_DECISION: `RIFA_XVLA_STAGE0_DESIGN_FAILURE`
- CALIBRATED_SCIENTIFIC_INTERPRETATION: RIFA v1 is not Stage-A-ready because one binary gripper flip violated the frozen action-delta gate and the full-versus-no-reliability action difference was practically negligible despite technically exceeding the preregistered minimum.
- Postmortem decision: `RIFA_GRIPPER_POSTPROCESS_DISCONTINUITY_CONFIRMED`

The tested formulation had valid X-VLA integration, exact clean passthrough, and valid optimization. Its dropout full-versus-ablation mean action RMS was only `1.0597695498731683e-06`, with no full-versus-ablation gripper flip. It has no authorized closed-loop evidence.

The failing row crossed the fixed `0.5` gripper threshold at chunk `0`, action index `12`: Base `0.5001511573791504 -> +1`, full `0.49706852436065674 -> -1`, and no-reliability `0.49706581234931946 -> -1`.

RIFA v1 is archived without a renamed v1.1. This is a decision about the tested formulation, not a claim that the whole reliability-conditioned family is impossible. RIFA Stage 0 was not rerun, training was not extended, Stage A was not launched, and no broad prior or natural-reset search was reopened.
