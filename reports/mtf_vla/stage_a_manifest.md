# MTF-VLA Stage A Manifest

Date: `2026-07-14 KST`

Final decision: `MTF_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`

- method: `MTF-VLA`
- config: `mtf_r20_ret100`
- proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`
- policies: `frozen_smolvla, frameskip_proxy_lora, uniform_retained_ratio_lora, mtf_no_retention_ablation, mtf_full`
- reset seeds: `[20261201, 20261202]`
- paired cases per policy: `10`
- planned episodes: `50`
- canonical payload sha256: `1BB86A8060F8CD057AF984423021CA582E87661CB5157C072EF34B6F587739E3`

## Tasks

- `libero_spatial/task_0`: pick up the black bowl between the plate and the ramekin and place it on the plate
- `libero_spatial/task_8`: pick up the black bowl next to the plate and place it on the plate
- `libero_object/task_6`: pick up the butter and place it in the basket
- `libero_goal/task_4`: put the bowl on top of the cabinet
- `libero_10/task_2`: turn on the stove and put the moka pot on it

## Frozen Rules

- five policies only: frozen SmolVLA, FrameSkip proxy, uniform retained-ratio LoRA, no-retention ablation, MTF full
- `frameskip_proxy_lora` is a faithful local proxy, not an official FrameSkip reproduction
- task/reset pairs are identical across policies and duplicate evaluation keys are zero
- policy order does not choose or perturb the reset identities; every episode uses `env.reset(seed=[reset_seed])`
- official LeRobot/LIBERO success condition is the primary closed-loop outcome
- no confirmatory-test tuning or checkpoint selection from Stage A outcomes
- exact matched task/reset pairs across all policies
- small differences, ties, and one- or two-episode gaps advance to Stage B
- permanent Stage A kill only under the preregistered catastrophic criteria

## Execution

- partial result path: `reports/mtf_vla/stage_a_partial_result.json`
- final result path: `reports/mtf_vla/stage_a_result.json`
- resume only missing `(policy, suite, task_id, reset_seed)` keys
