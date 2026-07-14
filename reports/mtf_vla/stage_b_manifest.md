# MTF-VLA Stage B Manifest

Date: `2026-07-14 KST`

Final decision: `MTF_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`

- method: `MTF-VLA`
- config: `mtf_r20_ret100`
- proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`
- policies: `frozen_smolvla, frameskip_proxy_lora, uniform_retained_ratio_lora, mtf_no_retention_ablation, mtf_full`
- reset seeds: `[20261203, 20261204]`
- paired cases per policy: `40`
- planned episodes: `200`
- canonical payload sha256: `3C9D9CCF835A3B9753B81C320E9390EC9DA516514563E4850C1DC4F19ACC5743`
- Stage A decision: `MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`

## Tasks

- `libero_spatial/task_0`: pick up the black bowl between the plate and the ramekin and place it on the plate
- `libero_spatial/task_2`: pick up the black bowl from table center and place it on the plate
- `libero_spatial/task_4`: pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
- `libero_spatial/task_6`: pick up the black bowl next to the cookie box and place it on the plate
- `libero_spatial/task_8`: pick up the black bowl next to the plate and place it on the plate
- `libero_object/task_0`: pick up the alphabet soup and place it in the basket
- `libero_object/task_2`: pick up the salad dressing and place it in the basket
- `libero_object/task_4`: pick up the ketchup and place it in the basket
- `libero_object/task_6`: pick up the butter and place it in the basket
- `libero_object/task_8`: pick up the chocolate pudding and place it in the basket
- `libero_goal/task_0`: open the middle drawer of the cabinet
- `libero_goal/task_2`: put the wine bottle on top of the cabinet
- `libero_goal/task_4`: put the bowl on top of the cabinet
- `libero_goal/task_6`: put the cream cheese in the bowl
- `libero_goal/task_8`: put the bowl on the plate
- `libero_10/task_0`: put both the alphabet soup and the tomato sauce in the basket
- `libero_10/task_2`: turn on the stove and put the moka pot on it
- `libero_10/task_4`: put the white mug on the left plate and put the yellow and white mug on the right plate
- `libero_10/task_6`: put the white mug on the plate and put the chocolate pudding to the right of the plate
- `libero_10/task_8`: put both moka pots on the stove

## Frozen Rules

- all 20 official tasks are included
- five policies only: frozen SmolVLA, FrameSkip proxy, uniform retained-ratio LoRA, no-retention ablation, MTF full
- `frameskip_proxy_lora` is a faithful local proxy, not an official FrameSkip reproduction
- task/reset pairs are identical across policies and duplicate evaluation keys are zero
- reset identities are fresh relative to Stage A
- official LeRobot/LIBERO success condition is the primary closed-loop outcome
- no confirmatory-test tuning or checkpoint selection from Stage A or Stage B outcomes
- one expansion to 80 paired episodes is allowed only if Stage B is genuinely unresolved

## Execution

- partial result path: `reports/mtf_vla/stage_b_partial_result.json`
- final result path: `reports/mtf_vla/stage_b_result.json`
- resume only missing `(policy, suite, task_id, reset_seed)` keys
