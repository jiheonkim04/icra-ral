# DAGR-VLA Stage B Manifest

Date: `2026-07-14 KST`

Final decision: `DAGR_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`

- method: `DAGR-VLA`
- config: `dagr_a020_route_mlp`
- proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`
- policies: `frozen_smolvla, dam_static_component_proxy, dagr_full, dagr_no_dynamic_route_ablation, gripper_transition_heuristic`
- reset seeds: `[20261207, 20261208]`
- paired cases per policy: `40`
- planned episodes: `200`
- canonical payload sha256: `2A14FA11271EC8FAD9BD91A1251952E9039A5BD297105BEBB78E27EFC4470A3B`
- Stage A decision: `DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`

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

- all 20 official tasks are used
- reset seeds are fresh relative to DAGR Stage A
- task/reset pairs are identical across policies and duplicate evaluation keys are zero
- `dam_static_component_proxy` remains a faithful transparent local proxy, not an official DAM-VLA reproduction
- no confirmatory-test tuning or checkpoint selection from Stage A or Stage B outcomes
- one expansion to 80 paired episodes is allowed only if Stage B is genuinely unresolved

## Execution

- partial result path: `reports/dagr_vla/stage_b_partial_result.json`
- final result path: `reports/dagr_vla/stage_b_result.json`
- resume only missing `(policy, suite, task_id, reset_seed)` keys
