# DAGR-VLA Stage A Manifest

Date: `2026-07-14 KST`

Final decision: `DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`

- method: `DAGR-VLA`
- config: `dagr_a020_route_mlp`
- proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`
- policies: `frozen_smolvla, dam_static_component_proxy, dagr_full, dagr_no_dynamic_route_ablation, gripper_transition_heuristic`
- reset seeds: `[20261205, 20261206]`
- paired cases per policy: `10`
- planned episodes: `50`
- canonical payload sha256: `8379E47D3C3C73E21ADDD285491750E7406B8389578C0003278E5E187EA27E7B`

## Tasks

- `libero_spatial/task_0`: pick up the black bowl between the plate and the ramekin and place it on the plate
- `libero_spatial/task_8`: pick up the black bowl next to the plate and place it on the plate
- `libero_object/task_6`: pick up the butter and place it in the basket
- `libero_goal/task_4`: put the bowl on top of the cabinet
- `libero_10/task_2`: turn on the stove and put the moka pot on it

## Frozen Rules

- five policies only: frozen SmolVLA, DAM-style static component proxy, DAGR full, no-dynamic-route ablation, and gripper-transition heuristic
- `dam_static_component_proxy` is a faithful transparent local proxy, not an official DAM-VLA reproduction
- task/reset pairs are identical across policies and duplicate evaluation keys are zero
- policy order does not choose or perturb reset identities
- official LIBERO success condition is the primary closed-loop outcome
- no confirmatory-test tuning or checkpoint selection from Stage A outcomes
- small differences, ties, and one- or two-episode gaps advance to Stage B
- permanent Stage A kill only under the preregistered catastrophic criteria

## Execution

- partial result path: `reports/dagr_vla/stage_a_partial_result.json`
- final result path: `reports/dagr_vla/stage_a_result.json`
- resume only missing `(policy, suite, task_id, reset_seed)` keys
