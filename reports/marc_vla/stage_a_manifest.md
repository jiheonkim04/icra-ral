# MARC-VLA Stage A Manifest

Date: `2026-07-15 KST`

Final decision: `MARC_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`

- method: `MARC-VLA`
- config: `marc_a020_gate_mlp`
- proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`
- policies: `frozen_smolvla, openvla_oft_l1_proxy, marc_full, marc_no_disagreement_gate_ablation, static_l1_mixture_baseline`
- reset seeds: `[20261209, 20261210]`
- paired cases per policy: `10`
- planned episodes: `50`
- canonical payload sha256: `3383E377CEDD2B44E7730AAD3617E64838786E7094B9CF60D39F9679DE97D74E`

## Tasks

- `libero_spatial/task_0`: pick up the black bowl between the plate and the ramekin and place it on the plate
- `libero_spatial/task_8`: pick up the black bowl next to the plate and place it on the plate
- `libero_object/task_6`: pick up the butter and place it in the basket
- `libero_goal/task_4`: put the bowl on top of the cabinet
- `libero_10/task_2`: turn on the stove and put the moka pot on it

## Frozen Rules

- five policies only: frozen SmolVLA, OpenVLA-OFT-style L1 proxy, MARC full, no-disagreement-gate ablation, and static L1 mixture
- `openvla_oft_l1_proxy` is a faithful transparent local proxy, not an official OpenVLA-OFT reproduction
- task/reset pairs are identical across policies and duplicate evaluation keys are zero
- policy order does not choose or perturb reset identities
- official LIBERO success condition is the primary closed-loop outcome
- no confirmatory-test tuning or checkpoint selection from Stage A outcomes
- small differences, ties, and one- or two-episode gaps advance to Stage B
- permanent Stage A kill only under the preregistered catastrophic, invalid-mechanism, or clear-dominance criteria

## Execution

- partial result path: `reports/marc_vla/stage_a_partial_result.json`
- final result path: `reports/marc_vla/stage_a_result.json`
- resume only missing `(policy, suite, task_id, reset_seed)` keys
