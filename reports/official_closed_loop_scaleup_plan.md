# Official Closed-Loop Scaleup Plan

Date: 2026-07-11 KST

## Frozen Design

- suites: `libero_spatial, libero_object, libero_goal, libero_10`
- task ids per suite: `[0, 2, 4, 6, 8]`
- reset seeds: `[20260711, 20260712, 20260713, 20260714, 20260715]`
- policies: `frozen_base`, `rank4_lora_seed_11`, `rank4_lora_seed_22`, `rank4_lora_seed_33`
- planned episodes: `400`
- batch size: `1`
- max parallel tasks: `1`
- control mode: official relative control
- static-mix duplicate rollouts: skipped because alpha is exactly `0.0`

## Manifest Hashes

- task manifest canonical payload sha256: `7ae264062b187e6f112ee31c0b1e541643cbe3a7625b174a288cee4e36ab2cbd`
- episode manifest canonical payload sha256: `9310ce47edfafc2b2f9f7129a6d1ca7ec149fbc54188b378e07d5777fdea47e4`

## Guardrails

No retraining, no seed selection after results, no static-mix duplicates, no old custom `LIBERO_7D` route, no exact-init replay bridge, no OpenVLA-OFT, and no rollout-outcome tuning.
