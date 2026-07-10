# Official LIBERO Rollout Pilot Plan

Date: 2026-07-10 KST

## Predeclared Scope

Run a bounded official closed-loop baseline pilot only after all four smoke policies execute.

## Policies

1. `frozen_base`
2. `rank4_lora_seed_11`
3. `rank4_lora_seed_22`
4. `rank4_lora_seed_33`

Static-mix duplicate policies were not run because canonical validation-selected alpha is exactly `0.0` for seeds 11, 22, and 33. They are recorded as `DEGENERATE_EQUIVALENT_TO_FROZEN_BASE`.

## Tasks

- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- Task ID: `0` in each suite
- Episodes: `3` per task per policy
- Planned total: `4 suites x 1 task x 3 episodes x 4 policies = 48 episodes`

## Execution Rules

- Same reset/evaluation seed across policies: `20260710`
- Batch size: `1`
- Max parallel tasks: `1`
- Control mode: official relative control
- No seed selection after rollout
- No static-alpha tuning
- No old custom `LIBERO_7D` route
- No full benchmark
- No retraining or checkpoint regeneration

## Primary And Secondary Metrics

- Primary: official task success rate
- Secondary: per-suite/task success, reward, exception count, CUDA memory, runtime, action validity, and failure patterns visible from official metrics/logs.
