# Official SmolVLA / LeRobot Autopilot State

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-lerobot-baseline`

## State 0 - Official Recipe Scout

Status: complete.

Outputs:

- `reports/official_smolvla_lerobot_task_definition.md`
- `reports/official_smolvla_lerobot_recipe_scout.md`
- `reports/official_smolvla_lerobot_requirements.md`
- `reports/official_smolvla_lerobot_risk_register.md`
- `reports/official_smolvla_lerobot_kill_criteria.md`
- `reports/official_smolvla_lerobot_autopilot_state.md`

Finding: official SmolVLA base recipe is locally runnable; official LIBERO recipe requires 8D state / 7D action and Linux/MuJoCo for eval.

## State 1 - Environment And Asset Feasibility

Status: complete.

Outputs:

- `reports/official_smolvla_lerobot_env_status.md`
- `reports/official_smolvla_lerobot_asset_matrix.md`
- `reports/official_smolvla_lerobot_model_load_status.md`

State 1 finding: official SmolVLA base loader/processor mini-repro is feasible, but official LIBERO reproduction is still pending.

This is intentionally narrower than official LIBERO reproduction. The local checkpoint is not a LIBERO checkpoint.

## State 2 - Official Mini-Reproduction

Status: complete for SmolVLA base loader/processor smoke; blocked for official LIBERO baseline.

Executed:

- official LeRobot `SmolVLAPolicy.from_pretrained`;
- official LeRobot `make_pre_post_processors`;
- local tokenizer override to cached SmolVLM2 files;
- one synthetic CPU forward pass;
- no custom LIBERO 7D adapter;
- no training;
- no rollout;
- no downloads.

## State 3 - Decision

Status: complete.

Final decision: `NEEDS_OFFICIAL_DATASET_CONVERSION`

Reason: official SmolVLA base loads and runs, but the target LIBERO path needs either official `smolvla_libero` plus official LeRobot LIBERO assets, or a clean local HDF5-to-LeRobot conversion that preserves the 8D state / 7D action convention.
