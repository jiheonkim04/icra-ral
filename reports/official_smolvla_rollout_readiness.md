# Official SmolVLA Rollout Readiness

Date: 2026-07-10 KST

Overall status: `NOT_READY`

Final decision: `LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`

This readiness check used local metadata, package inspection, CLI help, and source inspection only. It did not run training, GPU inference, simulator rollout, downloads, or experiments.

## Readiness Matrix

| Area | Status | Evidence |
| --- | --- | --- |
| Model revision | `READY` | `lerobot/smolvla_libero` locked to `31d453f7edd78c839a8bbc39744a292686daf0de` from local HF metadata |
| Dataset revision | `READY` | `lerobot/libero` locked to `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` from local HF metadata |
| Package versions | `PARTIAL` | versions pinned, package source commits unavailable from local wheel metadata |
| Baseline naming | `READY` | canonical policy names frozen in `reports/official_smolvla_baseline_naming_policy.md` |
| LoRA checkpoint persistence | `BLOCKED` | seeds `11`, `22`, and `33` have prediction JSONs but no persisted official adapter bundles |
| Action semantics | `READY_AS_SPEC` | official eval source order inspected and static-mix queue semantics frozen |
| Official eval entrypoint | `FOUND` | `lerobot-eval` maps to `lerobot.scripts.lerobot_eval:main` and lists `--env.type=libero` |
| Official eval dependencies | `BLOCKED` | local env lacks `libero` and `robosuite` |
| Native Windows official rollout | `UNPROVEN` | WSL/Linux or dependency fix required before rollout |
| Closed-loop rollout results | `NOT_RUN` | no official LIBERO success-rate rollout has been executed |

## Official Eval Env Status

Classification: `MISSING_OFFICIAL_EVAL_DEPENDENCY`

Reasons:

- local LeRobot help exposes `--env.type=libero`;
- local LeRobot source imports the official LIBERO env path;
- local Python environment does not have the `libero` package installed;
- local Python environment does not have `robosuite` installed;
- official native Windows rollout readiness remains unproven.

## Seed Checkpoint Status

| Seed | Status |
| --- | --- |
| `11` | `CHECKPOINT_MISSING` |
| `22` | `CHECKPOINT_MISSING` |
| `33` | `CHECKPOINT_MISSING` |

Prediction artifacts cannot replace these missing adapter checkpoint bundles.

## Rollout Readiness Conclusion

The protocol is now specific enough to execute later, but execution must stop before rollout until adapter bundles and official eval dependencies are fixed. The correct current decision is `LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`.
