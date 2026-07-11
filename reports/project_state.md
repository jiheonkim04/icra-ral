# Project State

Date: 2026-07-10 KST

Target branch: `main`

Implementation branch: `codex/audit-smolvla-lora-regen-drift`

Current decision: `PROTOCOL_DRIFT_FOUND`

## Current Route

The valid route remains official SmolVLA/LeRobot reproduction first. No closed-loop rollout is safe from the current evidence.

This audit diagnosed the old-vs-regenerated LoRA metric drift after commit `15649d6`. It did not retrain any seed, install simulator dependencies, run rollout, download assets, run OpenVLA-OFT, revive FCAR, design a new method, relax the frozen `0.002` tolerance, or overwrite historical metrics.

## Locked Inputs

- model: `lerobot/smolvla_libero`
  - revision: `31d453f7edd78c839a8bbc39744a292686daf0de`
  - local path: `C:\assets\checkpoints\smolvla_libero`
- dataset: `lerobot/libero`
  - revision: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
  - local path: `C:\assets\datasets\lerobot_libero`
- split manifest: unchanged across `5d48b1e` and `15649d6`
- metric protocol: unchanged across `5d48b1e` and `15649d6`

## Checkpoint Status

All three regenerated persisted adapter bundles remain complete and checksum verified:

| Seed | Checkpoint path | Status |
| ---: | --- | --- |
| `11` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_11` | `CHECKPOINT_COMPLETE_VERIFIED` |
| `22` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_22` | `CHECKPOINT_COMPLETE_VERIFIED` |
| `33` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_33` | `CHECKPOINT_COMPLETE_VERIFIED` |

## Drift Audit Result

Frame/label/protocol alignment:

- test frame IDs: identical
- task and episode IDs: identical
- ground-truth actions: identical
- split membership: identical
- frozen/base predictions: identical
- metric protocol file: identical
- static-alpha grid: identical, validation-only selection

Repeated disk evaluation:

| Seed | max action L2 repeat diff | rank4 metric diff | static metric diff | selected alpha identical |
| ---: | ---: | ---: | ---: | --- |
| `11` | `0.0` | `0.0` | `0.0` | `True` |
| `22` | `0.0` | `0.0` | `0.0` | `True` |
| `33` | `0.0` | `0.0` | `0.0` | `True` |

The current persisted checkpoints are internally stable under disk re-evaluation. The drift is not evaluation nondeterminism.

However, the saved regenerated artifact metrics do not exactly match the fixed-seed disk re-evaluation metrics. This shows that the evaluation RNG state was also an unpinned part of the protocol identity.

## Root Cause

The old `5d48b1e` run evaluated the trained in-memory policy and did not persist/reload adapter weights. The regenerated `15649d6` run assigns the PEFT wrapper return, saves adapter bundles, reloads them with `PeftModel.from_pretrained`, and evaluates that disk identity. The audit's fixed-seed disk re-evaluation is repeatable but differs from the saved regenerated artifact metrics, so evaluation RNG state must also be pinned in any future protocol.

Historical adapter weights, complete RNG state, and exact training sample order were not preserved, so the historical learned policy identity cannot be reconstructed exactly. The regenerated persisted checkpoints must not be described as identical historical models.

## Canonical Baseline Status

Current regenerated checkpoints are **not accepted as canonical** in this audit because real LoRA prediction-protocol differences were found. They can only become canonical after the PEFT in-memory vs persisted-reload difference and the evaluation RNG-state policy are fixed or explicitly adjudicated as a new re-baselining decision that preserves old metrics as historical.

## Key Reports

- `reports/official_smolvla_lora_drift_audit.md`
- `reports/official_smolvla_old_vs_regen_config_diff.md`
- `reports/official_smolvla_artifact_alignment_audit.md`
- `reports/official_smolvla_eval_determinism_check.md`
- `reports/official_smolvla_training_determinism_status.md`
- `reports/official_smolvla_canonical_checkpoint_proposal.md`
- `reports/official_smolvla_lora_drift_decision.md`

## Exact Next Step

Fix or explicitly adjudicate the PEFT in-memory versus persisted-reload protocol difference and evaluation RNG-state policy before canonicalizing or rolling out.

## 2026-07-10 Canonicalization Update

Current decision: `NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT`

Canonical persisted-checkpoint offline evaluation passed with intermediate decision `CANONICAL_BASELINES_READY_FOR_ROLLOUT`. The run evaluated frozen base plus rank-4 LoRA seeds 11/22/33 on the fixed val/test manifest under action-generation eval seeds `[101, 202, 303, 404, 505]`; it did not train, regenerate checkpoints, download dependencies, run rollout, revive FCAR, or use the old custom LIBERO_7D route.

Native Windows rollout remains blocked because `hf-libero`, `libero`, and `robosuite` are not installed in the active env. The next step is WSL/Linux official LeRobot LIBERO smoke using the canonical artifacts.

## 2026-07-10 WSL Official Rollout Pilot

Current decision: `OFFICIAL_ROLLOUT_BASELINE_READY`

The official WSL/LeRobot/LIBERO path is now working. Smoke completed `4/4` episodes and the bounded pilot completed `48/48` episodes across frozen base plus rank-4 LoRA seeds 11/22/33. All policy audits showed parameters and inputs on `cuda:0`; no CPU fallback, schema/action mismatch, or old custom `LIBERO_7D` route was used.

Pilot overall success:

- `frozen_base`: `75.0%`
- `rank4_lora_seed_11`: `83.3%`
- `rank4_lora_seed_22`: `66.7%`
- `rank4_lora_seed_33`: `75.0%`

Lower offline action L2 did not predict higher closed-loop success in the pilot. This result is enough to unlock larger official baseline rollout/failure mining, but not enough for method selection or best-seed selection.

## 2026-07-11 Official Closed-Loop Scaleup

Current decision: `OFFLINE_ONLINE_MISMATCH_CONFIRMED`

The predeclared official WSL/LeRobot/LIBERO closed-loop scaleup completed `400/400` planned episodes with frozen base plus persisted rank-4 LoRA seeds `11`, `22`, and `33`. No policy was retrained, no static-mix duplicate rollout was run, no old custom `LIBERO_7D` route was used, and no method was implemented.

CUDA/route audit:

- WSL saw `NVIDIA GeForce RTX 5080`
- policy parameters: `cuda:0`
- input tensors: `cuda:0`
- action chunks: `cuda:0`
- autocast fp16/bf16 active: `false`
- episode peak VRAM: approximately `926.638` to `928.365` MB
- infrastructure failures: `0`

Policy success:

- `frozen_base`: `74/100`, `74.0%`
- `rank4_lora_seed_11`: `74/100`, `74.0%`
- `rank4_lora_seed_22`: `68/100`, `68.0%`
- `rank4_lora_seed_33`: `66/100`, `66.0%`

Suite-level difficulty across all policies:

- `libero_10`: `45/100`, `45.0%`
- `libero_goal`: `71/100`, `71.0%`
- `libero_spatial`: `79/100`, `79.0%`
- `libero_object`: `87/100`, `87.0%`

Failure status:

- unsuccessful episodes preserved: `118`
- automatic failure category count: `ambiguous_or_unclassified = 118`
- strongest weak task slice: `libero_10/task_4`, `5/20` successes
- strongest repeated all-policy failures include `libero_10/task_4/seed_20260713`, `libero_10/task_4/seed_20260715`, and `libero_spatial/task_4` on seeds `20260712`, `20260713`, and `20260714`

The run shows task/reset-structured failures, but not a reliable mechanism-linked failure phase because failure videos or semantic phase traces were not captured. Offline action L2 remains diagnostic-only: the LoRA-only offline ordering does not safely select the better closed-loop seed.

Key reports:

- `reports/official_closed_loop_scaleup_plan.md`
- `reports/official_closed_loop_task_manifest.json`
- `reports/official_closed_loop_episode_manifest.json`
- `reports/official_closed_loop_scaleup_result.md`
- `reports/official_closed_loop_scaleup_result.json`
- `reports/official_closed_loop_failure_annotations.json`
- `reports/official_closed_loop_failure_taxonomy.md`
- `reports/official_closed_loop_seed_robustness.md`
- `reports/official_closed_loop_offline_online_analysis.md`
- `reports/official_closed_loop_method_gap_decision.md`

Exact next step: inspect official videos for the bounded review queue, starting with repeated all-policy failures on `libero_10/task_4` and `libero_spatial/task_4`. Do not design a method unless the visual phase evidence identifies a repeated success-critical mechanism.

## 2026-07-11 Closed-Loop Visual Method Gate

Current decision: `NO_SAFE_RA_L_METHOD_YET`

The bounded official video review completed `24/24` selected same-identity reruns with `0` errors. No policy was trained or tuned, no full 400-episode rerun was launched, no static-mix duplicate rollout was used, and the old custom `LIBERO_7D` route was not used.

Identity and outcome status:

- same suite/task/policy/reset identity preserved: `true`
- original-vs-rerun success matches: `16/24`
- original-vs-rerun success flips: `8/24`
- video evidence type: bounded rerun evidence, not exact original-frame replay

Visual mechanisms:

- `libero_spatial/task_4`: drawer-contained black bowl extraction fails at `stable_grasp` / `contact_transition`; rerun failures on reset seeds `20260713` and `20260714` for all four policies.
- `libero_10/task_4`: two-mug, two-plate sequence fails as `long_horizon_compounding`; reset seed `20260715` failed for all four policies, while seed `20260713` had rerun success flips for seeds `22` and `33`.

The hard method gate did not pass. The strongest spatial mechanism has only two independent rerun-failure reset seeds, and the `libero_10` mechanism is not the same physical failure. Recent VLA work also kills generic confidence, verification, correction, adaptive chunking, progress/recovery, failure-negative, and adapter-routing formulations.

Key reports:

- `reports/closed_loop_failure_video_inventory.md`
- `reports/closed_loop_failure_visual_annotations.json`
- `reports/closed_loop_failure_mechanism_summary.md`
- `reports/latest_vla_method_landscape_2026.md`
- `reports/closed_loop_failure_vs_recent_work.md`
- `reports/ral_method_candidate_spec.md`
- `reports/ral_method_experiment_matrix.md`
- `reports/ral_method_kill_criteria.md`
- `reports/closed_loop_method_gate_decision.md`

Exact next step: do not implement a method. Only reopen method design after one visible mechanism is shown in at least two tasks or at least three independent reset seeds, and after a non-generic novelty claim survives the recent-work audit.
