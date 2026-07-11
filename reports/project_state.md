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

## 2026-07-11 Cross-Backbone Cross-Benchmark Gate

Current decision: `SECOND_BACKBONE_OR_BENCHMARK_BLOCKED`

The new gate selected a second backbone and second benchmark without downloading assets, training, rolling out, or implementing a method.

Selected second backbone:

- `OpenVLA-OFT`
- checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- checkpoint size: `14.845` GiB
- license/access: MIT, public, non-gated
- State 1 decision: `SECOND_BACKBONE_DOWNLOAD_APPROVAL_REQUIRED`

Selected second benchmark:

- `LIBERO-PRO`
- official repo: `https://github.com/Zxy-MLlab/LIBERO-PRO`
- dataset size from Hugging Face API: `1,090,523` bytes for BDDL/init files
- license/access: MIT, public, non-gated
- State 2 decision: `SECOND_BENCHMARK_READY_AFTER_SECOND_BACKBONE`

No cross-model rollout ran because the selected OpenVLA-OFT checkpoint is a large download and local 16GB VRAM inference feasibility is not proven. The predeclared protocol is frozen in `reports/cross_model_failure_manifest.json` for a later approved run.

Key reports:

- `reports/cross_backbone_candidate_audit.md`
- `reports/cross_benchmark_candidate_audit.md`
- `reports/openvla_oft_local_feasibility.md`
- `reports/second_vla_selection.md`
- `reports/second_benchmark_selection.md`
- `reports/cross_model_failure_manifest.json`
- `reports/cross_model_failure_result.md`
- `reports/cross_model_failure_result.json`
- `reports/cross_model_visual_annotations.json`
- `reports/cross_model_failure_generality.md`
- `reports/cross_model_latest_work_comparison.md`
- `reports/cross_model_method_readiness_decision.md`

Exact next step: request explicit approval for the OpenVLA-OFT checkpoint download and decide whether to run inference locally with memory/offload safeguards or on lab GPUs. Do not implement a method before cross-backbone and LIBERO-PRO evidence exists.

## 2026-07-11 RTX 5080 Quantized OpenVLA-OFT Cross-Backbone Gate

Current decision: `FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE`

The approved `14.845` GiB checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` was downloaded exactly once, checksummed, and evaluated only as quantized INT4 OpenVLA-OFT on the local RTX 5080. No training, fine-tuning, full BF16 load, RLDS download, LIBERO-PRO download, CPU offload, or disk offload occurred.

Hard-slice outcome: OpenVLA-OFT INT4 completed `20/20` exact-init episodes with `20/20` successes and videos. The matched SmolVLA frozen-base exact-init rerun completed `20/20` with `11/20` successes, including hard-slice failures on `libero_spatial/task_4` (`1/5`) and `libero_10/task_4` (`1/5`).

Conclusion: the SmolVLA stable-grasp and long-horizon failures were not reproduced in the second architecture under this bounded quantized OpenVLA-OFT gate. LIBERO-PRO is not justified by this result.

Key reports:

- `reports/openvla_oft_int4_download_status.md`
- `reports/openvla_oft_int4_environment_lock.md`
- `reports/openvla_oft_int4_memory_preflight.md`
- `reports/openvla_oft_int4_policy_load_result.md`
- `reports/openvla_oft_int4_int8_consistency.md`
- `reports/openvla_oft_quantized_hard_slice_manifest.json`
- `reports/openvla_oft_quantized_hard_slice_result.md`
- `reports/openvla_oft_quantized_hard_slice_result.json`
- `reports/openvla_oft_quantized_visual_annotations.json`
- `reports/openvla_oft_quantized_cross_backbone_decision.md`

Exact next step: do not implement a method and do not proceed to LIBERO-PRO from this evidence. Archive the cross-backbone result as failure-not-reproduced unless a future full-precision or different second-backbone run is explicitly approved.

## 2026-07-11 Paper-First VLA Method Design

Current decision: `READY_TO_IMPLEMENT_PRIMARY_VLA_METHOD`

This no-experiment, no-implementation goal performed a paper-first VLA robotics ideation study from primary sources up to 2026-07-11. No GPU, model inference, simulator execution, rollout, training, large download, or method implementation occurred.

The selected primary method is `ECHO-VLA: Counterfactual Action-Effect Credit for Closed-Loop Vision-Language-Action Manipulation`.

Reason: recent literature closes generic confidence, verification, progress, correction, chunking, failure-negative, prior-preservation, and adapter-routing routes, but still leaves a deeper action-objective versus closed-loop-success mismatch. ECHO-VLA targets that mismatch by estimating phase-conditioned interventional predicate effects of action chunks and using them for training/guidance.

Key reports:

- `reports/paper_first_vla_landscape_2026.md`
- `reports/vla_shared_assumption_analysis.md`
- `reports/vla_implicit_gap_synthesis.md`
- `reports/vla_method_candidate_portfolio.md`
- `reports/vla_method_novelty_adversarial_review.md`
- `reports/vla_primary_method_spec.md`
- `reports/vla_primary_method_first_experiment.md`
- `reports/vla_primary_method_full_experiment_matrix.md`
- `reports/vla_primary_method_kill_criteria.md`
- `reports/vla_paper_contribution_outline.md`
- `reports/vla_method_design_decision.md`

Exact next step: implement only the bounded first ECHO-VLA prototype on SmolVLA using the predeclared four-task predicate-diversity set and frozen kill criteria. Do not run OpenVLA-OFT INT4, full LIBERO, LIBERO-PRO, or a broad robustness sweep before the SmolVLA gate passes.
