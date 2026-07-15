# Project State

## 2026-07-15 Epoch 4 Cycle 14 Current State

Active governance: `reports/current_research_governance.md`

Current branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `RAR_STAGE_0_STOP_DESIGN_FAILURE_CONTINUE_CYCLE_14`

Current epoch: `4`

Current cycle: `14`

G3P-VLA remains stopped before rollout as `DATA_OR_SUPERVISION_FAILURE`; do not rescue it by changing material-point labels, thresholds, source gates, validation search, or Stage 0 criteria.

Epoch 4 Cycle 12 generated exactly three candidates in `reports/epoch_4_cycle_12_candidate_generation.md` after the prior mechanism map in `reports/epoch_4_cycle_12_prior_mechanism_map.md` and selected `CALA-VLA`, Context-Gated Action-Latent Adapter for frozen SmolVLA.

CALA is anchored to CAC-VLA and tests whether a Base-preserving, zero-initialized context-gated latent-action interface can improve SmolVLA beyond Base, a CAC-style proxy, a no-context-gate ablation, and a task-mean latent-action baseline. Future 7D action segments are training labels only; confirmatory inference may not use future actions or privileged state.

The CALA Researcher A proposal is frozen in `reports/cala_vla/researcher_proposal.md` with proposal hash `5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76`.

Reviewer B attack is complete in `reports/cala_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

Researcher A rebuttal is complete in `reports/cala_vla/researcher_rebuttal.md` with decision `CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The CALA mathematical mechanism audit is frozen in `reports/cala_vla/mathematical_mechanism_audit.md` with decision `CALA_MATHEMATICAL_AUDIT_PREREGISTERED`.

The preregistration and prototype protocol are frozen in `reports/cala_vla/preregistration.md` and `reports/cala_vla/prototype_protocol.md`.

CALA Stage 0 is complete in `reports/cala_vla/development_audit.json` and `reports/cala_vla/development_audit.md` with final decision `DESIGN_FAILURE`. The hard stop was latent predictability: the full deployment-observable probe margin was `-0.01171824382857035`, and `action_history_only` beat it. No training, validation search, rollout, or confirmatory-test tuning happened.

Epoch 4 Cycle 13 generated exactly three candidates in `reports/epoch_4_cycle_13_candidate_generation.md` and selected `RAR-VLA`, Re-Anchored Autoregressive Residuals for frozen SmolVLA, anchored to AR-VLA.

The RAR-VLA Researcher A proposal is frozen in `reports/rar_vla/researcher_proposal.md` with proposal hash `723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56`.

Reviewer B attack is complete in `reports/rar_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

Researcher A rebuttal is complete in `reports/rar_vla/researcher_rebuttal.md` with decision `RAR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The mathematical mechanism audit is frozen in `reports/rar_vla/mathematical_mechanism_audit.md` with decision `RAR_MATHEMATICAL_AUDIT_PREREGISTERED`.

The preregistration and prototype protocol are frozen in `reports/rar_vla/preregistration.md` and `reports/rar_vla/prototype_protocol.md`.

RAR Stage 0 stopped as `DESIGN_FAILURE` in `reports/rar_vla/development_audit.json`: the residual predictability margin was `-0.03837609884238533`, with `zero_residual` beating the full legal causal probe. No training, validation search, rollout, or confirmatory-test tuning happened.

Exact next step: generate exactly three distinct Epoch 4 Cycle 14 candidates under current governance. Do not rescue RAR.

## 2026-07-13 Governance V2 Current State

Active governance: `reports/current_research_governance.md`

Current branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `EPOCH_4_CYCLE_1_RCV_KILLED_CONTINUE_CYCLE_2`

Current epoch: `4`

Current cycle: `2`

The previous fixed-cycle no-method stop is procedurally invalid under the active Goal. Epoch 1 is archived in `reports/epoch_1_corrected_adjudication.md`. Epoch 2 Cycle 1 `PTC-VLA` is archived in `reports/epoch_2_cycle_1_ptc_adjudication.md`. Epoch 2 Cycle 2 `SACF-VLA` is archived in `reports/epoch_2_cycle_2_sacf_adjudication.md`. Epoch 2 Cycle 3 `OCFN-VLA` is archived in `reports/epoch_2_cycle_3_ocfn_adjudication.md`. Epoch 3 Cycle 1 `CBFD-VLA` is archived in `reports/epoch_3_cycle_1_cbfd_adjudication.md`. Epoch 3 Cycle 2 `SCVC-VLA` is archived in `reports/epoch_3_cycle_2_scvc_adjudication.md`. Epoch 3 Cycle 3 `PSE-VLA` is archived in `reports/epoch_3_cycle_3_pse_adjudication.md`. Epoch 3 is synthesized in `reports/epoch_3_failure_synthesis.md`. Epoch 4 Cycle 1 `RCV-VLA` is archived in `reports/epoch_4_cycle_1_rcv_adjudication.md` as a valid current-formulation kill: the full method reached `20 / 40`, but the no-context ablation and stateless baseline each reached `24 / 40`. Epoch 4 Cycle 2 must continue under the post-PSE problem-first, external-prior-early, mathematically justified research-design gate and change at least two core dimensions relative to RCV.

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

## 2026-07-11 ECHO-VLA First Prototype Headroom Gate

Current decision: `NO_ECHO_CANDIDATE_HEADROOM`

Branch: `codex/implement-echo-vla-first-prototype`

Starting main commit: `5fcc87b93b627dbf09eb69676801e4412909bda4`

The targeted novelty adjudication passed only under the narrowed ECHO claim: phase-conditioned explicit physical-effect mediation learned from same-state action interventions and used for pre-execution candidate credit/ranking. ECHO was compared against Reflective VLA, Action-Effect Memory, Causal World Modeling / LingBot-VA, Pre-VLA, CoVer, Move-Then-Operate, Dream2Fix, and VLA-Corrector.

The same-state counterfactual protocol and effect schema were frozen, then a bounded candidate-headroom gate ran on official SmolVLA-LIBERO through WSL/CUDA. The gate used two initial tasks (`libero_spatial/task_0`, `libero_object/task_4`), two reset identities (`20260711`, `20260712`), `K=4` candidates, and horizon `4`. It generated `4` same-state intervention groups and `16` candidate records. All `4/4` group identity proofs passed, and non-gripper effect labels were populated for EEF displacement and target-distance change.

Oracle candidate selection did not improve over the default candidate:

- default success rate: `0.0`
- oracle success rate: `0.0`
- oracle improvement: `0.0` percentage points
- default-failure recoverable rate: `0.0`

No ECHO heads were trained and no closed-loop ECHO evaluation was run because the predeclared headroom kill fired before training.

Key reports:

- `reports/echo_vla_targeted_novelty_adjudication.md`
- `reports/echo_vla_counterfactual_data_protocol.md`
- `reports/echo_vla_effect_predicate_schema.md`
- `reports/echo_vla_candidate_headroom_result.md`
- `reports/echo_vla_first_prototype_plan.md`
- `reports/echo_vla_first_prototype_result.md`
- `reports/echo_vla_first_prototype_result.json`
- `reports/echo_vla_first_prototype_decision.md`

Exact next step: stop ECHO implementation under this candidate-generation protocol. Reopen only with a redesigned candidate generator or effect representation and a new predeclared headroom gate; do not train ECHO heads on the current no-headroom candidate set.

## ECHO Final Candidate Headroom Gate - 2026-07-11

- branch: `codex/echo-vla-final-candidate-headroom-gate`
- decision: `NO_ECHO_HEADROOM_CONFIRMED`
- official groups/candidates: `12` / `96`
- structured diagnostic candidates: `96`
- training happened: `False`
- OpenVLA used: `False`

## CensorCredit One-Repair Gate and Final Method - 2026-07-12

- branch: `codex/censorcredit-one-repair-and-final-method`
- base commit: `1f29a422945350e33ba3be0cb6150054735c49f6`
- CensorCredit exact diagnosis: `LABEL_OR_DATA_FAILURE`
- CensorCredit repair decision: `CENSORCREDIT_NO_VALID_REPAIR`
- CensorCredit repair attempted: `False`
- final method candidate: `Intervention-Set Action-Chunk Fine-Tuning (ISAC-VLA)`
- final method status: `FINAL_METHOD_KILLED_BEFORE_IMPLEMENTATION`
- final campaign decision: `NO_VALID_CENSORCREDIT_REPAIR_FINAL_METHOD_KILLED`
- main updated: `False`

Evidence:

- CensorCredit labels collapsed: `24/24` rows had matching censored and uncensored labels.
- Censored and uncensored learned weights were identical.
- The final distinct method was killed by near-exact prior-art overlap with SDP/TORL-VLA/ConRFT and by unavailable paired intervention/correction chunk data.

Key reports:

- `reports/censor_credit_exact_failure_diagnosis.md`
- `reports/censor_credit_repair_result.json`
- `reports/final_distinct_method_proposal.md`
- `reports/final_distinct_method_result.json`
- `reports/final_autonomous_method_decision.md`

## Implementation V2 Empirical Postmortem - 2026-07-12

- branch: `codex/implementation-v2-empirical-postmortem`
- base implementation commit: `1ff7e4d420dddae290105b07f8cd03acc987e123`
- final decision: `PROTOTYPE_EVIDENCE_INSUFFICIENT_FOR_TERMINAL_CLAIM`
- rollouts rerun: `False`
- training rerun: `False`
- thresholds changed: `False`
- main updated: `False`

Postmortem classifications:

- `PhaseBarrier-VLA`: `UNDERPOWERED_PROTOTYPE_INCONCLUSIVE`, because full PhaseBarrier changed actions but every variant scored `0/2` and the training positives were short-horizon effect-compatibility labels, not task-success labels.
- `CensorCredit-VLA`: `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, because censored and uncensored generated labels were identical for `24/24` records, yielding identical learned heads and a full method that collapsed to the uncensored ablation.

Key reports:

- `reports/phase_barrier_empirical_postmortem.md`
- `reports/censor_credit_empirical_postmortem.md`
- `reports/two_method_failure_comparison.md`
- `reports/final_method_mechanism_synthesis.md`
- `reports/final_method_decision.md`

Exact next step: preserve this postmortem branch. Do not treat `TWO_IMPLEMENTED_METHODS_KILLED` as a genuine two-method terminal scientific claim without a new, explicitly approved follow-up.

## PhaseBarrier Bounded Adjudication - 2026-07-12

- branch: `codex/phasebarrier-bounded-adjudication`
- base postmortem commit: `9620d1b5bea2555fe44bac2b8880a1d798699433`
- final decision: `PHASEBARRIER_COMPONENT_NOT_USEFUL`
- valid episodes: `100/100`
- training rerun for valid result: `False`
- original PhaseBarrier weights reused: `True`
- invalid retrained run preserved: `reports/phase_barrier_bounded_repair_invalid_retrained_result.json`
- main updated: `False`

Key outcome:

- frozen SmolVLA: `8/20`
- simple global damping: `0/20`
- no-phase ablation: `9/20`
- full PhaseBarrier: `0/20`

PhaseBarrier is archived and must not be rescued. The phase-conditioned component acted but was beaten by the no-phase ablation.

Key reports:

- `reports/phase_barrier_bounded_repair_plan.md`
- `reports/phase_barrier_power_and_sample_plan.md`
- `reports/phase_barrier_bounded_repair_manifest.json`
- `reports/phase_barrier_bounded_repair_result.json`
- `reports/phase_barrier_bounded_repair_result.md`
- `reports/phase_barrier_bounded_repair_decision.md`

Exact next step: do not continue PhaseBarrier. CensorCredit remains a documented implementation failure, but no CensorCredit repair was performed in this run.

## Autonomous Dual-Review RA-L Campaign - 2026-07-11

- branch: `codex/autonomous-dual-review-ral-research`
- final decision: `NO_METHOD_AFTER_3_VALID_CYCLES`
- cycles killed: `3`
- new downloads: `0 GiB`
- active GPU time in this batch: `0 h`
- paper-ready package produced: `False`
- main updated: `False`

Cycle outcomes:

- Cycle 01 action conditioning / representation: killed by CAC-VLA, ACoT-VLA, LaRA-VLA, ActionMap, LARA/LAWM/AEM proximity plus local ECHO/ActionMap no-headroom evidence.
- Cycle 02 intervention-censored correction credit: killed by TORL-VLA, SDP, AFIL, BORA, VLA-Corrector, Pre-VLA proximity plus missing robot/human/tactile data.
- Cycle 03 contact barrier / irreversibility: killed by VeriSpace, Pre-VLA, VLA-Corrector, AAC, SEAM, Legato, TORL-VLA proximity plus local contact/geometry baseline kills and non-cross-backbone hard-slice evidence.

Key reports:

- `reports/autonomous_campaign_state.md`
- `reports/autonomous_cycle_01_action_conditioning_kill.md`
- `reports/autonomous_cycle_02_censored_correction_kill.md`
- `reports/autonomous_cycle_03_contact_barrier_kill.md`
- `reports/autonomous_campaign_final_decision.md`

## Autonomous RA-L Research Implementation V2 - 2026-07-11

- branch: `codex/autonomous-ral-research-implementation-v2`
- previous terminal decision reclassified as: `PREMATURE_LITERATURE_ONLY_TERMINATION`
- final decision: `TWO_IMPLEMENTED_METHODS_KILLED`
- main updated: `False`
- new downloads: `0 GiB`

Implemented cycles:

- `PhaseBarrier-VLA`: trained a phase-conditioned feasibility-field action projection and ran closed-loop SmolVLA-LIBERO evaluation; final decision `PHASE_BARRIER_VALID_KILL`.
- `CensorCredit-VLA`: trained censored and uncensored temporal-credit action-history wrappers and ran closed-loop SmolVLA-LIBERO evaluation; final decision `CENSOR_CREDIT_VALID_KILL`.

Key reports:

- `reports/implementation_v2_reclassification.md`
- `reports/phase_barrier_vla_exact_overlap_matrix.md`
- `reports/phase_barrier_vla_prototype_result.json`
- `reports/censor_credit_vla_prototype_result.json`
- `reports/implementation_v2_campaign_state.json`
- `reports/implementation_v2_final_decision.md`

## ECHO Final Candidate Headroom Gate - 2026-07-11

- branch: `codex/echo-vla-final-candidate-headroom-gate`
- decision: `NO_ECHO_HEADROOM_CONFIRMED`
- official groups/candidates: `12` / `96`
- structured diagnostic candidates: `96`
- training happened: `False`
- OpenVLA used: `False`
