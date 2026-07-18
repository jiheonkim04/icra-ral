# Autonomous Compact Handoff

Updated: 2026-07-19 KST

## Authoritative state

- Branch: `codex/epoch5-official-prior-first`
- Source HEAD before Pivot Epoch 1 selection: `f11ec2135f39dba3ccf315ee5271aa152247cff7`
- Current pushed HEAD before the A2C2 closure commit: `73288b6`
- Epoch/cycle: `5 / 0`
- Campaign state: `AUTONOMOUS_CAMPAIGN_ACTIVE_STRATEGIC_PIVOT`
- Closed Pivot Epoch 1 thesis: action-chunk reactivity under asynchronous inference delay; local result `PRIOR_INFRASTRUCTURE_BLOCKED` without scientific adjudication.
- Active stage: exactly-two-candidate `PIVOT_EPOCH_2` selection; no Ours design is authorized.
- Novelty decision: `INCREMENTAL_BUT_POTENTIALLY_PUBLISHABLE`
- Active training/rollout worker: none.
- Paper-level decision: `KEY_COMPONENT_NOT_SUPPORTED`.
- Paper status: no `PROTOTYPE_GO`, `PAPER_CANDIDATE_GO`, or paper-ready package.
- Preserve pre-existing ignored `rollouts/2026_07_17/` and `rollouts/2026_07_18/`.

## Current authoritative reports

- Overlap audit JSON: `reports/action_consistent_missing_view_distillation_overlap_audit_result.json`
- Overlap audit Markdown: `reports/action_consistent_missing_view_distillation_overlap_audit_result.md`
- Simulation-only RA-L calibration: `reports/simulation_only_ral_evidence_calibration_result.json`
- Frozen method specification: `reports/action_consistent_missing_view_distillation_method_spec_result.json`
- Stage 0 result: `reports/action_consistent_missing_view_distillation_stage0_result.json`
- Exact scientific status: `reports/action_consistent_missing_view_distillation_exact_scientific_status.json`
- CUDA diagnosis: `reports/action_consistent_missing_view_distillation_cuda_device_diagnosis_result.json`
- Exceptional telemetry repair: `reports/action_consistent_missing_view_distillation_telemetry_device_repair_result.json`
- Numerical threshold freeze: `reports/action_consistent_missing_view_distillation_numerical_threshold_freeze_result.json`
- Microbatch preflight: `reports/action_consistent_missing_view_distillation_microbatch_preflight_result.json`
- Stage 0 execution contract result: `reports/action_consistent_missing_view_distillation_stage0_execution_contract_result.json`
- Resumed scientific Stage 0: `reports/action_consistent_missing_view_distillation_resumed_stage0_result.json`
- Stage 0 telemetry: `reports/action_consistent_missing_view_distillation_stage0_runtime_telemetry.json`
- Archive decision: `reports/action_consistent_missing_view_distillation_archive_decision.json`
- Strategic pivot recommendation: `reports/action_consistent_missing_view_distillation_strategic_pivot_recommendation.json`
- Full campaign audit: `reports/autonomous_research_full_history_audit.md`
- Current governance: `reports/current_research_governance.md`
- RL4IL prior result: `reports/rl4il_action_oracle_prior_closed_loop_rollout_result.json`
- RIFA final status: `reports/rifa_xvla_v1_archive_decision.json`
- CVLR final status: `reports/cvlr_xvla_exact_scientific_status.json`
- Pivot Epoch 1 selection: `reports/strategic_pivot_epoch1_selection_result.json`
- A2C2 frozen protocol: `reports/a2c2_prior/problem_verification_protocol.json`
- A2C2 accepted preflight: `reports/a2c2_prior/preflight_result.json`
- A2C2 accepted feature cache: `reports/a2c2_prior/cached_feature_result.json`
- A2C2 accepted Prior training: `reports/a2c2_prior/prior_module_training_result.json`
- A2C2 final local decision: `reports/a2c2_prior/problem_verification_result.json`

## Frozen scientific state

- Wrist-camera dropout is a locally verified claim-specific failure condition for frozen X-VLA.
- The `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT` partially improves the condition but leaves a task-dependent residual. Never relabel it as an official RL4IL reproduction.
- `RIFA_XVLA_STAGE0_DESIGN_FAILURE`; RIFA v1 archived; no RIFA Stage A.
- `CVLR_XVLA_STAGE0_DESIGN_FAILURE`; CVLR v1 archived; no CVLR Stage A.
- RIFA v1 had valid X-VLA integration, exact clean passthrough, valid optimization, and practically negligible full-versus-ablation effect. Its failing binary action delta was a gripper postprocess discontinuity.
- CVLR learned nontrivial cross-view wrist information and beat zero-fill and AWF reconstruction controls, but direct reconstructed-latent insertion destabilized X-VLA actions with 42 gripper flips across nine dropout rows.
- Direct reconstructed-image/token insertion is closed. Cross-view reconstruction is permitted only as training-time auxiliary supervision or a legal reliability statistic.
- Final action-consistent Stage 0 executed validly but returned `STAGE0_MECHANISM_NOT_SUPPORTED`; the method is archived and the wrist-dropout method-development axis is closed.
- Do not relax thresholds, retune/rerun either v1, or reinterpret either frozen decision.

## Closed search space

Do not reopen:

- natural-reset residual mining;
- broad prior or candidate search;
- Task75, R2P, SGL, OCR, AWF;
- RIFA v1 or CVLR v1;
- any other archived formulation;
- direct reconstructed-wrist-token insertion.

The final action-consistent missing-view direction ended in a valid scientific
failure. The wrist-dropout method-development axis is closed without a renamed
v2, rerun, retune, threshold relaxation, replacement candidate, or broad
search.

## Overlap-audit result

The audit inspected 23 primary papers/official records, including all 12 required named works and 11 direct or mechanism-matching additions.

Broad novelty is rejected:

- Acar et al. (RA-L 2023) already distill multi-camera teacher actions/features into a deployable single-camera manipulation policy.
- RME and DisDP already address complete sensor/camera dropout; DisDP includes multi-view shared/private features and action-sequence diffusion.
- VITA-VLA and ActDistill already distill VLA action hidden states/semantics; VITA-VLA separates continuous arm and binary gripper losses.
- VILA uses ground-truth action sequences for cross-view latent alignment.
- MVP-LAM uses cross-viewpoint reconstruction to learn action-centric latent actions for VLA pretraining.
- ReconVLA, WristWorld, MV-MWM, RPT, CRT, RoboNVS, and Imagination at Inference cover reconstruction/pretraining/insertion alternatives.

The narrow conditional claim that survived is:

> Frozen clean multi-view VLA teacher to parameter-efficient complete-wrist-dropout student, with action-chunk representation/continuous-action alignment, separate raw gripper-margin preservation, training-only cross-view wrist reconstruction, exact clean bypass, and deployment without teacher, future frame, retrieval library, or reconstructed-input insertion.

Strongest reviewer objection: this is an obvious combination and any gain may be generic wrist-dropout adaptation. The matched generic dropout adapter and no-reconstruction ablation are therefore mandatory.

## Comparator roles frozen before outcomes

- Base: frozen official X-VLA under wrist dropout.
- External prior: `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT`.
- Ours: action-consistent missing-view distillation.
- Mandatory key ablation: remove cross-view reconstruction supervision with matched teacher distillation and trainable capacity.
- Mandatory mechanism ablation when feasible: remove separate raw gripper-margin supervision.
- Mandatory simple control: ordinary wrist-dropout augmentation LoRA/adapter with matched data, effective batch, optimizer updates, and parameter budget.
- Archived diagnostics: AWF, CVLR v1 insertion, zero-fill.

## Deployment and validation prohibitions

At deployment, Ours may use no clean teacher, future frame, expert action, demonstration action oracle, reward, done/success flag, simulator object/contact/pose state, privileged reset identity, retrieval library, nearest-demonstration search, or reconstructed input insertion.

No physical robot manipulation experiment may be proposed or required. Only after a positive Stage B, a bounded non-actuated `CAMERA-ONLY REAL-IMAGE ACTION-STABILITY VALIDATION` may supplement—but never replace—official LIBERO closed-loop evidence.

## Simulation-only RA-L calibration

- The narrow paper claim is `ROBUST VLA MANIPULATION UNDER SIMULATED WRIST-CAMERA FAILURES`.
- Stage A starts with 3 tasks x 3 held-out identities and expands once to 5 identities per task only for a positive-uncertain, task-mixed, or boundary-near result.
- Stage B covers at least 4 tasks and 3 wrist-camera failure conditions. It starts with 60 paired failure rows per key policy and expands once to 80 only when the frozen performance/noninferiority interval overlaps its boundary.
- Against RL4IL, accept performance superiority, comparable/noninferior success plus one major structural deployment advantage, or comparable/noninferior success plus two moderate useful advantages. All margins, protocols, and major/moderate thresholds must be frozen before outcomes.
- A second backbone is optional strong generalization evidence. Camera-only validation is optional supplementary evidence. Neither is a hard `PAPER_CANDIDATE_GO` gate.
- A single-backbone simulation-only candidate requires stronger task, condition, ablation, paired-uncertainty, and resource evidence and may not claim real-world robustness, sensor reliability, hardware safety, sim-to-real transfer, or deployment readiness.

## Immediate next action

`PIVOT_EPOCH_1` closed locally as `PRIOR_INFRASTRUCTURE_BLOCKED`. A2C2
preflight, cached features, and 40k-step Prior training were valid, but the
same simulator RAM root persisted after its single verified correction. No
Base/Prior success result or Ours evidence exists, and the paper prior is not
disproved. Begin `PIVOT_EPOCH_2`: generate exactly two materially different
research theses outside wrist dropout and asynchronous-delay correction,
apply every hard filter, and select at most one. Do not reopen archived routes
or design Ours before a new closest-prior residual is verified.

## Final preflight boundary

- Final run: `runs/action_consistent_missing_view_distillation/noise_calibration_20260719T025602KST`
- Final result SHA-256: `d6b82e257ba01639ab79565d4995757dadf066d8cd5644b92920e8b828c0d76f`
- Fixed rows materialized: `12 / 12`
- Model / CUDA forwards / optimizer steps: `not loaded / 0 / 0`
- Confirmatory outcomes and physical manipulation: `none / none`
- Scientific mechanism status: unevaluated; do not reinterpret the execution failure as mechanism rejection or support.

The preceding bullets remain the preserved historical pre-resumption failure
boundary. They were superseded scientifically by the valid resumed Stage 0,
but remain immutable audit evidence.
