# Autonomous Compact Handoff

Updated: 2026-07-19 KST

## Authoritative state

- Branch: `codex/epoch5-official-prior-first`
- Source HEAD before the current overlap-audit commit: `ea3782a812e0472c0d05f57300e0d7a43dd67429`
- Current pushed HEAD before the method-specification commit: `1d746978b89a9931c92182e3c3a8679323dbd637`
- Epoch/cycle: `5 / 0`
- Campaign state: `AUTONOMOUS_CAMPAIGN_ACTIVE_FINAL_WRIST_DROPOUT_DIRECTION`
- Active method direction: `ACTION-CONSISTENT MISSING-VIEW DISTILLATION`
- Active stage: exact method and simulation-only evidence specification frozen locally; Stage 0 preregistration, noise calibration, and actual-path preflight not yet launched.
- Novelty decision: `INCREMENTAL_BUT_POTENTIALLY_PUBLISHABLE`
- Active training/rollout worker: none at the pre-audit snapshot.
- Paper status: no `PROTOTYPE_GO`, `PAPER_CANDIDATE_GO`, or paper-ready package.
- Preserve pre-existing ignored `rollouts/2026_07_17/` and `rollouts/2026_07_18/`.

## Current authoritative reports

- Overlap audit JSON: `reports/action_consistent_missing_view_distillation_overlap_audit_result.json`
- Overlap audit Markdown: `reports/action_consistent_missing_view_distillation_overlap_audit_result.md`
- Simulation-only RA-L calibration: `reports/simulation_only_ral_evidence_calibration_result.json`
- Frozen method specification: `reports/action_consistent_missing_view_distillation_method_spec_result.json`
- Full campaign audit: `reports/autonomous_research_full_history_audit.md`
- Current governance: `reports/current_research_governance.md`
- RL4IL prior result: `reports/rl4il_action_oracle_prior_closed_loop_rollout_result.json`
- RIFA final status: `reports/rifa_xvla_v1_archive_decision.json`
- CVLR final status: `reports/cvlr_xvla_exact_scientific_status.json`

## Frozen scientific state

- Wrist-camera dropout is a locally verified claim-specific failure condition for frozen X-VLA.
- The `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT` partially improves the condition but leaves a task-dependent residual. Never relabel it as an official RL4IL reproduction.
- `RIFA_XVLA_STAGE0_DESIGN_FAILURE`; RIFA v1 archived; no RIFA Stage A.
- `CVLR_XVLA_STAGE0_DESIGN_FAILURE`; CVLR v1 archived; no CVLR Stage A.
- RIFA v1 had valid X-VLA integration, exact clean passthrough, valid optimization, and practically negligible full-versus-ablation effect. Its failing binary action delta was a gripper postprocess discontinuity.
- CVLR learned nontrivial cross-view wrist information and beat zero-fill and AWF reconstruction controls, but direct reconstructed-latent insertion destabilized X-VLA actions with 42 gripper flips across nine dropout rows.
- Direct reconstructed-image/token insertion is closed. Cross-view reconstruction is permitted only as training-time auxiliary supervision or a legal reliability statistic.
- Do not relax thresholds, retune/rerun either v1, or reinterpret either frozen decision.

## Closed search space

Do not reopen:

- natural-reset residual mining;
- broad prior or candidate search;
- Task75, R2P, SGL, OCR, AWF;
- RIFA v1 or CVLR v1;
- any other archived formulation;
- direct reconstructed-wrist-token insertion.

This action-consistent missing-view direction is the final authorized method direction on the current wrist-dropout axis. If it ends in a robust scientific failure, close the axis without a renamed v2.

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

Freeze a separate Stage 0 preregistration using discovery/validation data only, estimate deterministic forward noise, run the actual-path microbatch preflight over `1,2,4,8`, and launch only the frozen contract. Full-model fine-tuning, CPU/disk offload, swap/pagefile training, confirmatory-outcome access, and outcome-dependent threshold changes remain prohibited.
