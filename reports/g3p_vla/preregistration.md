# G3P-VLA Preregistration

Date: 2026-07-15 KST

Method: `G3P-VLA`, Grounded 3D Point Injection for frozen SmolVLA.

Proposal hash: `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`

Mathematical audit: `reports/g3p_vla/mathematical_mechanism_audit.md`

Preregistered decision: `G3P_PREREGISTRATION_FROZEN_STAGE_0_PENDING`

## Frozen Documents

- prior mechanism map: `reports/epoch_4_cycle_11_prior_mechanism_map.md`
- candidate generation: `reports/epoch_4_cycle_11_candidate_generation.md`
- Researcher A proposal: `reports/g3p_vla/researcher_proposal.md`
- proposal hash: `reports/g3p_vla/proposal_hash.txt`
- Reviewer B attack: `reports/g3p_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/g3p_vla/researcher_rebuttal.md`
- mathematical audit: `reports/g3p_vla/mathematical_mechanism_audit.md`

## Evidence Partitions

`DISCOVERY`:

- historical closed-loop failures and EAC kill record;
- literature-derived G3P prior map;
- local source inventory;
- oracle geometry diagnostics and label construction on development identities only.

`VALIDATION`:

- Stage 0 source, label, predictability, gradient, Base-passthrough, and split-health audits;
- bounded validation search over the six named configurations below;
- all negative configurations and failed source variants must be saved.

`CONFIRMATORY_TEST`:

- official LIBERO paired manifests only after source gate, selected config, checkpoint identities, baselines, ablation, metrics, tasks, reset identities, and thresholds are frozen;
- confirmatory outcomes may not tune G3P.

## Frozen Method Definition

G3P predicts a deployment-observable task point from RGB/language/proprioception, converts it to a gripper-relative 3D displacement, and injects that signal through a bounded identity-preserving adapter around the frozen SmolVLA action interface.

The method may not use simulator object pose, placement pose, reset identity, reward, success, future observation, or confirmatory manifest metadata at inference.

## Stage 0: Development Audit

Stage 0 must complete before validation search, final adapter training, Stage A manifest freeze, or rollout.

Required outputs:

- `reports/g3p_vla/development_audit.json`
- `reports/g3p_vla/development_audit.md`
- `reports/g3p_vla/source_gate_manifest.json`
- `reports/g3p_vla/point_label_manifest.json`
- `reports/g3p_vla/split_manifest.json`

Required checks:

1. Source legality:
   - list all runtime fields;
   - prove inference uses only deployment RGB, language, proprioception, and Base outputs/features;
   - prove oracle geometry is not required at inference.

2. Split health:
   - zero train/validation/reserved-confirmatory overlap at frame, sample, task, episode, and reset identity levels where available;
   - duplicate sample keys equal `0`;
   - duplicate frame keys equal `0`.

3. Point-label health:
   - at least `500` scoreable development records unless the local artifact is smaller and explicitly justified;
   - at least `3` task keys represented;
   - point-valid positive fraction in `[0.05, 0.95]`;
   - coordinate variance nonzero in at least two spatial dimensions;
   - no single task contributes more than `0.35` of valid point labels.

4. Observability and headroom:
   - point prediction from deployment-observable inputs beats the strongest trivial predictor by at least `0.02` normalized score or equivalent preregistered margin;
   - trivial predictors include majority/no-point, language/task-only, phase-only, 2D-only when available, and nearest-object when available;
   - oracle diagnostic shows plausible spatial headroom.

5. Identity and action validity:
   - initial adapter action delta p95 at most `1e-6`;
   - Base action validity `1.0` on development records;
   - intended point/adapter parameters receive finite nonzero gradients in a small-batch smoke;
   - translation, rotation, and gripper deltas are separately bounded.

Stage 0 decisions:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- `DATA_OR_SUPERVISION_FAILURE`
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Stage 0 stops are pre-rollout development outcomes, not closed-loop scientific kills.

## Stage 1: Bounded Validation Search

Run only if Stage 0 passes.

Maximum six configurations:

1. `g3p_q30_a003_linear`: confidence quantile `0.30`, adapter scale `0.03`, linear point encoder
2. `g3p_q50_a003_linear`: confidence quantile `0.50`, adapter scale `0.03`, linear point encoder
3. `g3p_q70_a003_linear`: confidence quantile `0.70`, adapter scale `0.03`, linear point encoder
4. `g3p_q30_a006_mlp`: confidence quantile `0.30`, adapter scale `0.06`, one-hidden-layer point encoder
5. `g3p_q50_a006_mlp`: confidence quantile `0.50`, adapter scale `0.06`, one-hidden-layer point encoder
6. `g3p_q70_a006_mlp`: confidence quantile `0.70`, adapter scale `0.06`, one-hidden-layer point encoder

No other architecture, threshold, scale, coefficient, seed, or source variant may be added before confirmatory testing.

Validation score:

`S = 0.30 * point_predictability + 0.20 * clean_retention + 0.20 * bounded_action_validity + 0.15 * mechanism_activation + 0.10 * simple_baseline_margin + 0.05 * efficiency`

Save:

- all tried configurations;
- all negative results;
- selected config;
- checkpoint paths and checksums if training occurs;
- source gate manifest;
- validation metrics;
- selected config canonical JSON and hash.

## Stage A: Directional Screen

Use exactly five policies:

1. `frozen_smolvla`
2. `g3p_3d_point_proxy`
3. `g3p_full`
4. `g3p_no_3d_no_injection_ablation`
5. `simple_2d_phase_or_nearest_object_heuristic`

Use approximately `10` paired episodes per policy on a matched manifest frozen before rollout.

Stage A may permanently kill only for:

- mechanism invalidity;
- no headroom;
- catastrophic degradation;
- clear closest-prior proxy, ablation, or simple-baseline dominance;
- exact trivial equivalence.

Small differences, ties, and one- or two-episode gaps advance to Stage B.

## Stage B: Paired Prototype

Use at least `40` paired episodes per key policy.

Report:

- task-balanced official closed-loop success;
- paired wins/losses/ties;
- paired bootstrap confidence intervals;
- per-task breakdown;
- point-source activation;
- point prediction health;
- translation, rotation, and gripper deltas;
- clean retention;
- latency;
- VRAM;
- exceptions.

Allow one expansion to `80` only when Stage B is genuinely unresolved under current governance.

## GO Criteria

`PROTOTYPE_GO` requires:

- `g3p_full` beats `frozen_smolvla`;
- `g3p_full` beats `g3p_3d_point_proxy`;
- `g3p_full` beats `g3p_no_3d_no_injection_ablation`;
- `simple_2d_phase_or_nearest_object_heuristic` does not explain the gain;
- clean behavior is retained;
- point-source and action-delta evidence support the spatial mechanism;
- no privileged inference signal is used.

## Forbidden

- no confirmatory-test tuning;
- no hidden simulator/object-state inference;
- no broad novelty claim for 3D point injection;
- no official prior reproduction claim without equivalence audit;
- no adding configurations after validation or confirmatory outcomes;
- no adding extra baselines before the frozen five-policy comparison;
- no KL between deterministic 7D actions;
- no EAC, PESA, MARC, DAGR, MTF, RAC, CAVM, PSE, or RCV rescue.
