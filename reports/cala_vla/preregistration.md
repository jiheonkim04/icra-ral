# CALA-VLA Preregistration

Date: 2026-07-15 KST

Method: `CALA-VLA`, Context-Gated Action-Latent Adapter for frozen SmolVLA.

Proposal hash: `5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76`

Mathematical audit: `reports/cala_vla/mathematical_mechanism_audit.md`

Preregistered decision: `CALA_PREREGISTRATION_FROZEN_STAGE_0_PENDING`

## Frozen Documents

- prior mechanism map: `reports/epoch_4_cycle_12_prior_mechanism_map.md`
- candidate generation: `reports/epoch_4_cycle_12_candidate_generation.md`
- Researcher A proposal: `reports/cala_vla/researcher_proposal.md`
- proposal hash: `reports/cala_vla/proposal_hash.txt`
- Reviewer B attack: `reports/cala_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/cala_vla/researcher_rebuttal.md`
- mathematical audit: `reports/cala_vla/mathematical_mechanism_audit.md`

## Evidence Partitions

`DISCOVERY`:

- historical closed-loop failures and G3P Stage 0 data/supervision stop;
- literature-derived CAC/CALA prior map;
- local latent-action source inventory;
- deterministic latent encoder design and action-segment diagnostics on
  development identities only.

`VALIDATION`:

- Stage 0 source, latent-label, predictability, gradient, Base-passthrough,
  and split-health audits;
- bounded validation search over the six named configurations below;
- all negative configurations and failed source variants must be saved.

`CONFIRMATORY_TEST`:

- official LIBERO paired manifests only after source gate, selected config,
  checkpoint identities, baselines, ablation, metrics, tasks, reset identities,
  and thresholds are frozen;
- confirmatory outcomes may not tune CALA.

## Frozen Method Definition

CALA builds a deterministic latent-action label from future demonstration 7D
action segments on discovery/validation identities only. At inference, it
predicts that latent from current deployment RGB/proprioception/language/Base
features and injects it through a zero-initialized context-gated adapter around
the frozen SmolVLA action interface.

The method may not use future actions, future observations, latent labels,
simulator state, reset identity, reward, success, episode progress unavailable
at deployment, or confirmatory manifest metadata at inference.

## Stage 0: Development Audit

Stage 0 must complete before validation search, final adapter training, Stage A
manifest freeze, or rollout.

Required outputs:

- `reports/cala_vla/development_audit.json`
- `reports/cala_vla/development_audit.md`
- `reports/cala_vla/source_gate_manifest.json`
- `reports/cala_vla/latent_label_manifest.json`
- `reports/cala_vla/split_manifest.json`

Required checks:

1. Source legality:
   - list all runtime fields;
   - prove inference uses only deployment RGB, language, proprioception, and
     Base outputs/features;
   - prove future actions and latent labels are not required at inference.

2. Split health:
   - zero train/validation/reserved-confirmatory overlap at frame, sample, task,
     episode, and reset identity levels where available;
   - duplicate sample keys equal `0`;
   - duplicate frame keys equal `0`.

3. Latent-label health:
   - at least `500` scoreable development records unless the local artifact is
     smaller and explicitly justified;
   - at least `3` task keys represented;
   - nonzero variance in at least `3` latent dimensions;
   - no single task contributes more than `0.35` of scoreable latent labels;
   - no single latent cluster, sign pattern, or high/low bin covers more than
     `0.95` of scoreable records.

4. Observability and headroom:
   - latent prediction from deployment-observable inputs beats the strongest
     trivial predictor by at least `0.02` normalized score or equivalent
     preregistered margin;
   - trivial predictors include task-mean, language/task-only, phase-only,
     action-history/action-only, and majority/constant predictors;
   - diagnostic oracle or upper-bound proxy shows plausible action-latent
     headroom.

5. Identity and action validity:
   - initial adapter action delta p95 at most `1e-6`;
   - Base action validity `1.0` on development records;
   - intended latent/adapter parameters receive finite nonzero gradients in a
     small-batch smoke;
   - translation, rotation, and gripper deltas are separately bounded.

Stage 0 decisions:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- `DATA_OR_SUPERVISION_FAILURE`
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Stage 0 stops are pre-rollout development outcomes, not closed-loop scientific
kills.

## Stage 1: Bounded Validation Search

Run only if Stage 0 passes.

Maximum six configurations:

1. `cala_h8_g003_linear`: latent horizon `8`, gate scale `0.03`, linear adapter
2. `cala_h16_g003_linear`: latent horizon `16`, gate scale `0.03`, linear adapter
3. `cala_h32_g003_linear`: latent horizon `32`, gate scale `0.03`, linear adapter
4. `cala_h8_g006_mlp`: latent horizon `8`, gate scale `0.06`, one-hidden-layer adapter
5. `cala_h16_g006_mlp`: latent horizon `16`, gate scale `0.06`, one-hidden-layer adapter
6. `cala_h32_g006_mlp`: latent horizon `32`, gate scale `0.06`, one-hidden-layer adapter

No other architecture, horizon, threshold, scale, coefficient, seed, or source
variant may be added before confirmatory testing.

Validation score:

`S = 0.25 * latent_predictability + 0.20 * clean_retention + 0.20 * bounded_action_validity + 0.15 * mechanism_activation + 0.15 * simple_baseline_margin + 0.05 * efficiency`

Save:

- all tried configurations;
- all negative results;
- selected config;
- checkpoint paths and checksums if training occurs;
- source gate manifest;
- latent-label manifest;
- validation metrics;
- selected config canonical JSON and hash.

## Stage A: Directional Screen

Use exactly five policies:

1. `frozen_smolvla`
2. `cac_vla_latent_action_proxy`
3. `cala_full`
4. `cala_no_context_gate_ablation`
5. `task_mean_latent_action_baseline`

Use approximately `10` paired episodes per policy on a matched manifest frozen
before rollout.

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
- latent predictability and activation;
- gate values;
- full-versus-ablation action deltas;
- translation, rotation, and gripper deltas;
- clean retention;
- latency;
- VRAM;
- exceptions.

Allow one expansion to `80` only when Stage B is genuinely unresolved under
current governance.

## GO Criteria

`PROTOTYPE_GO` requires:

- `cala_full` beats `frozen_smolvla`;
- `cala_full` beats `cac_vla_latent_action_proxy`;
- `cala_full` beats `cala_no_context_gate_ablation`;
- `task_mean_latent_action_baseline` does not explain the gain;
- clean behavior is retained;
- latent/gate/action-delta evidence supports the intended mechanism;
- no privileged inference signal is used.

## Forbidden

- no confirmatory-test tuning;
- no future-action or latent-label inference;
- no hidden simulator-state inference;
- no broad novelty claim for latent actions or context gating;
- no official CAC-VLA reproduction claim without equivalence audit;
- no adding configurations after validation or confirmatory outcomes;
- no adding extra baselines before the frozen five-policy comparison;
- no KL between deterministic 7D actions;
- no G3P, EAC, PESA, MARC, DAGR, MTF, RAC, CAVM, PSE, or RCV rescue.
