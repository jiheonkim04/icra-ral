# RAR-VLA Preregistration

Date: 2026-07-15 KST

Method: `RAR-VLA`, Re-Anchored Autoregressive Residuals for frozen SmolVLA.

Proposal hash: `723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56`

Mathematical audit: `reports/rar_vla/mathematical_mechanism_audit.md`

Preregistered decision: `RAR_PREREGISTRATION_FROZEN_STAGE_0_PENDING`

## Frozen Documents

- prior mechanism map: `reports/epoch_4_cycle_13_prior_mechanism_map.md`
- candidate generation: `reports/epoch_4_cycle_13_candidate_generation.md`
- Researcher A proposal: `reports/rar_vla/researcher_proposal.md`
- proposal hash: `reports/rar_vla/proposal_hash.txt`
- Reviewer B attack: `reports/rar_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/rar_vla/researcher_rebuttal.md`
- mathematical audit: `reports/rar_vla/mathematical_mechanism_audit.md`

## Evidence Partitions

`DISCOVERY`:

- historical chunk, memory, route, latent, and action-history evidence;
- CALA Stage 0 result showing action-history-only as the strongest trivial
  latent-action predictor;
- latest AR-VLA, REMAC, TAS, ReactVLA, DSWAM, and ABot-M0 priors;
- causal action-memory diagnostics on development identities only.

`VALIDATION`:

- Stage 0 source legality, action-history headroom, residual predictability,
  inter-chunk/intra-chunk diagnostics, zero-delta identity, gradient, and split
  audits;
- bounded validation search over the six named RAR configurations below;
- all negative configurations and failed source variants must be saved.

`CONFIRMATORY_TEST`:

- official LIBERO paired manifests only after method, configuration, policies,
  baselines, ablation, tasks, reset identities, metrics, thresholds, and
  checkpoint identities are frozen;
- confirmatory outcomes may not tune RAR.

## Frozen Method Definition

RAR maintains a causal action-memory state from legal previous emitted actions,
previous Base chunks/actions, proprioception deltas, and task/language inputs.
It re-anchors this memory to each refreshed frozen SmolVLA Base chunk and emits
a bounded zero-initialized residual:

`a_rar_t = b_t + g_t * rhat_t`.

At initialization, RAR must be exact Base passthrough.

RAR may not use future actions, future observations, CALA latent labels,
simulator object pose, reset identity, reward, success, oracle phase, or
confirmatory manifest metadata at inference.

## Stage 0: Development Audit

Stage 0 must complete before validation search, final adapter training, Stage A
manifest freeze, or rollout.

Required outputs:

- `reports/rar_vla/development_audit.json`
- `reports/rar_vla/development_audit.md`
- `reports/rar_vla/source_gate_manifest.json`
- `reports/rar_vla/history_feature_manifest.json`
- `reports/rar_vla/split_manifest.json`

Required checks:

1. Causal source legality:
   - list all runtime fields;
   - prove inference uses only current or previous deployment values;
   - prove future actions, CALA latents, reset identities, object pose, and
     confirmatory outcomes are not required at inference.

2. Split health:
   - zero train/validation/reserved-confirmatory overlap at frame, sample, task,
     episode, and reset identity levels where available;
   - duplicate sample keys equal `0`;
   - duplicate frame keys equal `0`.

3. Action-history headroom:
   - at least `500` scoreable development records;
   - at least `3` task keys represented;
   - Base residual or discontinuity headroom at least `0.01` normalized L2;
   - inter-chunk and intra-chunk diagnostics reported separately.

4. Residual observability:
   - legal RAR features beat `ema_action_history_baseline` and linear-history
     baselines by at least `0.02` normalized residual score;
   - if EMA or linear history is strongest, stop as `DESIGN_FAILURE`.

5. Identity and action validity:
   - initial adapter action delta p95 at most `1e-6`;
   - Base action validity `1.0` on development records;
   - translation, rotation, and gripper deltas separately bounded;
   - expected residual/gate parameters receive finite nonzero gradients.

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

1. `rar_h4_s003_linear`: history horizon `4`, residual scale `0.03`, linear head
2. `rar_h8_s003_linear`: history horizon `8`, residual scale `0.03`, linear head
3. `rar_h16_s003_linear`: history horizon `16`, residual scale `0.03`, linear head
4. `rar_h4_s006_mlp`: history horizon `4`, residual scale `0.06`, one-hidden-layer head
5. `rar_h8_s006_mlp`: history horizon `8`, residual scale `0.06`, one-hidden-layer head
6. `rar_h16_s006_mlp`: history horizon `16`, residual scale `0.06`, one-hidden-layer head

No other architecture, horizon, threshold, scale, coefficient, seed, or source
variant may be added before confirmatory testing.

Validation score:

`S = 0.25 * residual_predictability + 0.20 * clean_retention + 0.20 * bounded_action_validity + 0.15 * localized_activation + 0.15 * ema_baseline_margin + 0.05 * efficiency`

## Stage A: Directional Screen

Use exactly five policies:

1. `frozen_smolvla`
2. `ar_vla_reanchored_expert_proxy`
3. `rar_full`
4. `rar_no_reanchor_memory_ablation`
5. `ema_action_history_baseline`

Use approximately `10` paired episodes per policy on a matched manifest frozen
before rollout.

Stage A may permanently kill only for mechanism invalidity, no headroom,
catastrophic degradation, clear prior/ablation/simple-baseline dominance, or
exact trivial equivalence.

## Stage B: Paired Prototype

Use at least `40` paired episodes per key policy.

Report task-balanced official closed-loop success, paired wins/losses/ties,
paired bootstrap confidence intervals, per-task breakdown, residual/gate
activation, inter-chunk/intra-chunk diagnostics, clean retention, latency, VRAM,
and exceptions.

## GO Criteria

`PROTOTYPE_GO` requires:

- `rar_full` beats `frozen_smolvla`;
- `rar_full` beats `ar_vla_reanchored_expert_proxy`;
- `rar_full` beats `rar_no_reanchor_memory_ablation`;
- `ema_action_history_baseline` does not explain the gain;
- clean behavior is retained;
- residual/gate/memory evidence supports the intended mechanism;
- no privileged inference signal is used.

## Forbidden

- no confirmatory-test tuning;
- no future-action or CALA-latent inference;
- no hidden simulator-state inference;
- no broad novelty claim for autoregressive action memory, re-anchoring,
  smoothing, or chunking;
- no official AR-VLA reproduction claim without equivalence audit;
- no adding configurations after validation or confirmatory outcomes;
- no KL between deterministic 7D actions;
- no rescue of CALA or earlier killed methods.
