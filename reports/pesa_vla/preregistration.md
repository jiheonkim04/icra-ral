# PESA-VLA Preregistration

Date: 2026-07-15 KST

Method: `PESA-VLA`.

This preregistration applies only to PESA-VLA after MARC-VLA is closed and archived. It does not modify MARC, DAGR, MTF, RAC, CAVM, PSE, or any earlier fixed-protocol result.

## Frozen Documents

- prior mechanism map: `reports/epoch_4_cycle_9_prior_mechanism_map.md`
- candidate generation: `reports/epoch_4_cycle_9_candidate_generation.md`
- Researcher A proposal: `reports/pesa_vla/researcher_proposal.md`
- proposal hash: `reports/pesa_vla/proposal_hash.txt`
- Reviewer B attack: `reports/pesa_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/pesa_vla/researcher_rebuttal.md`
- mathematical audit: `reports/pesa_vla/mathematical_mechanism_audit.md`

## Evidence Partitions

`DISCOVERY`:

- historical closed-loop reports and valid kills;
- literature-derived prior mechanism map;
- local stable prediction artifact used to inspect Base/expert action gaps;
- design debugging that does not touch confirmatory identities.

`VALIDATION`:

- train and validation splits from the official stable prediction artifact;
- bounded Stage 0 label, spectral, gradient, headroom, and clean-retention audit;
- bounded validation search over at most six configurations;
- simple-killer selection between standard LoRA and clean-retention LoRA;
- all attempted configurations and negative results must be saved.

`CONFIRMATORY_TEST`:

- frozen only after method, config, checkpoint identity, five-policy comparison, metrics, task/reset identities, and thresholds are written;
- may not be used to retune PESA or choose among PESA variants.

## Stage 0: Development Audit

Inputs:

- `reports/official_smolvla_stable_prediction_artifact.json`;
- optional local dataset state records already joined in that artifact;
- no confirmatory rollout outcomes.

Required outputs:

- `reports/pesa_vla/development_audit.json`
- `reports/pesa_vla/development_audit.md`
- `reports/pesa_vla/query_label_manifest.json`
- `reports/pesa_vla/spectral_activation_manifest.json`
- `reports/pesa_vla/split_manifest.json`

Checks:

1. At least `500` train+validation records are scoreable.
2. At least `3` task keys are represented.
3. Duplicate sample keys equal `0`.
4. Duplicate frame keys equal `0`.
5. Train/validation/reserved-test frame overlap equals `0`.
6. Train/validation/reserved reset-identity overlap equals `0` when reset identities are materialized.
7. Frozen Base 7D actions are available for all scoreable records.
8. Expert 7D actions are available for all scoreable records.
9. Base action validity is `1.0` on development records.
10. Standard LoRA or adapter development headroom over Base is positive by validation action proxy or another preregistered development metric.
11. PriorVLA-style proxy can be constructed with the same development data and comparable inference budget.
12. Query-label positive fraction in train and validation is in `[0.05, 0.95]`.
13. Query labels include at least `50` positives and `50` negatives in train.
14. No single task contributes more than `0.20` of query positives.
15. Validation query prediction beats the validation majority baseline by at least `0.02` accuracy.
16. Spectral active rank is not always minimum and not always maximum on validation.
17. Mean spectral activation fraction is in `[0.05, 0.95]`.
18. At least two task families have distinct mean active-rank profiles.
19. Full PESA validation action differs from the PriorVLA-style proxy by mean L2 at least `0.003`.
20. Full PESA validation action differs from the no-spectral/no-prior-query ablation by mean L2 at least `0.003`.
21. Full PESA validation action differs from the selected simple killer by mean L2 at least `0.003`.
22. Initial PESA emitted action delta p95 is at most `1e-6`.
23. Translation, rotation, and gripper deltas are finite and separately reported.
24. Expected adaptation, query, and spectral parameters receive finite nonzero gradients on a development batch.
25. No privileged inference field is required.

Stage 0 decisions:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- `DATA_OR_SUPERVISION_FAILURE`
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_FAILURE`

Do not proceed to training or rollout unless Stage 0 passes.

## Stage 1: Bounded Validation Search

Maximum configurations:

1. `pesa_eta070_a005_query_linear`: spectral energy threshold `0.70`, action cap `0.05`, linear query head
2. `pesa_eta085_a005_query_linear`: spectral energy threshold `0.85`, action cap `0.05`, linear query head
3. `pesa_eta070_a010_query_linear`: spectral energy threshold `0.70`, action cap `0.10`, linear query head
4. `pesa_eta085_a010_query_linear`: spectral energy threshold `0.85`, action cap `0.10`, linear query head
5. `pesa_eta070_a020_query_mlp`: spectral energy threshold `0.70`, action cap `0.20`, one-hidden-layer query head
6. `pesa_eta085_a020_query_mlp`: spectral energy threshold `0.85`, action cap `0.20`, one-hidden-layer query head

Fixed before search:

- `lambda_emit = 1.0`
- `lambda_ret = 0.50`
- `lambda_delta = 0.10`
- `lambda_spec = 0.05`
- `lambda_query = 1.0` when query labels are healthy

No other architecture, coefficient, threshold, or seed variant may be added to this method cycle before confirmatory testing.

Validation score:

- `25%` PriorVLA-style proxy validity and PESA full-versus-proxy distinction;
- `20%` query predictability above majority baseline;
- `20%` clean action retention and bounded deltas;
- `15%` full-versus-ablation and full-versus-simple-killer distinction;
- `10%` spectral activation health and noncollapse;
- `5%` action validity;
- `5%` compute and latency overhead.

Save:

- all tried configurations;
- all negative results;
- selected config;
- checkpoint path and checksum if training occurs;
- validation metrics;
- simple-killer selection rationale.

## Stage A: Catastrophic And Directional Screen

Use exactly five policies:

1. `frozen_smolvla`
2. `priorvla_style_proxy`
3. `pesa_full`
4. `pesa_no_spectral_no_prior_query_ablation`
5. `standard_lora_or_clean_retention_baseline`

Use approximately `10` paired episodes per policy on a matched manifest.

Stage A may permanently kill only for:

- mechanism invalidity;
- no headroom;
- catastrophic degradation;
- clear ablation or prior-proxy dominance;
- exact trivial equivalence.

Small differences, ties, and one- or two-episode gaps advance to Stage B.

## Stage B: Paired Prototype

Use at least `40` paired episodes per key policy.

Report:

- task-balanced success;
- success counts;
- paired wins/losses/ties;
- paired bootstrap confidence intervals;
- effect size;
- relative failure-rate reduction;
- per-task breakdown;
- query gate activation;
- active spectral rank distribution;
- translation, rotation, and gripper action deltas;
- clean retention;
- latency;
- VRAM;
- exceptions.

Allow one expansion to `80` only when Stage B is genuinely unresolved under current governance.

## GO Criteria

`PROTOTYPE_GO` requires:

- `pesa_full` beats `frozen_smolvla`;
- `pesa_full` beats `priorvla_style_proxy`;
- `pesa_full` beats `pesa_no_spectral_no_prior_query_ablation`;
- `standard_lora_or_clean_retention_baseline` does not explain the gain;
- clean behavior is retained;
- mechanism evidence supports read-only prior action plus spectral capacity plus prior-query retention;
- no privileged inference signal is used.

## Kill Or Failure Classifications

- `GENUINE_METHOD_KILL`: valid implementation/data, mechanism acts, and Stage B or valid catastrophic Stage A shows PESA fails against Base, prior proxy, simple baseline, or ablation.
- `SIMPLE_BASELINE_EXPLAINS_METHOD`: standard LoRA or clean-retention baseline matches or beats PESA.
- `KEY_COMPONENT_NOT_USEFUL`: no-spectral/no-prior-query ablation matches or beats PESA.
- `PRIOR_PROXY_DOMINATES`: `priorvla_style_proxy` matches or beats PESA on the claim axis.
- `SPECTRAL_COMPONENT_NOT_USEFUL`: spectral activation is nonacting or not needed beyond the prior proxy and simple baseline.
- `DATA_OR_SUPERVISION_FAILURE`: labels, targets, spectral records, Base actions, or split integrity fail before rollout.
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`: gradients, checkpoint reload, action validity, or adapter integration is invalid.
- `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`: Base, proxy, and diagnostic upper bounds leave no useful improvement target.

## Forbidden

- no confirmatory-test tuning;
- no adding configurations after test results;
- no hidden task/reset cherry-picking;
- no claiming official PriorVLA reproduction;
- no broad novelty claim for prior-preserving adaptation, LoRA, spectral LoRA, or expert routing;
- no MARC, DAGR, MTF, RAC, CAVM, or PSE rescue;
- no KL between deterministic 7D actions;
- no privileged inference input.
