# MARC-VLA Preregistration

Date: 2026-07-15 KST

Method: `MARC-VLA`.

This preregistration applies only to MARC-VLA after DAGR-VLA is closed and archived. It does not modify DAGR, MTF, RAC, CAVM, PSE, or any earlier fixed-protocol result.

## Frozen Documents

- prior mechanism map: `reports/epoch_4_cycle_8_prior_mechanism_map.md`
- candidate generation: `reports/epoch_4_cycle_8_candidate_generation.md`
- Researcher A proposal: `reports/marc_vla/researcher_proposal.md`
- proposal hash: `reports/marc_vla/proposal_hash.txt`
- Reviewer B attack: `reports/marc_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/marc_vla/researcher_rebuttal.md`
- mathematical audit: `reports/marc_vla/mathematical_mechanism_audit.md`

## Evidence Partitions

`DISCOVERY`:

- historical closed-loop reports and valid kills;
- literature-derived prior mechanism map;
- local stable prediction artifact used to inspect base/expert disagreement;
- design debugging that does not touch confirmatory identities.

`VALIDATION`:

- train and validation splits from the official stable prediction artifact;
- bounded Stage 0 disagreement-label health and gate-predictability audit;
- bounded validation search over at most six configurations;
- all attempted configurations and negative results must be saved.

`CONFIRMATORY_TEST`:

- frozen only after method, config, checkpoint identity, five-policy comparison, metrics, task/reset identities, and thresholds are written;
- may not be used to retune MARC or choose among MARC variants.

## Stage 0: Development Audit

Inputs:

- `reports/official_smolvla_stable_prediction_artifact.json`;
- optional local dataset state records already joined in that artifact;
- no confirmatory rollout outcomes.

Required outputs:

- `reports/marc_vla/development_audit.json`
- `reports/marc_vla/development_audit.md`
- `reports/marc_vla/disagreement_label_manifest.json`
- `reports/marc_vla/split_manifest.json`

Checks:

1. At least `500` train+validation records are disagreement-label scoreable.
2. At least `3` task keys are represented.
3. Duplicate sample keys equal `0`.
4. Duplicate frame keys equal `0`.
5. Train/validation/reserved-test frame overlap equals `0`.
6. Train and validation disagreement positive fraction in `[0.05, 0.95]`.
7. At least `50` positives and `50` negatives in train.
8. No single task contributes more than `0.20` of train positives.
9. Validation gate prediction beats the validation majority baseline by at least `0.02` accuracy.
10. The OpenVLA-OFT-style L1 proxy has finite action outputs and action validity `1.0`.
11. MARC full validation action differs from the L1 proxy by mean L2 at least `0.003`.
12. MARC full validation action differs from the no-gate ablation by mean L2 at least `0.003`.
13. MARC full validation action differs from the static mixture by mean L2 at least `0.003`.
14. Initial MARC action delta p95 is at most `1e-6`.
15. Base action validity is `1.0` on development records.
16. No privileged inference field is required.

Stage 0 decisions:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- `DATA_OR_SUPERVISION_FAILURE`
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_FAILURE`

Do not proceed to training or rollout unless Stage 0 passes.

## Stage 1: Bounded Validation Search

Maximum configurations:

1. `marc_a005_gate_linear`: correction alpha `0.05`, linear gate
2. `marc_a010_gate_linear`: correction alpha `0.10`, linear gate
3. `marc_a020_gate_linear`: correction alpha `0.20`, linear gate
4. `marc_a005_gate_mlp`: correction alpha `0.05`, one-hidden-layer gate
5. `marc_a010_gate_mlp`: correction alpha `0.10`, one-hidden-layer gate
6. `marc_a020_gate_mlp`: correction alpha `0.20`, one-hidden-layer gate

No other architecture, coefficient, threshold, or seed variant may be added to this method cycle before confirmatory testing.

Validation score:

- `25%` L1 proxy validity and MARC full-versus-proxy distinction;
- `25%` gate predictability above majority baseline;
- `20%` clean action retention and bounded deltas;
- `15%` full-versus-no-gate and full-versus-static-mixture distinction;
- `10%` action validity;
- `5%` compute and latency overhead.

Save:

- all tried configurations;
- all negative results;
- selected config;
- checkpoint path and checksum if training occurs;
- validation metrics.

## Stage A: Catastrophic And Directional Screen

Use exactly five policies:

1. `frozen_smolvla`
2. `openvla_oft_l1_proxy`
3. `marc_full`
4. `marc_no_disagreement_gate_ablation`
5. `static_l1_mixture_baseline`

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
- gate activation;
- action delta from Base;
- clean retention;
- latency;
- VRAM;
- exceptions.

Allow one expansion to `80` only when Stage B is genuinely unresolved under current governance.

## GO Criteria

`PROTOTYPE_GO` requires:

- `marc_full` beats `frozen_smolvla`;
- `marc_full` beats `openvla_oft_l1_proxy`;
- `marc_full` beats `marc_no_disagreement_gate_ablation`;
- `static_l1_mixture_baseline` does not explain the gain;
- clean behavior is retained;
- mechanism evidence supports state-dependent median-anchor correction;
- no privileged inference signal is used.

## Kill Or Failure Classifications

- `GENUINE_METHOD_KILL`: valid implementation/data, mechanism acts, and Stage B or valid catastrophic Stage A shows MARC fails against Base, prior proxy, simple baseline, or ablation.
- `SIMPLE_BASELINE_EXPLAINS_METHOD`: static L1 mixture matches or beats MARC.
- `KEY_COMPONENT_NOT_USEFUL`: no-disagreement-gate ablation matches or beats MARC.
- `PRIOR_PROXY_DOMINATES`: OpenVLA-OFT-style L1 proxy matches or beats MARC on the claim axis.
- `DATA_OR_SUPERVISION_FAILURE`: disagreement labels, targets, or split integrity fail before rollout.
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`: gradients, checkpoint reload, or action integration is invalid.
- `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`: Base, proxy, and diagnostic upper bounds leave no useful improvement target.

## Forbidden

- no confirmatory-test tuning;
- no adding configurations after test results;
- no hidden task/reset cherry-picking;
- no claiming official OpenVLA-OFT reproduction;
- no broad novelty claim for L1 continuous-action fine-tuning;
- no DAGR route rescue, MTF retention rescue, or RAC consequence-calibration rescue;
- no KL between deterministic 7D actions;
- no privileged inference input.
