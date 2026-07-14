# MTF-VLA Preregistration

Date: 2026-07-14 KST

Method: `MTF-VLA`.

This preregistration applies only to MTF-VLA after RAC-VLA is closed and archived. It does not modify RAC, PSE, CAVM, or any earlier fixed-protocol result.

## Frozen Documents

- candidate generation: `reports/epoch_4_cycle_6_candidate_generation.md`
- prior mechanism map: `reports/epoch_4_cycle_6_prior_mechanism_map.md`
- Researcher A proposal: `reports/mtf_vla/researcher_proposal.md`
- Reviewer B attack: `reports/mtf_vla/reviewer_attack.md`
- Researcher rebuttal: `reports/mtf_vla/researcher_rebuttal.md`
- mathematical audit: `reports/mtf_vla/mathematical_mechanism_audit.md`

## Evidence Partitions

`DISCOVERY`:

- historical official SmolVLA and LoRA closed-loop reports;
- existing repository traces and reports;
- literature mechanism map;
- data/score design.

`VALIDATION`:

- development-only frame and rollout identities selected before confirmatory testing;
- bounded validation search over at most six configurations;
- all attempted configurations and negative results must be saved.

`CONFIRMATORY_TEST`:

- frozen after Stage 0, validation search, checkpoint selection, baseline list, ablation, metrics, task/reset identities, and thresholds are written.
- confirmatory outcomes may not retune MTF.

## Stage 0: Development Audit

Inputs:

- official LIBERO demonstration or trace metadata available locally;
- frozen Base policy identity for retention target generation;
- no confirmatory rollout outcomes.

Required outputs:

- `reports/mtf_vla/development_audit.json`
- `reports/mtf_vla/development_audit.md`
- persisted split manifest;
- persisted frame-score summary;
- persisted base-retention target manifest or a hard-stop reason.

Stage 0 checks:

1. At least `500` scoreable frames or action records overall.
2. At least `3` task keys represented unless a narrower task axis is explicitly frozen before validation.
3. High and low score sets each contain at least `10%` of scoreable records.
4. Mean high score exceeds mean low score by at least `0.25`.
5. Gripper-transition positives are not collapsed: positive fraction must be greater than `0.005` and less than `0.80`.
6. Phase coverage includes at least `3` nonempty phase bins for each selected task.
7. Duplicate frame keys equal `0`.
8. Discovery, validation, and future confirmatory identity overlap equals `0` for all identities known at Stage 0.
9. Frozen-base retention targets can be generated, saved, and reloaded.
10. Adapter initialization is base-equivalent or near base-equivalent on a small action-delta smoke: p95 first-action delta must be at most `1e-4` before training.
11. The FrameSkip proxy can be constructed transparently. Any omitted official component must be listed.

Stage 0 decisions:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`
- `DATA_OR_SUPERVISION_FAILURE`
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_FAILURE`

Do not proceed to training or rollout unless Stage 0 passes.

## Stage 1: Bounded Validation Search

Maximum configurations:

1. retained high-frame ratio `0.20`, retention coefficient `0.25`
2. retained high-frame ratio `0.20`, retention coefficient `0.50`
3. retained high-frame ratio `0.20`, retention coefficient `1.00`
4. retained high-frame ratio `0.30`, retention coefficient `0.25`
5. retained high-frame ratio `0.30`, retention coefficient `0.50`
6. retained high-frame ratio `0.30`, retention coefficient `1.00`

Maximum training seeds:

- no more than `2` lightweight seeds for a selected configuration;
- no best-seed selection on confirmatory test.

Validation score:

- `35%` validation closed-loop success or closest feasible proxy;
- `25%` clean retention;
- `20%` milestone activation and score health;
- `10%` action validity and bounded deltas;
- `10%` compute overhead.

Save:

- all configs;
- all validation metrics;
- selected config;
- checkpoint path and checksum;
- negative results.

## Stage A: Catastrophic And Directional Screen

Use exactly five policies:

1. `base_smolvla`
2. `frameskip_proxy_lora`
3. `mtf_full`
4. `mtf_no_retention_ablation`
5. `uniform_retained_ratio_lora`

Use approximately `10` paired episodes per policy on a matched task/reset manifest.

Stage A may permanently kill only under active governance catastrophic criteria:

- full has `0 / 10` while a paired baseline has at least `4 / 10`;
- full is at least `30` absolute percentage points below a baseline or ablation;
- mechanism is valid and clearly harmful;
- no diagnostic headroom exists;
- implementation or supervision is invalid;
- exact trivial equivalence is proven.

Small differences, ties, and one- or two-episode gaps advance to Stage B.

## Stage B: Paired Prototype

Use at least `40` paired episodes per key policy.

Report:

- successes/counts;
- task-balanced success;
- paired wins/losses/ties;
- paired bootstrap confidence intervals;
- effect size;
- relative failure-rate reduction;
- per-task breakdown;
- clean retention;
- milestone activation;
- action-delta distribution;
- latency;
- VRAM.

Allow one expansion to `80` only when full is not clearly inferior, confidence remains unresolved, and useful improvement has not been excluded.

## GO Criteria

`PROTOTYPE_GO` requires:

- `mtf_full` beats `base_smolvla`;
- `mtf_full` beats `frameskip_proxy_lora`;
- `mtf_full` beats `mtf_no_retention_ablation`;
- `mtf_full` beats `uniform_retained_ratio_lora`;
- clean behavior is retained;
- mechanism is active and bounded;
- no privileged inference signal is used;
- the result is not caused by confirmatory best-seed selection.

## Kill Or Failure Classifications

- `GENUINE_METHOD_KILL`: valid implementation/data, mechanism acts, Stage B or valid catastrophic Stage A complete, and full clearly fails against prior, baseline, or ablation.
- `SIMPLE_BASELINE_EXPLAINS_METHOD`: uniform retained-ratio LoRA matches or beats full.
- `KEY_COMPONENT_NOT_USEFUL`: no-retention ablation matches or beats full.
- `DATA_OR_SUPERVISION_FAILURE`: scores, labels, targets, or splits are invalid before rollout.
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`: intended adapter or checkpoint behavior is invalid.
- `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`: Base, prior, and diagnostic upper bound leave no usable improvement target.

## Forbidden

- no confirmatory-test tuning;
- no adding hyperparameter variants after test results;
- no hiding failed validation configs;
- no describing MTF as official FrameSkip or StructVLA reproduction;
- no claiming LoRA itself is the contribution;
- no KL between deterministic 7D actions;
- no OpenVLA-OFT training before SmolVLA GO and a separate feasibility plan.
