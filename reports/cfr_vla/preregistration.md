# CFR-VLA Preregistration

Date: 2026-07-16 KST

Decision: `CFR_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Method: `CFR-VLA`, Continuous Full-Chunk Refinement for VLA action-flow
decoding.

Proposal hash:
`9E2FC510B2D97C869F18BE6C5B339CE034DD98223802078358320AA8BEF3D0AE`

Prerequisite documents:

- proposal: `reports/cfr_vla/researcher_proposal.md`
- Reviewer B attack: `reports/cfr_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/cfr_vla/researcher_rebuttal.md`
- mathematical audit: `reports/cfr_vla/mathematical_mechanism_audit.md`

No CFR training, validation search, rollout, simulator access, or confirmatory
test access has happened before this preregistration.

## Fixed Claim

CFR tests whether continuous Base-start identity-preserving full-chunk
refinement of SmolVLA `[50,7]` action chunks can improve a frozen SmolVLA
policy beyond the closest iterative-refinement prior/proxy, the no-iterative
ablation, and matched standard LoRA.

This is not a DFM-VLA invention claim, not adaptive chunk-size selection, not
action-manifold projection, and not LoRA as novelty.

## Evidence Partitions

`DISCOVERY`

- legal LIBERO demonstrations only;
- demonstrations `0..7` for each fixed development task;
- used for source inspection, feature extraction, Base chunk decoding, residual
  health, DFM proxy fitting, CFR target construction, and implementation
  debugging.

`VALIDATION`

- legal LIBERO demonstrations only;
- demonstrations `8..9` for each fixed development task;
- used for Stage 0 data/mechanism gates and, only after a Stage 0 pass, bounded
  validation search and final configuration selection.

`CONFIRMATORY_TEST`

- no confirmatory task/reset identities, rewards, success flags, done flags,
  object poses, or outcomes may be read before all policies, metrics,
  thresholds, manifests, and selected configuration are frozen;
- confirmatory outcomes may not retune CFR.

## Fixed Development Sources

Use the same fixed development tasks:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Minimum Stage 0 windows:

- at least `512` discovery windows;
- at least `128` validation windows;
- every task must contribute validation rows;
- no task may contribute more than `40%` of the validation audit subset.

## Stage 0 Audit

Stage 0 is development-only. It does not train a rollout policy, does not run a
closed-loop simulator, does not read success/reward/done outcomes, and does not
touch confirmatory reset identities.

Stage 0 must produce:

- `reports/cfr_vla/stage_0_manifest.json`;
- `reports/cfr_vla/stage_0_partial.json`;
- `reports/cfr_vla/stage_0_status.json`;
- `reports/cfr_vla/stage_0_heartbeat.json`;
- `reports/cfr_vla/stage_0_result.json`;
- `reports/cfr_vla/stage_0_result.md`;
- `reports/cfr_vla/stage_0_adjudication.md`;
- `reports/cfr_vla/stage_0_action_semantics.json`;
- stdout/stderr logs and exit code if launched as a worker.

Required Stage 0 checks:

1. proposal hash and source document hashes match;
2. official DFM-VLA asset/code status is recorded;
3. manifest keys are unique with zero missing, extra, duplicate, or split
   overlap keys;
4. feature/action/proprio/language/phase records are finite and aligned;
5. official SmolVLA / LIBERO action semantics are persisted before any
   action-validity decision;
6. Base decoded chunks are finite and have official shape;
7. Base-to-expert residual chunks are noncollapsed by dimension, task, phase,
   and timestep;
8. deployment-input residual/refinement probe beats task/phase residual
   baseline by at least `5%` relative validation Huber or `0.005` absolute
   normalized Huber;
9. DFM proxy is iterative full-sequence refinement, not one-shot residual;
10. DFM proxy leaves CFR residual headroom of at least `5%` relative validation
    Huber or `0.005` absolute normalized Huber;
11. `cfr_no_iterative_refinement` is implemented as one terminal residual and
    is distinct from CFR;
12. initialized and disk-reloaded CFR reproduces Base flow and actions within
    `1e-6`;
13. CFR has finite nonzero gradients on expected trainable parameters and zero
    frozen-parameter gradients;
14. loss-term magnitudes and gradient norms satisfy the mathematical audit
    scale rule;
15. postprocessed action validity is preserved under the frozen official
    semantics before rollout;
16. exceptions are zero.

## Stage 0 Stop Classes

Stage 0 must end with exactly one:

- `CFR_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `CFR_STAGE_0_NO_USABLE_HEADROOM`;
- `CFR_STAGE_0_DESIGN_FAILURE`;
- `CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `CFR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Classify failures as:

- collapsed or illegal source/labels/partitions:
  `CFR_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- no residual headroom over Base or DFM proxy:
  `CFR_STAGE_0_NO_USABLE_HEADROOM`;
- deployment inputs cannot predict useful refinement or iterative refinement is
  equivalent to no-iterative ablation: `CFR_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, reload, gradient, action semantics, action
  validity, persistence, or implementation defect:
  `CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- all gates pass: `CFR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

No Stage 0 stop is a closed-loop scientific kill.

## Bounded Validation Search

Allowed only after `CFR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Maximum six configurations:

1. CFR `K=2`, `lambda_v=0.3`, `g_max=0.10`;
2. CFR `K=4`, `lambda_v=0.3`, `g_max=0.10`;
3. CFR `K=4`, `lambda_v=1.0`, `g_max=0.10`;
4. `dfm_vla_continuous_refinement_proxy`;
5. `cfr_no_iterative_refinement`;
6. `standard_lora`.

One seed per configuration by default. A second seed is allowed only if the
predeclared validation score is genuinely unresolved and must be reported with
the first seed.

Frozen validation score:

`S = 0.30 * validation_success_or_proxy
   + 0.25 * CFR_minus_DFM_proxy_margin
   + 0.20 * clean_action_retention
   + 0.15 * postprocessed_action_validity
   + 0.10 * refinement_overhead_score`

If closed-loop validation is not feasible, `validation_success_or_proxy` must be
a frozen deployment-observable proxy documented before the search. Offline
terminal action L2 alone may not select the final configuration.

Tie break:

1. fewer refinement steps;
2. lower residual cap;
3. lower measured compute overhead.

## First Serious Comparison

Exactly five policy identities:

1. `smolvla_base`;
2. `dfm_vla_continuous_refinement_proxy` or official `dfm_vla` if installed;
3. `cfr_full`;
4. `cfr_no_iterative_refinement`;
5. `standard_lora`.

The closest prior/proxy must enter before any serious paper-viability claim.
The no-iterative ablation and standard-LoRA baseline cannot be removed after
seeing validation or confirmatory outcomes.

## Stage A / Stage B Policy

No Stage A manifest may be frozen until:

- Stage 0 passes;
- bounded validation selects exactly one CFR configuration;
- checkpoints for CFR, ablation, proxy if trainable, and standard LoRA are
  saved and disk-reload verified;
- action validity and clean retention pass under official semantics.

Stage A target:

- approximately `10` paired episodes per policy;
- five policies from the frozen first serious comparison;
- shared task/reset manifest.

Stage A may permanently kill only for mechanism invalidity, no headroom,
catastrophic degradation, clear ablation/prior/simple-baseline dominance, or
exact trivial equivalence.

Stage B target:

- at least `40` paired episodes per key policy;
- paired wins/losses/ties, bootstrap confidence interval, effect size,
  failure-rate reduction, per-task breakdown, mechanism activation, clean
  retention, and efficiency.

One expansion to `80` paired episodes per key policy is allowed only if Stage B
is genuinely unresolved under the active governance.

## Confirmatory Tuning Prohibition

Confirmatory outcomes cannot change:

- tasks or reset identities;
- DFM proxy definition;
- action-validity semantics;
- `K`, `g_max`, residual cap, or coefficients;
- stage thresholds;
- policy list;
- ablation list;
- standard-LoRA baseline;
- clean-retention rule;
- final checkpoint.

A redesign after confirmatory testing starts a new method cycle.

## Preregistration Decision

CFR may proceed to an executable prototype protocol. It may not proceed to
implementation, training, validation search, rollout, or confirmatory testing
until that protocol is frozen.
