# MCI-VLA Preregistration

Date: 2026-07-16 KST

Decision: `MCI_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Method: `MCI-VLA`, Multi-Consistency Invariance for Base-preserving SmolVLA.

Proposal SHA-256:
`88CB11CC6236D19BA05602217C65C1819A68BEA53B041E17BA12796403BA0B9A`

Prerequisite documents:

- proposal: `reports/mci_vla/researcher_proposal.md`
- Reviewer B attack: `reports/mci_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/mci_vla/researcher_rebuttal.md`
- mathematical audit: `reports/mci_vla/mathematical_mechanism_audit.md`

No MCI implementation, training, validation search, rollout, simulator
evaluation, or confirmatory-test access has happened before this
preregistration.

## Frozen Claim

MCI tests whether a frozen-SmolVLA, Base-preserving adapter trained with
RoVLA-style multi-consistency invariance can improve task-preserving robustness
over a RoVLA-style consistency proxy, a no-consistency-code ablation, and an
augmentation-only LoRA alternative.

The claim is not:

- official RoVLA reproduction unless official compatible RoVLA assets run
  locally;
- a new VLA backbone;
- ordinary LoRA as the method;
- generic data augmentation;
- object binding, short-horizon memory, critical-step residual repair, or
  chunk-boundary smoothing;
- use of reward, success, done, simulator state, object pose, reset identity,
  future observation, future expert action, or confirmatory outcome at
  inference;
- rescue or reinterpretation of CSPR or any closed method.

## Evidence Partitions

`DISCOVERY / TRAINING`

- legal LIBERO demonstrations, cached frozen SmolVLA Base chunks, current
  observation features, proprioception, task strings, and demonstration action
  chunks;
- tasks:
  - `libero_10/task_5`
  - `libero_goal/task_5`
  - `libero_object/task_3`
  - `libero_spatial/task_3`
- demo ids `0..7`;
- minimum usable rows: `512`;
- used for source checks, transformation generation, objective and gradient
  smoke, RoVLA proxy diagnostics, trivial baselines, and implementation
  debugging.

`VALIDATION`

- the same four development tasks;
- demo ids `8..9`;
- minimum usable rows: `128`;
- used for Stage 0 development gates and, only after Stage 0 passes, bounded
  validation search and final configuration selection;
- no confirmatory outcomes may be read.

`CONFIRMATORY TEST`

- untouched until method, configuration, policy list, ablation, task/reset
  identities, metrics, thresholds, manifests, and checkpoints are frozen;
- no confirmatory task/reset identity, reward, success, done, object pose,
  future observation, policy action, failure, partial outcome, or threshold may
  be read during Stage 0 or validation search;
- confirmatory outcomes may not retune MCI.

## Fixed Development Sources

Use existing local LIBERO demonstration rows and legal cached SmolVLA Base
chunks only. A Stage 0 source audit must write the exact source files, cache
hashes, and row keys before any training.

Required row properties:

- task identity is one of the four fixed development tasks;
- demo id is in `0..9`;
- Base chunk exists, loads finite, and has shape `[50, 7]`;
- demonstration action chunk exists, loads finite, and has shape `[50, 7]`;
- current RGB or cached visual features are legal current-observation inputs;
- proprioception is finite with dimension `8`;
- row key is unique;
- no split overlap between discovery and validation keys;
- no confirmatory identity is read.

If the `512 / 128` discovery/validation row minima cannot be produced without
duplicate keys, missing cache files, split overlap, or confirmatory reads,
Stage 0 stops as `MCI_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Frozen Mechanism

Use exactly the mathematical audit object:

- `H = 50`;
- `D = 7`;
- action chunk shape `[N, 50, 7]`;
- `d_z in {16, 32}`;
- `lambda_c in {0.25, 0.50, 1.00}`;
- objective terms `L_code`, `L_act`, `L_fit`, `L_keep`, `L_var`, and
  `L_bound`;
- representation variance floor `gamma_var = 0.5`;
- Huber beta values `0.05`;
- no deterministic-action KL;
- frozen SmolVLA Base weights;
- zero-initialized residual head and exact Base passthrough at initialization.

The six validation-search configurations are the only allowed configurations:

1. `mci_lc025_dz16`;
2. `mci_lc025_dz32`;
3. `mci_lc050_dz16`;
4. `mci_lc050_dz32`;
5. `mci_lc100_dz16`;
6. `mci_lc100_dz32`.

## Frozen Transformations

All transformation generators must be deterministic or fully logged before
Stage 0 begins.

Instruction transformations:

- deterministic task-preserving paraphrase templates;
- official equivalent task strings if available;
- no unlogged LLM prompt iteration after validation.

Observation/proprioception transformations:

- bounded brightness, contrast, crop/resize, and low-amplitude image noise;
- bounded proprioception jitter inside deployment-valid ranges.

Action-evolution transformations:

- small valid perturbations of the current Base chunk or legal action-generation
  feature path;
- no future expert action, reward, success, done, object pose, simulator state,
  reset identity, or confirmatory outcome.

Every transformed pair must keep task and action semantics unchanged.
Invalid transformations stop as `MCI_STAGE_0_DATA_OR_SUPERVISION_FAILURE` or
`MCI_STAGE_0_DESIGN_FAILURE`.

## First Serious Comparison

The first serious comparison remains exactly:

1. `smolvla_base`
2. `rovla_multiconsistency_proxy`
3. `mci_full`
4. `mci_no_consistency_code_ablation`
5. `augmentation_only_lora_killer`

Policy 2 must first attempt official RoVLA code/assets compatibility. If exact
execution is unavailable, it is a transparent local proxy and must document
every mismatch from official RoVLA. It must preserve instruction,
observation/proprioception, and action-evolution consistency.

Policy 4 removes the learned consistency code and consistency losses while
keeping the same adapter surface, data, action caps, clean-retention objective,
and training budget.

Policy 5 uses the same legal augmentations and matched budget without the
multi-consistency code mechanism.

## Stage 0 Purpose

Stage 0 is a development-only source, data, mathematical, implementation, and
mechanism audit. It is not a closed-loop scientific result and not a paper
claim.

Stage 0 determines whether:

- source rows and splits exist without leakage;
- transformation pairs are valid and noncollapsed;
- the consistency code is noncollapsed and observable from legal inputs;
- Base and the RoVLA proxy leave headroom on the robustness claim axis;
- MCI differs from Base, RoVLA proxy, no-code ablation, and augmentation-only
  LoRA in a bounded way;
- exact Base passthrough, checkpoint reload, gradient behavior, action
  validity, and clean retention hold.

## Stage 0 Required Artifacts

Stage 0 must produce under `reports/mci_vla/`:

- `stage_0_preflight.json`;
- `stage_0_manifest.json`;
- `stage_0_partial.json`;
- `stage_0_result.json`;
- `stage_0_result.md`;
- `stage_0_adjudication.md`;
- `stage_0_status.json`;
- `stage_0_heartbeat.json`;
- `stage_0_pid.txt`;
- `stage_0_exit_code.txt`;
- `stage_0_action_semantics.json`;
- `stage_0_official_prior_asset_check.json`;
- `stage_0_serializer_preflight.json`;
- stdout/stderr logs if launched detached.

## Stage 0 Required Metrics

Required metrics:

- planned and completed row counts;
- exception count;
- duplicate manifest keys, duplicate partial keys, missing keys, extra keys,
  and split-overlap keys;
- proposal hash match;
- no reward/success/done/object-pose/future-observation/confirmatory reads;
- discovery and validation row counts by task and demo id;
- Base and demonstration action shape, finite fraction, min, max, and
  postprocessor validity;
- visual feature and proprioception shape and finite fraction;
- transformation pair counts by family;
- positive and negative contrast counts by transformation family;
- task and demo coverage by transformation family;
- representation variance and noncollapse checks;
- mask or gate activation fraction by task and transformation family;
- trivial baseline scores for task identity, frame/demo phase audit-only
  proxy, action-magnitude statistics, and augmentation-family identity;
- legal consistency-signal predictor score versus strongest trivial baseline;
- RoVLA official/proxy status and mismatch list;
- RoVLA proxy score and remaining MCI headroom;
- augmentation-only LoRA score and remaining MCI headroom;
- MCI full versus Base, RoVLA proxy, no-code ablation, and augmentation-only
  LoRA;
- identity initialization and disk-reload max absolute error;
- finite nonzero gradients for expected consistency encoder, residual, and gate
  parameters;
- zero gradients for frozen SmolVLA Base parameters;
- objective magnitudes and weighted gradient norms for `L_code`, `L_act`,
  `L_fit`, `L_keep`, `L_var`, and `L_bound`;
- weighted objective gradient-norm ratio;
- action delta summaries by translation, rotation, and gripper groups;
- clean-retention deltas;
- normalized and postprocessed action-validity rate.

## Stage 0 Pass Gates

All must pass:

- proposal hash matches
  `88CB11CC6236D19BA05602217C65C1819A68BEA53B041E17BA12796403BA0B9A`;
- no privileged or confirmatory input access;
- manifest and partial row keys are unique and complete;
- split overlap is zero;
- discovery row count is at least `512`;
- validation row count is at least `128`;
- every fixed task contributes validation rows;
- no validation task contributes more than `40%` of validation rows;
- every transformation family has at least `32` validation pairs;
- positive and negative contrast counts are both at least `16`;
- representation standard deviation exceeds the `L_var` floor on at least
  `80%` of latent dimensions;
- consistency signal beats the strongest trivial baseline by at least `0.02`
  normalized validation score;
- Base leaves measurable transformed-pair action/representation headroom;
- RoVLA proxy leaves measurable residual headroom for MCI;
- augmentation-only LoRA does not explain MCI;
- MCI full beats the strongest of RoVLA proxy, no-code ablation, and
  augmentation-only LoRA by at least `0.005` normalized validation mechanism
  proxy;
- exact Base passthrough and disk reload max absolute error are `<= 1e-7`;
- expected trainable parameters receive finite nonzero gradients;
- frozen SmolVLA Base parameters receive no gradients;
- weighted objective gradient-norm ratio is at most `100x`;
- intervention fraction lies in `[0.02, 0.80]`;
- action deltas respect preregistered translation, rotation, and gripper caps;
- postprocessed action validity is `1.0`.

## Stage 0 Stop Classes

Stage 0 must end with exactly one:

- `MCI_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `MCI_STAGE_0_NO_HEADROOM`;
- `MCI_STAGE_0_IMPLEMENTATION_FAILURE`;
- `MCI_STAGE_0_DESIGN_FAILURE`;
- `MCI_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Classify failures as:

- cache, action-shape, feature, transformation, label/contrast, task-coverage,
  duplicate-key, split-overlap, or privileged-input failure:
  `MCI_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- no Base transformed-pair headroom, no RoVLA residual headroom, or no legal
  consistency-signal predictability: `MCI_STAGE_0_NO_HEADROOM`;
- hash, serialization, identity, reload, gradient, objective-scale,
  frozen-parameter, checkpoint, action-semantics, action-validity, persistence,
  or global-delta defect: `MCI_STAGE_0_IMPLEMENTATION_FAILURE`;
- constant-code mechanism, task-only shortcut, global destructive action
  changes, exact trivial equivalence, no-code ablation dominance, RoVLA proxy
  dominance, or augmentation-only LoRA explanation:
  `MCI_STAGE_0_DESIGN_FAILURE`;
- all gates pass: `MCI_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

No Stage 0 stop is a closed-loop scientific kill and no Stage 0 stop may be
rescued by changing transformations, row identities, thresholds, proxy
definition, objective terms, or pass gates after seeing results.

## Bounded Validation Search

Allowed only after `MCI_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Maximum six configurations:

1. `mci_lc025_dz16`;
2. `mci_lc025_dz32`;
3. `mci_lc050_dz16`;
4. `mci_lc050_dz32`;
5. `mci_lc100_dz16`;
6. `mci_lc100_dz32`.

No transformation family, task split, identity split, feature set, proxy
definition, comparator, threshold, action cap, or confirmatory identity may be
searched outside this budget.

Validation score:

`S_val = 0.30 * success_or_best_legal_proxy
       + 0.20 * clean_retention
       + 0.20 * consistency_activation
       + 0.15 * action_validity
       + 0.10 * prior_and_ablation_margin
       + 0.05 * compute_overhead`.

Ties break by clean retention, then lower Base-relative action delta, then
lower adapter parameter count.

## Confirmatory Gate

Confirmatory Stage A/B can begin only after:

- Stage 0 passes;
- bounded validation search selects one frozen configuration;
- checkpoint and config artifacts are saved;
- policy list, metrics, thresholds, tasks, reset identities, and manifests are
  frozen;
- no confirmatory result has been read.

Confirmatory outcomes may not retune MCI. A major redesign after confirmatory
access is a new method cycle.

## Immediate Next Stage

Proceed to executable prototype protocol before implementation, validation
search, rollout, or confirmatory-test access.
