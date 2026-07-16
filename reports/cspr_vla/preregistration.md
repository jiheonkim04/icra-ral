# CSPR-VLA Preregistration

Date: 2026-07-16 KST

Decision: `CSPR_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Method: `CSPR-VLA`, Critical-Step Selective Policy Refinement for SmolVLA.

Proposal SHA-256:
`CC83324F9AB37DAEEF4E2BA158C821F336383A8C4F96ADFFF4DE7B79E276D0D7`

Prerequisite documents:

- proposal: `reports/cspr_vla/researcher_proposal.md`
- Reviewer B attack: `reports/cspr_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/cspr_vla/researcher_rebuttal.md`
- mathematical audit: `reports/cspr_vla/mathematical_mechanism_audit.md`

No CSPR implementation, training, validation search, rollout, simulator
access, or confirmatory-test access has happened before this preregistration.

## Frozen Claim

CSPR tests whether critical-step selective action refinement can improve a
frozen SmolVLA action chunk by applying bounded residual corrections only on
deployment-observable high-criticality action cells while preserving exact
Base behavior elsewhere.

The claim is not:

- official DySL-VLA reproduction;
- a new VLA backbone;
- ordinary LoRA as the method;
- generic action residual imitation;
- global fine-tuning;
- a progress-guidance or ForesightFlow potential method;
- a rescue of DCCG or any prior closed method;
- use of demonstration time index, future expert action, reward, success,
  done, object pose, simulator state, future observation, or confirmatory
  identity at inference.

LoRA may only serve as low-compute implementation infrastructure for the
criticality predictor, residual head, or matched controls. Removing LoRA must
not remove the CSPR scientific mechanism.

## Evidence Partitions

`DISCOVERY / TRAINING`

- legal cached SmolVLA Base chunks, cached current-observation features,
  proprioception, task identity, and source LIBERO demonstration actions;
- tasks:
  - `libero_10/task_5`
  - `libero_goal/task_5`
  - `libero_object/task_3`
  - `libero_spatial/task_3`
- demo ids `0..7`;
- expected cached Base rows: `512`;
- used for cache/source checks, criticality-label construction, robust
  normalization, objective/gradient smoke, DySL proxy diagnostics, simple
  killer diagnostics, and implementation debugging.

`VALIDATION`

- same four development tasks;
- demo ids `8..9`;
- expected cached Base rows: `128`;
- used for Stage 0 development gates and, only after Stage 0 passes, bounded
  validation search and final configuration selection;
- no confirmatory outcomes may be read.

`CONFIRMATORY TEST`

- untouched until method, configuration, policy list, ablation, tasks, reset
  identities, metrics, thresholds, manifests, and checkpoints are frozen;
- no confirmatory task/reset identities, rewards, success flags, done flags,
  object poses, future observations, policy actions, failures, partial
  outcomes, or thresholds may be read during Stage 0 or validation search;
- confirmatory outcomes may not retune CSPR.

## Fixed Development Sources

Use the verified CCIF cached SmolVLA Base rows only unless a later
preregistered cache-coverage audit authorizes additional rows.

Fixed source cache:

- `reports/ccif_vla/stage_0_partial.json`
- `reports/ccif_vla/stage_0_manifest.json`

Required row properties:

- `model_or_probe == smolvla_base`;
- `base_chunk_cache_path` exists and loads a finite `[50, 7]` Base chunk;
- `feature_cache_path` exists and loads finite current visual features;
- task identity is one of the four fixed tasks;
- demo id is in `0..9`;
- row key is unique;
- no split overlap between discovery and validation keys.

If the verified `512 / 128` discovery/validation rows cannot be produced
without duplicate keys, missing cache files, split overlap, or confirmatory
identity reads, Stage 0 must stop as `CSPR_STAGE_0_DATA_FAILURE`.

## Frozen Mechanism

Use exactly the mathematical audit variables:

- `H = 50`;
- `D = 7`;
- Base action chunk `B in R^[N, 50, 7]`;
- demonstration action chunk `Y in R^[N, 50, 7]`, training/audit only;
- visual feature dimension `960`;
- proprioception dimension `8`;
- criticality target `C* in {0,1}^[N, 50, 7]`;
- criticality prediction `C in [0,1]^[N, 50, 7]`;
- residual proposal `R in R^[N, 50, 7]`;
- gate `G in [0,1]^[N, 50, 7]`;
- action `A = postprocess(B + G * Delta * tanh(R))`;
- objective terms `L_crit`, `L_fit`, `L_keep`, and `L_bound`;
- no deterministic-action KL.

Default Stage 0 smoke configuration:

- residual cap group: `mid`;
- `Delta_trans = 0.01`;
- `Delta_rot = 0.025`;
- `Delta_grip = 0.125`;
- criticality threshold: discovery `0.95` quantile;
- soft-gate temperature for gradient smoke: `0.05`;
- identity tolerance: `1e-7`;
- weighted objective gradient-norm ratio alert: `100.0`.

These Stage 0 constants may not change after Stage 0 begins.

## Frozen Criticality Label Construction

Criticality labels are development-only and may use `B`, `Y`, and legal
current-row features. They are not inference inputs.

For action cell `(t,d)` define:

`q_t,d = w_e * norm_err_t,d + w_k * norm_curv_t,d
       + w_a * norm_acc_t,d + w_g * grip_event_t,d`.

Fixed weights:

- `w_e = 1.0`;
- `w_k = 0.5`;
- `w_a = 0.5`;
- `w_g = 1.0`.

Discovery-only robust normalizers are used for Base-vs-demonstration error,
demonstration curvature, and demonstration acceleration. Validation must use
discovery normalizers only.

`C*_t,d = 1[q_t,d >= q_tau]`, where `q_tau` is the discovery `0.95` quantile
for Stage 0 smoke. If labels are all-zero, all-one, task-collapsed, or
phase-collapsed, Stage 0 stops as `CSPR_STAGE_0_DATA_FAILURE`.

Frame index may be reported as an audit baseline. It may not be used by CSPR
at inference.

## First Serious Comparison

The first serious comparison remains exactly:

1. `smolvla_base`
2. `dysl_action_importance_proxy`
3. `cspr_full`
4. `cspr_uniform_refinement_ablation`
5. `critical_step_threshold_simple_killer`

Policy 2 must first attempt official DySL-VLA code/assets compatibility. If
exact execution is unavailable, policy 2 is a transparent proxy that implements
action-importance-conditioned capacity allocation without CSPR's learned
residual action correction. It must document every mismatch from official
DySL-VLA.

Policy 4 keeps the same residual capacity, action caps, optimizer budget,
training rows, and clean-retention term while replacing learned criticality
with a uniform learned refinement gate.

Policy 5 is a nonlearned threshold over Base chunk curvature, velocity,
acceleration, and gripper transitions. It remains live as the strongest simple
reviewer-killer baseline.

## Stage 0 Purpose

Stage 0 is a development-only source, data, objective, implementation, and
mechanism audit. It is not a closed-loop scientific result and not a paper
claim.

Stage 0 determines whether:

- the fixed cache identities exist and match the preregistered split;
- criticality labels are noncollapsed with task and phase coverage;
- criticality is predictable from legal deployment-observable inputs above
  trivial baselines;
- Base and the DySL proxy leave headroom on the CSPR claim axis;
- CSPR differs from Base, DySL proxy, uniform-refinement ablation, and simple
  killer in a bounded way;
- exact Base passthrough, checkpoint reload, action validity, and clean
  retention hold.

## Stage 0 Required Artifacts

Stage 0 must produce under `reports/cspr_vla/`:

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
- Base chunk shape, finite fraction, min, max, and postprocessor validity;
- feature cache shape, finite fraction, and source hash status;
- criticality positive/negative counts globally and by task;
- criticality positive fraction by task and by frame-index quartile;
- all-zero/all-one checks;
- criticality score variance and robust-normalizer statistics;
- trivial baseline scores for gripper-transition threshold,
  curvature/velocity threshold, task-mean criticality, and frame-index audit
  proxy;
- legal criticality predictor score versus strongest trivial baseline;
- DySL official/proxy status and mismatch list;
- DySL proxy score and remaining CSPR headroom;
- simple-killer score and remaining CSPR headroom;
- CSPR full versus Base, DySL proxy, uniform-refinement ablation, and simple
  killer;
- identity initialization and disk-reload max absolute error;
- finite nonzero gradients for criticality predictor and residual head;
- zero gradients for frozen SmolVLA Base parameters;
- objective magnitudes and weighted gradient norms for `L_crit`, `L_fit`,
  `L_keep`, and `L_bound`;
- gate activation p50, p95, max, and intervention fraction;
- action delta summaries by translation, rotation, and gripper groups;
- clean-retention p95 deltas on low-criticality rows;
- action-validity rate after official postprocessing.

## Stage 0 Pass Gates

All must pass:

- proposal hash matches
  `CC83324F9AB37DAEEF4E2BA158C821F336383A8C4F96ADFFF4DE7B79E276D0D7`;
- no privileged or confirmatory input access;
- manifest and partial row keys are unique and complete;
- split overlap is zero;
- discovery row count is at least `512`;
- validation row count is at least `128`;
- every fixed task contributes validation rows;
- no validation task contributes more than `40%` of validation rows;
- validation positive count is at least `8`;
- validation negative count is at least `8`;
- validation positive fraction lies in `[0.02, 0.80]`;
- largest positive-task fraction is at most `0.75`;
- every criticality score component has finite discovery variance after safe
  fallback handling;
- legal criticality predictor beats the strongest trivial baseline by at least
  `0.02` normalized validation score;
- Base leaves measurable critical-cell residual headroom;
- DySL proxy leaves measurable residual headroom for CSPR;
- simple killer does not explain CSPR;
- CSPR full beats the strongest of DySL proxy, uniform ablation, and simple
  killer by at least `0.005` normalized validation proxy Huber on
  high-criticality cells;
- exact Base passthrough and disk reload max absolute error are `<= 1e-7`;
- expected CSPR parameters receive finite nonzero gradients;
- frozen SmolVLA Base parameters receive no gradients;
- weighted objective gradient-norm ratio is at most `100x`;
- intervention fraction lies in `[0.02, 0.80]`;
- action deltas respect the Stage 0 translation, rotation, and gripper caps;
- clean low-criticality rows have p95 deltas at most `10%` of each group cap;
- postprocessed action validity is `1.0`.

## Stage 0 Stop Classes

Stage 0 must end with exactly one:

- `CSPR_STAGE_0_DATA_FAILURE`;
- `CSPR_STAGE_0_NO_USABLE_HEADROOM`;
- `CSPR_STAGE_0_DESIGN_FAILURE`;
- `CSPR_STAGE_0_IMPLEMENTATION_FAILURE`;
- `CSPR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`;

Classify failures as:

- cache, action-shape, source-file, feature-cache, label, task-coverage,
  duplicate-key, split-overlap, privileged-input, or task-only-label failure:
  `CSPR_STAGE_0_DATA_FAILURE`;
- no Base residual activity, no DySL residual headroom, no simple-killer
  residual headroom, or no legal criticality predictability:
  `CSPR_STAGE_0_NO_USABLE_HEADROOM`;
- CSPR equivalent to uniform refinement, simple killer explains the gain, DySL
  proxy dominates the claim axis, global gate activation, or nonacting
  critical-step mechanism: `CSPR_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, reload, gradient, objective-scale,
  frozen-parameter, checkpoint, action-semantics, action-validity, persistence,
  global-delta, or exception defect: `CSPR_STAGE_0_IMPLEMENTATION_FAILURE`;
- all gates pass: `CSPR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

No Stage 0 stop is a closed-loop scientific kill and no Stage 0 stop may be
rescued by changing thresholds, label construction, task/demo identities,
proxy definition, action-validity semantics, or criticality features after
seeing results.

## Bounded Validation Search

Allowed only after `CSPR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Maximum six configurations:

1. `cspr_full_cap_small_tau_0_90`;
2. `cspr_full_cap_small_tau_0_95`;
3. `cspr_full_cap_mid_tau_0_90`;
4. `cspr_full_cap_mid_tau_0_95`;
5. `cspr_full_cap_large_tau_0_90`;
6. `cspr_full_cap_large_tau_0_95`.

Cap groups:

- small: `Delta_trans = 0.005`, `Delta_rot = 0.0125`,
  `Delta_grip = 0.0625`;
- mid: `Delta_trans = 0.01`, `Delta_rot = 0.025`,
  `Delta_grip = 0.125`;
- large: `Delta_trans = 0.02`, `Delta_rot = 0.05`,
  `Delta_grip = 0.25`.

No feature set, task split, label weight, proxy definition, threshold,
comparator, stop rule, or confirmatory identity may be searched outside this
budget.

Validation score:

`S_val = 0.35 * closed_loop_success_or_proxy
       + 0.20 * clean_retention
       + 0.15 * criticality_localization
       + 0.15 * prior_and_ablation_margin
       + 0.10 * action_validity
       + 0.05 * compute_overhead`.

Ties break by clean retention, then lower intervention fraction, then smaller
cap group, then higher criticality threshold.

Do not select purely by offline action L2.

## Worker And Resume Rules

Before any expensive command, inspect PID, heartbeat, status, partial, result,
log, manifest, and exit-code artifacts. If a CSPR worker is alive, monitor it
only. If it completed, adjudicate the result and do not rerun. If it died with
valid partial rows, resume only missing keys:

`(split, task_suite, task_identity, demo_id, frame_index, source_edge_sha256,
model_or_probe, config_label)`

Completed rows must not repeat. Duplicate-key and manifest checks must run
before accepting the result.

## Current Authorization

This preregistration authorizes prototype protocol drafting next. It does not
authorize implementation, validation search, training, rollout, or
confirmatory testing until the prototype protocol is frozen.
