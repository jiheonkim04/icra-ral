# CSPR-VLA Executable Prototype Protocol

Date: 2026-07-16 KST

Decision: `CSPR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`

Method: `CSPR-VLA`, Critical-Step Selective Policy Refinement for SmolVLA.

Proposal SHA-256:
`CC83324F9AB37DAEEF4E2BA158C821F336383A8C4F96ADFFF4DE7B79E276D0D7`

Frozen inputs:

- proposal: `reports/cspr_vla/researcher_proposal.md`
- Reviewer B attack: `reports/cspr_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/cspr_vla/researcher_rebuttal.md`
- mathematical audit: `reports/cspr_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/cspr_vla/preregistration.md`

No CSPR implementation, training, validation search, rollout, simulator
evaluation, or confirmatory-test tuning has happened before this protocol.

## Stage 0 Purpose

Stage 0 is a development-only source, data, objective, implementation, and
mechanism audit. It decides only whether CSPR may proceed to bounded validation
search.

It is not a closed-loop scientific result and cannot be interpreted as a paper
claim or confirmatory test.

## Required Command Contract

Implement Stage 0 with:

- helper module: `tca_map/smolvla/cspr_vla.py`;
- runner: `scripts/run_cspr_vla_stage0.py`;
- focused tests: `tests/test_cspr_vla.py`;
- serializer/preflight artifact:
  `reports/cspr_vla/stage_0_serializer_preflight.json`;
- primary result artifact: `reports/cspr_vla/stage_0_result.json`.

The runner must support the repository's WSL execution pattern:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e ./.venv/bin/python scripts/run_cspr_vla_stage0.py
```

The runner may support explicit `--data-root`, `--output-dir`, `--resume`,
`--max-rows`, and `--serializer-preflight` arguments. Defaults must use the
fixed CCIF cached SmolVLA Base rows and repository helpers for local path
resolution.

## Worker Safety And Resume

Before launching a worker, check existing PID, heartbeat/status, partial,
result, logs, manifest, and exit-code files.

- If an existing CSPR worker is alive, monitor it only.
- If a final result already exists, adjudicate that result and refuse duplicate
  execution.
- If a worker died and `stage_0_partial.json` parses, resume only missing row
  keys.
- If heartbeat is stale, verify PID, status, logs, partial JSON parseability,
  manifest integrity, and exit-code file before deciding it is dead.

Resume may add only missing manifest keys and may not repeat completed keys.
Duplicate manifest keys, duplicate partial keys, missing keys, extra keys, and
split-overlap keys must all be zero before accepting a final result.

If a Windows Efficiency Mode, VM throttling, or other resource-contention
interval occurs, record it in status/result artifacts. Wall-clock latency,
throughput, utilization, and efficiency measured during that interval may not
be used as final paper evidence. Synchronous offline rows may remain valid only
when no timeout, exception, semantic change, identity change, or duplicate row
occurred.

## Required Helper API

The helper module must provide deterministic utilities for:

- protocol constants for `H = 50`, `D = 7`, visual feature dimension `960`,
  proprioception dimension `8`, proposal hash, cap groups, threshold quantiles,
  soft-gate temperature, label weights, policy names, and stop decisions;
- canonical JSON serialization helpers;
- row-key construction and duplicate/missing/extra/split-overlap checks;
- loading and validating the fixed CCIF cached Base rows;
- Base chunk, demonstration chunk, feature cache, proprioception, and task
  identity shape/finite checks;
- official SmolVLA/LIBERO action semantics and postprocessor validity checks;
- official DySL-VLA asset/code inspection and proxy mismatch reporting;
- discovery-only robust normalizer fitting for Base error, curvature, and
  acceleration;
- criticality score and `C*` label construction with the frozen weights
  `w_e = 1.0`, `w_k = 0.5`, `w_a = 0.5`, and `w_g = 1.0`;
- label-health diagnostics, including positive/negative counts, variance,
  task coverage, frame-index quartile coverage, and all-zero/all-one checks;
- legal deployment-observable feature construction from current observation
  features, proprioception, task/language identity, and Base chunk summaries;
- trivial baseline diagnostics for gripper transition, Base curvature/velocity,
  task-mean criticality, and frame-index audit proxy;
- legal criticality-predictor diagnostics versus the strongest trivial
  baseline;
- transparent `dysl_action_importance_proxy` diagnostics without CSPR residual
  action correction;
- `cspr_uniform_refinement_ablation` with matched residual capacity, caps,
  optimizer budget, training rows, and clean-retention term;
- `critical_step_threshold_simple_killer` without learned residuals;
- identity-initialized CSPR residual and gate application;
- groupwise residual clipping for translation, rotation, and gripper
  dimensions;
- exact Base passthrough and disk-reload diagnostics;
- objective magnitude and gradient-norm diagnostics for `L_crit`, `L_fit`,
  `L_keep`, and `L_bound`;
- finite nonzero gradient diagnostics for the criticality predictor and
  residual head;
- zero-gradient diagnostics for frozen SmolVLA Base parameters;
- gate activation, intervention frequency, clean-retention, and action-delta
  summaries by translation, rotation, and gripper groups;
- action-validity metrics under persisted official semantics;
- Stage 0 decision taxonomy.

The helper must not import simulator environments, read reward/success/done
fields, use object poses, use future observations, use demonstration actions at
inference, use demonstration frame index at inference, or access confirmatory
identities.

## Required Artifacts

Stage 0 writes under `reports/cspr_vla/`:

- `stage_0_preflight.json`;
- `stage_0_manifest.json`;
- `stage_0_partial.json`;
- `stage_0_status.json`;
- `stage_0_heartbeat.json`;
- `stage_0_result.json`;
- `stage_0_result.md`;
- `stage_0_adjudication.md`;
- `stage_0_action_semantics.json`;
- `stage_0_official_prior_asset_check.json`;
- `stage_0_serializer_preflight.json`;
- `stage_0_implementation_blocker.json` on exception;
- `stage_0_pid.txt`;
- `stage_0_exit_code.txt`;
- `stage_0_stdout.log` and `stage_0_stderr.log` when launched detached.

Stage 0 writes caches under `runs/cspr_vla/stage0/` only if needed.

## Data Sources

Use only the fixed CCIF cached SmolVLA Base rows:

- `reports/ccif_vla/stage_0_partial.json`;
- `reports/ccif_vla/stage_0_manifest.json`.

Required row properties:

- `model_or_probe == smolvla_base`;
- `base_chunk_cache_path` exists and loads a finite `[50, 7]` Base chunk;
- `feature_cache_path` exists and loads finite current visual features;
- source demonstration action chunk loads as finite `[50, 7]`;
- task identity is one of the four fixed tasks;
- demo id is in `0..9`;
- row key is unique;
- no split overlap between discovery and validation keys.

Development tasks:

1. `libero_10/task_5`;
2. `libero_goal/task_5`;
3. `libero_object/task_3`;
4. `libero_spatial/task_3`.

Discovery demonstrations: `0..7`.

Validation demonstrations: `8..9`.

Minimum accepted final Stage 0 manifest:

- exactly or at least `512` discovery rows from the fixed cache;
- exactly or at least `128` validation rows from the fixed cache;
- every validation task has rows;
- no validation task fraction exceeds `0.40`;
- duplicate manifest keys `0`;
- duplicate partial keys `0`;
- missing manifest keys `0`;
- extra partial keys `0`;
- split-overlap keys `0`.

Confirmatory task/reset identities, rewards, success flags, done flags, object
poses, future observations, rollout outcomes, and confirmatory policy actions
are forbidden.

## Required Row Key

Every manifest and partial row must include a stable key containing:

`split | task_suite | task_identity | demo_id | frame_index |
source_edge_sha256 | model_or_probe | config_label`

If multiple development-only probes are audited in one run, the key must also
include the probe label. Completed keys may not be repeated during resume.

## Required Preflight

Before model-row work:

1. verify proposal hash equals
   `CC83324F9AB37DAEEF4E2BA158C821F336383A8C4F96ADFFF4DE7B79E276D0D7`;
2. verify required source documents exist;
3. verify the fixed CCIF cache and manifest parse and contain only legal
   development rows for CSPR;
4. persist official DySL-VLA asset/code status and whether policy 2 is official
   DySL-VLA or the transparent `dysl_action_importance_proxy`;
5. persist official SmolVLA/LIBERO action semantics;
6. verify JSON serialization of manifest rows, cache paths, action chunks,
   feature vectors, label metrics, criticality scores, gate values, gradient
   metrics, booleans, paths, and nested metric dictionaries;
7. verify CUDA and official SmolVLA checkpoint availability only when model
   decoding is required beyond existing caches;
8. persist preflight failures as implementation blockers without fabricating
   partial rows.

## Required Action Semantics

`stage_0_action_semantics.json` must include:

- model-native action shape;
- postprocessor/unnormalizer class and parameters;
- environment action shape;
- environment action-space low/high if exposed;
- gripper convention;
- finite checks;
- action-space or equivalent official environment validation result for Base;
- the final boolean action-validity definition applied to every policy/probe.

No ad hoc normalized `[-1,1]` validity-only rule is allowed as the hard gate.

## Fixed Policy And Probe Rows

Stage 0 is an offline development audit. It must include rows for:

1. `smolvla_base`;
2. `dysl_action_importance_proxy`;
3. `cspr_full`;
4. `cspr_uniform_refinement_ablation`;
5. `critical_step_threshold_simple_killer`;
6. `criticality_label_health_diagnostic`;
7. `criticality_predictability_diagnostic`;
8. `identity_passthrough_reload_diagnostic`;
9. `objective_gradient_scale_diagnostic`.

Only the first five are policy comparisons. Diagnostic rows cannot replace the
DySL prior as policy 2 and cannot be reported as inference methods.

## Frozen Stage 0 Constants

Default Stage 0 smoke configuration:

- residual cap group: `mid`;
- `Delta_trans = 0.01`;
- `Delta_rot = 0.025`;
- `Delta_grip = 0.125`;
- criticality threshold: discovery `0.95` quantile;
- soft-gate temperature for gradient smoke: `0.05`;
- identity tolerance: `1e-7`;
- weighted objective gradient-norm ratio alert: `100.0`.

Criticality score:

`q_t,d = w_e * norm_err_t,d + w_k * norm_curv_t,d
       + w_a * norm_acc_t,d + w_g * grip_event_t,d`.

Label:

`C*_t,d = 1[q_t,d >= q_tau]`.

Action:

`A = postprocess(B + G * Delta * tanh(R))`.

No deterministic-action KL is allowed.

## Required Metrics

Each result must report:

- planned and completed row counts;
- exception count and last exception;
- duplicate/missing/extra/split-overlap key counts;
- source hashes and manifest hash;
- proposal hash match;
- no reward/success/done/object-pose/future-observation/confirmatory reads;
- discovery and validation row counts by task and demo id;
- Base chunk shape, finite fraction, min, max, and postprocessor validity;
- feature cache shape, finite fraction, variance, and source hash status;
- source demonstration action shape and finite fraction;
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
- action-validity rate after official postprocessing;
- recorded resource-contention intervals, if any.

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
- `CSPR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

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

Validation score:

`S_val = 0.35 * closed_loop_success_or_proxy
       + 0.20 * clean_retention
       + 0.15 * criticality_localization
       + 0.15 * prior_and_ablation_margin
       + 0.10 * action_validity
       + 0.05 * compute_overhead`.

Ties break by clean retention, then lower intervention fraction, then smaller
cap group, then higher criticality threshold.

Do not select purely by offline action L2. Do not use confirmatory-test
outcomes to select or retune the final configuration.

## Current Authorization

This protocol authorizes Stage 0 implementation and serializer/preflight
testing next. It does not authorize bounded validation search, rollout,
confirmatory-test access, or paper-candidate claims until Stage 0 is completed
and adjudicated under this frozen protocol.
