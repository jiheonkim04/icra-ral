# AFID-VLA Executable Prototype Protocol

Date: 2026-07-16 KST

Decision: `AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`

Method: `AFID-VLA`, Action-Factor Instruction Densification for
Base-preserving SmolVLA.

Proposal SHA-256:
`B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`

Frozen inputs:

- proposal: `reports/afid_vla/researcher_proposal.md`
- Reviewer B attack: `reports/afid_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/afid_vla/researcher_rebuttal.md`
- mathematical audit: `reports/afid_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/afid_vla/preregistration.md`

No AFID implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this protocol.

## Stage 0 Purpose

Stage 0 is a development-only implementation, data, and mechanism audit. It
decides only whether AFID may proceed to bounded validation search.

It is not a closed-loop scientific result and cannot be interpreted as a paper
claim or confirmatory test.

## Required Command Contract

Implement Stage 0 with:

- helper module: `tca_map/smolvla/afid_vla.py`;
- runner: `scripts/run_afid_vla_stage0.py`;
- focused tests: `tests/test_afid_vla.py`;
- serializer/preflight artifact:
  `reports/afid_vla/stage_0_serializer_preflight.json`;
- primary result artifact: `reports/afid_vla/stage_0_result.json`.

The runner must support the repository's WSL execution pattern:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e ./.venv/bin/python scripts/run_afid_vla_stage0.py
```

The runner may support explicit `--checkpoint`, `--data-root`, `--output-dir`,
`--resume`, `--max-rows`, and `--serializer-preflight` arguments. Defaults must
use validated local SmolVLA/LIBERO paths discovered by existing repository
helpers.

## Required Helper API

The helper module must provide deterministic utilities for:

- protocol constants for `H=50`, `D=7`, proposal hash, thresholds, caps, and
  policy names;
- canonical JSON serialization helpers;
- row-key construction and duplicate/missing/extra/split-overlap checks;
- discovery-only residual-scale construction;
- Base chunk and demonstration chunk shape/finite checks;
- compact action-factor extraction for axis, direction, gripper type, gripper
  bin, rotation, and termination/settle labels;
- factor-label health diagnostics by split, task, phase, timestep, and action
  group;
- factor-mask construction from `M_factor = 1[abs(R_t) / S_d >= 0.50]`;
- factor-prediction metrics against majority and task/phase trivial baselines;
- factor-conditioned oracle Huber diagnostics;
- transparent FineVLA action-factor proxy metrics;
- identity-initialized AFID residual gate and residual application;
- no-factor ablation with matched capacity and optimizer budget;
- standard-LoRA proxy placeholder metrics for the reviewer-killer baseline;
- groupwise action clipping for translation, rotation, and gripper dimensions;
- clean-retention and inactive/low-confidence exact-Base metrics;
- action-delta summaries by translation, rotation, and gripper groups;
- finite nonzero gradient diagnostics for AFID trainable parameters;
- frozen-Base zero-gradient diagnostics;
- weighted objective gradient-norm ratio diagnostics;
- action-validity metrics under persisted official semantics;
- Stage 0 decision taxonomy.

The helper must not import simulator environments, read reward/success/done
fields, use object poses, use future observations, or access confirmatory
identities.

## Required Artifacts

Stage 0 writes under `reports/afid_vla/`:

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

Stage 0 writes feature/model caches under `runs/afid_vla/stage0/` only if
needed.

## Worker Safety And Resume

Before launching a worker, check existing PID, heartbeat/status, partial,
result, logs, and exit-code files.

- If an existing AFID worker is alive, monitor it only.
- If a final result already exists, adjudicate that result and refuse duplicate
  execution.
- If a worker died and `stage_0_partial.json` parses, resume only missing
  row keys.
- If heartbeat is stale, verify PID, status, logs, partial JSON parseability,
  and exit-code file before deciding it is dead.

Resume may add only missing manifest keys and may not repeat completed keys.
Duplicate manifest keys, duplicate partial keys, missing keys, extra keys, and
split-overlap keys must all be zero before accepting a final result.

## Data Sources

Use only legal LIBERO demonstrations for the fixed development tasks:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Discovery demonstrations: `0..7`.

Validation demonstrations: `8..9`.

Confirmatory task/reset identities, rewards, success flags, done flags, object
poses, future observations, and rollout outcomes are forbidden.

Minimum accepted final Stage 0 manifest:

- at least `512` discovery windows;
- at least `128` validation windows;
- every task has validation rows;
- no validation task fraction exceeds `0.40`;
- duplicate manifest keys `0`;
- duplicate partial keys `0`;
- missing manifest keys `0`;
- extra partial keys `0`;
- split-overlap keys `0`.

## Required Row Key

Every manifest and partial row must include a stable key containing:

`split | task_suite | task_id | demo_id | window_start | factor_key | policy`

If multiple development-only probes are audited in one run, the key must also
include the probe label and frozen configuration label. Completed keys may not
be repeated during resume.

## Required Preflight

Before model-row work:

1. verify proposal hash equals
   `B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`;
2. verify required source documents exist;
3. persist official FineVLA asset/code status and whether policy 2 is official
   FineVLA or the transparent `finevla_action_factor_proxy`;
4. persist official SmolVLA/LIBERO action semantics;
5. verify JSON serialization of manifest rows, action chunks, factor labels,
   factor masks, prediction metrics, gradient metrics, booleans, paths, and
   nested metric dictionaries;
6. verify CUDA and official SmolVLA checkpoint availability when model decoding
   is required;
7. persist preflight failures as implementation blockers without fabricating
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

No ad hoc `[-1,1]` validity-only rule is allowed as the hard gate.

## Fixed Policy And Probe Rows

Stage 0 is an offline development audit. It must include rows for:

1. `smolvla_base`;
2. `finevla_action_factor_proxy`;
3. `afid_full`;
4. `afid_no_factor_ablation`;
5. `standard_lora`;
6. `factor_conditioned_oracle_diagnostic`;
7. `task_phase_residual_diagnostic`;
8. `mask_only_residual_diagnostic`.

Only the first five are part of the first serious comparison. Diagnostic rows
cannot replace FineVLA as policy 2 and cannot be reported as inference
methods.

## Fixed Mechanism Constants

- `H = 50`;
- `D = 7`;
- `tau_axis_motion = 0.03`;
- `tau_dir = 0.01`;
- `tau_rot = 0.02`;
- `tau_grip_event = 0.20`;
- `tau_settle = 0.015`;
- `tau_residual_mask = 0.50`;
- `tau_conf = 0.60`;
- `tau_entropy = 0.75`;
- translation cap `0.02`;
- rotation cap `0.05`;
- gripper cap `0.25`;
- factor-mask global positive-fraction pass interval `[0.02, 0.80]`;
- validation task factor-mask positive-fraction pass interval `[0.01, 0.90]`;
- factor-prediction improvement threshold `0.05` macro-F1 or `0.05` accuracy
  over the best trivial baseline for each used factor;
- factor-conditioned oracle Huber reduction threshold `2%` over Base;
- gate activation fraction pass interval `[0.02, 0.80]`;
- initialized and disk-reloaded AFID must reproduce Base within `1e-6` when
  confidence is low or the mask is inactive;
- no deterministic-action KL.

These constants may not change after Stage 0 begins.

## Required Implementation Checks

Before any detached Stage 0 launch:

1. `py_compile` passes for helper and runner.
2. Focused unit tests in `tests/test_afid_vla.py` pass.
3. Serializer preflight writes
   `reports/afid_vla/stage_0_serializer_preflight.json`.
4. The preflight validates JSON serialization for actions, factor labels,
   masks, FineVLA proxy rows, gradient metrics, booleans, paths, and nested
   metric dictionaries.
5. The official action semantics artifact is created or updated with the same
   action dimension and postprocessing assumptions used by recent SmolVLA
   Stage 0 runners.
6. Current governance tests pass.
7. Governance checker passes.
8. No live or completed AFID worker exists unless the command is explicitly
   adjudicating the existing result.

## Required Result Metrics

The runner result JSON must include:

- `final_decision`;
- `completed_model_row_count`;
- `planned_model_row_count`;
- `exception_count`;
- `manifest_row_count`;
- `partial_row_count`;
- duplicate/missing/extra/split-overlap key counts;
- `key_sets_equal`;
- `proposal_hash_ok`;
- `serializer_preflight_ok`;
- `preflight_passed`;
- `closed_loop_experiment_happened = false`;
- `simulator_load_count = 0`;
- `confirmatory_records_read = 0`;
- `training_happened = false` unless a tiny Stage 0 smoke fit is explicitly
  reported as development-only;
- `validation_search_happened = false`;
- `horizon = 50`;
- `action_dimension = 7`;
- factor-label counts and health summaries;
- factor-mask positive fraction by split, task, phase, timestep, and group;
- usable factor count;
- factor-prediction accuracy and macro-F1 versus majority baseline;
- factor-prediction accuracy and macro-F1 versus task/phase baseline;
- factor-conditioned oracle Huber reduction over Base;
- FineVLA proxy score and residual headroom;
- AFID full versus Base, FineVLA proxy, no-factor ablation, and standard LoRA;
- `identity_max_abs_error`;
- `expected_parameter_gradient_nonzero`;
- `frozen_base_gradient_count`;
- `weighted_gradient_norm_ratio_max`;
- `gate_activation_fraction`;
- action-delta summaries for translation, rotation, and gripper groups;
- `action_validity_ok`;
- `clean_retention_ok`;
- `stage_0_is_closed_loop_scientific_kill = false`;
- `valid_scientific_result = false` unless later closed-loop confirmatory
  criteria are frozen and satisfied.

## Stage 0 Decision Contract

The runner must produce exactly one final decision:

- `AFID_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `AFID_STAGE_0_NO_USABLE_HEADROOM`;
- `AFID_STAGE_0_DESIGN_FAILURE`;
- `AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`;
- `AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Decision precedence:

1. source, hash, serialization, identity, reload, gradient, action-validity, or
   exception defects are implementation/objective-scale failures;
2. collapsed action factors, unusable factor masks, insufficient task coverage,
   duplicate keys, split overlap, or illegal label construction are
   data/supervision failures;
3. absent factor-conditioned residual headroom or no headroom beyond the
   FineVLA proxy is no-headroom;
4. unobservable factors, AFID equivalence to FineVLA proxy, no-factor ablation
   explanation, standard-LoRA explanation, global gate activation, or nonacting
   gate activation is design failure;
5. all gates pass means pass to bounded validation.

## Serializer Preflight

`--serializer-preflight` must:

- canonicalize one representative row key;
- round-trip tensors, factor labels, masks, and metrics through JSON-safe
  serialization;
- produce a deterministic SHA-256 fixture hash;
- persist `reports/afid_vla/stage_0_serializer_preflight.json`;
- include a healthy synthetic decision fixture equal to
  `AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION`;
- pass before full Stage 0 is launch-eligible.

## Current Authorization

This prototype protocol authorizes implementation and preflight validation
next. It does not authorize Stage 0 launch until implementation validation and
worker-safety checks are complete.
