# MHS-VLA Executable Prototype Protocol

Date: 2026-07-16 KST

Decision: `MHS_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`

Method: `MHS-VLA`, Mamba History State for Base-preserving SmolVLA.

Proposal SHA-256:
`BBDF67AE3EC4BD9D025707A8BB3A5008BAB5EB5C691D02D44516157802A87BF3`

Frozen inputs:

- proposal: `reports/mhs_vla/researcher_proposal.md`
- Reviewer B attack: `reports/mhs_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/mhs_vla/researcher_rebuttal.md`
- mathematical audit: `reports/mhs_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/mhs_vla/preregistration.md`

No MHS implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this protocol.

## Stage 0 Purpose

Stage 0 is a development-only implementation, data, and mechanism audit. It
decides only whether MHS may proceed to bounded validation search.

It is not a closed-loop scientific result and cannot be interpreted as a paper
claim or confirmatory test.

## Required Command Contract

Implement Stage 0 with:

- helper module: `tca_map/smolvla/mhs_vla.py`;
- runner: `scripts/run_mhs_vla_stage0.py`;
- focused tests: `tests/test_mhs_vla.py`;
- serializer/preflight artifact:
  `reports/mhs_vla/stage_0_serializer_preflight.json`;
- primary result artifact: `reports/mhs_vla/stage_0_result.json`.

The runner must support the repository's WSL execution pattern:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e ./.venv/bin/python scripts/run_mhs_vla_stage0.py
```

The runner may support explicit `--checkpoint`, `--data-root`, `--output-dir`,
`--resume`, `--max-rows`, and `--serializer-preflight` arguments. Defaults must
reuse validated local SmolVLA/LIBERO artifacts and cached Base chunks when
available.

## Required Helper API

The helper module must provide deterministic utilities for:

- protocol constants for `K=50`, `D=7`, `L=8`, proposal hash, caps,
  thresholds, and policy names;
- canonical JSON serialization helpers;
- row-key construction and duplicate/missing/extra/split-overlap checks;
- deterministic history identity construction from split, task, demo, and
  window start;
- Base chunk, demonstration chunk, and history-window shape/finite checks;
- legal history feature summaries from previous observations, proprioception,
  executed or demonstration training actions, instruction/task embedding, and
  Base chunk statistics;
- leave-one-out current-frame and history-neighbor diagnostics within the same
  split and task;
- frozen `m_i` and `z_i` construction;
- discovery-only `z_i` normalization statistics;
- label health diagnostics, including entropy, positive/negative counts,
  task coverage, and largest positive-task fraction;
- history-predictability diagnostics versus majority, task-only, and
  current-frame-only baselines;
- MTIL history-state proxy metrics without Base passthrough;
- no-history-state ablation metrics with matched residual/gate capacity;
- standard-LoRA proxy metrics for the single reviewer-killer baseline;
- identity-initialized MHS residual/gate application;
- groupwise action clipping for translation, rotation, and gripper dimensions;
- clean-retention and inactive Base metrics;
- action-delta summaries by translation, rotation, and gripper groups;
- finite nonzero gradient diagnostics for the history encoder, residual head,
  gate, and auxiliary head;
- frozen-Base zero-gradient diagnostics;
- weighted objective gradient-norm ratio diagnostics;
- action-validity metrics under persisted official semantics;
- Stage 0 decision taxonomy.

The helper must not import simulator environments, read reward/success/done
fields, use object poses, use future observations, use demonstration actions at
inference, or access confirmatory identities.

## Required Artifacts

Stage 0 writes under `reports/mhs_vla/`:

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

Stage 0 writes feature/model caches under `runs/mhs_vla/stage0/` only if
needed.

## Worker Safety And Resume

Before launching a worker, check existing PID, heartbeat/status, partial,
result, logs, and exit-code files.

- If an existing MHS worker is alive, monitor it only.
- If a final result already exists, adjudicate that result and refuse
  duplicate execution.
- If a worker died and `stage_0_partial.json` parses, resume only missing row
  keys.
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
poses, future observations, rollout outcomes, and confirmatory policy actions
are forbidden.

Minimum accepted final Stage 0 manifest:

- at least `512` discovery windows;
- at least `128` validation windows;
- at least `128` unmasked validation labels;
- every task has validation rows;
- no validation task fraction exceeds `0.40`;
- validation positive count at least `8`;
- validation negative count at least `8`;
- validation positive fraction in `[0.02, 0.80]`;
- largest positive-task fraction at most `0.75`;
- duplicate manifest keys `0`;
- duplicate partial keys `0`;
- missing manifest keys `0`;
- extra partial keys `0`;
- split-overlap keys `0`.

## Required Row Key

Every manifest and partial row must include a stable key containing:

`split | task_suite | task_id | demo_id | window_start | history_identity |
policy | config_label`

If multiple development-only probes are audited in one run, the key must also
include the probe label. Completed keys may not be repeated during resume.

## Required Preflight

Before model-row work:

1. verify proposal hash equals
   `BBDF67AE3EC4BD9D025707A8BB3A5008BAB5EB5C691D02D44516157802A87BF3`;
2. verify required source documents exist;
3. persist official MTIL asset/code status and whether policy 2 is official
   MTIL or the transparent `mtil_history_state_proxy`;
4. persist official SmolVLA/LIBERO action semantics;
5. verify JSON serialization of manifest rows, history identities, action
   chunks, history summaries, label metrics, gradient metrics, booleans, paths,
   and nested metric dictionaries;
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
2. `mtil_history_state_proxy`;
3. `mhs_full`;
4. `mhs_no_history_state_ablation`;
5. `standard_lora`;
6. `history_oracle_diagnostic`;
7. `current_frame_baseline_diagnostic`;
8. `task_only_baseline_diagnostic`;
9. `majority_baseline_diagnostic`.

Only the first five are part of the first serious comparison. Diagnostic rows
cannot replace MTIL as policy 2 and cannot be reported as inference methods.

## Fixed Mechanism Constants

- action horizon `K = 50`;
- action dimension `D = 7`;
- history length `L = 8`;
- history state dimension `128`;
- residual hidden dimension `128`;
- Huber delta `0.01`;
- Base residual threshold `0.02`;
- history usefulness margin `0.01`;
- history-predictability margin `0.02`;
- history-neighbor margin `0.01`;
- MHS-over-strongest-baseline validation proxy margin `0.005`;
- translation cap `0.02`;
- rotation cap `0.05`;
- gripper cap `0.25`;
- intervention fraction pass interval `[0.02, 0.80]`;
- initialized and disk-reloaded MHS must reproduce Base within `1e-7`;
- weighted objective gradient-norm ratio max `20x` median;
- no deterministic-action KL.

These constants may not change after Stage 0 begins.

## Required Implementation Checks

Before any detached Stage 0 launch:

1. `py_compile` passes for helper and runner.
2. Focused unit tests in `tests/test_mhs_vla.py` pass.
3. Serializer preflight writes
   `reports/mhs_vla/stage_0_serializer_preflight.json`.
4. The preflight validates JSON serialization for actions, history identities,
   history summaries, labels, proxy metrics, gradient metrics, booleans, paths,
   and nested metric dictionaries.
5. The official action semantics artifact is created or updated with the same
   action dimension and postprocessing assumptions used by recent SmolVLA
   Stage 0 runners.
6. Current governance tests pass.
7. Governance checker passes.
8. No live or completed MHS worker exists unless the command is explicitly
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
- `history_length = 8`;
- history-window counts and coverage summaries;
- label health summaries;
- `m_i` positive/negative counts and positive fraction;
- `z_i` normalization statistics;
- current-frame and history-neighbor errors;
- history-predictability metrics versus majority, task-only, and current-frame
  baselines;
- MTIL proxy score and residual headroom;
- MHS full versus Base, MTIL proxy, no-history ablation, and standard LoRA;
- `identity_max_abs_error`;
- checkpoint reload status;
- `expected_parameter_gradient_nonzero`;
- `frozen_base_gradient_count`;
- `weighted_gradient_norm_ratio_max`;
- objective magnitudes for `L_res`, `L_gate`, `L_hist`, `L_clean`, and
  `L_valid`;
- `intervention_fraction`;
- action-delta summaries for translation, rotation, and gripper groups;
- `action_validity_ok`;
- `clean_retention_ok`;
- `stage_0_is_closed_loop_scientific_kill = false`;
- `valid_scientific_result = false` unless later closed-loop confirmatory
  criteria are frozen and satisfied.

## Stage 0 Decision Contract

The runner must produce exactly one final decision:

- `MHS_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `MHS_STAGE_0_NO_USABLE_HEADROOM`;
- `MHS_STAGE_0_DESIGN_FAILURE`;
- `MHS_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Decision precedence:

1. source, hash, serialization, identity, reload, gradient, action-validity, or
   exception defects are implementation/optimization failures;
2. collapsed labels, invalid history identities, insufficient task coverage,
   duplicate keys, split overlap, illegal label construction, or task-only
   labels are data/supervision failures;
3. absent Base residual activity, absent history-over-current-frame benefit,
   unobservable history target, or no headroom beyond the MTIL proxy is
   no-headroom;
4. MHS equivalence to no-history ablation, standard-LoRA explanation, MTIL
   dominance, global intervention, or nonacting intervention is design failure;
5. all gates pass means pass to bounded validation.

## Serializer Preflight

`--serializer-preflight` must:

- canonicalize one representative row key;
- round-trip tensors, action chunks, history summaries, history identities,
  label metrics, proxy metrics, and nested diagnostics through JSON-safe
  serialization;
- produce a deterministic SHA-256 fixture hash;
- persist `reports/mhs_vla/stage_0_serializer_preflight.json`;
- include a healthy synthetic decision fixture equal to
  `MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION`;
- pass before full Stage 0 is launch-eligible.

## Current Authorization

This prototype protocol authorizes implementation and preflight validation
next. It does not authorize Stage 0 launch until implementation validation and
worker-safety checks are complete.
