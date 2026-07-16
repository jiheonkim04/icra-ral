# LCG-VLA Executable Prototype Protocol

Date: 2026-07-16 KST

Decision: `LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`

Method: `LCG-VLA`, Language-Contrastive Guidance for Base-preserving SmolVLA
actions.

Proposal SHA-256:
`F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E`

Frozen inputs:

- proposal: `reports/lcg_vla/researcher_proposal.md`
- Reviewer B attack: `reports/lcg_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/lcg_vla/researcher_rebuttal.md`
- mathematical audit: `reports/lcg_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/lcg_vla/preregistration.md`

No LCG implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this protocol.

## Stage 0 Purpose

Stage 0 is a development-only implementation/data/mechanism audit. It decides
only whether LCG may proceed to bounded validation search.

It is not a closed-loop scientific result.

## Required Command Contract

Implement Stage 0 with:

- helper module: `tca_map/smolvla/lcg_vla.py`;
- runner: `scripts/run_lcg_vla_stage0.py`;
- focused tests: `tests/test_lcg_vla.py`;
- serializer/preflight artifact:
  `reports/lcg_vla/stage_0_serializer_preflight.json`;
- primary result artifact: `reports/lcg_vla/stage_0_result.json`.

The runner must support the repository's WSL execution pattern:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e ./.venv/bin/python scripts/run_lcg_vla_stage0.py
```

The runner may support explicit `--checkpoint`, `--data-root`, `--output-dir`,
`--resume`, and `--max-rows` arguments. Defaults must use validated local
SmolVLA/LIBERO paths discovered by existing repository helpers.

## Required Helper API

The helper module must provide:

- protocol constants for `H=50`, `D=7`, `l_null=""`, caps, and proposal hash;
- canonical JSON serialization helpers;
- row-key construction and duplicate/missing/extra/split-overlap checks;
- discovery-only contrast-scale construction;
- language-contrast tensor and hard mask construction;
- groupwise action clipping;
- identity-initialized LCG gate/action application;
- CAG proxy application for `beta in {0.25, 0.5, 1.0}`;
- no-language-contrast ablation application;
- standard-LoRA proxy placeholder metrics for Stage 0 offline comparison;
- contrast/residual noncollapse diagnostics;
- contrast-residual Spearman diagnostic;
- clean-retention and inactive-gate exact-Base metrics;
- action-delta summaries by translation, rotation, and gripper groups;
- action-validity metrics under persisted official semantics;
- Stage 0 decision taxonomy.

The helper must not import simulator environments, read reward/success/done
fields, or access confirmatory identities.

## Required Artifacts

Stage 0 writes under `reports/lcg_vla/`:

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

Stage 0 writes feature/model caches under `runs/lcg_vla/stage0/` only if
needed.

## Worker Safety And Resume

Before launching a worker, check existing PID, heartbeat/status, partial,
result, logs, and exit-code files.

- If an existing worker is alive, monitor it only.
- If a final result already exists, adjudicate that result and refuse duplicate
  execution.
- If a worker died and `stage_0_partial.json` parses, resume only missing
  `row_key`s.
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

Every manifest and partial row must include a stable `row_key` containing:

`partition | suite | task_identity | source_edge_sha256 | demo_id | frame_index | instruction_variant | policy_probe`

If multiple LCG settings are audited in one run, the key must also include the
configuration label, `beta`, clean-retention coefficient, and gate/cap family.

## Required Preflight

Before model-row work:

1. verify proposal hash equals
   `F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E`;
2. verify required source documents exist;
3. persist official CAG asset/code status;
4. persist official SmolVLA/LIBERO action semantics;
5. verify JSON serialization of manifest rows, actions, contrast masks,
   residual diagnostics, gradient metrics, booleans, paths, and nested metric
   dictionaries;
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
2. `counterfactual_action_guidance_proxy`;
3. `lcg_full`;
4. `lcg_no_language_contrast_ablation`;
5. `standard_lora_proxy`;
6. `contrast_magnitude_only_gate`;
7. `task_phase_residual`;
8. `masked_residual_oracle_diagnostic`.

`standard_lora_proxy` and `masked_residual_oracle_diagnostic` are Stage 0
offline probes only. They are not closed-loop policies.

The first serious closed-loop comparison remains exactly five policies:
`smolvla_base`, `counterfactual_action_guidance_proxy`, `lcg_full`,
`lcg_no_language_contrast_ablation`, and `standard_lora`.

## Fixed Mechanism Constants

- `H = 50`;
- `D = 7`;
- `l_null = ""`;
- `tau_lang = 0.25`;
- translation cap `0.02`;
- rotation cap `0.05`;
- gripper cap `0.25`;
- CAG proxy beta candidates `{0.25, 0.5, 1.0}`;
- language-mask positive fraction pass interval `[0.05, 0.95]`;
- gate activation fraction pass interval `[0.02, 0.80]`;
- contrast-residual Spearman threshold `0.05`;
- contrast-conditioned probe improvement threshold `1%` over task/phase
  residual baseline;
- initialized and disk-reloaded LCG must reproduce Base within `1e-6`;
- no deterministic-action KL.

## Required Implementation Checks

Before any detached Stage 0 launch:

1. `py_compile` passes for helper and runner.
2. Focused unit tests in `tests/test_lcg_vla.py` pass.
3. Serializer preflight writes
   `reports/lcg_vla/stage_0_serializer_preflight.json`.
4. The preflight validates JSON serialization for actions, contrast scales,
   masks, CAG proxy rows, gradient metrics, booleans, paths, and nested metric
   dictionaries.
5. The official action semantics artifact is created or updated with the same
   action dimension and postprocessing assumptions used by recent SmolVLA Stage
   0 runners.
6. Current governance tests pass.
7. Governance checker passes.
8. No live or completed LCG worker exists unless the command is explicitly
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
- `null_instruction = ""`;
- `contrast_positive_fraction`;
- `contrast_residual_spearman`;
- `contrast_probe_beats_task_phase_baseline`;
- `best_cag_proxy_score`;
- `cag_proxy_residual_headroom`;
- `language_mask_all_zero`;
- `language_mask_all_one`;
- `identity_max_abs_error`;
- `expected_parameter_gradient_nonzero`;
- `frozen_base_gradient_count`;
- `weighted_gradient_norm_ratio_max`;
- `lora_explains`;
- `no_language_ablation_explains`;
- `action_validity_ok`;
- `clean_retention_ok`;
- `stage_0_is_closed_loop_scientific_kill = false`;
- `valid_scientific_result = false` unless later closed-loop confirmatory
  criteria are frozen and satisfied.

## Stage 0 Decision Contract

The runner must produce exactly one final decision:

- `LCG_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `LCG_STAGE_0_NO_USABLE_HEADROOM`;
- `LCG_STAGE_0_DESIGN_FAILURE`;
- `LCG_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`;
- `LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Decision precedence:

1. source, hash, serialization, identity, gradient, action-validity, or
   exception defects are implementation/objective-scale failures;
2. collapsed null branch, contrast, language mask, residual labels, or task
   coverage are data/supervision failures;
3. CAG proxy dominance or absent masked residual headroom is no-headroom;
4. contrast nonpredictiveness, CAG coefficient equivalence,
   no-language-ablation equivalence, standard-LoRA explanation, or global action
   editing is design failure;
5. all gates pass means pass to bounded validation.

## Serializer Preflight

`--serializer-preflight` must:

- canonicalize one representative row key;
- round-trip tensors and metrics through JSON-safe serialization;
- produce a deterministic SHA-256 fixture hash;
- persist `reports/lcg_vla/stage_0_serializer_preflight.json`;
- include a healthy synthetic decision fixture equal to
  `LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION`;
- pass before full Stage 0 is launch-eligible.

## Current Authorization

This prototype protocol authorizes implementation and preflight validation
next. It does not authorize Stage 0 launch until implementation validation and
worker-safety checks are complete.
