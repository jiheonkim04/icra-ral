# URF-VLA Executable Prototype Protocol

Date: 2026-07-16 KST

Decision: `URF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

Method: `URF-VLA`, Uncertainty-Routed Residual Flow for Base-preserving
SmolVLA chunks.

Proposal SHA-256:
`E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532`

Frozen inputs:

- proposal: `reports/urf_vla/researcher_proposal.md`
- Reviewer B attack: `reports/urf_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/urf_vla/researcher_rebuttal.md`
- mathematical audit: `reports/urf_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/urf_vla/preregistration.md`

No URF implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this protocol.

## Stage 0 Purpose

Stage 0 is a development-only implementation/data/mechanism audit. It decides
only whether URF may proceed to bounded validation search.

It is not a closed-loop scientific result.

## Required Command Contract

Implement Stage 0 with:

- helper module: `tca_map/smolvla/urf_vla.py`;
- runner: `scripts/run_urf_vla_stage0.py`;
- focused tests: `tests/test_urf_vla.py`;
- serializer/preflight artifact:
  `reports/urf_vla/stage_0_serializer_preflight.json`;
- primary result artifact: `reports/urf_vla/stage_0_result.json`.

The runner must support the repository's WSL execution pattern:

```powershell
wsl.exe --cd /mnt/c/Users/jiheo/tca_map -e ./.venv/bin/python scripts/run_urf_vla_stage0.py
```

The runner may also support explicit `--checkpoint`, `--data-root`,
`--output-dir`, `--resume`, and `--max-rows` arguments, but defaults must use
the validated local SmolVLA/LIBERO paths discovered by existing repository
helpers.

## Required Artifacts

Stage 0 writes under `reports/urf_vla/`:

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
- `stage_0_preflight.json`;
- `stage_0_implementation_blocker.json` on exception;
- `stage_0_pid.txt`;
- `stage_0_exit_code.txt`;
- `stage_0_stdout.log` and `stage_0_stderr.log` when launched detached.

Stage 0 writes feature/model caches under `runs/urf_vla/stage0/`.

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

`partition | suite | task_identity | source_edge_sha256 | demo_id | frame_index | policy_probe`

If multiple URF settings are audited in one run, the key must also include
`g_max`, `lambda_clean`, `tau_g_family`, and the proxy/variant label.

## Required Preflight

Before model-row work:

1. verify proposal hash equals
   `E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532`;
2. verify required source documents exist;
3. persist official SUREFlow asset/code status;
4. persist official SmolVLA/LIBERO action semantics;
5. verify JSON serialization of manifest rows, NumPy values, tensors,
   booleans, nested metric dictionaries, and route/uncertainty calibration
   bins;
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
2. `sureflow_uncertainty_residual_proxy` or official `sureflow` if installed
   and verified;
3. `urf_full`;
4. `urf_no_uncertainty_route_ablation`;
5. `standard_lora_proxy`;
6. `task_phase_residual`;
7. `residual_magnitude_route`;
8. `homoscedastic_residual`;
9. `stochastic_disagreement_route` if cheap enough under the local budget;
10. `perturbation_disagreement_route` if cheap enough under the local budget.

`standard_lora_proxy` is a Stage 0 offline proxy only. It may use a transparent
same-budget residual/adaptation probe, but it is not a closed-loop LoRA result.

The first serious closed-loop comparison remains exactly five policies:
`smolvla_base`, `sureflow_uncertainty_residual_proxy` or official `sureflow`,
`urf_full`, `urf_no_uncertainty_route_ablation`, and `standard_lora`.

## Fixed Mechanism Constants

- `H = 50`;
- `D = 7`;
- residual scale floor `1e-4`;
- residual scale ceiling `10.0`;
- log variance floor `-8`;
- log variance ceiling `4`;
- default residual cap `r_max = 2.0` normalized residual units;
- default `kappa = 1.0`;
- route positive fraction pass interval `[0.02, 0.80]`;
- uncertainty monotonicity Spearman threshold `rho >= 0.20` or binned
  monotonic fallback from preregistration;
- initialized residual and action gate are zero effect;
- initialized and disk-reloaded URF must reproduce Base within `1e-6`;
- no deterministic-action KL.

## Required Implementation Checks

Before any detached Stage 0 launch:

1. `py_compile` passes for helper and runner.
2. Focused unit tests in `tests/test_urf_vla.py` pass.
3. Serializer preflight writes
   `reports/urf_vla/stage_0_serializer_preflight.json`.
4. The preflight validates JSON serialization for actions, residual scales,
   route labels, uncertainty bins, calibration metrics, gradients, booleans,
   paths, and nested metric dictionaries.
5. The official action semantics artifact is created or updated with the same
   action dimension and postprocessing assumptions used by recent SmolVLA Stage
   0 runners.

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
- `residual_horizon = 50`;
- `action_dimension = 7`;
- `residual_scale_min`;
- `residual_scale_max`;
- `collapsed_residual_scale_count`;
- `residual_target_noncollapsed_by_group`;
- `route_label_positive_fraction`;
- `route_label_noncollapsed_by_task`;
- `heteroscedastic_residual_huber`;
- `homoscedastic_residual_huber`;
- `task_phase_residual_huber`;
- `hetero_beats_homoscedastic_relative`;
- `hetero_beats_task_phase_relative`;
- `uncertainty_strata_count`;
- `uncertainty_monotonicity_spearman`;
- `uncertainty_monotonicity_passed`;
- `sureflow_proxy_huber`;
- `urf_full_huber`;
- `urf_no_uncertainty_route_ablation_huber`;
- `standard_lora_proxy_huber`;
- `urf_minus_sureflow_proxy_relative`;
- `urf_minus_ablation_relative`;
- `base_to_expert_huber`;
- `base_residual_headroom_ok`;
- `route_activation_fraction`;
- `route_all_zero`;
- `route_all_one`;
- `route_globally_active`;
- `uncertainty_enters_route_gate`;
- `action_validity_ok`;
- `identity_max_abs_error`;
- `checkpoint_reload_ok`;
- `finite_objectives_and_gradients`;
- `urf_gradient_nonzero`;
- `frozen_parameter_gradient_count`;
- `weighted_gradient_norm_ratio_max`;
- `translation_delta_p95`;
- `rotation_delta_p95`;
- `gripper_delta_p95`;
- `timing_throughput_resource_evidence_eligible_for_paper = false`.

## Frozen Decision Rule

Return exactly one:

- `URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `URF_STAGE_0_NO_USABLE_HEADROOM`;
- `URF_STAGE_0_DESIGN_FAILURE`;
- `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Decision order:

1. If source artifacts, proposal hash, serialization, or split integrity fails:
   `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.
2. If residual scales, residual targets, route labels, uncertainty strata, or
   task/phase/action-group coverage collapse:
   `URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.
3. If Base residuals are too small, mostly postprocessor noise, or leave no
   plausible intervention target:
   `URF_STAGE_0_NO_USABLE_HEADROOM`.
4. If heteroscedastic residual prediction does not beat homoscedastic and
   task/phase residual baselines by the preregistered margins:
   `URF_STAGE_0_NO_USABLE_HEADROOM`.
5. If uncertainty does not enter the route gate:
   `URF_STAGE_0_DESIGN_FAILURE`.
6. If uncertainty strata fail the monotonicity gate:
   `URF_STAGE_0_DESIGN_FAILURE`.
7. If SUREFlow proxy leaves no URF headroom of at least `5%` relative Huber or
   `0.005` absolute normalized Huber:
   `URF_STAGE_0_NO_USABLE_HEADROOM`.
8. If `urf_no_uncertainty_route_ablation` is equivalent to URF full or explains
   the full signal:
   `URF_STAGE_0_DESIGN_FAILURE`.
9. If route activation is all-zero, all-one, or globally active outside the
   frozen `[0.02, 0.80]` activation interval:
   `URF_STAGE_0_DESIGN_FAILURE`.
10. If identity/reload, action validity, finite gradients, nonzero expected
    gradients, frozen-parameter gradient zero, or weighted gradient ratio
    `<= 100:1` fails:
    `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.
11. If action deltas are globally destructive or group deltas are not bounded:
    `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.
12. If exception count is nonzero:
    `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.
13. If all gates pass:
    `URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Any Stage 0 stop is a development result, not a closed-loop scientific kill.
No post-result threshold changes, task changes, route-label changes, residual
scale changes, proxy substitutions, action-validity reinterpretations, or
closed-method rescues are allowed.

## No Scientific Kill At Stage 0

Stage 0 failures are classified as data, no-headroom, design, or
implementation/optimization failures. They are not closed-loop scientific kills.

If Stage 0 passes, bounded validation search is the only allowed next stage. If
Stage 0 fails, archive the failure class and continue to the next method cycle
unless governance explicitly permits a measurement-invalid repair.

Only a pre-manifest serializer or launcher defect that produced no accepted
rows may receive one implementation repair under the identical protocol.

## Resource Contention Rule

Windows gaming / Efficiency Mode / resource-contention intervals must be
recorded separately. Timing, throughput, wall-clock efficiency, resource
utilization, and latency measurements overlapping or unresolved against those
intervals are not final paper evidence.

## Next Step

Implement and validate `tca_map/smolvla/urf_vla.py`,
`scripts/run_urf_vla_stage0.py`, and `tests/test_urf_vla.py` before launching
Stage 0.
