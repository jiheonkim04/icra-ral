# CCIF-VLA Prototype Protocol

Date: 2026-07-16 KST

Decision: `CCIF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

Proposal: `reports/ccif_vla/researcher_proposal.md`

Proposal SHA-256:
`2AFC40F050FD7F0D28507344358CBCB70BF27CC901C57474A501D3EB87E7FAA1`

Mathematical audit:
`reports/ccif_vla/mathematical_mechanism_audit.md`

Preregistration: `reports/ccif_vla/preregistration.md`

## Runner Contract

Implement Stage 0 with:

- helper module: `tca_map/smolvla/ccif_vla.py`;
- runner: `scripts/run_ccif_vla_stage0.py`;
- tests: `tests/test_ccif_vla.py`;
- serializer/preflight artifact:
  `reports/ccif_vla/stage_0_serializer_preflight.json`;
- primary result artifact: `reports/ccif_vla/stage_0_result.json`.

The runner must support:

- foreground dry run for serializer and unit-test validation;
- detached durable execution when Stage 0 is launched;
- missing-key-only resume if a valid partial exists after worker death;
- no duplicate completed `(model_or_probe, task, demo, frame)` rows;
- no rerun of completed rows after a valid partial.

## Fixed Stage 0 Artifact Paths

Stage 0 artifacts:

- `reports/ccif_vla/stage_0_manifest.json`;
- `reports/ccif_vla/stage_0_partial.json`;
- `reports/ccif_vla/stage_0_result.json`;
- `reports/ccif_vla/stage_0_result.md`;
- `reports/ccif_vla/stage_0_validation.json`;
- `reports/ccif_vla/stage_0_preflight.json`;
- `reports/ccif_vla/stage_0_status.json`;
- `reports/ccif_vla/stage_0_heartbeat.json`;
- `reports/ccif_vla/stage_0_pid.txt`;
- `reports/ccif_vla/stage_0_stdout.log`;
- `reports/ccif_vla/stage_0_stderr.log`;
- `reports/ccif_vla/stage_0_exit_code.txt`;
- `reports/ccif_vla/stage_0_action_semantics.json`;
- `reports/ccif_vla/stage_0_official_prior_asset_check.json`.

## Fixed Development Data

Use only:

- `libero_spatial/task_3`, demos `0..9`;
- `libero_object/task_3`, demos `0..9`;
- `libero_goal/task_5`, demos `0..9`;
- `libero_10/task_5`, demos `0..9`.

Discovery/training demos: `0..7`.

Validation demos: `8..9`.

Confirmatory identities: none read.

No simulator rollout, reward, success, done flag, object pose, future
observation, or confirmatory action access is allowed in Stage 0.

## Fixed Policy/Probe Rows

Stage 0 is an offline development audit. It must include rows for:

1. `smolvla_base`;
2. `coarse_to_control_continuous_proxy`;
3. `ccif_full`;
4. `ccif_no_coarse_intent_ablation`;
5. `standard_lora_proxy`;
6. `task_phase_mean_intent`;
7. `endpoint_only_intent`.

`standard_lora_proxy` is a Stage 0 offline proxy only. It may use the
repository's existing standard LoRA checkpoint statistics or a transparent
same-budget residual probe, but it is not a closed-loop LoRA result.

First serious closed-loop comparison still has exactly five policies:
`smolvla_base`, `coarse_to_control_continuous_proxy`, `ccif_full`,
`ccif_no_coarse_intent_ablation`, and `standard_lora`.

## Fixed Mechanism Constants

- `H = 50`;
- `D = 7`;
- `m = 31`;
- waypoint indices `[9, 19, 34, 49]`;
- `eps_c = 1e-6`;
- initialized residual and gate are zero effect;
- initialized and disk-reloaded CCIF must reproduce Base within `1e-6`;
- no deterministic-action KL.

The invalid duplicate-terminal `m = 37` intent draft is forbidden.

## Required Implementation Checks

Before any detached Stage 0 launch:

1. `py_compile` passes for helper and runner.
2. Focused unit tests in `tests/test_ccif_vla.py` pass.
3. Serializer preflight writes
   `reports/ccif_vla/stage_0_serializer_preflight.json`.
4. The preflight validates JSON serialization for NumPy arrays, tensors,
   booleans, paths, and nested metric dictionaries.
5. The official action semantics artifact is created or updated with the same
   action dimension and postprocessing assumptions used by recent official
   SmolVLA Stage 0 runners.

## Stage 0 Required Metrics

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
- `training_happened = false` unless a later Stage 0 smoke explicitly trains a
  tiny audit-only module and reports it as development-only;
- `validation_search_happened = false`;
- `intent_dimension = 31`;
- `waypoint_indices = [9, 19, 34, 49]`;
- `collapsed_intent_component_count`;
- `task_phase_mean_intent_huber`;
- `endpoint_only_intent_huber`;
- `deployment_intent_probe_huber`;
- `intent_probe_relative_improvement`;
- `base_to_expert_huber`;
- `coarse_to_control_proxy_huber`;
- `ccif_full_huber`;
- `ccif_no_intent_ablation_huber`;
- `ccif_beats_prior_relative`;
- `ccif_beats_prior_absolute_huber`;
- `ccif_beats_ablation_relative`;
- `ccif_beats_ablation_absolute_huber`;
- `action_validity_ok`;
- `identity_max_abs_error`;
- `checkpoint_reload_ok`;
- `finite_objectives_and_gradients`;
- `ccif_gradient_nonzero`;
- `frozen_parameter_gradient_count`;
- `weighted_gradient_norm_ratio_max`;
- `residual_activation_fraction`;
- `timing_throughput_resource_evidence_eligible_for_paper = false`.

## Frozen Decision Rule

Decision order:

1. If source artifacts, proposal hash, serialization, or split integrity fails:
   `CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.
2. If labels collapse or discovery/validation coverage is insufficient:
   `CCIF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.
3. If deployment-input intent probe does not beat task/phase mean by `5%`
   relative Huber or `0.005` absolute Huber:
   `CCIF_STAGE_0_DESIGN_FAILURE`.
4. If endpoint-only intent explains CCIF within the frozen margin:
   `CCIF_STAGE_0_DESIGN_FAILURE`.
5. If the prior proxy leaves no CCIF residual headroom of at least `5%`
   relative Huber or `0.005` absolute Huber:
   `CCIF_STAGE_0_NO_USABLE_HEADROOM`.
6. If CCIF does not beat `ccif_no_coarse_intent_ablation` by at least `5%`
   relative Huber or `0.005` absolute Huber:
   `CCIF_STAGE_0_NO_USABLE_HEADROOM`.
7. If identity/reload, action validity, finite gradients, nonzero expected
   gradients, frozen-parameter gradient zero, or weighted gradient ratio
   `<= 100:1` fails:
   `CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.
8. If all gates pass:
   `CCIF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Any Stage 0 stop is a development result, not a closed-loop scientific kill.
No post-result threshold changes, task changes, proxy substitutions, or TSC
rescue are allowed.

## Worker Safety

Before launching a detached Stage 0 worker, inspect existing PID, status,
heartbeat, partial, result, logs, and exit-code files.

- If a worker is alive, monitor only.
- If completed, adjudicate the existing result.
- If dead with valid partial, resume only missing keys.
- If heartbeat is stale, verify PID and logs before deciding it is dead.

No duplicate rows may be accepted.

## Next Step

Implement and validate `tca_map/smolvla/ccif_vla.py`,
`scripts/run_ccif_vla_stage0.py`, and `tests/test_ccif_vla.py` before launching
Stage 0.
