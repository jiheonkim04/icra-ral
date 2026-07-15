# FAMR-VLA Executable Prototype Protocol

Date: 2026-07-15 KST

Decision: `FAMR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_PENDING`

## Authorized Implementation Scope

Implement:

- pure FAMR grouping, scaling, response, solver, fidelity, validity, and
  manifest helpers in `tca_map/smolvla/famr_vla.py`;
- the real audit/training/search runner in `scripts/run_famr_vla_stage0.py`;
- focused pure and contract tests in `tests/test_famr_vla.py`;
- Stage 0 result files under `reports/famr_vla/`;
- durable runtime files under `runs/famr_vla/`.

Do not modify IARC artifacts, processors, action semantics, existing
checkpoints, prior results, or governance thresholds.

## Stage 0A Command

Run detached in WSL from repository root:

```bash
/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python \
  scripts/run_famr_vla_stage0.py \
  --mode audit \
  --checkpoint /mnt/c/assets/checkpoints/smolvla_libero \
  --libero-data-root /mnt/c/assets/data/libero \
  --stable-artifact reports/official_smolvla_stable_prediction_artifact.json \
  --run-root runs/famr_vla/stage0a \
  --report-root reports/famr_vla
```

Stage 0A may run the fixed 20-step micro fit but not the full endpoint, response
search, validation rollout, or confirmatory decode.

## Runtime Preflight

Before every command:

1. read both campaign-state JSON files;
2. locate the newest PID, heartbeat/status, partial, result, log, and exit files;
3. inspect Linux PID command lines and parent relationships;
4. parse partial JSON and read completed/planned/exception counts;
5. never duplicate a live or completed worker;
6. resume only missing manifest keys after verified worker death;
7. write resource-contention overlap metadata.

Preflight also verifies disk free space, CUDA availability, checkpoint files,
raw target HDF5 files, LIBERO BDDL files, Python environment, and campaign
stage/proposal hash.

## Pure Helper Contract

Tests must cover:

- coarse and fine parameter assignment is exhaustive and disjoint;
- coefficient bounds and deterministic projected solver;
- normalized Huber and retention objectives;
- effective LoRA `B` scaling identity at `c=0`, `c=1`, and intermediate `c`;
- response construction and direct fidelity metrics;
- practical-equivalence threshold;
- action validity absolute and Base-relative checks;
- task/reset manifest uniqueness and exact resume-key subtraction;
- duplicate, missing, extra, exception, and malformed-partial rejection;
- resource-overlap eligibility.

Pure tests use synthetic tensors only and do not load a model or simulator.

## Stage 0A Required Files

- `reports/famr_vla/stage_0a_result.json`
- `reports/famr_vla/stage_0a_result.md`
- `reports/famr_vla/stage_0a_adjudication.md`
- `reports/famr_vla/task_provenance_manifest.json`
- `reports/famr_vla/data_semantics_audit.json`
- `reports/famr_vla/parameter_group_manifest.json`
- `reports/famr_vla/checkpoint_manifest.json`
- `reports/famr_vla/implementation_blocker.json` when applicable
- `runs/famr_vla/stage0a/worker_pid.txt`
- `runs/famr_vla/stage0a/child_pid.txt`
- `runs/famr_vla/stage0a/heartbeat.json`
- `runs/famr_vla/stage0a/status.json`
- `runs/famr_vla/stage0a/partial_result.json`
- `runs/famr_vla/stage0a/stdout.log`
- `runs/famr_vla/stage0a/stderr.log`
- `runs/famr_vla/stage0a/exit_code.txt`
- `runs/famr_vla/stage0a/exact_resume_command.txt`

## Stage 0A Result Schema

The final JSON must include:

- method, proposal hash, git commit, command, timestamps;
- experiment boundaries and resource interval overlap;
- target task provenance and exact intersection count;
- HDF5 counts, split counts, duplicate counts, shapes, ranges, and semantics;
- expert replay/source success evidence;
- adapter config and trainable parameter manifest;
- zero-effect identity metrics;
- micro-fit before/after losses and gradient norms;
- checkpoint save/reload hashes and action errors;
- coarse/fine group assignment coverage;
- coefficient endpoint and effective-weight scaling tests;
- peak CUDA allocation;
- confirmatory observations/actions, both zero;
- exception count;
- final decision and exact next command.

## Stage 0A Decisions

`FAMR_STAGE_0A_PASS_ENDPOINT_TRAINING_ALLOWED` only when every frozen Stage 0A
gate passes.

Otherwise use one precise label:

- `FAMR_FATAL_PREIMPLEMENTATION`;
- `FAMR_IMPLEMENTATION_OR_DATA_FAILURE`;
- `FAMR_LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT` after the one allowed
  capacity check;
- `FAMR_UNDERPOWERED_ONE_CHECK_ALLOWED` only for the frozen rank-8 condition.

No closed-loop scientific kill is possible in Stage 0A.

## Later Commands

Only after Stage 0A pass:

```bash
python scripts/run_famr_vla_stage0.py --mode train-endpoint ...
python scripts/run_famr_vla_stage0.py --mode headroom ...
python scripts/run_famr_vla_stage0.py --mode response-search ...
```

Each mode receives and verifies the prior result hash. The implementation must
refuse out-of-order execution.

## Automatic Continuation

After each completed stage:

1. validate partial/final JSON and manifests;
2. adjudicate under the preregistered gate and false-negative safeguard;
3. update campaign state and reports;
4. commit and push the milestone;
5. continue to the next authorized stage without routine user approval;
6. after a valid stop, preserve the result and enter the next method cycle.
