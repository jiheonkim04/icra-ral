# NICE-VLA Prototype Protocol

Date: 2026-07-15 KST

Proposal hash:
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.

Decision: `NICE_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_PENDING`.

## Purpose

Implement the smallest executable artifact that proves the VLA-Corrector
source mapping, legal latent/action pair construction, normalized-innovation
math, calibration estimator, checkpoint persistence, and Base-preserving
monitor interface before any full extraction or rollout.

Stage 0A is an implementation and data audit. It is not a trained NICE policy,
not a VLA-Corrector reproduction, and not closed-loop scientific evidence.

## Files

- `tca_map/smolvla/nice_vla.py`: pure math, tiny smoke models, conformal
  estimator, condition construction, manifests, and validators;
- `scripts/run_nice_vla_stage0.py`: official data/model source and interface
  smoke with atomic durable reporting;
- `tests/test_nice_vla.py`: unit and property tests;
- `reports/nice_vla/stage_0a_pair_manifest.json`;
- `reports/nice_vla/stage_0a_partial.json`;
- `reports/nice_vla/stage_0a_result.json`;
- `reports/nice_vla/stage_0a_result.md`;
- `reports/nice_vla/stage_0a_validation.json`;
- `reports/nice_vla/stage_0a_pid.txt`;
- `reports/nice_vla/stage_0a_heartbeat.json`;
- `reports/nice_vla/stage_0a_status.json`;
- `reports/nice_vla/stage_0a_stdout.log`;
- `reports/nice_vla/stage_0a_stderr.log`;
- `reports/nice_vla/stage_0a_exit_code.txt`.

The installed official SmolVLA/LeRobot environment and cloned VLA-Corrector
source are read but never edited.

## Pure-Math Requirements

Deterministic tests verify:

- 18D condition construction and fixed deadband semantics;
- first-frame previous-action behavior;
- diagonal normalized innovation and NLL against dense covariance;
- rank-8 Woodbury score and determinant against dense covariance;
- positive definiteness and variance floor/ceiling;
- finite gradients to covariance parameters and none to detached residuals;
- episode 90th-percentile cluster score;
- finite-sample conformal quantile with ties and boundary coverages;
- task-balanced episode calibration;
- pair-manifest duplicate, missing, extra, and split-overlap detection;
- monitor-disabled queue and 7D action exact passthrough;
- checkpoint save/load determinism;
- action-validity summaries split into translation, rotation, and gripper.

Small dense references may use `n<=32`. Production functions must reject a
dense covariance request for larger residuals.

## Frozen Stage 0A Manifest

Resolve exactly:

- `libero_10/task_1`, demos 0 and 1;
- `libero_goal/task_1`, demos 0 and 1.

For each demonstration, valid starts are `0..T-11`. Choose 32 indices by
`floor(linspace(0,T-11,32))`; duplicate sampled indices are forbidden. Each
pair uses current and `t+10` frozen visual tokens and current normalized action.

Expected rows: 128. Pair rows are keyed by suite, task, resolved HDF5 path,
demo, current frame, and future frame. The full manifest is written before
latent decode. Resume skips only complete keys already in valid partial JSON.

## Official Source Map

The runner verifies the VLA-Corrector git commit, Apache-2.0 license, and hashes
the source files containing:

- `SiglipDynamicsDataset` pair construction;
- `SiglipResidualMLP`;
- training cosine objective;
- `CircuitBreaker` cosine and median-MAD logic.

It writes a machine-readable map of file path, symbol, and SHA256. A missing or
changed source is `IMPLEMENTATION_FAILURE`; it is not automatically recloned or
silently replaced during Stage 0A.

## Official Data And Latent Smoke

For every manifest row:

1. read `agentview_rgb`, `eye_in_hand_rgb`, and normalized 7D action from the
   same HDF5 demonstration;
2. use the official campaign SmolVLA preprocessor and frozen checkpoint;
3. extract the same frozen visual token site selected for the source-derived
   dynamics input;
4. verify both current and future tensor are finite and shape-identical;
5. construct `Delta z` and persist hashes plus non-sensitive summaries;
6. never read reward, done, success, simulator state, reset identity, or a
   validation/confirmatory task.

The first row records a deterministic Base action twice. Monitor-disabled
wrapper execution must preserve queue contents and postprocessed action with
max absolute error exactly `0.0`.

## Tiny Model Smoke

On the first fixed batch of at most eight rows:

1. instantiate the frozen tiny mean topology;
2. record pretraining mean cosine loss, MSE, and gradient norms;
3. run at most 20 AdamW steps;
4. freeze mean parameters;
5. instantiate diagonal covariance and record NLL, scale summaries, gradients,
   and frozen-parameter gradients;
6. run at most 20 AdamW steps;
7. build a deterministic rank-8 basis from available residuals;
8. execute rank-8 forward/backward without extra fitting;
9. save and disk reload all states;
10. require reload output error `<=1e-6`.

This smoke does not select a covariance family, coverage, seed, or checkpoint
for later science.

## Algebra And Calibration Smoke

The runner creates fixed seeded small tensors and reports direct-versus-fast
errors for Mahalanobis score and log determinant. Maximum error is `1e-5`.
Minimum dense eigenvalue is at least `v_floor-1e-8`.

Fixed episode-score fixtures test nearest-rank 90th percentile and conformal
order statistics at `0.90`, `0.95`, and `0.975`, including ties. The exact
expected indices and thresholds are serialized.

## Result Schema

The atomic partial and final result include:

- proposal hash and source commit;
- attempt number and repair-consumed flag;
- PID, start/end time, and runtime artifact paths;
- planned/completed/resumed/new row counts;
- exception count;
- duplicate, missing, extra, and split-overlap counts;
- source file hashes and license status;
- resolved task/HDF5/language mapping;
- demonstration lengths and sampled frame indices;
- measured `(L,D)`, action width, and `k`;
- pair finite/variance/action validity summaries;
- condition/deadband statistics;
- mean/covariance losses and gradient norms;
- variance and clamping summaries;
- algebra errors and eigenvalue bound;
- conformal fixture results;
- Base queue/action identity error;
- checkpoint reload error;
- privileged/validation/confirmatory read counts;
- training, validation-search, rollout, and confirmatory-tuning booleans;
- every individual pass gate and final decision.

## Stage 0A Decision

`NICE_STAGE_0A_PASS_STAGE_0B_ALLOWED` requires every preregistered gate to pass
with 128/128 rows, zero exceptions, zero duplicate/missing/extra/overlap keys,
and a parsed final JSON.

`NICE_STAGE_0A_DATA_FAILURE` records invalid source mapping, insufficient fixed
rows, collapsed legal action/latent signal, or data corruption.

`NICE_STAGE_0A_IMPLEMENTATION_FAILURE` records source mismatch, preprocessing,
shape, algebra, gradient, serialization, passthrough, or runtime defects. One
mechanical repair is allowed only under the preregistration.

`NICE_STAGE_0A_DESIGN_FAILURE` is permitted only when implementation and data
are valid but the legal deployment-observable latent/action signal is absent.

No Stage 0A failure is reported as closed-loop evidence. No failed result may
be rescued by changing data, math, thresholds, or gates.

## Runtime Procedure

Before launch, inspect campaign state and the newest NICE PID, heartbeat,
status, partial, result, log, and exit-code files. Verify worker liveness and
parse partial JSON. Check planned/completed rows, exceptions, duplicates, and
manifest keys.

If a worker is alive, monitor it only. If completed, adjudicate it without
rerun. If dead with valid partial, run only missing keys. A stale heartbeat
alone is never proof of death.

The detached WSL wrapper writes PID immediately, heartbeat/status atomically,
separate stdout/stderr, and exit code even on failure. The Python runner updates
partial JSON atomically after each completed pair.

## Verification Commands

Windows tests:

`python -m pytest tests/test_nice_vla.py tests/test_current_research_governance.py -q`

Official WSL Stage 0A:

`/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_nice_vla_stage0.py --mode stage0a`

The command is launched only after the runtime audit and implementation commit.

## Transition

After a valid Stage 0A pass, adjudicate, validate, commit, and push before
Stage 0B. After any valid failure, close NICE without rescue and continue to
the next method cycle automatically.
