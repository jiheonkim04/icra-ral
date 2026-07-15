# SPARC-VLA Prototype Protocol

Date: 2026-07-15 KST

Decision: `SPARC_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_PENDING`

## Purpose

Implement the smallest executable artifact that can prove SPARC's closed-form
operator and SmolVLA post-residual integration are real, Base-preserving, and
bounded before collecting `84` labeled discovery rollouts.

## Files

- `tca_map/smolvla/sparc_vla.py`: pure math, manifests, hashing, and gate state;
- `scripts/run_sparc_vla_stage0.py`: official model hook smoke and report;
- `tests/test_sparc_vla.py`: unit and property tests;
- `reports/sparc_vla/stage_0a_result.json`;
- `reports/sparc_vla/stage_0a_result.md`;
- `reports/sparc_vla/stage_0a_validation.json`.

The installed LeRobot package is read but never edited.

## Pure-Math Requirements

Tests use deterministic synthetic matrices to verify:

- equal-episode weights do not change when one episode is duplicated in
  length;
- conceptor symmetry and expected eigenvalue transform;
- alpha monotonicity of quota;
- AND-NOT finite output and eigenvalue tolerance;
- covariance aggregation differs from mean-conceptor aggregation on a
  noncommuting example;
- global and ten-step operators apply to `[B,50,720]` token tensors;
- beta zero is exact identity;
- canonical tensor hashes are stable;
- duplicate and missing manifest keys are detected;
- action-safety summaries separate translation, rotation, and gripper.

## Official Hook Smoke

Load frozen SmolVLA with the official campaign loader and one legal discovery
observation already available from local LIBERO demonstrations.

Run:

1. direct deterministic Base;
2. capture-only post-residual adapter at site `11`;
3. unconfigured serialized/reloaded adapter;
4. configured identity operator at beta `0.1`;
5. configured deterministic synthetic PSD operator close to identity at beta
   `0.1`;
6. removed hook followed by direct Base.

The identity operator is expected to match Base exactly even with nonzero beta.
The synthetic PSD operator must produce a finite nonzero internal delta; action
delta may be zero only if the output projection is causally insensitive on the
single row, in which case a second preregistered discovery observation is used.
If both rows have zero action consequence, classify `DESIGN_FAILURE` only after
hook correctness is independently proven; wiring uncertainty remains
`IMPLEMENTATION_FAILURE`.

## Adapter Semantics

The pre-hook is registered on action-expert layer `l+1` input layernorm. It
captures the incoming full post-residual tensor and, only under inference
`no_grad`, writes `H M^T` in-place so both attention and residual paths consume
the same tensor.

The adapter rejects:

- gradient-enabled forward;
- unexpected batch/token/width shape;
- more or fewer than ten denoising captures;
- nonfinite operator or tensor;
- operator count other than one global or ten per-step matrices;
- configured use before explicit state load;
- nested/double hook registration.

## Stage 0A Decision

`SPARC_STAGE_0A_PASS_DISCOVERY_COLLECTION_ALLOWED` requires every pure-math and
official hook criterion to pass with zero exceptions and a parsed result.

`SPARC_STAGE_0A_IMPLEMENTATION_FAILURE` preserves the failed artifact and
allows one bounded implementation repair without changing any scientific
choice.

`SPARC_STAGE_0A_DESIGN_FAILURE_NONACTING` is allowed only when hook fidelity,
operator application, tensor changes, and output path are proven correct on
both preregistered observations but the action remains exactly unchanged.

No data, geometry, validation, confirmatory, or rollout claim is made by Stage
0A.

## Verification Commands

Windows unit tests:

`python -m pytest tests/test_sparc_vla.py tests/test_current_research_governance.py -q`

Official WSL smoke:

`/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_sparc_vla_stage0.py --mode hook-smoke`

Before the WSL command, inspect current campaign state and newest durable
runtime files, verify no existing Linux research worker, parse any partial
JSON, and check completed/planned/exception/duplicate counts.

## Transition

After Stage 0A pass, commit and push the implementation and result. Then launch
the frozen Stage 0B discovery manifest detached with durable PID, heartbeat,
partial, result, logs, and exit code. Do not stop after documentation.
