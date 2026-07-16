# S2C-VLA Prototype Protocol

Date: 2026-07-16 KST

Decision: `S2C_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`

Proposal: `reports/s2c_vla/researcher_proposal.md`

Proposal SHA-256:
`399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3`

Mathematical audit: `reports/s2c_vla/mathematical_mechanism_audit.md`

Preregistration: `reports/s2c_vla/preregistration.md`

No S2C implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this protocol.

## Implementation Files

Implement:

- helper module: `tca_map/smolvla/s2c_vla.py`;
- Stage 0 runner: `scripts/run_s2c_vla_stage0.py`;
- focused tests: `tests/test_s2c_vla.py`.

The helper module must contain only deterministic utilities and lightweight
prototype logic for Stage 0. It must not import simulator environments or read
reward/success/done fields.

## Required Helper API

The helper module must provide:

- protocol constants for `H=50`, `stride=10`, `K=10`, `D=7`;
- action group constants and caps;
- proposal-hash constant;
- bridge target solver for the frozen quadratic objective;
- groupwise clipping;
- effective mask construction with exact identity initialization;
- S2C action application with future-zone exact Base passthrough;
- boundary, derivative, high-frequency, and action-delta metrics;
- gripper event detection and event-destruction checks;
- manifest key serialization and duplicate/missing/extra checks;
- partial/result JSON validation;
- decision taxonomy for Stage 0 stop classes.

## Stage 0 Runner Duties

The runner must support:

- `--serializer-preflight`;
- full Stage 0 audit;
- missing-key-only resume if partial rows already exist;
- PID, heartbeat, status, partial, result, result markdown, adjudication, and
  exit-code artifacts;
- no duplicate completed row generation;
- no confirmatory-test identity reads.

Before doing expensive work, the runner must inspect existing S2C artifacts:

1. newest PID;
2. heartbeat;
3. status;
4. partial result;
5. final result;
6. logs if present;
7. exit code.

If a worker is alive, the runner must refuse duplicate launch. If a completed
result exists, it must refuse rerun and report the existing result. If a valid
partial exists after a dead worker, it may resume only missing keys.

## Stage 0 Policies

Stage 0 rows must include:

1. `smolvla_base`;
2. `chunkflow_overlap_proxy`;
3. `s2c_full`;
4. `s2c_no_learned_overlap_mask_ablation`;
5. `standard_lora`;
6. optional `seam_previous_tail_diagnostic`;
7. optional `no_boundary_loss_diagnostic`.

Only the first five are part of the first serious comparison. Optional
diagnostics cannot replace ChunkFlow as policy 2.

## Data Sources

Allowed:

- existing verified SmolVLA Base action chunks and feature caches;
- LIBERO demonstration action chunks for development partitions;
- task and phase metadata on discovery/validation partitions;
- previous committed Base or S2C tail constructed from adjacent development
  rows.

Forbidden:

- reward;
- success;
- done;
- simulator rollout results;
- object poses at inference;
- future observations;
- expert future tail at inference;
- confirmatory identities.

## Required Artifacts

The runner must write:

- `reports/s2c_vla/stage_0_preflight.json`;
- `reports/s2c_vla/stage_0_manifest.json`;
- `reports/s2c_vla/stage_0_partial.json`;
- `reports/s2c_vla/stage_0_result.json`;
- `reports/s2c_vla/stage_0_result.md`;
- `reports/s2c_vla/stage_0_adjudication.md`;
- `reports/s2c_vla/stage_0_status.json`;
- `reports/s2c_vla/stage_0_heartbeat.json`;
- `reports/s2c_vla/stage_0_pid.txt`;
- `reports/s2c_vla/stage_0_exit_code.txt`;
- `reports/s2c_vla/stage_0_action_semantics.json`;
- `reports/s2c_vla/stage_0_official_prior_asset_check.json`;
- `reports/s2c_vla/stage_0_serializer_preflight.json`.

## Serializer Preflight

`--serializer-preflight` must:

- canonicalize one representative row key;
- round-trip tensors through JSON-safe serialization;
- produce a deterministic SHA-256 fixture hash;
- persist `reports/s2c_vla/stage_0_serializer_preflight.json`;
- pass before full Stage 0 is launch-eligible.

## Acceptance Before Stage 0 Launch

Before Stage 0 launch, all must pass:

- helper module compiles;
- runner compiles;
- focused tests pass;
- serializer preflight passes;
- governance tests pass;
- governance checker passes;
- no live or completed S2C worker exists.

This prototype protocol authorizes implementation and preflight validation
next. It does not authorize Stage 0 launch until implementation validation and
worker-safety checks are complete.
