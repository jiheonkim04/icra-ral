# EAC-VLA Prototype Protocol

Date: 2026-07-15 KST

Method: `EAC-VLA`

Protocol decision: `EAC_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

## Purpose

Run EAC as an honest fixed-protocol prototype. The first step is a development-only Stage 0 audit. No validation search, Stage A manifest, training, rollout, or confirmatory tuning is allowed before Stage 0 passes.

## Policy Identities

Future first comparison identities are frozen at the design level:

- `frozen_smolvla_fixed_queue`
- `aac_entropy_proxy`
- `eac_full`
- `eac_no_calibration_no_hysteresis_ablation`
- `fixed_short_replan_baseline`

The closest-prior proxy label is:

`faithful_transparent_local_proxy_not_official_aac_reproduction`

## Stage 0 Inputs

Allowed:

- official frozen SmolVLA policy and postprocessors;
- official development train/validation prediction records;
- official LIBERO task metadata for non-confirmatory development identities;
- current observation, action queue length, previous executed action, and frozen policy action chunks;
- repeated legal policy calls from the same observation if used to compute dispersion.

Forbidden:

- confirmatory test identities for tuning;
- simulator object-state, reward, success, future observation, target action, reset identity, or held-out outcomes at inference;
- any PESA query label or spectral adapter rescue.

## Stage 0 Outputs

Expected artifacts:

- `reports/eac_vla/stage_0_audit.json`
- `reports/eac_vla/stage_0_audit.md`
- optional `reports/eac_vla/queue_surface_manifest.json`
- optional `reports/eac_vla/dispersion_manifest.json`
- optional `reports/eac_vla/split_manifest.json`

Required fields:

- final decision;
- split counts and overlap counts;
- chunk shape and action dimension;
- queue observability/control result;
- action-value passthrough max absolute error;
- dispersion/entropy statistic summary;
- commitment-length distribution;
- latency/policy-call estimates;
- action validity;
- hard stop reasons;
- whether training happened;
- whether closed-loop rollout happened;
- whether confirmatory identities were used.

## Stage 0 Decision Labels

Allowed pass:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`

Allowed stops:

- `DESIGN_FAILURE`
- `DATA_OR_SUPERVISION_FAILURE`
- `IMPLEMENTATION_FAILURE`
- `NO_HEADROOM`

Stage 0 stop labels are pre-rollout development outcomes, not closed-loop scientific kills.

## Validation Search Protocol

Run only after Stage 0 pass.

Maximum search:

- six total configurations;
- one risk threshold factor with at most three values;
- one commitment/hysteresis factor with at most two values;
- no unbounded grid;
- no confirmatory-test tuning.

Save:

- all tried configurations;
- all negative results;
- selected configuration;
- selected score and component metrics;
- selected config hash or canonical JSON payload.

## Stage A Protocol

Stage A can launch only after:

- selected configuration is frozen;
- policy identities are implemented and preflighted;
- action-value passthrough is verified;
- matched task/reset manifest is frozen;
- no hidden test tuning is verified.

Planned comparison:

- approximately ten paired episodes per policy;
- identical task/reset identities across all five policies;
- task-balanced allocation;
- official LIBERO success condition;
- no post-hoc task/reset selection.

Stage A reports:

- success counts;
- task-balanced success;
- paired deltas;
- commitment statistics;
- action chunks generated;
- policy calls per step;
- latency/VRAM;
- action validity;
- exceptions.

## Stage B Protocol

Stage B runs if Stage A is positive or unresolved and not validly killed.

Stage B uses at least forty paired episodes per key policy and reports:

- paired wins/losses/ties;
- bootstrap confidence interval;
- failure-rate reduction;
- per-task breakdown;
- mechanism activation;
- clean retention;
- compute and latency.

One expansion to eighty paired cases per key policy is allowed only if genuinely unresolved.

## Implementation Boundaries

Implementation may add only the minimal reusable queue-scheduler integration needed to:

- observe or control queue length;
- execute selected chunk prefixes;
- preserve official preprocessing/postprocessing;
- record action-value equality;
- checkpoint partial rows;
- resume only missing evaluation keys.

Do not redesign EAC during implementation. Do not change action values. Do not add extra baselines before the first five-policy comparison.

## Current Next Action

Implement the Stage 0 development audit only. If Stage 0 stops, record the correct pre-rollout failure label and pivot; do not launch validation search or rollout.
