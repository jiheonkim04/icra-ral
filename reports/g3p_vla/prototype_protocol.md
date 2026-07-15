# G3P-VLA Prototype Protocol

Date: 2026-07-15 KST

Method: `G3P-VLA`

Proposal hash: `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`

Protocol decision: `G3P_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

## Purpose

Run G3P as an honest fixed-protocol prototype. The immediate next step is a development-only Stage 0 audit. No validation search, final adapter training, Stage A manifest, rollout, or confirmatory tuning is allowed before Stage 0 passes.

## Policy Identities

Future first comparison identities are frozen at the design level:

- `frozen_smolvla`
- `g3p_3d_point_proxy`
- `g3p_full`
- `g3p_no_3d_no_injection_ablation`
- `simple_2d_phase_or_nearest_object_heuristic`

The closest-prior proxy label is:

`faithful_transparent_local_proxy_not_official_direct_3d_point_injection_reproduction`

## Stage 0 Command

Planned command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_g3p_vla_development.py --mode audit
```

Stage 0 writes:

- `reports/g3p_vla/development_audit.json`
- `reports/g3p_vla/development_audit.md`
- `reports/g3p_vla/source_gate_manifest.json`
- `reports/g3p_vla/point_label_manifest.json`
- `reports/g3p_vla/split_manifest.json`

Stage 0 must not launch closed-loop rollout. Lightweight development-only probes are allowed only to assess point observability, source legality, gradients, and Base passthrough.

## Stage 0 Inputs

Allowed:

- official development train/validation records;
- official frozen SmolVLA predictions and Base actions on development identities;
- deployment RGB/language/proprioception fields;
- oracle geometry only for discovery/validation label construction and diagnostics.

Forbidden:

- confirmatory rollout identities for tuning;
- simulator object pose or target placement at inference;
- reset identity, reward, success, future observation, or held-out outcomes at inference;
- any prior method rescue labels or closed-loop outcomes as tuning signals.

## Stage 0 Decision Labels

Allowed pass:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`

Allowed stops:

- `DATA_OR_SUPERVISION_FAILURE`
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Stage 0 stop labels are pre-rollout development outcomes, not closed-loop scientific kills.

## Validation Search Protocol

Run only after Stage 0 pass.

The search may try at most the six named configurations in `reports/g3p_vla/preregistration.md`.

Save:

- all tried configurations;
- selected config;
- selected validation score and components;
- checkpoint paths and checksums;
- source gate manifest;
- point-label manifest;
- action-delta and clean-retention summaries;
- selected config hash.

## Training Artifacts

Every trained identity must have:

- config JSON;
- training seed;
- split manifest;
- point-label manifest;
- source gate manifest;
- checkpoint path;
- checksum;
- base checkpoint identity;
- validation metrics;
- action-delta metrics;
- point activation metrics;
- disk-reload verification.

Stage A cannot begin until `g3p_full`, `g3p_3d_point_proxy`, and `g3p_no_3d_no_injection_ablation` have disk-reloadable identities or are explicitly nontrainable by design.

## Closed-Loop Manifest

Stage A and Stage B must use matched paired manifests:

- identical task keys across policies;
- identical reset identities across policies;
- no overlap with validation identities;
- no duplicate `(policy, task, reset)` keys;
- official LIBERO success condition;
- no post-result task or reset selection.

The manifest is frozen before each stage begins.

## Metrics

Primary:

- official closed-loop task success;
- task-balanced success;
- paired full-minus-baseline success deltas.

Secondary:

- paired wins/losses/ties;
- paired bootstrap confidence interval;
- relative failure-rate reduction;
- per-task success;
- point predictability;
- point activation and confidence;
- source-gate status;
- translation, rotation, and gripper deltas from Base;
- residual norm and gate value;
- action validity;
- clean retention;
- latency;
- VRAM;
- training time.

## Resume Policy

For long-running WSL training or rollout:

- run detached;
- save Linux PID;
- save heartbeat/status JSON;
- save stdout and stderr logs;
- save partial result JSON;
- save exact resume command;
- resume only missing `(policy, task, reset)` keys after interruption.

## Scientific Decisions

G3P cannot become a paper candidate unless `g3p_full` beats Base, the closest-prior 3D-point proxy, the no-3D/no-injection ablation, and the simple 2D/phase/nearest-object heuristic under the matched protocol.

If the closest-prior proxy wins, G3P is not a useful local extension of the closest prior.

If the no-3D/no-injection ablation wins, the 3D point injection component is not useful.

If the simple heuristic wins, the result is explained by a cheaper non-3D or task-phase mechanism.

## Implementation Boundaries

Implementation may add only the minimal reusable development runner and modules needed to:

- build source and split manifests;
- construct legal point labels or source proxies on development identities;
- train or evaluate lightweight point observability probes;
- instantiate identity-preserving point adapters;
- run the six-config validation search if Stage 0 passes;
- preflight disk-reloaded policy identities before closed-loop rollout.

Do not redesign G3P during implementation. Do not add baselines before the frozen five-policy comparison. Do not use confirmatory identities for tuning.

## Current Next Action

Implement the Stage 0 development audit only. If Stage 0 stops, record the correct pre-rollout failure label and pivot; do not launch validation search, training, manifest freeze, or rollout.
