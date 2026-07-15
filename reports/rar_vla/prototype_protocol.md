# RAR-VLA Prototype Protocol

Date: 2026-07-15 KST

Method: `RAR-VLA`

Proposal hash: `723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56`

Protocol decision: `RAR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

## Purpose

Run RAR as an honest fixed-protocol prototype. The immediate next step is a
development-only Stage 0 audit. No validation search, final residual training,
Stage A manifest, rollout, or confirmatory tuning is allowed before Stage 0
passes.

## Policy Identities

Future first comparison identities are frozen at the design level:

- `frozen_smolvla`
- `ar_vla_reanchored_expert_proxy`
- `rar_full`
- `rar_no_reanchor_memory_ablation`
- `ema_action_history_baseline`

The closest-prior proxy label is:

`faithful_transparent_local_proxy_not_official_ar_vla_reproduction`

## Stage 0 Command

Planned command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_rar_vla_development.py --mode audit
```

Stage 0 writes:

- `reports/rar_vla/development_audit.json`
- `reports/rar_vla/development_audit.md`
- `reports/rar_vla/source_gate_manifest.json`
- `reports/rar_vla/history_feature_manifest.json`
- `reports/rar_vla/split_manifest.json`

Stage 0 must not launch closed-loop rollout. Lightweight development-only
probes are allowed only to assess source legality, action-history headroom,
residual observability, gradients, and Base passthrough.

## Stage 0 Inputs

Allowed:

- official development train/validation records;
- official frozen SmolVLA predictions and Base actions on development
  identities;
- current deployment RGB/language/proprioception fields;
- previous Base or emitted actions from the same trajectory;
- target actions only as development labels.

Forbidden:

- confirmatory rollout identities for tuning;
- future action segments or CALA latent labels at inference;
- simulator object pose or target placement at inference;
- reset identity, reward, success, future observation, or held-out outcomes at
  inference;
- any prior method rescue labels or closed-loop outcomes as tuning signals.

## Stage 0 Decision Labels

Allowed pass:

- `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`

Allowed stops:

- `DATA_OR_SUPERVISION_FAILURE`
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`
- `DESIGN_FAILURE`
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Stage 0 stop labels are pre-rollout development outcomes, not closed-loop
scientific kills.

## Validation Search Protocol

Run only after Stage 0 pass.

The search may try at most the six named configurations in
`reports/rar_vla/preregistration.md`.

Save all tried configurations, selected config, selected validation score and
components, checkpoint paths and checksums, source/history manifests,
action-delta metrics, residual/gate activation metrics, and selected config
hash.

## Training Artifacts

Every trained identity must have:

- config JSON;
- training seed;
- split manifest;
- history feature manifest;
- source gate manifest;
- checkpoint path;
- checksum;
- base checkpoint identity;
- validation metrics;
- action-delta metrics;
- residual/gate activation metrics;
- disk-reload verification.

Stage A cannot begin until `rar_full`, `ar_vla_reanchored_expert_proxy`, and
`rar_no_reanchor_memory_ablation` have disk-reloadable identities or are
explicitly nontrainable by design.

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
- residual predictability;
- residual/gate activation;
- inter-chunk and intra-chunk diagnostics;
- translation, rotation, and gripper deltas from Base;
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

RAR cannot become a paper candidate unless `rar_full` beats Base, the AR-style
proxy, the no-reanchor-memory ablation, and the EMA action-history baseline
under the matched protocol.

If the EMA baseline wins, RAR is explained by simple causal smoothing/history.
If the no-reanchor-memory ablation wins, re-anchoring is not useful. If the AR
proxy wins, RAR is not a useful local extension of the closest prior.

## Implementation Boundaries

Implementation may add only the minimal reusable development runner and modules
needed to:

- build source/history/split manifests;
- construct legal causal history features on development identities;
- train or evaluate lightweight residual observability probes;
- instantiate identity-preserving residual/gate wrappers;
- run the six-config validation search if Stage 0 passes;
- preflight disk-reloaded policy identities before closed-loop rollout.

Do not redesign RAR during implementation. Do not add baselines before the
frozen five-policy comparison. Do not use confirmatory identities for tuning.

## Current Next Action

Implement the Stage 0 development audit only. If Stage 0 stops, record the
correct pre-rollout failure label and pivot; do not launch validation search,
training, manifest freeze, or rollout.
