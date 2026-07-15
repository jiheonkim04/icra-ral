# CALA-VLA Prototype Protocol

Date: 2026-07-15 KST

Method: `CALA-VLA`

Proposal hash: `5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76`

Protocol decision: `CALA_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

## Purpose

Run CALA as an honest fixed-protocol prototype. The immediate next step is a
development-only Stage 0 audit. No validation search, final adapter training,
Stage A manifest, rollout, or confirmatory tuning is allowed before Stage 0
passes.

## Policy Identities

Future first comparison identities are frozen at the design level:

- `frozen_smolvla`
- `cac_vla_latent_action_proxy`
- `cala_full`
- `cala_no_context_gate_ablation`
- `task_mean_latent_action_baseline`

The closest-prior proxy label is:

`faithful_transparent_local_proxy_not_official_cac_vla_reproduction`

## Stage 0 Command

Planned command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_cala_vla_development.py --mode audit
```

Stage 0 writes:

- `reports/cala_vla/development_audit.json`
- `reports/cala_vla/development_audit.md`
- `reports/cala_vla/source_gate_manifest.json`
- `reports/cala_vla/latent_label_manifest.json`
- `reports/cala_vla/split_manifest.json`

Stage 0 must not launch closed-loop rollout. Lightweight development-only
probes are allowed only to assess latent observability, source legality,
gradients, and Base passthrough.

## Stage 0 Inputs

Allowed:

- official development train/validation records;
- official frozen SmolVLA predictions and Base actions on development
  identities;
- deployment RGB/language/proprioception fields;
- future demonstration 7D action segments only for discovery/validation label
  construction and diagnostics.

Forbidden:

- confirmatory rollout identities for tuning;
- future action segments or latent labels at inference;
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
`reports/cala_vla/preregistration.md`.

Save:

- all tried configurations;
- selected config;
- selected validation score and components;
- checkpoint paths and checksums;
- source gate manifest;
- latent-label manifest;
- action-delta and clean-retention summaries;
- selected config hash.

## Training Artifacts

Every trained identity must have:

- config JSON;
- training seed;
- split manifest;
- latent-label manifest;
- source gate manifest;
- checkpoint path;
- checksum;
- base checkpoint identity;
- validation metrics;
- action-delta metrics;
- latent/gate activation metrics;
- disk-reload verification.

Stage A cannot begin until `cala_full`, `cac_vla_latent_action_proxy`, and
`cala_no_context_gate_ablation` have disk-reloadable identities or are
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
- latent predictability;
- latent/gate activation;
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

CALA cannot become a paper candidate unless `cala_full` beats Base, the
CAC-style latent-action proxy, the no-context-gate ablation, and the
task-mean latent-action baseline under the matched protocol.

If the closest-prior proxy wins, CALA is not a useful local extension of the
closest prior.

If the no-context-gate ablation wins, the context gate is not useful.

If the task-mean latent-action baseline wins, the result is explained by a
cheaper task-prior mechanism.

## Implementation Boundaries

Implementation may add only the minimal reusable development runner and
modules needed to:

- build source and split manifests;
- construct legal latent labels on development identities;
- train or evaluate lightweight latent observability probes;
- instantiate identity-preserving CALA adapters;
- run the six-config validation search if Stage 0 passes;
- preflight disk-reloaded policy identities before closed-loop rollout.

Do not redesign CALA during implementation. Do not add baselines before the
frozen five-policy comparison. Do not use confirmatory identities for tuning.

## Current Next Action

Implement the Stage 0 development audit only. If Stage 0 stops, record the
correct pre-rollout failure label and pivot; do not launch validation search,
training, manifest freeze, or rollout.
