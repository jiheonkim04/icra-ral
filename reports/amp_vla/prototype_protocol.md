# AMP-VLA Prototype Protocol

Date: 2026-07-16 KST

Proposal SHA-256:
`67ACC693C706B76BC9FB84F9E59BA3DF9C0463A0BAFABE539312D0E232DFE9A4`.

Decision: `AMP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

## Purpose

Run one bounded development-only Stage 0 audit before any adapter training,
validation search, Stage A manifest freeze, rollout, simulator access, or
confirmatory identity decoding. The audit tests whether AMP has legal
discovery-only action-manifold construction, noncollapsed coordinates,
deployment-input predictability, ABot-proxy headroom, identity-preserving
integration, finite gradients, projection-vs-clipping distinction, and
postprocessed 7D action validity.

## Frozen Stage 0 Inputs

- task sources:
  - `libero_spatial/task_3`
  - `libero_object/task_3`
  - `libero_goal/task_5`
  - `libero_10/task_5`
- discovery demos: `0..7`
- validation demos: `8..9`
- confirmatory simulator identities read: `0`
- reward, success, done, object pose, and privileged simulator state reads: `0`
- closed-loop episodes: `0`
- optimizer steps beyond identity/gradient smokes: `0`

## Frozen Manifold Construction

Candidate manifold fitting uses only discovery demos. Validation rows may be
projected or scored, but validation rows may never refit `Phi`, `Decode`, `P`,
coordinate normalization, task/phase baselines, or projection statistics.

Default local manifold implementation:

- action chunk tensor: normalized `[50,7]`;
- action flattening: `[350]` after valid-step mask;
- coordinate model: discovery-only PCA/ridge decoder or equivalent
  deterministic low-dimensional model;
- candidate dimensions: `D_m in {8,16}`;
- coordinate standardization: discovery-only mean/std with nonzero variance
  floor;
- projection metric: masked coordinate-mean normalized action Huber;
- clipping diagnostic: normalized coordinate clipping and, when available,
  postprocessed 7D bound diagnostic.

No manifold family change, source split change, coordinate dimension change,
projection approximation change, variance floor change, or clipping diagnostic
change is allowed after Stage 0 begins.

## ABot-M0 Prior Status Check

Before Stage 0 scoring, persist an official-prior asset check:

- if official ABot-M0 assets, weights, inference code, and evaluation code are
  already installed and locally verified, label policy 2 as official;
- otherwise label policy 2 as `abot_m0_action_manifold_proxy` and list all
  deviations from official ABot-M0 before validation outcomes are interpreted.

Stage 0 does not require a new download. A missing official asset is not an AMP
failure; it fixes the first comparison prior as a transparent proxy.

## Required Stage 0 Artifacts

The runner must persist:

- `reports/amp_vla/stage_0_pid.txt`
- `reports/amp_vla/stage_0_heartbeat.json`
- `reports/amp_vla/stage_0_status.json`
- `reports/amp_vla/stage_0_preflight.json`
- `reports/amp_vla/stage_0_manifest.json`
- `reports/amp_vla/stage_0_partial.json`
- `reports/amp_vla/stage_0_result.json`
- `reports/amp_vla/stage_0_result.md`
- `reports/amp_vla/stage_0_validation.json`
- `reports/amp_vla/stage_0_adjudication.md`
- `reports/amp_vla/stage_0_official_prior_asset_check.json`
- `reports/amp_vla/stage_0_implementation_blocker.json` on exception
- stdout, stderr, and exit-code files for any detached execution

Before detached execution, a foreground serializer preflight must construct a
small manifest fixture containing ordinary lists for actions, features,
manifold statistics, projection diagnostics, and gradient-smoke records;
canonical-hash it; write it; parse it; and reproduce the hash.

Partial results must be valid JSON after every accepted row.

## Manifest And Resume

The full manifest must be enumerated before model inference. Manifest keys are:

`(partition, suite, task_identity, source_hash, demo_id, frame_index, latent_dim, policy_probe)`.

Resume is allowed only after verifying method, proposal hash, source hashes,
split identity, manifest hash, manifold-statistic hashes, and cached-feature
hashes. Resume may add only missing manifest keys and may not repeat completed
keys. If a final result exists, refuse duplicate execution. A stale heartbeat
alone never proves death; verify PID, status, logs, partial JSON parseability,
and exit-code file first.

Duplicate manifest keys, duplicate partial keys, missing keys, extra keys, and
split-overlap keys must all be zero before accepting a final Stage 0 result.

## Stage 0 Required Checks

Stage 0 must verify all gates in `reports/amp_vla/preregistration.md`.
Specifically:

- proposal, source, manifest, and manifold-statistic hashes match;
- action chunks, features, proprioception, language/task, phase, and timestamps
  are finite and aligned;
- at least `512` discovery windows and `128` validation windows exist;
- every task has validation rows and no task contributes more than `40%` of the
  audit subset;
- retained manifold coordinates have positive variance in every dimension;
- manifold reconstruction beats task/phase mean actions by at least `10%`
  validation Huber or `0.01` absolute normalized Huber;
- deployment-input coordinate prediction beats task/phase coordinate means by
  at least `5%` relative Huber or `0.005` absolute normalized Huber;
- ABot proxy leaves residual headroom for AMP of at least `5%` relative Huber
  or `0.005` absolute normalized Huber;
- projection is not explained by clipping or bound-only validity under the
  frozen manifold-consistency metric;
- `amp_no_manifold_projection` differs from AMP's projection path before
  rollout;
- initialized and disk-reloaded AMP reproduces Base flow and postprocessed
  actions within `1e-6`;
- postprocessed 7D LIBERO action validity is preserved;
- normalized validity, projection delta, residual norm, gate value,
  translation/rotation/gripper deltas, changed dimensions, and clipping
  diagnostics are reported;
- expected AMP parameters receive finite nonzero gradients;
- frozen Base parameters receive zero gradients;
- `L_flow`, `L_coord`, `L_proj`, and `L_clean` magnitudes and gradient norms
  are finite and scale-balanced;
- exceptions are zero.

## Action Validity Unit System

The hard gate is postprocessed 7D LIBERO action validity after the existing
SmolVLA processor/postprocessor path. Normalized action validity is diagnostic
only and must be reported separately. Clipping and bound-only projection are
diagnostics, not allowed rescue methods. No clipping rescue, threshold
widening, or post-hoc validity reinterpretation is allowed after Stage 0
begins.

## Stage 0 Decisions

- `AMP_STAGE_0_PASS_TO_BOUNDED_VALIDATION`
- `AMP_STAGE_0_DATA_OR_SUPERVISION_FAILURE`
- `AMP_STAGE_0_NO_USABLE_HEADROOM`
- `AMP_STAGE_0_DESIGN_FAILURE`
- `AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

No Stage 0 decision is a closed-loop scientific kill.

## Continuation Rules

If Stage 0 passes, run only the frozen bounded validation search from
`reports/amp_vla/preregistration.md`.

If Stage 0 fails after manifest or model-audit rows exist, do not repair by
changing manifold construction, coordinate dimension, projection approximation,
task sources, target construction, coefficients, thresholds, action-validity
policy, prior proxy strength, clipping diagnostic, or baseline list. Archive
the failure class and continue to the next method cycle unless governance
explicitly permits a measurement-invalid repair.

Only a pre-manifest serializer or launcher defect that produced no accepted
rows may receive one implementation repair under the identical protocol.

## First Serious Comparison

After Stage 0 and bounded validation selection, the first serious comparison
remains exactly:

1. `smolvla_base`
2. `abot_m0_action_manifold_proxy`
3. `amp_full`
4. `amp_no_manifold_projection`
5. `standard_lora`

No additional baseline may be added before Stage A unless it tests a concrete
reviewer objection, is decision-relevant, and is cheaper than proceeding under
the frozen comparison.
