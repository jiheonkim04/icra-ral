# RAP-VLA Prototype Protocol

Date: 2026-07-16 KST

Proposal SHA-256:
`E9C3672544E486E4D5BAA883917F8429DB0FB36982F3F5944AC26A85783D1008`.

Decision: `RAP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

## Purpose

Run one bounded development-only Stage 0 audit before any adapter training,
validation search, Stage A manifest freeze, rollout, simulator access, or
confirmatory identity decoding. The audit tests whether RAP has legal
discovery-only retrieval memory, noncollapsed action anchors, predictable
residual targets, identity-preserving integration, finite gradients, and
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

## Frozen Retrieval Construction

Candidate memory entries come only from discovery demos. Validation rows may be
queried for scoring but may never be inserted into the policy memory.

Retrieval feature `f_t` is frozen to the concatenation of:

- frozen SmolVLA current-observation visual-policy feature;
- current proprioception;
- deployment-observable task/instruction identity;
- normalized demonstration phase.

Continuous feature components are z-scored with discovery-only statistics.
Task identity is a hard same-task filter before nearest-neighbor selection.
Phase enters as a normalized scalar in the same z-scored feature vector. The
retrieval metric is squared Euclidean distance over this frozen vector.

Top-k is fixed to `k=8`. Anchor weights are uniform over the top-k legal memory
rows after the same-task filter. If fewer than eight same-task rows are legal,
use all available rows and report the count; the noncollapse gate still must
pass.

No FAISS tuning, feature replacement, top-k change, phase weighting change, or
task-filter change is allowed after Stage 0 begins.

## OptimusVLA Prior Status Check

Before Stage 0 scoring, persist an official-prior asset check:

- if official OptimusVLA assets, memory files, checkpoints, and evaluation code
  are already installed and locally verified, label policy 2 as official;
- otherwise label policy 2 as `optimusvla_memory_prior_proxy` and list all
  deviations from official OptimusVLA before validation outcomes are
  interpreted.

Stage 0 does not require a new download. A missing official asset is not a RAP
failure; it fixes the first comparison prior as a transparent proxy.

## Required Stage 0 Artifacts

The runner must persist:

- `reports/rap_vla/stage_0_pid.txt`
- `reports/rap_vla/stage_0_heartbeat.json`
- `reports/rap_vla/stage_0_status.json`
- `reports/rap_vla/stage_0_preflight.json`
- `reports/rap_vla/stage_0_manifest.json`
- `reports/rap_vla/stage_0_partial.json`
- `reports/rap_vla/stage_0_result.json`
- `reports/rap_vla/stage_0_result.md`
- `reports/rap_vla/stage_0_validation.json`
- `reports/rap_vla/stage_0_adjudication.md`
- `reports/rap_vla/stage_0_official_prior_asset_check.json`
- `reports/rap_vla/stage_0_implementation_blocker.json` on exception
- stdout, stderr, and exit-code files for any detached execution

Before detached execution, a foreground serializer preflight must construct a
small manifest fixture containing ordinary lists for features, actions,
normalization statistics, and retrieval records; canonical-hash it; write it;
parse it; and reproduce the hash.

Partial results must be valid JSON after every accepted row.

## Manifest And Resume

The full manifest must be enumerated before model inference. Manifest keys are:

`(partition, suite, task_identity, source_hash, demo_id, frame_index, top_k, policy_probe)`.

Resume is allowed only after verifying method, proposal hash, source hashes,
split identity, manifest hash, and any cached-feature hashes. Resume may add
only missing manifest keys and may not repeat completed keys. If a final result
exists, refuse duplicate execution. A stale heartbeat alone never proves death;
verify PID, status, logs, partial JSON parseability, and exit-code file first.

Duplicate manifest keys, duplicate partial keys, missing keys, extra keys, and
split-overlap keys must all be zero before accepting a final Stage 0 result.

## Stage 0 Required Checks

Stage 0 must verify all gates in `reports/rap_vla/preregistration.md`.
Specifically:

- proposal, source, manifest, and feature-cache hashes match;
- memory rows, feature rows, action chunks, proprioception, language/task,
  phase, and timestamps are finite and aligned;
- at least `512` discovery windows and `128` validation windows exist;
- every task has validation rows and no task contributes more than `40%` of the
  audit subset;
- median top-8 neighborhood contains at least `3` unique demonstrations;
- no single source row accounts for more than `25%` of top-1 retrievals;
- anchors beat task/phase mean chunks by at least `10%` validation action MSE
  or `0.01` normalized Huber;
- residual targets have positive variance in every valid action dimension;
- a deployment-input residual probe beats zero-residual prediction by at least
  `5%` relative validation Huber or `0.01` absolute normalized Huber;
- anchor-only/no-residual differs from RAP's residual path before rollout;
- initialized and disk-reloaded RAP reproduces Base flow and postprocessed
  actions within `1e-6`;
- postprocessed 7D LIBERO action validity is preserved;
- normalized validity, Base-relative delta, anchor delta, residual norm, gate
  value, changed dimensions, retrieval confidence, and memory overhead are
  reported;
- expected RAP parameters receive finite nonzero gradients;
- frozen Base parameters receive zero gradients;
- `L_flow`, `L_res`, `L_rap`, and `L_clean` magnitudes and gradient norms are
  finite and scale-balanced;
- exceptions are zero.

## Action Validity Unit System

The hard gate is postprocessed 7D LIBERO action validity after the existing
SmolVLA processor/postprocessor path. Normalized action validity is diagnostic
only and must be reported separately. No clipping rescue, threshold widening,
or post-hoc validity reinterpretation is allowed after Stage 0 begins.

## Stage 0 Decisions

- `RAP_STAGE_0_PASS_TO_BOUNDED_VALIDATION`
- `RAP_STAGE_0_DATA_OR_SUPERVISION_FAILURE`
- `RAP_STAGE_0_NO_USABLE_HEADROOM`
- `RAP_STAGE_0_DESIGN_FAILURE`
- `RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

No Stage 0 decision is a closed-loop scientific kill.

## Continuation Rules

If Stage 0 passes, run only the frozen bounded validation search from
`reports/rap_vla/preregistration.md`.

If Stage 0 fails after manifest or model-audit rows exist, do not repair by
changing memory construction, features, top-k, task sources, residual targets,
coefficients, thresholds, action-validity policy, prior proxy strength, or
baseline list. Archive the failure class and continue to the next method cycle
unless governance explicitly permits a measurement-invalid repair.

Only a pre-manifest serializer or launcher defect that produced no accepted
rows may receive one implementation repair under the identical protocol.

## First Serious Comparison

After Stage 0 and bounded validation selection, the first serious comparison
remains exactly:

1. `smolvla_base`
2. `optimusvla_memory_prior_proxy`
3. `rap_full`
4. `rap_anchor_only_no_residual`
5. `standard_lora`

No additional baseline may be added before Stage A unless it tests a concrete
reviewer objection, is decision-relevant, and is cheaper than proceeding under
the frozen comparison.
