# RAP-VLA Preregistration

Date: 2026-07-16 KST

Decision: `RAP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

This preregistration freezes RAP-VLA's evidence partitions, Stage 0 audit,
bounded validation search, first serious comparison, metrics, thresholds, and
stop rules. It does not authorize training, rollout, validation search, or
confirmatory-test access until the executable prototype protocol is written and
validated.

## Frozen Method Identity

Method: `RAP-VLA`, Retrieval-Anchored Prior residualization for VLA action
flows.

Proposal hash:
`E9C3672544E486E4D5BAA883917F8429DB0FB36982F3F5944AC26A85783D1008`.

Closest positive prior: OptimusVLA.

Scientific mechanism: retrieve legal action anchors from discovery memory
using deployment-observable current inputs, then train a bounded residual
action-flow path around the anchor. LoRA is only identity-preserving
implementation infrastructure.

RAP is not VDR, KITE, RAR, LIFT, EAC, HEST, HASTE, COVI, or any rescue of a
closed method.

## Evidence Partitions

`DISCOVERY`:

- build and inspect the candidate memory;
- fit feature normalization for retrieval;
- audit retrieval diversity and target variance;
- construct residual targets;
- debug tensor shapes, serializers, identity, and gradient path.

`VALIDATION`:

- query but never insert validation rows into candidate memory;
- select one RAP coefficient from the frozen bounded search;
- score clean retention, action validity, retrieval headroom, residual
  predictability, RAP-vs-anchor-only distinction, and memory overhead.

`CONFIRMATORY_TEST`:

- untouched until memory construction, feature normalization, top-k, coefficient
  choice, policy list, tasks, reset identities, metrics, and thresholds are
  frozen;
- confirmatory outcomes may not retune RAP, memory, retrieval features, task
  filters, top-k, coefficient, action-validity rule, baselines, or ablations.

## Fixed Development Sources

Use these four development task families:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Within each source:

- discovery/training demonstrations: `0..7`;
- validation demonstrations: `8..9`;
- confirmatory task/reset identities: sealed.

No reward, success, done flag, simulator state, object pose, reset identity, or
confirmatory outcome may enter Stage 0, memory construction, validation search,
or training.

## Stage 0 Development Audit

Stage 0 is development-only and must run before any optimizer step beyond
allowed identity/gradient smokes.

Required checks:

1. proposal hash and source hashes match;
2. discovery, validation, and reserved-test partitions are persisted;
3. memory row keys, feature keys, action chunks, proprioception, language/task,
   phase, and timestamps are finite and aligned;
4. duplicate, missing, extra, frame-overlap, and split-overlap keys are zero;
5. at least `512` discovery and `128` validation windows are available;
6. every task has validation rows and no task contributes more than `40%` of
   the audit subset;
7. top-k retrieval neighborhoods are noncollapsed:
   - median top-8 neighborhood has at least `3` unique demonstrations;
   - no single source row accounts for more than `25%` of all top-1 retrievals;
8. retrieved anchors beat task/phase mean chunks by at least `10%` validation
   action MSE or `0.01` normalized Huber;
9. residual targets have positive variance in every action dimension after
   masking invalid padded steps;
10. a deployment-input residual probe beats zero-residual prediction by at
    least `5%` relative validation Huber or `0.01` absolute normalized Huber;
11. anchor-only/no-residual differs from RAP's residual path before rollout;
12. initialized and disk-reloaded adapter reproduces Base flow and
    postprocessed actions within `1e-6`;
13. postprocessed 7D action validity is preserved;
14. normalized action validity, Base-relative deltas, residual norm, gate value,
    and dimensions changed are reported;
15. expected RAP parameters receive finite nonzero gradients;
16. frozen Base parameters receive zero gradients;
17. objective magnitudes and gradient norms are finite and scale-balanced;
18. exceptions are zero.

Stage 0 may not use simulator rollout, reward/success/done, confirmatory
identity decoding, or validation-selected hyperparameter changes.

## Stage 0 Stop Classes

Return `RAP_STAGE_0_DATA_OR_SUPERVISION_FAILURE` for source, alignment, overlap,
collapsed retrieval, collapsed residual target, or coverage failures.

Return `RAP_STAGE_0_NO_USABLE_HEADROOM` when retrieved anchors do not beat
task/phase means, Base has no relevant failure, or the OptimusVLA proxy leaves
no plausible residual failure.

Return `RAP_STAGE_0_DESIGN_FAILURE` when residuals are not predictable from
deployment inputs, anchor-only is equivalent to RAP, the residual branch is
nonacting, or the mechanism activates everywhere rather than relevant states.

Return `RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` for hash,
serialization, identity, persistence, gradient, frozen-parameter, action
validity, global action-delta, or exception defects.

Return `RAP_STAGE_0_PASS_TO_BOUNDED_VALIDATION` only if all gates pass.

These Stage 0 outcomes are not closed-loop scientific kills.

## Bounded Validation Search

Maximum six configurations:

1. RAP `lambda_r=0.1`, `g_max=0.25`;
2. RAP `lambda_r=0.3`, `g_max=0.25`;
3. RAP `lambda_r=1.0`, `g_max=0.25`;
4. transparent OptimusVLA memory-prior proxy;
5. `rap_anchor_only_no_residual`;
6. matched standard LoRA.

Frozen factors:

- retrieval feature definition;
- feature normalization;
- retrieval metric;
- top-k;
- task/phase filters;
- memory split;
- residual target formula;
- adapter rank;
- optimizer;
- training steps;
- checkpoint-selection score.

One seed per configuration unless a frozen run is genuinely unresolved; no more
than two seeds may then be used. No combinatorial grid is allowed.

Validation score:

`S = 0.30 * validation_success_or_proxy
   + 0.25 * RAP_minus_anchor_only_margin
   + 0.20 * clean_action_retention
   + 0.15 * postprocessed_action_validity
   + 0.10 * memory_overhead_score`.

If closed-loop validation is unavailable, `validation_success_or_proxy` must be
defined before execution by one frozen deployment-observable proxy. Offline
action L2 alone may not select the configuration. Tie break: smaller
`lambda_r`, then lower retrieval overhead.

## First Serious Comparison

Exactly five policies:

1. `smolvla_base`;
2. `optimusvla_memory_prior_proxy`;
3. `rap_full`;
4. `rap_anchor_only_no_residual`;
5. `standard_lora`.

The OptimusVLA policy is official only if official assets are installed and
verified locally before Stage 0. Otherwise it is a transparent proxy, and all
deviations from official OptimusVLA must be listed before validation outcomes
are interpreted.

## Stage A And Stage B

Stage A uses approximately ten paired episodes per policy after validation
selection and manifest freeze. It may kill only for mechanism invalidity, no
headroom, catastrophic degradation, clear prior or ablation dominance, or exact
trivial equivalence.

Stage B uses at least forty paired episodes per key policy with paired
wins/losses/ties, bootstrap confidence interval, effect size, failure-rate
reduction, per-task breakdown, mechanism activation, clean retention, latency,
and memory overhead. One expansion to eighty is allowed only if Stage B is
genuinely unresolved by its frozen rule.

## Paper-Candidate Gate

RAP becomes a serious paper candidate only if:

- RAP beats Base;
- RAP beats the OptimusVLA proxy on the matched claim axis;
- RAP beats anchor-only/no-residual;
- matched standard LoRA does not explain the gain;
- retrieved anchors and learned residuals are both active;
- clean behavior is retained;
- postprocessed 7D action validity is retained;
- memory overhead is reported and acceptable;
- novelty remains defensible.

Then verify the unchanged method on Quantized OpenVLA-OFT INT4 and add one
claim-specific second condition or benchmark.

## Resource Evidence Rule

Windows gaming / Efficiency Mode / resource-contention intervals remain
tracked separately. Timing, throughput, wall-clock efficiency, resource
utilization, and latency measurements overlapping or unresolved against those
intervals are not final paper evidence. Closed-loop task-success rows may
remain valid only if the simulator is synchronous, no timeout/exception occurs,
action semantics and task/reset identities are unchanged, and duplicate rows
are absent.
