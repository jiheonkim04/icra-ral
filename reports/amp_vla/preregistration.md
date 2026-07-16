# AMP-VLA Preregistration

Date: 2026-07-16 KST

Decision: `AMP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

This preregistration freezes AMP-VLA's evidence partitions, Stage 0 audit,
bounded validation search, first serious comparison, metrics, thresholds, and
stop rules. It does not authorize training, rollout, validation search, or
confirmatory-test access until the executable prototype protocol is written and
validated.

## Frozen Method Identity

Method: `AMP-VLA`, Action-Manifold Projection for VLA action-flow adaptation.

Proposal hash:
`67ACC693C706B76BC9FB84F9E59BA3DF9C0463A0BAFABE539312D0E232DFE9A4`.

Closest positive prior: ABot-M0.

Scientific mechanism: fit a discovery-only action manifold over legal
demonstration action chunks, then constrain SmolVLA adapter-induced action
changes through identity-preserving manifold projection and a bounded residual
gate. LoRA is only implementation infrastructure.

AMP is not RAP, VDR, KITE, HEST, HASTE, IARC, FAMR, PCAV, SPARC, NICE, COVI,
LIFT, EAC, or any rescue of a closed method.

## Evidence Partitions

`DISCOVERY`:

- fit action-manifold coordinate and decoder statistics;
- inspect action support, coordinate variance, reconstruction, task/phase
  coverage, and clipping diagnostics;
- construct the transparent ABot-M0 action-manifold proxy;
- debug tensor shapes, serializers, identity, projection, and gradient path.

`VALIDATION`:

- query validation rows only for development scoring;
- select one AMP configuration from the frozen bounded search;
- score clean retention, postprocessed action validity, manifold health,
  ABot-proxy headroom, AMP-vs-no-projection distinction, and projection
  overhead.

`CONFIRMATORY_TEST`:

- untouched until manifold construction, coordinate dimension, projection
  approximation, coefficient choice, policy list, tasks, reset identities,
  metrics, and thresholds are frozen;
- confirmatory outcomes may not retune AMP, ABot proxy, latent dimension,
  projection strength, coefficients, thresholds, tasks, reset identities,
  baselines, or ablations.

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
confirmatory outcome may enter Stage 0, manifold construction, validation
search, or training.

## Stage 0 Development Audit

Stage 0 is development-only and must run before any optimizer step beyond
allowed identity/gradient smokes.

Required checks:

1. proposal hash and source hashes match;
2. discovery, validation, and reserved-test partitions are persisted;
3. action chunks, feature keys, proprioception, language/task, phase, and
   timestamps are finite and aligned;
4. duplicate, missing, extra, frame-overlap, and split-overlap keys are zero;
5. at least `512` discovery and `128` validation windows are available;
6. every task has validation rows and no task contributes more than `40%` of
   the audit subset;
7. retained manifold coordinates have positive variance in every dimension;
8. manifold reconstruction beats task/phase mean actions by at least `10%`
   validation Huber or `0.01` absolute normalized Huber;
9. deployment-input coordinate prediction beats task/phase coordinate means by
   at least `5%` relative Huber or `0.005` absolute normalized Huber;
10. ABot proxy leaves residual headroom for AMP of at least `5%` relative
    Huber or `0.005` absolute normalized Huber;
11. projection is not explained by clipping or bound-only validity under the
    frozen manifold-consistency metric;
12. `amp_no_manifold_projection` differs from AMP before rollout;
13. initialized and disk-reloaded adapter reproduces Base flow and
    postprocessed actions within `1e-6`;
14. postprocessed 7D action validity is preserved;
15. normalized validity, postprocessed validity, projection delta, residual
    norm, gate value, translation/rotation/gripper deltas, and changed
    dimensions are reported;
16. expected AMP parameters receive finite nonzero gradients;
17. frozen Base parameters receive zero gradients;
18. objective magnitudes and gradient norms are finite and scale-balanced;
19. exceptions are zero.

Stage 0 may not use simulator rollout, reward/success/done, confirmatory
identity decoding, or validation-selected hyperparameter changes.

## Stage 0 Stop Classes

Return `AMP_STAGE_0_DATA_OR_SUPERVISION_FAILURE` for source, alignment, overlap,
collapsed action manifold, collapsed coordinate target, or coverage failures.

Return `AMP_STAGE_0_NO_USABLE_HEADROOM` when Base or ABot proxy leaves no
plausible AMP failure mode, manifold reconstruction does not beat task/phase
means, or projection is fully explained by clipping.

Return `AMP_STAGE_0_DESIGN_FAILURE` when manifold coordinates are not
predictable from deployment inputs, no-projection is equivalent to AMP, the
residual/gate path is nonacting, or the mechanism activates everywhere rather
than relevant states.

Return `AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` for hash,
serialization, identity, persistence, gradient, frozen-parameter, projection,
action-validity, global action-delta, or exception defects.

Return `AMP_STAGE_0_PASS_TO_BOUNDED_VALIDATION` only if all gates pass.

These Stage 0 outcomes are not closed-loop scientific kills.

## Bounded Validation Search

Maximum six configurations:

1. AMP `latent_dim=8`, `lambda_p=0.3`, `g_max=0.20`;
2. AMP `latent_dim=16`, `lambda_p=0.3`, `g_max=0.20`;
3. AMP `latent_dim=16`, `lambda_p=1.0`, `g_max=0.20`;
4. transparent ABot-M0 action-manifold proxy;
5. `amp_no_manifold_projection`;
6. matched standard LoRA.

Frozen factors:

- manifold family and coordinate standardization;
- projection approximation;
- feature definition;
- task/phase labels;
- source split;
- residual/gate parameterization;
- adapter rank;
- optimizer;
- training steps;
- checkpoint-selection score.

One seed per configuration unless a frozen run is genuinely unresolved; no more
than two seeds may then be used. No combinatorial grid is allowed.

Validation score:

`S = 0.30 * validation_success_or_proxy
   + 0.25 * AMP_minus_ABot_proxy_margin
   + 0.20 * clean_action_retention
   + 0.15 * postprocessed_action_validity
   + 0.10 * projection_overhead_score`.

If closed-loop validation is unavailable, `validation_success_or_proxy` must be
defined before execution by one frozen deployment-observable proxy. Offline
action L2 alone may not select the configuration. Tie break: lower latent
dimension, then smaller projection coefficient.

## First Serious Comparison

Exactly five policies:

1. `smolvla_base`;
2. `abot_m0_action_manifold_proxy`;
3. `amp_full`;
4. `amp_no_manifold_projection`;
5. `standard_lora`.

The ABot-M0 policy is official only if official assets are installed and
verified locally before Stage 0. Otherwise it is a transparent proxy, and all
deviations from official ABot-M0 must be listed before validation outcomes are
interpreted.

## Stage A And Stage B

Stage A uses approximately ten paired episodes per policy after validation
selection and manifest freeze. It may kill only for mechanism invalidity, no
headroom, catastrophic degradation, clear prior or ablation dominance, or exact
trivial equivalence.

Stage B uses at least forty paired episodes per key policy with paired
wins/losses/ties, bootstrap confidence interval, effect size, failure-rate
reduction, per-task breakdown, mechanism activation, clean retention, latency,
and projection overhead. One expansion to eighty is allowed only if Stage B is
genuinely unresolved by its frozen rule.

## Paper-Candidate Gate

AMP becomes a serious paper candidate only if:

- AMP beats Base;
- AMP beats the ABot-M0 proxy on the matched claim axis;
- AMP beats the no-projection ablation;
- matched standard LoRA does not explain the gain;
- manifold projection is active in relevant states rather than everywhere;
- clipping or bound-only validity does not explain the effect;
- clean behavior is retained;
- postprocessed 7D action validity is retained;
- projection overhead is reported and acceptable;
- novelty remains defensible.

Then verify the unchanged method on Quantized OpenVLA-OFT INT4 and add one
claim-specific second condition or benchmark.

## Resource Evidence Rule

Windows gaming / Efficiency Mode / resource-contention intervals remain tracked
separately. Timing, throughput, wall-clock efficiency, resource utilization,
and latency measurements overlapping or unresolved against those intervals are
not final paper evidence. Closed-loop task-success rows may remain valid only
if the simulator is synchronous, no timeout/exception occurs, action semantics
and task/reset identities are unchanged, and duplicate rows are absent.
