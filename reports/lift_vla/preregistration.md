# LIFT-VLA Preregistration

Date: 2026-07-15 KST

Proposal hash:
`3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F`

Mathematical audit: `reports/lift_vla/mathematical_mechanism_audit.md`

Decision: `LIFT_PREREGISTRATION_FROZEN_STAGE_0_PENDING`

Reviewer status: `APPROVE_WITH_FIXED_EMPIRICAL_RISKS`

## Claim

Under matched two-branch inference, pathwise language guidance through all ten
SmolVLA action-flow steps can improve scoreable same-scene counterfactual
grounding and task success over frozen Base, transparent final-action CAG, and a
matched-compute last-step-only ablation.

This is a narrow VLA action-flow transfer claim. It is not a new CFG algorithm,
official CAG reproduction, official LIBERO-CF reproduction, or LoRA method.

## Frozen Task Partition

The sorted official `libero_goal` BDDL target-task indices are partitioned before
any LIFT inference:

- discovery target tasks: `[0, 1, 2, 3]`;
- validation target tasks: `[4, 5, 6]`;
- confirmatory target tasks: `[7, 8, 9]`.

Source tasks may be paired only when their scene, entities, initial state, and
counterfactual target predicate pass the source gate. A target task cannot move
between partitions. Exact source/target/reset rows are persisted in
`reports/lift_vla/counterfactual_manifest.json`.

Reserved confirmatory BDDL metadata and unexecuted initial states may be hashed
to freeze identities. Confirmatory RGB observations, policy actions, grounding
outcomes, and success outcomes may not be decoded during Stage 0 or validation.

## Counterfactual Validity

Every manifest row must prove:

- same LIBERO scene;
- target objects, fixtures, and receptacles exist;
- target goal predicate is instantiated in the environment;
- target predicate differs from the source predicate;
- target grounding and success scorers refer to the target, not the source;
- initial-state and manifest-key overlap across partitions is zero.

String-only swaps and the repository's older `offline_proxy_only` pairs are
inadmissible. If at least `4 / 2 / 2` discovery/validation/confirmatory target
tasks cannot be retained, return `LIFT_DATA_OR_BENCHMARK_FAILURE`.

## Fixed Policies

1. `frozen_smolvla`
2. `training_free_cag_proxy`
3. `lift_full_pathwise_guidance`
4. `lift_last_step_only_ablation`

All variants use the same SmolVLA checkpoint, images, state, task text, initial
noise, ten Euler steps, native-space output, postprocessor, and environment
bridge. CAG, LIFT, and ablation each use exactly `20` vector-field evaluations.

No standard LoRA or fifth policy is allowed. Generic adaptation does not test
the inference-only mechanism.

## Fixed Empty-Language Branch

The empty branch passes the exact empty task string `""` through the canonical
SmolVLA newline processor and tokenizer. No alternate null prompt, padding mask,
negative prompt, or learned token may be tried.

The resulting IDs and attention mask are persisted in the Stage 0 result.

## Stage 0 Order

Stage 0 runs in this fixed order:

1. static BDDL and partition audit;
2. LIFT-specific source/scorer manifest construction;
3. load-only CUDA and exact-shape check;
4. one-discovery-observation Base repeatability and `omega = 1` identity check;
5. one-chunk Base/CAG/LIFT/ablation mechanism, action-validity, latency, and
   memory smoke;
6. discovery-only practical-threshold calculation and persistence;
7. bounded Base/CAG-only discovery/validation headroom rollout;
8. final Stage 0 adjudication.

If an earlier hard gate fails, later policy inference or rollout must not run.

## Stage 0 Gates

### Shapes And Identity

- native chunk: `[1, 50, 32]`;
- flow steps: `10`;
- language tokens: `[1, 48]`;
- canonical policy chunk: `[1, 50, 7]`;
- LIFT `omega = 1` native max error: `<= 1e-5`;
- LIFT `omega = 1` postprocessed max error: `<= 1e-5`;
- finite and range-valid action fraction: `1.0`.

### Activation And Practical Separation

- nonzero finite conditioned-minus-empty field on at least `80%` of scored
  discovery/validation states;
- discovery thresholds use the exact formulas in the mathematical audit;
- at selected nonidentity scales, LIFT-versus-CAG and LIFT-versus-ablation must
  exceed native and executed thresholds on at least `20%` of validation states;
- at least one target-relevant translation or rotation dimension changes.

Below-threshold behavior is
`LIFT_DESIGN_FAILURE_PRACTICAL_EQUIVALENCE`.

### Headroom

- Base counterfactual failure rate: `>= 0.20`;
- CAG residual failure rate: `>= 0.10`;
- at least one target-grounding miss under CAG;
- zero task removal based on observed outcomes.

Failure is `LIFT_NO_HEADROOM`.

### Compute

- peak allocated CUDA memory: `< 15.5 GiB`;
- no OOM or CPU fallback;
- median full-LIFT one-chunk latency over three measured calls: `<= 4` times
  Base;
- full, CAG, and ablation field-evaluation counts: exactly `20` each.

Failure is `LIFT_COMPUTE_INFEASIBLE`.

## Bounded Validation Search

Allowed only after Stage 0 passes. Exactly three scales:

1. `lift_w1.25`
2. `lift_w1.50`
3. `lift_w2.00`

No extra seeds for selection, schedules, null prompts, samplers, steps, or
architectures are allowed.

For each scale:

- `success_or_grounding = 0.5 * target_success_rate + 0.5 * target_grounding_rate`;
- `clean_retention = min(1, clean_success_rate / max(base_clean_success_rate, 0.01))`;
- `mechanism_separation = 0.5 * min(1, fraction_above_cag_threshold / 0.20) +
  0.5 * min(1, fraction_above_ablation_threshold / 0.20)`;
- `action_validity = finite_and_range_valid_fraction`;
- `efficiency = min(1, base_median_latency / lift_median_latency)`.

Selection score:

`S = 0.35 * success_or_grounding + 0.20 * clean_retention + 0.20 *
mechanism_separation + 0.15 * action_validity + 0.10 * efficiency`.

Ties within `1e-6` select smaller `omega`. All three results and negative
results are retained. The selected scale, checkpoint identity, source manifest,
and metrics are frozen before confirmatory testing.

## Clean Retention

Clean validation uses the original matching instruction and goal for the same
validation target-task identities. Clean success may not be replaced by action
L2. Translation, rotation, gripper, clipping, latency, and memory diagnostics are
reported separately.

## Stage A

Approximately ten paired episodes per policy on the sealed confirmatory target
tasks and reset identities. Policies share source state and initial action noise.

Permanent Stage A kill is allowed only for:

- mechanism invalidity or practical equivalence;
- no headroom;
- catastrophic degradation;
- clear CAG or ablation dominance;
- exact trivial equivalence.

Small differences advance to Stage B without retuning.

## Stage B

At least forty paired episodes per key policy. Report paired wins/losses/ties,
bootstrap confidence interval, effect size, failure-rate reduction, per-task
breakdown, target grounding, mechanism activation, clean retention, latency,
memory, and action validity.

One expansion to eighty is allowed only when the frozen Stage B criterion is
genuinely unresolved. No scale or method change is allowed.

## Paper-Candidate Gate

LIFT becomes a serious paper candidate only if:

- LIFT beats Base;
- LIFT beats transparent training-free CAG;
- LIFT beats matched-compute last-step guidance;
- clean behavior is retained;
- target-aware and flow-path evidence support the mechanism;
- novelty remains defensible after the required related-work audit.

After GO, verify Quantized OpenVLA-OFT INT4 plus LIFT, one second claim-specific
condition, recent baselines, latency, and compute. This later scale-up does not
alter the first SmolVLA protocol.

## Allowed Stage 0 Decisions

- `LIFT_STAGE_0_PASS_TO_BOUNDED_VALIDATION`
- `LIFT_DATA_OR_BENCHMARK_FAILURE`
- `LIFT_NO_HEADROOM`
- `LIFT_IMPLEMENTATION_FAILURE`
- `LIFT_DESIGN_FAILURE_PRACTICAL_EQUIVALENCE`
- `LIFT_COMPUTE_INFEASIBLE`

No other decision label may be invented after results.

