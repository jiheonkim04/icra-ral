# LIFT-VLA Mathematical Mechanism Audit

Date: 2026-07-15 KST

Proposal hash:
`3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `LIFT_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Audit decision: `LIFT_MATHEMATICAL_AUDIT_PREREGISTERED`

This audit freezes the mathematical and source-level contract before LIFT
implementation, validation search, manifest freeze, or rollout.

## Scope And Novelty

LIFT is not a new classifier-free-guidance equation. It tests whether applying
conditional-minus-unconditional guidance throughout a continuous SmolVLA action
flow produces a practically distinct and better VLA policy than applying CAG to
two completed action chunks.

The paper claim is allowed only if:

- pathwise LIFT is practically distinct from final-action CAG;
- LIFT beats CAG on a scoreable same-scene counterfactual claim axis;
- the matched-compute last-step ablation does not explain the gain;
- a pre-paper related-work search finds no equivalent released VLA action-flow
  implementation that already establishes the same result.

## Fixed Runtime Shapes

The local checkpoint and installed LeRobot path establish:

- batch size for local rollout: `B = 1`;
- raw environment image per real camera: `[B, 3, 256, 256]`;
- SmolVLA resized image per camera: `[B, 3, 512, 512]`;
- camera slots: `C = 3`, with two real LIBERO views and one mask-padded camera
  on the canonical path;
- visual embeddings per slot: `[B, 64, 960]`;
- language token IDs: `[B, 48]`;
- language attention mask: `[B, 48]`;
- raw policy state: `[B, 6]`;
- padded state: `[B, 32]`;
- prefix embeddings before padding: `[B, 241, 960]`, consisting of
  `3 * 64` visual tokens, `48` language tokens, and one state token;
- native noisy action chunk `x_k`: `[B, 50, 32]`;
- native vector fields `v_c`, `v_u`, and `v_lift`: `[B, 50, 32]`;
- flow time: `[B]`;
- flow steps: `K = 10`;
- Euler step: `dt = -0.1`;
- canonical unpadded policy chunk expected at runtime: `[B, 50, 7]`;
- executed policy-space first action: `[B, 7]`.

The implementation must measure and persist these shapes. A mismatch in native
`[50,32]`, step count `10`, language length `48`, or canonical output action
dimension `7` is `LIFT_IMPLEMENTATION_FAILURE`; it may not be silently padded,
truncated, or reinterpreted.

## Fixed Inputs

For one observation:

- `I = (I_1,I_2,I_3)`: resized image tensors and their camera masks;
- `s`: padded proprioceptive state;
- `z_l, m_l`: conditioned instruction tokens and mask;
- `z_0, m_0`: fixed empty-language tokens and mask produced by passing an empty
  task string through the same newline and tokenizer processors;
- `x_0 ~ N(0,I)`: one native action-noise tensor generated from the paired
  manifest seed.

All policies use identical `I`, camera masks, `s`, `x_0`, dtype, checkpoint,
step count, unpadding, postprocessor, and environment bridge. Only the declared
language branch and guidance location may differ.

## Prefix Caches

Let

- `P_c = Prefix(I,s,z_l,m_l)`;
- `P_u = Prefix(I,s,z_0,m_0)`.

Each prefix produces a pad mask and a frozen VLM key/value cache. The
conditioned and empty-language branches must use the same image and state
tensors. The empty-language branch is a training-free approximation to a
vision-action prior and is never described as a calibrated unconditional
density.

## Base Policy

Set `x_0^B = x_0`. For `k = 0,...,9`:

- `t_k = 1 - 0.1 k`;
- `v_k^B = v_theta(x_k^B,t_k;P_c)`;
- `x_(k+1)^B = x_k^B - 0.1 v_k^B`.

The native Base output is `A_B^native = x_10^B` with shape `[B,50,32]`.

## Transparent Training-Free CAG Proxy

Run two independent paths from the same `x_0`:

Conditioned path:

- `x_0^c = x_0`;
- `x_(k+1)^c = x_k^c - 0.1 v_theta(x_k^c,t_k;P_c)`.

Empty-language path:

- `x_0^u = x_0`;
- `x_(k+1)^u = x_k^u - 0.1 v_theta(x_k^u,t_k;P_u)`.

Mix once in native flow-output space:

`A_CAG^native = x_10^u + omega * (x_10^c - x_10^u)`.

No action unpadding, normalization, clipping, or 7D environment conversion may
occur before this mix. Same-noise coupling is the local paired fairness choice;
it is not attributed to the CAG authors unless source code later verifies it.

## LIFT Full

Set `x_0^L = x_0`. At every step evaluate both fields from the same current
LIFT latent:

- `v_k^c = v_theta(x_k^L,t_k;P_c)`;
- `v_k^u = v_theta(x_k^L,t_k;P_u)`;
- `v_k^L = v_k^u + omega * (v_k^c - v_k^u)`;
- `x_(k+1)^L = x_k^L - 0.1 v_k^L`.

The native LIFT output is `A_LIFT^native = x_10^L`.

At `omega = 1`, `v_k^L = v_k^c` algebraically and LIFT follows Base. The
implementation identity threshold is maximum absolute native and postprocessed
difference `<= 1e-5`.

## Matched-Compute Last-Step Ablation

Set `x_0^A = x_0`. At every step compute both fields:

- `v_k^c = v_theta(x_k^A,t_k;P_c)`;
- `v_k^u = v_theta(x_k^A,t_k;P_u)`.

For `k = 0,...,8`, discard `v_k^u` and update with `v_k^c`. At `k = 9`, use

`v_9^A = v_9^u + omega * (v_9^c - v_9^u)`.

Thus full LIFT, CAG, and the ablation each execute exactly `20` vector-field
evaluations. Base executes `10`. The ablation's extra discarded evaluations are
intentional compute matching and may not affect its latent state.

## Postprocessing Order

Each native output follows exactly one common path:

1. unpad native `[B,50,32]` to the loaded canonical policy action dimension;
2. apply the checkpoint's action postprocessor once;
3. obtain `[B,50,7]` canonical LIBERO policy-space actions;
4. pass executed actions through the same environment postprocessor once.

Variant-specific clipping, normalization, gripper filling, or bridge logic is
forbidden.

## No Training Objective

LIFT has no trainable parameters and no training loss. Therefore:

- there is no gradient path;
- no objective coefficient or gradient-norm search exists;
- no LoRA, QLoRA, adapter, or checkpoint is produced;
- KL divergence is not used between deterministic actions or vector fields.

The validation score is a selection statistic, not a differentiable objective.

## Mechanism Metrics

For each scored state, report:

- per-step field RMS:
  `d_k = rms(v_k^c - v_k^u)`;
- per-step field cosine between `v_k^c` and `v_k^u`;
- native chunk RMS `rms(A_LIFT^native - A_CAG^native)`;
- native chunk RMS `rms(A_LIFT^native - A_Abl^native)`;
- postprocessed first-action RMS for both comparisons;
- translation, rotation, and gripper deltas from Base;
- finite fraction, range-valid fraction, and clipping frequency;
- target grounding and task success when a valid scorer exists.

RMS is `sqrt(mean(delta^2))` over the declared tensor dimensions. It is a
distance diagnostic, not a probability divergence.

## Practical-Equivalence Threshold

The construction is frozen now; its values are calculated once from discovery
only and persisted before validation is decoded.

For native chunks:

- `e_native`: p99 RMS difference across repeated same-noise Base calls;
- `s_native`: median RMS magnitude of unpadded native Base chunks;
- `tau_native = max(100 * e_native, 0.01 * s_native, 1e-5)`.

For executed first actions:

- `e_exec`: p99 RMS difference across repeated same-noise postprocessed Base
  calls;
- `s_exec`: median RMS magnitude of Base first actions;
- `tau_exec = max(100 * e_exec, 0.01 * s_exec, 1e-5)`.

At the selected nonidentity scale, LIFT must exceed both thresholds versus CAG
and versus the ablation on at least `20%` of scoreable validation states. It must
also change at least one target-relevant translation or rotation dimension on
those states. Otherwise the result is
`LIFT_DESIGN_FAILURE_PRACTICAL_EQUIVALENCE`.

Threshold values are not recomputed on validation or confirmatory data.

## Counterfactual Source And Scorer Gate

The LIFT-specific manifest key is:

`suite/source_task/target_task/reset_seed`.

Each row must persist:

- source and target BDDL file hashes;
- source and target language;
- source and target goal predicates;
- object, fixture, and receptacle names;
- initial-state hash;
- feasibility audit;
- grounding-scorer definition;
- task-success-scorer definition;
- evidence partition.

Rows are valid only when the target goal can be instantiated and scored in the
source initial state. String-only swaps, cross-scene pairs, missing entities,
and rows whose success predicate remains the source goal are invalid.

Partitioning is by target task first, then reset identity. Target tasks may not
cross discovery, validation, and confirmatory partitions. Exact manifest keys
and initial-state hashes must also have zero overlap.

Minimum gate:

- at least `4` discovery target tasks;
- at least `2` validation target tasks;
- at least `2` untouched confirmatory target tasks;
- at least `10` scoreable discovery/validation episodes in total;
- zero invalid scorer rows;
- zero partition overlaps.

Failure is `LIFT_DATA_OR_BENCHMARK_FAILURE`.

## Headroom Gate

After one-chunk mechanism and compute smoke passes, bounded Base/CAG-only
development rollouts may establish headroom. Before LIFT scale selection:

- Base counterfactual failure rate must be at least `0.20` on the pooled
  discovery/validation audit;
- CAG residual failure rate must be at least `0.10`;
- at least one valid target-grounding miss must remain under CAG;
- no target task may be removed because of its observed outcome.

Failure is `LIFT_NO_HEADROOM`. These are development-only outcomes, not a
confirmatory scientific kill.

## Compute And Identity Gate

Before broad decoding, one canonical observation must establish:

- Base load and action chunk succeed on CUDA;
- LIFT `omega = 1` native and postprocessed max error `<= 1e-5`;
- full, CAG, and ablation each execute `20` field evaluations;
- finite and valid action fraction equals `1.0`;
- peak allocated CUDA memory remains below `15.5 GiB`;
- no CUDA out-of-memory or CPU fallback;
- median one-chunk LIFT latency from three measured calls is at most `4` times
  Base latency.

Sequential cache/branch scheduling is allowed only if it implements identical
equations and agrees with the reference within `1e-5`. Resource failure is
`LIFT_COMPUTE_INFEASIBLE`.

## Bounded Validation Search

Exactly three scales:

- `lift_w1.25`
- `lift_w1.50`
- `lift_w2.00`

No schedule, extra scale, alternate null prompt, seed sweep, sampler, layer
choice, or architecture variant is allowed.

For each scale, compute

`S = 0.35 success_or_grounding + 0.20 clean_retention + 0.20
mechanism_separation + 0.15 action_validity + 0.10 efficiency`.

Each component is normalized to `[0,1]` using formulas frozen in the
preregistration. Ties within `1e-6` select the smaller `omega`. Selection uses
validation only and is frozen before confirmatory testing.

## First Comparison

Exactly four policies:

1. `frozen_smolvla`
2. `training_free_cag_proxy`
3. `lift_full_pathwise_guidance`
4. `lift_last_step_only_ablation`

No fifth policy or standard-LoRA control is permitted in the first comparison.

## Allowed Pre-Rollout Decisions

- `LIFT_STAGE_0_PASS_TO_BOUNDED_VALIDATION`
- `LIFT_DATA_OR_BENCHMARK_FAILURE`
- `LIFT_NO_HEADROOM`
- `LIFT_IMPLEMENTATION_FAILURE`
- `LIFT_DESIGN_FAILURE_PRACTICAL_EQUIVALENCE`
- `LIFT_COMPUTE_INFEASIBLE`

None is reinterpreted as a closed-loop confirmatory scientific kill.

## Frozen Next Step

Freeze a preregistration and prototype protocol that instantiate this audit.
Implementation begins only afterward.

