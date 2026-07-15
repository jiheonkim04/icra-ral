# LIFT-VLA Researcher A Proposal

Date: 2026-07-15 KST

Method: `LIFT-VLA`, Language-Induced Flow Transport for frozen SmolVLA.

Decision: `LIFT_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Contribution type: `CROSS_DOMAIN_MECHANISM_TRANSFER`

Closest external prior: Counterfactual Action Guidance (CAG) and LIBERO-CF,
https://arxiv.org/abs/2602.17659.

## Research Claim

Frozen SmolVLA can underweight language when a visually plausible scene favors
a familiar action. Training-free CAG addresses this by mixing the final actions
of a language-conditioned policy and a language-dropped policy. LIFT tests a
narrower flow-model hypothesis: for a continuous flow-matching VLA, language
guidance should shape every latent action state along the transport path rather
than shift two completed action paths only at the end.

The paper-worthy claim is conditional and falsifiable:

> Under a matched two-branch inference budget, full-path language guidance in
> SmolVLA's action vector field improves same-scene instruction grounding and
> closed-loop task success over frozen Base, final-action CAG, and last-step-only
> field guidance while retaining clean action validity.

LIFT does not claim to invent classifier-free guidance. Its provisional novelty
is the transfer of pathwise classifier-free guidance to a continuous VLA action
flow, together with a controlled comparison against CAG's completed-action
mixing on the same claim axis.

## Scientific Method

Let:

- `o`: current legal RGB and proprioceptive observation;
- `l`: current language instruction;
- `empty`: the fixed empty-language input produced by the same tokenizer and
  preprocessor;
- `x_k in R^(B x H x D)`: SmolVLA's latent noisy action chunk at flow step `k`;
- `t_k in R^B`: the current integration time;
- `v_c(x_k,t_k,o,l) in R^(B x H x D)`: conditioned vector field;
- `v_u(x_k,t_k,o,empty) in R^(B x H x D)`: language-dropped vector field;
- `omega`: one scalar guidance scale selected on validation only;
- `K = 10`: the unchanged SmolVLA integration-step count;
- `dt = -1/K`: the unchanged Euler step.

For every step `k = 0,...,K-1`, LIFT computes

`v_lift = v_u + omega * (v_c - v_u)`

from the same `x_k`, observation, state, and integration time, then updates

`x_(k+1) = x_k + dt * v_lift`.

Both branches use the same frozen SmolVLA weights. They differ only in language
tokens and language attention masks. Images, proprioception, initial noise,
flow-step count, numerical integration, action unpadding, normalization, and 7D
LIBERO bridge are shared.

### Closest Prior

The transparent training-free CAG proxy computes two complete flows from the
same initial noise:

- `a_c = Flow(o,l,x_0)`;
- `a_u = Flow(o,empty,x_0)`;
- `a_cag = a_u + omega * (a_c - a_u)`.

CAG mixes completed actions. LIFT integrates a guided vector field from a shared
latent state. They are not generally equivalent because `v_theta` is nonlinear
in `x_k` and the two independently completed CAG branches visit different
latent states after their first update.

### Key Ablation

`lift_last_step_only_ablation` follows the conditioned Base path for steps
`0,...,K-2`, then applies the same conditional-minus-unconditional field rule
only at step `K-1`. It uses the same two-branch final-step compute needed to test
whether guidance over the complete transport path is necessary.

## Low-Compute Parameterization

LIFT is inference-only:

- frozen SmolVLA checkpoint;
- one conditioned and one empty-language prefix cache;
- two vector-field evaluations per flow step;
- mixed precision under the already validated SmolVLA runtime;
- no optimizer, adapter, trainable head, LoRA, QLoRA, or new checkpoint.

The closest-prior arm is matched to the same two full branches and ten flow steps.
Latency, peak allocated GPU memory, and branch count must be reported.

Standard LoRA is omitted because generic fine-tuning does not test whether
pathwise language guidance improves over final-action CAG for a frozen,
inference-only policy.

## Evidence Partitions

### Discovery

Discovery may use persisted local offline LIBERO records and official
LIBERO-Goal task metadata to:

- verify empty-language preprocessing;
- inspect conditioned-minus-unconditioned vector fields;
- establish that instructions produce noncollapsed field differences;
- identify valid same-scene/different-goal pairs;
- debug implementation and tensor shapes.

### Validation

Validation uses disjoint task/reset identities to:

- choose one guidance scale from the bounded list;
- check clean retention and action validity;
- establish Base and CAG residual headroom on the local proxy;
- choose the single final configuration.

### Confirmatory Test

Confirmatory tasks and reset identities remain sealed until method, scale,
policy list, metrics, thresholds, and manifests are frozen. Confirmatory results
may not retune LIFT. An official LIBERO-CF claim requires official benchmark
assets or a separately frozen and independently validated counterfactual suite;
the local development proxy may not be renamed LIBERO-CF.

## Stage 0 Development Audit

No validation search or rollout is allowed before all applicable checks pass.

1. Source and partition gate:
   - persist discovery, validation, and reserved-test identities;
   - prove zero overlap by task/reset/frame keys;
   - decode zero reserved-test observations in Stage 0;
   - record the exact empty-language token and mask construction.

2. Hook and branch fidelity:
   - verify the local code path uses `SmolVLAModel.sample_actions` and
     `denoise_step` without changing Base weights;
   - conditioned and unconditioned branches share image tensors, state, noise,
     step count, dtype, and postprocessor;
   - same-noise repeated calls are deterministic within a frozen tolerance.

3. Identity:
   - at `omega = 1`, LIFT must reproduce the conditioned Base chunk with maximum
     absolute difference at most `1e-5` before postprocessing and at most `1e-5`
     after postprocessing;
   - action valid fraction must equal `1.0`;
   - no queue, normalization, or action-dimension mismatch is allowed.

4. Mechanism activation:
   - `||v_c - v_u||` is finite and nonzero on at least `80%` of scored
     discovery/validation states;
   - report the per-step field difference, cosine relation, dimensions changed,
     and conditional/unconditional action difference;
   - LIFT at every nonidentity development scale must differ from Base, CAG, and
     last-step-only guidance on at least one scored state.

5. Headroom:
   - frozen Base must show nontrivial same-scene instruction sensitivity or
     grounding failure on a development-only LIBERO-Goal proxy;
   - final-action CAG must leave residual error or failure for LIFT to address;
   - instruction swaps must be feasible in the scene and independently audited;
   - cross-scene or absent-object swaps are excluded.

6. Disruption and efficiency:
   - report translation, rotation, and gripper deltas from Base;
   - report action bounds, NaN/Inf counts, latency, and peak GPU memory;
   - reject scales that change every action globally or violate the existing 7D
     bridge.

Allowed Stage 0 outcomes:

- `LIFT_STAGE_0_PASS_TO_BOUNDED_VALIDATION`
- `LIFT_DATA_OR_BENCHMARK_FAILURE`
- `LIFT_NO_HEADROOM`
- `LIFT_IMPLEMENTATION_FAILURE`
- `LIFT_DESIGN_FAILURE`
- `LIFT_COMPUTE_INFEASIBLE`

These pre-rollout outcomes are not closed-loop scientific kills.

## Bounded Validation Search

Only three configurations are allowed:

1. `lift_w1.25`
2. `lift_w1.50`
3. `lift_w2.00`

No schedule, layer choice, alternate null prompt, stochastic seed sweep,
additional coefficient, or architecture variant may be introduced in this
method cycle. `omega = 1` is the identity audit, not a selectable Ours
configuration.

At most two lightweight repeat seeds are allowed for diagnostics; the frozen
rollout manifest remains paired by initial state and initial action noise.

Validation score:

`S = 0.35 * validation_success_or_grounding + 0.20 * clean_retention + 0.20 *
mechanism_separation + 0.15 * action_validity + 0.10 * efficiency`

If closed-loop validation is not feasible, `validation_success_or_grounding`
must be replaced before execution by one frozen target-aware proxy. Offline
action L2 alone may not select the configuration.

## First Serious Comparison

Exactly four policies:

1. `frozen_smolvla`
2. `training_free_cag_proxy`
3. `lift_full_pathwise_guidance`
4. `lift_last_step_only_ablation`

| Comparison | Scientific question |
| --- | --- |
| Base vs Ours | Does full-path language-flow guidance improve frozen SmolVLA? |
| Prior vs Ours | Does pathwise guidance beat final-action CAG under matched two-branch inference? |
| Ablation vs Ours | Is guidance throughout the transport path necessary? |

No fifth policy is included. Standard LoRA is irrelevant, and another simple
inference policy would duplicate the CAG or last-step mechanism question.

## Stage A And Stage B

Stage A uses approximately ten paired episodes per policy. It may permanently
kill only for mechanism invalidity, no headroom, catastrophic degradation,
clear prior or ablation dominance, or exact trivial equivalence.

Stage B uses at least forty paired episodes per key policy with paired
wins/losses/ties, bootstrap confidence interval, effect size, failure-rate
reduction, per-task breakdown, mechanism activation, clean retention, latency,
and memory. One expansion to eighty is allowed only if the frozen Stage B rule
is genuinely unresolved.

## GO And Stop Logic

LIFT becomes a serious paper candidate only if:

- LIFT beats frozen SmolVLA;
- LIFT beats the transparent training-free CAG proxy on the same matched claim
  axis;
- LIFT beats last-step-only guidance;
- no action-invalidity or global-disruption explanation accounts for the gain;
- clean behavior is retained;
- pathwise field diagnostics support the intended mechanism;
- novelty remains defensible as a narrow VLA action-flow transfer.

Stop or kill the current formulation when:

- the Base has no counterfactual headroom;
- empty-language fields are collapsed, invalid, or not deployment-observable;
- full-path guidance is exactly equivalent to final-action CAG or the last-step
  ablation;
- CAG clearly dominates under the matched budget;
- LIFT causes catastrophic action or clean-retention degradation;
- the claim requires cross-scene, absent-object, privileged, or test-tuned
  instruction swaps;
- official benchmark equivalence cannot be established for a claim that
  requires it.

## Non-Claims

- LIFT does not invent classifier-free guidance.
- LIFT is not official CAG or official LIBERO-CF reproduction.
- LIFT is not a LoRA, QLoRA, PEFT, or adaptation-efficiency method.
- LIFT is not a trained vision-action prior.
- LIFT is not a new flow-matching backbone or numerical integrator.
- LIFT does not rescue COVI, RAR, CALA, G3P, EAC, MTF, or any prior stopped or
  killed method.

