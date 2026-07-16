# CCIF-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `CCIF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `CCIF-VLA`, Continuous Coarse Intent Field for base-preserving VLA
chunks.

Contribution type: `PRIOR_EXTENSION`.

This proposal starts Epoch 4 Cycle 29 after TSC-VLA is closed as
`TSC_STAGE_0_NO_USABLE_HEADROOM`. It does not repair TSC, change TSC
thresholds, change the TS-Mask proxy, retune TSC masks, reinterpret TSC partial
results, or relabel TSC as a closed-loop scientific kill.

## External Prior Anchor

Closest external prior: Coarse-to-Control,
https://arxiv.org/abs/2606.07107.

Official code/assets: official local code or checkpoints were not identified
during Cycle 29 candidate selection. Until such assets are installed and
verified, policy 2 is a transparent local proxy named
`coarse_to_control_continuous_proxy`, not an official Coarse-to-Control
reproduction.

Positive prior result: Coarse-to-Control reports action-token planning before
executable action-token generation, with a shared residual-VQ action vocabulary
for plan and execute tokens. Its arXiv report states `97.90%` average LIBERO
success and strong gains on long-horizon and real-world manipulation tasks.

Secondary priors:

- CAC-VLA, https://arxiv.org/abs/2607.04816
- CF-VLA, https://arxiv.org/html/2604.24622v1
- SUREFlow, https://arxiv.org/abs/2607.10504
- CoRE-VLA, https://arxiv.org/abs/2607.03693

## Claim

If some SmolVLA failures arise because the decoded continuous action chunk is
locally plausible but globally inconsistent with the intended coarse motor
direction, then a deployment-observable continuous coarse intent field can
improve Base-preserving residual action adaptation while retaining clean
behavior.

The claim is intentionally narrow. CCIF is not generic LoRA fine-tuning, not a
discrete action-token decoder, not an official Coarse-to-Control reproduction,
not task/instruction adapter routing, not uncertainty-only residual gating, and
not temporal-spatial masked completion. LoRA is only implementation
infrastructure for an identity-preserving coarse-intent and residual path.

## Evidence Partitions

`DISCOVERY`:

- derive coarse motor-intent labels from legal training demonstrations only;
- inspect Base chunk residuals, net displacement, net rotation, gripper
  endpoint, low-frequency waypoints, task/phase coverage, and action validity;
- fit the transparent Coarse-to-Control continuous proxy;
- debug serializer, identity initialization, reload, residual bounding, and
  gradient paths.

`VALIDATION`:

- select one CCIF configuration from the bounded validation search;
- compare CCIF against the closest-prior proxy, no-coarse-intent ablation, and
  standard LoRA using validation data or a frozen validation rollout/proxy;
- verify clean retention, intent activation, action validity, bounded deltas,
  reload, and no privileged inference inputs.

`CONFIRMATORY_TEST`:

- one frozen paired official LIBERO manifest after method, configuration,
  policy list, ablation, task/reset identities, metrics, and thresholds are
  frozen;
- confirmatory outcomes cannot retune CCIF, intent labels, residual caps,
  loss weights, thresholds, tasks, reset identities, or baselines.

## Scientific Method

For a legal demonstration timestep `t`, let:

- `o_t`: current legal RGB observations, proprioception, and language;
- `A_t in R^(50x7)`: normalized expert action chunk;
- `B_t in R^(50x7)`: frozen Base SmolVLA action chunk decoded from `o_t`;
- `x_t`: deployment-observable feature built from frozen SmolVLA visual tokens,
  proprioception, language/task identity, phase proxies, and the Base chunk;
- `c_t in R^m`: coarse motor-intent field derived from `A_t`;
- `c_hat_theta(x_t) in R^m`: predicted coarse motor intent;
- `Delta_phi(x_t, c_hat_theta, B_t) in R^(50x7)`: zero-initialized residual
  field;
- `g_phi(x_t, c_hat_theta, B_t) in [0, g_max]`: zero-initialized intent gate.

The coarse intent vector must be derivable from legal action chunks without
privileged inference inputs. The initial frozen field is:

`c_t = [mean_delta_xyz, terminal_delta_xyz, mean_delta_rpy,
terminal_delta_rpy, terminal_gripper, waypoint_summary]`.

The exact dimension `m`, waypoint count, normalization, and units must be
frozen in the mathematical mechanism audit before implementation.

CCIF predicts the coarse intent and applies a bounded residual:

`A_CCIF = B_t + g_phi(x_t, c_hat_theta, B_t) * ProjectIntent(Delta_phi, c_hat_theta)`

where `ProjectIntent` keeps residual directions aligned with the predicted
coarse intent under a preregistered continuous projection or penalty. The
initialized gate and residual are zero, so initialized CCIF is exactly Base.

Training objective:

`L = L_flow
   + lambda_c * Huber(c_hat_theta(x_t), c_t)
   + lambda_a * Huber(A_CCIF, A_t)
   + lambda_align * IntentAlign(Delta_phi, c_hat_theta)
   + lambda_clean * Huber(A_CCIF, B_t)_clean`

All terms are coordinate means in normalized action units or normalized intent
units after a small-batch magnitude and gradient-norm audit. No KL divergence
is used between deterministic 7D actions.

At inference, CCIF uses only current legal observations, proprioception,
language/task input, frozen Base features/actions, and learned parameters.
Rewards, success flags, done flags, object poses, future observations, reset
identities, and confirmatory outcomes are forbidden.

## Closest Prior And Controls

The transparent prior proxy, `coarse_to_control_continuous_proxy`, predicts the
same coarse action-space intent from deployment-observable inputs and then
generates an action chunk directly from that intent without CCIF's
Base-preserving residual clamp. It tests whether ordinary coarse-to-control
conditioning explains any gain.

The key ablation, `ccif_no_coarse_intent_ablation`, uses the same adapter,
optimizer, residual cap, clean-retention policy, and action loss, but removes
the predicted coarse intent from the residual path. This tests whether the
continuous coarse intent field is necessary.

Matched `standard_lora` receives the same demonstrations, optimizer budget,
rank, target modules, and clean-retention policy, but no coarse-intent target,
intent-alignment objective, or intent-conditioned residual field.

## Low-Compute Parameterization

Use the repository's validated low-compute SmolVLA adapter path with
identity-preserving initialization. Rank-4 LoRA or an equivalent zero-effect
adapter may parameterize `c_hat_theta`, `Delta_phi`, and `g_phi`, but the
scientific contribution is continuous coarse intent field conditioning.

Every residual branch initializes to zero. Every learned gate initializes to
Base passthrough. The initialized and disk-reloaded policy must reproduce Base
flow and postprocessed actions within `1e-6`. Residual magnitude is bounded by
a frozen residual cap, a frozen `g_max`, and official postprocessed 7D
action-validity checks.

## Fixed Development Sources

Use the same fixed development task families as recent SmolVLA development
cycles for continuity:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Within each source:

- discovery/training demonstrations: `0..7`;
- validation demonstrations: `8..9`;
- confirmatory task/reset identities: untouched until one configuration and all
  policies are frozen.

No reward, success, done flag, reset identity, simulator object pose, future
observation, or confirmatory outcome may enter label construction, validation
selection, or training.

## Pre-Experiment Gates

Before expensive training or rollout:

1. proposal and source hashes match;
2. action, Base chunk, proprioception, image-feature, language/task, phase, and
   timestamp records are finite and aligned;
3. duplicate, missing, extra, and split-overlap keys are zero;
4. at least `512` discovery and `128` validation windows are available;
5. every task has validation rows and no task contributes more than `40%` of
   the audit subset;
6. coarse-intent labels have nonzero variance in every retained component;
7. a deployment-input intent probe beats task/phase mean intent by at least
   `5%` relative Huber or `0.005` absolute normalized Huber;
8. Base-to-expert residual chunks have positive variance in every action
   dimension after valid-step masking;
9. the Coarse-to-Control continuous proxy leaves residual headroom for CCIF of
   at least `5%` relative Huber or `0.005` absolute normalized Huber;
10. intent-conditioned residual prediction improves validation proxy loss over
    `ccif_no_coarse_intent_ablation` before any rollout;
11. initialized and disk-reloaded adapter reproduces Base flow and
    postprocessed actions within `1e-6`;
12. CCIF differs from Base, prior proxy, and no-intent ablation after a small
    training smoke, but the difference is bounded rather than global;
13. postprocessed action validity is preserved before rollout;
14. expected CCIF parameters receive finite nonzero gradients and frozen Base
    parameters do not update;
15. exceptions are zero.

Failure classes:

- source, overlap, collapsed coarse-intent labels, or collapsed residual
  targets: `CCIF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- Base or the closest-prior proxy leaves no usable residual headroom:
  `CCIF_STAGE_0_NO_USABLE_HEADROOM`;
- coarse intent is not predictable from deployment inputs or the intent
  mechanism is equivalent to the no-intent ablation:
  `CCIF_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, persistence, gradient, residual-bound, or
  action-validity defect:
  `CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- all gates pass:
  `CCIF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

None of these Stage 0 stops is a closed-loop scientific kill.

## Bounded Validation Search

At most six trained development configurations:

1. CCIF `waypoints=2`, `lambda_c=0.3`, `g_max=0.10`;
2. CCIF `waypoints=4`, `lambda_c=0.3`, `g_max=0.10`;
3. CCIF `waypoints=4`, `lambda_c=1.0`, `g_max=0.10`;
4. `coarse_to_control_continuous_proxy`;
5. `ccif_no_coarse_intent_ablation`;
6. matched `standard_lora`.

The feature definition, task sources, split, intent target, adapter rank,
optimizer, step budget, residual cap, action-validity check, and checkpoint
selection rule are fixed before validation search. One seed per configuration
unless a fixed run is genuinely unresolved; no more than two seeds may then be
used before final selection.

Validation score for selecting the CCIF configuration:

`S = 0.30 * validation_success_or_proxy
   + 0.25 * CCIF_minus_prior_proxy_margin
   + 0.20 * clean_action_retention
   + 0.15 * postprocessed_action_validity
   + 0.10 * intent_overhead_score`.

If closed-loop validation is not feasible, `validation_success_or_proxy` must be
replaced before execution by one frozen deployment-observable proxy. Offline
action L2 alone may not select the configuration. Tie break: fewer waypoints,
then lower residual cap.

## First Serious Comparison

Exactly five policy identities:

1. `smolvla_base`;
2. `coarse_to_control_continuous_proxy`;
3. `ccif_full`;
4. `ccif_no_coarse_intent_ablation`;
5. `standard_lora`.

| Comparison | Scientific question |
| --- | --- |
| Base vs CCIF | Does continuous coarse intent conditioning improve SmolVLA? |
| Prior proxy vs CCIF | Does Base-preserving residualization beat direct coarse-to-action conditioning? |
| No-intent ablation vs CCIF | Is the coarse intent field necessary? |
| Standard LoRA vs CCIF | Is any gain explained by ordinary data-matched adaptation? |

## Paper-Candidate Gate

CCIF becomes a serious paper candidate only if frozen SmolVLA comparisons show
that CCIF beats Base, the Coarse-to-Control proxy or official Coarse-to-Control
if installed, the no-intent ablation, and standard LoRA while retaining clean
behavior, preserving postprocessed action validity, and showing that intent
activation is relevant-state selective rather than global.

Then verify the unchanged scientific method on Quantized OpenVLA-OFT INT4 and
add one claim-specific second condition or benchmark.

## Non-Claims

- CCIF is not official Coarse-to-Control unless official assets are installed
  and verified.
- CCIF is not a discrete action-token tokenizer or autoregressive action-token
  decoder.
- CCIF is not task/instruction adapter routing, TS-Mask, CFR, AMP, RAP, VDR,
  KITE, HASTE, HEST, IARC, FAMR, PCAV, SPARC, NICE, COVI, LIFT, or EAC rescue.
- CCIF is not generic LoRA, QLoRA, PEFT, or adaptation-efficiency work.
- CCIF is not a KL or probabilistic action-distribution method.
