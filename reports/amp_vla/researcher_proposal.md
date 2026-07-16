# AMP-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `AMP_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `AMP-VLA`, Action-Manifold Projection for VLA action-flow adaptation.

Contribution type: `PRIOR_EXTENSION`.

This proposal starts Epoch 4 Cycle 26 after the fixed RAP Stage 0
implementation/optimization failure. It does not repair RAP, change RAP
thresholds, clip RAP actions, reinterpret RAP action validity, reuse RAP
partial results as success evidence, or relabel RAP as a scientific kill.

## External Prior Anchor

Closest external prior: ABot-M0, https://arxiv.org/abs/2602.11236.

Official code/assets: https://github.com/amap-cvlab/ABot-Manipulation. The
repository reports released inference code, weights, training code, and data
for ABot-M0, plus ABot-M0.5 technical updates.

Positive prior result: ABot-M0 reports Action Manifold Learning for stable
continuous robot action prediction. ABot-M0.5 further reports strong
LIBERO-family and multi-benchmark manipulation results from temporal-granularity
alignment, action-space alignment, and inference consistency.

Secondary priors:

- PriorVLA, https://arxiv.org/abs/2605.10925
- InternVLA-M1, https://arxiv.org/abs/2510.13778
- Robometer, https://arxiv.org/abs/2603.02115

## Claim

If local SmolVLA adaptation failures are dominated by off-support or invalid
postprocessed action chunks, then constraining adapter-induced action changes
through a learned demonstration action manifold can improve closed-loop
manipulation success while preserving normal SmolVLA behavior.

The claim is intentionally narrow. AMP is not generic LoRA fine-tuning, not
nearest-demonstration replay, not a retrieval memory prior, not a spatial
waypoint method, not a reward model, and not an official ABot-M0 reproduction
unless official assets are installed and verified locally. LoRA is only
implementation infrastructure for an identity-preserving residual/gate path.

## Evidence Partitions

`DISCOVERY`:

- fit action-manifold statistics from training demonstrations only;
- inspect action support, action normalization, chunk phase structure, task
  coverage, and manifold dimension health;
- build a transparent ABot-M0-style local action-manifold projection proxy;
- debug serializer, projection, residual, and gate construction.

`VALIDATION`:

- select one manifold/residual configuration from the bounded search;
- verify projection headroom, residual predictability, clean retention,
  action validity, Base passthrough, reload, and gradient flow;
- select one final configuration using only the frozen validation score.

`CONFIRMATORY_TEST`:

- one frozen paired official LIBERO manifest after method, configuration,
  policies, ablation, task/reset identities, metrics, and thresholds are
  frozen;
- confirmatory outcomes cannot retune AMP, manifold construction, latent
  dimension, projection strength, coefficients, thresholds, or baselines.

## Scientific Method

For a legal demonstration timestep `t`, let:

- `o_t`: current legal RGB observations, proprioception, and language;
- `A_t in R^(50x7)`: normalized expert action chunk;
- `B_t in R^(50x7)`: frozen Base SmolVLA action chunk decoded from `o_t`;
- `x_t`: deployment-observable feature built from frozen SmolVLA visual
  tokens, proprioception, task/language identity, and phase;
- `Phi(A_t) in R^m`: fitted action-manifold coordinates for the legal expert
  chunk;
- `P(A)`: frozen projection from a candidate action chunk to the nearest local
  point on the discovered action manifold under normalized action Huber
  distance;
- `z_theta(o_t) in R^m`: predicted manifold coordinate;
- `Delta_theta(o_t) in R^(50x7)`: zero-initialized residual direction;
- `g_theta(o_t) in [0, g_max]`: zero-initialized residual gate.

Discovery rows fit the action-manifold transform only from legal expert action
chunks, optional phase/task identifiers, and legal current proprioception
summary statistics. Rewards, success flags, done flags, reset identities,
future simulator state, object poses, and confirmatory outcomes are forbidden.

AMP predicts a local manifold action:

`M_theta(o_t) = DecodeManifold(z_theta(o_t))`

and a bounded residual candidate:

`C_theta(o_t) = B_t + g_theta(o_t) * Delta_theta(o_t)`.

The final training-time action target path is projected back to demonstrated
support:

`A_hat_t = P_mix(C_theta(o_t), M_theta(o_t), alpha)`

where `P_mix` is a preregistered convex manifold-consistency operator. The
initialized gate is zero and the initialized projection weight is Base
passthrough, so the initialized policy is exactly Base.

The training objective is:

`L = L_flow
   + lambda_m * Huber(z_theta(o_t), Phi(A_t))
   + lambda_p * Huber(P_mix(C_theta(o_t), M_theta(o_t), alpha), A_t)
   + lambda_clean * Huber(A_hat_t, B_t)_clean`

All Huber terms are coordinate means in normalized action units after scale
audits. The projection term is not KL divergence; deterministic 7D actions and
SmolVLA flow vectors are not treated as probability distributions.

At inference, AMP uses only current legal observations, current proprioception,
language/task input, frozen Base features/actions, and the frozen
demonstration-derived manifold parameters. No privileged state or future
observation is required.

## Closest Prior And Controls

The transparent ABot-M0 proxy, `abot_m0_action_manifold_proxy`, uses the same
discovery-fitted local action manifold and projects Base or standard adapter
action chunks onto that manifold without AMP's learned identity-preserving
residual/gate. It is a faithful transparent local proxy, not an official
ABot-M0 reproduction unless official ABot assets are integrated and verified.

The key ablation, `amp_no_manifold_projection`, uses the same adapter, residual
gate, optimizer, and clean retention but removes the projection/support
constraint. This tests whether the action manifold itself is necessary.

Matched `standard_lora` receives the same demonstrations, optimizer budget,
rank, target modules, clean-retention policy, and ordinary flow objective, but
no action-manifold coordinate prediction or projection. This is the single
strongest simple reviewer-killer baseline because AMP updates policy behavior
through low-compute adapter infrastructure.

## Low-Compute Parameterization

Use the repository's validated low-compute SmolVLA adapter path with
identity-preserving initialization. Rank-4 LoRA or an equivalent zero-effect
adapter may implement `z_theta`, `Delta_theta`, and `g_theta`, but the
scientific contribution is action-manifold projection and support-preserving
adaptation.

Every trainable residual branch initializes to zero. Every learned gate
initializes to Base passthrough. Projection strength is bounded by a frozen
maximum residual norm and by postprocessed 7D action-validity checks.

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
observation, or confirmatory outcome may enter manifold fitting, target
construction, validation selection, or training.

## Pre-Experiment Gates

Before training:

1. proposal and source hashes match;
2. action, proprioception, image-feature, language/task, phase, and timestamp
   records are finite and aligned;
3. duplicate, missing, extra, and split-overlap keys are zero;
4. at least `512` discovery and `128` validation windows are available;
5. every task has validation rows and no task contributes more than `40%` of
   the audit subset;
6. action manifold coordinates have positive variance in every retained
   dimension;
7. reconstruction from manifold coordinates beats a task/phase mean action
   predictor by at least `10%` validation Huber or `0.01` absolute normalized
   Huber;
8. the transparent ABot-M0 proxy leaves enough residual headroom for AMP to
   improve by at least `5%` relative Huber or `0.005` absolute normalized
   Huber;
9. a deployment-input manifold-coordinate probe beats a trivial task/phase
   coordinate predictor by at least `5%` relative Huber or `0.005` absolute
   normalized Huber;
10. initialized and disk-reloaded adapter reproduces Base flow and
    postprocessed actions within `1e-6`;
11. AMP differs from Base and from `amp_no_manifold_projection` after a small
    training smoke, but the difference is bounded rather than global;
12. postprocessed action validity is preserved before rollout;
13. expected AMP parameters receive finite nonzero gradients and frozen Base
    parameters do not update;
14. exceptions are zero.

Failure classes:

- source, overlap, collapsed action manifold, or collapsed coordinate target:
  `AMP_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- Base or ABot proxy has no usable failure/headroom:
  `AMP_STAGE_0_NO_USABLE_HEADROOM`;
- manifold coordinates are not predictable from deployment inputs or AMP is
  equivalent to the no-projection ablation: `AMP_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, persistence, gradient, projection, or
  action-validity defect:
  `AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

None of these Stage 0 stops is a closed-loop scientific kill.

## Bounded Validation Search

At most six trained development configurations:

1. AMP `latent_dim=8`, `lambda_p=0.3`, `g_max=0.20`;
2. AMP `latent_dim=16`, `lambda_p=0.3`, `g_max=0.20`;
3. AMP `latent_dim=16`, `lambda_p=1.0`, `g_max=0.20`;
4. transparent ABot-M0 action-manifold proxy;
5. `amp_no_manifold_projection`;
6. matched standard LoRA.

The manifold family, feature definition, projection metric, task sources,
memory split, adapter rank, optimizer, steps, and checkpoint-selection rule are
fixed before validation search. One seed per configuration unless a fixed run
is genuinely unresolved; no more than two seeds may then be used before final
selection.

Validation score for selecting the AMP configuration:

`S = 0.30 * validation_success_or_proxy
   + 0.25 * AMP_minus_ABot_proxy_margin
   + 0.20 * clean_action_retention
   + 0.15 * postprocessed_action_validity
   + 0.10 * projection_overhead_score`.

If closed-loop validation is not feasible, `validation_success_or_proxy` must be
replaced before execution by one frozen deployment-observable proxy. Offline
action L2 alone may not select the configuration. Tie break: lower latent
dimension, then smaller projection coefficient.

## First Serious Comparison

Exactly five policy identities:

1. `smolvla_base`;
2. `abot_m0_action_manifold_proxy`;
3. `amp_full`;
4. `amp_no_manifold_projection`;
5. `standard_lora`.

| Comparison | Scientific question |
| --- | --- |
| Base vs AMP | Does support-preserving action-manifold adaptation improve SmolVLA? |
| ABot proxy vs AMP | Does learned identity-preserving residualization beat projection alone? |
| No-projection ablation vs AMP | Is the action-manifold constraint necessary? |
| Standard LoRA vs AMP | Is any gain explained by ordinary data-matched adaptation? |

## Paper-Candidate Gate

AMP becomes a serious paper candidate only if frozen SmolVLA comparisons show
that AMP beats Base, the ABot-M0 proxy, the no-projection ablation, and standard
LoRA while retaining clean behavior, preserving postprocessed action validity,
and showing that manifold projection is active in relevant states rather than
everywhere.

Then verify the unchanged scientific method on Quantized OpenVLA-OFT INT4 and
add one claim-specific second condition or benchmark.

## Non-Claims

- AMP is not official ABot-M0 unless official assets are installed and verified.
- AMP is not nearest-neighbor action replay.
- AMP is not a generic low-rank adaptation method.
- AMP is not RAP, VDR, KITE, HEST, HASTE, IARC, FAMR, PCAV, SPARC, NICE,
  COVI, LIFT, or EAC rescue.
- AMP is not a KL or probabilistic action-distribution method.
