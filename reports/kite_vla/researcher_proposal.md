# KITE-VLA Researcher A Proposal

Date: 2026-07-15 KST

Method: `KITE-VLA`, Kinematic Integration Targets for Execution.

Contribution type: `PRIOR_EXTENSION` plus
`CROSS_DOMAIN_MECHANISM_TRANSFER`.

## Problem

SmolVLA's ordinary conditional flow-matching loss supervises each action
coordinate but does not explicitly require the reconstructed clean action
chunk to realize the demonstrated future end-effector state. Small coherent
per-step errors can therefore accumulate over an open-loop action prefix and
shift approach or placement geometry.

The claim is narrow: on manipulation tasks where cumulative command-to-state
realization is predictable, directly supervising the generated chunk's
multi-horizon realization can improve closed-loop success while preserving the
ordinary inference path.

## Positive Prior Anchor

Closest prior: GeoPredict, https://arxiv.org/abs/2512.16811.

GeoPredict reports consistent improvements on RoboCasa Human-50, LIBERO, and
real-world manipulation using predictive kinematic and geometric priors. Its
trajectory module predicts future 3D robot-arm keypoints from policy
representations. No official code or checkpoint was verified locally, so the
first comparison uses a transparent kinematics-only proxy under the same
SmolVLA scaffold.

StyleVLA, https://arxiv.org/abs/2603.09482, is a cross-domain boundary: it
uses kinematic consistency for driving trajectories. KITE does not claim to
invent physics-informed trajectory supervision. Its proposed manipulation-VLA
difference is the frozen discovery-fitted action-to-state realization operator
and gradient path through the generated clean action chunk.

## Scientific Method

Let a demonstration provide observation `o_t`, raw action chunk
`A_t in R^(50x7)`, and end-effector state `s_t in R^6`.

For horizons `H in {5,20}`, define cumulative arm command

`c_(t,H) = sum_(j=0)^(H-1) A_(t+j,1:6)`

and future state displacement

`d_(t,H) = s_(t+H) - s_t`.

Using discovery rows only, standardize `c` and `d` and fit one ridge affine
realization operator `F_H` with fixed ridge coefficient `1e-4`. Freeze `F_H`
before any validation measurement or policy optimization.

SmolVLA uses

`X_u = u E + (1-u) A`

with velocity target `E-A`. For a predicted velocity `V_theta`, reconstruct
the clean normalized action estimate

`A_hat = X_u - u V_theta(X_u,u,o_t)`.

Apply the checkpoint's fixed differentiable action unnormalization, compute
`c_hat_(t,H)`, and predict

`d_hat_(t,H) = F_H(c_hat_(t,H))`.

The KITE objective is coordinate-mean Huber with delta `1.0` in
discovery-standardized state-displacement units:

`L_kite = mean_(H in {5,20}) Huber(norm(d_hat_(t,H)), norm(d_(t,H)))`.

Total training objective:

`L = L_flow + lambda_k L_kite`.

No KL divergence, reward model, memory, event label, candidate reranking,
action clipping, or inference-time correction is used.

## Low-Compute Parameterization

Use rank-4 zero-effect LoRA on the repository's validated SmolVLA target set.
Every LoRA B matrix initializes to zero. `F_H` is frozen and has no trainable
policy parameters. Mixed precision, batch size one, and accumulation are
implementation details, not contributions.

At inference, remove all training-only target construction and realization
loss code. The processor, solver, action horizon, action tensor, and output
path are exactly Base.

## Data Partitions

Fixed development task families:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Within each fixed HDF5 source:

- discovery/training demonstrations: `0..7`;
- validation demonstrations: `8..9`;
- confirmatory task/reset identities: sealed until one configuration and all
  policies are frozen.

No reward, success, done flag, or confirmatory identity may enter discovery,
operator fitting, headroom, training, or selection.

## Bounded Validation Search

Only `lambda_k in {0.1,0.3,1.0}` may vary: three total KITE configurations.
Rank, target modules, horizons, ridge coefficient, Huber delta, optimizer,
steps, data, and checkpoint-selection score are fixed.

The preregistered validation score combines:

- `40%` normalized multi-horizon realization improvement;
- `30%` offline clean-action reconstruction improvement;
- `20%` Base action retention;
- `10%` action validity.

No configuration with a nonfinite action, disk-reload failure, or clean
retention failure is eligible. Tie break: smaller `lambda_k`.

## First Serious Comparison

Exactly five policy identities:

1. `smolvla_base`;
2. `geopredict_kinematics_proxy`;
3. `kite_full`;
4. `kite_endpoint_only`;
5. `standard_lora`.

The GeoPredict proxy predicts the same future-state targets from the declared
SmolVLA representation with a small training-only head; its gradients shape
the same rank-4 LoRA but do not pass through generated actions. It is labeled a
transparent proxy, not an official reproduction.

The endpoint-only ablation uses only `H=20`. Standard LoRA receives the same
demonstrations, flow objective, optimizer, steps, rank, and target modules but
no kinematic target.

## Falsifiable Mechanism

Problem condition:

`spatially precise or goal-conditioned manipulation`

leads to

`per-step action errors whose cumulative command implies the wrong future
end-effector displacement`

which leads to

`approach, grasp, or placement endpoint drift and closed-loop failure`.

KITE:

`multi-horizon realization loss through generated clean actions`

should produce

`lower generated-action future-state error at H=5 and H=20`

then

`lower arm action error near spatial transitions without destructive global
action change`

and finally

`higher matched closed-loop task success than Base, the GeoPredict proxy, and
endpoint-only KITE`.

## Pre-Experiment Gates

Before adapter training:

1. exact proposal and source hashes;
2. finite aligned actions and `ee_states`;
3. no split overlap or duplicate keys;
4. at least `512` discovery and `96` validation windows per horizon;
5. positive discovery target variance in all six state coordinates;
6. no task contributes more than `40%` of sampled windows;
7. discovery-fitted `F_H` beats the discovery-mean target by at least `50%`
   validation MSE at both horizons;
8. frozen Base realized-state error exceeds demonstrated-action operator
   residual by at least `25%` in median or `0.02` normalized Huber;
9. initialized and disk-reloaded rank-4 LoRA reproduces Base flow and decoded
   actions within `1e-6`;
10. exceptions, nonfinite tensors, and invalid actions are zero.

Label/source failure is `DATA_FAILURE`; no Base deficit is `NO_HEADROOM`;
valid targets that cannot affect the policy are `DESIGN_FAILURE`; execution,
identity, alignment, persistence, or serialization defects are
`IMPLEMENTATION_FAILURE`. None is a scientific kill.

## Paper-Candidate Gate

KITE becomes a serious paper candidate only if frozen SmolVLA comparisons show
that KITE beats Base, the transparent GeoPredict proxy, endpoint-only KITE, and
standard LoRA on the matched claim axis while retaining clean behavior and
showing lower realized-state error. Then, and only then, port the unchanged
scientific method to Quantized OpenVLA-OFT INT4 through a compatible QLoRA
scaffold and add one claim-specific second condition.
