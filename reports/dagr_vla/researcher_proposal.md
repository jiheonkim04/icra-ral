# DAGR-VLA Researcher A Proposal

Date: 2026-07-14 KST

Method: `DAGR-VLA`, Dynamic Arm-Gripper Routing for frozen SmolVLA adaptation.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: DAM-VLA, https://arxiv.org/abs/2603.00926.

## Claim

DAGR-VLA tests whether a lightweight, identity-preserving arm/gripper action router can improve frozen SmolVLA closed-loop manipulation success beyond Base, a DAM-style static component proxy, a shared-residual ablation, and a simple gripper-transition heuristic.

The method is not a rescue of MTF-VLA. It does not change retained-frame ratio, does not tune `mtf_r20_ret100`, does not alter the MTF task/reset identities, and does not use MTF Stage B outcomes to change MTF. It starts a new method cycle with a different prior, representation, supervision, objective, and action-generation mechanism.

## Positive Prior Anchor

DAM-VLA reports that dynamic action routing, specialized arm/gripper action models, and dual-scale action weighting improve complex manipulation performance in simulated and real-world settings. The positive prior suggests that action components should not be treated as a homogeneous 7D vector when their timing and difficulty differ.

The local extension keeps SmolVLA frozen and adds only a small group route/residual module. The goal is not to reproduce the official DAM-VLA architecture. The fair local prior proxy is a static component-weighted arm/gripper adapter trained under the same data, backbone, and inference budget.

## Falsifiable Mechanism

Problem condition:

- SmolVLA emits 7D action chunks where translation, rotation, and gripper dimensions are optimized together.
- Manipulation failures often depend on only one action group, such as gripper timing or wrist/arm approach alignment.

Intermediate failure mechanism:

- A shared adapter or ordinary residual spreads capacity across all action dimensions.
- Small global action changes can miss the group-specific correction needed for contact, grasp, or release.

Policy behavior:

- The base action remains strong on many states but needs bounded corrections on specific action groups.
- Global correction risks disrupting clean behavior.

Closed-loop failure:

- The robot misses grasp/release timing, approaches with the wrong component, or changes gripper state at the wrong moment.

Proposed method:

- Infer group route logits from deployment-observable inputs.
- Apply clipped residuals separately to translation, rotation, and gripper groups.
- Initialize to exact base passthrough.

Intended internal change:

- The router activates sparsely and group-specifically.
- Residual heads specialize to different action-group errors.

Expected action behavior:

- Base-like action when no route is active.
- Bounded group-specific corrections when the route is active.

Expected closed-loop improvement:

- Better success on grasp/release and approach-sensitive episodes while retaining clean base behavior.

## Data And Supervision

Discovery and validation data may use existing official SmolVLA stable prediction artifacts and development split records. Confirmatory identities must remain held out until the Stage A/B manifests are frozen.

Required labels:

- base 7D action chunk from frozen SmolVLA;
- expert 7D action;
- 7D residual target `expert_action - base_action`;
- action-group masks for translation `[0:3]`, rotation `[3:6]`, and gripper `[6]`;
- route labels derived from group residual magnitude, action variance, and gripper sign or threshold transitions;
- task key, frame key, chunk phase, and split identity for overlap checks.

Stage 0 must reject before rollout if:

- any route label collapses to all-zero or all-one;
- any action group lacks sufficient positive and negative examples;
- route labels are not predictable above a trivial majority baseline from deployment inputs;
- full and ablation receive effectively identical targets;
- train/validation/test identity overlap is nonzero;
- initial residual action delta is not near zero;
- validation action deltas are globally destructive.

## Method Sketch

Let `a_base in R^7` be the frozen SmolVLA current action, `a_exp in R^7` the expert action, and `r = a_exp - a_base`.

Action groups:

- translation: dimensions `0:3`;
- rotation: dimensions `3:6`;
- gripper: dimension `6`.

A small router predicts `g = sigmoid(h(o_t, s_t, lang, phase)) in [0,1]^3`, one route gate per group. A residual module predicts `delta in R^7`, decomposed by group. The emitted action is:

`a_ours = clip_action(a_base + clip_group(g_trans * delta_trans, alpha_trans) + clip_group(g_rot * delta_rot, alpha_rot) + clip_group(g_grip * delta_grip, alpha_grip))`

All residual projections are initialized to zero, so the initial policy equals Base.

Training loss candidates:

- group-normalized Huber residual loss on `r`;
- route-label binary cross entropy or focal loss;
- clean action-delta regularizer;
- optional activation sparsity penalty selected only by validation.

No KL is computed between deterministic 7D action vectors.

## First Serious Comparison

Exactly five policies:

1. `frozen_smolvla`
2. `dam_static_component_proxy`
3. `dagr_full`
4. `dagr_no_dynamic_route_ablation`
5. `gripper_transition_heuristic`

`dam_static_component_proxy` is a faithful local proxy, not an official DAM-VLA reproduction. It preserves the key static arm/gripper component weighting idea but omits the official dynamic action architecture.

## Bounded Development Search

Default budget:

- at most `6` total configurations;
- at most `2` route architectures;
- at most `3` values for one residual alpha or route threshold;
- at most `2` lightweight seeds only if training is cheap enough;
- no combinatorial grid;
- no confirmatory-test use.

Validation score should combine:

- route-label health and route predictability;
- full-versus-ablation action difference;
- clean action retention;
- action validity;
- bounded route activation;
- validation closed-loop proxy when feasible;
- compute and latency.

## Required Ablations And Baselines

Key ablation:

- no-dynamic-route shared residual ablation. It receives the same residual targets and parameter budget where feasible, but no group-specific route gates.

Closest-prior proxy:

- static component-weighted adapter with arm/gripper loss weights but no dynamic route.

Simple killer:

- gripper-transition heuristic that only applies a simple bounded gripper timing bias near predicted gripper transitions.

## Stop Rules

Classify failures before rollout as:

- `DATA_OR_SUPERVISION_FAILURE` if route labels collapse, coverage fails, or labels are not observable;
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` if gradients, checkpoint reload, or target wiring fails;
- `DESIGN_FAILURE` if full and ablation are indistinguishable or the residual acts everywhere;
- `CONDITION_TOO_SEVERE_OR_NO_HEADROOM` if Base, prior proxy, and diagnostic upper bounds show no usable headroom.

Only a valid closed-loop Stage A/B result with active mechanism and fair baselines can kill the scientific current formulation.
