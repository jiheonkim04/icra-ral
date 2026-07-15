# VDR-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `VDR_MATHEMATICAL_AUDIT_PREREGISTERED`

Proposal hash:
`0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72`

## Variables And Shapes

- `o_t`: legal current observation dictionary containing RGB streams,
  proprioception, and language.
- `A_t`: expert normalized action chunk, shape `[50, 7]`, training target for
  ordinary flow only.
- `X_u`: noisy flow action chunk, shape `[50, 7]`.
- `V_theta(X_u,u,o_t)`: SmolVLA velocity prediction, shape `[50, 7]`.
- `Ahat_t = X_u - u V_theta(...)`: reconstructed clean generated action
  estimate, shape `[50, 7]`.
- `e_t = E(o_t)`: frozen current visual feature, shape `[960]`.
- `e_(t+H)`: frozen future visual feature, shape `[960]`, training target only.
- `p_t`: legal proprioceptive feature, shape `[d_p]`.
- `z_t`: legal task/language/phase feature, shape `[d_z]`.
- `P_K`: discovery-fitted PCA/whitening projection, `R^960 -> R^32`.
- `B_H(e_t,p_t,z_t)`: frozen actionless static predictor, shape `[32]`.
- `r_(t,H)`: dynamic residual target, shape `[32]`.
- `s(Ahat_t,H)`: generated-action summary for horizon `H`, shape `[m]`.
- `D_theta(h_t,s(Ahat_t,H),e_t,p_t,z_t)`: VDR residual head, shape `[32]`.

Frozen horizons: `H in {4,12}`.

## Target Construction

Feature change:

`y_(t,H) = P_K(e_(t+H) - e_t)`.

Discovery-only static prediction:

`b_(t,H) = B_H(e_t,p_t,z_t)`.

Dynamic residual target:

`r_(t,H) = y_(t,H) - b_(t,H)`.

The static predictor never consumes generated actions, future actions, reward,
success, reset identity, simulator object state, or confirmatory-test
identities.

## Objective

Primary VDR objective:

`L_vdr = mean_(H in {4,12}) mean_i Huber_delta=1(rhat_(t,H,i) - r_(t,H,i))`.

Total objective:

`L = L_flow + lambda_v L_vdr`.

Allowed `lambda_v`: `{0.1,0.3,1.0}`.

Units:

- `L_flow`: normalized action velocity units from the existing SmolVLA flow
  objective.
- `L_vdr`: whitened frozen visual-feature residual units.

Before full training, the runner must report:

- `L_flow` magnitude on a small batch;
- `L_vdr` magnitude on the same batch;
- gradient norm into VDR trainable parameters;
- gradient norm into LoRA/adapter parameters;
- zero gradient into frozen Base parameters;
- `L_vdr / L_flow` scale and any clipping or normalization justification.

## Gradient Path

The intended VDR gradient path is:

`L_vdr -> rhat -> D_theta -> h_t and s(Ahat_t,H) -> Ahat_t -> V_theta -> trainable adapter/LoRA parameters`.

Frozen paths:

- `E`;
- `P_K`;
- `B_H`;
- Base parameters not selected by the adapter;
- target features and discovery statistics.

## Mechanism Consequence

Expected internal change:

`generated action chunks encode visual dynamics not explained by static scene,
task, phase, or proprioception`.

Expected action behavior:

`bounded changes in arm dimensions in states where generated actions explain
the future-feature residual, with no global action disruption and no gripper
drift`.

Expected closed-loop consequence:

`higher success on matched manipulation tasks than Base, the FutureVLA proxy,
the no-action-residual ablation, and standard LoRA`.

## Closest Mathematical Alternatives

- full future-feature alignment without static subtraction;
- actionless static future-feature prediction;
- ordinary action L2 or flow loss only;
- KITE-style action-to-end-effector realization;
- COVI-style complementary feature reconstruction;
- deterministic-action KL, which is explicitly forbidden.

VDR uses Huber residual alignment because both arguments are real-valued
feature residual vectors with defined scale. KL is not valid here because no
normalized probability distributions are defined.

## Required Ablation

`vdr_no_action_residual` blocks generated-action information from the residual
head while keeping the same target, projection, static predictor, data, and
training budget as matched as possible. If this ablation matches or beats VDR,
the generated-action mechanism is not supported.

## Identity-Preserving Integration Audit

Before rollout:

- initialized adapter action delta p95 must be at most `1e-6`;
- disk-reloaded adapter action delta p95 must be at most `1e-6`;
- Base parameter hash must be unchanged;
- action validity must be `1.0`;
- translation, rotation, and gripper deltas must be separately reported after
  training;
- VDR activation must be context-dependent rather than global.

## Stage 0 Classification

- collapsed features, insufficient rows, source leakage, duplicate keys, or
  split overlap: `DATA_OR_SUPERVISION_FAILURE`;
- no Base or prior residual headroom: `NO_USABLE_HEADROOM`;
- action-conditioned residual probe fails over actionless probe:
  `DESIGN_FAILURE`;
- nonfinite gradients, wrong checkpoint, serialization defect, identity
  failure, disk reload failure, or invalid actions:
  `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

None is a scientific kill before closed-loop confirmatory evidence.
