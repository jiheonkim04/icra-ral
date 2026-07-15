# Epoch 4 Cycle 21 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_HEST_VLA`

Exactly three candidates were generated and scored under the active
performance-oriented, partitioned-evidence, false-negative-safeguard, and
post-COVI LoRA-role governance. NICE-VLA remains closed and unchanged.

## Candidate 1: HEST-VLA

Name: `HEST-VLA`, Hybrid Event-Spline Trajectories for VLA policies.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Spline Policy,
https://arxiv.org/abs/2606.07386.

### Scientific Method

Given a Base action chunk `A in R^(H x 7)`, HEST separates:

- six continuous arm increments `A_arm in R^(H x 6)`;
- one discontinuous gripper stream `A_grip in R^H`.

It integrates the arm increments into a cumulative path, fits a regularized
spline while constraining the first point and cumulative endpoint exactly,
differences the path back into `H` arm increments, and blends it with Base by a
validation-frozen magnitude. The gripper stream is copied bit-for-bit. If any
finite, support, shape, or endpoint check fails, HEST returns the entire Base
chunk exactly; it never clips individual coordinates.

This is a hybrid action representation, not another learned action code. Its
technical difference from Spline Policy is the explicit continuous/discrete
factorization and exact event preservation for a 7D manipulation interface.

Key ablation and closest-prior proxy: apply the same spline construction to all
seven coordinates, including gripper. Strongest simple reviewer-killer: a
three-tap moving average on arm increments with the original gripper stream.

### Low-Compute Parameterization

HEST is an inference-time analytic interface around a frozen SmolVLA. It has no
LoRA, QLoRA, learned head, or extra checkpoint. Standard LoRA is omitted because
generic weight adaptation does not test whether a hybrid continuous/discrete
trajectory representation preserves controller behavior better than an
all-channel spline or simple smoothing.

### Quality Screen

Provisional novelty:

- Spline Policy exposes continuous trajectory structure but does not make the
  six-arm/one-event distinction the method;
- HEST introduces a hybrid representation, exact endpoint constraints, exact
  event preservation, and whole-chunk Base fallback;
- the contribution fails if it is equivalent to moving-average smoothing or
  the all-channel spline proxy.

Prior-anchor strength:

- Spline Policy is a direct positive structured-action prior spanning
  flow-matching and VLA-style backbones;
- no official code was verified, so the local comparator is a transparent
  analytic proxy and prior fidelity is lower than for an official checkpoint.

Mechanism plausibility:

- fixed action chunks mix smooth arm motion and abrupt gripper events;
- one homogeneous smoother either leaves arm jerk or shifts gripper timing;
- shifted mode events or changed cumulative displacement cause grasp/release
  and terminal-pose errors in closed loop;
- HEST regularizes only cumulative arm motion, preserves cumulative arm
  endpoint and gripper events exactly, and rejects invalid transformations;
- intended action effect: lower arm jerk without event-time or endpoint drift;
- expected closed-loop effect: better controller fidelity and task success
  under a preregistered action-interface condition while retaining clean Base.

Data and supervision viability:

- official LIBERO demonstration action chunks provide all required development
  inputs without labels beyond the recorded 7D actions;
- exact simulator states permit direct development-only replay/control
  comparison against the original chunk;
- no future action, simulator state, reward, or task outcome is used at
  confirmatory inference.

Identity-preserving integration:

- blend magnitude zero is exactly Base;
- the whole-chunk fallback is exactly Base;
- horizon, action order, task, reset, observation schedule, and queue semantics
  do not change;
- no coordinate clipping is permitted.

Decisive experiment feasibility:

- a CPU-only algebra/source gate can test endpoints, event identity, action
  validity, nonacting behavior, and simple-baseline equivalence;
- a bounded exact-state replay can measure controller-state deviation before
  any VLA training or confirmatory rollout;
- at most three blend magnitudes form the only validation search.

Score:

- provisional novelty: `23 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `17 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `10 / 10`
- decisive experiment feasibility: `10 / 10`
- total: `93 / 100`

## Candidate 2: TASF-VLA

Name: `TASF-VLA`, Task-Anisotropic Set Flow for VLA adaptation.

Contribution type: `CROSS_DOMAIN_MECHANISM_TRANSFER`.

Closest external prior: Set-Supervised Diffusion Policy,
https://arxiv.org/abs/2606.01865.

### Scientific Method

Replace single-action flow targets with task-scaled ellipsoidal desired sets
estimated from discovery demonstrations, then train the SmolVLA adapter toward
the nearest valid set member. The closest-prior proxy uses SDP-style isotropic
sets; the key ablation is ordinary single-action flow matching.

### Low-Compute Parameterization

One fixed rank-4 SmolVLA LoRA scaffold would be shared by the prior proxy, Ours,
and ordinary adaptation. Standard LoRA is required because generic adaptation
with the same demonstrations is a plausible explanation.

### Quality Screen

- provisional novelty: anisotropic task-conditioned set geometry is distinct
  from a renamed scalar coefficient, but SDP already supports offline synthetic
  negatives and occupies set-valued action-chunk supervision;
- prior anchor: SDP has an RSS 2026 paper, MIT official code, and positive
  simulation/real-robot results;
- mechanism: set-valued targets could avoid penalizing acceptable action
  variation, but local conditional action sets are only weakly observed from
  one demonstration action per state;
- data viability: no human corrections exist, so the construction would depend
  on synthetic negatives or nearest-neighbor action statistics;
- identity: zero-effect LoRA initialization preserves Base before training;
- decisive experiment: set noncollapse and gradient distinction are cheap, but
  a fair action-equivalence claim is weaker without real corrections.

Score:

- provisional novelty: `14 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `78 / 100`

TASF is not selected. It is too close to the already rejected ISAC/SDP family
and lacks locally observed correction sets.

## Candidate 3: ACORN-VLA

Name: `ACORN-VLA`, Action Coarse-to-Residual Orthogonal Reasoning for VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: ACoT-VLA,
https://arxiv.org/abs/2601.11404.

### Scientific Method

Predict a low-frequency coarse action guide from SmolVLA features and constrain
the final flow adapter to model the orthogonal high-frequency residual. The
prior proxy predicts a coarse action guide without the orthogonal decomposition;
the key ablation is ordinary LoRA with the same demonstrations.

### Low-Compute Parameterization

A small coarse guide and fixed rank-4 LoRA would replace ACoT's unreleased
EAR/IAR implementation. Standard LoRA is required because the method updates
weights and receives the same demonstration supervision.

### Quality Screen

- provisional novelty: an explicit orthogonal residual decomposition differs
  from ACoT's dual action reasoners, but remains in the coarse action-guidance
  family;
- prior anchor: ACoT reports strong LIBERO/LIBERO-Plus/VLABench results and has
  an official repository, but core modules and checkpoints remain pending;
- mechanism: a coarse guide may reduce semantic-to-kinematic ambiguity, while
  an orthogonal residual preserves detail;
- data viability: action chunks provide targets, but local ECHO, RAR, CALA, and
  ActionMap evidence found no useful headroom or legal residual predictability;
- identity: zero-initialized conditioning can preserve Base;
- decisive experiment: predictability can be tested cheaply, but the family is
  already locally closed without a new source of headroom.

Score:

- provisional novelty: `15 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `79 / 100`

ACORN is not selected. It would reopen a closed action-conditioning family.

## Selection

Select exactly one: `HEST-VLA` with `93 / 100`.

Required next step: freeze the HEST Researcher A proposal, Reviewer B attack,
rebuttal, mathematical audit, preregistration, and Stage 0 protocol before code.
The first empirical gate must use discovery/validation actions and direct
exact-state replay/control fidelity. It may not use confirmatory reset
identities, change SmolVLA queue semantics, or treat offline action L2 as paper
evidence.
