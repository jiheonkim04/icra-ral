# G3P-VLA Researcher A Proposal

Date: 2026-07-15 KST

Decision: `G3P_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `G3P-VLA`, Grounded 3D Point Injection for frozen SmolVLA.

This proposal is written after the valid EAC Stage B current-formulation kill. It does not rescue, retune, reinterpret, or expand EAC. It starts a new method cycle with a different mechanism axis: source-gated spatial grounding at the action interface.

## External Prior Anchor

Closest external prior: Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization, https://arxiv.org/abs/2606.27663.

Positive prior result: the prior reports large LIBERO-PRO gains from converting a grounding signal into a 3D gripper-relative displacement and injecting it into the action head, with positive results on GR00T-N1.6 and `pi_0.5`.

Official code/checkpoint status: no official code or checkpoint for the closest prior is verified in this repository. Until official equivalence is established, the local closest-prior comparison is a faithful transparent proxy, not an official reproduction.

Secondary priors:

- RoboPoint: https://arxiv.org/abs/2406.10721
- RoboGround: https://arxiv.org/abs/2504.21530
- AffordanceVLA: https://arxiv.org/abs/2606.06155
- ActionMap as a related action-interface geometry prior: https://arxiv.org/abs/2606.06904 and https://github.com/showlab/ActionMap

## Claim

If local SmolVLA failures include spatial grounding errors, then a deployment-observable task target encoded as a gripper-relative 3D displacement and injected through an identity-preserving action-conditioning adapter can improve closed-loop LIBERO success beyond Base, a closest-prior 3D-point proxy, a no-3D/no-injection ablation, and one simple 2D/phase/nearest-object heuristic.

The claim is conditional on Stage 0 proving that the point source is legal, noncollapsed, observable from deployment inputs, split-clean, and useful enough to justify training or rollout.

## Evidence Partitions

`DISCOVERY`:

- inspect local failure/headroom conditions;
- construct candidate target-point labels;
- use oracle object-state or simulator geometry only as diagnostics and training-label sources;
- inspect source legality, label balance, and trivial baselines.

`VALIDATION`:

- select one point-confidence threshold, one adapter scale, and at most one architecture choice;
- choose one final configuration using the preregistered validation score;
- verify clean retention, action validity, point confidence, mechanism activation, and action-delta bounds.

`CONFIRMATORY_TEST`:

- one frozen paired official LIBERO manifest after method, checkpoint, baselines, ablation, tasks, reset identities, metrics, and thresholds are frozen;
- no confirmatory outcome may be used to retune G3P.

## Method

Inputs at inference:

- deployment RGB observations exposed by the official SmolVLA/LIBERO path;
- deployment proprioception, especially end-effector state;
- language instruction;
- Base SmolVLA features or actions when available through the local runner;
- no simulator object pose, reset identity, reward, success label, future observation, or hidden confirmatory-test metadata.

Training-only or diagnostic labels:

- oracle target object or placement position, if available from discovery/validation simulator metadata;
- image-space target-point pseudo-labels;
- gripper-relative displacement labels derived from legal development records;
- labels are forbidden for confirmatory inference.

Core representation:

- target point `p_t` in a legal spatial frame when available;
- gripper position `p_g`;
- displacement `d = p_t - p_g`;
- point confidence `c` from the deployable point predictor;
- unknown/no-point token when the point source is absent or below threshold.

Action conditioning:

- encode `d` and `c` with a small MLP;
- inject the embedding into a zero-initialized action-conditioning adapter around the SmolVLA action interface;
- initialize and gate the adapter so initial behavior is exact Base passthrough;
- bound translation, rotation, and gripper deltas separately;
- preserve action validity and clean behavior before rollout.

## Stage 0 Development Audit

Stage 0 must run before any expensive training, validation search, or rollout.

Required checks:

- zero overlap between discovery, validation, and reserved confirmatory identities;
- legal source inventory for RGB, proprioception, language, Base features, and any geometry labels;
- explicit proof that privileged simulator/object-state data is not required at inference;
- point-label positive/negative counts, variance, task coverage, phase coverage, and duplicate counts;
- no all-zero, all-one, single-task, or single-phase point targets;
- oracle headroom diagnostic showing the target variable is plausibly useful;
- point predictability from deployment inputs above a trivial majority, phase, or task heuristic;
- confidence noncollapse;
- Base passthrough with initial action delta p95 exactly or approximately zero;
- action validity and bound checks;
- no training, rollout, validation search, or confirmatory-test tuning during Stage 0.

Stage 0 hard stops:

- `DATA_OR_SUPERVISION_FAILURE` if legal deployable point labels or predictors are unavailable or collapsed;
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE` if Base, closest-prior proxy, and oracle show no useful spatial headroom;
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` if the source gate or adapter cannot preserve Base behavior;
- `DESIGN_FAILURE` if the point is not observable from deployment inputs or is explained by a trivial heuristic.

These stops are not closed-loop scientific kills.

## Bounded Validation Search

Default maximum: six total configurations.

Allowed factors:

- point confidence threshold: at most three values;
- adapter scale: at most three values;
- point encoder architecture: at most two choices;
- no combinatorial grid beyond six named configurations.

Validation score:

`score = 0.30 * point_predictability + 0.20 * clean_retention + 0.20 * bounded_action_validity + 0.15 * mechanism_activation + 0.10 * simple_baseline_margin + 0.05 * efficiency`

The score is development-only and cannot use confirmatory-test identities or outcomes.

## First Serious Comparison

Exactly five policies:

1. `frozen_smolvla`
2. `g3p_3d_point_proxy`
3. `g3p_full`
4. `g3p_no_3d_no_injection_ablation`
5. `simple_2d_phase_or_nearest_object_heuristic`

`g3p_3d_point_proxy` is a faithful transparent local proxy unless exact official equivalence is independently established. The simple heuristic must remain live through Stage A/B.

Stage A:

- approximately `10` paired episodes per policy;
- catastrophic screen only;
- no one- or two-episode permanent kill.

Stage B:

- at least `40` paired episodes per key policy;
- identical task/reset identities across policies;
- report task-balanced success, paired deltas, bootstrap CIs, paired wins/losses/ties, mechanism activation, clean retention, latency, and VRAM.

## Required Ablations And Simple Baseline

Key ablation:

- remove the 3D displacement and action-head injection while keeping any training/data path as matched as possible.

Closest-prior proxy:

- use the same legal point source and the closest feasible 3D injection mechanism, labeled as a proxy until official equivalence is proven.

Simple reviewer-killer:

- one strongest simple 2D point, task-phase, or nearest-object heuristic selected before confirmatory testing.

## Mathematical Commitments

The mathematical mechanism audit must define:

- variables and tensor shapes for images, proprioception, language, target point, gripper position, displacement, confidence, adapter embedding, Base action chunk, and adapted action chunk;
- formula for displacement and confidence gating;
- units and normalization of translation, rotation, and gripper terms;
- exact gradient path through point predictor and adapter;
- objective term magnitudes and gradient norms on a small batch;
- bounded action-delta formula;
- no KL between deterministic 7D actions.

## Safety And Integrity

G3P may not:

- use simulator object-state, target coordinates, reset identity, task success, future observation, or reward at inference;
- tune on confirmatory identities;
- reinterpret Stage 0 failure as a scientific closed-loop kill;
- rescue itself after a valid Stage B kill;
- add extra baselines before the first five-policy comparison unless a concrete reviewer objection makes them decision-relevant and cheaper than proceeding.

Immediate next step: Reviewer B attacks novelty, source legality, trivial baselines, local feasibility, and leakage risk before any implementation.
