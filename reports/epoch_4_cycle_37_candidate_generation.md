# Epoch 4 Cycle 37 Candidate Generation

Date: 2026-07-16 KST

Decision: `CSPR_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Candidate count: exactly `3`

Previous method: `DCCG-VLA`

Previous decision: `DCCG_STAGE_0_DATA_FAILURE`

Governance: current performance-oriented governance with one genuinely new
mechanism, LoRA only as implementation infrastructure, and the closest
external prior in the first serious comparison.

Local data constraint: use only locally available development identities
unless a later preregistered audit proves new coverage. The available cached
SmolVLA Base rows are `640` rows across `libero_10/task_5`,
`libero_goal/task_5`, `libero_object/task_3`, and `libero_spatial/task_3`,
with demo ids `0..9`. DCCG identities, cache source, thresholds, and
interpretation remain closed.

## Candidate 1: CSPR-VLA

Full name: Critical-Step Selective Policy Refinement for SmolVLA

Closest prior: DySL-VLA

Primary sources:

- https://arxiv.org/abs/2602.22896
- https://github.com/PKU-SEC-Lab/DYSL_VLA

Positive prior: DySL-VLA reports that action importance can drive dynamic
layer skipping, with official code, a `2.1%` CALVIN success-length gain,
`85.7x` fewer trainable parameters, and `3.75x` speedup at iso-accuracy.

Contribution type: `PRIOR_EXTENSION`

Scientific method: learn a deployment-observable critical-step score over
SmolVLA `[50, 7]` action chunks, then apply a zero-initialized bounded
residual only on action cells or timestep groups predicted to require high
precision. Noncritical cells are exact Base passthrough. LoRA may parameterize
the low-compute residual/gate implementation, but the scientific mechanism is
critical-step selective action refinement, not LoRA and not global fine-tuning.

Minimal difference from prior: DySL-VLA uses action importance to decide when
to skip or execute model layers. CSPR keeps DySL as policy 2 but transfers the
importance signal to the action interface: the model spends corrective action
capacity only on critical timesteps and dimensions, while preserving Base
elsewhere.

Mechanism chain:

- problem condition: many LIBERO failures concentrate around short high-
  precision intervals such as approach alignment, contact, grasp, release, or
  placement;
- intermediate failure mechanism: globally changing all action cells harms
  clean behavior, while leaving critical cells unchanged misses the few
  moments where Base needs precision;
- policy representation/action behavior: CSPR predicts critical action cells
  from current observation, proprioception, language, Base chunk, and cached
  visual features, then activates a bounded residual only there;
- expected closed-loop improvement: better contact and placement precision
  with near-Base behavior on easy or already-stable segments.

Data and supervision viability: the available cache has `640` SmolVLA Base
rows with `[50, 7]` chunks, frame indices, task identities, demo ids, cached
visual features, and source demonstration paths for the four cache-covered
tasks. Criticality labels can be derived from Base-vs-demo action error,
translation/rotation curvature, gripper transitions, and local action-change
energy. No success, reward, done, object pose, simulator state, future
observation, or held-out confirmatory identity is used at inference.

Identity-preserving integration: residual branch initializes to zero, gate
initializes to Base passthrough, residual caps are groupwise for translation,
rotation, and gripper, and an action-validity plus clean-retention audit is
required before any rollout.

First serious comparison:

1. `smolvla_base`
2. `dysl_action_importance_proxy`
3. `cspr_full`
4. `cspr_uniform_refinement_ablation`
5. `critical_step_threshold_simple_killer`

Scores:

- provisional novelty: `23 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `16 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `10 / 10`
- decisive experiment feasibility: `10 / 10`
- total: `90 / 100`

Rationale: CSPR is selected because it is prior-anchored, locally
reproducible under the available cached identities, identity-preserving by
construction, and genuinely changes the scientific mechanism from LoRA or
global fine-tuning to action-importance-conditioned selective refinement.

## Candidate 2: PGF-VLA

Full name: Progress-Gradient Field Guidance for SmolVLA

Closest prior: ProgressVLA

Primary sources:

- https://arxiv.org/abs/2603.27670
- https://arxiv.org/html/2603.27670v1

Positive prior: ProgressVLA reports robust progress estimation and
differentiable progress guidance, with gains on CALVIN, LIBERO, and
real-world deployment.

Contribution type: `PRIOR_EXTENSION`

Scientific method: learn a deployment-observable progress predictor and a
small action-gradient field that nudges generated action chunks toward
monotone task progress while preserving Base when predicted progress is
already adequate.

Minimal difference from prior: ProgressVLA maps predicted action tokens to
future latent visual states and then applies maximal-progress guidance. PGF
would use cached SmolVLA chunks and current-observation features to learn a
lighter progress-gradient proxy over action chunks.

Mechanism chain:

- problem condition: long-horizon tasks can stall after partial completion;
- intermediate failure mechanism: Base continues locally plausible motions
  without advancing progress;
- policy representation/action behavior: PGF predicts progress and applies a
  bounded guidance vector only when the predicted progress slope is too low;
- expected closed-loop improvement: fewer stalls and less redundant motion.

Data and supervision viability: progress labels can be derived from frame
index and demonstration ordering on the cache-covered demos. The risk is that
this becomes too close to Cycle 18 PCAV's progress-consequence axis, and a
frame-index target may encode demo timing rather than causal task progress.

Identity-preserving integration: zero guidance at initialization, Base
passthrough when progress slope is adequate, clean-retention and no-privileged
time-index inference gates.

First serious comparison:

1. `smolvla_base`
2. `progressvla_transparent_proxy`
3. `pgf_full`
4. `pgf_no_progress_gradient_ablation`
5. `frame_index_progress_simple_killer`

Scores:

- provisional novelty: `15 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `79 / 100`

Rationale: PGF is plausible, but historical overlap with PCAV and the danger
of privileged or noncausal progress supervision make it weaker than CSPR for
the next cycle.

## Candidate 3: POTR-VLA

Full name: Potential-Ordered Trajectory Reranking for SmolVLA

Closest prior: ForesightFlow

Primary sources:

- https://arxiv.org/abs/2606.04968
- https://arxiv.org/html/2606.04968v1

Positive prior: ForesightFlow jointly generates action chunks and
success-potential trajectories, ranks candidates without an external critic,
and reports simulation and real-world improvements plus `38%` training-compute
reduction.

Contribution type: `IMPLICIT_GAP_SOLUTION`

Scientific method: generate a small set of Base-near action candidates and
rank them with a learned local potential score, where the potential is trained
only from development data and is not used to tune confirmatory identities.

Minimal difference from prior: ForesightFlow trains a self-guided
flow-matching policy with success-potential coordinates and decoupled
advantage weighting. POTR would be a SmolVLA-compatible Base-near proxy that
uses an explicit potential head and bounded reranking rather than replacing
the flow policy.

Mechanism chain:

- problem condition: Base may sample multiple plausible chunks but choose a
  lower-quality one without a self-score;
- intermediate failure mechanism: the policy lacks a calibrated local
  potential for candidate ranking;
- policy representation/action behavior: POTR scores Base-near candidates and
  selects the highest potential candidate under action-delta and validity caps;
- expected closed-loop improvement: fewer avoidable low-potential chunks.

Data and supervision viability: current local cache lacks noncollapsed
success/failure/advantage labels for the cache-covered identities. Any
potential target would be a proxy from demonstrations, action smoothness, or
Base-vs-demo error, which is a weaker reproduction of ForesightFlow.

Identity-preserving integration: candidate set always includes exact Base;
selection defaults to Base unless potential margin and validity gates pass.

First serious comparison:

1. `smolvla_base`
2. `foresightflow_potential_proxy`
3. `potr_full`
4. `potr_no_potential_ablation`
5. `base_nearest_demo_rerank_killer`

Scores:

- provisional novelty: `21 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `3 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `77 / 100`

Rationale: POTR has a strong external prior, but the current local data does
not support faithful success-potential or advantage supervision without new
rollouts or hidden labels.

## Selection

Selected method: `CSPR-VLA`

Selected score: `90 / 100`

Selection decision: `CSPR_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

CSPR is selected because it is the strongest fixed-protocol next method after
DCCG's data/cache failure. It uses one new mechanism, keeps LoRA as
infrastructure only, fits the verified cached Base identities, preserves Base
by default, and puts the closest external prior, DySL-VLA, into the first
serious comparison. Unknown empirical performance is not a rejection reason.
No CSPR proposal, implementation, training, validation search, rollout, or
confirmatory-test access has happened.
