# DCCG-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `DCCG_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `DCCG-VLA`, Demonstration-Calibrated Coherence Guidance for SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

## Claim

Flow-based SmolVLA action chunks can be locally high-likelihood while still
containing within-chunk jitter, pauses, excessive jerk, or gripper-incoherent
motion. These action-coherence failures can disturb fine manipulation before
the next observation or replan is allowed to help.

DCCG tests whether a continuous coherence geometry estimated from existing
LIBERO demonstrations can guide SmolVLA action generation better than the
closest prior ACG's architecture-perturbation guidance, while preserving exact
Base behavior at zero guidance.

The paper claim, if supported, is narrow:

> Demonstration-calibrated action-coherence guidance improves flow-based VLA
> action generation over ACG-style generic incoherence guidance and simple
> action smoothing under matched SmolVLA action semantics.

DCCG does not claim a new VLA backbone, a new LoRA method, generic robustness,
generic adaptive chunking, a replacement policy, or a universal control
barrier.

## Positive Prior Anchor

Closest prior: ACG, Action Coherence Guidance for Flow-based VLA models.

Sources:

- https://arxiv.org/abs/2510.22201
- https://arxiv.org/html/2510.22201v2
- https://github.com/DAVIAN-Robotics/ACG
- https://davian-robotics.github.io/ACG/

ACG reports that flow/diffusion VLAs can fail from action incoherence caused
by demonstration noise, including jerks, pauses, jitter, instability, and
trajectory drift. Its mechanism constructs an incoherent action-generation
vector by disrupting temporal self-attention and guides sampling away from
that vector. It reports improved action coherence and success on RoboCasa,
DexMimicGen, and real-world SO-101 tasks, with public code.

ACG therefore enters as policy 2 in the first serious comparison. It is not
deferred until after internal ablations.

## Exact Technical Difference

ACG's negative direction is model-internal: replace temporal communication in
self-attention with an identity attention map, obtain an incoherent vector
field, and extrapolate away from it.

DCCG's negative direction is data-calibrated: fit a LIBERO demonstration
coherence manifold over legal 7D action chunks, then guide generated chunks
away from action sequences that are incoherent relative to that manifold.

DCCG preserves the prior's broad idea of action-generation guidance, but
changes the technical object:

1. ACG: incoherence from a hand-constructed attention perturbation.
2. DCCG: incoherence from a validated demonstration action-coherence energy.

This is not Gaussian smoothing, not a learned residual action head, not
VLA-Corrector latent-drift monitoring, and not A2C2 stepwise residual
correction.

## Falsifiable Mechanism Chain

Problem condition:

- SmolVLA emits a 50-step 7D action chunk;
- a chunk can have high imitation likelihood but poor within-chunk coherence;
- jitter, pauses, and jerk are especially harmful near grasp, insertion,
  alignment, or placement.

Intermediate failure mechanism:

- incoherent arm motion can nudge objects, lose alignment, or accumulate drift;
- incoherent gripper timing can open, close, or hover at the wrong moment;
- the next observation arrives too late to prevent the local disturbance.

Policy behavior:

- Base follows the generated chunk as-is;
- simple smoothing may reduce jitter but can distort task-critical contact or
  gripper events;
- ACG may improve generic action coherence but is not calibrated to LIBERO
  task/action regimes.

Proposed method:

- estimate a continuous coherence manifold from discovery demonstrations;
- score generated chunks by robust task/regime-normalized velocity,
  acceleration, jerk, pause, spectral-energy, and gripper-transition features;
- compute a bounded action-space gradient that reduces coherence energy;
- apply the guidance only when the validation-frozen gate says the chunk is
  outside the legal coherence manifold.

Expected result:

- DCCG reduces coherence defects while preserving gripper events;
- DCCG differs from Base, ACG, and simple smoothing in relevant states;
- DCCG beats Base, ACG, the no-demo-calibration ablation, and simple smoothing
  on the matched claim axis;
- clean behavior and action validity are retained.

## Legal Inputs

Allowed at training and validation:

- LIBERO demonstration action chunks;
- ordinary RGB observations, proprioception, task labels, and language;
- cached frozen SmolVLA Base action chunks;
- action-derived features such as velocity, acceleration, jerk, finite
  differences, gripper sign/magnitude, queue index, and chunk index;
- discovery and validation identities only.

Allowed at inference:

- ordinary SmolVLA inputs;
- the action chunk or flow sample currently being generated;
- action-derived coherence features from that chunk;
- a validation-frozen coherence gate and guidance scale.

Prohibited at inference:

- simulator object state or privileged `states` arrays;
- reward, success, done, or reset identity;
- future observations or future expert actions;
- human correction labels;
- confirmatory task labels, outcomes, or thresholds.

## Evidence Partitions

Discovery tasks:

- `libero_10/task_1`
- `libero_10/task_3`
- `libero_goal/task_1`
- `libero_goal/task_3`
- `libero_object/task_1`
- `libero_spatial/task_1`

Validation tasks:

- `libero_10/task_5`
- `libero_goal/task_5`
- `libero_object/task_3`
- `libero_spatial/task_3`

Confirmatory tasks:

- `libero_10/task_7`
- `libero_goal/task_7`
- `libero_object/task_5`
- `libero_spatial/task_5`

Demonstration partitions:

- discovery demonstrations `demo_0..demo_29` may fit coherence statistics and
  run diagnostics;
- validation demonstrations `demo_30..demo_39` may select one frozen
  configuration;
- confirmatory task demonstrations and rollout reset identities are sealed
  until method, metrics, policy list, and thresholds are frozen.

Rollout reset identities:

- discovery: `20263601..20263612`;
- validation: `20263621..20263632`;
- confirmatory Stage A/B: `20263641..20263650`;
- one unresolved-only expansion: `20263651..20263660`.

No confirmatory result may change the coherence features, bins, thresholds,
guidance scale, policy list, task list, metric, or decision rule.

## Data Health

Before expensive training or rollout, DCCG must report:

- task, demo, and sampled-window counts;
- action chunk shape exactly `[50, 7]`;
- finite action fraction and per-dimension ranges;
- duplicate key count and split overlap count;
- gripper transition count and task coverage;
- distribution of velocity, acceleration, jerk, pause, and spectral-energy
  features;
- noncollapsed coherence energy on discovery and validation;
- source HDF5 hashes and cached Base chunk hashes;
- no validation task contributing more than `25%` of sampled rows;
- no confirmatory reads.

MHS failed from collapsed binary labels. DCCG's primary supervision is
continuous, but it still must prove nonconstant feature variance and nontrivial
gate activation on validation data.

## Mathematical Definition

Let:

- `H = 50`: action chunk horizon;
- `D = 7`: action dimension;
- `A in R^(H x D)`: a postprocessed normalized action chunk;
- `A_B in R^(H x D)`: frozen Base SmolVLA chunk;
- `d in {trans, rot, grip}`: action groups;
- `Delta A_h = A_h - A_(h-1)` for `h = 1..H-1`;
- `Delta2 A_h = Delta A_h - Delta A_(h-1)` for `h = 2..H-1`;
- `Delta3 A_h = Delta2 A_h - Delta2 A_(h-1)` for `h = 3..H-1`.

For each chunk, define a coherence feature vector:

`s(A) = [p95(|Delta A_trans|), p95(|Delta2 A_trans|),
p95(|Delta3 A_trans|), p95(|Delta A_rot|), p95(|Delta2 A_rot|),
p95(|Delta3 A_rot|), pause_fraction_trans, high_frequency_energy_trans,
gripper_transition_count, gripper_reversal_count]`.

The exact feature implementation must be frozen in the mathematical audit.
Features are computed from actions only and do not require privileged state.

Discovery data define task/regime bins `b` from nonprivileged keys:

- task family;
- normalized chunk index within the demonstration for training labels only;
- action-regime features computable from the current generated chunk at
  inference: translation magnitude, rotation magnitude, gripper magnitude, and
  gripper sign-change indicator.

For each bin, compute robust center and scale:

`m_b = median(s(A_demo))`

`q_b = max(IQR(s(A_demo)), epsilon)`.

The DCCG coherence energy is:

`E_dccg(A, b) = mean_i Huber((s_i(A) - m_(b,i)) / q_(b,i), delta=1)`.

The gripper protection term is:

`E_grip(A, b) = Huber((n_transitions(A) - m_grip_b) / q_grip_b, 1)
              + Huber((n_reversals(A) - m_rev_b) / q_rev_b, 1)`.

Total energy:

`E(A, b) = E_dccg(A, b) + lambda_grip * E_grip(A, b)`.

No KL divergence is used. Deterministic 7D actions and SmolVLA flow vectors
are not treated as probability distributions.

## Guidance

Let `X_u in R^(H x D)` be the current flow action sample at solver step `u`
and `F_base(X_u, o, l, u)` be the frozen SmolVLA flow vector.

The DCCG guidance vector is:

`G_u = clip_group(grad_X E(X_u, b), c_trans, c_rot, c_grip)`.

The guided flow vector is:

`F_dccg = F_base - gamma * alpha_u * G_u`.

`alpha_u` is a frozen solver-step schedule. `gamma = 0` gives exact Base.
All clipping constants and schedules are frozen before validation search.

If the coherence gate is inactive, DCCG returns `F_base` exactly. If any
gradient, action, or score is nonfinite, DCCG falls back to Base and records an
implementation event.

## Objective Engineering

Stage 0 does not need to train a large model. It computes robust statistics and
tests differentiable energy and guidance on frozen action chunks.

If a small learned coherence scorer is later used, it may only approximate the
frozen feature energy and must obey:

- input variables and tensor shapes documented;
- loss scale and units documented;
- finite nonzero gradient on a small batch;
- no gradient into frozen SmolVLA unless explicitly generating a guided action
  sample;
- no learned term may overwhelm clean retention without validation evidence.

Required ablations:

- ACG official proxy or transparent local proxy as policy 2;
- no-demo-calibration ablation using global uncalibrated coherence features;
- simple action smoothing reviewer-killer;
- zero-guidance Base identity check.

## Stage 0A: Source And Coherence Smoke

Stage 0A may read at most:

- two discovery tasks;
- two demonstrations per task;
- 32 sampled action chunks per demonstration;
- cached Base chunks for the same rows;
- no validation or confirmatory task.

It must establish:

- source files and hashes;
- chunk shape `[50, 7]` and finite values;
- noncollapsed coherence features;
- gripper transition coverage;
- no duplicate row keys or split overlap;
- exact Base passthrough when `gamma = 0`;
- finite differentiable coherence energy and action gradients;
- group clipping preserves action validity;
- DCCG differs from simple smoothing and ACG-style generic guidance on
  diagnostic incoherent chunks;
- no privileged inference input and no confirmatory read.

One bounded implementation repair is allowed only for source, shape, gradient,
serialization, or hook defects before any scientific gate is applied. The
repair may not change the method, features, tasks, splits, thresholds,
comparators, or decision classes.

## Stage 0B: Development Headroom Audit

If Stage 0A passes, Stage 0B uses discovery and validation only to check:

- Base and ACG leave meaningful coherence/action-validity headroom;
- DCCG coherence energy separates expert chunks from synthetically jittered,
  paused, reversed, and gripper-corrupted chunks;
- DCCG beats zero-change, global-smoothing, and no-demo-calibration diagnostic
  baselines on validation separation;
- action deltas from Base are bounded;
- clean validation behavior is retained;
- gate activation is neither always on nor always off;
- ACG proxy, DCCG, ablation, and smoothing all use matched action semantics
  and inference budgets.

Failure here is `DATA_FAILURE`, `NO_HEADROOM`, `IMPLEMENTATION_FAILURE`, or
`DESIGN_FAILURE`, not a closed-loop scientific result.

## Bounded Validation Search

Maximum six total configurations:

- guidance scale `gamma`: `0.05`, `0.10`, `0.20`;
- gate quantile: `0.90`, `0.95`.

This is a `3 x 2` bounded search. No other feature set, binning method,
solver schedule, clipping cap, task split, or comparator is searched. If a
learned scorer is introduced, at most two lightweight seeds are allowed, and
the six-configuration maximum still holds.

Validation score:

`S_val = 0.40 * closed_loop_success_or_proxy
       + 0.20 * clean_retention
       + 0.15 * coherence_separation
       + 0.15 * action_validity
       + 0.10 * acg_and_smoothing_margin`.

All terms are scaled to `[0,1]`. Ties break by clean retention, then lower
guidance activation rate, then smaller `gamma`, then lower gate quantile.

All tried configurations and negative results are saved. The selected
configuration is frozen before confirmatory access.

## Mechanism Smoke Before Rollout

For Base, ACG, DCCG, ablation, and smoothing report:

- Base action chunk and final action chunk;
- coherence energy and gate value;
- gradient norm and clipped guidance norm;
- changed dimensions and per-group delta p95;
- gripper transition preservation;
- action validity;
- clean validation behavior;
- activation context and task/reset key.

Do not launch rollout if:

- coherence features collapse;
- no Base or ACG headroom exists;
- DCCG is identical to the ablation or smoothing;
- DCCG changes all actions globally;
- action validity fails;
- clean retention fails;
- any privileged inference input or confirmatory identity is used.

## First Serious Comparison

Exactly five policies:

1. `smolvla_base`
2. `acg_official_proxy`
3. `dccg_full`
4. `dccg_no_demo_calibration_ablation`
5. `action_smoothing_simple_killer`

Policy 2 uses official ACG code when feasible. If exact official equivalence
is blocked by model/runtime mismatch, the proxy must transparently implement
ACG's published perturbation-guidance mechanism under the same SmolVLA action
interface and be labeled a faithful local proxy.

Policy 4 removes demonstration calibration but keeps the same DCCG integration
surface and budget.

Policy 5 applies the strongest simple smoothing baseline that preserves action
shape and gripper-event legality under the same action caps.

## Confirmatory Stages

All policies use one paired manifest.

Stage A:

- approximately 10 paired episodes per policy;
- detects catastrophic degradation, exact equivalence, no headroom, invalid
  mechanism, or clear prior/ablation/smoothing dominance;
- small differences advance automatically;
- no hyperparameter, threshold, task, reset, or comparator changes.

Stage B:

- 40 paired episodes per key policy;
- paired wins/losses/ties;
- task-balanced success;
- paired bootstrap confidence interval;
- effect size and failure-rate reduction;
- per-task breakdown;
- coherence metrics and gate activation;
- clean retention;
- action validity;
- policy calls and compute overhead, excluding resource-contention intervals.

One expansion to 80 paired episodes per key policy is allowed only when the
frozen uncertainty rule declares the result unresolved.

## Paper-Candidate Gate

DCCG becomes a serious paper candidate only if:

- DCCG beats Base;
- DCCG beats ACG on the matched claim axis;
- DCCG beats the no-demo-calibration ablation;
- simple action smoothing does not explain the gain;
- clean behavior and action validity are retained;
- coherence evidence supports the intended explanation;
- novelty remains defensible after final literature refresh.

Then immediately verify:

- Quantized OpenVLA-OFT INT4 versus Quantized OpenVLA-OFT INT4 plus DCCG;
- one claim-specific second condition or benchmark;
- one or more newly relevant baselines when feasible;
- compute and latency outside resource-contention intervals;
- figure/table-ready mechanism and outcome evidence.

## Failure Classification

- `DATA_FAILURE`: invalid/collapsed coherence features, overlap, insufficient
  task/action-regime coverage, or missing legal inputs;
- `NO_HEADROOM`: Base or ACG leaves no plausible coherence or task-success
  gain;
- `IMPLEMENTATION_FAILURE`: source, hook, shape, reload, gradient, action
  validity, or JSON persistence defect;
- `DESIGN_FAILURE`: valid implementation but DCCG is nonacting, globally
  destructive, or explained by ACG, the ablation, or smoothing;
- `VALID_SCIENTIFIC_KILL`: frozen confirmatory method is valid and loses under
  the preregistered decision rule;
- `UNDERPOWERED_OR_UNRESOLVED`: valid evidence does not resolve the claim
  after the allowed expansion.

Implementation or data failure cannot be reported as a closed-loop scientific
result. A major redesign after confirmatory access is a new method cycle.

## Resource Policy

The Windows gaming and Efficiency Mode intervals remain recorded in
`reports/resource_contention_intervals.json`.

Latency, throughput, wall-clock efficiency, CUDA utilization, and resource
utilization are excluded whenever overlap is unknown or positive. Synchronous
closed-loop success rows may remain valid only after timeout, exception,
identity, action-semantics, duplicate-key, and manifest audits.

## Researcher A Recommendation

Proceed to independent Reviewer B attack. Do not implement DCCG, fit a learned
scorer, run validation search, launch rollout, or access confirmatory
identities until Reviewer B attack, rebuttal, mathematical audit,
preregistration, and prototype protocol are complete.
