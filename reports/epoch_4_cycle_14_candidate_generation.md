# Epoch 4 Cycle 14 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_COVI_VLA`

Governance applied: current performance-oriented and honest-positive-result
governance. Exactly three candidates were generated and scored. RAR-VLA remains
stopped as `DESIGN_FAILURE`; it must not be rescued by changing history
features, residual labels, thresholds, source gates, validation configs, or
baseline interpretation.

## Candidate 1: COVI-VLA

Name: `COVI-VLA`, Complementary Occlusion View Imagination for frozen SmolVLA.

Contribution type: `NEW_DEPLOYMENT_PROBLEM`.

Closest external prior: LIBERO-Occ / Viewpoint Imagination,
https://arxiv.org/abs/2606.10862.

Secondary priors: CamVLA, https://arxiv.org/abs/2607.05396; STRONG-VLA,
https://arxiv.org/abs/2604.10055.

Positive prior result: LIBERO-Occ reports that scene-induced occlusion produces
large performance drops for strong VLA baselines and that VIM improves occluded
LIBERO success by generating a complementary view from the occluded observation
without requiring extra deployment cameras.

Official code/checkpoint/reproducible mechanism: LIBERO-Occ reports an official
benchmark/code repository at https://github.com/litsh/Libero-Occ. Exact local
equivalence has not been established. The reproducible mechanism is
complementary-view inference from an occluded primary observation plus
action-conditioned use of the recovered visual evidence.

Assumption or limitation extended: VIM changes the VLA to generate visual tokens
and actions jointly. COVI-VLA tests a smaller frozen-SmolVLA-compatible
extension: learn an occlusion-conditioned complementary-view representation and
inject it through an identity-preserving gate, rather than replacing the
backbone or action head.

Minimal technical difference proposed by Ours:

- use development-only controlled occlusions on official LIBERO observations;
- use paired official camera streams as supervision for complementary-view
  feature recovery when permitted by the split;
- learn a compact visual feature imputer or adapter that predicts missing
  complementary-view evidence from legal deployment inputs;
- gate the adapter so the initial policy equals frozen SmolVLA;
- require clean retention and bounded feature/action consequences;
- compare against Base, a VIM-style transparent proxy, COVI full,
  no-imagined-view ablation, and a random-cutout clean-retention baseline.

Why it could improve the same claim axis: prior local methods repeatedly failed
on action-side corrections. COVI changes the axis to partially observable visual
evidence. The prior shows that the missing-view signal can be useful for
closed-loop success under scene-induced occlusion, and the local two-camera
LIBERO data gives a bounded way to audit whether a smaller representation-level
adapter has usable headroom before rollout.

### Quality Screen

Provisional novelty:

- Distinct from VIM because it is a frozen-backbone identity-preserving
  representation adapter, not a full generative VLA.
- Distinct from PatchGuard because the target condition is physically grounded
  scene-induced occlusion and complementary evidence recovery, not generic
  random image corruption.
- Distinct from RAR/CALA/G3P because it does not predict future actions,
  residual action history, waypoints, or material-point labels.
- Novelty risk remains: if cutout training or direct dual-camera passthrough
  explains the gain, the method must be killed.

Prior-anchor strength:

- Strong positive prior from LIBERO-Occ/VIM, including a benchmark and reported
  official code.
- CamVLA and STRONG-VLA reinforce that view and robustness shifts are live VLA
  deployment problems.
- Official local equivalence is unverified, so `vim_view_imagination_proxy`
  must be transparently labeled.

Mechanism plausibility:

- Problem condition -> task-relevant object or receptacle evidence is hidden
  from one view.
- Intermediate failure mechanism -> frozen SmolVLA receives insufficient visual
  evidence and emits plausible but spatially wrong actions.
- Policy behavior -> misses grasps, aims at the wrong region, or times out.
- Closed-loop failure -> occlusion-induced partial observability lowers task
  success.
- Proposed method -> infer a complementary-view representation and gate it into
  the frozen policy path.
- Intended internal change -> occlusion-specific visual evidence becomes
  available to the action generator while clean states remain near Base.
- Intended action behavior -> better object/receptacle localization under
  occlusion without global action disruption.
- Expected closed-loop improvement -> higher occluded task-balanced success
  than Base, VIM proxy, ablation, and random-cutout baseline with retained clean
  success.

Data and supervision viability:

- Official LIBERO records include two image streams and proprioception.
- Controlled image occlusions can be generated on discovery/validation splits
  without touching confirmatory identities.
- Stage 0 must verify that occlusion labels, view-completion targets, and
  Base-failure headroom are noncollapsed before any training or rollout.

Identity-preserving integration:

- Adapter gate initialized to Base passthrough.
- Feature residual or action consequence bounded.
- Clean-retention score is part of validation selection.

Decisive experiment feasibility:

- Stage 0 can audit occlusion headroom, split separation, label health,
  complementary-feature predictability, cutout-baseline strength, and zero-delta
  initialization.
- Stage A can use a fixed matched occlusion manifest with exactly five policies.
- A second condition is naturally available: clean LIBERO retention or a
  held-out occlusion severity/type.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `91 / 100`

## Candidate 2: SURF-VLA

Name: `SURF-VLA`, State-Uncertainty Residual Flow for frozen SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: SUREFlow,
https://arxiv.org/abs/2607.10504.

Secondary priors: ReactVLA, https://arxiv.org/abs/2606.14255; DFM-VLA,
https://arxiv.org/abs/2603.26320.

Positive prior result: SUREFlow reports robust manipulation gains by jointly
predicting action velocities and input-dependent residual uncertainty, enabling
selective refinement of unreliable action dimensions under noise, partial
observability, and stochastic initial conditions.

Official code/checkpoint/reproducible mechanism: SUREFlow reports source code at
https://github.com/tanvirnwu/SUREFlow. Exact local equivalence has not been
established. The reproducible mechanism is state-dependent uncertainty over
residual flow updates, not deterministic-action KL or a confidence head.

Assumption or limitation extended: SUREFlow trains a standalone residual
flow-matching policy. SURF-VLA would keep frozen SmolVLA and learn only a
small uncertainty-conditioned gate that allows bounded refinement when
deployment-observable evidence predicts unreliable action dimensions.

Minimal technical difference proposed by Ours:

- estimate uncertainty targets from development-only action dispersion,
  perturbation sensitivity, or flow-sample variability;
- predict dimension-wise residual uncertainty from legal observations and Base
  actions;
- apply a zero-initialized bounded residual only under uncertainty-gate
  activation;
- compare against Base, SUREFlow-style proxy, SURF full, no-uncertainty
  ablation, and deterministic fixed-noise or variance-threshold baseline.

Why it could improve the same claim axis: if failures arise from state-dependent
unreliable dimensions rather than globally wrong actions, uncertainty-gated
refinement could act selectively while avoiding the global residual failures
seen in prior local methods.

### Quality Screen

Provisional novelty:

- Meaningful if uncertainty is a learned state-dependent action-generation
  variable, not a post-hoc confidence score.
- Risk is substantial because many local output-residual and correction methods
  have already failed.

Prior-anchor strength:

- Strong recent positive prior with reported code and IROS 2026 acceptance.
- Backbone and action-interface mismatch remain unresolved.

Mechanism plausibility:

- Problem condition -> only some states and action dimensions are unreliable.
- Proposed method -> uncertainty-gated refinement acts only where needed.
- Expected behavior -> bounded improvement under noisy or partially observable
  states with clean retention.

Data and supervision viability:

- Flow-sample variability and perturbation sensitivity can be measured, but the
  labels may collapse or become another proxy for action magnitude.
- Stage 0 must prove uncertainty labels are observable and nontrivial.

Identity-preserving integration:

- Residual and gate initialized to zero.
- Per-dimension deltas bounded.

Decisive experiment feasibility:

- Stage 0 is feasible.
- A fair closest-prior comparison is harder than COVI because SUREFlow is a
  different backbone and training regime.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `7 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `86 / 100`

## Candidate 3: DFR-VLA

Name: `DFR-VLA`, Discrete Flow Refinement Adapter for frozen SmolVLA chunks.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: DFM-VLA,
https://arxiv.org/abs/2603.26320.

Secondary priors: Invertible Neural Network Adapter for One-Step Flow Matching,
https://arxiv.org/abs/2606.19194; ReactVLA, https://arxiv.org/abs/2606.14255.

Positive prior result: DFM-VLA reports that iterative action-token refinement
outperforms strong autoregressive, discrete diffusion, and continuous diffusion
baselines on CALVIN, LIBERO, and real-world manipulation while retaining high
inference efficiency.

Official code/checkpoint/reproducible mechanism: DFM-VLA reports a project page
at https://chris1220313648.github.io/DFM-VLA/. Exact local code/checkpoint
equivalence has not been established. The reproducible mechanism is iterative
velocity-field refinement over action tokens with deterministic validation.

Assumption or limitation extended: DFM-VLA operates on discrete action tokens.
DFR-VLA would test whether a lightweight tokenized residual refinement layer
around frozen continuous SmolVLA chunks can revise early action-bin mistakes
without replacing the policy.

Minimal technical difference proposed by Ours:

- quantize 7D action chunks into a small frozen codebook on discovery data;
- learn a development-only iterative refinement adapter over codebook indices;
- decode back to bounded continuous deltas around Base;
- compare against Base, DFM-style proxy, DFR full, no-iteration ablation, and
  simple PCA/codebook projection baseline.

Why it could improve the same claim axis: if Base emits locally wrong early
chunk components that are recoverable by iterative refinement, DFR could repair
sequence-level consistency without hand-tuned smoothing.

### Quality Screen

Provisional novelty:

- Distinct from generic residual correction only if the tokenized velocity
  refinement target is noncollapsed and beats codebook/PCA projection.
- High risk because it remains action-output-side and could repeat killed local
  correction routes.

Prior-anchor strength:

- Strong DFM-VLA positive prior on LIBERO.
- Local interface mismatch is substantial because SmolVLA continuous actions
  are not the same as DFM discrete action tokens.

Mechanism plausibility:

- Problem condition -> early discrete action decisions are wrong but
  sequence-level context can revise them.
- Proposed method -> iterative codebook velocity field updates the chunk before
  execution.
- Expected behavior -> fewer early locked-in action errors.

Data and supervision viability:

- Action chunks and targets exist, but the codebook may collapse to a smoothing
  or mean-action baseline.
- Stage 0 must prove codebook occupancy, refinement predictability, and
  action-headroom beyond projection baselines.

Identity-preserving integration:

- Adapter initialized to no refinement.
- Decoded deltas bounded by component.

Decisive experiment feasibility:

- Stage 0 is feasible.
- Full implementation is more complex and less directly tied to the current
  frozen SmolVLA interface than COVI.

Score:

- provisional novelty: `20 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `81 / 100`

## Selection

Selected method: `COVI-VLA`.

Selection reason:

- It changes the method axis away from action residual/history rescue and into
  scene-induced partial observability.
- It has the strongest positive external-prior anchor in the current candidate
  set: LIBERO-Occ/VIM reports both the problem headroom and a successful
  mechanism, with code available.
- It changes more than two core dimensions relative to RAR: problem condition,
  representation, supervision source, and claim axis all change.
- It has a clear simple-killer baseline, `random_cutout_clean_retention_baseline`,
  so ordinary augmentation cannot be hidden inside the method.
- It can be stopped before rollout if occlusion labels collapse, no Base
  headroom exists, complementary-view targets are not predictable from legal
  inputs, or the adapter disrupts clean behavior.
- Unknown empirical performance is not a rejection reason; Stage 0 can classify
  failure as `DATA_OR_SUPERVISION_FAILURE`,
  `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`, `DESIGN_FAILURE`, or
  `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` before rollout.

Immediate next steps:

1. Freeze a `COVI-VLA` Researcher A proposal and hash it.
2. Reviewer B attacks novelty and source fidelity against LIBERO-Occ/VIM,
   CamVLA, STRONG-VLA, random-cutout/cutmix robustness, PatchGuard-v1, and
   standard two-camera fusion.
3. Researcher A provides one rebuttal if the method remains nontrivial and
   locally feasible.
4. Write `reports/covi_vla/mathematical_mechanism_audit.md`, preregistration,
   and prototype protocol before any expensive training or rollout.
5. Implement only Stage 0 first: occlusion headroom, source legality,
   split separation, label/target health, complementary-view predictability
   above simple image-statistic and cutout baselines, Base passthrough,
   gradient-path smoke, clean retention, and no confirmatory-test identity use.
