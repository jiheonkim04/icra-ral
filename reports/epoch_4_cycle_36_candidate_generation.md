# Epoch 4 Cycle 36 Candidate Generation

Date: 2026-07-16 KST

Decision: `DCCG_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Candidate count: exactly `3`

Previous method: `MHS-VLA`

Previous decision: `MHS_STAGE_0_DATA_OR_SUPERVISION_FAILURE`

Governance: current post-COVI minimum-sufficient governance with one genuinely
new mechanism, LoRA only as implementation infrastructure, and the closest
external prior in the first serious comparison.

## Candidate 1: DCCG-VLA

Full name: Demonstration-Calibrated Coherence Guidance for SmolVLA

Closest prior: ACG

Primary sources:

- https://arxiv.org/abs/2510.22201
- https://arxiv.org/html/2510.22201v2
- https://github.com/DAVIAN-Robotics/ACG
- https://davian-robotics.github.io/ACG/

Positive prior: ACG reports that training-free perturbation guidance improves
action coherence and manipulation success on RoboCasa, DexMimicGen, and
real-world SO-101 tasks, with public code and stated applicability to
flow-based VLA models including SmolVLA.

Contribution type: `PRIOR_EXTENSION`

Scientific method: estimate a continuous LIBERO demonstration-calibrated
coherence manifold over SmolVLA action chunks, then use it as a bounded
flow-guidance signal during action generation. The method guides away from
demonstration-incoherent velocity, acceleration, jerk, and gripper-transition
patterns while preserving exact Base behavior at zero guidance. LoRA, if ever
used, may only expose a low-compute hook for measuring or parameterizing the
coherence score; it is not the scientific mechanism.

Minimal difference from prior: ACG constructs an incoherent direction by
perturbing temporal self-attention. DCCG keeps ACG as policy 2 but replaces the
hand-constructed incoherence direction with a data-calibrated LIBERO
coherence direction learned only from discovery/validation demonstrations and
cached Base chunks. This tests whether the action manifold itself, not merely
attention disruption, provides the useful guidance geometry.

Mechanism chain:

- problem condition: flow-based action chunks can contain jitter, pauses,
  jerk, or gripper-incoherent motion that remains high-likelihood under
  imitation-trained generation;
- intermediate failure mechanism: incoherent within-chunk motion causes object
  nudging, grasp fumbling, or trajectory drift before the next observation can
  repair it;
- policy representation/action behavior: DCCG scores generated chunks against
  a task/phase-local demonstration coherence manifold and applies bounded
  guidance only along incoherent action components;
- expected closed-loop improvement: fewer fine-manipulation failures from
  jitter/drift while retaining Base on already coherent chunks.

Data and supervision viability: existing LIBERO HDF5 demonstrations provide
7D action chunks, ordered frames, gripper events, task labels, and ordinary
inputs. Cached SmolVLA Base chunks provide the identity anchor. The targets are
continuous coherence statistics, so MHS-style binary label collapse is not
expected. No reward, success, done, object pose, simulator state, future image,
or held-out confirmatory identity is used at inference.

Identity-preserving integration: guidance scale initializes to zero; Base
output is exact when the guidance gate is inactive; continuous arm deltas and
gripper sign changes receive separate caps; gripper event timing is protected
against smoothing away legitimate open/close transitions.

First serious comparison:

1. `smolvla_base`
2. `acg_official_proxy`
3. `dccg_full`
4. `dccg_no_demo_calibration_ablation`
5. `action_smoothing_simple_killer`

Scores:

- provisional novelty: `23 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `92 / 100`

Rationale: DCCG is selected because it changes the failed MHS axis from
collapsed binary history labels to continuous action-generation coherence,
anchors to a positive flow-based VLA prior with code, enters ACG early as the
closest prior, and can be audited from existing LIBERO demonstrations without
privileged inference inputs.

## Candidate 2: DAF-VLA

Full name: Demonstration Affordance Field for Base-preserving SmolVLA

Closest prior: AFI, with GEAR-VLA as a geometry-aware secondary prior

Primary sources:

- https://arxiv.org/abs/2512.07472
- https://arxiv.org/html/2512.07472v1
- https://arxiv.org/abs/2606.08530
- https://arxiv.org/html/2606.08530v2

Positive prior: AFI reports gains from spatial affordance field intervention
under OOD scenarios and on LIBERO-Pro. GEAR-VLA reports strong generalization
from geometry-aware action representations on LIBERO, zero-shot LIBERO-Plus,
RoboTwin 2.0, and real-world settings.

Contribution type: `IMPLICIT_GAP_SOLUTION`

Scientific method: derive a nonprivileged feature-space affordance field from
LIBERO demonstrations by aligning visual features, end-effector motion, and
expert action endpoints. The field gates bounded Base-near action edits toward
demonstration-supported interaction regions only when Base appears to move
away from the local interaction manifold.

Minimal difference from prior: AFI uses explicit 3D SAFs and target/obstacle
geometry. DAF uses only demonstration-derived 2D/feature-space affordance
density available from existing LIBERO data and preserves frozen SmolVLA by
default. GEAR's 3D integration is treated as a positive geometry prior, not
claimed as reproduced.

Mechanism chain:

- problem condition: Base may replay a memorized motion when the actual visual
  interaction region differs from the training layout;
- intermediate failure mechanism: action chunks drift toward visually
  plausible but unsupported contact points;
- policy representation/action behavior: a demonstration affordance field
  identifies local regions where actions historically cause useful contact;
- expected closed-loop improvement: fewer wrong-region approaches and memory
  traps without replacing Base globally.

Data and supervision viability: existing LIBERO demonstrations contain RGB
observations, action chunks, and proprioception. The risk is that 2D
feature-space affordance is a weak proxy for AFI/GEAR's 3D geometry and may
collapse to action-density replay.

Identity-preserving integration: zero-initialized residual gate; Base
passthrough outside high-confidence interaction-field states; action caps and
clean retention required.

First serious comparison:

1. `smolvla_base`
2. `afi_saf_transparent_proxy`
3. `daf_full`
4. `daf_no_affordance_field_ablation`
5. `nearest_demo_action_density_killer`

Scores:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `17 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `84 / 100`

Rationale: DAF is scientifically interesting, but its local proxy is weaker
than the positive priors because the existing demonstrations do not directly
provide 3D SAFs or released GEAR checkpoints.

## Candidate 3: VIRG-VLA

Full name: Visual-Invariance Robustness Gate for Base-preserving SmolVLA

Closest prior: STRONG-VLA, with RobustVLA and LIBERO-Plus as secondary priors

Primary sources:

- https://arxiv.org/abs/2604.10055
- https://arxiv.org/abs/2510.00037
- https://arxiv.org/abs/2510.13626
- https://sylvestf.github.io/LIBERO-plus/
- https://arxiv.org/abs/2510.03827
- https://arxiv.org/html/2510.03827v2

Positive prior: STRONG-VLA reports gains from decoupled robustness learning
and clean realignment under multimodal perturbations across OpenVLA,
OpenVLA-OFT, and pi0 on LIBERO. RobustVLA reports LIBERO gains from
input-consistency and output-robustness objectives. LIBERO-PRO and
LIBERO-Plus document large robustness failures under controlled perturbations.

Contribution type: `CROSS_PAPER_SYNTHESIS`

Scientific method: train a Base-preserving gate that activates only when
synthetic perturbation-consistency diagnostics detect a likely visual shortcut.
The gate regularizes SmolVLA action chunks to stay invariant under
task-preserving perturbations while preserving clean Base behavior by default.

Minimal difference from prior: STRONG-VLA robustly fine-tunes the policy
through staged perturbation exposure and clean re-alignment. VIRG would keep
SmolVLA Base fixed and use a small identity-preserving gate around action
chunks, with LoRA only as optional infrastructure.

Mechanism chain:

- problem condition: apparent LIBERO competence may hide dependence on fixed
  camera, layout, background, or object appearance;
- intermediate failure mechanism: visual shortcuts produce action changes
  under task-preserving perturbations or action invariance under instruction-
  relevant changes;
- policy representation/action behavior: VIRG detects perturbation-sensitive
  action cells and gates only those cells toward clean-consistent behavior;
- expected closed-loop improvement: better robustness without sacrificing clean
  retention.

Data and supervision viability: synthetic perturbations can be generated from
existing LIBERO observations and action chunks, but prior campaign cycles
already touched perturbation replay, language contrast, and occlusion. The
local claim must avoid becoming a relabeled robustness LoRA.

Identity-preserving integration: gate initialized to zero effect; clean
retention objective; no activation unless perturbation-consistency score
exceeds a validation-frozen threshold.

First serious comparison:

1. `smolvla_base`
2. `strong_vla_perturbation_replay_proxy`
3. `virg_full`
4. `virg_no_invariance_gate_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `19 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `84 / 100`

Rationale: VIRG has good robustness priors, but overlap risk is high because
prior cycles already exercised perturbation replay, language contrast,
occlusion, and clean-retention objectives.

## Selection

Selected method: `DCCG-VLA`

Selected score: `92 / 100`

Selection decision: `DCCG_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

DCCG is selected because it is the strongest prior-anchored and locally
reproducible shift after MHS. It uses a continuous coherence mechanism rather
than collapsed binary labels, keeps LoRA out of the contribution, preserves
Base by default, and puts the closest external prior, ACG, into the first
serious comparison. Unknown empirical performance is not a rejection reason.
No DCCG proposal, implementation, training, validation search, rollout, or
confirmatory-test access has happened.
