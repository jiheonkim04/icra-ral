# Epoch 4 Cycle 22 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_HASTE_VLA`

Exactly three candidates were generated after the primary-source and local
mechanism audit. HEST remains closed unchanged.

## Candidate 1: HASTE-VLA

Name: `HASTE-VLA`, Hazard-Anchored Stage-Transition Encoding for VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: StaKe,
https://arxiv.org/abs/2606.26801.

### Scientific Method

From demonstration gripper commands, identify the next open/close transition.
For every training frame, HASTE supervises:

- a censored discrete hazard over time-to-next-transition;
- the normalized relative cumulative six-dimensional arm displacement from the
  current frame to that transition.

The auxiliary heads shape the adapted representation during training and are
not called at inference. The action flow path remains the ordinary SmolVLA
path. A zero-initialized rank-4 LoRA and clean-retention objective preserve Base
before training and penalize unnecessary drift.

Closest-prior proxy: binary stage classification plus absolute next-transition
keyframe regression on the same adapter and data scaffold.

Key ablation: relative event displacement without the hazard objective.

Strongest simple reviewer-killer: standard LoRA with the same action examples,
steps, rank, seed schedule, and clean-retention examples.

### Mechanism Chain

Frames near gripper transitions are disproportionately difficult, but ordinary
action loss treats all frames uniformly. A binary stage label does not express
remaining time, and an absolute keyframe target entangles current pose with
event geometry. Timing ambiguity and target-coordinate variance can therefore
leave the action representation weak around grasp/release boundaries.

The hazard target teaches when an event approaches; relative cumulative
displacement teaches where the arm must reach in current-centered action
coordinates. If those signals are observable, the shared adapter should reduce
near-event action error and closed-loop grasp/release failures while auxiliary
heads disappear from inference.

### Quality Screen

Provisional novelty:

- the censored hazard and relative event-displacement decomposition is
  mathematically distinct from StaKe's binary stage and absolute keyframe
  heads;
- no event memory, keyframe buffer, frame selection, action smoothing, or
  inference correction is introduced;
- novelty fails if the StaKe proxy or standard LoRA explains the gain.

Prior-anchor strength:

- StaKe directly reports positive event-structured VLA fine-tuning results;
- local labels reproduce its supervision source exactly, although public code
  was not verified;
- the comparison can share backbone, data, rank, steps, seeds, and inference
  budget.

Data and supervision viability:

- actions provide exact event and cumulative-displacement targets;
- event/censor balance, offset coverage, task coverage, target variance, and
  split overlap are auditable before training;
- auxiliary targets are training-only and require no privileged inference
  input.

Identity preservation:

- LoRA B matrices initialize to zero;
- auxiliary heads do not modify inference directly;
- clean-retention flow matching uses frozen Base targets;
- no output replacement, clipping, or runtime intervention occurs.

Decisive experiment feasibility:

- Stage 0 can test labels, Base near-event headroom, target predictability,
  objective scale, gradients, identity, and reload;
- six validation configurations are the full search budget;
- the five-policy paired comparison isolates prior, timing, and generic
  adaptation explanations.

Score:

- provisional novelty: `23 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `10 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `95 / 100`

## Candidate 2: SITE-VLA

Name: `SITE-VLA`, Spatially Indexed Transition Events for visual-trace VLAs.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest priors: TraceVLA and StaKe.

### Scientific Method

Use past-only tracked point trajectories as a TraceVLA-style visual interface,
then attach the most recent gripper event type and age to active trajectories
through a gated residual trace encoder. Compare against ordinary traces and an
event-only encoder.

### Quality Screen

- novelty is potentially meaningful because it spatially binds discrete events
  to visual motion, but it is crowded by EventVLA and KEMO;
- TraceVLA has official MIT code and checkpoints with positive results;
- local SmolVLA requires a new trace encoder and a transparent prior proxy,
  reducing comparison fidelity;
- CoTracker processing and dual-image integration increase compute and
  disruption risk;
- labels exist, but whether event type can be bound reliably to moving image
  points is uncertain.

Score:

- provisional novelty: `20 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `7 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `86 / 100`

SITE is not selected because the new interface is less identity-preserving and
has weaker local prior fidelity.

## Candidate 3: SAVR-VLA

Name: `SAVR-VLA`, Stage-Aware Visual Reward steering for frozen VLAs.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest priors: VLS and StaKe.

### Scientific Method

Condition a VLS-style differentiable geometric reward on an automatically
derived manipulation stage, then guide frozen SmolVLA flow samples only near a
predicted event boundary. Compare against unconditioned VLS guidance and a
fixed-strength reward baseline.

### Quality Screen

- the stage-conditioned reward is distinct, but the intervention remains in a
  crowded steering/correction family;
- VLS has strong positive CALVIN and LIBERO-PRO results and a code link;
- a faithful local reproduction needs RGB-D, SAM, DINOv2, VLM-generated reward
  code, multiple samples, and Feynman-Kac machinery;
- event labels exist for training, but reliable stage prediction and reward
  synthesis at deployment are separate failure points;
- inference compute and policy-disruption risk are much higher than HASTE.

Score:

- provisional novelty: `17 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `78 / 100`

SAVR is not selected because a fair local prior reproduction is not bounded.

## Selection

Select exactly one: `HASTE-VLA` with `95 / 100`.

Required next step: freeze the Researcher A proposal, Reviewer B attack,
rebuttal, mathematical audit, preregistration, and Stage 0 protocol before any
training or confirmatory access. The first gate must establish event-label
health and Base near-event headroom; exact labels alone do not authorize
training.
