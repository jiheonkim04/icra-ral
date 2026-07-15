# Epoch 4 Cycle 22 Prior And Mechanism Map

Date: 2026-07-15 KST

Decision: `CYCLE_22_PRIOR_MAP_COMPLETE`

## Campaign Boundary

HEST-VLA is closed unchanged as `HEST_STAGE_0A_IMPLEMENTATION_FAILURE`.
Support widening, clipping, fallback changes, rerun, Stage 0B, and HEST rescue
are forbidden. NICE and all earlier correction, verifier, timing, memory,
progress, latent-steering, robustness-retention, 3D-affordance, and structured
action formulations remain closed under their recorded scopes.

Cycle 22 must use a positive external prior, exact local supervision, an
identity-preserving integration, a bounded validation search, and one early
five-policy comparison. It may not reinterpret HEST's support failure as
closed-loop evidence.

## Positive Prior 1: StaKe

Paper: Improving Vision-Language-Action Model Fine-Tuning with Structured
Stage and Keyframe Supervision, https://arxiv.org/abs/2606.26801.

Project: https://hi-yuanxu.github.io/StaKe-Web/.

Positive result already demonstrated:

- StaKe derives stage and next-gripper-transition keyframe labels from
  demonstration gripper states without manual annotation;
- lightweight stage and keyframe heads supervise VLA representations during
  training while leaving the inference loop unchanged;
- the paper reports relative success gains of `14%` in bimanual simulation and
  `56%` on single-arm Franka tasks;
- the reported benefit grows on longer tasks with more gripper events.

Artifact status:

- the paper and detailed project method are public;
- the project code link does not expose a public implementation in the bounded
  audit, so a local implementation must be labeled a faithful transparent
  proxy, not an official reproduction.

Exact limitation extended:

- binary stage classification says which side of an event the policy occupies
  but not how soon the next transition should occur;
- predicting an absolute next-transition joint action entangles current pose,
  embodiment coordinates, and event displacement;
- a censored time-to-event hazard plus relative cumulative arm displacement
  separates when the transition should happen from where the arm should reach.

## Positive Prior 2: TraceVLA

Paper: TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for
Generalist Robotic Policies, https://arxiv.org/abs/2412.10345.

Official MIT code: https://github.com/umd-huang-lab/tracevla.

Official checkpoints: `furonghuang-lab/tracevla_7b`.

Positive result already demonstrated:

- TraceVLA uses CoTracker to overlay active past-only point trajectories;
- it reports a `10%` gain over OpenVLA on SimplerEnv and `3.5x` improvement on
  real-robot tasks;
- the official repository includes trace processing, fine-tuning, inference,
  and pretrained model paths.

Local implication:

- a visual trace extension is feasible in principle, but the local SmolVLA
  interface and compute budget do not provide a matched official TraceVLA
  checkpoint;
- adding event markers to traces would also approach EventVLA, KEMO, and StaKe;
- this route has higher integration and novelty risk than a direct StaKe
  extension.

## Novelty Boundaries: EventVLA And KEMO

EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon VLA Policies,
https://arxiv.org/abs/2606.20092, reports an average `+40%` success gain across
17 memory-requiring simulation tasks and four real-world bimanual tasks. It
predicts keyframe evidence utility and stores sparse visual events.

KEMO: Event-Driven Keyframe Memory for Long-Horizon Robot Manipulation with VLA
Policies, https://arxiv.org/abs/2606.23589, reports `+23.6%` task success and
`+34.1%` stage completion over its memory-free baseline. It detects kinematic
and visual events, stores ordered keyframe tokens, and weights transition-near
training samples.

Both occupy event-selected visual memory. Cycle 22 may not present sparse event
memory, keyframe storage, or transition-weighted frame sampling as novel. The
selected route must use no inference memory and must differ mathematically from
StaKe's binary-stage plus absolute-keyframe targets.

## Positive Prior 3: VLS

Paper: VLS: Steering Pretrained Robot Policies via Vision-Language Models,
https://arxiv.org/abs/2602.03973.

Project and code link: https://vision-language-steering.github.io/webpage/.

VLS reports a `31%` improvement on CALVIN and `13%` on LIBERO-PRO by using
VLM-generated differentiable rewards, gradient guidance, diversity, and
Feynman-Kac resampling. Its local burden is high: RGB-D grounding, SAM, DINOv2,
a capable code-generating VLM, multiple action samples, and inference-time
steering are required. It also approaches closed correction and progress-aware
guidance families.

## Local Data Audit

Official LIBERO HDF5 demonstrations contain, per timestep:

- `agentview_rgb` and `eye_in_hand_rgb`, each `128 x 128 x 3`;
- 7D postprocessed actions;
- end-effector pose, gripper state, joint state, robot state, and simulator
  state.

Gripper transition labels and cumulative arm displacement to the next event can
be derived from action records alone. HEST Stage 0A independently observed `23
/ 32` transition-containing validation windows across the four fixed task
families, so event supervision is not globally collapsed. Privileged state is
not required by the selected targets or by inference.

## Cycle 22 Opportunity

The best bounded opportunity is `HASTE-VLA`, Hazard-Anchored Stage-Transition
Encoding for VLA fine-tuning.

HASTE replaces StaKe's binary stage and absolute keyframe regression targets
with:

1. a censored discrete hazard over the next gripper transition offset;
2. a normalized relative cumulative six-dimensional arm displacement to that
   event.

Both heads are training-only. A rank-4 LoRA adapter is zero-effect at
initialization, and clean-retention flow matching constrains drift. The first
comparison is Base, a faithful StaKe proxy, HASTE, HASTE without hazard, and
data-matched standard LoRA.
