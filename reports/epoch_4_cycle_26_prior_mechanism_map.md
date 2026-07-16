# Epoch 4 Cycle 26 Prior Mechanism Map

Date: 2026-07-16 KST

Decision context: RAP-VLA Stage 0 closed as
`RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. Cycle 26 must not repair,
rescue, retune, or reinterpret RAP. The next method must use one genuinely new
mechanism, with LoRA only as implementation infrastructure.

## Primary-Source Anchors Checked

### ABot-M0 / ABot-M0.5

- Primary paper: `https://arxiv.org/abs/2602.11236`
- Official repository: `https://github.com/amap-cvlab/ABot-Manipulation`
- Positive prior: ABot-M0 reports Action Manifold Learning for stable,
  efficient continuous action prediction; the official repository reports
  released inference code, weights, training code, and data for ABot-M0, and
  later ABot-M0.5 results on LIBERO / LIBERO-Plus / RoboTwin-style benchmarks.
- Reproducible local proxy: learn a discovery-only low-dimensional action
  manifold from existing LIBERO demonstrations and compare Ours against a
  transparent action-manifold projection proxy under SmolVLA.
- Useful extension axis: preserve SmolVLA identity while constraining
  adaptation through a legal action manifold to reduce postprocessed action
  validity failures.

### PriorVLA

- Primary paper: `https://arxiv.org/abs/2605.10925`
- Official repository: `https://github.com/xinyuguo1566/PriorVLA`
- Positive prior: PriorVLA reports a frozen Prior Expert plus Adaptation Expert
  with expert queries, improving LIBERO, RoboTwin 2.0, and real-world results
  while updating fewer parameters than full fine-tuning.
- Official asset status: repository exists but states code will be released
  soon; no local official implementation is assumed.
- Reproducible local proxy: freeze SmolVLA as the Prior Expert, expose base
  flow/action summaries as expert queries, and train a small identity-preserving
  adaptation path.
- Useful extension axis: explicitly preserves pretrained motor priors during
  adaptation, but exact prior matching is weaker without released code.

### InternVLA-M1

- Primary paper: `https://arxiv.org/abs/2510.13778`
- Official repository: `https://github.com/InternRobotics/InternVLA-M1`
- Positive prior: InternVLA-M1 reports spatially guided VLA training, code and
  checkpoints, plus LIBERO and robot-control improvements from spatial
  grounding and spatial prompting.
- Reproducible local proxy: generate development-only waypoint/spatial
  supervision from LIBERO demonstration observations and robot proprioception,
  then condition SmolVLA adaptation on predicted spatial waypoints.
- Useful extension axis: bridge instruction-conditioned visual grounding to
  action; supervision is weaker locally because true object/pixel grounding
  labels are not guaranteed in the existing LIBERO HDF5 files.

### Robometer

- Primary paper: `https://arxiv.org/abs/2603.02115`
- LeRobot integration: `https://huggingface.co/docs/lerobot/en/robometer`
- Positive prior: Robometer reports trajectory-comparison reward modeling at
  scale; LeRobot exposes a published checkpoint that predicts dense progress
  and success from trajectory video plus task text without privileged robot
  state.
- Reproducible local proxy: use LIBERO frames and language to score progress or
  success, with synthetic temporal clips as negatives.
- Useful extension axis: reward-calibrated offline guidance could target
  progress and termination, but the existing campaign lacks true failed
  trajectories for robust preference supervision without generating synthetic
  negatives.

## Cycle 26 Design Bias

The strongest immediate lesson from RAP is that action validity is not a side
diagnostic. It can dominate an otherwise promising retrieval or residual signal.
Cycle 26 should therefore prefer a method whose core mechanism directly
constrains generated action chunks to a demonstrated, low-disruption action
support before any rollout.

This points to an ABot-M0-anchored action-manifold method as the best next
candidate family. It has a positive external prior, official assets, local
labels from existing LIBERO demonstrations, clear identity-preserving
integration, and a decisive Stage 0 audit that can fail before expensive
rollouts if the manifold is collapsed, nonacting, or harmful.
