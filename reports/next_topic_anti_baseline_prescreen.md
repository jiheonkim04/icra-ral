# Next Topic Anti-Baseline Pre-Screen

This screen must run before any new topic implementation.

## Invalid Topic Rules

A topic is invalid if:
- its first result is offline-only,
- a simple calibration, clipping, nearest, mean, random, safety, or probe heuristic baseline could plausibly solve it,
- it depends on native VLA competence before verifying that competence,
- it needs full VLA training or OpenVLA-OFT to get the first result,
- it cannot produce a rollout, replay, or direct control metric within 48 hours,
- it has no clear novelty against recent VLA/action/safety/deployment literature,
- it has no clear robotics evidence path.

## Required Pre-Screen Fields

For each candidate:
- task definition,
- latest-paper gap,
- method novelty,
- why not solved by a trivial baseline,
- strongest simple baseline that could kill it,
- first 48-hour executable test,
- exact kill criteria,
- expected simulator/data/model assets,
- whether rollout/control metric appears within 48 hours,
- why it can be RA-L-stable,
- why it might fail.

## Selection Rule

Recommend the topic with:
- highest chance of beating simple baselines,
- fastest real rollout/control metric,
- strongest novelty against current literature,
- lowest dependency on native VLA policy quality,
- clearest RA-L experiment table.

## Literature Context Checked

Recent work clusters around VLA safety evaluation, action-space design, action-chunking/latency, diffusion/action policies, object/attention guidance, and VLA surveys. The next topic must therefore avoid being just another action scaling, safety clipping, semantic target prior, or offline proxy result.

Context sources checked:
- Vision-Language-Action Safety: Threats, Challenges, Evaluations, and Mechanisms, arXiv:2604.23775.
- Demystifying Action Space Design for Robotic Manipulation Policies, arXiv:2602.23408.
- Set-Supervised Diffusion Policy: Learning Action-Chunking Diffusion through Corrections, arXiv:2606.01865.
- Real-Time Execution of Action Chunking Flow Policies, NeurIPS 2025.
- Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications.
