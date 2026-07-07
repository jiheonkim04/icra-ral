# Next Topic Candidates V3

No topic is implemented here. This is a filtered pre-screen only.

Recent related-work context checked:
- VLA deployment metadata/action-spec safety: `https://arxiv.org/html/2606.03724v1`
- VLA safety survey: `https://arxiv.org/abs/2604.23775`
- ForesightSafety-VLA diagnostic benchmark: `https://arxiv.org/html/2606.27079v2`
- Neuro-symbolic VLA safety guidance: `https://arxiv.org/html/2607.01378v1`
- PACE phase-aware chunk execution: `https://arxiv.org/html/2606.00537v1`
- action-space design study: `https://arxiv.org/abs/2602.23408`

## Candidate 1: Post-Intervention Resume-Point Selection

- task definition: after a safety stop, user stop, or stalled chunk, resume a manipulation trajectory from the current simulator state without resetting the task.
- novelty hypothesis: recovery requires joint reasoning over task phase, gripper state, EEF-object relation, and object progress; fixed rewind, nearest state, gripper reset, or raw next-action replay each handles only one part.
- strongest recent related work: VLA safety/deployment papers emphasize runtime safety and action execution, while PACE addresses when to replan chunks; resume after interruption is a different post-intervention recovery problem.
- strongest simple baseline suite: raw next action, no-op/abort, fixed rewind, fixed forward skip, nearest-progress demo, gripper reset, repeat-last/hold, safety-only stop, global scale, diagonal affine, event-locked retiming.
- why per-failure-mode simple baselines should not solve it: the first diagnostic must combine stop/hold, gripper-state mismatch, and phase offset so that a gripper-only, time-only, or nearest-progress rule cannot satisfy success/progress and action-validity metrics together.
- first 24-48 hour executable test: exact-init expert replay on one LIBERO task, inject bounded stop/hold interventions at pregrasp, postgrasp/prelift, and preplace checkpoints, then compare resume strategies on reward, success, done index, EEF-object progress, object movement, clip rate, and recovery over raw.
- kill criteria: kill if fixed rewind, nearest-progress demo, gripper reset, or event-locked retiming matches the proposed resume selector; kill if interventions do not degrade replay; kill if only offline resume-index accuracy improves.
- RA-L stability estimate: medium-high if the combined intervention cases create a simple-baseline-resistant recovery gap and then transfer across tasks/models.
- expected implementation risk: medium; state perturbation beyond hold/stop may require careful simulator-state handling, so the first test should start with non-destructive stop/hold interventions.
- recommended/not recommended: recommended.

## Candidate 2: Coupled Progress-Safety Arbitration For Candidate Chunks

- task definition: choose or minimally edit candidate action chunks so task progress, target-object specificity, and physical safety are all preserved under a single replay/control metric.
- novelty hypothesis: safety-only filters, progress-only heuristics, and nearest-target heuristics fail different axes; a valid method must optimize the coupled tradeoff rather than win one isolated metric.
- strongest recent related work: ForesightSafety-VLA and neuro-symbolic VLA safety guidance emphasize embodied safety and predictive constraints; the local gap is a tiny baseline-first LIBERO arbitration diagnostic with task-progress preservation as a required metric.
- strongest simple baseline suite: safety-only stop/filter, clipping-only, no-op, progress-only EEF-object heuristic, nearest-target, random candidate, action-norm threshold, single-step CBF-style filter, exact expert replay upper bound.
- why per-failure-mode simple baselines should not solve it: the first table must require simultaneous safety, progress, and intended-object movement; any baseline that is safe but deadlocked or progressive but unsafe fails the primary metric.
- first 24-48 hour executable test: construct candidate chunks from local HDF5 expert actions plus simple unsafe/wrong-direction perturbations, replay exact-init candidates, and score coupled success/progress/safety metrics without learned policy inference.
- kill criteria: kill if safety-only plus no-op matches coupled score, if progress-only matches without safety cost, if nearest-target matches intended-object progress, or if no replay/control metric appears.
- RA-L stability estimate: medium; strong if it shows coupled-metric gains across SafeLIBERO-like tasks, weaker if local hazards are too synthetic.
- expected implementation risk: medium-high; reliable safety/hazard metrics may need careful object/contact extraction.
- recommended/not recommended: not recommended first; keep as second option after resume-point selection because safety metrics may take longer to make non-toy.

## Candidate 3: Executable Policy Identity Replay Certificate

- task definition: certify whether a checkpoint, processor, action unnormalizer, gripper convention, controller frequency, and chunk execution rule instantiate the same executable robot policy before rollout scaling.
- novelty hypothesis: the failure is a composed deployment identity mismatch; one-at-a-time range checks or calibration checks can miss coupled action-spec interactions.
- strongest recent related work: Same Weights, Different Robot directly argues that action-space metadata is part of the executable policy, while action-space design work shows action conventions can dominate manipulation performance.
- strongest simple baseline suite: config checksum, action range check, no-op filtering, raw-vs-correct replay, gripper flip, mask/gripper check, global scale, diagonal affine, fixed chunk horizon, exact-init expert replay upper bound.
- why per-failure-mode simple baselines should not solve it: the first diagnostic must include coupled mismatches where every single-mode check passes but the composed replay/control behavior changes.
- first 24-48 hour executable test: use existing exact-init replay infrastructure to run a small paired spec-perturbation matrix and measure executable-equivalence failure, reward/success drift, action L2, clip rate, and gripper/object progress.
- kill criteria: kill if config/range/gripper/global-scale/diagonal-affine checks detect every failure, or if the result collapses to the prior ExecSpec calibration story.
- RA-L stability estimate: medium; the deployment framing is timely, but novelty risk is high because recent work is very close.
- expected implementation risk: low-medium; much infrastructure already exists, but avoiding overlap with killed ExecSpec is hard.
- recommended/not recommended: not recommended first because it is close to both recent deployment-safety work and the killed ExecSpec route.

## Recommendation

Recommended next candidate: Candidate 1, Post-Intervention Resume-Point Selection.

Reason: it has the fastest route to a real replay/control metric using existing exact-init infrastructure, targets a coupled post-intervention failure rather than separable phase/gripper/scale failures, and has an explicit first table where fixed rewind, nearest-progress demo, gripper reset, safety-only stop, global scale, diagonal affine, and the killed event-locked retimer can all kill the idea early.
