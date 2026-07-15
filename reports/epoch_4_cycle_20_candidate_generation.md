# Epoch 4 Cycle 20 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_NICE_VLA`

Exactly three candidates were generated and scored under the active
performance-oriented governance. SPARC-VLA remains frozen and closed. None of
these candidates changes or reruns SPARC.

## Candidate 1: NICE-VLA

Name: `NICE-VLA`, Normalized-Innovation Corrective Execution for VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: VLA-Corrector,
https://arxiv.org/abs/2607.01804.

Official code:
https://github.com/ZJU-OmniAI/vla-corrector at inspected commit
`9d23a0ba6fad562d3ed1a68fc52c8a12459abb41`.

Positive prior result: VLA-Corrector reports `+4.75` success points with
SmolVLA on MetaWorld, `+3.80` with PI0.5 on LIBERO few-shot, `+15.65` with
PI0.5 on MetaWorld, and `+17.7` on a real AgileX PiPER robot. It includes an
official LeRobot-based implementation.

### Scientific Difference

VLA-Corrector predicts a mean latent residual and detects drift with one
rolling cosine-error threshold. NICE preserves the official mean predictor,
queue truncation, and OGG recovery contract but replaces the monitor with an
action- and phase-conditioned predictive distribution.

NICE adds a heteroscedastic covariance head over the latent residual and scores
the observed innovation after normalization by predicted uncertainty. A
split-conformal quantile frozen on validation identities converts the score to
one trigger without using confirmatory data. This is a new statistical policy
monitor, not a changed cosine threshold.

### Mechanism Chain

- condition: expected visual dynamics vary across free-space motion, contact,
  grasp closure, transport, and release;
- prior failure mechanism: one global rolling cosine-MAD score treats all
  residual directions and scales alike;
- monitor behavior: expected high-variance motion can cause false interrupts,
  while a small but unlikely low-variance error can remain undetected;
- closed-loop consequence: unnecessary truncation creates discontinuity and
  missed drift leaves stale actions in the queue;
- NICE mechanism: predict mean and diagonal or low-rank-plus-diagonal latent
  residual covariance from legal deployment inputs;
- intended internal change: whiten residuals into normalized innovations and
  calibrate one validation-frozen conformal quantile;
- intended policy behavior: preserve the exact Base queue in expected states,
  then invoke the same prior recovery only for statistically unlikely drift;
- expected result: higher paired success and better precision/recall of
  critical interrupts than VLA-Corrector, the mean-only ablation, and fixed
  short-horizon replanning.

### Data And Supervision Viability

- `130` local task HDF5 files provide two RGB streams, 7D actions, episode
  boundaries, and `50` demonstrations per task;
- within-episode `k`-step visual residual targets are noncollapsed by
  construction and require no failure label;
- task, episode, and reset partitions can be frozen before extraction;
- raw simulator state is not an inference input;
- the same frozen visual encoder and action semantics are used for the prior,
  Ours, and ablation;
- the official repository provides extraction, mean-dynamics, training, and
  evaluation contracts but no pretrained corrector, so local training is
  required and fully reportable.

### Identity-Preserving Integration

- frozen SmolVLA weights never change;
- no trigger means the official Base queue and actions are bitwise unchanged;
- the covariance head only changes the interrupt decision;
- recovery uses the same bounded VLA-Corrector OGG path for the prior and Ours;
- action validity, Base-relative deltas, trigger rate, cooldown, and clean
  retention are hard pre-rollout gates.

### Bounded Search

At most six validation configurations:

- covariance architecture: diagonal or diagonal plus rank-8;
- conformal coverage: `0.90`, `0.95`, or `0.975`;
- all other mean-dynamics, persistence, cooldown, extraction, and OGG settings
  are frozen to the matched prior contract.

The preregistered validation score must combine closed-loop validation success
or the closest legal proxy, clean retention, interrupt calibration, action
validity, mechanism activation, and overhead. Offline action L2 alone cannot
select the configuration.

### Decisive Experiment

Stage 0 must establish partition integrity, latent-pair health, reload,
finite/nonzero gradients, covariance noncollapse, held-out calibration,
diagnostic mismatch detection, Base passthrough, bounded recovery actions, and
no confirmatory reads.

The first serious comparison uses exactly five policies:

1. `smolvla_base_fixed_horizon`
2. `vla_corrector_official_proxy`
3. `nice_full`
4. `nice_mean_only_global_error_ablation`
5. `fixed_short_horizon_replan`

The fifth policy is the strongest simple reviewer killer. No sixth internal
control is preregistered before the prior-first comparison.

### Score

- provisional novelty: `23 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `20 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `96 / 100`

## Candidate 2: WISP-VLA

Name: `WISP-VLA`, Weight-Inferred Safe Policy adapters for VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: WIZARD,
https://arxiv.org/abs/2606.07217.

Positive prior result: WIZARD reports up to about `2x` improvement on unseen
LIBERO dataset collections, up to about `14x` on unseen tasks, and real-robot
average success `0.33` versus `0.17`.

Scientific method: preserve WIZARD's instruction-plus-video weight inference,
but project each generated LoRA update into a development-only functional
trust region measured by Base-relative action changes. A zero gate returns the
frozen Base adapter exactly; a generated update is accepted only when a
separate calibration head predicts clean retention.

Mechanism chain: task evidence identifies a weight-manifold region; direct
weight generation can have correct cosine direction but harmful scale;
functional calibration bounds the induced action change; the generated
adapter specializes without global Base disruption.

Data and supervision audit: the local repository has only one full target LoRA
endpoint rather than a broad task-expert meta-dataset. Creating sufficient
task experts would require a large new training campaign. The paper/project
audit found no official code release. Required weight targets therefore do not
currently pass the local data gate.

Identity preservation: exact Base passthrough is possible through a zero
acceptance gate, but this does not repair the missing expert-target coverage.

Decisive experiment feasibility: a weight-space leave-one-task-out audit would
be decisive after at least sixteen independent task experts exist. That input
condition is not met now.

Score:

- provisional novelty: `23 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `3 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `84 / 100`

## Candidate 3: PRISM-Harness-VLA

Name: `PRISM-Harness-VLA`, Primitive-Range Inference and Skill Multiplexing for
frozen VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Harness VLA,
https://arxiv.org/abs/2607.08448.

Positive prior result: Harness VLA reports `+38.6` points on LIBERO-Pro,
`+25.4` points on RoboCasa365, and `58.4%` on RoboTwin C2R.

Scientific method: learn a calibrated operating-range classifier for a fixed
set of non-contact primitives and the frozen SmolVLA contact primitive. Add a
reversible handoff state that returns to SmolVLA when grounding confidence or
primitive reachability is low.

Mechanism chain: a monolithic VLA spends capacity on semantic grounding and
non-contact transport; a fixed analytic primitive can execute those phases;
an operating-range classifier selects only legally grounded primitives;
SmolVLA remains responsible for irregular contact-rich phases.

Data and supervision audit: local HDF5 trajectories can weakly label phase
transitions, but the runner lacks deployment-observable object grounding,
reachability, and analytic staging primitives. Simulator state would make the
prototype easy but would be a prohibited privileged inference input. No
official Harness code was found in the primary-source audit.

Identity preservation: the default router can select frozen SmolVLA, but the
analytic branch cannot yet be evaluated fairly without legal grounding.

Decisive experiment feasibility: a bounded primitive coverage and oracle audit
is possible, but a policy comparison is not feasible until the non-privileged
primitive interface exists.

Score:

- provisional novelty: `20 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `4 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `81 / 100`

## Selection

Select exactly one candidate: `NICE-VLA`, `96 / 100`.

Selection reason:

- it has direct positive SmolVLA evidence and an official LeRobot code path;
- its local supervision exists at scale and uses deployment-observable inputs;
- its difference from VLA-Corrector is a falsifiable predictive-distribution
  mechanism, not a renamed threshold;
- it changes the monitor signal and calibration relative to EAC and RCV while
  preserving the closest prior's recovery path for a matched comparison;
- it can fail cheaply at Stage 0 on covariance collapse, poor calibration,
  no diagnostic detection gain, or unsafe recovery actions;
- WISP lacks task-expert weight targets and PRISM lacks legal analytic
  primitive inputs.

## Frozen Next Boundary

Researcher A must now write one NICE proposal. Reviewer B must independently
attack novelty, mathematical validity, data partitioning, action safety, and
the EAC/RCV rescue risk. No implementation or labeled latent extraction begins
until proposal hash, rebuttal, mathematical audit, preregistration, and
prototype protocol are frozen.
