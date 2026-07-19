# Epoch 6 Fresh Opportunity Refresh

Refresh date: 2026-07-19 KST
Trigger: the initial three-card portfolio exhausted the thesis hard filter.
Outcome access: no Epoch 6 policy, contact-label, training, or rollout outcome
was exposed during this refresh.

## Why the initial portfolio is exhausted

- Causal low-motion preservation is too close to the pinned VLA-Arena code,
  which already uses simulator outcomes to select progressively restored
  post-transition actions.
- Outcome-separated checkpoint/configuration reporting is too close to
  RouterVLA's outcome-disjoint cross-fitting, N-SCORE's statistically rigorous
  robot-policy comparison, and ordinary independent holdout.
- Broad privileged-contact distillation into sensor-free deployment is occupied
  by FD-VLA and HapticVLA. Only a narrower object-environment contact-topology
  formulation remains technically distinguishable, and it has not passed a
  problem/supervision gate.
- Controller-dynamics adaptation is a near-exact collision with APEX.

## Local evidence synthesis

The historical workspace does not contain an undisclosed paper-grade positive
result. The strongest apparent residuals fail at least one factual hard filter:

| Local thread | Useful fact | Why it is not an active thesis |
|---|---|---|
| AFID action-factor supervision | Two usable factors, positive oracle reduction (`0.1493`) and residual headroom (`0.1647`) on its development audit | The factor predictor failed both majority and task-phase controls, and action deltas violated the frozen bound. The exact formulation cannot be rescued after outcome exposure. |
| CAVM memory | Small discovery near-miss (`24/58` versus nearest `23/58`, Base `22/58`) | Underpowered, adaptively explored, and memory/retrieval mechanisms are crowded; it is not fresh confirmatory evidence. |
| Task75 | X-VLA and SmolVLA failed one reset and task-level expert replay succeeded | No same-reset expert headroom, no runnable second prior, one anecdotal task/reset, and no distinct method thesis. |
| Standard SmolVLA | Historical clean competence (`48/80`) and trainable local path | Reusable Base infrastructure only; competence is not a contribution. |
| Exact-state LIBERO replay | Exact state restoration and official reward/success semantics are established | Reusable problem-label and oracle infrastructure only; expert replay is not policy success. |

The natural-reset residual search remains exhausted. Controlled new conditions
are allowed, but they require a current problem and novelty basis before any
outcome access.

## Static contact-topology feasibility audit

The narrow contact-topology backup is locally instrumentable without inventing
a new sensor interface:

- LIBERO's `ControlEnv.set_init_state` calls `set_state_from_flattened`,
  `sim.forward`, success checking, post-processing, and forced observation
  regeneration.
- Robosuite exposes the active MuJoCo contacts through
  `sim.data.contact[:sim.data.ncon]` and maps geom IDs through
  `sim.model.geom_id2name`.
- Robosuite models publish collision geom sets through `contact_geoms`, and the
  robot model exposes its own contact geoms. These provide an auditable way to
  exclude every robot/gripper geom and retain only object-environment pairs.
- Stored LIBERO HDF5 demonstrations contain flattened simulator states but no
  contact labels. Labels therefore require exact state restoration and contact
  extraction; they cannot be inferred from the old 2,800-row prediction cache.

This establishes implementation plausibility only. It does not establish label
prevalence, observability, failure relevance, action headroom, or closed-loop
value.

## Current-collision additions from the refresh

- Generic VLA uncertainty or failure detection is crowded by SAFE and
  *Shifting Uncertainty to Critical Moments*; a new aggregation heuristic is
  not enough.
- Backdoor/patch defense is additionally crowded by TrustVLA, while the local
  PatchGuard formulation was already baseline-dominated.
- Proprioceptive early fusion is occupied by ThinkProprio.
- Future-kinematic/keypoint auxiliary supervision is occupied by ELAN4D.
- Cross-embodiment action-prior pretraining and embodiment-canonical geometry
  are occupied by *Learning Action Priors for Cross-embodiment Robot
  Manipulation* and GEAR-VLA.
- Active perception has expanded beyond ActiveVLA to SaPaVe and CoMe-VLA. The
  local environment still lacks a movable-camera VLA interface and the official
  ActiveVLA repository still advertises future code/checkpoint/evaluation
  release rather than runnable artifacts.

## Selection rule for this refresh

A replacement card may be selected only if independent audits establish a
materially distinct central claim and intervention, a competent runnable Base,
legal inference signals, a closed-loop path, locally obtainable supervision,
and a cheap pre-Ours falsification. If no method card survives, benchmark and
systems archetypes must be assessed by their genuine contribution burden rather
than by repackaging historical failures.

## Refresh adjudication

The benchmark/systems audit found one residual worth falsifying before method
design: stochastic VLA action sampling can be coupled to request arrival order
when an evaluation stack draws from a process-global random stream. The pinned
X-VLA source contains an unaddressed `torch.randn` action-noise draw, while the
pinned VLA evaluation harness serializes model-server requests in arrival order
and exposes sharding and batching. Those source facts establish a plausible
causal path, not a material robotics result.

An independent hard-filter adjudication returned `SELECT_FOR_PROBLEM_GATE` for
schedule-invariant stochastic VLA evaluation. The selection is conditional on
two frozen falsifications: an outcome-suppressed action-level order audit, then
only after that passes, a paired closed-loop audit showing changed robotics
outcomes and conclusions. Counter-based randomness by itself is explicitly not
treated as novel.

Current decision: `ONE_ACTIVE_THESIS_AT_PROBLEM_VERIFICATION`. No keyed-noise
method, training, claim-relevant closed-loop outcome, confirmatory-manifest
access, or paper generation is authorized. The exact active protocol is
`reports/epoch6_schedule_invariant_evaluation/problem_verification_protocol.json`.
