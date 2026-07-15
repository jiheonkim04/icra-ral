# CALA-VLA Researcher A Proposal

Date: 2026-07-15 KST

Decision: `CALA_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `CALA-VLA`, Context-Gated Action-Latent Adapter for frozen SmolVLA.

This proposal is written after the fixed G3P-VLA Stage 0 result. It does not
rescue, retune, reinterpret, or relabel G3P. It starts a new method cycle with
a different mechanism axis: action-structured latent conditioning at the
SmolVLA action interface.

## External Prior Anchor

Closest external prior: CAC-VLA, Context-Gated Action Conditioning for
Vision-Language-Action Models, https://arxiv.org/abs/2607.04816.

Positive prior result: CAC-VLA reports `98.3%` average success on LIBERO and
`89.5%` on LIBERO-Plus from predicting latent actions out of
visual-language context and injecting them into the action expert through
context-gated conditioning.

Official code/checkpoint status: no official CAC-VLA code or checkpoint is
verified in this repository. Until official equivalence is established, the
local closest-prior comparison is a faithful transparent proxy, not an official
reproduction.

Secondary priors:

- VLS: https://arxiv.org/abs/2602.03973
- World Pilot: https://arxiv.org/abs/2606.12403
- STRONG-VLA as a robustness backup prior: https://arxiv.org/abs/2604.10055
- VLA Grounder as a language-conditioning related prior:
  https://arxiv.org/abs/2607.04517

## Claim

If local SmolVLA failures include a gap between visual-language context and
multi-step motor structure, then a deployment-observable latent-action
conditioning signal injected through an identity-preserving context gate can
improve closed-loop LIBERO success beyond Base, a CAC-style latent-action
proxy, a no-context-gate ablation, and one simple task-mean latent-action
baseline.

The claim is conditional on Stage 0 proving that latent-action labels are
split-clean, noncollapsed, task/phase-covered, predictable above trivial
baselines from deployment inputs, and useful enough to justify training or
rollout.

## Evidence Partitions

`DISCOVERY`:

- inspect local action-segment structure and failure/headroom conditions;
- construct candidate latent-action labels from future demonstration action
  segments;
- design the deterministic local action-latent encoder;
- inspect latent variance, task coverage, phase coverage, and trivial
  baselines.

`VALIDATION`:

- select one latent horizon, one context-gate scale, and at most one adapter
  architecture choice;
- choose one final configuration using the preregistered validation score;
- verify clean retention, action validity, latent predictability, mechanism
  activation, full-versus-ablation difference, and action-delta bounds.

`CONFIRMATORY_TEST`:

- one frozen paired official LIBERO manifest after method, checkpoint,
  baselines, ablation, tasks, reset identities, metrics, and thresholds are
  frozen;
- no confirmatory outcome may be used to retune CALA.

## Method

Inputs at inference:

- deployment RGB observations exposed by the official SmolVLA/LIBERO path;
- deployment proprioception;
- language instruction;
- Base SmolVLA features, hidden states, or action previews when available
  through the local runner;
- no future action segment, reset identity, reward, success label, future
  observation, simulator object pose, or hidden confirmatory-test metadata.

Training-only or diagnostic labels:

- future 7D action segments from discovery/validation demonstration records;
- deterministic latent-action codes derived from those segments by a frozen
  OAT-lite encoder;
- latent codes are supervision only and are forbidden for confirmatory
  inference.

Core representation:

- current observation-language feature vector `h_t`;
- future action segment `A_{t:t+H-1}` used only during training label
  construction;
- deterministic latent action `z_t = E(A_{t:t+H-1})`;
- predicted latent action `zhat_t = P(h_t)`;
- confidence or gate input `u_t` from current deployable features;
- context gate `g_t` initialized so the adapter is exact Base passthrough.

Action conditioning:

- encode `zhat_t` with a small projection module;
- inject it into the SmolVLA action interface through a zero-initialized
  context-gated residual adapter;
- initialize and gate the adapter so initial behavior is exact Base behavior;
- bound translation, rotation, and gripper deltas separately;
- preserve action validity and clean behavior before rollout.

## Stage 0 Development Audit

Stage 0 must run before any expensive training, validation search, manifest
freeze, or rollout.

Required checks:

- zero overlap between discovery, validation, and reserved confirmatory
  identities;
- legal source inventory for RGB, proprioception, language, Base features,
  action previews, demonstration actions, and latent labels;
- explicit proof that future actions and latent labels are unavailable at
  inference;
- latent-label variance, positive/negative or high/low contrast counts, task
  coverage, phase coverage, duplicate counts, and horizon coverage;
- no all-zero, all-one, single-task, or single-phase latent targets;
- latent predictability from deployment inputs above task-mean, action-only,
  phase-only, and majority/trivial baselines;
- diagnostic oracle/headroom showing latent action information can in principle
  improve the chosen validation proxy;
- Base passthrough with initial action delta p95 exactly or approximately zero;
- action validity and bound checks by translation, rotation, and gripper;
- no training, rollout, validation search, or confirmatory-test tuning during
  Stage 0.

Stage 0 hard stops:

- `DATA_OR_SUPERVISION_FAILURE` if latent labels are unavailable, collapsed,
  duplicated, split-leaking, or not covered across tasks/phases;
- `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE` if Base, closest-prior proxy,
  and diagnostic upper bound show no usable action-latent headroom;
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` if the source gate or adapter cannot
  preserve Base behavior;
- `DESIGN_FAILURE` if latent actions are not predictable from deployment inputs
  or are fully explained by a trivial task-mean or phase-only baseline.

These stops are not closed-loop scientific kills.

## Bounded Validation Search

Default maximum: six total configurations.

Allowed factors:

- latent horizon: at most three values;
- context-gate scale: at most three values;
- adapter architecture: at most two choices;
- no combinatorial grid beyond six named configurations.

Validation score:

`score = 0.25 * latent_predictability + 0.20 * clean_retention + 0.20 * bounded_action_validity + 0.15 * mechanism_activation + 0.15 * simple_baseline_margin + 0.05 * efficiency`

The score is development-only and cannot use confirmatory-test identities or
outcomes. It must not be pure offline action L2.

## First Serious Comparison

Exactly five policies:

1. `frozen_smolvla`
2. `cac_vla_latent_action_proxy`
3. `cala_full`
4. `cala_no_context_gate_ablation`
5. `task_mean_latent_action_baseline`

`cac_vla_latent_action_proxy` is a faithful transparent local proxy unless
exact official equivalence is independently established. The task-mean
latent-action baseline must remain live through Stage A/B.

Stage A:

- approximately `10` paired episodes per policy;
- catastrophic screen only;
- no one- or two-episode permanent kill.

Stage B:

- at least `40` paired episodes per key policy;
- identical task/reset identities across policies;
- report task-balanced success, paired deltas, bootstrap CIs, paired
  wins/losses/ties, latent/gate activation, clean retention, latency, and VRAM.

## Required Ablations And Simple Baseline

Key ablation:

- remove the context gate or force non-context latent injection while keeping
  the same latent labels and training budget as matched as possible.

Closest-prior proxy:

- implement the closest feasible CAC-style latent-action conditioning path,
  labeled as a transparent proxy until official equivalence is proven.

Simple reviewer-killer:

- `task_mean_latent_action_baseline`, which supplies a task- or
  instruction-conditioned latent prototype without current observation-specific
  context. If it explains the gain, CALA must be killed.

## Mathematical Commitments

The mathematical mechanism audit must define:

- variables and tensor shapes for images, proprioception, language, Base
  features, action segment, latent action, predicted latent, gate, Base action
  chunk, and adapted action chunk;
- formula for the deterministic latent-action encoder, prediction loss,
  context gate, hidden-state residual, and bounded action-delta penalty;
- units and normalization of translation, rotation, and gripper terms;
- exact gradient path through latent predictor and adapter;
- objective term magnitudes and gradient norms on a small batch;
- full-versus-ablation and full-versus-task-mean difference metrics;
- no KL between deterministic 7D actions.

## Safety And Integrity

CALA may not:

- use future actions, latent labels, simulator object-state, target
  coordinates, reset identity, task success, future observation, or reward at
  inference;
- tune on confirmatory identities;
- reinterpret Stage 0 failure as a scientific closed-loop kill;
- rescue itself after a valid Stage B kill;
- add extra baselines before the first five-policy comparison unless a concrete
  reviewer objection makes them decision-relevant and cheaper than proceeding.

Immediate next step: Reviewer B attacks novelty, CAC source fidelity, trivial
task-mean/action-history baselines, local SmolVLA integration feasibility, and
future-action leakage risk before any implementation.
