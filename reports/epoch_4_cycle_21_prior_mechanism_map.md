# Epoch 4 Cycle 21 Prior And Mechanism Map

Date: 2026-07-15 KST

Decision: `CYCLE_21_PRIOR_MAP_COMPLETE`

## Campaign Boundary

NICE-VLA is closed unchanged as
`NICE_STAGE_0B1_DATA_FAILURE_COLLAPSED_ACTION_REGIME_CONTRAST`. Its frozen
deadband, tasks, pairs, diagnostics, and method may not be rescued. Cycle 21
must also avoid the closed action-correction, verifier/ranker, memory,
progress, adaptive-chunk, affordance/3D, robustness-retention, latent-action,
conceptor, and uncertainty-correction formulations in the campaign ledgers.

The latest Windows gaming and Efficiency Mode interval is recorded in
`reports/resource_contention_intervals.json`. Timing, throughput, wall-clock,
and resource-utilization values with unknown or positive overlap are ineligible
for paper evidence.

## Positive Prior 1: Spline Policy

Paper: Spline Policy: A Structured Representation for Robot Policies,
https://arxiv.org/abs/2606.07386.

Positive result already demonstrated:

- the paper instantiates a spline output with transformer, diffusion,
  flow-matching, and VLA-style backbones;
- matched-backbone simulation keeps task scores in a comparable range while
  reducing output dimensionality;
- the spline exposes temporal resampling, derivative constraints, local
  trajectory editing, uncertainty propagation, and controller integration;
- real-robot deployments report `10 / 10` PushT and `9 / 10` toy-packing
  successes for the SP-pi0.5 example, while explicitly avoiding an
  architecture-superiority claim.

Artifact status:

- the primary paper provides the construction and analytic flow-field
  derivation;
- no official public implementation was verified in the bounded search;
- any local arm must therefore be labeled a transparent analytic proxy, never
  an official reproduction.

Exact limitation extended:

- the paper represents the policy output as a continuous spline but states
  that the representation may be unsuitable for highly discontinuous
  interactions;
- a 7D manipulation action mixes six continuous arm coordinates with a
  discontinuous gripper command;
- smoothing all seven coordinates treats a mode switch as if it were another
  continuous trajectory coordinate.

Local opportunity:

- represent cumulative six-dimensional arm motion with an endpoint-constrained
  spline;
- preserve the gripper command as an exact discrete event stream;
- reject the entire transformed chunk and return Base if any action validity
  check fails;
- compare first against an all-channel spline proxy and a simple moving-average
  arm smoother on direct exact-state replay/control fidelity.

This route is not the closed ACoT/CAC latent-action family: it does not learn an
intermediate action code or a new candidate generator. It is also not adaptive
chunk length or phase retiming: the horizon, observation schedule, queue
semantics, and action count remain fixed.

## Positive Prior 2: Set-Supervised Diffusion Policy

Paper: Set-Supervised Diffusion Policy: Learning Action-Chunking Diffusion
through Corrections, https://arxiv.org/abs/2606.01865.

Official code: https://github.com/ZhaotingLi/Set_Supervised_DP, MIT license.

Positive result already demonstrated:

- SDP reports consistent gains over ordinary diffusion-policy training in
  simulation, offline learning, and real-robot correction learning;
- the strongest gains occur under noisy correction data;
- the official repository includes offline and interactive training paths.

Local exclusion:

- ISAC-VLA was already rejected because paired negative policy chunks and
  positive corrective chunks are near-exactly occupied by SDP and local paired
  human correction data are unavailable;
- SDP also supports synthetic negative actions for ordinary demonstrations;
- changing an isotropic desired set into a task-scaled ellipsoid is therefore
  a narrow objective variation, not the strongest Cycle 21 novelty route.

This prior may appear as a rejected candidate anchor but may not be presented
as a fresh intervention-set contribution.

## Positive Prior 3: ACoT-VLA

Paper: ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models,
https://arxiv.org/abs/2601.11404.

Official repository: https://github.com/AgibotTech/ACoT-VLA, Apache-2.0.

Positive result already demonstrated:

- ACoT-VLA reports `98.5%` LIBERO average, `84.1%` LIBERO-Plus for the frozen
  variant, and `47.4` VLABench progress score;
- its explicit action reasoner predicts a coarse action trajectory and its
  implicit action reasoner supplies latent action guidance.

Local exclusion:

- the official repository still lists the core EAR/IAR release and LIBERO
  configuration/checkpoints as pending;
- the campaign already closed coarse action conditioning and latent action
  representation against CAC-VLA, ACoT-VLA, LaRA-VLA, ActionMap, and local
  ECHO/ActionMap no-headroom evidence;
- an orthogonal coarse/residual decomposition would remain too close to this
  closed family without new direct headroom.

## Adjacent Prior: RTR

Paper: Learning High-Frequency Continuous Action Chunks in Latent Space,
https://arxiv.org/abs/2605.24931.

Official code: https://github.com/tars-robotics/RTR, MIT license.

RTR positively demonstrates latent high-frequency action decoding and
Reuse-then-Refine under asynchronous real-robot execution. It is not selected
as a lead anchor because the local LIBERO simulator is synchronous, the frozen
SmolVLA protocol does not supply a `60 Hz` deployment claim, and manufacturing
an asynchronous/high-frequency condition would revisit closed delay and chunk
timing axes.

## Cycle 21 Opportunity

The only candidate in this screen that is both materially distinct from closed
local families and directly testable without new labels, a second resident VLA,
or privileged inference input is a hybrid continuous-discrete spline action
interface. Its first evidence must be controller-facing exact-state replay, not
offline action L2 alone. The method must beat both the closest all-channel
spline proxy and one simple arm smoother while preserving clean actions and
gripper events.
