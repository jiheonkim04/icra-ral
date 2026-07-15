# Epoch 4 Cycle 13 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the next method after `CALA-VLA` stopped before rollout as
`DESIGN_FAILURE`. CALA is not a closed-loop scientific kill, but its fixed
Stage 0 stop is valid for that protocol. It must not be rescued by changing
latent labels, prediction features, thresholds, source gates, validation
configs, or baseline interpretation.

## Local Constraints From Prior Results

The next method must not be:

- another future-action latent prediction or context-gated latent adapter rescue
  of CALA;
- another point-label, material-point, 3D-grounding, or waypoint source rescue
  of G3P;
- another adaptive chunking, queue commitment, entropy scheduler, fixed-replan,
  or retained-frame variant of EAC, RCV, or MTF;
- another median-anchor, static L1 mixture, component-route residual, reflective
  consequence wrapper, nearest-memory replay, or output-action correction route;
- another photometric-only perturbation ensemble or point/source threshold
  rescue.

CALA's failed audit is still informative: the strongest trivial predictor for
future action-latent structure was `action_history_only`. Cycle 13 should not
pretend causal action history is unimportant. A viable candidate may use action
history, but it must beat a simple action-history baseline under the frozen
protocol.

## Close Sources

### AR-VLA

Full title: AR-VLA: True Autoregressive Action Expert for Vision-Language-Action
Models.

URL: https://arxiv.org/abs/2603.10126

AUTHOR_STATED:

- AR-VLA proposes a standalone autoregressive Action Expert that generates a
  continuous causal action sequence while conditioning on refreshable
  vision-language prefixes.
- It explicitly targets the mismatch between fast control and slower reasoning,
  keeps long-lived action memory, and uses re-anchoring to account for
  perception staleness.
- The paper reports smoother trajectories while maintaining or exceeding
  state-of-the-art reactive VLA task success, and lists code/videos through the
  project website.

INDEPENDENTLY_INFERRED:

- The positive prior is action-generation structure, not generic action
  smoothing. The key mechanism is a causal action expert whose memory persists
  across observation refreshes while being re-anchored to fresh perceptual
  context.
- A local version should not replace SmolVLA's action head from scratch. The
  feasible extension is a small identity-preserving autoregressive residual or
  hidden-state adapter around frozen SmolVLA that defaults to Base and only
  acts when a causal history state predicts a bounded correction.
- Because CALA found action-history-only to be a strong trivial baseline, any
  AR-style method must compare directly against a frozen exponential smoothing
  or linear action-history residual baseline.

### ReactVLA

Full title: ReactVLA: Fast and Lightweight Reactive Robot Manipulation via
Improved Mean Flow Action Generation.

URL: https://arxiv.org/abs/2606.14255

AUTHOR_STATED:

- ReactVLA targets diffusion/VLA inference latency, combining improved Mean Flow
  action generation with Attention Residuals.
- It reports large-scale simulation and real-robot results, including
  outperforming similarly sized baselines such as SmolVLA and `pi0`, improving
  precision manipulation performance, and increasing inference speed.

INDEPENDENTLY_INFERRED:

- The strongest transferable lesson is that flow-action generation and residual
  feature routing can be made more reactive without abandoning the VLA action
  prior.
- Local feasibility is uncertain: a faithful implementation may need SmolVLA
  flow internals, multi-step action-sampling hooks, and a way to verify that
  any mean-flow correction is not just deterministic action smoothing.

### ABot-M0

Full title: ABot-M0: VLA Foundation Model for Robotic Manipulation with Action
Manifold Learning.

URL: https://arxiv.org/abs/2602.11236

AUTHOR_STATED:

- ABot-M0 proposes the Action Manifold Hypothesis: effective robot actions lie
  on a low-dimensional smooth manifold governed by physical and task
  constraints.
- Its Action Manifold Learning shifts learning from denoising toward predicting
  clean continuous action sequences on that manifold, improving decoding speed
  and policy stability.
- The paper reports code/pipeline release intent and additive benefits from
  modular components.

INDEPENDENTLY_INFERRED:

- The useful mechanism is not PCA alone; it is a representation constraint that
  projects action generation toward smooth physically feasible sequences.
- A local method must avoid reviving the killed local ActionMap route. It must
  include Base-preserving integration, a simple PCA/EMA projection killer
  baseline, and a Stage 0 headroom audit proving Base failures actually contain
  out-of-manifold action discontinuities.

### DSWAM

Full title: DSWAM: A Dual-System World Action Foundation Model for Fine-Grained
Robot Manipulation.

URL: https://arxiv.org/abs/2607.04927

AUTHOR_STATED:

- DSWAM contrasts world-action models with VLAs and combines a default WAM
  executor with an optional vision-language subtask planner.
- The executor uses action prediction and video co-training, while inference
  directly predicts action chunks without explicit future video generation.
- It adds asynchronous execution and real-time chunking for practical robot
  control.

INDEPENDENTLY_INFERRED:

- DSWAM is a strong conceptual prior for when language-level decomposition or
  world-aware execution is the missing mechanism.
- Local feasibility is weak for this campaign stage because no WAM checkpoint,
  video co-training corpus, or matched executor is available. DSWAM should
  inform reviewer questions, but it should not be selected unless a lightweight
  proxy can be decisive without pretending to reproduce the full world-action
  model.

## Cycle 13 Opportunity

The strongest immediate opportunity is `RAR-VLA`: Re-Anchored Autoregressive
Residuals for frozen SmolVLA.

It is anchored primarily to AR-VLA. The local extension is not a full AR-VLA
reproduction. It is a frozen-backbone, identity-preserving residual action
memory that uses only causal deployment-time information:

- current RGB/proprioception/language and Base action chunk;
- a bounded memory of previously emitted Base/Ours actions;
- a re-anchoring update when a new Base chunk arrives;
- zero-initialized residual/gate so initial behavior equals Base;
- no future actions, success labels, reset identities, simulator object state,
  or confirmatory outcomes at inference.

The critical Reviewer B baselines are:

- unmodified frozen SmolVLA;
- `ar_vla_reanchored_expert_proxy`, a faithful transparent proxy for the closest
  AR-style prior unless official local equivalence is established;
- `rar_full`;
- `rar_no_reanchor_memory_ablation`;
- `ema_action_history_baseline`, the simple reviewer-killer that tests whether
  ordinary causal smoothing/history alone explains any gain.

The Stage 0 gate must prove:

- Base has measurable action discontinuity or temporal inconsistency headroom
  on development identities;
- a deployment-observable causal memory predicts residual structure above the
  EMA/linear history baseline;
- the residual is bounded and identity-preserving at initialization;
- clean behavior is retained;
- no CALA latent labels or future action segments are used at inference.
