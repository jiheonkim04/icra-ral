# Epoch 4 Cycle 10 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the next method after `PESA-VLA` stopped at Stage 0 as a pre-rollout `DESIGN_FAILURE`. PESA is not a valid closed-loop method kill and must not be rescued by relabeling prior queries, lowering the query-probe margin, changing spectral thresholds, adding variants, or launching PESA validation search, training, or rollout.

## Local Constraints From Prior Results

The next method must not be:

- another prior-query, spectral-capacity, or PriorVLA-style PESA rescue;
- another median-anchor, static L1 mixture, or disagreement-gated MARC rescue;
- another dynamic arm/gripper route residual like DAGR;
- another milestone-frame sampling or retained-frame MTF rescue;
- another reflective consequence-calibration wrapper like RAC;
- another failure-aware residual field like FANG;
- another action-evolved state controller like EvoState;
- another nearest-memory contrastive action method like CAVM;
- another no-context receding-chunk replanner like RCV;
- another photometric perturbation ensemble like PSE;
- an ActionMap-style local proxy unless it first clears an official-source gate.

The repeated closed-loop pattern is now sharp: learned action wrappers and adapters can pass offline or validation checks yet still harm a strong queued SmolVLA policy. Cycle 10 therefore prioritizes methods that preserve the actual emitted 7D actions and change a different control surface first. The official rollout path exposes a `50 x 7` SmolVLA action chunk and an action queue, so adaptive chunk commitment is a real mechanism surface rather than a retrospective scoring trick.

## Close Sources

### Adaptive Action Chunking

Full title: Adaptive Action Chunking at Inference-time for Vision-Language-Action Models.

URLs:

- paper: https://arxiv.org/abs/2604.04161
- project/code index: https://lance-lot.github.io/adaptive-chunking.github.io/ and https://github.com/orgs/Adaptive-Action-Chunking/repositories

AUTHOR_STATED:

- Fixed VLA action chunking trades reactivity against temporal consistency.
- Small chunks can induce mode-jumping and jerky discontinuities; large chunks can delay response to new observations.
- AAC uses action entropy from current predictions to adapt chunk size at inference time.
- The paper is accepted by CVPR 2026 and reports substantial improvements over state-of-the-art alternatives across simulated and real-world manipulation tasks.
- The project page links public code and states the core rule: high entropy uses smaller chunks for reactivity; low entropy uses larger chunks for smoother commitment.

INDEPENDENTLY_INFERRED:

- The causal claim is not another action residual. It is an inference-scheduling claim: when to trust a predicted chunk before refreshing perception.
- A local faithful proxy is possible because the official SmolVLA/LIBERO runner already uses a policy action queue and produces full postprocessed action chunks.
- The direct external prior uses entropy over model predictions. Local SmolVLA uncertainty must therefore be verified from deployment-observable action-distribution signals, such as multi-noise flow samples, chunk self-consistency, or action-queue boundary instability. If the entropy proxy collapses, the method must stop before rollout.
- The strongest local reviewer killer is a simple fixed short replan interval or no-context replan baseline, because RCV already showed that naive replanning can outperform a more elaborate queued-validity mechanism.

CROSS_PAPER_SYNTHESIZED:

- RCV showed that changing chunk freshness can matter, but its learned current-state validity mechanism lost to no-context/stateless baselines.
- AAC supplies a stronger positive prior: chunk length should be controlled by action uncertainty, not by a generic learned wrapper or a fixed cadence.
- AR-VLA and AC2-VLA both reinforce that temporal/action context and refresh cadence are core VLA deployment variables, but EAC should first test the minimal AAC-like control surface: queue commitment length.

Mechanism fields:

- observation/input: official SmolVLA RGB/proprio/language batch, postprocessed `50 x 7` action chunk, optional repeated stochastic action chunks from the same deployment observation, current queue length, previous executed action, and no reward, simulator state, reset identity, or future observation;
- learned representation: either no learned representation or a validation-calibrated scalar chunk-commitment risk score built from action entropy, chunk variance, adjacent-step discontinuity, and queue boundary instability;
- supervision: development-only calibration labels derived from action-distribution uncertainty and action-smoothness diagnostics; no confirmatory success labels for tuning;
- objective: validation score over closed-loop proxy or bounded pilot, clean retention, mechanism activation, action validity, latency, and simple-baseline comparison; no KL over deterministic 7D actions;
- policy component changed: action-queue scheduling only, not SmolVLA weights or emitted action values;
- action-generation mechanism: use the exact frozen SmolVLA action values, but execute only a selected prefix length before refreshing the observation and generating a new chunk;
- inference-time intervention: choose commitment length from a frozen set such as `{1, 2, 4, 8, 16, 50}` using the frozen risk rule;
- assumed feedback: current observation and action predictions only;
- benchmark condition: official paired LIBERO manifest after Stage 0 and preflight pass;
- primary metric: task-balanced official closed-loop success, paired deltas, action chunks generated, latency, and clean retention;
- demonstrated causal link externally: action-entropy-conditioned chunk sizing improves VLA manipulation performance across multiple simulated and real-world settings;
- untested causal link locally: whether SmolVLA's action-distribution uncertainty is noncollapsed and whether adaptive queue commitment beats Base, an AAC proxy, a key ablation, and a fixed short-replan reviewer killer.

### Direct 3D Grounded Point Injection

Full title: Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization.

URL: https://arxiv.org/abs/2606.27663

AUTHOR_STATED:

- VLA test-time brittleness appears under spatial and task generalization shifts.
- A 3D grounded point represented as relative displacement to the gripper and injected into the action head through adaptive layer normalization is the key intervention.
- The module is lightweight and model-agnostic.
- The paper reports large LIBERO-PRO gains for GR00T-N1.6 and comparable gains for `pi_0.5`.

INDEPENDENTLY_INFERRED:

- The strongest causal prior is spatial grounding at the action head, not language prompting or visual prompting.
- A local method cannot use simulator object-state directly at inference. Object-state can only be a training label or oracle diagnostic.
- A feasible local proxy requires a deployment-observable 3D point predictor from RGB/proprio/language or a frozen external grounding source that is legal under repository governance.
- If point prediction is not above trivial baselines or the point injection behaves globally, the method must stop as a data/supervision failure before rollout.

CROSS_PAPER_SYNTHESIZED:

- GEAR-VLA and AffordanceVLA agree that geometry-aware intermediate representations can improve VLA generalization.
- Direct action-head injection is more locally compatible than full geometry-aware pretraining, but still needs a non-privileged grounding source.
- This family is strong but less immediately runnable than EAC because the current stable prediction artifact contains actions and proprioception, not validated RGB-to-3D target labels.

### AR-VLA

Full title: AR-VLA: True Autoregressive Action Expert for Vision-Language-Action Models.

URL: https://arxiv.org/abs/2603.10126

AUTHOR_STATED:

- AR-VLA introduces a standalone autoregressive action expert that generates continuous causal action sequences while conditioning on refreshable vision-language prefixes.
- It maintains long-lived action history instead of resetting temporal context with every new observation.
- It uses re-anchoring to account for perception staleness.
- The paper reports smoother trajectories while maintaining or exceeding state-of-the-art reactive VLA success, and code/videos are linked.

INDEPENDENTLY_INFERRED:

- The positive prior is a stronger temporal action-generation model, not merely output smoothing.
- A local full reproduction would replace or augment SmolVLA's action head and is therefore higher disruption than EAC.
- A lightweight local proxy could re-anchor Base chunks with a small causal composer, but it must be initialized as exact Base passthrough and compared to low-pass smoothing or fixed short replanning.

CROSS_PAPER_SYNTHESIZED:

- AR-VLA motivates temporal action syntax and re-anchoring, while AAC motivates adaptive refresh timing.
- If EAC fails because uncertainty cannot select useful chunk boundaries but temporal inconsistencies remain visible, an AR action expert may become a later method. It is not the first selection because it modifies action values and adds a trainable action generator.

### Affordance And Geometry Priors

Representative sources:

- Affordance Field Intervention: https://arxiv.org/abs/2512.07472
- AffordanceVLA: https://arxiv.org/abs/2606.06155
- GEAR-VLA: https://arxiv.org/abs/2606.08530

AUTHOR_STATED:

- AFI targets VLA memory traps with 3D spatial affordance fields, proprioceptive trap detection, affordance waypoints, and trajectory scoring; it reports average improvement on `pi_0`/`pi_0.5` real-world OOD settings and LIBERO-Pro.
- AffordanceVLA introduces Which2Act, Where2Act, and How2Act affordance forecasting as structured intermediate representations.
- GEAR-VLA uses geometry-aware representations, 3D integration, action tokenization, and embodiment canonicalization, reporting strong generalization across LIBERO, LIBERO-Plus, RoboTwin, real robots, and universal grasping.

INDEPENDENTLY_INFERRED:

- These are strong priors for explicit geometry and affordance, but their full data requirements are heavier than the current verified local stack.
- Local use would require non-privileged deployment-time affordance or geometry estimates. Simulator object-state or segmentation may be used only as training supervision or diagnostic oracle, never as hidden inference input.
- A reduced local method risks becoming another hand-engineered waypoint or visual preprocessing baseline unless the data/source gate is very strict.

CROSS_PAPER_SYNTHESIZED:

- Geometry/affordance should remain a live axis for a second condition or later method, especially if action-queue scheduling is insufficient.
- For Cycle 10, EAC is selected first because it has a validated intervention surface in the current runner and requires no new dense spatial labels.

## Cycle 10 Opportunity

The strongest post-PESA opportunity is `EAC-VLA`: Entropy-Calibrated Adaptive Chunking for frozen SmolVLA.

It is anchored primarily to AAC. Instead of training another adapter or residual, EAC preserves the exact frozen SmolVLA 7D action values and changes only the action-queue commitment length. The method uses a bounded development-only audit to verify that local SmolVLA action uncertainty is noncollapsed and predictive enough to choose short versus long commitments.

This changes the local method axis relative to PESA, MARC, DAGR, MTF, RAC, and RCV:

- representation: action-distribution uncertainty and queue-boundary risk rather than prior-query labels, disagreement gates, route logits, retained-frame scores, consequence histories, or current-state validity labels;
- supervision: development-only calibration of chunk commitment, not adapter imitation or residual regression;
- objective: clean action-value preservation and scheduling validity rather than action-value modification;
- policy generation: frozen SmolVLA emits the same action chunks, while the queue decides how many steps to execute before refresh;
- claim axis: whether uncertainty-calibrated chunk commitment improves official closed-loop success beyond fixed Base chunking, an AAC-style entropy proxy, a no-calibration ablation, and one fixed short-replan simple killer.

The critical Reviewer B baselines are:

- unmodified frozen SmolVLA with the official fixed action queue;
- a faithful transparent AAC-style proxy using entropy-only commitment;
- `EAC-VLA` full;
- an EAC no-calibration or no-hysteresis ablation;
- one strongest simple fixed short-replan or fixed-period queue-flush baseline.
