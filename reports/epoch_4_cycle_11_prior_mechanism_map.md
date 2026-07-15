# Epoch 4 Cycle 11 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the next method after `EAC-VLA` completed Stage B as `EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. EAC is a valid fixed-protocol current-formulation kill and must not be rescued by retuning `eac_q33_aggressive_1_4_50`, changing thresholds, changing tasks or reset identities, changing the five-policy comparison, reinterpreting partial results, or adding a post-hoc expansion.

## Local Constraints From Prior Results

The next method must not be:

- another adaptive chunking, queue-commitment, entropy/dispersion, or fixed-replan variant of EAC or RCV;
- another prior-query or spectral-capacity PESA rescue;
- another median-anchor, static L1 mixture, or disagreement-gated MARC rescue;
- another dynamic arm/gripper route residual like DAGR;
- another milestone-frame sampling or retained-frame MTF rescue;
- another reflective consequence-calibration wrapper like RAC;
- another failure-aware residual field like FANG;
- another action-evolved state controller like EvoState;
- another nearest-memory contrastive action method like CAVM;
- another photometric perturbation ensemble like PSE;
- an ActionMap mini-proxy unless the official source-fidelity gate is satisfied.

The repeated local pattern is now clear: methods that alter action values, learned residuals, route gates, priors, or queue schedules can pass offline checks yet fail to exceed Base, a closest-prior proxy, or a simple reviewer-killer in closed loop. Cycle 11 should change the mechanism axis toward explicit spatial grounding, while refusing privileged inference inputs and preserving Base behavior until the grounding signal is proven observable from deployment inputs.

## Close Sources

### Direct Action-Head Injection Of A Grounded 3D Point

Full title: Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization.

URL: https://arxiv.org/abs/2606.27663

AUTHOR_STATED:

- VLA brittleness appears along spatial generalization and task generalization axes.
- Existing 2D language-prompt or visual-prompt representations do not fully solve the limitation.
- The paper represents the grounding signal as a 3D point, computes relative displacement from gripper to target, encodes it with a two-layer MLP, and injects it into the action head via adaptive layer normalization.
- It reports large LIBERO-PRO gains on GR00T-N1.6 and comparable gains on `pi_0.5`, supporting a backbone-agnostic mechanism claim.

INDEPENDENTLY_INFERRED:

- The positive prior is not "add any spatial hint." The causal intervention is: make the task-relevant target a physical 3D displacement in the same geometric space as the action head.
- The method assumes an adequate grounding source. Local SmolVLA/LIBERO may not expose depth or a legal target detector by default.
- A local candidate must therefore separate the diagnostic oracle from the deployable method: simulator/object-state targets may audit headroom and create training labels on discovery/validation only, but cannot be used at inference or in confirmatory rollout.
- The local extension is valuable only if a deployment-observable RGB/proprio/language point predictor is noncollapsed and beats simple target heuristics before any action-head conditioning is trained.

CROSS_PAPER_SYNTHESIZED:

- RoboPoint and RoboGround show that language-conditioned spatial affordance or grounding masks can serve as useful intermediate representations.
- AffordanceVLA shows a more structured Which2Act, Where2Act, How2Act progression, reinforcing that object-centric and geometric intermediate signals can bridge perception and control.
- The 3D point injection paper provides the minimal action-head route that is more compatible with local SmolVLA than rebuilding a full affordance VLA or active perception stack.

Mechanism fields:

- observation/input: deployment RGB camera streams, proprioceptive end-effector state, instruction text, Base SmolVLA hidden/action interface when available, and no reset identity, future observation, reward, success label, or simulator object pose at inference;
- learned representation: predicted task-relevant target point or gripper-to-target relative displacement, with optional confidence and phase gate;
- supervision: discovery/validation-only labels from legal demonstration metadata, object-state diagnostics, or image-space pseudo-labels; confirmatory identities are excluded;
- objective: point predictability and calibration loss, optional action-conditioning loss, clean-retention loss, bounded action-delta penalty, and validation score combining mechanism activation, clean retention, action validity, and a small validation proxy;
- policy component changed: action-head conditioning or adapter only after source gate passes;
- action-generation mechanism: Base SmolVLA remains the default; spatial conditioning can only supply bounded context to the action pathway;
- inference-time intervention: inject a predicted non-privileged 3D displacement or calibrated unknown/no-point token into a zero-initialized action adapter;
- assumed feedback: current observation and proprioception only;
- benchmark condition: official paired LIBERO manifest after Stage 0 source/label, mathematical, validation, and mechanism-smoke gates;
- primary metric: task-balanced official closed-loop success, paired deltas, point-prediction health, action validity, clean retention, latency, and second-backbone feasibility;
- demonstrated causal link externally: direct 3D action-head injection improves spatial/task generalization on LIBERO-PRO across two VLA backbones;
- untested causal link locally: whether SmolVLA failures include usable spatial headroom and whether a non-privileged target-point predictor is observable enough to improve closed-loop success beyond a 2D point proxy, no-point ablation, and simple target/phase heuristics.

### ActionMap

Full title: ActionMap: Robot Policy Learning via Voxel Action Heatmap.

URLs:

- paper: https://arxiv.org/abs/2606.06904
- repository: https://github.com/showlab/ActionMap

AUTHOR_STATED:

- Most VLA action decoders remain single-point predictors over continuous actions, leaving action-space geometry underused.
- ActionMap replaces the native decoder with a voxel action heatmap over the action space.
- The paper reports improvements across LIBERO simulation and real-world Franka manipulation, including a reported LIBERO four-suite gain over OpenVLA-OFT's L1 head.
- The repository is currently a pre-release with the core action-head implementation and example plug-in code, not a full training/evaluation stack.

INDEPENDENTLY_INFERRED:

- The prior is strong on action-representation geometry, but exact official local reproduction remains source-blocked because there are no released checkpoints, official training commands, or official LIBERO logs.
- A new local ActionMap method would be risky after the archived mini-anchor unless it extracts real SmolVLA action-token hidden states and uses the official core head or a faithful line-by-line port.
- Without source fidelity, the method collapses to a local heatmap/candidate head already covered by reusable negative evidence.

CROSS_PAPER_SYNTHESIZED:

- ActionMap and 3D point injection both argue that geometry must reach the action-generation interface, not only the visual input.
- ActionMap changes the decoder distribution; 3D point injection preserves the VLA backbone and conditions the action head with a minimal spatial embedding. Given the local history of destructive action replacement, G3P has lower disruption risk than ActionMap for Cycle 11.

### AffordanceVLA And Grounding Priors

Representative sources:

- AffordanceVLA: https://arxiv.org/abs/2606.06155
- RoboGround: https://arxiv.org/abs/2504.21530
- RoboPoint: https://arxiv.org/abs/2406.10721
- MolmoAct: https://arxiv.org/abs/2508.07917

AUTHOR_STATED:

- AffordanceVLA introduces structured affordance forecasting through object-centric grounding, 2D interaction localization, and 3D geometric reasoning, with code and a project page listed on arXiv.
- RoboGround uses grounding masks as intermediate guidance for manipulation policies.
- RoboPoint predicts image keypoint affordances from language instructions and reports improved spatial-affordance accuracy and downstream task success.
- MolmoAct uses depth-aware perception tokens and mid-level spatial plans before low-level actions.

INDEPENDENTLY_INFERRED:

- These priors support the broader claim that spatial intermediate representations can improve robot policies.
- Their full mechanisms are heavier than the current local budget because they may require dense masks, external VLMs, depth, full foundation-model weights, or new training pipelines.
- They are best used as source and baseline priors for a minimal 3D-point injection method rather than selected as a full Cycle 11 implementation.

CROSS_PAPER_SYNTHESIZED:

- The selected path should not be another visual transform. The key local question is whether a deployable target point can be inferred and routed to the action pathway without replacing Base behavior.
- The closest-prior comparison must include a 2D point or visual-prompt proxy because the main external claim is that 3D action-head injection beats weaker 2D routing of the same grounding signal.

## Cycle 11 Opportunity

The strongest post-EAC opportunity is `G3P-VLA`: Grounded 3D Point Injection for frozen SmolVLA.

It is anchored primarily to Direct Action-Head Injection of a Grounded 3D Point. The local extension is not to assume privileged target geometry. Instead, it installs a hard source gate:

- discovery/validation may use oracle target positions only to measure headroom and label health;
- inference must use only deployment RGB, proprioception, language, and Base policy features;
- unknown or low-confidence grounding defaults to Base passthrough;
- action-head conditioning is zero-initialized and bounded;
- rollout is forbidden if the point predictor is collapsed, unobservable, privileged, or globally disruptive.

This changes the axis relative to EAC, PESA, MARC, DAGR, MTF, RAC, and RCV:

- representation: task-relevant spatial target displacement rather than uncertainty, spectral queries, median anchors, route labels, milestone frames, consequence histories, or queue-validity labels;
- supervision: source-gated spatial grounding labels and point predictability, not action residual selection alone;
- objective: point observability plus identity-preserving action conditioning, not action-value replacement as the main novelty;
- policy generation: Base SmolVLA remains default, with bounded action-head conditioning only when the deployment-observable point is confident;
- claim axis: whether non-privileged 3D spatial grounding at the action interface improves closed-loop success beyond Base, a 2D/visual-prompt proxy, no-point ablation, and one simple target/phase heuristic.

The critical Reviewer B baselines are:

- unmodified frozen SmolVLA;
- closest-prior 3D-point proxy using the same legal point source when available;
- `G3P-VLA` full;
- no-3D/no-action-head-injection ablation;
- one strongest simple 2D point, phase, or nearest-object heuristic baseline.
