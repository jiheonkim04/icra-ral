# Epoch 4 Cycle 5 Prior Mechanism Map

Date: 2026-07-14 KST

Purpose: select the next method after EvoState-VLA stopped at a preregistered Stage 0 design failure. EvoState is archived as `AUDIT_STOP_DESIGN_FAILURE` and must not be rescued by lowering the action-input improvement threshold or launching a rollout.

## Local Constraints From Prior Results

The next method must not be:

- another success/failure residual field like FANG;
- another nearest-memory contrastive action method like CAVM;
- another action-evolved next-state controller like EvoState;
- another no-context replanning or stateless chunk reset like RCV;
- another photometric perturbation ensemble like PSE unless delayed feedback or another non-PSE mechanism is central;
- another generic progress, value, verifier, selector, or action-ranker route.

EvoState showed that the local trace data contains enough state-transition signal to beat a constant predictor, but the action-conditioned transition model improved over an actionless model by only `0.024689`, below the frozen `0.05` action-input threshold. This kills the specific action-evolved state-correction formulation, not every method using observed consequences.

## Close Sources

### Reflective VLA

Full title: Reflective VLA: In-Context Action Consequences Make VLAs Generalize.

URL: https://arxiv.org/abs/2606.25215

AUTHOR_STATED:

- Most VLAs are reactive and assume the current observation fully specifies the action-relevant state.
- Deployment factors such as camera-to-robot geometry, robot calibration, and systematic actuation bias are hard to infer from one observation.
- Reflective VLA conditions decisions on observation-action-consequence triplets.
- Reported LIBERO-Plus and LIBERO-Plus-Hard gains are `5.4` and `4.2` percentage points over matched reactive baselines, with ablations indicating action consequences matter beyond history length.

INDEPENDENTLY_INFERRED:

- The central positive prior is not generic memory. It is that realized action effects identify hidden deployment-specific mappings.
- A direct local reproduction is infeasible because Reflective VLA changes the VLA architecture, routes modalities through a VLM under shared attention, and trains block-causal in-context policies.
- A local proxy can test the same causal axis with the verified SmolVLA interface: maintain a short deployment-observable history of state, action, and realized state-delta triplets; infer whether action effects are systematically shifted; and apply a bounded identity-preserving action calibration only when the inferred shift is stable.

CROSS_PAPER_SYNTHESIZED:

- EvoState failed because one-step action-conditioned next-state prediction did not add enough value over actionless dynamics in the clean trace distribution.
- Reflective VLA suggests a different target: identify deployment-specific action-effect mismatch from observed consequences, especially under controlled calibration or actuation shift.
- FEDO and SCVC warn that static inverse gains and generic feedback residuals can explain many wins, so the first comparison must include a simple online calibration killer and a no-consequence history ablation.

Mechanism fields:

- observation/input: current 8D robot state, base 7D action, previous action, task key, chunk phase, and a short history of `(state, action, next-state minus state)` triplets;
- learned representation: compact action-consequence calibration context summarizing deployment-specific action-effect mismatch;
- supervision: synthetic and real development-only action-effect consistency labels from non-confirmatory traces plus controlled action-channel perturbation diagnostics;
- objective: predict the perturbation/calibration context from consequence history, with clean passthrough and bounded action delta;
- policy component changed: inference-time action calibration wrapper around a frozen VLA action, not the VLA backbone;
- action-generation mechanism: residual calibration initialized to zero and gated by stable consequence evidence;
- inference-time intervention: base action passthrough unless the consequence context predicts a stable calibration mismatch;
- assumed feedback: deployment-observable proprioceptive state only;
- benchmark condition: controlled actuation or calibration shift, plus clean retention;
- primary metric: closed-loop success under the shift and clean task retention;
- demonstrated causal link externally: Reflective VLA reports that action-consequence context improves cross-environment generalization;
- untested causal link locally: whether low-dimensional LIBERO state consequences contain enough shift information to beat static inverse-gain and history-only baselines.

### ReactVLA

Full title: ReactVLA: Fast and Lightweight Reactive Robot Manipulation via Improved Mean Flow Action Generation.

URL: https://arxiv.org/abs/2606.14255

AUTHOR_STATED:

- Diffusion-based VLAs can be too slow for reactive closed-loop manipulation.
- ReactVLA combines improved Mean Flow action generation with attention residual routing.
- It reports stronger LIBERO/RoboIMI performance, up to `1.65x` task-performance improvement on precision tasks, more than `4x` inference speed improvement, and real-world latency below `38.6 ms`.

INDEPENDENTLY_INFERRED:

- ReactVLA is a strong positive prior for changing the action generator rather than adding a post-hoc verifier.
- Local reproduction would require model training or architecture modification that is too large for the current single-GPU campaign.
- A local proxy could only test a small flow-step or action-smoothing wrapper, which would risk collapsing into previous retiming or smoothing failures.

CROSS_PAPER_SYNTHESIZED:

- ReactVLA should remain a boundary condition: if a method claims reactivity but only adds a slow second model or repeated VLA calls, Reviewer B should attack latency and compare to simple low-latency baselines.

### GEAR-VLA, Qwen-VLA, And WLA

URLs:

- GEAR-VLA: https://arxiv.org/abs/2606.08530
- Qwen-VLA: https://arxiv.org/abs/2605.30280
- WLA: https://arxiv.org/abs/2606.05979

AUTHOR_STATED:

- GEAR-VLA learns geometry-aware action representations, semantic-aligned 3D integration, and embodiment canonicalization, reporting strong LIBERO, LIBERO-Plus, RoboTwin, and real-robot results.
- Qwen-VLA unifies manipulation, navigation, and trajectory-centric prediction through a DiT action decoder and large-scale pretraining, reporting high LIBERO and OOD scores.
- WLA jointly predicts textual subtasks, subgoal images, and robot actions; its world prediction can be disabled or used for test-time scaling.

INDEPENDENTLY_INFERRED:

- These are strong external priors for geometry, embodiment, and world-model representation learning.
- Their full mechanisms require large-scale pretraining, 3D/spatial backbones, world experts, or model weights not locally available.
- A local lightweight method should not pretend to reproduce them, but it can borrow one idea: deployment-specific action representation should be separated from the frozen backbone and audited for identity preservation.

CROSS_PAPER_SYNTHESIZED:

- A future paper route may be a small geometry or embodiment canonicalization adapter only if local observation labels and a fair baseline exist.
- For Cycle 5, these priors are too heavy for implementation unless reduced to a development audit rather than a closed-loop prototype.

### PDF, CAG, ProgressVLA, And AffordVLA

URLs:

- PDF: https://arxiv.org/abs/2604.18107
- CAG / LIBERO-CF: https://arxiv.org/abs/2602.17659
- ProgressVLA: https://arxiv.org/abs/2603.27670
- AffordVLA: https://arxiv.org/abs/2605.17517

AUTHOR_STATED:

- PDF reports test-time perturbation learning with delayed feedback, uncertainty-based augmentation, action voting, and a lightweight perturbation module.
- CAG reports dual VLA/VA guidance for counterfactual language failures.
- ProgressVLA reports robust progress estimation and differentiable progress guidance through an inverse dynamics world model.
- AffordVLA aligns VLA visual representations to manipulation-centric affordance teachers.

INDEPENDENTLY_INFERRED:

- PDF is close to PSE and should not be reduced to photometric voting.
- CAG is close to earlier counterfactual-language routes and local CAG-style proxies were weak.
- ProgressVLA and AffordVLA are strong, but local labels and teachers are not yet verified.

CROSS_PAPER_SYNTHESIZED:

- These methods are useful as reviewer boundaries and candidate anchors.
- They are not stronger than Reflective VLA for the immediate next cycle because Reflective's required signal, action-consequence triplets, is already available locally and directly addresses deployment-specific calibration.

## Cycle 5 Opportunity

The strongest next opportunity is `RAC-VLA`: Reflective Action-Consequence Calibration for frozen VLAs. It is a Reflective VLA anchored prior extension that does not train a new VLA. Instead, it learns a compact consequence-history calibration context and applies a bounded, zero-initialized action calibration when deployment-observable action effects indicate a stable action-channel mismatch.

This changes at least four dimensions relative to EvoState and FANG:

- representation: consequence-history calibration context rather than action-evolved next-state prior or success/failure action field;
- supervision: action-effect mismatch and calibration labels rather than terminal success/failure or raw next-state prediction;
- action-generation mechanism: identity-preserving calibration of the current action under inferred deployment shift rather than state-tracking inverse dynamics;
- claim axis: deployment calibration or actuation-shift robustness with clean retention.

The critical Reviewer B baselines are:

- base frozen SmolVLA under the same controlled shift;
- a local Reflective-history proxy;
- a no-consequence history ablation;
- one simple online affine or diagonal inverse-gain killer.
