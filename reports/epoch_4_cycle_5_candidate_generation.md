# Epoch 4 Cycle 5 Candidate Generation

Date: 2026-07-14 KST

Decision: `SELECT_RAC_VLA`

Governance applied: post-CAVM performance-oriented research design. Exactly three candidates were generated and scored. EvoState-VLA remains archived as `AUDIT_STOP_DESIGN_FAILURE` and must not be rescued.

## Candidate 1: RAC-VLA

Name: `RAC-VLA`, Reflective Action-Consequence Calibration for Frozen VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Reflective VLA, https://arxiv.org/abs/2606.25215.

Positive prior result: Reflective VLA reports that conditioning on observation-action-consequence triplets improves deployment generalization on LIBERO-Plus and LIBERO-Plus-Hard over matched reactive baselines, and that consequence triplets matter beyond generic history length.

Official code/checkpoint/reproducible mechanism: no compatible local checkpoint is available in this repository. The reproducible mechanism is the action-consequence context: use deployment-observable state, action, and realized state-delta triplets to infer hidden calibration or actuation mismatch before generating the next action.

Assumption extended: Reflective VLA retrains an architecture to ingest rich multimodal in-context triplets. Ours tests whether the same action-consequence principle can be made locally feasible as a frozen-policy calibration layer that preserves the base action by default.

Minimal technical difference proposed by Ours:

- learn a compact consequence-history calibration context from development traces;
- train it to identify action-effect mismatch under controlled synthetic action-channel perturbations and real trace variation;
- apply a zero-initialized bounded 7D calibration residual only when the consequence context is stable and validation-calibrated;
- default exactly to the base SmolVLA action when the gate is closed;
- evaluate under a controlled deployment-shift condition and a clean-retention condition.

Why it could improve the same claim axis: Reflective VLA's positive result indicates that past action consequences reveal deployment-specific mappings that a reactive policy cannot infer from one frame. RAC-VLA tests the minimal frozen-policy version of that claim in a resource-bounded LIBERO setting.

### Quality Screen

Provisional novelty:

- Distinct from Reflective VLA because it is not an in-context VLM architecture; it is a compact action-consequence calibration adapter for a frozen flow-action VLA.
- Distinct from EvoState because it does not require action-conditioned next-state prediction to beat actionless dynamics on clean traces; it targets deployment-shift calibration from realized consequences.
- Distinct from FEDO/SCVC because static inverse gain and simple affine correction are explicit killer baselines, and RAC must beat them rather than hide behind them.

Prior-anchor strength:

- Strong positive prior from Reflective VLA with reported LIBERO-Plus generalization gains and a consequence-vs-history ablation.
- Direct official reproduction is infeasible, but a faithful local proxy is transparent because action-consequence triplets are already present in `reports/cavm_vla/acquisition_records.jsonl`.

Mechanism plausibility:

- Problem condition -> the frozen policy is deployed under a controlled action-channel calibration shift or systematic actuation mismatch.
- Intermediate failure mechanism -> the current observation alone does not identify the mismatch before actions are executed.
- Policy behavior -> the base action is emitted in the wrong calibrated action space.
- Closed-loop failure -> repeated small calibration errors compound.
- Proposed method -> infer a stable calibration context from recent action consequences.
- Intended internal change -> update a low-dimensional deployment context summarizing how executed actions changed the observed robot state.
- Intended action behavior -> apply a small calibrated action residual or axis scaling only when the inferred context is stable.
- Expected closed-loop improvement -> higher shifted-condition success with clean passthrough.

Data and supervision viability:

- Required triplets exist: current 8D state, 7D action, previous action, task key, chunk phase, and next-state delta are available from development traces.
- Controlled perturbation labels can be generated synthetically for Stage 0 without touching confirmatory identities.
- Privileged object state and task success are not inference inputs.
- Stage 0 must verify noncollapsed perturbation labels, predictable action-effect mismatch, task coverage, and no leakage from confirmatory identities.

Identity-preserving integration:

- Calibration residual initialized to zero.
- Gate initialized to base passthrough.
- Action delta clipped per 7D action norm and per translation/rotation/gripper group.
- Clean validation action delta and clean rollout retention are hard gates.

Decisive experiment feasibility:

- Stage 0 audit: label health, perturbation-class predictability, consequence-vs-history ablation, action-delta bound, and clean passthrough.
- Validation search: no more than six configs over history horizon and residual coefficient.
- Stage A/B closed-loop: five policies only: Base, Reflective-history local proxy, RAC full, no-consequence history ablation, and online diagonal inverse-gain simple killer.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `88 / 100`

## Candidate 2: FlowPDF-VLA

Name: `FlowPDF-VLA`, Continuous-Action Delayed-Feedback Perturbation Learning for SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: PDF, Test-Time Perturbation Learning with Delayed Feedback for Vision-Language-Action Models, https://arxiv.org/abs/2604.18107.

Positive prior result: PDF reports LIBERO success gains through uncertainty-based augmentation, action voting, adaptive augmentation scheduling, and delayed-feedback perturbation learning without fine-tuning the base model.

Official code/checkpoint/reproducible mechanism: the paper provides a code URL. Local reproduction would require adapting the mechanism from action logits to continuous SmolVLA 7D flow actions.

Assumption extended: PDF assumes action logits and voting. Ours would define a mathematically valid continuous-action delayed-feedback objective using action-sample statistics, robust Huber/Mahalanobis distances, and no KL over deterministic 7D actions.

Minimal technical difference proposed by Ours:

- sample small observation perturbations and base flow actions;
- compute robust continuous-action consensus;
- learn a delayed-feedback perturbation scheduler from development rollouts;
- gate perturbation only when consensus is stable;
- compare directly with the strongest single perturbation and base action.

Why it could improve the same claim axis: PDF's positive result suggests test-time perturbations can reduce trajectory overfitting. A continuous-flow adaptation could make the claim compatible with SmolVLA.

### Quality Screen

Provisional novelty:

- Moderate. Continuous-action delayed feedback is a real technical adaptation, but it remains close to PDF.
- The previous PSE result is a serious local warning because bright-single explained the photometric ensemble.

Prior-anchor strength:

- Strong positive prior and code availability.

Mechanism plausibility:

- Problem condition -> subtle visual shifts or trajectory overfitting.
- Proposed method -> perturb observations, estimate stable continuous-action consensus, and learn delayed-feedback scheduling.
- Intended behavior -> reduce overconfident shifted actions while preserving clean base behavior.

Data and supervision viability:

- Inference-time perturbation is feasible.
- Delayed-feedback labels require additional development rollouts and careful partitioning.
- The method risks becoming another PSE unless delayed feedback is clearly active.

Identity-preserving integration:

- Base action passthrough when consensus confidence is low.
- Bounded action delta and clean retention required.

Decisive experiment feasibility:

- Technically feasible, but decisive novelty is weaker because PSE already tested a nearby route and lost to a simple perturbation baseline.

Score:

- provisional novelty: `15 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `7 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `78 / 100`

## Candidate 3: GeoAfford-Lite-VLA

Name: `GeoAfford-Lite-VLA`, Lightweight Geometry-Affordance Action Calibration.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: GEAR-VLA, https://arxiv.org/abs/2606.08530.

Positive prior result: GEAR-VLA reports strong generalization from geometry-aware action representations, semantic-aligned 3D integration, and embodiment canonicalization on LIBERO, LIBERO-Plus, RoboTwin 2.0, and real robots. AffordVLA provides a second positive prior for aligning VLA visual representations with manipulation-centric affordance representations.

Official code/checkpoint/reproducible mechanism: GEAR-VLA says code and models will be released, but no compatible local checkpoint is available here. AffordVLA requires a zero-shot affordance teacher not currently installed.

Assumption extended: the positive priors require large-scale 3D or affordance teachers. Ours would test whether a small state/action geometry-affordance proxy can provide a useful action calibration signal under local budget.

Minimal technical difference proposed by Ours:

- derive low-dimensional contact/approach geometry proxies from deployment-observable state and gripper behavior;
- align a small adapter to these proxies;
- use the adapter as a bounded action calibration signal;
- compare against pure state/action affine baselines and clean base behavior.

Why it could improve the same claim axis: manipulation success often depends on functional interaction geometry, and GEAR/AffordVLA show that geometry and affordance priors can improve generalization.

### Quality Screen

Provisional novelty:

- Good as a local lightweight synthesis, but it risks collapsing into a generic hand-engineered state feature adapter.

Prior-anchor strength:

- Strong external positives, but local reproduction is weak because true 3D/affordance representations are unavailable.

Mechanism plausibility:

- Problem condition -> visual features focus on global appearance rather than functional interaction regions.
- Proposed method -> geometry-affordance proxy captures approach/contact phase.
- Intended action behavior -> small phase-aware action corrections.
- Expected improvement -> better interaction precision.

Data and supervision viability:

- Low-dimensional proxy labels might be generated from state/gripper traces.
- True affordance and 3D teacher labels are absent.
- Stage 0 would need a label-health audit before any rollout.

Identity-preserving integration:

- Zero-initialized residual and validation-calibrated gate.
- Clean-retention hard stop.

Decisive experiment feasibility:

- Less decisive than RAC because the best local signal may be too weak and reviewers may reject the proxy as not faithful to the closest prior.

Score:

- provisional novelty: `20 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `17 / 20`
- technical mechanism quality: `15 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `75 / 100`

## Selection

Selected method: `RAC-VLA`.

Selection reason:

- It has the strongest combination of positive external prior, local data availability, and decisive experiment path.
- It is materially different from EvoState because it uses action-consequence history to infer deployment calibration, not next-state prediction for clean trace correction.
- It includes the dangerous simple explanations as first-class baselines: no-consequence history and online inverse-gain calibration.
- It is identity preserving by construction and can stop before rollout if consequence labels or calibration prediction collapse.

Immediate next steps:

1. Freeze a RAC-VLA proposal and hash it.
2. Reviewer B attacks novelty against Reflective VLA, FEDO/SCVC/static inverse-gain routes, ReactVLA, PDF, and generic feedback correction.
3. Write the mathematical audit and preregistration before implementation.
4. Run a bounded Stage 0 development audit using `reports/cavm_vla/acquisition_records.jsonl` and synthetic controlled action-channel perturbation labels.
