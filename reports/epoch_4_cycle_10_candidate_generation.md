# Epoch 4 Cycle 10 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_EAC_VLA`

Governance applied: post-CAVM performance-oriented governance and post-RAC honest positive-result governance. Exactly three candidates were generated and scored. PESA-VLA remains stopped at Stage 0 as `DESIGN_FAILURE`; it must not be rescued by retuning query labels, spectral thresholds, model capacity, or rollout criteria.

## Candidate 1: EAC-VLA

Name: `EAC-VLA`, Entropy-Calibrated Adaptive Chunking for frozen SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Adaptive Action Chunking, https://arxiv.org/abs/2604.04161.

Secondary priors: AR-VLA, https://arxiv.org/abs/2603.10126; AC2-VLA, https://arxiv.org/abs/2601.19634.

Positive prior result: AAC reports that action entropy can adapt VLA chunk size at inference time, improving manipulation performance across simulated and real-world tasks; the project page links public code repositories. AR-VLA reports smoother action trajectories while maintaining or exceeding state-of-the-art reactive VLA success, and AC2-VLA reports action-context-aware computation reuse with comparable success and lower compute.

Official code/checkpoint/reproducible mechanism: AAC has public code repositories. A local faithful proxy is possible without downloading anything now: the official SmolVLA/LIBERO runner exposes a `50 x 7` action chunk and an action queue, and the policy path can produce full action chunks. Stage 0 must verify whether stochastic or repeated chunk predictions provide a noncollapsed entropy/variance signal under deployment inputs.

Assumption or limitation extended: AAC assumes an action entropy signal that can be read from the deployed model. The local SmolVLA runner is not the same backbone/protocol, so EAC must first establish a valid local uncertainty proxy and compare against a transparent entropy-only AAC proxy.

Minimal technical difference proposed by Ours:

- keep frozen SmolVLA weights and emitted 7D action values unchanged;
- compute a deployment-observable queue-commitment risk score from action entropy or multi-sample chunk variance, chunk self-discontinuity, boundary disagreement, and previous-action jump risk;
- map the frozen risk score to a commitment length from a small set such as `{1, 2, 4, 8, 16, 50}`;
- use hysteresis or a retention band to avoid mode-jumping from rapid queue flushes;
- default to the official Base queue behavior when uncertainty is unavailable or below threshold;
- compare against Base, AAC entropy-only proxy, no-calibration/no-hysteresis ablation, and a fixed short-replan simple killer.

Why it could improve the same claim axis: the closest prior demonstrates that adaptive chunk size can improve VLA manipulation by balancing reactivity and temporal consistency. EAC tests the same axis on the local frozen SmolVLA 7D queue while preserving action values, directly addressing the campaign pattern where learned action modifications were destructive.

### Quality Screen

Provisional novelty:

- Distinct from AAC because the local method adds SmolVLA-specific uncertainty-source validation, queue-boundary risk, and hysteresis while preserving official 7D action values.
- Distinct from RCV because it is not a learned current-state validity replanner and must beat a fixed short-replan reviewer killer.
- Distinct from MTF because it does not change the training frame distribution or train adapters.
- Novelty risk remains: if entropy-only AAC or fixed short replanning matches EAC, the method collapses to the prior proxy or simple baseline.

Prior-anchor strength:

- Strong positive prior from AAC on the exact chunk-size claim axis.
- Public AAC code exists; local proxy is transparent if the uncertainty signal passes Stage 0.
- AR-VLA and AC2-VLA provide supporting evidence that action context and refresh cadence are central VLA deployment variables.

Mechanism plausibility:

- Problem condition -> SmolVLA emits long action chunks; large commitments can delay perception refresh, while too-frequent refresh can create discontinuities.
- Intermediate failure mechanism -> fixed queue commitment ignores model uncertainty and local contact/approach phases.
- Policy behavior -> the same good Base action values may be harmed by either stale execution or unnecessary queue flushing.
- Closed-loop failure -> late corrections, jerky chunk transitions, or contact timing errors reduce success.
- Proposed method -> choose commitment length from action uncertainty and boundary-risk signals.
- Intended internal change -> no policy-weight change; only queue scheduling changes.
- Intended action behavior -> long smooth commitments when confident, short refreshes when uncertain or boundary risk is high.
- Expected closed-loop improvement -> higher success or equal success with better smoothness/latency while retaining Base action validity.

Data and supervision viability:

- Official action chunks, queue length, previous action, proprioception, and task/instruction are available at deployment.
- No dense affordance labels, object-state inference inputs, success or reward labels, or confirmatory identities are required for Stage 0.
- Required uncertainty signal is unknown and must be audited before rollout; collapsed uncertainty is a valid design failure.

Identity-preserving integration:

- Exact emitted action values are preserved.
- Default behavior is the existing frozen SmolVLA queue when the risk score is invalid or below threshold.
- Intervention is bounded to queue flush/commit length, with action validity unchanged by construction.

Decisive experiment feasibility:

- Stage 0 audit verifies chunk shape, queue control surface, stochastic chunk support, entropy/variance noncollapse, fixed Base passthrough, no action-value modification, latency, and no confirmatory identity use.
- Bounded validation search uses at most six configurations over one threshold and one hysteresis/commitment map, selected by a preregistered score combining validation success proxy or small validation rollout, smoothness, action validity, mechanism activation, clean retention, and overhead.
- First serious comparison uses exactly five policies: Base fixed queue, AAC entropy-only proxy, EAC full, no-calibration/no-hysteresis ablation, and fixed short-replan simple killer.
- Second backbone path: if SmolVLA reaches GO, port the queue scheduler to Quantized OpenVLA-OFT INT4 without changing action values.
- Second condition: a transition/contact-heavy LIBERO slice or LIBERO-Pro-style spatial shift, frozen before use.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `93 / 100`

## Candidate 2: G3P-VLA

Name: `G3P-VLA`, Grounded 3D Point Injection for SmolVLA action heads.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Direct Action-Head Injection of A Grounded 3D Point, https://arxiv.org/abs/2606.27663.

Secondary priors: GEAR-VLA, https://arxiv.org/abs/2606.08530; AffordanceVLA, https://arxiv.org/abs/2606.06155.

Positive prior result: the closest prior reports that injecting a relative 3D grounded point into the action head substantially improves LIBERO-PRO task and position perturbation success for GR00T-N1.6 and `pi_0.5`. GEAR-VLA and AffordanceVLA report strong geometry/affordance-aware manipulation performance.

Official code/checkpoint/reproducible mechanism: direct official local code is not verified in this repo. A local proxy could use training-only object-state or scripted geometry labels to train a non-privileged RGB/proprio/language point predictor, then inject the predicted relative 3D displacement into a small identity-preserving adapter.

Assumption or limitation extended: the prior assumes adequate grounding is available. Local LIBERO evaluation cannot use simulator object-state, reset identity, or privileged target coordinates at inference. The central extension would be a strict non-privileged grounding-source gate before action-head injection.

Minimal technical difference proposed by Ours:

- derive 3D target-point labels only on discovery/development data;
- train or verify a deployment-observable point predictor from RGB/proprio/instruction;
- inject predicted relative gripper-to-target displacement into a small action adapter initialized as Base passthrough;
- reject if point prediction is trivial, unavailable, or globally disruptive;
- compare against a closest-prior 3D-point proxy, no-point ablation, and a simple 2D/phase/nearest-object baseline.

Why it could improve the same claim axis: the positive prior says direct 3D grounding at the action head can unlock spatial and task generalization. Local SmolVLA failures may include spatial grounding errors that action residuals cannot fix.

### Quality Screen

Provisional novelty:

- Meaningful if the 3D point is inferred from legal deployment inputs and injected directly into the action pathway.
- Weak if it uses simulator state at inference or degenerates into a task-id/phase lookup.

Prior-anchor strength:

- Strong recent positive prior with large reported LIBERO-PRO gains.
- Secondary geometry and affordance priors reinforce the claim axis.
- Local official reproduction is not established.

Mechanism plausibility:

- Problem condition -> object/task spatial shifts break implicit VLA grounding.
- Intermediate failure mechanism -> the policy lacks an explicit relative target displacement at the action head.
- Proposed method -> add a predicted 3D point embedding to the action adapter.
- Expected action behavior -> better approach, grasp, and placement correction under spatial shifts.

Data and supervision viability:

- Training labels may be derivable from LIBERO object-state, but inference must use RGB/proprio/instruction.
- Current stable prediction artifact does not already contain validated image-derived 3D point predictions.
- Stage 0 must prove noncollapsed point labels, no train/test overlap, and above-trivial deploy-input predictability.

Identity-preserving integration:

- A zero-initialized adapter can preserve Base initially.
- Risk is moderate because injecting a spatial embedding can alter many actions if the predictor is noisy.

Decisive experiment feasibility:

- Stage 0 source/label gate is decisive.
- Closed-loop comparison is feasible only after non-privileged point prediction and action-delta checks pass.

Score:

- provisional novelty: `23 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `88 / 100`

## Candidate 3: ARX-VLA

Name: `ARX-VLA`, Re-anchored Autoregressive Chunk Composer for frozen SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: AR-VLA, https://arxiv.org/abs/2603.10126.

Secondary prior: Adaptive Action Chunking, https://arxiv.org/abs/2604.04161.

Positive prior result: AR-VLA reports that a continuous causal autoregressive action expert with refreshable vision-language prefixes and re-anchoring produces smoother action trajectories while maintaining or exceeding state-of-the-art reactive VLA success.

Official code/checkpoint/reproducible mechanism: AR-VLA links code/videos, but no compatible local checkpoint is verified. A local proxy would train a tiny causal composer over Base action chunks, recent executed actions, and proprioception while keeping Base passthrough as initialization.

Assumption or limitation extended: AR-VLA replaces or substantially augments the action expert. The local extension would test a smaller re-anchored composer that corrects only chunk-to-chunk temporal syntax and must not overwrite Base actions globally.

Minimal technical difference proposed by Ours:

- train a small causal action composer on development demonstrations and Base chunks;
- re-anchor the composer to each refreshed SmolVLA chunk so perception staleness is explicit;
- initialize residual output to zero so initial behavior equals Base;
- compare against an AR-VLA-style local proxy, no-reanchor ablation, and a simple low-pass/fixed-period smoothing baseline.

Why it could improve the same claim axis: AR-VLA's positive prior says temporal action syntax matters for success and smoothness. A local re-anchored composer could reduce chunk discontinuities without changing the full VLA backbone.

### Quality Screen

Provisional novelty:

- Meaningful as a local, identity-preserving AR extension.
- Risk of collapsing into another residual smoother or low-pass filter is high.

Prior-anchor strength:

- Strong positive prior, RSS 2026 accepted, code/videos linked.
- Local official reproduction is not installed.

Mechanism plausibility:

- Problem condition -> chunked reactive policies reset temporal context.
- Intermediate failure mechanism -> action syntax discontinuities and stale perception create jerky or mistimed control.
- Proposed method -> maintain a causal action state and re-anchor at each vision refresh.
- Expected action behavior -> smoother chunk boundaries and better contact timing.

Data and supervision viability:

- Demonstration action sequences, Base chunks, and proprioception exist.
- Training a new action composer is heavier than EAC and risks offline action-L2 optimism.

Identity-preserving integration:

- Zero residual initialization is possible.
- High disruption risk remains because action values are modified.

Decisive experiment feasibility:

- Stage 0 can verify gradient flow, sequence prediction, action-delta bounds, and low-pass baseline comparison.
- Closed-loop comparison is heavier than EAC and may be explained by smoothing.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `7 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `83 / 100`

## Selection

Selected method: `EAC-VLA`.

Selection reason:

- It has the strongest combination of positive prior anchor, local intervention surface, identity preservation, and decisive Stage 0 feasibility.
- It changes the action-queue scheduling surface instead of training another residual, adapter, or action decoder.
- It directly addresses the RCV/MTF/MARC/DAGR/PESA pattern: the campaign has repeatedly found that learned action modification can be harmful or explained by simple baselines, while the official runner still exposes a meaningful chunk-commitment decision.
- It is better locally grounded than G3P, which needs a legal 3D grounding source, and less disruptive than ARX, which modifies action values.
- It can be killed before rollout if SmolVLA uncertainty is collapsed, queue control is unavailable, entropy-only AAC or fixed short replanning explains the benefit, or action scheduling causes unacceptable latency/smoothness harm.

Immediate next steps:

1. Freeze an EAC-VLA Researcher A proposal and hash it.
2. Reviewer B attacks novelty against AAC, RCV, AR-VLA, fixed short replanning, low-pass smoothing, and generic queue flushing.
3. Researcher A provides one rebuttal if the method remains nontrivial and locally feasible.
4. Write `reports/eac_vla/mathematical_mechanism_audit.md`, preregistration, and prototype protocol.
5. Implement only a Stage 0 development audit first: queue surface proof, entropy/variance signal health, Base passthrough, action-value preservation, latency, split integrity, and no confirmatory-test identity use.
