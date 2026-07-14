# Epoch 4 Cycle 4 Candidate Generation

Date: 2026-07-14 KST

Decision: `SELECT_EVOSTATE_VLA`

Governance applied: post-CAVM performance-oriented research design. Exactly three candidates were generated and scored. FANG-VLA remains fixed and archived as a valid Stage B kill.

## Candidate 1: EvoState-VLA

Name: `EvoState-VLA`, Action-Evolved State Guidance for Frozen Chunked VLAs.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: EvoScene-VLA, https://arxiv.org/abs/2605.21862.

Positive prior result: EvoScene reports improved chunked robot-control success by carrying an action-updated scene state across VLA calls and reconciling it with fresh observations.

Official code/checkpoint/reproducible mechanism: no official local checkpoint is available in this repository. The reproducible local mechanism is a transparent EvoScene/DREAM-style proxy over deployment-observable 8D robot state, 7D actions, task key, and chunk phase.

Assumption extended: EvoScene relies on scene tokens, 3D/depth teachers, and architecture-level action-decoder modification. Ours tests whether the core action-updated-belief mechanism can be made locally feasible and identity-preserving as a frozen-policy proprioceptive state controller.

Minimal technical difference proposed by Ours:

- train a compact action-conditioned transition model from non-confirmatory frozen SmolVLA trace transitions;
- maintain an internal predicted state during chunk execution;
- compare observed 8D state with the action-evolved predicted state;
- use a validation-calibrated controllability gate;
- apply a bounded inverse-dynamics correction toward the expected state only when the mismatch is predictable and controllable;
- default exactly to the base action when the gate is closed.

Why it could improve the same claim axis: EvoScene and DREAM-Chunk both show that action-updated latent dynamics can improve chunked control robustness. Local failures from RCV and FANG suggest that stateless replanning and success/failure residuals are not enough. EvoState targets the missing execution-state mismatch directly while preserving the frozen base policy.

### Quality Screen

Provisional novelty:

- Distinct from EvoScene because it removes scene-token architecture changes and tests a low-dimensional state-control version.
- Distinct from DREAM-Chunk because it corrects execution mismatch with an identity-preserving inverse-dynamics step rather than selecting among sampled chunks.
- Distinct from RCV because it maintains an action-evolved state prior and controllability gate rather than using no-context/stateless replanning.
- Distinct from FANG/CAVM because it uses next-state dynamics, not terminal success/failure action contrast.

Prior-anchor strength:

- Strong positive prior from EvoScene.
- DREAM-Chunk provides a second close positive prior and comparison target.
- AAC provides a simple chunking baseline boundary.
- A local faithful proxy is feasible because existing CAVM/FANG acquisition records contain `10801` non-confirmatory step rows with state, action, previous action, task key, chunk phase, identity, split, and success label.

Mechanism plausibility:

- Problem condition -> committed VLA chunks assume execution proceeds as expected.
- Intermediate failure mechanism -> contact, gripper timing, or small action realization errors make the actual state diverge from the chunk's expected state.
- Policy behavior -> the queued action continues as if the expected state were true.
- Closed-loop failure -> later actions compound the mismatch.
- Proposed method -> learn action-conditioned state evolution and controllability from development traces.
- Intended internal change -> maintain an action-evolved expected state and estimate whether current mismatch is correctable.
- Intended action behavior -> a small bounded correction that tracks the expected state when mismatch is reliable; otherwise base passthrough.
- Expected closed-loop improvement -> better robustness under controlled execution mismatch while retaining clean base behavior.

Data and supervision viability:

- Required labels are next-state transitions, which exist in `reports/cavm_vla/acquisition_records.jsonl`.
- The records are development-only identities and exclude confirmatory FANG/CAVM Stage B identities.
- No privileged object pose is required at inference.
- The audit must verify transition pairs, state variance, controllability rank, train/validation splits, and no hidden confirmatory identity use.

Identity-preserving integration:

- Correction output is zero by default.
- Gate threshold is selected on validation only.
- Correction is clipped in 7D action norm.
- Base action is emitted exactly when the learned mismatch confidence is low.
- Clean validation action delta, action bounds, and rollout retention are hard gates.

Decisive experiment feasibility:

- Stage 0 audit: transition-pair health, one-step prediction above constant/actionless baselines, controllable-subspace rank, and mismatch gate calibration.
- Stage 1 validation: no more than six configurations, selected by a score combining transition prediction, bounded action delta, clean retention proxy, and mechanism activation.
- Stage A/B closed-loop: five policies only: Base, DREAM-lite proxy, EvoState full, no-state-prior ablation, and static inverse-dynamics simple killer.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `87 / 100`

## Candidate 2: CPV-VLA

Name: `CPV-VLA`, Continuous Perturbation Voting for Flow-Based VLA Actions.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: PDF, Test-Time Perturbation Learning with Delayed Feedback for Vision-Language-Action Models, https://arxiv.org/abs/2604.18107.

Positive prior result: PDF reports LIBERO improvement from uncertainty-based observation perturbation, action voting, and delayed-feedback perturbation learning while keeping the base VLA frozen.

Official code/checkpoint/reproducible mechanism: the paper lists an official GitHub URL. A local reproduction would need adaptation because SmolVLA emits continuous 7D actions rather than discrete action logits.

Assumption extended: PDF assumes action logits and a voting/adaptation structure. Ours would define a mathematically valid continuous-action consensus and delayed-feedback perturbation head for flow-based 7D actions.

Minimal technical difference proposed by Ours:

- replace discrete majority voting with robust geometric-median consensus over continuous 7D action chunks;
- use action-sample covariance and Huber/Mahalanobis distances, not KL over deterministic actions;
- gate perturbation only when augmented views produce stable consensus;
- train any delayed-feedback head on validation/discovery only, never confirmatory identities.

Why it could improve the same claim axis: PDF's positive result suggests perturbation can reduce trajectory overfitting. Continuous consensus could make that mechanism valid for SmolVLA flow actions.

### Quality Screen

Provisional novelty:

- Moderate. The continuous-action formulation is real, but the mechanism is close to PDF and risks collapsing to PSE-style photometric ensembling.

Prior-anchor strength:

- Very strong positive prior with code link and LIBERO results.
- Local proxy feasible.

Mechanism plausibility:

- Problem condition -> visual shortcut or trajectory overfitting.
- Proposed method -> perturb observations and vote/consensus over actions.
- Expected action behavior -> stable action under harmless perturbations.
- Expected outcome -> less brittle execution.

Data and supervision viability:

- Inference-only and validation-only calibration are feasible.
- Delayed feedback learning would require additional development rollouts.
- The prior PSE kill is a serious warning: a simple bright perturbation baseline already explained a related effect.

Identity-preserving integration:

- Base action passthrough when consensus confidence is low.
- Bounded action delta and clean validation retention required.

Decisive experiment feasibility:

- Very feasible technically.
- Scientific decisiveness is weaker because a failure could repeat PSE, and a success must prove it is not just bright-single or averaging.

Score:

- provisional novelty: `14 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `15 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `78 / 100`

## Candidate 3: AffordProgress-VLA

Name: `AffordProgress-VLA`, Affordance-Conditioned Progress Guidance for Frozen VLAs.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: ProgressVLA, https://arxiv.org/abs/2603.27670.

Positive prior result: ProgressVLA reports that robust progress estimation plus inverse-dynamics world modeling improves CALVIN, LIBERO, and real-world manipulation success. AffordVLA reports improved VLA control by aligning visual representations with manipulation-centric affordance features.

Official code/checkpoint/reproducible mechanism: no compatible local checkpoint is available. A local method would need generated affordance/progress labels from development rollouts or a zero-shot teacher.

Assumption extended: ProgressVLA uses future visual latents and action-token guidance; AffordVLA uses affordance teachers. Ours would test a smaller local teacher that predicts manipulation affordance/progress from frozen SmolVLA observations and uses it to gate bounded action corrections.

Minimal technical difference proposed by Ours:

- generate development-only progress labels from successful traces;
- generate weak affordance labels from future gripper/contact state when available;
- train a compact affordance-progress representation;
- use it only as a bounded guidance signal with base passthrough by default.

Why it could improve the same claim axis: progress and affordance priors attack representation/action mismatch rather than terminal failure labels.

### Quality Screen

Provisional novelty:

- Good as a cross-paper synthesis of progress and affordance guidance.
- Risk: if reduced to a generic progress head, it is explicitly not enough.

Prior-anchor strength:

- Strong positive priors from ProgressVLA and AffordVLA.
- Local reproduction is less faithful because teachers and future visual latent gradients are unavailable.

Mechanism plausibility:

- Problem condition -> policy lacks a manipulation-centric intermediate state.
- Proposed method -> affordance/progress representation identifies useful interaction regions and subgoal movement.
- Intended action behavior -> bounded action changes toward progress-improving affordance.
- Expected improvement -> better long-horizon contact/release success.

Data and supervision viability:

- Progress labels from trace time are easy.
- Affordance labels are not yet verified locally.
- Inference observability and teacher quality must be audited before implementation.

Identity-preserving integration:

- Adapter/residual initialized to base passthrough.
- Clean retention penalty and action delta cap required.

Decisive experiment feasibility:

- Feasible only after a label-generation audit.
- More likely to become a representation study than a fast decisive prototype.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `12 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `78 / 100`

## Selection

Selected method: `EvoState-VLA`.

Selection reason:

- It is the most distinct from the just-killed FANG route and from CAVM.
- It is anchored to positive action-updated scene/world-model priors.
- It has directly available local supervision through non-confirmatory trace transitions.
- It can be audited cheaply before any large rollout.
- It has clear simple baselines: DREAM-lite proxy, no-state-prior ablation, static inverse dynamics, and Base.
- It preserves the pretrained policy by default and exposes a decisive mechanism-smoke path.

Immediate next steps:

1. Freeze an EvoState proposal and hash it.
2. Reviewer B attacks novelty against EvoScene, DREAM-Chunk, AAC, A2C2, and Health-conditioned VLA/fault-adaptation work.
3. Write mathematical audit and preregistration before implementation.
4. Run a bounded development audit using `reports/cavm_vla/acquisition_records.jsonl`.
