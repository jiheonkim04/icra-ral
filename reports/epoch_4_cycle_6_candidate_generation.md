# Epoch 4 Cycle 6 Candidate Generation

Date: 2026-07-14 KST

Decision: `SELECT_MTF_VLA`

Governance applied: post-RAC honest positive-result governance. Exactly three candidates were generated and scored. RAC-VLA remains archived as `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT` and must not be rescued.

## Candidate 1: MTF-VLA

Name: `MTF-VLA`, Milestone-Transition Focused VLA Adaptation.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: FrameSkip, https://arxiv.org/abs/2605.13757.

Secondary positive prior: StructVLA, https://arxiv.org/abs/2603.12553.

Positive prior result: FrameSkip reports improved success-retention trade-off across RoboCasa-GR1, SimplerEnv, and LIBERO, with macro-average success `76.15%` versus `66.50%` for full-frame training while retaining `20%` of unique frames. StructVLA reports strong sparse-milestone planning results, including `94.8%` LIBERO success, using gripper transitions and kinematic turning points as physically meaningful structured frames.

Official code/checkpoint/reproducible mechanism: no installed local official FrameSkip or StructVLA checkpoint is available. The reproducible local mechanism is transparent: compute frame importance from demonstration action variation, state or visual-action coherence when available, task-progress priors, gripper transitions, and kinematic turning points; train a lightweight SmolVLA adapter on selected frames while preserving base behavior on retention frames.

Assumption or limitation extended: FrameSkip changes the training sample distribution but does not explicitly preserve a strong pretrained policy on low-information frames. StructVLA uses sparse physical milestones inside a larger world-model planner, which is not locally cheap. MTF-VLA tests whether a local adapter can combine structured milestone emphasis with identity-preserving retention to improve closed-loop SmolVLA behavior.

Minimal technical difference proposed by Ours:

- compute milestone-transition scores from gripper transitions, action curvature, state velocity, and task-progress phase;
- build paired training batches containing high-milestone expert-action samples and low-milestone base-retention samples;
- train a lightweight adapter or LoRA with a weighted expert imitation loss plus a base-retention loss;
- validate the retained-frame ratio and one retention coefficient using development identities only;
- leave inference unchanged after the selected checkpoint is frozen.

Why it could improve the same claim axis: FrameSkip's positive result indicates that VLA success can improve when training emphasizes informative transition frames. MTF-VLA adds a policy-preserving constraint targeted at the local failure mode where ordinary LoRA did not improve closed-loop SmolVLA and sometimes degraded it.

### Quality Screen

Provisional novelty:

- Distinct from FrameSkip because it adds an explicit base-retention objective and paired high-transition/low-transition batches for adapter training.
- Distinct from StructVLA because it does not train a structured world model; it uses physically meaningful milestones as a supervision-selection mechanism.
- Distinct from generic LoRA because uniform-sampling LoRA is the mandatory simple killer and the novelty claim is the milestone-retention data objective.
- Distinct from PSE/RCV/RAC because inference has no perturbation ensemble, replanning, or consequence residual wrapper.

Prior-anchor strength:

- Strong positive prior from FrameSkip, directly on VLA training and LIBERO.
- StructVLA provides a second positive prior for gripper transitions and kinematic turning points as physically meaningful sparse frames.
- A faithful local FrameSkip proxy is possible even without official code, and any omissions must be listed before Stage 0.

Mechanism plausibility:

- Problem condition -> dense demonstration training overrepresents long low-change plateaus and underrepresents alignment, contact, grasp, and release transitions.
- Intermediate failure mechanism -> a small adapter spends capacity imitating abundant plateau behavior and disrupts a strong pretrained policy without improving rare success-critical transitions.
- Policy behavior -> generic LoRA changes actions without adding transition-specific competence, explaining prior closed-loop non-improvement.
- Closed-loop failure -> critical manipulation transitions remain unreliable or clean behavior degrades.
- Proposed method -> rebalance supervised learning toward physically meaningful milestones while explicitly retaining base behavior on plateau frames.
- Intended internal change -> adapter gradients concentrate on transition frames and are damped on retention frames.
- Intended action behavior -> bounded changes near transition states, base-like behavior elsewhere.
- Expected closed-loop improvement -> better transition-heavy task success while preserving clean base performance.

Data and supervision viability:

- Expert action labels exist in the official LeRobot LIBERO dataset.
- State, action, gripper, task, and phase traces exist in existing development records.
- Frame scores can be generated without privileged inference inputs.
- Frozen-base retention targets can be generated from the pretrained policy on development frames, then saved with checkpoint and hash.
- Stage 0 must verify noncollapsed high/low scores, task coverage, phase coverage, duplicate count, split overlap, and that no confirmatory test identities are used.

Identity-preserving integration:

- Adapter starts as base-equivalent or near base-equivalent.
- Retention loss keeps low-score frames close to frozen base actions.
- Validation hard gates check action delta, gripper delta, clean retention, and full-versus-ablation difference before rollout.
- Inference uses one policy call and does not require privileged state, reward, or trajectory history.

Decisive experiment feasibility:

- Stage 0 audit: label and score health, split proof, high/low contrast, prior-proxy reproducibility, base-retention target persistence, and preliminary action-delta smoke.
- Bounded validation search: at most six configs over retained-frame ratio and one retention coefficient, selected by a score combining validation closed-loop proxy or small validation rollout, clean retention, transition-score coverage, action validity, and compute.
- Stage A/B first comparison uses exactly five policies: Base, FrameSkip proxy, MTF full, no-retention ablation, and uniform retained-ratio LoRA simple killer.
- Second backbone path: if SmolVLA reaches GO, apply the same data-layer selection and retention idea to Quantized OpenVLA-OFT INT4 only after a risk-assessed feasibility plan; do not train OpenVLA-OFT before SmolVLA GO.
- Second condition: transition-heavy clean LIBERO slice or controlled demonstration-frame retention condition, preregistered before confirmatory use.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `91 / 100`

## Candidate 2: SMG-VLA

Name: `SMG-VLA`, Sparse Milestone-Guided Action Conditioning.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: StructVLA, https://arxiv.org/abs/2603.12553.

Positive prior result: StructVLA reports strong sparse structured-frame planning performance, including `94.8%` LIBERO success, by predicting physically meaningful structured frames from gripper transitions and kinematic turning points and mapping them into low-level actions.

Official code/checkpoint/reproducible mechanism: no compatible official local checkpoint is available. A local proxy would train a small milestone-phase predictor from demonstration traces and condition a lightweight action adapter on predicted next-milestone phase.

Assumption or limitation extended: StructVLA uses a two-stage generative world model with discrete structured-frame tokens. SMG-VLA would test whether milestone prediction alone can guide a frozen SmolVLA adapter without dense future prediction or a large world model.

Minimal technical difference proposed by Ours:

- label next milestone type and distance from gripper transitions and kinematic turning points;
- train a compact milestone predictor from RGB/proprioception/instruction;
- condition an identity-preserving adapter on the predicted milestone code;
- compare against a milestone-unconditioned adapter and a simple phase-progress baseline.

Why it could improve the same claim axis: StructVLA suggests sparse physical milestones bridge planning and control. A local adapter might use those milestones as a lower-cost control representation.

### Quality Screen

Provisional novelty:

- Stronger than pure data selection because it introduces an inference-time milestone representation.
- Risk of collapsing into another progress or phase head, which the project has repeatedly found weak unless tied to decisive action behavior.

Prior-anchor strength:

- Strong positive prior, but direct reproduction is infeasible.
- A local proxy preserves milestone labels but not the structured-frame world-model machinery.

Mechanism plausibility:

- Problem condition -> current observation lacks explicit physical transition state.
- Intermediate failure mechanism -> policy actions do not align with the next contact/gripper/turning milestone.
- Proposed method -> infer next milestone and condition action adapter on it.
- Intended action behavior -> better timing of contact, grasp, release, and reorientation actions.

Data and supervision viability:

- Milestone labels can be derived from action/state traces.
- Predicting them from deployment inputs above a trivial baseline is unproven.
- Inference-time milestone errors could globally disrupt actions.

Identity-preserving integration:

- Adapter can be initialized near base behavior, but active milestone conditioning risks acting broadly.
- Requires stronger clean-retention and action-delta gates than MTF.

Decisive experiment feasibility:

- Stage 0 would need a nontrivial predictor audit before any rollout.
- The first comparison is feasible but heavier than MTF because it adds a learned inference-time representation and one more failure mode.

Score:

- provisional novelty: `23 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `85 / 100`

## Candidate 3: TTPA-VLA

Name: `TTPA-VLA`, Test-Time Progress Adapter for Frozen VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: TT-VLA, https://arxiv.org/abs/2601.06748.

Positive prior result: TT-VLA reports that on-the-fly test-time reinforcement learning with dense task-progress rewards improves VLA adaptability, stability, and task success in simulated and real-world settings while preserving trained priors.

Official code/checkpoint/reproducible mechanism: no compatible local checkpoint is available. A faithful local proxy would require a frozen non-privileged progress reward predictor and a tiny test-time adapter update rule frozen before confirmatory testing.

Assumption or limitation extended: TT-VLA assumes access to useful dense progress feedback at deployment. TTPA-VLA would test whether a progress reward learned from LIBERO demonstrations can support tiny test-time updates without privileged simulator success signals at inference.

Minimal technical difference proposed by Ours:

- train a non-privileged progress predictor from discovery/validation demonstrations;
- freeze a tiny adapter and update rule before confirmatory testing;
- update only a small policy component during an episode using predicted dense progress;
- compare with frozen Base, TT-VLA-style progress proxy, no-update ablation, and a simple low-pass or retry baseline.

Why it could improve the same claim axis: TT-VLA's positive result suggests that deployment-time adaptation can help under changing conditions, but the local method must prove progress is observable without privileged labels.

### Quality Screen

Provisional novelty:

- Moderate. The test-time update mechanism is meaningful, but it remains close to TT-VLA and risks becoming a generic progress head.

Prior-anchor strength:

- Strong external prior, but local official reproduction is unavailable.
- A faithful proxy depends on a progress predictor that does not yet exist.

Mechanism plausibility:

- Problem condition -> policy encounters states not well covered by static supervised training.
- Intermediate failure mechanism -> no online correction signal shifts action behavior toward progress.
- Proposed method -> use predicted progress as dense reward for tiny test-time updates.
- Expected action behavior -> adapt within an episode while preserving priors.

Data and supervision viability:

- Demonstration time and success completion labels can generate weak progress labels.
- Labels may collapse to task time or phase rather than meaningful control progress.
- Progress must be predictable from RGB/proprioception/instruction without future state.

Identity-preserving integration:

- Updates can be bounded and initialized at base behavior.
- Test-time learning carries high disruption risk and must be frozen before confirmatory use.

Decisive experiment feasibility:

- Feasible only after a progress-label health audit.
- More expensive and riskier than MTF because every confirmatory episode includes online updates and additional leakage checks.

Score:

- provisional novelty: `19 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `78 / 100`

## Selection

Selected method: `MTF-VLA`.

Selection reason:

- It has the strongest positive prior and the cleanest local supervision path.
- It directly addresses a known local failure: ordinary rank-4 LoRA did not improve closed-loop SmolVLA, so the method must improve the training signal rather than merely attach another adapter.
- It is not selected merely because it is easy; it is selected because FrameSkip provides direct positive VLA evidence, StructVLA supplies a physically meaningful milestone prior, and the first experiment can decisively test the closest prior, the key ablation, and a uniform-sampling simple killer.
- It preserves inference simplicity and pretrained identity better than learned milestone conditioning or test-time RL.
- It changes the core problem, supervision, objective, and policy-generation surface relative to RAC, EvoState, FANG, CAVM, RCV, and PSE.

Immediate next steps:

1. Freeze an MTF-VLA Researcher A proposal and hash it.
2. Reviewer B attacks novelty against FrameSkip, StructVLA, ordinary data reweighting, standard LoRA, and simple uniform retained-ratio training.
3. Researcher A provides one rebuttal if the method is not exact duplication or trivial equivalence.
4. Write `reports/mtf_vla/mathematical_mechanism_audit.md`, preregistration, and prototype protocol.
5. Implement only the Stage 0 development audit first: split proof, label/score health, retained-frame balance, base-retention target persistence, and action-delta smoke before any expensive training or rollout.
