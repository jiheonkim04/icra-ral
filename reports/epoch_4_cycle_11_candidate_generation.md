# Epoch 4 Cycle 11 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_G3P_VLA`

Governance applied: post-CAVM performance-oriented governance and post-RAC honest positive-result governance. Exactly three candidates were generated and scored. EAC-VLA remains archived as `EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`; it must not be rescued by retuning commitment thresholds, scheduler maps, tasks, reset identities, policy list, or outcome interpretation.

## Candidate 1: G3P-VLA

Name: `G3P-VLA`, Grounded 3D Point Injection for frozen SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Direct Action-Head Injection of A Grounded 3D Point, https://arxiv.org/abs/2606.27663.

Secondary priors: RoboPoint, https://arxiv.org/abs/2406.10721; RoboGround, https://arxiv.org/abs/2504.21530; AffordanceVLA, https://arxiv.org/abs/2606.06155.

Positive prior result: the closest prior reports that representing a task-relevant grounding signal as a 3D gripper-relative displacement and injecting it into the action head substantially improves LIBERO-PRO task and position perturbation performance on GR00T-N1.6 and `pi_0.5`. RoboPoint, RoboGround, and AffordanceVLA independently support spatial grounding and affordance intermediates as useful robot-policy mechanisms.

Official code/checkpoint/reproducible mechanism: no official code or checkpoint for the closest prior was located during this Cycle 11 scan. The mechanism is reproducible from the primary paper: obtain a target point, lift it to 3D when legal depth/camera information exists, compute gripper-relative displacement, encode with a small MLP, and inject into the action head. Official-source status must be documented again before implementation. The local method may use only a faithful transparent proxy unless exact official equivalence is established.

Assumption or limitation extended: the closest prior assumes adequate grounding is available. Local SmolVLA/LIBERO cannot silently use simulator object pose, reset identity, task-success signals, or future observations at inference. G3P extends the prior with a non-privileged source gate and Base-passthrough behavior when the point is unavailable or unreliable.

Minimal technical difference proposed by Ours:

- build discovery/validation-only oracle labels to audit spatial headroom and label health;
- train or validate a deployment-observable point predictor from RGB, proprioception, language, and Base features only;
- lift the point to a gripper-relative 3D displacement only when legal depth/camera or calibrated local geometry is available; otherwise stop or use an explicitly declared 2D proxy baseline;
- inject the predicted displacement into a zero-initialized, bounded action-conditioning adapter;
- default to exact Base behavior when point confidence is low or the source gate fails;
- compare against Base, a closest-prior 3D-point proxy, G3P full, no-3D/no-injection ablation, and one strongest simple 2D/phase/nearest-object heuristic.

Why it could improve the same claim axis: the prior demonstrates that the representation and injection route of grounding, not just having a prompt, can unlock spatial and task generalization. Local SmolVLA has shown repeated failures from changing action values without stronger state grounding; a source-gated 3D point may supply the missing spatial variable while preserving Base as the default policy.

### Quality Screen

Provisional novelty:

- Distinct from the closest prior because the local contribution is a strict non-privileged source gate plus identity-preserving SmolVLA action conditioning.
- Distinct from FANG, CAVM, MARC, DAGR, MTF, RAC, and EAC because it does not start from failure memories, median anchors, component routing, frame selection, consequence histories, or queue scheduling.
- Novelty risk remains: if a 2D point proxy or simple nearest-object/phase heuristic explains the gain, the full method must be killed.

Prior-anchor strength:

- Very strong positive external effect on the same spatial/task generalization claim axis.
- Secondary grounding and affordance priors support the representation family.
- No closest-prior official code/checkpoint is currently verified, so local comparison must be called a faithful transparent proxy.

Mechanism plausibility:

- Problem condition -> SmolVLA fails when task-relevant object or placement geometry shifts.
- Intermediate failure mechanism -> visual-language semantics do not become a precise gripper-relative target variable at the action interface.
- Policy behavior -> Base may approach memorized or scene-triggered trajectories rather than the instruction-specified target.
- Closed-loop failure -> mis-approach, wrong-object contact, failed placement, or late correction.
- Proposed method -> infer a task-relevant target point from legal deployment inputs and condition the action head with a gripper-relative spatial embedding.
- Intended internal change -> action-conditioning layers receive a physically meaningful displacement while the VLM backbone remains unchanged.
- Intended action behavior -> approach and place motions become better aligned to the target when the point is confident, while uncertain states remain Base-like.
- Expected closed-loop improvement -> higher task-balanced success on spatially sensitive tasks with clean retention.

Data and supervision viability:

- RGB, proprioceptive end-effector state, language, Base actions, train/validation/test split manifests, and official rollout infrastructure exist locally.
- Oracle object-state or simulator labels may be available for diagnostics and training labels only, but inference use is prohibited.
- Stage 0 must prove target labels are noncollapsed, split-clean, phase/task-covered, and predictable above trivial baselines from deployment inputs.
- If no legal point source exists, this is a `DATA_OR_SUPERVISION_FAILURE`, not a closed-loop scientific kill.

Identity-preserving integration:

- The action-conditioning adapter is initialized as exact Base passthrough.
- Low-confidence or unavailable point predictions force Base behavior.
- Residual/action deltas are bounded and audited separately for translation, rotation, and gripper dimensions.

Decisive experiment feasibility:

- Stage 0 source/label audit can stop before training or rollout.
- Mathematical audit can define the point variable, tensor shapes, displacement formula, confidence gate, gradient path, and no privileged inference rule.
- Bounded validation search uses at most six configurations over point confidence threshold, adapter scale, and one architecture choice.
- First serious comparison uses exactly five policies: Base, closest-prior 3D-point proxy, G3P full, no-3D/no-injection ablation, and one simple 2D/phase/nearest-object heuristic.
- Second backbone path: if SmolVLA reaches GO, port the same point-conditioning gate to Quantized OpenVLA-OFT INT4.
- Second condition: a frozen spatial/task perturbation or LIBERO-PRO-style slice after source gate and SmolVLA prototype GO.

Score:

- provisional novelty: `23 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `90 / 100`

## Candidate 2: AMH-VLA

Name: `AMH-VLA`, ActionMap Hidden-State Heatmap decoder for SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: ActionMap, https://arxiv.org/abs/2606.06904 and https://github.com/showlab/ActionMap.

Positive prior result: ActionMap reports that a voxel action heatmap decoder improves LIBERO simulation and real-world Franka manipulation, including a reported gain over OpenVLA-OFT's L1 regression head on the LIBERO four-suite average.

Official code/checkpoint/reproducible mechanism: the repository is a pre-release. It exposes a core `HeatmapActionHead` and example hidden-state usage, but no full official training/evaluation stack, checkpoint, logs, or exact LIBERO commands. Local use must therefore begin with a source-fidelity gate rather than a claimed official reproduction.

Assumption or limitation extended: ActionMap assumes access to compatible VLA action-token hidden states and replacement of the native decoder. Local SmolVLA hidden-state/action-token extraction is not yet verified.

Minimal technical difference proposed by Ours:

- extract real SmolVLA hidden states at action-generating positions if available;
- attach the official core heatmap head or a faithful line-by-line port;
- initialize a Base-mixture safety path to prevent global action replacement;
- reject if source fidelity, hidden-state alignment, or simple mean/MLP/LoRA baselines explain the result.

Why it could improve the same claim axis: the external prior argues that action-space geometry is a real VLA performance lever. A local source-faithful heatmap head could test whether the previous mini-anchor failed because it was not close enough to the official action-token mechanism.

### Quality Screen

Provisional novelty:

- Meaningful only if real SmolVLA hidden/action-token states and the official head semantics are preserved.
- Weak if it falls back to the archived local ActionMap mini-anchor or ordinary action regression.

Prior-anchor strength:

- Strong positive prior and official core code preview.
- Exact official reproduction remains blocked by missing training/evaluation release.

Mechanism plausibility:

- Problem condition -> continuous action decoders ignore neighboring-action geometry.
- Intermediate failure mechanism -> single-point outputs overfit or choose brittle actions.
- Proposed method -> predict a voxel heatmap over action space.
- Expected action behavior -> more data-efficient and geometrically smoother action prediction.

Data and supervision viability:

- 7D action labels exist.
- Real hidden-state extraction is unverified.
- Prior local mini-anchor evidence makes source fidelity mandatory.

Identity-preserving integration:

- Base-mixture safeguard is possible but weakens closest-prior fidelity.
- Replacing the decoder has high disruption risk.

Decisive experiment feasibility:

- Stage 0 source gate is decisive.
- Closed-loop comparison is feasible only after hidden-state alignment, action-validity, and simple-baseline gates pass.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `83 / 100`

## Candidate 3: SAR-VLA

Name: `SAR-VLA`, Structured Affordance Representation for frozen SmolVLA.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: AffordanceVLA, https://arxiv.org/abs/2606.06155.

Secondary priors: RoboGround, https://arxiv.org/abs/2504.21530; RoboPoint, https://arxiv.org/abs/2406.10721; MolmoAct, https://arxiv.org/abs/2508.07917.

Positive prior result: AffordanceVLA reports strong simulation and real-world manipulation performance from Which2Act, Where2Act, and How2Act affordance forecasting. RoboGround, RoboPoint, and MolmoAct report positive results for grounding masks, affordance keypoints, and spatial plans.

Official code/checkpoint/reproducible mechanism: AffordanceVLA's arXiv entry lists code and a project page, but exact local compatibility with SmolVLA/LIBERO has not been established. RoboPoint and RoboGround supply reproducible grounding/affordance mechanisms, but not a ready local SmolVLA adapter.

Assumption or limitation extended: these priors often use dense labels, external models, depth, masks, or larger training pipelines. The local extension would compress the idea into a lightweight structured affordance token or map that is inferred from deployment RGB and fused conservatively.

Minimal technical difference proposed by Ours:

- train a small affordance representation from discovery/validation labels only;
- fuse it as an auxiliary token or gated observation side-channel;
- default to raw Base observation/policy behavior when confidence is low;
- compare against simple crop, edge/brightness, 2D point, and no-affordance baselines.

Why it could improve the same claim axis: if local failures are perception-grounding failures rather than action-syntax or queue failures, structured affordance cues may expose the relevant object/region before action generation.

### Quality Screen

Provisional novelty:

- Meaningful as a minimal deployment-observable affordance representation for frozen SmolVLA.
- High risk of becoming another PSE/GCAP-style visual transform or privileged segmentation wrapper.

Prior-anchor strength:

- Strong family-level prior from several recent papers.
- Closest local official reproduction path is less clear than G3P.

Mechanism plausibility:

- Problem condition -> policy attends to distractors or misses task-relevant regions.
- Intermediate failure mechanism -> semantic object identity and spatial actionability are not explicit.
- Proposed method -> expose a structured affordance cue to the policy.
- Expected action behavior -> better target selection and placement.

Data and supervision viability:

- RGB and language exist.
- Dense affordance or mask labels are not yet proven locally.
- Privileged inference risk is high and must be gated before rollout.

Identity-preserving integration:

- A gated auxiliary side-channel can default to raw Base behavior.
- Observation modification can still globally disturb Base if not carefully bounded.

Decisive experiment feasibility:

- Stage 0 can test label/source health and raw-vs-affordance action deltas.
- Full closed-loop experiment is heavier than G3P because there are more representation and source choices.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `85 / 100`

## Selection

Selected method: `G3P-VLA`.

Selection reason:

- It has the strongest positive-prior effect on a claim axis not yet tested locally: explicit gripper-relative spatial grounding at the action interface.
- It changes more than two core dimensions relative to EAC: representation, supervision, policy conditioning, and claim axis all change.
- It is more source-feasible than a full AffordanceVLA/RoboGround/MolmoAct-style representation stack and less source-blocked than official ActionMap reproduction.
- It preserves Base by construction until the source, label, and point-observability gates pass.
- Unknown empirical performance is not a rejection reason; the decisive Stage 0 gate can stop the method as `DATA_OR_SUPERVISION_FAILURE` or `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE` before rollout.

Immediate next steps:

1. Freeze a `G3P-VLA` Researcher A proposal and hash it.
2. Reviewer B attacks novelty and source legality against the closest 3D-point prior, RoboPoint/RoboGround/AffordanceVLA, visual/2D prompting, nearest-object heuristics, and prior local kills.
3. Researcher A provides one rebuttal if the method remains nontrivial and locally feasible.
4. Write `reports/g3p_vla/mathematical_mechanism_audit.md`, preregistration, and prototype protocol before any expensive training or rollout.
5. Implement only Stage 0 first: source legality, oracle headroom, label balance, RGB/proprio/language point predictability, split integrity, Base passthrough, action-delta bounds, and no confirmatory-test identity use.
