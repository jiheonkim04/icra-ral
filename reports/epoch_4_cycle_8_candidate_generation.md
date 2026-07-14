# Epoch 4 Cycle 8 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_MARC_VLA`

Governance applied: post-RAC honest positive-result governance. Exactly three candidates were generated and scored. DAGR-VLA remains archived as `DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD` and must not be rescued.

## Candidate 1: MARC-VLA

Name: `MARC-VLA`, Median-Anchored Regression Correction for frozen SmolVLA flow actions.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: OpenVLA-OFT, https://arxiv.org/abs/2502.19645 and https://openvla-oft.github.io/.

Secondary priors: ReactVLA, https://arxiv.org/abs/2606.14255; SnapFlow, https://arxiv.org/abs/2604.05656.

Positive prior result: OpenVLA-OFT reports that parallel decoding, action chunking, continuous actions, and L1 regression raise OpenVLA's LIBERO average success from `76.5%` to `97.1%` while increasing action generation throughput by `26x`. ReactVLA and SnapFlow report positive flow-action calibration or simplification results on LIBERO-family VLA settings.

Official code/checkpoint/reproducible mechanism: OpenVLA-OFT releases code and checkpoints, but direct local training exceeds the active local GPU budget. A faithful transparent local proxy is feasible: train a continuous L1/Huber action adapter on SmolVLA/LIBERO records and compare against it under the same data, backbone, and inference budget.

Assumption or limitation extended: OpenVLA-OFT replaces the original OpenVLA action head. MARC tests the minimal frozen-SmOLVLA extension: whether a robust deterministic median anchor can correct flow-action errors while preserving the pretrained flow policy by default.

Minimal technical difference proposed by Ours:

- train a robust median action anchor `m_t in R^7` from deployment-observable inputs and frozen base action features;
- train a disagreement gate from train-only base/expert disagreement labels;
- emit `a_base + g * clip(m_t - a_base)` rather than replacing base actions;
- initialize the emitted correction to exact base passthrough;
- compare against the OpenVLA-OFT-style L1 proxy, a no-gate ablation, and a static Base/L1-proxy mixture simple killer.

Why it could improve the same claim axis: OpenVLA-OFT's positive result suggests deterministic L1-style continuous action generation can be more reliable than more expressive generative action heads on noisy demonstrations. MARC tests whether that median-style anchor is useful as a bounded correction to SmolVLA flow actions rather than a full replacement.

### Quality Screen

Provisional novelty:

- Distinct from OpenVLA-OFT because it does not replace the VLA action head; it adds a base-preserving median anchor around frozen SmolVLA.
- Distinct from DAGR because it has no arm/gripper dynamic routing and no group-specific residual heads.
- Distinct from MTF because it does not select training frames or add base-retention sampling.
- Novelty risk remains: if the gate does not matter, the method reduces to a plain L1 adapter or static mixture.

Prior-anchor strength:

- Strong positive prior from OpenVLA-OFT, with official code and checkpoints.
- Secondary priors support action-generation calibration in flow/action-generation VLAs.
- A faithful local proxy is implementable and must be included early.

Mechanism plausibility:

- Problem condition -> SmolVLA flow actions may reproduce noisy or suboptimal action modes from demonstrations or sampling-time flow errors.
- Intermediate failure mechanism -> a flow-generated action chunk can be locally plausible but closed-loop brittle when the conditional action distribution has bad modes.
- Policy behavior -> Base is strong overall, so full replacement is risky.
- Closed-loop failure -> occasional inaccurate approach, insertion, grasp, or release actions compound through the episode.
- Proposed method -> learn a robust L1/Huber median anchor and apply it only when an observable disagreement gate predicts useful correction.
- Intended internal change -> median anchor captures central expert action tendency; gate avoids global disruption.
- Intended action behavior -> base-like action by default, bounded correction toward a robust anchor in high-disagreement states.
- Expected closed-loop improvement -> improved success on states where the base flow action is off-mode, with clean retention.

Data and supervision viability:

- Expert actions, base actions, state, task keys, phases, and split identities already exist in stable SmolVLA prediction artifacts.
- L1/Huber anchor targets are ordinary 7D actions, not dense affordance labels or simulator reward.
- Disagreement labels can be generated from train-only base/expert action differences.
- Privileged simulator success is not required at inference.

Identity-preserving integration:

- Correction output is zero-initialized.
- Gate bias starts at base passthrough.
- Correction norm is clipped.
- Clean validation action delta and action validity are hard gates.

Decisive experiment feasibility:

- Stage 0 audit verifies headroom, noncollapsed disagreement labels, L1 proxy strength, full-versus-proxy separability, initial identity, and split integrity.
- Bounded validation search uses at most six configs over correction alpha and gate architecture.
- First serious comparison uses exactly five policies: Base, OpenVLA-OFT-style L1 proxy, MARC full, MARC no-gate ablation, and static validation-selected Base/L1 mixture.
- Second backbone path: if SmolVLA reaches GO, port the median anchor to Quantized OpenVLA-OFT INT4 as a frozen-head correction with the same 7D semantics.
- Second condition: noisy-demonstration or low-data LIBERO slice, frozen before use.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `91 / 100`

## Candidate 2: AMH-VLA

Name: `AMH-VLA`, ActionMap Heatmap Head source-gated SmolVLA adaptation.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: ActionMap, https://arxiv.org/abs/2606.06904 and https://github.com/showlab/ActionMap.

Positive prior result: ActionMap reports that replacing single-point action decoders with voxel heatmap action heads improves LIBERO and real-world Franka manipulation, including a reported `+8.2%` over OpenVLA-OFT's L1 head on the LIBERO four-suite average.

Official code/checkpoint/reproducible mechanism: the official repository currently provides a core `HeatmapActionHead` preview, but not a complete training stack. A source-gated local reproduction would need real SmolVLA hidden action-token states and an official-style heatmap loss, not the old failed CPU mini-proxy.

Assumption or limitation extended: ActionMap replaces a VLA decoder. AMH would test whether the official core heatmap head can be attached to local SmolVLA features under a bounded source gate.

Minimal technical difference proposed by Ours:

- locate a valid SmolVLA hidden state corresponding to action prediction;
- attach the official core heatmap head or a line-by-line faithful port;
- train only the heatmap head or a small adapter under fixed 7D action normalization;
- compare to L1 regression and MLP action heads before rollout.

Why it could improve the same claim axis: ActionMap's positive result suggests action-space geometry is a real lever. A faithful official-source gate could determine whether the earlier local mini-anchor failed because the proxy was too weak rather than because the representation is useless.

### Quality Screen

Provisional novelty:

- Strong if a real source-gated heatmap head is integrated with SmolVLA hidden states.
- Weak if it falls back to the already killed local mini-anchor proxy.

Prior-anchor strength:

- Strong positive prior and official core code preview.
- Full training stack and checkpoints are unavailable.

Mechanism plausibility:

- Problem condition -> single-point action decoders ignore geometric proximity between neighboring actions.
- Proposed method -> predict action heatmaps over discretized translation, rotation, and gripper spaces.
- Expected action behavior -> smoother data-efficient action decoding.

Data and supervision viability:

- 7D actions exist.
- Hidden action-token extraction is unverified.
- Prior local proxy failed mean-action and MLP gates, so source fidelity is mandatory.

Identity-preserving integration:

- Harder than MARC because replacing the decoder risks disrupting Base.
- A mixture with Base would be required for initial identity, but that weakens ActionMap fidelity.

Decisive experiment feasibility:

- Stage 0 source gate is decisive, but may fail before method implementation.
- Closed-loop experiment is feasible only after hidden-state and heatmap training gates pass.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `83 / 100`

## Candidate 3: LAAF-VLA

Name: `LAAF-VLA`, Lightweight Affordance-Aligned Flow correction.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: AffordanceVLA, https://arxiv.org/abs/2606.06155 and https://github.com/Skywalker-yqz/AffordanceVLA/.

Secondary prior: GEAR-VLA, https://arxiv.org/abs/2606.08530.

Positive prior result: AffordanceVLA reports strong simulation and real-world performance from Which2Act, Where2Act, and How2Act affordance forecasting. GEAR-VLA reports strong geometry-aware action representation results across LIBERO, LIBERO-Plus, RoboTwin, and real robots.

Official code/checkpoint/reproducible mechanism: AffordanceVLA releases a repository with architecture/config scaffolding, but the dataset-specific loaders are not bundled and the staged training is multi-GPU. GEAR-VLA code is currently a placeholder.

Assumption or limitation extended: the positive priors assume dense affordance or 3D geometry supervision. LAAF would test whether train-only weak affordance labels derived from local demonstrations can align a small flow correction module without privileged inference inputs.

Minimal technical difference proposed by Ours:

- generate train-only weak contact/approach labels from expert gripper state, end-effector motion, and phase;
- train a small affordance-latent predictor from deployment inputs;
- use it only as a bounded correction gate around Base;
- reject before rollout if labels are collapsed, phase-only, or unpredictable above trivial baselines.

Why it could improve the same claim axis: affordance priors suggest manipulation succeeds when actions are tied to what, where, and how to interact. A weak local affordance signal could improve flow action correction if it survives strict label-health gates.

### Quality Screen

Provisional novelty:

- Meaningful if weak affordance labels produce a distinct, observable latent.
- High risk of collapsing into a hand-engineered phase/contact heuristic.

Prior-anchor strength:

- Strong external results, but weak local reproduction.
- Official code is incomplete for local one-GPU reproduction.

Mechanism plausibility:

- Problem condition -> action errors cluster near contact/approach moments.
- Proposed method -> predict an affordance/contact latent and gate corrections only there.
- Expected action behavior -> better interaction timing and placement.

Data and supervision viability:

- Weak labels can be generated from existing traces.
- Dense 2D/3D affordance labels are not locally available.
- Stage 0 is likely to expose phase-only or gripper-only collapse.

Identity-preserving integration:

- Zero-initialized correction and base-passthrough gate are feasible.
- Clean retention must be a hard gate.

Decisive experiment feasibility:

- Stage 0 label-health gate is cheap.
- Fair prior comparison is weak because the local method is far from official AffordanceVLA.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `17 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `80 / 100`

## Selection

Selected method: `MARC-VLA`.

Selection reason:

- It has the strongest combination of positive external prior, local feasibility, identity-preserving integration, and a decisive five-policy comparison.
- It directly targets a strong prior claim axis from OpenVLA-OFT: deterministic L1-style continuous action generation can outperform more expressive action generation under noisy demonstrations.
- It is less dependent on unavailable dense labels or hidden-token source gates than AMH or LAAF.
- It can be killed cleanly before rollout if the L1 proxy or static mixture explains the method, if disagreement labels collapse, if the gate is not observable, or if corrections globally disrupt Base.

Immediate next steps:

1. Freeze a MARC-VLA Researcher A proposal and hash it.
2. Reviewer B attacks novelty against OpenVLA-OFT, ordinary L1 adapters, static Base/L1 mixtures, DAGR, MTF no-retention, and generic residual correction.
3. Researcher A provides one rebuttal if the method remains nontrivial and locally feasible.
4. Write `reports/marc_vla/mathematical_mechanism_audit.md`, preregistration, and prototype protocol.
5. Implement only a Stage 0 development audit first: label/headroom health, L1 proxy quality, full-versus-proxy separation, initial identity, action-delta bounds, and split integrity before expensive training or rollout.
