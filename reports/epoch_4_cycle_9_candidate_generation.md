# Epoch 4 Cycle 9 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_PESA_VLA`

Governance applied: post-CAVM performance-oriented governance and post-RAC honest positive-result governance. Exactly three candidates were generated and scored. MARC-VLA remains archived as `MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE` and must not be rescued.

## Candidate 1: PESA-VLA

Name: `PESA-VLA`, Prior-Expert Spectral Adaptation for frozen SmolVLA 7D policies.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: PriorVLA, https://arxiv.org/abs/2605.10925.

Secondary priors: LoRA-SP, https://arxiv.org/abs/2603.07404; VLA-GSE, https://arxiv.org/abs/2605.06175 and https://github.com/YuhuaJiang2002/VLA-GSE.

Positive prior result: PriorVLA reports a read-only prior expert plus adaptation expert that reaches `99.1%` average LIBERO success and improves OOD/few-shot adaptation. LoRA-SP reports adaptive spectral capacity over pi0 and SmolVLA real-robot tasks, including up to `31.6%` multi-task success improvement over standard LoRA. VLA-GSE reports strong PEFT transfer and `81.2%` zero-shot success on LIBERO-Plus with an official code release.

Official code/checkpoint/reproducible mechanism: PriorVLA official code is not established in the current repo, but the prior/adaptation-expert mechanism can be transparently proxied. VLA-GSE has official code. LoRA-SP provides an explicit SVD-style energy rule. The local repo already has a working SmolVLA 7D adapter/LoRA path, disk-reloadable policy identity checks, and official LIBERO rollout infrastructure.

Assumption or limitation extended: PriorVLA preserves a pretrained prior while adapting, but direct official reproduction is not locally established. LoRA-SP/VLA-GSE allocate adapter capacity but do not by themselves make the frozen pretrained policy a read-only action prior at inference. PESA tests their minimal local synthesis under a strict clean-retention and Base-passthrough integration.

Minimal technical difference proposed by Ours:

- keep frozen SmolVLA as a read-only prior expert that always emits the default 7D action chunk;
- train a spectral adaptation expert over the fixed LIBERO 7D action interface;
- use per-input/layer spectral energy scores to activate only the needed adapter directions;
- condition adapter use on a prior-query embedding derived from deployment-observable inputs and the Base action;
- initialize inference to exact Base passthrough and enforce bounded clean-retention deltas;
- compare against a PriorVLA-style proxy, standard fixed-rank LoRA, a no-spectral/no-prior-query ablation, and one simple clean-retention adaptation baseline.

Why it could improve the same claim axis: the closest priors show that preserving pretrained priors and allocating VLA adaptation capacity can substantially improve downstream robot performance. PESA makes that prior preservation explicit for the local frozen SmolVLA 7D policy and prevents the globally destructive action changes observed in several previous local kills.

### Quality Screen

Provisional novelty:

- Distinct from PriorVLA because the local mechanism adds spectral capacity activation and a hard Base-passthrough clean-retention gate around a frozen SmolVLA 7D policy.
- Distinct from LoRA-SP because the frozen prior expert remains an explicit action source rather than merely an initialization.
- Distinct from VLA-GSE because it does not claim generic expert specialization; the testable mechanism is prior-preserving spectral adaptation under official closed-loop evaluation.
- Novelty risk remains: if a standard LoRA or PriorVLA-style proxy matches the full method, PESA collapses to ordinary PEFT and must be killed.

Prior-anchor strength:

- Strong positive prior from PriorVLA, LoRA-SP, and VLA-GSE.
- VLA-GSE has official code; LoRA-SP has an explicit mathematical mechanism; PriorVLA is transparent enough for a faithful local proxy.
- The closest-prior proxy can be compared under the same local backbone, data, task/reset manifest, and inference budget.

Mechanism plausibility:

- Problem condition -> frozen SmolVLA is strong but leaves task-specific residual failure; naive adapters or residual wrappers can globally disrupt good actions.
- Intermediate failure mechanism -> fixed-rank or fully active adapters learn a narrow downstream distribution and overwrite useful pretrained motor priors.
- Policy behavior -> Base succeeds on many paired resets, so the method must learn when and how much adaptation is allowed.
- Closed-loop failure -> unnecessary action shifts degrade otherwise solvable trajectories.
- Proposed method -> keep Base as a read-only prior expert, activate spectral adapter directions only when prior-query evidence supports adaptation, and regularize clean deltas.
- Intended internal change -> adaptation capacity is concentrated in input-relevant directions while Base behavior remains the default.
- Intended action behavior -> Base-like actions on clean or uncertain states, bounded adapter improvement on states where training/validation show adaptation headroom.
- Expected closed-loop improvement -> improved task-balanced success when standard LoRA has headroom but would otherwise disrupt clean Base behavior.

Data and supervision viability:

- Expert 7D actions, Base actions, task keys, train/validation/reserved splits, and policy identity infrastructure already exist.
- No dense affordance labels, simulator rewards, future observations, or privileged inference inputs are required.
- Standard LoRA/simple adaptation baselines are feasible and mandatory.
- Confirmation identities remain reserved and cannot be used for selection.

Identity-preserving integration:

- Initial behavior is exact Base passthrough.
- Prior expert is frozen and read-only.
- Adapter output is bounded by validation-selected delta limits.
- Clean-retention action deltas and action validity are hard gates before rollout.

Decisive experiment feasibility:

- Stage 0 audit verifies standard LoRA headroom, PriorVLA-proxy headroom, split integrity, action interface identity, initial Base passthrough, gradient flow, and non-catastrophic action deltas.
- Bounded validation search uses at most six configurations over spectral energy threshold and query head shape.
- First serious comparison uses exactly five policies: Base, PriorVLA-style proxy, PESA full, no-spectral/no-prior-query ablation, and the strongest simple fixed-rank LoRA or clean-retention baseline.
- Second backbone path: if SmolVLA reaches GO, port the adapter/scorer interface to Quantized OpenVLA-OFT INT4 with the same 7D semantics.
- Second condition: low-data or controlled LIBERO-Plus-style shift, frozen before use.

Score:

- provisional novelty: `20 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `90 / 100`

## Candidate 2: AMH-VLA

Name: `AMH-VLA`, ActionMap Heatmap Head source-gated SmolVLA adaptation.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: ActionMap, https://arxiv.org/abs/2606.06904 and https://github.com/showlab/ActionMap.

Positive prior result: ActionMap reports that a voxel heatmap action head improves LIBERO and real-world Franka manipulation, including `+8.2%` over OpenVLA-OFT's L1 regression head on the LIBERO four-suite average.

Official code/checkpoint/reproducible mechanism: the official repository is a pre-release and exposes the core `HeatmapActionHead` plus an example that gathers hidden states at action-token positions. It does not yet provide a full training stack. Local continuation is permitted only through an official-source gate using real SmolVLA hidden/action states, not the old failed mini-proxy.

Assumption or limitation extended: ActionMap replaces a decoder in existing VLA backbones. AMH tests whether the official core heatmap head can be attached to local SmolVLA 7D hidden states under a bounded source-fidelity gate.

Minimal technical difference proposed by Ours:

- identify real SmolVLA hidden states corresponding to action prediction;
- attach the official core heatmap head or a line-by-line faithful port;
- train only the heatmap head or a minimal adapter under fixed 7D action normalization;
- reject before rollout if mean-action, fixed-rank LoRA, or cheap MLP baselines match it, or if heatmap predictions collapse.

Why it could improve the same claim axis: ActionMap's positive result says action-space geometry is a real VLA performance lever. A source-gated local reproduction could determine whether the previous local mini-anchor failed because the proxy was too weak.

### Quality Screen

Provisional novelty:

- Strong if official hidden-state and heatmap semantics are preserved.
- Weak if it falls back to the previously killed local CPU heatmap/candidate proxy.

Prior-anchor strength:

- Strong positive prior and official core-code preview.
- Full training stack and checkpoints are not yet available.

Mechanism plausibility:

- Problem condition -> single-point action decoders ignore geometric proximity of neighboring actions.
- Proposed method -> predict translation/rotation/gripper heatmaps over the action space.
- Expected action behavior -> smoother and more data-efficient action decoding.

Data and supervision viability:

- 7D action labels exist.
- Real action-token hidden-state extraction is unverified.
- Prior local proxy failed mean-action and MLP gates, making source fidelity mandatory.

Identity-preserving integration:

- Harder than PESA because replacing the action decoder risks Base disruption.
- A Base-mixture safeguard is feasible, but may weaken ActionMap fidelity and must be treated as an ablation.

Decisive experiment feasibility:

- Stage 0 source gate is decisive and cheap.
- Closed-loop experiment is feasible only after hidden-state, heatmap-training, collapse, and simple-baseline gates pass.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `83 / 100`

## Candidate 3: SVO-VLA

Name: `SVO-VLA`, Semantic Virtual Observation adaptation for fixed-camera SmolVLA.

Contribution type: `CROSS_DOMAIN_MECHANISM_TRANSFER`.

Closest external prior: SEVO, https://arxiv.org/abs/2605.11114.

Secondary priors: SaPaVe, https://arxiv.org/abs/2603.12193; ActiveVLA, https://arxiv.org/abs/2601.08325.

Positive prior result: SEVO reports large cross-environment transfer gains for ACT and SmolVLA by improving the observation stream through body-fixed cameras, active illumination, segmentation overlay, and diverse data collection. SaPaVe reports up to `31.25%` higher real-world success than recent VLA baselines with decoupled camera/manipulation actions. ActiveVLA reports active view/zoom improvements for precise 3D manipulation.

Official code/checkpoint/reproducible mechanism: no local official active-camera or illumination stack is verified. A faithful local proxy would need a non-privileged segmentation/crop/overlay source from deployment RGB, not simulator object labels.

Assumption or limitation extended: these priors assume active perception hardware, camera control, or external segmentation. SVO would test a fixed-camera, deployment-observable semantic virtual observation overlay/crop around local SmolVLA without changing the action policy architecture.

Minimal technical difference proposed by Ours:

- create a non-privileged semantic overlay or crop from deployment RGB only;
- train or calibrate SmolVLA adapter behavior against raw-RGB and overlay/crop variants;
- reject if a simple crop, random erasing, Sobel/edge boost, or raw-RGB LoRA baseline matches it;
- do not use simulator segmentation, BDDL target labels, task IDs, reset identities, or reward at inference.

Why it could improve the same claim axis: the active/semantic perception priors show that improving what the policy sees can matter as much as changing the action model. If local failures are perception-limited, a semantic virtual observation could improve closed-loop success without another action residual.

### Quality Screen

Provisional novelty:

- Meaningful if the local overlay is deployment-observable and not a privileged simulator mask.
- Weak if it reduces to Sobel, random erasing, or another PSE/GCAP-like visual transform.

Prior-anchor strength:

- Strong external results for perception-side intervention.
- Local official reproduction is weak because active cameras and physical illumination are unavailable in the current LIBERO rollout.

Mechanism plausibility:

- Problem condition -> fixed RGB can hide object identity, boundaries, or task-relevant workspace regions.
- Proposed method -> expose object/region cues through a semantic virtual observation channel.
- Expected action behavior -> improved grasp/placement targeting under visual ambiguity while preserving raw behavior elsewhere.

Data and supervision viability:

- RGB observations exist.
- Non-privileged segmentation/overlay source is not yet verified.
- Prior PSE, GCAP, and PatchGuard visual routes impose high simple-baseline risk.

Identity-preserving integration:

- The safest integration is a two-view or gated overlay defaulting to raw RGB.
- Any overlay that globally changes all observations receives a high disruption penalty.

Decisive experiment feasibility:

- Stage 0 source gate can verify non-privileged overlay generation, raw-vs-overlay action delta, and simple visual baselines.
- Closed-loop feasibility depends on a valid segmentation/crop source and visual headroom.

Score:

- provisional novelty: `19 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `15 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `78 / 100`

## Selection

Selected method: `PESA-VLA`.

Selection reason:

- It has the strongest combination of positive external prior, local feasibility, identity-preserving integration, and decisive five-policy comparison.
- It responds directly to the MARC/DAGR/MTF failure pattern: global action changes and learned wrappers can be destructive unless the pretrained policy remains an explicit, read-only prior.
- It is better anchored and more locally executable than SVO, and less source-blocked than AMH.
- It can be killed cleanly before rollout if standard LoRA, a PriorVLA-style proxy, a no-spectral ablation, or a simple clean-retention baseline explains the gain.

Immediate next steps:

1. Freeze a PESA-VLA Researcher A proposal and hash it.
2. Reviewer B attacks novelty against PriorVLA, LoRA-SP, VLA-GSE, standard LoRA, CLARE, MTF no-retention, and generic PEFT expert routing.
3. Researcher A provides one rebuttal if the method remains nontrivial and locally feasible.
4. Write `reports/pesa_vla/mathematical_mechanism_audit.md`, preregistration, and prototype protocol.
5. Implement only a Stage 0 development audit first: standard LoRA/prior-proxy headroom, split integrity, initial identity, gradient flow, action-delta bounds, clean retention, and no confirmatory-test identity use before any expensive training or rollout.
