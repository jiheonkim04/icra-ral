# Epoch 4 Cycle 9 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the next method after the valid MARC-VLA Stage A kill. MARC-VLA is archived as `MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE` and must not be rescued by retuning `marc_a020_gate_mlp`, changing correction thresholds, changing policy identities, changing task/reset identities, or reinterpreting the frozen Stage A result.

## Local Constraints From Prior Results

The next method must not be:

- another median-anchor, static L1 mixture, or disagreement-gated MARC rescue;
- another dynamic arm/gripper routing or route-gated group residual method like DAGR;
- another milestone-frame selection, retained-frame sampling, or no-retention MTF rescue;
- another reflective consequence-calibration wrapper like RAC;
- another failure-aware residual field like FANG;
- another action-evolved state controller like EvoState;
- another nearest-memory contrastive action method like CAVM;
- another no-context receding-chunk replanner like RCV;
- another photometric perturbation ensemble like PSE;
- another local ActionMap-style proxy extension unless it first clears an official-source gate;
- an ECHO-style candidate reranker without a new candidate generator and a fresh headroom gate.

MARC showed that a learned bounded action correction can be active, disk-reloadable, and validation-clean while still catastrophically losing closed loop. Cycle 9 therefore prioritizes methods whose positive prior is stronger than "small residual around SmolVLA" and whose first development audit can determine whether the method is merely ordinary LoRA, ordinary heatmap decoding, or ordinary semantic observation preprocessing.

## Close Sources

### PriorVLA

Full title: PriorVLA: Prior-Preserving Adaptation for Vision-Language-Action Models.

URL: https://arxiv.org/abs/2605.10925

AUTHOR_STATED:

- PriorVLA keeps a frozen prior expert as a read-only source and trains an adaptation expert for downstream specialization.
- Expert Queries capture scene priors from the pretrained VLM and motor priors from the Prior Expert.
- The paper reports stronger adaptation than full fine-tuning and state-of-the-art VLA baselines, including `99.1%` average success on LIBERO and large few-shot/OOD gains.

INDEPENDENTLY_INFERRED:

- The central positive prior is not generic LoRA. It is the explicit read-only pretrained prior expert plus a trainable adaptation expert, so downstream learning is not allowed to overwrite the prior source.
- Direct official reproduction is not yet established in this repo. A faithful transparent local proxy is feasible: freeze the local SmolVLA policy as prior expert, train a 7D LIBERO adaptation expert, and compare against standard 7D LoRA and no-prior ablations under the same task/reset manifest.
- Any Cycle 9 extension must beat this proxy and a plain LoRA/simple adaptation baseline. Otherwise the method is just ordinary PEFT with extra names.

CROSS_PAPER_SYNTHESIZED:

- The repeated local kills show that methods which globally perturb strong Base actions often fail closed loop even when offline metrics improve.
- PriorVLA supplies the strongest directly relevant positive prior for preserving a pretrained VLA while adapting to downstream robot data.
- LoRA-SP and VLA-GSE supply complementary evidence that the adapter capacity itself should be allocated by spectral/task structure rather than a fixed low-rank knob.

Mechanism fields:

- observation/input: official SmolVLA RGB observations, language instruction, proprioception/state, frozen Base 7D action chunk, train/validation split identity, and task key only for development grouping;
- learned representation: prior-query embedding, spectral adapter basis, per-input/layer energy scores, adaptation action, and clean-retention risk gate;
- supervision: expert 7D action chunks, frozen Base action chunks, train-only validation split metadata, and clean-retention action deltas;
- objective: imitation loss for the adaptation expert, spectral concentration/energy loss, prior-retention loss, clean-action delta regularization, and optional gate BCE only if train-only disruption labels are noncollapsed;
- policy component changed: parameter-efficient adapter/expert path around SmolVLA, with the frozen prior expert left read-only;
- action-generation mechanism: default Base action unless the prior-query/spectral adapter passes bounded intervention and retention checks;
- inference-time intervention: one Base policy call plus a lightweight adapter/expert path; no simulator state, reset identity, reward, future observation, or success oracle;
- benchmark condition: official paired LIBERO manifest with exactly five policies after development gates pass;
- primary metric: task-balanced official closed-loop success and paired deltas;
- demonstrated causal link externally: preserving pretrained priors and adaptive rank/capacity improves VLA adaptation in published LIBERO, RoboTwin, real-robot, and SmolVLA/pi0 settings;
- untested causal link locally: whether a prior-preserving spectral adapter can improve the frozen local SmolVLA 7D closed-loop policy beyond standard LoRA, a PriorVLA-style proxy, and a simple clean-retention baseline.

### LoRA-SP

Full title: Adaptive Capacity Allocation for Vision Language Action Fine-tuning.

URL: https://arxiv.org/abs/2603.07404

AUTHOR_STATED:

- LoRA-SP replaces a fixed-rank LoRA update with an SVD-style vector bank and a small router whose nonnegative scores act like singular values.
- The active rank is selected by an energy target `E(k) >= eta`.
- The paper reports that LoRA-SP matches or exceeds full fine-tuning with fewer trainable parameters and improves multi-task success by up to `31.6%` over standard LoRA across pi0 and SmolVLA real-robot tasks.

INDEPENDENTLY_INFERRED:

- LoRA-SP is a strong positive prior for the capacity-allocation part of a local method, not for a new robot failure mode by itself.
- A faithful local proxy is feasible without full official reproduction: implement the spectral energy rule over the existing local 7D adapter path and compare to fixed-rank LoRA under matched parameter, data, and inference budgets.
- The key reviewer-killer is standard LoRA at reasonable ranks; if it matches or beats the spectral path, there is no local method.

CROSS_PAPER_SYNTHESIZED:

- PriorVLA protects the prior source; LoRA-SP decides how much trainable capacity should be active for the current input and layer.
- Their combination targets a local failure pattern: fixed small adapters or correction heads either under-act or globally disrupt Base.

### VLA-GSE

Full title: VLA-GSE: Boosting Parameter-Efficient Fine-Tuning in VLA with Generalized and Specialized Experts.

URLs:

- paper: https://arxiv.org/abs/2605.06175
- code: https://github.com/YuhuaJiang2002/VLA-GSE

AUTHOR_STATED:

- VLA-GSE spectrally decomposes a frozen backbone into generalized shared experts and specialized routed experts.
- It reports strong PEFT performance under a small trainable-parameter budget, including `81.2%` average zero-shot success on LIBERO-Plus and real-world manipulation gains.
- Official code is linked from the paper record.

INDEPENDENTLY_INFERRED:

- VLA-GSE makes generic expert specialization too crowded to claim as novelty.
- It remains valuable as a mechanism prior for spectral decomposition and a fair comparison family.
- A Cycle 9 method must not claim "experts" or "routing" alone; the local novelty must be the specific prior-preserving, spectral-capacity, clean-retention integration and must be killed if ordinary VLA-GSE-style or fixed LoRA baselines explain it.

CROSS_PAPER_SYNTHESIZED:

- PriorVLA's read-only prior expert and VLA-GSE's generalized/specialized decomposition agree that the pretrained model should remain an explicit participant, not merely an initialization.
- The local method should therefore preserve Base behavior by construction rather than rely on a post-hoc action-delta penalty after training.

### ActionMap

Full title: ActionMap: Robot Policy Learning via Voxel Action Heatmap.

URLs:

- paper: https://arxiv.org/abs/2606.06904
- pre-release code: https://github.com/showlab/ActionMap

AUTHOR_STATED:

- ActionMap replaces a single-point VLA decoder with a voxel heatmap action head.
- It reports gains across LIBERO and real-world Franka manipulation, including `+8.2%` over OpenVLA-OFT's L1 regression head on the LIBERO four-suite average.
- The public repository is a pre-release that exposes the core `HeatmapActionHead`, expects VLA hidden states at action-token positions, and notes that fuller code is coming soon.

INDEPENDENTLY_INFERRED:

- ActionMap remains a strong action-representation prior, but the local mini-anchor already failed against mean-action and cheap MLP baselines with candidate collapse.
- Any future ActionMap candidate must begin with an official-source gate using real hidden action-token states and the official-style heatmap head, not the old CPU mini-proxy.

CROSS_PAPER_SYNTHESIZED:

- ActionMap is the best alternative if Cycle 9 prioritizes action representation over adaptation.
- Its feasibility remains lower than a prior-preserving adapter because SmolVLA hidden action-token extraction and full heatmap training are not yet validated locally.

### ActiveVLA, SaPaVe, And SEVO

URLs:

- ActiveVLA: https://arxiv.org/abs/2601.08325
- SaPaVe: https://arxiv.org/abs/2603.12193
- SEVO: https://arxiv.org/abs/2605.11114

AUTHOR_STATED:

- ActiveVLA adds active perception through 3D critical-region localization, active view selection, and 3D zoom-in for precise manipulation.
- SaPaVe decouples camera and manipulation actions, trains semantic camera control first, and reports up to `31.25%` higher real-world success than recent VLA baselines.
- SEVO improves cross-environment robustness without changing the policy architecture by combining body-fixed cameras, active illumination, segmentation overlay, and diversified data collection; it reports large transfer gains for ACT and SmolVLA on low-cost hardware.

INDEPENDENTLY_INFERRED:

- These papers are strong priors for perception-side intervention, not action residuals.
- Local LIBERO lacks an immediately verified active camera/control channel, and privileged simulator segmentation must not be used as an inference signal.
- A local method would need a deployment-observable semantic virtual observation or view/crop selection gate before any rollout.

CROSS_PAPER_SYNTHESIZED:

- The positive prior is real, but the local pathway is weaker because prior PSE/GCAP/PatchGuard visual routes were killed and the current official rollout stack is fixed-camera.
- This family is kept as a candidate, not selected first, unless a source/feasibility gate shows non-privileged visual overlays or view changes are available.

## Cycle 9 Opportunity

The strongest immediate opportunity is `PESA-VLA`: Prior-Expert Spectral Adaptation for frozen SmolVLA 7D policies.

It is anchored primarily to PriorVLA's read-only prior/adaptation expert separation, with LoRA-SP and VLA-GSE as secondary spectral-capacity priors. Instead of adding another residual action correction, PESA keeps frozen SmolVLA as a prior expert and trains a parameter-efficient spectral adaptation expert that can only move away from Base under bounded, validation-selected prior-query conditions. The initial behavior is exact Base passthrough.

This changes the local method axis relative to MARC, DAGR, MTF, and RAC:

- representation: prior-query embedding plus spectral capacity scores rather than median anchors, route logits, milestone frames, or consequence histories;
- supervision: ordinary 7D action adaptation plus prior retention, not disagreement labels, frame retention labels, or reflective correction labels;
- objective: spectral capacity concentration and clean prior retention rather than action residual regression alone;
- policy generation: an adaptation expert constrained by a frozen prior expert rather than post-hoc correction of the emitted action;
- claim axis: whether explicit prior-preserving spectral adaptation improves local SmolVLA 7D closed-loop success beyond standard LoRA, a PriorVLA-style proxy, and a simple retention baseline.

The critical Reviewer B baselines are:

- unmodified frozen SmolVLA;
- a faithful transparent PriorVLA-style proxy without spectral capacity allocation;
- `PESA-VLA` full;
- `PESA` no-spectral/no-prior-query ablation;
- one strongest simple adaptation baseline, expected to be standard fixed-rank 7D LoRA or a validation-selected clean-retention LoRA mixture.
