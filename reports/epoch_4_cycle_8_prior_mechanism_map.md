# Epoch 4 Cycle 8 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the next method after the closed DAGR-VLA Stage B kill. DAGR-VLA is archived as `DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD` and must not be rescued by retuning `dagr_a020_route_mlp`, changing route thresholds, changing task/reset identities, changing the policy list, or reinterpreting the fixed Stage B result.

## Local Constraints From Prior Results

The next method must not be:

- another dynamic arm/gripper routing or route-gated group residual method like DAGR;
- another milestone-frame selection, retained-frame sampling, or no-retention MTF rescue;
- another reflective consequence-calibration wrapper like RAC;
- another failure-aware residual field like FANG;
- another action-evolved state controller like EvoState;
- another nearest-memory contrastive action method like CAVM;
- another no-context receding-chunk replanner like RCV;
- another photometric perturbation ensemble like PSE;
- another local ActionMap-style proxy extension unless it first clears an official-source gate.

DAGR showed that a learned residual module can beat a weak closest-prior proxy yet still lose badly to Base and a simple gripper heuristic. Cycle 8 therefore prioritizes a stronger positive prior whose first comparison includes a plain L1/adapter proxy and a static mixture baseline, so an ordinary adapter cannot be hidden as novelty.

## Close Sources

### OpenVLA-OFT

Full title: Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success.

URLs:

- paper: https://arxiv.org/abs/2502.19645
- project/code/models: https://openvla-oft.github.io/

AUTHOR_STATED:

- OpenVLA-OFT combines parallel decoding, action chunking, continuous actions, and an L1 regression objective.
- It reports a LIBERO four-suite average improvement from OpenVLA's `76.5%` to `97.1%`, with `26x` action-generation throughput.
- The project releases code and pretrained model checkpoints.
- The authors argue that L1 regression can regularize noisy demonstrations by committing to a median-like action mode where diffusion policies can reproduce suboptimal behavior.

INDEPENDENTLY_INFERRED:

- The central positive prior is not "LoRA" by itself. It is the continuous deterministic L1 action-generation recipe plus chunked parallel decoding.
- A direct local official OpenVLA-OFT training run is not currently feasible on the active 16GB local GPU budget; the project lists LIBERO fine-tuning memory above the local envelope.
- A faithful transparent local proxy is feasible: train a continuous L1/Huber action adapter on the same SmolVLA/LIBERO records, preserve official 7D action semantics, and compare it directly against any proposed method.
- The proposal must beat this plain L1 proxy and a static validation-selected mixture, otherwise the method is just ordinary regression or static mixing.

CROSS_PAPER_SYNTHESIZED:

- OpenVLA-OFT's L1 finding directly attacks flow/diffusion-style action generation on noisy demonstration data.
- SnapFlow and ReactVLA independently suggest that flow-action generation can be improved by calibrating or simplifying the flow path, but their main benefits are speed and flow-step efficiency rather than a clear local success headroom guarantee.
- A small median anchor around SmolVLA is a lower-resource way to test whether deterministic L1/Huber action anchors can correct local flow-action errors without replacing the pretrained policy.

Mechanism fields:

- observation/input: official SmolVLA RGB observations, language instruction or transparent task-language proxy in development, 8D state when available, base 7D action chunk, task key, chunk phase;
- learned representation: robust median action anchor `m_t in R^7`, disagreement gate `g_t in [0,1]`, and bounded correction `Delta_t`;
- supervision: expert 7D action chunks, frozen-base 7D action chunks, train-only residual and action scales;
- objective: L1/Huber anchor loss, base-retention loss, gate-label BCE from train-only base/expert disagreement, and action-delta regularization;
- policy component changed: small external median-anchor/gate module around frozen SmolVLA actions;
- action-generation mechanism: base action plus a clipped, gate-scaled move toward the median anchor;
- inference-time intervention: one base policy call plus a small adapter; no reward, simulator state, reset identity, or future observation;
- benchmark condition: official paired LIBERO manifest with exactly five policies;
- primary metric: task-balanced official closed-loop success and paired deltas;
- demonstrated causal link externally: continuous L1 action recipes can substantially improve LIBERO success and speed over the original OpenVLA action decoder;
- untested causal link locally: whether a bounded median anchor can improve frozen SmolVLA beyond a faithful L1 proxy, a no-gate ablation, and a static-mix simple baseline.

### ReactVLA

Full title: ReactVLA: Fast and Lightweight Reactive Robot Manipulation via Improved Mean-Flow Action Generation.

URLs:

- paper: https://arxiv.org/abs/2606.14255
- project: https://game-loader.github.io/ReactVLA/

AUTHOR_STATED:

- ReactVLA uses improved Mean Flow action generation and Attention Residuals.
- It reports stronger LIBERO performance than similarly sized baselines, including SmolVLA, and much lower latency.
- The project page reports LIBERO average success `88.0` for ReactVLA versus `87.3` for SmolVLA and latency `18.3 ms` versus `74.1 ms`.

INDEPENDENTLY_INFERRED:

- ReactVLA is a strong positive prior for changing action generation rather than adding post-hoc wrappers.
- Direct local reproduction would require architecture-level retraining that is heavier than the current cycle.
- The usable local lesson is that action-generation calibration should remain lightweight and latency-audited.

CROSS_PAPER_SYNTHESIZED:

- ReactVLA makes latency a required metric for any Cycle 8 action-generation method.
- It is not enough to add multiple SmolVLA calls or a large search wrapper; the method must preserve or clearly justify inference overhead.

### SnapFlow

Full title: SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation.

URL: https://arxiv.org/abs/2604.05656

AUTHOR_STATED:

- SnapFlow compresses multi-step flow denoising into one forward pass through progressive self-distillation.
- It uses two-step Euler shortcut velocity targets and a zero-initialized target-time embedding.
- It reports `98.75%` average LIBERO success on pi0.5, matching or slightly exceeding a 10-step teacher, and reports an `8.3%` SmolVLA MSE reduction plus speedup.

INDEPENDENTLY_INFERRED:

- The positive prior is flow-path calibration, not a residual wrapper.
- Full self-distillation is locally expensive and primarily improves latency; the local simulator already reports low per-policy latency for current wrappers.
- SnapFlow is therefore a secondary source for the selected method's identity-preserving calibration discipline, not the closest prior.

### ActionMap

Full title: ActionMap: Robot Policy Learning via Voxel Action Heatmap.

URLs:

- paper: https://arxiv.org/abs/2606.06904
- project/code preview: https://github.com/showlab/ActionMap

AUTHOR_STATED:

- ActionMap replaces a single-point VLA action decoder with a voxel action heatmap.
- It reports cross-backbone LIBERO and real-world Franka gains, including `+8.2%` over OpenVLA-OFT's L1 head on the LIBERO four-suite average.
- The current repository is a pre-release with a core `HeatmapActionHead` preview, not a complete training stack.

INDEPENDENTLY_INFERRED:

- ActionMap is a strong action-representation prior, but the local mini-anchor diagnostic failed against mean-action and cheap MLP baselines.
- A future revival needs an official-source gate using real hidden action-token states, not another tiny local heatmap proxy.
- Cycle 8 may list ActionMap as a candidate, but a selected method must not depend on the old failed local proxy.

### AffordanceVLA And GEAR-VLA

URLs:

- AffordanceVLA paper/code: https://arxiv.org/abs/2606.06155 and https://github.com/Skywalker-yqz/AffordanceVLA/
- GEAR-VLA paper/code placeholder: https://arxiv.org/abs/2606.08530 and https://github.com/babynabeauty/GEAR-VLA

AUTHOR_STATED:

- AffordanceVLA introduces Which2Act, Where2Act, and How2Act structured affordance forecasting and reports strong simulated and real-world results.
- Its released repository includes LIBERO/CALVIN configs but notes that dataset-specific loaders are not bundled and that full stages are multi-GPU.
- GEAR-VLA reports geometry-aware action representations and state-of-the-art or strong generalization results across LIBERO, LIBERO-Plus, RoboTwin, and real robots; the code repository currently says "coming soon".

INDEPENDENTLY_INFERRED:

- These are strong priors for structured intermediate supervision, but direct local reproduction is weak because dense affordance, 3D, and data-loader requirements are not locally available.
- A local weak-affordance proxy would be at high risk of becoming another phase/contact heuristic unless a Stage 0 label-health gate is very strict.

## Cycle 8 Opportunity

The strongest immediate opportunity is `MARC-VLA`: Median-Anchored Regression Correction for frozen SmolVLA flow actions.

It is anchored primarily to OpenVLA-OFT's positive L1 continuous-action result, with ReactVLA/SnapFlow as secondary flow-action calibration context. Instead of replacing SmolVLA or adding another route-specific residual, MARC trains a robust median action anchor and an observable disagreement gate. At inference it emits the frozen base action plus a clipped, gate-scaled correction toward the median anchor. The initial behavior is exact base passthrough.

This changes at least four dimensions relative to DAGR and MTF:

- representation: robust median anchor and base-anchor disagreement gate rather than action-group route logits or milestone-frame scores;
- supervision: L1/Huber action-anchor targets and train-only disagreement labels rather than group residual route labels or retained-frame labels;
- objective: robust median-style regression with base-retention regularization rather than group-routed residual loss or retention sampling;
- policy generation: bounded move toward a deterministic action anchor rather than component-specific residual routing;
- claim axis: whether an OpenVLA-OFT-style deterministic action anchor can improve SmolVLA flow actions without replacing the pretrained policy.

The critical Reviewer B baselines are:

- unmodified frozen SmolVLA;
- a faithful transparent OpenVLA-OFT-style L1 continuous-action proxy;
- `MARC-VLA` full;
- `MARC` no-disagreement-gate ablation;
- one static validation-selected Base/L1-proxy mixture simple killer.
