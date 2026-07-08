# Latest Anchor Paper Matrix

Scan date: 2026-07-08.

Scope: latest VLA action-decoder, semantic grounding, paraphrase robustness, low-resource adapter/quantization, and action-chunk correction context. Sources were inspected through arXiv or official project/GitHub pages only. No PDFs were downloaded to the repo, no datasets were downloaded, and no code was run.

## Anchor Matrix

| Anchor | Primary source checked | What it advances | What it does not solve for this project | Impact on Target-Grounded ActionMap |
| --- | --- | --- | --- | --- |
| ActionMap | arXiv:2606.06904, revised 2026-06-10, https://arxiv.org/abs/2606.06904; project/code page https://github.com/showlab/ActionMap | Replaces single-point continuous action decoders with a voxel action heatmap head; reports LIBERO and real Franka gains, including gains over OpenVLA-OFT L1 regression. | Does not explicitly condition the heatmap on semantic target/object candidates or test paraphrase/object lexical robustness as the central mechanism. | Main action-decoder anchor. Candidate must beat ActionMap alone and cannot claim novelty as heatmap-only. |
| Direct action-head injection of a grounded 3D point | arXiv:2606.27663, submitted 2026-06-26, https://arxiv.org/abs/2606.27663 | Injects one grounded 3D point into the action head through adaptive layer normalization; reports large LIBERO-PRO gains for GR00T-N1.6 and pi0.5 under task/position perturbations. | Uses a single grounding point, not a semantic target-conditioned voxel action heatmap; does not directly address paraphrase/object lexical robustness. | Mandatory baseline risk. Target-Grounded ActionMap must beat single-point and destination-only target injection. |
| LIBERO-Para | arXiv HTML 2603.28301v1, https://arxiv.org/html/2603.28301v1; project page https://cau-hai-lab.github.io/LIBERO-Para/ | Introduces controlled paraphrase robustness benchmark and PRIDE; finds 22-52 pp degradation under paraphrasing, object lexical variation as dominant, and mostly planning-level failures. | Diagnostic benchmark and metrics, not an action decoder method. Does not solve robustness with target-conditioned action heatmaps. | Supplies the object lexical/paraphrase failure axis and metrics. Candidate must beat canonicalization-only and preserve counterfactual sensitivity. |
| OpenVLA-OFT | arXiv:2502.19645, revised 2025-04-28, https://arxiv.org/abs/2502.19645; project https://openvla-oft.github.io/ | Strong optimized fine-tuning recipe: parallel decoding, action chunking, continuous action representation, L1 objective, strong LIBERO success and throughput. | Not target-grounded heatmap decoding; local full training is outside current safe scope. | Strong baseline and future real VLA route. LoRA/adapters may use OFT-style recipe only as tool/context. |
| LoRA-SP | arXiv:2603.07404, submitted 2026-03-08, https://arxiv.org/abs/2603.07404 | Shows VLA adaptation may require task-varying and higher-rank capacity; rank-adaptive LoRA improves multi-task real-robot success over standard LoRA. | Adapter capacity method, not semantic target-grounded action decoding. | Supports using adapters carefully later; forbids making LoRA itself the novelty. |
| QVLA | arXiv:2602.03782, submitted 2026-02-03, https://arxiv.org/abs/2602.03782 | Action-centric quantization with channel-wise bit allocation; reports lower VRAM while retaining OpenVLA-OFT performance. | Compression/deployment method, not target grounding or paraphrase robustness. | Context for low-resource constraints only. |
| ActQuant | arXiv:2605.24011, revised 2026-06-04, https://arxiv.org/abs/2605.24011 | Action-guided mixed-precision PTQ for sub-4-bit VLA deployment; reports LIBERO and real UR3 evidence. | Quantization method, not semantic target heatmap or decoder-level grounding. | Context for efficient future evaluation; not novelty. |
| DyQ-VLA | arXiv:2603.07904, revised 2026-03-14, https://arxiv.org/abs/2603.07904 | Dynamic temporal-aware VLA quantization using kinematic proxies; reports memory retention and speedups. | Deployment efficiency, not target-grounded action decoding. | Context for low-resource evaluation and adapter/quantization constraints only. |
| CAC-VLA | arXiv HTML 2607.04816v1, posted 2026-07-07, https://arxiv.org/html/2607.04816v1 | Learns VLM-native latent-action predictions and context-gated action-expert conditioning; reports LIBERO, LIBERO-Plus, and initial real-world evidence. | Latent-action conditioning, not explicit semantic target/object heatmap conditioning; does not isolate object lexical paraphrase robustness as the main decoder design. | Very fresh novelty-risk anchor. Candidate must not be described merely as "context-gated action conditioning." |
| Decoupling declarative from procedural in VLA / w2 VLA | arXiv HTML 2606.21496v1, https://arxiv.org/html/2606.21496v1 | Decouples where/what information, uses spatial localization heatmaps and FiLM-style conditioning, and tests skill-object transfer in low-data settings. | Final action head is not an ActionMap-style voxel action heatmap; focus is skill transfer and decoupled state-token modulation rather than target-conditioned action-distribution decoding. | Major novelty-risk anchor. Target-Grounded ActionMap must differentiate as semantic target prior -> voxel action heatmap distribution, not generic where/what FiLM. |
| GuidedVLA | arXiv HTML 2605.12369v1, https://arxiv.org/html/2605.12369v1 | Guides action-decoder learning with specialized heads for object grounding, spatial geometry, and temporal skill logic. | Not specifically ActionMap-style target-conditioned voxel heatmap; uses manually defined auxiliary factors and broader attention specialization. | Adds baseline/related-work pressure: object grounding alone is no longer novel. |
| RoVLA | arXiv HTML 2605.19678v1, https://arxiv.org/html/2605.19678v1 | Multi-consistency constraints for robustness under instruction, trajectory, and observation transformations; includes paraphrase-style robustness. | Generic consistency/invariance framework, not target-conditioned heatmap action decoder. | Candidate cannot claim generic paraphrase consistency as novelty; it must show decoder-specific target grounding gains beyond canonicalization and generic consistency. |
| OA-WAM | arXiv HTML 2605.06481v1, https://arxiv.org/html/2605.06481v1 | Object-addressable world action modeling for locating target objects under scene shifts. | World-action model and object-slot prediction, not an ActionMap heatmap decoder. | Useful adjacent object-addressability context; not the primary route. |
| Gaze2Act | arXiv HTML 2605.30282v1, https://arxiv.org/html/2605.30282v1 | Uses gaze-derived object masks and gaze points for coarse-to-fine target specification. | Requires human gaze/interactivity; not language-only semantic target grounding or heatmap decoding. | Supports need to beat point/mask target signals; not a direct target. |
| VLA-Corrector | arXiv:2607.01804, submitted 2026-07-02, https://arxiv.org/abs/2607.01804; project https://zju-omniai.github.io/vla-corrector/ | Detect-and-correct inference for action chunks using latent visual deviation monitoring and corrective replanning. | Action-chunk correction, not target-grounded action heatmap decoding. | Fallback context only if target-grounded ActionMap is rejected. |
| A2C2 | arXiv:2509.23224, submitted 2025-09-27, https://arxiv.org/abs/2509.23224 | Lightweight real-time correction head for VLA action chunks, improving reactivity under delay/horizon stress. | Correction head, not semantic target heatmap or paraphrase robustness. | Baseline/related-work context for any future correction route, not current candidate. |
| RTC | arXiv:2506.07339, https://arxiv.org/abs/2506.07339 | Asynchronous real-time chunking for flow/diffusion policies without retraining. | Latency/chunk execution method, not semantic target grounding. | Correction-route baseline only. |

## Gap Synthesis

The defensible gap is narrow but real:

- ActionMap makes action geometry explicit through voxel heatmaps, but does not explicitly condition the heatmap on semantic target/object candidates.
- Direct 3D point injection makes grounding explicit, but only as a single point injected into the action head, not as a target-conditioned action distribution.
- LIBERO-Para shows that object lexical variation and paraphrasing break VLA task identification, but it is a benchmark/diagnostic rather than an action-decoder method.
- w2 VLA, GuidedVLA, CAC-VLA, and RoVLA raise the novelty bar. The new claim cannot be "add semantic conditioning" or "add consistency"; it must be a decoder-level claim about semantic target priors shaping an ActionMap-style voxel action heatmap, with counterfactual target sensitivity and object lexical robustness.

## Candidate Positioning

Working title:

**Language-Grounded Action Heatmap for VLA Manipulation**

Claim boundary:

The method would combine an ActionMap-style voxel action heatmap decoder with explicit target/object semantic prior conditioning through FiLM, AdaLN, or residual gating, plus counterfactual target consistency and paraphrase/object lexical robustness objectives. LoRA/adapters may be used later only to train this head efficiently.

Immediate gating decision:

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`
