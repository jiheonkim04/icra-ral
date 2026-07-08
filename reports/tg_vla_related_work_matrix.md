# TG-VLA Related Work Matrix

Date: 2026-07-09 KST

Primary sources checked in this run:

- ActionMap: https://arxiv.org/abs/2606.06904
- Direct grounded 3D point action-head injection: https://arxiv.org/html/2606.27663v1
- LIBERO-Para: https://arxiv.org/abs/2603.28301 and https://huggingface.co/datasets/HAI-Lab/LIBERO-Para
- OpenVLA-OFT: https://openvla-oft.github.io/
- SmolVLA: https://arxiv.org/abs/2506.01844 and https://huggingface.co/lerobot/smolvla_base
- LoRA-SP: https://arxiv.org/abs/2603.07404
- QVLA: https://arxiv.org/abs/2602.03782
- ActQuant: https://arxiv.org/abs/2605.24011

| Anchor | What it solves | What it does not solve | Strongest reported baseline or comparison | Is TG-VLA genuinely different? | Could a simple baseline explain TG-VLA? |
| --- | --- | --- | --- | --- | --- |
| ActionMap | Replaces single-point action prediction with a voxel heatmap action head. Reports gains over OpenVLA-OFT L1 regression on LIBERO and validates that action representation is a strong lever. | Does not explicitly bind instruction-resolved object/target priors to action generation; not primarily a paraphrase/object lexical robustness method. | OpenVLA-OFT L1 regression and another distinct VLA backbone at matched training steps. | Potentially, if TG-VLA targets semantic/object grounding in the action pathway rather than action-space heatmap structure alone. | Yes. ActionMap plus standard LoRA or paraphrase augmentation could absorb the gain unless TG-VLA beats it on wrong-target and object-lexical metrics. |
| Direct grounded 3D point action-head injection | Lifts a 2D target point to 3D, computes gripper-to-target displacement, and injects the spatial embedding into the action head via AdaLN. Reports large LIBERO-PRO task and position perturbation gains. | Uses an externally grounded point and focuses on spatial/task generalization; does not directly solve object lexical robustness or target resolution from paraphrased language without oracle/grounding assumptions. | GR00T-N1.6 and another VLA backbone under LIBERO/LIBERO-PRO perturbations; compares 2D prompt/visual variants and 3D action-head injection. | Only if TG-VLA resolves target/object priors from language and visible candidates without leakage, and adds consistency/sensitivity objectives beyond a single 3D point. This is the closest novelty threat. | Very likely. A single 3D target point or destination point baseline is mandatory if TG-VLA uses any spatial target representation. |
| LIBERO-Para | Controlled benchmark for paraphrase robustness, independently varying action and object expressions; reports 22-52 pp degradation and object lexical variation as a dominant failure mode. | It is an evaluation benchmark and metric package, not a training method or action-pathway architecture. | Seven VLA configurations from 0.6B to 7.5B; PRIDE difficulty metric. | Yes, if TG-VLA turns the benchmark failure into a real adapter/action-pathway method. | Yes. Prior local PRISM evidence was killed because canonicalization-only beat the proposed consistency method. |
| OpenVLA-OFT | Strong VLA fine-tuning recipe with parallel decoding, action chunking, continuous action representation, and L1 regression. Reports 97.1% average LIBERO success and large speed gains over base OpenVLA. | Too heavy for local RTX 5080 training; does not specifically target target/object lexical robustness with action-path target grounding. | pi_0, MDT, Seer, DiT Policy, Octo, Diffusion Policy, and OpenVLA baselines. | TG-VLA can differ only as a low-resource target-grounded adapter axis, not as a general fine-tuning recipe. | Yes. OpenVLA-OFT-style L1/LoRA fine-tuning may match TG-VLA unless target metrics are isolated. |
| SmolVLA | Compact VLA intended for single-GPU training and consumer hardware/CPU deployment; model card exposes inference and training-step APIs. | It is a backbone/tooling candidate, not a target-grounding method. | Larger VLA models and standard simulated/real benchmarks in the SmolVLA paper. | TG-VLA uses SmolVLA as the local executable backbone. | Yes. SmolVLA standard fine-tuning or LoRA must be a baseline. |
| LoRA-SP | Rank-adaptive VLA fine-tuning; reports that standard small-rank LoRA can be mismatched for VLA transfer and that adaptive capacity improves over standard LoRA. | Does not target object/target grounding or paraphrase/counterfactual target sensitivity. | Standard LoRA, full fine-tuning, multiple VLA backbones including SmolVLA and pi_0. | TG-VLA differs only if the adapter content is target/action-path grounding rather than adaptive PEFT capacity. | Yes. LoRA-SP could dominate a naive fixed-rank TG-VLA adapter. |
| QVLA | Action-centric channel-wise quantization for VLA deployment; measures final action sensitivity under quantization. | Compression/deployment method, not target grounding or language robustness training. | SmoothQuant and other quantization baselines on LIBERO/OpenVLA-OFT. | TG-VLA is orthogonal; QVLA is a possible future deployment constraint. | Less direct, but quantization-sensitive action metrics should not be mistaken for target grounding. |
| ActQuant | Sub-4-bit action-guided mixed-precision PTQ; reports high retention for OpenVLA-OFT and pi_0.5 with large memory reduction. | Compression only; does not solve target/object grounding. | PTQ baselines and compressed OpenVLA-OFT/pi_0.5. | Orthogonal to TG-VLA. | Not likely to explain target robustness, but useful as low-resource context. |

## Novelty Gate

The direct 3D point action-head injection paper is the major pre-mortem risk. TG-VLA cannot be framed as "grounding injected into the action head" alone. The defensible gap is narrower:

- target/object prior is resolved from instruction semantics and visible/object names rather than supplied as an oracle point,
- the action-path adapter is trained to preserve same-target paraphrase consistency,
- counterfactual target changes are explicitly separated,
- canonicalization-only and single-point injection are required baselines.

If the method collapses to a target point, prompt target name, lexical normalization, or generic LoRA capacity, it is not novel enough.
