# TG-7D Adapter Related Work Matrix

| Anchor | Relevance | TG-7D distinction | Required anti-baseline |
| --- | --- | --- | --- |
| ActionMap | Voxel heatmap action head improves VLA action representation on LIBERO and real robot tasks. Source: https://arxiv.org/abs/2606.06904 | TG-7D is not a heatmap decoder; it tests instruction-resolved target priors in a fixed 7D adapter. | Standard LoRA/action imitation and future ActionMap-style head if scaled. |
| Direct grounded 3D point action-head injection | Injects grounded 3D target points into the action head. Source: https://arxiv.org/html/2606.27663v1 | TG-7D may not rely on oracle 3D points; priors must come from instruction text plus visible names. | Single-point/destination oracle if any spatial point is used. |
| LIBERO-Para | Provides paraphrase/object lexical benchmark and PRIDE-style difficulty evidence. Source: https://arxiv.org/abs/2603.28301 and https://huggingface.co/datasets/HAI-Lab/LIBERO-Para | TG-7D is a method gate using LIBERO-Para as evaluation metadata. | Canonicalization-only and simple paraphrase augmentation. |
| OpenVLA-OFT | Strong fine-tuning recipe with continuous actions and high LIBERO success. Source: https://arxiv.org/abs/2502.19645 | TG-7D is a bounded SmolVLA adapter path, not OpenVLA-OFT. | OpenVLA-OFT remains a nonlocal paper baseline; do not run locally here. |
| SmolVLA | Compact VLA backbone suited to consumer hardware. Source: https://arxiv.org/abs/2506.01844 | TG-7D uses SmolVLA as the executable backbone. | Standard SmolVLA 7D LoRA/adapter. |
| Old Target-Prior TCA | Earlier target-prior route had reusable prior/leakage audits but weak proxy/action heads. | TG-7D must use fixed real SmolVLA/LIBERO_7D path, not old TCA-Select or weak MLP route. | Mean-action, ridge/MLP, standard LoRA. |
| Canonicalization / paraphrase augmentation | Strong simple language baselines; prior PRISM route was killed by canonicalization. | TG-7D must beat them rather than rename them. | Canonicalization-only and simple paraphrase augmentation. |
