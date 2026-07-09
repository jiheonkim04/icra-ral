# SmolVLA 7D Standard Replay Baseline Plan

- Use local LIBERO HDF5 tasks only.
- Use train/eval demo holdout with no train/eval demo leakage.
- Train fixed LIBERO_7D mean, ridge, MLP, frozen/base adapter, rank-4 LoRA, and rank-8 LoRA baselines.
- Replay expert, mean, ridge, MLP, and best LoRA on held-out exact-init demos.
- Diagnose clipping and action range; do not change the method.
