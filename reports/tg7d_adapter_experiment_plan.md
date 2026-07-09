# TG-7D Adapter Experiment Plan

- STATE 1: prove a no-leakage LIBERO-Para/object/counterfactual split exists.
- STATE 2: run a tiny fixed LIBERO_7D rank-4 adapter gate only if STATE 1 is green.
- Required arms: mean-action, ridge/MLP, standard SmolVLA 7D LoRA/adapter, canonicalization-only, simple paraphrase augmentation, TG-7D Adapter, and oracle target upper bound.
- Metrics: clean/paraphrase/object action L2, translation L2, rotation L2, gripper error/accuracy, train/eval gap proxy, target consistency, counterfactual sensitivity, trainable params, VRAM, runtime.
- No OpenVLA-OFT, downloads, full benchmark, old TCA-Select, old 6D/SO100 action path, hard-coded gripper fill, or paper claims.
