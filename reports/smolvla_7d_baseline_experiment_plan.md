# SmolVLA 7D Baseline Experiment Plan

- Use fixed LIBERO_7D labels only.
- Use train-split-only 7D normalization.
- Compare mean, per-task mean, persistence, ridge, MLP, frozen state-proj adapter, no-LoRA adapter, and rank-4/rank-8 state-proj LoRA adapters.
- Do not use SO100 action normalizer, old 6D action labels, hard-coded gripper fill, rollout benchmark, OpenVLA-OFT, or downloads.
- Run optional replay/progress only if a bounded executable 7D adapter bridge is available and action metrics beat mean and MLP/ridge.
