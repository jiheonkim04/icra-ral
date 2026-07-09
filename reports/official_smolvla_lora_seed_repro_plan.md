# Official SmolVLA Rank-4 LoRA Seed Reproduction Plan

Date: 2026-07-10 KST

Purpose: audit standard rank-4 LoRA seed robustness under the fixed official SmolVLA-LIBERO manifest and metric protocol.

Boundary:

- no new method
- no FCAR revival or tuning
- no routing model
- no simulator rollout or full benchmark
- no OpenVLA-OFT
- no downloads
- no old custom LIBERO_7D route
- no static-alpha tuning on test

Preflight:

- model path: `C:\assets\checkpoints\smolvla_libero`
- dataset path: `C:\assets\datasets\lerobot_libero`
- manifest path: `reports\official_smolvla_split_manifest.json`
- existing artifact path: `reports\official_smolvla_stable_prediction_artifact.json`
- selected seeds: `[11, 22, 33]`
- device plan: `CUDA rank-4 LoRA seed training; stop with CPU_FALLBACK_BUG if params or tensors remain on CPU`
- static merge reproduced: `{'selected_alpha': 0.5, 'test_action_l2': 0.08113506}`
- estimated runtime: `about one LoRA-only 2800-record evaluation per seed; bounded by a three-hour cap`
