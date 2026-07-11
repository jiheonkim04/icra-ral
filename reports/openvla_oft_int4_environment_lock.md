
# OpenVLA-OFT INT4 Environment Lock

- machine scope: `single_local_rtx5080_only`
- GPU: `NVIDIA GeForce RTX 5080`
- CUDA available: `True`
- PyTorch: `2.10.0+cu128` compiled CUDA `12.8`
- bitsandbytes: `0.49.2`
- accelerate: `1.14.0`
- transformers: `4.40.1`
- OpenVLA-OFT source: `e4287e94541f459edc4feabc4e181f537cd569a8` at `/mnt/c/assets/repos/openvla-oft`
- Transformers fork commit: `bc339d9ad707454c0c115970db43c260067c61ab`
- dlimp fork commit: `040105d256bd28866cc6620621a3d5f7b6b91b46`
- LIBERO source: `8f1084e3132a39270c3a13ebe37270a43ece2a01` at `/home/jiheon/assets/repos/LIBERO`
- full BF16 load attempted: `false`
- CPU/disk offload used: `false` / `false`
- training/fine-tuning: `false`
- old custom `LIBERO_7D` path: `false`

Compatibility deviations are recorded in `runs/openvla_oft_int4/env_lock.json`; the main one is retaining PyTorch `2.10.0+cu128` for RTX 5080 support while pinning the OpenVLA source and custom Transformers fork.
