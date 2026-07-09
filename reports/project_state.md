# Project State

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-mini-repro`

Current decision: `READY_FOR_OFFICIAL_BASELINE_SCALEUP`

## Current Route

The archived custom SmolVLA 7D adapter route remains stopped. The valid route is official SmolVLA/LeRobot reproduction first, with official preprocessing, normalization, action conventions, dataset format, and evaluation stack.

## Evidence Summary

- User-approved official assets were downloaded: `lerobot/smolvla_libero` and `lerobot/libero`.
- Visible downloaded asset size is `2,842,253,889` bytes, about `2.647 GiB`.
- Official `lerobot/libero` uses 8D state, 7D action, two 256x256 video keys, LeRobot metadata/stats, and train split `0:1693`.
- Official `smolvla_libero` loads through `SmolVLAPolicy.from_pretrained`.
- Official pre/postprocessors load through `make_pre_post_processors`.
- LeRobot dataset sample loading works locally with `video_backend='pyav'`.
- One-sample and five-sample official offline action-prediction smokes worked on CUDA.
- Tiny standard LoRA smoke worked: rank 4, batch size 1, 5 steps, loss `0.003114 -> 0.003007`, peak VRAM about `1.103 GiB`.
- Official simulator eval was not run; `lerobot-eval --env.type=libero` still requires WSL/Linux/MuJoCo readiness.
- OpenVLA-OFT, full benchmark, long training, and custom `LIBERO_7D` adapter route were not used.

## Conclusion

`READY_FOR_OFFICIAL_BASELINE_SCALEUP`

Next valid step: create a bounded official baseline scaleup run using the downloaded `smolvla_libero` + `lerobot/libero` assets, rank-4 LoRA, batch size 1, fixed small step count, full CUDA device/memory/autocast logging, and no simulator benchmark until WSL/Linux eval readiness is checked.
