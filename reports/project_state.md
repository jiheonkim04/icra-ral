# Project State

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-failure-mining`

Current decision: `GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING`

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
- Bounded official rank-4 LoRA baseline scaleup worked on CUDA: batch size 1, 100 steps, train loss `0.005532921 -> 0.003888785`, trainable params `185,664`, peak CUDA allocation `1104.506 MB`, total runtime `40.813 sec`.
- Frozen/base mini-holdout action L2 was `0.081655363`; rank-4 LoRA mini-holdout action L2 was `0.072837438`.
- Frozen/base mini-holdout eval loss was `0.008015549`; rank-4 LoRA mini-holdout eval loss was worse at `0.020719278`. Keep this mixed signal in future comparisons.
- Official failure mining over 200 held-out frames across 5 task groups found that rank-4 LoRA worsened aggregate action L2 (`0.106514960 -> 0.118024259`) and eval loss (`0.011978370 -> 0.012148290`) versus frozen/base.
- The same run found mixed task/frame interference: LoRA helped `98` frames and hurt `102`, with task-mean help/hurt count `2` / `3`. Mean-action prior was much worse (`1.144859722` action L2), so the pattern is not explained by a trivial prior.
- Strongest method-worthy gap is task/frame-level adapter interference, but kill risk is high because frozen/base is strong and MoIRA-style modular routing is a close recent-paper baseline.
- Official simulator eval was not run; `lerobot-eval --env.type=libero` still requires WSL/Linux/MuJoCo readiness.
- OpenVLA-OFT, full benchmark, long training, and custom `LIBERO_7D` adapter route were not used.

## Conclusion

`GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING`

Next valid step: design, but do not implement, a task-conditional adapter-routing plan on the official SmolVLA-LIBERO path. The plan must explicitly compare against frozen/base, standard rank-4 LoRA, mean-action prior, and MoIRA-style routing, and must keep the high kill risk visible.
