# Project State

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-routing-design-gate`

Current decision: `GO_DESIGN_FRAME_CONDITIONAL_ROUTING`

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
- Routing oracle design gate found meaningful frame-level headroom but tiny task-level headroom: frame oracle action L2 `0.084582188` improves over frozen/base by `0.021932772` / `20.5912597%`, while task oracle action L2 `0.106079976` improves by only `0.000434984` / `0.4083783%`.
- Pure task/instruction routing is killed by MoIRA-style routing and by tiny task-oracle headroom. The surviving design direction is frame-conditional adapter retention with frozen/base as an explicit expert.
- Official simulator eval was not run; `lerobot-eval --env.type=libero` still requires WSL/Linux/MuJoCo readiness.
- OpenVLA-OFT, full benchmark, long training, and custom `LIBERO_7D` adapter route were not used.

## Conclusion

`GO_DESIGN_FRAME_CONDITIONAL_ROUTING`

Next valid step: create the first Frame-Conditional Adapter Retention experiment plan on the official SmolVLA-LIBERO path. Do not run it until frozen/base, rank-4 LoRA, mean-action prior, frame oracle, task oracle, MoIRA-style instruction router, adapter soup/merge, and kill criteria are predeclared.
