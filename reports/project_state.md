# Project State

Date: 2026-07-10 KST

Branch: `codex/implement-fcar-tiny-gate`

Current decision: `FCAR_KILLED_BY_STATIC_BASELINE`

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
- FCAR first-experiment implementation completed: the script saved compact official per-frame base/LoRA predictions to `reports/fcar_prediction_artifact.json`, regenerated the fixed rank-4 LoRA baseline only because the artifact was missing, and trained only the FCAR tiny CPU gate.
- FCAR gate-test split was episode-disjoint: train `120` frames, val `40` frames, test `40` frames. Test tasks were task `5` and task `8`.
- FCAR test action L2 was `0.100144625`, improving over frozen/base test action L2 `0.123998278` by `0.023853653` / `19.2370841%` and recovering `41.216345%` of frame-oracle headroom on the gate-test split.
- FCAR did not beat rank-4 LoRA test action L2 `0.076191123`.
- FCAR did not beat the predeclared adapter-soup/static-merge baseline: val-selected static mixture `w=0.5` reached test action L2 `0.091179973`.
- MoIRA-style task/instruction router routed all gate-test tasks to frozen/base from train evidence and reached test action L2 `0.123998278`, so it did not kill FCAR; the static baseline did.
- Official simulator eval was not run; `lerobot-eval --env.type=libero` still requires WSL/Linux/MuJoCo readiness.
- OpenVLA-OFT, full benchmark, long training, and custom `LIBERO_7D` adapter route were not used.

## Conclusion

`FCAR_KILLED_BY_STATIC_BASELINE`

Do not scale FCAR from this result. The first tiny-gate method is killed by the predeclared static mixture baseline and also loses to rank-4 LoRA on the gate-test split. Keep the saved official per-frame prediction artifact and result reports as reusable evidence, but do not claim an FCAR method contribution from this run.
