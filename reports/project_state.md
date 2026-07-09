# Project State

Date: 2026-07-10 KST

Branch: `codex/official-smolvla-robust-baseline-sweep`

Current decision: `METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD`

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
- Post-FCAR robust baseline sweep over `5` deterministic episode-disjoint folds used the saved official prediction artifact. No new training, downloads, GPU work, rollouts, OpenVLA-OFT, or FCAR tuning happened in the sweep.
- Robust sweep mean/std action L2: frozen/base `0.106514933` / `0.030256808`, rank-4 LoRA `0.118024225` / `0.023707422`, mean-action `1.144859705` / `0.018515874`, frame oracle `0.084582167` / `0.027591676`, task oracle `0.106079936` / `0.029986441`, MoIRA-style task router `0.106514933` / `0.030256808`, val-selected static mix `0.105142674` / `0.026514373`.
- Realistic baseline win counts across the five folds were frozen/base `2` and val-selected static mix `3`; rank-4 LoRA won `0` folds but beat frozen/base in `2` folds, so LoRA behavior is split-dependent.
- Frame oracle won all `5` folds and still had mean headroom `0.021932766` over frozen/base, while task oracle headroom remained tiny at `0.000434997`.
- Static mix remained a reviewer-killer for FCAR, but the base/static/LoRA rank order is too split-dependent to design a stable new method from this evidence.
- Official simulator eval was not run; `lerobot-eval --env.type=libero` still requires WSL/Linux/MuJoCo readiness.
- OpenVLA-OFT, full benchmark, long training, and custom `LIBERO_7D` adapter route were not used.

## Conclusion

`METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD`

Do not scale or tune FCAR. Do not design a new method yet. First build a more stable official split/metric protocol because the current baseline ranking changes across episode-disjoint folds even though frame-oracle headroom remains.
