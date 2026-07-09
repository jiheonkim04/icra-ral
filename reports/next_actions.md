# Next Actions

Date: 2026-07-10 KST

Current decision: `METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD`

## Immediate Next Action

Build a more stable official split/metric protocol before any new method design.

Result boundary:

- official assets only: `C:\assets\checkpoints\smolvla_libero` and `C:\assets\datasets\lerobot_libero`;
- compact official per-frame base/LoRA predictions were regenerated and saved at `reports/fcar_prediction_artifact.json`;
- FCAR tiny gate was implemented and trained only as a small CPU gate;
- no SmolVLA backbone training happened;
- fixed rank-4 LoRA was regenerated only as the required baseline artifact source;
- FCAR gate-test action L2 was `0.100144625`;
- frozen/base gate-test action L2 was `0.123998278`;
- rank-4 LoRA gate-test action L2 was `0.076191123`;
- val-selected static mixture `w=0.5` gate-test action L2 was `0.091179973`;
- final decision is `FCAR_KILLED_BY_STATIC_BASELINE`;
- post-FCAR robust baseline sweep used `5` deterministic episode-disjoint folds from the official prediction artifact;
- no new training, GPU work, downloads, OpenVLA-OFT, simulator rollout, full benchmark, FCAR tuning, or new method implementation happened in the sweep;
- robust sweep mean/std action L2: frozen/base `0.106514933` / `0.030256808`, rank-4 LoRA `0.118024225` / `0.023707422`, mean-action `1.144859705` / `0.018515874`, frame oracle `0.084582167` / `0.027591676`, task oracle `0.106079936` / `0.029986441`, MoIRA-style router `0.106514933` / `0.030256808`, val-selected static mix `0.105142674` / `0.026514373`;
- realistic win counts were frozen/base `2` and val-selected static mix `3`;
- rank-4 LoRA beat frozen/base in `2` / `5` folds but won no realistic fold, so LoRA behavior is split-dependent;
- frame oracle won all `5` folds and still has mean headroom `0.021932766`, while task oracle remains weak with mean headroom `0.000434997`;
- final post-FCAR decision is `METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD`;
- no archived custom `LIBERO_7D` adapter route;
- no OpenVLA-OFT;
- no full benchmark or simulator rollout until WSL/Linux/MuJoCo readiness is handled separately.

Preserve the FCAR and robust-sweep artifacts for audit. Do not continue FCAR scaleup, tune FCAR, or design a new method until the official split/metric protocol is made stable. Official simulator eval remains a separate WSL/Linux/MuJoCo readiness milestone. Do not substitute the archived custom replay bridge as official eval.
