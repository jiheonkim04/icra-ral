# Official SmolVLA Stable Protocol Plan

Date: 2026-07-10 KST

Purpose: freeze the split and metric protocol before any new official SmolVLA-LIBERO baseline or method work.

Hard boundary:

- no new method design
- no FCAR revival or tuning
- no simulator rollout or full benchmark
- no OpenVLA-OFT
- no old custom LIBERO_7D route
- no new large downloads

Stable protocol target:

- train frames: `1200`
- val frames: `400`
- test frames: `1200`
- tasks per split: `{'train': 40, 'val': 40, 'test': 40}`

Decision rule: do not design methods until a larger official prediction artifact is generated and evaluated under this manifest and metric protocol.
