# Official SmolVLA-LIBERO Baseline Scaleup Result

- decision: `READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA`
- status: `completed`
- downloads performed: `False`
- training performed: `True`
- rollouts performed: `False`
- OpenVLA-OFT executed: `False`
- CPU fallback occurred: `False`
- schema mismatch: `False`

## Failed Attempt Log

- first attempt: failed before optimizer step with exit code `31` because this LeRobot `policy.forward()` returned a tuple-shaped loss field and the first runner version only handled scalar/dict loss.
- fix: update the runner to extract scalar loss from tuple/list outputs.
- second attempt: completed the same bounded rank-4 LoRA run successfully.

## Dataset Audit

- total episodes: `1693`
- total frames: `273465`
- total tasks: `40`
- splits: `{'train': '0:1693'}`
- action dim: `7`
- state dim: `8`
- image streams: `{'observation.images.image': [256, 256, 3], 'observation.images.image2': [256, 256, 3]}`
- data deterministic: `True`
- labels/action stats loaded: `True`

## Training

- LoRA rank: `4`
- batch size: `1`
- requested/completed steps: `100` / `100`
- trainable params: `185664`
- loss before/after: `0.005532921` / `0.003888785`
- loss decrease fraction: `0.297155155`
- last grad norm: `0.015808909`
- steps/sec: `5.531585`

## Evaluation

- frozen/base action L2 mean: `0.081655363`
- LoRA action L2 mean: `0.072837438`
- frozen/base eval loss mean: `0.008015549`
- LoRA eval loss mean: `0.020719278`
- LoRA action finite: `True`

Interpretation: the mini-holdout action L2 improved after LoRA, while the chunk eval loss became worse than frozen/base. Keep both signals; do not treat this as a benchmark or paper-grade success.

## Runtime

- total elapsed sec: `40.813`
- training elapsed sec: `18.078`
- CUDA available: `True`
- CUDA device: `NVIDIA GeForce RTX 5080`
- CUDA max allocated MB: `1104.506`
- RSS final MB: `2862.723`

Exact next step: Start method design only on the official SmolVLA-LIBERO path, using this rank-4 baseline as the minimum frozen/base and LoRA comparison anchor.
