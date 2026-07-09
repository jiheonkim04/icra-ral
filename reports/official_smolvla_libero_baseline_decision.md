# Official SmolVLA-LIBERO Baseline Decision

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-baseline-scaleup`

Final decision: `READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA`

## Boundary

- experiments happened: yes
- training happened: yes, standard rank-4 LoRA only
- loss computed: yes
- GPU used: yes, CUDA on `NVIDIA GeForce RTX 5080`
- CPU fallback: no
- downloads happened: no
- simulator rollout/full benchmark: no
- OpenVLA-OFT: no
- custom `LIBERO_7D` adapter route: no
- new method variants: no

## STATE 1 Dataset/Sample Audit

- official checkpoint: `C:\assets\checkpoints\smolvla_libero`
- official dataset: `C:\assets\datasets\lerobot_libero`
- available episodes: `1693`
- available frames/samples: `273465`
- available tasks: `40`
- official split metadata: `{"train": "0:1693"}`
- official eval split present: no
- offline mini-holdout used: episode `1` from the official train split, for diagnostic metrics only
- action dimension: `7`
- action stats range: min `[-0.9375, -0.9375, -0.9375, -0.258214, -0.375, -0.3675, -1.0]`, max `[0.9375, 0.9375, 0.9375, 0.355714, 0.375, 0.375, 1.0]`
- state dimension: `8`
- image streams: `observation.images.image` and `observation.images.image2`, each `[256, 256, 3]`
- processor outputs: action `[1, 50, 7]`, state `[1, 8]`, two CUDA image tensors `[1, 3, 256, 256]`, language tokens `[1, 48]`
- labels/action stats loaded: yes
- data loading deterministic: yes, repeated sample max absolute diff `0.0`
- schema mismatch: no

## STATE 2 Baseline Scaleup

Frozen/base official SmolVLA, no training:

- eval loss mean: `0.008015549`
- action L2 mean: `0.081655363`
- translation L2 mean: `0.079455563`
- rotation L2 mean: `0.015266377`
- gripper abs mean: `0.003841186`
- gripper sign accuracy: `1.0`
- action range: `[-1.008845806, 0.180931509]`

Standard rank-4 LoRA:

- batch size: `1`
- requested/completed steps: `100` / `100`
- trainable params: `185,664`
- total params: `450,231,840`
- loss before/after: `0.005532921` / `0.003888785`
- loss decrease fraction: `0.297155155`
- last grad norm: `0.015808909`
- nonzero grad tensors: `74`
- training runtime: `18.078 sec`
- steps/sec: `5.531585`
- total runtime: `40.813 sec`
- peak CUDA allocated: `1104.506 MB`

LoRA offline mini-holdout metrics:

- eval loss mean: `0.020719278`
- action L2 mean: `0.072837438`
- translation L2 mean: `0.071438076`
- rotation L2 mean: `0.011711556`
- gripper abs mean: `0.004226804`
- gripper sign accuracy: `1.0`
- action range: `[-1.007751584, 0.144551635]`
- finite outputs: yes

Important mixed signal: action L2 improved versus frozen/base, but chunk eval loss worsened. This is still sufficient for method-design readiness under the predeclared rule because the objective here is official path stability, not a benchmark claim.

## Decision Rationale

Use `READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA` because:

- official model/dataset/processor path remained stable;
- rank-4 LoRA completed on CUDA without OOM;
- train loss decreased by more than the predeclared 10% threshold;
- labels/action metrics were available;
- no CPU fallback occurred;
- VRAM/runtime were far inside budget;
- mini-holdout action L2 was not worse than frozen/base.

This decision allows future method design only on the official SmolVLA/LeRobot path and only with this frozen/base plus rank-4 LoRA baseline retained as the comparison anchor.

## Exact Next Step

Start method design only on the official SmolVLA-LIBERO path. The first method-design plan must predeclare comparisons against:

- frozen/base official SmolVLA;
- standard rank-4 LoRA official baseline from this run;
- any stronger longer official baseline if it is run before the method;
- official simulator rollout/eval separately, only after WSL/Linux/MuJoCo readiness.
