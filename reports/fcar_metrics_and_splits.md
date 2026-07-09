# FCAR Metrics And Splits

Date: 2026-07-09 KST

## Data Source

Official local assets only:

- model: `C:\assets\checkpoints\smolvla_libero`
- dataset: `C:\assets\datasets\lerobot_libero`
- VLM dependency: `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`

No official eval split is present in the local metadata, so the first experiment uses a deterministic diagnostic split inside the official train split.

## Split Plan

Reuse the 200-frame failure-mining selection first:

- train episode for LoRA recreation: episode `0`;
- diagnostic held-out episodes: `1, 4, 2, 3, 7, 9, 8, 13, 14, 15`;
- held-out task groups: `5`;
- held-out frames: `200`.

For FCAR gate training:

- split the 200 frames deterministically, e.g. `60%` gate-train, `20%` gate-val, `20%` gate-test;
- group by episode where possible so adjacent frames do not leak across train/test;
- if expanding to 1000 frames, predeclare task/episode list before seeing results.

## Primary Metric

Postprocessed 7D action L2 against official raw action label on held-out gate-test frames.

## Secondary Metrics

- normalized eval loss if meaningful for selected policy/mix;
- translation L2;
- rotation L2;
- gripper absolute error;
- gripper sign accuracy;
- per-task breakdown;
- per-phase breakdown;
- help/hurt count vs frozen/base;
- help/hurt count vs rank-4 LoRA;
- fraction of frames routed to base vs LoRA;
- gate calibration: predicted alpha vs oracle help probability;
- action range validity;
- train/eval gap;
- runtime.

## Success Thresholds

Hard success gate:

- beat frozen/base by at least `5%` relative action L2 improvement or `0.005` absolute action L2 improvement;
- beat standard rank-4 LoRA;
- beat MoIRA-style task/instruction router;
- beat adapter soup/static merge;
- beat mean-action prior.

Soft target:

- recover at least `30%` of frame-oracle gain over frozen/base.

Current numeric targets:

- frozen/base L2: `0.106514960`
- frame oracle L2: `0.084582188`
- oracle gain: `0.021932772`
- `30%` recovery target: `0.099935128` or lower
- `50%` recovery target: `0.095548574` or lower

## Reporting Requirements

Every FCAR run must report:

- all baseline metrics in one table;
- route fraction to base/LoRA;
- calibration and confusion matrix for oracle help labels;
- top FCAR helps and hurts;
- whether FCAR beats frozen/base and by how much;
- whether FCAR beats task/instruction routing and static merge.
