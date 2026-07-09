# SmolVLA 7D Action Range Kill Summary

Date: 2026-07-09 KST

## Final Strategic Decision

`OFFICIAL_VLA_RECIPE_REPRODUCTION_REQUIRED`

The custom SmolVLA 7D adapter route is archived as stopped. The action-range gate improved validity but failed as a sufficient infrastructure fix and must not be treated as method progress.

## Original Custom SmolVLA 7D Adapter Hypothesis

The hypothesis was that a lightweight local SmolVLA adapter could preserve the native checkpoint, add a LIBERO-compatible 7D action head, learn the 7th gripper dimension, and transfer offline action improvements into exact-init LIBERO control. If this worked, method work could then be built on top of the fixed local VLA baseline.

## Positive Evidence

- PEFT, bitsandbytes, CUDA, and RTX 5080 SmolVLA LoRA smokes worked.
- Local SmolVLA LoRA injection worked on `C:\assets\checkpoints\smolvla`.
- The LIBERO 7D interface was fixed enough for bounded offline adapter training:
  - native SmolVLA 6D/SO100 action path preserved separately,
  - LIBERO labels kept as 7D,
  - train-split-only LIBERO 7D normalization,
  - learned 7th gripper output,
  - one-sample and one-demo overfit passed.
- Expert replay stable set exists:
  - 6 expert-success eligible cases,
  - expert replay after feature fix: `6/6`.
- Live/HDF5 feature schema mismatch was fixed:
  - feature L2 `2.248343 -> 0.033195`,
  - teacher-forced action L2 `2.285072 -> 0.843733`.

## Decisive Negative Evidence

- The learned adapter failed exact-init replay after the feature fix:
  - adapter success `0/6`,
  - adapter progress `-0.041091`,
  - mean-action progress `0.038336`,
  - ridge progress `-0.173863`.
- Offline improvement did not transfer to control.
- Action range fixing improved validity but worsened action quality:
  - clip rate `0.15625 -> 0.0`,
  - controller-valid proxy `0.84375 -> 1.0`,
  - offline action L2 worsened `0.795274 -> 0.976681`.
- Range fix made replay worse:
  - replay progress `-0.041091 -> -0.902509`.
- Clip-only postprocessing matched or beat the trained range-fixed adapter:
  - clip-only replay progress `-0.041091`,
  - range-fixed replay progress `-0.902509`.

## Exact Stop Criterion Triggered

`CLIP_ONLY_BASELINE_DOMINATES`

The predeclared range-fix gate required action validity improvement without severe offline degradation and with replay/progress improvement over simple baselines. The range-fixed adapter met the validity part but failed the transfer and baseline-dominance parts.

## Stopped Route

Stop iterating the custom SmolVLA 7D adapter pipeline as the main RA-L path. Do not tune another custom adapter variant, gripper convention, range penalty, or postprocessor as if this route is nearly solved.

## Why No New Method Should Be Built Here

This route now has multiple repaired pieces and still fails at control transfer. The remaining failure is not a single known bug:

- interface mismatch was fixed,
- expert replay was stabilized,
- live feature schema was fixed,
- action range/controller validity was fixed,
- a simple clip-only baseline matched or beat the trained range-fix variant.

Building a new method on this stack would risk optimizing around a locally idiosyncratic adapter pipeline rather than demonstrating VLA method value.

## Required To Resume

Resume VLA method work only after an official SmolVLA/LeRobot/OpenVLA-style recipe is reproduced locally:

- official model/training recipe,
- official preprocessing and normalization,
- official action/gripper conventions,
- official eval or replay stack,
- baseline success before method modifications.

No paper claim or new RA-L method should start before that baseline is established.
