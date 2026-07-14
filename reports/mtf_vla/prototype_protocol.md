# MTF-VLA Prototype Protocol

Date: 2026-07-14 KST

Method: `MTF-VLA`, Milestone-Transition Focused VLA Adaptation.

## Variant Set

The first serious comparison contains exactly five policies:

1. `base_smolvla`
   - Frozen official SmolVLA policy.

2. `frameskip_proxy_lora`
   - Closest external-prior proxy.
   - Uses a transparent FrameSkip-style frame score at the selected retained ratio.
   - Does not use MTF's base-retention objective.
   - Must list any omitted official FrameSkip component.

3. `mtf_full`
   - Uses structured milestone-transition selection.
   - Uses base-retention frames and the selected retention coefficient.
   - Uses the frozen selected checkpoint.

4. `mtf_no_retention_ablation`
   - Same milestone-transition selection and training budget as MTF full.
   - Sets retention coefficient to `0`.

5. `uniform_retained_ratio_lora`
   - One strongest simple reviewer-killer baseline.
   - Same adapter family, retained-frame ratio, training budget, and selection rule, but uniformly sampled frames.

## Stage 0 Artifacts

The Stage 0 audit must write:

- `reports/mtf_vla/development_audit.json`
- `reports/mtf_vla/development_audit.md`
- frame-score manifest or summary;
- partition manifest;
- base-retention target manifest or hard-stop reason.

Stage 0 must not launch confirmatory rollout.

## Training And Validation Artifacts

Every trained policy must have:

- config JSON;
- training seed;
- source split manifest;
- checkpoint path;
- checksum;
- base checkpoint identity;
- dataset identity;
- validation metrics;
- action-delta metrics;
- clean-retention metrics.

## Closed-Loop Manifest

Stage A and Stage B must use a matched paired manifest:

- identical task keys across policies;
- identical reset identities across policies;
- no task/reset identity overlap with validation identities;
- no post-result task cherry-picking.

The manifest is frozen before each stage begins.

## Metrics

Primary:

- closed-loop success;
- task-balanced success;
- paired full-minus-baseline success deltas.

Secondary:

- paired wins/losses/ties;
- paired bootstrap confidence interval;
- relative failure-rate reduction;
- per-task success;
- clean-retention success;
- action delta from Base;
- translation, rotation, and gripper delta;
- action validity;
- milestone activation distribution;
- latency;
- VRAM;
- training time.

## Resume Policy

For long-running WSL training or rollout:

- run detached;
- save PID;
- save heartbeat;
- save logs;
- save partial result;
- save exact resume command;
- resume only missing `(variant, task, identity)` keys after interruption.

## Scientific Decisions

MTF cannot become a paper candidate unless full MTF beats Base, FrameSkip proxy, no-retention ablation, and uniform retained-ratio LoRA under the matched protocol.

If the FrameSkip proxy wins, MTF is not a useful extension of the closest prior.

If uniform retained-ratio LoRA wins, the result is explained by simple adapter training.

If no-retention wins, the retention component is not useful.
