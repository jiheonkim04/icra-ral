# PatchGuard-VLA Kill Summary

Date: 2026-07-09 KST

## Final Decision

`KILL_BASELINE_DOMINATED`

PatchGuard-VLA is killed as the current main RA-L route. This kills the PatchGuard method claim, not the local LoRA environment.

## Original PatchGuard Hypothesis

PatchGuard-VLA proposed a kinematic-consistent defense against physical patch attacks on VLA policies. The intended claim was that a robot-state-aware action-path defense could suppress patch-induced phantom embodiment or visual-proprioceptive hijacking while preserving clean task behavior.

LoRA, QLoRA, PEFT, and bitsandbytes were implementation tools only. They were not the contribution.

## Strongest Positive Evidence

- A real local SmolVLA policy path was used.
- The patch effect was measurable in STATE 1.
- Max attacked policy-action L1 vs clean was `0.181765`.
- Max attacked translation-action L2 vs clean was `0.213965`.
- Kinematic/proprioceptive signal was available.
- Cutout did not fully remove the fixed-patch effect in the initial STATE 1 diagnostic.
- PEFT installed and worked in STATE 1B: `peft 0.19.1`.
- bitsandbytes installed and worked in STATE 1B: `bitsandbytes 0.49.2`.
- bitsandbytes 4-bit and 8-bit CUDA kernel smokes passed.
- CUDA/PyTorch on RTX 5080 worked: PyTorch `2.10.0+cu128`, CUDA runtime `12.8`.
- Local SmolVLA LoRA injection worked on `C:\assets\checkpoints\smolvla`.
- LoRA targets were `state_proj`, `action_in_proj`, and `action_out_proj`.
- Trainable LoRA params: `9984`.
- Tiny batch-size-1 rank-4 training smoke ran for 10 steps per variant.
- Loss was computed.
- VRAM peak was `2224.845` MB.
- Runtime was `57.438` sec.

## Decisive Negative Evidence

- Standard LoRA attack-divergence metric: `0.144186`.
- Generic adversarial LoRA attack-divergence metric: `0.142803`.
- PatchGuard kinematic LoRA attack-divergence metric: `0.13356`.
- Cutout/random-erasing metric: `0.02973`.
- PatchGuard did not beat both required baselines.
- PatchGuard did not beat the generic adversarial LoRA baseline under the archive decision criterion.
- PatchGuard did not beat the cutout/random-erasing baseline.

## Exact Kill Criterion Triggered

The STATE 1B baseline gate failed:

`PatchGuard LoRA does not beat both generic adversarial LoRA and cutout/random-erasing.`

Exact decision: `KILL_BASELINE_DOMINATED`.

## Baselines That Killed It

- `fixed_patch_cutout_defense` / random-erasing style cutout: attack-divergence metric `0.02973`.
- `generic_adv_aug_lora`: attack-divergence metric `0.142803`.

PatchGuard produced `0.13356`, which was not enough because the predeclared gate required a robust win over both generic adversarial LoRA and cutout/random-erasing.

## Why PatchGuard Should Not Continue As RA-L-Stable

A RA-L-stable defense route needs a method-level advantage beyond generic augmentation and simple image erasing. STATE 1B showed the environment and adapter path were not the blocker. Once the environment blocker was removed, the PatchGuard-specific objective failed the declared baseline-dominance gate.

Continuing to STATE 2 would spend more GPU time on a method whose first real adapter smoke is already explained by cheaper baselines. That would repeat the project anti-pattern: plausible mechanism, working infrastructure, but no win over the strongest simple baseline.

## Why The LoRA Environment Remains Reusable

The negative method decision leaves a positive infrastructure result:

- PEFT works locally.
- bitsandbytes works locally, including CUDA 4-bit and 8-bit smokes.
- RTX 5080 memory is sufficient for small SmolVLA LoRA experiments.
- Local SmolVLA LoRA injection works on action-path modules.
- The repo now has a bounded dependency, adapter, and tiny-training smoke path.

This supports future standard SmolVLA LoRA baseline reproduction. It does not support a new custom method yet.

## Execution Boundary For This Archive Pass

No new experiment, training, GPU job, download, rollout, OpenVLA-OFT execution, or paper claim occurred in this archive pass. This summary archives previously committed STATE 1 and STATE 1B evidence.
