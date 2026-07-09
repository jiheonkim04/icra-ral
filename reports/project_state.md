# Project State

Date: 2026-07-09 KST

Branch:

`codex/archive-patchguard-and-smolvla-lora-baseline`

Current branch base:

`5ff597c Archive PatchGuard route and record LoRA status`

Current decision:

`KILL_MEAN_BASELINE_DOMINATED`

## Current Bounded Run Boundary

- PatchGuard archived: yes.
- LoRA environment status recorded: yes.
- Experiments happened: yes, one bounded standard SmolVLA LoRA baseline.
- Training happened: yes, rank-4 PEFT LoRA only.
- Loss computation happened: yes.
- GPU happened: yes, RTX 5080 CUDA.
- Downloads happened: no.
- Rollout/replay happened: no.
- OpenVLA-OFT happened: no.
- Full VLA fine-tuning happened: no.
- New method implementation happened: no.
- Paper claims happened: no.

## PatchGuard Status

PatchGuard-VLA remains archived as `KILL_BASELINE_DOMINATED`. This kills the PatchGuard method claim, not the LoRA environment.

## SmolVLA LoRA Baseline Status

STATE 1 standard LoRA baseline ran on:

- model: `C:\assets\checkpoints\smolvla`
- dataset: `C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5`
- split: `deterministic_demo_holdout`
- train demos: `demo_0`, `demo_1`, `demo_2`
- eval demos: `demo_3`, `demo_4`
- train/eval records: `9 / 6`

Result:

- LoRA rank: `4`
- trainable params: `9984`
- optimizer steps: `60`
- loss start/end: `0.06359 / 0.008743`
- loss decreased meaningfully: yes
- VRAM peak MB: `1190.228`
- runtime sec: `43.765`
- mean-action eval action L2: `0.486561`
- frozen/base SmolVLA eval action L2: `1.6029`
- standard LoRA eval action L2: `0.940196`
- LoRA beats frozen/base: yes
- LoRA beats mean-action: no

## Conclusion

Standard LoRA can train and improve over frozen/base SmolVLA in this bounded local setup, but it does not beat the mean-action baseline. This blocks any method on top of SmolVLA LoRA until the baseline/action-interface issue is resolved.
