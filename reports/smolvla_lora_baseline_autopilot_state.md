# SmolVLA LoRA Baseline Autopilot State

Date: 2026-07-09 KST

Branch: `codex/archive-patchguard-and-smolvla-lora-baseline`

## Current Route

SmolVLA LoRA Baseline Reproduction.

This is the active route only as a baseline foundation. It is not a new paper method.

## Starting Context

PatchGuard-VLA is archived as `KILL_BASELINE_DOMINATED`. The positive reusable artifact is that PEFT, bitsandbytes, CUDA on RTX 5080, and local SmolVLA LoRA injection worked.

## STATE 1 Plan

Run a bounded standard LoRA baseline on local LIBERO HDF5 data:

- one task,
- deterministic demo-level split,
- at least three train demos if available,
- mean-action baseline,
- frozen/base SmolVLA baseline,
- standard LoRA baseline,
- rank 4,
- batch size 1,
- at most 60 default optimization steps.

## Safety State

- OpenVLA-OFT happened: no.
- Full benchmark happened: no.
- Rollout happened: no.
- Large downloads happened: no.
- New method invented: no.
- PatchGuard continued: no.
- GPU training happened: yes, only for the bounded standard LoRA baseline.
- Loss computed: yes.

## Result Location

The runner will write:

- `reports/smolvla_lora_baseline_state1_result.json`
- `reports/smolvla_lora_baseline_state1_result.md`

## STATE 1 Result

Decision: `KILL_MEAN_BASELINE_DOMINATED`

Dataset and split:

- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo`
- HDF5: `C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5`
- split: `deterministic_demo_holdout`
- train demos: `demo_0`, `demo_1`, `demo_2`
- eval demos: `demo_3`, `demo_4`
- train/eval records: `9 / 6`

Key evidence:

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

Interpretation: standard LoRA learned the small training objective but did not clear the held-out mean-action gate. Do not start a method on top of this local baseline yet.

## Diagnosis Result

Decision: `ACTION_INTERFACE_BUG`

The follow-up diagnosis found:

- raw HDF5 timesteps: `13298`
- larger deterministic demo-holdout split possible: `300 / 100`
- HDF5 action dim: `7`
- SmolVLA model action shape: `[6]`
- SO100-style checkpoint action normalizer mismatched local LIBERO action scale
- label reconstruction and chunk alignment passed
- one-sample overfit failed
- one-demo overfit failed
- best LoRA action L2: `0.912258`
- best small MLP/ridge action L2: `0.401848`

Conclusion: fix the action interface before any method work.
