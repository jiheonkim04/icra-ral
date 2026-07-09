# SmolVLA LoRA Baseline Experiment Plan

Date: 2026-07-09 KST

## STATE 1: Bounded Standard LoRA Baseline

Runner:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_SMOLVLA_LORA_BASELINE="1"
$env:ALLOW_SMOLVLA_LORA_BASELINE_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\230_smolvla_lora_baseline.ps1 -MaxSteps 60 -MaxTrainDemos 3 -MaxEvalDemos 2 -RecordsPerDemo 3 -LoraRank 4
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_SMOLVLA_LORA_BASELINE -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_SMOLVLA_LORA_BASELINE_TRAINING -ErrorAction SilentlyContinue
```

The runner writes:

- `reports/smolvla_lora_baseline_state1_result.json`
- `reports/smolvla_lora_baseline_state1_result.md`

## Dataset

Default local HDF5:

`C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5`

Use a deterministic demo-level split with no train/eval demo overlap. Default split is three train demos, two eval demos, and three sampled timesteps per demo.

## Required Variants

- mean-action baseline from train records,
- frozen/base SmolVLA select-action baseline,
- standard PEFT LoRA action imitation baseline.

No new method variant is allowed.

## Metrics

- train loss curve,
- eval action L2,
- first-six-dimension action L2,
- translation L2,
- rotation L2,
- gripper error,
- gripper accuracy,
- train/eval gap,
- VRAM peak,
- runtime,
- trainable parameter count,
- whether LoRA beats mean-action,
- whether LoRA beats frozen/base SmolVLA,
- failure cases by action dimension.

## Continue Gate

Continue beyond baseline-only planning only if:

- real SmolVLA LoRA training runs without OOM,
- loss decreases meaningfully,
- eval action metric improves over mean-action,
- eval action metric improves over frozen/base SmolVLA,
- VRAM stays within the RTX 5080 budget,
- runtime stays reasonable.

## Stop Boundary

Stop immediately if the run becomes OpenVLA-OFT, full VLA fine-tuning, rollout, full benchmark, method invention, PatchGuard continuation, or large-asset download.

## Observed STATE 1 Result

The bounded runner completed with decision `KILL_MEAN_BASELINE_DOMINATED`.

- train/eval records: `9 / 6`
- loss start/end: `0.06359 / 0.008743`
- LoRA learned loss: yes
- mean-action eval action L2: `0.486561`
- frozen/base eval action L2: `1.6029`
- standard LoRA eval action L2: `0.940196`
- LoRA beat frozen/base: yes
- LoRA beat mean-action: no

Consequence: no new method should be started on top of the current local LoRA setup.
