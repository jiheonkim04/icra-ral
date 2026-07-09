# PatchGuard-VLA Kill Criteria

Date: 2026-07-09 KST

## Hard Stops

Kill or block PatchGuard-VLA if any of the following is true:

- no real VLA model can be used locally,
- local evidence is toy-only,
- clean-vs-patched action divergence is not measurable,
- random patch and fixed visible patch have no nontrivial effect,
- no EEF/joint/proprioceptive or arm-pose signal is available,
- random erasing/cutout fully removes the measured patch effect,
- only generic visual augmentation explains the benefit,
- there is no feasible path to a real adapter or LoRA smoke,
- the run would require full benchmark, full fine-tuning, large downloads, OpenVLA-OFT, or long GPU jobs.

## Diagnostic Thresholds

STATE 1 treats patch effect as nontrivial if either:

- max attacked policy-action L1 vs clean is at least `0.01`, or
- max attacked translation-action L2 vs clean is at least `0.01`.

STATE 1 treats cutout as baseline-dominating when:

- fixed visible patch is nontrivial, and
- cutout mean policy-action L1 vs clean is less than or equal to the larger of `0.005` and 25 percent of the fixed-patch mean policy-action L1.

These thresholds are feasibility thresholds only, not paper metrics.

STATE 1B treats the prior adapter-tooling `TOO_HEAVY_LOCAL` as an installable environment blocker. If `peft` and `bitsandbytes` install and their CUDA smokes pass, the route is no longer blocked by adapter availability.

## Decision Mapping

| Condition | Exact decision |
| --- | --- |
| local source/model/runtime unavailable | `SOURCE_BLOCKED` |
| EEF/joint/proprio or non-leaking kinematic signal unavailable | `KILL_NO_KINEMATIC_SIGNAL` |
| random/fixed patch does not change real VLA actions | `KILL_ATTACK_NOT_REPRODUCIBLE` |
| cutout/random-erasing or generic visual baseline removes the effect | `KILL_BASELINE_DOMINATED` |
| attack/signal exists but real adapter path is not locally feasible under constraints | `TOO_HEAVY_LOCAL` |
| all continue criteria pass | `READY_FOR_PATCHGUARD_LORA_SMOKE` |

## STATE 1B Decision Mapping

| Condition | Exact decision |
| --- | --- |
| PEFT cannot be installed/imported after the approved minimal install | `ENV_BLOCKED_INSTALL_FAILED` |
| bitsandbytes/QLoRA is blocked but ordinary PEFT LoRA works | `QLORA_BLOCKED_BUT_LORA_POSSIBLE` |
| no trainable LoRA adapter path can be attached to local SmolVLA | `KILL_NO_ADAPTER_PATH` |
| bounded rank-4, batch-size-1 smoke exceeds the local VRAM/runtime budget | `TOO_HEAVY_LOCAL` |
| PatchGuard LoRA does not beat both generic adversarial LoRA and cutout/random-erasing | `KILL_BASELINE_DOMINATED` |
| environment, adapter injection, tiny training, and baseline gates pass | `READY_FOR_PATCHGUARD_LORA_STATE2` |

## READY Means

`READY_FOR_PATCHGUARD_LORA_SMOKE` authorizes only a future separately approved adapter smoke. `READY_FOR_PATCHGUARD_LORA_STATE2` authorizes only a future explicitly approved STATE 2 pilot. Neither authorizes full training, full benchmark, OpenVLA-OFT, or paper claims.

