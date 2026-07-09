# PatchGuard-VLA STATE 1B Result

Bounded environment and tiny LoRA feasibility gate. This is not a full benchmark, rollout, OpenVLA-OFT run, or paper claim.

- final decision: `KILL_BASELINE_DOMINATED`
- dependency install happened: `True`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5080`
- PEFT: `0.19.1`
- bitsandbytes: `0.49.2`
- bitsandbytes 4-bit smoke: `True`
- LoRA injection happened: `True`
- tiny training smoke happened: `True`
- loss computed: `True`
- VRAM peak MB: `2224.845`
- runtime sec: `57.438`
- clean metric: `0.422465`
- patched metric: `0.134391`
- cutout/random erasing metric: `0.02973`
- generic adversarial LoRA metric: `0.142803`
- PatchGuard metric: `0.13356`
- PatchGuard beats baseline: `False`

## Variant Metrics

| variant | attack divergence | clean metric | fixed patch | random patch | cutout | loss start | loss end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard_clean_lora | 0.144186 | 0.352031 | 0.16103 | 0.127341 | 0.042131 | 0.072097 | 0.038002 |
| generic_adv_aug_lora | 0.142803 | 0.352705 | 0.162334 | 0.123272 | 0.043473 | 0.064754 | 0.036388 |
| patchguard_kinematic_lora | 0.13356 | 0.422465 | 0.155522 | 0.111598 | 0.031044 | 0.067332 | 0.043161 |

Exact next step: Do not proceed to STATE 2; PatchGuard did not beat generic adversarial augmentation and cutout baselines in the tiny smoke.
