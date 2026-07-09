# Next Actions

Date: 2026-07-09 KST

Current decision:

`KILL_BASELINE_DOMINATED`

## Immediate Next Action

Archive PatchGuard-VLA as a main RA-L route and preserve the LoRA environment as reusable infrastructure.

## Why

PatchGuard cleared the environment and adapter gates, so the final blocker is not local tooling:

- PEFT worked.
- bitsandbytes worked.
- CUDA on RTX 5080 worked.
- SmolVLA LoRA injection worked.
- Tiny training ran and loss was computed.

The method then failed the baseline gate:

- generic adversarial LoRA metric: `0.142803`,
- PatchGuard metric: `0.13356`,
- cutout/random-erasing metric: `0.02973`,
- PatchGuard did not beat generic adversarial LoRA under the archive decision criterion,
- PatchGuard did not beat cutout/random-erasing,
- PatchGuard did not beat both required baselines.

## Recommended Next Step

Run a real SmolVLA LoRA baseline reproduction on an official or standard task split.

The goal is to understand standard LoRA behavior before inventing any new method:

- clean retention,
- perturbation behavior,
- generic augmentation behavior,
- memory/runtime scaling,
- stable metrics for later comparisons.

## Allowed Next Work

- Standard SmolVLA LoRA baseline reproduction.
- Environment documentation and reproducibility checks.
- Baseline-first planning that predeclares standard LoRA, generic augmentation, cutout/random-erasing, and no-adaptation controls.

## Disallowed Next Work

Do not:

- proceed to PatchGuard STATE 2,
- run more PatchGuard training,
- invent a new defense method,
- start another local proxy idea,
- run rollout from PatchGuard evidence,
- run OpenVLA-OFT,
- download large assets,
- make paper claims from STATE 1B.
