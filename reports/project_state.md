# Project State

Date: 2026-07-09 KST

Branch:

`main`

Current main commit before this archive pass:

`67beb80 Run PatchGuard STATE 1B LoRA gate`

Current decision:

`KILL_BASELINE_DOMINATED`

## Current Archive Pass Boundary

- Experiments happened in this archive pass: no.
- Training happened in this archive pass: no.
- Loss computation happened in this archive pass: no.
- Rollout/replay happened in this archive pass: no.
- Downloads happened in this archive pass: no.
- GPU use happened in this archive pass: no.
- OpenVLA-OFT happened in this archive pass: no.
- New defense method implementation happened in this archive pass: no.
- Paper claims happened in this archive pass: no.

## Current Route Status

PatchGuard-VLA is archived as a main RA-L route.

This kills the PatchGuard method claim, not the local LoRA environment.

## Prior PatchGuard Evidence

STATE 1 found the original vulnerability and signal:

- patch effect measured: yes,
- max attacked policy-action L1 vs clean: `0.181765`,
- max attacked translation-action L2 vs clean: `0.213965`,
- kinematic/proprioceptive signal available: yes,
- cutout did not fully solve the fixed-patch effect in the initial diagnostic.

STATE 1B unblocked the environment and tested the tiny adapter path:

- PEFT installed and worked: `0.19.1`,
- bitsandbytes installed and worked: `0.49.2`,
- bitsandbytes 4-bit and 8-bit CUDA smokes passed,
- PyTorch/CUDA on RTX 5080 worked,
- SmolVLA LoRA injection worked,
- tiny batch-size-1 rank-4 training smoke ran,
- loss was computed,
- VRAM peak: `2224.845` MB,
- runtime: `57.438` sec.

## Decisive Negative Evidence

- standard LoRA metric: `0.144186`,
- generic adversarial LoRA metric: `0.142803`,
- PatchGuard metric: `0.13356`,
- cutout/random-erasing metric: `0.02973`,
- PatchGuard did not beat generic adversarial LoRA under the archive decision criterion,
- PatchGuard did not beat cutout/random-erasing,
- PatchGuard did not beat the required baseline set.

## Conclusion

PatchGuard should not proceed to STATE 2. The method-specific claim failed after the environment blocker was resolved.

The useful surviving project state is that real SmolVLA LoRA is now locally feasible. The next valid step is standard SmolVLA LoRA baseline reproduction on an official or standard task split, not a new local proxy method.
