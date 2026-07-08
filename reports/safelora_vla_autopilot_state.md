# SafeLoRA-VLA Autopilot State

Date: 2026-07-08

## Scope

- Mode: bounded feasibility gate only
- Branch: `codex/safelora-vla-state0-state2`
- Starting main commit: `7b1b0f7 Add SafeManip feasibility scout`
- STATE 2: disabled
- Experiments happened: no
- Training happened: no
- Large download happened: no
- GPU job happened: no
- OpenVLA-OFT happened: no

## Local Readiness Observed

- Local RAM: about 24.9 GB
- GPU inventory: NVIDIA GeForce RTX 5080, 16303 MiB reported by `nvidia-smi`
- SmolVLA local asset readiness check: green for existing local assets and
  previously validated load/interface/feature-cache/tiny-head status
- SmolVLA memory estimate from local checker: 12000 MB load plus 2048 MB
  headroom, fits the 16 GB RTX 5080 budget
- QLoRA guard check: not locally feasible now because `peft` and
  `bitsandbytes` are absent without installing packages

## Source Gate Summary

- SafeManip: best temporal-property benchmark, but too heavy locally.
- LIBERO-Safety: best public official safety dataset/code candidate, but no
  clear property-conditioned LoRA path and no explicit tiny official subset.
- ForesightSafety-VLA: strong metrics on paper, but no official code/data path
  found in this gate.
- SafeVLA-Bench: useful related benchmark, but no official code package found
  in this gate.
- Local standard LIBERO: useful engineering fallback only, not paper evidence
  for this route.

## Decision

`NO_CLEAR_LORA_PATH`

The next state is not STATE 2. The next state is blocker resolution:

- identify official property-level unsafe labels or official rollout logs,
- confirm a bounded dataset subset/download path,
- confirm a real SmolVLA/OpenVLA LoRA implementation path,
- rerun the source/LoRA gate before training.
