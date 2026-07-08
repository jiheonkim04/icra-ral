# SafeLoRA-VLA LoRA Feasibility

Date: 2026-07-08

No LoRA/QLoRA training was run.

## Local Hardware And Asset Checks

Read-only local checks found:

- system RAM: about 24.9 GB,
- GPU inventory: NVIDIA GeForce RTX 5080 with 16303 MiB reported by
  `nvidia-smi`,
- local SmolVLA asset directories exist under `C:\assets`,
- `scripts/13_check_smolvla_adapter_smoke.ps1` reports
  `ready_for_smolvla_adapter_smoke=true`,
- that checker reports load-only, single-sample interface, feature-cache, and
  tiny head-only smoke statuses as previously passed,
- memory estimate: 12000 MB load plus 2048 MB headroom, fitting the 16 GB local
  budget.

These are engineering readiness checks, not benchmark results.

## QLoRA Check

`scripts/35_check_qlora_feasibility.ps1` was run in check-only mode. It reported:

- `peft`: unavailable,
- `bitsandbytes`: unavailable,
- QLoRA locally feasible now: false,
- safe to run QLoRA now: false,
- recommended next step: defer until blockers are resolved without unapproved
  installs or CUDA/PyTorch changes.

## OpenVLA / OpenVLA-OFT

OpenVLA supports LoRA in principle. OpenVLA-OFT reports strong LIBERO results
but its project page says training jobs used 8 A100 or H100 GPUs with 80 GB
memory for 50k-150k gradient steps over 1-2 days. That is outside this local
bounded gate.

OpenVLA-OFT is therefore not a local training target.

## SmolVLA

SmolVLA is the best low-compute candidate model because it is about 450M
parameters and the local asset readiness guard is green. However:

- official SmolVLA docs describe ordinary fine-tuning, not property-conditioned
  safety LoRA,
- the local repo's existing LoRA runners are proxy/tiny/cached-feature
  scaffolds, not official benchmark SafeLoRA training,
- no official LIBERO-Safety property-conditioned safe/unsafe pair source is
  green,
- QLoRA tooling is unavailable locally.

## Feasibility Verdict

LoRA as a software technique is plausible. SafeLoRA-VLA as an official
benchmark-backed property-conditioned LoRA run is not yet feasible.

Decision contribution: `NO_CLEAR_LORA_PATH`.
