# Next Actions

Date: 2026-07-09 KST

Current decision:

`KILL_MEAN_BASELINE_DOMINATED`

## Immediate Next Action

Stop method work. Diagnose why standard SmolVLA LoRA loses to mean-action on the held-out local split.

## Why

The baseline did learn the training objective:

- loss start/end: `0.06359 / 0.008743`
- loss decreased meaningfully: yes
- trainable params: `9984`
- VRAM peak MB: `1190.228`
- runtime sec: `43.765`

But it failed the required mean-action gate:

- mean-action eval action L2: `0.486561`
- standard LoRA eval action L2: `0.940196`
- frozen/base SmolVLA eval action L2: `1.6029`

LoRA beat frozen/base SmolVLA, but it did not beat the trivial action prior.

## Allowed Next Work

- Baseline-only action normalization/provenance audit.
- Baseline-only split/sampling audit.
- Official SmolVLA training-recipe comparison if it can be done without large downloads or OpenVLA-OFT.
- A rerun of standard LoRA only if it changes the baseline protocol, not the method.

## Disallowed Next Work

Do not:

- invent a new method,
- continue PatchGuard,
- start Target-Grounded ActionMap, SafeLoRA, PRISM, ActionMap, or another route,
- run OpenVLA-OFT,
- run rollout from this evidence,
- download large assets,
- make paper claims,
- treat local proxy evidence as final paper evidence.
