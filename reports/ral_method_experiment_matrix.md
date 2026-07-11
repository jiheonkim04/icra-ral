# RA-L Method Experiment Matrix

Date: 2026-07-11 KST

Status: inactive because no method passed the gate.

## Activation Rule

This matrix only activates if a later visual gate produces one mechanism that satisfies at least one condition:

- same visible mechanism in at least two tasks; or
- same visible mechanism in at least three independent reset seeds within one hard task.

The mechanism must also survive the recent-work comparison in `reports/closed_loop_failure_vs_recent_work.md`.

## Required Baselines If Reopened

- official `frozen_base`
- official `rank4_lora_seed_11`
- official `rank4_lora_seed_22`
- official `rank4_lora_seed_33`
- simple frequent-replan baseline
- simple fixed smaller action-chunk baseline, if executable in the official path
- closest recent-work proxy for the selected mechanism, such as VLA-Corrector/AAC/SEAM/VeriSpace/SPR depending on the claim

## Required Backbones

- primary: official SmolVLA / LeRobot / LIBERO path used in this repo
- second backbone candidate: OpenVLA-OFT, https://arxiv.org/abs/2502.19645

No method can be paper-facing with SmolVLA-only evidence.

## Required Benchmarks

Primary benchmark:

- official LIBERO hard slices from the current scaleup

Second benchmark candidates:

- LIBERO-Plus for controlled robustness perturbations, https://arxiv.org/abs/2510.13626
- LIBERO-Occ if the later mechanism is visual/occlusion-linked, https://arxiv.org/abs/2606.10862
- RoboTwin 2.0 only if the mechanism is plausibly embodiment-general, https://arxiv.org/abs/2506.18088
- CALVIN only if the mechanism is long-horizon language composition rather than drawer contact, https://arxiv.org/abs/2112.03227

## Required Metrics

- closed-loop success rate with Wilson intervals
- paired reset-level win/tie/loss against frozen base and LoRA seeds
- per-task success, not only aggregate suite success
- failure-phase recurrence rate from videos
- intervention-trigger false-positive/false-negative rate, if any trigger exists
- compute/runtime overhead
- exact count of rerun outcome flips

## Current Matrix Decision

Do not run this matrix now. It is a guardrail for a later reopened method gate, not approval to implement a method.
