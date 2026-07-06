# Pivot Options

The current low-compute RA-L route is killed. Honest options are below.

## A. New Project With Full Online VLA Action-Head Training

Start a new project that explicitly trains an online VLA action head or policy adapter, using enough compute and data to produce rollout-quality actions.

This would be a new project, not a small continuation of the current low-compute route.

Requirements:

- stronger online action model,
- real training budget,
- simulator rollouts,
- proper benchmark comparison,
- no OpenVLA-OFT local shortcut unless compute is available,
- no claims until rollout results exist.

## B. New Project Focused On Benchmark / Diagnostic Rather Than RA-L Control Improvement

Reframe the work as a diagnostic benchmark or tooling paper: target-prior leakage, action-source provenance, offline proxy vs rollout mismatch, and bridge validation for low-compute VLA research.

This is more honest than forcing a control-improvement claim.

## C. Use Logs As Negative Result / Internal Research Note

Preserve the project as an internal negative result:

- fixed-prior TCA can look strong offline,
- rollout bridge can be correct,
- expert replay can succeed,
- but weak online action heads can still kill the method.

This option has research value and prevents repeating the same route.

## D. Abandon And Start A More Rollout-First Topic

Stop investing in this method family and pick a new topic where rollout feedback is available earlier.

This may be the cleanest option if the goal is a publishable robotics paper under limited time.

## Non-Options

- Do not revive TCA-Select as the main novelty from current evidence.
- Do not claim representation collapse.
- Do not submit the current evidence as RA-L-ready.
- Do not run more rollout variants from the current weak 7D head.

