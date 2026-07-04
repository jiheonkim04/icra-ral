# LIBERO Learned-Policy Rollout Readiness Plan

This is the next rung after the bounded LIBERO/RoboSuite zero-action diagnostic rollout.

The goal is to decide whether a tiny learned-policy LIBERO rollout can run locally without inventing a risky Windows/WSL bridge or overstating evidence. This plan is readiness-only. It does not load SmolVLA, run inference, train, use GPU, create simulator environments, execute rollouts, download assets, execute OpenVLA-OFT, or make paper claims.

## Preferred Execution Topology

Use a WSL-only process for the first tiny learned-policy rollout:

- LIBERO/RoboSuite simulator runs in WSL,
- the selected WSL venv remains `/home/jiheon/.venvs/tca_map_sim`,
- local SmolVLA checkpoint files remain under `C:\assets\checkpoints\smolvla`,
- WSL sees those assets through `/mnt/c/...`,
- no Windows-policy/WSL-simulator IPC bridge is introduced until WSL-only readiness is exhausted.

The Windows-policy/WSL-simulator bridge is deliberately not treated as ready because it would add process, latency, synchronization, image serialization, and failure-mode complexity before the simpler topology is tested.

## Readiness Criteria

`scripts\66_plan_libero_policy_rollout_readiness.ps1` reports execution-ready only if:

- bounded LIBERO/RoboSuite diagnostic rollout has already passed,
- `LIBERO_ROOT`, `ROBOSUITE_ROOT`, `LIBERO_DATA_ROOT`, `SMOLVLA_CKPT`, and `HF_HOME` paths exist,
- SmolVLA config and weights are present,
- the tokenizer/processor dependency under `HF_HOME` is present,
- the selected WSL Python can see the lightweight SmolVLA runtime modules by module-spec lookup,
- the planned tiny rollout is inside budget: max 5 tasks, max 10 steps per task, max 30 minutes, max 14 GB VRAM,
- no token, login, payment, license click-through, OpenVLA-OFT, GPU training, or paper claim is required.

## Next Step Semantics

If the planner says `proceed`, create a separately gated tiny learned-policy rollout runner. The initial runner should use one LIBERO task, at most 5 to 10 steps, CPU by default, no training, no GPU by default, no OpenVLA-OFT, and no paper claim.

If the planner says `reduce_scope`, do not run learned-policy rollout yet. The next safe task is WSL SmolVLA runtime setup/readiness in the existing WSL venv.

If the planner says `stop`, resolve the listed local prerequisite blocker first.

## Evidence Label

Passing this readiness plan is not paper-grade evidence. A future tiny learned-policy rollout would still be a local pilot or bounded benchmark diagnostic unless it runs a documented benchmark protocol with verified outputs, baselines, and honest evidence labels.
