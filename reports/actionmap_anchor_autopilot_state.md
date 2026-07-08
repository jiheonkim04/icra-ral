# ActionMap Anchor Autopilot State

## Current State

- Branch: `codex/actionmap-anchor-reproduction-state0-state1`
- Base: local and pushed `main` at `5dac9f0`
- Milestone: STATE 0 docs plus STATE 1 mini ActionMap-style feasibility diagnostic completed
- Decision: `kill`
- Next state: `kill_or_reframe_anchor_reproduction`
- Heavy training: not allowed
- Full VLA fine-tuning: not allowed
- OpenVLA-OFT: blocked
- GPU jobs: not part of STATE 1
- Simulator rollouts: not part of STATE 1
- Evidence label: mini ActionMap-style local HDF5 action-head feasibility

## STATE 1 Result

- usable local HDF5 demos: `8`
- train/eval records: `1008 / 432`
- split: `deterministic_per_demo_time_holdout`
- mean-action action L2: `0.466767673`
- linear/L1 action L2: `0.812610317`
- simple MLP action L2: `0.501926707`
- ActionMap-style action L2: `0.529931357`
- oracle nearest-candidate action L2: `0.065653208`
- ActionMap-style beats mean/linear: `false / true`
- cheap MLP matches or beats ActionMap-style: `true`
- candidate collapse: `true`, unique translation/rotation/gripper bins `5 / 1 / 2`
- replay/control metric: `false`
- GPU/download/OpenVLA-OFT: `false / false / false`
- official ActionMap reproduction / extension / failure mining: `false / false / false`

Triggered kill gates:

- mean-action baseline matches or beats the ActionMap-style heatmap head,
- cheap MLP action head matches or beats the ActionMap-style heatmap head,
- ActionMap-style head collapsed to too few action candidates.

## Executable

Safe runner:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\210_actionmap_anchor_diagnostic.ps1
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

The runner trains tiny CPU NumPy heads only. It refuses download, GPU, rollout, simulator, heavy-import, runtime-install, OpenVLA, and OpenVLA-OFT gates.

## Current Recommendation

Do not invent an ActionMap extension or proceed to STATE 2 failure mining from this evidence. The local mini-anchor approximation did not pass the simple-baseline and candidate-collapse gates.
