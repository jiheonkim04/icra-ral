# Project State

Date: 2026-07-09 KST

Branch: `codex/smolvla-7d-adapter-replay-bridge`

Current decision: `EXPERT_REPLAY_BLOCKED`

## Current Route

SmolVLA 7D Adapter Executable Replay Bridge is the active control-validity gate.

The local language/target route remains killed: do not continue TG-7D, TCA, PRISM, PatchGuard, SafeLoRA, or canonicalization work from the prior route.

## Fixed 7D Foundation

- fixed LIBERO_7D action interface is the only allowed path;
- best prior baseline: rank-8 state-proj LoRA + 7D adapter action L2 `0.494959`;
- mean-action L2 `1.082453`, MLP L2 `0.518738`, ridge L2 `0.890603` on the fixed-interface baseline;
- no old 6D/SO100 action label path and no hard-coded gripper fill.

## Replay Bridge Status

- adapter artifact reloadable: `True`
- training happened: `True`
- loss computed: `True`
- replay/control happened: `False`
- offline held-out replay-demo mean/action/ridge/adapter L2: `1.104166` / `0.893329` / `0.464353`
- env acceptance status: `blocked: ModuleNotFoundError: No module named 'mujoco'`

## Conclusion

`EXPERT_REPLAY_BLOCKED`

Install or activate the local `mujoco` Python dependency for LIBERO/RoboSuite in the `tca_map` environment, then rerun this same replay bridge; do not start a new method.
