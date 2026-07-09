# Project State

Date: 2026-07-09 KST

Branch: `codex/smolvla-7d-replay-mujoco-unblock`

Current decision: `READY_FOR_METHOD_AFTER_REPLAY_BRIDGE`

## Current Route

SmolVLA 7D Adapter Replay Bridge is unblocked for one bounded exact-init LIBERO demo.

The local language/target route remains killed: do not continue TG-7D, TCA, PRISM, PatchGuard, SafeLoRA, or canonicalization work from the prior route.

## Fixed 7D Foundation

- fixed LIBERO_7D action interface is the only allowed path;
- best prior baseline: rank-8 state-proj LoRA + 7D adapter action L2 `0.494959`;
- mean-action L2 `1.082453`, MLP L2 `0.518738`, ridge L2 `0.890603` on the fixed-interface baseline;
- no old 6D/SO100 action label path and no hard-coded gripper fill.

## Replay Bridge Status

- adapter artifact reloadable: `True`
- training happened: `False`
- loss computed: `False`
- replay/control happened: `True`
- offline held-out replay-demo mean/action/ridge/adapter L2: `1.104166` / `0.893329` / `0.464353`
- env acceptance status: `accepted_by_env_step`
- expert exact-init replay: reward_sum `1.0`, success `True`, first_done_index `250`
- mean/ridge/adapter progress proxy: `0.106222` / `0.167573` / `0.234297`
- adapter clip rate element/step: `0.061303` / `0.429119`
- adapter controller-valid proxy rate: `0.570881`

## Conclusion

`READY_FOR_METHOD_AFTER_REPLAY_BRIDGE`

The reusable result is an executable real SmolVLA/LIBERO 7D replay path on the local Windows RTX setup. The next valid research step is standard SmolVLA LoRA baseline reproduction on an official or standard task split, not a new method.
