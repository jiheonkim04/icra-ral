# SmolVLA 7D Adapter Replay Bridge Result

Final decision: `EXPERT_REPLAY_BLOCKED`

This is a bounded control-validity gate for the already-fixed 7D baseline, not a method claim.

## Summary

- branch: `codex/smolvla-7d-adapter-replay-bridge`
- experiments happened: `True`
- training happened: `True`
- loss computed: `True`
- replay/control happened: `False`
- downloads happened: `False`
- OpenVLA-OFT happened: `False`
- model/adapter used: `smolvla_state_proj_lora_rank8_7d_adapter`
- dataset/demo used: `C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5::demo_30`
- adapter artifact: `runs\smolvla_7d_replay_bridge\smolvla_state_proj_lora_rank8_7d_adapter.pt`
- adapter reloadable: `True`
- output exactly 7D: `True`
- train-split-only normalization: `True`
- learned gripper output: `True`
- unnormalize correct: `True`
- replay env acceptance: `blocked: ModuleNotFoundError: No module named 'mujoco'`

## Offline Replay-Demo Metrics

- expert action L2: `0.0`
- mean-action L2: `1.104166`
- ridge L2: `0.893329`
- SmolVLA 7D adapter L2: `0.464353`
- adapter translation / rotation / gripper error: `0.252312 / 0.068959 / 0.33562`
- adapter clip rate element/step: `0.061303 / 0.429119`
- adapter controller-valid proxy rate: `0.570881`

## Replay

- replay executed: `False`
- replay reason: `Failed to import or configure LIBERO/RoboSuite exact-init environment.`
- replay error: `ModuleNotFoundError: No module named 'mujoco'`
- expert replay reward/success: `None`
- mean-action replay result: `None`
- MLP/ridge replay result: `None`
- SmolVLA 7D adapter replay result: `None`
- action L2 vs replay progress relationship: not assessed because exact-init replay/control did not execute
- exact next step: Install or activate the local `mujoco` Python dependency for LIBERO/RoboSuite in the `tca_map` environment, then rerun this same replay bridge; do not start a new method.
