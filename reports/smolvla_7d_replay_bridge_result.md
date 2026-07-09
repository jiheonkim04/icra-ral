# SmolVLA 7D Adapter Replay Bridge Result

Final decision: `READY_FOR_METHOD_AFTER_REPLAY_BRIDGE`

This is a bounded control-validity gate for the already-fixed 7D baseline, not a method claim.

## Summary

- branch: `codex/smolvla-7d-replay-mujoco-unblock`
- experiments happened: `True`
- training happened: `False`
- loss computed: `False`
- replay/control happened: `True`
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
- replay env acceptance: `accepted_by_env_step`

## Offline Replay-Demo Metrics

- expert action L2: `0.0`
- mean-action L2: `1.104166`
- ridge L2: `0.893329`
- SmolVLA 7D adapter L2: `0.464353`
- adapter translation / rotation / gripper error: `0.252312 / 0.068959 / 0.33562`
- adapter clip rate element/step: `0.061303 / 0.429119`
- adapter controller-valid proxy rate: `0.570881`

## Replay

- replay executed: `True`
- replay reason: `bounded exact-init replay attempted`
- replay error: `None`
- expert replay reward/success: reward_sum `1.0`, final_success `True`, first_done_index `250`, progress_proxy `0.229161`
- mean-action replay result: reward_sum `0.0`, final_success `False`, progress_proxy `0.106222`
- ridge replay result: reward_sum `0.0`, final_success `False`, progress_proxy `0.167573`
- SmolVLA 7D adapter replay result: reward_sum `0.0`, final_success `False`, progress_proxy `0.234297`
- action L2 vs replay progress relationship: adapter improves over mean/ridge offline and beats both on the target-distance progress proxy, but it does not yet solve the task.
- exact next step: reproduce a real SmolVLA LoRA baseline on an official or standard task split before starting any new method.
