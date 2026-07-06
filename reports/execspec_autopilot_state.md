# ExecSpec-Repair Autopilot State

- branch: `codex/execspec-repair-state0-state1`
- current stage: `STATE 1 complete`
- last completed stage: `STATE 1 executable mismatch reproduction`
- continue/kill decision: `continue`
- next milestone: `STATE 2 calibrated repair replay`
- rollout/replay happened: `true`
- training happened: `false`
- loss computed: `false`
- GPU/download/OpenVLA-OFT happened: `false`
- evidence level: `bounded exact-init diagnostic`
- mismatch reproduced: `true`
- strongest mismatch: `gripper_sign_flip`
- replay degradation: `true`
- simple baseline beaten: `true`, supervised calibration metric only

## Resume Command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\163_execspec_mismatch_diagnostic.ps1
$env:ALLOW_EXECSPEC_MISMATCH_REPLAY="1"
powershell -ExecutionPolicy Bypass -File scripts\164_execspec_exact_init_mismatch_replay.ps1
Remove-Item Env:\ALLOW_EXECSPEC_MISMATCH_REPLAY -ErrorAction SilentlyContinue
```

## Last Result Summary

```json
{
  "hdf5_strongest_mismatch": "gripper_sign_flip",
  "hdf5_action_l2_mean": 2.0,
  "hdf5_gripper_mismatch_rate": 1.0,
  "expert_replay_reward_success": "1.0 / true",
  "gripper_flip_replay_reward_success": "0.0 / false",
  "translation_scale_replay_reward_success": "0.0 / false",
  "replay_degradation": true,
  "paper_grade": false
}
```
