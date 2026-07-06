# ExecSpec-Repair Autopilot State

- branch: `codex/execspec-repair-state2`
- current stage: `STATE 2 complete`
- last completed stage: `STATE 2 calibrated repair replay`
- continue/kill decision: `continue`
- next milestone: `STATE 3 replay/rollout validation`
- rollout/replay happened: `true`
- training happened: `false`
- loss computed: `false`
- GPU/download/OpenVLA-OFT happened: `false`
- evidence level: `bounded exact-init diagnostic`
- mismatch reproduced: `true`
- strongest mismatch: `gripper_sign_flip`
- replay degradation: `true`
- eval leakage detected: `false`
- full repair beats identity/clipping/global affine: `true / true / true`
- calibrated repair replay recovery: `true`

## Resume Command

```powershell
$env:ALLOW_EXECSPEC_CALIBRATED_REPAIR_REPLAY="1"
powershell -ExecutionPolicy Bypass -File scripts\165_execspec_calibrated_repair.ps1
Remove-Item Env:\ALLOW_EXECSPEC_CALIBRATED_REPAIR_REPLAY -ErrorAction SilentlyContinue
```

## Last Result Summary

```json
{
  "calibration_demo_count": 5,
  "eval_demo_count": 1,
  "calibration_action_samples": 1403,
  "eval_action_samples": 272,
  "leakage_detected": false,
  "best_repair_method": "diagonal_affine_calibration",
  "full_repair_beats_identity": true,
  "full_repair_beats_clipping_only": true,
  "full_repair_beats_global_affine": true,
  "full_repair_mean_recovery_fraction": 1.0,
  "gripper_sign_flip_full_repair_reward_success": "1.0 / true",
  "translation_scale_full_repair_reward_success": "1.0 / true",
  "repair_improves_replay_reward_or_success": true,
  "paper_grade": false
}
```
