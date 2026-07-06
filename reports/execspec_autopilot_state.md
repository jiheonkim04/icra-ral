# ExecSpec-Repair Autopilot State

- branch: `codex/execspec-repair-state3`
- current stage: `STATE 3 complete`
- last completed stage: `STATE 3 multi-demo replay validation`
- continue/kill decision: `kill_or_reframe`
- next milestone: `reframe ExecSpec-Repair around mismatch-specific value or select a new route`
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
- multi-demo replay recovery: `17 / 19 degraded cases`
- simple baseline match count: `4`

## Resume Command

```powershell
$env:ALLOW_EXECSPEC_STATE3_REPLAY_VALIDATION="1"
powershell -ExecutionPolicy Bypass -File scripts\166_execspec_replay_validation.ps1
Remove-Item Env:\ALLOW_EXECSPEC_STATE3_REPLAY_VALIDATION -ErrorAction SilentlyContinue
```

## Last Result Summary

```json
{
  "calibration_demo_count": 5,
  "eval_demo_count": 3,
  "calibration_action_samples": 1403,
  "eval_action_samples": 805,
  "leakage_detected": false,
  "best_repair_method": "diagonal_affine_calibration",
  "full_repair_beats_identity": true,
  "full_repair_beats_clipping_only": true,
  "full_repair_beats_global_affine": true,
  "full_repair_mean_recovery_fraction": 1.0,
  "exact_init_replay_cases": 21,
  "degraded_replay_cases": 19,
  "success_recovered_cases": 17,
  "success_recovery_rate": 0.894736842,
  "simple_baseline_match_count": 4,
  "default_reset_expert_success": false,
  "default_reset_full_repair_success": false,
  "decision": "kill_or_reframe",
  "paper_grade": false
}
```
