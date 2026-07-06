# ExecSpec-Repair Autopilot State

- branch: `codex/execspec-repair-state3-5`
- current stage: `STATE 3.5 complete`
- last completed stage: `STATE 3.5 baseline dominance and reframe audit`
- continue/kill decision: `kill`
- next milestone: `archive ExecSpec-Repair or select a new rollout-first route`
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
- best single simple baseline: `diagonal_affine_calibration`
- full minus best single simple baseline: `0.0`
- repair selector/routing meaningful: `false`

## Resume Command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\167_execspec_baseline_dominance_audit.ps1
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
  "best_single_simple_baseline": "diagonal_affine_calibration",
  "best_single_simple_baseline_success_recovery_rate": 0.894736842,
  "best_single_simple_baseline_action_recovery": 1.0,
  "full_gain_over_best_single_simple_baseline": 0.0,
  "selector_success_recovery_rate": 0.894736842,
  "selector_gain_over_best_single_simple_baseline": 0.0,
  "decision": "kill",
  "paper_grade": false
}
```
