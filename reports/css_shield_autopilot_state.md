# CSS-Shield Autopilot State

- current main commit: `68ae7b824974141cc33cb4407db49915e39728bb`
- current stage: `STATE 3`
- last completed stage: `STATE 2`
- continue/kill decision: `continue`
- next milestone: `RA-L strength check`
- why next: State 2 randomized semantic/safety batch passed; next check is whether evidence has publishable strength.
- hard blockers: `[]`
- rollout happened: `True`
- training happened: `False`
- LoRA training happened: `False`
- loss computed: `False`
- GPU/download/heavy import/OpenVLA-OFT: `False` / `False` / `True` / `False`
- evidence level: `diagnostic_only`
- exact resume command: `powershell -ExecutionPolicy Bypass -File scripts\160_css_shield_autopilot_next.ps1`

## Last Result Summary

```json
{
  "diagnostic_only": true,
  "key_metric": {
    "state2_full_vs_clipping_unsafe_delta": 0.25,
    "state2_full_vs_clipping_wrong_target_delta": 0.7,
    "state2_full_vs_safety_wrong_target_delta": 0.7
  },
  "paper_grade": false,
  "passed": true,
  "state1_5_decision": {
    "continue": true,
    "full_beats_clipping_on_semantic_or_safety_metric": true,
    "full_beats_safety_on_semantic_metric": true,
    "full_stop_all": false,
    "kill_now": false,
    "reason": "semantic wrong-target shielding beat safety-only and clipping-only without stop-all behavior",
    "reframe": false,
    "semantic_only_catches_wrong_target": true,
    "state": "STATE 1.5"
  },
  "state2_decision": {
    "continue": true,
    "full_beats_clipping_on_semantic_or_safety_metric": true,
    "full_beats_safety_on_semantic_metric": true,
    "full_stop_all": false,
    "kill_now": false,
    "reason": "semantic wrong-target shielding beat safety-only and clipping-only without stop-all behavior",
    "reframe": false,
    "semantic_only_catches_wrong_target": true,
    "state": "STATE 2"
  }
}
```
