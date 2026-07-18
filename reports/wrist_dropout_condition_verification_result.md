# Wrist-Camera Dropout Condition Verification

- Decision: `CLAIM_CONDITION_OFFICIAL_PRIOR_DEGRADATION_VERIFIED`
- Condition: `wrist_camera_dropout_partial_observation`
- Run: `runs/xvla_prior/claim_condition_wrist_dropout_task5_discovery_20260718T1455KST`
- Policy: frozen official X-VLA-Libero only.
- No Ours method, optimizer, checkpoint, or control rollout was selected or executed before this condition check.

## Result

- Clean baseline successes: `2/2`
- Wrist-dropout successes: `0/2`
- Success drop: `2`
- Condition verified: `True`

| identity | clean baseline success | dropout success | dropout steps | chunks | result |
|---:|---:|---:|---:|---:|---|
| 20260731 | True | False | 900 | 30 | `runs/xvla_prior/claim_condition_wrist_dropout_task5_discovery_20260718T1455KST/identity_20260731/result.json` |
| 20260732 | True | False | 900 | 30 | `runs/xvla_prior/claim_condition_wrist_dropout_task5_discovery_20260718T1455KST/identity_20260732/result.json` |

Next action: Select one method candidate within the steer budget.
