# AWF-XVLA Stage 0 Result

- Decision: `AWF_XVLA_STAGE0_NO_GO`
- Method: `Agentview-Wrist Fill for X-VLA`
- Condition: `wrist_camera_dropout_partial_observation`
- Run: `runs/xvla_prior/awf_xvla_stage0_wrist_dropout_task5_discovery_20260718T1502KST`
- No training, optimizer step, checkpoint, privileged state, reward/done/success trigger, or broad sweep was used.

## Result

- Clean baseline: `2/2`
- Frozen-prior wrist dropout: `0/2`
- AWF-XVLA wrist dropout: `0/2`
- Stage 0 GO: `False`

| identity | AWF success | steps | chunks | mitigation-triggered steps | result |
|---:|---:|---:|---:|---:|---|
| 20260731 | False | 900 | 30 | 900 | `runs/xvla_prior/awf_xvla_stage0_wrist_dropout_task5_discovery_20260718T1502KST/identity_20260731/result.json` |
| 20260732 | False | 900 | 30 | 900 | `runs/xvla_prior/awf_xvla_stage0_wrist_dropout_task5_discovery_20260718T1502KST/identity_20260732/result.json` |

Next action: Archive AWF-XVLA and do not tune it on the discovery failures.
