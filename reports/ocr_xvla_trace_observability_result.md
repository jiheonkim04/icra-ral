# OCR-XVLA Bounded Trace Observability Result

- Decision: `OCR_TRIGGER_OBSERVABILITY_FAIL`
- Trace run: `runs/xvla_prior/ocr_trace_acquisition_task5_discovery_20260718T1437KST`
- Policy: frozen official X-VLA prior only; no Ours rollout, no optimizer, no checkpoint.
- Discovery identities only: residual `20260727/20260730/20260733`; clean-retention `20260731/20260732`.
- Held-out identities not used: `20260734/20260735/20260736/20260737`.

## Preregistered test

- Attempt window: first gripper-close step, fallback post-step20 +1.5cm EEF z-rise; `120` steps.
- PASS requires a legal RGB/proprio feature to strictly separate residual failures from clean-retention successes above the action-history-only baseline.
- Reward, done, success, simulator object/contact state, privileged pose, and future observations were not trigger features.

## Best separations

- Action-history-only best: `action_history_chunk_count_window`, strict=True, normalized_gap=0.33333322222225925
- Legal observation/proprio best: `agentview_mean_frame_delta_from_attempt`, strict=True, normalized_gap=0.3044140909296114
- Observation/proprio above action baseline: `False`

## Per-identity trace summary

| identity | role | completed | success label for offline scoring only | steps | chunks | attempt window | trace |
|---:|---|---:|---:|---:|---:|---|---|
| 20260727 | residual_failure | True | False | 900 | 30 | 37..157 | `runs/xvla_prior/ocr_trace_acquisition_task5_discovery_20260718T1437KST/identity_20260727/legal_trace/libero_spatial_task5_identity20260727_trace.npz` |
| 20260730 | residual_failure | True | False | 900 | 30 | 35..155 | `runs/xvla_prior/ocr_trace_acquisition_task5_discovery_20260718T1437KST/identity_20260730/legal_trace/libero_spatial_task5_identity20260730_trace.npz` |
| 20260731 | clean_retention_success | True | True | 88 | 3 | 35..88 | `runs/xvla_prior/ocr_trace_acquisition_task5_discovery_20260718T1437KST/identity_20260731/legal_trace/libero_spatial_task5_identity20260731_trace.npz` |
| 20260732 | clean_retention_success | True | True | 128 | 5 | 36..128 | `runs/xvla_prior/ocr_trace_acquisition_task5_discovery_20260718T1437KST/identity_20260732/legal_trace/libero_spatial_task5_identity20260732_trace.npz` |
| 20260733 | residual_failure | True | False | 900 | 30 | 35..155 | `runs/xvla_prior/ocr_trace_acquisition_task5_discovery_20260718T1437KST/identity_20260733/legal_trace/libero_spatial_task5_identity20260733_trace.npz` |

## Interpretation

The one permitted trace test did not identify a legal no-progress trigger above the action-history-only baseline; OCR is archived under the user steer.

This result does not reopen SGL-XVLA and does not execute OCR-XVLA as an intervention policy.
