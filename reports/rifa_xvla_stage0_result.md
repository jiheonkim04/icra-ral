# RIFA-XVLA Stage 0 Result

- Decision: `RIFA_XVLA_STAGE0_DESIGN_FAILURE`
- Execution classification: `OURS_VLA_TRAINING`
- Execution valid: `True`
- CUDA PID: `396`
- Peak VRAM MiB: `4189.80126953125`
- X-VLA forward calls: `75`
- Full trainable parameters: `214660`
- Full / ablation optimizer steps: `6 / 6`
- Dropout full-vs-Base max RMS: `0.1380136224897776`
- Dropout full-vs-ablation mean RMS: `1.0597695498731683e-06`
- Clean full-vs-Base max RMS: `0.0`

## Frozen gates

| gate | pass |
|---|---|
| `real_xvla_forward_path` | `True` |
| `cuda_execution` | `True` |
| `trainable_parameters_nonzero_and_matched` | `True` |
| `finite_nonzero_gradients` | `True` |
| `optimizer_steps_exact` | `True` |
| `weights_changed` | `True` |
| `checkpoint_write_and_disk_reload` | `True` |
| `base_preserving_initialization` | `True` |
| `missing_modality_signal_observable` | `True` |
| `rl4il_reliability_features_nonconstant` | `True` |
| `full_vs_no_reliability_difference` | `True` |
| `bounded_action_delta` | `False` |
| `clean_validation_retained` | `True` |
| `action_outputs_finite` | `True` |

No closed-loop Ours rollout or official success measurement occurred in Stage 0. The frozen X-VLA base, RL4IL checkpoints, and CLIP encoders remained frozen.
