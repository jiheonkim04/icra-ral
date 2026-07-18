# CVLR-XVLA Stage 0 Result

- Decision: `CVLR_XVLA_STAGE0_DESIGN_FAILURE`
- Execution valid: `True`
- CUDA PID: `355`
- Peak VRAM MiB: `3537.66650390625`
- Trainable parameters: `422144`
- Optimizer steps: `96`
- Validation MSE full / zero / AWF: `0.44344032059113186 / 0.9941876629988352 / 1.559069037437439`
- Clean translation / rotation max RMS: `0.0 / 0.0`
- Dropout gripper flips full vs Base: `42`

## Frozen gates

| gate | pass |
|---|---|
| `target_records_valid` | `True` |
| `split_integrity` | `True` |
| `real_xvla_forward_path` | `True` |
| `cuda_execution` | `True` |
| `trainable_parameter_count_exact` | `True` |
| `finite_nonzero_gradients` | `True` |
| `optimizer_steps_exact` | `True` |
| `weights_changed` | `True` |
| `checkpoint_write_and_disk_reload` | `True` |
| `xvla_frozen` | `True` |
| `wrist_insertion_path_active` | `True` |
| `reconstruction_meaningfully_beats_controls` | `True` |
| `prediction_noncollapsed` | `True` |
| `meaningful_full_vs_no_reconstruction_action_effect` | `True` |
| `exact_clean_bypass` | `True` |
| `semantic_action_safety` | `False` |
| `action_outputs_finite` | `True` |

Continuous translation, continuous rotation, raw gripper score, and final discrete gripper flips were evaluated separately.
No closed-loop rollout, official success measurement, threshold tuning, or privileged inference input occurred in Stage 0.
