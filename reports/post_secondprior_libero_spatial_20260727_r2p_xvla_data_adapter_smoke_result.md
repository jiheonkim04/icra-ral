# R2P-XVLA Data-Adapter Smoke

Decision: `R2P_XVLA_DATA_ADAPTER_SMOKE_PASS`

This was a pre-optimizer interface check for `R2P-XVLA`. It materialized two LIBERO-spatial task-5 demos into the local X-VLA reader format and instantiated the official X-VLA dataset reader. It did not load a VLA model, train, run backward, run an optimizer step, write a checkpoint, launch a simulator, or evaluate Ours.

## Runtime artifact

- Result: `runs/xvla_prior/r2p_xvla_task5_data_adapter_smoke_20260718T0417KST/result.json`
- Result SHA-256: `c0e44013d31f364beeea134c7991f55fb12d3643e4746961d993eeb2f19288e6`
- Source HDF5: `C:/assets/data/libero/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5`
- X-VLA root: `C:/assets/repos/X-VLA`

## Converted demos

| Demo | Steps | `abs_action_6d` shape | Phase counts source / transit / target |
| --- | ---: | --- | --- |
| `demo_0` | 137 | `[137, 10]` | `67 / 23 / 47` |
| `demo_40` | 117 | `[117, 10]` | `61 / 14 / 42` |

Combined phase coverage: source `128`, transit `37`, target `89`.

## Official reader smoke

| Field | Shape | Dtype |
| --- | --- | --- |
| `action` | `[30, 20]` | `torch.float32` |
| `proprio` | `[20]` | `torch.float32` |
| `image_input` | `[3, 3, 224, 224]` | `torch.float32` |
| `image_mask` | `[3]` | `torch.bool` |
| `domain_id` | `[]` | `torch.int64` |

Language instruction:

`pick up the black bowl on the ramekin and place it on the plate`

## No-training statement

- Training happened: `false`
- Optimizer step happened: `false`
- Checkpoint written: `false`
- Model loaded: `false`
- Backward happened: `false`
- Closed-loop Ours evaluation happened: `false`
- Simulator rollout happened: `false`
- LoRA/QLoRA training happened: `false`

## Validation

- `py_compile`: passed for `data_adapter_smoke.py` and its focused test.
- Focused pytest: `2 passed`.
- Real smoke command completed with exit code `0`.

## Next action

Run a one-batch `R2P-XVLA` gradient smoke under the frozen spec without `optimizer.step`, checkpoint writes, downloads, simulator rollouts, or closed-loop Ours evaluation.
