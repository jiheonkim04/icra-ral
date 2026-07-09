# Official SmolVLA / LeRobot Model Load Status

Date: 2026-07-09 KST

## Summary

The local SmolVLA base checkpoint loads with the official LeRobot policy loader and runs one finite synthetic CPU forward pass with LeRobot's official policy processor factory.

This is a CPU-only diagnostic, not LoRA training, not evaluation, not rollout, and not paper evidence.

## Loader

- Loader: `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained`
- Processor factory: `lerobot.policies.factory.make_pre_post_processors`
- Checkpoint: `C:\assets\checkpoints\smolvla`
- VLM/tokenizer dependency: local `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`
- Downloads: no
- OpenVLA-OFT: no
- Custom LIBERO 7D adapter: no

## Config

| field | value |
| --- | --- |
| policy type | `smolvla` |
| state shape | `[6]` |
| action shape | `[6]` |
| visual input shape | `[3, 256, 256]` for three cameras |
| chunk size | `50` |
| action steps | `50` |
| inference diffusion steps for smoke | `1` |
| max state/action dim | `32 / 32` |
| `load_vlm_weights` | `false` |

## Normalizer Status

The checkpoint contains 6D SO100 action normalizer tensors:

- `so100.buffer.action.mean/std`
- `so100-blue.buffer.action.mean/std`
- `so100-red.buffer.action.mean/std`

No LIBERO 7D action normalizer was found in the local checkpoint.

## One-Sample Forward Result

- Model loaded: yes
- Parameter count: `450,046,176`
- Trainable parameter count in loaded policy object: `99,880,992`
- Parameter device: `cpu`
- Input tensor devices: `cpu`
- Autocast GPU enabled: `false`
- Autocast CPU enabled: `false`
- CUDA available: yes
- CUDA memory allocated before/after: `0.0 MB / 0.0 MB`
- CUDA max memory allocated: `0.0 MB`
- Single-sample inference runtime: `1.735 sec`
- End-to-end smoke runtime: `30.922 sec`
- Raw action shape: `[1, 6]`
- Postprocessed action shape: `[1, 6]`
- Raw action finite: yes
- Postprocessed action finite: yes
- Action preview: `[-0.032545, -0.181919, -0.077109, 0.011333, 0.151391, -0.164538]`

## Interpretation

This proves local official SmolVLA base loader/processor feasibility. It does not prove official LIBERO baseline feasibility because the checkpoint is 6D SO100-style while LeRobot LIBERO expects 8D state and 7D actions.

## LoRA Feasibility Check

LeRobot exposes PEFT through `PreTrainedPolicy.wrap_with_peft`, and SmolVLA defines official default target modules:

```text
(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))
```

A no-training, CPU count-only wrap on the local SmolVLA architecture found:

| LoRA rank | total params | trainable params | trainable % |
| ---: | ---: | ---: | ---: |
| 4 | 450,231,840 | 185,664 | 0.0412 |
| 16 | 450,788,832 | 742,656 | 0.1647 |

Interpretation: official SmolVLA PEFT attachment is technically feasible and batch size 1 should be plausible on the RTX 5080 16GB for a tiny smoke, but training is blocked until the dataset/action convention is official-compatible. Any future LoRA training must log model/input devices, CUDA allocated/max memory, and autocast/fp16/bf16 state.
