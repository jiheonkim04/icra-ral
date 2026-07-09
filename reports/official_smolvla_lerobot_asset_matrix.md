# Official SmolVLA / LeRobot Asset Matrix

Date: 2026-07-09 KST

## Local Assets

| asset | path | present | status |
| --- | --- | ---: | --- |
| Local SmolVLA checkpoint | `C:\assets\checkpoints\smolvla` | yes | complete base-style checkpoint |
| SmolVLA config | `C:\assets\checkpoints\smolvla\config.json` | yes | 6D state/action |
| SmolVLA weights | `C:\assets\checkpoints\smolvla\model.safetensors` | yes | 906,712,520 bytes |
| Policy preprocessor | `policy_preprocessor.json` | yes | tokenizer + normalizer |
| Policy postprocessor | `policy_postprocessor.json` | yes | unnormalizer |
| Action normalizer tensors | pre/postprocessor safetensors | yes | SO100 6D tensors |
| VLM tokenizer/cache | `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct` | yes | local tokenizer/processor/config/weights |
| Local LIBERO HDF5 | `C:\assets\data\libero` | yes | raw HDF5 data, not official LeRobot dataset format |
| Official `lerobot/libero` cache | `C:\assets\hf_home\datasets--lerobot--libero` | no | not downloaded |
| Official `lerobot/smolvla_libero` checkpoint | `C:\assets\checkpoints\smolvla_libero` | no | not downloaded |

## Local Checkpoint Schema

| field | value |
| --- | --- |
| policy type | `smolvla` |
| state input | `observation.state`, shape `[6]` |
| image inputs | three camera tensors, shape `[3, 256, 256]` |
| action output | `action`, shape `[6]` |
| normalization | VISUAL identity, STATE mean/std, ACTION mean/std |
| action normalizer provenance | SO100-named tensors |
| LIBERO compatible as-is | no |

## External Official Assets Identified

No external asset was downloaded in this pass.

| asset | source | approximate size | gated | note |
| --- | --- | ---: | ---: | --- |
| `lerobot/smolvla_libero` | Hugging Face model hub | 0.844 GB | false | official LIBERO SmolVLA checkpoint candidate |
| `lerobot/libero` | Hugging Face dataset hub | 1.803 GB | false | LeRobot-format LIBERO dataset candidate |
| `HuggingFaceVLA/libero` | Hugging Face dataset hub | 32.528 GB | false | too large for this bounded pass |

## Asset Decision

The local base checkpoint is sufficient for a bounded official SmolVLA loader/processor mini-repro. It is not sufficient for official LIBERO baseline reproduction. The official LIBERO route requires either new official assets or a clean dataset/checkpoint alignment plan.

