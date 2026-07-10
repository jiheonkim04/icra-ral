# Official LoRA Policy Loading Integration

Date: 2026-07-10 KST

## Problem

The frozen base loads through the official LeRobot policy factory. The persisted LoRA adapter directories did not load directly through the naive official path because adapter metadata still contains Windows-local paths, for example:

- `base_model_name_or_path`: `C:\assets\checkpoints\smolvla_libero`
- `vlm_model_name`: `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`

Direct loading raised a Hugging Face path validation error in WSL.

## Minimal Integration

The runner in `tca_map/smolvla/official_wsl_libero_rollout.py` keeps the official path intact:

1. Load the locked frozen base with `make_policy`.
2. Build official policy/env preprocessors and postprocessors.
3. Read adapter config with `PeftConfig.from_pretrained`.
4. Resolve only `peft_config.base_model_name_or_path` in memory to `/home/jiheon/assets/checkpoints/smolvla_libero`.
5. Wrap the official base policy with `PeftModel.from_pretrained`.
6. Move the wrapped policy to `cuda`.

No adapter files were edited, no checksums changed, and no observations/actions/control-mode semantics were altered.

## Audit Result

All four policies loaded and executed:

| Policy | Loaded | Class | Parameter device | Input device | Action chunk | PEFT |
| --- | --- | --- | --- | --- | --- | --- |
| `frozen_base` | yes | `SmolVLAPolicy` | `cuda:0` | `cuda:0` | `[1, 50, 7]` | no |
| `rank4_lora_seed_11` | yes | `PeftModel` | `cuda:0` | `cuda:0` | `[1, 50, 7]` | yes |
| `rank4_lora_seed_22` | yes | `PeftModel` | `cuda:0` | `cuda:0` | `[1, 50, 7]` | yes |
| `rank4_lora_seed_33` | yes | `PeftModel` | `cuda:0` | `cuda:0` | `[1, 50, 7]` | yes |

Autocast and AMP were inactive for the audits; parameters were loaded as `torch.bfloat16`, and action chunks were finite `torch.float32`.

## Test Coverage

`tests/test_official_wsl_libero_rollout.py` covers:

- alpha-zero static mix is exactly frozen base
- static-mix duplicate records are classified rather than run
- final-decision precedence
- pilot-only final decision
- official rename map
- smoke-only report writing does not clobber pilot result with an empty placeholder
