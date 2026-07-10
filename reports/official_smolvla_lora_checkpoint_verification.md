# Official SmolVLA LoRA Checkpoint Verification

Date: 2026-07-10 KST

Final decision: `LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT`

## Seed Verification

| seed | status | path | adapter checksum recorded | disk reload | action schema | CPU fallback |
| ---: | --- | --- | --- | --- | --- | --- |
| 11 | `CHECKPOINT_COMPLETE_VERIFIED` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_11` | `True` | `True` | `7D` | `False` |
| 22 | `CHECKPOINT_COMPLETE_VERIFIED` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_22` | `True` | `True` | `7D` | `False` |
| 33 | `CHECKPOINT_COMPLETE_VERIFIED` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_33` | `True` | `True` | `7D` | `False` |

## Locked Revision Check

`{'model_expected_revision': '31d453f7edd78c839a8bbc39744a292686daf0de', 'model_local_metadata_revisions': ['31d453f7edd78c839a8bbc39744a292686daf0de'], 'dataset_expected_revision': 'a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4', 'dataset_local_metadata_revisions': ['a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4']}`

## Required Files

`['adapter_config.json', 'adapter_model.safetensors', 'training_manifest.json', 'eval_preprocessor_postprocessor_refs.json', 'source_repro_lock.yaml', 'sha256_manifest.json']`
