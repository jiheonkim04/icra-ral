# Official SmolVLA Training Determinism Status

Date: 2026-07-10 KST

Training happened in this audit: `False`
Optional single-seed probe ran: `False`

## Historical Identity Answers

1. Were the old adapter weights persisted? `False`
2. Was the complete historical RNG/DataLoader state persisted? `False`
3. Was the exact historical sample order persisted? `False`
4. Can the old learned policy identity be reconstructed? `False`
5. Is exact old metric reproduction scientifically possible? `False`
6. Is the observed difference config drift or ordinary retraining variance? `configuration/protocol drift is directly observed: historical in-memory evaluation did not persist/reload adapters and did not assign wrap_with_peft return; regenerated evaluation uses assigned PEFT wrapper and PeftModel.from_pretrained.`

## Loss Sequence Evidence

All old-vs-regenerated loss sequences identical: `True`

| seed | loss sequence identical | old adapter path | regenerated adapter path |
| ---: | --- | --- | --- |
| 11 | `True` | `None` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_11` |
| 22 | `True` | `None` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_22` |
| 33 | `True` | `None` | `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_33` |

Terminology:

- historical run: `5d48b1e ephemeral in-memory seed-reproduction run`
- regenerated persisted run: `15649d6 run that saved/reloaded rank-4 adapter bundles`
- canonical persisted checkpoint: `not accepted in this audit because protocol drift was found`
