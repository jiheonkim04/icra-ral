# Official SmolVLA Artifact And Evaluation Alignment

Date: 2026-07-10 KST

Protocol drift from frame/label/base alignment: `False`

| seed | records | split/label/target/base aligned | max target diff | max frozen/base diff | max old-vs-regen LoRA action L2 | mean old-vs-regen LoRA action L2 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 11 | 2800 | `True` | 0.0 | 0.0 | 2.048022083 | 0.072587957 |
| 22 | 2800 | `True` | 0.0 | 0.0 | 2.048055725 | 0.072563036 |
| 33 | 2800 | `True` | 0.0 | 0.0 | 2.066805781 | 0.072501573 |

## Alignment Verdict

- test frame IDs: identical
- task and episode IDs: identical
- ground-truth actions: identical
- split membership: identical
- frozen/base predictions: identical
- metric protocol file: identical across historical and regenerated commits
- static-alpha grid: `[0.0, 0.25, 0.5, 0.75, 1.0]`, validation-only selection
- test leakage: not introduced by this audit; split manifest leakage checks remain the authority

The frame/label/base protocol aligns. The observed drift is in the LoRA prediction path, not in labels or split membership.
