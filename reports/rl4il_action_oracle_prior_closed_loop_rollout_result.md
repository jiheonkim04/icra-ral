# RL4IL Action-Oracle Prior Closed-Loop Rollout Result

- Execution classification: `PRIOR_CLOSED_LOOP_ROLLOUT`
- Implementation label: `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT`
- Decision: `RL4IL_ACTION_ORACLE_PRIOR_LOCAL_RESIDUAL_ESTABLISHED`
- Valid: `True`
- Clean successes: `4/9`
- mask_1 successes: `3/9`
- Module forward count: `99`
- Peak VRAM MiB: `632.029296875`

This is an external RL4IL retrieval/imputation prior rollout, not Ours and not VLA fine-tuning.

| suite/task | identity | condition | success | steps | retrieved demo | imputation |
|---|---:|---|---|---:|---|---|
| `libero_goal/task0` | 20260733 | `clean` | `True` | 117 | `demo_18` | `False` |
| `libero_goal/task0` | 20260733 | `mask_1_in_hand_dropout` | `False` | 195 | `demo_17` | `True` |
| `libero_goal/task0` | 20260734 | `clean` | `True` | 117 | `demo_18` | `False` |
| `libero_goal/task0` | 20260734 | `mask_1_in_hand_dropout` | `False` | 195 | `demo_17` | `True` |
| `libero_goal/task0` | 20260735 | `clean` | `True` | 125 | `demo_23` | `False` |
| `libero_goal/task0` | 20260735 | `mask_1_in_hand_dropout` | `False` | 195 | `demo_17` | `True` |
| `libero_object/task0` | 20260733 | `clean` | `False` | 153 | `demo_32` | `False` |
| `libero_object/task0` | 20260733 | `mask_1_in_hand_dropout` | `True` | 183 | `demo_8` | `True` |
| `libero_object/task0` | 20260734 | `clean` | `True` | 133 | `demo_32` | `False` |
| `libero_object/task0` | 20260734 | `mask_1_in_hand_dropout` | `True` | 142 | `demo_17` | `True` |
| `libero_object/task0` | 20260735 | `clean` | `False` | 153 | `demo_32` | `False` |
| `libero_object/task0` | 20260735 | `mask_1_in_hand_dropout` | `True` | 142 | `demo_17` | `True` |
| `libero_spatial/task5` | 20260731 | `clean` | `False` | 121 | `demo_2` | `False` |
| `libero_spatial/task5` | 20260731 | `mask_1_in_hand_dropout` | `False` | 121 | `demo_2` | `True` |
| `libero_spatial/task5` | 20260732 | `clean` | `False` | 121 | `demo_2` | `False` |
| `libero_spatial/task5` | 20260732 | `mask_1_in_hand_dropout` | `False` | 121 | `demo_2` | `True` |
| `libero_spatial/task5` | 20260735 | `clean` | `False` | 121 | `demo_2` | `False` |
| `libero_spatial/task5` | 20260735 | `mask_1_in_hand_dropout` | `False` | 121 | `demo_2` | `True` |
