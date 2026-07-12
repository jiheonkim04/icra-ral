# Epoch 3 Cycle 1 CBFD-VLA Adjudication

Date: 2026-07-12 KST

Decision: `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`

This is a valid current-formulation kill for `CBFD-VLA`, not a terminal campaign decision.

## Method Tested

`CBFD-VLA` used Quantized OpenVLA-OFT INT4 as a bounded training-time teacher for successful traces on two SmolVLA-hard LIBERO tasks. The student policies were lightweight state/task action adapters evaluated without loading the teacher at inference.

## Evidence

Teacher acquisition:

- planned episodes: `10`
- completed episodes: `10`
- successful teacher episodes: `10`
- teacher trace rows: `1765`

Student training:

- teacher rows: `1765`
- retention rows: `192`
- direct distillation proxy checkpoint: `4f54a416015634460ceb8ca52586d2f0321a5c9955530e754fbc240b0689298c`
- no-retention checkpoint: `2551f23174dd46d9732f7d6bd09eeb2ce3cbafec3e8cd885eddc93ae33bab3af`
- full CBFD checkpoint: `14c95a910adb9d980e3fb93e47ccc99af857a51fc2dc5eaab466045462ff8c20`

Stage A completed `50 / 50` held-out episodes with zero exceptions:

| Policy | Success | Task-balanced |
| --- | ---: | ---: |
| frozen SmolVLA | `7 / 10` | `0.70` |
| direct distillation proxy | `0 / 10` | `0.00` |
| teacher trace memory | `0 / 10` | `0.00` |
| CBFD no-retention | `0 / 10` | `0.00` |
| CBFD full | `0 / 10` | `0.00` |

Per-task full CBFD:

- `libero_spatial/task_4`: `0 / 5`
- `libero_10/task_4`: `0 / 5`

Mechanism activation:

- mean action delta full versus direct distillation: `1.244676`
- mean action delta full versus teacher memory: `1.652989`

## Ruling

The implementation was valid enough to make a scientific decision:

- teacher acquisition succeeded;
- student training passed;
- Stage A completed with zero exceptions;
- the full method acted differently from direct distillation and memory baselines;
- no teacher was used at student inference.

Under `reports/current_research_governance.md`, Stage A may permanently kill when the full method has `0 / 10` success while a paired baseline has at least `4 / 10`. CBFD full had `0 / 10`; frozen SmolVLA had `7 / 10`.

Do not rescue this CBFD formulation by changing hidden size, retention weights, teacher train identities, memory distance, held-out identities, or by converting it into online teacher routing. The failed assumption is that successful cross-backbone teacher traces can be transferred through a lightweight state/task student adapter without destroying closed-loop behavior.

Next action: continue to Epoch 3 Cycle 2 under `reports/current_research_governance.md`.
