# Epoch 3 Cycle 2 SCVC-VLA Adjudication

Date: 2026-07-12 KST

Decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

This is a valid current-formulation kill for `SCVC-VLA`, not a terminal campaign decision.

## Method Tested

`SCVC-VLA` tested calibration-derived image canonicalization under a predeclared synthetic sensor shift. The frozen SmolVLA policy produced all actions; SCVC only transformed preprocessed image tensors before inference.

## Evidence

Synthetic mechanism:

- final decision: `SYNTHETIC_MECHANISM_PASS`

Calibration:

- final decision: `CALIBRATION_PASS`
- calibration rows: `10`
- camera streams: `observation.images.camera1`, `observation.images.camera2`

Stage A:

- final decision: `STAGE_A_NON_GO_TO_STAGE_B_REQUIRED`
- episodes: `50 / 50`
- full SCVC: `4 / 10`
- shifted frozen SmolVLA: `4 / 10`
- known inverse affine: `5 / 10`
- SCVC no-temporal: `5 / 10`

Stage B completed `200 / 200` held-out episodes with zero exceptions:

| Policy | Success | Task-balanced |
| --- | ---: | ---: |
| clean frozen SmolVLA | `10 / 40` | `0.25` |
| shifted frozen SmolVLA | `20 / 40` | `0.50` |
| known inverse affine | `10 / 40` | `0.25` |
| SCVC no-temporal | `10 / 40` | `0.25` |
| SCVC full | `11 / 40` | `0.275` |

Paired comparison against the strongest baseline, `shifted_frozen_smolvla`:

- paired wins: `4`
- paired losses: `13`
- paired ties: `23`
- paired success delta: `-0.225`
- paired bootstrap CI: `[-0.425, -0.025]`

Mechanism activation:

- shifted image MSE versus clean: `0.01981`
- full output MSE versus clean: `0.016791`
- mean full image delta versus shifted input: `0.111153`

## Ruling

The method acted and slightly reduced image-level error, but this did not transfer to closed-loop success. The strongest baseline was the uncorrected shifted frozen policy at `20 / 40`, while full SCVC reached only `11 / 40`. The paired confidence interval versus shifted frozen was entirely negative.

Under `reports/current_research_governance.md`, this is a permanent Stage B kill: the full method is clearly worse than a simple baseline, and useful improvement is excluded by paired evidence.

Do not rescue SCVC by retuning gain, bias, temporal blend, calibration identities, target stats, or by rebranding it as another visual normalization method.

Next action: continue to Epoch 3 Cycle 3 under `reports/current_research_governance.md`.
