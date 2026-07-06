# ExecSpec STATE 2 Calibrated Repair

This is bounded diagnostic evidence only. It is not benchmark success or paper-grade evidence.

- decision: `continue`
- calibration demos: `5`
- eval demos: `1`
- calibration action samples: `1403`
- eval action samples: `272`
- eval leakage detected: `False`
- task count: `6`
- mismatch types: `gripper_sign_flip, translation_scale_mismatch, rotation_scale_mismatch, global_action_scale_mismatch, per_dimension_scale_mismatch, gripper_threshold_0_1_mismatch, range_clipping_mismatch`
- best repair method: `diagonal_affine_calibration`
- full beats identity: `True`
- full beats clipping-only: `True`
- full beats global affine: `True`
- full mean recovery fraction: `1.0`
- mean action L2 identity/clipping/global/full: `0.565447642` / `0.565447642` / `0.308794194` / `0.0`
- full repair per-mismatch beat counts identity/clipping/global: `7/7` / `7/7` / `5/7`
- replay/rollout happened: `True`
- replay improved reward/success: `True`
- next state: `STATE 3 replay/rollout validation`

## Held-Out Action Metrics

| mismatch | generated as | plausible source | wrong L2 | wrong T/R/G | wrong clip/valid | full L2 | full T/R/G | full clip/valid | recovery | full beats id/clip/global | replay degradation before repair |
| --- | --- | --- | ---: | --- | --- | ---: | --- | --- | ---: | --- | --- |
| gripper_sign_flip | Closed/open gripper sign convention is inverted. | common binary gripper open/close convention mismatch | 2.0 | 0.0/0.0/1.0 | 0.0/1.0 | 0.0 | 0.0/0.0/0.0 | 0.0/1.0 | 1.0 | true/true/true | true; expert 1.0/true, wrong 0.0/false |
| translation_scale_mismatch | Cartesian translation dimensions are over-scaled before the controller. | policy/controller unit or normalization scale mismatch | 0.36397 | 0.36397/0.0/0.0 | 0.334559/0.665441 | 0.0 | 0.0/0.0/0.0 | 0.0/1.0 | 1.0 | true/true/true | true; expert 1.0/true, wrong 0.0/false |
| rotation_scale_mismatch | Rotation dimensions are under-scaled before the controller. | axis-angle / delta-rotation normalization mismatch | 0.064211 | 0.0/0.064211/0.0 | 0.0/1.0 | 0.0 | 0.0/0.0/0.0 | 0.0/1.0 | 1.0 | true/true/true | not replayed in STATE 2 |
| global_action_scale_mismatch | All action dimensions use an incorrect global unnormalization scale. | incorrect action unnormalizer scale | 0.223911 | 0.211687/0.042808/0.0 | 1.0/0.0 | 0.0 | 0.0/0.0/0.0 | 0.0/1.0 | 1.0 | true/true/false | not replayed in STATE 2 |
| per_dimension_scale_mismatch | Each action dimension uses a different incorrect metadata scale. | stale per-dimension action statistics | 0.190567 | 0.18382/0.028214/0.0 | 0.0625/0.9375 | 0.0 | 0.0/0.0/0.0 | 0.0/1.0 | 1.0 | true/true/true | not replayed in STATE 2 |
| gripper_threshold_0_1_mismatch | A 0/1 gripper convention is sent to a -1/1 controller interface. | binary gripper threshold exported with wrong range | 0.610294 | 0.0/0.0/0.610294 | 0.0/1.0 | 0.0 | 0.0/0.0/0.0 | 0.0/1.0 | 1.0 | true/true/true | not replayed in STATE 2 |
| range_clipping_mismatch | Action range metadata is too wide, causing controller clipping. | controller range mismatch or missing action validity certificate | 0.50518 | 0.465885/0.128423/0.0 | 1.0/0.0 | 0.0 | 0.0/0.0/0.0 | 0.0/1.0 | 1.0 | true/true/false | not replayed in STATE 2 |

## Exact-Init Replay Metrics

### gripper_sign_flip

- degradation recovered: `True`
- total simulator steps performed: `2121`
- HDF5 first reward/done/signal index: `271` / `271` / `271`

| variant | reward | success | done index | trajectory length | valid | clip | action L2 | gripper error | T/R drift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| correct_7d_expert_action_replay | 1.0 | true | 260 | 261 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0/0.0 |
| wrong_executable_spec_replay | 0.0 | false | n/a | 272 | 1.0 | 0.0 | 2.0 | 1.0 | 0.0/0.0 |
| clipping_only | 0.0 | false | n/a | 272 | 1.0 | 0.0 | 2.0 | 1.0 | 0.0/0.0 |
| global_affine_calibration | 0.0 | false | n/a | 272 | 1.0 | 0.0 | 0.884073 | 0.0 | 0.660036/0.149621 |
| diagonal_affine_calibration | 1.0 | true | 260 | 261 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0/0.0 |
| gripper_only_calibration | 1.0 | true | 260 | 261 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0/0.0 |
| split_trg_calibration | 1.0 | true | 260 | 261 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0/0.0 |
| full_execspec_repair | 1.0 | true | 260 | 261 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0/0.0 |

### translation_scale_mismatch

- degradation recovered: `True`
- total simulator steps performed: `2132`
- HDF5 first reward/done/signal index: `271` / `271` / `271`

| variant | reward | success | done index | trajectory length | valid | clip | action L2 | gripper error | T/R drift |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| correct_7d_expert_action_replay | 1.0 | true | 260 | 261 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0/0.0 |
| wrong_executable_spec_replay | 0.0 | false | n/a | 272 | 0.665441 | 0.334559 | 0.36397 | 0.0 | 0.36397/0.0 |
| clipping_only | 0.0 | false | n/a | 272 | 1.0 | 0.0 | 0.36397 | 0.0 | 0.36397/0.0 |
| global_affine_calibration | 0.0 | false | n/a | 272 | 1.0 | 0.0 | 0.350258 | 0.0 | 0.196605/0.040743 |
| diagonal_affine_calibration | 1.0 | true | 260 | 261 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0/0.0 |
| gripper_only_calibration | 0.0 | false | n/a | 272 | 1.0 | 0.0 | 0.36397 | 0.0 | 0.36397/0.0 |
| split_trg_calibration | 1.0 | true | 260 | 261 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0/0.0 |
| full_execspec_repair | 1.0 | true | 260 | 261 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0/0.0 |
