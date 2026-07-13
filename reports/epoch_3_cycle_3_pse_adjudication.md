# Epoch 3 Cycle 3 PSE-VLA Adjudication

Date: 2026-07-13 KST

Decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

## Method

`PSE-VLA`, Photometric Sensor-Ensemble VLA, averaged frozen SmolVLA first action-chunk predictions over a fixed transform bank:

- `identity`
- `bright_low_contrast`
- `dark_high_contrast`

The proposal hash was `3F15D6E3ADCF340C490FBD5656051DFD101136D592F5A6B5D773ABF0E5308CAD`.

## Execution Boundary

- training happened: `false`
- teacher data used: `false`
- privileged inference used: `false`
- synthetic mechanism smoke: passed
- Stage A: `50 / 50` episodes, zero exceptions
- Stage B 40-paired result: completed and preserved
- expanded Stage B: `400 / 400` episodes, zero exceptions, `80` paired episodes per variant

## Stage A

Stage A result: `STAGE_A_NON_GO_TO_STAGE_B_REQUIRED`.

- `clean_frozen_smolvla`: `7 / 10`
- `bright_single`: `8 / 10`
- `dark_single`: `6 / 10`
- `pse_duplicate_clean`: `6 / 10`
- `pse_full`: `7 / 10`

The method was not eligible for a permanent Stage A kill under current governance, so Stage B was required.

## Stage B 40-Paired Result

The first Stage B result was unresolved and was archived separately:

- `clean_frozen_smolvla`: `28 / 40`
- `bright_single`: `27 / 40`
- `dark_single`: `26 / 40`
- `pse_duplicate_clean`: `26 / 40`
- `pse_full`: `27 / 40`
- paired full minus clean CI: `[-0.2000, 0.1250]`

Current governance allowed exactly one expansion to at most `80` paired episodes per key policy.

## Expanded Stage B

- `clean_frozen_smolvla`: `48 / 80`, task-balanced `0.6000`
- `bright_single`: `51 / 80`, task-balanced `0.6375`
- `dark_single`: `46 / 80`, task-balanced `0.5750`
- `pse_duplicate_clean`: `44 / 80`, task-balanced `0.5500`
- `pse_full`: `50 / 80`, task-balanced `0.6250`

Strongest baseline: `bright_single`.

Paired full minus `bright_single`:

- wins: `6`
- losses: `7`
- ties: `67`
- delta: `-0.0125`
- bootstrap CI: `[-0.1000, 0.0750]`

## Ruling

PSE-VLA is a valid current-formulation kill.

The mechanism acted, but the full method did not beat the strongest baseline. The paired upper confidence bound versus `bright_single` was `0.075`, excluding the preregistered useful `+0.10` prototype improvement after the maximum allowed expansion. No further PSE internal controls or expansions are allowed.

Next action: preserve this result, apply the post-PSE research-design governance requested by the user, and begin Epoch 4 with a problem-first, external-prior-early candidate search.
