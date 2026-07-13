# PSE-VLA Stage B Validation

Date: 2026-07-13 KST

Decision: `VALID_80_PAIRED_MANIFEST`

## Integrity Checks

- total rows: `400`
- unique `(variant, task, identity)` keys: `400`
- duplicate keys: `0`
- rows per variant: `80`
- all variants use identical task/reset manifest: `true`
- preserved 40-case result: `reports/pse_vla/stage_b_40_result.json`
- expanded 80-paired result: `reports/pse_vla/stage_b_result.json`

## Expanded Stage B Result

- `clean_frozen_smolvla`: `48 / 80`, task-balanced `0.6000`
- `bright_single`: `51 / 80`, task-balanced `0.6375`
- `dark_single`: `46 / 80`, task-balanced `0.5750`
- `pse_duplicate_clean`: `44 / 80`, task-balanced `0.5500`
- `pse_full`: `50 / 80`, task-balanced `0.6250`

Strongest baseline: `bright_single`.

Paired full minus `bright_single`:

- paired count: `80`
- wins: `6`
- losses: `7`
- ties: `67`
- delta: `-0.0125`
- bootstrap CI: `[-0.1000, 0.0750]`

The PSE mechanism was active: `pse_full` mean postprocessed action delta versus clean was `0.076743`.

## Validation Ruling

The expanded result is valid for final Stage B adjudication. The full method does not beat the strongest baseline, and the paired upper confidence bound excludes the preregistered useful `+0.10` prototype improvement.
