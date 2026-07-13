# PSE-VLA Preregistration

Date: 2026-07-12 KST

Decision: `IMPLEMENTATION_PREREGISTERED`

## Fixed Transform Bank

Use exactly these transforms on preprocessed SmolVLA image tensors:

- `identity`: `x`
- `bright_low_contrast`: `clip(0.42 * x + 0.28, 0, 1)`
- `dark_high_contrast`: `clip(1.25 * x - 0.10, 0, 1)`

No Stage A or Stage B result may change these values.

## Stage A

Held-out identities:

- `20260741..20260745`

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Policies:

1. `clean_frozen_smolvla`
2. `bright_single`
3. `dark_single`
4. `pse_duplicate_clean`
5. `pse_full`

Episode count:

- `50` total episodes.

Primary metric:

- task-balanced closed-loop success.

Mechanism metrics:

- mean postprocessed action delta of `pse_full` versus clean;
- mean postprocessed action delta of `pse_full` versus bright and dark;
- duplicate-clean delta versus clean;
- transform count per step.

## Stage A Decision Rules

Permanent Stage A kill follows `reports/current_research_governance.md`.

Additionally:

- if `pse_duplicate_clean` is exactly equivalent to `pse_full`, the aggregation mechanism is not useful;
- if either single transform exactly explains `pse_full`, the ensemble is trivial;
- otherwise, single-transform or duplicate-clean ties in Stage A are diagnostic and the method proceeds to Stage B unless a current-governance Stage A kill fires.

## Stage B

If required, initial Stage B uses identities:

- `20260741..20260760`

This gives `40` paired episodes per policy and `200` total episodes.

The first five identities are the Stage A identities. Stage B is treated as the expanded predeclared paired set, not as a separate fresh-test claim.

## Stage B Expansion

The initial `40` paired episodes per policy finished unresolved on 2026-07-13 KST:

- `clean_frozen_smolvla`: `28 / 40`
- `bright_single`: `27 / 40`
- `dark_single`: `26 / 40`
- `pse_duplicate_clean`: `26 / 40`
- `pse_full`: `27 / 40`
- paired full minus clean CI: `[-0.20, 0.125]`

Per `reports/current_research_governance.md`, run exactly one expansion to `80` paired episodes per policy using the full contiguous exact-init range:

- `20260721..20260760`

This adds identities `20260721..20260740` for every variant. They are not selected from PSE outcomes. No third expansion is allowed.

Stage B reports paired full-minus-baseline wins/losses/ties and a paired bootstrap confidence interval against every non-full variant.
