# PSE-VLA Stage B Expansion Preregistration

Date: 2026-07-13 KST

Decision: `STAGE_B_UNRESOLVED_EXPAND_TO_80_REQUIRED`

## Initial Stage B 40-Paired Result

- `clean_frozen_smolvla`: `28 / 40`
- `bright_single`: `27 / 40`
- `dark_single`: `26 / 40`
- `pse_duplicate_clean`: `26 / 40`
- `pse_full`: `27 / 40`
- paired full minus clean: wins `5`, losses `6`, ties `29`, delta `-0.025`, CI `[-0.20, 0.125]`

The result is not `PROTOTYPE_GO`. It is also not a permanent kill because the paired upper confidence bound against the strongest baseline does not exclude a useful improvement by the active Stage B rule.

## Expansion

Run the one allowed expansion to `80` paired episodes per policy:

- identities: `20260721..20260760`
- variants: unchanged
- tasks: `libero_spatial/task_4`, `libero_10/task_4`
- total expanded episodes: `400`

The already completed `20260741..20260760` episodes remain part of the expanded paired set. The new episodes are exactly `20260721..20260740` for every variant.

## Decision Rule

After expansion, no third expansion is allowed.

Use `reports/current_research_governance.md`:

- `PROTOTYPE_GO` only if full PSE beats the strongest baseline and ablation with at least 10 absolute points, or paired evidence is consistently positive with meaningful failure-rate reduction;
- permanent kill if full is clearly worse, the paired upper confidence bound excludes a useful improvement, or a baseline/ablation explains the method;
- otherwise archive as unresolved after the maximum expansion.
