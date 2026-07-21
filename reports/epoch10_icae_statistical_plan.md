# Epoch 10 ICAE statistical plan

Status: frozen before checkpoint comparison or prospective closed-loop labels.

## Units and aggregation

The intervention row is the paired state-level unit. Rows are grouped by raw demonstration episode and task. Checkpoint uncertainty is clustered by whole training-seed lineage because the three snapshots from one seed are correlated repeated measures. Task scores are row means; overall scores are equal-task macro-averages. Lower ICAE and action-error baselines are predicted to associate with higher closed-loop success.

## Mechanics and assay gates

- Select the shortest `H` in `{4, 8, 16}` with at least 95% nominal-twin equivalence, at least 90% valid nominal rows, and pooled medium-perturbation state deterioration no smaller than small-perturbation deterioration.
- Treat a branch/cache duplicate as material when its absolute scalar difference exceeds `1e-6` or an action component differs by more than `1e-6`.
- Require Spearman rank repeatability of at least `0.8` between the two predeclared interleaved mechanics subsets for the six development checkpoints.
- Require all 12 adapters to pass finite 7-D action, official normalization/postprocessing, disk-load, and safety-limited scene checks.
- Require the six held-out checkpoint identities to have no unexplained raw out-of-support action-unit rate above 15%.

The sole allowed repair has already been consumed by the uniform environment-wrapper accessor fix. Any remaining assay-gate failure terminates ICAE as `EPOCH10_ICAE_INTERVENTION_ASSAY_INVALID`.

## Stage 0 development

The development checkpoint panel is all three snapshots from seeds `101` and `202` (six identities). The intervention panel is 12 states per task across four tasks. Closed-loop labels use 15 common official initial-state indices (`20`–`34`) per task and checkpoint, with standard LIBERO success semantics.

Before opening those labels, freeze two ICAE variants: signed raw deterioration as primary and a bounded score `clip(d, -2, 2)` as secondary. Baselines are raw MSE, MAE, action-dimension-normalized MSE, validation loss when well-defined, arm/gripper-separated MSE with fixed equal arm/gripper macro-weighting, state-only criticality times isotropic action magnitude, unpaired ICAE, and state-shuffled ICAE. CI-MSE is included only if its official implementation is faithfully compatible; otherwise the incompatibility is documented and no proxy is substituted.

Report Spearman and Kendall association, within-task pairwise concordance with success, best-checkpoint selection regret, and top-half selection precision. Success ties use exact equality; score ties within `1e-8` receive half credit in concordance. Bootstrap intervals resample whole checkpoint lineages and whole demonstration episodes, stratified by task, with 10,000 deterministic replicates.

Proceed to held-out evaluation only if success spans at least 20 absolute percentage points or forms three distinguishable bands, ICAE has the expected negative association, within-task concordance is at least 0.60, ICAE point concordance exceeds raw normalized MSE, and ICAE is not dominated on every endpoint by the strongest faithfully implemented equal-input baseline. Tied outcomes terminate as `EPOCH10_ICAE_DEVELOPMENT_PANEL_NONIDENTIFIABLE`; distinguishable outcomes without ICAE headroom after the already-consumed repair terminate as `EPOCH10_ICAE_STAGE0_NO_PREDICTIVE_HEADROOM`.

## Official holdout and confirmation

Held-out checkpoint actions and intervention rows are completed and hashed before success labels for initial-state indices `35`–`49` are opened. Confirmatory inference uses the prospectively frozen metrics, exclusions, tie handling, tasks, seeds, and checkpoint subset. Lineage-clustered uncertainty and per-suite results accompany every pooled result. No favorable checkpoint, state, task, or metric substitution is allowed after outcome access.
