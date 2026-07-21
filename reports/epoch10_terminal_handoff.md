# Epoch 10 terminal handoff

Terminal state: `EPOCH10_ICAE_INTERVENTION_ASSAY_INVALID`

Branch: `codex/epoch10-icae-vla-evaluation`

Epoch 10 reached a named scoped scientific terminal during Phase C. The checkpoint panel and four-suite task substrate passed, but none of the frozen horizons passed exact restore/sham equivalence or perturbation monotonicity after the one allowed repair. The decisive source is `reports/epoch10_icae_mechanics_calibration.json`; the compact gate decision is `reports/epoch10_icae_assay_adjudication.json`.

No new checkpoint simulator outcome or closed-loop success label was opened. Development Stage 0, official holdout, one-shot confirmation, and the positive paper package are intentionally absent. Running them now would cross a failed mandatory gate and violate the prompt.

The negative is localized: exact HDF5 physics vectors restore perfectly, but the controller retains internal state that is not captured by the flattened MuJoCo vector. An independently designed future campaign could test fresh-controller-per-branch execution or explicit controller-state serialization. That is outside this terminal campaign because it would require a second assay repair.

The 12 prospective checkpoint bundles remain at `C:\assets\checkpoints\epoch10_icae_panel\rank4`. They are valid saved artifacts, but their scientific simulator outcomes remain sealed. Pre-existing untracked rollout directories remain outside the Epoch 10 commits.
