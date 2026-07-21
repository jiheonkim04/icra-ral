# Epoch 10 ICAE-VLA result

Terminal state: `EPOCH10_ICAE_INTERVENTION_ASSAY_INVALID`

The prospective checkpoint and task-coverage prerequisites passed, but the exact-state intervention assay did not. Twelve real LoRA checkpoints were trained from four whole-seed lineages at three predeclared stages, all four LIBERO suites were executable, and all 128 raw demonstration physics states restored with zero state-vector error after the single wrapper-access repair.

The decisive mechanics gate then failed at every candidate horizon:

| Continuation H | Nominal/sham equivalent | Valid nominal rows | Max nominal-twin state L2 | Medium − small pooled state deterioration, grouped 95% interval | Gate |
|---:|---:|---:|---:|---:|---|
| 4 | 46.875% | 100.000% | 3.440729 | 0.000353 [−0.006605, 0.005771] | fail |
| 8 | 81.250% | 96.875% | 0.109095 | 0.003396 [−0.006226, 0.012234] | fail |
| 16 | 93.750% | 90.625% | 1.447421 | 0.001301 [−0.006566, 0.008819] | fail |

The required nominal/sham rate was at least 95%, the branch-order materiality tolerance was `1e-6`, and perturbation monotonicity had to hold with grouped uncertainty in the expected direction. None passed.

The failure is not a raw-state restore failure: the repaired preflight restored 128/128 HDF5 vectors exactly. The missing state is controller-internal. Reusing an environment and setting its flattened MuJoCo state does not reset the controller's retained goal/integrator state, so two nominal branches can diverge depending on which branch ran before them. Fresh-controller-per-branch execution or explicit controller-state restoration is a plausible future assay design, but it would be a second mechanics repair. The frozen protocol authorized only one, already consumed by the complete wrapper-access rerun.

No prospective checkpoint simulator outcome, development closed-loop label, or official held-out outcome was opened. Checkpoint ranking and Stage 0 were therefore not run, and no positive paper package was produced. This is a scoped negative for ICAE as executed, not a claim that short-intervention VLA evaluation is impossible in general.
