# Target-Prior TCA-Map Kill Summary

## Original Hypothesis

Target-conditioned action decoding, fixed target priors, and Distributional TCA-Select could provide a low-compute path to counterfactual robustness for VLA manipulation.

## Strongest Positive Evidence

- Fixed-prior TCA had strong offline proxy evidence.
- Prior-source audit passed and found no inference-time leakage.
- TCA beat ActionMap in several offline and 7D diagnostic comparisons.
- The LIBERO 7D bridge and expert replay sanity checks were validated.

## Decisive Negative Evidence

- Online 7D action-quality gate failed.
- Best redesigned head: `small_cpu_mlp_fixed_prior_tca_7d`.
- Mean-action baseline eval 7D L2: `0.57299313`.
- Best redesigned eval 7D L2: `0.669078005`.
- The best online 7D TCA head did not beat the mean-action baseline.
- Valid rollout-level support was not established.
- TCA-Select had no measurable headroom.

## Kill Criterion Triggered

The method action source failed the pre-rollout online action-quality gate. Offline fixed-prior gains were not enough to justify RA-L-stable submission.

## Reusable Artifacts

- LIBERO counterfactual split and offline proxy evaluation tools.
- Target-prior source audit.
- ActionMap/TCA comparison scripts.
- 7D action bridge and adapter checks.
- Expert replay sanity and online action diagnostics.
- Research-integrity logging discipline.

## Why It Should Not Continue As RA-L-Stable

The route depends on offline proxy advantages without a validated online action source. Continuing would risk converting a diagnostic result into an unsupported robotics-control claim.

