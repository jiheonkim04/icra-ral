# NICE-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Proposal hash:
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.

Reviewer decision answered:
`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

Decision: `NICE_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

Researcher A accepts every binding condition. The frozen proposal is not
edited; this rebuttal narrows and operationalizes it.

## Narrowed Contribution

NICE does not claim invention of heteroscedastic regression, Mahalanobis
innovation, conformal calibration, latent-dynamics monitoring, queue
truncation, or recovery. It tests one prior extension:

> under a matched VLA-Corrector execution path, does an action-conditioned
> residual covariance and validation-frozen normalized-innovation trigger
> improve interrupt timing and closed-loop success over the prior's global
> cosine-error monitor?

Parity with the prior, the mean-only ablation, or fixed short-horizon
replanning kills the contribution under the later frozen rule.

## Prior Fidelity

The local comparison is named `vla_corrector_official_proxy`. It remains a
transparent official-code-derived proxy unless exact checkpoint and execution
equivalence are demonstrated. The source commit is
`9d23a0ba6fad562d3ed1a68fc52c8a12459abb41`.

Stage 0A will persist a mechanism map for the official latent hook, pair
construction, current-action conditioning, mean target, `k=10`, action
normalization, rolling threshold, queue truncation, recovery, and OGG. NICE,
prior, and ablation share mean checkpoint, queue semantics, recovery policy,
OGG budget, postprocessing, and decode budget.

## Episode-Balanced Calibration

Calibration episodes are disjoint from every training episode. For each
calibration episode, compute the fixed within-episode upper quantile of valid
natural-pair scores. Each task contributes the same number of episode scores.
The conformal multiset is therefore episode-level and task-balanced, never a
frame-level pool. The finite-sample `ceil((m+1)*(1-alpha))` order statistic is
used, clipped to `m`.

No task may contribute more than 25% of training, calibration, diagnostic, or
validation pairs. Duplicate pair keys and split overlap are hard failures.

## Fixed Conditioning

The conditioning vector uses normalized current and previous 7D actions,
translation norm, rotation norm, gripper magnitude, and one gripper-transition
indicator. Its deadband is the discovery-only median absolute nonzero gripper
change, computed once from the frozen extraction manifest and persisted before
validation. There is no searched deadband.

Task outcomes, reset identities, simulator state, reward, done, future
observations, and future actions are prohibited inference inputs.

## Covariance And Numerical Commitments

The mean model is fit and frozen first. The only covariance choices are:

1. diagonal;
2. diagonal plus rank-8 with a discovery-only PCA basis.

Scale floors, ceilings, jitter, rank, basis, inverses, log determinants, and
finite checks are frozen in the mathematical audit. A Gaussian NLL is only a
proper scoring objective for the residual scale model; no Gaussian truth claim
is made. Empirical coverage and rank diagnostics are required.

No Student-t, mixture, kernel, learned basis, new rank, or alternate distance
may rescue this cycle.

## Diagnostic Status

Temporal-offset, cross-episode, and action-regime-swap mismatches are synthetic
or constructed development diagnostics. They can establish that the monitor
acts and can stop the method for nonobservability. They cannot establish
closed-loop headroom, select test identities, or serve as central paper
evidence.

## Frozen Search Utility

The proposal's intended equation is clarified as:

`S_val = 0.45*success_proxy + 0.20*clean_retention
         + 0.15*interrupt_F1 + 0.10*coverage_score
         + 0.10*action_validity - 0.05*normalized_overhead`.

The positive terms sum to one; overhead is an additional penalty, so this is
a utility with range `[-0.05,1]`, not a probability. Each configuration is
scored by the arithmetic mean over both fixed lightweight seeds. A seed is
never selected or discarded.

The search remains exactly six configurations: two covariance families crossed
with coverage `{0.90,0.95,0.975}`. There is no threshold, `k`, horizon,
persistence, cooldown, recovery, OGG, mean, task, or reset variant.

## Safety And Headroom

Before confirmatory access, validation-only paired rollouts must report:

- Base, prior, Ours, ablation, and fixed-short success;
- prior-relative residual headroom;
- clean retention;
- interrupt and recovery frequency by queue position;
- translation, rotation, and gripper deltas;
- action-bound validity and exceptions;
- exact monitor-disabled Base identity;
- one diagnostic upper bound labeled oracle-only.

No headroom after the prior, or parity with fixed short-horizon replanning,
stops the method before confirmatory testing under the frozen adjudication.

## Data And Compute Bound

Stage 0A is capped at two discovery tasks, two demonstrations per task, and 32
sampled frames per demonstration. It may read at most 128 current-frame records
and construct at most 128 valid `k=10` pairs. It may run at most 20 tiny mean
optimizer steps and 20 tiny covariance optimizer steps. It reads zero
validation or confirmatory records.

Full development extraction and training require a separate Stage 0B pass and
durable manifest. The 100 GB corpus is never scanned opportunistically during
configuration selection.

## Repair, Runtime, And Evidence Integrity

One Stage 0A repair is permitted only for code, schema, shape, path, or
serialization. It cannot change scientific choices or gates. Failed attempts,
partial JSON, logs, and exit codes remain preserved.

Detached workers use PID, heartbeat, status, atomic partial result, final
result, logs, and exit code. Resume executes missing manifest keys only.
Duplicate completed rows are forbidden.

Timing, throughput, and utilization overlapping or ambiguously overlapping a
recorded resource-contention interval are quarantined and cannot select a
configuration or support the paper. Valid synchronous task-success rows require
zero timeout/exception and unchanged action/task/reset semantics.

## Transition

Proceed to the mathematical audit, preregistration, and prototype protocol.
No labeled extraction, model training beyond the tiny Stage 0A interface
smoke, rollout, validation search, or confirmatory access is yet authorized.
