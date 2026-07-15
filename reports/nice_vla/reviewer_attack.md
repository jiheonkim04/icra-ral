# NICE-VLA Independent Reviewer B Attack

Date: 2026-07-15 KST

Proposal hash reviewed:
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

The proposal is not an exact duplication of the closest prior, but its claim
can collapse into a calibrated-threshold engineering change unless the
distributional model, calibration unit, prior fidelity, and closed-loop
comparison are fixed before implementation. No labeled latent extraction,
training, validation search, or confirmatory access is authorized yet.

## Independent Prior Attack

### Closest prior: VLA-Corrector

Primary sources:

- paper: https://arxiv.org/abs/2607.01804;
- project: https://zju-omniai.github.io/vla-corrector/;
- official code: https://github.com/ZJU-OmniAI/vla-corrector;
- inspected code commit: `9d23a0ba6fad562d3ed1a68fc52c8a12459abb41`.

VLA-Corrector already owns frozen-VLA latent-dynamics monitoring, a rolling
median-plus-MAD circuit breaker, action-queue truncation, recovery, and OGG.
It reports positive SmolVLA and PI0.5 results. NICE can claim only that an
action-conditioned predictive distribution and validation-frozen normalized
innovation improve the prior's monitor under otherwise matched execution.

### Historical campaign overlap

EAC and RCV already tested adaptive replanning or recovery scheduling in this
campaign and were not positive paper candidates. NICE is not a rescue of
either method only if it preserves VLA-Corrector's official mean/recovery path
and changes the monitor statistic through an explicit conditional covariance
model. Fixed short-horizon replanning must remain the simple killer. Any
post-result threshold change closes NICE as an invalid rescue attempt.

### Related uncertainty and calibration concepts

Heteroscedastic Gaussian residual models, Mahalanobis innovation, and split
conformal calibration are individually standard. Provisional novelty lies in
their action-conditioned integration into a frozen VLA latent-dynamics circuit
breaker and the matched evidence, not in any one mathematical ingredient.

## Major Attacks

### 1. The Method Could Be Only A Threshold Swap

If NICE learns no meaningful conditional covariance, its score is a monotone
rescaling of VLA-Corrector's error followed by a different threshold. The key
ablation must use the same frozen mean and recovery path with a global scalar
error threshold. NICE must beat that ablation and the official-prior proxy;
Base improvement alone is insufficient.

### 2. Prior Fidelity Is Not Yet Executable

The proposal cites official code but does not freeze exact latent hook,
preprocessing, action normalization, horizon `k`, mean architecture, warmup,
persistence, cooldown, queue truncation, or OGG budget. The local arm must be
labeled a transparent official-code-derived proxy unless exact equivalence is
demonstrated. Stage 0A must persist source commit and a line-level mechanism
map before any model fit.

### 3. Frame-Level Conformal Calibration Would Leak Correlation

Frames from one episode and task are highly correlated. Treating them as IID
calibration samples would make nominal coverage misleading and let long tasks
dominate. Calibration must use episode-cluster scores with equal task weight,
fixed episode quotas, and a finite-sample split-conformal quantile. No frame
from a calibration episode may train either mean or covariance.

### 4. The Diagnostic Mismatches May Be Artificially Easy

Temporal offsets, cross-episode futures, and action-regime swaps can be
detected without identifying real closed-loop correction opportunities. Their
AUROC is a development mechanism check only. It cannot establish task-success
headroom, cannot select confirmatory identities, and cannot appear as the
paper's central performance result.

### 5. High-Dimensional Gaussian NLL Is Numerically Fragile

The flattened residual dimension is `L*D`, potentially tens of thousands.
Unconstrained diagonal scales can collapse or explode; a learned low-rank term
can be singular or absorb mean error. The audit must freeze floors, ceilings,
parameterization, rank, basis construction, log-determinant algebra, jitter,
and finite checks. The mean must be frozen before covariance fitting.

### 6. Gaussian Likelihood May Be Misspecified

Latent residuals can be multimodal, heavy-tailed, and token-correlated. A
Gaussian NLL is acceptable as a scoring model only, not a calibrated density
claim. Report empirical coverage and rank statistics. If the score provides no
gain over cosine error, classify `DESIGN_FAILURE`; do not add a mixture model,
Student-t likelihood, or new distance within this cycle.

### 7. Action-Regime Conditioning Can Encode Unstable Heuristics

Gripper sign changes near zero and raw norm scales can create brittle phase
labels. The conditioning vector must be continuous except for one discovery-
frozen gripper transition with a fixed deadband. No task outcome, reset
identity, simulator state, future action, or queue-success label may enter.

### 8. Calibration Coverage Alone Does Not Prove Safe Integration

The monitor can interrupt too often while retaining nominal residual coverage.
Require interrupt frequency, queue position, persistence, recovery frequency,
translation/rotation/gripper deltas, clean validation success, action bounds,
and exact monitor-disabled Base identity. A globally active circuit breaker is
a failure even if AUROC is high.

### 9. OGG And Recovery Can Confound The Claimed Gain

The official prior's OGG and recovery behavior can create gains independently
of normalized innovation. Prior, Ours, and mean-only ablation must share the
same mean checkpoint, OGG budget, recovery policy, queue semantics, and action
postprocessing. Log trigger-to-recovery transitions so the monitor effect is
separable from the recovery controller.

### 10. The Validation Utility Needs Exact Semantics

The displayed equation omits `0.45*` before `success_proxy`, while the next
sentence assigns that weight. Freeze the intended formula explicitly. The
positive weights sum to one and overhead is an additional penalty, so this is
a utility in `[-0.05,1]`, not a normalized probability. Define every term and
tie-break before search.

### 11. Search And Data Budgets Are Large Enough To Drift

Six configurations times two seeds can become twelve effective selection
attempts. Seeds may measure stability but may not be cherry-picked. Select by
the mean preregistered utility across both seeds, preserve every trial, and
allow no threshold, horizon, architecture, task, or reset substitution.

The local LIBERO corpus is about 100 GB. Freeze a bounded extraction manifest
and a compute ceiling before reading it. Stage 0A must remain tiny and cannot
silently become the full VLA-Corrector reproduction.

### 12. Headroom Must Be Closed-Loop And Prior-Relative

Diagnostic mismatch detection is not headroom. Before confirmatory testing,
validation-only paired rollouts must show Base failure, residual failure after
the VLA-Corrector proxy, and a useful diagnostic upper bound. If fixed
short-horizon replanning matches the attainable gain, classify `NO_HEADROOM`
or simple-baseline dominance as preregistered.

### 13. Partitions Must Be Enforced By Code

Task strings, demonstration indices, reset identities, and extracted pair
keys need one manifest validator. Confirmatory tasks must remain unread before
the selected configuration, policy list, thresholds, and checkpoint hashes
are frozen. Duplicate `(suite,task,demo,frame,k)` and
`(policy,task,reset_identity)` keys are hard failures.

### 14. Stage 0A Repair Scope Must Stay Mechanical

The one allowed repair may fix only code, schema, shape, path, or
serialization. It may not change loss, covariance families, `k`, data source,
hook, sample count, action ranges, gates, or pass thresholds. All failed
attempts and partial rows must remain durable.

### 15. Efficiency Claims Are Currently Quarantined

The recorded gaming and Windows Efficiency Mode intervals contaminate timing
and utilization evidence when overlap is positive or unknown. Such evidence
cannot select a configuration or support a paper claim. Synchronous success
rows survive only after timeout, exception, semantics, identity, duplicate,
and manifest checks.

## Required Rebuttal Commitments

Researcher A must bind the method to all of the following:

1. official-code-derived transparent prior labeling and exact source commit;
2. episode-cluster, task-balanced split-conformal calibration;
3. frozen mean before covariance fit;
4. diagonal and diagonal-plus-rank-8 as the only covariance choices;
5. explicit numerical floors, ceilings, Woodbury algebra, and finite gates;
6. diagnostic mismatches treated only as development smoke;
7. exactly six configurations selected by the two-seed mean utility;
8. corrected utility equation and deterministic tie-breaks;
9. matched OGG, recovery, inference budget, and action postprocessing;
10. fixed-short-horizon headroom and simple-killer comparison;
11. code-enforced partitions, manifests, duplicate checks, and zero test use;
12. no latency or resource evidence from quarantined intervals.

## Reviewer B Decision

`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

The mechanism is sufficiently distinct to continue only under the listed
constraints. Failure to accept any item rejects NICE before implementation.
