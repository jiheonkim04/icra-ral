# NICE-VLA Researcher A Proposal

Date: 2026-07-15 KST

Decision: `NICE_RESEARCHER_PROPOSAL_READY_FOR_INDEPENDENT_REVIEW`

Method: `NICE-VLA`, Normalized-Innovation Corrective Execution for VLAs.

Contribution type: `PRIOR_EXTENSION`.

## Claim

A frozen action-chunked VLA should interrupt an executing chunk when observed
visual dynamics are unlikely under the action-conditioned predictive
distribution, not merely when a global cosine error exceeds a rolling robust
threshold.

NICE tests whether heteroscedastic normalized innovation improves the timing
of VLA-Corrector's existing truncation and recovery path while preserving
frozen SmolVLA behavior whenever no interrupt occurs.

The paper claim, if supported, is narrow:

> Action-conditioned predictive uncertainty and validation-frozen innovation
> calibration improve event-triggered corrective execution over the closest
> mean-only latent-drift monitor on matched flow-based VLAs.

NICE does not claim generic uncertainty estimation, generic conformal safety,
a new LoRA method, a new VLA backbone, or generic adaptive action chunking.

## Positive Prior Anchor

Closest prior: VLA-Corrector,
https://arxiv.org/abs/2607.01804.

Official code: https://github.com/ZJU-OmniAI/vla-corrector, inspected at commit
`9d23a0ba6fad562d3ed1a68fc52c8a12459abb41` under Apache-2.0.

The prior freezes the VLA, predicts short-horizon visual latent residuals,
detects persistent prediction/observation mismatch, truncates the stale queue,
and applies Online Gradient Guidance to the next recovery query.

Reported positive results include:

- MetaWorld SmolVLA `61.90 -> 66.65`, `+4.75` points;
- MetaWorld PI0.5 `48.70 -> 64.35`, `+15.65` points;
- LIBERO PI0.5 few-shot `94.00 -> 97.80`, `+3.80` points;
- real AgileX PiPER `55.6 -> 73.3`, `+17.7` points.

The prior is therefore entered as policy 2 in the first serious comparison,
not deferred until after internal controls.

## Exact Technical Difference

The inspected prior circuit breaker uses:

`e_t = 1 - cosine(delta_z_pred, delta_z_real)`

and triggers when `e_t` exceeds an episode-local rolling
`median + k_MAD * MAD` threshold after warmup.

NICE preserves the prior's:

- frozen visual encoder;
- mean latent-dynamics target;
- action normalization and `k`-step pair construction;
- queue truncation semantics;
- persistence and cooldown semantics;
- OGG recovery objective and guidance budget;
- postprocessing and 7D action interface.

NICE changes only the monitor model and its calibration:

1. fit the matched prior mean predictor `mu_phi(z_t, a_t)`;
2. freeze `mu_phi`;
3. fit a covariance predictor from discovery residuals;
4. compute a normalized innovation score;
5. freeze one split-conformal threshold selected only on validation data;
6. feed the resulting interrupt event into the same truncation and OGG path.

This is not a retuned cosine threshold. The new technical object is an
input-conditioned predictive covariance and its induced innovation geometry.

## Falsifiable Mechanism Chain

Problem condition:

- frozen SmolVLA executes multiple actions before observing and replanning;
- normal visual dynamics have different residual scale and correlation under
  free motion, contact, gripper transition, transport, and release.

Intermediate failure mechanism:

- a mean-only global angular error does not represent expected residual scale;
- expected high-variance motion can cause false interrupts;
- a smaller residual in a low-variance direction can be a true drift event but
  remain below the global threshold.

Policy behavior:

- false interrupts cause avoidable queue discontinuity and extra recovery;
- false negatives continue stale actions after visual dynamics have diverged.

Closed-loop failure:

- unnecessary or missed correction near contact and alignment reduces task
  completion.

Proposed method:

- predict residual mean and covariance from frozen visual tokens, current 7D
  action, previous 7D action, and deterministic action-regime features;
- whiten observed residuals by predicted covariance;
- compare the normalized innovation to a validation-frozen conformal quantile;
- require the same persistence rule as the closest prior;
- invoke the same truncation and OGG recovery path.

Expected result:

- improved interrupt calibration and critical-phase precision/recall;
- fewer false interrupts at equal or better missed-drift rate;
- higher paired closed-loop success than Base, VLA-Corrector, the key
  mean-only ablation, and fixed short-horizon replanning;
- exact Base action behavior on non-triggered steps.

## Legal Inputs

Allowed at training and inference:

- frozen SmolVLA image tokens from the two official RGB observations;
- current normalized 7D action;
- previous normalized 7D action;
- action-regime features derived from those actions: translation norm,
  rotation norm, gripper magnitude, and gripper sign-change indicator;
- current queue index and queue length;
- task instruction only through the frozen policy's ordinary input path;
- episode-local monitor history required by the frozen persistence/cooldown
  rule.

Prohibited at inference:

- simulator object state or privileged `states` arrays;
- future images or future actions;
- reward, success, done, or reset identity as model input;
- confirmatory labels or outcomes;
- a human correction or manual phase label;
- target-task failure labels.

## Evidence Partitions

Task identities are frozen before extraction.

Discovery tasks:

- `libero_10/task_1`
- `libero_10/task_3`
- `libero_goal/task_1`
- `libero_goal/task_3`
- `libero_object/task_1`
- `libero_spatial/task_1`

Validation tasks:

- `libero_10/task_5`
- `libero_goal/task_5`
- `libero_object/task_3`
- `libero_spatial/task_3`

Confirmatory tasks:

- `libero_10/task_7`
- `libero_goal/task_7`
- `libero_object/task_5`
- `libero_spatial/task_5`

Offline development data:

- discovery task demonstrations `demo_0..demo_29` may train or diagnose;
- validation task demonstrations `demo_30..demo_39` may calibrate and select;
- all other demonstration/task combinations are unread during method
  selection;
- confirmatory tasks are not used for extraction, training, or calibration.

Rollout reset identities:

- discovery: `20262001..20262012`;
- validation: `20262021..20262032`;
- confirmatory Stage A/B: `20262041..20262050`;
- one unresolved-only expansion: `20262051..20262060`.

No result on confirmatory tasks or reset identities may change the model,
covariance architecture, coverage, threshold, persistence, cooldown, OGG,
policy list, task list, metric, or decision rule.

## Data Health

The local root `C:/assets/data/libero` contains 130 task HDF5 files. Audited
files contain 50 demonstrations, hundreds of frames per demonstration, two
`128 x 128` RGB streams, 7D actions, proprioception, reward/done records, and
episode boundaries.

Before training, the extractor must report:

- task and demo counts;
- valid within-episode `k`-step pair counts;
- duplicate frame/pair keys;
- split overlap;
- action finite fraction and component ranges;
- latent finite fraction, variance, and token shape;
- action-regime balance;
- censor frequency at episode tails;
- no all-zero residual or variance target;
- source file hashes and official-code commit.

Raw simulator states are ignored by the method and excluded from extracted
model inputs.

## Model

Let frozen visual tokens be `z_t in R^(L x D)`, with `L` and `D` measured and
persisted by Stage 0A rather than assumed. Let `a_t in R^7` be the normalized
executed action and let `k=10` frames match the closest-prior default.

Target residual:

`Delta z_t = z_(t+k) - z_t`.

Matched mean model:

`mu_phi(z_t, a_t) in R^(L x D)`.

NICE residual:

`r_t = vec(Delta z_t - mu_phi(z_t, a_t)) in R^n`, where `n=L*D`.

Action-regime condition:

`c_t = [a_t, a_(t-1), ||a_t[0:3]||_2, ||a_t[3:6]||_2,
         |a_t[6]|, 1[sign(a_t[6]) != sign(a_(t-1)[6])]]`.

Two allowed covariance architectures are:

1. diagonal:
   `Sigma_theta = diag(softplus(s_theta(z_t,c_t)) + sigma_min^2)`;
2. diagonal plus rank-8:
   `Sigma_theta = diag(d_theta) + B_8 diag(lambda_theta) B_8^T`,
   where `B_8` is a discovery-only PCA residual basis and all predicted scales
   are positive by softplus.

The mean checkpoint is frozen before covariance fitting. Covariance gradients
do not update SmolVLA or `mu_phi`.

Normalized innovation:

`q_t = r_t^T Sigma_theta^(-1) r_t / n`.

For the low-rank architecture, inversion and log determinant use the Woodbury
identity and matrix determinant lemma. No dense `n x n` covariance is formed.

The validation set freezes a split-conformal empirical quantile `tau_c` for
coverage `c`. NICE triggers only when the same persistence rule used by the
closest prior observes `q_t > tau_c`.

## Objective Engineering

Mean training uses the closest-prior cosine objective and reports MSE as a
diagnostic. The matched prior and Ours share the same frozen mean checkpoint.

The covariance objective is Gaussian residual negative log likelihood:

`L_cov = 0.5 * mean_batch((r^T Sigma^(-1) r + logdet(Sigma)) / n)`.

The constant `log(2*pi)` term is omitted because it has no gradient and is
identical across configurations.

Required pre-training batch audit:

- exact shapes and dtypes;
- mean cosine-loss magnitude;
- covariance NLL magnitude;
- mean and covariance gradient norms;
- finite fraction of scores and gradients;
- minimum/maximum diagonal scale;
- low-rank eigenvalue bounds when applicable;
- proof that Base and mean parameters receive no covariance gradient.

No KL is used. Deterministic 7D actions and SmolVLA flow vectors are not
treated as normalized action probability distributions.

## Stage 0A: Source And Interface Smoke

Stage 0A may read at most:

- two discovery tasks;
- two demonstrations per task;
- 32 sampled frames per demonstration after within-episode censoring;
- no validation or confirmatory task.

It must establish:

- official code commit and license provenance;
- exact HDF5-to-official-preprocessor mapping;
- frozen SmolVLA latent extraction and measured `(L,D)`;
- action dimension exactly 7 and `k=10` pair construction;
- deterministic disk reload of a tiny mean and covariance checkpoint;
- finite nonzero covariance gradients on a small real batch;
- diagonal and low-rank algebra against direct small-matrix references;
- covariance positive definiteness;
- split-conformal quantile correctness;
- exact Base queue/action passthrough with monitor disabled;
- no confirmatory read;
- no privileged input;
- duplicate pair count zero.

One bounded implementation repair is allowed only if Stage 0A exposes a code,
schema, shape, or serialization defect before any scientific gate is applied.
The repair may not change method, objective, tasks, splits, architectures,
coverage values, thresholds, or success criteria. All attempts are preserved.

Stage 0A failure is classified as `IMPLEMENTATION_OR_DATA_FAILURE`, not a
scientific kill, unless the implementation is valid and the required legal
signal is proven absent.

## Stage 0B: Development Headroom Audit

If Stage 0A passes, extract the frozen discovery/validation latent manifest and
fit the matched mean model plus a bounded covariance smoke.

Natural matched validation pairs are compared with three diagnostic mismatch
families constructed only from validation data:

1. same-episode temporal offset future latent;
2. same-task cross-episode future latent;
3. action-regime swap while preserving current latent.

These are diagnostic oracles, not deployment conditions or paper outcomes.

Required gates:

- mean predictor exceeds a zero-change and task-mean residual baseline;
- covariance scales are noncollapsed and finite;
- nominal empirical coverage is within `0.03` of requested coverage;
- NICE diagnostic mismatch AUROC exceeds `0.60` and exceeds matched prior
  cosine-error AUROC by at least `0.03` on the preregistered aggregate;
- no one task supplies more than `25%` of sampled pairs;
- trigger score remains nonconstant across all validation tasks;
- monitor-disabled Base actions remain bitwise identical;
- matched prior and Ours use the same mean checkpoint and recovery budget.

If Base and fixed short-horizon validation rollouts show no success headroom,
classify `NO_HEADROOM` and stop before confirmatory testing.

## Bounded Validation Search

Maximum six total configurations:

- covariance: diagonal, diagonal plus rank-8;
- coverage: `0.90`, `0.95`, `0.975`.

At most two lightweight training seeds per configuration:
`20262011`, `20262012`.

No other architecture, coefficient, history, `k`, persistence, cooldown, or
OGG variant is searched. Mean-dynamics training is shared.

One final configuration is selected by this frozen validation score:

`S_val = success_proxy + 0.20*clean_retention + 0.15*interrupt_F1
         + 0.10*coverage_score + 0.10*action_validity
         - 0.05*normalized_overhead`.

Each term is scaled to `[0,1]`; `success_proxy` receives weight `0.45`. If
legal validation closed-loop success is available, it replaces the proxy.
Ties are broken by higher clean retention, then lower trigger rate, then the
diagonal architecture, then higher coverage.

All configurations, seeds, checkpoints, and negative results are saved. The
selected checkpoint and all settings are frozen before confirmatory access.

## Mechanism Smoke Before Rollout

For Base, prior, Ours, and ablation report:

- current and predicted latent residual summary;
- cosine error;
- NICE innovation score and conformal threshold;
- predicted diagonal scale and low-rank scales;
- trigger decision, persistence state, and cooldown state;
- queue length before and after decision;
- Base action chunk, recovery action chunk, and component-wise delta;
- translation, rotation, gripper, and full 7D delta statistics;
- absolute action validity and Base-relative range validity;
- activation context and task/reset key.

Required pre-rollout gates:

- checkpoint persists and disk reloads exactly;
- all intended parameters receive finite nonzero gradients;
- mean and covariance validation objectives behave sensibly;
- Ours differs from prior and mean-only ablation on relevant validation states;
- monitor is not always on or always off;
- non-triggered Base action and queue behavior are exact;
- triggered recovery deltas are bounded and action-valid;
- no privileged inference input and no confirmatory identity read.

## First Serious Comparison

Exactly five policies:

1. `smolvla_base_fixed_horizon`
2. `vla_corrector_official_proxy`
3. `nice_full`
4. `nice_mean_only_global_error_ablation`
5. `fixed_short_horizon_replan`

Policy 2 uses the inspected official mean/cosine-MAD monitor and official
truncation/OGG path, adapted transparently to the same local frozen SmolVLA.
It is labeled an official-code local proxy, not claimed as an exact paper
reproduction without the authors' checkpoints and full training data.

Policy 4 keeps NICE training and integration but removes covariance
normalization and conformal calibration. It uses one validation-frozen global
residual threshold.

Policy 5 is the mandatory reviewer killer because RCV and EAC showed that a
fixed short horizon can explain an adaptive scheduler. It uses the same policy
call and action semantics but no learned monitor or OGG.

## Confirmatory Stages

All policies use one paired manifest.

Stage A:

- approximately 10 paired episodes per policy;
- detects catastrophic degradation, exact equivalence, no headroom, invalid
  mechanism, or clear prior/ablation dominance;
- small differences automatically advance;
- no hyperparameter or threshold changes are allowed.

Stage B:

- 40 paired episodes per key policy;
- paired wins/losses/ties;
- task-balanced success;
- paired bootstrap confidence interval;
- effect size and failure-rate reduction;
- per-task breakdown;
- trigger precision/recall proxy and critical-phase concentration;
- clean retention;
- action validity;
- policy calls and compute overhead, excluding resource-contention intervals.

One expansion to 80 paired episodes per key policy is allowed only when the
frozen Stage B uncertainty rule declares the result unresolved. No third
expansion is allowed.

## Paper-Candidate Gate

NICE becomes a serious paper candidate only if:

- NICE beats Base on the matched claim condition;
- NICE beats VLA-Corrector;
- NICE beats the mean-only ablation;
- fixed short-horizon replanning does not explain the gain;
- clean behavior and action validity are retained;
- innovation evidence supports the intended interrupt mechanism;
- novelty remains defensible after final literature refresh.

Then immediately verify:

- Quantized OpenVLA-OFT INT4 versus Quantized OpenVLA-OFT INT4 plus NICE;
- one claim-specific second condition or benchmark;
- one or more newly relevant baselines when feasible;
- compute and latency outside contention intervals;
- figure/table-ready mechanism and outcome evidence.

## Failure Classification

- `DATA_FAILURE`: invalid/collapsed pairs, overlap, missing legal input, or
  insufficient task/action-regime coverage;
- `IMPLEMENTATION_FAILURE`: source, shape, reload, gradient, queue, or action
  integration defect;
- `NO_HEADROOM`: valid Base/prior evaluation leaves no plausible gain;
- `DESIGN_FAILURE`: valid implementation but normalized innovation is
  noninformative or cannot improve the matched monitor on development data;
- `VALID_SCIENTIFIC_KILL`: frozen confirmatory method is valid and loses under
  the preregistered decision rule;
- `UNDERPOWERED_OR_UNRESOLVED`: valid evidence does not resolve the claim after
  the allowed expansion.

Implementation or data failure cannot be reported as a closed-loop scientific
result. A major redesign after confirmatory access is a new method cycle.

## Resource Policy

The two Windows gaming and Efficiency Mode intervals remain recorded in
`reports/resource_contention_intervals.json`.

Latency, throughput, wall-clock efficiency, CUDA utilization, and resource
utilization are excluded whenever overlap is unknown or positive. Synchronous
closed-loop success rows may remain valid only after timeout, exception,
identity, action-semantics, duplicate, and manifest audits.

## Researcher A Recommendation

Proceed to independent Reviewer B attack. Do not implement NICE, extract
labeled latents, train a corrector, or access confirmatory identities until the
proposal hash is frozen and Reviewer B attack, rebuttal, mathematical audit,
preregistration, and prototype protocol are complete.
