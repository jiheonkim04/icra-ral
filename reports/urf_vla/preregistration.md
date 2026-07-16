# URF-VLA Preregistration

Date: 2026-07-16 KST

Decision: `URF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

Method: `URF-VLA`, Uncertainty-Routed Residual Flow for Base-preserving
SmolVLA chunks.

Proposal SHA-256:
`E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532`

Prerequisite documents:

- proposal: `reports/urf_vla/researcher_proposal.md`
- Reviewer B attack: `reports/urf_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/urf_vla/researcher_rebuttal.md`
- mathematical audit: `reports/urf_vla/mathematical_mechanism_audit.md`

No URF training, validation search, rollout, simulator access, or confirmatory
test access has happened before this preregistration.

## Fixed Claim

URF tests whether a heteroscedastic residual model plus an explicit
uncertainty-dependent route gate can safely apply bounded Base-to-expert
residual transport to a frozen SmolVLA `[50,7]` action chunk.

The claim is not:

- inventing uncertainty-aware residual flow matching;
- official SUREFlow reproduction unless official assets are installed and
  verified;
- Guided Action Flow-style critic guidance;
- generic residual LoRA;
- an uncertainty-estimation paper;
- or a rescue of CCIF, TSC, CFR, AMP, RAP, or VDR.

## Evidence Partitions

`DISCOVERY / TRAINING`

- legal LIBERO demonstrations only;
- demonstrations `0..7` for each fixed development task;
- used for residual-scale fitting, route-label construction, Base chunk
  decoding, SUREFlow proxy fitting, URF fitting, ablation fitting, standard
  LoRA fitting, implementation debugging, and small gradient/magnitude audits;
- may not include confirmatory reset identities or outcomes.

`VALIDATION`

- legal LIBERO demonstrations only;
- demonstrations `8..9` for each fixed development task;
- used for Stage 0 data/mechanism gates and, only after a Stage 0 pass, bounded
  validation search and final configuration selection;
- may not use confirmatory outcomes.

`CONFIRMATORY TEST`

- untouched until method, configuration, policy list, ablation, tasks, reset
  identities, metrics, thresholds, manifests, and checkpoints are frozen;
- no confirmatory task/reset identities, rewards, success flags, done flags,
  object poses, future observations, or policy actions may be read during Stage
  0 or validation search;
- confirmatory outcomes may not retune URF.

## Fixed Development Sources

Use these four source task families:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Discovery/training demonstrations: `0..7`.

Validation demonstrations: `8..9`.

Minimum Stage 0 windows:

- at least `512` discovery windows;
- at least `128` validation windows;
- every task must contribute validation rows;
- no task may contribute more than `40%` of the Stage 0 validation subset.

## Frozen Residual And Route Construction

Use exactly the mathematical audit variables:

- Base and expert action chunks: `[B,50,7]`;
- normalized residual target: `Y_t = (A_t - B_t) / s_d`;
- residual scale: `s_d = clamp(p95(|A_t - B_t| by coordinate), 1e-4, 10.0)`
  fitted on discovery/training rows only;
- log variance: `ell_theta` parameterizes log variance in normalized residual
  units;
- variance bounds: `log(v_min) = -8`, `log(v_max) = 4`;
- route target: `Z_route = 1[abs(Y_t) >= tau_route_d]`;
- route threshold:
  `tau_route_d = max(p50(|Y_t| for coordinate d), 0.25)` fitted on
  discovery/training rows only;
- route logit must include uncertainty, either through
  `Q_theta = q_base_theta + alpha_m abs(mu_theta) - alpha_u sqrt(v_theta) - tau_g`
  or the equivalent lower-confidence-bound form from the audit.

Any implementation where `ell_theta` does not affect the route gate is not URF
and must stop as `URF_STAGE_0_DESIGN_FAILURE`.

## Stage 0 Development Audit

Stage 0 happens before expensive training, bounded validation search, rollout,
or confirmatory testing. It may perform bounded small-fit smokes needed to
measure gradients, route activation, residual prediction, and proxy headroom.

Required Stage 0 artifact paths:

- `reports/urf_vla/stage_0_manifest.json`;
- `reports/urf_vla/stage_0_partial.json`;
- `reports/urf_vla/stage_0_status.json`;
- `reports/urf_vla/stage_0_heartbeat.json`;
- `reports/urf_vla/stage_0_result.json`;
- `reports/urf_vla/stage_0_result.md`;
- `reports/urf_vla/stage_0_adjudication.md`;
- `reports/urf_vla/stage_0_action_semantics.json`;
- `reports/urf_vla/stage_0_serializer_preflight.json`;
- PID, stdout, stderr, and exit-code files if launched as a detached worker.

Required Stage 0 checks:

1. proposal hash and source document hashes match;
2. official SUREFlow asset/code status is recorded;
3. manifest keys are unique with zero duplicate, missing, extra, or split
   overlap keys;
4. feature, action, proprioception, language/task, phase, and Base chunk
   records are finite and aligned;
5. official SmolVLA / LIBERO action semantics are persisted before any
   action-validity decision;
6. Base decoded chunks are finite and have official shape `[50,7]`;
7. residual scales are noncollapsed by action coordinate;
8. residual targets are noncollapsed by task, phase, timestep, and
   translation/rotation/gripper group;
9. route labels are noncollapsed by task and action group;
10. route positive fraction is between `0.02` and `0.80` after valid masking;
11. heteroscedastic residual prediction beats homoscedastic residual and
    task/phase residual baselines on validation by at least `5%` relative
    normalized residual Huber or `0.005` absolute normalized residual Huber;
12. predicted uncertainty strata are noncollapsed and monotonic with actual
    residual prediction error on validation;
13. monotonicity must satisfy Spearman `rho >= 0.20` between predicted
    `sqrt(v_theta)` and `abs(Y_t - mu_theta)`, or binned means must be
    nondecreasing with at most one adjacent violation below `5%` relative
    magnitude;
14. SUREFlow proxy is heteroscedastic and leaves URF residual headroom of at
    least `5%` relative validation Huber or `0.005` absolute normalized Huber;
15. `urf_no_uncertainty_route_ablation` is implemented and distinct from URF;
16. after a small fit, URF full differs from Base and from the no-uncertainty
    ablation in a bounded way;
17. route activation is neither all-zero, all-one, nor globally active; route
    activation fraction after small fit must lie in `[0.02, 0.80]`;
18. stochastic-sampling disagreement and perturbation-disagreement route
    diagnostics are saved when computationally cheap; if skipped, the result
    must record the cost reason;
19. initialized and disk-reloaded URF reproduces Base actions within `1e-6`;
20. expected URF parameters receive finite nonzero gradients;
21. frozen Base parameters receive zero gradients;
22. weighted objective gradient-norm ratio across required trainable terms is
    at most `100:1`;
23. postprocessed action validity is preserved under frozen official semantics;
24. action deltas are bounded by group and not globally destructive;
25. no confirmatory records, reset identities, rewards, success flags, done
    flags, object poses, future observations, or policy outcomes are read;
26. exceptions are zero.

## Stage 0 Stop Classes

Stage 0 must end with exactly one:

- `URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `URF_STAGE_0_NO_USABLE_HEADROOM`;
- `URF_STAGE_0_DESIGN_FAILURE`;
- `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Classify failures as:

- source, alignment, overlap, residual-scale, residual-target, route-label,
  uncertainty-strata, or task-coverage failure:
  `URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- no Base residual headroom, no SUREFlow-proxy residual headroom, or no
  heteroscedastic residual advantage:
  `URF_STAGE_0_NO_USABLE_HEADROOM`;
- uncertainty not entering the route gate, nonmonotonic uncertainty,
  all-zero/all-one/global route activation, no-uncertainty ablation
  equivalence, or failure to infer the mechanism from legal deployment inputs:
  `URF_STAGE_0_DESIGN_FAILURE`;
- hash, serialization, identity, reload, gradient, frozen-parameter,
  action-semantics, action-validity, persistence, global-delta, or exception
  defect: `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- all gates pass: `URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

No Stage 0 stop is a closed-loop scientific kill and no Stage 0 stop may be
rescued by changing thresholds, route labels, residual scales, proxy
definitions, task/reset identities, or action-validity semantics after seeing
results.

## Bounded Validation Search

Allowed only after `URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Maximum six configurations:

1. URF `g_max=0.05`, `lambda_clean=0.2`, audit-default route threshold;
2. URF `g_max=0.10`, `lambda_clean=0.2`, audit-default route threshold;
3. URF `g_max=0.10`, `lambda_clean=1.0`, audit-default route threshold;
4. `sureflow_uncertainty_residual_proxy` or official `sureflow` if installed
   and verified;
5. `urf_no_uncertainty_route_ablation`;
6. matched `standard_lora`.

One seed per configuration by default. A second seed is allowed only if the
predeclared validation score is genuinely unresolved and must be reported with
the first seed. No combinatorial grid is allowed.

Frozen validation score:

`S = 0.30 * validation_success_or_proxy
   + 0.25 * URF_minus_SUREFlow_proxy_margin
   + 0.15 * uncertainty_monotonicity_score
   + 0.15 * clean_action_retention
   + 0.10 * postprocessed_action_validity
   + 0.05 * route_overhead_score`.

If closed-loop validation is not feasible, `validation_success_or_proxy` must
be a frozen deployment-observable proxy documented in the executable prototype
protocol before execution. Offline action L2 alone may not select the final
configuration.

Tie break:

1. lower `g_max`;
2. lower clean-retention disruption;
3. lower measured compute overhead.

## First Serious Comparison

Exactly five policy identities:

1. `smolvla_base`;
2. `sureflow_uncertainty_residual_proxy` or official `sureflow` if installed
   and verified;
3. `urf_full`;
4. `urf_no_uncertainty_route_ablation`;
5. `standard_lora`.

The closest prior/proxy must enter before any serious paper-viability claim.
The no-uncertainty ablation and standard-LoRA baseline cannot be removed after
seeing validation or confirmatory outcomes.

Guided Action Flow remains listed as the closest frozen-SmolVLA
action-intervention prior. It is not a first-stage policy unless Reviewer B
reopens the comparison before confirmatory testing.

## Stage A / Stage B Policy

No Stage A manifest may be frozen until:

- Stage 0 passes;
- bounded validation selects exactly one URF configuration;
- checkpoints for URF, ablation, proxy if trainable, and standard LoRA are
  saved and disk-reload verified;
- action validity and clean retention pass under official semantics.

Stage A target:

- approximately `10` paired episodes per policy;
- five policies from the frozen first serious comparison;
- shared task/reset manifest.

Stage A may permanently kill only for mechanism invalidity, no headroom,
catastrophic degradation, clear prior, ablation, or simple-baseline dominance,
or exact trivial equivalence.

Stage B target:

- at least `40` paired episodes per key policy;
- paired wins/losses/ties, bootstrap confidence interval, effect size,
  failure-rate reduction, per-task breakdown, mechanism activation, clean
  retention, and efficiency.

One expansion to `80` paired episodes per key policy is allowed only if Stage B
is genuinely unresolved under active governance.

## Confirmatory Tuning Prohibition

Confirmatory outcomes cannot change:

- tasks or reset identities;
- SUREFlow proxy definition;
- action-validity semantics;
- residual scale construction;
- route-label threshold;
- `g_max`, `lambda_clean`, `tau_g`, or loss coefficients;
- uncertainty monotonicity thresholds;
- stage thresholds;
- policy list;
- ablation list;
- standard-LoRA baseline;
- clean-retention rule;
- final checkpoint.

A redesign after confirmatory testing starts a new method cycle.

## Resource Evidence Rule

Windows gaming / Efficiency Mode / resource-contention intervals remain tracked
separately. Timing, throughput, wall-clock efficiency, resource utilization,
and latency measurements overlapping or unresolved against those intervals are
not final paper evidence. Closed-loop task-success rows may remain valid only
if the simulator is synchronous, no timeout or exception occurs, action
semantics and task/reset identities are unchanged, and duplicate rows are
absent.

## Preregistration Decision

URF may proceed to an executable prototype protocol. It may not proceed to
implementation, training, validation search, rollout, or confirmatory testing
until that protocol is frozen.
