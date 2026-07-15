# SPARC-VLA Researcher A Proposal

Date: 2026-07-15 KST

Decision: `SPARC_RESEARCHER_PROPOSAL_READY_FOR_REVIEW`

## Method Identity

Name: `SPARC-VLA`, Success-Preserving Aggregated Reusable Conceptors for VLA
policies.

Contribution type: `IMPLICIT_GAP_SOLUTION` plus `PRIOR_EXTENSION`.

Scientific method: task-balanced multi-source failure conceptor aggregation,
composed with a target-specific success conceptor and applied as a bounded
multiplicative action-expert gate.

Low-compute parameterization: a closed-form `720 x 720` conceptor operator and
a forward hook on frozen SmolVLA. No learned SPARC parameter and no Base weight
update are required.

Closest external prior: COAST,
https://arxiv.org/abs/2605.17144.

The proposal treats the paper as a reproducible mechanism source, not an
official code reproduction. Any local hook or checkpoint difference is listed
in a proxy-fidelity manifest.

## Research Question

When target failures are unavailable for fitting, can a frozen VLA be steered
more effectively by preserving target-success geometry while suppressing a
task-balanced aggregate of reusable source-failure geometry, compared with
Base and a complete single-source COAST transfer gate?

The paper-axis claim is narrow:

> In the no-target-failure fitting regime, separating task-specific target
> success geometry from transferable multi-source failure geometry improves
> cross-task conceptor steering while preserving clean Base behavior.

SPARC does not claim conceptors, action-expert activation steering, Boolean
AND-NOT, multiplicative gating, or cross-task failure transfer individually.

## Positive Prior And Exact Gap

COAST constructs success and failure conceptors from action-expert rollout
activations and applies `success AND NOT failure` as a multiplicative residual-
stream gate. It reports broad positive results across flow, autoregressive, and
diffusion policies.

COAST also reports that success subspaces are relatively task-specific while
failure-subspace containment predicts cross-task transfer. Nevertheless, its
cross-task arm transfers a complete source contrastive conceptor:

`source success AND NOT source failure`.

SPARC changes exactly this construction. For target task `T` and source tasks
`S_1 ... S_J`, it constructs:

`target success AND NOT balanced multi-source failure`.

This removes source-specific success geometry, keeps target-specific success
geometry, and prevents tasks with more frames from dominating the failure
estimate.

## Separation From Closed Campaign Methods

- Not CAVM: no nearest-neighbor action memory, no 7D success/failure action
  means, and no state-action retrieval at inference.
- Not FANG: no learned 7D action field, residual head, or action-space target.
- Not PCAV: no multi-noise candidates, support verifier, reranking, or adapted
  candidate generator.
- Not CALA: no prediction of a future action latent from deployment context.
- Not FAMR: no endpoint response model, merge coefficient, or stopped
  checkpoint is loaded.

SPARC starts from the untouched Base checkpoint.

## Frozen Backbone And Local Hook

Checkpoint: `/mnt/c/assets/checkpoints/smolvla_libero`.

The config has `16` VLM/action-expert layers, action-expert width `720`, action
chunk `50 x 7`, and `10` flow denoising steps.

The hook point is the output of
`policy.model.vlm_with_expert.lm_expert.layers[layer].mlp`. The unmodified
SmolVLA implementation invokes that module at every denoising step before the
residual addition. Stage 0 must prove the observed tensor shape and whether
the paper's residual-stream convention corresponds to gating this MLP output
or the post-add residual. Reviewer B may require the latter; the mathematical
audit freezes one semantically faithful hook before any labeled fit.

No installed package source is edited. The hook is registered and removed by
the experiment adapter.

## Frozen Tasks

Target tasks use known mixed-outcome official SmolVLA tasks but exclude the two
previously disallowed SmolVLA-specific failure claims:

1. `libero_10/task_8`: put both moka pots on the stove;
2. `libero_10/task_6`: put the white mug on the plate and put the chocolate
   pudding to the right of the plate;
3. `libero_goal/task_8`: put the bowl on the plate.

Historical Base discovery evidence is respectively `1/5`, `2/5`, and `3/5`
successes. Those rows contain no captured SPARC activations and are not final
evidence; they only establish plausible outcome headroom.

Source tasks are:

1. `libero_10/task_0`: put both the alphabet soup and tomato sauce in the
   basket;
2. `libero_goal/task_0`: open the middle drawer of the cabinet;
3. `libero_object/task_2`: pick up the salad dressing and place it in the
   basket;
4. `libero_spatial/task_8`: pick up the black bowl next to the plate and place
   it on the plate.

The target set, source set, and two prohibited task identities are disjoint.
No task may move between source and target after labeled activation collection
begins.

## Evidence Partitions

Every identity key is `(partition, policy, suite, task_id, reset_seed)`.

- `DISCOVERY`: reset seeds `20261901..20261912`;
- `VALIDATION`: reset seeds `20261921..20261926`;
- `CONFIRMATORY_TEST`: reset seeds `20261941..20261960`.

The ranges are disjoint from each other and from the historical five-reset
screen. A manifest is materialized before the first episode in each partition.

Discovery may be used to verify labels, select the semantic hook convention,
compute geometric layer/aperture diagnostics, construct all conceptors, and
debug implementation.

Validation may select one of at most six frozen configurations and evaluate
mechanism activation, action validity, clean retention, and matched success.

Confirmatory test is opened once after method, configuration, checkpoints,
policy list, tasks, reset identities, metrics, thresholds, and kill rules are
frozen. Its outcomes cannot retune SPARC.

## Activation Records

For every Base replan step, capture the action-expert activation tensor at each
eligible layer and denoising step. The expected shape before token pooling is
`[batch=1, action_tokens=50, hidden=720]`. Mean pooling over action tokens gives
`x in R^720` per denoising step.

Each persisted record includes:

- policy, task, reset identity, outcome, replan index, and denoising step;
- layer and hook convention;
- activation dtype, shape, finite fraction, norm, and hash;
- emitted native and postprocessed `50 x 7` action chunk hashes;
- episode success, timeout, exception, and synchronous-simulator status.

Outcome labels are episode-level simulator completion labels. No object pose,
reward value, reset identity, future observation, or privileged state is used
by the inference gate.

## Conceptor Construction

For activation rows `X in R^(N x 720)` from one class, compute the class mean
`mu in R^720`, centered rows `X_c = X - mu`, and covariance:

`R = X_c^T X_c / N`, shape `720 x 720`.

At aperture `alpha > 0`:

`C(R, alpha) = R (R + alpha^-2 I)^-1`.

The implementation uses symmetric eigendecomposition in float64, clamps only
roundoff-scale eigenvalue violations documented before execution, and stores
the mean, covariance hash, eigenvalues, effective rank, and conceptor.

For each target `T`, target-success covariance is estimated only from
successful target discovery episodes:

`R_s^T` and `C_s^T = C(R_s^T, alpha)`.

For each source task `j`, estimate the failure covariance from failed source
discovery episodes after centering within that task:

`R_f^j`.

Aggregate with equal task weight:

`R_f^src = (1 / J) sum_j R_f^j`.

Then:

`C_f^src = C(R_f^src, alpha)`.

The SPARC operator is:

`C_sparc = pinv(pinv(C_s^T) + pinv(I - C_f^src) - I)`.

The exact numerical tolerance and final PSD/eigenvalue projection policy are
frozen in the mathematical audit. Any correction beyond documented numerical
roundoff is an implementation failure, not an unreported method change.

## Inference Gate

For steering strength `beta in [0, 1]`:

`M = (1 - beta) I + beta C_sparc`.

At the frozen action-expert layer and strategy, replace activation `h` by
`M h`. Global steering uses one operator at all denoising steps. Per-step
steering constructs and applies one operator per step.

The persisted adapter initializes with `configured = false` and `beta = 0`.
While unconfigured it does not transform activations. The initial action must
match direct Base bit-for-bit before and after disk reload.

## Closest-Prior Proxy

`coast_single_source_transfer_proxy` uses the same hook, activation records,
conceptor implementation, layer/aperture procedure, strategy, beta candidates,
action semantics, and inference budget as SPARC.

For each source task it constructs the prior's complete operator:

`C_coast^j = C_s^j AND NOT C_f^j`.

One source is selected per target by maximum validation-only failure
containment, with a stable lexicographic task tie break. It is then frozen. No
confirmatory target outcome selects a source.

This arm is a faithful transparent SmolVLA proxy, not an official COAST
checkpoint or published-number comparison.

## Key Ablation And Simple Control

`sparc_source_failure_only` keeps the balanced source failure covariance and
all selected inference settings but removes target-success preservation:

`C_ablate = I - C_f^src`.

It tests whether pooled failure suppression alone explains SPARC.

`standard_lora_target_success` uses only the same successful target discovery
episodes available to `C_s^T`. Its data, steps, optimizer, rank, target
modules, checkpoint selection, and inference budget are frozen before
training. It tests ordinary positive-only adaptation and is not the method.

A target-success-only conceptor and pooled CAA vector are offline mechanism
diagnostics. They do not become extra first-comparison rollout policies.

## Bounded Validation Search

Layer and aperture are selected geometrically on discovery activations using
the COAST quota and overlap procedure. Exact candidate layers, aperture list,
overlap band, and tie breaks are frozen in the mathematical audit before
labeled extraction. This deterministic geometric reduction is not selected by
validation success.

Validation evaluates exactly six configurations:

- strategy in `{global, per_step}`;
- `beta` in `{0.1, 0.3, 0.5}`.

One seed is sufficient because conceptor fitting is closed-form. Standard LoRA
uses its own preregistered training seed and may use a second seed only after
one configuration is selected, as permitted by governance.

The validation score combines paired success, clean retention, action
validity, bounded mechanism activation, and compute overhead. It cannot be
selected by offline action L2 alone. Ties prefer higher clean retention, then
lower beta, global strategy, lower overhead, and lexicographic config id.

## Pre-Experiment Headroom And Data Audit

Before any steered validation rollout, require:

1. at least `3` successful target discovery episodes per target;
2. at least `3` failed source discovery episodes per source;
3. nonzero activation variance and finite full-rank-tolerant covariance;
4. no task or reset overlap among partitions;
5. no all-success/all-failure label collapse in any required fit;
6. target-failure containment by `C_f^src` above a preregistered shuffled-task
   null on discovery target failures;
7. target-success retention by `C_sparc` above the source-success retention of
   the selected complete source COAST gate;
8. nonzero but bounded action consequence at `beta = 0.1`;
9. exact Base identity at `beta = 0`;
10. action-space validity and clean discovery retention.

Target failures used in items 6-7 are diagnostic-only discovery rows. They do
not enter `R_f^src`, `C_s^T`, configuration selection, or inference.

If the required success/failure counts are absent after the frozen discovery
manifest, classify `DATA_FAILURE`. Do not change tasks, labels, or identities
inside the same cycle.

If geometry is finite but source failure containment has no headroom, classify
`NO_HEADROOM` or `DESIGN_FAILURE` as preregistered. If the hook is nonacting,
shape-invalid, or numerically unstable, classify `IMPLEMENTATION_FAILURE`.
None is silently promoted to a closed-loop scientific kill.

## Identity And Action-Safety Audit

For every method row report:

- Base action and Ours action;
- activation before/after norm and gate-induced delta;
- beta, layer, denoising step, and strategy;
- conceptor effective rank and eigenvalue range;
- changed action dimensions;
- translation, rotation, and gripper delta from Base;
- finite fraction, maximum absolute action, Base-relative bound exceedance;
- clean-versus-target intervention behavior.

Before validation rollout, freeze numerical postprocessed action-validity
limits from Base discovery/validation rows. No after-result clipping, threshold
change, or action repair is allowed.

## First Serious Comparison

The matched policy order is:

1. `smolvla_base`;
2. `coast_single_source_transfer_proxy`;
3. `sparc_full`;
4. `sparc_source_failure_only`;
5. `standard_lora_target_success`.

All policies share task/reset identities. Stage A uses approximately `10`
paired episodes per policy. Small differences advance to Stage B. Stage B uses
at least `40` paired episodes per key policy and one expansion to `80` only if
the frozen unresolved rule fires.

Report successes/counts, paired wins/losses/ties, paired bootstrap confidence
interval, effect size, failure-rate reduction, per-task outcomes, activation
geometry, action consequences, clean retention, and uncontaminated efficiency.

SPARC becomes a paper candidate only if it beats Base, the COAST proxy, the
key ablation, and the relevant LoRA explanation while retaining clean
behavior and supporting the intended geometry. Quantized OpenVLA-OFT INT4 and
one claim-specific second condition follow immediately only then.

## Durable Execution And Resource Quarantine

Every job longer than a unit test writes PID, heartbeat, status, atomic partial
JSON, final JSON, stdout/stderr log, and exit code. Resume only missing keys.
Never duplicate a living worker.

Every rollout key `(policy, suite, task_id, reset_seed)` must be unique and
match its frozen manifest. A stale heartbeat is insufficient evidence of
death; PID and logs must also be checked.

The recorded Windows gaming/Efficiency Mode intervals remain resource-
contention intervals. Overlapping or unknown-overlap latency, throughput,
wall-clock efficiency, and utilization are excluded from paper evidence.
Synchronous, exception-free, timeout-free success rows with unchanged action
semantics and zero duplicates may remain valid.

## Proposal Decision

This proposal authorizes an independent Reviewer B prior-art and mechanism
attack. It does not authorize implementation, labeled activation collection,
validation search, confirmatory decoding, or rollout until the proposal is
hashed and the required rebuttal, mathematical audit, preregistration, and
prototype protocol are frozen.
