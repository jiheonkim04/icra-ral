# PCAV-VLA Researcher A Proposal

Date: 2026-07-15 KST

Decision: `PCAV_RESEARCHER_PROPOSAL_READY_FOR_REVIEW`

## Method Identity

Name: `PCAV-VLA`, Progress-Conditioned Anti-exploration Verification for VLA
policies.

Contribution type: `CROSS_PAPER_SYNTHESIS` plus `PRIOR_EXTENSION`.

Scientific method: support-constrained, progress-conditioned candidate action
verification with exact Base abstention.

Low-compute implementation: frozen SmolVLA, fixed multi-noise candidate
generation, one coupled support head, one latent consequence model, and one
progress head. LoRA is not the method and is not required for PCAV.

Closest external prior: TACO,
https://arxiv.org/abs/2512.02834, official code at
https://github.com/breez3young/TACO.

Single extension prior: ProgressVLA,
https://arxiv.org/abs/2603.27670.

Direct contemporary comparison prior: VLA-ATTC,
https://arxiv.org/abs/2605.01194.

Generate-and-verify reference: RoboMonkey,
https://arxiv.org/abs/2506.17811, official code at
https://github.com/robomonkey-vla/RoboMonkey.

## Research Question

For a frozen flow-based SmolVLA adapted to a target manipulation task, can a
small action verifier improve closed-loop success by choosing a
progress-advancing candidate only from successful demonstration support, while
returning the deterministic Base action whenever the evidence is weak?

The paper-axis claim is narrower than generic VLA improvement:

> Separating action support from predicted task progress improves candidate
> selection over Base and support-only anti-exploration without sacrificing
> clean Base behavior.

## Positive Prior And Gap

TACO demonstrates that a coupled pseudo-count head can identify high-support
action candidates and improve several VLAs, including OpenVLA and pi0.5 on
LIBERO-Long. Its key assumption is that high demonstration density correlates
with task success.

That assumption is incomplete for long-horizon execution. A common
approach/hold/low-motion action can be in support yet wrong for the current
phase. ProgressVLA independently shows that an explicit progress estimator and
latent action-consequence model can guide manipulation actions toward task
completion.

PCAV changes one technical decision: TACO's maximum-support candidate becomes
an eligibility filter, and action-conditioned predicted progress selects among
eligible candidates. Base is always eligible and remains selected unless the
best candidate clears a validation-frozen advantage margin.

This is not FAMR rescue. PCAV does not use the FAMR endpoint, parameter groups,
merge coefficients, functional response model, or failed action-validity
threshold. It starts from the untouched SmolVLA Base checkpoint.

## Frozen Backbone And Tasks

Base checkpoint:

`/mnt/c/assets/checkpoints/smolvla_libero`

Base pretraining dataset:

`lerobot/libero`, 40 task identities recorded in
`reports/famr_vla/task_provenance_manifest.json`.

Target tasks use official local LIBERO-90 demonstrations and have zero
normalized identity overlap with those 40 pretraining tasks:

1. `KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf`
2. `LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray`
3. `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy`

FAMR decoded only discovery demonstrations for these tasks and never ran
headroom, validation, or confirmatory rollout. PCAV receives no FAMR action
endpoint and no FAMR test outcome.

## Evidence Partitions

For each target-task demonstration file:

- `DISCOVERY`: episode indices `0..29`;
- `VALIDATION`: episode indices `30..39`;
- `CONFIRMATORY_OFFLINE`: episode indices `40..49`.

Discovery may be used for candidate observability, label construction, support
and consequence training, progress training, and mechanistic debugging.

Validation may be used for at most six frozen configurations, action validity,
clean retention, activation, and one paired closed-loop proxy when affordable.

Confirmatory-offline rows are not decoded until the architecture, selected
configuration, checkpoints, policy list, metrics, thresholds, and closed-loop
manifest are frozen. Confirmatory outcomes may not retune PCAV.

Closed-loop discovery, validation, and confirmatory reset identities will be
disjoint and explicitly materialized before the first closed-loop candidate is
run. No reset identity may appear in more than one partition or policy row may
appear twice.

## Candidate Construction

At observation `o_t` and instruction `ell`, frozen SmolVLA produces one
deterministic Base chunk `a_0` and `N-1` alternative chunks from fixed,
preregistered noise seeds. The candidate set is:

`A_t = {a_0, a_1, ..., a_(N-1)}`.

Candidate count is initially `N=4` for Stage 0. Validation may compare only
`N in {4, 8}` under the six-configuration budget.

All candidates use the same observation, instruction, proprioception,
postprocessor, action horizon, action dimensions, and SmolVLA solver steps.
Only the initial flow noise differs. Candidate index zero and its noise are
frozen as Base.

The candidate audit records raw/native and postprocessed chunks, noise hashes,
finite values, unique hashes, pairwise translation/rotation/gripper distances,
action bounds, and Base identity.

## Support Estimator

The support head follows the reproducible TACO mechanism rather than claiming
official checkpoint equivalence:

1. pass successful discovery `(observation, instruction, action)` rows through
   frozen SmolVLA with fixed noising levels;
2. retain an internal feature associated with the prediction nearest the
   demonstration action;
3. assign fixed Rademacher vectors;
4. fit a small coupled head by squared regression;
5. convert the head norm to a monotone pseudo-count/support score.

Only discovery data trains this head. The support threshold is a validation
percentile and is never selected on confirmatory data.

The closest-prior arm, `taco_support_proxy`, uses identical candidates and
selects maximum support exactly, with no PCAV progress model or abstention
margin. It is reported as a transparent proxy, not an official reproduction.

## Consequence Model

Let frozen deployment-observable context be `z_t` and proprioception be `s_t`.
A lightweight consequence model receives `z_t`, `s_t`, and a candidate chunk
`a_i`, and predicts the frozen context change over a fixed future offset
`delta`:

`F_omega(z_t, s_t, a_i) -> delta_z_hat_i`.

Training tuples come only from ordered discovery demonstrations:

`(z_t, s_t, a_t:t+H, z_t+delta)`.

The fixed offset, context extraction location, and dimensionality are frozen in
the mathematical audit before training. No simulator object pose or success
signal is an input or target.

## Progress Model

The progress head maps a frozen context and instruction to a scalar:

`P_phi(z_t, ell) in [0, 1]`.

It is trained on within-episode temporal ordering and local continuity. For two
frames from the same episode with `j > i + gap`, the later frame is preferred.
Pairs are symmetrized so input order cannot reveal the label.

Normalized frame index may be used only as a discovery training target and
diagnostic. It is never available at inference. The decisive signal is
held-out within-episode ordering accuracy against task-only,
proprioception-only, and normalized-time trivial baselines.

Candidate predicted progress is:

`q_i = P_phi(z_t + F_omega(z_t, s_t, a_i), ell)`.

## PCAV Decision Rule

The decision is lexicographic:

1. Base candidate `a_0` is always retained.
2. Any nonfinite or action-invalid alternative is removed.
3. Alternatives below the validation-frozen support threshold are removed.
4. Let `a_star` maximize predicted progress among the remaining candidates.
5. Return `a_star` only if `q_star - q_0 > m`, where `m` is the
   validation-frozen abstention margin.
6. Otherwise return `a_0` exactly.

There is no arbitrary sum of support and progress scores. Support defines
eligibility; progress defines preference; the margin defines abstention.

## Identity And Action Safety

The following are invariants, not tunable paper outcomes:

- Base appears in every candidate set;
- invalid head output, no eligible alternative, nonfinite score, or low margin
  returns Base;
- candidate actions use the official Base postprocessor;
- no clipping is introduced after seeing action-validity results;
- absolute and Base-relative action-validity limits are frozen before the first
  trained-head audit;
- all Base parameters remain frozen and hash-identical;
- no privileged inference input is allowed.

Before training, zero-initialized consequence/progress output and an infinite
abstention margin must reproduce Base bitwise after serialization and reload.

## Stage 0 Development Audit

Stage 0 is bounded and uses discovery rows only.

### Stage 0A: Data And Candidate Observability

Required checks:

- all three target tasks and requested episode ranges exist;
- frame/action/state/image fields have the expected shapes and finite values;
- episode/frame keys and row hashes contain no duplicates or cross-partition
  overlap;
- target task identities remain disjoint from Base pretraining identities;
- at least `24` fixed discovery observations are decoded, balanced `8` per
  task and spanning early/middle/late episode thirds;
- four fixed-noise candidates per row persist and disk-reload;
- Base candidate exactly matches direct Base inference;
- candidate diversity is noncollapsed in translation or rotation on more than
  half the rows;
- candidate actions pass frozen absolute and Base-relative validity gates;
- confirmatory observations decoded and actions computed both equal zero.

Exact candidate equivalence is a `DESIGN_FAILURE`. Missing/corrupt data,
postprocessing mismatch, invalid action semantics, or a failed reload is an
`IMPLEMENTATION_OR_DATA_FAILURE`. Neither class is a closed-loop scientific
kill.

### Stage 0B: Minimum Head Capacity

Only after Stage 0A passes:

- fit each small head for a fixed `20` optimizer-step micro audit;
- verify finite nonzero gradients and checkpoint reload;
- verify support targets, temporal pairs, and consequence targets are
  noncollapsed;
- show support score differs between demonstration-near and deliberately
  perturbed candidates;
- show consequence prediction improves over zero-delta on the fixed discovery
  subset;
- show progress ordering exceeds chance on discovery holdout;
- show `pcav_full`, support-only, progress-only, and Base are not accidentally
  identical on every acting row;
- show the full decision remains sparse and Base-passthrough when evidence is
  weak.

An underpowered ordering or consequence result is
`UNDERPOWERED_OR_UNRESOLVED`, not a permanent kill. Collapsed labels or an
incapable observable signal are `DATA_OR_SUPERVISION_FAILURE`. A valid robust
design failure requires adequate targets, capacity, optimization, and a
preregistered decisive comparison.

## Bounded Validation Search

Maximum six total configurations:

1. support percentile `50`, progress margin `0.00`, `N=4`;
2. support percentile `50`, progress margin `0.05`, `N=4`;
3. support percentile `70`, progress margin `0.00`, `N=4`;
4. support percentile `70`, progress margin `0.05`, `N=4`;
5. selected support/margin with `N=8`;
6. selected support/margin with the second lightweight seed and selected `N`.

The head architecture, context feature, future offset, losses, optimizer, and
training schedule do not enter this search. The tie break is lower
intervention frequency, then lower overhead, then smaller `N`.

The validation score must combine:

- paired validation closed-loop success or the closest feasible proxy;
- clean Base retention;
- support/progress mechanism activation;
- action validity;
- compute overhead.

Every attempted configuration and negative result is saved. One configuration
and checkpoint are frozen before confirmatory evaluation.

## First Serious Comparison

Exactly five policies:

1. `smolvla_base`
2. `taco_support_proxy`
3. `pcav_full`
4. `pcav_progress_only`
5. `standard_lora_new_task`

`pcav_progress_only` uses the same candidates, consequence/progress heads, and
abstention margin but does not apply the support eligibility threshold. It is
the key ablation for the anti-exploration constraint.

Standard LoRA is included because matched target-task adaptation remains a
plausible simple explanation. It does not define PCAV's method identity.

## Decisive Experiment

Stage A uses approximately `10` paired reset identities per policy with one
matched manifest. It may permanently stop the formulation only for mechanism
invalidity, no usable headroom, catastrophic degradation, clear closest-prior
or ablation dominance, or exact trivial equivalence.

Small directional differences advance to Stage B. Stage B uses at least `40`
paired reset identities per key policy and reports paired wins/losses/ties,
bootstrap confidence intervals, effect size, failure-rate reduction, per-task
results, activation, clean retention, and compute. One expansion to `80` is
allowed only if the frozen Stage B result is genuinely unresolved.

## Paper-Candidate Gate

PCAV becomes a serious paper candidate only if:

- `pcav_full` beats Base;
- `pcav_full` beats `taco_support_proxy`;
- `pcav_full` beats `pcav_progress_only`;
- standard LoRA does not explain the gain;
- clean behavior and action validity are retained;
- support and progress evidence match the intended mechanism;
- novelty remains defensible against TACO, ProgressVLA, VLA-ATTC, and
  RoboMonkey.

After GO, test the same frozen method on Quantized OpenVLA-OFT INT4, add one
claim-specific second condition, and prepare paper-scale statistics and
artifacts.

## Current Boundary

This proposal authorizes independent Reviewer B analysis only. It authorizes no
training, validation search, confirmatory decoding, or closed-loop rollout.
