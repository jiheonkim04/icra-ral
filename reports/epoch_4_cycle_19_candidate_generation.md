# Epoch 4 Cycle 19 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_SPARC_VLA`

Exactly three materially distinct candidates were generated and scored under
the active post-CAVM and post-COVI governance. The prior reconstruction is in
`reports/epoch_4_cycle_19_prior_mechanism_map.md`.

PCAV and FAMR remain closed unchanged. Unknown empirical performance is not a
rejection reason.

## Candidate 1: SPARC-VLA

Name: Success-Preserving Aggregated Reusable Conceptors for VLA policies.

Contribution type: `IMPLICIT_GAP_SOLUTION` plus `PRIOR_EXTENSION`.

Closest external prior: COAST,
https://arxiv.org/abs/2605.17144.

Positive prior anchor: COAST reports more than `20` absolute points mean
simulation gain and more than `40` points mean real-robot gain across three
policy families. It also shows cross-task steering and a correlation between
source-target failure-subspace containment and transfer gain.

Official code/checkpoint status: no official repository was identified during
the preselection audit. The paper provides full equations, pseudocode,
activation-hook details, fitting/evaluation partitions, and hyperparameter
selection rules, so a faithful transparent SmolVLA proxy is reproducible.

### Scientific Difference

COAST self-fitting constructs a target success AND NOT target failure
conceptor. Its cross-task experiment applies a complete source contrastive
conceptor to the target. SPARC instead follows the prior's observed asymmetry:

- preserve the target task's success subspace;
- estimate reusable failure geometry from several source tasks;
- average source failure covariances with equal task weight;
- compose target success AND NOT aggregated source failure;
- apply the same Base-interpolated multiplicative gate.

SPARC therefore needs target successes but no target failure label for its
fitted intervention. The minimal new mechanism is balanced multi-source
failure aggregation plus target-success preservation. It is not a renamed
loss, action residual, memory, or hyperparameter variant.

### Mechanism Chain

- a target task has too few failures, no safe failure collection, or
  deployment arrives before target failures are labeled;
- a complete source contrastive gate transfers source-specific success
  geometry along with reusable failure suppression;
- source success geometry can attenuate target-relevant action-expert
  directions because success subspaces are task-specific;
- balanced source failure covariance estimates only the reusable negative
  geometry without allowing a high-frame-count task to dominate;
- the target-success conceptor restores target-specific positive geometry;
- Boolean AND-NOT yields a bounded soft subspace operator;
- identity interpolation limits action disruption;
- expected result: better target success than Base and single-source COAST
  transfer while retaining clean Base behavior.

### Data And Integration

- source successes/failures and target successes come from disjoint discovery
  rollout identities under the frozen Base checkpoint;
- target failures may be inspected only in a discovery headroom diagnostic and
  are excluded from the SPARC fit;
- validation identities choose one frozen configuration from at most six;
- confirmatory identities remain sealed until every policy and threshold is
  frozen;
- a forward hook on a SmolVLA action-expert MLP captures and gates the existing
  `720`-dimensional activation at every denoising step;
- `beta = 0` is exact Base and the persisted unconfigured policy defaults to
  that state.

### Decisive Experiment

The first serious matched comparison is:

1. `smolvla_base`;
2. `coast_single_source_transfer_proxy`;
3. `sparc_full`;
4. `sparc_source_failure_only`;
5. `standard_lora_target_success`.

The key ablation removes target-success preservation while retaining identical
balanced source-failure data. Standard LoRA is included conditionally because
SPARC receives target-success data; it tests whether ordinary positive-only
adaptation explains the result. LoRA remains an implementation/control, not
the SPARC contribution.

### Score

- provisional novelty: `24 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `96 / 100`

## Candidate 2: BASALT-VLA

Name: Base-Anchored Spherical Actor Latent Tuning for VLA policies.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Latent Policy Steering,
https://arxiv.org/abs/2603.05296, with official code at
https://github.com/jellyho/LPS.

Positive prior anchor: LPS reports `56.2%` real-robot DROID success, above
Flow-BC (`31.2%`), MF-BC (`28.7%`), and DSRL (`35.0%`), by differentiating an
original-action-space critic through a one-step MeanFlow policy into a
spherical latent actor.

Scientific difference: represent the learned spherical actor as a
zero-initialized tangent residual around the Base latent, constrain its
geodesic radius, and use a clean-retention term so the initial and unsupported
policy is exact Base.

Mechanism plausibility: critic gradients identify higher-value original
actions; tangent-space residuals move the latent toward them; a trust radius
and Base gate prevent global replacement. The chain is falsifiable through
critic calibration, latent displacement, action delta, and success.

Data/supervision audit: the repository has expert demonstrations and binary
episode outcomes, but not yet a faithful rewarded mixed transition buffer or
a validated one-step MeanFlow conversion for SmolVLA. Collecting and auditing
that buffer is feasible but substantially less bounded than SPARC's closed-
form fit.

Identity preservation: exact zero residual, a frozen maximum geodesic radius,
and Base fallback outside critic support.

Decisive experiment: Base, faithful LPS proxy, BASALT, no-anchor ablation, and
behavior-cloning LoRA on one matched task. It requires critic calibration and
one-step conversion audits before any rollout.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `89 / 100`

## Candidate 3: GWAP-VLA

Name: Gated World-Action Priors for VLA policies.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: World Pilot,
https://arxiv.org/abs/2606.12403, with official code at
https://github.com/ZefuLin/WorldPilot.

Positive prior anchor: World Pilot reports `84.7%` on LIBERO-Plus and positive
real-robot results by injecting world-action scene evolution and anticipated
trajectory information into the policy.

Scientific difference: inject the world-action trajectory token through a
zero-initialized confidence gate, returning exact Base when world-action
prediction disagreement is high. The claim would be reliable use of a strong
world prior, not a new world model.

Mechanism plausibility: a world-action token supplies future scene/action
structure; confidence gating suppresses hallucinated or out-of-support priors;
the remaining interventions should improve long-horizon consistency without
destroying clean behavior.

Data/supervision audit: the official implementation is available, but local
SmolVLA requires either a compatible pretrained world-action checkpoint or a
new WAM distillation/pretraining stage. Existing local demonstrations can
supervise the latter, but Cycle 12 CALA already showed that weakly predictable
action-latent targets are a real risk.

Identity preservation: zero-initialized gate, bounded token residual, and
deterministic Base fallback.

Decisive experiment: Base, transparent World Pilot proxy, GWAP, always-on
world-prior ablation, and a no-world temporal-token baseline. The checkpoint
and integration cost make this less bounded than SPARC.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `85 / 100`

## Selection

`SPARC-VLA` is selected with `96 / 100`.

SPARC is the strongest combination of positive anchor, locally faithful
mechanism, identity-preserving integration, viable labels, and a decisive
matched comparison. It directly tests an unresolved implication of COAST's
own cross-task geometry instead of attaching a speculative local network.

The provisional novelty risk is explicit: Reviewer B may find that COAST's
cross-task transfer and Boolean conceptor algebra already make SPARC an obvious
recombination. Selection therefore authorizes only a frozen proposal,
independent prior-art attack, rebuttal, mathematical audit, preregistration,
and bounded Stage 0. It does not authorize confirmatory testing.

## Baseline Rationale

| Policy | Distinct scientific question |
| --- | --- |
| `smolvla_base` | Does any steering improve the untouched policy? |
| `coast_single_source_transfer_proxy` | Does SPARC beat the closest prior's no-target-failure transfer mechanism? |
| `sparc_full` | Does target-success preservation plus balanced reusable failure suppression work? |
| `sparc_source_failure_only` | Is target-success preservation necessary beyond pooled failure suppression? |
| `standard_lora_target_success` | Does ordinary positive-only adaptation explain any gain from target-success data? |

No sixth policy is preregistered. Target-success-only conceptor steering is a
representation/action diagnostic before rollout, not an additional policy in
the first serious comparison.
