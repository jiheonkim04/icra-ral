# SPARC-VLA Independent Reviewer B Attack

Date: 2026-07-15 KST

Proposal hash reviewed:
`CC2F9ACCE2A26EC438C58F2854ADC95134354C245CAD8ED961D29A895DBC697D`.

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

The proposal is not rejected as exact prior-art duplication. Its novelty is
thin, however, and its implementation fidelity is not yet established. No
labeled activation collection or steered rollout is authorized before the
required rebuttal and mathematical audit are frozen.

## Independent Prior Search

### Closest prior: COAST

Primary source: https://arxiv.org/abs/2605.17144

COAST already owns:

- success/failure action-expert conceptors;
- Boolean `success AND NOT failure` composition;
- global and per-denoising-step multiplicative gating;
- geometric layer/aperture selection;
- cross-task transfer without refitting;
- the empirical conclusion that failure geometry transfers while success
  geometry is comparatively task-specific;
- positive-only, failure-only, CAA, SAE, random, and LoRA comparisons.

SPARC's provisional difference is therefore only the operator's data
construction: target success plus balanced multi-source failure. That is a
logical response to COAST's reported geometry and may be viewed as obvious.
The paper claim survives only if the exact construction beats a faithfully
matched COAST transfer arm and the failure-only ablation in a no-target-
failure-fit regime.

### SmolVLA mechanistic prior

Not All Features Are Created Equal,
https://arxiv.org/abs/2603.19233, studies SmolVLA among six VLAs. It reports
that expert pathways encode motor programs, cross-task activation injection can
carry spatially bound source behavior, and per-token processing is important
for action fidelity on most architectures.

This strengthens SPARC's motivation for removing source-success geometry, but
also attacks its safety: action-expert subspace transfer can import source
motor programs, and token mean pooling may erase the action-token structure
that matters causally.

### Failure-guidance prior

AFIL, https://arxiv.org/abs/2605.08434, trains success and failure action
generators and adaptively steers diffusion/flow sampling using their distance.
It is not a conceptor method and is not exact duplication, but it owns a broad
failure-informed flow-guidance claim. SPARC must remain an inference-time
closed-form representation-geometry contribution, not claim generic negative
guidance.

### Concurrent frozen-VLA prior

Harness VLA, https://arxiv.org/abs/2607.08448, uses task execution traces,
success rules, and failure models around frozen VLA primitives. It operates at
an agent/planner level and does not duplicate conceptor steering. It does mean
SPARC cannot claim that frozen-policy failure knowledge is itself new.

The search found no primary source that explicitly constructs a task-balanced
multi-source failure conceptor and composes it with a target-success conceptor
for VLA action-expert steering.

## Major Attacks

### 1. Novelty May Be An Obvious Boolean Recombination

COAST supplies every mathematical operator and the exact empirical asymmetry
used to motivate SPARC. Averaging source covariances and swapping
`C_s^source` for `C_s^target` may be an obvious engineering follow-up rather
than a contribution.

The rebuttal must narrow novelty to a falsifiable low-target-failure-fit claim,
identify why complete source transfer is structurally mismatched, and state
that failure-only or single-source parity kills the contribution. A positive
result against Base alone is insufficient.

### 2. The Proposed MLP Hook Is Not COAST's Residual-Stream Gate

The installed SmolVLA path computes:

`after_first_residual = attention_output + hidden_states`

then:

`next_hidden = mlp(layernorm(after_first_residual)) + after_first_residual`.

A hook on `layer.mlp` captures only the MLP branch. Multiplying that branch by
the conceptor is not equivalent to multiplying the full post-add residual
stream described by COAST.

The rebuttal must specify a wrapper or hook that captures and replaces the
full action-expert hidden state between layers without editing installed
source. Stage 0 must prove that capture-only mode is bit-exact Base and that
the steering operator acts on the intended tensor.

### 3. Token Mean Pooling Can Hide Destructive Token Effects

COAST mean-pools tokens for conceptor fitting but applies the width-space
operator to token activations. The SmolVLA mechanistic prior warns that token-
level structure can matter for action fidelity.

SPARC must retain every `50 x 720` token tensor for diagnostics, fit from
episode-balanced token summaries exactly as preregistered, apply the same
`720 x 720` operator independently to every token, and report per-token norm
and action-index consequences. A single pooled action delta is insufficient.

### 4. Failed Episodes Contain Successful-Looking Prefixes

An episode-level failure label marks every activation in a failed rollout as
failure. Long episodes can contain correct approach, grasp, and transport
segments before one terminal error. Pooling all frames can make
`R_f^src` a task-duration or common-control covariance rather than reusable
failure geometry.

Equal task weighting alone does not fix within-task episode-length dominance.
The rebuttal must freeze equal episode weight, equal denoising-step weight, a
fixed maximum number of replans per episode, and a terminal-prefix sensitivity
diagnostic. It may not invent privileged step-level failure labels.

### 5. The Aggregate Covariance Needs A Mathematical Justification

`C(mean_j R_j)` is not equal to `mean_j C(R_j)`, and conceptor construction is
nonlinear in both covariance and aperture. A broad source covariance can
become nearly identity and make `I - C_f^src` globally suppressive.

The audit must compare the proposed covariance aggregate with a diagnostic
mean-conceptor aggregate, report eigenvalue spectrum, quota, effective rank,
condition number, and source-task contribution, and explain why covariance
aggregation is the scientific method. The diagnostic may not become an
unbounded seventh configuration.

### 6. Target-Success Preservation Can Suppress Recovery States

Only successful on-policy target episodes populate `C_s^T`. These may cover a
narrow nominal path and exclude states from which recovery is possible. AND
with `C_s^T` can therefore suppress useful target directions even if source
failure removal is valid.

Require target-success coverage by phase/replan index, leave-one-episode-out
stability, and clean/shift action consequences. Three episodes is a numerical
minimum, not evidence of adequate coverage.

### 7. The No-Target-Failure Claim Is Easy To Overstate

The proposal permits target failures for headroom diagnostics, validation
success, and selection of the COAST source proxy. The method fit excludes
target failure activations, but the development process is not target-failure-
blind.

Every report must say `no target failure activations in the SPARC operator
fit`, not `no target failures required` or `zero-failure adaptation`. Count and
disclose all target failure episodes used for diagnostics and validation.

### 8. Geometric Selection Is Underspecified

The proposal defers exact layer candidates, aperture list, overlap band,
pseudoinverse tolerance, eigenvalue correction, and tie breaks. These choices
can materially change the operator.

Freeze all values in the mathematical audit before labeled extraction. Layer
and aperture selection for SPARC may use only `C_s^T` and `C_f^src`, not target
failure activations. Target failures remain diagnostic-only.

### 9. The Source And Target Tasks Are Not A New-Task Test

All selected tasks are from the 40-task distribution associated with the Base
checkpoint. The initial claim is therefore low-failure-label steering on known
task identities, not new-task generalization.

If SPARC reaches paper-candidate status, the required second condition must
include held-out task identities or a second benchmark. The current bounded
prototype remains useful but cannot support a new-task headline.

### 10. The Fit Manifest Is Large Enough To Bias Source Choice

Seven tasks times twelve discovery resets is `84` Base episodes before any
validation. Source tasks were chosen using historical mixed-outcome evidence.
That is permitted discovery design, but all task-selection evidence and reset
ranges must be disclosed. The full fixed manifest must run; do not stop after
obtaining favorable class counts.

### 11. Prior Fairness Requires Matched Budget Accounting

SPARC receives four source tasks plus target successes. A single-source COAST
proxy receives much less fitting data unless budgets are explicitly matched.
The prior arm must either receive an equal episode/activation budget or the
comparison must be labeled as multi-source-data versus single-source prior.

At minimum report both episode count and retained activation count. A second
budget-matched COAST aggregate cannot be introduced after results unless it is
already the one key prior proxy.

### 12. Standard LoRA Is Necessary But Can Be Unfair

Including standard LoRA is correct because SPARC receives target-success data.
COAST's filtered-BC baseline uses successful on-policy trajectories. Local
LoRA must receive exactly the same successful observation-action pairs, not
expert demonstrations or additional target data, and must use a frozen
checkpoint rule. Weak micro-training is implementation evidence, not a
scientific LoRA comparison.

### 13. Geometry Gates Need Numerical Decision Rules

The proposal names shuffled containment, target-success retention, bounded
action consequence, and clean retention without values. Freeze formulas,
null construction, confidence interval or quantile, minimum effect, and action
limits before collection.

If a label quota, extraction, covariance, or hook fails, classify
`DATA_FAILURE` or `IMPLEMENTATION_FAILURE`. If adequately sampled geometry has
no target-success/failure separation or source-failure containment, classify
`NO_HEADROOM` or `DESIGN_FAILURE`. Do not call either a closed-loop scientific
kill.

### 14. Action Safety Must Be Base-Relative And Per Component

A bounded activation delta can still rotate the output action sharply. Freeze
translation, rotation, gripper, absolute-range, Base-relative exceedance, and
simulator acceptance gates. Report all components and do not repair actions by
post-hoc clipping.

### 15. Durable Execution Rules Must Cover Activation Rows

The proposal's rollout key is episode-level, but activation extraction adds
replan, layer, denoising step, and token axes. Freeze separate unique keys and
hashes for episode rows and activation records. Resume only missing episode
keys; never append a second activation set for an already complete episode.

The Windows gaming/Efficiency Mode intervals remain excluded from timing and
resource evidence exactly as proposed.

## Required Rebuttal

Researcher A must provide:

1. a narrow novelty and claim boundary;
2. a faithful post-residual SmolVLA capture/gate design;
3. token-level application and diagnostics;
4. episode-balanced failure aggregation and prefix sensitivity;
5. mathematical justification and diagnostic alternative aggregation;
6. target-success coverage/stability rules;
7. precise disclosure of target-failure use;
8. frozen geometric/numerical selection rules;
9. known-task claim limits and second-condition path;
10. complete fixed-manifest and budget accounting;
11. matched filtered-BC LoRA specification;
12. numerical headroom and action-safety gates;
13. activation-record identity and resume rules;
14. false-negative classifications that separate data, implementation,
    underpowered, no-headroom, design, and genuine scientific failure.

## Reviewer Decision

`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

SPARC may proceed only if the frozen proposal remains unchanged and the
rebuttal resolves these issues through executable constraints rather than
post-result discretion.
