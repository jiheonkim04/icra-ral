# LIFT-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Frozen proposal: `reports/lift_vla/researcher_proposal.md`

Proposal hash:
`3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F`

Reviewer attack: `reports/lift_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Rebuttal decision: `LIFT_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Researcher A accepts every essential Reviewer B constraint. None changes the
central method, adds a policy, expands the guidance-scale search, or uses a
confirmatory identity.

## Accepted Novelty Boundary

LIFT will claim only an empirical and mechanistic VLA action-flow result:

> pathwise language guidance through a continuous SmolVLA action flow can
> differ from and improve over final-action CAG under matched branches and
> field-evaluation budget.

LIFT will not claim a new CFG equation, flow-matching objective, numerical
integrator, sampler, or general guidance framework. Per-denoising-step language
guidance in released VLA work remains a mandatory related-work audit before a
paper novelty claim. Practical equivalence to CAG is a valid design kill.

## Accepted Benchmark Gate

The existing repository-wide offline counterfactual reports will not be used as
the LIFT benchmark. They are explicitly `offline_proxy_only` and contain pairs
that do not satisfy the LIFT source gate.

Before rollout, LIFT will create a new persisted manifest that:

- uses same-scene task pairs only;
- verifies target objects, fixtures, receptacles, and predicates in each initial
  state;
- validates the counterfactual BDDL goal and success predicate, not just a text
  replacement;
- records target-grounding and task-success scorers;
- partitions task/reset identities into discovery, validation, and sealed
  confirmatory sets with zero overlap;
- rejects cross-scene, absent-object, impossible, and original-goal-only pairs.

Until official LIBERO-CF assets are obtained and checksum-verified, all local
results will be labeled a custom development or evaluation proxy. Failure to
construct a scoreable feasible manifest returns
`LIFT_DATA_OR_BENCHMARK_FAILURE`. Offline action separation alone cannot pass
the headroom gate.

## Accepted Native-Space CAG Definition

All four policies start from the same sampled native SmolVLA noise tensor.

The transparent CAG proxy will:

1. complete conditioned and empty-language flows in native
   `B x H x D` SmolVLA action space;
2. mix those native completed chunks once;
3. apply the same action unpadding, normalizer/postprocessor, and 7D LIBERO
   bridge used by every other policy.

It will not mix separately clipped or postprocessed 7D actions. If the CAG paper
does not specify noise coupling, same-noise coupling will be disclosed as the
local variance-control choice rather than attributed to the authors.

## Accepted Matched-Compute Ablation

`lift_last_step_only_ablation` will compute both conditioned and
empty-language fields at all ten flow steps. For steps `0,...,8`, it discards
the empty-language field and performs the Base conditioned update. At step `9`,
it applies the same guidance equation as full LIFT.

This keeps `20` field evaluations for both full and ablation while changing
only where guidance affects the trajectory. No additional ablation or policy is
introduced.

## Accepted Practical-Separation Audit

The mathematical audit and preregistration will freeze:

- repeated same-noise numerical error;
- native full-chunk and executed-first-action separation metrics;
- a practical threshold derived from discovery-only action scale and numerical
  error, then held fixed for validation and confirmatory testing;
- per-step field-difference norms and cosine relations;
- LIFT-versus-CAG and LIFT-versus-ablation chunk separation;
- target-aware grounding or success consequences.

Action L2 alone may not establish mechanism or select a configuration. Falling
below the practical threshold returns
`LIFT_DESIGN_FAILURE_PRACTICAL_EQUIVALENCE`.

## Accepted Headroom Requirement

Before validation search, Stage 0 must show that:

- Base fails meaningfully on scoreable feasible counterfactual pairs;
- final-action CAG leaves residual grounding or task-success failure;
- a target-aware diagnostic supports a plausible benefit from pathwise
  conditioning;
- guidance is not merely increasing action magnitude or gripper switching.

Language sensitivity alone does not count. Failure returns `LIFT_NO_HEADROOM`.

## Accepted Compute Gate

A load-only and one-chunk smoke will precede broad decoding. It must report peak
GPU allocation, latency, field-evaluation count, action validity, and Base
identity at `omega = 1`.

Sequential branch evaluation is allowed only as an implementation schedule for
the same equations and only if it matches the reference schedule within the
frozen tolerance. An unresolved 16GB or latency limit returns
`LIFT_COMPUTE_INFEASIBLE`, not a scientific kill.

## Accepted Controls

The first serious comparison remains exactly four policies:

1. `frozen_smolvla`
2. `training_free_cag_proxy`
3. `lift_full_pathwise_guidance`
4. `lift_last_step_only_ablation`

The strongest alternative explanation is final-action CAG. Standard LoRA is an
irrelevant experiment because no policy weights, data, or trainable parameters
change. A trained VA branch is useful future scale-up evidence, not essential
prototype evidence. Dynamic schedules, alternate null prompts, extra samplers,
and a fifth policy remain forbidden in this cycle.

## Mathematical Audit Requirements

The next artifact must formalize, before implementation:

- exact native and postprocessed tensor shapes;
- conditional and empty-language token/mask construction;
- shared-noise coupling;
- Base, CAG, LIFT, and matched-compute ablation equations;
- action unpadding, normalization, and bridge order;
- identity and numerical tolerances;
- practical-separation construction without test access;
- feasible-counterfactual source and scorer gate;
- peak-memory and latency gate;
- three-scale validation search and fixed selection score;
- exact stop classifications.

No implementation, broad decode, validation search, rollout, or confirmatory
access is authorized by this rebuttal alone.

