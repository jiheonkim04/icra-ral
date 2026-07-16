# CFR-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `CFR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Proposal: `reports/cfr_vla/researcher_proposal.md`

Proposal hash:
`9E2FC510B2D97C869F18BE6C5B339CE034DD98223802078358320AA8BEF3D0AE`

Reviewer attack: `reports/cfr_vla/reviewer_attack.md`

Reviewer decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Researcher A accepts all Reviewer B conditions. CFR remains live only under the
narrowed formulation below.

## Accepted Novelty Boundary

Accepted condition:
`CFR_NOVELTY_NARROWED_TO_CONTINUOUS_BASE_START_IDENTITY_REFINEMENT`.

CFR does not claim to invent iterative action refinement, full-sequence
correction, or flow refinement for robot actions. DFM-VLA owns the closest
positive prior: iterative full-sequence refinement via discrete flow matching
over action tokens.

CFR's claim is narrowed to:

- continuous `[50,7]` SmolVLA action chunks;
- starting from a frozen Base decoded chunk;
- bounded residual velocity refinement before execution;
- exact identity-preserving initialization and disk reload;
- local demonstration supervision without privileged inference inputs;
- matched early comparison against DFM-VLA or a transparent DFM proxy.

Any report, mathematical audit, preregistration, implementation, or paper draft
that states a broader claim is invalid.

## Accepted DFM Prior Policy

`dfm_vla_continuous_refinement_proxy` remains policy 2 unless official DFM-VLA
code/assets are locally installed and verified before the first serious
comparison. If official DFM-VLA becomes available before confirmatory testing,
it replaces or augments the proxy without using confirmatory outcomes to retune
CFR.

The proxy must implement iterative full-sequence refinement. A one-shot
residual predictor cannot serve as the DFM proxy. It may only serve as
`cfr_no_iterative_refinement`, the key ablation.

CFR must beat the DFM proxy or official DFM-VLA on the matched claim axis before
paper viability.

## Accepted Simple-Killer And Ablation Policy

The first serious comparison remains:

1. `smolvla_base`;
2. `dfm_vla_continuous_refinement_proxy` or official `dfm_vla` if installed;
3. `cfr_full`;
4. `cfr_no_iterative_refinement`;
5. `standard_lora`.

`cfr_no_iterative_refinement` is the key ablation. `standard_lora` is the
single simple reviewer-killer baseline because CFR uses trainable adapter
infrastructure. Offline terminal action Huber alone cannot select, validate, or
kill CFR. Mechanism evidence must show stepwise refinement, not a hidden
one-shot terminal residual.

## Accepted Action-Validity Policy

Before any CFR Stage 0 execution, the mathematical audit and prototype protocol
must define action validity from official LIBERO / SmolVLA postprocessor
semantics and environment action limits. The definition must not be inherited
blindly from RAP or AMP if that definition is inconsistent with official Base
behavior.

The same frozen action-validity definition must apply to Base, DFM proxy,
`cfr_no_iterative_refinement`, `standard_lora`, and CFR. If Base fails before
CFR acts, the result is an implementation/optimization stop, not a scientific
kill. No clipping rescue or post-hoc bound change is allowed.

## Accepted Data And Supervision Gates

CFR may use expert future action chunks only as training targets on discovery
and validation demonstrations. At inference, it may use only legal current
observations, proprioception, language/task input, Base features/actions, and
learned parameters.

Required before bounded validation:

- residual targets are noncollapsed by action dimension, task, phase, and
  timestep;
- deployment-input refinement probes beat task/phase residual baselines;
- DFM proxy residual headroom is positive;
- the false-negative safeguard remains active for weak offline evidence.

Failure here is `CFR_STAGE_0_DATA_OR_SUPERVISION_FAILURE`,
`CFR_STAGE_0_NO_USABLE_HEADROOM`, or `CFR_STAGE_0_DESIGN_FAILURE`, not a
closed-loop scientific kill.

## Accepted Mathematical Audit Requirements

The mathematical audit must specify:

- exact variables and tensor shapes;
- objective formulas and units;
- small-batch term magnitudes;
- gradient norms and gradient paths;
- which paths use `stopgrad`;
- how unrolled refinement carries gradients;
- action-validity semantics;
- why Huber/vector-field consistency is used instead of JS, Wasserstein, MMD,
  Mahalanobis distance, KL, or trajectory discrepancy.

No KL may be computed directly between deterministic 7D action vectors or
SmolVLA flow vectors.

## Accepted Inference-Legality Rule

Accepted condition:
`NO_PRIVILEGED_INFERENCE_INPUTS_CONFIRMED`.

CFR cannot use future observations, rewards, success flags, done flags, object
poses, reset identities, simulator hidden state, or confirmatory outcomes at
inference. If any implementation path requires such signals, CFR must stop
before rollout.

## Rebuttal Decision

All Reviewer B conditions are accepted. CFR is not killed before implementation,
but it may proceed only to mathematical mechanism audit. No preregistration,
validation search, training, rollout, or confirmatory evaluation is allowed
until the mathematical audit freezes the objective, action-validity semantics,
proxy definition, gradient-scale checks, and ablation requirements.
