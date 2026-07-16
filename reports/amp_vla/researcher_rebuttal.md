# AMP-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `AMP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Researcher A accepts Reviewer B's conditional pass in full. AMP remains a
single-mechanism method: identity-preserving action-manifold projection for
SmolVLA action-flow adaptation. It does not rescue RAP, VDR, KITE, HEST,
HASTE, IARC, FAMR, PCAV, SPARC, NICE, COVI, LIFT, or EAC.

## Accepted Constraints

### ABot-M0 Proximity

AMP will not claim to invent action manifolds or broad efficient robot learning.
The novelty claim is narrowed to frozen-SmolVLA identity-preserving
manifold-constrained residual adaptation under the existing LIBERO/SmolVLA
interface.

The first serious comparison remains exactly:

1. `smolvla_base`;
2. `abot_m0_action_manifold_proxy`;
3. `amp_full`;
4. `amp_no_manifold_projection`;
5. `standard_lora`.

Before executable Stage 0, AMP will check whether official ABot-M0 assets can
be integrated within local budget. If not, policy 2 remains explicitly named a
transparent local proxy, not an official ABot-M0 reproduction. Every deviation
from official ABot-M0 action manifold learning must be listed before Stage 0
execution.

### Projection Versus Clipping

AMP accepts that projection can be confused with clipping. Stage 0 and later
validation must therefore report both action-bound validity and manifold
consistency.

The hard deployment validity gate is postprocessed 7D LIBERO action validity.
Raw normalized action validity remains a scale and serializer diagnostic. AMP
will also include a clipping or bound-only diagnostic whenever projection could
be explained by coordinate clipping.

No clipping rescue, bound widening, unit-system switch, threshold change, or
post-hoc validity reinterpretation is allowed after Stage 0 begins.

### Manifold Health

AMP stops before rollout if the action manifold is collapsed, nonpredictive, or
equivalent to task/phase means.

Stage 0 must persist:

- retained dimension;
- explained variance;
- reconstruction Huber;
- per-task and phase coverage;
- coordinate variance;
- task/phase mean action baseline;
- task/phase mean coordinate baseline;
- duplicate, missing, extra, and split-overlap audits.

The manifold reconstruction must beat the task/phase mean action predictor by
the frozen margin, and deployment-input coordinate prediction must beat the
task/phase coordinate predictor by the frozen margin.

### ABot Proxy Headroom

AMP accepts that beating Base alone is insufficient. The transparent ABot-M0
proxy remains the closest-prior comparator and must be frozen before AMP
validation performance is known.

The proxy will match AMP's action normalization, action chunking, manifold
fitting data, inference budget, action postprocessing, task/reset manifest, and
projection metric as closely as locally possible. If the ABot proxy matches or
beats AMP in the first serious comparison, AMP does not become a paper
candidate.

### Standard LoRA

Matched standard LoRA remains mandatory. It receives the same demonstrations,
optimizer steps, rank, target modules, clean-retention coefficient where
applicable, ordinary flow objective, and checkpoint-selection budget. If
standard LoRA matches or beats AMP, AMP does not become a paper candidate.

### Identity-Preserving Integration

AMP will initialize and disk-reload to exact Base behavior within `1e-6`.
Projection and residual influence must be bounded and relevant-state selective.

Mechanism smoke must report:

- Base action;
- AMP action;
- projection delta;
- residual norm;
- gate value;
- changed dimensions;
- activation context;
- action validity;
- translation, rotation, and gripper deltas.

If AMP catastrophically changes all actions or acts everywhere, it stops as
`AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

### Objective Engineering

The mathematical audit must define `Phi`, `DecodeManifold`, `P`, `P_mix`,
`lambda_m`, `lambda_p`, `lambda_clean`, tensor shapes, units, scale estimates,
gradient paths, and gradient norm ratios.

AMP will not use KL between deterministic actions or SmolVLA flow vectors. If a
projection operator is nondifferentiable, the audit must specify where gradients
flow and which differentiable surrogate supervises trainable parameters.

### Validation Score

Offline action L2 or Huber alone cannot select the final configuration. The
validation score must include clean retention, postprocessed action validity,
mechanism activation, AMP-minus-ABot-proxy margin, and AMP-minus-ablation
margin. All tried configurations and negative results must be saved.

Confirmatory outcomes may not retune latent dimension, projection strength,
thresholds, coefficients, task identities, reset identities, or baselines.

## Rebuttal To Reviewer Concerns

The strongest reviewer concern is fair: an action manifold prior is already an
ABot-M0 claim axis. AMP therefore narrows the contribution to the setting where
SmolVLA already has a pretrained action flow and the new mechanism constrains
only the adapter-induced residual around that Base behavior.

The second key concern is that projection can degrade into clipping. AMP accepts
that a range-only improvement is not a paper mechanism. Stage 0 must separate
bound validity, clipping diagnostics, and demonstration-support consistency
before any rollout.

The third concern is that action manifolds can collapse into phase/task means.
AMP accepts task/phase action and coordinate baselines as hard gates. If the
deployment-observable coordinate probe cannot beat those baselines, the method
stops as a design or data failure rather than proceeding to validation search.

## Decision

AMP proceeds to mathematical mechanism audit and preregistration. The audit must
formalize:

- action-manifold variables and source partitions;
- ABot-M0 proxy status and deviations;
- projection, clipping diagnostic, and no-projection ablation;
- identity-preserving residual/gate initialization;
- objective formulas, scales, tensor shapes, units, and gradients;
- action-validity unit system and hard gates;
- bounded validation search;
- Stage 0 stop classes.

No training, validation search, rollout, simulator access, or confirmatory-test
access is authorized by this rebuttal.
