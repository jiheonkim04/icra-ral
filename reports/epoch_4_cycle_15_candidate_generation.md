# Epoch 4 Cycle 15 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_LIFT_VLA`

Exactly three candidates were generated and scored under the active
performance-oriented governance. COVI remains closed without scientific kill,
and no COVI, RAR, CALA, G3P, scheduler, residual, or output-correction rescue is
included.

## Candidate 1: LIFT-VLA

Name: `LIFT-VLA`, Language-Induced Flow Transport for frozen SmolVLA.

Contribution type: `CROSS_DOMAIN_MECHANISM_TRANSFER`.

Closest external prior: Counterfactual Action Guidance (CAG) and LIBERO-CF,
https://arxiv.org/abs/2602.17659.

Positive external result: training-free CAG reports average gains of `9.7`
points in grounding and `3.6` points in success on LIBERO-CF; its separately
trained vision-action branch reports `15.5` and `8.5` point gains. The paper also
reports a `17.2` point average real-world success gain.

Official code/checkpoint status: the paper reports
https://vla-va.github.io/, but no usable official repository or benchmark assets
were found during this audit. The local closest-prior arm must be a transparent
implementation of the published training-free action-mixing rule.

### Scientific Method

Given shared noise `x_1`, observation `o`, instruction `l`, and empty instruction
`empty`, compute at every SmolVLA integration step:

- `v_c = v_theta(x_t, t, o, l)`;
- `v_u = v_theta(x_t, t, o, empty)`;
- `v_lift = v_u + omega * (v_c - v_u)`;
- `x_(t+dt) = x_t + dt * v_lift`.

The closest-prior arm independently completes the conditioned and unconditioned
flows from the same noise, then mixes final actions. The key ablation applies
field guidance only on the final denoising step. LIFT therefore tests whether
language must shape the complete action-transport path, rather than only the
final action or final field update.

Removing the phrase LoRA does not change this method: no policy adaptation is
part of the contribution.

### Low-Compute Parameterization

- frozen SmolVLA weights;
- two prefix caches and two vector-field evaluations per integration step;
- shared observation, noise, flow-step count, and postprocessor across Prior and
  Ours;
- no training checkpoint, LoRA, QLoRA, adapter, or auxiliary head.

Standard LoRA omission: generic fine-tuning does not test the inference-only
claim that pathwise language guidance improves over final-action CAG under a
matched two-branch budget.

### Quality Screen

Provisional novelty:

- The mechanism is not claimed as new classifier-free guidance.
- The narrow novelty is its transfer to continuous VLA action-flow transport
  and the matched test against CAG's completed-action mixing.
- It is more than a renamed scalar loss because it changes every latent action
  state traversed during inference.
- Novelty fails if full-path and final-action guidance are empirically or
  numerically equivalent.

Prior-anchor strength:

- CAG reports positive counterfactual grounding and success gains across
  multiple VLA families and real-world tasks.
- The exact training-free CAG equation can be implemented transparently even
  though official code was unavailable.
- Prior and Ours can share the same frozen backbone, task, reset, initial noise,
  branch count, flow steps, and postprocessing.

Mechanism plausibility:

- Problem condition -> an instruction conflicts with a visually dominant
  training-task shortcut.
- Intermediate failure -> language has weak influence on the action-flow vector
  field and the latent path enters the vision-prior basin early.
- Action behavior -> the completed chunk targets the familiar object or task.
- Closed-loop failure -> the policy executes the visually likely task instead
  of the requested feasible task.
- Proposed method -> amplify the conditional-minus-unconditional vector field
  from the shared latent state at every flow step.
- Intended internal change -> the complete latent action path remains aligned
  with instruction-induced flow rather than receiving only a terminal shift.
- Intended action behavior -> earlier and more coherent target-directed chunk
  changes with bounded validity.
- Expected result -> higher language grounding and task success than Base,
  final-action CAG, and last-step-only guidance.

Data and supervision viability:

- No training labels are required.
- Stage 0 must use discovery and validation identities only to establish
  nontrivial language sensitivity, same-noise branch reproducibility, valid
  empty-language preprocessing, and counterfactual headroom.
- Official LIBERO-Goal provides a local same-scene/different-goal development
  proxy. It must not be mislabeled as official LIBERO-CF.
- A paper claim on LIBERO-CF requires official assets or a frozen, independently
  validated local counterfactual manifest before confirmatory testing.

Identity-preserving integration:

- `omega = 1` analytically reduces to the conditional Base field.
- Guidance is bounded by action validity and development-only delta checks.
- No privileged signal or future observation is used.

Decisive experiment feasibility:

- Stage 0 can compare Base, final-action CAG, full-path LIFT, and last-step LIFT
  on paired offline action and flow diagnostics before rollout.
- A maximum of three predeclared guidance scales is enough for validation-only
  selection.
- Stage A requires only four matched policies and approximately ten paired
  episodes each.

Score:

- provisional novelty: `20 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `90 / 100`

## Candidate 2: MOTAL-VLA

Name: `MOTAL-VLA`, Motion-Token Alignment for low-compute VLA adaptation.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Robotic VLA Benefits from Joint Learning with Motion
Image Diffusion, https://arxiv.org/abs/2512.18007.

Positive external result: the prior reports `97.5%` average LIBERO success,
`58.0%` RoboTwin success, and a `23%` real-world improvement from joint action
and optical-flow motion-image learning.

Official mechanism and mismatch: the reproducible mechanism is dense
optical-flow supervision coupled to action learning through a shared VLM. The
published method uses a large DiT motion head, DROID warm-up, full-model joint
training, and eight H200 GPUs. A local lightweight implementation would be a
transparent proxy, not an official reproduction.

### Scientific Method

Predict one compact horizon-aligned motion token from the policy representation
and align it to cached optical-flow features while retaining the normal action
flow objective. The token is training-only; the normal action path remains at
inference. The key ablation uses the same adaptation scaffold and action data
without motion alignment.

### Low-Compute Parameterization

- cached RAFT-style optical-flow features from discovery/validation frames;
- one small projection head and a fixed default SmolVLA LoRA scaffold;
- frozen backbone except the declared adapter targets;
- one bounded capacity diagnostic only if the adapter fails the subset-fit gate.

Standard LoRA is required as the conditional fifth policy because Ours updates
weights, uses extra supervision, and ordinary data-matched adaptation is a
plausible explanation.

### Quality Screen

Provisional novelty:

- The compact token is a meaningful low-compute extension, but much of the
  scientific idea belongs to the closest prior.
- Novelty collapses if a generic auxiliary target or standard LoRA gives the same
  result.

Prior-anchor strength:

- Strong positive multi-benchmark prior.
- Local fidelity is limited by the published method's `400M` motion head,
  pretraining, and full optimization budget.

Mechanism plausibility:

- Sparse action imitation may not encode pixel-space dynamics.
- Horizon-aligned flow supervision can make the representation sensitive to
  task-relevant motion and improve temporal control.

Data and supervision viability:

- Future frame pairs exist locally and optical flow can be cached without use at
  inference.
- Stage 0 must establish noncollapsed motion, task/phase coverage, subset fit,
  objective-scale balance, and no split leakage.

Identity-preserving integration:

- LoRA and the motion projection initialize to preserve Base behavior.
- Clean-retention loss and action-delta limits are required.

Decisive experiment feasibility:

- Offline Stage 0 is feasible.
- A faithful prior proxy and enough joint adaptation to reproduce the positive
  mechanism are substantially harder than the LIFT comparison.

Score:

- provisional novelty: `18 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `82 / 100`

## Candidate 3: GRAFT-VLA

Name: `GRAFT-VLA`, Geometry-Referenced Action Flow Tokens for low-compute VLA
adaptation.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: GeoPredict,
https://arxiv.org/abs/2512.16811.

Positive external result: GeoPredict reports `96.5%` average LIBERO success
versus `93.9%` for its reproduced base, with additional positive RoboCasa
Human-50 and real-world results.

Official mechanism and mismatch: GeoPredict predicts multi-step robot keypoint
trajectories and future 3D Gaussian geometry, using depth rendering as
training-only supervision. A local single-token geometry target would be a
transparent minimum-sufficient extension, not an official reproduction.

### Scientific Method

Use simulator-derived training-only workspace geometry to supervise a compact
geometry query that conditions SmolVLA action-flow features. The key ablation
uses the identical adaptation scaffold without geometry supervision.

### Low-Compute Parameterization

- cached development-only depth/keypoint targets;
- one geometry query and small decoder;
- fixed default SmolVLA LoRA scaffold with a frozen base;
- no 3D decoder or privileged input at inference.

Standard LoRA is required because the method changes policy weights, adds
supervision, and must rule out ordinary adaptation with the same demonstrations
and compute.

### Quality Screen

Provisional novelty:

- A compact geometry-referenced flow token differs from full GeoPredict.
- It is nevertheless close to prior local G3P and future-geometry routes, so the
  burden of showing a noncollapsed new mechanism is high.

Prior-anchor strength:

- Strong positive same-benchmark prior with a continuous action-flow backbone.
- The local proxy omits GeoPredict's trajectory hierarchy and predictive 3DGS,
  weakening fidelity.

Mechanism plausibility:

- 2D-reactive action features can miss workspace clearance and depth.
- Predictive geometry supervision could make the action representation encode
  task-relevant 3D structure and improve precise manipulation.

Data and supervision viability:

- Local records do not directly include the full calibrated future depth and 3D
  keypoint labels.
- Simulator replay is possible but costly, and the earlier G3P point-label
  failure raises a concrete collapse risk.

Identity-preserving integration:

- Query influence and LoRA initialize at Base behavior.
- Privileged depth, keypoints, and simulator state are training-only.

Decisive experiment feasibility:

- A source and label audit is possible.
- Producing a fair prior proxy, healthy labels, and a decisive rollout within one
  RTX 5080 budget is less feasible than either alternative.

Score:

- provisional novelty: `17 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `77 / 100`

## Selection

`LIFT-VLA` is selected with `90 / 100`.

It has the strongest combination of positive prior, same-backbone matched
comparison, minimum-sufficient implementation, identity preservation, and
decisive local experiment. Its novelty is deliberately narrow: transferring
pathwise classifier-free guidance into SmolVLA's continuous action-flow
transport and testing it against final-action CAG. Unknown empirical performance
is not a rejection reason.

## Baseline Rationale

| Comparison | Scientific question |
| --- | --- |
| Base vs Ours | Does full-path language-flow guidance improve frozen SmolVLA? |
| Prior vs Ours | Does guiding the evolving flow path improve over CAG's final-action mixing under matched two-branch inference? |
| Ablation vs Ours | Is guidance across the complete transport path necessary, rather than only the final flow step? |

No fifth policy is retained. Standard LoRA is irrelevant to an inference-only,
frozen-backbone mechanism, and another inference baseline would duplicate the
question already answered by the closest prior or key ablation.

## Frozen Next Gate

Researcher A must now write one bounded proposal. Before rollout it must prove,
on discovery/validation identities only:

- conditioned and language-dropped branches use identical observations, noise,
  integration steps, and postprocessing;
- `omega = 1` reproduces Base within a frozen tolerance;
- conditional-minus-unconditional fields are finite and noncollapsed;
- full-path LIFT differs from both final-action CAG and last-step-only guidance;
- all action dimensions remain valid and deltas are bounded;
- a local same-scene/different-goal proxy has meaningful Base and CAG headroom;
- no official LIBERO-CF equivalence is claimed without benchmark assets;
- no confirmatory identity is decoded during development or validation search.

