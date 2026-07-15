# Epoch 4 Cycle 24 Candidate Generation

Date: 2026-07-16 KST

Exactly three candidates are evaluated under the active prior-first,
performance-oriented, minimum-sufficient-method governance. KITE repair or
rescue is not a candidate.

## Candidate 1: VDR-VLA

Name: `VDR-VLA`, Visuomotor Dynamic Residual alignment for VLA policies.

Contribution type: `PRIOR_EXTENSION`.

Closest positive prior: FutureVLA.

### Scientific Method

Train SmolVLA with a training-only dynamic future-feature residual objective.
For each discovery frame, a frozen visual encoder produces current and future
feature vectors. A discovery-only actionless ridge predictor estimates the
future feature change explainable from current visual state, proprioception,
language/task identity, and phase, but not generated actions. VDR subtracts
that actionless prediction from the actual future feature change and supervises
the policy to predict only the residual dynamic component from its generated
clean action chunk and current policy representation.

Unlike FutureVLA, VDR does not pretrain a new full joint architecture. Unlike
COVI, it does not reconstruct a complementary view or use occlusion labels.
Unlike KITE, it does not map cumulative commands to end-effector displacement.
Unlike PTC/CALA/RAR, it does not supervise future action latents or action
history residuals.

Mechanism chain:

`static-scene-dominated imitation -> action flow can match per-step actions
without representing visual consequences of motion -> unstable grasp or
placement dynamics -> closed-loop failure`

`dynamic-only future-feature residual alignment -> generated clean action
chunk must explain the motor-induced part of future visual change -> better
visuomotor temporal representation with bounded action changes -> improved
task success`.

### Quality Screen

- Provisional novelty: dynamic-only residual target differs from FutureVLA's
  full joint predictive architecture and from COVI/KITE/PTC closed routes.
- Prior anchor: FutureVLA reports positive downstream VLA improvements from
  joint visuomotor prediction and latent alignment.
- Data viability: local HDF5 demonstrations contain current/future images,
  proprioception, language/task identity, and action chunks; Stage 0A audits
  target variance and action-conditioned residual predictability before any
  training.
- Identity: zero-effect rank-4 LoRA or adapter; no inference module; Base
  passthrough audited before rollout.
- Decisive experiment: Base, transparent FutureVLA latent-alignment proxy,
  VDR full, no-action-residual ablation, and matched standard LoRA.

Score:

| Criterion | Weight | Score |
| --- | ---: | ---: |
| Provisional novelty | 25 | 23 |
| Problem importance | 15 | 14 |
| Positive prior anchor | 20 | 18 |
| Technical mechanism | 20 | 19 |
| Data/supervision feasibility | 10 | 9 |
| Decisive experiment feasibility | 10 | 9 |
| Total | 100 | 92 |

## Candidate 2: HIC-VLA

Name: `HIC-VLA`, History-Intent Commitment for aliased VLA chunks.

Contribution type: `PRIOR_EXTENSION`.

Closest positive prior: IntentVLA.

### Scientific Method

Learn a compact history-intent representation from recent legal observation
history and use it to condition SmolVLA chunk generation through an
identity-preserving gate. The local claim is narrower than IntentVLA: improve
only cases where current observations are aliased but recent history selects a
consistent local continuation.

The first audit would search discovery/validation data for noncollapsed
aliasing pairs where current visual-proprioceptive features are near while
future action chunks differ and recent history resolves the difference.

### Quality Screen

- Provisional novelty: differs from CIRR/ACoT by using recent observation
  history commitment rather than counterfactual coarse action intents.
- Prior anchor: IntentVLA has a strong positive paper result and public
  AliasBench task code, but the model code is currently forthcoming.
- Data viability: local LIBERO may not contain enough true aliasing contrast;
  this must be audited before implementation.
- Identity: possible with zero-initialized gate and Base passthrough.
- Decisive experiment: Base, IntentVLA-style history proxy, HIC full,
  no-history ablation, and an action-history-only simple baseline.

Score:

| Criterion | Weight | Score |
| --- | ---: | ---: |
| Provisional novelty | 25 | 20 |
| Problem importance | 15 | 14 |
| Positive prior anchor | 20 | 19 |
| Technical mechanism | 20 | 17 |
| Data/supervision feasibility | 10 | 7 |
| Decisive experiment feasibility | 10 | 8 |
| Total | 100 | 85 |

## Candidate 3: ALC-VLA

Name: `ALC-VLA`, Algebraic Latent Consistency for SmolVLA flow training.

Contribution type: `PRIOR_EXTENSION`.

Closest positive prior: ALAM.

### Scientific Method

Construct local frozen visual transition embeddings from frame triplets and
regularize them with composition and reversal consistency. During policy
training, co-generate action chunks and low-dimensional transition embeddings
with a joint flow objective, using no latent-to-action decoder at inference.

The local version would be intentionally smaller than ALAM and would compare
against an ALAM-style transition proxy under the same SmolVLA scaffold.

### Quality Screen

- Provisional novelty: weak for this campaign because PTC already explored
  local transition-latent action generation and CALA explored latent-action
  conditioning.
- Prior anchor: ALAM is strong and reports positive MetaWorld, LIBERO, and
  real-world gains, but no official code was verified locally.
- Data viability: frame triplets exist locally, but reliable visual transition
  embeddings require a stronger visual-feature implementation than a cheap
  proxy.
- Identity: possible through training-only co-generation with no inference
  module.
- Decisive experiment: Base, ALAM-style proxy, ALC full, no-algebraic
  transition ablation, and standard LoRA.

Score:

| Criterion | Weight | Score |
| --- | ---: | ---: |
| Provisional novelty | 25 | 14 |
| Problem importance | 15 | 13 |
| Positive prior anchor | 20 | 19 |
| Technical mechanism | 20 | 16 |
| Data/supervision feasibility | 10 | 7 |
| Decisive experiment feasibility | 10 | 7 |
| Total | 100 | 76 |

## Selection

Select exactly one candidate: `VDR-VLA`, `92 / 100`.

VDR has the strongest balance of positive prior anchor, local data viability,
identity-preserving integration, and novelty separation from the most recent
closed routes. HIC is promising but may fail the aliasing-contrast audit before
training. ALC has the strongest external prior but is too close to the
already-closed transition/latent-action family in this campaign.

## Baseline Rationale

| Comparison | Scientific question |
| --- | --- |
| Base vs VDR | Does dynamic visuomotor residual alignment improve SmolVLA? |
| FutureVLA proxy vs VDR | Does subtracting actionless static prediction improve over full future-latent alignment? |
| No-action-residual ablation vs VDR | Is generated-action conditioning necessary for the residual target? |
| Standard LoRA vs VDR | Is any gain explained by ordinary adaptation on the same demonstrations and steps? |

The first serious comparison contains exactly five policies: Base,
transparent FutureVLA latent-alignment proxy, VDR full, no-action-residual
ablation, and standard LoRA.
