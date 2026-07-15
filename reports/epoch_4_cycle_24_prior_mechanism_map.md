# Epoch 4 Cycle 24 Prior And Mechanism Map

Date: 2026-07-16 KST

## Preserved Boundary

KITE-VLA is closed unchanged as `KITE_STAGE_0A_IMPLEMENTATION_FAILURE`.
Its result is not a scientific kill, but Stage 0B, rerun, action clipping,
threshold change, coefficient change, and KITE rescue are forbidden. Cycle 24
must not repair KITE's reconstructed-action range issue or relabel its partial
results.

Closed or high-risk neighborhoods remain live boundaries:

- PTC/CALA latent-action or future-action conditioning;
- RAR action-history residual memory;
- COVI complementary-view reconstruction under occlusion;
- HEST/HASTE hybrid smoothing or event timing;
- KITE action-to-future-end-effector realization;
- PCAV candidate verification or progress filtering;
- generic LoRA/QLoRA adaptation as a contribution.

## Positive Primary Priors

### FutureVLA

Primary source: https://arxiv.org/abs/2603.10712

FutureVLA reports that joint visuomotor predictive modeling improves VLA
frameworks by decoupling visual state preservation from temporal action
modeling, using joint visuomotor gating and latent embedding alignment. Its
core positive result is that VLA training benefits when future prediction is
not dominated by static visual reconstruction and when motor dynamics are
aligned with visual constraints.

No official code or checkpoint was verified locally. A fair local comparison
therefore uses a transparent FutureVLA-style latent-alignment proxy, not an
official reproduction.

Mechanism map:

| Axis | FutureVLA |
| --- | --- |
| observation/input | current visual observations, robot state, language, heterogeneous manipulation data |
| representation | joint visuomotor embeddings with separated visual and motor streams |
| supervision | future predictive latent alignment and post-training embedding alignment |
| objective | predictive visuomotor embedding alignment, not raw action L2 alone |
| policy component changed | training representations; downstream inference architecture can remain unchanged |
| action-generation mechanism | better action generation through internalized temporal physical priors |
| inference-time intervention | none required by the post-training alignment claim |
| demonstrated causal link | predictive visuomotor pretraining/alignment improves downstream VLA performance |
| untested local link | whether a small SmolVLA-compatible dynamic-residual objective is observable and useful locally |

### IntentVLA

Primary source: https://arxiv.org/abs/2605.14712

Official repository: https://github.com/ZGC-EmbodyAI/IntentVLA

IntentVLA reports a short-horizon intent representation for aliased robot
manipulation. The current public repository releases AliasBench task code and
states that model code is forthcoming. The paper reports positive gains on
AliasBench, SimplerEnv, LIBERO-Long, and RoboCasa by conditioning chunk
generation on recent visual history rather than current-frame input alone.

Mechanism map:

| Axis | IntentVLA |
| --- | --- |
| observation/input | recent visual history, current observation, language |
| representation | compact short-horizon intent |
| supervision | imitation data with aliased current observations and consistent recent context |
| objective | history-conditioned chunk generation |
| policy component changed | VLA conditioning path for chunk generation |
| action-generation mechanism | preserve local intent commitment across adjacent replanning steps |
| inference-time intervention | history-conditioned policy, no external oracle |
| demonstrated causal link | better rollout stability on aliasing benchmarks and standard manipulation benchmarks |
| untested local link | whether local LIBERO demonstrations have enough noncollapsed aliasing contrast |

### ALAM

Primary source: https://arxiv.org/abs/2605.10819

ALAM reports algebraically consistent latent transitions from action-free
videos, regularized by composition and reversal, then transferred to VLA
learning through joint flow matching with robot actions. It reports large
reductions in transition-structure errors and positive downstream gains on
MetaWorld, LIBERO, and real-world manipulation.

Mechanism map:

| Axis | ALAM |
| --- | --- |
| observation/input | frame triplets, action-free video, downstream VLA demonstrations |
| representation | locally additive and reversible latent transition space |
| supervision | reconstruction plus composition and reversal consistency |
| objective | algebraic latent-transition pretraining plus joint flow matching |
| policy component changed | action and latent-transition co-generation |
| action-generation mechanism | structured transition geometry informs action generation |
| inference-time intervention | no latent-to-action decoder needed in the reported transfer |
| demonstrated causal link | algebraic latent transitions improve VLA policy learning |
| untested local link | whether a local SmolVLA version would differ enough from closed PTC/CALA routes |

### ManiFlow And FreqPolicy

Primary sources:

- https://arxiv.org/abs/2509.01819
- https://arxiv.org/abs/2506.08822

Official ManiFlow code: https://github.com/allenai/maniflow

ManiFlow reports improved manipulation performance from consistency flow
training and adaptive multimodal conditioning. FreqPolicy reports
frequency-domain consistency for efficient flow-based visuomotor policies.
They remain useful secondary priors for action-flow objectives, but generic
flow consistency and frequency consistency are crowded and close to HFC-style
routes from Cycle 23.

## Local Failure Synthesis

KITE established that discovery demonstrations contain learnable
action-to-state realization signal, but its prototype failed an implementation
range-validity gate before any optimizer step. The next method should keep the
useful lesson that action generation should be tied to physical consequences,
while changing the target away from end-effector realization and avoiding any
post-hoc repair of KITE's invalid reconstructed actions.

PTC, CALA, and RAR warn that future action latents or action-history labels
can be predictable mostly from trivial action baselines. A new method must
therefore audit whether the proposed dynamic signal is predictable from
deployment inputs and whether action-conditioned information beats an
actionless static predictor by a preregistered margin before training.

COVI warns that full visual reconstruction or complementary-view prediction
can become a heavy implementation target. A new visual prior should avoid
reconstructing entire future images and should target only the dynamic feature
component not explained by static scene context.

## Design Implication

Cycle 24 should prioritize a FutureVLA-anchored dynamic-residual formulation:
fit a frozen actionless visual predictor on discovery data, subtract it from
future visual-feature change, and train the policy only against the remaining
motor-explained dynamic residual. This changes the mechanism axis from KITE's
end-effector realization to dynamic visuomotor representation alignment, from
COVI's complementary-view reconstruction to dynamic-only residual prediction,
and from CALA/PTC future-action latents to future visual-feature consequences.
