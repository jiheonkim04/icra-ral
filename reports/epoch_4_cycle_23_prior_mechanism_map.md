# Epoch 4 Cycle 23 Prior And Mechanism Map

Date: 2026-07-15 KST

## Preserved Boundary

HASTE-VLA is closed unchanged as `HASTE_STAGE_0A_IMPLEMENTATION_FAILURE`.
Cycle 23 may not repair its serializer, rerun its event labels, reuse its
hazard target, or reinterpret it as a scientific result. HEST smoothing,
HASTE event timing, RAR action-history memory, COVI cross-view reconstruction,
and prior candidate-reranking routes also remain closed.

## Positive Primary Priors

### GeoPredict

Primary source: https://arxiv.org/abs/2512.16811

Project page: https://jingjingqian75.github.io/GeoPredict-Page/

GeoPredict reports consistent gains on RoboCasa Human-50, LIBERO, and
real-world manipulation by using predictive robot kinematics and 3D Gaussian
geometry as training-time supervision. Its trajectory module predicts
multi-step 3D robot-arm keypoints from policy representations. A public project
page is available; no official training code or checkpoint was verified for
the local audit. The fair local baseline is therefore a transparent
kinematics-only proxy, not an official reproduction.

### ACoT-VLA And ERVLA

Primary sources:

- https://arxiv.org/abs/2601.11404
- https://arxiv.org/abs/2606.03784

Official ACoT code: https://github.com/AgibotTech/ACoT-VLA

ACoT-VLA reports `98.5%`, `84.1%`, and `47.4%` success on LIBERO,
LIBERO-Plus, and VLABench using coarse action intents plus implicit action
reasoning. ERVLA reports `86.9%` on LIBERO-Plus and `53.2%` on VLABench using
action-grounded reasoning as representation-shaping supervision with
reasoning dropout. These are positive anchors for action-intent supervision,
but a new candidate must differ from positive-only coarse trajectory
prediction.

### ManiFlow And FreqPolicy

Primary sources:

- https://openreview.net/forum?id=etSYDtRO0Z
- https://arxiv.org/abs/2506.08822

Official ManiFlow code: https://github.com/allenai/maniflow

ManiFlow reports strong gains from continuous-time flow consistency across
image, point-cloud, bimanual, and dexterous tasks. FreqPolicy imposes
frequency-domain consistency and reports evaluation on 53 tasks plus a VLA
integration on 40 LIBERO tasks without performance degradation. These are
positive anchors for continuous-flow consistency, but generic consistency or
frequency regularization is already prior art.

## Direct Novelty Boundaries

### StyleVLA

Primary source: https://arxiv.org/abs/2603.09482

StyleVLA applies kinematic consistency to autonomous-driving trajectories.
That is a cross-domain precedent for physics-informed trajectory supervision,
not a manipulation result. KITE must claim only the manipulation-VLA extension
that links generated action chunks to measured future end-effector state under
a discovery-fitted realization operator.

### FlowPolicy

Primary source: https://arxiv.org/abs/2412.04987

Official code: https://github.com/zql-kk/FlowPolicy

FlowPolicy already constrains velocity consistency along generative flows.
KITE may not claim generic flow straightness or few-step consistency. Its claim
axis is action-to-future-state realization.

## Local Discovery Audit

Four development task families used demonstrations `0..7` for discovery and
`8..9` for validation. For horizons `5`, `10`, and `20`, an affine map from
cumulative six-dimensional commands to measured future `ee_states` displacement
reduced validation MSE over the discovery-mean predictor by:

| Horizon | Discovery rows | Validation rows | Relative MSE improvement |
| ---: | ---: | ---: | ---: |
| 5 | 2686 | 621 | 0.8715 |
| 10 | 1053 | 242 | 0.8918 |
| 20 | 502 | 115 | 0.9168 |

This audit used no reward, success, done flag, simulator rollout, or
confirmatory identity. It establishes that the action-to-state realization
target is noncollapsed and locally learnable before proposal selection.

## Failure-Informed Design Rules

1. Use structured JSON serialization helpers from the beginning; never hash
   raw NumPy objects.
2. The scientific method must survive removal of the word LoRA.
3. Use one core action-realization objective and no decorative divergence.
4. Keep the auxiliary path training-only and the inference action path
   identical to Base.
5. Put the closest prior in the first serious comparison.
6. Require an executable data/headroom gate before adapter training.
