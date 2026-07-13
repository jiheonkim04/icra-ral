# PSE-VLA Reviewer B Attack

Date: 2026-07-12 KST

Reviewed fixed proposal hash: `3F15D6E3ADCF340C490FBD5656051DFD101136D592F5A6B5D773ABF0E5308CAD`

## Independent Search

Primary sources checked:

- TTT-VLA: https://arxiv.org/abs/2606.03127
- Domain Arithmetic / DART: https://arxiv.org/abs/2607.00666
- Test-Time Perturbation Learning with Delayed Feedback / PDF: https://arxiv.org/abs/2604.18107
- CoVer-VLA / Scaling Verification: https://arxiv.org/abs/2602.12281
- Better Aggregation in Test-Time Augmentation: https://arxiv.org/abs/2011.11156

## Novelty Attack

PSE is close to old computer-vision test-time augmentation and to recent VLA deployment-time adaptation. The proposal is weak as a paper unless closed-loop evidence shows that action-space averaging across fixed photometric views improves manipulation success beyond the best single transformed view.

The closest VLA threat is PDF because it uses augmented observations and action voting at test time. PSE differs only if it remains a no-training, continuous-action, frozen-policy ensemble with no perturbation head, no delayed feedback, no uncertainty scheduler, and no logit/action voting classifier. If PSE needs feedback, scoring, learning, or adaptive transform selection, the novelty collapses into PDF/TTT/verification territory.

TTT-VLA and DART are broader deployment adaptation methods. They are not exact duplicates because they update latent prompts or weights, while PSE changes only inference-time observation sampling and 7D action aggregation. They still set a strong reviewer expectation: a zero-training ensemble must beat simple single-transform and duplicate-ensemble baselines to be interesting.

## Simplest Equivalent Method

The simplest equivalent method is the best single photometric transform:

`a_t = first(pi(T_best(o_t), q_t, l))`.

The second simplest equivalent method is duplicate clean averaging:

`a_t = mean(first(pi(o_t, q_t, l)), first(pi(o_t, q_t, l)), first(pi(o_t, q_t, l)))`.

Both are required. If either matches full PSE, the method is not useful.

## Leakage And Triviality Risks

- Do not tune transform gains after Stage A.
- Do not choose transforms using held-out rollout success.
- Do not use reward, success, simulator state, object pose, future observations, or any feedback at inference.
- Do not let SmolVLA action queue state make the transform order a hidden intervention.
- Do not claim calibration or domain adaptation; no calibration set exists in PSE.
- Do not claim robustness unless the second condition explicitly tests photometric shift.

## Pre-Implementation Decision

Decision: `IMPLEMENTATION_ALLOWED_WITH_STRONG_KILLER_BASELINES`

Reason: PSE is fragile but not a near-exact duplicate across all axes. It is locally cheap, changes inference-time action generation, and can be cleanly killed by the best single transform and duplicate-clean aggregation baselines.
