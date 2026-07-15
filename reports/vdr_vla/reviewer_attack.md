# VDR-VLA Reviewer B Attack

Date: 2026-07-16 KST

Role: Reviewer B

Frozen proposal hash:
`0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Primary Sources Checked

- FutureVLA: https://arxiv.org/abs/2603.10712
- IntentVLA: https://arxiv.org/abs/2605.14712
- IntentVLA code: https://github.com/ZGC-EmbodyAI/IntentVLA
- ALAM: https://arxiv.org/abs/2605.10819
- ManiFlow: https://arxiv.org/abs/2509.01819
- ManiFlow code: https://github.com/allenai/maniflow
- FreqPolicy: https://arxiv.org/abs/2506.08822

## Novelty Attack

VDR is close to FutureVLA because both use future predictive visuomotor
supervision and avoid changing the deployed inference architecture. VDR is
also near COVI because both touch visual representations, and near KITE because
both supervise a future consequence of the generated action chunk.

Reviewer B does not kill it before implementation because the actual mechanism
is narrower and different:

- no full FutureVLA joint architecture or heterogeneous pretraining;
- no complementary-view reconstruction or occlusion condition from COVI;
- no end-effector state realization operator from KITE;
- no latent-action label, future action code, or action-history residual from
  PTC/CALA/RAR;
- no inference-time reranker, verifier, scheduler, residual correction, or
  clipping.

The defensible claim is only that subtracting a frozen actionless static
future-feature predictor yields a dynamic residual whose action-conditioned
prediction can improve SmolVLA.

## Baseline Attack

The following explanations must remain live:

- a FutureVLA-style full future-latent alignment proxy may match or beat VDR;
- the no-action-residual ablation may match VDR if generated actions add no
  information;
- standard LoRA may explain any gain because VDR updates policy weights on the
  same demonstrations;
- static scene/task/phase features may explain the residual target;
- a cheap visual-feature predictor may be too noisy to support policy training.

## Leakage Attack

VDR may use future visual features only for discovery, validation, and training
target construction. At inference it must not use future frames, future
features, future actions, success, reward, reset identity, simulator object
state, BDDL predicates, confirmatory-test identity, or held-out outcome.

The static predictor and PCA whitening statistics must be discovery-only. The
validation split may select one VDR coefficient but must not refit target
construction.

## Mathematical Attack

The dynamic residual is not a probability distribution. VDR may not use KL
between residual vectors, deterministic actions, or SmolVLA flow vectors. The
audit must define units, projection, whitening, Huber scale, gradient path, and
loss magnitudes. The action-conditioned residual probe must be compared against
an actionless residual probe before training.

## Implementation Attack

The method is invalid if:

- the visual-feature hook changes across rows or is not disk reproducible;
- future-feature targets leak into inference;
- PCA/whitening is fitted on validation or confirmatory identities;
- action summaries use expert future actions rather than reconstructed clean
  generated actions;
- action validity fails before rollout;
- adapter identity is not exact at initialization and reload;
- standard LoRA receives different data or optimizer steps.

## Reviewer Decision

VDR is conditionally allowed to proceed to rebuttal and mathematical audit.
It must narrow its claim, keep the FutureVLA proxy and no-action-residual
ablation live, require the action-conditioned residual probe to beat actionless
trivial predictors, and classify failed pre-rollout evidence as data,
headroom, design, or implementation failure rather than a scientific kill.
