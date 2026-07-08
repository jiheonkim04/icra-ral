# Next Topic Selection Criteria

Current reset: no new custom method topic should be started from local proxy diagnostics. RA-L-stable work now requires reproducing an official benchmark/source anchor first.

Allowed next anchors:
- SafeManip official benchmark reproduction,
- LIBERO-Safety official benchmark reproduction,
- ForesightSafety-VLA reproduction,
- ActionMap reproduction,
- VLA-Corrector reproduction.

Recommended first anchor: SafeManip official benchmark reproduction, with LIBERO-Safety as fallback if SafeManip is source-blocked.

Any new topic must satisfy all requirements before implementation:
- real rollout, replay, or direct control metric within 24-48 hours,
- strong simple-baseline suite specified before method implementation,
- reason why per-failure-mode simple heuristics cannot solve the target failures,
- direct robotics metric, not offline proxy only,
- plausible path to multi-task and multi-model evaluation,
- novelty against recent VLA/action/safety/deployment papers,
- kill criteria defined before implementation.
- official anchor baseline reproduced before custom method design.

## Invalid Topic Rules

A topic is invalid if:
- its first result is offline-only,
- it depends on native VLA competence before verifying that competence,
- it needs full VLA training, OpenVLA-OFT, downloads, GPU, or heavy imports for the first result,
- it cannot produce a replay/control metric within 24-48 hours,
- it has no direct robotics evidence path,
- it is already solved by calibration, clipping, nearest, mean, random, safety, fixed-shift, gripper-only, linear-warp, or replay-leakage baselines,
- each targeted failure mode can be solved by a separate obvious simple baseline.
- it improves symbolic, proxy, monitor, or offline constraint satisfaction while failing direct replay/control utility against a simple baseline.
- it proposes data augmentation but generated actions are not controller-valid before training.
- it proposes contact/object-pose retargeting but simple object-relative retargeting preserves trajectory/contact metrics better.
- it proposes richer action-head geometry but active single-point, source-only, destination-only, source+destination, or no-geometry baselines match or beat it on the first held-out action metric.
- it proposes or relies on an action-decoder anchor but mean-action, linear/L1, or cheap MLP action heads match or beat the anchor-style head on held-out 7D action L2.
- it proposes or relies on a heatmap/candidate action head but candidate predictions collapse to trivial bins before replay/control evidence appears.
- it proposes language or paraphrase robustness but canonicalization-only beats it on held-out paraphrase robustness, PRIDE, or difficulty-weighted robustness.
- it improves paraphrase consistency by weakening counterfactual object/target sensitivity.
- it starts from local proxy diagnostics before reproducing an official benchmark/source anchor.
- it proposes temporal safety preferences while safety-only/risk-only monitor scoring, stop-on-risk, or generic DPO/preference labels match the method signal.

## Baseline Gate

A method must beat:
- the best single simple baseline,
- the best per-failure-mode simple baseline,
- and the relevant no-method/raw/negative controls.

Passing only against the weakest baseline is a kill condition, not progress.

Symbolic or proxy improvement is also a kill condition when reward, success, safe-success, done/progress, or direct replay/control utility does not beat simple baselines.

Data-augmentation topics must additionally beat random action jitter, random pose jitter, image-only/metadata-only augmentation where applicable, and simple object-relative retargeting before training. Invalid augmented actions are a kill condition, not a reason to train a stronger learner.

Action-head geometry topics must additionally beat active single-point injection, source-only, destination-only, source+destination, and no-geometry action-head baselines before replay scale-up or full VLA fine-tuning.

Action-decoder anchor topics must additionally beat mean-action, linear/L1, and cheap MLP baselines and pass a candidate-diversity/collapse check before failure mining or extension work.

Language-robustness topics must additionally beat canonicalization-only on held-out paraphrase robustness and preserve counterfactual object/target sensitivity. Consistency-only gains are not enough.

Temporal safety preference topics must additionally reproduce an official safety benchmark/source first and beat safety-only/risk-only monitor scoring, stop-on-risk, clipping-only, reward-penalty, and generic DPO/preference labels while preserving measurable task utility.

## Required First Table

Every new topic must predeclare:
- task and failure modes,
- method-free controls,
- strongest single simple baseline,
- per-failure-mode simple baselines,
- oracle/replay-leakage upper bounds clearly labeled invalid as method evidence,
- action validity, controller-valid action rate, and clip rate for any augmentation method,
- canonicalization-only and counterfactual sensitivity checks for any language robustness method,
- direct success/reward/done/progress/safety metrics,
- exact continue and kill criteria.
