# Next Topic Selection Criteria

Any new topic must satisfy all requirements before implementation:
- real rollout, replay, or direct control metric within 24-48 hours,
- strong simple-baseline suite specified before method implementation,
- reason why per-failure-mode simple heuristics cannot solve the target failures,
- direct robotics metric, not offline proxy only,
- plausible path to multi-task and multi-model evaluation,
- novelty against recent VLA/action/safety/deployment papers,
- kill criteria defined before implementation.

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
- it proposes language or paraphrase robustness but canonicalization-only beats it on held-out paraphrase robustness, PRIDE, or difficulty-weighted robustness.
- it improves paraphrase consistency by weakening counterfactual object/target sensitivity.

## Baseline Gate

A method must beat:
- the best single simple baseline,
- the best per-failure-mode simple baseline,
- and the relevant no-method/raw/negative controls.

Passing only against the weakest baseline is a kill condition, not progress.

Symbolic or proxy improvement is also a kill condition when reward, success, safe-success, done/progress, or direct replay/control utility does not beat simple baselines.

Data-augmentation topics must additionally beat random action jitter, random pose jitter, image-only/metadata-only augmentation where applicable, and simple object-relative retargeting before training. Invalid augmented actions are a kill condition, not a reason to train a stronger learner.

Language-robustness topics must additionally beat canonicalization-only on held-out paraphrase robustness and preserve counterfactual object/target sensitivity. Consistency-only gains are not enough.

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
