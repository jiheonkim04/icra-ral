# Simple Baseline Failure Patterns

Lesson: a method that only beats no-method is not enough. The route must beat the strongest trivial baseline for its claimed failure mode.

## Baseline-Kill Table

| Route | Claimed failure mode | Baseline that killed or weakened it | Pattern |
| --- | --- | --- | --- |
| Target-Prior TCA-Map | target-conditioned action grounding | mean-action | action decoder weaker than a trivial action prior |
| CSS-Shield | semantic wrong-target safety | safety-only | semantic component did not add value beyond generic safety intervention |
| ExecSpec-Repair | executable-spec mismatch | diagonal affine | broad repair collapsed to simple calibration |
| AMP-GD | active disambiguating probes | informative-probe heuristic, random-probe, safety-only | active probe logic did not beat simple probe/safety behavior |
| ResetSpec-Retarget | object-pose/reset retargeting | global scale | state-dependent retargeting lost to action-only scaling |
| Phase-Locked Retiming | temporal phase retiming | gripper-only, fixed shift, repeat-last, linear warp, diagonal affine | each sub-failure was explained by a separate simple timing/action baseline |
| TL-ChunkRepair | temporal safety/property action-chunk repair | no-repair, clipping-only, safety-only, repeat-last/hold, fixed delay shift | symbolic violation reduction did not translate into replay/control utility |
| ContactTube-Aug | contact-preserving demonstration augmentation | simple object-relative translation retargeting | proposed augmentation was not controller-valid enough and preserved the contact tube worse than simple retargeting |
| PRISM-VLA | paraphrase-robust language-action consistency | canonicalization-only | stronger lexical normalization beat PRISM on primary held-out paraphrase and PRIDE metrics |

## Mandatory Early Baselines

Every new topic must predeclare and test the relevant subset of:
- mean-action,
- no-action/no-method,
- clipping-only,
- safety-only,
- nearest-target,
- random-probe,
- informative-probe heuristic,
- diagonal affine,
- global scale,
- nearest-demo,
- simple object-relative retargeting,
- random action jitter and random pose jitter for augmentation claims,
- canonicalization-only for language robustness and paraphrase robustness claims,
- exact-init expert replay upper bound,
- oracle/replay-leakage upper bound clearly labeled as invalid method evidence.

## Per-Failure-Mode Baseline Rule

A topic is invalid if each targeted failure mode can be solved by a separate obvious simple baseline. A method must beat:
- the best single simple baseline,
- the best per-failure-mode simple baseline,
- and the relevant raw/no-method controls.

Do not average across failure modes to hide that gripper-only, fixed-shift, linear-warp, nearest, safety-only, or calibration baselines solved their own slice.

## Symbolic/Proxy Utility Rule

A method is invalid for RA-L-stable continuation if it improves symbolic constraints, proxy scores, monitor satisfaction, or offline-only metrics while failing reward, success, safe-success, done/progress, or direct replay/control utility against a simple baseline.

## Data-Augmentation Validity Rule

A data-augmentation method is invalid for continuation if generated actions are not controller-valid or if a simple object-relative retargeting baseline preserves trajectory/contact metrics better. Do not train on augmented demonstrations after this failure; training would test learner robustness to invalid supervision, not augmentation value.

## Language Robustness Canonicalization Rule

A language-robustness method is invalid for continuation if canonicalization-only beats it on held-out paraphrase robustness, PRIDE, or difficulty-weighted robustness. A method is also invalid if it improves paraphrase consistency by weakening counterfactual object/target sensitivity.

## Anti-Pattern

Do not start from a clever method and add baselines later. Start from the strongest simple baseline, then ask whether a method can plausibly beat it within 48 to 72 hours.
