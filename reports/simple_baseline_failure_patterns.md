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
| ContactSet-VLA | contact-set action-head geometry injection | active single 3D point, destination-only, no-geometry | richer point sets made the offline action head worse than simpler geometry or no geometry |
| ActionMap Anchor Reproduction | heatmap/candidate action decoding anchor | mean-action, cheap MLP | local ActionMap-style candidate head lost the first reproduction gate and collapsed candidate diversity |
| SafeTrace-VLA | temporal safety preference optimization | safety-only/risk-only monitor scoring, generic DPO/preference proxy | temporal preference labels were solved by simple monitor risk and generic preference optimization |

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
- safety-only/risk-only monitor scoring and generic DPO/preference labels for safety-preference claims.

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

## Action-Head Geometry Rule

An action-head geometry method is invalid for continuation if active single-point injection, source-only point injection, destination-only point injection, source+destination two-point injection, or no-geometry action-head baselines match or beat the richer geometry encoder on the first held-out action metric. Do not scale to VLA fine-tuning or replay when simpler geometry is already stronger.

## Action-Decoder Anchor Rule

A local action-decoder anchor is invalid for failure mining or extension work if mean-action, linear/L1, or cheap MLP action heads match or beat the anchor-style head on held-out 7D action L2, or if heatmap/candidate predictions collapse to trivial bins. An oracle candidate upper bound can show discretization headroom, but it is invalid as method evidence.

## Temporal Safety Preference Rule

A temporal safety preference method is invalid if safety-only/risk-only monitor scoring, stop-on-risk, or a generic DPO/preference proxy matches the proposed temporal preference objective. The method must also preserve measurable task utility on an official safety benchmark/source; local proxy risk labels alone are not RA-L-stable evidence.

## Anti-Pattern

Do not start from a clever method and add baselines later. Start from the strongest simple baseline, then ask whether a method can plausibly beat it within 48 to 72 hours.

Additional reset after SafeTrace-VLA: do not start a new custom method from local proxy diagnostics. Reproduce an official benchmark/source anchor first, then consider method design only after the official baseline is working.
