# Final Topic Shortlist

This shortlist is literature-first. It is not approval to implement.

## Rank 1: Constraint-Validated Spline VLA Action Interface

Anchor: Spline Policy, ActionMap, OpenVLA-OFT, pi0/pi0.5, GR00T N1/N1.6.

Core gap: recent VLA action representations improve decoding, but fixed action chunks and unconstrained decoders still do not guarantee controller-valid, temporally resamplable, contact-aware execution. Spline Policy opens the representation direction, but the repo should only consider a topic if the first evidence is direct replay/control and controller-validity, not offline action L2.

Why different from killed routes: it is not target prior, not contact-set injection, not symbolic repair, not paraphrase consistency, and not ActionMap tuning. The claim would be a structured trajectory interface with controller-validity gates, and it must beat timing/retarget baselines before any VLA scale-up.

Evidence needed before implementation:
- exact task family and policy interface,
- primary replay/control metric,
- controller-valid action rate and clip rate,
- raw chunk/no-repair baseline,
- fixed shift, linear time warp, gripper-only, diagonal affine, global scale, and object-relative retargeting baselines,
- kill rule if any simple baseline matches the spline interface on replay/control or validity.

## Rank 2: Early Failure Detection With Evidence-Calibrated Stop/Retry

Anchor: VLA-FAIL, partial-observation adversarial VLA attacks, TTT-VLA, VLA-JEPA.

Core gap: VLA-FAIL pushes failure detection, but deployment needs to know whether a detector improves downstream safe utility at matched intervention cost, not merely whether it detects distribution shift.

Why different from killed routes: the first claim would be early detection and abstention utility, not action repair. It must beat no-repair, safety-only, clipping, nearest-demo, always-abstain, and detector-only baselines.

Evidence needed before implementation:
- failure taxonomy and intervention budget,
- early detection metric plus downstream safe-success/reward/progress,
- no-repair, safety-only, clipping, always-abstain, and nearest-demo controls,
- cross-task or cross-perturbation split,
- kill rule if safety-only or no-repair matches utility.

## Parked

- Declarative/procedural disentanglement: interesting, but PRISM-VLA makes canonicalization risk high.
- Phase-aware continual replay: PHASER is too recent and strong; novelty window is narrow.
- Semi-supervised/JEPA pretraining: scientifically attractive but too heavy for first evidence.
- Test-time latent prompt optimization: promising but likely interaction-heavy.
- One-step action generation: latest high-noise schedule paper already covers the obvious latency axis.

## Recommendation

Recommended topic for the next literature deep dive: `Constraint-Validated Spline VLA Action Interface`.

Do not implement yet. The next step is a one-page evidence contract: exact tasks, direct metrics, baselines, kill criteria, and minimum cross-model/dataset requirement. If that contract cannot identify a direct replay/control metric within 24-48 hours, reject the topic before coding.

