# ResetSpec-Retarget Kill Criteria

Continue only if:
- exact-init expert replay succeeds,
- default-reset or perturbed raw replay fails or degrades,
- object-relative retargeting improves success, reward, done index, or meaningful trajectory/object progress,
- object-relative retargeting beats diagonal-affine, global-scale, clipping, and feasible nearest-demo baselines.

Kill if:
- default-reset raw replay already succeeds,
- object poses or EEF poses are unavailable,
- object-relative retargeting does not improve replay/progress,
- diagonal-affine, global-scale, clipping, or nearest-demo replay matches or beats object-relative retargeting,
- only offline metrics appear without replay/control metrics,
- the result depends on eval labels, BDDL target fields, dataset target labels, task IDs, filenames, manifest target fields, OpenVLA-OFT, GPU, downloads, or paper-grade claims.

STATE 1 outcome: killed. Object-relative retargeting improved progress over default raw replay but did not beat the fixed global-scale baseline, which succeeded from default reset.
