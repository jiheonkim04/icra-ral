# AMP-GD Kill Criteria

Continue only if State 1 produces a rollout/control metric and AMP-GD:
- reduces wrong-target rate relative to no-probe and random-probe,
- beats safety-only/clipping-only on target disambiguation or wrong-target rate,
- keeps probe/utility cost bounded,
- commits after probing rather than reducing errors by stopping,
- yields interpretable failure cases,
- has a credible path from toy control evidence to LIBERO/RoboSuite.

Kill or reframe immediately if:
- no rollout/control metric is produced,
- no-probe or nearest-target is near-perfect,
- random-probe matches AMP-GD,
- safety-only/clipping-only matches AMP-GD,
- micro-probing only helps by avoiding commitment,
- probe cost destroys utility,
- evidence is purely synthetic with no path to a real simulator,
- any result depends on eval labels, BDDL oracle target fields, task IDs, filenames, manifest targets, native VLA competence, GPU, downloads, or OpenVLA-OFT.

