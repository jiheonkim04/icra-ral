# AMP-GD Experiment Plan

State 0 creates only concise task, risk, decision, and kill-gate files.

State 1 produces the first direct rollout/control metric. Priority A is a LIBERO/RoboSuite exact-init scene if intended and distractor object states plus safe active-probe response are observable. If robust active object-response observability is blocked, use a minimal local toy point-world diagnostic, label it as toy, and keep a concrete path back to LIBERO/RoboSuite.

State 1 baselines:
- no-probe greedy target choice,
- random-probe,
- safety-only/clipping-only,
- nearest-target heuristic,
- AMP-GD micro-probe.

Required metrics:
- target disambiguation accuracy,
- wrong-target rate,
- success rate,
- unsafe/collision rate,
- probe cost,
- extra path length or extra steps,
- utility drop versus no-probe,
- belief entropy reduction,
- intervention/probe rate,
- runtime overhead.

Minimum run: at least 20 randomized trials; prefer 50 or more when cheap. Use at least two target classes or two distractor configurations.

