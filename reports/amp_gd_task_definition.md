# AMP-GD Task Definition

Active Micro-Probe Goal Disambiguation asks a robot to delay commitment when language-conditioned target identity is ambiguous. The robot may either commit to a target or execute one small safe information-gathering action, update its target belief from the resulting observation/state, and then commit.

Inputs:
- current robot state or observation,
- instruction text,
- candidate semantic targets,
- distractor targets,
- safety constraints.

Outputs:
- commit action toward one target, or
- bounded micro-probe action plus a post-probe target commitment.

Primary claim target: active micro-probes reduce wrong-target decisions under target ambiguity better than no-probe, random-probe, safety-only/clipping-only, and nearest-target baselines, with bounded utility cost.

Inference-time rule: the policy may use observation/state, instruction text, candidate geometry, and probe observations. It must not use eval labels, BDDL oracle target fields, task IDs, filenames, manifest target fields, or hidden intended-target labels.

