# AMP-GD Autopilot State

- branch: `codex/amp-gd-state2-libero-port`
- current stage: `KILL_OR_REFRAME`
- last completed stage: `STATE 2`
- current evidence level: `toy_audit_plus_tiny_libero_control_metric`
- toy rollout/control metric happened: `true`
- LIBERO/RoboSuite object-observable inventory happened: `true`
- LIBERO/RoboSuite micro-probe diagnostic happened: `true`
- trials: toy `60` per profile; LIBERO `1` task and `5` policy variants
- training happened: `false`
- loss computed: `false`
- GPU/download/heavy VLA/OpenVLA-OFT happened: `false`
- continue/kill decision: `kill_or_reframe`
- exact resume command: none for AMP-GD as a main RA-L route

State 2 killed or reframed AMP-GD as the main route. Toy evidence was matched by deterministic/entropy-greedy informative-probe heuristics, and the tiny LIBERO/RoboSuite diagnostic did not show AMP-GD value beyond simple baselines. Object observability was real and non-leaking, but the available scene exposed language-resolved object positions rather than an active ambiguity signal.
