# ECHO-VLA First Prototype Decision

Final decision: `NO_ECHO_CANDIDATE_HEADROOM`

## Basis

- novelty adjudication: `ECHO_NOVELTY_SURVIVES_TARGETED_GATE`
- candidate oracle headroom: `{'group_count': 4, 'candidate_count_total': 16, 'default_success_rate': 0.0, 'oracle_success_rate': 0.0, 'oracle_improvement_pp': 0.0, 'default_failure_group_count': 4, 'default_failure_recoverable_count': 0, 'default_failure_recoverable_rate': 0.0, 'materially_better_group_count': 0, 'passes_headroom_gate': False, 'hard_kill_reason': 'oracle improvement <10pp or fewer than 15% of default-failure states contain a successful/materially better candidate'}`
- data generated: `{'same_state_intervention_groups': 4, 'candidate_records': 16, 'tasks': [{'suite': 'libero_spatial', 'task_id': 0, 'instruction': 'pick up the black bowl between the plate and the ramekin and place it on the plate'}, {'suite': 'libero_object', 'task_id': 4, 'instruction': 'pick up the ketchup and place it in the basket'}], 'reset_identities': [20260711, 20260712], 'candidate_count': 4, 'horizon': 4}`
- components trained: `none_headroom_gate_first`
- prototype baselines: `['frozen_smolvla_default_candidate', 'random_candidate_selector', 'simple_phase_predicate_heuristic', 'direct_success_or_value_head', 'pre_vla_style_validity_advantage_proxy', 'echo_no_counterfactual', 'echo_no_phase', 'echo_full']`
- closed-loop results: `not_run_headroom_gate_first`
- effect/ranking results: `not_trained`
- latency/VRAM: `{'elapsed_seconds': 140.301, 'cuda_memory': {'allocated_bytes': 935880704, 'max_allocated_bytes': 971650560, 'allocated_mb': 892.525, 'max_allocated_mb': 926.638}}`

## Exact Next Step

Stop ECHO implementation or redesign candidate generation/effect representation before training.

## Implementation Boundary

The branch added the ECHO protocol module, tests, WSL launcher, and headroom runner. It intentionally did not train the phase head, effect head, compatibility head, direct value baseline, Pre-VLA-style proxy, or ECHO ablations because the headroom gate is the required prerequisite for any training.
