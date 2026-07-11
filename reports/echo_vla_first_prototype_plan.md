# ECHO-VLA First Prototype Plan

Status: candidate-headroom gate first. No ECHO heads are trained unless this gate passes.

## Scope

- backbone: frozen official SmolVLA-LIBERO
- max tasks in first gate: `2`
- reset identities per task: `2`
- candidate count: `4`
- horizon: `4`
- OpenVLA-OFT: not used
- full benchmark: not run
- full SmolVLA backbone training: not allowed

## Baselines To Use After Headroom

1. frozen_smolvla_default_candidate
2. random_candidate_selector
3. simple_phase_predicate_heuristic
4. direct_success_or_value_head
5. pre_vla_style_validity_advantage_proxy
6. echo_no_counterfactual
7. echo_no_phase
8. echo_full
