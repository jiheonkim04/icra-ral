# PhaseBarrier-VLA Prototype Protocol

Date: 2026-07-12 KST

## Method

PhaseBarrier-VLA trains a phase-conditioned linear feasibility margin from short exact-state simulator interventions. At deployment it receives only current observation-derived proprio features, phase inferred from episode fraction, and the current postprocessed SmolVLA action. It reshapes the action continuously; it does not rank candidates, query future success, or replan.

## Fixed Split

- tasks: `[('libero_spatial', 4), ('libero_10', 4)]`
- training identities: `[20260711]`
- eval identities: `[20260712, 20260713, 20260714, 20260715, 20260716, 20260717, 20260718, 20260719, 20260720, 20260721]`
- loaded training/result JSON: `reports/phase_barrier_vla_prototype_result.json`
- training state fractions: `0.0,0.35,0.65`
- short intervention horizon: `4`
- max eval steps override: `0` (`0` means official max)

## Variants

1. `frozen_smolvla`
2. `pre_vla_style_halt_proxy`
3. `simple_global_damping`
4. `phase_barrier_no_phase_ablation`
5. `phase_barrier_full`

## GO/KILL

- Route A: full method improves task-balanced success by at least 5 absolute percentage points over the strongest non-ablation baseline and beats the no-phase ablation.
- Route B: full method beats strongest baseline and ablation, and relative failure rate decreases by at least 10%.
- Kill: full method fails both routes, simple baseline matches/beats it, ablation matches/beats it, or infrastructure/runtime invalidates measurement.
