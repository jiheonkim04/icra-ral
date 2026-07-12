# PhaseBarrier-VLA Bounded Repair Plan

Date: 2026-07-12 KST

Decision before execution: `RUN_ONE_BOUNDED_SAMPLE_SIZE_ADJUDICATION`

## Frozen Method

This repair preserves PhaseBarrier-VLA exactly. It does not change:

- model architecture;
- linear barrier representation;
- observation/action/proprio/phase inputs;
- short exact-state intervention supervision;
- closed-form ridge objective;
- phase-conditioned projection intervention;
- original two targeted tasks;
- original baselines and key ablation;
- GO/KILL thresholds.

Allowed repair used here: insufficient rollout sample size only.

The runner changes are evaluation/checkpoint infrastructure only: extend the deterministic reset identity list, write bounded-repair output filenames, and load the original saved PhaseBarrier weights from `reports/phase_barrier_vla_prototype_result.json`. The action projection, label generation, training solve used in the original prototype, and policy transforms are unchanged.

## Original Evidence Gate

The original postmortem verified that the method acted and did not collapse:

- full PhaseBarrier success: `0/2`;
- full mean action delta: `0.111434`;
- full mean shaped steps: `357.5`;
- full shaped steps: `715/800`;
- phase model learned-parameter width: `13`;
- phase model L2 norm: `1.666286`;
- exceptions: `0`.

Therefore the larger rollout budget is eligible. This is not a rescue of a nonacting implementation.

## Bounded Repeat Protocol

- Backbone: official frozen SmolVLA-LIBERO.
- Tasks: `libero_spatial/task_4`, `libero_10/task_4`.
- Training identity: `20260711`.
- Eval identities: `20260712` through `20260721`.
- Initial-state indices: `1` through `10` for eval; training remains index `0`.
- Variants: `frozen_smolvla`, `pre_vla_style_halt_proxy`, `simple_global_damping`, `phase_barrier_no_phase_ablation`, `phase_barrier_full`.
- Episodes: `20` per policy, `10` per task per policy, `100` total.
- Maximum allowed: `300` total episodes.
- Max eval steps override: `0`, official max.
- Original learned weights: loaded from `reports/phase_barrier_vla_prototype_result.json`, not retrained.
- Reruns: no unsuccessful episode reruns; only infrastructure crash reruns would be separately recorded.

## Fixed GO/KILL Criteria

`PHASEBARRIER_PROTOTYPE_GO` if Route A or Route B passes and mechanism activation is noncollapsed.

Route A:

- full method improves task-balanced closed-loop success by at least `5` percentage points over strongest non-ablation baseline;
- full method beats the no-phase ablation.

Route B:

- paired evidence favors full;
- relative failure rate decreases by at least `10%`;
- full beats the no-phase ablation;
- clean/control performance is retained.

Other fixed decisions:

- `PHASEBARRIER_GENUINE_METHOD_KILL` if implementation and activation are valid, sufficient paired evaluation completes, and full fails to beat strongest baseline or ablation.
- `PHASEBARRIER_KILLED_BY_SIMPLE_BASELINE` if the simple baseline matches or beats full.
- `PHASEBARRIER_COMPONENT_NOT_USEFUL` if the no-phase ablation matches or beats full.
- `PHASEBARRIER_IMPLEMENTATION_FAILURE` if the method does not affect policy behavior as intended.
- `PHASEBARRIER_RESULT_STILL_INCONCLUSIVE` only if this bounded paired sample remains statistically unresolved; no further PhaseBarrier repeat is allowed.

## Expected Runtime

The original run used `487.086` seconds for training plus `10` rollout episodes. The bounded repeat uses `100` rollout episodes. Expected runtime is approximately `45` to `75` minutes on the same WSL/CUDA RTX 5080 path, within the bounded local budget.
