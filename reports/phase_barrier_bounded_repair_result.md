# PhaseBarrier-VLA Bounded Repair Result

Date: 2026-07-12 KST

Final PhaseBarrier decision: `PHASEBARRIER_COMPONENT_NOT_USEFUL`

## Execution Boundary

- Branch: `codex/phasebarrier-bounded-adjudication`
- Original implementation/postmortem commit: `9620d1b5bea2555fe44bac2b8880a1d798699433`
- Valid result JSON: `reports/phase_barrier_bounded_repair_result.json`
- Invalid checkpoint-mismatch run preserved as: `reports/phase_barrier_bounded_repair_invalid_retrained_result.json`
- Training rerun for valid result: `False`
- Original saved PhaseBarrier weights loaded from: `reports/phase_barrier_vla_prototype_result.json`
- Closed-loop episodes completed: `100/100`
- Exceptions: `0`
- Rollout reruns after result: `0`

The first bounded attempt was rejected before decision because it retrained the barrier and produced a different training identity: `1` positive label instead of the original `8`. The valid result below uses the original saved `phase_model` and `no_phase_model` weights.

## Original Evidence Reconstructed

- Training states: `5`
- Training records: `20`
- Validation examples: `0`
- Original labels: `8` positive, `12` negative
- Training steps: closed-form ridge solve, no SGD steps
- Learned parameters: full phase model `13` weights, no-phase ablation `9` weights
- Loss curves: not applicable / not saved
- Gradient norms: not applicable, no SGD
- Original full success: `0/2`
- Original full action delta: `0.111434`
- Original full shaped steps: `357.5` mean, `715/800` aggregate

## Valid Bounded Protocol

- Backbone: official frozen SmolVLA-LIBERO
- Tasks: `libero_spatial/task_4`, `libero_10/task_4`
- Eval identities: `20260712` through `20260721`
- Variants: frozen, Pre-VLA-style halt proxy, simple global damping, no-phase ablation, full PhaseBarrier
- Episodes: `20` per policy, `10` per task per policy, `100` total
- Primary metric: task-balanced official closed-loop success

## Success Counts

| Variant | Success | Task-balanced success | Wilson 95% CI | Mean action delta | Mean shaped steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | `8/20` | `0.40` | `[0.219, 0.613]` | `0.000000` | `0.0` |
| `pre_vla_style_halt_proxy` | `0/20` | `0.00` | `[0.000, 0.161]` | `0.505738` | `355.05` |
| `simple_global_damping` | `0/20` | `0.00` | `[0.000, 0.161]` | `0.113719` | `400.0` |
| `phase_barrier_no_phase_ablation` | `9/20` | `0.45` | `[0.258, 0.658]` | `0.012180` | `196.75` |
| `phase_barrier_full` | `0/20` | `0.00` | `[0.000, 0.161]` | `0.105796` | `357.45` |

Per-task success:

| Variant | `libero_spatial/task_4` | `libero_10/task_4` |
| --- | ---: | ---: |
| `frozen_smolvla` | `5/10` | `3/10` |
| `pre_vla_style_halt_proxy` | `0/10` | `0/10` |
| `simple_global_damping` | `0/10` | `0/10` |
| `phase_barrier_no_phase_ablation` | `5/10` | `4/10` |
| `phase_barrier_full` | `0/10` | `0/10` |

## Paired Statistics

Full PhaseBarrier versus:

| Comparator | Full wins | Full losses | Ties |
| --- | ---: | ---: | ---: |
| `frozen_smolvla` | `0` | `8` | `12` |
| `pre_vla_style_halt_proxy` | `0` | `0` | `20` |
| `simple_global_damping` | `0` | `0` | `20` |
| `phase_barrier_no_phase_ablation` | `0` | `9` | `11` |

Relative failure-rate reduction versus frozen: `-66.6667%`.

## Mechanism Verification

The mechanism did not collapse:

- Full PhaseBarrier shaped `20/20` episodes.
- Mean full shaped steps: `357.45`.
- Mean full action delta: `0.105796`.
- Full PhaseBarrier modified at least `3379` contact/transport-phase steps by a conservative lower-bound calculation from shaped-step counts and deterministic phase intervals.
- Full is not equivalent to the no-phase ablation: full mean action delta `0.105796`, ablation `0.012180`; full success `0/20`, ablation `9/20`.
- Full is not equivalent to pure clipping or global damping: global damping changed every step and still matched full at `0/20`, but the no-phase ablation retained `9/20` success with much smaller action deltas.

Success gains did not occur in mechanism-activated episodes because full PhaseBarrier had no successful episodes.

## Runtime

- Total elapsed: `2539.94` seconds
- Peak CUDA allocation: `926.638` MB
- Policy route: official `SmolVLAPolicy`, `cuda:0`, relative control
- Old custom `LIBERO_7D` route used: `False`

## Interpretation

The bounded sample is sufficient to reject PhaseBarrier-VLA as implemented. The original saved phase-conditioned component acted strongly, but it destroyed all successes on the paired held-out set. The key no-phase ablation beat the full method by `45` percentage points and won all `9` non-tied paired comparisons.

This is not an implementation non-action. It is a component-level failure: the phase-conditioned barrier component is not useful under the frozen method and protocol.
