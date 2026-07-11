# PhaseBarrier-VLA Empirical Postmortem

Date: 2026-07-12 KST

Postmortem classification: `UNDERPOWERED_PROTOTYPE_INCONCLUSIVE`

This supersedes treating `PHASE_BARRIER_VALID_KILL` as a method-level kill. The prototype ran and the wrapper changed actions, but the evidence is too small and too success-sparse to distinguish a failed idea from a weak smoke test on an over-hard slice.

## Source Artifacts

- Protocol: `reports/phase_barrier_vla_prototype_protocol.md`
- Result JSON: `reports/phase_barrier_vla_prototype_result.json`
- Result report: `reports/phase_barrier_vla_prototype_result.md`
- Method code: `tca_map/smolvla/phase_barrier_vla.py`
- Runner: `scripts/run_phase_barrier_vla_prototype.py`
- Tests: `tests/test_phase_barrier_vla.py`

## Hypothesis And Component

Hypothesis: a phase-conditioned feasibility margin can identify when the frozen SmolVLA action is physically risky, then continuously project the action away from phase-inappropriate execution without candidate ranking, replanning, or privileged inference.

Technical component: a closed-form ridge linear margin over action/proprio/phase features. The full method uses phase one-hot features; the key ablation removes phase. At inference, negative margins shrink XY translation and rotation, adjust Z by phase, and leave the gripper dimension unchanged.

Backbone: frozen official SmolVLA-LIBERO, policy class `SmolVLAPolicy`, `cuda:0`, relative control, action chunk shape `[1, 50, 7]`.

## Split And Training Data

- Tasks: `libero_spatial/task_4` and `libero_10/task_4`.
- Train identity: `[20260711]`.
- Eval identity: `[20260712]`.
- Training fractions requested: `0.0,0.35,0.65`.
- Captured training states: `5`, not `6`.
- Training records: `20` records from `5` states x `4` candidate actions.
- Candidate counts: `default=5`, `global_damping_0p70=5`, `translation_scale_1p35=5`, `contact_z_boost=5`.
- Labels: `8` positive, `12` negative.
- Important imbalance: every positive label was in phase `contact`; `approach=8` negative, `contact=8` positive, `transport=4` negative, and no `placement` training records were captured.
- No candidate achieved actual short-horizon task success: `task_success_within_short_horizon=False` for `20/20` rows. Positives came from the effect-compatibility threshold, not terminal success.

There was no validation split. The only held-out closed-loop split was the single reset identity `20260712`.

## Training Procedure

Training was not SGD. The runner generated short exact-state intervention records, then solved one weighted ridge linear system for the phase model and one for the no-phase model with `l2=1e-3`.

Loss curves: none saved and none expected from this closed-form solve.

Gradient evidence: not applicable. Nonzero learned-parameter evidence exists:

- Phase model: width `13`, L2 norm `1.666286`, max absolute weight `1.331981`.
- No-phase model: width `9`, L2 norm `37.870051`, max absolute weight `22.634854`.

The unit tests check that a synthetic PhaseBarrier model separates simple good/bad records, that risky actions are changed while safe actions are not, that baselines are distinct, and that phase inference is deterministic.

## Closed-Loop Evidence

All variants ran `2` held-out episodes, one per task. Wilson intervals below are 95% binomial intervals for raw success count.

| Variant | Success | Task-balanced success | Wilson 95% CI | Mean action delta | Mean shaped steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | `0/2` | `0.0` | `[0.000, 0.658]` | `0.000000` | `0.0` |
| `pre_vla_style_halt_proxy` | `0/2` | `0.0` | `[0.000, 0.658]` | `0.494227` | `341.5` |
| `simple_global_damping` | `0/2` | `0.0` | `[0.000, 0.658]` | `0.114796` | `400.0` |
| `phase_barrier_no_phase_ablation` | `0/2` | `0.0` | `[0.000, 0.658]` | `0.004760` | `175.0` |
| `phase_barrier_full` | `0/2` | `0.0` | `[0.000, 0.658]` | `0.111434` | `357.5` |

Per-task success for every variant:

- `libero_spatial/task_4`: `0/1`.
- `libero_10/task_4`: `0/1`.

Runtime and memory:

- Total elapsed: `487.086` seconds.
- Peak CUDA allocation: `926.638` MB.
- Exceptions: `0`.

## Action Verification

The full method did act:

- `libero_spatial/task_4`: shaped `255/280` steps, mean action delta `0.141024`, mean margin `-0.438403`.
- `libero_10/task_4`: shaped `460/520` steps, mean action delta `0.081844`, mean margin `-0.441524`.

The no-phase ablation barely acted despite very large positive average margins:

- `libero_spatial/task_4`: shaped `127/280` steps, mean action delta `0.004157`, mean margin `11.674931`.
- `libero_10/task_4`: shaped `223/520` steps, mean action delta `0.005364`, mean margin `11.279003`.

The simple global damping baseline produced action-delta magnitude similar to full PhaseBarrier (`0.114796` versus `0.111434`) and also failed `0/2`. That tie is not enough to classify the method as simple-baseline-explained because the shared `0/2` outcome has almost no statistical resolution.

Action chunk distribution, gripper behavior, detailed action direction, and trajectory/failure phase cannot be reconstructed from the saved JSON because per-step action traces were not persisted. From code, PhaseBarrier intentionally changes dimensions `0:6` and leaves the gripper dimension unchanged.

## Classification Rationale

`UNDERPOWERED_PROTOTYPE_INCONCLUSIVE` is the correct classification.

Why not `GENUINE_METHOD_KILL`: although the implementation ran and full PhaseBarrier materially changed actions, the held-out evaluation is only `2` episodes and all success intervals are broad. The training labels also did not include any short-horizon task-success positives, only effect-compatibility positives concentrated in contact.

Why not `NO_MECHANISM_HEADROOM`: no oracle or diagnostic upper bound was run for this exact projection policy. The fact that all variants failed is not an oracle headroom proof.

Why not `SIMPLE_BASELINE_EXPLAINS_METHOD`: a simple damping baseline matched full at `0/2`, but with every variant at `0/2` this is an underpowered tie, not an explanation.

Why not `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`: the head learned nonzero parameters and full PhaseBarrier changed actions on most steps. The saved evidence does not show a collapsed integration.

Bottom line: this prototype is a negative smoke result, not a review-resistant method kill.
