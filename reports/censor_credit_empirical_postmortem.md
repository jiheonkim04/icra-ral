# CensorCredit-VLA Empirical Postmortem

Date: 2026-07-12 KST

Postmortem classification: `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

This supersedes treating `CENSOR_CREDIT_VALID_KILL` as a method-level kill. The rollout produced a weak positive signal over frozen/simple baselines, but the intended censored-credit mechanism did not get a distinct learned target: the censored and uncensored labels were identical for every training record, the saved models have identical weights, and the full method behaved as the uncensored ablation.

## Source Artifacts

- Protocol: `reports/censor_credit_vla_prototype_protocol.md`
- Result JSON: `reports/censor_credit_vla_prototype_result.json`
- Result report: `reports/censor_credit_vla_prototype_result.md`
- Method code: `tca_map/smolvla/censored_credit_vla.py`
- Runner: `scripts/run_censor_credit_vla_prototype.py`
- Tests: `tests/test_censored_credit_vla.py`

## Hypothesis And Component

Hypothesis: temporal credit should be censored at intervention boundaries so that actions before a recovery are not wrongly credited for the eventual recovered outcome.

Technical component: two closed-form ridge temporal trust heads over current action, previous action, action delta, gripper, and step fraction:

- `uncensored_recovery_ablation`: labels a prefix using the recovered-outcome score.
- `censor_credit_full`: labels a prefix using prefix-only credit.

At inference, both heads use the same temporal hold/blend rule: if the learned margin is negative, blend the current action toward the previous action with `hold_strength=0.70`.

Backbone: frozen official SmolVLA-LIBERO, policy class `SmolVLAPolicy`, `cuda:0`, relative control, action chunk shape `[1, 50, 7]`.

## Split And Training Data

- Tasks: `libero_spatial/task_4` and `libero_10/task_4`.
- Train identity: `[20260711]`.
- Eval identity: `[20260712]`.
- Training fractions requested: `0.0,0.35,0.65`.
- Captured training states: `6`.
- Training records: `24` records from `6` states x `4` candidate actions.
- Candidate counts: `default=6`, `ema=6`, `jump_plus=6`, `hold_previous=6`.
- Censored positives: `4/24`.
- Uncensored positives: `4/24`.
- `prefix_success=False` for `24/24`.
- `recovered_success=False` for `24/24`.
- Label-pair table: `(-1,-1)=20`, `(1,1)=4`, with no record where censored and uncensored labels differed.

There was no validation split. The only held-out closed-loop split was the single reset identity `20260712`.

## Training Procedure

Training was not SGD. The runner generated short exact-state prefix/recovery records, then solved one weighted ridge linear system for each temporal trust head with `l2=1e-3`.

Loss curves: none saved and none expected from this closed-form solve.

Gradient evidence: not applicable. Nonzero learned-parameter evidence exists, but it is identical for the two heads:

- Censored model: width `8`, L2 norm `12.127745`, max absolute weight `8.211099`.
- Uncensored model: width `8`, L2 norm `12.127745`, max absolute weight `8.211099`.
- Saved weights are identical element by element.

The unit tests check that a synthetic temporal head separates smooth versus jump records, that low-margin blending uses the previous action, and that the EMA and jump-proxy baselines are distinct.

## Closed-Loop Evidence

All variants ran `2` held-out episodes, one per task. Wilson intervals below are 95% binomial intervals for raw success count.

| Variant | Success | Task-balanced success | Wilson 95% CI | Mean action delta | Mean shaped steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frozen_smolvla` | `0/2` | `0.0` | `[0.000, 0.658]` | `0.000000` | `0.0` |
| `vla_corrector_jump_proxy` | `0/2` | `0.0` | `[0.000, 0.658]` | `1.864512` | `357.5` |
| `simple_temporal_ema` | `0/2` | `0.0` | `[0.000, 0.658]` | `0.041235` | `400.0` |
| `uncensored_recovery_ablation` | `1/2` | `0.5` | `[0.095, 0.905]` | `0.113220` | `278.5` |
| `censor_credit_full` | `1/2` | `0.5` | `[0.095, 0.905]` | `0.119921` | `296.5` |

Per-task success:

- `libero_spatial/task_4`: frozen `0/1`, jump proxy `0/1`, EMA `0/1`, uncensored ablation `1/1`, full `1/1`.
- `libero_10/task_4`: all variants `0/1`.

Runtime and memory:

- Total elapsed: `465.897` seconds.
- Peak CUDA allocation: `926.638` MB.
- Exceptions: `0`.

## Action Verification

The full method changed actions relative to frozen, but it did not act distinctly from the uncensored ablation:

- Full: mean action delta `0.119921`, mean shaped steps `296.5`.
- Uncensored ablation: mean action delta `0.113220`, mean shaped steps `278.5`.
- Full and ablation shared the same success pattern: success on `libero_spatial/task_4`, failure on `libero_10/task_4`.
- Full and ablation used the same hold/blend transform with identical learned weights.

Episode details:

- `libero_spatial/task_4`: ablation succeeded in `131` steps with `77` shaped steps; full succeeded in `132` steps with `98` shaped steps.
- `libero_10/task_4`: ablation failed after `520` steps with `480` shaped steps; full failed after `520` steps with `495` shaped steps.

Action chunk distribution, detailed action direction, gripper trajectory, and failure phase cannot be reconstructed from the saved JSON because per-step action traces were not persisted. From code, temporal hold/blend blends the full 7D current action toward the full 7D previous action, including the gripper dimension.

## Classification Rationale

`IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` is the correct classification.

Why not `GENUINE_METHOD_KILL`: the intended new component was censored temporal credit, but the generated training targets did not distinguish censored from uncensored credit. The full method therefore never received an empirical test as a distinct learned mechanism.

Why not `SIMPLE_BASELINE_EXPLAINS_METHOD`: simple EMA and the jump proxy both scored `0/2`; the matching method was the key ablation, not a simple baseline.

Why not only `UNDERPOWERED_PROTOTYPE_INCONCLUSIVE`: the rollout count is also tiny, but the more concrete defect appears before statistics: the censored and uncensored heads are identical because their labels are identical.

Why not `MEASUREMENT_INVALID`: there is no leakage or protocol defect in the train/eval split. The issue is that the generated supervision did not instantiate the intended mechanism.

Bottom line: CensorCredit-VLA was not genuinely killed. The particular implementation collapsed to its uncensored ablation.
