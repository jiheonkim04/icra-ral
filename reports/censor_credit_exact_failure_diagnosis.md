# CensorCredit Exact Failure Diagnosis

Date: 2026-07-12 KST
Branch: `codex/censorcredit-one-repair-and-final-method`
Base commit: `1f29a422945350e33ba3be0cb6150054735c49f6`

Final diagnosis: `LABEL_OR_DATA_FAILURE`

Allowed repair categories from the user objective:

- `CONCRETE_IMPLEMENTATION_BUG`
- `CONCRETE_OPTIMIZATION_BUG`

Because the exact failure is neither allowed category, no CensorCredit repair is permitted.

## Mechanism Audited

CensorCredit-VLA trained two closed-form ridge temporal trust heads over temporal action features:

- Full censored head: label a prefix as positive only if the one-step prefix succeeds or the prefix-only effect score exceeds `0.03`.
- Uncensored recovery ablation: label the same prefix as positive if the recovered rollout succeeds or the recovered effect score exceeds `0.03`.

The heads are fitted by ridge regression over `CensorRecord(features, label)` records. At inference, both variants call the same temporal hold transform:

- `uncensored_recovery_ablation`: score features with the uncensored model, then call `temporal_hold_blend`.
- `censor_credit_full`: score features with the censored model, then call `temporal_hold_blend`.

Therefore the intended new component exists only if the censored and uncensored training labels produce different learned heads.

## Exact Evidence

Source artifacts:

- `reports/censor_credit_vla_prototype_result.json`
- `reports/censor_credit_empirical_postmortem.md`
- `scripts/run_censor_credit_vla_prototype.py`
- `tca_map/smolvla/censored_credit_vla.py`

Training evidence:

- training records: `24`
- training states: `6`
- censored positives: `4/24`
- uncensored positives: `4/24`
- prefix successes: `0/24`
- recovered successes: `0/24`
- label-pair counts: `(-1,-1)=20`, `(1,1)=4`
- rows with censored/uncensored label disagreement: `0`

Learned-model evidence:

- censored weights: `[-8.211099468732627, 8.007013840134432, -2.946969746171567, -1.780154245290029, 0.31521261992376537, 0.7934892463627556, 0.14765500244966767, -1.7156391161055606]`
- uncensored weights: `[-8.211099468732627, 8.007013840134432, -2.946969746171567, -1.780154245290029, 0.31521261992376537, 0.7934892463627556, 0.14765500244966767, -1.7156391161055606]`
- weights equal element by element: `True`
- threshold equal: `0.0`

Closed-loop behavior:

- frozen SmolVLA: `0/2`
- simple temporal EMA: `0/2`
- jump-hold proxy: `0/2`
- uncensored recovery ablation: `1/2`
- CensorCredit full: `1/2`
- full mean action delta: `0.119921`
- uncensored ablation mean action delta: `0.113220`

The full variant changed actions relative to frozen, but the change was produced by the same learned head and same hold/blend transform as the uncensored ablation. The intended censored-credit distinction was never instantiated.

## Audit Answers

1. Checkpoint identity: no separate repaired checkpoint is involved. The rollout used in-memory fitted ridge models serialized in the report.
2. Old model loading: not applicable to the two trust heads; both heads were fitted during the run from the same row table.
3. Gradient existence: not applicable. Training is a closed-form ridge solve, not an SGD optimization.
4. Loss decrease: not applicable. No iterative loss curve is expected from the solver.
5. Nonzero learned parameters: yes, the heads have nonzero weights.
6. Distinct learned component: no. The two intended heads are exactly identical.
7. Label separability: failed. Censored and uncensored labels match on every row.
8. Action change: yes versus frozen, but not as a distinct censored-credit component.
9. Ablation gate: failed. The uncensored ablation matches the full method.
10. Privileged input: no privileged eval input was needed to explain the failure.
11. Optimization bug: no evidence. The optimizer solved exactly the labels it was given.
12. Implementation bug: no single concrete code bug was isolated. The code executed the predeclared label formulas.
13. Data failure: yes. The generated intervention records did not create any prefix/recovery disagreement.

## Classification

This is `LABEL_OR_DATA_FAILURE`, not `CONCRETE_IMPLEMENTATION_BUG` and not `CONCRETE_OPTIMIZATION_BUG`.

The one permissible repair would require a specific bug such as a wrong checkpoint, an unused trained head, a sign error, an optimizer failure, or a transform wiring error. The evidence instead shows that the supervision itself collapsed: every row gave the censored and uncensored heads the same label, so identical models are the expected outcome.

The CensorCredit branch must therefore proceed with:

`CENSORCREDIT_NO_VALID_REPAIR`
