# CensorCredit Repair Preregistration

Date: 2026-07-12 KST
Branch: `codex/censorcredit-one-repair-and-final-method`
Base commit: `1f29a422945350e33ba3be0cb6150054735c49f6`

Preregistration decision: `CENSORCREDIT_NO_VALID_REPAIR_PREREGISTERED`

## Gate

The user objective permits exactly one CensorCredit repair only if the exact cause is one of:

- `CONCRETE_IMPLEMENTATION_BUG`
- `CONCRETE_OPTIMIZATION_BUG`

The exact diagnosis in `reports/censor_credit_exact_failure_diagnosis.md` is:

`LABEL_OR_DATA_FAILURE`

## Repair Status

No repair is preregistered, implemented, trained, or evaluated.

The only plausible fixes would require changing the data generation, recovery horizon, recovery policy, candidate set, label threshold, task selection, or supervision source so that prefix-only and recovered-outcome labels differ. Those changes are method/data redesigns, not a bounded repair of a concrete implementation or optimization bug.

## Frozen Non-Actions

- Do not tune `hold_strength`.
- Do not tune the `0.03` label threshold.
- Do not rerun rollouts under a relabeled CensorCredit protocol.
- Do not swap in a new recovery policy and call it a repair.
- Do not train a new CensorCredit checkpoint.

Next step:

`CENSORCREDIT_NO_VALID_REPAIR`
