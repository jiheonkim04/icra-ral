# SafeLoRA-VLA Kill Criteria

Date: 2026-07-08

## Immediate Kill Gates

Kill or stop before training if any condition holds:

- no official benchmark/source can provide safety metrics beyond local proxy
  labels,
- no bounded official subset or clearly bounded official sample path exists,
- the training path requires full VLA fine-tuning or heavy OpenVLA-OFT,
- property labels are not available or cannot be derived without hidden oracle
  leakage,
- utility retention cannot be measured,
- only safe demonstrations exist and no unsafe/property violation pairs can be
  generated from official rollouts or official annotations,
- generic DPO/ORPO with the same pair set is expected to match SafeLoRA,
- safety-only, stop-on-risk, clipping-only, or no-op baselines are expected to
  match SafeLoRA on safe-success,
- the plan cannot produce a publishable comparison table.

## Future Quantitative Continuation Criteria

If a later user-approved STATE 2 runs, SafeLoRA may continue only if:

- SafeLoRA improves official safe-success over base and standard LoRA;
- SafeLoRA reduces temporal/process safety metrics over base and standard LoRA;
- task success drop versus standard LoRA remains within a predeclared tolerance;
- no-op/abort/stop rate does not explain the safety gain;
- safety-only or stop-on-risk does not match SafeLoRA safe-success within the
  tolerance;
- generic DPO/ORPO does not match SafeLoRA aggregate or property-wise safety
  gains within the tolerance;
- at least one property-wise improvement is not explained by frequency or
  label imbalance.

Suggested first tolerance for a smoke, to be revised before training:

- safe-success improvement: at least 5 percentage points over standard LoRA,
- task-success retention: no more than 5 percentage point drop versus standard
  LoRA,
- no-op/abort increase: no more than 5 percentage points versus standard LoRA,
- baseline match threshold: within 2 percentage points counts as matched.

## Current Gate Trigger

This run triggers the pre-training stop condition:

`NO_CLEAR_LORA_PATH`

The official benchmark candidates do not yet provide a clear, bounded,
property-conditioned LoRA/QLoRA training path under the current constraints.
