# RAC-VLA Development Audit

Date: `2026-07-14`

Proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`

Final decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`

- closed-loop experiment happened: `False`
- training happened: `False`
- consequence pairs: `10769`
- labeled examples: `53685`
- train examples: `32640`
- validation examples: `21045`
- duplicate perturbation keys: `0`
- full validation accuracy: `0.5857448325017819`
- action-only validation accuracy: `0.368496079828938`
- no-consequence validation accuracy: `0.3744832501781896`
- full-vs-best-baseline accuracy margin: `0.21126158232359227`
- gate positive fraction: `0.16830601092896175`
- clean gate positive fraction: `0.0`
- clean action delta p95: `0.0`
- shifted action delta p95: `0.010000000000000002`
- validation action validity: `1.0`

Task consequence counts:

```json
{
  "libero_10/task_4": 7194,
  "libero_spatial/task_4": 3575
}
```

Perturbation label counts:

```json
{
  "gripper_bias": 10737,
  "identity": 10737,
  "x_attenuate": 10737,
  "xy_swap": 10737,
  "y_attenuate": 10737
}
```

Hard stop reasons:
- none

Next step: Run bounded six-config validation search.
