# FANG-VLA Development Audit

Date: `2026-07-14`

Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`

Final decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`

- closed-loop experiment happened: `False`
- training happened: `False`
- development records: `10801`
- train records: `6568`
- validation records: `4233`
- duplicate development keys: `0`
- validation gateable fraction: `1.0`
- validation median action-field separation: `0.1243454626282238`

Hard stop reasons:
- none

Class counts:

```json
{
  "libero_10/task_4": {
    "failure": 12,
    "success": 4
  },
  "libero_spatial/task_4": {
    "failure": 10,
    "success": 6
  }
}
```

Next step: Run bounded train-validate search.
