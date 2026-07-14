# EvoState-VLA Development Audit

Date: `2026-07-14`

Proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`

Final decision: `AUDIT_STOP_DESIGN_FAILURE`

- closed-loop experiment happened: `False`
- training happened: `False`
- transition pairs: `10769`
- train transition pairs: `6548`
- validation transition pairs: `4221`
- duplicate transition keys: `0`
- transition improvement vs constant: `0.7153085205910045`
- transition improvement vs actionless: `0.024689372539669806`
- controllability effective rank: `7`
- gate positive fraction: `0.28760957119166075`
- validation action delta p95: `0.04157738591349467`
- validation action validity: `1.0`

Task transition counts:

```json
{
  "libero_10/task_4": 7194,
  "libero_spatial/task_4": 3575
}
```

Hard stop reasons:
- `transition model improvement vs actionless below minimum: 0.024689`

Next step: Do not roll out EvoState; archive the hard stop and continue.
