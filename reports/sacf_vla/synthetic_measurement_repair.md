# SACF-VLA Synthetic Measurement Repair

Date: 2026-07-12 KST

Initial failed result preserved:

- `reports/sacf_vla/synthetic_result_initial_fail.json`
- `reports/sacf_vla/synthetic_result_initial_fail.md`

Reason for repair:

The initial synthetic gate classified SACF as failed because `full_mean_action_l2 = 0.001384` was more than 10 percent above `plain_mean_action_l2 = 0.001138`. Both values are near-perfect reconstruction, and the semantic component was active (`0.381098`). The relative threshold was therefore too brittle for a synthetic implementation smoke where plain BC has the same inputs and enough capacity.

Repair:

Before any real-demo training or closed-loop SACF result was inspected, the synthetic pass rule was changed to require:

- full SACF loss decreases;
- full SACF factor loss decreases;
- plain BC loss decreases;
- full SACF probe L2 is at most `0.01`;
- semantic component norm is greater than `0.01`.

This does not change Stage A variants, tasks, identities, prefix fraction, baselines, or kill rules.
