# NICE-VLA Stage 0B1 Result

Decision: `NICE_STAGE_0B1_DATA_FAILURE_COLLAPSED_ACTION_REGIME_CONTRAST`.

The worker completed and cached all `1792 / 1792` frozen pair keys with zero
duplicate, missing, or extra keys, then exited `1` during the action-regime
diagnostic. The partial JSON is valid and exactly equals the manifest, so no
pair is rerun.

The frozen discovery deadband is `2.0`. Evaluation transition counts are:

- `libero_10/task_5`: `78 / 2` no-transition/transition;
- `libero_goal/task_5`: `79 / 1`;
- `libero_object/task_3`: `80 / 0`;
- `libero_spatial/task_3`: `80 / 0`.

Two validation tasks therefore have collapsed action-regime contrast. This
fails the preregistered data-health gate. Changing deadband, tasks, sampler, or
diagnostic would be a new method cycle.

Mean and covariance training artifacts were produced before the ordering flaw
exposed the data gate, but no mechanism metrics are accepted or interpreted.
No simulator rollout, task outcome, reward, done, reset identity, or
confirmatory record was read. Stage 0B2 and NICE rescue are forbidden.
