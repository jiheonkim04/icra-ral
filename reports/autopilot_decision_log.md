# Autopilot Decision Log

## 2026-07-06 - 7D Bridge And Limited Fixed-Prior Rollout Diagnostic

Decision: preserve full `7D` HDF5 actions for rollout candidates while keeping the existing `ACTION_PREFIX_DIM=4` offline proxy path unchanged.

Rationale: the previous rollout gate was red because the rollout path reused `4D` offline proxy records. Padding or inventing gripper/rotation values would make the diagnostic invalid. The local HDF5 demos already contain `7D` LIBERO actions, so the safe fix is to preserve those actions only for rollout readiness/diagnostics.

Outcome: readiness gate became green. A bounded fixed-prior diagnostic ran `30` simulator steps across ActionMap-style mean, fixed semantic target-prior TCA, and oracle upper-bound variants. Reward and success stayed zero for all variants. The result supports the action bridge enough to proceed with cautious diagnosis, but does not support a rollout performance claim.

Integrity note: no training, LoRA training, loss computation, GPU job, download, heavy VLA import, OpenVLA-OFT execution, benchmark rollout, or paper-grade claim occurred.


## 2026-07-06 - Zero-Reward Rollout Diagnosis And Expert Replay Sanity

Decision: treat the first fixed-prior rollout's zero reward as unresolved until expert replay is checked beyond the short 10-step horizon.

Rationale: the previous fixed-prior rollout diagnostic used a bounded 10-step horizon and returned zero reward/success for all variants. The new diagnostic replayed zero action, ActionMap-style mean actions, HDF5 expert actions, and fixed-prior proxy actions on the same task/init-state for horizons `10`, `25`, and `50`.

Outcome: all variants still had reward `0.0` and success `false` through 50 steps. HDF5 metadata reports the first positive reward/done at step `271`, so zero reward through 50 steps is consistent with sparse reward / short horizon rather than conclusive policy failure. Fixed-prior proxy actions were identical to expert replay actions in this diagnostic, confirming the 7D action bridge path. The naive target-distance metric matched `moka_pot_1_pos`, but did not show fixed-prior target-directed advantage over the ActionMap-style mean baseline; wrong-target movement was unavailable because the counterfactual object was not present in this environment observation key set.

Integrity note: this was bounded diagnostic rollout evidence only. No training, LoRA training, loss computation, GPU job, download, heavy VLA import, OpenVLA-OFT execution, benchmark rollout, or paper-grade claim occurred.

## 2026-07-06 - Full-Demo Expert Replay Sanity

Decision: classify the `7D` action bridge, exact HDF5 init-state replay, and raw HDF5 action convention as green for matched-init longer-horizon diagnostics.

Rationale: the earlier 10/25/50-step zero-reward diagnostic was inconclusive because the HDF5 demonstration first reward/done index was `271`. The full-demo sanity check replayed the same task long enough to cover that success window and compared exact-init zero action, exact-init HDF5 expert replay, and default-reset HDF5 expert replay.

Outcome: exact-init HDF5 expert replay reached reward/done/success at observed index `260`, while the HDF5 reward/done index is `271`. Zero action under the same exact init state stayed at reward `0.0` and success `false`. HDF5 expert actions from default reset also stayed at reward `0.0` and success `false`. The timing mismatch is acceptable for diagnostic purposes and should be documented, but the result shows that raw `7D` HDF5 actions and `env.set_init_state(init_state)` can reproduce task success when replayed long enough.

Integrity note: this was bounded diagnostic rollout evidence only. No training, LoRA training, loss computation, GPU job, download, heavy VLA import, OpenVLA-OFT execution, benchmark rollout, or paper-grade claim occurred.
