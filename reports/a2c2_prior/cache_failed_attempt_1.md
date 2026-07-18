# A2C2 Cached Feature Probe Failed Attempt 1

Date: `2026-07-19 KST`

Classification: `DATA_PIPELINE_DEFECT`

The process loaded the frozen Base, then stopped before the first cache row or
model forward because LeRobot `0.4.4` does not expose the LeRobot `0.2`
`episode_data_index` attribute. The empty HDF5 artifact contained zero dataset
keys and zero rows. No training, rollout, or scientific outcome occurred.

The root-bounded repair derives the same half-open, subset-local episode
boundaries from the authoritative `hf_dataset["episode_index"]` column and
requires that the resulting contiguous runs exactly match the 40 frozen
episode IDs. It changes no episode, frame, anchor, offset, supervision,
condition, mechanism, budget, threshold, or decision rule.
