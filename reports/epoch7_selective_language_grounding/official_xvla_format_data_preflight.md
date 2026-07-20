# Official X-VLA-format LIBERO-Goal data preflight

Date: 2026-07-20

Status: `OFFICIAL_XVLA_FORMAT_GOAL_DATA_RETAINED_AND_PARTITIONED`.

The public Apache-2.0 dataset [`2toINF/Libero-XVLA-format`](https://huggingface.co/datasets/2toINF/Libero-XVLA-format) was acquired at immutable revision `27ddd36538ee4812bd31fd8b494f8d7c6a11ef9d`, restricted to `libero_goal/**` and the README. The retained subset contains 428 per-demonstration HDF5 files totaling 1,899,116,312 bytes across all ten Goal tasks. The authors' release contains 34–50 converted demonstrations per task; the variation is preserved and not filled from a different format.

Every retained file used by the partition audit contains the official fields `abs_action_6d[T,10]`, `proprio[T,9]`, two encoded image streams, and a scalar language instruction. Action/proprio arrays are finite and modality lengths match. The frozen loader reproduces the released `LiberoHandler`: discard the first image, sample a 31-point 30 Hz/1 s absolute trajectory, append a zero 10D right arm, use point zero as 20D proprio, and use points 1–30 as the clean-action target.

The hash-bound partition manifest is `method_partition_manifest.json`:

- 348 train, 40 validation, and 40 sealed confirmatory demonstrations;
- 2,892 train, 600 validation, and 600 sealed confirmatory LIBERO-Para rows;
- exact assignment digest `041e3cb49f2daf72fe5dd55e71cc7c5dd0bfddc1e11d2dc2707181457110bfd1`;
- 30 outcome-independent Base-energy samples spanning ten tasks and all three paraphrase families.

No model, simulator, reward, done, success, training, optimizer, Ours, or confirmatory content was used to acquire or partition the data. The retained subset consumes 1.769 GiB, leaving approximately 101.7 GiB above the required Windows reserve at the post-download check.
