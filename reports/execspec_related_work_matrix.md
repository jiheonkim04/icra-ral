# ExecSpec-Repair Related Work Matrix

## Working Buckets

- VLA deployment/action normalization: compare against papers and repos that document action heads, normalization metadata, and postprocessors.
- Robot controller interfaces: compare against OSC/pose controller conventions, delta/absolute action assumptions, and action clipping.
- Gripper convention repair: compare against binary gripper adapters, sign conventions, thresholding, and open/close timing diagnostics.
- Runtime calibration and system identification: compare against lightweight action-space calibration, not full VLA fine-tuning.
- Evaluation diagnostics: compare against exact-init replay, HDF5 replay sanity, and action-distribution audits.

## Baseline Commitments

Required simple baselines are identity, clipping-only, naive global scale, and per-dimension calibration. Later replay states must keep exact expert replay and zero/held controls when simulator execution is safe.
