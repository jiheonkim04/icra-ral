# ExecSpec-Repair Kill Criteria

Kill or reframe ExecSpec-Repair if any of these persist:

- no local executable mismatch can be reproduced from HDF5 action streams or exact-init replay,
- mismatch metrics are indistinguishable from identity/clipping-only baselines,
- minimal calibration cannot beat naive global scaling on any plausible mismatch,
- exact-init replay cannot be run and no concrete simulator/data fix exists,
- the only positive result uses future expert actions as rollout actions,
- the project requires full VLA fine-tuning, OpenVLA-OFT, large downloads, or GPU-heavy training before the first credible metric,
- claims would need paper-grade rollout evidence before diagnostic replay support exists.
