# SafeTrace-VLA Autopilot State

- branch: `codex/safetrace-vla-state0-state1`
- allowed states: `STATE 0-1`
- wall-clock budget: preferred 2 hours, hard 4 hours
- large downloads: forbidden in this run
- GPU: forbidden unless separately green; not expected
- OpenVLA-OFT: forbidden
- full VLA fine-tuning: forbidden
- current state: STATE 1 complete
- final output: `KILL`
- kill reason: safety-only/risk-only scoring matched the SafeTrace preference objective on generated pairs
- temporal metric: produced on local standard LIBERO HDF5 proxy only
- preference pairs: 800 valid, 10 nontrivial
- generic DPO proxy accuracy: 1.0
- SafeTrace proxy accuracy: 1.0
- task success labels: unavailable in sampled local HDF5 traces
- final output must be exactly one of `CONTINUE_TO_STATE_2`, `KILL`, or `SOURCE_BLOCKED`
