# Autonomous RA-L Time Budget

Governance correction:

- total GPU time cap: `24 h`
- per-cycle wall-clock cap: `12 h`
- single uncheckpointed command cap: `4 h`

Cycle 1 recorded GPU time before Stage A: approximately `0.05 h`.

Stage A planned command:

- expected duration: under `4 h`
- planned episodes: `50`
- checkpointed inputs: full and no-history DICD adapter checkpoints
- output artifacts: `reports/dicd_vla/stage_a_result.json`, `reports/dicd_vla/stage_a_result.md`

Reviewer B approval for Stage A: yes, because it is the cheapest preregistered decisive experiment.
