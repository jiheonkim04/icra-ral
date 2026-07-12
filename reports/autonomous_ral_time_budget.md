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

Stage A actual runtime:

- stopped uncheckpointed infrastructure launch: approximately `1.0 h`
- checkpointed Stage A result launch: `5637.278 s`, approximately `1.57 h`
- approximate Cycle 1 GPU time consumed including prior smoke work: `2.62 h`
- remaining campaign GPU budget: approximately `21.38 h`

Cycle 1 is closed. Cycle 2 remains under the `12 h` per-cycle wall-clock cap and the campaign-level `24 h` GPU cap.

Cycle 2 pre-Stage-A runtime:

- synthetic smoke: local CPU/GPU-light, not counted as material GPU time
- failed real-trace reporting attempt: approximately `0.04 h`
- successful real-trace training: `174.563 s`, approximately `0.05 h`
- approximate campaign GPU time consumed so far: `2.72 h`
- approximate remaining campaign GPU budget: `21.28 h`

FEDO Stage A planned command is checkpointed and expected to remain under `4 h`.
