# Epoch 3 Failure Synthesis

Date: 2026-07-13 KST

Decision: `EPOCH_3_SYNTHESIZED_KILLS_EPOCH_4_PIVOT_REQUIRED`

## Related Methods

Epoch 3 tested three related observation/data-side methods:

- `CBFD-VLA`: cross-backbone teacher-feature distillation and retention from Quantized OpenVLA-OFT INT4 traces.
- `SCVC-VLA`: sensor-statistic canonicalization under fixed photometric shift.
- `PSE-VLA`: fixed photometric view ensembling with postprocessed 7D action averaging.

## Outcomes

`CBFD-VLA` was killed in Stage A:

- frozen SmolVLA: `7 / 10`
- full CBFD: `0 / 10`
- decision: `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`

`SCVC-VLA` was killed in Stage B:

- shifted frozen SmolVLA: `20 / 40`
- full SCVC: `11 / 40`
- paired CI versus shifted frozen: `[-0.4250, -0.0250]`
- decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

`PSE-VLA` was killed after the allowed Stage B expansion:

- bright single transform: `51 / 80`
- full PSE: `50 / 80`
- paired CI versus bright single: `[-0.1000, 0.0750]`
- decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

## Synthesis

The common failure pattern is that local observation-side or teacher-data interventions can make the frozen policy behave differently, but closed-loop success is explained by simpler baselines:

- direct frozen SmolVLA outperformed CBFD;
- shifted frozen SmolVLA outperformed SCVC;
- a single fixed photometric transform outperformed PSE.

Epoch 4 must change at least two core dimensions relative to Epoch 3. It must not be a cosmetic variant of teacher distillation, sensor-statistic canonicalization, or photometric action ensembling.

The next method search must follow the post-PSE governance update: problem-first, external-prior-early, mechanism-explicit, and mathematically justified.
