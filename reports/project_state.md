# Project State

Date: 2026-07-10 KST

Target branch: `main`

Implementation branch: `codex/official-smolvla-lora-seed-repro`

Current decision: `STATIC_MERGE_ROBUST_BASELINE_READY`

## Current Route

The archived custom SmolVLA 7D adapter route remains stopped. The valid route is official SmolVLA/LeRobot reproduction first, using official preprocessing, normalization, action conventions, dataset format, and evaluation stack.

This state update executed official rank-4 LoRA seed reproduction under the fixed stable protocol. It did not design a new method, revive FCAR, tune FCAR, train a routing model, run simulator rollout, run a full benchmark, run OpenVLA-OFT, download new assets, use the old custom `LIBERO_7D` route, or make paper claims.

## Fixed Protocol

- split manifest: `reports/official_smolvla_split_manifest.json`
- metric protocol: `reports/official_smolvla_metric_protocol.md`
- stable base artifact: `reports/official_smolvla_stable_prediction_artifact.json`
- train: `80` episodes / `1200` frames
- validation: `40` episodes / `400` frames
- test: `80` episodes / `1200` frames
- tasks: `40`
- leakage checks: passed

## Seed Reproduction Status

- seeds run: `11`, `22`, `33`
- per-seed artifacts:
  - `reports/official_smolvla_lora_seed_11_prediction_artifact.json`
  - `reports/official_smolvla_lora_seed_22_prediction_artifact.json`
  - `reports/official_smolvla_lora_seed_33_prediction_artifact.json`
- result reports:
  - `reports/official_smolvla_lora_seed_repro_plan.md`
  - `reports/official_smolvla_lora_seed_repro_result.md`
  - `reports/official_smolvla_lora_seed_repro_result.json`
  - `reports/official_smolvla_lora_seed_repro_table.md`
  - `reports/official_smolvla_lora_seed_repro_decision.md`

## Execution Boundary

- experiments happened: `True`
- training happened: `True`
- trained components: standard rank-4 LoRA baseline seeds only
- SmolVLA backbone trained: `False`
- GPU used: `True`, RTX 5080 CUDA
- downloads happened: `False`
- OpenVLA-OFT happened: `False`
- full benchmark / simulator rollout happened: `False`
- official model/dataset used: `True`
- old custom `LIBERO_7D` route used: `False`
- new method implemented: `False`
- FCAR tuned: `False`
- paper claims made: `False`
- CPU fallback: `False`

CUDA/device audit:

- model parameter device: `cuda:0`
- input tensor devices: `cuda:0`
- peak CUDA allocation: about `1105.569 MB`
- autocast cpu/cuda: `False` / `False`

## Seed Metrics

Primary metric is raw 7D action L2 after official SmolVLA postprocessing.

| seed | frozen/base | rank-4 LoRA | static mix | frame oracle | task oracle | MoIRA router | realistic winner |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `11` | `0.085558433` | `0.084128699` | `0.077354597` | `0.066234143` | `0.078372683` | `0.085719423` | static mix |
| `22` | `0.085558433` | `0.090162398` | `0.080789904` | `0.070815707` | `0.082495298` | `0.089507871` | static mix |
| `33` | `0.085558433` | `0.090426934` | `0.083704791` | `0.070301761` | `0.082546947` | `0.089208622` | static mix |

Mean/std across seeds:

- frozen/base: `0.085558433` / `0.0`
- rank-4 LoRA: `0.088239344` / `0.002908670`
- mean-action prior: `1.197255124` / `0.0`
- static mix: `0.080616431` / `0.002595356`
- task oracle: `0.081138309` / `0.001955707`
- frame oracle: `0.069117204` / `0.002049401`
- MoIRA-style task router: `0.088145305` / `0.001719703`

## Seed Robustness Analysis

- static mix is the realistic winner in `3` / `3` seeds.
- rank-4 LoRA beats frozen/base in only seed `11`.
- rank-4 LoRA does not beat static mix in any seed.
- realistic task win counts summed over seeds: static mix `93`, frozen/base `20`, rank-4 LoRA `7`.
- LoRA seed variance action L2 std: `0.002908670`, range `0.006298235`.
- frame oracle headroom after static remains in all seeds: mean `0.011499227`, min `0.009974197`, max `0.013403030`.
- task oracle headroom is not consistently meaningful: mean `0.004420124`, values `[0.007185750, 0.003063135, 0.003011486]`.
- MoIRA-style task/instruction router remains weak.
- FCAR remains killed and must not be revived from this evidence.

## Conclusion

`STATIC_MERGE_ROBUST_BASELINE_READY`

Validation-selected static merge is now the main realistic baseline for any later planning gate. A future method-design run, if allowed later, must explicitly beat static merge under this fixed protocol and must not use frame oracle as realistic performance.
