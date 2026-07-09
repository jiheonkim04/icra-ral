# Project State

Date: 2026-07-10 KST

Target branch: `main`

Implementation branch: `codex/official-smolvla-stable-artifact-eval`

Current decision: `NEEDS_LONGER_LORA_BASELINE_REPRO`

## Current Route

The archived custom SmolVLA 7D adapter route remains stopped. The valid route is official SmolVLA/LeRobot reproduction first, using official preprocessing, normalization, action conventions, dataset format, and evaluation stack.

This state update executed the fixed official split/metric protocol. It did not design a new method, revive FCAR, tune FCAR, run simulator rollout, run a full benchmark, run OpenVLA-OFT, download new assets, use the old custom `LIBERO_7D` route, or make paper claims.

## Stable Artifact Status

- Stable prediction artifact generated: `reports/official_smolvla_stable_prediction_artifact.json`
- Artifact size: `7,219,361` bytes
- Artifact records: `2800`
- Fixed manifest: `reports/official_smolvla_split_manifest.json`
- Metric protocol: `reports/official_smolvla_metric_protocol.md`
- Result reports:
  - `reports/official_smolvla_stable_prediction_artifact_status.md`
  - `reports/official_smolvla_stable_artifact_eval_result.md`
  - `reports/official_smolvla_stable_artifact_eval_result.json`
  - `reports/official_smolvla_stable_baseline_table.md`
  - `reports/official_smolvla_stable_artifact_decision.md`

Manifest scope:

- tasks: `40`
- train: `80` episodes / `1200` frames
- validation: `40` episodes / `400` frames
- test: `80` episodes / `1200` frames
- train/validation/test episode leakage checks: passed

## Execution Boundary

- experiments happened: `True`
- training happened: `True`
- trained components: standard rank-4 LoRA baseline only
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

CUDA/device audit:

- model parameter device: `cuda:0`
- input tensor devices: `cuda:0`
- model parameter dtype: `torch.bfloat16`
- peak CUDA allocation: `1104.506 MB`
- autocast cpu/cuda: `False` / `False`
- CPU fallback: `False`

Rank-4 LoRA regeneration:

- train split: fixed manifest train split
- train frames available: `1200`
- steps: `100`
- trainable params: `185,664`
- loss before/after: `0.008257858` / `0.002369085`
- nonzero grad tensors at final step: `74`
- training elapsed: `17.953 sec`

## Stable Test Metrics

Primary metric is raw 7D action L2 after official SmolVLA postprocessing.

| baseline | action L2 | task-balanced L2 | translation L2 | rotation L2 | gripper abs | gripper sign acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen/base | `0.085558433` | `0.085558433` | `0.069736605` | `0.013744150` | `0.022632962` | `0.993333333` |
| rank-4 LoRA | `0.091230140` | `0.091230140` | `0.070690943` | `0.013045764` | `0.027685609` | `0.990833333` |
| mean-action prior | `1.197255124` | `1.197255124` | `0.606959130` | `0.077452536` | `0.995449574` | `0.545833333` |
| frame oracle | `0.068470215` | `0.068470215` | `0.056971588` | `0.012921991` | `0.017395659` | `0.995833333` |
| task oracle | `0.079386015` | `0.079386015` | `0.068160808` | `0.013377581` | `0.017816481` | `0.995833333` |
| MoIRA-style task router | `0.092209764` | `0.092209764` | `0.070046466` | `0.013393855` | `0.029344422` | `0.990000000` |
| val-selected static mix | `0.081135060` | `0.081135060` | `0.063464903` | `0.011991432` | `0.024354280` | `0.994166667` |

Static mixture:

- alpha grid: `[0.0, 0.25, 0.5, 0.75, 1.0]`
- selected alpha: `0.5`
- selection split: validation
- test-set tuning: `False`

## Stability Analysis

- Frozen/base is still competitive: `True`
- Rank-4 LoRA is robustly better than frozen/base: `False`
- Rank-4 LoRA is worse than frozen/base on aggregate: `True`
- Rank-4 LoRA beats frozen/base on `16` / `40` tasks, but not overall.
- Static mix beats both frozen/base and rank-4 LoRA on aggregate: `True`
- Realistic task win counts: static mix `29`, frozen/base `7`, rank-4 LoRA `4`
- MoIRA-style task router remains weak: `True`
- Frame oracle headroom over frozen/base: `0.017088218`
- Frame oracle headroom after static mix: `0.012664845`
- Task oracle headroom over frozen/base: `0.006172418`
- Task oracle no longer looks weak under the larger stable artifact.
- The larger artifact resolves the previous split-instability blocker enough to move the blocker to LoRA seed robustness.

## Conclusion

`NEEDS_LONGER_LORA_BASELINE_REPRO`

The stable artifact and baseline table now exist. The next blocker is not split construction; it is single-seed rank-4 LoRA robustness under the fixed manifest. Do not design a new method yet. Run independent standard rank-4 LoRA seeds under the fixed manifest first.
