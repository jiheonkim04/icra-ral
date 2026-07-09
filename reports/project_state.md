# Project State

Date: 2026-07-10 KST

Target branch: `main`

Implementation branch: `codex/official-smolvla-stable-protocol`

Current decision: `NEEDS_LARGER_PREDICTION_ARTIFACT`

## Current Route

The archived custom SmolVLA 7D adapter route remains stopped. The valid route is official SmolVLA/LeRobot reproduction first, using official preprocessing, normalization, action conventions, dataset format, and evaluation stack.

This state update is protocol-building only. No new method, FCAR v2, FCAR tuning, simulator rollout, full benchmark, OpenVLA-OFT run, old custom `LIBERO_7D` route, or paper claim was made.

## Stable Protocol Status

- Official stable protocol reports were created:
  - `reports/official_smolvla_stable_protocol_plan.md`
  - `reports/official_smolvla_split_manifest.md`
  - `reports/official_smolvla_split_manifest.json`
  - `reports/official_smolvla_metric_protocol.md`
  - `reports/official_smolvla_prediction_artifact_plan.md`
  - `reports/official_smolvla_stable_protocol_result.md`
  - `reports/official_smolvla_stable_protocol_result.json`
  - `reports/official_smolvla_stable_protocol_decision.md`
- The split manifest was built from official LeRobot LIBERO metadata at `C:\assets\datasets\lerobot_libero`.
- The manifest is task-stratified and episode-disjoint across train, validation, and test.
- Manifest scope: `40` tasks, train `80` episodes / `1200` frames, validation `40` episodes / `400` frames, test `80` episodes / `1200` frames.
- Leakage checks passed: train/validation, train/test, and validation/test episode sets are disjoint.
- The planned stable prediction artifact size is `2800` records.
- The larger stable prediction artifact was not generated in this run.

## Metric Protocol

- Primary metric: aggregate raw 7D action L2 after official SmolVLA postprocessing.
- Required breakdowns: translation dims `0-2`, rotation dims `3-5`, gripper dim `6`, task-balanced means, frame-weighted means, help/hurt counts, route/static-alpha reporting, and action-range validity.
- Required uncertainty: episode and task bootstrap intervals.
- Static mixture alpha must be selected on validation only and frozen before test.
- No method design should proceed until the larger artifact is generated and the fixed metric report is produced from it.

## Instability Diagnosis

The post-FCAR robust sweep used the saved official prediction artifact with `5` deterministic episode-disjoint folds and only `40` test frames per fold. It found:

- frozen/base action L2 mean/std: `0.106514933` / `0.030256808`
- rank-4 LoRA action L2 mean/std: `0.118024225` / `0.023707422`
- mean-action action L2 mean/std: `1.144859705` / `0.018515874`
- frame oracle action L2 mean/std: `0.084582167` / `0.027591676`
- task oracle action L2 mean/std: `0.106079936` / `0.029986441`
- MoIRA-style task router action L2 mean/std: `0.106514933` / `0.030256808`
- val-selected static mix action L2 mean/std: `0.105142674` / `0.026514373`
- realistic win counts: frozen/base `2`, val-selected static mix `3`
- rank-4 LoRA beat frozen/base in `2` / `5` folds but won no realistic fold
- frame oracle won all `5` folds and retained mean headroom `0.021932766`
- task oracle headroom remained tiny at `0.000434997`

The diagnosis is that FCAR remains killed, and the next blocker is not method quality. The blocker is unstable split/metric evidence caused by too-small prediction artifacts, task imbalance, tiny validation slices for static-alpha selection, and missing task-balanced/bootstrap reporting.

## Execution Boundary

- experiments happened: `False`
- training happened: `False`
- trained components: `[]`
- GPU used: `False`
- downloads happened: `False`
- OpenVLA-OFT happened: `False`
- full benchmark / simulator rollout happened: `False`
- official route used: `True`
- official dataset metadata used: `True`
- official model execution happened: `False`
- old custom `LIBERO_7D` route used: `False`
- new method implemented: `False`
- FCAR tuned: `False`
- paper claims made: `False`

## Conclusion

`NEEDS_LARGER_PREDICTION_ARTIFACT`

The stable official split and metric protocol now exist. The exact next step is to generate the larger official prediction artifact under `reports/official_smolvla_split_manifest.json`, then score frozen/base, rank-4 LoRA, static mixture, MoIRA-style task router, and oracles under the frozen metric protocol.
