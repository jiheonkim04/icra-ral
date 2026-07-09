# FCAR Implementation TODO

Date: 2026-07-09 KST

Do not implement in this run. This is the exact next-run TODO.

## Files To Add

- `tca_map/smolvla/fcar_tiny_gate.py`
- `scripts/245_fcar_tiny_gate.ps1`
- `tests/test_fcar_tiny_gate.py`
- `reports/fcar_tiny_gate_result.json`
- `reports/fcar_tiny_gate_result.md`

## Data Artifacts To Reuse

- official checkpoint: `C:\assets\checkpoints\smolvla_libero`
- official dataset: `C:\assets\datasets\lerobot_libero`
- official VLM dependency: `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`
- existing routing gate report: `reports/official_smolvla_routing_design_gate.json`

## Prediction Artifact Plan

Saved per-frame base/LoRA predictions are currently missing.

First implementation step:

1. regenerate frozen/base and rank-4 LoRA predictions using official code path;
2. save compact per-frame artifact under `reports/fcar_prediction_artifact.json`;
3. include sample keys, task, episode, frame, base action, LoRA action, target action, base/LoRA losses, and allowed gate features;
4. do not save model weights or dataset files.

## Tiny Gate Architecture

Start with a small MLP:

- input: concatenated allowed features;
- hidden sizes: `[64, 32]`;
- output: scalar alpha logit;
- activation: ReLU or GELU;
- alpha: sigmoid(logit).

Use a logistic-regression baseline before the MLP if implementation time is tight.

## Training Loop

- deterministic seed: `0`;
- split by episode where possible;
- train/val/test split: `60/20/20`;
- optimizer: AdamW;
- max epochs: small fixed cap, e.g. `100`;
- early stop on validation action L2;
- CPU acceptable; GPU optional;
- runtime under 30 minutes.

## Evaluation Loop

Evaluate:

- frozen/base;
- rank-4 LoRA;
- mean-action prior;
- frame oracle;
- task oracle;
- MoIRA-style instruction/task router;
- static mixture grid;
- FCAR.

## JSON Report Schema

Required top-level keys:

- `policy`
- `paths`
- `split`
- `feature_schema`
- `baselines`
- `fcar_config`
- `metrics`
- `oracle_recovery`
- `calibration`
- `kill_criteria`
- `final_decision`

## Validation Commands

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest -q tests\test_fcar_tiny_gate.py
git diff --check
powershell -ExecutionPolicy Bypass -File scripts\99_tree_check.ps1
```

Run safe runner if the implementation touches shared code.

## Stop Criteria

- prediction artifact cannot be regenerated from official assets;
- FCAR needs disallowed inference inputs;
- FCAR fails to beat frozen/base;
- FCAR fails to beat static mixture or MoIRA-style router;
- any method run drifts into simulator rollout, OpenVLA-OFT, full benchmark, or custom route.
