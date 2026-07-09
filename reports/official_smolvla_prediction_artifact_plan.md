# Official SmolVLA Prediction Artifact Plan

Date: 2026-07-10 KST

- status: `planned_not_generated`
- target prediction records: `2800`
- recommended output: `reports/official_smolvla_stable_prediction_artifact.json`

Reason not generated:

Generating official SmolVLA predictions for the 2800-frame manifest would require a larger bounded GPU run and rank-4 LoRA artifact regeneration; this protocol run freezes the split/metrics first.

Required contents:

- frozen/base predictions
- rank-4 LoRA predictions
- raw 7D labels
- official state/action metadata
- task and instruction identifiers from official metadata
- split membership from reports/official_smolvla_split_manifest.json
- per-frame action errors
- normalized eval loss when available
- CUDA/device/VRAM/runtime audit if LoRA is regenerated

Exact next command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\248_official_smolvla_prediction_artifact_from_manifest.ps1 -SplitManifest reports\official_smolvla_split_manifest.json -Output reports\official_smolvla_stable_prediction_artifact.json
```
