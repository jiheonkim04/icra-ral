# Official SmolVLA Stable Protocol Decision

Date: 2026-07-10 KST

Final decision: `NEEDS_LARGER_PREDICTION_ARTIFACT`

Reason: Fixed manifest and metric protocol are ready, but no larger official prediction artifact has been generated under them.

Exact next step: powershell -ExecutionPolicy Bypass -File scripts\248_official_smolvla_prediction_artifact_from_manifest.ps1 -SplitManifest reports\official_smolvla_split_manifest.json -Output reports\official_smolvla_stable_prediction_artifact.json
