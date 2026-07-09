# Official SmolVLA Stable Protocol Result

Date: 2026-07-10 KST

- final decision: `NEEDS_LARGER_PREDICTION_ARTIFACT`
- experiments happened: `False`
- training happened: `False`
- trained components: `[]`
- GPU/download/OpenVLA-OFT happened: `False` / `False` / `False`
- official model/dataset used: `True`
- old custom route used: `False`

## Instability Diagnosis

- too_few_heldout_frames: `{'likely': True, 'evidence': 'Previous sweep used 5 folds with 40 test frames per fold from a 200-frame artifact.'}`
- task_imbalance: `{'likely': True, 'evidence': 'Previous fold tests each covered one task pair; stable manifest now covers all eligible official tasks in each split.'}`
- episode_leakage: `{'likely': False, 'evidence': 'Previous and new manifests use episode-disjoint splits; the problem is coverage, not known leakage.'}`
- insufficient_episode_disjoint_coverage: `{'likely': True, 'evidence': 'Previous artifact used 10 held-out episodes across 5 tasks; stable manifest selects 200 episodes across 40 tasks.'}`
- lora_regeneration_mismatch: `{'likely': 'unresolved', 'evidence': 'Robust sweep reused one rank-4 LoRA artifact and did not retrain LoRA per fold or seed.'}`
- small_prediction_artifact: `{'likely': True, 'evidence': 'Current artifact has 200 frames; stable manifest requires 2800 prediction records.'}`
- metric_definition_variance: `{'likely': True, 'evidence': 'Action L2 rank order changed by fold; task-balanced and bootstrap intervals were not fixed before FCAR.'}`
- action_component_imbalance: `{'likely': True, 'evidence': 'Raw 7D L2 mixes translation, rotation, and gripper units; component metrics must be reported separately.'}`
- gripper_or_rotation_weighting: `{'likely': True, 'evidence': 'Previous large errors often involved gripper; raw aggregate can hide component-specific behavior.'}`
- task_level_distribution_shift: `{'likely': True, 'evidence': 'Rank-4 LoRA beat frozen/base in only 2/5 folds and won no realistic fold.'}`
- static_validation_instability: `{'likely': True, 'evidence': 'Validation-selected static alpha won 3/5 folds but was selected from very small validation slices.'}`

Must fix before method design:

- use the fixed task-stratified episode-disjoint manifest
- generate larger official base/LoRA prediction artifacts under the manifest
- report both frame-weighted and task-balanced metrics
- select static alpha on validation only and freeze before test
- add episode/task bootstrap intervals
- run independent rank-4 LoRA seeds only after the manifest and metrics are frozen
- keep FCAR killed unless a future frozen-criteria report beats static merge

## Split Manifest

- status: `created`
- frame counts: `{'train': 1200, 'val': 400, 'test': 1200}`
- leakage checks: `{'episode_disjoint_train_val': True, 'episode_disjoint_train_test': True, 'episode_disjoint_val_test': True}`

## Metric Protocol

- status: `created`
- primary metric: `aggregate raw 7D action L2 after official SmolVLA postprocessing`

## Artifact

- status: `planned_not_generated`
- exact next command: `powershell -ExecutionPolicy Bypass -File scripts\248_official_smolvla_prediction_artifact_from_manifest.ps1 -SplitManifest reports\official_smolvla_split_manifest.json -Output reports\official_smolvla_stable_prediction_artifact.json`
