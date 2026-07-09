# Official SmolVLA Split Manifest

Date: 2026-07-10 KST

- JSON manifest: `reports/official_smolvla_split_manifest.json`
- source: `official_lerobot_libero_metadata`
- eligible tasks: `40`
- frame counts: `{'train': 1200, 'val': 400, 'test': 1200}`
- episode counts: `{'train': 80, 'val': 40, 'test': 80}`
- task counts: `{'train': 40, 'val': 40, 'test': 40}`
- leakage checks: `{'episode_disjoint_train_val': True, 'episode_disjoint_train_test': True, 'episode_disjoint_val_test': True}`

## Sampling Rule

- seed: `0`
- task_stratified: `True`
- episode_disjoint: `True`
- task_order: `ascending official task_index`
- episode_order: `ascending official episode_index within each task`
- train_episodes_per_task: `2`
- val_episodes_per_task: `1`
- test_episodes_per_task: `2`
- train_frames_per_episode: `15`
- val_frames_per_episode: `10`
- test_frames_per_episode: `15`
- frame_sampling: `linearly spaced integer offsets from first to last frame of each selected episode`
- max_frames_per_episode: `15`

The manifest is task-stratified, episode-disjoint, and deterministic. It is a protocol artifact only; no model inference or training happened while creating it.
