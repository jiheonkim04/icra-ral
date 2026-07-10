# Official SmolVLA Canonical Eval Policy

Date: 2026-07-10 KST

- Policies: `frozen_base`, `rank4_lora_seed_11`, `rank4_lora_seed_22`, `rank4_lora_seed_33`, and validation-selected action-space static mixes.
- Persisted disk-reloaded policies only: `true`.
- Training/checkpoint regeneration: `false`.
- Old custom LIBERO_7D route: `false`.
- Action-generation eval seeds: `[101, 202, 303, 404, 505]`.
- RNG formula: `seed64 = int(sha256(json({namespace, eval_seed, immutable_frame_identity}, sort_keys=True, compact)).hexdigest()[:16], 16) & ((1 << 63) - 1)`.
- RNG identity fields: `['split', 'sample_id', 'dataset_local_index', 'dataset_global_index', 'episode_index', 'frame_index', 'episode_length', 'task_index']`.
- Labels excluded from RNG: `['target_action', 'action_l2', 'eval_loss', 'success', 'reward']`.
- Static-mix alpha grid: `[0.0, 0.25, 0.5, 0.75, 1.0]`.
- Static-mix alpha selection split: `val` only.
- Test outcomes used for seed/alpha selection: `false`.
- Repeat determinism mode: `smoke_first_5_frames_per_split`.
- Offline action-L2 uses the current/first postprocessed action vector; full postprocessed action chunks are generated and hashed.
