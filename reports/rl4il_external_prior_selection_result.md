# RL4IL External Prior Selection Result

- Decision: `RL4IL_EXTERNAL_PRIOR_SELECTED_FEATURE_SMOKE_PASS_PRIOR_ROLLOUT_REQUIRED`
- Prerequisite decision: `WRIST_DROPOUT_REPEATED_PROBLEM_CONFIRMED`
- Selected prior: RL4IL, “Reinforcement Learning-Guided Retrieval with Soft Fusion for Robust Multimodal Imitation Learning under Missing Modalities”
- Primary sources: https://arxiv.org/html/2606.15514v1 and https://github.com/h-ismkhan/Reinforcement-Learning-via-kNN-for-Robotic-Learning-with-Missing-Camera
- Local clone: `C:\assets\repos\RL4IL-Missing-Camera`
- Local clone HEAD: `e1dd5b741ebb6b392bfd1f8cbb61bad82417e9bd`

## Why RL4IL is the selected external prior

RL4IL is the closest runnable external prior found for the confirmed condition because it directly targets complete camera dropout and missing modalities in robot imitation learning. Its paper and official code use LIBERO Spatial/Object/Goal with agent-view RGB, in-hand RGB, and language, so `mask_1` corresponds to the local wrist/in-hand camera dropout condition.

The mechanism is nontrivial and prior-grounded: frozen CLIP modality encoders, modality-fair normalization, kNN/BFS candidate sets, PPO-guided retrieval, soft cross-attention fusion, per-camera PPO donor retrieval, and soft imputation heads. Inference replays retrieved expert demonstration actions open-loop rather than training a new VLA policy.

## Official code inspection

- Files checked: `README.md`, `rl4il-sptl.py`, `rl4il-obj.py`, `rl4il-goal.py`, `rl4il-epoch.py`, `rl4il-topk.py`.
- `py_compile` passed for all five Python scripts.
- Default dataset root is hardcoded as `/.../datasets`; local runs must redirect to `/mnt/c/assets/data/libero`.
- Dropout configs are `mask_0` for agent camera missing and `mask_1` for in-hand camera missing.
- Default budget is heavy: PPO `30`, fusion `30`, imputation PPO `50`, soft imputation `30` epochs; rollout default is `3` seeds x `25` rollouts/task, `260` max steps.
- `SKIP_TRAINING=True` exists, but no official checkpoint files were bundled in the cloned repository.

## Local no-training smoke

The local readiness smoke used the official RL4IL module only for import, HDF5 loading, compilation, and frozen CLIP feature extraction. No training, optimizer step, checkpoint write, simulator rollout, or Ours method occurred.

| check | result |
|---|---|
| LIBERO Goal task0 HDF5 | 50 demos, expected `agentview_rgb` and `eye_in_hand_rgb` keys present |
| LIBERO Object task0 HDF5 | 50 demos, expected `agentview_rgb` and `eye_in_hand_rgb` keys present |
| LIBERO Spatial task5 HDF5 | 50 demos, expected `agentview_rgb` and `eye_in_hand_rgb` keys present |
| CLIP smoke on 2 Spatial task5 demos | cam0 `[2,512]`, cam1 `[2,512]`, language `[2,512]`, labels `[2]` |
| CUDA | NVIDIA GeForce RTX 5080 |
| Peak CUDA max allocated | 612.6103515625 MiB |

Environment versions: scikit-learn `1.7.2`, gym `0.26.2`, transformers `4.57.6`, torch `2.10.0+cu128`.

## Comparator status

RL4IL is selected as the external prior, but it has not yet satisfied the full “prior comparator” requirement locally. The paper addresses and reports improvements for the same semantic condition, but local matched prior performance, local residual gap, and exact protocol deltas versus the frozen X-VLA paired-identity manifest are not established yet.

Required next action: preregister and execute a bounded RL4IL `mask_1` prior training/rollout or checkpoint-based evaluation before generating any learned Ours method.
