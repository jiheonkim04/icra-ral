# Official LeRobot LIBERO Asset Plan

Date: 2026-07-09 KST

## Assets Audited

Hub metadata was inspected without downloading full assets.

| asset | type | access | license | files | size |
| --- | --- | --- | --- | ---: | ---: |
| `lerobot/smolvla_libero` | model | public, not gated | Apache-2.0 | 9 | 0.844 GiB |
| `lerobot/libero` | dataset | public, not gated | Apache-2.0 | 457 | 1.803 GiB |
| `HuggingFaceVLA/libero` | dataset | public, not gated | Apache-2.0 | 383 | 32.528 GiB |

Local disk at audit time:

- `C:\` free bytes: `402,327,212,032`
- approximate free space: `374.7 GiB`

## Selected Official Asset Pair

For `lerobot/smolvla_libero`, the checkpoint `train_config.json` names:

```text
dataset.repo_id = lerobot/libero
```

Therefore the selected official asset pair is:

- model: `lerobot/smolvla_libero`
- dataset: `lerobot/libero`

Combined model + selected dataset size:

```text
0.844 GiB + 1.803 GiB = 2.647 GiB
```

This is within local disk capacity but exceeds the objective's `2GB` no-approval threshold. Do not download until user approval is explicit.

## Official Dataset Metadata

`lerobot/libero` metadata:

- `robot_type`: `panda`
- `fps`: `10.0`
- total episodes: `1693`
- total frames: `273465`
- total tasks: `40`
- images:
  - `observation.images.image`: video, `[256, 256, 3]`
  - `observation.images.image2`: video, `[256, 256, 3]`
- state:
  - `observation.state`: float32, `[8]`
- action:
  - `action`: float32, `[7]`
- metadata:
  - `meta/info.json`
  - `meta/stats.json`
  - `meta/tasks.parquet`
  - `meta/episodes/chunk-000/file-000.parquet`

`HuggingFaceVLA/libero` has the same state/action shape but stores images as `image` dtype in parquet and is much larger.

## Official Commands

Metadata-only inspection:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -c "from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata; m=LeRobotDatasetMetadata('lerobot/libero'); print(m.info); print(m.features)"
```

Official model load after approval:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; p=SmolVLAPolicy.from_pretrained('lerobot/smolvla_libero'); print(p.config.input_features); print(p.config.output_features)"
```

Official eval after approval and on Linux/WSL:

```bash
export MUJOCO_GL=egl
lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env.max_parallel_tasks=1
```

Documented official PEFT/LoRA command:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=your_hub_name/my_libero_smolvla \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.output_features=null \
  --policy.input_features=null \
  --policy.optimizer_lr=1e-3 \
  --policy.scheduler_decay_lr=1e-4 \
  --env.type=libero \
  --env.task=libero_spatial \
  --steps=100000 \
  --batch_size=32 \
  --peft.method_type=LORA \
  --peft.r=64 \
  --peft.lora_alpha=64
```

Do not run this local PEFT command under the current state.

## Approval-Gated Download Command

Exact next command only after explicit user approval:

```powershell
$env:HF_HOME='C:\assets\hf_home'
huggingface-cli download lerobot/smolvla_libero --local-dir C:\assets\checkpoints\smolvla_libero
huggingface-cli download lerobot/libero --repo-type dataset --local-dir C:\assets\datasets\lerobot_libero
```

Expected persistent download size:

```text
2.647 GiB plus filesystem/cache overhead
```

## Windows / WSL Status

- Dataset metadata and local conversion planning are viable on Windows.
- Official LIBERO eval requires Linux/WSL because LeRobot LIBERO requires Linux and MuJoCo.
- Official training/eval with simulator callbacks should be treated as WSL/Linux-only until a separate simulator readiness check passes.

