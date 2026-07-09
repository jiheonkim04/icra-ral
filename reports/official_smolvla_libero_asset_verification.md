# Official SmolVLA-LIBERO Asset Verification

Date: 2026-07-09 KST

## Checkpoint Verification

Path:

```text
C:\assets\checkpoints\smolvla_libero
```

Visible files:

- `.gitattributes`
- `README.md`
- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_preprocessor_step_5_normalizer_processor.safetensors`
- `policy_postprocessor.json`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`
- `train_config.json`

Checkpoint status:

- config present: yes
- model weights present: yes, `906,712,520` bytes
- preprocessor JSON present: yes
- postprocessor JSON present: yes
- normalizer/unnormalizer safetensors present: yes
- train config present: yes

Checkpoint schema:

- policy type: `smolvla`
- config action output: `[7]`
- config state input: `[6]`
- config image inputs: `camera1`, `camera2`, `camera3`, each `[3, 256, 256]`
- preprocessor image rename:
  - `observation.images.image` -> `observation.images.camera1`
  - `observation.images.image2` -> `observation.images.camera2`
- normalizer stats include official LIBERO 8D `observation.state` and 7D `action`.

Interpretation: the apparent config-vs-dataset state wrinkle did not block the official smoke. The official processor/model path accepted the downloaded dataset sample with 8D state and produced 7D actions.

## Dataset Verification

Path:

```text
C:\assets\datasets\lerobot_libero
```

Dataset metadata:

- `robot_type`: `panda`
- `fps`: `10.0`
- total episodes: `1693`
- total frames: `273465`
- total tasks: `40`
- split: `{"train": "0:1693"}`

Dataset files:

- `meta/info.json`: present
- `meta/stats.json`: present
- `meta/tasks.parquet`: present
- `meta/episodes/chunk-000/file-000.parquet`: present
- `data/chunk-000/*.parquet`: present
- `videos/observation.images.image/chunk-000/*.mp4`: present
- `videos/observation.images.image2/chunk-000/*.mp4`: present

Dataset schema:

- `observation.images.image`: video, `[256, 256, 3]`
- `observation.images.image2`: video, `[256, 256, 3]`
- `observation.state`: float32, `[8]`
- `action`: float32, `[7]`
- `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`: present

Dataset loader:

- `LeRobotDatasetMetadata('lerobot/libero', root=C:\assets\datasets\lerobot_libero)`: pass
- `LeRobotDataset(..., episodes=[0], video_backend='pyav')`: pass
- sample count for episode 0: `214`
- sample image shapes: `[3, 256, 256]` for both official image keys
- sample state shape: `[8]`
- sample action shape: `[7]`
- sample task text: present

Video readability:

- `torchcodec`: not installed
- `video_reader`: unavailable in this torchvision build
- `pyav`: available and successfully decoded samples

Use `video_backend='pyav'` for local Windows dataset sample smoke unless `torchcodec` is separately installed later.

## Action / State Schema Verdict

- Official downloaded dataset action dimension: 7D.
- Official downloaded checkpoint action output: 7D.
- Official downloaded dataset state dimension: 8D.
- Official downloaded checkpoint/processor accepted 8D state in smoke.

Schema status: pass for mini-repro.
