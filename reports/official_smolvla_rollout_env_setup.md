# Official SmolVLA Rollout Env Setup

Date: 2026-07-10 KST

- rollout environment decision: `NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT`
- can run native smoke now: `False`
- blocked reason: Official LIBERO rollout dependencies are missing in the active native environment: ['hf-libero', 'libero', 'robosuite']
- official entrypoint: `lerobot.scripts.lerobot_eval` / `C:\Users\jiheo\miniconda3\envs\tca_map\Scripts\lerobot-eval.exe`
- adapter loading path: `lerobot.policies.factory.make_policy with cfg.policy.use_peft=True and PeftModel.from_pretrained`
- rendering backend required: `OffScreenRenderEnv via LIBERO/RoboSuite; WSL/Linux should set MUJOCO_GL=osmesa or another verified offscreen backend before smoke.`
- compatible LIBERO package: `{'package': 'hf-libero', 'version_spec_from_installed_lerobot_metadata': '>=0.1.3,<0.2.0', 'source': "installed LeRobot 0.4.4 package metadata: extra == 'libero'", 'installed': False}`
- compatible RoboSuite package: `{'package': 'robosuite', 'version_spec_from_installed_lerobot_metadata': 'not directly declared; expected through hf-libero/LIBERO dependency resolution', 'source': 'not established in active env because hf-libero/libero/robosuite are not installed', 'installed': False}`
- compatible MuJoCo package: `{'package': 'mujoco', 'installed_version': '2.3.7', 'source': 'active conda environment package metadata'}`
- package changes: `[]`
- additional downloads performed: `False`

## Package Versions

- accelerate: `1.14.0`
- gymnasium: `1.3.0`
- hf-libero: `NOT_INSTALLED`
- huggingface_hub: `0.35.3`
- lerobot: `0.4.4`
- libero: `NOT_INSTALLED`
- mujoco: `2.3.7`
- peft: `0.19.1`
- robosuite: `NOT_INSTALLED`
- safetensors: `0.8.0`
- torch: `2.10.0+cu128`
- transformers: `4.57.6`

## LeRobot LIBERO Requirements

- `gymnasium<2.0.0,>=1.1.1`
- `lerobot[transformers-dep]; extra == "libero"`
- `hf-libero<0.2.0,>=0.1.3; extra == "libero"`
- `lerobot[libero]; extra == "all"`
