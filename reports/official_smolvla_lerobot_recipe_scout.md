# Official SmolVLA / LeRobot Recipe Scout

Date: 2026-07-09 KST

## Sources Checked

- SmolVLA docs: https://huggingface.co/docs/lerobot/en/smolvla
- SmolVLA base model card: https://huggingface.co/lerobot/smolvla_base
- SmolVLA LIBERO model card: https://huggingface.co/lerobot/smolvla_libero
- LeRobot GitHub README: https://github.com/huggingface/lerobot
- LeRobot LIBERO docs: https://huggingface.co/docs/lerobot/libero
- LeRobot environment processor docs: https://huggingface.co/docs/lerobot/env_processor
- LeRobot LIBERO dataset card: https://huggingface.co/datasets/lerobot/libero

## Official SmolVLA Recipe

The official SmolVLA docs describe SmolVLA as a base model that should be fine-tuned on a user's LeRobot dataset. The documented training entrypoint is `lerobot-train`, with `--policy.path=lerobot/smolvla_base`, `--dataset.repo_id=...`, `--batch_size=64`, `--steps=20000`, `--policy.device=cuda`, and optional WandB logging.

The local installed LeRobot package exposes these console scripts:

- `lerobot-train`
- `lerobot-eval`
- `lerobot-record`
- `lerobot-replay`
- supporting calibration/dataset utilities

The installed help surface also exposes `--policy.type=smolvla`, `--policy.pretrained_path`, `--dataset.repo_id`, `--env.type=libero`, `--env.task`, `--eval.batch_size`, `--eval.n_episodes`, and PEFT/LoRA options.

## Official LIBERO Recipe

The current LeRobot LIBERO docs state:

- LIBERO requires Linux.
- LeRobot uses MuJoCo for LIBERO simulation.
- The official evaluation command uses `lerobot-eval --env.type=libero`.
- LIBERO control mode must match the checkpoint, with `relative` as default.
- Policy observations are `observation.state`, `observation.images.image`, and `observation.images.image2`.
- LIBERO actions are continuous `Box(-1, 1, shape=(7,))`.

The environment processor docs describe the LIBERO processor as:

- flattening end-effector position, axis-angle orientation, and gripper qpos into an 8D state;
- rotating images to match the HuggingFaceVLA/libero dataset convention;
- separating environment-specific processing from policy-specific normalization/postprocessing.

## Local LeRobot LIBERO Package Status

Installed LeRobot version: `0.4.4`.

Local package inspection shows:

- `LiberoEnv` exists.
- Default `LiberoEnv.task`: `libero_10`.
- Default `LiberoEnv.control_mode`: `relative`.
- Default `LiberoEnv` action shape: `[7]`.
- Default LIBERO observation source features include agentview image, eye-in-hand image, EEF position, EEF quaternion, gripper qpos/qvel, and joint pos/vel.
- `make_env_pre_post_processors` and `LiberoProcessorStep` are installed.

## Local Checkpoint Recipe Status

Local checkpoint: `C:\assets\checkpoints\smolvla`.

The checkpoint README matches the `lerobot/smolvla_base` model-card pattern. The checkpoint config declares:

- `type`: `smolvla`
- `observation.state`: `[6]`
- three visual inputs: `[3, 256, 256]`
- output `action`: `[6]`
- `normalization_mapping`: visual identity, state mean/std, action mean/std
- `chunk_size`: `50`
- `n_action_steps`: `50`
- VLM dependency: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

The pre/postprocessor normalizer tensors are SO100-named and 6D:

- `so100.buffer.action.mean/std`: `[6]`
- `so100-blue.buffer.action.mean/std`: `[6]`
- `so100-red.buffer.action.mean/std`: `[6]`

## External Official Assets Identified

These were inspected by Hugging Face Hub metadata only. No download was performed.

| asset | type | gated | approximate size | license/status |
| --- | --- | ---: | ---: | --- |
| `lerobot/smolvla_libero` | model | false | 0.844 GB | model card says Apache-2.0 |
| `lerobot/smolvla_base` | model | false | 0.852 GB | local copy exists |
| `lerobot/libero` | dataset | false | 1.803 GB | dataset card Apache-2.0 |
| `HuggingFaceVLA/libero` | dataset | false | 32.528 GB | too large for this pass without separate approval |

## Scout Conclusion

Official SmolVLA base loading and preprocessing can be tested locally. Official LIBERO reproduction is not complete with the current local checkpoint because the local checkpoint is 6D SO100-style while LeRobot LIBERO expects 8D state and 7D actions. The next official-compatible path is either:

1. risk-assessed acquisition of the official `lerobot/smolvla_libero` checkpoint and a bounded LIBERO-shaped smoke, or
2. a clean conversion plan from local LIBERO HDF5 data into the LeRobot LIBERO 8D/7D convention.

