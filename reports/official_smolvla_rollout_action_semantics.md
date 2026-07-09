# Official SmolVLA Rollout Action Semantics

Date: 2026-07-10 KST

Status: `FROZEN_NOT_EXECUTABLE_YET`

This is a source-inspection record only. It did not run model inference or simulator rollout.

## Required Entrypoint

Official rollout must use:

```powershell
lerobot-eval --env.type=libero
```

or the exact installed equivalent confirmed by local source inspection:

- console script: `lerobot-eval`
- import target: `lerobot.scripts.lerobot_eval:main`
- installed script path: `C:\Users\jiheo\miniconda3\envs\tca_map\Scripts\lerobot-eval.exe`

The local help output lists `libero` as a supported env type, and the source imports `lerobot.envs.libero`.

## Disallowed Rollout Paths

Official rollout must not use:

- old exact-init replay bridge
- custom `LIBERO_7D` adapter
- local normalization
- hard-coded gripper conversion
- adapter-weight soup or model-weight merge

## Observed LeRobot Eval Order

Local source inspection of `lerobot/scripts/lerobot_eval.py` shows the rollout path:

1. environment preprocessor
2. policy preprocessor
3. `policy.select_action(observation)`
4. policy postprocessor
5. environment postprocessor
6. `env.step(action_numpy)`

Local source inspection of `lerobot/policies/smolvla/modeling_smolvla.py` shows that SmolVLA manages an internal action queue. The configured chunk size and number of action steps are both `50`, and the action dimension is `7`.

## Frozen Base Semantics

`frozen_base` is the official SmolVLA-LIBERO checkpoint at:

- `C:\assets\checkpoints\smolvla_libero`
- Hugging Face revision `31d453f7edd78c839a8bbc39744a292686daf0de`

It must run through the official LeRobot preprocessor, SmolVLA action queue, official policy postprocessor, official LIBERO env postprocessor, and official `env.step` path.

## Rank-4 LoRA Semantics

`rank4_lora` is the base policy plus a persisted rank-4 LoRA adapter checkpoint bundle. The adapter must be loaded through the official-compatible policy path and must share the same official preprocessor/postprocessor references as the base rollout.

Because seed-specific adapter bundles are currently missing, this policy is not rollout-executable yet.

## Static Mix Semantics

`validation_selected_action_space_static_mix` is:

```text
a_mix = alpha * a_lora + (1 - alpha) * a_base
```

where `alpha` is selected using validation evidence only.

For official rollout, static mix must operate on corresponding base and LoRA action chunks, not on two independently advancing `select_action` queues. The admissible wrapper must:

1. receive the same official preprocessed observation at queue refill time;
2. compute the corresponding frozen-base and rank-4-LoRA action chunks;
3. apply the official policy postprocessor to both corresponding chunks;
4. mix corresponding postprocessed chunk actions elementwise with the validation-selected alpha;
5. serve the mixed queue through the official environment postprocessor and `env.step` path.

Mixing only the one-step outputs of two independent policy queues is disallowed because queue refills could desynchronize the corresponding base and LoRA chunks.

## Compute Accounting

Static mix is more expensive than a single policy. It must be reported as such.

Required rollout logs:

- model forward passes per queue refill
- effective model forward passes per env step
- latency per action or action chunk
- peak VRAM
- wall-clock time
- episodes
- steps
- success rate
- exceptions

Expected static-mix accounting:

- queue refill: `2` policy forward passes, one base and one LoRA
- non-refill env step while mixed queue is nonempty: `0` new policy forward passes
- memory accounting: report peak VRAM for the combined wrapper, not only one subpolicy
