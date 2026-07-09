# Official SmolVLA / LeRobot Next Decision

Date: 2026-07-09 KST

Final decision: `NEEDS_OFFICIAL_DATASET_CONVERSION`

## Reason

The local official SmolVLA base path is usable:

- official LeRobot loader works;
- official LeRobot processor factory works with local tokenizer override;
- model loads;
- one synthetic forward pass returns finite `[1, 6]` action;
- PEFT/bitsandbytes/CUDA environment is present.

However, the target LIBERO route is not yet an official baseline:

- local checkpoint action/state schema is 6D SO100-style;
- LeRobot LIBERO expects 8D state and 7D actions;
- the local LIBERO HDF5 files are not the official LeRobot dataset format;
- official LIBERO evaluation requires Linux/MuJoCo and was not run;
- no official `lerobot/smolvla_libero` checkpoint is present locally.

## Official Compatibility Checklist

- Official package/source: LeRobot `0.4.4` is installed.
- Official model class/path: `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy`.
- Official loader used: `SmolVLAPolicy.from_pretrained`.
- Official processor factory used: `lerobot.policies.factory.make_pre_post_processors`.
- Checkpoint completeness: complete for SmolVLA base loading; `config.json`, `model.safetensors`, policy preprocessor/postprocessor JSON, and normalizer safetensors are present.
- Processor/preprocessor status: present and loadable with local SmolVLM tokenizer override.
- Action normalizer status: present in checkpoint pre/postprocessor safetensors, but SO100-named and 6D.
- Official local checkpoint action convention: 6D SO100-style action, with no LIBERO 7D gripper-action normalizer found.
- Official LIBERO action convention: 8D state and continuous 7D relative-control action through LeRobot `LiberoEnv`.
- Dataset compatibility: local LIBERO HDF5 is not official LeRobot dataset format as-is; conversion is required before real reproduction.
- Conversion feasibility: a tiny sample can be planned in LeRobot dataset format if it preserves LIBERO 8D state, image keys, 7D action, task text, and split/stat provenance without using the archived adapter.
- Evaluation compatibility: official eval path is `lerobot-eval --env.type=libero`; it is rollout/simulator evaluation via LeRobot LIBERO on Linux/MuJoCo. The current MuJoCo/RoboSuite replay bridge is custom-only and is not official reproduction evidence.
- Local eval status: official LIBERO eval was not run on this Windows pass.
- LoRA feasibility: PEFT can attach through LeRobot's official `wrap_with_peft` path after official-compatible data exists.
- Official default LoRA targets: `(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))`.
- Count-only LoRA parameter check: rank 4 gives `185,664` trainable parameters; rank 16 gives `742,656` trainable parameters.
- VRAM feasibility: RTX 5080 16GB should be enough for a tiny batch-1 official-compatible smoke, but this must be verified with parameter/input device and CUDA-memory logging.
- bitsandbytes status: available and CUDA-smoke tested; useful for low-memory training, but not a substitute for official preprocessing/action compatibility.

## Exact Next Step

Create a bounded official-compatible LIBERO alignment plan before any method work.

The plan should choose one route:

1. Official asset route: risk-assess and acquire `lerobot/smolvla_libero` and the small `lerobot/libero` LeRobot-format dataset, then run a metadata/single-sample official LIBERO-shaped smoke.
2. Conversion route: convert a tiny local LIBERO HDF5 subset into the official LeRobot LIBERO 8D state / 7D action convention, then run official processor/load smoke without using the archived custom adapter route.

Do not start RA-L method work until one of those routes produces a clean official-compatible baseline.

Because the final decision is not `READY_FOR_OFFICIAL_SMOLVLA_MINI_REPRO`, there is no training command to run yet.
