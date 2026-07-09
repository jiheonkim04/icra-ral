# Official SmolVLA-LIBERO Mini-Reproduction Result

Date: 2026-07-09 KST

## Execution Boundary

- Downloads happened: yes, only the two approved official assets.
- GPU used: yes, for official model forward and tiny LoRA smoke.
- Training happened: yes, only rank-4 LoRA smoke for 5 steps.
- Full benchmark happened: no.
- Simulator rollout happened: no.
- OpenVLA-OFT happened: no.
- Custom `LIBERO_7D` adapter route used: no.
- Paper claims: no.

## Official Model / Processor / Dataset Smoke

Paths:

- checkpoint: `C:\assets\checkpoints\smolvla_libero`
- dataset: `C:\assets\datasets\lerobot_libero`
- VLM dependency resolved locally: `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`

Official APIs:

- `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained`
- `lerobot.policies.factory.make_pre_post_processors`
- `lerobot.datasets.lerobot_dataset.LeRobotDataset`

Load/forward status:

- model loaded: yes
- parameter device: `cuda:0`
- parameter dtype: `torch.bfloat16`
- processor/preprocessor loaded: yes
- dataset loaded: yes
- sample video backend: `pyav`
- raw dataset state/action: `[8]` / `[7]`
- preprocessed state/action: `[1, 8]` / `[1, 7]` for one-step inference sample
- input tensor devices after preprocessing: `cuda:0`
- autocast GPU/CPU: false / false
- selected action shape: `[1, 7]`
- postprocessed action shape: `[1, 7]`
- output finite: yes
- one-sample runtime: `1.053 sec` action selection after load/preprocess
- CUDA peak during one-sample forward: `923.243 MB`
- RSS after one-sample forward: `2561.828 MB`

One-sample offline action metric:

- action L2: `0.080831`
- translation L2: `0.079664`
- rotation L2: `0.007863`
- gripper absolute error: `0.011206`
- postprocessed action preview: `[0.073823, -0.051641, -0.018556, -0.001997, 0.000617, -0.00758, -1.011206]`

## Five-Sample Offline Mini Evaluation Smoke

This is an offline labeled-sample smoke, not a simulator benchmark.

Samples: first 5 frames from episode 0.

Summary:

- action L2 mean/max: `0.072885` / `0.105014`
- translation L2 mean/max: `0.071989` / `0.104230`
- rotation L2 mean/max: `0.006936` / `0.010011`
- gripper abs mean/max: `0.007376` / `0.011630`
- gripper sign accuracy: `1.0`
- finite outputs: yes
- prediction range over five samples: `[-1.011630, 0.086202]`
- runtime: `10.697751 sec`
- CUDA peak: `923.243164 MB`
- RSS: `2451.035156 MB`

## Tiny LoRA Smoke

Configuration:

- PEFT method: LoRA
- rank: `4`
- batch size: `1`
- optimizer steps: `5`
- optimizer: `AdamW`
- learning rate: `1e-4`
- dataset sample: official `lerobot/libero`, episode 0, with official SmolVLA action chunk delta timestamps
- action chunk: `[1, 50, 7]`
- action pad mask: `[1, 50]`

First attempt:

- failed before optimizer step because the official preprocessor produced action chunk shape `[50, 7]`.
- fix: add the missing batch dimension to action chunk and pad mask, producing `[1, 50, 7]` and `[1, 50]`.
- no custom action adapter, custom normalizer, or gripper fill was used.

Successful smoke:

- total params: `450,231,840`
- trainable params: `185,664`
- trainable percent: `0.041237`
- trainable tensors with gradients: `74`
- nonzero gradient tensors after step 1 onward: `74`
- loss before: `0.003114`
- loss after: `0.003007`
- loss delta: `-0.000107`
- post-LoRA action shape: `[1, 7]`
- post-LoRA action finite: yes
- post-LoRA action range: `[-0.993084, -0.001292]`
- runtime: `9.903 sec`
- CUDA peak: `1102.960 MB`
- RSS final: `2839.043 MB`
- autocast GPU/CPU: false / false

## Official Eval Status

Offline mini evaluation works on Windows with the downloaded assets and `pyav`.

Official simulator/environment evaluation via `lerobot-eval --env.type=libero` still requires Linux/WSL/MuJoCo readiness and was not run in this pass.

## Decision

Final decision: `READY_FOR_OFFICIAL_BASELINE_SCALEUP`

Reason: approved assets downloaded, official model loaded, official dataset loaded, pre/postprocessing worked, mini offline evaluation worked, and tiny rank-4 LoRA smoke worked within local VRAM/runtime budget.

Exact next step:

Create a bounded official baseline scaleup script/run that uses the downloaded `smolvla_libero` + `lerobot/libero` assets, keeps batch size `1`, rank `4`, runtime under 30 minutes, logs CUDA memory/devices/autocast, and runs a small fixed number of LoRA steps before any WSL/Linux simulator eval.
