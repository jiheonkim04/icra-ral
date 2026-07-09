# Official SmolVLA Stable Prediction Artifact Status

Date: 2026-07-10 KST

- status: `completed`
- final decision: `NEEDS_LONGER_LORA_BASELINE_REPRO`
- model path: `C:\assets\checkpoints\smolvla_libero`
- dataset path: `C:\assets\datasets\lerobot_libero`
- manifest path: `reports\official_smolvla_split_manifest.json`
- metric protocol path: `reports\official_smolvla_metric_protocol.md`
- output artifact path: `reports\official_smolvla_stable_prediction_artifact.json`
- artifact generated: `True`
- artifact size bytes: `7219361`
- artifact record count: `2800`
- device plan: `CUDA rank-4 LoRA regeneration; stop with CPU_FALLBACK_BUG if params or tensors remain on CPU`
- estimated frame counts: `{'test': 1200, 'train': 1200, 'val': 400}`
- estimated runtime: `bounded by two-hour cap; previous 200-frame artifact took about 228 seconds`
