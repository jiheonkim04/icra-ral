# Official SmolVLA / LeRobot Risk Register

Date: 2026-07-09 KST

| risk | severity | status | mitigation |
| --- | --- | --- | --- |
| Local checkpoint is 6D SO100-style, not official LIBERO 8D/7D | high | active | Do not call local checkpoint an official LIBERO baseline. Use official `smolvla_libero` or convert data cleanly. |
| Archived custom LIBERO 7D adapter route could be accidentally reused | high | controlled | Explicitly forbid it in all official baseline reports. |
| Plain `python` resolves to Windows Store alias | medium | controlled | Use `C:\Users\jiheo\miniconda3\envs\tca_map\python.exe`. |
| Official LIBERO evaluation requires Linux/MuJoCo | high | active | Do not run Windows LIBERO eval; use WSL/Linux readiness only after separate risk assessment. |
| Official LIBERO training data may be large | medium | active | `lerobot/libero` is about 1.803 GB; `HuggingFaceVLA/libero` is about 32.528 GB and should not be downloaded in this pass. |
| Tokenizer/preprocessor offline resolution can fail when saved Hub IDs are not mapped locally | medium | controlled | Use a local tokenizer override pointing to the cached SmolVLM2 directory for offline mini-repro. |
| LoRA training could silently run on CPU | high | controlled | CUDA checks were run. Future LoRA training must log parameter/input devices, CUDA memory, and autocast state. |
| Mini-repro could be overclaimed | high | controlled | Label it as CPU-only official-loader synthetic smoke, not training, rollout, benchmark, or paper evidence. |

## Risk Assessment For Executed Mini-Repro

- Task: official LeRobot SmolVLA base load and one-sample synthetic forward.
- Source: local checkpoint and local cached VLM tokenizer.
- Downloads: none.
- Expected size: existing local files only.
- Runtime: under 1 minute.
- RAM/VRAM: CPU model load, 0 MB CUDA allocation.
- Device: intentionally CPU-only diagnostic.
- Token/license/payment: none.
- Decision: proceed.
- Reason: bounded, local, no training, no rollout, no OpenVLA-OFT, and no paper claim.

