# Next Actions

Date: 2026-07-09 KST

Current decision: `READY_FOR_OFFICIAL_ASSET_APPROVAL`

## Immediate Next Action

Await explicit approval before the official asset route:

```powershell
$env:HF_HOME='C:\assets\hf_home'
huggingface-cli download lerobot/smolvla_libero --local-dir C:\assets\checkpoints\smolvla_libero
huggingface-cli download lerobot/libero --repo-type dataset --local-dir C:\assets\datasets\lerobot_libero
```

If approval is not granted, the next no-download milestone is to implement a tiny local HDF5-to-LeRobot conversion utility using the plan in `reports/local_hdf5_to_lerobot_conversion_plan.md`.

Do not use the archived custom LIBERO 7D adapter route as evidence. Do not run LoRA/training until the official asset sample or tiny converted sample passes a shape/processor smoke.
