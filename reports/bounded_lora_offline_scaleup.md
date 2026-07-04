# Bounded LIBERO Offline LoRA Scale-Up

This task adds a separately gated CPU-only runner for the required LoRA experimental track over local LIBERO HDF5 snippets.

It remains an offline proxy diagnostic only. It is not standard success, not rollout success, not a SmolVLA model-load result, and not paper-grade evidence.

The runner is implemented by:

- `scripts\126_bounded_lora_offline_scaleup.ps1`
- `tca_map.datasets.libero_offline_lora_scaleup`

Policy:

- requires task-local `ALLOW_TINY_TRAINING=1`,
- trains only tiny NumPy LoRA matrices,
- freezes the base representation,
- uses local LIBERO HDF5 action snippets only,
- caps execution to 16 counterfactual pairs, 64 records, 64 update steps, LoRA rank 4, and a 900-second enforced runtime cap,
- stays CPU-only with max GPU memory reported as zero,
- does not download, install packages, import heavy VLA models, load SmolVLA, run model inference, run simulators, rollout, execute OpenVLA-OFT, access tokens, or make paper claims.

Command:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\126_bounded_lora_offline_scaleup.ps1
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

Expected interpretation: if the runner passes, refresh the offline TCA-Map/LoRA evidence table to include this bounded scale-up. Keep learned-policy rollout scaling and paper claims blocked until a separate valid rollout/checkpoint path exists.
