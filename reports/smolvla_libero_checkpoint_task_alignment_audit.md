# SmolVLA/LIBERO Checkpoint-Task Alignment Audit

This report defines a report-only checkpoint/task alignment audit after the reset-only and HDF5-init-state learned-policy diagnostics all produced zero reward.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\104_audit_smolvla_libero_checkpoint_task_alignment.ps1
```

The audit reads only local SmolVLA config/preprocessor files, local LIBERO BDDL names, and existing diagnostic reports. It must not download, install, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper-grade claims.

Expected interpretation:

- `decision=no_go_rollout_scaling`: current learned-policy rollout scaling remains blocked, but a planning-only offline demonstration-conditioned action decoding gate is a safe next step.
- `decision=stop`: required local audit inputs are missing or an execution gate was set.

This is diagnostic evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence.
