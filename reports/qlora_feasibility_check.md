# QLoRA Feasibility Check

## Purpose

This check determines whether QLoRA is feasible under the current low-compute local policy without installing packages, changing CUDA/PyTorch, importing heavy VLA models, loading models, training, using GPU, downloading assets, running rollouts, executing OpenVLA-OFT, or making paper claims.

QLoRA is a required feasibility track if memory/tooling allows. It is not the main novelty.

## Command

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\35_check_qlora_feasibility.ps1
```

It writes an ignored runtime report:

```text
reports\qlora_feasibility_report.json
```

## Feasibility Criteria

QLoRA is locally feasible only if:

- the QLoRA config passes local policy guards,
- no full backbone fine-tuning is enabled,
- no OpenVLA-OFT execution is required,
- required adapter modules remain target fusion, action head projection, or small adapter layers,
- QLoRA tooling is present without installing new packages,
- no CUDA/PyTorch major changes are required,
- memory estimate stays under the 14GB local target.

If tooling is missing or Windows-native support is uncertain, QLoRA remains a required tracked arm but should be deferred to Linux/WSL/cloud handoff rather than forcing local package changes.
