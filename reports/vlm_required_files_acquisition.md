# VLM Required Files Acquisition

This stage acquires only the bounded SmolVLM2 dependency files needed to plan a later VLM-enabled SmolVLA load smoke.

Official source:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

Target directory:

```text
C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct
```

Cache directory:

```text
C:\assets\hf_home
```

The prerequisite metadata-only planner is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\111_plan_vlm_enabled_loading_risk.ps1
```

The acquisition runner is:

```powershell
$env:ALLOW_DOWNLOADS="1"
powershell -ExecutionPolicy Bypass -File scripts\112_acquire_vlm_required_files.ps1
Remove-Item Env:\ALLOW_DOWNLOADS -ErrorAction SilentlyContinue
```

The runner is intentionally narrow. It does not load models, run inference, run training, run rollouts, use GPU jobs, execute OpenVLA-OFT, read tokens, install packages, or make paper-grade claims. It only proceeds when the source is official/public/ungated, file sizes are known and bounded, disk-after budget remains above 250GB, and no execution gate other than `ALLOW_DOWNLOADS=1` is set.

Passing this stage means the files are locally present for a future VLM-enabled load-smoke plan. It is not evidence that VLM-enabled policy loading works, and it is not manipulation evidence.
