# VLM-Enabled Loading Risk Plan

This report defines the metadata-only risk gate for enabling the VLM path of the local SmolVLA checkpoint.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\111_plan_vlm_enabled_loading_risk.ps1
```

The planner uses the official Hugging Face model repo referenced by the local SmolVLA config:

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

It may query Hugging Face metadata to estimate file size and license/token risk, but it does not download model weights, install packages, load models, run inference, create simulator environments, rollout, train, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims.

The planner should record:

- source repo and whether it is official/public,
- license and gated/private status,
- required root model weight and tokenizer/config files,
- expected new disk use,
- free disk after acquisition,
- whether token/login/license/payment is required,
- expected CPU load-smoke RAM/VRAM/runtime,
- whether a later acquisition plan is safe.

If green, the next step is a separately gated VLM weight acquisition plan for required files only. The actual VLM-enabled load smoke remains a later gate.

Current local metadata result:

- source: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
- official/public: true,
- gated/private: false/false,
- license: `apache-2.0`,
- root model weight: `model.safetensors`,
- estimated required files: 12,
- estimated new disk: `1.895GB`,
- root weight size: `1.891GB`,
- free disk after estimate: about `419GB`,
- token/login/license/payment required: false,
- decision: `proceed`,
- ready for VLM weight acquisition plan: true,
- ready for VLM-enabled load smoke now: false.

This result authorizes only a future acquisition plan/runner for required VLM files. It does not authorize immediate model loading without a separate bounded load-smoke gate.
