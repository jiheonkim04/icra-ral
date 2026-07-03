# Next Actions

## Immediate Autonomous Behavior

Codex should self-check current state instead of asking the user to confirm it.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\14_plan_smolvla_download.ps1
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
```

If checkpoint files are missing, report the missing file classes and stop at the checkpoint-file gate.

If checkpoint files are present, verify readiness and continue through the risk-assessed bounded SmolVLA pilot path. Load-only construction has passed; the next bounded step is selected by risk assessment. Do not perform rollout, simulator execution, real benchmark evaluation, token access, OpenVLA-OFT execution, or work outside the SmolVLA autonomous pilot budget unless the risk assessment says proceed.

The planning command is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\15_plan_smolvla_load_only_smoke.ps1
```

The bounded execution scaffold is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\16_smolvla_load_only_smoke.ps1
```

It will not load a model without `ALLOW_HEAVY_IMPORT=1`. Runtime dependencies are now installed, and the bounded load-only smoke has passed on CPU. `ALLOW_HEAVY_IMPORT=1` may be set by Codex only inside risk-assessed bounded SmolVLA tasks.

Check runtime dependency readiness:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
```

Run the bounded single-sample interface smoke only after load-only passes:

```powershell
$env:ALLOW_HEAVY_IMPORT="1"
$env:ALLOW_SINGLE_SAMPLE_INFERENCE="1"
powershell -ExecutionPolicy Bypass -File scripts\28_smolvla_single_sample_interface_smoke.ps1
Remove-Item Env:\ALLOW_SINGLE_SAMPLE_INFERENCE -ErrorAction SilentlyContinue
Remove-Item Env:\ALLOW_HEAVY_IMPORT -ErrorAction SilentlyContinue
```

Recheck the completed runtime install state without importing heavy VLA models:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\18_plan_smolvla_runtime_install.ps1
```

Plan or validate the dummy feature-cache interface without SmolVLA imports:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\19_plan_feature_cache.ps1
```

Run the eval-only cached-feature smoke with dummy cached features:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\25_eval_feature_cache_smoke.ps1 -PrepareDummyCache
```

Plan the tiny head-only pilot risk boundary without training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\26_plan_tiny_head_only_pilot.ps1
```

Run the bounded tiny head-only smoke runner:

```powershell
$env:ALLOW_TINY_TRAINING="1"
powershell -ExecutionPolicy Bypass -File scripts\29_tiny_head_only_smoke.ps1 -PrepareDummyCache
Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
```

This is a tiny CPU head-only smoke on cached/dummy features. It is not a paper-grade result.

Summarize the current risk-gate choices:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\27_summarize_hard_stop_status.ps1
```

Generate the go/no-go status summary:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\31_generate_go_no_go_report.ps1
```

## Current Asset State

The approved SmolVLA checkpoint source has been acquired:

```text
lerobot/smolvla_base
```

Target directory:

```text
C:\assets\checkpoints\smolvla
```

Cache directory:

```text
C:\assets\hf_home
```

Detected groups:

- config: present (`config.json`),
- weights: present (`model.safetensors` and processor safetensors),
- tokenizer/processor dependency: present under `C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct`.

The dependency was acquired with full model weights avoided. Retained files include tokenizer JSON/config, vocab/merges, special tokens, chat template, preprocessor config, processor config, and config.

Bounded load-only status:

- passed on CPU,
- local checkpoint loaded,
- `load_vlm_weights=false`,
- no model inference,
- no training,
- no rollout,
- no OpenVLA-OFT execution,
- no downloads.

Verify files without loading the model:

```powershell
Test-Path C:\assets\checkpoints\smolvla
Get-ChildItem C:\assets\checkpoints\smolvla
Test-Path C:\assets\checkpoints\smolvla\config.json
Get-ChildItem C:\assets\checkpoints\smolvla -Filter *.safetensors
Get-ChildItem C:\assets\checkpoints\smolvla -Filter *.bin
```

## Expected Order

1. Manual SmolVLA acquisition checklist.
2. Readiness recheck.
3. Load-only adapter smoke planning.
4. Runtime dependency install completed under the earlier install gate. Done.
5. Run risk-assessed SmolVLA load-only heavy import/model construction. Done.
6. Create or run single-sample SmolVLA interface smoke with synthetic or dummy inputs. Done.
7. Feature cache interface validation with dummy cached features. Done.
8. Eval-only cached-feature head/metric smoke. Done.
9. Tiny head-only pilot planning and budget check. Done.
10. Tiny head-only smoke runner with bounded steps, no rollout, no OpenVLA-OFT, no paper claim, runtime<=30 minutes, and VRAM<=14GB. Done.
11. Summarize risk-gate choices. Done.
12. Generate go/no-go status summary. Done.
13. Required LoRA adapter construction plan. Done.
14. Required LoRA tiny smoke scaffold. Done.
15. Required TCA-Map + LoRA comparison plan. Done.
16. QLoRA feasibility check. Done.
17. Update LoRA/QLoRA go/no-go status. Done.
18. Bounded local pilot execution is risk-assessed autonomous if inside limits. Done: head-only ActionMap vs TCA-Map comparison report.
19. Tiny LoRA smoke runner inside risk-assessed budget. Done.
20. Tiny LoRA comparison report. Done.
21. Consolidated local pilot status/report inside risk-assessed budget. Done.
22. Approval-based hard-stops replaced by risk-assessed autonomous execution. Done.
23. LIBERO/LIBERO-CF-style dataset readiness/tiny subset risk planner. Done.
24. Simulator readiness planning without simulator import/render/rollout. Done.
25. Local pilot budget alignment to 300 steps. Done.
26. Bounded cached-feature local pilot extension inside the 300-step policy. Done.
27. Next: update consolidated status/go-no-go summaries to include the bounded extension report. Done when `scripts\39_generate_local_pilot_status.ps1` and `scripts\31_generate_go_no_go_report.ps1` report it.
28. Candidate next stages after status consolidation: bounded simulator import-smoke if local WSL2/Linux paths are green, real dataset/tiny-subset setup if local paths or official source are ready, or larger compute handoff planning.

Current status: the bounded tiny head-only smoke, ActionMap vs TCA-Map head-only comparison report, tiny LoRA smoke runner, tiny LoRA comparison report, consolidated local pilot status, go/no-go summary, required LoRA adapter construction plan, LoRA tiny-smoke scaffold, TCA-Map + LoRA comparison plan, QLoRA feasibility check, LoRA/QLoRA go/no-go update, LIBERO dataset risk planner, simulator readiness planner, and 300-step local pilot budget alignment have passed. LoRA/QLoRA are required experimental tracks after the head-only path, but not the main novelty. The repository now uses risk-assessed autonomous execution instead of broad approval-based hard-stops. Downloads, GPU tasks, bounded training, real dataset setup, simulator readiness, and bounded rollout should proceed automatically if a risk assessment is green and should stop only if risk is ambiguous, outside budget, external/irreversible, OpenVLA-OFT-related, or paper-claim-related.

Planning command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\32_plan_lora_adapter_construction.ps1
powershell -ExecutionPolicy Bypass -File scripts\33_plan_lora_tiny_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts\34_plan_lora_comparison.ps1
powershell -ExecutionPolicy Bypass -File scripts\35_check_qlora_feasibility.ps1
powershell -ExecutionPolicy Bypass -File scripts\36_compare_head_only_tiny_pilot.ps1
$env:ALLOW_TINY_TRAINING="1"; powershell -ExecutionPolicy Bypass -File scripts\37_tiny_lora_smoke.ps1 -PrepareDummyCache; Remove-Item Env:\ALLOW_TINY_TRAINING -ErrorAction SilentlyContinue
powershell -ExecutionPolicy Bypass -File scripts\38_compare_tiny_lora_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\39_generate_local_pilot_status.ps1
```

## Required LoRA/QLoRA Progression

A. SmolVLA load-only smoke. Done.
B. Single-sample interface smoke. Done.
C. Frozen/head-only TCA-Map tiny pilot. Done as bounded smoke.
D. Required LoRA adapter construction plan. Done.
E. Required LoRA tiny smoke scaffold. Done; bounded tiny execution runner and comparison report exist.
F. Required TCA-Map + LoRA comparison plan. Done.
G. QLoRA feasibility check. Done.
H. Go/no-go report. Done.

LoRA/QLoRA are required adaptation tracks, not optional nice-to-have items. Full fine-tuning remains forbidden locally.

## Self-Check Cases

Case A: SmolVLA checkpoint path missing or not configured. Codex reports exact missing path/config, updates state/action docs if needed, and stops at asset path gate.

Case B: SmolVLA path exists but config/tokenizer dependency/weights are missing. Codex reports exact missing file classes, updates state/action docs if needed, and stops at checkpoint-file gate.

Case C: Config/tokenizer dependency/weights and runtime dependencies are present and adapter-smoke-ready. Codex updates state/action docs if needed and continues to the next risk-assessed bounded SmolVLA pilot task. Since load-only construction, single-sample interface smoke, and dummy feature-cache/interface validation have passed, the next task is selected by risk assessment.

Case D: Checker fails due to Windows/PATH/tooling. Codex diagnoses and fixes minimally on a new branch if safe, then validates again.

Case E: Risk gate reached. Codex runs risk assessment. If the task is inside budget, proceed; if outside/ambiguous/external irreversible/OpenVLA/paper-claim-related, stop and report the blocker.

## Risk-Assessed Next Gates

Codex must not ask for routine approval when risk can be checked automatically. Before downloads, GPU, training, dataset setup, simulator readiness, or rollout, write/print a risk assessment with task, source, expected size, target path, disk free before/after estimate, expected runtime, expected RAM/VRAM, budget, source status, token/license/payment status, decision, and reason.

Structured helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\41_risk_assess_task.ps1 -Task "next concrete task" -Category "generic"
```

Proceed automatically if inside budget. Stop and report if risk is ambiguous or outside budget.

Default autonomous budgets:

- download <=80GB per task and keep >=100GB free disk,
- GPU VRAM <=14GB, runtime <=30 minutes, batch size 1,
- SmolVLA-only local training, frozen backbone or LoRA/QLoRA adapter only, max 300 steps after smaller smoke is stable,
- dataset setup only from official/documented unambiguous sources without token/login/payment/license click-through,
- simulator readiness/import-render smoke only if already installed locally and <=10 minutes,
- bounded rollout only after readiness smoke, task count <=5, runtime <=30 minutes, no OpenVLA-OFT, no paper claim.

Always stop before:

- OpenVLA-OFT execution until a separate risk budget exists,
- token/secret/API key access,
- paid services,
- license click-through,
- external upload/submission/publishing,
- deleting user files outside approved cache/repo cleanup,
- system-wide CUDA/PyTorch/driver changes,
- admin/system-level installers,
- paper-level empirical claims.

Codex should not ask routine questions such as whether files were placed, whether readiness should be checked, whether pytest should run, which branch is current, whether git is clean, or what is missing. It should inspect and report.

## Readiness Target

Proceed to a new load-only adapter smoke task only if:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
```

## Later Task

After readiness, planning, load-only smoke, single-sample interface smoke, and feature-cache/interface validation are true, continue on a new branch for a tiny head-only smoke runner. That branch may run only bounded head-only smoke and must not train a backbone, rollout, evaluate real datasets, or execute OpenVLA-OFT.

After the LoRA/QLoRA go/no-go update, tiny LoRA smoke runner, tiny LoRA comparison report, consolidated local pilot status report, risk-assessed policy update, LIBERO dataset risk planner, simulator readiness planner, 300-step budget alignment, and bounded local pilot extension, continue autonomously by choosing the next concrete task and running its risk assessment. The current next concrete task is status consolidation for the bounded extension. It must remain summary-only and must not use real datasets, import simulators, render, rollout, download, train, execute OpenVLA-OFT, or make paper claims. Stop only if the assessment is ambiguous/outside budget or reaches external irreversible, OpenVLA-OFT, token/secret/payment/license, system-level, or paper-claim gates.
