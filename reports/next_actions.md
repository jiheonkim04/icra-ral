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

If checkpoint files are present, verify readiness and continue through the standing-approved bounded SmolVLA pilot path. Load-only construction has passed; the next bounded step is a single-sample interface smoke with synthetic or dummy inputs. Do not perform rollout, simulator execution, real benchmark evaluation, token access, OpenVLA-OFT execution, or work outside the SmolVLA autonomous pilot budget without explicit approval.

The planning command is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\15_plan_smolvla_load_only_smoke.ps1
```

The bounded execution scaffold is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\16_smolvla_load_only_smoke.ps1
```

It will not load a model without `ALLOW_HEAVY_IMPORT=1`. Runtime dependencies are now installed, and the bounded load-only smoke has passed on CPU. `ALLOW_HEAVY_IMPORT=1` may be set by Codex only inside standing-approved bounded SmolVLA tasks.

Check runtime dependency readiness:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
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

Plan the tiny head-only pilot approval boundary without training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\26_plan_tiny_head_only_pilot.ps1
```

Summarize the current hard-stop approval choices:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\27_summarize_hard_stop_status.ps1
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
4. Runtime dependency install completed under explicit approval.
5. Run standing-approved SmolVLA load-only heavy import/model construction. Done.
6. Create or run single-sample SmolVLA interface smoke with synthetic or dummy inputs.
7. Feature cache interface validation with dummy cached features.
8. Eval-only cached-feature head/metric smoke.
9. Tiny head-only pilot planning and budget check.
10. Tiny head-only smoke if within standing-approved budget.
11. Summarize hard-stop approval choices.
12. Later simulator rollout after LIBERO/RoboSuite/simulator paths pass checks.

Current hard-stop: none for the next bounded single-sample interface smoke if readiness remains true and runtime stays within the autonomous pilot budget. Future package upgrades, CUDA/PyTorch major changes, OpenVLA-OFT, rollouts, simulator execution, real benchmark evaluation, tokens, multi-seed work, or paper claims still require separate explicit approval.

## Self-Check Cases

Case A: SmolVLA checkpoint path missing or not configured. Codex reports exact missing path/config, updates state/action docs if needed, and stops at asset path gate.

Case B: SmolVLA path exists but config/tokenizer dependency/weights are missing. Codex reports exact missing file classes, updates state/action docs if needed, and stops at checkpoint-file gate.

Case C: Config/tokenizer dependency/weights and runtime dependencies are present and adapter-smoke-ready. Codex updates state/action docs if needed and continues to the next standing-approved bounded SmolVLA pilot task. Since load-only construction has passed, the next task is single-sample interface smoke.

Case D: Checker fails due to Windows/PATH/tooling. Codex diagnoses and fixes minimally on a new branch if safe, then validates again.

Case E: Dangerous gate reached. Codex stops and asks for explicit approval with risk explanation.

## Blocked Steps Requiring Explicit Approval

Codex must stop before true hard-stop gates:

- OpenVLA-OFT download/import/load/execution,
- LIBERO/RoboSuite/RoboCasa/dataset download,
- simulator execution,
- rollout,
- real benchmark evaluation,
- training longer than 15 minutes or more than 100 steps,
- any job expected to exceed 30 minutes,
- using more than 14GB VRAM,
- changing CUDA/PyTorch major versions,
- installing large unplanned packages,
- token or secret access.
- multi-seed experiment,
- paper-level empirical claim,
- external submission/upload/publishing.

Codex should not ask routine questions such as whether files were placed, whether readiness should be checked, whether pytest should run, which branch is current, whether git is clean, or what is missing. It should inspect and report.

## Readiness Target

Proceed to a new load-only adapter smoke task only if:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
```

## Later Task

After readiness, planning, and load-only smoke are true, continue on a new branch for the standing-approved single-sample SmolVLA interface smoke. That branch should remain interface-only, use synthetic or dummy inputs, and must not train, rollout, evaluate datasets, or execute OpenVLA-OFT.
