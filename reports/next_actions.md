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

If checkpoint files are present, verify readiness and prepare a load-only adapter smoke plan. Do not perform heavy import, GPU inference, download, training, rollout, simulator execution, token access, or OpenVLA-OFT execution without explicit approval.

The planning command is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\15_plan_smolvla_load_only_smoke.ps1
```

The bounded execution scaffold is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\16_smolvla_load_only_smoke.ps1
```

It will not load a model without `ALLOW_HEAVY_IMPORT=1`, and it currently blocks on missing runtime dependencies rather than installing packages automatically.

Check runtime dependency readiness:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\17_check_smolvla_runtime_deps.ps1
```

Plan the explicit runtime install approval boundary without installing anything:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\18_plan_smolvla_runtime_install.ps1
```

Plan or validate the dummy feature-cache interface without SmolVLA imports:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\19_plan_feature_cache.ps1
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
4. Resolve runtime dependency plan for PyTorch/Transformers/LeRobot without changing CUDA/PyTorch automatically.
5. Request explicit approval before installing or changing runtime packages.
6. Feature cache interface validation with dummy cached features.
7. Tiny head-only pilot.
8. Later simulator rollout after LIBERO/RoboSuite/simulator paths pass checks.

Current hard-stop: installing or changing PyTorch/CUDA/LeRobot/Transformers/Safetensors requires explicit user approval.

## Self-Check Cases

Case A: SmolVLA checkpoint path missing or not configured. Codex reports exact missing path/config, updates state/action docs if needed, and stops at asset path gate.

Case B: SmolVLA path exists but config/tokenizer dependency/weights are missing. Codex reports exact missing file classes, updates state/action docs if needed, and stops at checkpoint-file gate.

Case C: Config/tokenizer dependency/weights are present and adapter-smoke-ready. Codex updates state/action docs and prepares the next safe load-only adapter smoke plan, then stops before heavy import or model load approval. The plan now exists; actual model loading is still gated.

Case D: Checker fails due to Windows/PATH/tooling. Codex diagnoses and fixes minimally on a new branch if safe, then validates again.

Case E: Dangerous gate reached. Codex stops and asks for explicit approval with risk explanation.

## Blocked Steps Requiring Explicit Approval

Codex must stop before:

- any actual download,
- setting `ALLOW_DOWNLOADS=1`,
- setting `ALLOW_HEAVY_IMPORT=1`,
- GPU inference,
- training,
- rollout,
- simulator execution,
- heavy SmolVLA/OpenVLA import,
- OpenVLA-OFT execution,
- token or secret access.

Codex should not ask routine questions such as whether files were placed, whether readiness should be checked, whether pytest should run, which branch is current, whether git is clean, or what is missing. It should inspect and report.

## Readiness Target

Proceed to a new load-only adapter smoke task only if:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
```

## Later Task

After readiness and planning are true, create a new branch for the separately approved SmolVLA load-only execution smoke. That branch should remain load/interface-only and must not infer, train, rollout, or execute OpenVLA-OFT.
