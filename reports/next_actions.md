# Next Actions

## Immediate Safe Steps

1. Read `reports/smolvla_manual_acquisition_checklist.md`.
2. Manually place a valid SmolVLA-compatible checkpoint under:

```text
C:\assets\checkpoints\smolvla
```

3. Verify files without loading the model:

```powershell
Test-Path C:\assets\checkpoints\smolvla
Get-ChildItem C:\assets\checkpoints\smolvla
Test-Path C:\assets\checkpoints\smolvla\config.json
Get-ChildItem C:\assets\checkpoints\smolvla -Filter *.safetensors
Get-ChildItem C:\assets\checkpoints\smolvla -Filter *.bin
```

4. Rerun readiness checks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\11_check_real_assets.ps1
powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1
```

## Expected Order

1. Manual SmolVLA acquisition checklist.
2. User places checkpoint files manually or explicitly approves a download task.
3. Readiness recheck.
4. Load-only adapter smoke planning.
5. Heavy import/GPU only with explicit approval.
6. Feature cache planning and implementation.
7. Tiny head-only pilot.
8. Later simulator rollout after LIBERO/RoboSuite/simulator paths pass checks.

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

## Readiness Target

Proceed to a new load-only adapter smoke task only if:

```text
ready_for_smolvla_path_check=true
smolvla_checkpoint_files_present=true
ready_for_smolvla_adapter_smoke=true
```

## Later Task

After readiness is true, create a new branch for SmolVLA load-only adapter smoke. That branch should remain load/interface-only and must not train, rollout, or execute OpenVLA-OFT.
