param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$DataRoot = "C:\assets\data\libero\libero_10",
    [string]$LiberoRoot = "C:\assets\repos\LIBERO",
    [string]$RobosuiteRoot = "C:\assets\repos\robosuite",
    [string]$AdapterDir = "runs\smolvla_7d_standard_replay_baseline",
    [string]$ExactInitReportPath = "reports\exact_init_expert_replay_stabilization.json",
    [string]$ReportPath = "reports\smolvla_7d_offline_to_control_gap.json",
    [int]$MaxReplaySteps = 320,
    [int]$PostSignalMargin = 16,
    [int]$CameraSize = 64
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:MUJOCO_GL = if ($env:MUJOCO_GL) { $env:MUJOCO_GL } else { "glfw" }

Write-Host "SmolVLA 7D offline-to-control gap diagnosis"
Write-Host "Repo root: $RepoRoot"
Write-Host "Requires ALLOW_SMOLVLA_7D_OFFLINE_TO_CONTROL_GAP=1."
Write-Host "Does not train, download, run OpenVLA-OFT, continue TG/PatchGuard/SafeLoRA/PRISM, invent a method, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

& $Python -m tca_map.smolvla_lora_baseline.offline_to_control_gap `
    --data-root $DataRoot `
    --libero-root $LiberoRoot `
    --robosuite-root $RobosuiteRoot `
    --adapter-dir $AdapterDir `
    --exact-init-report-path $ExactInitReportPath `
    --report-path $ReportPath `
    --max-replay-steps $MaxReplaySteps `
    --post-signal-margin $PostSignalMargin `
    --camera-size $CameraSize
exit $LASTEXITCODE
