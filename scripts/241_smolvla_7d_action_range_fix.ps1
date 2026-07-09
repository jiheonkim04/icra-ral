param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$DataRoot = "C:\assets\data\libero\libero_10",
    [string]$SmolVlaCkpt = "C:\assets\checkpoints\smolvla",
    [string]$LiberoRoot = "C:\assets\repos\LIBERO",
    [string]$RobosuiteRoot = "C:\assets\repos\robosuite",
    [string]$AdapterDir = "runs\smolvla_7d_standard_replay_baseline",
    [string]$ExactInitReportPath = "reports\exact_init_expert_replay_stabilization.json",
    [string]$ReportPath = "reports\smolvla_7d_action_range_fix.json",
    [int]$MaxTasks = 2,
    [int]$TrainDemosPerTask = 5,
    [int]$EvalDemosPerTask = 2,
    [int]$RecordsPerDemo = 8,
    [int]$AdapterSteps = 800,
    [int]$MlpSteps = 800,
    [int]$AdapterHiddenDim = 128,
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

Write-Host "SmolVLA 7D action range and controller-validity fix"
Write-Host "Repo root: $RepoRoot"
Write-Host "Requires ALLOW_SMOLVLA_7D_ACTION_RANGE_FIX=1."
Write-Host "Does not download, run OpenVLA-OFT, run a full benchmark, revive TG/PatchGuard/SafeLoRA/PRISM/ActionMap, invent a method, use eval-label calibration, hard-code gripper fill, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

& $Python -m tca_map.smolvla_lora_baseline.action_range_fix `
    --data-root $DataRoot `
    --smolvla-ckpt $SmolVlaCkpt `
    --libero-root $LiberoRoot `
    --robosuite-root $RobosuiteRoot `
    --adapter-dir $AdapterDir `
    --exact-init-report-path $ExactInitReportPath `
    --report-path $ReportPath `
    --max-tasks $MaxTasks `
    --train-demos-per-task $TrainDemosPerTask `
    --eval-demos-per-task $EvalDemosPerTask `
    --records-per-demo $RecordsPerDemo `
    --adapter-steps $AdapterSteps `
    --mlp-steps $MlpSteps `
    --adapter-hidden-dim $AdapterHiddenDim `
    --max-replay-steps $MaxReplaySteps `
    --post-signal-margin $PostSignalMargin `
    --camera-size $CameraSize
exit $LASTEXITCODE
