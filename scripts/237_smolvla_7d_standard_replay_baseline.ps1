param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$DataRoot = "C:\assets\data\libero\libero_10",
    [string]$SmolVlaCkpt = "C:\assets\checkpoints\smolvla",
    [string]$LiberoRoot = "C:\assets\repos\LIBERO",
    [string]$RobosuiteRoot = "C:\assets\repos\robosuite",
    [string]$ReportPath = "reports\smolvla_7d_standard_replay_baseline_result.json",
    [int]$MaxTasks = 2,
    [int]$TrainDemosPerTask = 5,
    [int]$EvalDemosPerTask = 2,
    [int]$ReplayDemosPerTask = 1,
    [int]$RecordsPerDemo = 8,
    [int]$AdapterSteps = 800,
    [int]$MlpSteps = 800,
    [int]$AdapterHiddenDim = 128,
    [int]$MaxReplaySteps = 280,
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

Write-Host "SmolVLA 7D standard replay baseline reproduction"
Write-Host "Repo root: $RepoRoot"
Write-Host "Requires ALLOW_SMOLVLA_7D_STANDARD_REPLAY_BASELINE=1, ALLOW_SMOLVLA_7D_STANDARD_REPLAY_BASELINE_TRAINING=1, and ALLOW_SMOLVLA_7D_STANDARD_REPLAY_BASELINE_REPLAY=1."
Write-Host "Does not download, run a full benchmark, run OpenVLA-OFT, continue TG/PatchGuard/SafeLoRA/PRISM, invent a method, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

& $Python -m tca_map.smolvla_lora_baseline.standard_replay_baseline `
    --data-root $DataRoot `
    --smolvla-ckpt $SmolVlaCkpt `
    --libero-root $LiberoRoot `
    --robosuite-root $RobosuiteRoot `
    --report-path $ReportPath `
    --max-tasks $MaxTasks `
    --train-demos-per-task $TrainDemosPerTask `
    --eval-demos-per-task $EvalDemosPerTask `
    --replay-demos-per-task $ReplayDemosPerTask `
    --records-per-demo $RecordsPerDemo `
    --adapter-steps $AdapterSteps `
    --mlp-steps $MlpSteps `
    --adapter-hidden-dim $AdapterHiddenDim `
    --max-replay-steps $MaxReplaySteps `
    --camera-size $CameraSize
exit $LASTEXITCODE
