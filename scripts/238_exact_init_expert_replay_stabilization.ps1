param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$DataRoot = "C:\assets\data\libero\libero_10",
    [string]$LiberoRoot = "C:\assets\repos\LIBERO",
    [string]$RobosuiteRoot = "C:\assets\repos\robosuite",
    [string]$AdapterDir = "runs\smolvla_7d_standard_replay_baseline",
    [string]$PriorResultPath = "reports\smolvla_7d_standard_replay_baseline_result.json",
    [string]$ReportPath = "reports\exact_init_expert_replay_stabilization.json",
    [int]$MaxTasks = 2,
    [int]$TrainDemosPerTask = 5,
    [int]$EvalDemosPerTask = 2,
    [int]$RecordsPerDemo = 8,
    [int]$CandidateDemosPerTask = 4,
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

Write-Host "Exact-init expert replay stabilization and eligibility-set construction"
Write-Host "Repo root: $RepoRoot"
Write-Host "Requires ALLOW_EXACT_INIT_EXPERT_REPLAY_STABILIZATION=1."
Write-Host "Set ALLOW_EXACT_INIT_EXPERT_REPLAY_STABILIZATION_LEARNED=1 only after the expert eligibility set is green."
Write-Host "Does not download, run a full benchmark, run OpenVLA-OFT, continue TG/PatchGuard/SafeLoRA, invent a method, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

& $Python -m tca_map.smolvla_lora_baseline.exact_init_replay_stabilization `
    --data-root $DataRoot `
    --libero-root $LiberoRoot `
    --robosuite-root $RobosuiteRoot `
    --adapter-dir $AdapterDir `
    --prior-result-path $PriorResultPath `
    --report-path $ReportPath `
    --max-tasks $MaxTasks `
    --train-demos-per-task $TrainDemosPerTask `
    --eval-demos-per-task $EvalDemosPerTask `
    --records-per-demo $RecordsPerDemo `
    --candidate-demos-per-task $CandidateDemosPerTask `
    --max-replay-steps $MaxReplaySteps `
    --post-signal-margin $PostSignalMargin `
    --camera-size $CameraSize
exit $LASTEXITCODE
