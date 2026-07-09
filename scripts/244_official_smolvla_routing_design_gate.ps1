param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CheckpointPath = "C:\assets\checkpoints\smolvla_libero",
    [string]$DatasetRoot = "C:\assets\datasets\lerobot_libero",
    [string]$HfHome = "C:\assets\hf_home",
    [string]$VlmRoot = "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct",
    [int]$Steps = 100,
    [int]$MaxEvalSamples = 200,
    [string]$JsonReportPath = "reports\official_smolvla_routing_design_gate.json",
    [string]$ReportDir = "reports"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HOME = $HfHome
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "Official SmolVLA-LIBERO routing design gate"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner regenerates bounded base/LoRA per-frame errors for oracle analysis only. It does not implement a method, run rollouts, full benchmark, OpenVLA-OFT, or downloads."
Write-Host "Requires task-local gates ALLOW_HEAVY_IMPORT=1 and ALLOW_GPU_TRAINING=1 after a green risk assessment."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.official_libero_routing_design_gate `
    --checkpoint-path $CheckpointPath `
    --dataset-root $DatasetRoot `
    --hf-home $HfHome `
    --vlm-root $VlmRoot `
    --steps $Steps `
    --max-eval-samples $MaxEvalSamples `
    --report-json $JsonReportPath `
    --report-dir $ReportDir

exit $LASTEXITCODE
