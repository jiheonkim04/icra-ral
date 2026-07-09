param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CheckpointPath = "C:\assets\checkpoints\smolvla_libero",
    [string]$DatasetRoot = "C:\assets\datasets\lerobot_libero",
    [string]$HfHome = "C:\assets\hf_home",
    [string]$VlmRoot = "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct",
    [int]$Steps = 100,
    [int]$MaxEvalSamples = 200,
    [string]$PredictionArtifact = "reports\fcar_prediction_artifact.json",
    [string]$JsonReportPath = "reports\fcar_tiny_gate_result.json",
    [string]$MarkdownReportPath = "reports\fcar_tiny_gate_result.md",
    [string]$DecisionReportPath = "reports\fcar_tiny_gate_decision.md"
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

Write-Host "FCAR tiny-gate runner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner trains only the FCAR tiny gate, and regenerates the fixed rank-4 LoRA prediction baseline only if the per-frame artifact is missing."
Write-Host "It does not train the SmolVLA backbone, download assets, run rollouts, full benchmark, OpenVLA-OFT, or the old custom LIBERO_7D route."
Write-Host "If prediction regeneration is needed, set ALLOW_HEAVY_IMPORT=1 and ALLOW_GPU_TRAINING=1 for this command."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python scripts\run_fcar_tiny_gate.py `
    --checkpoint-path $CheckpointPath `
    --dataset-root $DatasetRoot `
    --hf-home $HfHome `
    --vlm-root $VlmRoot `
    --steps $Steps `
    --max-eval-samples $MaxEvalSamples `
    --prediction-artifact $PredictionArtifact `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath `
    --decision-md $DecisionReportPath

exit $LASTEXITCODE
