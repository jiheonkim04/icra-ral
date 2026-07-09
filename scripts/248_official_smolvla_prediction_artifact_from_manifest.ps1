param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CheckpointPath = "C:\assets\checkpoints\smolvla_libero",
    [string]$DatasetRoot = "C:\assets\datasets\lerobot_libero",
    [string]$HfHome = "C:\assets\hf_home",
    [string]$VlmRoot = "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct",
    [string]$SplitManifest = "reports\official_smolvla_split_manifest.json",
    [string]$MetricProtocol = "reports\official_smolvla_metric_protocol.md",
    [string]$Output = "reports\official_smolvla_stable_prediction_artifact.json",
    [string]$JsonReportPath = "reports\official_smolvla_stable_artifact_eval_result.json",
    [string]$MarkdownResultPath = "reports\official_smolvla_stable_artifact_eval_result.md",
    [string]$StatusPath = "reports\official_smolvla_stable_prediction_artifact_status.md",
    [string]$BaselineTablePath = "reports\official_smolvla_stable_baseline_table.md",
    [string]$DecisionPath = "reports\official_smolvla_stable_artifact_decision.md",
    [int]$Steps = 100,
    [int]$Seed = 0,
    [int]$ChunkSize = 50,
    [string]$VideoBackend = "pyav",
    [switch]$NoEvalLoss,
    [switch]$Force
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

Write-Host "Official SmolVLA-LIBERO stable prediction artifact runner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner uses the fixed manifest and metric protocol from the stable protocol commit."
Write-Host "It may train only the standard rank-4 LoRA baseline on the manifest train split."
Write-Host "It does not implement a method, revive FCAR, download assets, run rollouts, full benchmark, OpenVLA-OFT, or the old custom LIBERO_7D route."
Write-Host "CUDA is required for LoRA regeneration; CPU fallback is reported as CPU_FALLBACK_BUG."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$cmd = @(
    "-m", "tca_map.smolvla.official_libero_stable_artifact_eval",
    "--checkpoint-path", $CheckpointPath,
    "--dataset-root", $DatasetRoot,
    "--hf-home", $HfHome,
    "--vlm-root", $VlmRoot,
    "--split-manifest", $SplitManifest,
    "--metric-protocol", $MetricProtocol,
    "--output-artifact", $Output,
    "--report-json", $JsonReportPath,
    "--result-md", $MarkdownResultPath,
    "--status-md", $StatusPath,
    "--baseline-table-md", $BaselineTablePath,
    "--decision-md", $DecisionPath,
    "--steps", "$Steps",
    "--seed", "$Seed",
    "--chunk-size", "$ChunkSize",
    "--video-backend", $VideoBackend
)

if ($NoEvalLoss) {
    $cmd += "--no-include-eval-loss"
}
if ($Force) {
    $cmd += "--force"
}

& $Python @cmd
exit $LASTEXITCODE
