param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CheckpointPath = "C:\assets\checkpoints\smolvla_libero",
    [string]$DatasetRoot = "C:\assets\datasets\lerobot_libero",
    [string]$HfHome = "C:\assets\hf_home",
    [string]$VlmRoot = "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct",
    [string]$SplitManifest = "reports\official_smolvla_split_manifest.json",
    [string]$MetricProtocol = "reports\official_smolvla_metric_protocol.md",
    [string]$StableArtifact = "reports\official_smolvla_stable_prediction_artifact.json",
    [string]$SeedArtifactPattern = "reports\official_smolvla_lora_seed_{seed}_prediction_artifact.json",
    [string]$JsonReportPath = "reports\official_smolvla_lora_seed_repro_result.json",
    [string]$MarkdownResultPath = "reports\official_smolvla_lora_seed_repro_result.md",
    [string]$PlanPath = "reports\official_smolvla_lora_seed_repro_plan.md",
    [string]$TablePath = "reports\official_smolvla_lora_seed_repro_table.md",
    [string]$DecisionPath = "reports\official_smolvla_lora_seed_repro_decision.md",
    [string]$Seeds = "11,22,33",
    [int]$Steps = 100,
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

Write-Host "Official SmolVLA-LIBERO rank-4 LoRA seed reproduction"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner trains only standard rank-4 LoRA baseline seeds under the fixed stable manifest."
Write-Host "It reuses the stable frozen/base artifact and evaluates each seed under the fixed metric protocol."
Write-Host "It does not implement a method, revive FCAR, train routing, download assets, run rollouts, full benchmark, OpenVLA-OFT, or the old custom LIBERO_7D route."
Write-Host "CUDA is required; CPU fallback is reported as CPU_FALLBACK_BUG."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$cmd = @(
    "-m", "tca_map.smolvla.official_libero_lora_seed_repro",
    "--checkpoint-path", $CheckpointPath,
    "--dataset-root", $DatasetRoot,
    "--hf-home", $HfHome,
    "--vlm-root", $VlmRoot,
    "--split-manifest", $SplitManifest,
    "--metric-protocol", $MetricProtocol,
    "--stable-artifact", $StableArtifact,
    "--seed-artifact-pattern", $SeedArtifactPattern,
    "--report-json", $JsonReportPath,
    "--result-md", $MarkdownResultPath,
    "--plan-md", $PlanPath,
    "--table-md", $TablePath,
    "--decision-md", $DecisionPath,
    "--seeds", $Seeds,
    "--steps", "$Steps",
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
