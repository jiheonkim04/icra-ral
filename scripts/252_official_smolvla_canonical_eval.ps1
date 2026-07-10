param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CheckpointPath = "C:\assets\checkpoints\smolvla_libero",
    [string]$DatasetRoot = "C:\assets\datasets\lerobot_libero",
    [string]$HfHome = "C:\assets\hf_home",
    [string]$VlmRoot = "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct",
    [string]$SplitManifest = "reports\official_smolvla_split_manifest.json",
    [string]$CheckpointManifest = "reports\official_smolvla_lora_checkpoint_manifest.json",
    [string]$EvalSeeds = "101,202,303,404,505",
    [string]$LoraSeeds = "11,22,33",
    [int]$ChunkSize = 50,
    [string]$VideoBackend = "pyav",
    [int]$ProgressEvery = 100,
    [int]$RepeatSmokeCount = 5,
    [switch]$NoFullRepeat,
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

Write-Host "Official SmolVLA canonical persisted-checkpoint evaluator"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner does not train, regenerate checkpoints, run FCAR, run rollout, run OpenVLA-OFT, or use the old LIBERO_7D route."
Write-Host "CUDA inference is required. If CUDA is available but params/tensors are CPU, it reports CPU_FALLBACK_BUG."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$cmd = @(
    "-m", "tca_map.smolvla.official_canonical_eval",
    "--checkpoint-path", $CheckpointPath,
    "--dataset-root", $DatasetRoot,
    "--hf-home", $HfHome,
    "--vlm-root", $VlmRoot,
    "--split-manifest", $SplitManifest,
    "--checkpoint-manifest", $CheckpointManifest,
    "--eval-seeds", $EvalSeeds,
    "--lora-seeds", $LoraSeeds,
    "--chunk-size", "$ChunkSize",
    "--video-backend", $VideoBackend,
    "--progress-every", "$ProgressEvery",
    "--repeat-smoke-count", "$RepeatSmokeCount"
)

if ($NoFullRepeat) {
    $cmd += "--no-verify-full-repeat"
}
if ($Force) {
    $cmd += "--force"
}

& $Python @cmd
exit $LASTEXITCODE
