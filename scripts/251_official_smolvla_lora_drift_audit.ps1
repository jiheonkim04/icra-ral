param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CheckpointPath = "C:\assets\checkpoints\smolvla_libero",
    [string]$DatasetRoot = "C:\assets\datasets\lerobot_libero",
    [string]$HfHome = "C:\assets\hf_home",
    [string]$VlmRoot = "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct",
    [string]$SplitManifest = "reports\official_smolvla_split_manifest.json",
    [string]$MetricProtocol = "reports\official_smolvla_metric_protocol.md",
    [string]$StableArtifact = "reports\official_smolvla_stable_prediction_artifact.json",
    [string]$OldResultJson = "reports\official_smolvla_lora_seed_repro_result.json",
    [string]$RegeneratedResultJson = "reports\official_smolvla_lora_checkpoint_regen_result.json",
    [string]$CheckpointManifest = "reports\official_smolvla_lora_checkpoint_manifest.json",
    [string]$Seeds = "11,22,33",
    [int]$ChunkSize = 50,
    [string]$VideoBackend = "pyav",
    [int]$ProgressEvery = 200,
    [switch]$SkipEval
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

Write-Host "Official SmolVLA-LIBERO LoRA drift audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner audits old-vs-regenerated LoRA drift and re-evaluates persisted checkpoints from disk."
Write-Host "It does not train, install simulator dependencies, run rollout, download assets, run OpenVLA-OFT, revive FCAR, or design a method."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$cmd = @(
    "-m", "tca_map.smolvla.official_lora_drift_audit",
    "--checkpoint-path", $CheckpointPath,
    "--dataset-root", $DatasetRoot,
    "--hf-home", $HfHome,
    "--vlm-root", $VlmRoot,
    "--split-manifest", $SplitManifest,
    "--metric-protocol", $MetricProtocol,
    "--stable-artifact", $StableArtifact,
    "--old-result-json", $OldResultJson,
    "--regenerated-result-json", $RegeneratedResultJson,
    "--checkpoint-manifest", $CheckpointManifest,
    "--seeds", $Seeds,
    "--chunk-size", "$ChunkSize",
    "--video-backend", $VideoBackend,
    "--progress-every", "$ProgressEvery"
)

if ($SkipEval) {
    $cmd += "--skip-eval"
}

& $Python @cmd
exit $LASTEXITCODE
