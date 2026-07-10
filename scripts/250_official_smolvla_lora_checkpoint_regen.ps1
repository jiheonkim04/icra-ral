param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$CheckpointPath = "C:\assets\checkpoints\smolvla_libero",
    [string]$DatasetRoot = "C:\assets\datasets\lerobot_libero",
    [string]$HfHome = "C:\assets\hf_home",
    [string]$VlmRoot = "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct",
    [string]$CheckpointOutputRoot = "C:\assets\checkpoints\smolvla_libero_lora\rank4",
    [string]$SplitManifest = "reports\official_smolvla_split_manifest.json",
    [string]$MetricProtocol = "reports\official_smolvla_metric_protocol.md",
    [string]$StableArtifact = "reports\official_smolvla_stable_prediction_artifact.json",
    [string]$PriorResultJson = "reports\official_smolvla_lora_seed_repro_result.json",
    [string]$SourceReproLock = "configs\official_smolvla_repro_lock.yaml",
    [string]$SeedArtifactPattern = "reports\official_smolvla_seed_{seed}_prediction_artifact.json",
    [string]$JsonReportPath = "reports\official_smolvla_lora_checkpoint_regen_result.json",
    [string]$MarkdownResultPath = "reports\official_smolvla_lora_checkpoint_regen_result.md",
    [string]$PlanPath = "reports\official_smolvla_lora_checkpoint_regen_plan.md",
    [string]$TablePath = "reports\official_smolvla_lora_checkpoint_regen_table.md",
    [string]$DecisionPath = "reports\official_smolvla_lora_checkpoint_regen_decision.md",
    [string]$CheckpointManifestPath = "reports\official_smolvla_lora_checkpoint_manifest.json",
    [string]$VerificationPath = "reports\official_smolvla_lora_checkpoint_verification.md",
    [string]$ComparisonPath = "reports\official_smolvla_lora_reproduction_comparison.md",
    [string]$Seeds = "11,22,33",
    [int]$Steps = 100,
    [double]$LearningRate = 0.0002,
    [double]$ReproductionTolerance = 0.002,
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

Write-Host "Official SmolVLA-LIBERO rank-4 LoRA checkpoint regeneration"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner regenerates only standard rank-4 LoRA adapter checkpoints for the fixed seeds."
Write-Host "It saves adapter bundles, reloads adapters from disk, generates seed-specific prediction artifacts, and compares against frozen prior metrics."
Write-Host "It does not install libero/robosuite, initialize a simulator, run rollout, download assets, run OpenVLA-OFT, revive FCAR, or design a method."
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
    "--checkpoint-output-root", $CheckpointOutputRoot,
    "--split-manifest", $SplitManifest,
    "--metric-protocol", $MetricProtocol,
    "--stable-artifact", $StableArtifact,
    "--prior-result-json", $PriorResultJson,
    "--source-repro-lock", $SourceReproLock,
    "--seed-artifact-pattern", $SeedArtifactPattern,
    "--report-json", $JsonReportPath,
    "--result-md", $MarkdownResultPath,
    "--plan-md", $PlanPath,
    "--table-md", $TablePath,
    "--decision-md", $DecisionPath,
    "--checkpoint-manifest", $CheckpointManifestPath,
    "--verification-md", $VerificationPath,
    "--comparison-md", $ComparisonPath,
    "--seeds", $Seeds,
    "--steps", "$Steps",
    "--lr", "$LearningRate",
    "--reproduction-tolerance", "$ReproductionTolerance",
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
