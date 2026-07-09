param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$DatasetRoot = "C:\assets\datasets\lerobot_libero",
    [int]$Seed = 0,
    [string]$JsonReportPath = "reports\official_smolvla_stable_protocol_result.json",
    [string]$MarkdownReportPath = "reports\official_smolvla_stable_protocol_result.md",
    [string]$PlanPath = "reports\official_smolvla_stable_protocol_plan.md",
    [string]$SplitManifestPath = "reports\official_smolvla_split_manifest.md",
    [string]$SplitManifestJsonPath = "reports\official_smolvla_split_manifest.json",
    [string]$MetricProtocolPath = "reports\official_smolvla_metric_protocol.md",
    [string]$ArtifactPlanPath = "reports\official_smolvla_prediction_artifact_plan.md",
    [string]$DecisionPath = "reports\official_smolvla_stable_protocol_decision.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Host "Official SmolVLA-LIBERO stable split/metric protocol builder"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runner reads official LeRobot metadata and previous reports only."
Write-Host "It does not train, tune FCAR, implement a new method, download assets, run rollouts, full benchmark, OpenVLA-OFT, or the old custom LIBERO_7D route."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.official_libero_stable_protocol `
    --dataset-root $DatasetRoot `
    --seed $Seed `
    --result-json $JsonReportPath `
    --result-md $MarkdownReportPath `
    --plan-md $PlanPath `
    --split-manifest-md $SplitManifestPath `
    --split-manifest-json $SplitManifestJsonPath `
    --metric-md $MetricProtocolPath `
    --artifact-plan-md $ArtifactPlanPath `
    --decision-md $DecisionPath

exit $LASTEXITCODE
