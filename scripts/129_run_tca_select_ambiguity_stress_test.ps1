param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_report.json",
    [string]$JsonReportPath = "reports\tca_select_ambiguity_stress_report.json",
    [string]$MarkdownReportPath = "reports\tca_select_ambiguity_stress_report.md",
    [int]$MaxPairs = 16,
    [int]$MaxRecords = 64,
    [int]$CandidateCount = 8,
    [double]$Temperature = 0.5,
    [int]$MaxRuntimeSeconds = 300
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Offline TCA-Select ambiguity stress test"
Write-Host "Repo root: $RepoRoot"
Write-Host "This CPU-only script uses existing local LIBERO counterfactual artifacts. It does not train, download, import heavy VLA models, load models, infer with SmolVLA, use GPU jobs, rollout, execute simulators, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT"
)

$setDangerousGates = @()
foreach ($gate in $dangerousGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setDangerousGates += $gate
    }
}

if ($setDangerousGates.Count -gt 0) {
    Write-Host ("Refusing to run offline TCA-Select ambiguity stress test while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

& $Python -m tca_map.smolvla.tca_select_ambiguity_stress `
    --manifest $ManifestPath `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath `
    --max-pairs $MaxPairs `
    --max-records $MaxRecords `
    --candidate-count $CandidateCount `
    --temperature $Temperature `
    --max-runtime-seconds $MaxRuntimeSeconds
exit $LASTEXITCODE
