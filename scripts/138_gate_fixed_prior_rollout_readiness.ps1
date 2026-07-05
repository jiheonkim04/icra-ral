param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$JsonReportPath = "reports\libero_fixed_prior_rollout_readiness_gate_report.json",
    [string]$MarkdownReportPath = "reports\libero_fixed_prior_rollout_readiness_gate_report.md",
    [int]$MaxPairs = 32,
    [int]$MaxActionSteps = 16,
    [int]$EnvActionDim = 7
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Fixed-prior rollout readiness gate"
Write-Host "Repo root: $RepoRoot"
Write-Host "This gate performs no rollout, training, model loading, heavy VLA import, GPU job, download, OpenVLA-OFT execution, or paper-grade claim."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

if (-not (Test-Path -LiteralPath $ManifestPath)) {
    Write-Host "Refusing: manifest not found: $ManifestPath"
    exit 11
}

if ($MaxPairs -lt 1 -or $MaxPairs -gt 32) {
    Write-Host "Refusing: MaxPairs must be between 1 and 32."
    exit 12
}

if ($MaxActionSteps -lt 1 -or $MaxActionSteps -gt 32) {
    Write-Host "Refusing: MaxActionSteps must be between 1 and 32."
    exit 13
}

if ($EnvActionDim -lt 1 -or $EnvActionDim -gt 16) {
    Write-Host "Refusing: EnvActionDim must be between 1 and 16."
    exit 14
}

$forbiddenGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP"
)

$setForbidden = @()
foreach ($gate in $forbiddenGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setForbidden += $gate
    }
}

if ($setForbidden.Count -gt 0) {
    Write-Host ("Refusing to run fixed-prior rollout readiness gate while execution gates are set: " + ($setForbidden -join ", "))
    exit 20
}

& $Python -m tca_map.datasets.libero_fixed_prior_rollout_readiness `
    --manifest $ManifestPath `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath `
    --max-pairs $MaxPairs `
    --max-action-steps $MaxActionSteps `
    --env-action-dim $EnvActionDim

exit $LASTEXITCODE
