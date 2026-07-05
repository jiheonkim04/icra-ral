param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$JsonReportPath = "reports\libero_publishability_gate_audit_report.json",
    [string]$MarkdownReportPath = "reports\libero_publishability_gate_audit_report.md",
    [string]$Seeds = "11,23,37",
    [int]$MaxPairs = 32,
    [int]$MaxActionSteps = 16,
    [int]$MaxSteps = 64,
    [int]$MaxRuntimeSeconds = 900,
    [int]$MaxSamples = 64,
    [int]$Rank = 4
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO publishability gate audit"
Write-Host "Repo root: $RepoRoot"
Write-Host "This bounded audit uses only local LIBERO offline proxy data and CPU NumPy heads/LoRA. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

$seedValues = @($Seeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($MaxSamples -ne 64) {
    Write-Host "Refusing: publishability gate audit script must run the 64-record split."
    exit 11
}

if ($seedValues.Count -lt 1 -or $seedValues.Count -gt 3) {
    Write-Host "Refusing: 64-record publishability audit seed count must be between 1 and 3."
    exit 12
}

if ($MaxSteps -lt 1 -or $MaxSteps -gt 300) {
    Write-Host "Refusing: MaxSteps must be between 1 and 300."
    exit 13
}

if ($MaxRuntimeSeconds -lt 1 -or $MaxRuntimeSeconds -gt 900) {
    Write-Host "Refusing: MaxRuntimeSeconds must be between 1 and 900."
    exit 14
}

if ($Rank -lt 1 -or $Rank -gt 16) {
    Write-Host "Refusing: Rank must be between 1 and 16."
    exit 15
}

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF",
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
    Write-Host ("Refusing to run publishability gate audit while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

if ($env:ALLOW_TINY_TRAINING -ne "1") {
    Write-Host "Refusing: ALLOW_TINY_TRAINING=1 is required for bounded CPU offline audit reconstruction."
    exit 21
}

& $Python -m tca_map.datasets.libero_publishability_gate_audit `
    --manifest $ManifestPath `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath `
    --seeds $Seeds `
    --max-pairs $MaxPairs `
    --max-action-steps $MaxActionSteps `
    --max-steps $MaxSteps `
    --max-runtime-seconds $MaxRuntimeSeconds `
    --max-samples $MaxSamples `
    --rank $Rank

exit $LASTEXITCODE
