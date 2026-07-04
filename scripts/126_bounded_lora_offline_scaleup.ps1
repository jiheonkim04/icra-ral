param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_report.json",
    [string]$JsonReportPath = "reports\bounded_lora_offline_scaleup_report.json",
    [string]$MarkdownReportPath = "reports\bounded_lora_offline_scaleup_report.md",
    [int]$MaxPairs = 16,
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

Write-Host "Bounded LIBERO offline LoRA scale-up"
Write-Host "Repo root: $RepoRoot"
Write-Host "This bounded script trains only tiny NumPy LoRA adapter weights on local LIBERO HDF5 action snippets."
Write-Host "It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

if ($MaxPairs -lt 1 -or $MaxPairs -gt 16) {
    Write-Host "Refusing: MaxPairs must be between 1 and 16."
    exit 11
}

if ($MaxActionSteps -lt 1 -or $MaxActionSteps -gt 32) {
    Write-Host "Refusing: MaxActionSteps must be between 1 and 32."
    exit 12
}

if ($MaxSteps -lt 1 -or $MaxSteps -gt 64) {
    Write-Host "Refusing: MaxSteps must be between 1 and 64 for bounded offline LoRA scale-up."
    exit 13
}

if ($MaxRuntimeSeconds -lt 1 -or $MaxRuntimeSeconds -gt 900) {
    Write-Host "Refusing: MaxRuntimeSeconds must be between 1 and 900. This is stricter than the 20-minute planning budget."
    exit 14
}

if ($MaxSamples -lt 1 -or $MaxSamples -gt 64) {
    Write-Host "Refusing: MaxSamples must be between 1 and 64 for bounded offline LoRA scale-up."
    exit 15
}

if ($Rank -lt 1 -or $Rank -gt 4) {
    Write-Host "Refusing: Rank must be between 1 and 4 for bounded offline LoRA scale-up."
    exit 16
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
    Write-Host ("Refusing to run bounded offline LoRA scale-up while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

if ($env:ALLOW_TINY_TRAINING -ne "1") {
    Write-Host "Refusing: ALLOW_TINY_TRAINING=1 is required for bounded LIBERO offline LoRA scale-up."
    exit 21
}

& $Python -m tca_map.datasets.libero_offline_lora_scaleup `
    --manifest $ManifestPath `
    --report-json $JsonReportPath `
    --report-md $MarkdownReportPath `
    --max-pairs $MaxPairs `
    --max-action-steps $MaxActionSteps `
    --max-steps $MaxSteps `
    --max-runtime-seconds $MaxRuntimeSeconds `
    --max-samples $MaxSamples `
    --rank $Rank

exit $LASTEXITCODE
