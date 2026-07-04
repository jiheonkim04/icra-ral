param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_report.json",
    [string]$JsonReportPath = "reports\libero_offline_lora_comparison_report.json",
    [string]$MarkdownReportPath = "reports\libero_offline_lora_comparison_report.md",
    [int]$MaxPairs = 4,
    [int]$MaxActionSteps = 16,
    [int]$MaxSteps = 16,
    [int]$MaxRuntimeSeconds = 900,
    [int]$MaxSamples = 16,
    [int]$Rank = 4
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO offline LoRA comparison"
Write-Host "Repo root: $RepoRoot"
Write-Host "This bounded script trains only tiny NumPy LoRA adapter weights on local LIBERO HDF5 action snippets. It does not download assets, run GPU jobs, import heavy VLA models, load models, infer, rollout, execute simulators, access tokens, execute OpenVLA-OFT, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

if ($MaxSteps -lt 1 -or $MaxSteps -gt 100) {
    Write-Host "Refusing: MaxSteps must be between 1 and 100 for tiny LoRA comparison."
    exit 11
}

if ($MaxRuntimeSeconds -lt 1 -or $MaxRuntimeSeconds -gt 900) {
    Write-Host "Refusing: MaxRuntimeSeconds must be between 1 and 900."
    exit 12
}

if ($MaxSamples -lt 1 -or $MaxSamples -gt 200) {
    Write-Host "Refusing: MaxSamples must be between 1 and 200 for tiny LoRA comparison."
    exit 13
}

if ($Rank -lt 1 -or $Rank -gt 16) {
    Write-Host "Refusing: Rank must be between 1 and 16 for tiny LoRA comparison."
    exit 14
}

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF"
)

$setDangerousGates = @()
foreach ($gate in $dangerousGates) {
    $value = [Environment]::GetEnvironmentVariable($gate)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $setDangerousGates += $gate
    }
}

if ($setDangerousGates.Count -gt 0) {
    Write-Host ("Refusing to run LIBERO offline LoRA comparison while dangerous gates are set: " + ($setDangerousGates -join ", "))
    exit 20
}

if ($env:ALLOW_TINY_TRAINING -ne "1") {
    Write-Host "Refusing: ALLOW_TINY_TRAINING=1 is required for bounded tiny LIBERO offline LoRA comparison."
    exit 21
}

& $Python -m tca_map.datasets.libero_offline_lora_comparison `
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
