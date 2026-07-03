param(
    [string]$BudgetFile = "configs\compute_budget.yaml",
    [string[]]$Config = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

function Read-ScalarValue {
    param(
        [string]$Path,
        [string]$Key,
        [object]$Default = $null
    )
    if (-not (Test-Path -LiteralPath $Path)) { return $Default }
    foreach ($rawLine in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $line = $rawLine.Trim()
        if ($line -match "^$([regex]::Escape($Key))\s*:\s*(.+)$") {
            $value = $Matches[1].Trim().Trim('"').Trim("'")
            if ($value -match '^(true|false)$') { return [bool]::Parse($value) }
            if ($value -match '^-?\d+(\.\d+)?$') { return [double]$value }
            return $value
        }
    }
    return $Default
}

function Get-ConfigFiles {
    if ($Config.Count -gt 0) { return $Config }
    return Get-ChildItem -LiteralPath "configs" -Filter "*.yaml" |
        Where-Object { $_.Name -notin @("paths.example.yaml", "paths.local.yaml.example", "compute_budget.yaml") } |
        ForEach-Object { $_.FullName }
}

function Test-TruthyLine {
    param([string[]]$Lines, [string]$Key)
    foreach ($line in $Lines) {
        if ($line -match "^\s*$([regex]::Escape($Key))\s*:\s*true\s*$") { return $true }
    }
    return $false
}

function Get-NumericLineValue {
    param([string[]]$Lines, [string]$Key)
    foreach ($line in $Lines) {
        if ($line -match "^\s*$([regex]::Escape($Key))\s*:\s*([0-9]+(\.[0-9]+)?)\s*$") { return [double]$Matches[1] }
    }
    return $null
}

$maxGrid = Read-ScalarValue -Path $BudgetFile -Key "max_heatmap_grid_initial" -Default 8
$maxTrainable = Read-ScalarValue -Path $BudgetFile -Key "max_trainable_params_millions_initial" -Default 50
$maxSteps = Read-ScalarValue -Path $BudgetFile -Key "max_local_pilot_steps_initial" -Default 300
$errors = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
$checked = New-Object System.Collections.Generic.List[string]

if (-not (Test-Path -LiteralPath $BudgetFile)) {
    $errors.Add("Missing compute budget file: $BudgetFile") | Out-Null
}

foreach ($path in Get-ConfigFiles) {
    if (-not (Test-Path -LiteralPath $path)) {
        $errors.Add("Missing config: $path") | Out-Null
        continue
    }
    $checked.Add($path) | Out-Null
    $lines = Get-Content -LiteralPath $path -Encoding UTF8
    $text = ($lines -join "`n").ToLowerInvariant()

    foreach ($forbiddenKey in @("openvla_oft_full_finetune", "openvla_oft_full_rollout", "openvla_oft_multiseed_sweep", "high_resolution_voxel_heatmap", "full_finetune", "full_rollout", "multiseed_sweep", "train_backbone")) {
        if (Test-TruthyLine -Lines $lines -Key $forbiddenKey) {
            $errors.Add("$path enables forbidden key ${forbiddenKey}: true.") | Out-Null
        }
    }

    $grid = Get-NumericLineValue -Lines $lines -Key "grid_size"
    if ($null -ne $grid -and $grid -gt $maxGrid) {
        $errors.Add("$path sets grid_size=$grid above max_heatmap_grid_initial=$maxGrid.") | Out-Null
    }

    $params = Get-NumericLineValue -Lines $lines -Key "trainable_params_millions_estimate"
    if ($null -ne $params -and $params -gt $maxTrainable) {
        $errors.Add("$path estimates trainable_params_millions=$params above limit=$maxTrainable.") | Out-Null
    }

    $steps = Get-NumericLineValue -Lines $lines -Key "max_steps"
    if ($null -ne $steps -and $steps -gt $maxSteps) {
        $errors.Add("$path sets max_steps=$steps above max_local_pilot_steps_initial=$maxSteps.") | Out-Null
    }

    $openvlaMentioned = $text.Contains("openvla")
    $openvlaExplicitlyDisabled = $text.Contains("openvla_oft_enabled: false") -or ($text.Contains("openvla_oft:") -and $text.Contains("enabled: false"))
    $openvlaActive = $openvlaMentioned -and -not $openvlaExplicitlyDisabled
    if ($openvlaActive -and -not ($text.Contains("frozen") -or $text.Contains("load") -or $text.Contains("smoke"))) {
        $warnings.Add("$path mentions active OpenVLA without an obvious frozen/load/smoke context.") | Out-Null
    }
    if ($openvlaActive -and ($text.Contains("train: true") -or $text.Contains("train_heads: true"))) {
        $warnings.Add("$path mentions active OpenVLA and training. Verify this is not OpenVLA training.") | Out-Null
    }
}

$report = [ordered]@{
    budget_file = $BudgetFile
    checked_configs = @($checked)
    limits = [ordered]@{
        max_heatmap_grid_initial = $maxGrid
        max_trainable_params_millions_initial = $maxTrainable
        max_local_pilot_steps_initial = $maxSteps
    }
    passed = $errors.Count -eq 0
    warnings = @($warnings)
    errors = @($errors)
    gpu_jobs_performed = $false
    downloads_performed = $false
    rollouts_performed = $false
}

$report | ConvertTo-Json -Depth 6
if ($errors.Count -gt 0) { exit 1 }
exit 0
