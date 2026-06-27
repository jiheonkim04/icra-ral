param(
    [string]$PathsFile = "configs\paths.local.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

function Read-AssetConfig {
    param([string]$Path)

    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $values
    }

    $inAssets = $false
    foreach ($rawLine in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $line = $rawLine.TrimEnd()
        if ($line -match '^\s*#' -or $line.Trim().Length -eq 0) {
            continue
        }
        if ($line -match '^assets\s*:') {
            $inAssets = $true
            continue
        }
        if ($inAssets -and $line -match '^\S' -and $line -notmatch '^assets\s*:') {
            break
        }
        if ($inAssets -and $line -match '^\s+([A-Za-z0-9_]+)\s*:\s*(.*)$') {
            $key = $Matches[1]
            $value = $Matches[2].Trim()
            $value = $value.Trim('"').Trim("'")
            if ($value -and $value.ToLowerInvariant() -ne "null") {
                $values[$key] = $value
            }
        }
    }
    return $values
}

function Get-ConfiguredValue {
    param(
        [hashtable]$Config,
        [string]$Key,
        [string]$EnvName
    )

    $envValue = [Environment]::GetEnvironmentVariable($EnvName)
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        return @{ Value = $envValue; Source = "env:$EnvName" }
    }
    if ($Config.ContainsKey($Key)) {
        return @{ Value = $Config[$Key]; Source = $PathsFile }
    }
    return @{ Value = $null; Source = $null }
}

$assetSpecs = @(
    @{ Key = "openvla_oft_ckpt"; Env = "OPENVLA_OFT_CKPT"; Label = "OpenVLA-OFT checkpoint or local model directory" },
    @{ Key = "smolvla_ckpt"; Env = "SMOLVLA_CKPT"; Label = "SmolVLA checkpoint or local model directory" },
    @{ Key = "libero_root"; Env = "LIBERO_ROOT"; Label = "LIBERO source checkout" },
    @{ Key = "libero_data_root"; Env = "LIBERO_DATA_ROOT"; Label = "LIBERO data/demos root" },
    @{ Key = "robosuite_root"; Env = "ROBOSUITE_ROOT"; Label = "RoboSuite checkout/install root" },
    @{ Key = "data_root"; Env = "DATA_ROOT"; Label = "General data root" },
    @{ Key = "checkpoint_root"; Env = "CHECKPOINT_ROOT"; Label = "Checkpoint root" },
    @{ Key = "hf_home"; Env = "HF_HOME"; Label = "Hugging Face cache root" }
)

$config = Read-AssetConfig -Path $PathsFile
$status = @{}
$missing = New-Object System.Collections.Generic.List[string]

foreach ($spec in $assetSpecs) {
    $resolved = Get-ConfiguredValue -Config $config -Key $spec.Key -EnvName $spec.Env
    $value = $resolved.Value
    $exists = $false
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $exists = Test-Path -LiteralPath $value
    }
    $status[$spec.Key] = @{
        env = $spec.Env
        label = $spec.Label
        configured = -not [string]::IsNullOrWhiteSpace($value)
        exists = [bool]$exists
        source = $resolved.Source
        value_redacted = $(if ($value) { "set" } else { $null })
    }
    if (-not $exists) {
        $missing.Add($spec.Env) | Out-Null
    }
}

$readyForSmolVlaSmoke = [bool]($status["smolvla_ckpt"].exists -and ($status["hf_home"].exists -or $status["checkpoint_root"].exists))
$readyForOpenVlaOftSmoke = [bool]($status["openvla_oft_ckpt"].exists -and $status["hf_home"].exists -and $status["checkpoint_root"].exists)
$readyForLiberoRollout = [bool]($status["libero_root"].exists -and $status["libero_data_root"].exists -and $status["robosuite_root"].exists)

if ($readyForSmolVlaSmoke) {
    $recommendedNextStep = "Run a separate approved SmolVLA load-only adapter smoke task. Do not train."
} elseif ($readyForOpenVlaOftSmoke) {
    $recommendedNextStep = "OpenVLA-OFT assets are present, but SmolVLA-first is still recommended on RTX 5080 16GB."
} else {
    $recommendedNextStep = "Configure missing local paths, preferably SMOLVLA_CKPT plus HF_HOME or CHECKPOINT_ROOT first."
}

$report = [ordered]@{
    policy = [ordered]@{
        local_paths_only = $true
        downloads_performed = $false
        gpu_jobs_performed = $false
        heavy_model_imports_performed = $false
        real_rollouts_performed = $false
    }
    ready_for_smolvla_smoke = $readyForSmolVlaSmoke
    ready_for_openvla_oft_smoke = $readyForOpenVlaOftSmoke
    ready_for_libero_rollout = $readyForLiberoRollout
    missing_assets = @($missing)
    recommended_next_step = $recommendedNextStep
    assets = $status
}

Write-Host "TCA-Map real asset readiness check"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script does not download assets, import heavy VLA models, run GPU jobs, or run rollouts."
$report | ConvertTo-Json -Depth 8
exit 0
