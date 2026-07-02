param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$AssetRoot = "C:\assets"
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
            $value = $Matches[2].Trim().Trim('"').Trim("'")
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
        [string]$EnvName,
        [string]$DefaultValue
    )

    $envValue = [Environment]::GetEnvironmentVariable($EnvName)
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        return @{ Value = $envValue; Source = "env:$EnvName" }
    }
    if ($Config.ContainsKey($Key)) {
        return @{ Value = $Config[$Key]; Source = $PathsFile }
    }
    return @{ Value = $DefaultValue; Source = "default" }
}

$config = Read-AssetConfig -Path $PathsFile
$checkpointRoot = Get-ConfiguredValue -Config $config -Key "checkpoint_root" -EnvName "CHECKPOINT_ROOT" -DefaultValue (Join-Path $AssetRoot "checkpoints")
$hfHome = Get-ConfiguredValue -Config $config -Key "hf_home" -EnvName "HF_HOME" -DefaultValue (Join-Path $AssetRoot "hf_home")
$smolVlaCkpt = Get-ConfiguredValue -Config $config -Key "smolvla_ckpt" -EnvName "SMOLVLA_CKPT" -DefaultValue (Join-Path $checkpointRoot.Value "smolvla")

$allowDownloads = $env:ALLOW_DOWNLOADS -eq "1"
$allowCreateDirs = $env:ALLOW_CREATE_DIRS -eq "1"
$dirs = @($checkpointRoot.Value, $hfHome.Value, $smolVlaCkpt.Value)

Write-Host "SmolVLA asset preparation"
Write-Host "Repo root: $RepoRoot"
Write-Host "Dry run: $(-not $allowCreateDirs)"
Write-Host "Downloads allowed gate set: $allowDownloads"
Write-Host "Downloads performed: false"
Write-Host "Tokens are never read from or written to committed files."
Write-Host ""

foreach ($dir in $dirs) {
    if ($allowCreateDirs) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "created_or_exists: $dir"
    } else {
        Write-Host "would_create: $dir"
    }
}

Write-Host ""
Write-Host "Resolved local paths:"
Write-Host "SMOLVLA_CKPT=$($smolVlaCkpt.Value) [$($smolVlaCkpt.Source)]"
Write-Host "CHECKPOINT_ROOT=$($checkpointRoot.Value) [$($checkpointRoot.Source)]"
Write-Host "HF_HOME=$($hfHome.Value) [$($hfHome.Source)]"

if ($allowDownloads) {
    Write-Host ""
    Write-Host "ALLOW_DOWNLOADS=1 is set, but this scaffold does not download automatically."
    Write-Host "Use your authenticated Hugging Face workflow outside this script, then rerun scripts\13_check_smolvla_adapter_smoke.ps1."
} else {
    Write-Host ""
    Write-Host "To allow a future explicit download-capable script, set ALLOW_DOWNLOADS=1. This script still performs no downloads."
}

Write-Host ""
Write-Host "Suggested configs/paths.local.yaml block:"
Write-Host "assets:"
Write-Host "  smolvla_ckpt: `"$($smolVlaCkpt.Value.Replace('\', '/'))`""
Write-Host "  checkpoint_root: `"$($checkpointRoot.Value.Replace('\', '/'))`""
Write-Host "  hf_home: `"$($hfHome.Value.Replace('\', '/'))`""
Write-Host ""
Write-Host "Next safe check:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\13_check_smolvla_adapter_smoke.ps1"
exit 0
