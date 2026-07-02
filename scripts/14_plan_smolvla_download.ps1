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
$requiredConfig = @("config.json")
$requiredTokenizerAny = @("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.model", "sentencepiece.bpe.model")
$acceptedTokenizerAlso = @("special_tokens_map.json")
$requiredWeightsAny = @("model.safetensors", "pytorch_model.bin", "*.safetensors", "*.bin")

$report = [ordered]@{
    policy = [ordered]@{
        dry_run_only = $true
        allow_downloads_gate_set = $allowDownloads
        downloads_performed = $false
        directories_created = $false
        gpu_jobs_performed = $false
        training_performed = $false
        real_rollouts_performed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
    }
    intended_paths = [ordered]@{
        smolvla_ckpt = [ordered]@{
            env = "SMOLVLA_CKPT"
            value = $smolVlaCkpt.Value
            source = $smolVlaCkpt.Source
            exists = $(if ($smolVlaCkpt.Value) { Test-Path -LiteralPath $smolVlaCkpt.Value } else { $false })
        }
        checkpoint_root = [ordered]@{
            env = "CHECKPOINT_ROOT"
            value = $checkpointRoot.Value
            source = $checkpointRoot.Source
            exists = $(if ($checkpointRoot.Value) { Test-Path -LiteralPath $checkpointRoot.Value } else { $false })
        }
        hf_home = [ordered]@{
            env = "HF_HOME"
            value = $hfHome.Value
            source = $hfHome.Source
            exists = $(if ($hfHome.Value) { Test-Path -LiteralPath $hfHome.Value } else { $false })
        }
    }
    required_files = [ordered]@{
        config = $requiredConfig
        tokenizer_any = $requiredTokenizerAny
        tokenizer_also_accepted_by_readiness_checker = $acceptedTokenizerAlso
        weights_any = $requiredWeightsAny
    }
    readiness_semantics = [ordered]@{
        path_ready_is_not_adapter_smoke_ready = $true
        adapter_smoke_requires_config_tokenizer_weights = $true
        adapter_smoke_requires_hf_home_or_checkpoint_root = $true
        smolvla_smoke_is_interface_validation_only = $true
        paper_grade_requires_real_benchmark_data_and_rollouts_later = $true
    }
    recommended_next_step = "Manually place a SmolVLA-compatible checkpoint under SMOLVLA_CKPT, then run scripts\11_check_real_assets.ps1 and scripts\13_check_smolvla_adapter_smoke.ps1."
}

Write-Host "SmolVLA checkpoint acquisition plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "Dry run only: true"
Write-Host "ALLOW_DOWNLOADS set: $allowDownloads"
Write-Host "Downloads performed: false"
Write-Host "Heavy VLA imports performed: false"
Write-Host "GPU jobs performed: false"
Write-Host "Training performed: false"
Write-Host "Rollouts performed: false"
Write-Host "OpenVLA-OFT executed: false"
Write-Host ""
Write-Host "Intended paths:"
Write-Host "SMOLVLA_CKPT=$($smolVlaCkpt.Value) [$($smolVlaCkpt.Source)]"
Write-Host "CHECKPOINT_ROOT=$($checkpointRoot.Value) [$($checkpointRoot.Source)]"
Write-Host "HF_HOME=$($hfHome.Value) [$($hfHome.Source)]"
Write-Host ""
Write-Host "Required files:"
Write-Host "- config: $($requiredConfig -join ', ')"
Write-Host "- tokenizer_any: $($requiredTokenizerAny -join ', ')"
Write-Host "- weights_any: $($requiredWeightsAny -join ', ')"
Write-Host ""
if ($allowDownloads) {
    Write-Host "ALLOW_DOWNLOADS=1 is set, but this planner still performs no downloads."
} else {
    Write-Host "ALLOW_DOWNLOADS is not set. This is the expected planning-only state."
}
Write-Host ""
$report | ConvertTo-Json -Depth 8
exit 0
