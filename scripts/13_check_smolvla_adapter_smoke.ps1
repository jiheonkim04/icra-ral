param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe"
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

function Test-AnyFile {
    param(
        [string]$Root,
        [string[]]$Names,
        [string[]]$Patterns
    )

    $matches = New-Object System.Collections.Generic.List[string]
    if (-not (Test-Path -LiteralPath $Root)) {
        return @()
    }
    foreach ($name in $Names) {
        $candidate = Join-Path $Root $name
        if (Test-Path -LiteralPath $candidate) {
            $matches.Add($name) | Out-Null
        }
    }
    foreach ($pattern in $Patterns) {
        Get-ChildItem -LiteralPath $Root -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
            $matches.Add($_.Name) | Out-Null
        }
    }
    return @($matches | Select-Object -Unique)
}

$config = Read-AssetConfig -Path $PathsFile
$smolVla = Get-ConfiguredValue -Config $config -Key "smolvla_ckpt" -EnvName "SMOLVLA_CKPT"
$checkpointRoot = Get-ConfiguredValue -Config $config -Key "checkpoint_root" -EnvName "CHECKPOINT_ROOT"
$hfHome = Get-ConfiguredValue -Config $config -Key "hf_home" -EnvName "HF_HOME"

$allowHeavyImport = $env:ALLOW_HEAVY_IMPORT -eq "1"
$ckptPath = $smolVla.Value
$ckptExists = -not [string]::IsNullOrWhiteSpace($ckptPath) -and (Test-Path -LiteralPath $ckptPath)

$configFiles = if ($ckptExists) { Test-AnyFile -Root $ckptPath -Names @("config.json") -Patterns @() } else { @() }
$tokenizerFiles = if ($ckptExists) {
    Test-AnyFile -Root $ckptPath -Names @("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model", "sentencepiece.bpe.model") -Patterns @()
} else { @() }
$weightFiles = if ($ckptExists) {
    Test-AnyFile -Root $ckptPath -Names @("model.safetensors", "pytorch_model.bin", "model-00001-of-00001.safetensors", "pytorch_model-00001-of-00001.bin") -Patterns @("*.safetensors", "*.bin")
} else { @() }

$gpuName = $null
$gpuMemoryMb = $null
try {
    $nvidia = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($LASTEXITCODE -eq 0 -and $nvidia) {
        $parts = $nvidia.Split(",")
        $gpuName = $parts[0].Trim()
        $gpuMemoryMb = [int]$parts[1].Trim()
    }
} catch {
    $gpuName = $null
    $gpuMemoryMb = $null
}

$estimatedLoadMb = 12000
$requiredHeadroomMb = 2048
$memoryFits = $false
if ($gpuMemoryMb) {
    $memoryFits = ($gpuMemoryMb -ge ($estimatedLoadMb + $requiredHeadroomMb))
}

$lightweightAdapterImportOk = $false
$lightweightAdapterImportError = $null
if (Test-Path -LiteralPath $Python) {
    $importOutput = & $Python -c "from tca_map.adapters.lora_policy import validate_lora_policy_config; print('lightweight_adapter_guard_import_ok')" 2>&1
    $lightweightAdapterImportOk = ($LASTEXITCODE -eq 0)
    if (-not $lightweightAdapterImportOk) {
        $lightweightAdapterImportError = ($importOutput -join "`n")
    }
} else {
    $lightweightAdapterImportError = "Python interpreter not found: $Python"
}

$ready = [bool](
    $ckptExists -and
    $configFiles.Count -gt 0 -and
    $tokenizerFiles.Count -gt 0 -and
    $weightFiles.Count -gt 0 -and
    $memoryFits -and
    $lightweightAdapterImportOk
)

if ($ready) {
    $recommended = "Ready for a separately approved SmolVLA load-only adapter smoke. Do not train."
} elseif (-not $ckptExists) {
    $recommended = "Configure SMOLVLA_CKPT to a local checkpoint directory first."
} elseif ($configFiles.Count -eq 0 -or $tokenizerFiles.Count -eq 0 -or $weightFiles.Count -eq 0) {
    $recommended = "Checkpoint path exists but expected config/tokenizer/weights files are incomplete."
} elseif (-not $memoryFits) {
    $recommended = "Memory estimate does not leave enough RTX 5080 16GB headroom for local smoke."
} else {
    $recommended = "Fix lightweight adapter guard import before any real adapter smoke."
}

$report = [ordered]@{
    policy = [ordered]@{
        downloads_performed = $false
        gpu_training_performed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        real_rollouts_performed = $false
        libero_required = $false
        training_performed = $false
        heavy_import_gate_set = $allowHeavyImport
    }
    ready_for_smolvla_adapter_smoke = $ready
    smolvla_ckpt = [ordered]@{
        configured = -not [string]::IsNullOrWhiteSpace($ckptPath)
        exists = $ckptExists
        source = $smolVla.Source
        value_redacted = $(if ($ckptPath) { "set" } else { $null })
    }
    cache_roots = [ordered]@{
        checkpoint_root_configured = -not [string]::IsNullOrWhiteSpace($checkpointRoot.Value)
        checkpoint_root_exists = $(if ($checkpointRoot.Value) { Test-Path -LiteralPath $checkpointRoot.Value } else { $false })
        hf_home_configured = -not [string]::IsNullOrWhiteSpace($hfHome.Value)
        hf_home_exists = $(if ($hfHome.Value) { Test-Path -LiteralPath $hfHome.Value } else { $false })
    }
    expected_files = [ordered]@{
        config_found = @($configFiles)
        tokenizer_found = @($tokenizerFiles)
        weights_found = @($weightFiles)
    }
    adapter_check = [ordered]@{
        lightweight_adapter_guard_import_ok = $lightweightAdapterImportOk
        lightweight_adapter_guard_import_error = $lightweightAdapterImportError
        actual_smolvla_heavy_import_attempted = $false
    }
    memory_estimate = [ordered]@{
        gpu_name = $gpuName
        gpu_memory_total_mb = $gpuMemoryMb
        estimated_load_mb = $estimatedLoadMb
        required_headroom_mb = $requiredHeadroomMb
        fits_rtx_5080_16gb_budget = $memoryFits
    }
    recommended_next_step = $recommended
}

Write-Host "SmolVLA adapter smoke readiness check"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script does not download assets, train, run rollouts, require LIBERO, import SmolVLA, or execute OpenVLA-OFT."
if ($allowHeavyImport) {
    Write-Host "ALLOW_HEAVY_IMPORT=1 is set, but this scaffold still skips actual SmolVLA heavy import."
}
$report | ConvertTo-Json -Depth 8
exit 0
