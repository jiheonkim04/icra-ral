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

function Test-LoadOnlySmokePassed {
    param([string]$ExpectedSmolVlaPath)

    $reportPath = Join-Path $RepoRoot "reports\smolvla_load_only_smoke_report.json"
    if ([string]::IsNullOrWhiteSpace($ExpectedSmolVlaPath) -or -not (Test-Path -LiteralPath $reportPath)) {
        return $false
    }
    try {
        $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $reportedPath = [string]$report.paths.smolvla_ckpt
        if ([string]::IsNullOrWhiteSpace($reportedPath)) {
            return $false
        }
        $expectedFullPath = [System.IO.Path]::GetFullPath($ExpectedSmolVlaPath)
        $reportedFullPath = [System.IO.Path]::GetFullPath($reportedPath)
        return [bool]($report.result.passed -and $expectedFullPath -eq $reportedFullPath)
    } catch {
        return $false
    }
}

function Test-SingleSampleInterfacePassed {
    param([string]$ExpectedSmolVlaPath)

    $reportPath = Join-Path $RepoRoot "reports\smolvla_single_sample_interface_report.json"
    if ([string]::IsNullOrWhiteSpace($ExpectedSmolVlaPath) -or -not (Test-Path -LiteralPath $reportPath)) {
        return $false
    }
    try {
        $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $reportedPath = [string]$report.paths.smolvla_ckpt
        if ([string]::IsNullOrWhiteSpace($reportedPath)) {
            return $false
        }
        $expectedFullPath = [System.IO.Path]::GetFullPath($ExpectedSmolVlaPath)
        $reportedFullPath = [System.IO.Path]::GetFullPath($reportedPath)
        return [bool]($report.result.passed -and $expectedFullPath -eq $reportedFullPath)
    } catch {
        return $false
    }
}

function Test-FeatureCacheEvalPassed {
    $reportPath = Join-Path $RepoRoot "reports\feature_cache_eval_report.json"
    if (-not (Test-Path -LiteralPath $reportPath)) {
        return $false
    }
    try {
        $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        return [bool]$report.cache_valid
    } catch {
        return $false
    }
}

function Test-TinyHeadOnlySmokePassed {
    $reportPath = Join-Path $RepoRoot "reports\tiny_head_only_smoke_report.json"
    if (-not (Test-Path -LiteralPath $reportPath)) {
        return $false
    }
    try {
        $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        return [bool]$report.tiny_head_only_smoke_passed
    } catch {
        return $false
    }
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

function Get-SmolVlaTokenizerDependencyName {
    param([string]$Root)

    $preprocessorPath = Join-Path $Root "policy_preprocessor.json"
    if (-not (Test-Path -LiteralPath $preprocessorPath)) {
        return $null
    }

    try {
        $preprocessor = Get-Content -LiteralPath $preprocessorPath -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($step in @($preprocessor.steps)) {
            if ($step.registry_name -eq "tokenizer_processor" -and $step.config.tokenizer_name) {
                return [string]$step.config.tokenizer_name
            }
        }
    } catch {
        return $null
    }
    return $null
}

function Get-DependencyRoots {
    param(
        [string]$DependencyName,
        [string[]]$BaseRoots
    )

    $roots = New-Object System.Collections.Generic.List[string]
    if ([string]::IsNullOrWhiteSpace($DependencyName)) {
        return @()
    }

    $parts = $DependencyName -split "/"
    if ($parts.Count -lt 2) {
        return @()
    }

    $org = $parts[0]
    $repo = $parts[1]
    foreach ($base in $BaseRoots) {
        if ([string]::IsNullOrWhiteSpace($base) -or -not (Test-Path -LiteralPath $base)) {
            continue
        }

        $plainRoot = Join-Path (Join-Path $base $org) $repo
        $roots.Add($plainRoot) | Out-Null

        $hubRoot = Join-Path $base ("models--{0}--{1}" -f $org, $repo)
        $roots.Add($hubRoot) | Out-Null
        $snapshots = Join-Path $hubRoot "snapshots"
        if (Test-Path -LiteralPath $snapshots) {
            Get-ChildItem -LiteralPath $snapshots -Directory -ErrorAction SilentlyContinue | ForEach-Object {
                $roots.Add($_.FullName) | Out-Null
            }
        }
    }

    return @($roots | Select-Object -Unique)
}

function Find-ExternalTokenizerDependency {
    param(
        [string]$DependencyName,
        [string[]]$BaseRoots
    )

    $candidateRoots = Get-DependencyRoots -DependencyName $DependencyName -BaseRoots $BaseRoots
    foreach ($root in $candidateRoots) {
        $tokenizerFiles = Test-AnyFile -Root $root -Names @(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "sentencepiece.bpe.model",
            "chat_template.json",
            "chat_template.jinja",
            "preprocessor_config.json",
            "processor_config.json",
            "config.json"
        ) -Patterns @()
        if ($tokenizerFiles.Count -gt 0) {
            return @{
                name = $DependencyName
                found = $true
                root = $root
                files = @($tokenizerFiles)
                candidate_roots = @($candidateRoots)
            }
        }
    }

    return @{
        name = $DependencyName
        found = $false
        root = $null
        files = @()
        candidate_roots = @($candidateRoots)
    }
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
$tokenizerDependencyName = if ($ckptExists) { Get-SmolVlaTokenizerDependencyName -Root $ckptPath } else { $null }
$externalTokenizerDependency = Find-ExternalTokenizerDependency -DependencyName $tokenizerDependencyName -BaseRoots @($hfHome.Value, $checkpointRoot.Value)
$tokenizerPresent = [bool]($tokenizerFiles.Count -gt 0 -or $externalTokenizerDependency.found)

$gpuName = $null
$gpuMemoryMb = $null
try {
    $nvidiaOutput = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    $nvidiaExitCode = $LASTEXITCODE
    if (($nvidiaExitCode -eq 0) -and $nvidiaOutput) {
        $nvidia = @($nvidiaOutput)[0]
        $parts = $nvidia.Split(",")
        if ($parts.Count -ge 2) {
            $gpuName = $parts[0].Trim()
            $gpuMemoryMb = [int]$parts[1].Trim()
        }
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
    $tokenizerPresent -and
    $weightFiles.Count -gt 0 -and
    $memoryFits -and
    $lightweightAdapterImportOk
)
$loadOnlySmokePassed = Test-LoadOnlySmokePassed -ExpectedSmolVlaPath $ckptPath
$singleSampleInterfacePassed = Test-SingleSampleInterfacePassed -ExpectedSmolVlaPath $ckptPath
$featureCacheEvalPassed = Test-FeatureCacheEvalPassed
$tinyHeadOnlySmokePassed = Test-TinyHeadOnlySmokePassed

if ($ready -and $tinyHeadOnlySmokePassed) {
    $recommended = "Tiny head-only smoke has passed. Treat it as interface validation only; stop before real dataset training, rollouts, simulator execution, OpenVLA-OFT, or paper claims."
} elseif ($ready -and $featureCacheEvalPassed) {
    $recommended = "Ready for a tiny head-only smoke runner with strict caps. Do not rollout or execute OpenVLA-OFT."
} elseif ($ready -and $singleSampleInterfacePassed) {
    $recommended = "Ready for tiny feature-cache/interface validation. Do not train or rollout."
} elseif ($ready -and $loadOnlySmokePassed) {
    $recommended = "Ready for the standing-approved single-sample SmolVLA interface smoke with synthetic or dummy inputs. Do not train or rollout."
} elseif ($ready) {
    $recommended = "Ready for the standing-approved bounded SmolVLA load-only adapter smoke. Do not train."
} elseif (-not $ckptExists) {
    $recommended = "Configure SMOLVLA_CKPT to a local checkpoint directory first."
} elseif ($configFiles.Count -eq 0 -or -not $tokenizerPresent -or $weightFiles.Count -eq 0) {
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
    smolvla_load_only_smoke_passed = $loadOnlySmokePassed
    smolvla_single_sample_interface_passed = $singleSampleInterfacePassed
    feature_cache_eval_smoke_passed = $featureCacheEvalPassed
    tiny_head_only_smoke_passed = $tinyHeadOnlySmokePassed
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
        external_tokenizer_dependency = [ordered]@{
            name = $externalTokenizerDependency.name
            found = $externalTokenizerDependency.found
            root = $externalTokenizerDependency.root
            files_found = @($externalTokenizerDependency.files)
            candidate_roots = @($externalTokenizerDependency.candidate_roots)
        }
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
