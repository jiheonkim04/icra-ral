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

$smolVlaPath = (Get-ConfiguredValue -Config $config -Key "smolvla_ckpt" -EnvName "SMOLVLA_CKPT").Value
$checkpointRootPath = (Get-ConfiguredValue -Config $config -Key "checkpoint_root" -EnvName "CHECKPOINT_ROOT").Value
$hfHomePath = (Get-ConfiguredValue -Config $config -Key "hf_home" -EnvName "HF_HOME").Value
$smolVlaPathConfigured = -not [string]::IsNullOrWhiteSpace($smolVlaPath)
$smolVlaPathExists = [bool]($smolVlaPathConfigured -and (Test-Path -LiteralPath $smolVlaPath))
$smolVlaConfigFiles = if ($smolVlaPathExists) { Test-AnyFile -Root $smolVlaPath -Names @("config.json") -Patterns @() } else { @() }
$smolVlaTokenizerFiles = if ($smolVlaPathExists) {
    Test-AnyFile -Root $smolVlaPath -Names @("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model", "sentencepiece.bpe.model") -Patterns @()
} else { @() }
$smolVlaWeightFiles = if ($smolVlaPathExists) {
    Test-AnyFile -Root $smolVlaPath -Names @("model.safetensors", "pytorch_model.bin", "model-00001-of-00001.safetensors", "pytorch_model-00001-of-00001.bin") -Patterns @("*.safetensors", "*.bin")
} else { @() }
$smolVlaTokenizerDependencyName = if ($smolVlaPathExists) { Get-SmolVlaTokenizerDependencyName -Root $smolVlaPath } else { $null }
$smolVlaExternalTokenizerDependency = Find-ExternalTokenizerDependency -DependencyName $smolVlaTokenizerDependencyName -BaseRoots @($hfHomePath, $checkpointRootPath)
$smolVlaTokenizerPresent = [bool]($smolVlaTokenizerFiles.Count -gt 0 -or $smolVlaExternalTokenizerDependency.found)
$smolVlaCheckpointFilesPresent = [bool]($smolVlaConfigFiles.Count -gt 0 -and $smolVlaTokenizerPresent -and $smolVlaWeightFiles.Count -gt 0)
$readyForSmolVlaPathCheck = [bool]($smolVlaPathConfigured -and $smolVlaPathExists)
$readyForSmolVlaAdapterSmoke = [bool](
    $smolVlaPathConfigured -and
    $smolVlaPathExists -and
    $smolVlaCheckpointFilesPresent -and
    ($status["hf_home"].exists -or $status["checkpoint_root"].exists)
)
$readyForSmolVlaSmoke = $readyForSmolVlaAdapterSmoke
$readyForOpenVlaOftSmoke = [bool]($status["openvla_oft_ckpt"].exists -and $status["hf_home"].exists -and $status["checkpoint_root"].exists)
$readyForLiberoRollout = [bool]($status["libero_root"].exists -and $status["libero_data_root"].exists -and $status["robosuite_root"].exists)
$loadOnlySmokePassed = Test-LoadOnlySmokePassed -ExpectedSmolVlaPath $smolVlaPath
$singleSampleInterfacePassed = Test-SingleSampleInterfacePassed -ExpectedSmolVlaPath $smolVlaPath

if ($readyForSmolVlaAdapterSmoke -and $singleSampleInterfacePassed) {
    $recommendedNextStep = "Continue to tiny feature-cache/interface validation. Do not train or rollout."
} elseif ($readyForSmolVlaAdapterSmoke -and $loadOnlySmokePassed) {
    $recommendedNextStep = "Continue to the standing-approved single-sample SmolVLA interface smoke with synthetic or dummy inputs. Do not train or rollout."
} elseif ($readyForSmolVlaAdapterSmoke) {
    $recommendedNextStep = "Continue to the standing-approved bounded SmolVLA load-only adapter smoke. Do not train."
} elseif ($readyForSmolVlaPathCheck -and -not $smolVlaCheckpointFilesPresent) {
    $recommendedNextStep = "SmolVLA path exists, but config/tokenizer/weights files are missing. This is path-ready only, not adapter-smoke-ready."
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
    smolvla_path_configured = $smolVlaPathConfigured
    smolvla_path_exists = $smolVlaPathExists
    smolvla_checkpoint_files_present = $smolVlaCheckpointFilesPresent
    ready_for_smolvla_path_check = $readyForSmolVlaPathCheck
    ready_for_smolvla_adapter_smoke = $readyForSmolVlaAdapterSmoke
    smolvla_load_only_smoke_passed = $loadOnlySmokePassed
    smolvla_single_sample_interface_passed = $singleSampleInterfacePassed
    smolvla_expected_files = [ordered]@{
        config_found = @($smolVlaConfigFiles)
        tokenizer_found = @($smolVlaTokenizerFiles)
        weights_found = @($smolVlaWeightFiles)
        external_tokenizer_dependency = [ordered]@{
            name = $smolVlaExternalTokenizerDependency.name
            found = $smolVlaExternalTokenizerDependency.found
            root = $smolVlaExternalTokenizerDependency.root
            files_found = @($smolVlaExternalTokenizerDependency.files)
            candidate_roots = @($smolVlaExternalTokenizerDependency.candidate_roots)
        }
    }
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
