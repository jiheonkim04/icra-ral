param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$Source = "",
    [switch]$OfficialSource,
    [double]$ExpectedSizeGb = -1,
    [string]$TargetPath = "",
    [switch]$TokenRequired,
    [switch]$LicenseClickThroughRequired,
    [switch]$PaymentRequired,
    [string]$JsonReportPath = "reports\libero_dataset_risk_report.json",
    [string]$MarkdownReportPath = "reports\libero_dataset_risk_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO dataset risk planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not download datasets, run GPU jobs, train, rollout, import simulators or heavy VLA models, access tokens, or execute OpenVLA-OFT."

$downloadBudgetGb = 80.0
$diskSafetyMarginGb = 100.0

if ([string]::IsNullOrWhiteSpace($Source)) {
    $Source = "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets"
    $OfficialSource = $true
    $ExpectedSizeGb = 100.0
}

function Read-AssetConfig {
    param([string]$Path)
    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) { return $values }
    $inAssets = $false
    foreach ($rawLine in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $line = $rawLine.TrimEnd()
        if ($line -match '^\s*#' -or $line.Trim().Length -eq 0) { continue }
        if ($line -match '^assets\s*:') {
            $inAssets = $true
            continue
        }
        if ($inAssets -and $line -match '^\S' -and $line -notmatch '^assets\s*:') { break }
        if ($inAssets -and $line -match '^\s+([A-Za-z0-9_]+)\s*:\s*(.*)$') {
            $key = $Matches[1]
            $value = $Matches[2].Trim().Trim('"').Trim("'")
            if ($value -and $value.ToLowerInvariant() -ne "null") { $values[$key] = $value }
        }
    }
    return $values
}

function Get-ConfiguredValue {
    param([hashtable]$Config, [string]$Key, [string]$EnvName)
    $envValue = [Environment]::GetEnvironmentVariable($EnvName)
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        return @{ Value = $envValue; Source = "env:$EnvName" }
    }
    if ($Config.ContainsKey($Key)) {
        return @{ Value = $Config[$Key]; Source = $PathsFile }
    }
    return @{ Value = $null; Source = $null }
}

function Get-FreeDiskGb {
    param([string]$Path)
    try {
        $root = [System.IO.Path]::GetPathRoot($Path)
        if ([string]::IsNullOrWhiteSpace($root)) { $root = [System.IO.Path]::GetPathRoot($RepoRoot) }
        $driveName = $root.Substring(0, 1)
        $drive = Get-PSDrive -Name $driveName -ErrorAction Stop
        return [Math]::Round(($drive.Free / 1GB), 3)
    } catch {
        return $null
    }
}

function Get-PathStatus {
    param([hashtable]$Config, [string]$Key, [string]$EnvName, [string]$Label)
    $configured = Get-ConfiguredValue -Config $Config -Key $Key -EnvName $EnvName
    $value = $configured.Value
    $exists = $false
    if (-not [string]::IsNullOrWhiteSpace($value)) { $exists = Test-Path -LiteralPath $value }
    return [ordered]@{
        label = $Label
        env = $EnvName
        configured = -not [string]::IsNullOrWhiteSpace($value)
        exists = [bool]$exists
        source = $configured.Source
        value_redacted = if ([string]::IsNullOrWhiteSpace($value)) { $null } else { "set" }
        path = $value
    }
}

function Get-DatasetFileSample {
    param([string]$Root)
    if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path -LiteralPath $Root)) { return @() }
    $extensions = @(".hdf5", ".h5", ".npz", ".pkl", ".json", ".jsonl")
    try {
        return @(
            Get-ChildItem -LiteralPath $Root -File -Recurse -Depth 3 -ErrorAction SilentlyContinue |
                Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() } |
                Select-Object -First 20 -ExpandProperty FullName
        )
    } catch {
        return @()
    }
}

$config = Read-AssetConfig -Path $PathsFile
$paths = [ordered]@{
    libero_root = Get-PathStatus -Config $config -Key "libero_root" -EnvName "LIBERO_ROOT" -Label "LIBERO source checkout"
    libero_data_root = Get-PathStatus -Config $config -Key "libero_data_root" -EnvName "LIBERO_DATA_ROOT" -Label "LIBERO data/demos root"
    robosuite_root = Get-PathStatus -Config $config -Key "robosuite_root" -EnvName "ROBOSUITE_ROOT" -Label "RoboSuite checkout/install root"
    data_root = Get-PathStatus -Config $config -Key "data_root" -EnvName "DATA_ROOT" -Label "General data root"
}

$targetCandidate = $TargetPath
if ([string]::IsNullOrWhiteSpace($targetCandidate)) {
    if ($paths.libero_data_root.configured) {
        $targetCandidate = $paths.libero_data_root.path
    } elseif ($paths.data_root.configured) {
        $targetCandidate = Join-Path $paths.data_root.path "libero"
    } else {
        $targetCandidate = "C:\assets\data\libero"
    }
}
$targetFullPath = if ([System.IO.Path]::IsPathRooted($targetCandidate)) { $targetCandidate } else { Join-Path $RepoRoot $targetCandidate }

$datasetFiles = Get-DatasetFileSample -Root $paths.libero_data_root.path
$dataFilesDetected = $datasetFiles.Count -gt 0
$readyForPathCheck = [bool]($paths.libero_root.exists -and $paths.libero_data_root.exists)
$readyForOfflineSubset = [bool]($readyForPathCheck -and $dataFilesDetected)
$readyForRolloutPathCheck = [bool]($readyForPathCheck -and $paths.robosuite_root.exists)

$diskFreeBeforeGb = Get-FreeDiskGb -Path $targetFullPath
$estimatedSizeForAfter = if ($ExpectedSizeGb -gt 0) { $ExpectedSizeGb } else { 0.0 }
$diskFreeAfterEstimateGb = if ($null -eq $diskFreeBeforeGb) { $null } else { [Math]::Round(($diskFreeBeforeGb - $estimatedSizeForAfter), 3) }

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if (-not $readyForOfflineSubset) {
    if (-not $paths.libero_root.exists) { $stopReasons.Add("LIBERO_ROOT is missing or does not exist") }
    if (-not $paths.libero_data_root.exists) { $stopReasons.Add("LIBERO_DATA_ROOT is missing or does not exist") }
    if ($paths.libero_data_root.exists -and -not $dataFilesDetected) { $stopReasons.Add("LIBERO_DATA_ROOT exists but no lightweight dataset marker files were found within depth 3") }
}
if (-not $readyForOfflineSubset) {
    if ([string]::IsNullOrWhiteSpace($Source)) { $stopReasons.Add("dataset source is missing; document an official source before acquisition") }
    if (-not $OfficialSource) { $stopReasons.Add("dataset source is not marked official/documented") }
    if ($ExpectedSizeGb -lt 0) { $stopReasons.Add("expected dataset size is unknown; do a listing or metadata-only plan first") }
    if ($ExpectedSizeGb -gt $downloadBudgetGb) { $stopReasons.Add("expected dataset size exceeds $downloadBudgetGb GB local task budget") }
    if ($null -eq $diskFreeBeforeGb) {
        $stopReasons.Add("disk free space could not be evaluated")
    } elseif ($diskFreeAfterEstimateGb -lt $diskSafetyMarginGb) {
        $stopReasons.Add("disk free after estimate would be below $diskSafetyMarginGb GB")
    }
    if ($TokenRequired) { $stopReasons.Add("token/secret/API key access is required") }
    if ($LicenseClickThroughRequired) { $stopReasons.Add("license click-through is required") }
    if ($PaymentRequired) { $stopReasons.Add("payment or paid service is required") }
}
$normalizedTargetFullPath = $targetFullPath.Replace("/", "\").ToLowerInvariant()
if (-not ($normalizedTargetFullPath.StartsWith("c:\assets"))) {
    $warnings.Add("target path is outside C:\assets; continue only if this is an approved local asset root")
}

$acquisitionRiskGreen = [bool](
    -not [string]::IsNullOrWhiteSpace($Source) -and
    $OfficialSource -and
    $ExpectedSizeGb -ge 0 -and
    $ExpectedSizeGb -le $downloadBudgetGb -and
    $null -ne $diskFreeAfterEstimateGb -and
    $diskFreeAfterEstimateGb -ge $diskSafetyMarginGb -and
    -not $TokenRequired -and
    -not $LicenseClickThroughRequired -and
    -not $PaymentRequired
)

$decision = if ($readyForOfflineSubset -or $acquisitionRiskGreen) { "proceed" } else { "stop" }
$recommendedNextStep = if ($readyForOfflineSubset) {
    "Create a metadata-only or tiny LIBERO/LIBERO-CF-style subset manifest; do not train, rollout, or make paper claims."
} elseif ($acquisitionRiskGreen) {
    "A future dataset acquisition task may proceed with task-local ALLOW_DOWNLOADS=1, but this planner did not download anything."
} elseif ($readyForPathCheck -and -not $dataFilesDetected -and $ExpectedSizeGb -gt $downloadBudgetGb) {
    "LIBERO/RoboSuite paths are ready, but no tiny dataset files are present and the official full dataset exceeds the local task budget. Document or place a tiny subset under LIBERO_DATA_ROOT before offline dataset smoke."
} else {
    "Provide an official dataset source, expected size, and valid local LIBERO paths, or place a tiny local subset under LIBERO_DATA_ROOT."
}

$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
        downloads_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        rollouts_performed = $false
        simulator_executed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    source = [ordered]@{
        value = $Source
        official_or_documented = [bool]$OfficialSource
        token_required = [bool]$TokenRequired
        license_click_through_required = [bool]$LicenseClickThroughRequired
        payment_required = [bool]$PaymentRequired
    }
    target_path = $targetFullPath
    expected_size_gb = $ExpectedSizeGb
    disk_free_before_gb = $diskFreeBeforeGb
    disk_free_after_estimate_gb = $diskFreeAfterEstimateGb
    budgets = [ordered]@{
        download_soft_limit_gb = $downloadBudgetGb
        disk_safety_margin_gb = $diskSafetyMarginGb
    }
    paths = $paths
    dataset_probe = [ordered]@{
        data_files_detected = [bool]$dataFilesDetected
        sample_files = @($datasetFiles)
        max_depth_checked = 3
    }
    ready_for_libero_path_check = $readyForPathCheck
    ready_for_libero_offline_subset = $readyForOfflineSubset
    ready_for_libero_dataset_acquisition = $acquisitionRiskGreen
    ready_for_libero_rollout_path_check = $readyForRolloutPathCheck
    ready_for_libero_rollout = $false
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    decision = $decision
    recommended_next_step = $recommendedNextStep
}

$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
$markdown = @(
    "# LIBERO Dataset Risk Report",
    "",
    "- decision: $decision",
    "- source: $Source",
    "- official/documented source: $([bool]$OfficialSource)",
    "- target path: $targetFullPath",
    "- expected size GB: $ExpectedSizeGb",
    "- disk free before GB: $diskFreeBeforeGb",
    "- disk free after estimate GB: $diskFreeAfterEstimateGb",
    "- ready for LIBERO path check: $readyForPathCheck",
    "- ready for offline subset: $readyForOfflineSubset",
    "- ready for dataset acquisition: $acquisitionRiskGreen",
    "- ready for rollout: false",
    "",
    "This report is planning-only. It performs no downloads, GPU jobs, training, rollouts, simulator execution, heavy VLA imports, token access, OpenVLA-OFT execution, or paper-grade claims."
) -join "`n"
$markdown | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8

$report | ConvertTo-Json -Depth 8
