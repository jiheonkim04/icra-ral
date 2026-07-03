param(
    [string]$Task = "unspecified",
    [string]$Category = "generic",
    [string]$Source = "",
    [string]$TargetPath = "C:\assets",
    [double]$ExpectedSizeGb = -1,
    [double]$ExpectedRuntimeMinutes = 0,
    [double]$ExpectedVramGb = 0,
    [double]$ExpectedRamGb = 0,
    [int]$MaxSteps = 0,
    [int]$BatchSize = 1,
    [int]$TaskCount = 0,
    [switch]$OfficialSource,
    [switch]$TokenRequired,
    [switch]$LicenseClickThroughRequired,
    [switch]$PaymentRequired,
    [switch]$ExternalUploadRequired,
    [switch]$SystemChangeRequired,
    [switch]$AdminInstallerRequired,
    [switch]$OpenVlaRequired,
    [switch]$PaperClaim,
    [switch]$FullFinetuneRequired,
    [switch]$SimulatorInstalled,
    [string]$JsonReportPath = "reports\risk_assessment_report.json",
    [string]$MarkdownReportPath = "reports\risk_assessment_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Risk assessment"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script assesses risk only. It does not download, install, run GPU jobs, train, rollout, import heavy VLA models, access tokens, or execute OpenVLA-OFT."

$downloadBudgetGb = 80.0
$diskSafetyMarginGb = 100.0
$maxVramGb = 14.0
$maxRuntimeMinutes = 30.0
$maxTrainingSteps = 300
$maxBatchSize = 1
$maxRolloutTasks = 5

$targetFullPath = if ([System.IO.Path]::IsPathRooted($TargetPath)) {
    $TargetPath
} else {
    Join-Path $RepoRoot $TargetPath
}

$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) {
    $JsonReportPath
} else {
    Join-Path $RepoRoot $JsonReportPath
}
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) {
    $MarkdownReportPath
} else {
    Join-Path $RepoRoot $MarkdownReportPath
}

function Get-FreeDiskGb {
    param([string]$Path)
    try {
        $root = [System.IO.Path]::GetPathRoot($Path)
        if ([string]::IsNullOrWhiteSpace($root)) {
            $root = [System.IO.Path]::GetPathRoot($RepoRoot)
        }
        $driveName = $root.Substring(0, 1)
        $drive = Get-PSDrive -Name $driveName -ErrorAction Stop
        return [Math]::Round(($drive.Free / 1GB), 3)
    } catch {
        return $null
    }
}

$diskFreeBeforeGb = Get-FreeDiskGb -Path $targetFullPath
$estimatedSizeForAfter = if ($ExpectedSizeGb -gt 0) { $ExpectedSizeGb } else { 0.0 }
$diskFreeAfterEstimateGb = if ($null -eq $diskFreeBeforeGb) { $null } else { [Math]::Round(($diskFreeBeforeGb - $estimatedSizeForAfter), 3) }

$normalizedCategory = $Category.Trim().ToLowerInvariant()
$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if ($TokenRequired) { $stopReasons.Add("token/secret/API key access is required") }
if ($LicenseClickThroughRequired) { $stopReasons.Add("license click-through is required") }
if ($PaymentRequired) { $stopReasons.Add("payment or paid service is required") }
if ($ExternalUploadRequired) { $stopReasons.Add("external upload/submission/publishing is required") }
if ($SystemChangeRequired) { $stopReasons.Add("system-wide CUDA/PyTorch/driver change is required") }
if ($AdminInstallerRequired) { $stopReasons.Add("admin/system-level installer is required") }
if ($OpenVlaRequired) { $stopReasons.Add("OpenVLA-OFT execution/download/import/load is outside the current automatic risk budget") }
if ($PaperClaim) { $stopReasons.Add("paper-level empirical claim is forbidden automatically") }

if ($ExpectedRuntimeMinutes -gt $maxRuntimeMinutes) {
    $stopReasons.Add("expected runtime exceeds $maxRuntimeMinutes minutes")
}
if ($ExpectedVramGb -gt $maxVramGb) {
    $stopReasons.Add("expected VRAM exceeds $maxVramGb GB")
}
if ($BatchSize -gt $maxBatchSize) {
    $stopReasons.Add("batch size exceeds $maxBatchSize")
}

if ($normalizedCategory -in @("download", "dataset")) {
    if ([string]::IsNullOrWhiteSpace($Source)) {
        $stopReasons.Add("source is missing")
    }
    if (-not $OfficialSource) {
        $stopReasons.Add("source is not marked official/documented")
    }
    if ($ExpectedSizeGb -lt 0) {
        $stopReasons.Add("expected size is unknown; do dry-run/listing first")
    }
    if ($ExpectedSizeGb -gt $downloadBudgetGb) {
        $stopReasons.Add("expected download size exceeds $downloadBudgetGb GB")
    }
    if ($null -eq $diskFreeBeforeGb) {
        $stopReasons.Add("disk free space could not be evaluated")
    } elseif ($diskFreeAfterEstimateGb -lt $diskSafetyMarginGb) {
        $stopReasons.Add("disk free after estimate would be below $diskSafetyMarginGb GB")
    }
    if (-not ($targetFullPath.ToLowerInvariant().StartsWith("c:\assets"))) {
        $warnings.Add("target path is outside C:\assets; ensure it is an approved asset/cache root")
    }
}

if ($normalizedCategory -eq "training") {
    if ($FullFinetuneRequired) {
        $stopReasons.Add("full fine-tuning is required")
    }
    if ($MaxSteps -gt $maxTrainingSteps) {
        $stopReasons.Add("max steps exceeds $maxTrainingSteps")
    }
}

if ($normalizedCategory -eq "simulator") {
    if (-not $SimulatorInstalled) {
        $stopReasons.Add("simulator is not marked installed locally")
    }
    if ($ExpectedRuntimeMinutes -gt 10) {
        $stopReasons.Add("simulator readiness smoke exceeds 10 minutes")
    }
}

if ($normalizedCategory -eq "rollout") {
    if (-not $SimulatorInstalled) {
        $stopReasons.Add("simulator is not marked installed locally")
    }
    if ($TaskCount -gt $maxRolloutTasks) {
        $stopReasons.Add("rollout task count exceeds $maxRolloutTasks")
    }
}

$decision = if ($stopReasons.Count -eq 0) { "proceed" } else { "stop" }
$reason = if ($stopReasons.Count -eq 0) {
    "Risk assessment is inside the documented local budget."
} else {
    ($stopReasons -join "; ")
}

$report = [ordered]@{
    policy = [ordered]@{
        risk_assessment_only = $true
        downloads_performed = $false
        installs_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        rollouts_performed = $false
        simulator_executed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    task = $Task
    category = $normalizedCategory
    source = $Source
    source_official_or_documented = [bool]$OfficialSource
    target_path = $targetFullPath
    expected_size_gb = $ExpectedSizeGb
    disk_free_before_gb = $diskFreeBeforeGb
    disk_free_after_estimate_gb = $diskFreeAfterEstimateGb
    expected_runtime_minutes = $ExpectedRuntimeMinutes
    expected_ram_gb = $ExpectedRamGb
    expected_vram_gb = $ExpectedVramGb
    max_steps = $MaxSteps
    batch_size = $BatchSize
    task_count = $TaskCount
    budget = [ordered]@{
        download_soft_limit_gb = $downloadBudgetGb
        disk_safety_margin_gb = $diskSafetyMarginGb
        max_vram_gb = $maxVramGb
        max_runtime_minutes = $maxRuntimeMinutes
        max_training_steps = $maxTrainingSteps
        max_batch_size = $maxBatchSize
        max_rollout_tasks = $maxRolloutTasks
    }
    gates = [ordered]@{
        token_required = [bool]$TokenRequired
        license_click_through_required = [bool]$LicenseClickThroughRequired
        payment_required = [bool]$PaymentRequired
        external_upload_required = [bool]$ExternalUploadRequired
        system_change_required = [bool]$SystemChangeRequired
        admin_installer_required = [bool]$AdminInstallerRequired
        openvla_required = [bool]$OpenVlaRequired
        paper_claim = [bool]$PaperClaim
        full_finetune_required = [bool]$FullFinetuneRequired
        simulator_installed = [bool]$SimulatorInstalled
    }
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    decision = $decision
    reason = $reason
}

$jsonDir = Split-Path -Parent $jsonFullPath
$mdDir = Split-Path -Parent $markdownFullPath
New-Item -ItemType Directory -Force -Path $jsonDir | Out-Null
New-Item -ItemType Directory -Force -Path $mdDir | Out-Null

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8

$markdown = @(
    "# Risk Assessment Report",
    "",
    "- task: $Task",
    "- category: $normalizedCategory",
    "- source: $Source",
    "- target path: $targetFullPath",
    "- expected size GB: $ExpectedSizeGb",
    "- disk free before GB: $diskFreeBeforeGb",
    "- disk free after estimate GB: $diskFreeAfterEstimateGb",
    "- expected runtime minutes: $ExpectedRuntimeMinutes",
    "- expected RAM GB: $ExpectedRamGb",
    "- expected VRAM GB: $ExpectedVramGb",
    "- decision: $decision",
    "- reason: $reason",
    "",
    "This is assessment-only. It performs no download, install, GPU job, training, rollout, simulator execution, heavy import, OpenVLA-OFT execution, token access, or paper claim."
)
$markdown -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8

$report | ConvertTo-Json -Depth 8
