param(
    [string]$JsonReportPath = "reports\libero_robosuite_source_resolution_report.json",
    [string]$MarkdownReportPath = "reports\libero_robosuite_source_resolution_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$downloadBudgetGb = 180.0
$diskSafetyMarginGb = 250.0

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

function New-Source {
    param(
        [string]$Name,
        [string]$Task,
        [string]$Url,
        [string]$DocumentationUrl,
        [string]$TargetPath,
        [double]$ExpectedSizeGb,
        [string]$SizeMethod,
        [string]$License,
        [bool]$Official,
        [bool]$TokenRequired,
        [bool]$LicenseClickThroughRequired,
        [bool]$PaymentRequired,
        [bool]$SimulatorWillRun,
        [bool]$RolloutWillRun
    )

    $diskFreeBefore = Get-FreeDiskGb -Path $TargetPath
    $diskFreeAfter = if ($null -eq $diskFreeBefore) { $null } else { [Math]::Round(($diskFreeBefore - $ExpectedSizeGb), 3) }
    $reasons = New-Object System.Collections.Generic.List[string]
    if (-not $Official) { $reasons.Add("source is not official/documented") | Out-Null }
    if ($ExpectedSizeGb -lt 0) { $reasons.Add("expected size is unknown") | Out-Null }
    if ($ExpectedSizeGb -gt $downloadBudgetGb) { $reasons.Add("expected size exceeds $downloadBudgetGb GB task budget") | Out-Null }
    if ($null -eq $diskFreeAfter) {
        $reasons.Add("disk free space could not be evaluated") | Out-Null
    } elseif ($diskFreeAfter -lt $diskSafetyMarginGb) {
        $reasons.Add("disk free after estimate would be below $diskSafetyMarginGb GB") | Out-Null
    }
    if ($TokenRequired) { $reasons.Add("login/token is required") | Out-Null }
    if ($LicenseClickThroughRequired) { $reasons.Add("license click-through is required") | Out-Null }
    if ($PaymentRequired) { $reasons.Add("payment is required") | Out-Null }
    if ($SimulatorWillRun) { $reasons.Add("simulator execution is outside source-resolution scope") | Out-Null }
    if ($RolloutWillRun) { $reasons.Add("rollout is outside source-resolution scope") | Out-Null }

    $decision = if ($reasons.Count -eq 0) { "proceed" } else { "stop" }
    $reasonText = if ($decision -eq "proceed") { "official/documented source, known size, no token/license/payment gate, and disk budget is green" } else { ($reasons -join "; ") }

    return [ordered]@{
        name = $Name
        task = $Task
        source_url = $Url
        documentation_url = $DocumentationUrl
        official_or_documented = $Official
        expected_size_gb = $ExpectedSizeGb
        size_estimation_method = $SizeMethod
        target_path = $TargetPath
        disk_free_before_gb = $diskFreeBefore
        disk_free_after_estimate_gb = $diskFreeAfter
        login_token_required = $TokenRequired
        license_click_through_required = $LicenseClickThroughRequired
        payment_required = $PaymentRequired
        expected_runtime = if ($Name -eq "libero_full_dataset") { "dataset acquisition may take many minutes depending on network throughput" } else { "minutes for shallow repo clone" }
        simulator_will_run = $SimulatorWillRun
        rollout_will_run = $RolloutWillRun
        license = $License
        decision = $decision
        reason = $reasonText
    }
}

$sources = [ordered]@{
    libero_repo = New-Source `
        -Name "libero_repo" `
        -Task "Acquire official LIBERO source checkout only" `
        -Url "https://github.com/Lifelong-Robot-Learning/LIBERO.git" `
        -DocumentationUrl "https://github.com/Lifelong-Robot-Learning/LIBERO" `
        -TargetPath "C:\assets\repos\LIBERO" `
        -ExpectedSizeGb 0.35 `
        -SizeMethod "GitHub repository API size was about 323128 KB on 2026-07-04; shallow clone should be smaller." `
        -License "MIT" `
        -Official $true `
        -TokenRequired $false `
        -LicenseClickThroughRequired $false `
        -PaymentRequired $false `
        -SimulatorWillRun $false `
        -RolloutWillRun $false
    robosuite_repo = New-Source `
        -Name "robosuite_repo" `
        -Task "Acquire official RoboSuite source checkout only" `
        -Url "https://github.com/ARISE-Initiative/robosuite.git" `
        -DocumentationUrl "https://github.com/ARISE-Initiative/robosuite" `
        -TargetPath "C:\assets\repos\robosuite" `
        -ExpectedSizeGb 0.70 `
        -SizeMethod "GitHub repository API size was about 644964 KB on 2026-07-04; shallow clone should be smaller." `
        -License "MIT with bundled DeepMind MuJoCo Apache-2.0 notice" `
        -Official $true `
        -TokenRequired $false `
        -LicenseClickThroughRequired $false `
        -PaymentRequired $false `
        -SimulatorWillRun $false `
        -RolloutWillRun $false
    libero_full_dataset = New-Source `
        -Name "libero_full_dataset" `
        -Task "Acquire full official LIBERO demonstration dataset" `
        -Url "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets" `
        -DocumentationUrl "https://github.com/Lifelong-Robot-Learning/LIBERO" `
        -TargetPath "C:\assets\data\libero" `
        -ExpectedSizeGb 100.0 `
        -SizeMethod "Hugging Face dataset card reports total file size as 100 GB." `
        -License "apache-2.0" `
        -Official $true `
        -TokenRequired $false `
        -LicenseClickThroughRequired $false `
        -PaymentRequired $false `
        -SimulatorWillRun $false `
        -RolloutWillRun $false
}

$repoDecisions = @($sources.libero_repo.decision, $sources.robosuite_repo.decision)
$repoSetupDecision = if ($repoDecisions -notcontains "stop") { "proceed" } else { "stop" }
$fullDatasetDecision = $sources.libero_full_dataset.decision

$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
        downloads_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        simulator_executed = $false
        rollouts_performed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    budgets = [ordered]@{
        download_soft_limit_gb = $downloadBudgetGb
        disk_safety_margin_gb = $diskSafetyMarginGb
    }
    sources = $sources
    ready_for_repo_setup = [bool]($repoSetupDecision -eq "proceed")
    ready_for_full_dataset_download = [bool]($fullDatasetDecision -eq "proceed")
    ready_for_tiny_or_metadata_only_dataset_setup = $true
    decision = $repoSetupDecision
    reason = if ($repoSetupDecision -eq "proceed") {
        "LIBERO and RoboSuite source checkout acquisition is inside budget; official LIBERO full dataset acquisition is inside the LIBERO-only 180GB budget if disk remains above 250GB, and must use scripts\49_acquire_libero_data.ps1 with task-local ALLOW_DOWNLOADS=1."
    } else {
        "One or more required source checkouts failed risk assessment."
    }
    recommended_next_step = if ($repoSetupDecision -eq "proceed") {
        "Run scripts\46_prepare_libero_robosuite_sources.ps1 for source checkout if needed, then use scripts\49_acquire_libero_data.ps1 for the official LIBERO dataset only after a green risk assessment."
    } else {
        "Fix source ambiguity, size, disk, license, token, or payment blockers before any acquisition."
    }
}

$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8

$md = @(
    "# LIBERO/RoboSuite Source Resolution Report",
    "",
    "- decision: $($report.decision)",
    "- ready for repo setup: $($report.ready_for_repo_setup)",
    "- ready for full dataset download: $($report.ready_for_full_dataset_download)",
    "- ready for tiny or metadata-only dataset setup: true",
    "",
    "## Sources",
    "",
    "- LIBERO repo: $($sources.libero_repo.source_url) -> $($sources.libero_repo.decision)",
    "- RoboSuite repo: $($sources.robosuite_repo.source_url) -> $($sources.robosuite_repo.decision)",
    "- Full LIBERO dataset: $($sources.libero_full_dataset.source_url) -> $($sources.libero_full_dataset.decision)",
    "",
    "The full LIBERO dataset is official/documented and now inside the LIBERO-only 180 GB acquisition budget if at least 250 GB disk remains after acquisition. This planner still performs no downloads.",
    "",
    "This report is planning-only. It performs no downloads, GPU jobs, training, simulator execution, rollouts, heavy VLA imports, token access, OpenVLA-OFT execution, or paper-grade claims."
) -join "`n"
$md | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8

Write-Host "LIBERO/RoboSuite official source resolution"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only and performs no acquisition."
$report | ConvertTo-Json -Depth 8
