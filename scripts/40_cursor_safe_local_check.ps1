$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\jiheo\tca_map"
$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe"
$ReportJson = Join-Path $RepoRoot "reports\cursor_safe_check_report.json"
$ReportMarkdown = Join-Path $RepoRoot "reports\cursor_safe_check_report.md"
$SmokeReport = Join-Path $RepoRoot "reports\smoke_report.json"

function Get-StatusPath {
    param([string]$Line)

    if ([string]::IsNullOrWhiteSpace($Line)) {
        return ""
    }
    if ($Line.Length -lt 4) {
        return $Line
    }
    return $Line.Substring(3).Trim().Trim('"')
}

function Convert-ToStringArray {
    param([object]$Value)

    if ($null -eq $Value) {
        return @()
    }
    return @($Value | Where-Object { $null -ne $_ } | ForEach-Object { $_.ToString() })
}

function Test-AllowedRuntimeStatus {
    param([string]$Line)

    $path = (Get-StatusPath -Line $Line).Replace("\", "/")
    if ([string]::IsNullOrWhiteSpace($path)) {
        return $true
    }
    $allowed = @(
        "reports/preflight_report.json",
        "reports/smoke_report.json",
        "reports/dummy_train_metrics.json",
        "reports/dummy_eval_metrics.json",
        "reports/missing_assets_runtime.json",
        "reports/missing_assets_runtime.md",
        "reports/cursor_safe_check_report.json",
        "reports/cursor_safe_check_report.md"
    )
    if ($allowed -contains $path) {
        return $true
    }
    if ($path -eq ".pytest_cache" -or $path.StartsWith(".pytest_cache/")) {
        return $true
    }
    if ($path -eq "__pycache__" -or $path.EndsWith("/__pycache__") -or $path.Contains("/__pycache__/")) {
        return $true
    }
    if ($path.EndsWith(".pyc")) {
        return $true
    }
    return $false
}

function Get-SourceStatus {
    $lines = Convert-ToStringArray -Value (& git status --porcelain)
    return @($lines | Where-Object { $_ -and -not (Test-AllowedRuntimeStatus -Line $_) })
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Script,
        [bool]$Required = $true
    )

    Write-Host ""
    Write-Host "== $Name =="
    $started = Get-Date
    $exitCode = 0
    $errorMessage = $null
    $outputLines = @()
    $global:LASTEXITCODE = 0

    try {
        $rawOutput = @(& $Script 2>&1)
        $outputLines = @($rawOutput | ForEach-Object { $_.ToString() })
        foreach ($line in $outputLines) {
            Write-Host $line
        }
        if ($null -ne $LASTEXITCODE) {
            $exitCode = [int]$LASTEXITCODE
        } elseif (-not $?) {
            $exitCode = 1
        }
    } catch {
        $exitCode = 1
        $errorMessage = $_.Exception.Message
        Write-Host "ERROR: $errorMessage" -ForegroundColor Red
    }

    $finished = Get-Date
    $passed = ($exitCode -eq 0)
    Write-Host "== $Name completed: $(if ($passed) { 'PASS' } else { 'FAIL' }) (exit $exitCode) =="

    return [ordered]@{
        name = $Name
        required = $Required
        passed = $passed
        exit_code = $exitCode
        started_at = $started.ToString("o")
        finished_at = $finished.ToString("o")
        duration_seconds = [math]::Round(($finished - $started).TotalSeconds, 3)
        error = $errorMessage
        output_tail = @($outputLines | Select-Object -Last 40)
    }
}

if (-not (Test-Path -LiteralPath $RepoRoot)) {
    throw "Repo root not found: $RepoRoot"
}

Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Cursor safe local check"
Write-Host "Repo root: $RepoRoot"
Write-Host "Python: $Python"
Write-Host "Policy: no GPU jobs, downloads, rollouts, real training, heavy VLA imports, or OpenVLA-OFT execution."
Write-Host ""

Write-Host "git branch --show-current"
$branch = (& git branch --show-current) -join "`n"
Write-Host $branch

Write-Host ""
Write-Host "git log -1 --oneline"
$commit = (& git log -1 --oneline) -join "`n"
Write-Host $commit

Write-Host ""
Write-Host "git status --short"
$initialGitStatus = @(Convert-ToStringArray -Value (& git status --short))
if ($initialGitStatus.Count -gt 0) {
    $initialGitStatus | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "(clean)"
}

$initialSourceStatus = @(Convert-ToStringArray -Value (Get-SourceStatus))
$steps = New-Object System.Collections.Generic.List[object]

$steps.Add((Invoke-Step -Name "tree_check" -Required $true -Script {
    & powershell -ExecutionPolicy Bypass -File "scripts\99_tree_check.ps1"
})) | Out-Null

$steps.Add((Invoke-Step -Name "compute_budget" -Required $true -Script {
    & powershell -ExecutionPolicy Bypass -File "scripts\30_enforce_compute_budget.ps1"
})) | Out-Null

$steps.Add((Invoke-Step -Name "pytest" -Required $true -Script {
    & $Python -m pytest -q
})) | Out-Null

$steps.Add((Invoke-Step -Name "preflight" -Required $true -Script {
    & powershell -ExecutionPolicy Bypass -File "scripts\00_preflight.ps1" -Python $Python
})) | Out-Null

$steps.Add((Invoke-Step -Name "dummy_train_smoke" -Required $true -Script {
    & powershell -ExecutionPolicy Bypass -File "scripts\04_train_smoke.ps1" -Python $Python
})) | Out-Null

$steps.Add((Invoke-Step -Name "dummy_eval_smoke" -Required $true -Script {
    & powershell -ExecutionPolicy Bypass -File "scripts\05_eval_smoke.ps1" -Python $Python
})) | Out-Null

$steps.Add((Invoke-Step -Name "real_asset_readiness" -Required $false -Script {
    & powershell -ExecutionPolicy Bypass -File "scripts\11_check_real_assets.ps1"
})) | Out-Null

Write-Host ""
Write-Host "reports\smoke_report.json"
$smokeReportExists = Test-Path -LiteralPath $SmokeReport
$smokeReportObject = $null
if ($smokeReportExists) {
    $smokeText = Get-Content -LiteralPath $SmokeReport -Raw -Encoding UTF8
    Write-Host $smokeText
    try {
        $smokeReportObject = $smokeText | ConvertFrom-Json
    } catch {
        Write-Host "WARNING: smoke_report.json exists but could not be parsed as JSON: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "MISSING: reports\smoke_report.json" -ForegroundColor Red
}

$finalGitStatus = @(Convert-ToStringArray -Value (& git status --short))
$finalSourceStatus = @(Convert-ToStringArray -Value (Get-SourceStatus))
$sourceStatusDelta = @(Compare-Object -ReferenceObject @($initialSourceStatus) -DifferenceObject @($finalSourceStatus) | ForEach-Object {
    "$($_.SideIndicator) $($_.InputObject)"
})
$sourceStatusChanged = ($sourceStatusDelta.Count -gt 0)
$stepResults = @($steps.ToArray())

$hardFailures = New-Object System.Collections.Generic.List[string]
foreach ($step in $stepResults) {
    if ($step.required -and -not $step.passed) {
        $hardFailures.Add($step.name) | Out-Null
    }
}
if (-not $smokeReportExists) {
    $hardFailures.Add("missing_smoke_report") | Out-Null
}
if ($sourceStatusChanged) {
    $hardFailures.Add("unexpected_source_modifications") | Out-Null
}

$createdAt = (Get-Date).ToString("o")
$report = [ordered]@{
    created_at = $createdAt
    repo_root = $RepoRoot
    python = $Python
    git = [ordered]@{
        branch = $branch
        commit = $commit
        initial_status_short = @($initialGitStatus)
        final_status_short = @($finalGitStatus)
        initial_source_status = @($initialSourceStatus)
        final_source_status = @($finalSourceStatus)
        source_status_delta = @($sourceStatusDelta)
        source_status_changed = $sourceStatusChanged
    }
    policy = [ordered]@{
        downloads_performed = $false
        gpu_jobs_performed = $false
        real_training_performed = $false
        dummy_train_smoke_performed = $true
        heavy_vla_imports_performed = $false
        openvla_oft_executed = $false
        real_rollouts_performed = $false
    }
    steps = @($stepResults)
    smoke_report_path = "reports/smoke_report.json"
    smoke_report_exists = $smokeReportExists
    smoke_report = $smokeReportObject
    hard_failures = @($hardFailures)
    passed = ($hardFailures.Count -eq 0)
}

$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ReportJson -Encoding UTF8

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Cursor Safe Local Check Report") | Out-Null
$md.Add("") | Out-Null
$md.Add("- Created: $createdAt") | Out-Null
$md.Add("- Branch: $branch") | Out-Null
$md.Add("- Commit: $commit") | Out-Null
$md.Add(('- Python: `{0}`' -f $Python)) | Out-Null
$md.Add(("- Passed: {0}" -f ($hardFailures.Count -eq 0))) | Out-Null
$md.Add("") | Out-Null
$md.Add("## Steps") | Out-Null
foreach ($step in $stepResults) {
    $stepStatus = if ($step.passed) { "PASS" } else { "FAIL" }
    $md.Add(("- {0}: {1} (exit {2}, required={3})" -f $step.name, $stepStatus, $step.exit_code, $step.required)) | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Hard Failures") | Out-Null
if ($hardFailures.Count -gt 0) {
    foreach ($failure in $hardFailures) {
        $md.Add("- $failure") | Out-Null
    }
} else {
    $md.Add("- none") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Source Status Delta") | Out-Null
if ($sourceStatusDelta.Count -gt 0) {
    foreach ($line in $sourceStatusDelta) {
        $md.Add(('- `{0}`' -f $line)) | Out-Null
    }
} else {
    $md.Add("- none") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Expected Missing-Asset Conditions") | Out-Null
$md.Add('The check does not fail merely because `safe_to_run_pilot_gpu`, `ready_for_smolvla_smoke`, `ready_for_openvla_oft_smoke`, or `ready_for_libero_rollout` is false.') | Out-Null
$md | Set-Content -LiteralPath $ReportMarkdown -Encoding UTF8

Write-Host ""
Write-Host "Wrote $ReportJson"
Write-Host "Wrote $ReportMarkdown"

if ($hardFailures.Count -gt 0) {
    Write-Host ""
    Write-Host "Cursor safe local check failed:" -ForegroundColor Red
    $hardFailures | ForEach-Object { Write-Host "- $_" -ForegroundColor Red }
    exit 1
}

Write-Host ""
Write-Host "Cursor safe local check passed."
exit 0
