param(
    [string]$JsonReportPath = "reports\libero_robosuite_setup_report.json",
    [string]$MarkdownReportPath = "reports\libero_robosuite_setup_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "LIBERO/RoboSuite bounded source setup"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script may shallow-clone official source repos only when ALLOW_DOWNLOADS=1."
Write-Host "It does not download the full LIBERO dataset, run simulators, run rollouts, train, use GPU, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

$allowDownloads = [Environment]::GetEnvironmentVariable("ALLOW_DOWNLOADS") -eq "1"
if (-not $allowDownloads) {
    throw "ALLOW_DOWNLOADS=1 is required for bounded source checkout acquisition. This protects against accidental downloads."
}

function Invoke-GitSafe {
    param([string[]]$Arguments)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "git"
    $psi.Arguments = (($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_.Replace('"', '\"')) + '"' } else { $_ }
    }) -join " ")
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $process = [System.Diagnostics.Process]::Start($psi)
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    $text = (($stdout, $stderr) -join "`n").Trim()
    return [ordered]@{
        returncode = $process.ExitCode
        output = $text
        ok = $process.ExitCode -eq 0
    }
}

function Assert-ApprovedTarget {
    param([string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $approvedRoot = [System.IO.Path]::GetFullPath("C:\assets\repos")
    if (-not $fullPath.StartsWith($approvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing target outside approved repo asset root: $fullPath"
    }
    return $fullPath
}

function Ensure-ShallowClone {
    param(
        [string]$Name,
        [string]$Url,
        [string]$TargetPath
    )

    $target = Assert-ApprovedTarget -Path $TargetPath
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    if (Test-Path -LiteralPath $target) {
        $children = @(Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue)
        $gitDir = Join-Path $target ".git"
        if ($children.Count -eq 0) {
            Remove-Item -LiteralPath $target -Force
        } elseif (Test-Path -LiteralPath $gitDir) {
            $remote = Invoke-GitSafe -Arguments @("-C", $target, "remote", "get-url", "origin")
            if (-not $remote.ok -or $remote.output.Trim() -ne $Url) {
                throw "$Name target exists but origin does not match official URL."
            }
            return [ordered]@{
                name = $Name
                url = $Url
                target_path = $target
                action = "already_present"
                downloaded = $false
                ok = $true
                output = "Existing checkout with matching origin."
            }
        } else {
            throw "$Name target exists and is not an empty directory or git checkout: $target"
        }
    }

    $clone = Invoke-GitSafe -Arguments @("clone", "--depth", "1", $Url, $target)
    if (-not $clone.ok) {
        throw "$Name shallow clone failed: $($clone.output)"
    }

    return [ordered]@{
        name = $Name
        url = $Url
        target_path = $target
        action = "shallow_cloned"
        downloaded = $true
        ok = $true
        output = $clone.output
    }
}

$sourceReportPath = Join-Path $RepoRoot "reports\libero_robosuite_source_resolution_report.json"
if (-not (Test-Path -LiteralPath $sourceReportPath)) {
    & powershell -ExecutionPolicy Bypass -File scripts\45_resolve_libero_robosuite_sources.ps1 | Out-Null
}
$sourceReport = Get-Content -LiteralPath $sourceReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not [bool]$sourceReport.ready_for_repo_setup) {
    throw "Source resolution is not green for repo setup. Run scripts\45_resolve_libero_robosuite_sources.ps1 and inspect blockers."
}

$libero = Ensure-ShallowClone -Name "LIBERO" -Url "https://github.com/Lifelong-Robot-Learning/LIBERO.git" -TargetPath "C:\assets\repos\LIBERO"
$robosuite = Ensure-ShallowClone -Name "RoboSuite" -Url "https://github.com/ARISE-Initiative/robosuite.git" -TargetPath "C:\assets\repos\robosuite"
$dataRoot = "C:\assets\data\libero"
New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null
$marker = Join-Path $dataRoot "_NO_FULL_DATASET_DOWNLOADED.txt"
if (-not (Test-Path -LiteralPath $marker)) {
    @(
        "This directory is path-ready only.",
        "The full LIBERO dataset was not downloaded by this source-setup script. Use scripts\49_acquire_libero_data.ps1 for the official LIBERO-only acquisition gate with the 180 GB budget and 250 GB free-after requirement.",
        "Place a documented tiny subset or rerun a future risk-assessed dataset acquisition task before offline dataset smoke."
    ) -join "`n" | Set-Content -LiteralPath $marker -Encoding UTF8
}

$report = [ordered]@{
    policy = [ordered]@{
        task_local_allow_downloads = $true
        source_repos_only = $true
        full_dataset_downloaded = $false
        gpu_jobs_performed = $false
        training_performed = $false
        simulator_executed = $false
        rollouts_performed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    risk_assessment = [ordered]@{
        task = "bounded LIBERO/RoboSuite source checkout setup"
        source_urls = @($libero.url, $robosuite.url)
        official_or_documented = $true
        expected_size_or_method = "source repos only, estimated under 1.1 GB total from GitHub API sizes; full LIBERO dataset excluded"
        target_paths = @($libero.target_path, $robosuite.target_path, $dataRoot)
        login_token_required = $false
        license_click_through_required = $false
        payment_required = $false
        expected_runtime = "minutes"
        simulator_will_run = $false
        rollout_will_run = $false
        decision = "proceed"
        reason = "official source repo setup is inside budget; full dataset is not downloaded"
    }
    actions = [ordered]@{
        libero = $libero
        robosuite = $robosuite
        data_root = [ordered]@{
            path = $dataRoot
            action = "created_or_present"
            full_dataset_downloaded = $false
            marker = $marker
        }
    }
    ready_for_libero_repo_path_check = [bool](Test-Path -LiteralPath "C:\assets\repos\LIBERO")
    ready_for_robosuite_repo_path_check = [bool](Test-Path -LiteralPath "C:\assets\repos\robosuite")
    ready_for_libero_data_path_check = [bool](Test-Path -LiteralPath $dataRoot)
    ready_for_libero_offline_subset = $false
    ready_for_simulator_import_smoke = $false
    ready_for_rollout = $false
    recommended_next_step = "Run dataset and simulator readiness planners. Continue to tiny real/offline dataset interface smoke only after a real tiny subset file is present under LIBERO_DATA_ROOT."
}

$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8

$md = @(
    "# LIBERO/RoboSuite Source Setup Report",
    "",
    "- decision: proceed",
    "- LIBERO action: $($libero.action)",
    "- RoboSuite action: $($robosuite.action)",
    "- data root action: created_or_present",
    "- full dataset downloaded: false",
    "- simulator executed: false",
    "- rollout performed: false",
    "",
    "This setup is path/source readiness only. It is not an offline dataset result, not rollout readiness, and not paper-grade evidence."
) -join "`n"
$md | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8

$report | ConvertTo-Json -Depth 8
