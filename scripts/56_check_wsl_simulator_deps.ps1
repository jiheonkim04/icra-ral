param(
    [string]$JsonReportPath = "reports\wsl_simulator_dependency_report.json",
    [string]$MarkdownReportPath = "reports\wsl_simulator_dependency_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "WSL simulator dependency check"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is check-only. It does not install packages, download assets, render, rollout, train, run GPU jobs, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

function Invoke-SafeCommand {
    param(
        [string[]]$Command,
        [int]$TimeoutSec = 60,
        [System.Text.Encoding]$Encoding = [System.Text.Encoding]::UTF8
    )
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Command[0]
        if ($Command.Count -gt 1) {
            $psi.Arguments = (($Command[1..($Command.Count - 1)] | ForEach-Object {
                if ($_ -match '[\s"]') { '"' + ($_.Replace('"', '\"')) + '"' } else { $_ }
            }) -join " ")
        }
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.StandardOutputEncoding = $Encoding
        $psi.StandardErrorEncoding = $Encoding
        $process = [System.Diagnostics.Process]::Start($psi)
        $completed = $process.WaitForExit($TimeoutSec * 1000)
        if (-not $completed) {
            try { $process.Kill() } catch {}
            return [ordered]@{ ok = $false; timed_out = $true; returncode = $null; stdout = ""; stderr = "command timed out after $TimeoutSec seconds" }
        }
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        return [ordered]@{
            ok = $process.ExitCode -eq 0
            timed_out = $false
            returncode = $process.ExitCode
            stdout = $stdout.Trim().Replace("`0", "")
            stderr = $stderr.Trim().Replace("`0", "")
        }
    } catch {
        return [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = $_.Exception.Message }
    }
}

function Read-JsonIfPresent {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $text = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
        return $text | ConvertFrom-Json
    } catch {
        return $null
    }
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
$wslInstalled = $null -ne $wslCommand
$pythonVersion = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "python3", "--version") } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$pipVersion = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "python3", "-m", "pip", "--version") } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$ensurePip = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "python3", "-m", "ensurepip", "--version") } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$numpyProbe = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "python3", "-c", "import numpy; print(numpy.__version__)") } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$aptProbe = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "which", "apt") } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$userSite = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "python3", "-m", "site", "--user-site") } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }

$importReport = Read-JsonIfPresent -Path (Join-Path $RepoRoot "reports\bounded_simulator_import_smoke_report.json")
$missingFromImport = New-Object System.Collections.Generic.List[string]
if ($null -ne $importReport -and $null -ne $importReport.import_results) {
    foreach ($item in $importReport.import_results) {
        if ($item.error -match "No module named '([^']+)'") {
            $missingFromImport.Add($Matches[1])
        }
    }
}

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
if (-not $wslInstalled) { $stopReasons.Add("wsl command is not available") }
if (-not $pythonVersion.ok) { $stopReasons.Add("WSL python3 is not available") }
if (-not $numpyProbe.ok) { $stopReasons.Add("WSL python3 cannot import numpy") }
if (-not $pipVersion.ok -and -not $ensurePip.ok) {
    $stopReasons.Add("WSL python3 has neither pip nor ensurepip; automatic dependency installation would require a separate system/package-management risk gate")
}
if ($aptProbe.ok) {
    $warnings.Add("apt is available in WSL, but apt-based installs are system-level/admin-adjacent and must not run automatically from this checker")
}

$readyForUserLevelPipInstall = [bool]($wslInstalled -and $pythonVersion.ok -and ($pipVersion.ok -or $ensurePip.ok))
$readyForSimulatorImportRetry = [bool]($wslInstalled -and $pythonVersion.ok -and $numpyProbe.ok)
$decision = if ($readyForSimulatorImportRetry) { "proceed" } else { "stop" }
$recommendedNextStep = if ($readyForSimulatorImportRetry) {
    "Rerun scripts\55_bounded_simulator_import_smoke.ps1 with task-local ALLOW_SIMULATOR_IMPORT_SMOKE=1."
} elseif ($readyForUserLevelPipInstall) {
    "Run a separate dependency risk assessment for user-level WSL Python package installation, then install only the missing small dependencies if green."
} else {
    "Configure WSL Python packaging manually or through a separately approved system-level setup; do not render or rollout."
}

$report = [ordered]@{
    policy = [ordered]@{
        check_only = $true
        installs_performed = $false
        downloads_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        render_smoke_performed = $false
        rollouts_performed = $false
        simulator_environment_steps_performed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    wsl = [ordered]@{
        command_installed = [bool]$wslInstalled
        python3_version = $pythonVersion
        pip_version = $pipVersion
        ensurepip = $ensurePip
        numpy_probe = $numpyProbe
        apt_probe = $aptProbe
        user_site = $userSite
    }
    source_import_report_present = $null -ne $importReport
    missing_modules_from_import_smoke = @($missingFromImport)
    ready_for_user_level_pip_install = $readyForUserLevelPipInstall
    ready_for_simulator_import_retry = $readyForSimulatorImportRetry
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
    "# WSL Simulator Dependency Report",
    "",
    "- decision: $decision",
    "- ready for user-level pip install: $readyForUserLevelPipInstall",
    "- ready for simulator import retry: $readyForSimulatorImportRetry",
    "- missing modules from import smoke: $(@($missingFromImport) -join ', ')",
    "- recommended next step: $recommendedNextStep",
    "",
    "This report is check-only. It performs no installs, downloads, render smoke, rollouts, GPU jobs, training, heavy VLA imports, OpenVLA-OFT execution, token access, or paper claims."
) -join "`n"
$markdown | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8

$report | ConvertTo-Json -Depth 8
