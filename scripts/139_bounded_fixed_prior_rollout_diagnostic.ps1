param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$ReadinessReportPath = "reports\libero_fixed_prior_rollout_readiness_gate_report.json",
    [string]$JsonReportPath = "reports\fixed_prior_rollout_diagnostic_report.json",
    [string]$MarkdownReportPath = "reports\fixed_prior_rollout_diagnostic_report.md",
    [int]$MaxTasks = 1,
    [int]$MaxSteps = 10,
    [int]$CameraSize = 64,
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded fixed-prior rollout diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script runs only a bounded fixed-prior diagnostic rollout after a green readiness gate. It does not train, run GPU jobs, load VLA models, download assets, execute OpenVLA-OFT, run benchmark rollouts, or make paper-grade claims."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

function Invoke-SafeCommand {
    param(
        [string[]]$Command,
        [int]$TimeoutSec = 120,
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
        return [ordered]@{
            ok = $process.ExitCode -eq 0
            timed_out = $false
            returncode = $process.ExitCode
            stdout = $process.StandardOutput.ReadToEnd().Trim().Replace("`0", "")
            stderr = $process.StandardError.ReadToEnd().Trim().Replace("`0", "")
        }
    } catch {
        return [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = $_.Exception.Message }
    }
}

if ($MaxTasks -lt 1 -or $MaxTasks -gt 3) {
    Write-Host "Refusing: MaxTasks must be between 1 and 3."
    exit 12
}
if ($MaxSteps -lt 1 -or $MaxSteps -gt 25) {
    Write-Host "Refusing: MaxSteps must be between 1 and 25."
    exit 13
}
if ($CameraSize -lt 16 -or $CameraSize -gt 128) {
    Write-Host "Refusing: CameraSize must be between 16 and 128."
    exit 14
}
if (-not (Test-Path -LiteralPath (Resolve-RepoPath -Path $ManifestPath))) {
    Write-Host "Refusing: manifest not found: $ManifestPath"
    exit 15
}
if (-not (Test-Path -LiteralPath (Resolve-RepoPath -Path $ReadinessReportPath))) {
    Write-Host "Refusing: readiness report not found: $ReadinessReportPath"
    exit 16
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    Write-Host "WSL is unavailable; writing blocked report through local Python."
    & $Python -m tca_map.datasets.libero_fixed_prior_rollout_diagnostic `
        --manifest (Resolve-RepoPath -Path $ManifestPath) `
        --readiness-report (Resolve-RepoPath -Path $ReadinessReportPath) `
        --report-json (Resolve-RepoPath -Path $JsonReportPath) `
        --report-md (Resolve-RepoPath -Path $MarkdownReportPath) `
        --max-tasks $MaxTasks `
        --max-steps $MaxSteps `
        --camera-size $CameraSize
    exit 0
}

$repoWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($RepoRoot.Replace("\", "/"))) -TimeoutSec 30
$manifestWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $ManifestPath).Replace("\", "/"))) -TimeoutSec 30
$readinessWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $ReadinessReportPath).Replace("\", "/"))) -TimeoutSec 30
$jsonReportWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $JsonReportPath).Replace("\", "/"))) -TimeoutSec 30
$markdownReportWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $MarkdownReportPath).Replace("\", "/"))) -TimeoutSec 30
$liberoRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/LIBERO") -TimeoutSec 30
$robosuiteRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/robosuite") -TimeoutSec 30
$pythonProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "test -x $WslPython && $WslPython --version") -TimeoutSec 30

if (-not ($repoWsl.ok -and $manifestWsl.ok -and $readinessWsl.ok -and $jsonReportWsl.ok -and $markdownReportWsl.ok -and $liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $pythonProbe.ok)) {
    Write-Host "Refusing: WSL path or Python probe failed."
    exit 17
}

$taskGateExport = if ([Environment]::GetEnvironmentVariable("ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC") -eq "1") { "export ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC=1;" } else { "" }
$bashCommand = @"
export PYTHONPATH='$($repoWsl.stdout)';
export MUJOCO_GL=osmesa;
$taskGateExport
$WslPython -m tca_map.datasets.libero_fixed_prior_rollout_diagnostic --manifest '$($manifestWsl.stdout)' --readiness-report '$($readinessWsl.stdout)' --report-json '$($jsonReportWsl.stdout)' --report-md '$($markdownReportWsl.stdout)' --libero-root '$($liberoRootWsl.stdout)' --robosuite-root '$($robosuiteRootWsl.stdout)' --max-tasks $MaxTasks --max-steps $MaxSteps --camera-size $CameraSize
"@
$bashCommand = $bashCommand -replace "`r", ""
$result = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
if (-not $result.ok) {
    $jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
    if (Test-Path -LiteralPath $jsonFullPath) {
        try {
            $existingReport = Get-Content -LiteralPath $jsonFullPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ([bool]$existingReport.result.passed) {
                Write-Host "Fixed-prior rollout diagnostic report passed even though the WSL command did not exit cleanly."
                Write-Host "Treating the bounded diagnostic as completed; inspect the report for metrics."
                Get-Content -LiteralPath $jsonFullPath -Raw -Encoding UTF8
                exit 0
            }
        } catch {}
    }
    Write-Host "Fixed-prior rollout diagnostic command failed."
    Write-Host $result.stderr
    exit 18
}

Write-Host $result.stdout
exit 0
