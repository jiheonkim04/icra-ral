param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$ReadinessReportPath = "reports\libero_fixed_prior_rollout_readiness_gate_report.json",
    [string]$JsonReportPath = "reports\resetspec_state1_result.json",
    [string]$MarkdownReportPath = "reports\resetspec_state1_result.md",
    [int]$MaxStepsCap = 320,
    [int]$PostSignalMargin = 20,
    [int]$CameraSize = 64,
    [double]$GlobalScale = 0.85,
    [int]$Seed = 0,
    [int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "ResetSpec-Retarget STATE 1 bounded diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "No downloads, installs, GPU jobs, training, model loading, OpenVLA-OFT, benchmark sweep, or paper-grade claims."
Write-Host "Risk assessment: task=bounded LIBERO reset-mismatch replay/retarget diagnostic; source=local LIBERO/RoboSuite/HDF5 assets; expected_size_gb=0; target_path=$RepoRoot; expected_runtime_minutes<=30; expected_ram_gb<=4; expected_vram_gb=0; token_license_payment=none; decision=proceed if readiness gate is green and task-local gate is set."

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
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $completed = $process.WaitForExit($TimeoutSec * 1000)
        if (-not $completed) {
            try { $process.Kill() } catch {}
            return [ordered]@{ ok = $false; timed_out = $true; returncode = $null; stdout = ""; stderr = "command timed out after $TimeoutSec seconds" }
        }
        return [ordered]@{
            ok = $process.ExitCode -eq 0
            timed_out = $false
            returncode = $process.ExitCode
            stdout = $stdoutTask.Result.Trim().Replace("`0", "")
            stderr = $stderrTask.Result.Trim().Replace("`0", "")
        }
    } catch {
        return [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = $_.Exception.Message }
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}
if (-not (Test-Path -LiteralPath (Resolve-RepoPath -Path $ManifestPath))) {
    Write-Host "Refusing: manifest not found: $ManifestPath"
    exit 11
}
if (-not (Test-Path -LiteralPath (Resolve-RepoPath -Path $ReadinessReportPath))) {
    Write-Host "Refusing: readiness report not found: $ReadinessReportPath"
    exit 12
}
if ($MaxStepsCap -lt 1 -or $MaxStepsCap -gt 320) {
    Write-Host "Refusing: MaxStepsCap must be between 1 and 320."
    exit 13
}
if ($PostSignalMargin -lt 0 -or $PostSignalMargin -gt 50) {
    Write-Host "Refusing: PostSignalMargin must be between 0 and 50."
    exit 14
}
if ($CameraSize -lt 16 -or $CameraSize -gt 128) {
    Write-Host "Refusing: CameraSize must be between 16 and 128."
    exit 15
}

if ([Environment]::GetEnvironmentVariable("ALLOW_RESETSPEC_RETARGET") -ne "1") {
    & $Python -m tca_map.resetspec.retarget `
        --manifest (Resolve-RepoPath -Path $ManifestPath) `
        --readiness-report (Resolve-RepoPath -Path $ReadinessReportPath) `
        --report-json (Resolve-RepoPath -Path $JsonReportPath) `
        --report-md (Resolve-RepoPath -Path $MarkdownReportPath) `
        --max-steps-cap $MaxStepsCap `
        --post-signal-margin $PostSignalMargin `
        --camera-size $CameraSize `
        --global-scale $GlobalScale `
        --seed $Seed
    exit 0
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    Write-Host "Refusing: WSL is unavailable for simulator replay."
    exit 16
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

$bashCommand = @"
export PYTHONPATH='$($repoWsl.stdout)';
export MUJOCO_GL=osmesa;
export CUDA_VISIBLE_DEVICES=;
export ALLOW_RESETSPEC_RETARGET=1;
$WslPython -m tca_map.resetspec.retarget --manifest '$($manifestWsl.stdout)' --readiness-report '$($readinessWsl.stdout)' --report-json '$($jsonReportWsl.stdout)' --report-md '$($markdownReportWsl.stdout)' --libero-root '$($liberoRootWsl.stdout)' --robosuite-root '$($robosuiteRootWsl.stdout)' --max-steps-cap $MaxStepsCap --post-signal-margin $PostSignalMargin --camera-size $CameraSize --global-scale $GlobalScale --seed $Seed
"@
$bashCommand = $bashCommand -replace "`r", ""
$result = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
if (-not $result.ok) {
    Write-Host "ResetSpec-Retarget diagnostic command failed."
    Write-Host $result.stderr
    exit 18
}

Write-Host $result.stdout
exit 0
