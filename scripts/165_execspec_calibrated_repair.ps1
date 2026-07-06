param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$DataRoot = "C:\assets\data\libero",
    [string]$JsonReportPath = "reports\execspec_state2_calibrated_repair.json",
    [string]$MarkdownReportPath = "reports\execspec_state2_calibrated_repair.md",
    [int]$MaxCalibrationDemos = 5,
    [int]$MaxEvalDemos = 1,
    [int]$MaxActionsPerDemo = 300,
    [int]$MaxStepsCap = 300,
    [int]$PostSignalMargin = 20,
    [int]$CameraSize = 64,
    [string]$ReplayMismatches = "gripper_sign_flip,translation_scale_mismatch",
    [int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "ExecSpec STATE 2 calibrated repair"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script fits repair layers on calibration HDF5 demos and evaluates held-out action metrics."
Write-Host "Exact-init replay runs only under ALLOW_EXECSPEC_CALIBRATED_REPAIR_REPLAY=1."
Write-Host "It does not download, install, load VLA models, infer, train neural models, use GPU, execute OpenVLA-OFT, run benchmark rollouts, or make paper claims."

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

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}
if (-not (Test-Path -LiteralPath (Resolve-RepoPath -Path $ManifestPath))) {
    Write-Host "Refusing: manifest not found: $ManifestPath"
    exit 11
}
if (-not (Test-Path -LiteralPath $DataRoot)) {
    Write-Host "Refusing: data root not found: $DataRoot"
    exit 12
}
if ($MaxCalibrationDemos -lt 1 -or $MaxCalibrationDemos -gt 20) {
    Write-Host "Refusing: MaxCalibrationDemos must be between 1 and 20."
    exit 13
}
if ($MaxEvalDemos -lt 1 -or $MaxEvalDemos -gt 5) {
    Write-Host "Refusing: MaxEvalDemos must be between 1 and 5."
    exit 14
}
if ($MaxActionsPerDemo -lt 2 -or $MaxActionsPerDemo -gt 512) {
    Write-Host "Refusing: MaxActionsPerDemo must be between 2 and 512."
    exit 15
}
if ($MaxStepsCap -lt 1 -or $MaxStepsCap -gt 320) {
    Write-Host "Refusing: MaxStepsCap must be between 1 and 320."
    exit 16
}
if ($PostSignalMargin -lt 0 -or $PostSignalMargin -gt 50) {
    Write-Host "Refusing: PostSignalMargin must be between 0 and 50."
    exit 17
}
if ($CameraSize -lt 16 -or $CameraSize -gt 128) {
    Write-Host "Refusing: CameraSize must be between 16 and 128."
    exit 18
}

if ([Environment]::GetEnvironmentVariable("ALLOW_EXECSPEC_CALIBRATED_REPAIR_REPLAY") -ne "1") {
    & $Python -m tca_map.execspec.repair `
        --manifest (Resolve-RepoPath -Path $ManifestPath) `
        --data-root $DataRoot `
        --report-json (Resolve-RepoPath -Path $JsonReportPath) `
        --report-md (Resolve-RepoPath -Path $MarkdownReportPath) `
        --max-calibration-demos $MaxCalibrationDemos `
        --max-eval-demos $MaxEvalDemos `
        --max-actions-per-demo $MaxActionsPerDemo `
        --max-steps-cap $MaxStepsCap `
        --post-signal-margin $PostSignalMargin `
        --camera-size $CameraSize `
        --replay-mismatches $ReplayMismatches
    exit $LASTEXITCODE
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    Write-Host "Refusing: WSL is unavailable for exact-init replay."
    exit 19
}

$repoWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($RepoRoot.Replace("\", "/"))) -TimeoutSec 30
$manifestWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $ManifestPath).Replace("\", "/"))) -TimeoutSec 30
$dataRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($DataRoot.Replace("\", "/"))) -TimeoutSec 30
$jsonReportWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $JsonReportPath).Replace("\", "/"))) -TimeoutSec 30
$markdownReportWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $MarkdownReportPath).Replace("\", "/"))) -TimeoutSec 30
$liberoRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/LIBERO") -TimeoutSec 30
$robosuiteRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/robosuite") -TimeoutSec 30
$pythonProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "test -x $WslPython && $WslPython --version") -TimeoutSec 30

if (-not ($repoWsl.ok -and $manifestWsl.ok -and $dataRootWsl.ok -and $jsonReportWsl.ok -and $markdownReportWsl.ok -and $liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $pythonProbe.ok)) {
    Write-Host "Refusing: WSL path or Python probe failed."
    exit 20
}

$bashCommand = @"
export PYTHONPATH='$($repoWsl.stdout)';
export MUJOCO_GL=osmesa;
export ALLOW_EXECSPEC_CALIBRATED_REPAIR_REPLAY=1;
$WslPython -m tca_map.execspec.repair --manifest '$($manifestWsl.stdout)' --data-root '$($dataRootWsl.stdout)' --report-json '$($jsonReportWsl.stdout)' --report-md '$($markdownReportWsl.stdout)' --libero-root '$($liberoRootWsl.stdout)' --robosuite-root '$($robosuiteRootWsl.stdout)' --max-calibration-demos $MaxCalibrationDemos --max-eval-demos $MaxEvalDemos --max-actions-per-demo $MaxActionsPerDemo --max-steps-cap $MaxStepsCap --post-signal-margin $PostSignalMargin --camera-size $CameraSize --replay-mismatches '$ReplayMismatches'
"@
$bashCommand = $bashCommand -replace "`r", ""
$result = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
if (-not $result.ok) {
    Write-Host "ExecSpec STATE 2 calibrated repair command failed."
    Write-Host $result.stderr
    exit 21
}

Write-Host $result.stdout
exit 0
