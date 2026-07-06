param(
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$ManifestPath = "reports/libero_offline_counterfactual_split_scaled_report.json",
    [string]$JsonReportPath = "reports/css_shield_minimal_rollout_diagnostic_report.json",
    [string]$MarkdownReportPath = "reports/css_shield_minimal_rollout_diagnostic_report.md",
    [string]$ProposalSource = "native_or_synthetic",
    [int]$MaxSteps = 5,
    [int]$CaseIndex = 0,
    [int]$CameraSize = 64,
    [double]$MaxTranslationNorm = 0.20,
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "CSS-Shield minimal rollout diagnostic via WSL"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runs a bounded LIBERO/RoboSuite exact-init diagnostic. It does not train, download, use GPU, execute OpenVLA-OFT, or make paper-grade claims."

if ($MaxSteps -lt 1 -or $MaxSteps -gt 25) {
    Write-Host "Refusing: MaxSteps must be between 1 and 25."
    exit 12
}
if ($ProposalSource -notin @("native_or_synthetic", "native_smolvla", "synthetic_counterfactual_probe")) {
    Write-Host "Refusing: ProposalSource must be native_or_synthetic, native_smolvla, or synthetic_counterfactual_probe."
    exit 13
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    Write-Host "Refusing: wsl command not found."
    exit 14
}

function ConvertTo-WslPath {
    param([string]$Path)
    $full = [System.IO.Path]::GetFullPath($Path)
    if ($full -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    return ($full -replace '\\', '/')
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

$repoWsl = ConvertTo-WslPath -Path $RepoRoot
if ([string]::IsNullOrWhiteSpace($repoWsl)) {
    Write-Host "Refusing: could not convert repo path to WSL path."
    exit 15
}
$liberoWsl = ConvertTo-WslPath -Path "C:\assets\repos\LIBERO"
$robosuiteWsl = ConvertTo-WslPath -Path "C:\assets\repos\robosuite"
$manifestFull = Resolve-RepoPath -Path $ManifestPath
$jsonFull = Resolve-RepoPath -Path $JsonReportPath
$mdFull = Resolve-RepoPath -Path $MarkdownReportPath
$manifestWsl = ConvertTo-WslPath -Path $manifestFull
$jsonWsl = ConvertTo-WslPath -Path $jsonFull
$mdWsl = ConvertTo-WslPath -Path $mdFull

$cmd = @"
set -e
cd '$repoWsl'
export PYTHONPATH='$repoWsl':'$liberoWsl':'$robosuiteWsl'
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=osmesa
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=
export ALLOW_CSS_SHIELD_ROLLOUT=1
if [ ! -x $WslPython ]; then
  echo "Refusing: WSL Python not executable: $WslPython"
  exit 16
fi
$WslPython -m tca_map.css_shield.minimal_rollout_diagnostic \
  --manifest '$manifestWsl' \
  --report-json '$jsonWsl' \
  --report-md '$mdWsl' \
  --proposal-source '$ProposalSource' \
  --max-steps $MaxSteps \
  --case-index $CaseIndex \
  --camera-size $CameraSize \
  --max-translation-norm $MaxTranslationNorm
"@

$cmd = $cmd -replace "`r", ""

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "wsl"
$psi.Arguments = "bash -lc " + '"' + ($cmd.Replace('"', '\"')) + '"'
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
$process = [System.Diagnostics.Process]::Start($psi)
$completed = $process.WaitForExit($TimeoutSeconds * 1000)
$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()
Write-Host $stdout
if (-not [string]::IsNullOrWhiteSpace($stderr)) {
    Write-Host $stderr
}
if (-not $completed) {
    try { $process.Kill() } catch {}
    Write-Host "Refusing: CSS-Shield WSL diagnostic timed out after $TimeoutSeconds seconds."
    exit 17
}
exit $process.ExitCode
