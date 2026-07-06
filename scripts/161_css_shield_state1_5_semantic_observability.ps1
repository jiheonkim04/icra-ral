param(
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$ManifestPath = "reports/libero_offline_counterfactual_split_scaled_report.json",
    [string]$ReportJsonPath = "reports/css_shield_state1_5_semantic_diagnostic_report.json",
    [string]$ReportMarkdownPath = "reports/css_shield_state1_5_semantic_diagnostic_report.md",
    [string]$InventoryJsonPath = "reports/css_shield_state1_5_object_inventory.json",
    [string]$InventoryMarkdownPath = "reports/css_shield_state1_5_object_inventory.md",
    [int]$CameraSize = 64,
    [double]$MaxTranslationNorm = 0.20,
    [int]$State2Trials = 20,
    [switch]$IncludeNative,
    [switch]$RunState2IfGreen,
    [int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "CSS-Shield State 1.5 semantic observability diagnostic via WSL"
Write-Host "Repo root: $RepoRoot"
Write-Host "This is bounded diagnostic execution only: no training, no downloads, no GPU job, no OpenVLA-OFT, and no paper-grade claim."

if ($State2Trials -lt 1 -or $State2Trials -gt 50) {
    Write-Host "Refusing: State2Trials must be between 1 and 50."
    exit 12
}

if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
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
$liberoWsl = ConvertTo-WslPath -Path "C:\assets\repos\LIBERO"
$robosuiteWsl = ConvertTo-WslPath -Path "C:\assets\repos\robosuite"
$manifestWsl = ConvertTo-WslPath -Path (Resolve-RepoPath -Path $ManifestPath)
$reportJsonWsl = ConvertTo-WslPath -Path (Resolve-RepoPath -Path $ReportJsonPath)
$reportMdWsl = ConvertTo-WslPath -Path (Resolve-RepoPath -Path $ReportMarkdownPath)
$inventoryJsonWsl = ConvertTo-WslPath -Path (Resolve-RepoPath -Path $InventoryJsonPath)
$inventoryMdWsl = ConvertTo-WslPath -Path (Resolve-RepoPath -Path $InventoryMarkdownPath)
$includeNativeArg = if ($IncludeNative) { "--include-native" } else { "" }
$runState2Arg = if ($RunState2IfGreen) { "--run-state2-if-green" } else { "" }

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
export ALLOW_CSS_SHIELD_STATE15=1
if [ ! -x $WslPython ]; then
  echo "Refusing: WSL Python not executable: $WslPython"
  exit 16
fi
$WslPython -m tca_map.css_shield.semantic_observability \
  --manifest '$manifestWsl' \
  --report-json '$reportJsonWsl' \
  --report-md '$reportMdWsl' \
  --inventory-json '$inventoryJsonWsl' \
  --inventory-md '$inventoryMdWsl' \
  --camera-size $CameraSize \
  --max-translation-norm $MaxTranslationNorm \
  --state2-trials $State2Trials \
  $includeNativeArg \
  $runState2Arg
"@

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
    Write-Host "Refusing: CSS-Shield State 1.5 diagnostic timed out after $TimeoutSeconds seconds."
    exit 17
}
exit $process.ExitCode
