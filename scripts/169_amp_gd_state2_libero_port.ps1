param(
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$JsonReportPath = "reports\amp_gd_state2_report.json",
    [string]$MarkdownReportPath = "reports\amp_gd_state2_result.md",
    [string]$InventoryJsonPath = "reports\amp_gd_state2_libero_inventory.json",
    [int]$ToyTrials = 60,
    [string]$Seeds = "11,23,37",
    [int]$CaseIndex = 0,
    [int]$CameraSize = 64,
    [int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "AMP-GD State 2 toy robustness plus LIBERO object-observable port"
Write-Host "Repo root: $RepoRoot"
Write-Host "Bounded diagnostic only: no training, no downloads, no GPU, no heavy VLA imports, no OpenVLA-OFT, no paper-grade claim."

if ($ToyTrials -lt 20 -or $ToyTrials -gt 120) {
    Write-Host "Refusing: ToyTrials must be between 20 and 120."
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
$reportJsonWsl = ConvertTo-WslPath -Path (Resolve-RepoPath -Path $JsonReportPath)
$reportMdWsl = ConvertTo-WslPath -Path (Resolve-RepoPath -Path $MarkdownReportPath)
$inventoryJsonWsl = ConvertTo-WslPath -Path (Resolve-RepoPath -Path $InventoryJsonPath)

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
export ALLOW_AMP_GD_STATE2=1
if [ ! -x $WslPython ]; then
  echo "Refusing: WSL Python not executable: $WslPython"
  exit 16
fi
$WslPython -m tca_map.amp_gd.state2_libero_port \
  --toy-trials $ToyTrials \
  --seeds '$Seeds' \
  --manifest '$manifestWsl' \
  --report-json '$reportJsonWsl' \
  --report-md '$reportMdWsl' \
  --inventory-json '$inventoryJsonWsl' \
  --libero-root '$liberoWsl' \
  --robosuite-root '$robosuiteWsl' \
  --case-index $CaseIndex \
  --camera-size $CameraSize \
  --run-libero-probe
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
    Write-Host "Refusing: AMP-GD State 2 diagnostic timed out after $TimeoutSeconds seconds."
    exit 17
}
exit $process.ExitCode
