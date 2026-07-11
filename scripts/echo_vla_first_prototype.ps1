param(
    [string]$WslPython = "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python",
    [int]$MaxTasks = 2,
    [int]$MaxStatesPerTask = 2,
    [int]$CandidateCount = 4,
    [int]$Horizon = 4,
    [double]$PerturbScale = 0.025,
    [int]$TimeoutSeconds = 3600
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "ECHO-VLA first prototype candidate-headroom gate"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runs frozen SmolVLA candidate proposal plus bounded same-state LIBERO candidate interventions. It does not train OpenVLA-OFT, use OpenVLA-OFT, run a full benchmark, download, or train the SmolVLA backbone."

if ($MaxTasks -lt 1 -or $MaxTasks -gt 4) {
    Write-Host "Refusing: MaxTasks must be 1..4"
    exit 10
}
if ($MaxStatesPerTask -lt 1 -or $MaxStatesPerTask -gt 5) {
    Write-Host "Refusing: MaxStatesPerTask must be 1..5"
    exit 11
}
if ($CandidateCount -lt 2 -or $CandidateCount -gt 8) {
    Write-Host "Refusing: CandidateCount must be 2..8"
    exit 12
}
if ($Horizon -lt 1 -or $Horizon -gt 16) {
    Write-Host "Refusing: Horizon must be 1..16"
    exit 13
}
if ($PerturbScale -lt 0.0 -or $PerturbScale -gt 0.10) {
    Write-Host "Refusing: PerturbScale must be 0..0.10"
    exit 14
}

if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Host "Refusing: wsl command not found."
    exit 15
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

$repoWsl = ConvertTo-WslPath -Path $RepoRoot
$cmd = @"
set -e
cd '$repoWsl'
if [ ! -x '$WslPython' ]; then
  echo 'Refusing: WSL Python not executable: $WslPython'
  exit 16
fi
export PYTHONPATH='$repoWsl'
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
'$WslPython' scripts/run_echo_vla_first_prototype.py headroom --max-tasks $MaxTasks --max-states-per-task $MaxStatesPerTask --candidate-count $CandidateCount --horizon $Horizon --perturb-scale $PerturbScale --report-dir reports
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
    Write-Host "Refusing: ECHO-VLA first prototype timed out after $TimeoutSeconds seconds."
    exit 17
}
exit $process.ExitCode
