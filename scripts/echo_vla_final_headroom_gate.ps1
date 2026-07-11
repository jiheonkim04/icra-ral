param(
    [string]$WslPython = "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python",
    [int]$CandidateCount = 8,
    [int]$MaxHorizon = 16,
    [int]$TimeoutSeconds = 14400,
    [switch]$DeterminismOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "ECHO-VLA final candidate-headroom gate"
Write-Host "Repo root: $RepoRoot"
Write-Host "This runs frozen official SmolVLA stochastic candidate generation, exact same-state LIBERO interventions, downstream frozen-policy continuation, and separate structured perturbation diagnostics. It does not train ECHO, train SmolVLA, run OpenVLA-OFT, run a full benchmark, or download assets."

if ($CandidateCount -ne 8) {
    Write-Host "Refusing: CandidateCount must be exactly 8 for the final gate."
    exit 10
}
if ($MaxHorizon -lt 4 -or $MaxHorizon -gt 16) {
    Write-Host "Refusing: MaxHorizon must be 4..16."
    exit 11
}
if ($TimeoutSeconds -lt 600 -or $TimeoutSeconds -gt 28800) {
    Write-Host "Refusing: TimeoutSeconds must be 600..28800."
    exit 12
}
if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Host "Refusing: wsl command not found."
    exit 13
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
$determinismArg = ""
if ($DeterminismOnly) {
    $determinismArg = "--determinism-only"
}
$cmd = @"
set -e
cd '$repoWsl'
if [ ! -x '$WslPython' ]; then
  echo 'Refusing: WSL Python not executable: $WslPython'
  exit 14
fi
export PYTHONPATH='$repoWsl'
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
'$WslPython' scripts/run_echo_vla_final_headroom_gate.py --candidate-count $CandidateCount --max-horizon $MaxHorizon --report-dir reports $determinismArg
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
    Write-Host "Refusing: ECHO-VLA final headroom gate timed out after $TimeoutSeconds seconds."
    exit 15
}
exit $process.ExitCode
