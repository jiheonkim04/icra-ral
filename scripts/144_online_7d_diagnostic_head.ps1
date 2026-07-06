param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$JsonReportPath = "reports\online_7d_diagnostic_head_report.json",
    [string]$MarkdownReportPath = "reports\online_7d_diagnostic_head_report.md",
    [int]$MaxSteps = 25,
    [int]$TrainMaxSteps = 64,
    [int]$SampleStride = 4,
    [int]$CameraSize = 64,
    [int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Online 7D ActionMap/TCA diagnostic head"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script trains tiny CPU 7D diagnostic heads from HDF5 training labels and only runs bounded matched-init rollout under ALLOW_ONLINE_7D_DIAGNOSTIC_HEAD_ROLLOUT=1."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

function Invoke-SafeCommand {
    param([string[]]$Command, [int]$TimeoutSec = 120)
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
        $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
        $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
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

if ($MaxSteps -lt 1 -or $MaxSteps -gt 25) {
    Write-Host "Refusing: MaxSteps must be between 1 and 25 for this first diagnostic."
    exit 12
}
if ($TrainMaxSteps -lt 1 -or $TrainMaxSteps -gt 256) {
    Write-Host "Refusing: TrainMaxSteps must be between 1 and 256."
    exit 13
}
if ($SampleStride -lt 1 -or $SampleStride -gt 32) {
    Write-Host "Refusing: SampleStride must be between 1 and 32."
    exit 14
}
if ($CameraSize -lt 16 -or $CameraSize -gt 128) {
    Write-Host "Refusing: CameraSize must be between 16 and 128."
    exit 15
}

if ([Environment]::GetEnvironmentVariable("ALLOW_ONLINE_7D_DIAGNOSTIC_HEAD_ROLLOUT") -ne "1") {
    & $Python -m tca_map.smolvla.online_7d_diagnostic_head `
        --manifest (Resolve-RepoPath -Path $ManifestPath) `
        --report-json (Resolve-RepoPath -Path $JsonReportPath) `
        --report-md (Resolve-RepoPath -Path $MarkdownReportPath) `
        --max-steps $MaxSteps `
        --train-max-steps $TrainMaxSteps `
        --sample-stride $SampleStride `
        --camera-size $CameraSize
    exit $LASTEXITCODE
}

if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Host "Refusing: WSL is unavailable for simulator diagnosis."
    exit 16
}

$repoWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($RepoRoot.Replace("\", "/"))) -TimeoutSec 30
$manifestWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $ManifestPath).Replace("\", "/"))) -TimeoutSec 30
$jsonReportWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $JsonReportPath).Replace("\", "/"))) -TimeoutSec 30
$markdownReportWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $MarkdownReportPath).Replace("\", "/"))) -TimeoutSec 30
$smolvlaWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/checkpoints/smolvla") -TimeoutSec 30
$checkpointRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/checkpoints") -TimeoutSec 30
$hfHomeWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/hf_home") -TimeoutSec 30
$liberoRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/LIBERO") -TimeoutSec 30
$robosuiteRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/robosuite") -TimeoutSec 30
$pythonProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "test -x $WslPython && $WslPython --version") -TimeoutSec 30
$stdoutLogWin = Join-Path $RepoRoot "runs\online_7d_diagnostic_head\runner_stdout.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $stdoutLogWin) | Out-Null
$stdoutLogWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($stdoutLogWin.Replace("\", "/"))) -TimeoutSec 30

if (-not ($repoWsl.ok -and $manifestWsl.ok -and $jsonReportWsl.ok -and $markdownReportWsl.ok -and $smolvlaWsl.ok -and $checkpointRootWsl.ok -and $hfHomeWsl.ok -and $liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $pythonProbe.ok -and $stdoutLogWsl.ok)) {
    Write-Host "Refusing: WSL path or Python probe failed."
    exit 17
}

$bashCommand = @"
export PYTHONPATH='$($repoWsl.stdout)';
export HF_HUB_OFFLINE=1;
export TRANSFORMERS_OFFLINE=1;
export MUJOCO_GL=osmesa;
export CUDA_VISIBLE_DEVICES='';
export ALLOW_ONLINE_7D_DIAGNOSTIC_HEAD_ROLLOUT=1;
$WslPython -m tca_map.smolvla.online_7d_diagnostic_head --manifest '$($manifestWsl.stdout)' --report-json '$($jsonReportWsl.stdout)' --report-md '$($markdownReportWsl.stdout)' --smolvla-ckpt '$($smolvlaWsl.stdout)' --checkpoint-root '$($checkpointRootWsl.stdout)' --hf-home '$($hfHomeWsl.stdout)' --libero-root '$($liberoRootWsl.stdout)' --robosuite-root '$($robosuiteRootWsl.stdout)' --max-steps $MaxSteps --train-max-steps $TrainMaxSteps --sample-stride $SampleStride --camera-size $CameraSize --device cpu > '$($stdoutLogWsl.stdout)' 2>&1
rc=`$?
echo "online 7D diagnostic head command finished with rc=`$rc"
tail -n 30 '$($stdoutLogWsl.stdout)' || true
exit `$rc
"@
$bashCommand = $bashCommand -replace "`r", ""
$result = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
if (-not $result.ok) {
    Write-Host "Online 7D diagnostic head command failed."
    Write-Host $result.stderr
    Write-Host $result.stdout
    exit 18
}

Write-Host $result.stdout
exit 0
