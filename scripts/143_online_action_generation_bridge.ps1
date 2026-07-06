param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$JsonReportPath = "reports\online_action_generation_bridge_report.json",
    [string]$MarkdownReportPath = "reports\online_action_generation_bridge_report.md",
    [int]$MaxSteps = 25,
    [int]$CameraSize = 64,
    [int]$TimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Online action-generation bridge"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script inventories online action sources and, only under ALLOW_ONLINE_ACTION_BRIDGE_ROLLOUT=1, runs a bounded CPU native online matched-init diagnostic. It does not train, use GPU, download, execute OpenVLA-OFT, run a full benchmark, or make paper claims."

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

if ($MaxSteps -lt 1 -or $MaxSteps -gt 25) {
    Write-Host "Refusing: MaxSteps must be between 1 and 25 for the first online bridge diagnostic."
    exit 12
}
if ($CameraSize -lt 16 -or $CameraSize -gt 128) {
    Write-Host "Refusing: CameraSize must be between 16 and 128."
    exit 13
}

if ([Environment]::GetEnvironmentVariable("ALLOW_ONLINE_ACTION_BRIDGE_ROLLOUT") -ne "1") {
    & $Python -m tca_map.smolvla.online_action_generation_bridge `
        --manifest (Resolve-RepoPath -Path $ManifestPath) `
        --report-json (Resolve-RepoPath -Path $JsonReportPath) `
        --report-md (Resolve-RepoPath -Path $MarkdownReportPath) `
        --max-steps $MaxSteps `
        --camera-size $CameraSize
    exit 0
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    Write-Host "Refusing: WSL is unavailable for simulator diagnosis."
    exit 14
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
$stdoutLogWin = Join-Path $RepoRoot "runs\online_action_generation_bridge\runner_stdout.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $stdoutLogWin) | Out-Null
$stdoutLogWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($stdoutLogWin.Replace("\", "/"))) -TimeoutSec 30

if (-not ($repoWsl.ok -and $manifestWsl.ok -and $jsonReportWsl.ok -and $markdownReportWsl.ok -and $smolvlaWsl.ok -and $checkpointRootWsl.ok -and $hfHomeWsl.ok -and $liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $pythonProbe.ok -and $stdoutLogWsl.ok)) {
    Write-Host "Refusing: WSL path or Python probe failed."
    exit 15
}

$bashCommand = @"
export PYTHONPATH='$($repoWsl.stdout)';
export HF_HUB_OFFLINE=1;
export TRANSFORMERS_OFFLINE=1;
export MUJOCO_GL=osmesa;
export ALLOW_ONLINE_ACTION_BRIDGE_ROLLOUT=1;
export CUDA_VISIBLE_DEVICES='';
$WslPython -m tca_map.smolvla.online_action_generation_bridge --manifest '$($manifestWsl.stdout)' --report-json '$($jsonReportWsl.stdout)' --report-md '$($markdownReportWsl.stdout)' --smolvla-ckpt '$($smolvlaWsl.stdout)' --checkpoint-root '$($checkpointRootWsl.stdout)' --hf-home '$($hfHomeWsl.stdout)' --libero-root '$($liberoRootWsl.stdout)' --robosuite-root '$($robosuiteRootWsl.stdout)' --max-steps $MaxSteps --camera-size $CameraSize --device cpu > '$($stdoutLogWsl.stdout)' 2>&1
rc=`$?
echo "online action-generation bridge command finished with rc=`$rc"
tail -n 20 '$($stdoutLogWsl.stdout)' || true
exit `$rc
"@
$bashCommand = $bashCommand -replace "`r", ""
$result = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
if (-not $result.ok) {
    $jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
    if (Test-Path -LiteralPath $jsonFullPath) {
        try {
            $existingReport = Get-Content -LiteralPath $jsonFullPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ([bool]$existingReport.result.passed) {
                Write-Host "Online action-generation bridge report passed even though the wrapper command did not exit cleanly."
                Write-Host "Report path: $jsonFullPath"
                exit 0
            }
        } catch {}
    }
    Write-Host "Online action-generation bridge command failed."
    Write-Host $result.stderr
    Write-Host $result.stdout
    exit 16
}

Write-Host $result.stdout
exit 0
