param(
    [ValidateSet("interventions", "rollouts")]
    [string]$Mode = "interventions",
    [string]$Distribution = "Ubuntu-22.04",
    [string]$Repo = "C:\Users\jiheo\tca_map",
    [int]$InterventionBatchSize = 60,
    [int]$RolloutBlockBatchSize = 4
)

$ErrorActionPreference = "Stop"
$linuxRepo = "/mnt/c/Users/jiheo/tca_map"
$linuxPython = "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
$runDir = if ($Mode -eq "interventions") {
    Join-Path $Repo "runs\epoch10b_stage0_interventions"
} else {
    Join-Path $Repo "runs\epoch10b_stage0_rollouts"
}
$completionManifest = if ($Mode -eq "interventions") {
    Join-Path $Repo "reports\epoch10b_stage0_intervention_manifest.json"
} else {
    Join-Path $Repo "reports\epoch10b_stage0_rollout_manifest.json"
}
$rawPath = if ($Mode -eq "interventions") {
    Join-Path $runDir "branches.jsonl"
} else {
    Join-Path $runDir "development_episodes.jsonl"
}
$monitorPath = Join-Path $runDir "host_monitor.json"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Get-HostRamPercent {
    Add-Type -AssemblyName Microsoft.VisualBasic
    $computer = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
    return [math]::Round((1.0 - [double]$computer.AvailablePhysicalMemory / [double]$computer.TotalPhysicalMemory) * 100.0, 4)
}

function Get-RawRowCount {
    if (-not (Test-Path -LiteralPath $rawPath)) { return 0 }
    return (Get-Content -LiteralPath $rawPath | Measure-Object -Line).Lines
}

function Test-Complete {
    if (-not (Test-Path -LiteralPath $completionManifest)) { return $false }
    try {
        $manifest = Get-Content -Raw -LiteralPath $completionManifest | ConvertFrom-Json
        if ($Mode -eq "interventions") {
            return $manifest.status -eq "EPOCH10B_STAGE0_DEVELOPMENT_INTERVENTIONS_COMPLETE"
        }
        return $manifest.status -eq "EPOCH10B_STAGE0_DEVELOPMENT_ROLLOUTS_COMPLETE"
    }
    catch { return $false }
}

$existingWorker = wsl.exe -d $Distribution -e bash -lc "pgrep -af '[r]un_epoch10b_stage0.py' || true" 2>$null
if ($existingWorker) {
    throw "duplicate Epoch 10B Stage 0 worker detected: $existingWorker"
}
$forbiddenNames = @("eldenring", "Cyberpunk2077", "steam", "python", "python3")
$forbidden = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $forbiddenNames -contains $_.ProcessName })
if ($forbidden.Count -gt 0) {
    throw "conflicting game or scientific worker detected: $($forbidden.ProcessName -join ', ')"
}

$campaignStarted = Get-Date
$campaignPeak = 0.0
$softCrossed = $false
$hardCrossed = $false
$batchRecords = @()
$terminalRunnerExit = $null
while (-not (Test-Complete)) {
    wsl.exe --terminate $Distribution 2>$null | Out-Null
    $recoveryDeadline = (Get-Date).AddSeconds(90)
    do {
        Start-Sleep -Seconds 3
        $baseline = Get-HostRamPercent
    } while ($baseline -ge 80.0 -and (Get-Date) -lt $recoveryDeadline)
    if ($baseline -ge 80.0) {
        throw "clean-restart host RAM baseline remained at or above 80%: $baseline%"
    }

    $beforeCount = Get-RawRowCount
    $batchStarted = Get-Date
    $monitorJob = Start-Job -ArgumentList $Distribution -ScriptBlock {
        param($DistributionName)
        Add-Type -AssemblyName Microsoft.VisualBasic
        $computer = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
        while ($true) {
            $used = [math]::Round((1.0 - [double]$computer.AvailablePhysicalMemory / [double]$computer.TotalPhysicalMemory) * 100.0, 4)
            [PSCustomObject]@{ Timestamp = (Get-Date).ToString("o"); UsedPercent = $used }
            if ($used -ge 90.0) {
                wsl.exe --terminate $DistributionName 2>$null | Out-Null
                break
            }
            Start-Sleep -Seconds 2
        }
    }
    try {
        if ($Mode -eq "interventions") {
            wsl.exe -d $Distribution -e bash -lc "cd $linuxRepo && export MUJOCO_GL=egl PYTHONPATH=$linuxRepo && $linuxPython scripts/run_epoch10b_stage0.py interventions --max-new-branches $InterventionBatchSize"
        } else {
            wsl.exe -d $Distribution -e bash -lc "cd $linuxRepo && export MUJOCO_GL=egl PYTHONPATH=$linuxRepo HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false && $linuxPython scripts/run_epoch10b_stage0.py rollouts --max-new-rollout-blocks $RolloutBlockBatchSize"
        }
        $runnerExit = $LASTEXITCODE
    }
    finally {
        Stop-Job -Job $monitorJob -ErrorAction SilentlyContinue
        $samples = @(Receive-Job -Job $monitorJob -ErrorAction SilentlyContinue | Where-Object { $_.PSObject.Properties.Name -contains "UsedPercent" })
        Remove-Job -Job $monitorJob -Force -ErrorAction SilentlyContinue
    }
    $batchPeak = if ($samples.Count -gt 0) { ($samples | Measure-Object -Property UsedPercent -Maximum).Maximum } else { Get-HostRamPercent }
    $campaignPeak = [math]::Max($campaignPeak, [double]$batchPeak)
    if ($batchPeak -ge 80.0) { $softCrossed = $true }
    if ($batchPeak -ge 90.0) { $hardCrossed = $true }
    $afterCount = Get-RawRowCount
    $batchRecords += [ordered]@{
        batch_started_at = $batchStarted.ToString("o")
        baseline_host_ram_percent = $baseline
        peak_host_ram_percent = $batchPeak
        rows_before = $beforeCount
        rows_after = $afterCount
        runner_exit = $runnerExit
    }
    Write-Output ("epoch10b stage0 {0} batch: rows={1}, added={2}, baseline={3}%, peak={4}%, exit={5}" -f $Mode, $afterCount, ($afterCount - $beforeCount), $baseline, $batchPeak, $runnerExit)
    wsl.exe --terminate $Distribution 2>$null | Out-Null
    if ($hardCrossed) { $terminalRunnerExit = 90; break }
    if ($runnerExit -ne 0) { $terminalRunnerExit = $runnerExit; break }
    if ($afterCount -le $beforeCount -and -not (Test-Complete)) {
        $terminalRunnerExit = 5
        break
    }
}

$monitor = [ordered]@{
    schema_version = 1
    campaign = "epoch10b_icae_fresh_controller"
    mode = $Mode
    started_at = $campaignStarted.ToString("o")
    ended_at = (Get-Date).ToString("o")
    peak_host_ram_percent = $campaignPeak
    soft_warning_crossed = $softCrossed
    hard_stop_crossed = $hardCrossed
    terminal_runner_exit = $terminalRunnerExit
    completion_manifest = $completionManifest
    complete = Test-Complete
    batches = $batchRecords
}
$monitorJson = ($monitor | ConvertTo-Json -Depth 8) + [Environment]::NewLine
[System.IO.File]::WriteAllText($monitorPath, $monitorJson, [System.Text.UTF8Encoding]::new($false))
wsl.exe --terminate $Distribution 2>$null | Out-Null
if ($hardCrossed) { exit 90 }
if ($null -ne $terminalRunnerExit) { exit $terminalRunnerExit }
exit 0
