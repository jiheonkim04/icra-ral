param(
    [string]$Repo = "C:\Users\jiheo\tca_map",
    [string]$Distribution = "Ubuntu-22.04",
    [string]$LinuxRepo = "/mnt/c/Users/jiheo/tca_map",
    [string]$LinuxPython = "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"
)

$ErrorActionPreference = "Stop"
$runDir = Join-Path $Repo "runs\epoch10b_manifest_erratum"
$attemptId = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $runDir ("runner_{0}_stdout.log" -f $attemptId)
$stderrPath = Join-Path $runDir ("runner_{0}_stderr.log" -f $attemptId)
$monitorPath = Join-Path $runDir "host_monitor.json"
$originalRawPath = Join-Path $Repo "runs\epoch10b_mechanics_certification\branches.jsonl"
$panelStatePath = Join-Path $runDir "panel_state.json"
$expectedOriginalRawHash = "A2F2992D03FAE52177408F057BA311B4F522B3955D91BA78DCD74A165E55CED7"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Get-HostRamPercent {
    $os = Get-CimInstance Win32_OperatingSystem
    return [math]::Round((1.0 - ([double]$os.FreePhysicalMemory / [double]$os.TotalVisibleMemorySize)) * 100.0, 4)
}

function Get-ProtectedLedger([string]$Root) {
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root)
    $files = @(Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File | Sort-Object FullName)
    $lines = foreach ($file in $files) {
        $rootPrefix = $resolvedRoot.TrimEnd('\') + '\'
        $relative = $file.FullName.Substring($rootPrefix.Length).Replace('\', '/')
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
        "$relative|$hash"
    }
    $payload = if ($lines.Count -gt 0) { ($lines -join "`n") + "`n" } else { "" }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $ledgerHash = [System.BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
    return [ordered]@{
        path = $resolvedRoot
        file_count = $files.Count
        bytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
        ledger_sha256 = $ledgerHash
    }
}

function Get-WslMemory {
    $text = (& wsl.exe -d $Distribution -e cat /proc/meminfo 2>$null) -join "`n"
    $values = @{}
    foreach ($name in @('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')) {
        if ($text -match "(?m)^${name}:\s+(\d+)\s+kB") {
            $values[$name] = [int64]$Matches[1] * 1024
        }
        else {
            $values[$name] = 0
        }
    }
    return [ordered]@{
        memory_used_bytes = [math]::Max([int64]0, [int64]($values.MemTotal - $values.MemAvailable))
        swap_used_bytes = [math]::Max([int64]0, [int64]($values.SwapTotal - $values.SwapFree))
    }
}

function Get-GpuSample {
    $raw = (& nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    $parts = @($raw -split ',' | ForEach-Object { $_.Trim() })
    return [ordered]@{
        memory_used_mib = if ($parts.Count -ge 1) { [double]$parts[0] } else { 0.0 }
        utilization_percent = if ($parts.Count -ge 2) { [double]$parts[1] } else { 0.0 }
        temperature_c = if ($parts.Count -ge 3) { [double]$parts[2] } else { 0.0 }
        power_w = if ($parts.Count -ge 4) { [double]$parts[3] } else { 0.0 }
    }
}

$startTime = Get-Date
$baselineRam = Get-HostRamPercent
if ($baselineRam -ge 80.0) {
    throw "host RAM baseline is not below the frozen 80% soft threshold: $baselineRam"
}

$runningWsl = (& wsl.exe --list --running 2>$null) -join ""
if ($runningWsl -match 'Ubuntu-22.04') {
    throw "Ubuntu-22.04 was already running before the bounded erratum panel"
}

$blockedProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and (
        $_.Name -match '^(python|python3|mujoco|robosuite|libero|steam|eldenring|Cyberpunk2077)\.exe$'
    )
})
if ($blockedProcesses.Count -gt 0) {
    throw "an unrelated scientific/game process is active before launch"
}

$computeAppsRaw = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null | Where-Object { $_.Trim() })
$computeApps = @($computeAppsRaw | Where-Object { $_ -match '(?i)python|wsl|libero|mujoco|robosuite|steam|game' })
if ($computeApps.Count -gt 0) {
    throw "an unrelated GPU compute process is active before launch"
}

$protectedBefore = [ordered]@{
    rollouts_2026_07_17 = Get-ProtectedLedger (Join-Path $Repo "rollouts\2026_07_17")
    rollouts_2026_07_18 = Get-ProtectedLedger (Join-Path $Repo "rollouts\2026_07_18")
    adapter_panel_rank4 = Get-ProtectedLedger "C:\assets\checkpoints\epoch10_icae_panel\rank4"
}

$preflight = & "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe" (Join-Path $Repo "scripts\run_epoch10b_manifest_erratum.py") preflight
if ($LASTEXITCODE -ne 0 -or $preflight -notmatch 'ERRATUM_PREFLIGHT_PASS') {
    throw "erratum frozen-input preflight failed"
}

$samples = [System.Collections.Generic.List[object]]::new()
$softWarning = $false
$controlledStop = $false
$runnerExit = $null
$process = $null
try {
    $arguments = @(
        '-d', $Distribution,
        '--cd', $LinuxRepo,
        '-e', '/usr/bin/env',
        "MUJOCO_GL=egl",
        "PYTHONPATH=$LinuxRepo",
        $LinuxPython,
        'scripts/run_epoch10b_manifest_erratum.py',
        'run-panel'
    )
    $process = Start-Process -FilePath "wsl.exe" -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    while (-not $process.HasExited) {
        $ram = Get-HostRamPercent
        $wsl = Get-WslMemory
        $gpu = Get-GpuSample
        $samples.Add([ordered]@{
            timestamp = (Get-Date).ToString('o')
            host_ram_percent = $ram
            wsl_memory_used_bytes = $wsl.memory_used_bytes
            wsl_swap_used_bytes = $wsl.swap_used_bytes
            gpu_memory_used_mib = $gpu.memory_used_mib
            gpu_utilization_percent = $gpu.utilization_percent
            gpu_temperature_c = $gpu.temperature_c
            gpu_power_w = $gpu.power_w
        })
        if ($ram -ge 80.0) {
            $softWarning = $true
            $controlledStop = $true
            & wsl.exe --shutdown 2>$null | Out-Null
            break
        }
        Start-Sleep -Seconds 1
        $process.Refresh()
    }
    $process.WaitForExit()
    $process.Refresh()
    $runnerExit = [int]$process.ExitCode
}
finally {
    & wsl.exe --shutdown 2>$null | Out-Null
}

Start-Sleep -Seconds 2
$protectedAfter = [ordered]@{
    rollouts_2026_07_17 = Get-ProtectedLedger (Join-Path $Repo "rollouts\2026_07_17")
    rollouts_2026_07_18 = Get-ProtectedLedger (Join-Path $Repo "rollouts\2026_07_18")
    adapter_panel_rank4 = Get-ProtectedLedger "C:\assets\checkpoints\epoch10_icae_panel\rank4"
}
$protectedMatch = (($protectedBefore | ConvertTo-Json -Depth 8 -Compress) -eq ($protectedAfter | ConvertTo-Json -Depth 8 -Compress))
$originalRawHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $originalRawPath).Hash
$panelState = if (Test-Path -LiteralPath $panelStatePath) { Get-Content -Raw -LiteralPath $panelStatePath | ConvertFrom-Json } else { $null }
$erratumRawPath = Join-Path $runDir "frame60_primary_panel.jsonl"
$rowResourceSamples = [System.Collections.Generic.List[object]]::new()
if (Test-Path -LiteralPath $erratumRawPath) {
    $rowNumber = 0
    foreach ($line in Get-Content -LiteralPath $erratumRawPath) {
        $rowNumber += 1
        $row = $line | ConvertFrom-Json
        foreach ($label in @('resource_before', 'resource_after')) {
            $sample = $row.$label
            $rowResourceSamples.Add([ordered]@{
                row_number = $rowNumber
                branch_key = $row.branch_key
                sample = $label
                host_ram_percent = [double]$sample.host_ram_percent
                wsl_memory_used_bytes = [int64]$sample.wsl_mem_total_bytes - [int64]$sample.wsl_mem_available_bytes
                wsl_swap_used_bytes = [int64]$sample.wsl_swap_used_bytes
                gpu_memory_used_mib = [double]$sample.gpu_vram_used_mib
            })
        }
    }
}

$scientificWindowStart = if (Test-Path -LiteralPath $erratumRawPath) { (Get-Item -LiteralPath $erratumRawPath).CreationTime } else { $startTime }
$events = @(Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$scientificWindowStart; Level=1,2} -ErrorAction SilentlyContinue | Where-Object {
    $_.ProviderName -match 'Ntfs|disk|WHEA|Kernel-Power'
})
$ramValues = @([double]$baselineRam) + @($samples | ForEach-Object { [double]$_['host_ram_percent'] }) + @($rowResourceSamples | ForEach-Object { [double]$_['host_ram_percent'] })
$swapValues = @([int64]0) + @($samples | ForEach-Object { [int64]$_['wsl_swap_used_bytes'] }) + @($rowResourceSamples | ForEach-Object { [int64]$_['wsl_swap_used_bytes'] })
$wslValues = @([int64]0) + @($samples | ForEach-Object { [int64]$_['wsl_memory_used_bytes'] }) + @($rowResourceSamples | ForEach-Object { [int64]$_['wsl_memory_used_bytes'] })
$vramValues = @([double]0) + @($samples | ForEach-Object { [double]$_['gpu_memory_used_mib'] }) + @($rowResourceSamples | ForEach-Object { [double]$_['gpu_memory_used_mib'] })
$peakRam = ($ramValues | Measure-Object -Maximum).Maximum
$peakSwap = ($swapValues | Measure-Object -Maximum).Maximum
$peakWsl = ($wslValues | Measure-Object -Maximum).Maximum
$peakVram = ($vramValues | Measure-Object -Maximum).Maximum
$pass = [bool](
    $runnerExit -eq 0 -and
    -not $controlledStop -and
    $peakRam -lt 80.0 -and
    $peakSwap -eq 0 -and
    $protectedMatch -and
    $originalRawHash -eq $expectedOriginalRawHash -and
    $events.Count -eq 0 -and
    $null -ne $panelState -and
    $panelState.status -eq 'ERRATUM_PANEL_COMPLETE' -and
    $panelState.all_valid_finite_unique
)

$monitor = [ordered]@{
    schema_version = 1
    campaign = 'epoch10b_manifest_adjudicator_erratum'
    started_at = $startTime.ToString('o')
    ended_at = (Get-Date).ToString('o')
    pass = $pass
    runner_exit = $runnerExit
    frozen_soft_host_ram_percent = 80.0
    immediate_stop_before_percent = 90.0
    baseline_host_ram_percent = $baselineRam
    peak_host_ram_percent = $peakRam
    soft_warning_crossed = $softWarning
    controlled_stop = $controlledStop
    peak_wsl_memory_used_bytes = [int64]$peakWsl
    peak_wsl_swap_used_bytes = [int64]$peakSwap
    peak_gpu_vram_used_mib = [double]$peakVram
    wsl_shutdown_unconditional = $true
    original_raw_sha256 = $originalRawHash.ToLowerInvariant()
    original_raw_unchanged = ($originalRawHash -eq $expectedOriginalRawHash)
    protected_manifests_before = $protectedBefore
    protected_manifests_after = $protectedAfter
    protected_manifests_unchanged = $protectedMatch
    material_system_event_count = $events.Count
    material_system_events = @($events | Select-Object TimeCreated, ProviderName, Id, LevelDisplayName, Message)
    scientific_event_window_started_at = $scientificWindowStart.ToString('o')
    preflight_compute_apps_query = $computeAppsRaw
    preflight_unrelated_compute_apps = $computeApps
    panel_state_path = 'runs/epoch10b_manifest_erratum/panel_state.json'
    panel_complete = [bool]($null -ne $panelState -and $panelState.status -eq 'ERRATUM_PANEL_COMPLETE')
    stdout_path = $stdoutPath.Substring($Repo.Length + 1).Replace('\', '/')
    stderr_path = $stderrPath.Substring($Repo.Length + 1).Replace('\', '/')
    wrapper_samples = $samples
    scientific_row_resource_samples = $rowResourceSamples
}
$monitorJson = $monitor | ConvertTo-Json -Depth 12
[System.IO.File]::WriteAllText($monitorPath, $monitorJson + "`n", [System.Text.UTF8Encoding]::new($false))

if (-not $pass) {
    throw "Epoch 10B erratum host guard failed; inspect $monitorPath"
}
Write-Output ("EPOCH10B_ERRATUM_PANEL_PASS rows={0} peak_ram={1} swap={2} monitor={3}" -f $panelState.row_count, $peakRam, $peakSwap, $monitorPath)
