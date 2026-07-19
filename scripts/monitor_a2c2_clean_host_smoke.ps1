param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(8, 10, 12, 14)]
    [int]$CapGiB,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z0-9_]+$")]
    [string]$RunId,

    [string]$RepoRoot = "C:\Users\jiheo\tca_map"
)

$ErrorActionPreference = "Stop"
$WslConfig = "C:\Users\jiheo\.wslconfig"
$RunRoot = Join-Path $RepoRoot "runs\a2c2_prior\clean_host_resource_smokes\$RunId"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Get-HostMemorySample {
    $os = Get-CimInstance Win32_OperatingSystem
    $cs = Get-CimInstance Win32_ComputerSystem
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $page = Get-CimInstance Win32_PageFileUsage
    $total = [int64]$cs.TotalPhysicalMemory
    $available = [int64]$os.FreePhysicalMemory * 1KB
    return [ordered]@{
        timestamp = (Get-Date).ToString("o")
        physical_total_bytes = $total
        physical_used_bytes = $total - $available
        physical_available_bytes = $available
        used_fraction = ($total - $available) / $total
        committed_bytes = [int64]$memory.CommittedBytes
        commit_limit_bytes = [int64]$memory.CommitLimit
        page_reads_per_sec = [int64]$memory.PageReadsPersec
        page_writes_per_sec = [int64]$memory.PageWritesPersec
        pages_input_per_sec = [int64]$memory.PagesInputPersec
        pages_output_per_sec = [int64]$memory.PagesOutputPersec
        pagefile_current_usage_mib = ($page | Measure-Object CurrentUsage -Sum).Sum
        pagefile_peak_usage_mib = ($page | Measure-Object PeakUsage -Sum).Sum
        pagefile_allocated_mib = ($page | Measure-Object AllocatedBaseSize -Sum).Sum
    }
}

if (-not (Test-Path -LiteralPath $WslConfig)) {
    throw "Expected temporary .wslconfig is absent"
}
$configText = Get-Content -LiteralPath $WslConfig -Raw
if ($configText -notmatch "memory=${CapGiB}GB") {
    throw ".wslconfig does not request ${CapGiB}GB"
}
if ($configText -notmatch "swap=0") {
    throw ".wslconfig does not disable swap"
}
if ($configText -notmatch "guiApplications=false") {
    throw ".wslconfig does not disable unused WSLg"
}
$configHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $WslConfig).Hash

$staleWorkers = @(Get-Process | Where-Object { $_.ProcessName -in @("python", "python3", "wsl") })
if ($staleWorkers.Count -gt 0) {
    throw "Stale research-capable worker exists before launch: $($staleWorkers.ProcessName -join ',')"
}
$heavyPattern = "League|RiotClient|VALORANT|Shipping|steam|Discord|chrome|Battle.net|Overwatch|Fortnite|Minecraft|Genshin|Roblox|OP.GG|GGQ|OneDrive|Dropbox|GoogleDrive"
$heavy = @(Get-Process | Where-Object { $_.ProcessName -match $heavyPattern } | Select-Object Id, ProcessName)
if ($heavy.Count -gt 0) {
    throw "Background-heavy process became active before launch: $($heavy.ProcessName -join ',')"
}

$baselineFirst = Get-HostMemorySample
Start-Sleep -Seconds 2
$baseline = Get-HostMemorySample
if ([double]$baseline.used_fraction -gt 0.65) {
    throw "A2C2_CLEAN_HOST_BASELINE_UNSAFE: $($baseline.used_fraction)"
}
if ([double]$baseline.pagefile_current_usage_mib -ne [double]$baselineFirst.pagefile_current_usage_mib -or [int64]$baseline.page_writes_per_sec -gt 0) {
    throw "A2C2_CLEAN_HOST_PAGEFILE_UNSTABLE"
}

$stdout = Join-Path $RunRoot "cap_${CapGiB}gb.stdout.log"
$stderr = Join-Path $RunRoot "cap_${CapGiB}gb.stderr.log"
$hostOutput = Join-Path $RunRoot "cap_${CapGiB}gb_host.json"
$internalJsonRelative = "reports/a2c2_prior/${RunId}_resource_smoke_cap_${CapGiB}gb_internal.json"
$internalMdRelative = "reports/a2c2_prior/${RunId}_resource_smoke_cap_${CapGiB}gb_internal.md"
$internalOutput = Join-Path $RepoRoot ($internalJsonRelative -replace "/", "\")
if (Test-Path -LiteralPath $internalOutput) {
    throw "Stale internal clean-host smoke report must be preserved and not overwritten: $internalOutput"
}
if (Test-Path -LiteralPath $hostOutput) {
    throw "Stale host clean-host smoke report must be preserved and not overwritten: $hostOutput"
}

$argumentList = @(
    "-d", "Ubuntu-22.04", "--", "bash",
    "/mnt/c/Users/jiheo/tca_map/scripts/run_a2c2_clean_host_smoke_wsl.sh",
    "$CapGiB", "$configHash", "$internalJsonRelative", "$internalMdRelative"
)
$process = Start-Process -FilePath "wsl.exe" -ArgumentList $argumentList -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr

$samples = New-Object System.Collections.Generic.List[object]
$peakUsedFraction = [double]$baseline.used_fraction
$peakCommittedBytes = [int64]$baseline.committed_bytes
$peakPagefileCurrentMiB = [double]$baseline.pagefile_current_usage_mib
$peakPageWritesPerSec = [int64]$baseline.page_writes_per_sec
$hostCeilingTerminated = $false

while (-not $process.HasExited) {
    $sample = Get-HostMemorySample
    $samples.Add($sample)
    $peakUsedFraction = [math]::Max($peakUsedFraction, [double]$sample.used_fraction)
    $peakCommittedBytes = [math]::Max($peakCommittedBytes, [int64]$sample.committed_bytes)
    $peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$sample.pagefile_current_usage_mib)
    $peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$sample.page_writes_per_sec)
    if ([double]$sample.used_fraction -gt 0.82) {
        $hostCeilingTerminated = $true
        & wsl.exe --terminate Ubuntu-22.04 | Out-Null
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        break
    }
    Start-Sleep -Milliseconds 500
    $process.Refresh()
}
$process.WaitForExit()
$process.Refresh()
$rawExitCode = $process.ExitCode
$afterChild = Get-HostMemorySample
$samples.Add($afterChild)

$internal = $null
if (Test-Path -LiteralPath $internalOutput) {
    $internal = Get-Content -LiteralPath $internalOutput -Raw | ConvertFrom-Json
}
if ($hostCeilingTerminated) {
    $meminfo = @("unavailable_after_forced_distro_terminate")
    $dmesg = @("unavailable_after_forced_distro_terminate")
} else {
    $meminfo = & wsl.exe -d Ubuntu-22.04 -- bash -lc "awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo" 2>&1
    $dmesg = & wsl.exe -d Ubuntu-22.04 -- bash -lc "dmesg --color=never 2>/dev/null | grep -Ei 'out of memory|oom-kill|killed process' | tail -30 || true" 2>&1
}
& wsl.exe --shutdown
Start-Sleep -Seconds 3
$postShutdown = Get-HostMemorySample
$samples.Add($postShutdown)

$peakUsedFraction = [math]::Max($peakUsedFraction, [double]$afterChild.used_fraction)
$peakUsedFraction = [math]::Max($peakUsedFraction, [double]$postShutdown.used_fraction)
$peakCommittedBytes = [math]::Max($peakCommittedBytes, [int64]$afterChild.committed_bytes)
$peakCommittedBytes = [math]::Max($peakCommittedBytes, [int64]$postShutdown.committed_bytes)
$peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$afterChild.pagefile_current_usage_mib)
$peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$postShutdown.pagefile_current_usage_mib)
$peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$afterChild.page_writes_per_sec)
$peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$postShutdown.page_writes_per_sec)
$pagefileGrowthMiB = $peakPagefileCurrentMiB - [double]$baseline.pagefile_current_usage_mib
$memoryReleaseVerified = [double]$postShutdown.used_fraction -le ([double]$baseline.used_fraction + 0.05)
$internalPeakFraction = if ($null -ne $internal) { [double]($internal.peak_resources.peak_wsl_used_fraction) } else { 0.0 }
$internalOomLines = if ($null -ne $internal) { @($internal.new_kernel_oom_lines) } else { @() }
$internalPass = [bool]($null -ne $internal -and $internal.internal_pass)
$offloadVerified = [bool]($null -ne $internal -and $internal.no_cpu_or_disk_model_offload)
$swapZero = [bool]($null -ne $internal -and [int64]$internal.meminfo_after.swap_total_bytes -eq 0)

if ($hostCeilingTerminated) {
    $decision = "A2C2_RESOURCE_SMOKE_FAIL_WINDOWS_CEILING"
} elseif ($internalOomLines.Count -gt 0) {
    $decision = "A2C2_RESOURCE_SMOKE_FAIL_KERNEL_OOM"
} elseif ($null -ne $internal -and $internalPeakFraction -gt 0.95 -and -not $internalPass) {
    $decision = "A2C2_RESOURCE_SMOKE_FAIL_CAP_TOO_LOW"
} elseif (-not $memoryReleaseVerified -or $pagefileGrowthMiB -gt 0) {
    $decision = "A2C2_RESOURCE_SMOKE_FAIL_MEMORY_LEAK"
} elseif ($internalPass -and $offloadVerified -and $swapZero -and $peakUsedFraction -le 0.82) {
    $decision = "A2C2_RESOURCE_SMOKE_PASS"
} else {
    $decision = "A2C2_RESOURCE_SMOKE_FAIL_UNRELATED_IMPLEMENTATION"
}

$payload = [ordered]@{
    schema_version = 1
    date_kst = (Get-Date).ToString("o")
    authority_state = "A2C2_CLEAN_HOST_RESOURCE_FEASIBILITY_REOPENED"
    execution_type = "VLA_INFERENCE"
    purpose = "CLEAN_HOST_RESOURCE_ONLY_ACTUAL_PATH_SMOKE_HOST_MONITOR"
    run_id = $RunId
    cap_gib = $CapGiB
    wslconfig_sha256 = $configHash
    command_pid = $process.Id
    command_exit_code_raw = $rawExitCode
    host_ceiling_terminated = $hostCeilingTerminated
    concurrency = [ordered]@{
        policy_processes = 1
        policy_instances = 1
        libero_environments = 1
        eval_batch_size = 1
        parallel_tasks = 1
        telemetry_monitors = 1
    }
    prefetch = [ordered]@{
        model_checkpoint = $false
        training_data = $false
        environment = $false
        next_task = $false
        video = $false
        observation_history = $false
    }
    model_residency_count = 1
    baseline_first = $baselineFirst
    baseline = $baseline
    peak = [ordered]@{
        used_fraction = $peakUsedFraction
        committed_bytes = $peakCommittedBytes
        pagefile_current_usage_mib = $peakPagefileCurrentMiB
        page_writes_per_sec = $peakPageWritesPerSec
    }
    after_child = $afterChild
    post_wsl_shutdown = $postShutdown
    pagefile_current_growth_mib = $pagefileGrowthMiB
    memory_release_verified = $memoryReleaseVerified
    sample_count = $samples.Count
    samples = $samples
    wsl_meminfo = @($meminfo)
    kernel_oom_lines = @($dmesg)
    internal_report_path = $internalJsonRelative
    internal_decision = if ($null -ne $internal) { $internal.final_decision } else { $null }
    internal_pass = $internalPass
    no_cpu_or_disk_model_offload = $offloadVerified
    swap_zero = $swapZero
    stdout_path = $stdout
    stderr_path = $stderr
    scientific_outcome_persisted_or_counted = $false
    final_decision = $decision
}
$payload | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $hostOutput -Encoding utf8
$payload | ConvertTo-Json -Depth 9

if ($decision -eq "A2C2_RESOURCE_SMOKE_PASS") {
    exit 0
}
if ($decision -eq "A2C2_RESOURCE_SMOKE_FAIL_WINDOWS_CEILING") {
    exit 82
}
exit 1
