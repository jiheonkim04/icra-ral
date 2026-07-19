param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("smoke", "panel")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-zA-Z0-9_]+$")]
    [string]$RunId,

    [string]$RepoRoot = "C:\Users\jiheo\tca_map"
)

$ErrorActionPreference = "Stop"
$WslConfig = "C:\Users\jiheo\.wslconfig"
$RunRoot = Join-Path $RepoRoot "runs\a2c2_fidelity_corrected\$RunId"
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
if ($configText -notmatch "memory=12GB" -or $configText -notmatch "swap=0" -or $configText -notmatch "guiApplications=false") {
    throw "Temporary .wslconfig does not match frozen 12GB/swap0/no-WSLg contract"
}
$configHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $WslConfig).Hash

$staleWorkers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "^(python|python3|wsl)\.exe$" -and $_.CommandLine -match "a2c2_fidelity_corrected"
})
if ($staleWorkers.Count -gt 0) {
    throw "Stale corrected-A2C2 worker exists before launch"
}
$competingResearch = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "^(python|python3)\.exe$" -and $_.CommandLine -match "(run_|train_|rollout|smolvla|libero)"
})
if ($competingResearch.Count -gt 0) {
    throw "Competing research worker exists before launch"
}
$gamePattern = "League|RiotClient|VALORANT|Shipping|steam|Battle.net|Overwatch|Fortnite|Minecraft|Genshin|Roblox"
$activeGames = @(Get-Process | Where-Object { $_.ProcessName -match $gamePattern } | Select-Object Id, ProcessName)
if ($activeGames.Count -gt 0) {
    throw "Active game/heavy interactive process exists: $($activeGames.ProcessName -join ',')"
}
$backgroundHeavy = @(Get-Process | Where-Object {
    $_.ProcessName -match "chrome|Discord|OP.GG|GGQ|OneDrive|Dropbox|GoogleDrive"
} | Select-Object Id, ProcessName, @{Name="working_set_mib"; Expression={[math]::Round($_.WorkingSet64 / 1MB, 3)}})

$baselineFirst = Get-HostMemorySample
Start-Sleep -Seconds 2
$baseline = Get-HostMemorySample
if ([double]$baseline.used_fraction -gt 0.65) {
    throw "A2C2_CORRECTED_CLEAN_HOST_BASELINE_UNSAFE: $($baseline.used_fraction)"
}
if ([double]$baseline.pagefile_current_usage_mib -ne [double]$baselineFirst.pagefile_current_usage_mib) {
    throw "A2C2_CORRECTED_PAGEFILE_BASELINE_UNSTABLE"
}

$stdout = Join-Path $RunRoot "$Mode.stdout.log"
$stderr = Join-Path $RunRoot "$Mode.stderr.log"
$hostOutput = Join-Path $RunRoot "host_monitor.json"
$internalOutput = Join-Path $RunRoot "result.json"
if (Test-Path -LiteralPath $hostOutput) {
    throw "Host monitor output already exists and will not be overwritten: $hostOutput"
}
if (Test-Path -LiteralPath $internalOutput) {
    throw "Internal output already exists and will not be overwritten: $internalOutput"
}

$argumentList = @(
    "-d", "Ubuntu-22.04", "--", "bash",
    "/mnt/c/Users/jiheo/tca_map/scripts/run_a2c2_fidelity_corrected_wsl.sh",
    $Mode, $RunId
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
$peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$afterChild.pagefile_current_usage_mib)
$peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$afterChild.page_writes_per_sec)
$pagefileGrowthMiB = $peakPagefileCurrentMiB - [double]$baseline.pagefile_current_usage_mib
$memoryReleaseVerified = [double]$postShutdown.used_fraction -le ([double]$baseline.used_fraction + 0.05)
$swapZero = [bool]($null -ne $internal -and [int64]$internal.swap_total_bytes_at_end -eq 0)
$offloadVerified = [bool]($null -ne $internal -and $internal.model_load_audit.no_cpu_or_disk_offload)
$internalDecision = if ($null -ne $internal) { [string]$internal.final_decision } else { $null }
$smokePass = $Mode -eq "smoke" -and $internalDecision -eq "A2C2_CORRECTED_ACTUAL_PATH_SMOKE_PASS"
$panelAllowed = @(
    "CORRECTED_A2C2_PRIOR_IMPROVES_AND_LEAVES_RESIDUAL",
    "CORRECTED_A2C2_PRIOR_SATURATES_DELAY",
    "CORRECTED_A2C2_PRIOR_NO_IMPROVEMENT",
    "CORRECTED_A2C2_BASE_NOT_COMPETENT",
    "CORRECTED_A2C2_EVALUATION_INVALID"
)
$panelComplete = $Mode -eq "panel" -and $null -ne $internal -and [int]$internal.completed_scientific_rows -eq 45 -and $internalDecision -in $panelAllowed

if ($hostCeilingTerminated) {
    $decision = "A2C2_CORRECTED_HOST_FAIL_WINDOWS_CEILING"
} elseif (@($dmesg | Where-Object { $_ -match "out of memory|oom-kill|killed process" }).Count -gt 0) {
    $decision = "A2C2_CORRECTED_HOST_FAIL_KERNEL_OOM"
} elseif (-not $memoryReleaseVerified -or $pagefileGrowthMiB -gt 0) {
    $decision = "A2C2_CORRECTED_HOST_FAIL_MEMORY_OR_PAGEFILE"
} elseif (($smokePass -or $panelComplete) -and $offloadVerified -and $swapZero -and $peakUsedFraction -le 0.82) {
    $decision = if ($Mode -eq "smoke") { "A2C2_CORRECTED_HOST_SMOKE_PASS" } else { "A2C2_CORRECTED_HOST_PANEL_PASS" }
} else {
    $decision = "A2C2_CORRECTED_HOST_FAIL_IMPLEMENTATION_OR_EXECUTION"
}

$payload = [ordered]@{
    schema_version = 1
    date_kst = (Get-Date).ToString("o")
    mode = $Mode
    run_id = $RunId
    fidelity_label = "A2C2_FIDELITY_CORRECTED_LOCAL_PORT"
    wsl_cap_gib = 12
    wslconfig_sha256 = $configHash
    command_pid = $process.Id
    command_exit_code_raw = $rawExitCode
    host_ceiling_terminated = $hostCeilingTerminated
    active_games = $activeGames
    recorded_background_heavy_processes = $backgroundHeavy
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
    internal_result_path = $internalOutput
    internal_decision = $internalDecision
    no_cpu_or_disk_model_offload = $offloadVerified
    swap_zero = $swapZero
    stdout_path = $stdout
    stderr_path = $stderr
    scientific_outcome_persisted_or_counted = [bool]($Mode -eq "panel")
    final_decision = $decision
}
$payload | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $hostOutput -Encoding UTF8
$payload | ConvertTo-Json -Depth 9

if ($decision -in @("A2C2_CORRECTED_HOST_SMOKE_PASS", "A2C2_CORRECTED_HOST_PANEL_PASS")) {
    exit 0
}
if ($decision -eq "A2C2_CORRECTED_HOST_FAIL_WINDOWS_CEILING") {
    exit 82
}
exit 1
