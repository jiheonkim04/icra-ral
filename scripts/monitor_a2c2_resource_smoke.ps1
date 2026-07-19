param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(6, 8, 10, 12, 14)]
    [int]$CapGiB,

    [string]$RepoRoot = "C:\Users\jiheo\tca_map"
)

$ErrorActionPreference = "Stop"
$WslConfig = "C:\Users\jiheo\.wslconfig"
$RunRoot = Join-Path $RepoRoot "runs\a2c2_prior\resource_smokes"
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
$configHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $WslConfig).Hash
$baseline = Get-HostMemorySample
if ([double]$baseline.used_fraction -gt 0.70) {
    throw "A2C2_HOST_MEMORY_BASELINE_UNSAFE: $($baseline.used_fraction)"
}

$gamePattern = "League|RiotClient|VALORANT|Shipping|steam|Battle.net|Overwatch|Fortnite|Minecraft|Genshin|Roblox|elden|cyberpunk|witcher|dota|cs2"
$games = @(Get-Process | Where-Object { $_.ProcessName -match $gamePattern } | Select-Object Id, ProcessName)
if ($games.Count -gt 0) {
    throw "Game process became active before launch: $($games.ProcessName -join ',')"
}

$stdout = Join-Path $RunRoot "cap_${CapGiB}gb.stdout.log"
$stderr = Join-Path $RunRoot "cap_${CapGiB}gb.stderr.log"
$internalOutput = Join-Path $RepoRoot "reports\a2c2_prior\resource_smoke_cap_${CapGiB}gb_internal.json"
if (Test-Path -LiteralPath $internalOutput) {
    throw "Stale internal smoke report must be archived before launch: $internalOutput"
}
$argumentList = @(
    "-d", "Ubuntu-22.04", "--", "bash",
    "/mnt/c/Users/jiheo/tca_map/scripts/run_a2c2_resource_smoke_wsl.sh",
    "$CapGiB", "$configHash"
)
$process = Start-Process -FilePath "wsl.exe" -ArgumentList $argumentList -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr

$samples = New-Object System.Collections.Generic.List[object]
$peakUsedFraction = [double]$baseline.used_fraction
$peakCommittedBytes = [int64]$baseline.committed_bytes
$peakPagefileCurrentMiB = [double]$baseline.pagefile_current_usage_mib
$peakPageReadsPerSec = [int64]$baseline.page_reads_per_sec
$peakPageWritesPerSec = [int64]$baseline.page_writes_per_sec
$hostCeilingTerminated = $false

while (-not $process.HasExited) {
    $sample = Get-HostMemorySample
    $samples.Add($sample)
    $peakUsedFraction = [math]::Max($peakUsedFraction, [double]$sample.used_fraction)
    $peakCommittedBytes = [math]::Max($peakCommittedBytes, [int64]$sample.committed_bytes)
    $peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$sample.pagefile_current_usage_mib)
    $peakPageReadsPerSec = [math]::Max($peakPageReadsPerSec, [int64]$sample.page_reads_per_sec)
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
if ($hostCeilingTerminated) {
    $exitCode = 82
    $exitCodeSource = "host_ceiling_guard"
} elseif ($null -ne $rawExitCode) {
    $exitCode = [int]$rawExitCode
    $exitCodeSource = "process"
} elseif (Test-Path -LiteralPath $internalOutput) {
    $internal = Get-Content -LiteralPath $internalOutput -Raw | ConvertFrom-Json
    $exitCode = if ($internal.final_decision -eq "A2C2_RESOURCE_SMOKE_INTERNAL_PASS") { 0 } else { 1 }
    $exitCodeSource = "internal_report_fallback"
} else {
    $exitCode = 125
    $exitCodeSource = "missing_process_code_and_internal_report"
}
$after = Get-HostMemorySample
$samples.Add($after)

if ($hostCeilingTerminated) {
    $meminfo = @("unavailable_after_forced_distro_terminate")
    $dmesg = @("unavailable_after_forced_distro_terminate")
} else {
    $meminfo = & wsl.exe -d Ubuntu-22.04 -- bash -lc "awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo" 2>&1
    $dmesg = & wsl.exe -d Ubuntu-22.04 -- bash -lc "dmesg --color=never 2>/dev/null | grep -Ei 'out of memory|oom-kill|killed process' | tail -30 || true" 2>&1
}

$payload = [ordered]@{
    schema_version = 1
    date_kst = (Get-Date).ToString("o")
    execution_type = "VLA_INFERENCE"
    purpose = "RESOURCE_ONLY_ACTUAL_PATH_SMOKE_HOST_MONITOR"
    cap_gib = $CapGiB
    wslconfig_sha256 = $configHash
    command_pid = $process.Id
    command_exit_code = $exitCode
    command_exit_code_source = $exitCodeSource
    host_ceiling_terminated = $hostCeilingTerminated
    baseline = $baseline
    peak = [ordered]@{
        used_fraction = $peakUsedFraction
        committed_bytes = $peakCommittedBytes
        pagefile_current_usage_mib = $peakPagefileCurrentMiB
        page_reads_per_sec = $peakPageReadsPerSec
        page_writes_per_sec = $peakPageWritesPerSec
    }
    after = $after
    pagefile_current_growth_mib = $peakPagefileCurrentMiB - [double]$baseline.pagefile_current_usage_mib
    sample_count = $samples.Count
    samples = $samples
    wsl_meminfo = @($meminfo)
    kernel_oom_lines = @($dmesg)
    stdout_path = $stdout
    stderr_path = $stderr
}
$hostOutput = Join-Path $RunRoot "cap_${CapGiB}gb_host.json"
$payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $hostOutput -Encoding utf8
$payload | ConvertTo-Json -Depth 8

if ($hostCeilingTerminated) {
    & wsl.exe --shutdown
    exit 82
}
exit $exitCode
