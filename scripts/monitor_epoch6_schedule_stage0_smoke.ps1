param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z0-9_]+$")]
    [string]$RunId,

    [string]$RepoRoot = "C:\Users\jiheo\tca_map",

    [switch]$AllowWslCacheDropAfterChild
)

$ErrorActionPreference = "Stop"
$ExpectedProtocolSha256 = "E5BA74354A1947A00045879A4815CCD09856F127E6809CF8BF649F10E2359946"
$RunRoot = Join-Path $RepoRoot "runs\epoch6_schedule_invariant_evaluation\stage0\$RunId"
$StageRoot = Join-Path $RepoRoot "runs\epoch6_schedule_invariant_evaluation\stage0"
$RunRootWsl = "/mnt/c/Users/jiheo/tca_map/runs/epoch6_schedule_invariant_evaluation/stage0/$RunId"
$HostOutput = Join-Path $RunRoot "resource_smoke_host.json"
$HostHeartbeat = Join-Path $RunRoot "resource_smoke_host_heartbeat.json"
$HostLock = Join-Path $StageRoot "host_resource_smoke.global.lock.json"
$HostLockWsl = "/mnt/c/Users/jiheo/tca_map/runs/epoch6_schedule_invariant_evaluation/stage0/host_resource_smoke.global.lock.json"
$InternalOutput = Join-Path $RunRoot "resource_smoke.json"
$ChildExitFile = Join-Path $RunRoot "resource_smoke_child_exit_code.txt"
$Stdout = Join-Path $RunRoot "resource_smoke_host_child.stdout.log"
$Stderr = Join-Path $RunRoot "resource_smoke_host_child.stderr.log"
$ProtocolPath = Join-Path $RepoRoot "reports\epoch6_schedule_invariant_evaluation\problem_verification_protocol.json"

if (-not (Test-Path -LiteralPath $RunRoot)) {
    throw "Prepared run directory is absent: $RunRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $RunRoot "fixture_manifest.json"))) {
    throw "Fixture must be captured before the resource smoke"
}
foreach ($path in @($HostOutput, $InternalOutput, $ChildExitFile, $HostHeartbeat, $HostLock, $Stdout, $Stderr)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite preserved resource-smoke artifact: $path"
    }
}
$ProtocolSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ProtocolPath).Hash
if ($ProtocolSha256 -ne $ExpectedProtocolSha256) {
    throw "Protocol hash mismatch: expected $ExpectedProtocolSha256, got $ProtocolSha256"
}

function Get-GpuSample {
    $line = (& nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | Select-Object -First 1)
    $parts = $line -split "," | ForEach-Object { $_.Trim() }
    if ($parts.Count -ne 4) {
        throw "Unexpected nvidia-smi response: $line"
    }
    return [ordered]@{
        name = $parts[0]
        total_mib = [int64]$parts[1]
        used_mib = [int64]$parts[2]
        free_mib = [int64]$parts[3]
    }
}

function Get-HostMemorySample {
    $os = Get-CimInstance Win32_OperatingSystem
    $cs = Get-CimInstance Win32_ComputerSystem
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $page = Get-CimInstance Win32_PageFileUsage
    $total = [int64]$cs.TotalPhysicalMemory
    $available = [int64]$os.FreePhysicalMemory * 1KB
    return [ordered]@{
        timestamp = (Get-Date).ToString("o")
        timestamp_unix = [DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0
        physical_total_bytes = $total
        physical_used_bytes = $total - $available
        physical_available_bytes = $available
        used_fraction = ($total - $available) / $total
        committed_bytes = [int64]$memory.CommittedBytes
        commit_limit_bytes = [int64]$memory.CommitLimit
        page_writes_per_sec = [int64]$memory.PageWritesPersec
        pages_output_per_sec = [int64]$memory.PagesOutputPersec
        pagefile_current_usage_mib = [double](($page | Measure-Object CurrentUsage -Sum).Sum)
        pagefile_allocated_mib = [double](($page | Measure-Object AllocatedBaseSize -Sum).Sum)
        gpu = Get-GpuSample
    }
}

$wslWorkers = @(& wsl.exe -d Ubuntu-22.04 -- bash -lc "ps -eo pid=,args= | grep -E 'python.*(tca_map|run_.*stage)' | grep -v grep || true")
if ($wslWorkers.Count -gt 0 -and ($wslWorkers -join "").Trim()) {
    throw "Research-capable WSL worker exists before smoke: $($wslWorkers -join '; ')"
}

$baselineFirst = Get-HostMemorySample
Start-Sleep -Seconds 2
$baseline = Get-HostMemorySample
if ([double]$baseline.used_fraction -gt 0.65) {
    throw "Unsafe host-memory baseline: $($baseline.used_fraction)"
}
if (
    [double]$baseline.pagefile_current_usage_mib -ne [double]$baselineFirst.pagefile_current_usage_mib -or
    [int64]$baseline.page_writes_per_sec -gt 0 -or
    [int64]$baseline.pages_output_per_sec -gt 0
) {
    throw "Windows pagefile baseline is not stable"
}

$arguments = @(
    "-d", "Ubuntu-22.04", "--", "bash",
    "/mnt/c/Users/jiheo/tca_map/scripts/run_epoch6_schedule_stage0_smoke_wsl.sh",
    $RunRootWsl,
    $HostLockWsl
)
$lockPayload = [ordered]@{
    status = "active"
    monitor_pid = $PID
    created_at = (Get-Date).ToString("o")
    protocol_sha256 = $ProtocolSha256
    run_id = $RunId
}
$lockBytes = [System.Text.Encoding]::UTF8.GetBytes(($lockPayload | ConvertTo-Json -Depth 3) + "`n")
$lockStream = [System.IO.File]::Open($HostLock, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
try {
    $lockStream.Write($lockBytes, 0, $lockBytes.Length)
} finally {
    $lockStream.Dispose()
}
$process = $null
$pagefileActivityTerminated = $false
try {
$process = Start-Process -FilePath "wsl.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
$samples = New-Object System.Collections.Generic.List[object]
$peakUsedFraction = [double]$baseline.used_fraction
$peakCommittedBytes = [int64]$baseline.committed_bytes
$peakPagefileCurrentMiB = [double]$baseline.pagefile_current_usage_mib
$peakPageWritesPerSec = [int64]$baseline.page_writes_per_sec
$peakPagesOutputPerSec = [int64]$baseline.pages_output_per_sec
$peakGpuUsedMiB = [int64]$baseline.gpu.used_mib
$hostCeilingTerminated = $false

while (-not $process.HasExited) {
    $sample = Get-HostMemorySample
    $samples.Add($sample)
    $peakUsedFraction = [math]::Max($peakUsedFraction, [double]$sample.used_fraction)
    $peakCommittedBytes = [math]::Max($peakCommittedBytes, [int64]$sample.committed_bytes)
    $peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$sample.pagefile_current_usage_mib)
    $peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$sample.page_writes_per_sec)
    $peakPagesOutputPerSec = [math]::Max($peakPagesOutputPerSec, [int64]$sample.pages_output_per_sec)
    $peakGpuUsedMiB = [math]::Max($peakGpuUsedMiB, [int64]$sample.gpu.used_mib)
    $heartbeatPayload = [ordered]@{
        status = "running"
        monitor_pid = $PID
        child_pid = $process.Id
        updated_at = (Get-Date).ToString("o")
        sample_count = $samples.Count
        current = $sample
        peak_used_fraction = $peakUsedFraction
        peak_pagefile_current_usage_mib = $peakPagefileCurrentMiB
        peak_page_writes_per_sec = $peakPageWritesPerSec
        peak_pages_output_per_sec = $peakPagesOutputPerSec
    }
    $heartbeatTemporary = "$HostHeartbeat.tmp"
    $heartbeatPayload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $heartbeatTemporary -Encoding utf8
    Move-Item -Force -LiteralPath $heartbeatTemporary -Destination $HostHeartbeat
    $currentPagefileGrowthMiB = [double]$sample.pagefile_current_usage_mib - [double]$baseline.pagefile_current_usage_mib
    if (
        $currentPagefileGrowthMiB -gt 0 -or
        [int64]$sample.page_writes_per_sec -gt 0 -or
        [int64]$sample.pages_output_per_sec -gt 0
    ) {
        $pagefileActivityTerminated = $true
        & wsl.exe -d Ubuntu-22.04 -- bash -lc "pkill -TERM -f 'run_epoch6_schedule_invariance_stage0.py --mode resource-smoke' || true" | Out-Null
        Start-Sleep -Seconds 2
        & wsl.exe -d Ubuntu-22.04 -- bash -lc "pkill -KILL -f 'run_epoch6_schedule_invariance_stage0.py --mode resource-smoke' || true" | Out-Null
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        break
    }
    if ([double]$sample.used_fraction -gt 0.82) {
        $hostCeilingTerminated = $true
        & wsl.exe -d Ubuntu-22.04 -- bash -lc "pkill -TERM -f 'run_epoch6_schedule_invariance_stage0.py --mode resource-smoke' || true" | Out-Null
        Start-Sleep -Seconds 2
        & wsl.exe -d Ubuntu-22.04 -- bash -lc "pkill -KILL -f 'run_epoch6_schedule_invariance_stage0.py --mode resource-smoke' || true" | Out-Null
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        break
    }
    Start-Sleep -Milliseconds 500
    $process.Refresh()
}
$process.WaitForExit()
$process.Refresh()
$childExitSource = "powershell_process"
$childExitCode = $process.ExitCode
if (Test-Path -LiteralPath $ChildExitFile) {
    $childExitCode = [int](Get-Content -Raw -LiteralPath $ChildExitFile).Trim()
    $childExitSource = "bash_persisted"
}
$afterChild = Get-HostMemorySample
$samples.Add($afterChild)
if ($AllowWslCacheDropAfterChild) {
    & wsl.exe -d Ubuntu-22.04 -u root -- sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
    Start-Sleep -Seconds 20
} else {
    Start-Sleep -Seconds 3
}
$afterRelease = Get-HostMemorySample
$samples.Add($afterRelease)

foreach ($sample in @($afterChild, $afterRelease)) {
    $peakUsedFraction = [math]::Max($peakUsedFraction, [double]$sample.used_fraction)
    $peakCommittedBytes = [math]::Max($peakCommittedBytes, [int64]$sample.committed_bytes)
    $peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$sample.pagefile_current_usage_mib)
    $peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$sample.page_writes_per_sec)
    $peakPagesOutputPerSec = [math]::Max($peakPagesOutputPerSec, [int64]$sample.pages_output_per_sec)
    $peakGpuUsedMiB = [math]::Max($peakGpuUsedMiB, [int64]$sample.gpu.used_mib)
}

$pagefileGrowthMiB = $peakPagefileCurrentMiB - [double]$baseline.pagefile_current_usage_mib
$pagefileWriteActivity = [bool]($peakPageWritesPerSec -gt 0 -or $peakPagesOutputPerSec -gt 0)
$memoryReleaseVerified = [bool]([int64]$afterRelease.physical_used_bytes -le ([int64]$baseline.physical_used_bytes + 1GB))
$gpuReleaseVerified = [bool]([int64]$afterRelease.gpu.used_mib -le ([int64]$baseline.gpu.used_mib + 256))
$internal = if (Test-Path -LiteralPath $InternalOutput) { Get-Content -LiteralPath $InternalOutput -Raw | ConvertFrom-Json } else { $null }
$internalValid = [bool](
    $null -ne $internal -and
    $internal.status -eq "ACTUAL_PATH_RESOURCE_SMOKE_PASS" -and
    [int64]$internal.model_inference_calls -eq 1 -and
    [int64]$internal.resource_monitor.maximum_swap_used_bytes -eq 0 -and
    @($internal.resource_monitor.exceptions).Count -eq 0
)

if ($hostCeilingTerminated) {
    $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_HOST_CEILING"
} elseif ($pagefileActivityTerminated) {
    $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_PAGEFILE_ACTIVITY"
} elseif ($childExitCode -ne 0 -or -not $internalValid) {
    $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_CHILD_OR_INTERNAL"
} elseif ($pagefileGrowthMiB -gt 0 -or $pagefileWriteActivity) {
    $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_PAGEFILE_ACTIVITY"
} elseif (-not $memoryReleaseVerified -or -not $gpuReleaseVerified) {
    $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_TEARDOWN"
} elseif ($peakUsedFraction -le 0.82) {
    $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_PASS"
} else {
    $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_UNCLASSIFIED"
}

$payload = [ordered]@{
    schema_version = "epoch6.schedule_stage0.host_resource_smoke.v1"
    completed_at = (Get-Date).ToString("o")
    protocol_sha256 = $ProtocolSha256
    monitor_script_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash
    child_pid = $process.Id
    child_exit_code = $childExitCode
    child_exit_code_source = $childExitSource
    wsl_cache_drop_after_child_requested = [bool]$AllowWslCacheDropAfterChild
    host_ceiling_terminated = $hostCeilingTerminated
    pagefile_activity_terminated = $pagefileActivityTerminated
    baseline_first = $baselineFirst
    baseline = $baseline
    peak = [ordered]@{
        used_fraction = $peakUsedFraction
        committed_bytes = $peakCommittedBytes
        pagefile_current_usage_mib = $peakPagefileCurrentMiB
        page_writes_per_sec = $peakPageWritesPerSec
        pages_output_per_sec = $peakPagesOutputPerSec
        gpu_used_mib = $peakGpuUsedMiB
    }
    after_child = $afterChild
    after_release = $afterRelease
    pagefile_current_growth_mib = $pagefileGrowthMiB
    pagefile_write_activity = $pagefileWriteActivity
    memory_release_verified = $memoryReleaseVerified
    gpu_release_verified = $gpuReleaseVerified
    sample_count = $samples.Count
    samples = $samples
    internal_report_path = $InternalOutput
    internal_report_sha256 = if ($null -ne $internal) { (Get-FileHash -Algorithm SHA256 -LiteralPath $InternalOutput).Hash } else { $null }
    internal_valid = $internalValid
    scientific_gate_rows = 0
    simulator_actions_executed = 0
    reward_success_done_read = $false
    final_decision = $decision
}
$temporary = "$HostOutput.tmp"
$payload | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $temporary -Encoding utf8
Move-Item -LiteralPath $temporary -Destination $HostOutput
$finalHeartbeat = [ordered]@{
    status = "completed"
    monitor_pid = $PID
    child_pid = $process.Id
    updated_at = (Get-Date).ToString("o")
    final_decision = $decision
}
$heartbeatTemporary = "$HostHeartbeat.tmp"
$finalHeartbeat | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $heartbeatTemporary -Encoding utf8
Move-Item -Force -LiteralPath $heartbeatTemporary -Destination $HostHeartbeat
$payload | ConvertTo-Json -Depth 5

if ($decision -eq "EPOCH6_STAGE0_RESOURCE_SMOKE_PASS") {
    exit 0
}
if ($hostCeilingTerminated) {
    exit 82
}
exit 1
} finally {
    if ($null -ne $process) {
        $process.Refresh()
        if (-not $process.HasExited) {
            & wsl.exe -d Ubuntu-22.04 -- bash -lc "pkill -TERM -f 'run_epoch6_schedule_invariance_stage0.py --mode resource-smoke' || true" | Out-Null
            Start-Sleep -Seconds 2
            & wsl.exe -d Ubuntu-22.04 -- bash -lc "pkill -KILL -f 'run_epoch6_schedule_invariance_stage0.py --mode resource-smoke' || true" | Out-Null
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path -LiteralPath $HostLock) {
        $releasedLock = Join-Path $RunRoot ("released_host_resource_smoke_lock_" + [DateTimeOffset]::Now.ToUnixTimeSeconds() + ".json")
        Move-Item -LiteralPath $HostLock -Destination $releasedLock -ErrorAction SilentlyContinue
    }
}
