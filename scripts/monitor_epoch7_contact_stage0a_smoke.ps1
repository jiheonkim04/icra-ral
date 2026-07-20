param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z0-9_]+$")]
    [string]$RunId,

    [string]$RepoRoot = "C:\Users\jiheo\tca_map"
)

$ErrorActionPreference = "Stop"
$ExpectedProtocolSha256 = "7FA28AAEEAC9886F36DD5CCD059CA7AC4CD65B21FABFBBCA4AFFA53B0A256240"
$ExpectedAmendmentSha256 = "7CCDCE5D9AA0B24C356AF873D0481AF76312D3C7FCF6871C4CA80FD6621ACFEB"
$BaselineUsedFractionMax = 0.70
$PeakUsedFractionMax = 0.85
$PagefileGrowthMiBMax = 16.0
$ReleasePhysicalBytesMax = 2GB
$ReleaseGpuMiBMax = 256

$StageRoot = Join-Path $RepoRoot "runs\epoch7_contact_transition_topology\stage0a"
$RunRoot = Join-Path $StageRoot $RunId
$RunRootWsl = "/mnt/c/Users/jiheo/tca_map/runs/epoch7_contact_transition_topology/stage0a/$RunId"
$HostLock = Join-Path $StageRoot "host_resource_smoke.global.lock.json"
$HostLockWsl = "/mnt/c/Users/jiheo/tca_map/runs/epoch7_contact_transition_topology/stage0a/host_resource_smoke.global.lock.json"
$HostOutput = Join-Path $RunRoot "resource_smoke_host.json"
$HostHeartbeat = Join-Path $RunRoot "resource_smoke_host_heartbeat.json"
$InternalOutput = Join-Path $RunRoot "resource_smoke.json"
$ChildExitFile = Join-Path $RunRoot "resource_smoke_child_exit_code.txt"
$Stdout = Join-Path $RunRoot "resource_smoke_host_child.stdout.log"
$Stderr = Join-Path $RunRoot "resource_smoke_host_child.stderr.log"
$ProtocolPath = Join-Path $RepoRoot "reports\epoch6_contact_transition_topology\problem_verification_protocol.json"
$AmendmentPath = Join-Path $RepoRoot "reports\epoch7_contact_transition_topology\resource_rule_amendment.json"

if (-not (Test-Path -LiteralPath (Join-Path $RunRoot "static_preflight.json"))) {
    throw "A passed static preflight is required before the contact resource smoke"
}
foreach ($path in @($HostOutput, $HostHeartbeat, $InternalOutput, $ChildExitFile, $Stdout, $Stderr, $HostLock)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite preserved contact resource-smoke artifact: $path"
    }
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $ProtocolPath).Hash -ne $ExpectedProtocolSha256) {
    throw "Frozen contact protocol hash mismatch"
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $AmendmentPath).Hash -ne $ExpectedAmendmentSha256) {
    throw "Frozen Epoch 7 resource amendment hash mismatch"
}

function Get-GpuSample {
    $line = (& nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | Select-Object -First 1)
    $parts = $line -split "," | ForEach-Object { $_.Trim() }
    if ($parts.Count -ne 4) { throw "Unexpected nvidia-smi response: $line" }
    return [ordered]@{
        name = $parts[0]
        total_mib = [int64]$parts[1]
        used_mib = [int64]$parts[2]
        free_mib = [int64]$parts[3]
    }
}

function Get-HostSample {
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
        page_writes_per_sec = [int64]$memory.PageWritesPersec
        pages_output_per_sec = [int64]$memory.PagesOutputPersec
        pagefile_current_usage_mib = [double](($page | Measure-Object CurrentUsage -Sum).Sum)
        pagefile_allocated_mib = [double](($page | Measure-Object AllocatedBaseSize -Sum).Sum)
        gpu = Get-GpuSample
    }
}

function Stop-ExactSmokeChild([object]$Process) {
    & wsl.exe -d Ubuntu-22.04 -- bash -lc "pkill -TERM -f 'run_epoch6_contact_topology_stage0a.py --mode resource-smoke' || true" | Out-Null
    Start-Sleep -Seconds 2
    & wsl.exe -d Ubuntu-22.04 -- bash -lc "pkill -KILL -f 'run_epoch6_contact_topology_stage0a.py --mode resource-smoke' || true" | Out-Null
    if ($null -ne $Process) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

$wslWorkers = @(& wsl.exe -d Ubuntu-22.04 -- bash -lc "ps -eo pid=,args= | grep -E 'python.*(run_epoch6_contact|contact_topology_stage0a)' | grep -v grep || true")
if ($wslWorkers.Count -gt 0 -and ($wslWorkers -join "").Trim()) {
    throw "Research-capable contact WSL worker exists before smoke: $($wslWorkers -join '; ')"
}

$baselineFirst = Get-HostSample
Start-Sleep -Seconds 2
$baseline = Get-HostSample
if ([double]$baseline.used_fraction -gt $BaselineUsedFractionMax) {
    throw "Unsafe contact-smoke host baseline: $($baseline.used_fraction)"
}
if (
    [int64]$baselineFirst.page_writes_per_sec -gt 0 -or
    [int64]$baselineFirst.pages_output_per_sec -gt 0 -or
    [int64]$baseline.page_writes_per_sec -gt 0 -or
    [int64]$baseline.pages_output_per_sec -gt 0
) {
    throw "Windows paging writes are active at baseline"
}

$lockPayload = [ordered]@{
    status = "active"
    monitor_pid = $PID
    created_at = (Get-Date).ToString("o")
    protocol_sha256 = $ExpectedProtocolSha256
    resource_amendment_sha256 = $ExpectedAmendmentSha256
    run_id = $RunId
}
$lockBytes = [System.Text.Encoding]::UTF8.GetBytes(($lockPayload | ConvertTo-Json -Depth 3) + "`n")
$lockStream = [System.IO.File]::Open($HostLock, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
try { $lockStream.Write($lockBytes, 0, $lockBytes.Length) } finally { $lockStream.Dispose() }

$arguments = @(
    "-d", "Ubuntu-22.04", "--", "bash",
    "/mnt/c/Users/jiheo/tca_map/scripts/run_epoch6_contact_stage0a_smoke_wsl.sh",
    $RunRootWsl,
    $HostLockWsl
)
$process = $null
$hostCeilingTerminated = $false
$pagingWriteTerminated = $false
$pagefileGrowthTerminated = $false
try {
    $process = Start-Process -FilePath "wsl.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    $samples = New-Object System.Collections.Generic.List[object]
    $peakUsedFraction = [double]$baseline.used_fraction
    $peakPagefileCurrentMiB = [double]$baseline.pagefile_current_usage_mib
    $peakPageWritesPerSec = [int64]$baseline.page_writes_per_sec
    $peakPagesOutputPerSec = [int64]$baseline.pages_output_per_sec
    $peakGpuUsedMiB = [int64]$baseline.gpu.used_mib
    while (-not $process.HasExited) {
        $sample = Get-HostSample
        $samples.Add($sample)
        $peakUsedFraction = [math]::Max($peakUsedFraction, [double]$sample.used_fraction)
        $peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$sample.pagefile_current_usage_mib)
        $peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$sample.page_writes_per_sec)
        $peakPagesOutputPerSec = [math]::Max($peakPagesOutputPerSec, [int64]$sample.pages_output_per_sec)
        $peakGpuUsedMiB = [math]::Max($peakGpuUsedMiB, [int64]$sample.gpu.used_mib)
        $heartbeat = [ordered]@{
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
        $heartbeat | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $heartbeatTemporary -Encoding utf8
        Move-Item -Force -LiteralPath $heartbeatTemporary -Destination $HostHeartbeat
        $hostCeilingTerminated = [bool]([double]$sample.used_fraction -gt $PeakUsedFractionMax)
        $pagingWriteTerminated = [bool](
            [int64]$sample.page_writes_per_sec -gt 0 -or
            [int64]$sample.pages_output_per_sec -gt 0
        )
        $pagefileGrowthTerminated = [bool](
            [double]$sample.pagefile_current_usage_mib -
            [double]$baseline.pagefile_current_usage_mib -gt $PagefileGrowthMiBMax
        )
        if ($hostCeilingTerminated -or $pagingWriteTerminated -or $pagefileGrowthTerminated) {
            Stop-ExactSmokeChild $process
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
    $afterChild = Get-HostSample
    $samples.Add($afterChild)
    & wsl.exe -d Ubuntu-22.04 -u root -- sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
    Start-Sleep -Seconds 5
    $afterRelease = Get-HostSample
    $samples.Add($afterRelease)
    foreach ($sample in @($afterChild, $afterRelease)) {
        $peakUsedFraction = [math]::Max($peakUsedFraction, [double]$sample.used_fraction)
        $peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$sample.pagefile_current_usage_mib)
        $peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$sample.page_writes_per_sec)
        $peakPagesOutputPerSec = [math]::Max($peakPagesOutputPerSec, [int64]$sample.pages_output_per_sec)
        $peakGpuUsedMiB = [math]::Max($peakGpuUsedMiB, [int64]$sample.gpu.used_mib)
    }
    $pagefileGrowthMiB = $peakPagefileCurrentMiB - [double]$baseline.pagefile_current_usage_mib
    $pagefileWriteActivity = [bool]($peakPageWritesPerSec -gt 0 -or $peakPagesOutputPerSec -gt 0)
    $memoryReleaseVerified = [bool]([int64]$afterRelease.physical_used_bytes -le ([int64]$baseline.physical_used_bytes + $ReleasePhysicalBytesMax))
    $gpuReleaseVerified = [bool]([int64]$afterRelease.gpu.used_mib -le ([int64]$baseline.gpu.used_mib + $ReleaseGpuMiBMax))
    $internal = if (Test-Path -LiteralPath $InternalOutput) { Get-Content -Raw -LiteralPath $InternalOutput | ConvertFrom-Json } else { $null }
    $internalValid = [bool](
        $null -ne $internal -and
        $internal.status -eq "ACTUAL_PATH_CONTACT_RESOURCE_SMOKE_PASS" -and
        $internal.protocol_sha256 -eq $ExpectedProtocolSha256 -and
        [int64]$internal.contact_label_gate_rows -eq 0 -and
        [int64]$internal.forbidden_dataset_access_count -eq 0 -and
        [int64]$internal.simulator_actions_executed -eq 0 -and
        [int64]$internal.success_check_calls -eq 0 -and
        -not [bool]$internal.reward_success_done_read -and
        [int64]$internal.resources_after.swap_used_bytes -eq 0
    )
    if ($hostCeilingTerminated -or $peakUsedFraction -gt $PeakUsedFractionMax) {
        $decision = "EPOCH7_CONTACT_STAGE0A_RESOURCE_SMOKE_FAIL_HOST_CEILING"
    } elseif ($pagingWriteTerminated -or $pagefileWriteActivity) {
        $decision = "EPOCH7_CONTACT_STAGE0A_RESOURCE_SMOKE_FAIL_PAGING_WRITES"
    } elseif ($pagefileGrowthTerminated -or $pagefileGrowthMiB -gt $PagefileGrowthMiBMax) {
        $decision = "EPOCH7_CONTACT_STAGE0A_RESOURCE_SMOKE_FAIL_PAGEFILE_GROWTH"
    } elseif ($childExitCode -ne 0 -or -not $internalValid) {
        $decision = "EPOCH7_CONTACT_STAGE0A_RESOURCE_SMOKE_FAIL_CHILD_OR_INTERNAL"
    } elseif (-not $memoryReleaseVerified -or -not $gpuReleaseVerified) {
        $decision = "EPOCH7_CONTACT_STAGE0A_RESOURCE_SMOKE_FAIL_CONTROLLED_RELEASE"
    } else {
        $decision = "EPOCH7_CONTACT_STAGE0A_RESOURCE_SMOKE_PASS"
    }
    $payload = [ordered]@{
        schema_version = "epoch7.contact_topology.host_resource_smoke.v1"
        completed_at = (Get-Date).ToString("o")
        protocol_sha256 = $ExpectedProtocolSha256
        resource_amendment_sha256 = $ExpectedAmendmentSha256
        monitor_script_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash
        child_pid = $process.Id
        child_exit_code = $childExitCode
        child_exit_code_source = $childExitSource
        wsl_shutdown_after_child_requested = $false
        wsl_cache_drop_after_child_requested = $true
        host_ceiling_terminated = $hostCeilingTerminated
        paging_write_terminated = $pagingWriteTerminated
        pagefile_growth_terminated = $pagefileGrowthTerminated
        thresholds = [ordered]@{
            baseline_used_fraction_max = $BaselineUsedFractionMax
            peak_used_fraction_max = $PeakUsedFractionMax
            pagefile_allocation_growth_mib_max = $PagefileGrowthMiBMax
            physical_used_bytes_above_baseline_after_release_max = $ReleasePhysicalBytesMax
            gpu_used_mib_above_baseline_after_release_max = $ReleaseGpuMiBMax
        }
        baseline_first = $baselineFirst
        baseline = $baseline
        peak = [ordered]@{
            used_fraction = $peakUsedFraction
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
    if ($decision -eq "EPOCH7_CONTACT_STAGE0A_RESOURCE_SMOKE_PASS") { exit 0 }
    exit 1
} finally {
    if ($null -ne $process -and -not $process.HasExited) {
        Stop-ExactSmokeChild $process
    }
    if (Test-Path -LiteralPath $HostLock) {
        Remove-Item -LiteralPath $HostLock -Force
    }
}
