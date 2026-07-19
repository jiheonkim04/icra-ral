param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z0-9_]+$")]
    [string]$RunId,

    [string]$RepoRoot = "C:\Users\jiheo\tca_map",

    [switch]$CalibratedResourceGovernance,

    [switch]$AllowWslCacheDropAfterChild
)

$ErrorActionPreference = "Stop"
$ExpectedProtocolSha256 = "E5BA74354A1947A00045879A4815CCD09856F127E6809CF8BF649F10E2359946"
$ExpectedResourceAmendmentSha256 = "E98AED352765CEDA55607A2182ECE7B6E44B499DCC0253DA66313C72A6F3C601"
$ResourceGovernanceMode = "CALIBRATED_OUTCOME_FREE_V1"
$IdleControlDurationSeconds = 60
$IdleControlSampleIntervalSeconds = 1
$RuntimeSampleIntervalMilliseconds = 500
$SustainedPagingMinConsecutiveSamples = 3
$RunRoot = Join-Path $RepoRoot "runs\epoch6_schedule_invariant_evaluation\stage0\$RunId"
$StageRoot = Join-Path $RepoRoot "runs\epoch6_schedule_invariant_evaluation\stage0"
$RunRootWsl = "/mnt/c/Users/jiheo/tca_map/runs/epoch6_schedule_invariant_evaluation/stage0/$RunId"
$HostOutput = Join-Path $RunRoot "resource_smoke_host.json"
$IdleControlOutput = Join-Path $RunRoot "resource_smoke_idle_control.json"
$HostHeartbeat = Join-Path $RunRoot "resource_smoke_host_heartbeat.json"
$HostLock = Join-Path $StageRoot "host_resource_smoke.global.lock.json"
$HostLockWsl = "/mnt/c/Users/jiheo/tca_map/runs/epoch6_schedule_invariant_evaluation/stage0/host_resource_smoke.global.lock.json"
$InternalOutput = Join-Path $RunRoot "resource_smoke.json"
$ChildExitFile = Join-Path $RunRoot "resource_smoke_child_exit_code.txt"
$Stdout = Join-Path $RunRoot "resource_smoke_host_child.stdout.log"
$Stderr = Join-Path $RunRoot "resource_smoke_host_child.stderr.log"
$ProtocolPath = Join-Path $RepoRoot "reports\epoch6_schedule_invariant_evaluation\problem_verification_protocol.json"
$ResourceAmendmentPath = Join-Path $RepoRoot "reports\epoch6_schedule_invariant_evaluation\resource_governance_amendment_v1.json"

if (-not $CalibratedResourceGovernance) {
    throw "The preserved strict smoke has already failed; explicitly select -CalibratedResourceGovernance"
}
if (-not (Test-Path -LiteralPath $RunRoot)) {
    throw "Prepared run directory is absent: $RunRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $RunRoot "fixture_manifest.json"))) {
    throw "Fixture must be captured before the resource smoke"
}
foreach ($path in @(
    $HostOutput,
    $IdleControlOutput,
    $InternalOutput,
    $ChildExitFile,
    $HostHeartbeat,
    $HostLock,
    $Stdout,
    $Stderr
)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite preserved resource-smoke artifact: $path"
    }
}
$ProtocolSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ProtocolPath).Hash
if ($ProtocolSha256 -ne $ExpectedProtocolSha256) {
    throw "Protocol hash mismatch: expected $ExpectedProtocolSha256, got $ProtocolSha256"
}
$ResourceAmendmentSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ResourceAmendmentPath).Hash
if ($ResourceAmendmentSha256 -ne $ExpectedResourceAmendmentSha256) {
    throw "Resource-amendment hash mismatch: expected $ExpectedResourceAmendmentSha256, got $ResourceAmendmentSha256"
}

function Write-JsonAtomic {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] $Payload,
        [int]$Depth = 9
    )
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -Force -LiteralPath $temporary -Destination $Path
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

function Test-PagingActive {
    param([Parameter(Mandatory = $true)] $Sample)
    return [bool](
        [int64]$Sample.page_writes_per_sec -gt 0 -or
        [int64]$Sample.pages_output_per_sec -gt 0
    )
}

function Get-MaxConsecutivePagingSamples {
    param([object[]]$Samples)
    $current = 0
    $maximum = 0
    foreach ($sample in $Samples) {
        if (Test-PagingActive -Sample $sample) {
            $current += 1
            $maximum = [math]::Max($maximum, $current)
        } else {
            $current = 0
        }
    }
    return [int]$maximum
}

function Get-WslRunningNames {
    $lines = @(& wsl.exe --list --running --quiet 2>$null)
    return @(
        $lines |
            ForEach-Object { ([string]$_).Replace([string][char]0, "").Trim() } |
            Where-Object { $_ }
    )
}

function Wait-WslStopped {
    param([int]$TimeoutSeconds)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (@(Get-WslRunningNames).Count -eq 0) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return [bool](@(Get-WslRunningNames).Count -eq 0)
}

$runningWsl = @(Get-WslRunningNames)
if ($runningWsl.Count -gt 0) {
    $wslWorkers = @(& wsl.exe -d Ubuntu-22.04 -- bash -lc "ps -eo pid=,args= | grep -E 'python.*(tca_map|run_.*stage)' | grep -v grep || true")
    if ($wslWorkers.Count -gt 0 -and ($wslWorkers -join "").Trim()) {
        throw "Research-capable WSL worker exists before calibrated smoke: $($wslWorkers -join '; ')"
    }
}
& wsl.exe --shutdown
if (-not (Wait-WslStopped -TimeoutSeconds 30)) {
    throw "WSL did not stop before the idle control"
}

$idleSamples = New-Object System.Collections.Generic.List[object]
for ($index = 0; $index -lt $IdleControlDurationSeconds; $index += 1) {
    if (@(Get-WslRunningNames).Count -ne 0) {
        throw "WSL started during the no-research idle control"
    }
    $idleSamples.Add((Get-HostMemorySample))
    if ($index -lt ($IdleControlDurationSeconds - 1)) {
        Start-Sleep -Seconds $IdleControlSampleIntervalSeconds
    }
}
$idleMaximumUsedFraction = [double](($idleSamples | ForEach-Object { [double]$_.used_fraction } | Measure-Object -Maximum).Maximum)
$idleMaximumConsecutivePaging = Get-MaxConsecutivePagingSamples -Samples @($idleSamples)
$idleValid = [bool](
    $idleSamples.Count -eq $IdleControlDurationSeconds -and
    $idleMaximumUsedFraction -le 0.65 -and
    $idleMaximumConsecutivePaging -lt $SustainedPagingMinConsecutiveSamples -and
    @(Get-WslRunningNames).Count -eq 0
)
$idlePayload = [ordered]@{
    schema_version = "epoch6.schedule_stage0.idle_control.v1"
    completed_at = (Get-Date).ToString("o")
    resource_governance_mode = $ResourceGovernanceMode
    resource_amendment_sha256 = $ResourceAmendmentSha256
    duration_seconds = $IdleControlDurationSeconds
    sample_interval_seconds = $IdleControlSampleIntervalSeconds
    sample_count = $idleSamples.Count
    wsl_stopped_throughout = $true
    maximum_used_fraction = $idleMaximumUsedFraction
    pagefile_current_usage_mib_min = [double](($idleSamples | ForEach-Object { [double]$_.pagefile_current_usage_mib } | Measure-Object -Minimum).Minimum)
    pagefile_current_usage_mib_max = [double](($idleSamples | ForEach-Object { [double]$_.pagefile_current_usage_mib } | Measure-Object -Maximum).Maximum)
    maximum_page_writes_per_sec = [int64](($idleSamples | ForEach-Object { [int64]$_.page_writes_per_sec } | Measure-Object -Maximum).Maximum)
    maximum_pages_output_per_sec = [int64](($idleSamples | ForEach-Object { [int64]$_.pages_output_per_sec } | Measure-Object -Maximum).Maximum)
    maximum_consecutive_paging_active_samples = $idleMaximumConsecutivePaging
    sustained_paging_min_consecutive_samples = $SustainedPagingMinConsecutiveSamples
    samples = $idleSamples
    valid = $idleValid
}
Write-JsonAtomic -Path $IdleControlOutput -Payload $idlePayload
if (-not $idleValid) {
    throw "The no-research idle control did not establish a clean calibrated baseline"
}

$baselineFirst = $idleSamples[0]
$baseline = $idleSamples[$idleSamples.Count - 1]
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
    resource_amendment_sha256 = $ResourceAmendmentSha256
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
$sustainedPagingTerminated = $false
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
    $currentPagingConsecutive = 0
    $maximumRuntimeConsecutivePaging = 0

    while (-not $process.HasExited) {
        $sample = Get-HostMemorySample
        $samples.Add($sample)
        $peakUsedFraction = [math]::Max($peakUsedFraction, [double]$sample.used_fraction)
        $peakCommittedBytes = [math]::Max($peakCommittedBytes, [int64]$sample.committed_bytes)
        $peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$sample.pagefile_current_usage_mib)
        $peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$sample.page_writes_per_sec)
        $peakPagesOutputPerSec = [math]::Max($peakPagesOutputPerSec, [int64]$sample.pages_output_per_sec)
        $peakGpuUsedMiB = [math]::Max($peakGpuUsedMiB, [int64]$sample.gpu.used_mib)
        if (Test-PagingActive -Sample $sample) {
            $currentPagingConsecutive += 1
            $maximumRuntimeConsecutivePaging = [math]::Max($maximumRuntimeConsecutivePaging, $currentPagingConsecutive)
        } else {
            $currentPagingConsecutive = 0
        }
        $heartbeatPayload = [ordered]@{
            status = "running"
            monitor_pid = $PID
            child_pid = $process.Id
            updated_at = (Get-Date).ToString("o")
            sample_count = $samples.Count
            current = $sample
            peak_used_fraction = $peakUsedFraction
            maximum_consecutive_paging_active_samples = $maximumRuntimeConsecutivePaging
        }
        Write-JsonAtomic -Path $HostHeartbeat -Payload $heartbeatPayload -Depth 6
        if ($currentPagingConsecutive -ge $SustainedPagingMinConsecutiveSamples) {
            $sustainedPagingTerminated = $true
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
        Start-Sleep -Milliseconds $RuntimeSampleIntervalMilliseconds
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
    $memoryReleaseVerified = [bool]([int64]$afterRelease.physical_used_bytes -le ([int64]$baseline.physical_used_bytes + 1GB))

    & wsl.exe --shutdown
    $shutdownInvoked = $true
    $restoreSamples = New-Object System.Collections.Generic.List[object]
    $cleanStateRestored = $false
    $restoreDeadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $restoreDeadline) {
        Start-Sleep -Seconds 1
        $restoreSample = Get-HostMemorySample
        $restoreSamples.Add($restoreSample)
        $restorePaging = Get-MaxConsecutivePagingSamples -Samples @($restoreSamples)
        $wslStopped = [bool](@(Get-WslRunningNames).Count -eq 0)
        $cleanStateRestored = [bool](
            $wslStopped -and
            [double]$restoreSample.used_fraction -le 0.65 -and
            [int64]$restoreSample.gpu.used_mib -le ([int64]$baseline.gpu.used_mib + 256) -and
            $restorePaging -lt $SustainedPagingMinConsecutiveSamples
        )
        if ($cleanStateRestored) {
            break
        }
    }
    $afterCleanState = $restoreSamples[$restoreSamples.Count - 1]

    foreach ($sample in @($afterChild, $afterRelease) + @($restoreSamples)) {
        $peakUsedFraction = [math]::Max($peakUsedFraction, [double]$sample.used_fraction)
        $peakCommittedBytes = [math]::Max($peakCommittedBytes, [int64]$sample.committed_bytes)
        $peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$sample.pagefile_current_usage_mib)
        $peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$sample.page_writes_per_sec)
        $peakPagesOutputPerSec = [math]::Max($peakPagesOutputPerSec, [int64]$sample.pages_output_per_sec)
        $peakGpuUsedMiB = [math]::Max($peakGpuUsedMiB, [int64]$sample.gpu.used_mib)
    }

    $maximumRuntimeConsecutivePaging = Get-MaxConsecutivePagingSamples -Samples @($samples)
    $maximumRestoreConsecutivePaging = Get-MaxConsecutivePagingSamples -Samples @($restoreSamples)
    $sustainedPagingDetected = [bool](
        $maximumRuntimeConsecutivePaging -ge $SustainedPagingMinConsecutiveSamples -or
        $maximumRestoreConsecutivePaging -ge $SustainedPagingMinConsecutiveSamples
    )
    $pagefileGrowthMiB = $peakPagefileCurrentMiB - [double]$baseline.pagefile_current_usage_mib
    $pagefileWriteActivity = [bool]($peakPageWritesPerSec -gt 0 -or $peakPagesOutputPerSec -gt 0)
    $gpuReleaseVerified = [bool]([int64]$afterCleanState.gpu.used_mib -le ([int64]$baseline.gpu.used_mib + 256))
    $stdoutText = if (Test-Path -LiteralPath $Stdout) { Get-Content -Raw -LiteralPath $Stdout } else { "" }
    $stderrText = if (Test-Path -LiteralPath $Stderr) { Get-Content -Raw -LiteralPath $Stderr } else { "" }
    $oomOrKillSignatureDetected = [bool](($stdoutText + "`n" + $stderrText) -match '(?i)out of memory|cuda[^\r\n]*oom|killed process|oom-kill')
    $internal = if (Test-Path -LiteralPath $InternalOutput) { Get-Content -LiteralPath $InternalOutput -Raw | ConvertFrom-Json } else { $null }
    $internalValid = [bool](
        $null -ne $internal -and
        $internal.status -eq "ACTUAL_PATH_RESOURCE_SMOKE_PASS" -and
        [int64]$internal.model_inference_calls -eq 1 -and
        [bool]$internal.raw_chunk_finite -and
        [int64]$internal.resource_monitor.maximum_swap_used_bytes -eq 0 -and
        @($internal.resource_monitor.exceptions).Count -eq 0 -and
        @($internal.runtime.parameter_devices).Count -eq 1 -and
        [string]$internal.runtime.parameter_devices[0] -eq "cuda:0" -and
        [bool]$internal.runtime.device_map_requested -eq $false -and
        [bool]$internal.runtime.cpu_or_disk_model_offload -eq $false
    )

    if ($hostCeilingTerminated) {
        $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_HOST_CEILING"
    } elseif ($sustainedPagingTerminated -or $sustainedPagingDetected) {
        $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_SUSTAINED_PAGING"
    } elseif ($childExitCode -ne 0 -or -not $internalValid) {
        $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_CHILD_OR_INTERNAL"
    } elseif ($oomOrKillSignatureDetected) {
        $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_OOM_OR_KILL"
    } elseif (-not $cleanStateRestored -or -not $gpuReleaseVerified) {
        $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_CLEAN_STATE_RESTORE"
    } elseif ($peakUsedFraction -le 0.82) {
        $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_PASS_CALIBRATED"
    } else {
        $decision = "EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_UNCLASSIFIED"
    }

    $allocationClassification = if ($pagefileGrowthMiB -gt 0 -and -not $sustainedPagingDetected) {
        "NONFATAL_ALLOCATION_ONLY_ABSENT_PRESSURE"
    } elseif ($pagefileGrowthMiB -gt 0) {
        "ALLOCATION_WITH_PAGING_PRESSURE"
    } else {
        "NO_POSITIVE_ALLOCATION_DELTA"
    }
    $payload = [ordered]@{
        schema_version = "epoch6.schedule_stage0.host_resource_smoke.v2"
        completed_at = (Get-Date).ToString("o")
        protocol_sha256 = $ProtocolSha256
        resource_governance_mode = $ResourceGovernanceMode
        resource_amendment_sha256 = $ResourceAmendmentSha256
        monitor_script_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash
        child_pid = $process.Id
        child_exit_code = $childExitCode
        child_exit_code_source = $childExitSource
        wsl_cache_drop_after_child_requested = [bool]$AllowWslCacheDropAfterChild
        wsl_shutdown_after_child_invoked = $shutdownInvoked
        host_ceiling_terminated = $hostCeilingTerminated
        sustained_paging_terminated = $sustainedPagingTerminated
        idle_control_report_path = $IdleControlOutput
        idle_control_report_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $IdleControlOutput).Hash
        idle_control_valid = $idleValid
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
        after_bounded_release = $afterRelease
        after_clean_state_restore = $afterCleanState
        pagefile_current_growth_mib = $pagefileGrowthMiB
        pagefile_allocation_classification = $allocationClassification
        pagefile_write_activity_observed = $pagefileWriteActivity
        sustained_paging_min_consecutive_samples = $SustainedPagingMinConsecutiveSamples
        maximum_runtime_consecutive_paging_active_samples = $maximumRuntimeConsecutivePaging
        maximum_restore_consecutive_paging_active_samples = $maximumRestoreConsecutivePaging
        sustained_paging_detected = $sustainedPagingDetected
        memory_release_verified_in_old_bounded_window = $memoryReleaseVerified
        clean_state_restored = $cleanStateRestored
        gpu_release_verified = $gpuReleaseVerified
        oom_or_kill_signature_detected = $oomOrKillSignatureDetected
        sample_count = $samples.Count
        samples = $samples
        restore_sample_count = $restoreSamples.Count
        restore_samples = $restoreSamples
        internal_report_path = $InternalOutput
        internal_report_sha256 = if ($null -ne $internal) { (Get-FileHash -Algorithm SHA256 -LiteralPath $InternalOutput).Hash } else { $null }
        internal_valid = $internalValid
        scientific_gate_rows = 0
        simulator_actions_executed = 0
        reward_success_done_read = $false
        final_decision = $decision
    }
    Write-JsonAtomic -Path $HostOutput -Payload $payload
    $finalHeartbeat = [ordered]@{
        status = "completed"
        monitor_pid = $PID
        child_pid = $process.Id
        updated_at = (Get-Date).ToString("o")
        final_decision = $decision
    }
    Write-JsonAtomic -Path $HostHeartbeat -Payload $finalHeartbeat -Depth 4
    [ordered]@{
        final_decision = $decision
        idle_control_valid = $idleValid
        child_exit_code = $childExitCode
        internal_valid = $internalValid
        peak_used_fraction = $peakUsedFraction
        pagefile_current_growth_mib = $pagefileGrowthMiB
        pagefile_allocation_classification = $allocationClassification
        sustained_paging_detected = $sustainedPagingDetected
        clean_state_restored = $cleanStateRestored
        oom_or_kill_signature_detected = $oomOrKillSignatureDetected
    } | ConvertTo-Json -Depth 3

    if ($decision -eq "EPOCH6_STAGE0_RESOURCE_SMOKE_PASS_CALIBRATED") {
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
