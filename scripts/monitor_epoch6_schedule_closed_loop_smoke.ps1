param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z0-9_]+$")]
    [string]$RunId,

    [string]$RepoRoot = "C:\Users\jiheo\tca_map"
)

$ErrorActionPreference = "Stop"
$ExpectedProtocolSha256 = "E5BA74354A1947A00045879A4815CCD09856F127E6809CF8BF649F10E2359946"
$ExpectedExecutionManifestSha256 = "93DEA8F3A9EAAE864A0A691B11312492E9E2D585606859597228E6F23D0FB242"
$ExpectedResourceAmendmentSha256 = "E98AED352765CEDA55607A2182ECE7B6E44B499DCC0253DA66313C72A6F3C601"
$IdleSeconds = 60
$SustainedPagingSamples = 3
$RunRoot = Join-Path $RepoRoot "runs\epoch6_schedule_invariant_evaluation\closed_loop\$RunId"
$StageRoot = Join-Path $RepoRoot "runs\epoch6_schedule_invariant_evaluation\closed_loop"
$RunRootWsl = "/mnt/c/Users/jiheo/tca_map/runs/epoch6_schedule_invariant_evaluation/closed_loop/$RunId"
$InternalOutput = Join-Path $RunRoot "closed_loop_resource_smoke.json"
$HostOutput = Join-Path $RunRoot "closed_loop_resource_smoke_host.json"
$IdleOutput = Join-Path $RunRoot "closed_loop_resource_smoke_idle_control.json"
$Heartbeat = Join-Path $RunRoot "closed_loop_resource_smoke_host_heartbeat.json"
$ExitFile = Join-Path $RunRoot "closed_loop_resource_smoke_child_exit_code.txt"
$Stdout = Join-Path $RunRoot "closed_loop_resource_smoke_host_child.stdout.log"
$Stderr = Join-Path $RunRoot "closed_loop_resource_smoke_host_child.stderr.log"
$HostLock = Join-Path $StageRoot "host_resource_smoke.global.lock.json"
$HostLockWsl = "/mnt/c/Users/jiheo/tca_map/runs/epoch6_schedule_invariant_evaluation/closed_loop/host_resource_smoke.global.lock.json"
$ProtocolPath = Join-Path $RepoRoot "reports\epoch6_schedule_invariant_evaluation\problem_verification_protocol.json"
$ExecutionManifestPath = Join-Path $RepoRoot "reports\epoch6_schedule_invariant_evaluation\closed_loop_execution_manifest.json"
$ResourceAmendmentPath = Join-Path $RepoRoot "reports\epoch6_schedule_invariant_evaluation\resource_governance_amendment_v1.json"

if (-not (Test-Path -LiteralPath (Join-Path $RunRoot "closed_loop_static_preflight.json"))) {
    throw "Closed-loop preflight is required before the resource smoke"
}
foreach ($path in @($InternalOutput, $HostOutput, $IdleOutput, $Heartbeat, $ExitFile, $Stdout, $Stderr, $HostLock)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite preserved closed-loop smoke artifact: $path"
    }
}
$ProtocolSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ProtocolPath).Hash
$ExecutionManifestSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ExecutionManifestPath).Hash
$ResourceAmendmentSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ResourceAmendmentPath).Hash
if ($ProtocolSha256 -ne $ExpectedProtocolSha256) { throw "Protocol hash mismatch" }
if ($ExecutionManifestSha256 -ne $ExpectedExecutionManifestSha256) { throw "Execution-manifest hash mismatch" }
if ($ResourceAmendmentSha256 -ne $ExpectedResourceAmendmentSha256) { throw "Resource-amendment hash mismatch" }

function Write-JsonAtomic {
    param([string]$Path, $Payload, [int]$Depth = 9)
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -Force -LiteralPath $temporary -Destination $Path
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
        commit_limit_bytes = [int64]$memory.CommitLimit
        page_writes_per_sec = [int64]$memory.PageWritesPersec
        pages_output_per_sec = [int64]$memory.PagesOutputPersec
        pagefile_current_usage_mib = [double](($page | Measure-Object CurrentUsage -Sum).Sum)
        pagefile_allocated_mib = [double](($page | Measure-Object AllocatedBaseSize -Sum).Sum)
        gpu = Get-GpuSample
    }
}

function Test-PagingActive {
    param($Sample)
    return [bool]([int64]$Sample.page_writes_per_sec -gt 0 -or [int64]$Sample.pages_output_per_sec -gt 0)
}

function Get-MaxConsecutivePaging {
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
    return @(
        @(& wsl.exe --list --running --quiet 2>$null) |
            ForEach-Object { ([string]$_).Replace([string][char]0, "").Trim() } |
            Where-Object { $_ }
    )
}

& wsl.exe --shutdown
$deadline = (Get-Date).AddSeconds(30)
while (@(Get-WslRunningNames).Count -ne 0 -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
if (@(Get-WslRunningNames).Count -ne 0) { throw "WSL did not stop before idle control" }

$idleSamples = New-Object System.Collections.Generic.List[object]
for ($index = 0; $index -lt $IdleSeconds; $index += 1) {
    if (@(Get-WslRunningNames).Count -ne 0) { throw "WSL started during idle control" }
    $idleSamples.Add((Get-HostSample))
    if ($index -lt ($IdleSeconds - 1)) { Start-Sleep -Seconds 1 }
}
$idleMaxRam = [double](($idleSamples | ForEach-Object { [double]$_.used_fraction } | Measure-Object -Maximum).Maximum)
$idleMaxConsecutive = Get-MaxConsecutivePaging -Samples $idleSamples.ToArray()
$idleValid = [bool]($idleSamples.Count -eq 60 -and $idleMaxRam -le 0.65 -and $idleMaxConsecutive -lt 3)
$idlePayload = [ordered]@{
    schema_version = "epoch6.schedule_closed_loop.idle_control.v1"
    completed_at = (Get-Date).ToString("o")
    execution_manifest_sha256 = $ExecutionManifestSha256
    resource_amendment_sha256 = $ResourceAmendmentSha256
    sample_count = $idleSamples.Count
    maximum_used_fraction = $idleMaxRam
    pagefile_current_usage_mib_min = [double](($idleSamples | ForEach-Object { [double]$_.pagefile_current_usage_mib } | Measure-Object -Minimum).Minimum)
    pagefile_current_usage_mib_max = [double](($idleSamples | ForEach-Object { [double]$_.pagefile_current_usage_mib } | Measure-Object -Maximum).Maximum)
    maximum_page_writes_per_sec = [int64](($idleSamples | ForEach-Object { [int64]$_.page_writes_per_sec } | Measure-Object -Maximum).Maximum)
    maximum_pages_output_per_sec = [int64](($idleSamples | ForEach-Object { [int64]$_.pages_output_per_sec } | Measure-Object -Maximum).Maximum)
    maximum_consecutive_paging_active_samples = $idleMaxConsecutive
    samples = $idleSamples
    valid = $idleValid
}
Write-JsonAtomic -Path $IdleOutput -Payload $idlePayload
if (-not $idleValid) { throw "Closed-loop idle control is not valid" }

$baseline = $idleSamples[$idleSamples.Count - 1]
$lockPayload = [ordered]@{
    status = "active"
    monitor_pid = $PID
    run_id = $RunId
    created_at = (Get-Date).ToString("o")
    protocol_sha256 = $ProtocolSha256
    execution_manifest_sha256 = $ExecutionManifestSha256
}
$lockBytes = [System.Text.Encoding]::UTF8.GetBytes(($lockPayload | ConvertTo-Json -Depth 3) + "`n")
$stream = [System.IO.File]::Open($HostLock, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
try { $stream.Write($lockBytes, 0, $lockBytes.Length) } finally { $stream.Dispose() }

$arguments = @(
    "-d", "Ubuntu-22.04", "--", "bash",
    "/mnt/c/Users/jiheo/tca_map/scripts/run_epoch6_schedule_closed_loop_smoke_wsl.sh",
    $RunRootWsl,
    $HostLockWsl
)
$process = $null
$hostCeilingTerminated = $false
$sustainedPagingTerminated = $false
try {
    $process = Start-Process -FilePath "wsl.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    $samples = New-Object System.Collections.Generic.List[object]
    $pagingConsecutive = 0
    while (-not $process.HasExited) {
        $sample = Get-HostSample
        $samples.Add($sample)
        if (Test-PagingActive -Sample $sample) { $pagingConsecutive += 1 } else { $pagingConsecutive = 0 }
        Write-JsonAtomic -Path $Heartbeat -Payload ([ordered]@{
            status = "running"
            updated_at = (Get-Date).ToString("o")
            child_pid = $process.Id
            sample_count = $samples.Count
            current = $sample
            paging_consecutive = $pagingConsecutive
        }) -Depth 6
        if ([double]$sample.used_fraction -gt 0.82) {
            $hostCeilingTerminated = $true
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            break
        }
        if ($pagingConsecutive -ge $SustainedPagingSamples) {
            $sustainedPagingTerminated = $true
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            break
        }
        Start-Sleep -Milliseconds 500
        $process.Refresh()
    }
    $process.WaitForExit()
    $process.Refresh()
    $childExitCode = $process.ExitCode
    if (Test-Path -LiteralPath $ExitFile) { $childExitCode = [int](Get-Content -Raw -LiteralPath $ExitFile).Trim() }
    $afterChild = Get-HostSample
    $samples.Add($afterChild)
    & wsl.exe -d Ubuntu-22.04 -u root -- sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
    Start-Sleep -Seconds 20
    $afterCacheDrop = Get-HostSample
    $samples.Add($afterCacheDrop)
    & wsl.exe --shutdown
    $restoreSamples = New-Object System.Collections.Generic.List[object]
    $cleanStateRestored = $false
    $restoreDeadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $restoreDeadline) {
        Start-Sleep -Seconds 1
        $restore = Get-HostSample
        $restoreSamples.Add($restore)
        $cleanStateRestored = [bool](
            @(Get-WslRunningNames).Count -eq 0 -and
            [double]$restore.used_fraction -le 0.65 -and
            [int64]$restore.gpu.used_mib -le ([int64]$baseline.gpu.used_mib + 256) -and
            (Get-MaxConsecutivePaging -Samples $restoreSamples.ToArray()) -lt 3
        )
        if ($cleanStateRestored) { break }
    }
    $afterRestore = $restoreSamples[$restoreSamples.Count - 1]
    $allRuntimeSamples = @($samples.ToArray()) + @($restoreSamples.ToArray())
    $peakRam = [double](($allRuntimeSamples | ForEach-Object { [double]$_.used_fraction } | Measure-Object -Maximum).Maximum)
    $peakPagefile = [double](($allRuntimeSamples | ForEach-Object { [double]$_.pagefile_current_usage_mib } | Measure-Object -Maximum).Maximum)
    $peakPageWrites = [int64](($allRuntimeSamples | ForEach-Object { [int64]$_.page_writes_per_sec } | Measure-Object -Maximum).Maximum)
    $peakPagesOutput = [int64](($allRuntimeSamples | ForEach-Object { [int64]$_.pages_output_per_sec } | Measure-Object -Maximum).Maximum)
    $runtimeMaxConsecutive = Get-MaxConsecutivePaging -Samples $samples.ToArray()
    $restoreMaxConsecutive = Get-MaxConsecutivePaging -Samples $restoreSamples.ToArray()
    $sustainedPagingDetected = [bool]($runtimeMaxConsecutive -ge 3 -or $restoreMaxConsecutive -ge 3)
    $pagefileGrowth = $peakPagefile - [double]$baseline.pagefile_current_usage_mib
    $stdoutText = if (Test-Path $Stdout) { Get-Content -Raw -LiteralPath $Stdout } else { "" }
    $stderrText = if (Test-Path $Stderr) { Get-Content -Raw -LiteralPath $Stderr } else { "" }
    $oomDetected = [bool](($stdoutText + "`n" + $stderrText) -match '(?i)out of memory|cuda[^\r\n]*oom|killed process|oom-kill')
    $internal = if (Test-Path $InternalOutput) { Get-Content -Raw -LiteralPath $InternalOutput | ConvertFrom-Json } else { $null }
    $internalValid = [bool](
        $null -ne $internal -and
        $internal.status -eq "CLOSED_LOOP_ACTUAL_PATH_RESOURCE_SMOKE_PASS" -and
        [int64]$internal.simultaneous_env_instances -eq 4 -and
        [int64]$internal.model_inference_calls -eq 1 -and
        [bool]$internal.raw_chunk_finite -and
        [int64]$internal.success_check_calls -eq 0 -and
        [int64]$internal.resource_monitor.maximum_swap_used_bytes -eq 0 -and
        @($internal.resource_monitor.exceptions).Count -eq 0 -and
        @($internal.runtime.parameter_devices).Count -eq 1 -and
        [string]$internal.runtime.parameter_devices[0] -eq "cuda:0" -and
        [bool]$internal.runtime.cpu_or_disk_model_offload -eq $false
    )
    if ($hostCeilingTerminated) {
        $decision = "EPOCH6_CLOSED_LOOP_RESOURCE_SMOKE_FAIL_HOST_CEILING"
    } elseif ($sustainedPagingTerminated -or $sustainedPagingDetected) {
        $decision = "EPOCH6_CLOSED_LOOP_RESOURCE_SMOKE_FAIL_SUSTAINED_PAGING"
    } elseif ($childExitCode -ne 0 -or -not $internalValid) {
        $decision = "EPOCH6_CLOSED_LOOP_RESOURCE_SMOKE_FAIL_CHILD_OR_INTERNAL"
    } elseif ($oomDetected) {
        $decision = "EPOCH6_CLOSED_LOOP_RESOURCE_SMOKE_FAIL_OOM_OR_KILL"
    } elseif (-not $cleanStateRestored) {
        $decision = "EPOCH6_CLOSED_LOOP_RESOURCE_SMOKE_FAIL_CLEAN_STATE_RESTORE"
    } else {
        $decision = "EPOCH6_CLOSED_LOOP_RESOURCE_SMOKE_PASS_CALIBRATED"
    }
    $payload = [ordered]@{
        schema_version = "epoch6.schedule_closed_loop.host_resource_smoke.v1"
        completed_at = (Get-Date).ToString("o")
        protocol_sha256 = $ProtocolSha256
        execution_manifest_sha256 = $ExecutionManifestSha256
        resource_amendment_sha256 = $ResourceAmendmentSha256
        monitor_script_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash
        idle_control_valid = $idleValid
        idle_control_report_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $IdleOutput).Hash
        child_exit_code = $childExitCode
        internal_valid = $internalValid
        baseline = $baseline
        peak_used_fraction = $peakRam
        peak_pagefile_current_usage_mib = $peakPagefile
        peak_page_writes_per_sec = $peakPageWrites
        peak_pages_output_per_sec = $peakPagesOutput
        pagefile_current_growth_mib = $pagefileGrowth
        pagefile_allocation_classification = if ($pagefileGrowth -gt 0 -and -not $sustainedPagingDetected) { "NONFATAL_ALLOCATION_ONLY_ABSENT_PRESSURE" } elseif ($pagefileGrowth -gt 0) { "ALLOCATION_WITH_PAGING_PRESSURE" } else { "NO_POSITIVE_ALLOCATION_DELTA" }
        maximum_runtime_consecutive_paging_active_samples = $runtimeMaxConsecutive
        maximum_restore_consecutive_paging_active_samples = $restoreMaxConsecutive
        sustained_paging_detected = $sustainedPagingDetected
        host_ceiling_terminated = $hostCeilingTerminated
        oom_or_kill_signature_detected = $oomDetected
        clean_state_restored = $cleanStateRestored
        after_child = $afterChild
        after_cache_drop = $afterCacheDrop
        after_clean_state_restore = $afterRestore
        samples = $samples
        restore_samples = $restoreSamples
        internal_report_sha256 = if ($null -ne $internal) { (Get-FileHash -Algorithm SHA256 -LiteralPath $InternalOutput).Hash } else { $null }
        final_decision = $decision
    }
    Write-JsonAtomic -Path $HostOutput -Payload $payload
    Write-JsonAtomic -Path $Heartbeat -Payload ([ordered]@{status="completed";updated_at=(Get-Date).ToString("o");final_decision=$decision}) -Depth 4
    [ordered]@{
        final_decision = $decision
        idle_control_valid = $idleValid
        child_exit_code = $childExitCode
        internal_valid = $internalValid
        peak_used_fraction = $peakRam
        pagefile_current_growth_mib = $pagefileGrowth
        sustained_paging_detected = $sustainedPagingDetected
        clean_state_restored = $cleanStateRestored
    } | ConvertTo-Json -Depth 3
    if ($decision -eq "EPOCH6_CLOSED_LOOP_RESOURCE_SMOKE_PASS_CALIBRATED") { exit 0 }
    exit 1
} finally {
    if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
    & wsl.exe --shutdown 2>$null
    if (Test-Path -LiteralPath $HostLock) {
        Move-Item -LiteralPath $HostLock -Destination (Join-Path $RunRoot ("released_host_resource_smoke_lock_" + [DateTimeOffset]::Now.ToUnixTimeSeconds() + ".json")) -ErrorAction SilentlyContinue
    }
}
