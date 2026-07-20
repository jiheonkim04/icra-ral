param(
    [string]$RunId = "run_20260720_2039_kst"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunParent = Join-Path $RepoRoot "reports\epoch8_two_shard_actual_arrival"
$RunDir = Join-Path $RunParent $RunId
$LockPath = Join-Path $RunParent "epoch8_two_shard_resource.global.lock.json"
$HostReport = Join-Path $RunDir "two_shard_resource_smoke_host.json"
$StdoutPath = Join-Path $RunDir "two_shard_resource_smoke.stdout.log"
$StderrPath = Join-Path $RunDir "two_shard_resource_smoke.stderr.log"
$HeartbeatPath = Join-Path $RunDir "two_shard_resource_heartbeat.json"
$InternalPath = Join-Path $RunDir "two_shard_resource_smoke.json"
$ProtocolHash = "4C5CD89AAE4C36B978FF3DBBD3658D2217A88115E19F3FACBFA60D18DA96904B"
$UsedFractionCeiling = 0.82

New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
if (Test-Path -LiteralPath $LockPath) {
    throw "Refusing to overwrite an existing resource lock: $LockPath"
}
if (Test-Path -LiteralPath $HostReport) {
    throw "Refusing to overwrite an existing host report: $HostReport"
}

$lock = [ordered]@{
    schema_version = "epoch8.two_shard_actual_arrival.resource_lock.v1"
    status = "active"
    run_id = $RunId
    protocol_sha256 = $ProtocolHash
    monitor_pid = $PID
    created_at = (Get-Date).ToString("o")
}
$lock | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $LockPath -Encoding utf8

$samples = New-Object System.Collections.Generic.List[object]
$peakUsedFraction = 0.0
$peakPageWrites = 0.0
$peakPagesOutput = 0.0
$consecutivePaging = 0
$maximumConsecutivePaging = 0
$ceilingBreached = $false
$child = $null
$childExitCode = $null
$monitorException = $null

try {
    $arguments = @(
        "-d", "Ubuntu-22.04", "--",
        "/usr/bin/env",
        "EPOCH8_TWO_SHARD_RESOURCE_LOCK=/mnt/c/Users/jiheo/tca_map/reports/epoch8_two_shard_actual_arrival/epoch8_two_shard_resource.global.lock.json",
        "/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python",
        "/mnt/c/Users/jiheo/tca_map/scripts/run_epoch8_two_shard_actual_arrival.py",
        "--mode", "resource-smoke",
        "--run-dir", "/mnt/c/Users/jiheo/tca_map/reports/epoch8_two_shard_actual_arrival/$RunId",
        "--child"
    )
    $child = Start-Process -FilePath "wsl.exe" -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
    while (-not $child.HasExited) {
        $os = Get-CimInstance Win32_OperatingSystem
        $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
        $usedFraction = 1.0 - ([double]$os.FreePhysicalMemory / [double]$os.TotalVisibleMemorySize)
        $pageWrites = [double]$memory.PageWritesPersec
        $pagesOutput = [double]$memory.PagesOutputPersec
        $peakUsedFraction = [math]::Max($peakUsedFraction, $usedFraction)
        $peakPageWrites = [math]::Max($peakPageWrites, $pageWrites)
        $peakPagesOutput = [math]::Max($peakPagesOutput, $pagesOutput)
        if ($pageWrites -gt 0 -or $pagesOutput -gt 0) {
            $consecutivePaging += 1
        } else {
            $consecutivePaging = 0
        }
        $maximumConsecutivePaging = [math]::Max($maximumConsecutivePaging, $consecutivePaging)
        $samples.Add([ordered]@{
            captured_at = (Get-Date).ToString("o")
            used_fraction = $usedFraction
            page_writes_per_sec = $pageWrites
            pages_output_per_sec = $pagesOutput
        })
        if ($usedFraction -gt $UsedFractionCeiling) {
            $ceilingBreached = $true
            $linuxPid = $null
            if (Test-Path -LiteralPath $HeartbeatPath) {
                try {
                    $linuxPid = (Get-Content -Raw -LiteralPath $HeartbeatPath | ConvertFrom-Json).pid
                } catch {
                    $linuxPid = $null
                }
            }
            if ($null -ne $linuxPid) {
                wsl.exe -d Ubuntu-22.04 -- bash -lc "pkill -TERM -P $linuxPid 2>/dev/null || true; kill -TERM $linuxPid 2>/dev/null || true" | Out-Null
            }
            Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue
            break
        }
        Start-Sleep -Milliseconds 500
        $child.Refresh()
    }
    if (-not $child.HasExited) {
        $child.WaitForExit(10000) | Out-Null
    } else {
        $child.WaitForExit()
    }
    $child.Refresh()
    if ($child.HasExited) {
        $childExitCode = $child.ExitCode
    }
} catch {
    $monitorException = $_.Exception.Message
    if ($null -ne $child -and -not $child.HasExited) {
        Stop-Process -Id $child.Id -Force -ErrorAction SilentlyContinue
    }
} finally {
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
}

$internal = $null
$internalHash = $null
if (Test-Path -LiteralPath $InternalPath) {
    $internal = Get-Content -Raw -LiteralPath $InternalPath | ConvertFrom-Json
    $internalHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $InternalPath).Hash
}
$sustainedPaging = $maximumConsecutivePaging -ge 3
$pass = (
    -not $ceilingBreached -and
    -not $sustainedPaging -and
    $null -eq $monitorException -and
    $childExitCode -eq 0 -and
    $null -ne $internal -and
    $internal.status -eq "TWO_SHARD_ACTUAL_PATH_RESOURCE_SMOKE_PASS" -and
    $internal.resource_monitor.maximum_swap_used_bytes -eq 0 -and
    $internal.resource_monitor.exceptions.Count -eq 0
)
$report = [ordered]@{
    schema_version = "epoch8.two_shard_actual_arrival.host_resource_smoke.v1"
    completed_at = (Get-Date).ToString("o")
    run_id = $RunId
    protocol_sha256 = $ProtocolHash
    child_exit_code = $childExitCode
    peak_used_fraction = $peakUsedFraction
    used_fraction_ceiling = $UsedFractionCeiling
    ceiling_breached = $ceilingBreached
    peak_page_writes_per_sec = $peakPageWrites
    peak_pages_output_per_sec = $peakPagesOutput
    maximum_consecutive_paging_samples = $maximumConsecutivePaging
    sustained_paging_detected = $sustainedPaging
    sample_count = $samples.Count
    monitor_exception = $monitorException
    internal_report_sha256 = $internalHash
    final_decision = $(if ($pass) { "TWO_SHARD_RESOURCE_SMOKE_PASS" } else { "TWO_SHARD_RESOURCE_BLOCKED_CURRENT_HOST" })
    samples = $samples
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $HostReport -Encoding utf8
[pscustomobject]$report | Select-Object final_decision,peak_used_fraction,ceiling_breached,sustained_paging_detected,child_exit_code,sample_count
if (-not $pass) {
    exit 1
}
