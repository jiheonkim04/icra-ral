param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("rollout-base", "rollout-prior")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z0-9_]+$")]
    [string]$RunId,

    [ValidateRange(1, 20)]
    [int]$Attempt = 1,

    [string]$RepoRoot = "C:\Users\jiheo\tca_map"
)

$ErrorActionPreference = "Stop"
$WslConfig = "C:\Users\jiheo\.wslconfig"
$RunRoot = Join-Path $RepoRoot "runs\a2c2_prior\clean_host_panel\$RunId"
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
    throw "The validated 12GB/swap0/WSLg-off configuration is not active"
}
$configHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $WslConfig).Hash
if ($configHash -ne "A7CC4F707936DBBCE335F298BF0E968804F956E4DBE71A17ECEF5775CC708445") {
    throw "Unexpected validated .wslconfig hash: $configHash"
}

$staleWorkers = @(Get-Process | Where-Object { $_.ProcessName -in @("python", "python3", "wsl") })
if ($staleWorkers.Count -gt 0) {
    throw "Stale research-capable worker exists before panel launch: $($staleWorkers.ProcessName -join ',')"
}
$heavyPattern = "League|RiotClient|VALORANT|Shipping|steam|Discord|chrome|Battle.net|Overwatch|Fortnite|Minecraft|Genshin|Roblox|OP.GG|GGQ|OneDrive|Dropbox|GoogleDrive"
$heavy = @(Get-Process | Where-Object { $_.ProcessName -match $heavyPattern } | Select-Object Id, ProcessName)
if ($heavy.Count -gt 0) {
    throw "Background-heavy process became active before panel launch: $($heavy.ProcessName -join ',')"
}

$baselineFirst = Get-HostMemorySample
Start-Sleep -Seconds 2
$baseline = Get-HostMemorySample
if ([double]$baseline.used_fraction -gt 0.65) {
    throw "A2C2_CLEAN_HOST_BASELINE_UNSAFE: $($baseline.used_fraction)"
}

$stage = if ($Mode -eq "rollout-base") { "base" } else { "prior" }
$stdout = Join-Path $RunRoot "${stage}_attempt_${Attempt}.stdout.log"
$stderr = Join-Path $RunRoot "${stage}_attempt_${Attempt}.stderr.log"
$hostOutput = Join-Path $RunRoot "${stage}_attempt_${Attempt}_host.json"
$partialPath = Join-Path $RunRoot "${stage}_rollout_partial.json"
$resultPath = Join-Path $RepoRoot "reports\a2c2_prior\${RunId}_${stage}_closed_loop_result.json"
if (Test-Path -LiteralPath $hostOutput) {
    throw "Stale panel host report must be preserved and not overwritten: $hostOutput"
}

$argumentList = @(
    "-d", "Ubuntu-22.04", "--", "bash",
    "/mnt/c/Users/jiheo/tca_map/scripts/run_a2c2_clean_host_panel_wsl.sh",
    "$Mode", "$RunId"
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
    Start-Sleep -Milliseconds 750
    $process.Refresh()
}
$process.WaitForExit()
$process.Refresh()
$rawExitCode = $process.ExitCode
$afterChild = Get-HostMemorySample
$samples.Add($afterChild)

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

foreach ($sample in @($afterChild, $postShutdown)) {
    $peakUsedFraction = [math]::Max($peakUsedFraction, [double]$sample.used_fraction)
    $peakCommittedBytes = [math]::Max($peakCommittedBytes, [int64]$sample.committed_bytes)
    $peakPagefileCurrentMiB = [math]::Max($peakPagefileCurrentMiB, [double]$sample.pagefile_current_usage_mib)
    $peakPageWritesPerSec = [math]::Max($peakPageWritesPerSec, [int64]$sample.page_writes_per_sec)
}

$partial = if (Test-Path -LiteralPath $partialPath) { Get-Content -LiteralPath $partialPath -Raw | ConvertFrom-Json } else { $null }
$result = if (Test-Path -LiteralPath $resultPath) { Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json } else { $null }
$expectedDecision = if ($stage -eq "base") { "A2C2_BASE_CLOSED_LOOP_ACCEPTED" } else { "A2C2_PRIOR_CLOSED_LOOP_ACCEPTED" }
$stageAccepted = [bool]($null -ne $result -and $result.final_decision -eq $expectedDecision)
$memoryReleaseVerified = [double]$postShutdown.used_fraction -le ([double]$baseline.used_fraction + 0.05)

$payload = [ordered]@{
    schema_version = 1
    date_kst = (Get-Date).ToString("o")
    authority_state = "A2C2_CLEAN_HOST_RESOURCE_FEASIBILITY_REOPENED"
    job_classification = "VLA_CLOSED_LOOP_ROLLOUT"
    run_id = $RunId
    stage = $stage
    attempt = $Attempt
    wsl_cap_gib = 12
    wslconfig_sha256 = $configHash
    command_pid = $process.Id
    command_exit_code_raw = $rawExitCode
    host_ceiling_terminated = $hostCeilingTerminated
    execution = [ordered]@{
        policy_processes = 1
        full_backbone_instances = 1
        prior_module_instances = if ($stage -eq "prior") { 1 } else { 0 }
        simultaneous_full_backbones = 1
        eval_batch_size = 1
        parallel_tasks = 1
        live_environments = 1
        concurrency_mode = "SEQUENTIAL"
        prefetch_mode = "DISABLED"
    }
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
    pagefile_current_growth_mib = $peakPagefileCurrentMiB - [double]$baseline.pagefile_current_usage_mib
    memory_release_verified = $memoryReleaseVerified
    sample_count = $samples.Count
    samples = $samples
    wsl_meminfo = @($meminfo)
    kernel_oom_lines = @($dmesg)
    partial_path = $partialPath
    result_path = $resultPath
    completed_episode_rows = if ($null -ne $partial) { [int]$partial.completed_episode_rows } else { 0 }
    planned_episode_rows = if ($null -ne $partial) { [int]$partial.planned_episode_rows } else { if ($stage -eq "base") { 30 } else { 15 } }
    stage_result_decision = if ($null -ne $result) { $result.final_decision } else { $null }
    stage_accepted = $stageAccepted
    stdout_path = $stdout
    stderr_path = $stderr
    final_decision = if ($hostCeilingTerminated) { "A2C2_PANEL_HOST_CEILING_INTERRUPTED_RESUME_MISSING_KEYS" } elseif ($stageAccepted) { "A2C2_PANEL_STAGE_ACCEPTED" } else { "A2C2_PANEL_STAGE_INCOMPLETE_OR_INVALID" }
}
$payload | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath $hostOutput -Encoding utf8
$payload | ConvertTo-Json -Depth 9

if ($stageAccepted -and -not $hostCeilingTerminated) {
    exit 0
}
if ($hostCeilingTerminated) {
    exit 82
}
exit 1
