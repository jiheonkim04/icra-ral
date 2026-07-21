param()

$ErrorActionPreference = "Stop"
$repo = "C:\Users\jiheo\tca_map"
$outputDir = Join-Path $repo "reports\epoch9e_joint_certification"
$monitorPath = Join-Path $outputDir "host_resource_monitor.json"
$stdoutPath = Join-Path $outputDir "runner_stdout.log"
$stderrPath = Join-Path $outputDir "runner_stderr.log"
$resultPath = Join-Path $outputDir "result.json"
$hostCeilingPercent = 82.0

if (Test-Path -LiteralPath $monitorPath) {
    throw "Refusing to overwrite $monitorPath"
}
if (Test-Path -LiteralPath $resultPath) {
    throw "Refusing to overwrite or resume the one-shot joint result"
}
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$existing = & wsl.exe -e bash -lc "pgrep -af '[r]un_epoch9e_joint_certification.py' || true"
if ($existing) {
    throw "An Epoch 9E joint runner is already active: $existing"
}

function Get-HostSample {
    $os = Get-CimInstance Win32_OperatingSystem
    $totalBytes = [int64]$os.TotalVisibleMemorySize * 1024
    $freeBytes = [int64]$os.FreePhysicalMemory * 1024
    $usedBytes = $totalBytes - $freeBytes
    $gpuUsed = 0.0
    try {
        $gpuRows = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpuRows) {
            $gpuUsed = [double](($gpuRows | ForEach-Object { [double]$_.Trim() } | Measure-Object -Sum).Sum)
        }
    } catch {
        $gpuUsed = 0.0
    }
    return [pscustomobject]@{
        timestamp = (Get-Date -Format o)
        host_used_physical_bytes = $usedBytes
        host_ram_percent = 100.0 * $usedBytes / $totalBytes
        gpu_used_mib_system_wide = $gpuUsed
    }
}

$bashCommand = "set -e; cd /mnt/c/Users/jiheo/tca_map; export MUJOCO_GL=egl; export PYTHONUNBUFFERED=1; exec /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_epoch9e_joint_certification.py"
$argumentString = "-e bash -lc `"$bashCommand`""
$baseline = Get-HostSample
$peak = $baseline
$peakGpu = $baseline.gpu_used_mib_system_wide
$ceilingBreached = $false
$terminationRequested = $false
$startedAt = Get-Date -Format o
$process = Start-Process -FilePath "wsl.exe" -ArgumentList $argumentString -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

while (-not $process.HasExited) {
    Start-Sleep -Seconds 1
    $process.Refresh()
    $sample = Get-HostSample
    if ($sample.host_used_physical_bytes -gt $peak.host_used_physical_bytes) {
        $peak = $sample
    }
    if ($sample.gpu_used_mib_system_wide -gt $peakGpu) {
        $peakGpu = $sample.gpu_used_mib_system_wide
    }
    if ($sample.host_ram_percent -ge $hostCeilingPercent) {
        $ceilingBreached = $true
        $terminationRequested = $true
        $terminateCommand = 'pids=$(pgrep -f ''^/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_epoch9e_joint_certification.py'' || true); if [ -n "$pids" ]; then kill -TERM $pids; fi'
        & wsl.exe -e bash -lc $terminateCommand | Out-Null
        break
    }
}

if ($terminationRequested -and -not $process.HasExited) {
    try {
        Wait-Process -Id $process.Id -Timeout 15
    } catch {
        $killCommand = 'pids=$(pgrep -f ''^/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_epoch9e_joint_certification.py'' || true); if [ -n "$pids" ]; then kill -KILL $pids; fi'
        & wsl.exe -e bash -lc $killCommand | Out-Null
        Wait-Process -Id $process.Id -Timeout 10
    }
}

$process.WaitForExit()
$final = Get-HostSample
$resultHash = if (Test-Path -LiteralPath $resultPath) { (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath).Hash } else { $null }
$monitor = [ordered]@{
    schema_version = "epoch9e.joint_host_resource_monitor.v1"
    started_at = $startedAt
    completed_at = (Get-Date -Format o)
    command = $bashCommand
    one_shot = $true
    resume = $false
    runner_exit_code = [int]$process.ExitCode
    baseline_host_ram_percent = [double]$baseline.host_ram_percent
    peak_host_ram_percent = [double]$peak.host_ram_percent
    final_host_ram_percent = [double]$final.host_ram_percent
    peak_host_used_physical_bytes = [int64]$peak.host_used_physical_bytes
    baseline_gpu_used_mib = [double]$baseline.gpu_used_mib_system_wide
    peak_gpu_used_mib = [double]$peakGpu
    final_gpu_used_mib = [double]$final.gpu_used_mib_system_wide
    host_ram_ceiling_percent = $hostCeilingPercent
    host_ram_ceiling_breached = $ceilingBreached
    targeted_runner_termination_requested = $terminationRequested
    scientific_result_path = "reports/epoch9e_joint_certification/result.json"
    scientific_result_sha256_after_runner = $resultHash
    stdout_path = "reports/epoch9e_joint_certification/runner_stdout.log"
    stdout_sha256 = if (Test-Path -LiteralPath $stdoutPath) { (Get-FileHash -Algorithm SHA256 -LiteralPath $stdoutPath).Hash } else { $null }
    stderr_path = "reports/epoch9e_joint_certification/runner_stderr.log"
    stderr_sha256 = if (Test-Path -LiteralPath $stderrPath) { (Get-FileHash -Algorithm SHA256 -LiteralPath $stderrPath).Hash } else { $null }
}
$temporary = "$monitorPath.tmp"
$monitor | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
Move-Item -LiteralPath $temporary -Destination $monitorPath
$monitor | ConvertTo-Json -Depth 8
if ($process.ExitCode -ne 0 -or $ceilingBreached) {
    exit 1
}
exit 0
