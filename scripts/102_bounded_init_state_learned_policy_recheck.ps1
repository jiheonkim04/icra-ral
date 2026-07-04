param(
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$TaskSuite = "libero_10",
    [int]$StartTaskId = 0,
    [int]$TaskCount = 1,
    [int]$MaxStepsPerTask = 3,
    [int]$CameraSize = 64,
    [string]$ActionAdapterStrategy = "policy_6d_delta_pose_plus_gripper_close",
    [double]$ActionScale = 1.0,
    [int]$TimeoutSeconds = 1800,
    [string]$JsonReportPath = "reports\init_state_learned_policy_recheck_report.json",
    [string]$MarkdownReportPath = "reports\init_state_learned_policy_recheck_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded init-state learned-policy LIBERO recheck"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script runs one diagnostic learned-policy rollout from a local HDF5 init_state only. It does not train, run GPU jobs, download assets, execute OpenVLA-OFT, run multi-seed evaluation, or make benchmark/SOTA/paper-grade claims."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

function Invoke-SafeCommand {
    param([string[]]$Command, [int]$TimeoutSec = 60)
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Command[0]
        if ($Command.Count -gt 1) {
            $psi.Arguments = (($Command[1..($Command.Count - 1)] | ForEach-Object {
                if ($_ -match '[\s"]') { '"' + ($_.Replace('"', '\"')) + '"' } else { $_ }
            }) -join " ")
        }
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
        $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
        $process = [System.Diagnostics.Process]::Start($psi)
        $completed = $process.WaitForExit($TimeoutSec * 1000)
        if (-not $completed) {
            try { $process.Kill() } catch {}
            return [ordered]@{ ok = $false; timed_out = $true; returncode = $null; stdout = ""; stderr = "command timed out after $TimeoutSec seconds" }
        }
        return [ordered]@{
            ok = $process.ExitCode -eq 0
            timed_out = $false
            returncode = $process.ExitCode
            stdout = $process.StandardOutput.ReadToEnd().Trim().Replace("`0", "")
            stderr = $process.StandardError.ReadToEnd().Trim().Replace("`0", "")
        }
    } catch {
        return [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = $_.Exception.Message }
    }
}

function Read-JsonFile {
    param([string]$Path)
    $fullPath = Resolve-RepoPath -Path $Path
    $text = [System.IO.File]::ReadAllText($fullPath, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
    return $text | ConvertFrom-Json
}

function Write-Reports {
    param([object]$Report)
    $jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
    $markdownFullPath = Resolve-RepoPath -Path $MarkdownReportPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# Bounded Init-State Learned-Policy Recheck Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.bounded_init_state_learned_policy_recheck_passed)",
        "- model load performed: $($Report.policy.model_load_performed)",
        "- learned policy inference performed: $($Report.policy.learned_policy_inference_performed)",
        "- diagnostic rollouts performed: $($Report.policy.diagnostic_rollouts_performed)",
        "- HDF5 init state set in environment: $($Report.policy.hdf5_init_state_set_in_environment)",
        "- task count: $($Report.policy.task_count)",
        "- max steps per task: $($Report.policy.max_steps_per_task)",
        "- benchmark rollouts performed: false",
        "- paper-grade claim: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is init-state learned-policy diagnostic evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 14
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_init_state_learned_policy_recheck = $true
            task_local_gate_required = "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1"
            task_suite = $TaskSuite
            start_task_id = $StartTaskId
            task_count = $TaskCount
            max_steps_per_task = $MaxStepsPerTask
            camera_size = $CameraSize
            action_adapter_strategy = $ActionAdapterStrategy
            action_scale = $ActionScale
            downloads_performed = $false
            installs_performed = $false
            gpu_jobs_performed = $false
            training_performed = $false
            heavy_model_imports_performed = $false
            model_load_performed = $false
            model_inference_performed = $false
            learned_policy_inference_performed = $false
            simulator_environment_created = $false
            diagnostic_rollouts_performed = $false
            benchmark_rollouts_performed = $false
            multi_seed_performed = $false
            openvla_oft_executed = $false
            tokens_read_or_written = $false
            benchmark_claims_made = $false
            sota_claims_made = $false
            paper_grade_claims_made = $false
            hdf5_init_state_set_in_environment = $false
        }
        plan = $null
        wsl = [ordered]@{}
        rollout_command = $null
        rollout_result_raw = $null
        rollout_result = $null
        bounded_init_state_learned_policy_recheck_passed = $false
        ready_for_rollout_scaling = $false
        ready_for_paper_claim = $false
        decision = "stop"
        reason = $null
        recommended_next_step = $null
    }
}

$dangerousGateNames = @(
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
    "ALLOW_HDF5_REPLAY_DIAGNOSTIC"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })

$report = New-BaseReport
if ([Environment]::GetEnvironmentVariable("ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK") -ne "1") {
    $report.reason = "ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1 is required after a green init-state learned-policy recheck plan."
    $report.recommended_next_step = "Run scripts\101_plan_init_state_learned_policy_recheck.ps1 first, then set ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1 only for this bounded task if the plan says proceed."
    Write-Reports -Report $report
    exit 0
}
if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "init-state learned-policy recheck refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset broad rollout, benchmark, OpenVLA, GPU, tiny-rollout, matrix, HDF5 replay, and training gates before this runner."
    Write-Reports -Report $report
    exit 0
}
if ($TaskCount -ne 1 -or $MaxStepsPerTask -lt 1 -or $MaxStepsPerTask -gt 5) {
    $report.reason = "init-state learned-policy recheck allows exactly one task and 1..5 steps per task."
    $report.recommended_next_step = "Use TaskCount=1 and MaxStepsPerTask<=5 for this recheck rung."
    Write-Reports -Report $report
    exit 0
}

$runDir = Join-Path $RepoRoot "runs\init_state_learned_policy_recheck"
$planJson = Join-Path $runDir "plan_report.json"
$planMd = Join-Path $runDir "plan_report.md"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$savedGate = [Environment]::GetEnvironmentVariable("ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK")
Remove-Item Env:\ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK -ErrorAction SilentlyContinue
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\101_plan_init_state_learned_policy_recheck.ps1") `
        -TaskCount $TaskCount `
        -MaxStepsPerTask $MaxStepsPerTask `
        -JsonReportPath $planJson `
        -MarkdownReportPath $planMd | Out-Null
} finally {
    if ($null -ne $savedGate) { $env:ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK = $savedGate }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planJson)) {
    $report.reason = "init-state learned-policy recheck planner failed"
    $report.recommended_next_step = "Fix the planner before bounded init-state recheck."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planJson
$report.plan = $plan
if (-not $plan.ready_for_bounded_init_state_learned_policy_recheck_runner) {
    $report.reason = "planner did not authorize bounded init-state learned-policy recheck"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}
if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Configure WSL before bounded init-state recheck."
    Write-Reports -Report $report
    exit 0
}

$hdf5Path = [string]$plan.prerequisites.hdf5_replay.hdf5_path
if ([string]::IsNullOrWhiteSpace($hdf5Path)) {
    $report.reason = "planner did not report an HDF5 init-state path"
    $report.recommended_next_step = "Regenerate the HDF5 replay report before bounded init-state recheck."
    Write-Reports -Report $report
    exit 0
}

$repoWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($RepoRoot.Replace("\", "/"))) -TimeoutSec 30
$smolvlaWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/checkpoints/smolvla") -TimeoutSec 30
$checkpointRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/checkpoints") -TimeoutSec 30
$hfHomeWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/hf_home") -TimeoutSec 30
$liberoRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/LIBERO") -TimeoutSec 30
$robosuiteRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/robosuite") -TimeoutSec 30
$liberoDataRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/data/libero") -TimeoutSec 30
$reportPathWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $JsonReportPath).Replace("\", "/"))) -TimeoutSec 30
$pythonProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "test -x $WslPython && $WslPython --version") -TimeoutSec 30
$stdoutLogWin = Join-Path $runDir "runner_stdout.log"
$stdoutLogWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($stdoutLogWin.Replace("\", "/"))) -TimeoutSec 30
if ($hdf5Path -like "/mnt/*") {
    $hdf5PathWsl = [ordered]@{ ok = $true; stdout = $hdf5Path; stderr = ""; returncode = 0; timed_out = $false }
} else {
    $hdf5PathWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($hdf5Path.Replace("\", "/"))) -TimeoutSec 30
}
$report.wsl = [ordered]@{
    repo_root = $repoWsl
    smolvla_ckpt = $smolvlaWsl
    checkpoint_root = $checkpointRootWsl
    hf_home = $hfHomeWsl
    libero_root = $liberoRootWsl
    robosuite_root = $robosuiteRootWsl
    libero_data_root = $liberoDataRootWsl
    hdf5_init_state_path = $hdf5PathWsl
    report_path = $reportPathWsl
    stdout_log_path = $stdoutLogWsl
    python = $pythonProbe
}
if (-not ($repoWsl.ok -and $smolvlaWsl.ok -and $checkpointRootWsl.ok -and $hfHomeWsl.ok -and $liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $liberoDataRootWsl.ok -and $hdf5PathWsl.ok -and $reportPathWsl.ok -and $stdoutLogWsl.ok -and $pythonProbe.ok)) {
    $report.reason = "WSL path or Python probe failed"
    $report.recommended_next_step = "Fix WSL path/Python readiness before bounded init-state learned-policy recheck."
    Write-Reports -Report $report
    exit 0
}

$innerReportPath = Resolve-RepoPath -Path $JsonReportPath
Remove-Item -LiteralPath $innerReportPath -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stdoutLogWin -ErrorAction SilentlyContinue
$bashCommand = @"
export PYTHONPATH='$($repoWsl.stdout)';
export HF_HUB_OFFLINE=1;
export TRANSFORMERS_OFFLINE=1;
export MUJOCO_GL=osmesa;
export ALLOW_INIT_STATE_LEARNED_POLICY_RECHECK=1;
$WslPython -m tca_map.smolvla.libero_learned_policy_rollout --smolvla-ckpt '$($smolvlaWsl.stdout)' --checkpoint-root '$($checkpointRootWsl.stdout)' --hf-home '$($hfHomeWsl.stdout)' --libero-root '$($liberoRootWsl.stdout)' --robosuite-root '$($robosuiteRootWsl.stdout)' --libero-data-root '$($liberoDataRootWsl.stdout)' --task-suite '$TaskSuite' --start-task-id $StartTaskId --task-count $TaskCount --max-steps-per-task $MaxStepsPerTask --camera-size $CameraSize --device cpu --action-adapter-strategy '$ActionAdapterStrategy' --action-scale $ActionScale --hdf5-init-state-path '$($hdf5PathWsl.stdout)' --require-hdf5-init-state --report-path '$($reportPathWsl.stdout)' > '$($stdoutLogWsl.stdout)' 2>&1
printf 'bounded init-state learned-policy recheck command finished\n'
"@
$bashCommand = $bashCommand -replace "`r", ""
$rollout = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
$report.rollout_command = [ordered]@{ timeout_seconds = $TimeoutSeconds; device = "cpu"; task_suite = $TaskSuite; task_count = $TaskCount; max_steps_per_task = $MaxStepsPerTask; action_adapter_strategy = $ActionAdapterStrategy; hdf5_init_state_path = $hdf5PathWsl.stdout }
$report.rollout_result_raw = $rollout

if (Test-Path -LiteralPath $innerReportPath) {
    try {
        $inner = Read-JsonFile -Path $JsonReportPath
        $report.rollout_result = $inner
        $passed = [bool]$inner.result.passed
        $report.bounded_init_state_learned_policy_recheck_passed = $passed
        $report.policy.heavy_model_imports_performed = [bool]$inner.policy.heavy_model_imports_performed
        $report.policy.model_load_performed = [bool]$inner.policy.model_load_performed
        $report.policy.model_inference_performed = [bool]$inner.policy.model_inference_performed
        $report.policy.learned_policy_inference_performed = [bool]$inner.policy.learned_policy_inference_performed
        $report.policy.simulator_environment_created = [bool]$inner.policy.simulator_environment_created
        $report.policy.diagnostic_rollouts_performed = [bool]$inner.policy.diagnostic_rollouts_performed
        $report.policy.hdf5_init_state_set_in_environment = [bool]$inner.hdf5_init_state.set_in_environment
        $report.decision = if ($passed) { "proceed" } else { "stop" }
        $report.reason = if ($passed) { "bounded init-state learned-policy LIBERO recheck passed" } else { [string]$inner.result.blocked_reason }
    } catch {
        $report.reason = "failed to parse inner init-state learned-policy recheck report: $($_.Exception.Message)"
    }
} else {
    $report.reason = "bounded init-state learned-policy recheck command failed or timed out"
}

$report.ready_for_rollout_scaling = $false
$report.recommended_next_step = if ($report.bounded_init_state_learned_policy_recheck_passed) {
    "Generate a diagnostic metric summary comparing init-state recheck against prior reset-only learned-policy results. Keep evidence diagnostic/local-pilot only."
} else {
    "Inspect the init-state learned-policy recheck blocker before any larger matrix or claims."
}

Write-Reports -Report $report
exit 0
