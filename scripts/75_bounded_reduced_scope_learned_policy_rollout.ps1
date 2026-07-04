param(
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$TaskSuite = "libero_10",
    [int]$StartTaskId = 0,
    [int]$TaskCount = 1,
    [int]$MaxStepsPerTask = 10,
    [int]$CameraSize = 64,
    [int]$TimeoutSeconds = 1800,
    [string]$JsonReportPath = "reports\bounded_reduced_scope_learned_policy_rollout_report.json",
    [string]$MarkdownReportPath = "reports\bounded_reduced_scope_learned_policy_rollout_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded reduced-scope learned-policy LIBERO rollout"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script runs a one-task longer diagnostic rollout only. It does not train, run GPU jobs, download assets, execute OpenVLA-OFT, run multi-seed evaluation, or make benchmark/SOTA/paper-grade claims."

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
        "# Bounded Reduced-Scope Learned-Policy Rollout Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.bounded_reduced_scope_learned_policy_rollout_passed)",
        "- model load performed: $($Report.policy.model_load_performed)",
        "- learned policy inference performed: $($Report.policy.learned_policy_inference_performed)",
        "- diagnostic rollouts performed: $($Report.policy.diagnostic_rollouts_performed)",
        "- task count: $($Report.policy.task_count)",
        "- max steps per task: $($Report.policy.max_steps_per_task)",
        "- benchmark rollouts performed: false",
        "- paper-grade claim: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is reduced-scope learned-policy diagnostic evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 14
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_reduced_scope_learned_policy_rollout = $true
            task_local_gate_required = "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX=1"
            task_suite = $TaskSuite
            start_task_id = $StartTaskId
            task_count = $TaskCount
            max_steps_per_task = $MaxStepsPerTask
            camera_size = $CameraSize
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
        }
        plan = $null
        wsl = [ordered]@{}
        rollout_command = $null
        rollout_result_raw = $null
        rollout_result = $null
        bounded_reduced_scope_learned_policy_rollout_passed = $false
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
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })

$report = New-BaseReport
if ([Environment]::GetEnvironmentVariable("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX") -ne "1") {
    $report.reason = "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX=1 is required after a reduced-scope matrix plan."
    $report.recommended_next_step = "Run scripts\74_plan_bounded_learned_policy_rollout_matrix.ps1 first, then set ALLOW_BOUNDED_LEARNED_POLICY_MATRIX=1 only for this bounded task if the plan says reduce_scope or proceed."
    Write-Reports -Report $report
    exit 0
}
if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "reduced-scope learned-policy rollout refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset broad rollout, benchmark, OpenVLA, GPU, tiny-rollout, and training gates before this runner."
    Write-Reports -Report $report
    exit 0
}
if ($TaskCount -ne 1 -or $MaxStepsPerTask -gt 10 -or $MaxStepsPerTask -lt 1) {
    $report.reason = "reduced-scope runner allows exactly one task and 1..10 steps per task."
    $report.recommended_next_step = "Use TaskCount=1 and MaxStepsPerTask<=10 for this reduced-scope rung."
    Write-Reports -Report $report
    exit 0
}

$planJson = Join-Path $RepoRoot "runs\bounded_learned_policy_rollout_matrix\plan_report.json"
$planMd = Join-Path $RepoRoot "runs\bounded_learned_policy_rollout_matrix\plan_report.md"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $planJson) | Out-Null
$savedGate = [Environment]::GetEnvironmentVariable("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX")
Remove-Item Env:\ALLOW_BOUNDED_LEARNED_POLICY_MATRIX -ErrorAction SilentlyContinue
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\74_plan_bounded_learned_policy_rollout_matrix.ps1") `
        -ReducedScopeTasks $TaskCount `
        -MaxStepsPerTask $MaxStepsPerTask `
        -JsonReportPath $planJson `
        -MarkdownReportPath $planMd | Out-Null
} finally {
    if ($null -ne $savedGate) { $env:ALLOW_BOUNDED_LEARNED_POLICY_MATRIX = $savedGate }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planJson)) {
    $report.reason = "bounded rollout matrix planner failed"
    $report.recommended_next_step = "Fix the planner before reduced-scope rollout."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planJson
$report.plan = $plan
if (-not $plan.ready_for_reduced_scope_learned_policy_runner) {
    $report.reason = "planner did not authorize reduced-scope learned-policy rollout"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}
if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Configure WSL before reduced-scope learned-policy rollout."
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
$stdoutLogWin = Join-Path $RepoRoot "runs\bounded_learned_policy_rollout_matrix\runner_stdout.log"
$stdoutLogWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($stdoutLogWin.Replace("\", "/"))) -TimeoutSec 30
$report.wsl = [ordered]@{
    repo_root = $repoWsl
    smolvla_ckpt = $smolvlaWsl
    checkpoint_root = $checkpointRootWsl
    hf_home = $hfHomeWsl
    libero_root = $liberoRootWsl
    robosuite_root = $robosuiteRootWsl
    libero_data_root = $liberoDataRootWsl
    report_path = $reportPathWsl
    stdout_log_path = $stdoutLogWsl
    python = $pythonProbe
}
if (-not ($repoWsl.ok -and $smolvlaWsl.ok -and $checkpointRootWsl.ok -and $hfHomeWsl.ok -and $liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $liberoDataRootWsl.ok -and $reportPathWsl.ok -and $stdoutLogWsl.ok -and $pythonProbe.ok)) {
    $report.reason = "WSL path or Python probe failed"
    $report.recommended_next_step = "Fix WSL path/Python readiness before reduced-scope learned-policy rollout."
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
export ALLOW_BOUNDED_LEARNED_POLICY_MATRIX=1;
$WslPython -m tca_map.smolvla.libero_learned_policy_rollout --smolvla-ckpt '$($smolvlaWsl.stdout)' --checkpoint-root '$($checkpointRootWsl.stdout)' --hf-home '$($hfHomeWsl.stdout)' --libero-root '$($liberoRootWsl.stdout)' --robosuite-root '$($robosuiteRootWsl.stdout)' --libero-data-root '$($liberoDataRootWsl.stdout)' --task-suite '$TaskSuite' --start-task-id $StartTaskId --task-count $TaskCount --max-steps-per-task $MaxStepsPerTask --camera-size $CameraSize --device cpu --report-path '$($reportPathWsl.stdout)' > '$($stdoutLogWsl.stdout)' 2>&1
printf 'bounded reduced-scope learned-policy rollout command finished\n'
"@
$rollout = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
$report.rollout_command = [ordered]@{ timeout_seconds = $TimeoutSeconds; device = "cpu"; task_suite = $TaskSuite; task_count = $TaskCount; max_steps_per_task = $MaxStepsPerTask }
$report.rollout_result_raw = $rollout

if (Test-Path -LiteralPath $innerReportPath) {
    try {
        $inner = Read-JsonFile -Path $JsonReportPath
        $report.rollout_result = $inner
        $passed = [bool]$inner.result.passed
        $report.bounded_reduced_scope_learned_policy_rollout_passed = $passed
        $report.policy.heavy_model_imports_performed = [bool]$inner.policy.heavy_model_imports_performed
        $report.policy.model_load_performed = [bool]$inner.policy.model_load_performed
        $report.policy.model_inference_performed = [bool]$inner.policy.model_inference_performed
        $report.policy.learned_policy_inference_performed = [bool]$inner.policy.learned_policy_inference_performed
        $report.policy.simulator_environment_created = [bool]$inner.policy.simulator_environment_created
        $report.policy.diagnostic_rollouts_performed = [bool]$inner.policy.diagnostic_rollouts_performed
        $report.decision = if ($passed) { "proceed" } else { "stop" }
        $report.reason = if ($passed) { "bounded reduced-scope learned-policy LIBERO rollout passed" } else { [string]$inner.result.blocked_reason }
    } catch {
        $report.reason = "failed to parse inner reduced-scope learned-policy rollout report: $($_.Exception.Message)"
    }
} else {
    $report.reason = "bounded reduced-scope learned-policy rollout command failed or timed out"
}

$report.recommended_next_step = if ($report.bounded_reduced_scope_learned_policy_rollout_passed) {
    "Generate a reduced-scope rollout metric summary. Keep evidence diagnostic/local-pilot only."
} else {
    "Inspect the reduced-scope learned-policy rollout blocker before any larger matrix or claims."
}

Write-Reports -Report $report
exit 0
