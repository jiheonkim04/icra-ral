param(
    [string]$PlanReportPath = "reports\libero_robosuite_diagnostic_rollout_plan_report.json",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$TaskSuite = "libero_10",
    [int]$StartTaskId = 0,
    [int]$TaskCount = 1,
    [int]$MaxStepsPerTask = 3,
    [int]$CameraSize = 64,
    [int]$TimeoutSeconds = 600,
    [string]$JsonReportPath = "reports\bounded_libero_robosuite_diagnostic_rollout_report.json",
    [string]$MarkdownReportPath = "reports\bounded_libero_robosuite_diagnostic_rollout_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded LIBERO/RoboSuite diagnostic rollout"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script runs only bounded zero-action LIBERO/RoboSuite diagnostic rollouts. It does not run learned policy inference, train, run GPU jobs, install packages, download assets, import heavy VLA models, access tokens, execute OpenVLA-OFT, run multi-seed evaluation, or make benchmark/SOTA/paper-grade claims."

function Invoke-SafeCommand {
    param(
        [string[]]$Command,
        [int]$TimeoutSec = 120,
        [System.Text.Encoding]$Encoding = [System.Text.Encoding]::UTF8
    )
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
        $psi.StandardOutputEncoding = $Encoding
        $psi.StandardErrorEncoding = $Encoding
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
    $fullPath = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
    $text = [System.IO.File]::ReadAllText($fullPath, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
    return $text | ConvertFrom-Json
}

function Write-Reports {
    param([object]$Report)
    $jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
    $markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# Bounded LIBERO/RoboSuite Diagnostic Rollout Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.bounded_libero_robosuite_diagnostic_rollout_passed)",
        "- diagnostic rollouts performed: $($Report.policy.diagnostic_rollouts_performed)",
        "- benchmark rollouts performed: false",
        "- tasks completed: $($Report.rollout_result.tasks_completed)",
        "- total steps performed: $($Report.rollout_result.total_steps_performed)",
        "- paper-grade claim: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is bounded simulator diagnostic evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 12
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_libero_robosuite_diagnostic_rollout = $true
            task_local_gate_required = "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT=1"
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
            zero_action_policy_only = $true
            simulator_environment_attempted = $false
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
        runtime = [ordered]@{
            wsl_python = $WslPython
            timeout_seconds = $TimeoutSeconds
            expected_vram_gb = 0
            mujoco_gl = "osmesa"
        }
        risk_plan = $null
        wsl = [ordered]@{}
        rollout_probe = $null
        rollout_result = [ordered]@{ ok = $false; tasks_completed = 0; total_steps_performed = 0 }
        progress_log_path = $null
        progress_events = @()
        bounded_libero_robosuite_diagnostic_rollout_passed = $false
        ready_for_benchmark_rollout = $false
        ready_for_paper_claim = $false
        decision = "stop"
        reason = $null
        recommended_next_step = $null
    }
}

$dangerousGateNames = @(
    "ALLOW_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_TINY_ROLLOUT"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })
$report = New-BaseReport
$runDir = Join-Path $RepoRoot "runs\libero_robosuite_diagnostic_rollout"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

if ($TaskCount -lt 1 -or $TaskCount -gt 5) {
    $report.reason = "TaskCount must be between 1 and 5."
    $report.recommended_next_step = "Use TaskCount<=5 before running bounded diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}
if ($MaxStepsPerTask -lt 1 -or $MaxStepsPerTask -gt 5) {
    $report.reason = "MaxStepsPerTask must be between 1 and 5."
    $report.recommended_next_step = "Use MaxStepsPerTask<=5 before running bounded diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}
if ($CameraSize -lt 16 -or $CameraSize -gt 128) {
    $report.reason = "CameraSize must be between 16 and 128 for diagnostic rollout."
    $report.recommended_next_step = "Use CameraSize<=128 before running bounded diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}
if ([Environment]::GetEnvironmentVariable("ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT") -ne "1") {
    $report.reason = "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT=1 is required after a green diagnostic rollout risk assessment."
    $report.recommended_next_step = "Run scripts\64_plan_libero_robosuite_diagnostic_rollout.ps1, then set ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT=1 only for this bounded task if the plan says proceed."
    Write-Reports -Report $report
    exit 0
}
if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "bounded LIBERO/RoboSuite diagnostic rollout refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset broad rollout, OpenVLA, heavy-import, simulator, and training gates before diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

$planJson = Join-Path $runDir "libero_robosuite_diagnostic_rollout_plan_report.json"
$planMd = Join-Path $runDir "libero_robosuite_diagnostic_rollout_plan_report.md"
$savedGate = [Environment]::GetEnvironmentVariable("ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT")
Remove-Item Env:\ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT -ErrorAction SilentlyContinue
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\64_plan_libero_robosuite_diagnostic_rollout.ps1") `
        -TaskCount $TaskCount `
        -MaxStepsPerTask $MaxStepsPerTask `
        -ExpectedRuntimeMinutes 15 `
        -ExpectedVramGb 0 `
        -JsonReportPath $planJson `
        -MarkdownReportPath $planMd | Out-Null
} finally {
    if ($null -ne $savedGate) {
        $env:ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT = $savedGate
    }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planJson)) {
    $report.reason = "LIBERO/RoboSuite diagnostic rollout planner failed"
    $report.recommended_next_step = "Fix planning report before any diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planJson
$report.risk_plan = $plan
if (-not $plan.ready_for_libero_robosuite_diagnostic_rollout_execution) {
    $report.reason = "diagnostic rollout planner did not authorize bounded execution"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Install/configure WSL2 before LIBERO/RoboSuite diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

$pythonSelector = "if [ -x $WslPython ]; then printf '%s' $WslPython; else printf '%s' python3; fi"
$selectedWslPython = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $pythonSelector) -TimeoutSec 30
$wslPythonExecutable = if ($selectedWslPython.ok -and -not [string]::IsNullOrWhiteSpace($selectedWslPython.stdout)) { $selectedWslPython.stdout } else { "python3" }
$pythonVersion = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable --version") -TimeoutSec 30
$liberoRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/LIBERO") -TimeoutSec 30
$robosuiteRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/robosuite") -TimeoutSec 30
$dataRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/data/libero") -TimeoutSec 30
$report.wsl = [ordered]@{
    selected_python = $selectedWslPython
    python_executable = $wslPythonExecutable
    python_version = $pythonVersion
    libero_root = $liberoRootWsl
    robosuite_root = $robosuiteRootWsl
    libero_data_root = $dataRootWsl
}
if (-not ($pythonVersion.ok -and $liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $dataRootWsl.ok)) {
    $report.reason = "WSL path conversion or python probe failed"
    $report.recommended_next_step = "Fix WSL path/python readiness before LIBERO/RoboSuite diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

$probeScriptWin = Join-Path $runDir "probe_libero_robosuite_diagnostic_rollout.py"
$probeScript = @'
import json
import os
import sys
import time
import traceback

started = time.perf_counter()
task_suite_name = sys.argv[1]
start_task_id = int(sys.argv[2])
task_count = int(sys.argv[3])
max_steps = int(sys.argv[4])
camera_size = int(sys.argv[5])
progress_path = os.environ.get("TCA_MAP_DIAGNOSTIC_PROGRESS")


def mark(stage, **extra):
    event = {
        "stage": stage,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    event.update(extra)
    if progress_path:
        with open(progress_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def compact_text(value, limit=800):
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"

result = {
    "ok": False,
    "error": None,
    "traceback_tail": None,
    "elapsed_seconds": None,
    "python": sys.version.split()[0],
    "task_suite": task_suite_name,
    "start_task_id": start_task_id,
    "task_count": task_count,
    "max_steps_per_task": max_steps,
    "camera_size": camera_size,
    "tasks_completed": 0,
    "diagnostic_rollouts_completed": 0,
    "total_steps_performed": 0,
    "libero_robosuite_env_created": False,
    "benchmark_rollout_performed": False,
    "learned_policy_inference_performed": False,
    "training_performed": False,
    "gpu_job_performed": False,
    "openvla_oft_executed": False,
    "paper_grade_claim_made": False,
    "task_summaries": [],
}

try:
    mark("start")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    libero_root = os.environ["TCA_MAP_LIBERO_ROOT_WSL"]
    robosuite_root = os.environ["TCA_MAP_ROBOSUITE_ROOT_WSL"]
    data_root = os.environ["TCA_MAP_LIBERO_DATA_ROOT_WSL"]
    mark("paths_loaded", libero_root=libero_root, robosuite_root=robosuite_root, data_root=data_root)
    for module_name in list(sys.modules):
        if module_name == "libero" or module_name.startswith("libero."):
            del sys.modules[module_name]
    sys.path = [
        path for path in sys.path
        if not path.startswith(libero_root)
    ]
    for path in [robosuite_root, libero_root]:
        if path:
            sys.path.insert(0, path)

    mark("before_imports")
    import glob
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv
    mark("after_imports", numpy_version=np.__version__)

    bddl_dir = os.path.join(libero_root, "libero", "libero", "bddl_files", task_suite_name)
    bddl_files = sorted(glob.glob(os.path.join(bddl_dir, "*.bddl")))
    mark("bddl_files_scanned", bddl_dir=bddl_dir, bddl_count=len(bddl_files))
    if not bddl_files:
        raise FileNotFoundError(f"no BDDL files found under {bddl_dir}")
    num_tasks = len(bddl_files)
    if start_task_id < 0 or start_task_id + task_count > num_tasks:
        raise ValueError(f"task range [{start_task_id}, {start_task_id + task_count}) is outside suite size {num_tasks}")

    for offset in range(task_count):
        task_id = start_task_id + offset
        bddl_file = bddl_files[task_id]
        task_name = os.path.splitext(os.path.basename(bddl_file))[0]
        mark("task_start", task_id=task_id, task_name=task_name)

        env = None
        task_started = time.perf_counter()
        summary = {
            "task_id": task_id,
            "task_name": task_name,
            "language": task_name.replace("_", " "),
            "bddl_file": bddl_file,
            "init_states_file": None,
            "init_state_loaded": False,
            "init_state_error": "skipped: direct BDDL diagnostic avoids LIBERO benchmark torch dependency",
            "env_created": False,
            "reset_ok": False,
            "set_init_state_ok": False,
            "steps_performed": 0,
            "reward_sum": 0.0,
            "done_seen": False,
            "success_check": None,
            "agentview_image_shape": None,
            "agentview_image_mean": None,
            "elapsed_seconds": None,
            "error": None,
        }
        try:
            env_args = {
                "bddl_file_name": bddl_file,
                "camera_heights": camera_size,
                "camera_widths": camera_size,
            }
            mark("before_env_create", task_id=task_id, task_name=task_name)
            env = OffScreenRenderEnv(**env_args)
            result["libero_robosuite_env_created"] = True
            summary["env_created"] = True
            mark("after_env_create", task_id=task_id, task_name=task_name)
            mark("before_seed", task_id=task_id, task_name=task_name)
            env.seed(0)
            mark("after_seed", task_id=task_id, task_name=task_name)
            mark("before_reset", task_id=task_id, task_name=task_name)
            obs = env.reset()
            summary["reset_ok"] = True
            mark("after_reset", task_id=task_id, task_name=task_name)
            action = [0.0] * 7
            for step_id in range(max_steps):
                mark("before_step", task_id=task_id, task_name=task_name, step_id=step_id)
                obs, reward, done, info = env.step(action)
                summary["steps_performed"] += 1
                result["total_steps_performed"] += 1
                try:
                    summary["reward_sum"] += float(reward)
                except Exception:
                    pass
                summary["done_seen"] = bool(summary["done_seen"] or done)
                mark("after_step", task_id=task_id, task_name=task_name, step_id=step_id)
            try:
                mark("before_success_check", task_id=task_id, task_name=task_name)
                summary["success_check"] = bool(env.check_success())
                mark("after_success_check", task_id=task_id, task_name=task_name)
            except Exception:
                summary["success_check"] = None
            image = obs.get("agentview_image") if isinstance(obs, dict) else None
            if image is not None:
                arr = np.asarray(image)
                summary["agentview_image_shape"] = list(arr.shape)
                summary["agentview_image_mean"] = float(arr.mean())
            result["tasks_completed"] += 1
            result["diagnostic_rollouts_completed"] += 1
            mark("task_done", task_id=task_id, task_name=task_name, steps_performed=summary["steps_performed"])
        except Exception as task_exc:
            summary["error"] = compact_text(f"{type(task_exc).__name__}: {task_exc}")
            mark("task_error", task_id=task_id, task_name=task_name, error=summary["error"])
        finally:
            if env is not None:
                try:
                    mark("before_env_close", task_id=task_id, task_name=task_name)
                    env.close()
                    mark("after_env_close", task_id=task_id, task_name=task_name)
                except Exception:
                    pass
            summary["elapsed_seconds"] = round(time.perf_counter() - task_started, 6)
            result["task_summaries"].append(summary)

    result["ok"] = (
        result["tasks_completed"] == task_count
        and result["diagnostic_rollouts_completed"] == task_count
        and result["total_steps_performed"] == task_count * max_steps
        and result["libero_robosuite_env_created"]
        and not result["benchmark_rollout_performed"]
        and not result["learned_policy_inference_performed"]
        and not result["training_performed"]
        and not result["gpu_job_performed"]
        and not result["openvla_oft_executed"]
        and not result["paper_grade_claim_made"]
    )
except Exception as exc:
    result["error"] = compact_text(f"{type(exc).__name__}: {exc}")
    result["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
    mark("fatal_error", error=result["error"])

result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
mark("done", ok=result["ok"], tasks_completed=result["tasks_completed"], total_steps_performed=result["total_steps_performed"])
print(json.dumps(result, indent=2, sort_keys=True))
'@
$probeScript | Set-Content -LiteralPath $probeScriptWin -Encoding UTF8
$probeScriptWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($probeScriptWin.Replace("\", "/"))) -TimeoutSec 30
if (-not $probeScriptWsl.ok) {
    $report.reason = "failed to map diagnostic rollout probe script into WSL"
    $report.recommended_next_step = "Fix WSL path mapping before diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

$progressWin = Join-Path $runDir "libero_robosuite_diagnostic_progress.jsonl"
Remove-Item -LiteralPath $progressWin -ErrorAction SilentlyContinue
$progressWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($progressWin.Replace("\", "/"))) -TimeoutSec 30
if ($progressWsl.ok) {
    $report["progress_log_path"] = $progressWin
}

$bashCommand = "export TCA_MAP_LIBERO_ROOT_WSL='$($liberoRootWsl.stdout)'; export TCA_MAP_ROBOSUITE_ROOT_WSL='$($robosuiteRootWsl.stdout)'; export TCA_MAP_LIBERO_DATA_ROOT_WSL='$($dataRootWsl.stdout)'; export TCA_MAP_DIAGNOSTIC_PROGRESS='$($progressWsl.stdout)'; export MUJOCO_GL=osmesa; $wslPythonExecutable '$($probeScriptWsl.stdout)' '$TaskSuite' $StartTaskId $TaskCount $MaxStepsPerTask $CameraSize"
$rolloutProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
$report.policy.simulator_environment_attempted = $true
$report.rollout_probe = $rolloutProbe

if (Test-Path -LiteralPath $progressWin) {
    $events = New-Object System.Collections.Generic.List[object]
    Get-Content -LiteralPath $progressWin -Encoding UTF8 | ForEach-Object {
        if (-not [string]::IsNullOrWhiteSpace($_)) {
            try {
                $events.Add(($_ | ConvertFrom-Json)) | Out-Null
            } catch {}
        }
    }
    $report["progress_events"] = @($events.ToArray())
}

if ($rolloutProbe.ok) {
    try {
        $stdoutText = [string]$rolloutProbe.stdout
        $jsonStart = $stdoutText.IndexOf("{")
        if ($jsonStart -lt 0) {
            throw "WSL diagnostic rollout stdout did not contain a JSON object"
        }
        $parsed = $stdoutText.Substring($jsonStart) | ConvertFrom-Json
        $report.rollout_result = $parsed
        $report.bounded_libero_robosuite_diagnostic_rollout_passed = [bool]$parsed.ok
        $report.policy.simulator_environment_created = [bool]$parsed.libero_robosuite_env_created
        $report.policy.diagnostic_rollouts_performed = [bool]$parsed.ok
        $report.decision = if ($parsed.ok) { "proceed" } else { "stop" }
        if ($parsed.ok) {
            $report.reason = "bounded LIBERO/RoboSuite diagnostic rollout passed"
        } else {
            $failureReason = [string]$parsed.error
            if ([string]::IsNullOrWhiteSpace($failureReason) -and $null -ne $parsed.task_summaries -and $parsed.task_summaries.Count -gt 0) {
                $failureReason = [string]$parsed.task_summaries[0].error
            }
            if ([string]::IsNullOrWhiteSpace($failureReason)) {
                $failureReason = "unknown diagnostic rollout failure"
            }
            $report.reason = "bounded LIBERO/RoboSuite diagnostic rollout failed: $failureReason"
        }
    } catch {
        $report.reason = "failed to parse WSL diagnostic rollout JSON: $($_.Exception.Message)"
    }
} else {
    $report.reason = "WSL diagnostic rollout command failed"
}

$report.ready_for_benchmark_rollout = $false
$report.ready_for_paper_claim = $false
$report.recommended_next_step = if ($report.bounded_libero_robosuite_diagnostic_rollout_passed) {
    "Update status reports and plan the next bounded diagnostic. Stop before benchmark rollouts, multi-seed rollouts, OpenVLA-OFT, full training, external upload, or paper-level claims."
} else {
    "Diagnostic rollout did not pass. Diagnose simulator/env errors before benchmark rollout or claims."
}

Write-Reports -Report $report
exit 0
