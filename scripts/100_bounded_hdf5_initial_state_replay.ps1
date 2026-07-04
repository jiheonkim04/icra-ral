param(
    [string]$PlanReportPath = "reports\hdf5_initial_state_replay_plan_report.json",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$Hdf5Path = "",
    [string]$TaskSuite = "libero_10",
    [int]$TaskId = 0,
    [int]$MaxReplaySteps = 1,
    [int]$CameraSize = 64,
    [int]$TimeoutSeconds = 600,
    [string]$JsonReportPath = "reports\bounded_hdf5_initial_state_replay_report.json",
    [string]$MarkdownReportPath = "reports\bounded_hdf5_initial_state_replay_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded HDF5 initial-state replay diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script runs only a one-demo HDF5 initial-state/first-action replay diagnostic. It does not load learned policies, infer, train, run GPU jobs, install packages, download assets, access tokens, execute OpenVLA-OFT, run benchmark/multi-seed rollout, or make paper claims."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

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
        "# Bounded HDF5 Initial-State Replay Diagnostic Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.bounded_hdf5_initial_state_replay_passed)",
        "- simulator environment created: $($Report.policy.simulator_environment_created)",
        "- set init state ok: $($Report.replay_result.set_init_state_ok)",
        "- replay steps performed: $($Report.replay_result.steps_performed)",
        "- reward sum: $($Report.replay_result.reward_sum)",
        "- learned-policy inference performed: false",
        "- benchmark rollout performed: false",
        "- paper-grade claim: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is a bounded HDF5 replay diagnostic only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 14
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_hdf5_initial_state_replay = $true
            task_local_gate_required = "ALLOW_HDF5_REPLAY_DIAGNOSTIC=1"
            task_suite = $TaskSuite
            task_id = $TaskId
            max_replay_steps = $MaxReplaySteps
            camera_size = $CameraSize
            downloads_performed = $false
            installs_performed = $false
            gpu_jobs_performed = $false
            training_performed = $false
            heavy_model_imports_performed = $false
            learned_policy_load_performed = $false
            learned_policy_inference_performed = $false
            model_load_performed = $false
            model_inference_performed = $false
            simulator_environment_attempted = $false
            simulator_environment_created = $false
            hdf5_replay_diagnostic_performed = $false
            rollouts_performed = $false
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
        plan = $null
        wsl = [ordered]@{}
        replay_probe = $null
        replay_result = [ordered]@{ ok = $false; steps_performed = 0; reward_sum = 0.0; set_init_state_ok = $false }
        progress_log_path = $null
        progress_events = @()
        bounded_hdf5_initial_state_replay_passed = $false
        ready_for_learned_policy_rollout_recheck = $false
        ready_for_rollout_scaling = $false
        ready_for_paper_claim = $false
        decision = "stop"
        reason = $null
        recommended_next_step = $null
    }
}

$dangerousGateNames = @(
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
    "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })
$report = New-BaseReport
$runDir = Join-Path $RepoRoot "runs\hdf5_initial_state_replay"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

if ($MaxReplaySteps -ne 1) {
    $report.reason = "The first bounded HDF5 replay runner allows exactly one replay step."
    $report.recommended_next_step = "Use MaxReplaySteps=1 for the first HDF5 initial-state replay diagnostic."
    Write-Reports -Report $report
    exit 0
}
if ($CameraSize -lt 16 -or $CameraSize -gt 128) {
    $report.reason = "CameraSize must be between 16 and 128 for bounded HDF5 replay."
    $report.recommended_next_step = "Use CameraSize<=128 before running bounded HDF5 replay."
    Write-Reports -Report $report
    exit 0
}
if ([Environment]::GetEnvironmentVariable("ALLOW_HDF5_REPLAY_DIAGNOSTIC") -ne "1") {
    $report.reason = "ALLOW_HDF5_REPLAY_DIAGNOSTIC=1 is required after a green HDF5 replay plan."
    $report.recommended_next_step = "Run scripts\98_plan_hdf5_initial_state_replay.ps1 first, then set ALLOW_HDF5_REPLAY_DIAGNOSTIC=1 only for this bounded replay task if the plan says proceed."
    Write-Reports -Report $report
    exit 0
}
if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "bounded HDF5 replay refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset rollout, benchmark, OpenVLA, heavy-import, GPU, adapter-diagnostic, matrix, and training gates before replay."
    Write-Reports -Report $report
    exit 0
}

$savedGate = [Environment]::GetEnvironmentVariable("ALLOW_HDF5_REPLAY_DIAGNOSTIC")
Remove-Item Env:\ALLOW_HDF5_REPLAY_DIAGNOSTIC -ErrorAction SilentlyContinue
$planJson = Join-Path $runDir "hdf5_initial_state_replay_plan_report.json"
$planMd = Join-Path $runDir "hdf5_initial_state_replay_plan_report.md"
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\98_plan_hdf5_initial_state_replay.ps1") `
        -JsonReportPath $planJson `
        -MarkdownReportPath $planMd | Out-Null
} finally {
    if ($null -ne $savedGate) {
        $env:ALLOW_HDF5_REPLAY_DIAGNOSTIC = $savedGate
    }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planJson)) {
    $report.reason = "HDF5 initial-state replay planner failed"
    $report.recommended_next_step = "Fix the replay planner before bounded replay execution."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planJson
$report.plan = $plan
if (-not $plan.ready_for_bounded_hdf5_replay_runner) {
    $report.reason = "HDF5 replay planner did not authorize bounded replay execution"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}
if ([string]::IsNullOrWhiteSpace($Hdf5Path)) {
    $Hdf5Path = [string]$plan.hdf5_inputs.hdf5_path
}
if ([string]::IsNullOrWhiteSpace($Hdf5Path) -or -not (Test-Path -LiteralPath $Hdf5Path)) {
    $report.reason = "HDF5 path is missing or does not exist: $Hdf5Path"
    $report.recommended_next_step = "Fix local HDF5 path before bounded replay execution."
    Write-Reports -Report $report
    exit 0
}

if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Configure WSL before bounded HDF5 replay."
    Write-Reports -Report $report
    exit 0
}

$pythonSelector = "if [ -x $WslPython ]; then printf '%s' $WslPython; else printf '%s' python3; fi"
$selectedWslPython = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $pythonSelector) -TimeoutSec 30
$wslPythonExecutable = if ($selectedWslPython.ok -and -not [string]::IsNullOrWhiteSpace($selectedWslPython.stdout)) { $selectedWslPython.stdout } else { "python3" }
$pythonVersion = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable --version") -TimeoutSec 30
$h5pyProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable -c 'import h5py, numpy; print(h5py.__version__)'") -TimeoutSec 60
$liberoRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/LIBERO") -TimeoutSec 30
$robosuiteRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/repos/robosuite") -TimeoutSec 30
$hdf5PathWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($Hdf5Path.Replace("\", "/"))) -TimeoutSec 30
$report.wsl = [ordered]@{
    selected_python = $selectedWslPython
    python_executable = $wslPythonExecutable
    python_version = $pythonVersion
    h5py_probe = $h5pyProbe
    libero_root = $liberoRootWsl
    robosuite_root = $robosuiteRootWsl
    hdf5_path = $hdf5PathWsl
}
if (-not ($pythonVersion.ok -and $h5pyProbe.ok -and $liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $hdf5PathWsl.ok)) {
    $report.reason = "WSL path conversion or Python/HDF5 probe failed"
    $report.recommended_next_step = "Fix WSL path/Python/HDF5 readiness before bounded replay."
    Write-Reports -Report $report
    exit 0
}

$probeScriptWin = Join-Path $runDir "probe_hdf5_replay.py"
$probeScript = @'
import json
import os
import sys
import time
import traceback

started = time.perf_counter()
task_suite_name = sys.argv[1]
task_id = int(sys.argv[2])
max_steps = int(sys.argv[3])
camera_size = int(sys.argv[4])
hdf5_path = sys.argv[5]
progress_path = os.environ.get("TCA_MAP_HDF5_REPLAY_PROGRESS")


def mark(stage, **extra):
    event = {"stage": stage, "elapsed_seconds": round(time.perf_counter() - started, 6)}
    event.update(extra)
    if progress_path:
        with open(progress_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def compact(value, limit=800):
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"


result = {
    "ok": False,
    "error": None,
    "traceback_tail": None,
    "elapsed_seconds": None,
    "python": sys.version.split()[0],
    "task_suite": task_suite_name,
    "task_id": task_id,
    "camera_size": camera_size,
    "hdf5_path": hdf5_path,
    "demo_name": None,
    "init_state_shape": None,
    "first_action": None,
    "first_gripper_action": None,
    "env_created": False,
    "reset_ok": False,
    "set_init_state_ok": False,
    "steps_requested": max_steps,
    "steps_performed": 0,
    "reward_sum": 0.0,
    "done_seen": False,
    "success_check": None,
    "agentview_image_shape": None,
    "agentview_image_mean": None,
    "learned_policy_inference_performed": False,
    "training_performed": False,
    "gpu_job_performed": False,
    "benchmark_rollout_performed": False,
    "openvla_oft_executed": False,
    "paper_grade_claim_made": False,
}

try:
    mark("start")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    libero_root = os.environ["TCA_MAP_LIBERO_ROOT_WSL"]
    robosuite_root = os.environ["TCA_MAP_ROBOSUITE_ROOT_WSL"]
    for module_name in list(sys.modules):
        if module_name == "libero" or module_name.startswith("libero."):
            del sys.modules[module_name]
    sys.path = [path for path in sys.path if not path.startswith(libero_root)]
    for path in [robosuite_root, libero_root]:
        if path:
            sys.path.insert(0, path)
    mark("before_imports")
    import glob
    import h5py
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv
    mark("after_imports")

    with h5py.File(hdf5_path, "r") as handle:
        demo_name = sorted(handle["data"].keys())[0]
        demo = handle["data"][demo_name]
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
        actions = np.asarray(demo["actions"][:max_steps], dtype=np.float64)
        result["demo_name"] = demo_name
        result["init_state_shape"] = list(init_state.shape)
        result["first_action"] = [float(x) for x in actions[0]]
        result["first_gripper_action"] = float(actions[0][-1])
    mark("hdf5_loaded", demo_name=result["demo_name"], init_state_shape=result["init_state_shape"])

    bddl_dir = os.path.join(libero_root, "libero", "libero", "bddl_files", task_suite_name)
    bddl_files = sorted(glob.glob(os.path.join(bddl_dir, "*.bddl")))
    if task_id < 0 or task_id >= len(bddl_files):
        raise ValueError(f"task_id {task_id} outside BDDL suite size {len(bddl_files)}")
    bddl_file = bddl_files[task_id]
    mark("before_env_create", bddl_file=bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=camera_size,
        camera_widths=camera_size,
    )
    result["env_created"] = True
    mark("after_env_create")
    try:
        env.seed(0)
        env.reset()
        result["reset_ok"] = True
        mark("after_reset")
        obs = env.set_init_state(init_state)
        result["set_init_state_ok"] = True
        mark("after_set_init_state")
        for step_id, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            result["steps_performed"] += 1
            result["reward_sum"] += float(reward)
            result["done_seen"] = bool(result["done_seen"] or done)
            mark("after_step", step_id=step_id, reward=float(reward), done=bool(done))
        try:
            result["success_check"] = bool(env.check_success())
        except Exception:
            result["success_check"] = None
        if isinstance(obs, dict) and "agentview_image" in obs:
            image = np.asarray(obs["agentview_image"])
            result["agentview_image_shape"] = list(image.shape)
            result["agentview_image_mean"] = float(image.mean())
    finally:
        try:
            env.close()
        except Exception:
            pass

    result["ok"] = (
        result["env_created"]
        and result["reset_ok"]
        and result["set_init_state_ok"]
        and result["steps_performed"] == max_steps
        and not result["learned_policy_inference_performed"]
        and not result["training_performed"]
        and not result["gpu_job_performed"]
        and not result["benchmark_rollout_performed"]
        and not result["openvla_oft_executed"]
        and not result["paper_grade_claim_made"]
    )
except Exception as exc:
    result["error"] = compact(f"{type(exc).__name__}: {exc}")
    result["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
    mark("fatal_error", error=result["error"])

result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
mark("done", ok=result["ok"], steps_performed=result["steps_performed"])
print(json.dumps(result, indent=2, sort_keys=True))
'@
$probeScript | Set-Content -LiteralPath $probeScriptWin -Encoding UTF8
$probeScriptWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($probeScriptWin.Replace("\", "/"))) -TimeoutSec 30
if (-not $probeScriptWsl.ok) {
    $report.reason = "failed to map HDF5 replay probe script into WSL"
    $report.recommended_next_step = "Fix WSL path mapping before bounded replay."
    Write-Reports -Report $report
    exit 0
}

$progressWin = Join-Path $runDir "hdf5_initial_state_replay_progress.jsonl"
Remove-Item -LiteralPath $progressWin -ErrorAction SilentlyContinue
$progressWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($progressWin.Replace("\", "/"))) -TimeoutSec 30
if ($progressWsl.ok) {
    $report.progress_log_path = $progressWin
}

$bashCommand = "export TCA_MAP_LIBERO_ROOT_WSL='$($liberoRootWsl.stdout)'; export TCA_MAP_ROBOSUITE_ROOT_WSL='$($robosuiteRootWsl.stdout)'; export TCA_MAP_HDF5_REPLAY_PROGRESS='$($progressWsl.stdout)'; export MUJOCO_GL=osmesa; $wslPythonExecutable '$($probeScriptWsl.stdout)' '$TaskSuite' $TaskId $MaxReplaySteps $CameraSize '$($hdf5PathWsl.stdout)'"
$bashCommand = $bashCommand -replace "`r", ""
$replayProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
$report.policy.simulator_environment_attempted = $true
$report.replay_probe = $replayProbe

if (Test-Path -LiteralPath $progressWin) {
    $events = New-Object System.Collections.Generic.List[object]
    Get-Content -LiteralPath $progressWin -Encoding UTF8 | ForEach-Object {
        if (-not [string]::IsNullOrWhiteSpace($_)) {
            try {
                $events.Add(($_ | ConvertFrom-Json)) | Out-Null
            } catch {}
        }
    }
    $report.progress_events = @($events.ToArray())
}

if ($replayProbe.ok) {
    try {
        $stdoutText = [string]$replayProbe.stdout
        $jsonStart = $stdoutText.IndexOf("{")
        if ($jsonStart -lt 0) {
            throw "WSL HDF5 replay stdout did not contain a JSON object"
        }
        $parsed = $stdoutText.Substring($jsonStart) | ConvertFrom-Json
        $report.replay_result = $parsed
        $report.bounded_hdf5_initial_state_replay_passed = [bool]$parsed.ok
        $report.policy.simulator_environment_created = [bool]$parsed.env_created
        $report.policy.hdf5_replay_diagnostic_performed = [bool]$parsed.ok
        $report.decision = if ($parsed.ok) { "proceed" } else { "stop" }
        if ($parsed.ok) {
            $report.reason = "bounded HDF5 initial-state replay diagnostic passed"
        } else {
            $failureReason = [string]$parsed.error
            if ([string]::IsNullOrWhiteSpace($failureReason)) {
                $failureReason = "unknown HDF5 replay failure"
            }
            $report.reason = "bounded HDF5 replay diagnostic failed: $failureReason"
        }
    } catch {
        $report.reason = "failed to parse WSL HDF5 replay JSON: $($_.Exception.Message)"
    }
} else {
    $report.reason = "WSL HDF5 replay command failed"
}

$report.ready_for_learned_policy_rollout_recheck = [bool]$report.bounded_hdf5_initial_state_replay_passed
$report.ready_for_rollout_scaling = $false
$report.ready_for_paper_claim = $false
$report.recommended_next_step = if ($report.bounded_hdf5_initial_state_replay_passed) {
    "Plan a narrow learned-policy rollout recheck that uses a documented initial-state convention. Keep it one task, diagnostic-only, no benchmark claim, no multi-seed, no OpenVLA-OFT."
} else {
    "Diagnose HDF5 replay failure before any learned-policy rollout recheck."
}

Write-Reports -Report $report
exit 0
