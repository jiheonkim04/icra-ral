param(
    [string]$ResetStepReportPath = "reports\bounded_simulator_reset_step_smoke_report.json",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [int]$TaskCount = 1,
    [int]$MaxEpisodes = 1,
    [int]$MaxStepsPerEpisode = 5,
    [int]$TimeoutSeconds = 120,
    [string]$JsonReportPath = "reports\bounded_tiny_diagnostic_rollout_report.json",
    [string]$MarkdownReportPath = "reports\bounded_tiny_diagnostic_rollout_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded tiny diagnostic rollout"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script runs only tiny MuJoCo diagnostic rollouts. It does not create LIBERO/RoboSuite benchmark environments, run training, run GPU jobs, install packages, download assets, import heavy VLA models, access tokens, execute OpenVLA-OFT, run multi-seed rollouts, or make benchmark/SOTA/paper-grade claims."

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
    $text = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
    return $text | ConvertFrom-Json
}

function Write-Reports {
    param([object]$Report)
    $jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
    $markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# Bounded Tiny Diagnostic Rollout Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.bounded_tiny_diagnostic_rollout_passed)",
        "- rollout attempted: $($Report.policy.rollout_attempted)",
        "- rollouts performed: $($Report.policy.rollouts_performed)",
        "- diagnostic tasks completed: $($Report.rollout_result.tasks_completed)",
        "- total steps performed: $($Report.rollout_result.total_steps_performed)",
        "- LIBERO/RoboSuite benchmark env created: false",
        "- paper-grade claim: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is tiny diagnostic simulator plumbing evidence only. It is not LIBERO success, standard success, benchmark evidence, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 10
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_tiny_diagnostic_rollout = $true
            task_local_gate_required = "ALLOW_TINY_ROLLOUT=1"
            task_count = $TaskCount
            max_episodes = $MaxEpisodes
            max_steps_per_episode = $MaxStepsPerEpisode
            downloads_performed = $false
            installs_performed = $false
            gpu_jobs_performed = $false
            training_performed = $false
            heavy_model_imports_performed = $false
            model_load_performed = $false
            model_inference_performed = $false
            policy_inference_performed = $false
            simulator_imports_performed = $false
            rollout_attempted = $false
            rollouts_performed = $false
            multi_seed_performed = $false
            libero_robosuite_benchmark_env_created = $false
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
        }
        risk_plan = $null
        wsl = [ordered]@{}
        rollout_probe = $null
        rollout_result = [ordered]@{ ok = $false; tasks_completed = 0; total_steps_performed = 0 }
        bounded_tiny_diagnostic_rollout_passed = $false
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
    "ALLOW_SIMULATOR_RENDER_SMOKE"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })
$report = New-BaseReport
$runDir = Join-Path $RepoRoot "runs\tiny_diagnostic_rollout"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

if ($TaskCount -lt 1 -or $TaskCount -gt 5) {
    $report.reason = "TaskCount must be between 1 and 5."
    $report.recommended_next_step = "Use TaskCount<=5 before running bounded tiny diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}
if ($MaxEpisodes -ne 1) {
    $report.reason = "MaxEpisodes must be exactly 1 for this bounded diagnostic rollout."
    $report.recommended_next_step = "Use MaxEpisodes=1 before running bounded tiny diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}
if ($MaxStepsPerEpisode -lt 1 -or $MaxStepsPerEpisode -gt 5) {
    $report.reason = "MaxStepsPerEpisode must be between 1 and 5."
    $report.recommended_next_step = "Use MaxStepsPerEpisode<=5 before running bounded tiny diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

if ([Environment]::GetEnvironmentVariable("ALLOW_TINY_ROLLOUT") -ne "1") {
    $report.reason = "ALLOW_TINY_ROLLOUT=1 is required after a green tiny diagnostic rollout risk assessment."
    $report.recommended_next_step = "Run scripts\62_plan_tiny_diagnostic_rollout.ps1, then set ALLOW_TINY_ROLLOUT=1 only for this bounded diagnostic rollout task if the plan says proceed."
    Write-Reports -Report $report
    exit 0
}
if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "bounded tiny diagnostic rollout refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset broad rollout, OpenVLA, heavy-import, simulator, and training gates before diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

$planJson = Join-Path $runDir "tiny_diagnostic_rollout_plan_report.json"
$planMd = Join-Path $runDir "tiny_diagnostic_rollout_plan_report.md"
$savedTinyRolloutGate = [Environment]::GetEnvironmentVariable("ALLOW_TINY_ROLLOUT")
Remove-Item Env:\ALLOW_TINY_ROLLOUT -ErrorAction SilentlyContinue
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\62_plan_tiny_diagnostic_rollout.ps1") `
        -ResetStepReportPath $ResetStepReportPath `
        -TaskCount $TaskCount `
        -MaxEpisodes $MaxEpisodes `
        -MaxStepsPerEpisode $MaxStepsPerEpisode `
        -ExpectedRuntimeMinutes 30 `
        -ExpectedVramGb 0 `
        -JsonReportPath $planJson `
        -MarkdownReportPath $planMd | Out-Null
} finally {
    if ($null -ne $savedTinyRolloutGate) {
        $env:ALLOW_TINY_ROLLOUT = $savedTinyRolloutGate
    }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planJson)) {
    $report.reason = "tiny diagnostic rollout planner failed"
    $report.recommended_next_step = "Fix planning report before any diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planJson
$report.risk_plan = $plan
if (-not $plan.ready_for_tiny_diagnostic_rollout_execution) {
    $report.reason = "tiny diagnostic rollout planner did not authorize bounded execution"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Install/configure WSL2 before bounded tiny diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

$pythonSelector = "if [ -x $WslPython ]; then printf '%s' $WslPython; else printf '%s' python3; fi"
$selectedWslPython = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $pythonSelector) -TimeoutSec 30
$wslPythonExecutable = if ($selectedWslPython.ok -and -not [string]::IsNullOrWhiteSpace($selectedWslPython.stdout)) { $selectedWslPython.stdout } else { "python3" }
$pythonVersion = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable --version") -TimeoutSec 30
$mujocoProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable -c 'import mujoco; print(mujoco.__version__)'") -TimeoutSec 60
$report.wsl = [ordered]@{
    selected_python = $selectedWslPython
    python_executable = $wslPythonExecutable
    python_version = $pythonVersion
    mujoco_probe = $mujocoProbe
}
if (-not ($pythonVersion.ok -and $mujocoProbe.ok)) {
    $report.reason = "selected WSL Python cannot import mujoco"
    $report.recommended_next_step = "Fix WSL venv MuJoCo readiness before bounded tiny diagnostic rollout."
    Write-Reports -Report $report
    exit 0
}

$probeScriptWin = Join-Path $runDir "probe_tiny_diagnostic_rollout.py"
$probeScript = @'
import json
import sys
import time

started = time.perf_counter()
task_count = int(sys.argv[1])
max_episodes = int(sys.argv[2])
max_steps = int(sys.argv[3])
result = {
    "ok": False,
    "error": None,
    "elapsed_seconds": None,
    "python": sys.version.split()[0],
    "mujoco_version": None,
    "task_count": task_count,
    "max_episodes": max_episodes,
    "max_steps_per_episode": max_steps,
    "tasks_completed": 0,
    "rollouts_completed": 0,
    "total_steps_performed": 0,
    "toy_mujoco_env_created": False,
    "libero_robosuite_benchmark_env_created": False,
    "policy_inference_performed": False,
    "training_performed": False,
    "benchmark_claims_made": False,
    "task_summaries": [],
}

try:
    import mujoco

    result["mujoco_version"] = getattr(mujoco, "__version__", "unknown")
    for task_id in range(task_count):
        xml = f"""
        <mujoco model="tca_map_tiny_diagnostic_rollout_{task_id}">
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <worldbody>
            <body name="body" pos="{0.01 * task_id:.4f} 0 0.2">
              <freejoint/>
              <geom type="sphere" size="0.05" mass="{0.1 + 0.01 * task_id:.4f}" rgba="0.1 0.4 0.8 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        result["toy_mujoco_env_created"] = True
        task_summary = {
            "task_id": task_id,
            "episodes_completed": 0,
            "steps_performed": 0,
            "qpos_start_z": None,
            "qpos_end_z": None,
        }
        for _episode in range(max_episodes):
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            task_summary["qpos_start_z"] = float(data.qpos[2]) if len(data.qpos) > 2 else None
            for _ in range(max_steps):
                mujoco.mj_step(model, data)
                task_summary["steps_performed"] += 1
                result["total_steps_performed"] += 1
            task_summary["qpos_end_z"] = float(data.qpos[2]) if len(data.qpos) > 2 else None
            task_summary["episodes_completed"] += 1
            result["rollouts_completed"] += 1
        result["tasks_completed"] += 1
        result["task_summaries"].append(task_summary)

    result["ok"] = (
        result["tasks_completed"] == task_count
        and result["rollouts_completed"] == task_count * max_episodes
        and result["total_steps_performed"] == task_count * max_episodes * max_steps
        and not result["libero_robosuite_benchmark_env_created"]
        and not result["policy_inference_performed"]
        and not result["training_performed"]
        and not result["benchmark_claims_made"]
    )
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
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

$bashCommand = "$wslPythonExecutable '$($probeScriptWsl.stdout)' $TaskCount $MaxEpisodes $MaxStepsPerEpisode"
$rolloutProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
$report.policy.rollout_attempted = $true
$report.rollout_probe = $rolloutProbe

if ($rolloutProbe.ok) {
    try {
        $stdoutText = [string]$rolloutProbe.stdout
        $jsonStart = $stdoutText.IndexOf("{")
        if ($jsonStart -lt 0) {
            throw "WSL rollout probe stdout did not contain a JSON object"
        }
        $parsed = $stdoutText.Substring($jsonStart) | ConvertFrom-Json
        $report.rollout_result = $parsed
        $report.bounded_tiny_diagnostic_rollout_passed = [bool]$parsed.ok
        $report.policy.rollouts_performed = [bool]$parsed.ok
        $report.decision = if ($parsed.ok) { "proceed" } else { "stop" }
        $report.reason = if ($parsed.ok) { "bounded tiny diagnostic MuJoCo rollout passed" } else { "bounded tiny diagnostic MuJoCo rollout failed: $($parsed.error)" }
    } catch {
        $report.reason = "failed to parse WSL diagnostic rollout JSON: $($_.Exception.Message)"
    }
} else {
    $report.reason = "WSL diagnostic rollout probe command failed"
}

$report.ready_for_benchmark_rollout = $false
$report.ready_for_paper_claim = $false
$report.recommended_next_step = if ($report.bounded_tiny_diagnostic_rollout_passed) {
    "Generate/refresh local status reports. Stop before benchmark rollouts, multi-seed rollouts, OpenVLA-OFT, training, external upload, or paper-level claims."
} else {
    "Diagnostic rollout did not pass. Stop before benchmark rollouts or claims; record the blocker."
}

Write-Reports -Report $report
exit 0
