param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [ValidateSet("auto", "windows", "wsl", "linux")]
    [string]$RuntimePlatform = "auto",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [int]$MaxSteps = 3,
    [int]$TimeoutSeconds = 120,
    [string]$JsonReportPath = "reports\bounded_simulator_reset_step_smoke_report.json",
    [string]$MarkdownReportPath = "reports\bounded_simulator_reset_step_smoke_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded simulator reset/step smoke"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script performs at most a tiny MuJoCo reset/step smoke. It does not create LIBERO/RoboSuite environments, rollout, train, run GPU jobs, install packages, download assets, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

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
        "# Bounded Simulator Reset/Step Smoke Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.bounded_simulator_reset_step_smoke_passed)",
        "- reset/step attempted: $($Report.policy.reset_step_smoke_attempted)",
        "- reset/step performed: $($Report.policy.reset_step_smoke_performed)",
        "- steps performed: $($Report.step_result.steps_performed)",
        "- rollouts performed: false",
        "- LIBERO/RoboSuite env created: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is a tiny MuJoCo reset/step smoke only. It is not rollout evidence, benchmark evidence, standard success, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 10
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_reset_step_smoke = $true
            task_local_gate_required = "ALLOW_SIMULATOR_RESET_STEP=1"
            max_steps = $MaxSteps
            installs_performed = $false
            downloads_performed = $false
            gpu_jobs_performed = $false
            training_performed = $false
            heavy_model_imports_performed = $false
            model_load_performed = $false
            model_inference_performed = $false
            policy_inference_performed = $false
            simulator_imports_performed = $false
            render_smoke_performed = $false
            reset_step_smoke_attempted = $false
            reset_step_smoke_performed = $false
            libero_robosuite_env_created = $false
            rollouts_performed = $false
            openvla_oft_executed = $false
            tokens_read_or_written = $false
            paper_grade_claims_made = $false
        }
        runtime = [ordered]@{
            requested_runtime_platform = $RuntimePlatform
            effective_runtime_platform = $null
            timeout_seconds = $TimeoutSeconds
            wsl_python = $WslPython
        }
        risk_plan = $null
        wsl = [ordered]@{}
        step_probe = $null
        step_result = [ordered]@{ ok = $false; steps_performed = 0 }
        bounded_simulator_reset_step_smoke_passed = $false
        ready_for_tiny_diagnostic_rollout_plan = $false
        ready_for_rollout = $false
        decision = "stop"
        reason = $null
        recommended_next_step = $null
    }
}

$dangerousGateNames = @(
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })

$gateValue = [Environment]::GetEnvironmentVariable("ALLOW_SIMULATOR_RESET_STEP")
$report = New-BaseReport
$runDir = Join-Path $RepoRoot "runs\simulator_reset_step_smoke"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

if ($MaxSteps -lt 1 -or $MaxSteps -gt 5) {
    $report.reason = "MaxSteps must be between 1 and 5 for bounded reset/step smoke."
    $report.recommended_next_step = "Use MaxSteps<=5 and rerun only after a green reset/step risk plan."
    Write-Reports -Report $report
    exit 0
}

if ($gateValue -ne "1") {
    $report.reason = "ALLOW_SIMULATOR_RESET_STEP=1 is required after a green reset/step risk assessment."
    $report.recommended_next_step = "Run scripts\58_plan_simulator_render_reset.ps1, then set ALLOW_SIMULATOR_RESET_STEP=1 only for this bounded reset/step smoke task if the plan says proceed."
    Write-Reports -Report $report
    exit 0
}

if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "bounded reset/step smoke refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset rollout, heavy-import, training, and OpenVLA gates before reset/step smoke."
    Write-Reports -Report $report
    exit 0
}

$planJson = Join-Path $runDir "simulator_render_reset_plan_report.json"
$planMd = Join-Path $runDir "simulator_render_reset_plan_report.md"
$savedResetGate = [Environment]::GetEnvironmentVariable("ALLOW_SIMULATOR_RESET_STEP")
Remove-Item Env:\ALLOW_SIMULATOR_RESET_STEP -ErrorAction SilentlyContinue
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\58_plan_simulator_render_reset.ps1") -PathsFile $PathsFile -RuntimePlatform $RuntimePlatform -JsonReportPath $planJson -MarkdownReportPath $planMd | Out-Null
} finally {
    if ($null -ne $savedResetGate) {
        $env:ALLOW_SIMULATOR_RESET_STEP = $savedResetGate
    }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planJson)) {
    $report.reason = "simulator render/reset-step planner failed"
    $report.recommended_next_step = "Fix planning report before any reset/step smoke."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planJson
$report.risk_plan = $plan
$report.runtime.effective_runtime_platform = $plan.risk_assessment.target_runtime_platform

if (-not $plan.ready_for_bounded_reset_step_smoke_plan) {
    $report.reason = "render/reset-step planner did not allow bounded reset/step smoke"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}

if ($plan.risk_assessment.target_runtime_platform -ne "wsl") {
    $report.reason = "bounded reset/step smoke currently supports WSL from Windows only"
    $report.recommended_next_step = "Use WSL2/Linux for reset/step smoke; keep native Windows as planning-only."
    Write-Reports -Report $report
    exit 0
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Install/configure WSL2 before simulator reset/step smoke."
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
    $report.recommended_next_step = "Fix WSL venv MuJoCo readiness before bounded reset/step smoke."
    Write-Reports -Report $report
    exit 0
}

$probeScriptWin = Join-Path $runDir "probe_reset_step.py"
$probeScript = @'
import json
import sys
import time

started = time.perf_counter()
result = {
    "ok": False,
    "error": None,
    "elapsed_seconds": None,
    "python": sys.version.split()[0],
    "mujoco_version": None,
    "steps_requested": 0,
    "steps_performed": 0,
    "qpos_initial": None,
    "qpos_final": None,
    "qvel_final": None,
    "libero_robosuite_env_created": False,
    "rollout_performed": False,
}

try:
    import mujoco

    steps = int(sys.argv[1])
    result["steps_requested"] = steps
    result["mujoco_version"] = getattr(mujoco, "__version__", "unknown")
    xml = """
    <mujoco model="tca_map_reset_step_smoke">
      <option timestep="0.002"/>
      <worldbody>
        <body name="body" pos="0 0 0.2">
          <freejoint/>
          <geom type="sphere" size="0.05" mass="0.1" rgba="0.1 0.4 0.8 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    result["qpos_initial"] = [float(x) for x in data.qpos[: min(3, len(data.qpos))]]
    for _ in range(steps):
        mujoco.mj_step(model, data)
        result["steps_performed"] += 1
    result["qpos_final"] = [float(x) for x in data.qpos[: min(3, len(data.qpos))]]
    result["qvel_final"] = [float(x) for x in data.qvel[: min(3, len(data.qvel))]]
    result["ok"] = result["steps_performed"] == steps
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
print(json.dumps(result, indent=2, sort_keys=True))
'@
$probeScript | Set-Content -LiteralPath $probeScriptWin -Encoding UTF8
$probeScriptWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($probeScriptWin.Replace("\", "/"))) -TimeoutSec 30
if (-not $probeScriptWsl.ok) {
    $report.reason = "failed to map reset/step probe script into WSL"
    $report.recommended_next_step = "Fix WSL path mapping before reset/step smoke."
    Write-Reports -Report $report
    exit 0
}

$bashCommand = "$wslPythonExecutable '$($probeScriptWsl.stdout)' $MaxSteps"
$stepProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
$report.policy.reset_step_smoke_attempted = $true
$report.step_probe = $stepProbe

if ($stepProbe.ok) {
    try {
        $stdoutText = [string]$stepProbe.stdout
        $jsonStart = $stdoutText.IndexOf("{")
        if ($jsonStart -lt 0) {
            throw "WSL reset/step probe stdout did not contain a JSON object"
        }
        $parsed = $stdoutText.Substring($jsonStart) | ConvertFrom-Json
        $report.step_result = $parsed
        $report.bounded_simulator_reset_step_smoke_passed = [bool]$parsed.ok
        $report.policy.reset_step_smoke_performed = [bool]$parsed.ok
        $report.decision = if ($parsed.ok) { "proceed" } else { "stop" }
        $report.reason = if ($parsed.ok) { "bounded MuJoCo reset/step smoke passed" } else { "bounded MuJoCo reset/step probe failed: $($parsed.error)" }
    } catch {
        $report.reason = "failed to parse WSL reset/step probe JSON: $($_.Exception.Message)"
    }
} else {
    $report.reason = "WSL reset/step probe command failed"
}

$report.ready_for_tiny_diagnostic_rollout_plan = [bool]$report.bounded_simulator_reset_step_smoke_passed
$report.ready_for_rollout = $false
$report.recommended_next_step = if ($report.bounded_simulator_reset_step_smoke_passed) {
    "Create a separate bounded tiny diagnostic rollout risk assessment. Do not rollout or claim standard success until that separate gate passes."
} else {
    "Reset/step smoke did not pass. Stop before rollout or benchmark claims; record the reset/step blocker."
}

Write-Reports -Report $report
exit 0
