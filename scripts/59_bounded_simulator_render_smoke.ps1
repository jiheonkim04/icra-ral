param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [ValidateSet("auto", "windows", "wsl", "linux")]
    [string]$RuntimePlatform = "auto",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [ValidateSet("osmesa")]
    [string]$MuJoCoGl = "osmesa",
    [int]$TimeoutSeconds = 120,
    [string]$JsonReportPath = "reports\bounded_simulator_render_smoke_report.json",
    [string]$MarkdownReportPath = "reports\bounded_simulator_render_smoke_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded simulator render smoke"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script performs at most a tiny MuJoCo offscreen render smoke. It does not create or step LIBERO/RoboSuite environments, rollout, train, run GPU jobs, install packages, download assets, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

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
            return [ordered]@{
                ok = $false
                timed_out = $true
                returncode = $null
                stdout = ""
                stderr = "command timed out after $TimeoutSec seconds"
            }
        }
        return [ordered]@{
            ok = $process.ExitCode -eq 0
            timed_out = $false
            returncode = $process.ExitCode
            stdout = $process.StandardOutput.ReadToEnd().Trim().Replace("`0", "")
            stderr = $process.StandardError.ReadToEnd().Trim().Replace("`0", "")
        }
    } catch {
        return [ordered]@{
            ok = $false
            timed_out = $false
            returncode = $null
            stdout = ""
            stderr = $_.Exception.Message
        }
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
        "# Bounded Simulator Render Smoke Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.bounded_simulator_render_smoke_passed)",
        "- render attempted: $($Report.policy.render_smoke_attempted)",
        "- render performed: $($Report.policy.render_smoke_performed)",
        "- reset/step performed: false",
        "- rollouts performed: false",
        "- MuJoCo GL: $($Report.runtime.mujoco_gl)",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is a tiny offscreen render smoke only. It is not reset/step evidence, not rollout evidence, and not paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 10
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_render_smoke = $true
            task_local_gate_required = "ALLOW_SIMULATOR_RENDER_SMOKE=1"
            installs_performed = $false
            downloads_performed = $false
            gpu_jobs_performed = $false
            training_performed = $false
            heavy_model_imports_performed = $false
            model_load_performed = $false
            model_inference_performed = $false
            simulator_imports_performed = $false
            render_smoke_attempted = $false
            render_smoke_performed = $false
            reset_step_smoke_performed = $false
            rollouts_performed = $false
            simulator_environment_steps_performed = $false
            openvla_oft_executed = $false
            tokens_read_or_written = $false
            paper_grade_claims_made = $false
        }
        runtime = [ordered]@{
            requested_runtime_platform = $RuntimePlatform
            effective_runtime_platform = $null
            timeout_seconds = $TimeoutSeconds
            mujoco_gl = $MuJoCoGl
            wsl_python = $WslPython
        }
        risk_plan = $null
        wsl = [ordered]@{}
        render_probe = $null
        render_result = $null
        bounded_simulator_render_smoke_passed = $false
        ready_for_reset_step_smoke_plan = $false
        ready_for_rollout = $false
        decision = "stop"
        reason = $null
        recommended_next_step = $null
    }
}

$dangerousGateNames = @(
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING"
)
$dangerousGatesSet = @(
    $dangerousGateNames |
        Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" }
)

$gateValue = [Environment]::GetEnvironmentVariable("ALLOW_SIMULATOR_RENDER_SMOKE")
$report = New-BaseReport
$runDir = Join-Path $RepoRoot "runs\simulator_render_smoke"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

if ($gateValue -ne "1") {
    $report.reason = "ALLOW_SIMULATOR_RENDER_SMOKE=1 is required after a green render-smoke risk assessment."
    $report.recommended_next_step = "Run scripts\58_plan_simulator_render_reset.ps1, then set ALLOW_SIMULATOR_RENDER_SMOKE=1 only for this bounded render-smoke task if the plan says proceed."
    Write-Reports -Report $report
    exit 0
}

if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "bounded render smoke refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset reset/step, rollout, heavy-import, training, and OpenVLA gates before render smoke."
    Write-Reports -Report $report
    exit 0
}

$planJson = Join-Path $runDir "simulator_render_reset_plan_report.json"
$planMd = Join-Path $runDir "simulator_render_reset_plan_report.md"
$savedRenderGate = [Environment]::GetEnvironmentVariable("ALLOW_SIMULATOR_RENDER_SMOKE")
Remove-Item Env:\ALLOW_SIMULATOR_RENDER_SMOKE -ErrorAction SilentlyContinue
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\58_plan_simulator_render_reset.ps1") -PathsFile $PathsFile -RuntimePlatform $RuntimePlatform -JsonReportPath $planJson -MarkdownReportPath $planMd | Out-Null
} finally {
    if ($null -ne $savedRenderGate) {
        $env:ALLOW_SIMULATOR_RENDER_SMOKE = $savedRenderGate
    }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planJson)) {
    $report.reason = "simulator render/reset-step planner failed"
    $report.recommended_next_step = "Fix planning report before any render smoke."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planJson
$report.risk_plan = $plan
$report.runtime.effective_runtime_platform = $plan.risk_assessment.target_runtime_platform

if (-not $plan.ready_for_bounded_render_smoke_plan) {
    $report.reason = "render/reset-step planner did not allow bounded render smoke"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}

if ($plan.risk_assessment.target_runtime_platform -ne "wsl") {
    $report.reason = "bounded render smoke currently supports WSL from Windows only"
    $report.recommended_next_step = "Use WSL2/Linux for render smoke; keep native Windows as planning-only."
    Write-Reports -Report $report
    exit 0
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Install/configure WSL2 before simulator render smoke."
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
    $report.recommended_next_step = "Fix WSL venv MuJoCo readiness before bounded render smoke; do not install system graphics packages without a separate risk gate."
    Write-Reports -Report $report
    exit 0
}

$probeScriptWin = Join-Path $runDir "probe_render.py"
$probeScript = @'
import json
import os
import sys
import time

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "osmesa")

started = time.perf_counter()
result = {
    "ok": False,
    "error": None,
    "elapsed_seconds": None,
    "python": sys.version.split()[0],
    "mujoco_gl": os.environ["MUJOCO_GL"],
    "image_shape": None,
    "image_mean": None,
}

try:
    import mujoco

    xml = """
    <mujoco model="tca_map_render_smoke">
      <visual>
        <global offwidth="64" offheight="64"/>
      </visual>
      <worldbody>
        <light pos="0 0 3"/>
        <camera name="fixed" pos="0 -2 1" xyaxes="1 0 0 0 0.5 1"/>
        <geom name="floor" type="plane" size="1 1 0.01" rgba="0.2 0.3 0.4 1"/>
        <geom name="target" type="sphere" pos="0 0 0.2" size="0.2" rgba="0.8 0.1 0.1 1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data, camera="fixed")
        image = renderer.render()
    finally:
        renderer.close()

    result["ok"] = True
    result["mujoco_version"] = getattr(mujoco, "__version__", "unknown")
    result["image_shape"] = list(image.shape)
    result["image_mean"] = float(image.mean())
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"

result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
print(json.dumps(result, indent=2, sort_keys=True))
'@
$probeScript | Set-Content -LiteralPath $probeScriptWin -Encoding UTF8
$probeScriptWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($probeScriptWin.Replace("\", "/"))) -TimeoutSec 30
if (-not $probeScriptWsl.ok) {
    $report.reason = "failed to map render probe script into WSL"
    $report.recommended_next_step = "Fix WSL path mapping before render smoke."
    Write-Reports -Report $report
    exit 0
}

$bashCommand = "export MUJOCO_GL='$MuJoCoGl'; $wslPythonExecutable '$($probeScriptWsl.stdout)'"
$renderProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
$report.policy.render_smoke_attempted = $true
$report.render_probe = $renderProbe

if ($renderProbe.ok) {
    try {
        $stdoutText = [string]$renderProbe.stdout
        $jsonStart = $stdoutText.IndexOf("{")
        if ($jsonStart -lt 0) {
            throw "WSL render probe stdout did not contain a JSON object"
        }
        $parsed = $stdoutText.Substring($jsonStart) | ConvertFrom-Json
        $report.render_result = $parsed
        $report.bounded_simulator_render_smoke_passed = [bool]$parsed.ok
        $report.policy.render_smoke_performed = [bool]$parsed.ok
        $report.decision = if ($parsed.ok) { "proceed" } else { "stop" }
        if ($parsed.ok) {
            $report.reason = "bounded MuJoCo offscreen render smoke passed"
        } elseif ([string]$parsed.error -match "glGetError|OSMesa|OpenGL|GLContext") {
            $report.reason = "bounded MuJoCo offscreen render probe failed; likely OSMesa/offscreen GL is unavailable or misconfigured: $($parsed.error)"
        } else {
            $report.reason = "bounded MuJoCo offscreen render probe failed: $($parsed.error)"
        }
    } catch {
        $report.reason = "failed to parse WSL render probe JSON: $($_.Exception.Message)"
    }
} else {
    $report.reason = "WSL render probe command failed"
}

$report.ready_for_reset_step_smoke_plan = [bool]$report.bounded_simulator_render_smoke_passed
$report.ready_for_rollout = $false
$report.recommended_next_step = if ($report.bounded_simulator_render_smoke_passed) {
    "Create a separate bounded reset/step smoke branch. Do not rollout or claim standard success."
} else {
    "Render smoke did not pass. Stop before system graphics changes, reset/step, rollout, or benchmark claims; record the render blocker."
}

Write-Reports -Report $report
exit 0
