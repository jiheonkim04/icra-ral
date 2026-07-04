param(
    [string]$VenvPath = '$HOME/.venvs/tca_map_sim',
    [string]$PlanReportPath = "reports\wsl_smolvla_runtime_setup_plan_report.json",
    [int]$TimeoutSeconds = 1800,
    [string]$JsonReportPath = "reports\wsl_smolvla_runtime_setup_report.json",
    [string]$MarkdownReportPath = "reports\wsl_smolvla_runtime_setup_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "WSL SmolVLA runtime dependency setup"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is bounded to WSL venv-local package setup. It does not load SmolVLA, run inference, train, rollout, use GPU, execute OpenVLA-OFT, access tokens, or make paper claims."

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
    $Report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# WSL SmolVLA Runtime Setup Report",
        "",
        "- decision: $($Report.decision)",
        "- executed: $($Report.execution.executed)",
        "- setup passed: $($Report.execution.setup_passed)",
        "- missing modules after setup: $(@($Report.execution.missing_modules_after) -join ', ')",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is WSL venv package setup evidence only. It is not model-load, inference, rollout, training, benchmark, SOTA, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 12
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_wsl_smolvla_runtime_setup = $true
            task_local_gate_required = "ALLOW_WSL_SMOLVLA_RUNTIME_SETUP=1"
            package_installs_performed = $false
            package_downloads_performed = $false
            apt_installs_performed = $false
            sudo_used = $false
            heavy_model_imports_performed = $false
            model_load_performed = $false
            model_inference_performed = $false
            gpu_jobs_performed = $false
            training_performed = $false
            rollouts_performed = $false
            simulator_environment_created = $false
            openvla_oft_executed = $false
            tokens_read_or_written = $false
            paper_grade_claims_made = $false
        }
        plan = $null
        execution = [ordered]@{
            gate_present = [Environment]::GetEnvironmentVariable("ALLOW_WSL_SMOLVLA_RUNTIME_SETUP") -eq "1"
            executed = $false
            setup_passed = $false
            commands = @()
            module_probe_after = $null
            missing_modules_after = @()
        }
        decision = "stop"
        reason = $null
        recommended_next_step = $null
    }
}

$dangerousGateNames = @(
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })

$report = New-BaseReport
if (-not $report.execution.gate_present) {
    $report.reason = "ALLOW_WSL_SMOLVLA_RUNTIME_SETUP=1 is required for bounded WSL SmolVLA runtime setup."
    $report.recommended_next_step = "Run scripts\67_plan_wsl_smolvla_runtime_setup.ps1 first, then set ALLOW_WSL_SMOLVLA_RUNTIME_SETUP=1 only for this setup task if the plan says proceed."
    Write-Reports -Report $report
    exit 0
}
if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "WSL SmolVLA runtime setup refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset rollout, OpenVLA, heavy-import, GPU, simulator, and training gates before package setup."
    Write-Reports -Report $report
    exit 0
}

$planRunPath = Join-Path $RepoRoot "runs\wsl_smolvla_runtime_setup\plan_report.json"
$planRunMd = Join-Path $RepoRoot "runs\wsl_smolvla_runtime_setup\plan_report.md"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $planRunPath) | Out-Null

& powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\67_plan_wsl_smolvla_runtime_setup.ps1") `
    -VenvPath $VenvPath `
    -JsonReportPath $planRunPath `
    -MarkdownReportPath $planRunMd | Out-Null
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planRunPath)) {
    $report.reason = "WSL SmolVLA runtime setup planner failed"
    $report.recommended_next_step = "Fix the planner before package setup."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planRunPath
$report.plan = $plan
if ($plan.decision -ne "proceed") {
    $report.reason = "planner did not authorize setup: $($plan.reason)"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}
if (-not $plan.setup_required) {
    $report.execution.setup_passed = $true
    $report.decision = "proceed"
    $report.reason = "WSL SmolVLA runtime setup is already complete"
    $report.recommended_next_step = "Rerun scripts\66_plan_libero_policy_rollout_readiness.ps1."
    Write-Reports -Report $report
    exit 0
}

if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Configure WSL before runtime setup."
    Write-Reports -Report $report
    exit 0
}

$installScript = @'
set -euo pipefail
VENV_PATH="$1"
PY="$VENV_PATH/bin/python"
if [ ! -x "$PY" ]; then
  echo "Missing WSL venv python: $PY" >&2
  exit 20
fi
"$PY" -m pip install --upgrade setuptools wheel
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
"$PY" -m pip install transformers==4.57.6 safetensors==0.8.0 huggingface_hub==0.35.3 accelerate==1.14.0 num2words==0.5.14
"$PY" -m pip install --no-deps lerobot==0.4.4
'@

$runDir = Join-Path $RepoRoot "runs\wsl_smolvla_runtime_setup"
$installWin = Join-Path $runDir "install_wsl_smolvla_runtime.sh"
$probeWin = Join-Path $runDir "probe_wsl_smolvla_runtime_after.py"
$installScript | Set-Content -LiteralPath $installWin -Encoding UTF8
@'
import importlib.util
import json
import sys

required = [
    "torch",
    "torchvision",
    "transformers",
    "lerobot",
    "safetensors",
    "huggingface_hub",
    "accelerate",
    "num2words",
]
specs = {}
for name in required:
    try:
        specs[name] = importlib.util.find_spec(name) is not None
    except Exception:
        specs[name] = False
print(json.dumps({
    "ok": all(specs.values()),
    "python": sys.version.split()[0],
    "module_specs": specs,
    "heavy_imports_performed": False,
    "model_load_performed": False,
    "model_inference_performed": False,
    "gpu_jobs_performed": False,
    "training_performed": False,
    "rollouts_performed": False,
    "openvla_oft_executed": False,
}, sort_keys=True))
'@ | Set-Content -LiteralPath $probeWin -Encoding UTF8

$installWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($installWin.Replace("\", "/"))) -TimeoutSec 30
$probeWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($probeWin.Replace("\", "/"))) -TimeoutSec 30
if (-not ($installWsl.ok -and $probeWsl.ok)) {
    $report.reason = "failed to map WSL setup scripts"
    $report.recommended_next_step = "Fix WSL path mapping before package setup."
    Write-Reports -Report $report
    exit 0
}

$install = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "bash '$($installWsl.stdout)' '$VenvPath'") -TimeoutSec $TimeoutSeconds
$report.execution.executed = $true
$report.policy.package_installs_performed = $true
$report.policy.package_downloads_performed = $true
$report.execution.commands = @([ordered]@{ label = "install_wsl_smolvla_runtime"; result = $install })

if (-not $install.ok) {
    $report.reason = "WSL SmolVLA runtime package setup failed"
    $report.recommended_next_step = "Inspect setup stderr/stdout and reduce package scope or adjust pinned versions without changing CUDA/driver/system packages."
    Write-Reports -Report $report
    exit 0
}

$probe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$VenvPath/bin/python '$($probeWsl.stdout)'") -TimeoutSec 120
$report.execution.module_probe_after = $probe
if ($probe.ok) {
    try {
        $start = ([string]$probe.stdout).IndexOf("{")
        if ($start -lt 0) { throw "probe stdout did not contain JSON" }
        $parsed = ([string]$probe.stdout).Substring($start) | ConvertFrom-Json
        $missing = New-Object System.Collections.Generic.List[string]
        $parsed.module_specs.PSObject.Properties | ForEach-Object {
            if (-not [bool]$_.Value) { $missing.Add($_.Name) | Out-Null }
        }
        $report.execution.missing_modules_after = @($missing)
        $report.execution.setup_passed = [bool]($parsed.ok -and $missing.Count -eq 0)
    } catch {
        $report.reason = "failed to parse WSL runtime module probe after setup: $($_.Exception.Message)"
    }
}

if ($report.execution.setup_passed) {
    $report.decision = "proceed"
    $report.reason = "WSL SmolVLA runtime module-spec setup passed"
    $report.recommended_next_step = "Rerun scripts\66_plan_libero_policy_rollout_readiness.ps1; if it is green, create the tiny learned-policy rollout runner."
} else {
    $report.reason = "WSL SmolVLA runtime setup did not satisfy all module specs"
    $report.recommended_next_step = "Inspect missing modules and install only the minimal missing WSL venv packages after a new risk assessment."
}

Write-Reports -Report $report
exit 0
