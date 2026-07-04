param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [ValidateSet("auto", "windows", "wsl", "linux")]
    [string]$RuntimePlatform = "auto",
    [int]$TimeoutSeconds = 120,
    [string]$JsonReportPath = "reports\bounded_simulator_import_smoke_report.json",
    [string]$MarkdownReportPath = "reports\bounded_simulator_import_smoke_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded simulator import smoke"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script performs at most bounded import-only checks. It does not install packages, download assets, render, rollout, train, run GPU jobs, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

function Convert-PathForWslArg {
    param([string]$Path)
    return $Path.Replace("\", "/")
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
            return [ordered]@{
                ok = $false
                timed_out = $true
                returncode = $null
                stdout = ""
                stderr = "command timed out after $TimeoutSec seconds"
            }
        }
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        return [ordered]@{
            ok = $process.ExitCode -eq 0
            timed_out = $false
            returncode = $process.ExitCode
            stdout = $stdout.Trim().Replace("`0", "")
            stderr = $stderr.Trim().Replace("`0", "")
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
    $text = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
    $text = $text.TrimStart([char]0xFEFF)
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
        "# Bounded Simulator Import Smoke Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.bounded_simulator_import_smoke_passed)",
        "- runtime platform: $($Report.runtime.effective_runtime_platform)",
        "- imports attempted: $($Report.policy.simulator_imports_attempted)",
        "- render smoke performed: false",
        "- rollouts performed: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is import-only readiness evidence. It is not standard success, not rollout success, and not paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 10
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_import_smoke = $true
            task_local_gate_required = "ALLOW_SIMULATOR_IMPORT_SMOKE=1"
            downloads_performed = $false
            installs_performed = $false
            gpu_jobs_performed = $false
            training_performed = $false
            heavy_model_imports_performed = $false
            model_load_performed = $false
            model_inference_performed = $false
            simulator_imports_attempted = $false
            simulator_imports_performed = $false
            render_smoke_performed = $false
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
        }
        planner = $null
        wsl = [ordered]@{}
        import_results = @()
        bounded_simulator_import_smoke_passed = $false
        ready_for_render_smoke = $false
        ready_for_rollout = $false
        decision = "stop"
        reason = $null
        recommended_next_step = $null
    }
}

$gateValue = [Environment]::GetEnvironmentVariable("ALLOW_SIMULATOR_IMPORT_SMOKE")
$report = New-BaseReport
$runDir = Join-Path $RepoRoot "runs\simulator_import_smoke"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

if ($gateValue -ne "1") {
    $report.reason = "ALLOW_SIMULATOR_IMPORT_SMOKE=1 is required after a green risk assessment before simulator imports are attempted."
    $report.recommended_next_step = "Run risk assessment and set ALLOW_SIMULATOR_IMPORT_SMOKE=1 only for a bounded import-smoke task. Do not render or rollout."
    Write-Reports -Report $report
    exit 0
}

$plannerJson = Join-Path $runDir "simulator_readiness_plan_report.json"
$plannerMd = Join-Path $runDir "simulator_readiness_plan_report.md"
& powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\43_plan_simulator_readiness.ps1") -PathsFile $PathsFile -RuntimePlatform $RuntimePlatform -JsonReportPath $plannerJson -MarkdownReportPath $plannerMd | Out-Null
if ($LASTEXITCODE -ne 0) {
    $report.reason = "simulator readiness planner failed"
    $report.recommended_next_step = "Fix simulator readiness planner failure before any import smoke."
    Write-Reports -Report $report
    exit 0
}

$planner = Read-JsonFile -Path $plannerJson
$report.planner = $planner
$report.runtime.effective_runtime_platform = $planner.host.effective_runtime_platform

if (-not $planner.ready_for_simulator_import_smoke) {
    $report.reason = "simulator readiness planner did not allow import smoke"
    $report.recommended_next_step = $planner.recommended_next_step
    Write-Reports -Report $report
    exit 0
}

if ($planner.host.effective_runtime_platform -ne "wsl") {
    $report.reason = "bounded simulator import smoke currently supports WSL from Windows only"
    $report.recommended_next_step = "Use WSL2/Linux for simulator import smoke; keep native Windows as planning-only."
    Write-Reports -Report $report
    exit 0
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -eq $wslCommand) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Install/configure WSL2 before simulator import smoke."
    Write-Reports -Report $report
    exit 0
}

$liberoRootWin = $planner.paths.libero_root.path
$robosuiteRootWin = $planner.paths.robosuite_root.path
$liberoRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", (Convert-PathForWslArg -Path $liberoRootWin)) -TimeoutSec 30
$robosuiteRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", (Convert-PathForWslArg -Path $robosuiteRootWin)) -TimeoutSec 30
$pythonVersion = Invoke-SafeCommand -Command @("wsl", "python3", "--version") -TimeoutSec 30
$report.wsl = [ordered]@{
    libero_root = $liberoRootWsl
    robosuite_root = $robosuiteRootWsl
    python3_version = $pythonVersion
}

if (-not ($liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $pythonVersion.ok)) {
    $report.reason = "WSL path conversion or python3 probe failed"
    $report.recommended_next_step = "Fix WSL path/python readiness before simulator import smoke."
    Write-Reports -Report $report
    exit 0
}

$probeScriptWin = Join-Path $runDir "probe_imports.py"
$probeScript = @'
import importlib
import json
import os
import sys
import time

libero_root = os.environ["TCA_MAP_LIBERO_ROOT_WSL"]
robosuite_root = os.environ["TCA_MAP_ROBOSUITE_ROOT_WSL"]
for path in [libero_root, robosuite_root]:
    if path and path not in sys.path:
        sys.path.insert(0, path)

modules = ["robosuite", "libero"]
results = []
started = time.perf_counter()
for name in modules:
    item = {"module": name, "ok": False, "error": None, "elapsed_seconds": None}
    module_started = time.perf_counter()
    try:
        importlib.import_module(name)
        item["ok"] = True
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
    item["elapsed_seconds"] = round(time.perf_counter() - module_started, 6)
    results.append(item)

print(json.dumps({
    "python": sys.version.split()[0],
    "sys_path_prefix": sys.path[:4],
    "imports": results,
    "elapsed_seconds": round(time.perf_counter() - started, 6),
}, indent=2, sort_keys=True))
'@
$probeScript | Set-Content -LiteralPath $probeScriptWin -Encoding UTF8
$probeScriptWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", (Convert-PathForWslArg -Path $probeScriptWin)) -TimeoutSec 30
if (-not $probeScriptWsl.ok) {
    $report.reason = "failed to map probe script into WSL"
    $report.recommended_next_step = "Fix WSL path mapping before simulator import smoke."
    Write-Reports -Report $report
    exit 0
}

$bashCommand = "export TCA_MAP_LIBERO_ROOT_WSL='$($liberoRootWsl.stdout)'; export TCA_MAP_ROBOSUITE_ROOT_WSL='$($robosuiteRootWsl.stdout)'; export MUJOCO_GL=disable; python3 '$($probeScriptWsl.stdout)'"
$importProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
$report.policy.simulator_imports_attempted = $true
$report.policy.simulator_imports_performed = $true
$report.import_probe = $importProbe

if ($importProbe.ok) {
    try {
        $stdoutText = [string]$importProbe.stdout
        $jsonStart = $stdoutText.IndexOf("{")
        if ($jsonStart -lt 0) {
            throw "WSL import probe stdout did not contain a JSON object"
        }
        $parsed = $stdoutText.Substring($jsonStart) | ConvertFrom-Json
        $report.import_results = @($parsed.imports)
        $allImportsOk = $true
        foreach ($item in $report.import_results) {
            if (-not $item.ok) { $allImportsOk = $false }
        }
        $report.bounded_simulator_import_smoke_passed = [bool]$allImportsOk
        $report.decision = if ($allImportsOk) { "proceed" } else { "stop" }
        $report.reason = if ($allImportsOk) { "bounded import-only smoke passed" } else { "one or more simulator package imports failed" }
    } catch {
        $report.reason = "failed to parse WSL import probe JSON: $($_.Exception.Message)"
    }
} else {
    $report.reason = "WSL import probe command failed"
}

$report.ready_for_render_smoke = $false
$report.ready_for_rollout = $false
$report.recommended_next_step = if ($report.bounded_simulator_import_smoke_passed) {
    "Create a separate bounded render-smoke risk gate if needed. Do not rollout or claim standard success."
} else {
    "Resolve simulator import errors in WSL/Linux before any render smoke or rollout."
}

Write-Reports -Report $report
exit 0
