param(
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$JsonReportPath = "reports\wsl_smolvla_single_action_smoke_report.json",
    [string]$MarkdownReportPath = "reports\wsl_smolvla_single_action_smoke_report.md",
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded WSL SmolVLA single-action smoke"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script runs one WSL CPU synthetic SmolVLA action only. It does not create simulator environments, rollout, train, use GPU, download, execute OpenVLA-OFT, access tokens, or make paper claims."

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
    $Report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# WSL SmolVLA Single-Action Smoke Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.wsl_smolvla_single_action_smoke_passed)",
        "- model load performed: $($Report.policy.model_load_performed)",
        "- single action inference performed: $($Report.policy.single_action_inference_performed)",
        "- rollouts performed: false",
        "- paper-grade claim: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is bounded WSL model-load/action interface evidence only. It is not rollout evidence, benchmark success, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 12
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            bounded_wsl_single_action_smoke = $true
            task_local_gate_required = "ALLOW_WSL_SMOLVLA_SINGLE_ACTION=1"
            downloads_performed = $false
            installs_performed = $false
            simulator_environment_created = $false
            rollouts_performed = $false
            benchmark_rollouts_performed = $false
            heavy_model_imports_performed = $false
            model_load_performed = $false
            single_action_inference_performed = $false
            model_inference_performed = $false
            gpu_jobs_performed = $false
            training_performed = $false
            openvla_oft_executed = $false
            tokens_read_or_written = $false
            paper_grade_claims_made = $false
        }
        plan = $null
        wsl = [ordered]@{}
        smoke_command = $null
        smoke_result = $null
        wsl_smolvla_single_action_smoke_passed = $false
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
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })

$report = New-BaseReport
if ([Environment]::GetEnvironmentVariable("ALLOW_WSL_SMOLVLA_SINGLE_ACTION") -ne "1") {
    $report.reason = "ALLOW_WSL_SMOLVLA_SINGLE_ACTION=1 is required after a green WSL single-action smoke plan."
    $report.recommended_next_step = "Run scripts\69_plan_wsl_smolvla_single_action_smoke.ps1 first, then set ALLOW_WSL_SMOLVLA_SINGLE_ACTION=1 only for this bounded task if the plan says proceed."
    Write-Reports -Report $report
    exit 0
}
if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "WSL SmolVLA single-action smoke refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset rollout, OpenVLA, GPU, simulator, and training gates before WSL single-action smoke."
    Write-Reports -Report $report
    exit 0
}

$planJson = Join-Path $RepoRoot "runs\wsl_smolvla_single_action_smoke\plan_report.json"
$planMd = Join-Path $RepoRoot "runs\wsl_smolvla_single_action_smoke\plan_report.md"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $planJson) | Out-Null
$savedGate = [Environment]::GetEnvironmentVariable("ALLOW_WSL_SMOLVLA_SINGLE_ACTION")
Remove-Item Env:\ALLOW_WSL_SMOLVLA_SINGLE_ACTION -ErrorAction SilentlyContinue
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\69_plan_wsl_smolvla_single_action_smoke.ps1") `
        -JsonReportPath $planJson `
        -MarkdownReportPath $planMd | Out-Null
} finally {
    if ($null -ne $savedGate) { $env:ALLOW_WSL_SMOLVLA_SINGLE_ACTION = $savedGate }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planJson)) {
    $report.reason = "WSL single-action smoke planner failed"
    $report.recommended_next_step = "Fix the planner before WSL single-action smoke."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planJson
$report.plan = $plan
if (-not $plan.ready_for_wsl_smolvla_single_action_smoke) {
    $report.reason = "planner did not authorize WSL single-action smoke"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}

if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Configure WSL before WSL SmolVLA single-action smoke."
    Write-Reports -Report $report
    exit 0
}

$repoWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($RepoRoot.Replace("\", "/"))) -TimeoutSec 30
$ckptWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/checkpoints/smolvla") -TimeoutSec 30
$ckptRootWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/checkpoints") -TimeoutSec 30
$hfHomeWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", "C:/assets/hf_home") -TimeoutSec 30
$reportWslPath = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ((Resolve-RepoPath -Path $JsonReportPath).Replace("\", "/"))) -TimeoutSec 30
$pythonProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "test -x $WslPython && $WslPython --version") -TimeoutSec 30
$report.wsl = [ordered]@{
    repo_root = $repoWsl
    smolvla_ckpt = $ckptWsl
    checkpoint_root = $ckptRootWsl
    hf_home = $hfHomeWsl
    report_path = $reportWslPath
    python = $pythonProbe
}
if (-not ($repoWsl.ok -and $ckptWsl.ok -and $ckptRootWsl.ok -and $hfHomeWsl.ok -and $reportWslPath.ok -and $pythonProbe.ok)) {
    $report.reason = "WSL path or Python probe failed"
    $report.recommended_next_step = "Fix WSL path/Python readiness before single-action smoke."
    Write-Reports -Report $report
    exit 0
}

$bashCommand = @"
export PYTHONPATH='$($repoWsl.stdout)';
export ALLOW_HEAVY_IMPORT=1;
export ALLOW_SINGLE_SAMPLE_INFERENCE=1;
export HF_HUB_OFFLINE=1;
export TRANSFORMERS_OFFLINE=1;
export HF_HOME='$($hfHomeWsl.stdout)';
$WslPython -m tca_map.smolvla.single_sample_interface_smoke --smolvla-ckpt '$($ckptWsl.stdout)' --checkpoint-root '$($ckptRootWsl.stdout)' --hf-home '$($hfHomeWsl.stdout)' --report-path '$($reportWslPath.stdout)' --device cpu --task 'put the object on the target'
"@
$innerReportPath = Resolve-RepoPath -Path $JsonReportPath
Remove-Item -LiteralPath $innerReportPath -ErrorAction SilentlyContinue
$smoke = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSeconds
$report.smoke_command = [ordered]@{ timeout_seconds = $TimeoutSeconds; device = "cpu"; synthetic_input_only = $true }
$report.smoke_result = $smoke

if (Test-Path -LiteralPath $innerReportPath) {
    try {
        $inner = Read-JsonFile -Path $JsonReportPath
        $passed = [bool]$inner.result.passed
        $report.wsl_smolvla_single_action_smoke_passed = $passed
        $report.policy.heavy_model_imports_performed = [bool]$inner.policy.heavy_model_imports_performed
        $report.policy.model_load_performed = [bool]$inner.policy.model_load_performed
        $report.policy.single_action_inference_performed = [bool]$inner.policy.single_sample_model_inference_performed
        $report.policy.model_inference_performed = [bool]$inner.policy.model_inference_performed
        $report.decision = if ($passed) { "proceed" } else { "stop" }
        $report.reason = if ($passed) { "WSL SmolVLA single-action smoke passed" } else { [string]$inner.result.blocked_reason }
        if (-not $smoke.ok -and [string]::IsNullOrWhiteSpace($report.reason)) {
            $report.reason = "WSL SmolVLA single-action smoke command failed with return code $($smoke.returncode)"
        }
    } catch {
        $report.reason = "failed to parse inner WSL single-action report: $($_.Exception.Message)"
    }
} else {
    $report.reason = "WSL SmolVLA single-action smoke command failed or timed out"
}

$report.recommended_next_step = if ($report.wsl_smolvla_single_action_smoke_passed) {
    "Create the separately gated tiny learned-policy LIBERO rollout runner with max 1 task and max 1-5 steps."
} else {
    "Inspect the WSL single-action smoke blocker before any learned-policy rollout."
}

Write-Reports -Report $report
exit 0
