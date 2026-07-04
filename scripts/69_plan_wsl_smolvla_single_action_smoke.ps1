param(
    [string]$PolicyReadinessReportPath = "reports\libero_policy_rollout_readiness_plan_report.json",
    [string]$WslRuntimePlanPath = "reports\wsl_smolvla_runtime_setup_plan_report.json",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [int]$ExpectedRuntimeMinutes = 10,
    [double]$ExpectedVramGb = 0,
    [string]$JsonReportPath = "reports\wsl_smolvla_single_action_smoke_plan_report.json",
    [string]$MarkdownReportPath = "reports\wsl_smolvla_single_action_smoke_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "WSL SmolVLA single-action smoke planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not load models, import heavy VLA code, run inference, train, rollout, use GPU, download, execute OpenVLA-OFT, access tokens, or make paper claims."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

function Read-JsonFileIfPresent {
    param([string]$Path)
    $fullPath = Resolve-RepoPath -Path $Path
    if (-not (Test-Path -LiteralPath $fullPath)) {
        return [ordered]@{ present = $false; path = $fullPath; data = $null; error = $null }
    }
    try {
        $text = [System.IO.File]::ReadAllText($fullPath, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
        return [ordered]@{ present = $true; path = $fullPath; data = ($text | ConvertFrom-Json); error = $null }
    } catch {
        return [ordered]@{ present = $true; path = $fullPath; data = $null; error = $_.Exception.Message }
    }
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

function Get-FreeDiskGb {
    try {
        $drive = Get-PSDrive -Name "C" -ErrorAction Stop
        return [math]::Round(($drive.Free / 1GB), 3)
    } catch {
        return $null
    }
}

function Write-Reports {
    param([object]$Report)
    $jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
    $markdownFullPath = Resolve-RepoPath -Path $MarkdownReportPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# WSL SmolVLA Single-Action Smoke Plan Report",
        "",
        "- decision: $($Report.decision)",
        "- ready for execution: $($Report.ready_for_wsl_smolvla_single_action_smoke)",
        "- expected runtime minutes: $($Report.risk_assessment.expected_runtime_minutes)",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is planning-only. It performs no model load, inference, rollout, training, GPU job, download, OpenVLA-OFT execution, token access, or paper claim."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 12
}

$dangerousGateNames = @(
    "ALLOW_WSL_SMOLVLA_SINGLE_ACTION",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })

$policyRead = Read-JsonFileIfPresent -Path $PolicyReadinessReportPath
$runtimeRead = Read-JsonFileIfPresent -Path $WslRuntimePlanPath
$policyReady = $false
if ($policyRead.present -and -not $policyRead.error -and $null -ne $policyRead.data) {
    $policyReady = [bool]$policyRead.data.ready_for_tiny_learned_policy_rollout_execution
}
$runtimeReady = $false
if ($runtimeRead.present -and -not $runtimeRead.error -and $null -ne $runtimeRead.data) {
    $runtimeReady = [bool]$runtimeRead.data.ready_for_wsl_smolvla_runtime
}

$wslPythonProbe = if ($null -ne (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "test -x $WslPython && $WslPython --version") -TimeoutSec 30
} else {
    [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" }
}

$stopReasons = New-Object System.Collections.Generic.List[string]
if ($dangerousGatesSet.Count -gt 0) { $stopReasons.Add("planning-only single-action smoke refuses execution gates: $($dangerousGatesSet -join ', ')") }
if (-not $policyReady) { $stopReasons.Add("learned-policy rollout readiness planner is not execution-ready") }
if (-not $runtimeReady) { $stopReasons.Add("WSL SmolVLA runtime is not ready") }
if (-not $wslPythonProbe.ok) { $stopReasons.Add("selected WSL Python is not executable") }
if ($ExpectedRuntimeMinutes -gt 30) { $stopReasons.Add("expected runtime exceeds 30 minutes") }
if ($ExpectedVramGb -gt 14) { $stopReasons.Add("expected VRAM exceeds 14 GB") }

$decision = if ($stopReasons.Count -eq 0) { "proceed" } else { "stop" }
$reason = if ($stopReasons.Count -eq 0) {
    "WSL-only SmolVLA runtime and local assets are ready for one synthetic single-action smoke before rollout."
} else {
    $stopReasons -join "; "
}

$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
        downloads_performed = $false
        installs_performed = $false
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
    risk_assessment = [ordered]@{
        task = "bounded WSL SmolVLA single-action smoke planning"
        command = "scripts\70_bounded_wsl_smolvla_single_action_smoke.ps1"
        source = "local SmolVLA checkpoint and WSL venv runtime"
        target_output_paths = @("reports\wsl_smolvla_single_action_smoke_report.json", "reports\wsl_smolvla_single_action_smoke_report.md")
        expected_size_gb = 0
        disk_free_before_gb = Get-FreeDiskGb
        expected_runtime_minutes = $ExpectedRuntimeMinutes
        expected_ram_gb = 10
        expected_vram_gb = $ExpectedVramGb
        token_login_license_payment_needed = $false
        simulator_will_run = $false
        rollout_will_run = $false
        model_load_will_run = $true
        single_action_inference_will_run = $true
        training_will_run = $false
        openvla_oft_will_run = $false
        decision = $decision
        reason = $reason
    }
    prerequisites = [ordered]@{
        policy_readiness = [ordered]@{ report_present = [bool]$policyRead.present; report_path = $policyRead.path; report_error = $policyRead.error; ready = $policyReady }
        wsl_runtime = [ordered]@{ report_present = [bool]$runtimeRead.present; report_path = $runtimeRead.path; report_error = $runtimeRead.error; ready = $runtimeReady }
        wsl_python = $wslPythonProbe
    }
    dangerous_execution_gates_set = @($dangerousGatesSet)
    ready_for_wsl_smolvla_single_action_smoke = [bool]($decision -eq "proceed")
    ready_for_tiny_learned_policy_rollout = $false
    ready_for_paper_claim = $false
    stop_reasons = @($stopReasons)
    decision = $decision
    recommended_next_step = if ($decision -eq "proceed") {
        "Run scripts\70_bounded_wsl_smolvla_single_action_smoke.ps1 with task-local ALLOW_WSL_SMOLVLA_SINGLE_ACTION=1."
    } else {
        "Resolve listed blockers before any WSL SmolVLA load/action smoke."
    }
}

Write-Reports -Report $report
exit 0
