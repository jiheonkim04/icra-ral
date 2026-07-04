param(
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$TaskSuite = "libero_10",
    [int]$StartTaskId = 0,
    [int]$TaskCount = 1,
    [int]$MaxStepsPerVariant = 10,
    [int]$CameraSize = 64,
    [string]$ActionAdapterStrategy = "policy_6d_delta_pose_plus_gripper_zero_hold",
    [double]$ActionScale = 1.0,
    [string[]]$PromptStrategies = @("stem_spaces", "bddl_language", "bddl_language_period"),
    [int]$TimeoutSecondsPerVariant = 1800,
    [string]$JsonReportPath = "reports\prompt_format_diagnostic_report.json",
    [string]$MarkdownReportPath = "reports\prompt_format_diagnostic_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded prompt-format diagnostic"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script runs only a bounded one-task prompt-format diagnostic. It does not download, train, run GPU jobs, execute OpenVLA-OFT, run multi-seed evaluation, or make benchmark/SOTA/paper-grade claims."

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

function ConvertTo-SafeName {
    param([string]$Value)
    return ($Value -replace '[^A-Za-z0-9_.-]', '_')
}

function Write-Reports {
    param([object]$Report)
    $jsonFullPath = Resolve-RepoPath -Path $JsonReportPath
    $markdownFullPath = Resolve-RepoPath -Path $MarkdownReportPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# Prompt-Format Diagnostic Report",
        "",
        "- decision: $($Report.decision)",
        "- passed: $($Report.prompt_format_diagnostic_passed)",
        "- prompt strategies: $($Report.prompt_strategies -join ', ')",
        "- variants completed: $($Report.result.variants_completed)",
        "- best prompt strategy: $($Report.result.best_prompt_strategy)",
        "- best diagnostic success rate: $($Report.result.best_diagnostic_success_rate)",
        "- best reward sum: $($Report.result.best_reward_sum)",
        "- rollout scaling ready: false",
        "- benchmark claim: false",
        "- paper-grade claim: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is prompt-format diagnostic evidence only. It is not standard success, benchmark success, SOTA evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 16
}

function New-BaseReport {
    return [ordered]@{
        policy = [ordered]@{
            prompt_format_diagnostic = $true
            task_local_gate_required = "ALLOW_PROMPT_FORMAT_DIAGNOSTIC=1"
            task_suite = $TaskSuite
            start_task_id = $StartTaskId
            task_count = $TaskCount
            max_steps_per_variant = $MaxStepsPerVariant
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
        action_adapter_strategy = $ActionAdapterStrategy
        action_scale = $ActionScale
        prompt_strategies = @($PromptStrategies)
        plan = $null
        wsl = [ordered]@{}
        variants = @()
        result = [ordered]@{
            variants_completed = 0
            best_prompt_strategy = $null
            best_diagnostic_success_rate = 0.0
            best_reward_sum = 0.0
        }
        prompt_format_diagnostic_passed = $false
        ready_for_rollout_scaling = $false
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
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
    "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC",
    "ALLOW_ACTION_SCALE_DIAGNOSTIC"
)
$dangerousGatesSet = @($dangerousGateNames | Where-Object { [Environment]::GetEnvironmentVariable($_) -eq "1" })
$allowedPromptStrategies = @("stem_spaces", "bddl_language", "bddl_language_period")

$report = New-BaseReport
if ([Environment]::GetEnvironmentVariable("ALLOW_PROMPT_FORMAT_DIAGNOSTIC") -ne "1") {
    $report.reason = "ALLOW_PROMPT_FORMAT_DIAGNOSTIC=1 is required after a green prompt-format diagnostic plan."
    $report.recommended_next_step = "Run scripts\86_plan_prompt_format_diagnostic.ps1 first, then set ALLOW_PROMPT_FORMAT_DIAGNOSTIC=1 only for this bounded task if the plan says proceed."
    Write-Reports -Report $report
    exit 0
}
if ($dangerousGatesSet.Count -gt 0) {
    $report.reason = "prompt-format diagnostic refuses unrelated execution gates: $($dangerousGatesSet -join ', ')"
    $report.recommended_next_step = "Unset broad rollout, benchmark, OpenVLA, GPU, matrix, adapter-strategy, action-scale, tiny-rollout, and training gates before this runner."
    Write-Reports -Report $report
    exit 0
}
if ($TaskCount -ne 1 -or $MaxStepsPerVariant -gt 10 -or $MaxStepsPerVariant -lt 1 -or $PromptStrategies.Count -gt 3 -or $PromptStrategies.Count -lt 1) {
    $report.reason = "prompt-format runner allows exactly one task, 1..10 steps per variant, and 1..3 prompt variants."
    $report.recommended_next_step = "Use TaskCount=1, MaxStepsPerVariant<=10, and no more than 3 prompt strategies."
    Write-Reports -Report $report
    exit 0
}
foreach ($strategy in $PromptStrategies) {
    if ($allowedPromptStrategies -notcontains $strategy) {
        $report.reason = "unsupported prompt strategy: $strategy"
        $report.recommended_next_step = "Use stem_spaces, bddl_language, or bddl_language_period."
        Write-Reports -Report $report
        exit 0
    }
}
if ([double]::IsNaN($ActionScale) -or [double]::IsInfinity($ActionScale) -or $ActionScale -le 0 -or $ActionScale -gt 2.0) {
    $report.reason = "action scale must be finite and in the range (0, 2]."
    $report.recommended_next_step = "Use conservative action scale 1.0 for prompt-format diagnostics."
    Write-Reports -Report $report
    exit 0
}

$planJson = Join-Path $RepoRoot "runs\prompt_format_diagnostic\plan_report.json"
$planMd = Join-Path $RepoRoot "runs\prompt_format_diagnostic\plan_report.md"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $planJson) | Out-Null
$savedGate = [Environment]::GetEnvironmentVariable("ALLOW_PROMPT_FORMAT_DIAGNOSTIC")
Remove-Item Env:\ALLOW_PROMPT_FORMAT_DIAGNOSTIC -ErrorAction SilentlyContinue
try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\86_plan_prompt_format_diagnostic.ps1") `
        -JsonReportPath $planJson `
        -MarkdownReportPath $planMd | Out-Null
} finally {
    if ($null -ne $savedGate) { $env:ALLOW_PROMPT_FORMAT_DIAGNOSTIC = $savedGate }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $planJson)) {
    $report.reason = "prompt-format diagnostic planner failed"
    $report.recommended_next_step = "Fix the planner before prompt-format diagnostics."
    Write-Reports -Report $report
    exit 0
}

$plan = Read-JsonFile -Path $planJson
$report.plan = $plan
if (-not $plan.ready_for_prompt_format_diagnostic_runner) {
    $report.reason = "planner did not authorize prompt-format diagnostic runner"
    $report.recommended_next_step = $plan.recommended_next_step
    Write-Reports -Report $report
    exit 0
}
if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
    $report.reason = "wsl command not found"
    $report.recommended_next_step = "Configure WSL before prompt-format diagnostics."
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
$pythonProbe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "test -x $WslPython && $WslPython --version") -TimeoutSec 30
$report.wsl = [ordered]@{
    repo_root = $repoWsl
    smolvla_ckpt = $smolvlaWsl
    checkpoint_root = $checkpointRootWsl
    hf_home = $hfHomeWsl
    libero_root = $liberoRootWsl
    robosuite_root = $robosuiteRootWsl
    libero_data_root = $liberoDataRootWsl
    python = $pythonProbe
}
if (-not ($repoWsl.ok -and $smolvlaWsl.ok -and $checkpointRootWsl.ok -and $hfHomeWsl.ok -and $liberoRootWsl.ok -and $robosuiteRootWsl.ok -and $liberoDataRootWsl.ok -and $pythonProbe.ok)) {
    $report.reason = "WSL path or Python probe failed"
    $report.recommended_next_step = "Fix WSL path/Python readiness before prompt-format diagnostics."
    Write-Reports -Report $report
    exit 0
}

$variantReports = @()
foreach ($promptStrategy in $PromptStrategies) {
    $safeStrategy = ConvertTo-SafeName -Value $promptStrategy
    $innerReportWin = Join-Path $RepoRoot "runs\prompt_format_diagnostic\$safeStrategy.json"
    $stdoutLogWin = Join-Path $RepoRoot "runs\prompt_format_diagnostic\$safeStrategy.stdout.log"
    Remove-Item -LiteralPath $innerReportWin -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stdoutLogWin -ErrorAction SilentlyContinue
    $innerReportWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($innerReportWin.Replace("\", "/"))) -TimeoutSec 30
    $stdoutLogWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($stdoutLogWin.Replace("\", "/"))) -TimeoutSec 30
    if (-not ($innerReportWsl.ok -and $stdoutLogWsl.ok)) {
        $variantReports += [ordered]@{ prompt_strategy = $promptStrategy; passed = $false; reason = "failed to map WSL report/log paths" }
        continue
    }
    $bashCommand = @"
export PYTHONPATH='$($repoWsl.stdout)';
export HF_HUB_OFFLINE=1;
export TRANSFORMERS_OFFLINE=1;
export MUJOCO_GL=osmesa;
export ALLOW_PROMPT_FORMAT_DIAGNOSTIC=1;
$WslPython -m tca_map.smolvla.libero_learned_policy_rollout --smolvla-ckpt '$($smolvlaWsl.stdout)' --checkpoint-root '$($checkpointRootWsl.stdout)' --hf-home '$($hfHomeWsl.stdout)' --libero-root '$($liberoRootWsl.stdout)' --robosuite-root '$($robosuiteRootWsl.stdout)' --libero-data-root '$($liberoDataRootWsl.stdout)' --task-suite '$TaskSuite' --start-task-id $StartTaskId --task-count $TaskCount --max-steps-per-task $MaxStepsPerVariant --camera-size $CameraSize --device cpu --prompt-strategy '$promptStrategy' --action-adapter-strategy '$ActionAdapterStrategy' --action-scale $ActionScale --report-path '$($innerReportWsl.stdout)' > '$($stdoutLogWsl.stdout)' 2>&1
printf 'prompt format diagnostic finished: $promptStrategy\n'
"@
    $bashCommand = $bashCommand -replace "`r", ""
    $rollout = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $bashCommand) -TimeoutSec $TimeoutSecondsPerVariant
    $variant = [ordered]@{
        prompt_strategy = $promptStrategy
        action_adapter_strategy = $ActionAdapterStrategy
        action_scale = $ActionScale
        command = [ordered]@{ timeout_seconds = $TimeoutSecondsPerVariant; task_suite = $TaskSuite; task_count = $TaskCount; max_steps_per_variant = $MaxStepsPerVariant; device = "cpu" }
        rollout_result_raw = $rollout
        report_path = $innerReportWin
        stdout_log_path = $stdoutLogWin
        passed = $false
        diagnostic_success_rate = 0.0
        reward_sum = 0.0
        observed_language = $null
        observed_prompt_strategy = $null
        last_env_action_preview = @()
        error = $null
    }
    if (Test-Path -LiteralPath $innerReportWin) {
        try {
            $inner = Read-JsonFile -Path $innerReportWin
            $variant.passed = [bool]$inner.result.passed
            $tasks = @($inner.tasks)
            $successCount = @($tasks | Where-Object { $_.success_check -eq $true }).Count
            $variant.diagnostic_success_rate = if ($tasks.Count -gt 0) { [math]::Round($successCount / $tasks.Count, 6) } else { 0.0 }
            $variant.reward_sum = [math]::Round((@($tasks | ForEach-Object { [double]$_.reward_sum }) | Measure-Object -Sum).Sum, 6)
            if ($tasks.Count -gt 0) {
                $lastTask = $tasks[-1]
                $variant.observed_language = $lastTask.language
                $variant.observed_prompt_strategy = $lastTask.prompt_strategy
                $variant.last_env_action_preview = @($lastTask.last_env_action_preview)
            }
            $report.policy.heavy_model_imports_performed = $report.policy.heavy_model_imports_performed -or [bool]$inner.policy.heavy_model_imports_performed
            $report.policy.model_load_performed = $report.policy.model_load_performed -or [bool]$inner.policy.model_load_performed
            $report.policy.model_inference_performed = $report.policy.model_inference_performed -or [bool]$inner.policy.model_inference_performed
            $report.policy.learned_policy_inference_performed = $report.policy.learned_policy_inference_performed -or [bool]$inner.policy.learned_policy_inference_performed
            $report.policy.simulator_environment_created = $report.policy.simulator_environment_created -or [bool]$inner.policy.simulator_environment_created
            $report.policy.diagnostic_rollouts_performed = $report.policy.diagnostic_rollouts_performed -or [bool]$inner.policy.diagnostic_rollouts_performed
        } catch {
            $variant.error = "failed to parse inner report: $($_.Exception.Message)"
        }
    } else {
        $variant.error = "inner report was not written"
    }
    $variantReports += $variant
}

$report.variants = $variantReports
$completed = @($variantReports | Where-Object { $_.passed -eq $true })
$report.result.variants_completed = $completed.Count
if ($completed.Count -gt 0) {
    $best = @($completed | Sort-Object -Property @{ Expression = { [double]$_.diagnostic_success_rate }; Descending = $true }, @{ Expression = { [double]$_.reward_sum }; Descending = $true })[0]
    $report.result.best_prompt_strategy = $best.prompt_strategy
    $report.result.best_diagnostic_success_rate = [double]$best.diagnostic_success_rate
    $report.result.best_reward_sum = [double]$best.reward_sum
}
$report.prompt_format_diagnostic_passed = ($completed.Count -eq $PromptStrategies.Count)
$report.decision = if ($report.prompt_format_diagnostic_passed) { "proceed" } else { "stop" }
$report.reason = if ($report.prompt_format_diagnostic_passed) {
    "bounded prompt-format diagnostic completed for all requested prompt strategies"
} else {
    "one or more prompt-format diagnostic variants failed"
}
$report.recommended_next_step = if ($report.prompt_format_diagnostic_passed) {
    "Summarize prompt-format diagnostics and decide whether camera-source or state-sufficiency diagnostics are needed. Keep evidence diagnostic only."
} else {
    "Inspect failed prompt-format variants before additional diagnostics."
}

Write-Reports -Report $report
exit 0
