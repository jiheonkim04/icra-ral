param(
    [string]$DiagnosticReportPath = "reports\bounded_libero_robosuite_diagnostic_rollout_report.json",
    [string]$LiberoRoot = "C:\assets\repos\LIBERO",
    [string]$RobosuiteRoot = "C:\assets\repos\robosuite",
    [string]$LiberoDataRoot = "C:\assets\data\libero",
    [string]$SmolVlaCheckpoint = "C:\assets\checkpoints\smolvla",
    [string]$HfHome = "C:\assets\hf_home",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [string]$WslRuntimeProbeReportPath = "",
    [int]$TaskCount = 1,
    [int]$MaxStepsPerTask = 5,
    [int]$ExpectedRuntimeMinutes = 30,
    [double]$ExpectedVramGb = 0,
    [switch]$SkipLiveWslProbe,
    [string]$JsonReportPath = "reports\libero_policy_rollout_readiness_plan_report.json",
    [string]$MarkdownReportPath = "reports\libero_policy_rollout_readiness_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO learned-policy rollout readiness planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not load SmolVLA, import heavy VLA models, run inference, train, use GPU, create simulator environments, rollout, download assets, access tokens, execute OpenVLA-OFT, or make paper claims."

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
    param(
        [string[]]$Command,
        [int]$TimeoutSec = 60,
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

function Test-AnyFile {
    param(
        [string]$Root,
        [string[]]$Patterns
    )
    foreach ($pattern in $Patterns) {
        $found = Get-ChildItem -LiteralPath $Root -Filter $pattern -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $found) { return $true }
    }
    return $false
}

function Get-FreeDiskGb {
    param([string]$Path)
    try {
        $root = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue).Path)
        if ([string]::IsNullOrWhiteSpace($root)) { $root = [System.IO.Path]::GetPathRoot($Path) }
        $driveName = $root.TrimEnd("\").TrimEnd(":")
        $drive = Get-PSDrive -Name $driveName -ErrorAction Stop
        return [math]::Round(($drive.Free / 1GB), 3)
    } catch {
        return $null
    }
}

function Get-WslRuntimeProbe {
    param([string]$ProbeReportPath)
    if (-not [string]::IsNullOrWhiteSpace($ProbeReportPath)) {
        $read = Read-JsonFileIfPresent -Path $ProbeReportPath
        if ($read.present -and -not $read.error -and $null -ne $read.data) {
            return [ordered]@{ source = "provided_report"; skipped = $false; ok = [bool]$read.data.ok; data = $read.data; error = $null }
        }
        return [ordered]@{ source = "provided_report"; skipped = $false; ok = $false; data = $null; error = "provided WSL runtime probe report is missing or unreadable" }
    }
    if ($SkipLiveWslProbe) {
        return [ordered]@{ source = "skipped"; skipped = $true; ok = $false; data = $null; error = "live WSL runtime probe skipped by parameter" }
    }

    $wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
    if ($null -eq $wslCommand) {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "wsl command not found" }
    }

    $runDir = Join-Path $RepoRoot "runs\libero_policy_rollout_readiness"
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $probeScriptWin = Join-Path $runDir "probe_wsl_policy_runtime.py"
    $probeScript = @'
import importlib.util
import json
import sys

required_modules = [
    "torch",
    "torchvision",
    "transformers",
    "lerobot",
    "safetensors",
    "huggingface_hub",
    "accelerate",
    "num2words",
    "draccus",
    "datasets",
    "imageio",
    "diffusers",
    "serial",
    "deepdiff",
    "av",
    "einops",
]

module_specs = {}
for name in required_modules:
    try:
        module_specs[name] = importlib.util.find_spec(name) is not None
    except Exception as exc:
        module_specs[name] = False

report = {
    "ok": all(module_specs.values()),
    "python": sys.version.split()[0],
    "required_modules": required_modules,
    "module_specs": module_specs,
    "heavy_imports_performed": False,
    "model_load_performed": False,
    "model_inference_performed": False,
    "gpu_jobs_performed": False,
    "training_performed": False,
    "rollouts_performed": False,
    "openvla_oft_executed": False,
}
print(json.dumps(report, indent=2, sort_keys=True))
'@
    $probeScript | Set-Content -LiteralPath $probeScriptWin -Encoding UTF8
    $probeScriptWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($probeScriptWin.Replace("\", "/"))) -TimeoutSec 30
    if (-not $probeScriptWsl.ok) {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "failed to map probe script into WSL: $($probeScriptWsl.stderr)" }
    }

    $pythonSelector = "if [ -x $WslPython ]; then printf '%s' $WslPython; else printf '%s' ''; fi"
    $selected = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $pythonSelector) -TimeoutSec 30
    if (-not $selected.ok -or [string]::IsNullOrWhiteSpace($selected.stdout)) {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "selected WSL python is unavailable: $WslPython" }
    }
    $cmd = "$($selected.stdout) '$($probeScriptWsl.stdout)'"
    $probe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $cmd) -TimeoutSec 60
    if (-not $probe.ok) {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "WSL runtime dependency probe failed: $($probe.stderr)" }
    }
    try {
        $start = ([string]$probe.stdout).IndexOf("{")
        if ($start -lt 0) { throw "probe stdout did not contain JSON" }
        $parsed = ([string]$probe.stdout).Substring($start) | ConvertFrom-Json
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = [bool]$parsed.ok; data = $parsed; error = $null }
    } catch {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "failed to parse WSL runtime dependency probe: $($_.Exception.Message)" }
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
        "# LIBERO Learned-Policy Rollout Readiness Plan Report",
        "",
        "- decision: $($Report.decision)",
        "- readiness plan green: $($Report.ready_for_tiny_learned_policy_rollout_plan)",
        "- execution green: $($Report.ready_for_tiny_learned_policy_rollout_execution)",
        "- WSL-only topology ready: $($Report.topologies.wsl_only_policy_and_sim.ready)",
        "- Windows-policy/WSL-sim bridge ready: false",
        "- paper claim ready: false",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is planning/readiness evidence only. It does not run a policy, simulator rollout, training, GPU job, heavy VLA import, OpenVLA-OFT execution, or paper claim."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 12
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

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if ($dangerousGatesSet.Count -gt 0) {
    $stopReasons.Add("planning-only policy rollout readiness refuses execution environment gates: $($dangerousGatesSet -join ', ')")
}
if ($TaskCount -lt 1 -or $TaskCount -gt 5) {
    $stopReasons.Add("tiny learned-policy rollout planning is capped at 5 tasks")
}
if ($MaxStepsPerTask -lt 1 -or $MaxStepsPerTask -gt 10) {
    $stopReasons.Add("tiny learned-policy rollout planning is capped at 10 steps per task")
}
if ($ExpectedRuntimeMinutes -gt 30) {
    $stopReasons.Add("tiny learned-policy rollout planning exceeds the 30 minute local budget")
}
if ($ExpectedVramGb -gt 14) {
    $stopReasons.Add("tiny learned-policy rollout planning exceeds the 14 GB VRAM budget")
}

$diagnosticRead = Read-JsonFileIfPresent -Path $DiagnosticReportPath
$diagnosticPassed = $false
if ($diagnosticRead.present -and -not $diagnosticRead.error -and $null -ne $diagnosticRead.data) {
    $diagnosticPassed = [bool]$diagnosticRead.data.bounded_libero_robosuite_diagnostic_rollout_passed
} else {
    $stopReasons.Add("bounded LIBERO/RoboSuite diagnostic rollout report is missing or unreadable")
}
if (-not $diagnosticPassed) {
    $stopReasons.Add("bounded LIBERO/RoboSuite diagnostic rollout has not passed")
}

foreach ($path in @($LiberoRoot, $RobosuiteRoot, $LiberoDataRoot, $SmolVlaCheckpoint, $HfHome)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $stopReasons.Add("required local path does not exist: $path")
    }
}

$smolvlaConfigPresent = Test-Path -LiteralPath (Join-Path $SmolVlaCheckpoint "config.json")
$smolvlaWeightsPresent = $false
if (Test-Path -LiteralPath $SmolVlaCheckpoint) {
    $smolvlaWeightsPresent = (Test-AnyFile -Root $SmolVlaCheckpoint -Patterns @("model.safetensors", "pytorch_model.bin", "*.safetensors", "*.bin"))
}
$processorPresent = (Test-Path -LiteralPath (Join-Path $SmolVlaCheckpoint "policy_preprocessor.json")) -or (Test-Path -LiteralPath (Join-Path $SmolVlaCheckpoint "preprocessor_config.json"))
$tokenizerDependencyRoot = Join-Path $HfHome "HuggingFaceTB\SmolVLM2-500M-Video-Instruct"
$tokenizerDependencyPresent = Test-Path -LiteralPath $tokenizerDependencyRoot
if (-not $smolvlaConfigPresent) { $stopReasons.Add("SmolVLA config.json is missing") }
if (-not $smolvlaWeightsPresent) { $stopReasons.Add("SmolVLA weights file is missing") }
if (-not $processorPresent) { $warnings.Add("SmolVLA processor metadata file was not found in checkpoint root") }
if (-not $tokenizerDependencyPresent) { $stopReasons.Add("SmolVLA tokenizer/processor dependency is missing under HF_HOME") }

$runtimeProbe = Get-WslRuntimeProbe -ProbeReportPath $WslRuntimeProbeReportPath
$wslPolicyRuntimeReady = [bool]$runtimeProbe.ok
if (-not $wslPolicyRuntimeReady) {
    $runtimeDetail = $runtimeProbe.error
    $missingModules = New-Object System.Collections.Generic.List[string]
    if ($null -ne $runtimeProbe.data -and $null -ne $runtimeProbe.data.module_specs) {
        $runtimeProbe.data.module_specs.PSObject.Properties | ForEach-Object {
            if (-not [bool]$_.Value) { $missingModules.Add($_.Name) | Out-Null }
        }
    }
    if ($missingModules.Count -gt 0) {
        $runtimeDetail = "missing WSL modules: $($missingModules -join ', ')"
    }
    if ([string]::IsNullOrWhiteSpace($runtimeDetail)) {
        $runtimeDetail = "runtime probe did not report all required modules as available"
    }
    $warnings.Add("WSL SmolVLA policy runtime is not verified: $runtimeDetail")
}

$localPrereqsInsideBudget = [bool]($stopReasons.Count -eq 0)
$wslOnlyReady = [bool]($localPrereqsInsideBudget -and $wslPolicyRuntimeReady)
$decision = if ($wslOnlyReady) {
    "proceed"
} elseif ($localPrereqsInsideBudget) {
    "reduce_scope"
} else {
    "stop"
}
$reason = if ($wslOnlyReady) {
    "WSL-only simulator and SmolVLA policy runtime prerequisites are green for a separately gated tiny learned-policy rollout."
} elseif ($localPrereqsInsideBudget) {
    "Local simulator/checkpoint prerequisites are green, but WSL SmolVLA runtime is not verified; reduce scope to a WSL policy-runtime setup/readiness task before learned-policy rollout."
} else {
    $stopReasons -join "; "
}

$freeDiskGb = Get-FreeDiskGb -Path $RepoRoot
$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
        downloads_performed = $false
        installs_performed = $false
        simulator_environment_created = $false
        rollouts_performed = $false
        benchmark_rollouts_performed = $false
        learned_policy_inference_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        heavy_model_imports_performed = $false
        model_load_performed = $false
        model_inference_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        benchmark_claims_made = $false
        sota_claims_made = $false
        paper_grade_claims_made = $false
    }
    risk_assessment = [ordered]@{
        task = "tiny LIBERO learned-policy rollout readiness planning"
        command = "scripts\67_bounded_libero_policy_rollout.ps1 (future runner; not executed by this planner)"
        source = "local official LIBERO/RoboSuite source and data plus local SmolVLA checkpoint"
        model_source = "local lerobot/smolvla_base checkpoint only"
        dataset_source = "local official yifengzhu-hf/LIBERO-datasets files"
        expected_size_gb = 0
        current_free_disk_gb = $freeDiskGb
        target_output_paths = @("reports\libero_policy_rollout_readiness_plan_report.json", "reports\libero_policy_rollout_readiness_plan_report.md", "runs\libero_policy_rollout_readiness")
        expected_runtime_minutes = $ExpectedRuntimeMinutes
        expected_ram_gb = 10
        expected_vram_gb = $ExpectedVramGb
        task_count = $TaskCount
        max_steps_per_task = $MaxStepsPerTask
        token_login_license_payment_needed = $false
        simulator_would_run_in_future_runner = $true
        rollout_would_run_in_future_runner = $true
        training_would_run = $false
        openvla_oft_would_run = $false
        stop_condition = "stop if WSL policy runtime is missing, runtime exceeds 30 minutes, VRAM estimate exceeds 14 GB, token/license/payment is required, or OpenVLA-OFT becomes required"
        fallback_plan = "reduce scope to WSL SmolVLA runtime setup/readiness or offline proxy evaluation"
        decision = $decision
        reason = $reason
    }
    prerequisites = [ordered]@{
        diagnostic_rollout = [ordered]@{ report_present = [bool]$diagnosticRead.present; report_path = $diagnosticRead.path; report_error = $diagnosticRead.error; passed = $diagnosticPassed }
        local_paths = [ordered]@{
            libero_root = [ordered]@{ path = $LiberoRoot; exists = [bool](Test-Path -LiteralPath $LiberoRoot) }
            robosuite_root = [ordered]@{ path = $RobosuiteRoot; exists = [bool](Test-Path -LiteralPath $RobosuiteRoot) }
            libero_data_root = [ordered]@{ path = $LiberoDataRoot; exists = [bool](Test-Path -LiteralPath $LiberoDataRoot) }
            smolvla_checkpoint = [ordered]@{ path = $SmolVlaCheckpoint; exists = [bool](Test-Path -LiteralPath $SmolVlaCheckpoint); config_present = $smolvlaConfigPresent; weights_present = $smolvlaWeightsPresent; processor_metadata_present = $processorPresent }
            hf_home = [ordered]@{ path = $HfHome; exists = [bool](Test-Path -LiteralPath $HfHome); tokenizer_dependency_root = $tokenizerDependencyRoot; tokenizer_dependency_present = $tokenizerDependencyPresent }
        }
        wsl_runtime_probe = $runtimeProbe
    }
    topologies = [ordered]@{
        wsl_only_policy_and_sim = [ordered]@{
            ready = $wslOnlyReady
            reason = if ($wslOnlyReady) { "same WSL process can host simulator and local SmolVLA policy runtime" } else { "WSL policy runtime is missing or local prerequisites are incomplete" }
        }
        windows_policy_wsl_sim_bridge = [ordered]@{
            ready = $false
            reason = "not implemented; cross-process Windows-policy/WSL-simulator bridge would add latency and IPC risk before WSL-only readiness is exhausted"
        }
    }
    dangerous_execution_gates_set = @($dangerousGatesSet)
    ready_for_tiny_learned_policy_rollout_plan = $localPrereqsInsideBudget
    ready_for_tiny_learned_policy_rollout_execution = $wslOnlyReady
    ready_for_benchmark_rollout = $wslOnlyReady
    ready_for_paper_claim = $false
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    decision = $decision
    recommended_next_step = if ($wslOnlyReady) {
        "Create a separately gated tiny learned-policy rollout runner with task-local ALLOW_POLICY_ROLLOUT=1, max 1 task, max 5-10 steps, no training, no GPU by default, and no paper claim."
    } elseif ($localPrereqsInsideBudget) {
        "Plan WSL SmolVLA runtime dependency setup/readiness in the existing /home/jiheon/.venvs/tca_map_sim venv; do not run policy rollout yet."
    } else {
        "Resolve listed local prerequisite blockers before any learned-policy rollout planning."
    }
}

Write-Reports -Report $report
exit 0
