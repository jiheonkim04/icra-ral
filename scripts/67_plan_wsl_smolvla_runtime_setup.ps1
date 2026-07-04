param(
    [string]$VenvPath = '$HOME/.venvs/tca_map_sim',
    [string]$PolicyReadinessReportPath = "reports\libero_policy_rollout_readiness_plan_report.json",
    [string]$JsonReportPath = "reports\wsl_smolvla_runtime_setup_plan_report.json",
    [string]$MarkdownReportPath = "reports\wsl_smolvla_runtime_setup_plan_report.md",
    [int]$ExpectedRuntimeMinutes = 30,
    [double]$ExpectedSizeGb = 8,
    [double]$ExpectedVramGb = 0,
    [switch]$SkipLiveWslProbe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "WSL SmolVLA runtime setup planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not install packages, download assets, import heavy VLA models, load models, run inference, train, rollout, use GPU, execute OpenVLA-OFT, access tokens, or make paper claims."

$RequiredModules = @("torch", "torchvision", "transformers", "lerobot", "safetensors", "huggingface_hub", "accelerate", "num2words", "draccus", "datasets", "imageio", "diffusers", "serial", "deepdiff", "av", "einops")
$TorchPackages = @("torch==2.10.0", "torchvision==0.25.0")
$RuntimePackages = @("transformers==4.57.6", "safetensors==0.8.0", "huggingface_hub==0.35.3", "accelerate==1.14.0", "num2words==0.5.14", "draccus==0.10.0", "datasets==4.8.5", "imageio[ffmpeg]==2.37.3", "diffusers==0.35.2", "pyserial==3.5", "deepdiff==8.6.2", "av==15.1.0", "einops==0.8.2")
$NoDepsPackages = @("lerobot==0.4.4")

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

function Get-FreeDiskGb {
    try {
        $drive = Get-PSDrive -Name "C" -ErrorAction Stop
        return [math]::Round(($drive.Free / 1GB), 3)
    } catch {
        return $null
    }
}

function Get-WslModuleProbe {
    if ($SkipLiveWslProbe) {
        return [ordered]@{ source = "skipped"; skipped = $true; ok = $false; data = $null; error = "live WSL probe skipped by parameter" }
    }
    if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "wsl command not found" }
    }
    $probePython = @"
import importlib.util
import json
import sys

required = $($RequiredModules | ConvertTo-Json -Compress)
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
"@
    $runDir = Join-Path $RepoRoot "runs\wsl_smolvla_runtime_setup"
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $probeWin = Join-Path $runDir "probe_wsl_smolvla_runtime.py"
    $probePython | Set-Content -LiteralPath $probeWin -Encoding UTF8
    $probeWsl = Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", ($probeWin.Replace("\", "/"))) -TimeoutSec 30
    if (-not $probeWsl.ok) {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "failed to map probe into WSL" }
    }
    $selector = "if [ -x $VenvPath/bin/python ]; then printf '%s' $VenvPath/bin/python; else printf '%s' ''; fi"
    $selected = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $selector) -TimeoutSec 30
    if (-not $selected.ok -or [string]::IsNullOrWhiteSpace($selected.stdout)) {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "selected WSL venv python is unavailable: $VenvPath/bin/python" }
    }
    $probe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$($selected.stdout) '$($probeWsl.stdout)'") -TimeoutSec 60
    if (-not $probe.ok) {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "WSL module probe failed: $($probe.stderr)" }
    }
    try {
        $start = ([string]$probe.stdout).IndexOf("{")
        if ($start -lt 0) { throw "probe stdout did not contain JSON" }
        $parsed = ([string]$probe.stdout).Substring($start) | ConvertFrom-Json
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = [bool]$parsed.ok; data = $parsed; error = $null }
    } catch {
        return [ordered]@{ source = "live_wsl"; skipped = $false; ok = $false; data = $null; error = "failed to parse WSL module probe: $($_.Exception.Message)" }
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
        "# WSL SmolVLA Runtime Setup Plan Report",
        "",
        "- decision: $($Report.decision)",
        "- setup required: $($Report.setup_required)",
        "- ready after current probe: $($Report.ready_for_wsl_smolvla_runtime)",
        "- expected size GB: $($Report.risk_assessment.expected_size_gb)",
        "- expected runtime minutes: $($Report.risk_assessment.expected_runtime_minutes)",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is planning-only. It performs no install, download, heavy import, model load, inference, training, rollout, GPU job, OpenVLA-OFT execution, token access, or paper claim."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 12
}

$policyRead = Read-JsonFileIfPresent -Path $PolicyReadinessReportPath
$policyPlanReady = $false
if ($policyRead.present -and -not $policyRead.error -and $null -ne $policyRead.data) {
    $policyPlanReady = [bool]$policyRead.data.ready_for_tiny_learned_policy_rollout_plan
}

$freeDiskGb = Get-FreeDiskGb
$diskAfter = if ($null -eq $freeDiskGb) { $null } else { [math]::Round(($freeDiskGb - $ExpectedSizeGb), 3) }
$moduleProbe = Get-WslModuleProbe
$missingModules = New-Object System.Collections.Generic.List[string]
if ($null -ne $moduleProbe.data -and $null -ne $moduleProbe.data.module_specs) {
    $moduleProbe.data.module_specs.PSObject.Properties | ForEach-Object {
        if (-not [bool]$_.Value) { $missingModules.Add($_.Name) | Out-Null }
    }
} elseif (-not $moduleProbe.ok) {
    foreach ($module in $RequiredModules) { $missingModules.Add($module) | Out-Null }
}

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
if (-not $policyPlanReady) { $stopReasons.Add("learned-policy rollout local prerequisites are not ready; run scripts\66_plan_libero_policy_rollout_readiness.ps1 first") }
if ($ExpectedRuntimeMinutes -gt 30) { $stopReasons.Add("expected runtime exceeds 30 minutes") }
if ($ExpectedVramGb -gt 14) { $stopReasons.Add("expected VRAM exceeds 14 GB") }
if ($ExpectedSizeGb -gt 80) { $stopReasons.Add("expected package download/install size exceeds 80 GB") }
if ($null -eq $freeDiskGb) { $stopReasons.Add("disk free space could not be evaluated") }
elseif ($diskAfter -lt 100) { $stopReasons.Add("disk free after estimate would be below 100 GB") }
if ($SkipLiveWslProbe) { $warnings.Add("live WSL module probe skipped; setup plan is not execution-ready without a live probe") }

$setupRequired = [bool]($missingModules.Count -gt 0)
$decision = if ($stopReasons.Count -gt 0) { "stop" } else { "proceed" }
$reason = if ($stopReasons.Count -gt 0) {
    $stopReasons -join "; "
} elseif ($setupRequired) {
    "WSL SmolVLA runtime modules are missing but package setup is inside the local risk budget."
} else {
    "WSL SmolVLA runtime modules are already available."
}

$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
        installs_performed = $false
        package_downloads_performed = $false
        heavy_model_imports_performed = $false
        model_load_performed = $false
        model_inference_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        rollouts_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    risk_assessment = [ordered]@{
        task = "bounded WSL SmolVLA runtime package setup"
        command = "scripts\68_setup_wsl_smolvla_runtime_deps.ps1"
        source = "PyTorch official CPU wheel index plus PyPI packages matching the Windows SmolVLA runtime"
        target_path = $VenvPath
        expected_size_gb = $ExpectedSizeGb
        disk_free_before_gb = $freeDiskGb
        disk_free_after_estimate_gb = $diskAfter
        expected_runtime_minutes = $ExpectedRuntimeMinutes
        expected_ram_gb = 4
        expected_vram_gb = $ExpectedVramGb
        official_documented_sources = $true
        token_login_license_payment_needed = $false
        cuda_driver_system_changes = $false
        simulator_will_run = $false
        rollout_will_run = $false
        model_load_will_run = $false
        decision = $decision
        reason = $reason
    }
    package_plan = [ordered]@{
        venv_path = $VenvPath
        required_modules = @($RequiredModules)
        torch_cpu_index_url = "https://download.pytorch.org/whl/cpu"
        torch_packages = @($TorchPackages)
        runtime_packages = @($RuntimePackages)
        no_deps_packages = @($NoDepsPackages)
        preserve_simulator_numpy = $true
        notes = "LeRobot is installed with --no-deps in the setup script to avoid broad simulator venv dependency drift before load-only WSL smoke."
    }
    prerequisite_policy_readiness = [ordered]@{
        report_present = [bool]$policyRead.present
        report_path = $policyRead.path
        report_error = $policyRead.error
        ready_for_tiny_learned_policy_rollout_plan = $policyPlanReady
    }
    wsl_module_probe = $moduleProbe
    missing_modules = @($missingModules)
    setup_required = $setupRequired
    ready_for_wsl_smolvla_runtime = [bool]($decision -eq "proceed" -and -not $setupRequired)
    dangerous_execution_gates_set = @()
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    decision = $decision
    recommended_next_step = if ($decision -eq "proceed" -and $setupRequired) {
        "Run scripts\68_setup_wsl_smolvla_runtime_deps.ps1 with task-local ALLOW_WSL_SMOLVLA_RUNTIME_SETUP=1."
    } elseif ($decision -eq "proceed") {
        "Rerun scripts\66_plan_libero_policy_rollout_readiness.ps1; if it is green, create the tiny learned-policy rollout runner."
    } else {
        "Resolve listed blockers before WSL SmolVLA runtime setup."
    }
}

Write-Reports -Report $report
exit 0
