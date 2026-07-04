param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$WslPython = '$HOME/.venvs/tca_map_sim/bin/python',
    [switch]$Execute,
    [int]$TimeoutSeconds = 300,
    [string]$JsonReportPath = "reports\wsl_simulator_source_link_report.json",
    [string]$MarkdownReportPath = "reports\wsl_simulator_source_link_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "WSL simulator source link"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script links existing local simulator source checkouts into the existing WSL venv. It does not create a repo-local venv, download packages, install dependencies, render, reset/step, rollout, train, run GPU jobs, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

function Read-AssetConfig {
    param([string]$Path)
    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) { return $values }
    $inAssets = $false
    foreach ($rawLine in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $line = $rawLine.TrimEnd()
        if ($line -match '^\s*#' -or $line.Trim().Length -eq 0) { continue }
        if ($line -match '^assets\s*:') {
            $inAssets = $true
            continue
        }
        if ($inAssets -and $line -match '^\S' -and $line -notmatch '^assets\s*:') { break }
        if ($inAssets -and $line -match '^\s+([A-Za-z0-9_]+)\s*:\s*(.*)$') {
            $key = $Matches[1]
            $value = $Matches[2].Trim().Trim('"').Trim("'")
            if ($value -and $value.ToLowerInvariant() -ne "null") { $values[$key] = $value }
        }
    }
    return $values
}

function Get-ConfiguredValue {
    param([hashtable]$Config, [string]$Key, [string]$EnvName)
    $envValue = [Environment]::GetEnvironmentVariable($EnvName)
    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        return @{ Value = $envValue; Source = "env:$EnvName" }
    }
    if ($Config.ContainsKey($Key)) {
        return @{ Value = $Config[$Key]; Source = $PathsFile }
    }
    return @{ Value = $null; Source = $null }
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

function Convert-PathForWslArg {
    param([string]$Path)
    return $Path.Replace("\", "/")
}

function Test-PathValue {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    return Test-Path -LiteralPath $Path
}

function Write-Reports {
    param([object]$Report)
    $jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
    $markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# WSL Simulator Source Link Report",
        "",
        "- decision: $($Report.decision)",
        "- executed: $($Report.execution.executed)",
        "- setup passed: $($Report.execution.setup_passed)",
        "- WSL python: $($Report.target.wsl_python_executable)",
        "- RoboSuite direct import after: $($Report.probes_after.robosuite_direct.ok)",
        "- LIBERO direct import after: $($Report.probes_after.libero_direct.ok)",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report covers editable source linking only. It is not render evidence, reset/step evidence, rollout evidence, or paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 10
}

$config = Read-AssetConfig -Path $PathsFile
$robosuiteConfigured = Get-ConfiguredValue -Config $config -Key "robosuite_root" -EnvName "ROBOSUITE_ROOT"
$liberoConfigured = Get-ConfiguredValue -Config $config -Key "libero_root" -EnvName "LIBERO_ROOT"
$liberoDataConfigured = Get-ConfiguredValue -Config $config -Key "libero_data_root" -EnvName "LIBERO_DATA_ROOT"
$robosuiteRoot = $robosuiteConfigured.Value
$liberoRoot = $liberoConfigured.Value
$liberoDataRoot = $liberoDataConfigured.Value

$wslInstalled = $null -ne (Get-Command wsl -ErrorAction SilentlyContinue)
$selectedPython = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "if [ -x $WslPython ]; then printf '%s' $WslPython; else printf '%s' python3; fi") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$wslPythonExecutable = if ($selectedPython.ok -and -not [string]::IsNullOrWhiteSpace($selectedPython.stdout)) { $selectedPython.stdout } else { "python3" }
$pythonVersion = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable --version") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$pipVersion = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable -m pip --version") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$sitePackages = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable -c 'import site; print(site.getsitepackages()[0])'") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$robosuiteWsl = if ($wslInstalled -and (Test-PathValue -Path $robosuiteRoot)) { Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", (Convert-PathForWslArg -Path $robosuiteRoot)) -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "ROBOSUITE_ROOT missing or path does not exist" } }
$liberoWsl = if ($wslInstalled -and (Test-PathValue -Path $liberoRoot)) { Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", (Convert-PathForWslArg -Path $liberoRoot)) -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "LIBERO_ROOT missing or path does not exist" } }
$liberoDataWsl = if ($wslInstalled -and (Test-PathValue -Path $liberoDataRoot)) { Invoke-SafeCommand -Command @("wsl", "wslpath", "-a", (Convert-PathForWslArg -Path $liberoDataRoot)) -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "LIBERO_DATA_ROOT missing or path does not exist" } }

$robosuiteBefore = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable -c 'import robosuite; print(getattr(robosuite, `"__file__`", `"ok`"))'") -TimeoutSec 60 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$liberoBefore = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable -c 'import libero; print(getattr(libero, `"__file__`", `"ok`"))'") -TimeoutSec 60 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }

$stopReasons = New-Object System.Collections.Generic.List[string]
if (-not $wslInstalled) { $stopReasons.Add("wsl command is not available") }
if (-not $pythonVersion.ok) { $stopReasons.Add("selected WSL Python is not executable") }
if (-not $pipVersion.ok) { $stopReasons.Add("selected WSL Python does not have pip") }
if (-not $sitePackages.ok) { $stopReasons.Add("selected WSL Python site-packages path could not be resolved") }
if (-not (Test-PathValue -Path $robosuiteRoot)) { $stopReasons.Add("ROBOSUITE_ROOT is missing or does not exist") }
if (-not (Test-PathValue -Path $robosuiteRoot) -or -not (Test-Path -LiteralPath (Join-Path $robosuiteRoot "setup.py"))) { $stopReasons.Add("ROBOSUITE_ROOT does not contain setup.py") }
if (-not (Test-PathValue -Path $liberoRoot)) { $stopReasons.Add("LIBERO_ROOT is missing or does not exist") }
if (-not (Test-PathValue -Path $liberoRoot) -or -not (Test-Path -LiteralPath (Join-Path $liberoRoot "setup.py"))) { $stopReasons.Add("LIBERO_ROOT does not contain setup.py") }
if (-not (Test-PathValue -Path $liberoDataRoot)) { $stopReasons.Add("LIBERO_DATA_ROOT is missing or does not exist") }
if (-not $robosuiteWsl.ok) { $stopReasons.Add("ROBOSUITE_ROOT could not be mapped into WSL") }
if (-not $liberoWsl.ok) { $stopReasons.Add("LIBERO_ROOT could not be mapped into WSL") }
if (-not $liberoDataWsl.ok) { $stopReasons.Add("LIBERO_DATA_ROOT could not be mapped into WSL") }

$gateOk = [Environment]::GetEnvironmentVariable("ALLOW_WSL_SIM_SOURCE_LINK") -eq "1"
$decision = if ($stopReasons.Count -eq 0) { "proceed" } else { "stop" }

$report = [ordered]@{
    policy = [ordered]@{
        source_link_only = $true
        task_local_gate_required = "ALLOW_WSL_SIM_SOURCE_LINK=1 when -Execute is used"
        uses_existing_wsl_venv = $true
        creates_repo_local_venv = $false
        pip_no_index = $true
        pip_no_deps = $true
        installs_performed = $false
        downloads_performed = $false
        render_smoke_performed = $false
        reset_step_smoke_performed = $false
        rollouts_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    risk_assessment = [ordered]@{
        task = "link local simulator source checkouts into existing WSL venv"
        source = "local official LIBERO and RoboSuite source checkouts"
        target = $WslPython
        expected_size_gb = 0
        expected_runtime_minutes = 2
        expected_vram_gb = 0
        token_license_payment_needed = $false
        cuda_driver_system_graphics_changes = $false
        command = "$wslPythonExecutable -m pip install --no-index --no-deps --no-build-isolation -e $($robosuiteWsl.stdout) -e $($liberoWsl.stdout); write $($liberoWsl.stdout)/libero to site-packages/tca_map_libero_source.pth; write noninteractive ~/.libero/config.yaml"
        decision = $decision
        reason = if ($stopReasons.Count -eq 0) { "local source checkouts and existing WSL venv are present; editable no-deps linking is bounded and offline" } else { $stopReasons -join "; " }
    }
    paths = [ordered]@{
        robosuite_root_configured = -not [string]::IsNullOrWhiteSpace($robosuiteRoot)
        robosuite_root_source = $robosuiteConfigured.Source
        robosuite_root_exists = [bool](Test-PathValue -Path $robosuiteRoot)
        robosuite_root_wsl = $robosuiteWsl
        libero_root_configured = -not [string]::IsNullOrWhiteSpace($liberoRoot)
        libero_root_source = $liberoConfigured.Source
        libero_root_exists = [bool](Test-PathValue -Path $liberoRoot)
        libero_root_wsl = $liberoWsl
        libero_data_root_configured = -not [string]::IsNullOrWhiteSpace($liberoDataRoot)
        libero_data_root_source = $liberoDataConfigured.Source
        libero_data_root_exists = [bool](Test-PathValue -Path $liberoDataRoot)
        libero_data_root_wsl = $liberoDataWsl
    }
    target = [ordered]@{
        wsl_python = $selectedPython
        wsl_python_executable = $wslPythonExecutable
        python_version = $pythonVersion
        pip_version = $pipVersion
        site_packages = $sitePackages
    }
    probes_before = [ordered]@{
        robosuite_direct = $robosuiteBefore
        libero_direct = $liberoBefore
    }
    execution = [ordered]@{
        requested = [bool]$Execute
        gate_present = [bool]$gateOk
        executed = $false
        setup_passed = $false
        link_result = $null
        libero_pth_result = $null
        libero_config_result = $null
    }
    probes_after = [ordered]@{
        robosuite_direct = $robosuiteBefore
        libero_direct = $liberoBefore
    }
    stop_reasons = @($stopReasons)
    warnings = @()
    decision = $decision
    recommended_next_step = $null
}

if ($Execute -and -not $gateOk) {
    $report.decision = "stop"
    $report.stop_reasons += "ALLOW_WSL_SIM_SOURCE_LINK=1 is required when -Execute is used"
    $report.risk_assessment.decision = "stop"
    $report.risk_assessment.reason = ($report.stop_reasons -join "; ")
}

if ($Execute -and $report.decision -eq "proceed") {
    $linkCommand = "$wslPythonExecutable -m pip install --no-index --no-deps --no-build-isolation -e '$($robosuiteWsl.stdout)' -e '$($liberoWsl.stdout)'"
    $linkResult = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $linkCommand) -TimeoutSec $TimeoutSeconds
    $pthCommand = "printf '%s\n' '$($liberoWsl.stdout)/libero' > '$($sitePackages.stdout)/tca_map_libero_source.pth'"
    $liberoPthResult = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $pthCommand) -TimeoutSec 30
    $liberoBenchmarkRoot = "$($liberoWsl.stdout)/libero/libero"
    $configCommand = @(
        "mkdir -p '`$HOME/.libero'",
        "cat > '`$HOME/.libero/config.yaml' <<'YAML'",
        "benchmark_root: $liberoBenchmarkRoot",
        "bddl_files: $liberoBenchmarkRoot/bddl_files",
        "init_states: $liberoBenchmarkRoot/init_files",
        "datasets: $($liberoDataWsl.stdout)",
        "assets: $liberoBenchmarkRoot/assets",
        "YAML"
    ) -join "`n"
    $liberoConfigResult = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $configCommand) -TimeoutSec 30
    $report.execution.executed = $true
    $report.execution.link_result = $linkResult
    $report.execution.libero_pth_result = $liberoPthResult
    $report.execution.libero_config_result = $liberoConfigResult
    $report.policy.installs_performed = $true

    $robosuiteAfter = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable -c 'import robosuite; print(getattr(robosuite, `"__file__`", `"ok`"))'") -TimeoutSec 60
    $liberoAfter = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$wslPythonExecutable -c 'import libero; print(getattr(libero, `"__file__`", `"ok`"))'") -TimeoutSec 60
    $report.probes_after.robosuite_direct = $robosuiteAfter
    $report.probes_after.libero_direct = $liberoAfter
    $report.execution.setup_passed = [bool]($linkResult.ok -and $liberoPthResult.ok -and $liberoConfigResult.ok -and $robosuiteAfter.ok -and $liberoAfter.ok)
}

$report.recommended_next_step = if ($report.execution.setup_passed -or (-not $Execute -and $report.decision -eq "proceed")) {
    "Rerun scripts\56_check_wsl_simulator_deps.ps1 and then scripts\55_bounded_simulator_import_smoke.ps1 with ALLOW_SIMULATOR_IMPORT_SMOKE=1."
} elseif ($Execute -and -not $gateOk) {
    "Set ALLOW_WSL_SIM_SOURCE_LINK=1 only for this bounded local source-link task."
} else {
    "Resolve listed source-link blockers. Do not create a repo-local venv, download dependencies, render, reset/step, rollout, train, use GPU, execute OpenVLA-OFT, or make paper claims."
}

Write-Reports -Report $report
exit 0
