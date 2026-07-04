param(
    [string]$VenvPath = '$HOME/.venvs/tca_map_sim',
    [string[]]$Packages = @("numpy"),
    [string]$PipBootstrapUrl = "https://bootstrap.pypa.io/get-pip.py",
    [switch]$Execute,
    [switch]$ClearVenv,
    [int]$TimeoutSeconds = 600,
    [string]$JsonReportPath = "reports\wsl_simulator_dependency_setup_report.json",
    [string]$MarkdownReportPath = "reports\wsl_simulator_dependency_setup_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "WSL simulator dependency setup"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is bounded to WSL Python packaging setup. It does not run GPU jobs, train, render, rollout, import heavy VLA models, access tokens, execute OpenVLA-OFT, or make paper claims."

$PackageList = @(
    $Packages |
        ForEach-Object { [string]$_ -split "," } |
        ForEach-Object { $_.Trim() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)

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

function Write-Reports {
    param([object]$Report)
    $jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
    $markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null
    $Report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
    $lines = @(
        "# WSL Simulator Dependency Setup Report",
        "",
        "- decision: $($Report.risk_assessment.decision)",
        "- executed: $($Report.execution.executed)",
        "- venv path: $($Report.target.venv_path)",
        "- packages: $(@($Report.target.packages) -join ', ')",
        "- pip bootstrap source: $($Report.target.pip_bootstrap_url)",
        "- setup passed: $($Report.execution.setup_passed)",
        "- recommended next step: $($Report.recommended_next_step)",
        "",
        "This report is for bounded WSL Python packaging setup only. It is not simulator rollout evidence and not paper-grade evidence."
    )
    $lines -join "`n" | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8
    $Report | ConvertTo-Json -Depth 10
}

function Get-ImportModuleName {
    param([string]$Package)
    $base = ($Package -replace "\[.*\]", "" -replace "[<>=!~].*$", "").Trim()
    switch ($base.ToLowerInvariant()) {
        "pillow" { return "PIL" }
        "opencv-python" { return "cv2" }
        "pyyaml" { return "yaml" }
        default { return ($base -replace "-", "_") }
    }
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
$wslInstalled = $null -ne $wslCommand
$pythonVersion = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "python3", "--version") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$venvHelp = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "python3", "-m", "venv", "--help") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$pipVersion = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "python3", "-m", "pip", "--version") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$aptProbe = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "which", "apt") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$sudoNoPassword = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "sudo", "-n", "true") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$diskProbe = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "df -BG `$HOME | tail -1") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$venvWithoutPipProbe = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "rm -rf /tmp/tca_map_venv_probe; python3 -m venv --without-pip /tmp/tca_map_venv_probe; rc=`$?; rm -rf /tmp/tca_map_venv_probe; exit `$rc") -TimeoutSec 60 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$curlProbe = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "which", "curl") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$wgetProbe = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "which", "wget") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }
$venvProbe = if ($wslInstalled) { Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "test -x $VenvPath/bin/python && $VenvPath/bin/python -m pip --version") -TimeoutSec 30 } else { [ordered]@{ ok = $false; timed_out = $false; returncode = $null; stdout = ""; stderr = "wsl command not found" } }

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
if (-not $wslInstalled) { $stopReasons.Add("wsl command is not available") }
if (-not $pythonVersion.ok) { $stopReasons.Add("WSL python3 is not available") }
if (-not $venvHelp.ok -or -not $venvWithoutPipProbe.ok) { $stopReasons.Add("WSL python3 venv support is not available; apt setup would be required and must stop if sudo password is required") }
if (-not $curlProbe.ok -and -not $wgetProbe.ok) { $stopReasons.Add("neither curl nor wget is available for venv-local pip bootstrap") }
if ($PackageList.Count -eq 0) { $stopReasons.Add("no packages requested") }
if (-not $sudoNoPassword.ok) { $warnings.Add("sudo -n is not available without password; this script will not use sudo or apt") }
if ($aptProbe.ok) { $warnings.Add("apt exists, but this script uses the venv path only and does not install apt packages") }

$riskDecision = if ($stopReasons.Count -eq 0) { "proceed" } else { "stop" }
$riskReason = if ($stopReasons.Count -eq 0) { "WSL python3 and venv are available; setup can proceed without sudo, tokens, CUDA/driver changes, render, rollout, or OpenVLA-OFT." } else { $stopReasons -join "; " }
$gateOk = [Environment]::GetEnvironmentVariable("ALLOW_WSL_SIM_DEPS") -eq "1"

$report = [ordered]@{
    policy = [ordered]@{
        bounded_wsl_python_packaging_setup = $true
        task_local_gate_required = "ALLOW_WSL_SIM_DEPS=1 when -Execute is used"
        apt_installs_performed = $false
        sudo_used = $false
        gpu_jobs_performed = $false
        training_performed = $false
        render_smoke_performed = $false
        rollouts_performed = $false
        simulator_environment_steps_performed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    risk_assessment = [ordered]@{
        task = "bounded WSL simulator Python dependency setup"
        environment = "WSL"
        command = "create or reuse $VenvPath with python3 -m venv --without-pip; bootstrap pip only if missing; $VenvPath/bin/python -m pip install $(@($PackageList) -join ' ')"
        expected_install_download_size_gb = 0.2
        target_environment_path = $VenvPath
        disk_free_probe = $diskProbe
        expected_runtime_minutes = 10
        expected_ram_gb = 1
        expected_vram_gb = 0
        sudo_needed = $false
        sudo_without_password_available = [bool]$sudoNoPassword.ok
        token_license_payment_needed = $false
        cuda_driver_system_graphics_changes = $false
        decision = $riskDecision
        reason = $riskReason
    }
    wsl = [ordered]@{
        command_installed = [bool]$wslInstalled
        python3_version = $pythonVersion
        venv_available = $venvHelp
        global_pip_version = $pipVersion
        apt_probe = $aptProbe
        sudo_no_password_probe = $sudoNoPassword
        venv_without_pip_probe = $venvWithoutPipProbe
        curl_probe = $curlProbe
        wget_probe = $wgetProbe
    }
    target = [ordered]@{
        venv_path = $VenvPath
        packages = @($PackageList)
        pip_bootstrap_url = $PipBootstrapUrl
        venv_probe_before = $venvProbe
        clear_venv_requested = [bool]$ClearVenv
    }
    execution = [ordered]@{
        requested = [bool]$Execute
        gate_present = [bool]$gateOk
        executed = $false
        setup_passed = $false
        commands = @()
        package_probe_after = @()
    }
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    recommended_next_step = $null
}

if ($Execute -and -not $gateOk) {
    $report.stop_reasons += "ALLOW_WSL_SIM_DEPS=1 is required when -Execute is used"
    $report.risk_assessment.decision = "stop"
    $report.risk_assessment.reason = ($report.stop_reasons -join "; ")
}

if ($Execute -and $report.risk_assessment.decision -eq "proceed") {
    $createMode = if ($ClearVenv) { "--clear --without-pip" } else { "--without-pip" }
    $createCommand = if ($ClearVenv) {
        "python3 -m venv $createMode $VenvPath"
    } else {
        "if [ ! -x $VenvPath/bin/python ]; then python3 -m venv $createMode $VenvPath; fi"
    }
    $createVenv = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $createCommand) -TimeoutSec $TimeoutSeconds
    $pipAfterCreate = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "test -x $VenvPath/bin/python && $VenvPath/bin/python -m pip --version") -TimeoutSec 30
    $downloadPip = [ordered]@{ ok = $true; timed_out = $false; returncode = 0; stdout = "skipped; pip already available"; stderr = "" }
    $bootstrapPip = [ordered]@{ ok = $true; timed_out = $false; returncode = 0; stdout = "skipped; pip already available"; stderr = "" }
    if (-not $pipAfterCreate.ok) {
        $downloadPip = if ($curlProbe.ok) {
            Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "curl -fsSL '$PipBootstrapUrl' -o $VenvPath/get-pip.py") -TimeoutSec $TimeoutSeconds
        } else {
            Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "wget -q '$PipBootstrapUrl' -O $VenvPath/get-pip.py") -TimeoutSec $TimeoutSeconds
        }
        $bootstrapPip = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$VenvPath/bin/python $VenvPath/get-pip.py") -TimeoutSec $TimeoutSeconds
    }
    $upgradePip = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$VenvPath/bin/python -m pip install --upgrade setuptools wheel") -TimeoutSec $TimeoutSeconds
    $packageCommand = "$VenvPath/bin/python -m pip install $(@($PackageList) -join ' ')"
    $installPackages = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", $packageCommand) -TimeoutSec $TimeoutSeconds
    $report.execution.executed = $true
    $report.execution.python_package_downloads_performed = $true
    $report.execution.commands = @(
        [ordered]@{ label = "create_venv"; result = $createVenv },
        [ordered]@{ label = "pip_probe_after_create"; result = $pipAfterCreate },
        [ordered]@{ label = "download_get_pip"; result = $downloadPip },
        [ordered]@{ label = "bootstrap_pip"; result = $bootstrapPip },
        [ordered]@{ label = "upgrade_pip"; result = $upgradePip },
        [ordered]@{ label = "install_packages"; result = $installPackages }
    )
    $setupPassed = [bool]($createVenv.ok -and $downloadPip.ok -and $bootstrapPip.ok -and $upgradePip.ok -and $installPackages.ok)
    $report.execution.setup_passed = $setupPassed
    foreach ($package in $PackageList) {
        $moduleName = Get-ImportModuleName -Package $package
        $probe = Invoke-SafeCommand -Command @("wsl", "bash", "-lc", "$VenvPath/bin/python -c `"import $moduleName as module; print(getattr(module, '__version__', 'ok'))`"") -TimeoutSec 60
        $report.execution.package_probe_after += [ordered]@{
            package = $package
            import_module = $moduleName
            result = $probe
        }
        if (-not $probe.ok) { $setupPassed = $false }
    }
    $report.execution.setup_passed = $setupPassed
}

$report.recommended_next_step = if ($report.execution.setup_passed) {
    "Rerun scripts\56_check_wsl_simulator_deps.ps1, then rerun scripts\55_bounded_simulator_import_smoke.ps1 with ALLOW_SIMULATOR_IMPORT_SMOKE=1."
} elseif ($report.risk_assessment.decision -eq "proceed" -and -not $Execute) {
    "Set ALLOW_WSL_SIM_DEPS=1 task-locally and rerun this script with -Execute if proceeding with bounded WSL venv setup."
} elseif ($Execute -and -not $gateOk) {
    "Set ALLOW_WSL_SIM_DEPS=1 only for this bounded WSL simulator dependency setup task."
} else {
    "Resolve the listed WSL dependency setup blockers; stop before sudo password, token/license/payment, CUDA/driver, graphics-stack, OpenVLA-OFT, render, or rollout gates."
}

Write-Reports -Report $report
exit 0
