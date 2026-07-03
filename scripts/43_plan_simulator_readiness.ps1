param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [ValidateSet("auto", "windows", "wsl", "linux")]
    [string]$RuntimePlatform = "auto",
    [string]$JsonReportPath = "reports\simulator_readiness_plan_report.json",
    [string]$MarkdownReportPath = "reports\simulator_readiness_plan_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Simulator readiness risk planner"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not install packages, download assets, import simulators, run render smoke, run rollouts, run GPU jobs, train, import heavy VLA models, access tokens, or execute OpenVLA-OFT."

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

function Get-PathStatus {
    param([hashtable]$Config, [string]$Key, [string]$EnvName, [string]$Label)
    $configured = Get-ConfiguredValue -Config $Config -Key $Key -EnvName $EnvName
    $value = $configured.Value
    $exists = $false
    if (-not [string]::IsNullOrWhiteSpace($value)) { $exists = Test-Path -LiteralPath $value }
    return [ordered]@{
        label = $Label
        env = $EnvName
        configured = -not [string]::IsNullOrWhiteSpace($value)
        exists = [bool]$exists
        source = $configured.Source
        value_redacted = if ([string]::IsNullOrWhiteSpace($value)) { $null } else { "set" }
        path = $value
    }
}

function Invoke-SafeCommand {
    param([string[]]$Command)
    try {
        $output = & $Command[0] @($Command[1..($Command.Count - 1)]) 2>&1
        $text = (($output | ForEach-Object { $_.ToString() }) -join "`n").Replace("`0", "")
        return [ordered]@{
            ok = $LASTEXITCODE -eq 0
            returncode = $LASTEXITCODE
            output = $text
        }
    } catch {
        return [ordered]@{
            ok = $false
            returncode = $null
            output = $_.Exception.Message
        }
    }
}

function Get-WslProbe {
    $wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
    $wslInstalled = $null -ne $wslCommand
    $status = if ($wslInstalled) { Invoke-SafeCommand @("wsl", "--status") } else { [ordered]@{ ok = $false; returncode = $null; output = "wsl command not found" } }
    $distros = if ($wslInstalled) { Invoke-SafeCommand @("wsl", "--list", "--verbose") } else { [ordered]@{ ok = $false; returncode = $null; output = "wsl command not found" } }
    $ubuntuDetected = $false
    if ($distros.output) { $ubuntuDetected = $distros.output -match "Ubuntu" }
    return [ordered]@{
        command_installed = [bool]$wslInstalled
        status = $status
        distros = $distros
        ubuntu_detected = [bool]$ubuntuDetected
    }
}

$config = Read-AssetConfig -Path $PathsFile
$paths = [ordered]@{
    libero_root = Get-PathStatus -Config $config -Key "libero_root" -EnvName "LIBERO_ROOT" -Label "LIBERO source checkout"
    libero_data_root = Get-PathStatus -Config $config -Key "libero_data_root" -EnvName "LIBERO_DATA_ROOT" -Label "LIBERO data/demos root"
    robosuite_root = Get-PathStatus -Config $config -Key "robosuite_root" -EnvName "ROBOSUITE_ROOT" -Label "RoboSuite checkout/install root"
    data_root = Get-PathStatus -Config $config -Key "data_root" -EnvName "DATA_ROOT" -Label "General data root"
}

$wslProbe = Get-WslProbe
$isWindowsHost = [System.Environment]::OSVersion.Platform.ToString() -match "Win"
$effectivePlatform = $RuntimePlatform
if ($RuntimePlatform -eq "auto") {
    if (-not $isWindowsHost) {
        $effectivePlatform = "linux"
    } elseif ($wslProbe.command_installed -and $wslProbe.ubuntu_detected) {
        $effectivePlatform = "wsl"
    } else {
        $effectivePlatform = "windows"
    }
}

$linuxLikeRuntime = $effectivePlatform -in @("linux", "wsl")
$readyForSimulatorPathCheck = [bool]($paths.libero_root.exists -and $paths.robosuite_root.exists)
$readyForDatasetPathCheck = [bool]$paths.libero_data_root.exists
$readyForImportSmoke = [bool]($readyForSimulatorPathCheck -and $linuxLikeRuntime)
$readyForRenderSmoke = $false
$readyForRollout = $false

$stopReasons = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if (-not $paths.libero_root.exists) { $stopReasons.Add("LIBERO_ROOT is missing or does not exist") }
if (-not $paths.robosuite_root.exists) { $stopReasons.Add("ROBOSUITE_ROOT is missing or does not exist") }
if (-not $linuxLikeRuntime) { $stopReasons.Add("simulator import smoke should run in WSL2/Linux, not native Windows") }
if (-not $paths.libero_data_root.exists) { $warnings.Add("LIBERO_DATA_ROOT is missing; import smoke may still be planned, but dataset-backed rollout remains blocked") }
if ($effectivePlatform -eq "windows") { $warnings.Add("native Windows remains a planning/readiness path for simulators; use WSL2/Linux for actual import/render smoke") }
if ($effectivePlatform -eq "wsl" -and -not $wslProbe.ubuntu_detected) { $warnings.Add("runtime platform was set to wsl, but Ubuntu was not detected by the host WSL probe") }

$decision = if ($readyForImportSmoke) { "proceed" } else { "stop" }
$recommendedNextStep = if ($readyForImportSmoke) {
    "Create a separate bounded simulator import-smoke branch. It may import simulator modules only, must avoid rollouts, render loops, policy execution, GPU jobs, OpenVLA-OFT, downloads, and paper claims."
} else {
    "Configure local LIBERO_ROOT and ROBOSUITE_ROOT, prefer WSL2/Linux, then rerun this planner before any simulator import smoke."
}

$report = [ordered]@{
    policy = [ordered]@{
        planning_only = $true
        installs_performed = $false
        downloads_performed = $false
        simulator_imports_performed = $false
        render_smoke_performed = $false
        rollouts_performed = $false
        gpu_jobs_performed = $false
        training_performed = $false
        heavy_model_imports_performed = $false
        openvla_oft_executed = $false
        tokens_read_or_written = $false
        paper_grade_claims_made = $false
    }
    host = [ordered]@{
        os_platform = [System.Environment]::OSVersion.Platform.ToString()
        os_version = [System.Environment]::OSVersion.VersionString
        requested_runtime_platform = $RuntimePlatform
        effective_runtime_platform = $effectivePlatform
        linux_like_runtime = [bool]$linuxLikeRuntime
    }
    wsl = $wslProbe
    paths = $paths
    ready_for_simulator_path_check = $readyForSimulatorPathCheck
    ready_for_dataset_path_check = $readyForDatasetPathCheck
    ready_for_simulator_import_smoke = $readyForImportSmoke
    ready_for_simulator_render_smoke = $readyForRenderSmoke
    ready_for_libero_rollout = $readyForRollout
    warnings = @($warnings)
    stop_reasons = @($stopReasons)
    decision = $decision
    recommended_next_step = $recommendedNextStep
}

$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $jsonFullPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $markdownFullPath) | Out-Null

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8
$markdown = @(
    "# Simulator Readiness Plan Report",
    "",
    "- decision: $decision",
    "- requested runtime platform: $RuntimePlatform",
    "- effective runtime platform: $effectivePlatform",
    "- ready for simulator path check: $readyForSimulatorPathCheck",
    "- ready for dataset path check: $readyForDatasetPathCheck",
    "- ready for simulator import smoke: $readyForImportSmoke",
    "- ready for simulator render smoke: false",
    "- ready for LIBERO rollout: false",
    "",
    "This report is planning-only. It performs no installs, downloads, simulator imports, render smoke, rollouts, GPU jobs, training, heavy VLA imports, token access, OpenVLA-OFT execution, or paper-grade claims."
) -join "`n"
$markdown | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8

$report | ConvertTo-Json -Depth 8
