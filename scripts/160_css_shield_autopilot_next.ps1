param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [switch]$Continuous,
    [int]$MaxStates = 5,
    [double]$MaxHours = 6,
    [string]$StopAfterState = "",
    [switch]$DryRun,
    [switch]$IncludeNative,
    [int]$State2Trials = 20,
    [int]$State4Trials = 50
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "CSS-Shield bounded continuous autopilot controller"
Write-Host "Repo root: $RepoRoot"
Write-Host "Policy: no downloads, no GPU jobs, no training, no OpenVLA-OFT, no paper-grade claim."

function Get-State {
    if (-not (Test-Path -LiteralPath "reports\css_shield_autopilot_state.json")) {
        return [pscustomobject]@{ current_stage = "STATE 1.5"; continue_kill_decision = "continue" }
    }
    return Get-Content -Raw -LiteralPath "reports\css_shield_autopilot_state.json" | ConvertFrom-Json
}

function Invoke-State15Or2 {
    param([int]$Trials)
    $stateArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\161_css_shield_state1_5_semantic_observability.ps1",
        "-RunState2IfGreen",
        "-State2Trials", "$Trials"
    )
    if ($IncludeNative) { $stateArgs += "-IncludeNative" }
    if ($DryRun) {
        Write-Host "DRY RUN: powershell $($stateArgs -join ' ')"
        return
    }
    powershell @stateArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $mainCommit = (git rev-parse HEAD).Trim()
    & $Python -m tca_map.css_shield.autopilot_next --main-commit $mainCommit
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-State3 {
    if ($DryRun) {
        Write-Host "DRY RUN: complete STATE3"
        return
    }
    $mainCommit = (git rev-parse HEAD).Trim()
    & $Python -m tca_map.css_shield.autopilot_next --main-commit $mainCommit --complete-state STATE3
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-State4 {
    $stateArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\161_css_shield_state1_5_semantic_observability.ps1",
        "-RunState2IfGreen",
        "-State2Trials", "$State4Trials",
        "-ReportJsonPath", "reports/css_shield_state4_scale_diagnostic_report.json",
        "-ReportMarkdownPath", "reports/css_shield_state4_scale_diagnostic_report.md",
        "-InventoryJsonPath", "reports/css_shield_state4_object_inventory.json",
        "-InventoryMarkdownPath", "reports/css_shield_state4_object_inventory.md"
    )
    if ($IncludeNative) { $stateArgs += "-IncludeNative" }
    if ($DryRun) {
        Write-Host "DRY RUN: powershell $($stateArgs -join ' ')"
        return
    }
    powershell @stateArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $mainCommit = (git rev-parse HEAD).Trim()
    & $Python -m tca_map.css_shield.autopilot_next --main-commit $mainCommit --complete-state STATE4
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-State5 {
    if ($DryRun) {
        Write-Host "DRY RUN: complete STATE5"
        return
    }
    $mainCommit = (git rev-parse HEAD).Trim()
    & $Python -m tca_map.css_shield.autopilot_next --main-commit $mainCommit --complete-state STATE5
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$started = Get-Date
$statesRun = 0
while ($true) {
    if ($statesRun -ge $MaxStates) {
        Write-Host "Stopping: MaxStates reached ($MaxStates)."
        break
    }
    $elapsedHours = ((Get-Date) - $started).TotalHours
    if ($elapsedHours -ge $MaxHours) {
        Write-Host "Stopping: MaxHours reached ($MaxHours)."
        break
    }
    $state = Get-State
    $stage = [string]$state.current_stage
    $decision = [string]$state.continue_kill_decision
    Write-Host "Current CSS-Shield stage: $stage"

    if ($decision -like "kill*") {
        Write-Host "Stopping: current state decision is $decision."
        break
    }
    if ($stage -eq "COMPLETE") {
        Write-Host "Stopping: autopilot package is complete."
        break
    }
    if ($StopAfterState -and $statesRun -gt 0 -and $StopAfterState -eq $stage) {
        Write-Host "Stopping before $stage due to StopAfterState."
        break
    }

    switch ($stage) {
        "STATE 1.5" { Invoke-State15Or2 -Trials $State2Trials }
        "STATE 2" { Invoke-State15Or2 -Trials $State2Trials }
        "STATE 3" { Invoke-State3 }
        "STATE 4" { Invoke-State4 }
        "STATE 5" { Invoke-State5 }
        default {
            Write-Host "Stopping: no executable next state for '$stage'."
            break
        }
    }
    $statesRun += 1
    if (-not $Continuous) {
        Write-Host "Stopping: one-shot mode completed one state. Use -Continuous to keep going."
        break
    }
}

exit 0
