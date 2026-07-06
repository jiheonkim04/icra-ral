param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [switch]$Continuous,
    [int]$MaxStates = 5,
    [double]$MaxHours = 6,
    [switch]$DryRun,
    [int]$State2MaxSteps = 20,
    [int]$State3Tasks = 3,
    [int]$State3MaxSteps = 10,
    [int]$TimeoutSeconds = 3600
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "CSS-Shield Phase 2 bounded autopilot"
Write-Host "Repo root: $RepoRoot"
Write-Host "Policy: no downloads, no GPU jobs, no training, no OpenVLA-OFT, no paper-grade claim."

if ($State2MaxSteps -lt 1 -or $State2MaxSteps -gt 25) {
    Write-Host "Refusing: State2MaxSteps must be between 1 and 25."
    exit 12
}
if ($State3Tasks -lt 1 -or $State3Tasks -gt 5) {
    Write-Host "Refusing: State3Tasks must be between 1 and 5."
    exit 13
}
if ($State3MaxSteps -lt 1 -or $State3MaxSteps -gt 25) {
    Write-Host "Refusing: State3MaxSteps must be between 1 and 25."
    exit 14
}

function Get-Phase2State {
    if (-not (Test-Path -LiteralPath "reports\css_shield_phase2_state.json")) {
        return [pscustomobject]@{
            current_state = "PHASE2_STATE 1"
            continue_kill_decision = "continue"
        }
    }
    return Get-Content -Raw -LiteralPath "reports\css_shield_phase2_state.json" | ConvertFrom-Json
}

function Invoke-Phase2StateUpdate {
    param([string]$StateName)
    $mainCommit = (git rev-parse HEAD).Trim()
    if ($DryRun) {
        Write-Host "DRY RUN: $Python -m tca_map.css_shield.phase2_autopilot --main-commit $mainCommit --complete-state $StateName"
        return
    }
    & $Python -m tca_map.css_shield.phase2_autopilot --main-commit $mainCommit --complete-state $StateName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-NativeDiagnostic {
    if ($DryRun) {
        Write-Host "DRY RUN: scripts\151_css_shield_minimal_rollout_diagnostic_wsl.ps1 native phase2"
        return
    }
    powershell -ExecutionPolicy Bypass -File scripts\151_css_shield_minimal_rollout_diagnostic_wsl.ps1 `
        -JsonReportPath "reports/css_shield_phase2_native_action_diagnostic_report.json" `
        -MarkdownReportPath "reports/css_shield_phase2_native_action_diagnostic_report.md" `
        -ProposalSource "native_smolvla" `
        -MaxSteps $State2MaxSteps `
        -CaseIndex 0 `
        -TimeoutSeconds $TimeoutSeconds
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Invoke-Phase2StateUpdate -StateName "STATE2"
}

function Invoke-MultiTaskDiagnostic {
    if ($DryRun) {
        Write-Host "DRY RUN: Phase 2 multitask diagnostic for $State3Tasks tasks"
        return
    }
    for ($idx = 0; $idx -lt $State3Tasks; $idx++) {
        $jsonPath = "reports/css_shield_phase2_multitask_task_$idx`_report.json"
        $mdPath = "reports/css_shield_phase2_multitask_task_$idx`_report.md"
        Write-Host "Running Phase 2 task index $idx"
        powershell -ExecutionPolicy Bypass -File scripts\151_css_shield_minimal_rollout_diagnostic_wsl.ps1 `
            -JsonReportPath $jsonPath `
            -MarkdownReportPath $mdPath `
            -ProposalSource "native_smolvla" `
            -MaxSteps $State3MaxSteps `
            -CaseIndex $idx `
            -TimeoutSeconds $TimeoutSeconds
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    Invoke-Phase2StateUpdate -StateName "STATE3"
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

    $state = Get-Phase2State
    $stage = [string]$state.current_state
    $decision = [string]$state.continue_kill_decision
    Write-Host "Current CSS-Shield Phase 2 state: $stage"

    if ($stage -eq "COMPLETE") {
        Write-Host "Stopping: Phase 2 package is complete."
        break
    }
    if (($decision -like "kill*") -and ($stage -ne "PHASE2_STATE 5")) {
        Write-Host "Stopping before more execution: current decision is $decision."
        break
    }

    switch ($stage) {
        "PHASE2_STATE 1" { Invoke-Phase2StateUpdate -StateName "STATE1" }
        "PHASE2_STATE 2" { Invoke-NativeDiagnostic }
        "PHASE2_STATE 3" { Invoke-MultiTaskDiagnostic }
        "PHASE2_STATE 4" { Invoke-Phase2StateUpdate -StateName "STATE4" }
        "PHASE2_STATE 5" { Invoke-Phase2StateUpdate -StateName "STATE5" }
        default {
            Write-Host "Stopping: no executable next state for '$stage'."
            break
        }
    }

    $statesRun += 1
    if (-not $Continuous) {
        Write-Host "Stopping: one-shot mode completed one Phase 2 state. Use -Continuous to keep going."
        break
    }
}

exit 0
