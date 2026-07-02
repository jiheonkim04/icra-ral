param(
    [string]$PathsFile = "configs\paths.local.yaml",
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ReportPath = "reports\smolvla_load_only_smoke_plan_report.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

function ConvertFrom-JsonOutput {
    param([string]$Text)

    $start = $Text.IndexOf("{")
    if ($start -lt 0) {
        throw "No JSON object found in checker output."
    }
    return $Text.Substring($start) | ConvertFrom-Json
}

$allowHeavyImport = $env:ALLOW_HEAVY_IMPORT -eq "1"
$reportFullPath = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path $RepoRoot $ReportPath
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $reportFullPath) | Out-Null

$checkerScript = Join-Path $RepoRoot "scripts\13_check_smolvla_adapter_smoke.ps1"
$checkerOutput = & powershell -ExecutionPolicy Bypass -File $checkerScript -PathsFile $PathsFile -Python $Python 2>&1 | Out-String
$checkerReport = ConvertFrom-JsonOutput -Text $checkerOutput

$policy = [ordered]@{
    plan_only = $true
    downloads_performed = $false
    model_load_performed = $false
    model_inference_performed = $false
    gpu_jobs_performed = $false
    gpu_training_performed = $false
    training_performed = $false
    real_rollouts_performed = $false
    heavy_model_imports_performed = $false
    openvla_oft_executed = $false
    tokens_read_or_written = $false
    heavy_import_gate_set = $allowHeavyImport
}

$readyForPlan = [bool]$checkerReport.ready_for_smolvla_adapter_smoke
$unsafeGateSet = [bool]$allowHeavyImport
if ($unsafeGateSet) {
    $recommended = "Unset ALLOW_HEAVY_IMPORT for planning-only checks. A later task must explicitly approve heavy import/model load."
} elseif ($readyForPlan) {
    $recommended = "Prepare a separate approved SmolVLA load-only smoke execution task. Do not train, infer, rollout, or execute OpenVLA-OFT."
} else {
    $recommended = "Resolve SmolVLA readiness before planning any load-only execution."
}

$report = [ordered]@{
    policy = $policy
    repo = [ordered]@{
        root = $RepoRoot
        branch = (& git branch --show-current 2>$null)
        commit = (& git log -1 --oneline 2>$null)
    }
    readiness = [ordered]@{
        ready_for_smolvla_adapter_smoke = $readyForPlan
        checker = $checkerReport
    }
    next_gate = [ordered]@{
        explicit_approval_required = $true
        required_gate = "ALLOW_HEAVY_IMPORT=1"
        gate_is_currently_set = $allowHeavyImport
        execution_script_to_create_later = "scripts/16_smolvla_load_only_smoke.ps1"
        permitted_future_scope_after_approval = @(
            "heavy import/load only",
            "no inference",
            "no training",
            "no rollouts",
            "no OpenVLA-OFT"
        )
    }
    load_only_smoke_plan = [ordered]@{
        objective = "Validate that local SmolVLA checkpoint, tokenizer dependency, adapter guard, and memory budget are sufficient for a later load-only smoke."
        expected_inputs = @(
            "SMOLVLA_CKPT",
            "HF_HOME",
            "CHECKPOINT_ROOT"
        )
        expected_safety_controls = @(
            "separate explicit approval before ALLOW_HEAVY_IMPORT=1",
            "walltime cap",
            "max GPU memory recording",
            "no model inference",
            "no optimizer or training loop",
            "no rollout or simulator import",
            "no OpenVLA-OFT imports"
        )
        success_criteria = @(
            "load-only task exits zero",
            "max GPU memory recorded",
            "no downloads",
            "no inference",
            "no training",
            "no rollouts",
            "no OpenVLA-OFT execution"
        )
        likely_failure_modes = @(
            "missing Python package for SmolVLA/LeRobot",
            "Windows CUDA or PyTorch compatibility issue",
            "SmolVLA config device defaults to cuda and triggers OOM",
            "system RAM pressure while loading weights",
            "dependency path mismatch between HF_HOME and SmolVLA preprocessor config"
        )
    }
    recommended_next_step = $recommended
}

$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportFullPath -Encoding UTF8

Write-Host "SmolVLA load-only smoke plan"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script is planning-only. It does not import SmolVLA, load models, run inference, train, run rollouts, download assets, or execute OpenVLA-OFT."
if ($unsafeGateSet) {
    Write-Host "Refusing planning run because ALLOW_HEAVY_IMPORT=1 is set."
}
$report | ConvertTo-Json -Depth 12

if ($unsafeGateSet) {
    exit 2
}
exit 0
