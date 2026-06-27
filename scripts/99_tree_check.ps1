$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$required = @(
    "AGENTS.md",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "configs/paths.example.yaml",
    "configs/paths.local.yaml.example",
    "configs/compute_budget.yaml",
    "configs/smolvla_feature_cache_pilot.yaml",
    "configs/tca_map_head_only_lowcompute.yaml",
    "configs/actionmap_head_only_lowcompute.yaml",
    "reports/missing_assets.md",
    "reports/linux_setup_todo.md",
    "reports/local_run_instructions.md",
    "reports/real_asset_setup_plan.md",
    "reports/no_large_openvla_strategy.md",
    "reports/low_compute_experiment_plan.md",
    "reports/reviewer2_no_large_openvla_risk.md",
    "scripts/00_preflight.ps1",
    "scripts/00_preflight.sh",
    "scripts/04_train_smoke.ps1",
    "scripts/05_eval_smoke.ps1",
    "scripts/11_check_real_assets.ps1",
    "scripts/11_check_real_assets.sh",
    "scripts/30_enforce_compute_budget.ps1",
    "scripts/30_enforce_compute_budget.sh",
    "scripts/99_tree_check.ps1",
    "tca_map/__init__.py"
)

$missing = @()
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        $missing += $path
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Missing scaffold files:"
    $missing | ForEach-Object { Write-Host "- $_" }
    exit 1
}

Write-Host "Scaffold tree check passed."
