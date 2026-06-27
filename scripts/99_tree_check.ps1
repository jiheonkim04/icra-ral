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
    "reports/missing_assets.md",
    "reports/linux_setup_todo.md",
    "reports/local_run_instructions.md",
    "reports/real_asset_setup_plan.md",
    "reports/local_papergrade_plan.md",
    "reports/hardware_upgrade_plan.md",
    "reports/local_experiment_matrix.md",
    "reports/cloud_handoff_manifest.md",
    "reports/cloud_handoff_manifest.json",
    "scripts/00_preflight.ps1",
    "scripts/00_preflight.sh",
    "scripts/04_train_smoke.ps1",
    "scripts/05_eval_smoke.ps1",
    "scripts/11_check_real_assets.ps1",
    "scripts/11_check_real_assets.sh",
    "scripts/20_system_readiness.ps1",
    "scripts/20_system_readiness.sh",
    "scripts/21_make_asset_dirs.ps1",
    "scripts/21_make_asset_dirs.sh",
    "scripts/22_plan_local_experiment_matrix.ps1",
    "scripts/22_plan_local_experiment_matrix.sh",
    "scripts/23_cloud_handoff_manifest.ps1",
    "scripts/23_cloud_handoff_manifest.sh",
    "scripts/24_wsl2_setup_check.ps1",
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
