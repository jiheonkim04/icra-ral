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
    "configs/tca_select_inference.yaml",
    "configs/tca_select_ablation.yaml",
    "configs/distributional_tca_select.yaml",
    "configs/lora_adapter_lowcompute.yaml",
    "configs/qlora_adapter_lowcompute.yaml",
    "reports/missing_assets.md",
    "reports/linux_setup_todo.md",
    "reports/local_run_instructions.md",
    "reports/real_asset_setup_plan.md",
    "reports/tca_select_method.md",
    "reports/final_method_spec_distributional_tca_map.md",
    "reports/action_decoder_landscape.md",
    "reports/mg_select_vs_distributional_tca_select.md",
    "reports/lora_inference_ablation_plan.md",
    "reports/lora_vs_inference_trick_strategy.md",
    "reports/publishability_criteria.md",
    "reports/reviewer2_tca_select_lora_risk.md",
    "scripts/00_preflight.ps1",
    "scripts/00_preflight.sh",
    "scripts/04_train_smoke.ps1",
    "scripts/05_eval_smoke.ps1",
    "scripts/11_check_real_assets.ps1",
    "scripts/11_check_real_assets.sh",
    "scripts/99_tree_check.ps1",
    "tca_map/__init__.py",
    "tca_map/inference/__init__.py",
    "tca_map/inference/tca_select.py",
    "tca_map/adapters/__init__.py",
    "tca_map/adapters/lora_policy.py",
    "tests/test_tca_select.py",
    "tests/test_distributional_tca_select.py",
    "tests/test_lora_config_guards.py"
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
