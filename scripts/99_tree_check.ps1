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
    "configs/smolvla_smoke.yaml",
    "configs/smolvla_feature_cache_pilot.yaml",
    "configs/tca_map_head_only_lowcompute.yaml",
    "configs/actionmap_head_only_lowcompute.yaml",
    "configs/tca_select_inference.yaml",
    "configs/tca_select_ablation.yaml",
    "configs/distributional_tca_select.yaml",
    "configs/lora_adapter_lowcompute.yaml",
    "configs/qlora_adapter_lowcompute.yaml",
    "reports/missing_assets.md",
    "reports/linux_setup_todo.md",
    "reports/local_run_instructions.md",
    "reports/next_actions.md",
    "reports/project_state.md",
    "reports/real_asset_setup_plan.md",
    "reports/smolvla_asset_setup.md",
    "reports/smolvla_download_plan.md",
    "reports/smolvla_load_only_smoke_plan.md",
    "reports/smolvla_manual_acquisition_checklist.md",
    "reports/smolvla_runtime_dependency_plan.md",
    "reports/smolvla_runtime_install_request.md",
    "reports/no_large_openvla_strategy.md",
    "reports/low_compute_experiment_plan.md",
    "reports/reviewer2_no_large_openvla_risk.md",
    "reports/local_papergrade_plan.md",
    "reports/hardware_upgrade_plan.md",
    "reports/local_experiment_matrix.md",
    "reports/cloud_handoff_manifest.md",
    "reports/cloud_handoff_manifest.json",
    "reports/codex_delegation_manual.md",
    "reports/decision_log.md",
    "reports/integration_lowcompute_distributional_stack.md",
    "reports/tca_select_method.md",
    "reports/final_method_spec_distributional_tca_map.md",
    "reports/action_decoder_landscape.md",
    "reports/mg_select_vs_distributional_tca_select.md",
    "reports/lora_inference_ablation_plan.md",
    "reports/lora_vs_inference_trick_strategy.md",
    "reports/publishability_criteria.md",
    "reports/reviewer2_tca_select_lora_risk.md",
    "reports/risk_register.md",
    "scripts/00_preflight.ps1",
    "scripts/00_preflight.sh",
    "scripts/04_train_smoke.ps1",
    "scripts/05_eval_smoke.ps1",
    "scripts/11_check_real_assets.ps1",
    "scripts/11_check_real_assets.sh",
    "scripts/12_prepare_smolvla_assets.ps1",
    "scripts/12_prepare_smolvla_assets.sh",
    "scripts/13_check_smolvla_adapter_smoke.ps1",
    "scripts/13_check_smolvla_adapter_smoke.sh",
    "scripts/14_plan_smolvla_download.ps1",
    "scripts/14_plan_smolvla_download.sh",
    "scripts/15_plan_smolvla_load_only_smoke.ps1",
    "scripts/16_smolvla_load_only_smoke.ps1",
    "scripts/17_check_smolvla_runtime_deps.ps1",
    "scripts/18_plan_smolvla_runtime_install.ps1",
    "scripts/20_system_readiness.ps1",
    "scripts/20_system_readiness.sh",
    "scripts/21_make_asset_dirs.ps1",
    "scripts/21_make_asset_dirs.sh",
    "scripts/22_plan_local_experiment_matrix.ps1",
    "scripts/22_plan_local_experiment_matrix.sh",
    "scripts/23_cloud_handoff_manifest.ps1",
    "scripts/23_cloud_handoff_manifest.sh",
    "scripts/24_wsl2_setup_check.ps1",
    "scripts/30_enforce_compute_budget.ps1",
    "scripts/30_enforce_compute_budget.sh",
    "scripts/99_tree_check.ps1",
    "tca_map/__init__.py",
    "tca_map/inference/__init__.py",
    "tca_map/inference/tca_select.py",
    "tca_map/adapters/__init__.py",
    "tca_map/adapters/lora_policy.py",
    "tca_map/smolvla/__init__.py",
    "tca_map/smolvla/load_only_smoke.py",
    "tests/test_tca_select.py",
    "tests/test_distributional_tca_select.py",
    "tests/test_smolvla_asset_readiness.py",
    "tests/test_smolvla_download_plan.py",
    "tests/test_smolvla_load_only_smoke_plan.py",
    "tests/test_smolvla_load_only_smoke_scaffold.py",
    "tests/test_smolvla_runtime_deps_check.py",
    "tests/test_smolvla_runtime_install_plan.py",
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
