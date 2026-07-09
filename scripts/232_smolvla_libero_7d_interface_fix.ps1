param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$Hdf5Path = "C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
    [string]$SmolVlaCkpt = "C:\assets\checkpoints\smolvla",
    [string]$ReportPath = "reports\smolvla_libero_7d_interface_fix.json",
    [int]$OneSampleSteps = 500,
    [int]$OneDemoSteps = 800,
    [int]$AdapterSteps = 1000,
    [int]$BaselineMlpSteps = 800,
    [int]$AdapterHiddenDim = 128
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "SmolVLA-LIBERO 7D action interface fix"
Write-Host "Repo root: $RepoRoot"
Write-Host "Requires ALLOW_SMOLVLA_LIBERO_7D_INTERFACE_FIX=1 and ALLOW_SMOLVLA_LIBERO_7D_INTERFACE_TRAINING=1."
Write-Host "Runs bounded schema, normalization, learned-gripper adapter, overfit, and larger-split diagnostics only."
Write-Host "Does not download, rollout, run OpenVLA-OFT, continue PatchGuard, invent a method, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

& $Python -m tca_map.smolvla_lora_baseline.libero_7d_interface_fix `
    --hdf5-path $Hdf5Path `
    --smolvla-ckpt $SmolVlaCkpt `
    --report-path $ReportPath `
    --one-sample-steps $OneSampleSteps `
    --one-demo-steps $OneDemoSteps `
    --adapter-steps $AdapterSteps `
    --baseline-mlp-steps $BaselineMlpSteps `
    --adapter-hidden-dim $AdapterHiddenDim
exit $LASTEXITCODE
