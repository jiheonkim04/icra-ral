param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$Hdf5Path = "C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
    [string]$ReportPath = "reports\smolvla_lora_baseline_diagnosis.json",
    [int]$OverfitSteps = 80,
    [int]$CapacitySteps = 50,
    [int]$LoraRank = 4
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "SmolVLA LoRA baseline diagnosis"
Write-Host "Repo root: $RepoRoot"
Write-Host "Requires ALLOW_HEAVY_IMPORT=1, ALLOW_SMOLVLA_LORA_BASELINE_DIAGNOSIS=1, and ALLOW_SMOLVLA_LORA_BASELINE_DIAGNOSIS_TRAINING=1."
Write-Host "Runs bounded data/split, action-interface, overfit, and capacity diagnostics only. It does not download, rollout, run OpenVLA-OFT, continue PatchGuard, invent a method, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

& $Python -m tca_map.smolvla_lora_baseline.diagnosis `
    --hdf5-path $Hdf5Path `
    --report-path $ReportPath `
    --overfit-steps $OverfitSteps `
    --capacity-steps $CapacitySteps `
    --lora-rank $LoraRank
exit $LASTEXITCODE
