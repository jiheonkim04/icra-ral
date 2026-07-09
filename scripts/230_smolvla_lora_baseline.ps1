param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$Hdf5Path = "C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
    [string]$ReportPath = "reports\smolvla_lora_baseline_state1_result.json",
    [int]$MaxSteps = 60,
    [int]$MaxTrainDemos = 3,
    [int]$MaxEvalDemos = 2,
    [int]$RecordsPerDemo = 3,
    [int]$LoraRank = 4
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "SmolVLA standard LoRA baseline STATE 1"
Write-Host "Repo root: $RepoRoot"
Write-Host "Requires ALLOW_HEAVY_IMPORT=1, ALLOW_SMOLVLA_LORA_BASELINE=1, and ALLOW_SMOLVLA_LORA_BASELINE_TRAINING=1."
Write-Host "Runs a bounded standard LoRA baseline on local LIBERO HDF5 data only. It does not download, rollout, run OpenVLA-OFT, invent a method, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

& $Python -m tca_map.smolvla_lora_baseline.diagnostic `
    --hdf5-path $Hdf5Path `
    --report-path $ReportPath `
    --max-steps $MaxSteps `
    --max-train-demos $MaxTrainDemos `
    --max-eval-demos $MaxEvalDemos `
    --records-per-demo $RecordsPerDemo `
    --lora-rank $LoraRank
exit $LASTEXITCODE
