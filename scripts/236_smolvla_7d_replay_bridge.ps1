param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$Hdf5Path = "C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
    [string]$SmolVlaCkpt = "C:\assets\checkpoints\smolvla",
    [string]$AdapterArtifact = "runs\smolvla_7d_replay_bridge\smolvla_state_proj_lora_rank8_7d_adapter.pt",
    [string]$ReportPath = "reports\smolvla_7d_replay_bridge_result.json",
    [string]$DataRoot = "C:\assets\data\libero",
    [string]$LiberoRoot = "C:\assets\repos\LIBERO",
    [string]$RobosuiteRoot = "C:\assets\repos\robosuite",
    [int]$AdapterSteps = 800,
    [int]$AdapterHiddenDim = 128,
    [int]$MaxReplaySteps = 280,
    [int]$CameraSize = 64
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "SmolVLA 7D adapter executable replay bridge"
Write-Host "Repo root: $RepoRoot"
Write-Host "Requires ALLOW_SMOLVLA_7D_REPLAY_BRIDGE=1."
Write-Host "If the adapter artifact is missing, also requires ALLOW_SMOLVLA_7D_REPLAY_BRIDGE_TRAINING=1."
Write-Host "Exact-init replay/control is attempted only when ALLOW_SMOLVLA_7D_REPLAY_BRIDGE_REPLAY=1."
Write-Host "Does not download, run OpenVLA-OFT, run a full benchmark, continue TG-7D/TCA/PRISM/PatchGuard/SafeLoRA, invent a method, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 20
}

& $Python -m tca_map.smolvla_lora_baseline.replay_bridge `
    --hdf5-path $Hdf5Path `
    --smolvla-ckpt $SmolVlaCkpt `
    --adapter-artifact $AdapterArtifact `
    --report-path $ReportPath `
    --data-root $DataRoot `
    --libero-root $LiberoRoot `
    --robosuite-root $RobosuiteRoot `
    --adapter-steps $AdapterSteps `
    --adapter-hidden-dim $AdapterHiddenDim `
    --max-replay-steps $MaxReplaySteps `
    --camera-size $CameraSize
exit $LASTEXITCODE
