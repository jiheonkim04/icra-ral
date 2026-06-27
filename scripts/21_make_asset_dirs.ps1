param(
    [string]$AssetRoot = "C:\assets"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$allowCreate = $env:ALLOW_CREATE_DIRS -eq "1"
$dirs = @(
    Join-Path $AssetRoot "checkpoints",
    Join-Path $AssetRoot "data",
    Join-Path $AssetRoot "repos",
    Join-Path $AssetRoot "hf_home"
)

Write-Host "TCA-Map asset directory planner"
Write-Host "Dry run: $(-not $allowCreate)"
Write-Host "No downloads are performed."

foreach ($dir in $dirs) {
    if ($allowCreate) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "created_or_exists: $dir"
    } else {
        Write-Host "would_create: $dir"
    }
}

$yaml = @"
assets:
  openvla_oft_ckpt: "C:/assets/checkpoints/openvla-oft"
  smolvla_ckpt: "C:/assets/checkpoints/smolvla"
  libero_root: "C:/assets/repos/LIBERO"
  libero_data_root: "C:/assets/data/libero"
  robosuite_root: "C:/assets/repos/robosuite"
  data_root: "C:/assets/data"
  checkpoint_root: "C:/assets/checkpoints"
  hf_home: "C:/assets/hf_home"
  wandb_api_key: null
"@

Write-Host ""
Write-Host "Matching configs/paths.local.yaml template:"
Write-Host $yaml
Write-Host ""
Write-Host "To actually create directories, rerun with:"
Write-Host '$env:ALLOW_CREATE_DIRS="1"; powershell -ExecutionPolicy Bypass -File scripts\21_make_asset_dirs.ps1'
