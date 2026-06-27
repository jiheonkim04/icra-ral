param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "TCA-Map preflight (PowerShell)"
Write-Host "Repo root: $RepoRoot"
& $Python -m tca_map.launch.preflight
