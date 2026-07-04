param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$SourceRepo = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    [string]$RiskReportPath = "reports\vlm_enabled_loading_risk_plan_report.json",
    [string]$TargetRoot = "C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct",
    [string]$HfHome = "C:\assets\hf_home",
    [string]$JsonReportPath = "reports\vlm_required_files_acquisition_report.json",
    [string]$MarkdownReportPath = "reports\vlm_required_files_acquisition_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "VLM required files acquisition"
Write-Host "Repo root: $RepoRoot"
Write-Host "Source: $SourceRepo"
Write-Host "Target directory: $TargetRoot"
Write-Host "Cache directory: $HfHome"
Write-Host "Task scope: SmolVLA dependency file acquisition only"
Write-Host "This script requires ALLOW_DOWNLOADS=1 and downloads only selected files from the official SmolVLM2 dependency."
Write-Host "It does not load models, infer, train, rollout, use GPU jobs, execute OpenVLA-OFT, access tokens, or make paper claims."

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python interpreter not found: $Python"
    exit 10
}

& $Python -m tca_map.smolvla.vlm_required_files_acquisition `
    --source-repo $SourceRepo `
    --risk-report $RiskReportPath `
    --target-root $TargetRoot `
    --hf-home $HfHome `
    --json-report $JsonReportPath `
    --markdown-report $MarkdownReportPath
exit $LASTEXITCODE
