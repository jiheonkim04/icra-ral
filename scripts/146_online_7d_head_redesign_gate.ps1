param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$ManifestPath = "reports\libero_offline_counterfactual_split_scaled_report.json",
    [string]$JsonReportPath = "reports\online_7d_head_redesign_gate_report.json",
    [string]$MarkdownReportPath = "reports\online_7d_head_redesign_gate_report.md",
    [int]$MaxSteps = 25,
    [int]$TrainMaxSteps = 64,
    [int]$SampleStride = 4,
    [int]$TeacherMaxSteps = 300,
    [int]$MlpSteps = 200
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Bounded 7D action-head redesign gate"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script trains/evaluates bounded CPU 7D heads offline/teacher-forced only."
Write-Host "It does not download, use GPU, train LoRA, run rollout, import heavy VLA models, or execute OpenVLA-OFT."

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $RepoRoot $Path
}

if ($MaxSteps -lt 1 -or $MaxSteps -gt 25) {
    Write-Host "Refusing: MaxSteps must be between 1 and 25."
    exit 12
}
if ($TrainMaxSteps -lt 1 -or $TrainMaxSteps -gt 256) {
    Write-Host "Refusing: TrainMaxSteps must be between 1 and 256."
    exit 13
}
if ($SampleStride -lt 1 -or $SampleStride -gt 32) {
    Write-Host "Refusing: SampleStride must be between 1 and 32."
    exit 14
}
if ($TeacherMaxSteps -lt 1 -or $TeacherMaxSteps -gt 512) {
    Write-Host "Refusing: TeacherMaxSteps must be between 1 and 512."
    exit 15
}
if ($MlpSteps -lt 1 -or $MlpSteps -gt 300) {
    Write-Host "Refusing: MlpSteps must be between 1 and 300."
    exit 16
}

& $Python -m tca_map.smolvla.online_7d_head_redesign_gate `
    --manifest (Resolve-RepoPath -Path $ManifestPath) `
    --report-json (Resolve-RepoPath -Path $JsonReportPath) `
    --report-md (Resolve-RepoPath -Path $MarkdownReportPath) `
    --max-steps $MaxSteps `
    --train-max-steps $TrainMaxSteps `
    --sample-stride $SampleStride `
    --teacher-max-steps $TeacherMaxSteps `
    --mlp-steps $MlpSteps

exit $LASTEXITCODE
