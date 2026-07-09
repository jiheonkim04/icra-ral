param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$TG7DReport = "reports\tg7d_adapter_state_gate.json",
    [string]$LiberoDataRoot = "C:\assets\data\libero",
    [string]$LiberoParaMetadataCsv = "C:\assets\data\libero_para\libero_para_metadata.csv",
    [string]$ReportPath = "reports\post_canonicalization_residual_mining.json"
)

$ErrorActionPreference = "Stop"

if ($env:ALLOW_TG7D_ADAPTER_TRAINING -eq "1" -or $env:ALLOW_DOWNLOADS -eq "1" -or $env:ALLOW_OPENVLA_OFT -eq "1" -or $env:ALLOW_ROLLOUTS -eq "1") {
    Write-Output "Refusing residual mining because a forbidden training/download/rollout gate is set."
    exit 20
}

& $Python -m tca_map.tg7d_adapter.residual_mining `
    --tg7d-report $TG7DReport `
    --libero-data-root $LiberoDataRoot `
    --libero-para-metadata-csv $LiberoParaMetadataCsv `
    --report-path $ReportPath

exit $LASTEXITCODE
