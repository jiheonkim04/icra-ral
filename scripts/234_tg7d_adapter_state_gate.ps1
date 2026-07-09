param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$LiberoDataRoot = "C:\assets\data\libero",
    [string]$LiberoParaMetadataCsv = "C:\assets\data\libero_para\libero_para_metadata.csv",
    [string]$SmolVLACkpt = "C:\assets\checkpoints\smolvla",
    [string]$ReportPath = "reports\tg7d_adapter_state_gate.json"
)

$ErrorActionPreference = "Stop"

if ($env:ALLOW_DOWNLOADS -eq "1" -or $env:ALLOW_OPENVLA_OFT -eq "1" -or $env:ALLOW_ROLLOUTS -eq "1") {
    Write-Error "Refusing TG-7D gate because a forbidden broad execution gate is set."
    exit 20
}

& $Python -m tca_map.tg7d_adapter.state_gate `
    --libero-data-root $LiberoDataRoot `
    --libero-para-metadata-csv $LiberoParaMetadataCsv `
    --smolvla-ckpt $SmolVLACkpt `
    --report-path $ReportPath

exit $LASTEXITCODE
