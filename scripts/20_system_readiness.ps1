param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$ReportsDir = Join-Path $RepoRoot "reports"
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null

function Invoke-SafeCommand {
    param([string[]]$Command)
    try {
        $output = & $Command[0] @($Command[1..($Command.Count - 1)]) 2>&1
        return @{
            available = $LASTEXITCODE -eq 0
            returncode = $LASTEXITCODE
            output = (($output | ForEach-Object { $_.ToString() }) -join "`n")
        }
    } catch {
        return @{
            available = $false
            returncode = $null
            output = $_.Exception.Message
        }
    }
}

function Get-EnvValue {
    param([string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    return @{
        configured = -not [string]::IsNullOrWhiteSpace($value)
        value_redacted = $(if ($value) { "set" } else { $null })
    }
}

$cpu = $null
$ramGb = $null
try {
    $cpuInfo = Get-CimInstance Win32_Processor | Select-Object -First 1
    $cpu = $cpuInfo.Name
    $system = Get-CimInstance Win32_ComputerSystem
    $ramGb = [math]::Round($system.TotalPhysicalMemory / 1GB, 2)
} catch {
    $cpu = "unavailable: $($_.Exception.Message)"
}

$disk = @()
try {
    $disk = Get-PSDrive -PSProvider FileSystem | ForEach-Object {
        @{
            name = $_.Name
            root = $_.Root
            free_gb = [math]::Round($_.Free / 1GB, 2)
            used_gb = [math]::Round($_.Used / 1GB, 2)
        }
    }
} catch {
    $disk = @(@{ error = $_.Exception.Message })
}

$gpu = Invoke-SafeCommand @("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
$wslStatus = Invoke-SafeCommand @("wsl", "--status")
$wslList = Invoke-SafeCommand @("wsl", "--list", "--verbose")
$pythonVersion = Invoke-SafeCommand @($Python, "--version")

$torchCode = @'
import json
try:
    import torch
    result = {
        "import_ok": True,
        "version": getattr(torch, "__version__", None),
        "cuda_is_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }
except Exception as exc:
    result = {
        "import_ok": False,
        "error": str(exc),
        "cuda_is_available": False,
        "torch_cuda_version": None,
    }
print(json.dumps(result))
'@
$torchRaw = Invoke-SafeCommand @($Python, "-c", $torchCode)
$torch = @{ import_ok = $false; error = $torchRaw.output; cuda_is_available = $false; torch_cuda_version = $null }
try {
    if ($torchRaw.available -and $torchRaw.output) {
        $torch = $torchRaw.output | ConvertFrom-Json
    }
} catch {
    $torch = @{ import_ok = $false; error = "Could not parse torch probe: $($torchRaw.output)"; cuda_is_available = $false; torch_cuda_version = $null }
}

$report = [ordered]@{
    policy = [ordered]@{
        downloads_performed = $false
        gpu_training_performed = $false
        heavy_vla_imports_performed = $false
        real_rollouts_performed = $false
    }
    os = [ordered]@{
        platform = [System.Environment]::OSVersion.Platform.ToString()
        version = [System.Environment]::OSVersion.VersionString
    }
    cpu = $cpu
    total_system_ram_gb = $ramGb
    free_disk = $disk
    gpu = $gpu
    conda_env = $env:CONDA_DEFAULT_ENV
    python = [ordered]@{
        command = $Python
        version = $pythonVersion.output
        available = $pythonVersion.available
    }
    torch = $torch
    wsl2 = [ordered]@{
        status = $wslStatus
        distros = $wslList
    }
    paths = [ordered]@{
        paths_local_yaml_exists = Test-Path -LiteralPath "configs\paths.local.yaml"
        HF_HOME = Get-EnvValue "HF_HOME"
        CHECKPOINT_ROOT = Get-EnvValue "CHECKPOINT_ROOT"
        DATA_ROOT = Get-EnvValue "DATA_ROOT"
    }
}

$outPath = Join-Path $ReportsDir "system_readiness.json"
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outPath -Encoding UTF8
Write-Host "System readiness written to $outPath"
$report | ConvertTo-Json -Depth 8
