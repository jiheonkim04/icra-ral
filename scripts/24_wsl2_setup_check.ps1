$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
$ReportsDir = Join-Path $RepoRoot "reports"
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
$OutPath = Join-Path $ReportsDir "wsl2_status.md"

function Invoke-SafeCommand {
    param([string[]]$Command)
    try {
        $output = & $Command[0] @($Command[1..($Command.Count - 1)]) 2>&1
        return @{
            ok = $LASTEXITCODE -eq 0
            returncode = $LASTEXITCODE
            output = (($output | ForEach-Object { $_.ToString() }) -join "`n")
        }
    } catch {
        return @{
            ok = $false
            returncode = $null
            output = $_.Exception.Message
        }
    }
}

$wslCommand = Get-Command wsl -ErrorAction SilentlyContinue
$wslInstalled = $null -ne $wslCommand
$status = if ($wslInstalled) { Invoke-SafeCommand @("wsl", "--status") } else { @{ ok = $false; output = "wsl command not found" } }
$distros = if ($wslInstalled) { Invoke-SafeCommand @("wsl", "--list", "--verbose") } else { @{ ok = $false; output = "wsl command not found" } }
$ubuntuExists = $false
if ($distros.output) {
    $ubuntuExists = $distros.output -match "Ubuntu"
}
$wslNvidia = if ($wslInstalled -and $ubuntuExists) {
    Invoke-SafeCommand @("wsl", "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
} else {
    @{ ok = $false; output = "Skipped because WSL or Ubuntu is missing." }
}

$md = @"
# WSL2 Status

This report only checks WSL2 status and prints setup commands. It does not install WSL, install Ubuntu, download assets, run GPU jobs, or run rollouts.

## Summary

- WSL command installed: $wslInstalled
- Ubuntu distro detected: $ubuntuExists
- NVIDIA visible inside WSL: $($wslNvidia.ok)

## `wsl --status`

```text
$($status.output)
```

## `wsl --list --verbose`

```text
$($distros.output)
```

## WSL NVIDIA check

```text
$($wslNvidia.output)
```

## Install commands if missing

Run these manually in an elevated PowerShell only if WSL/Ubuntu is missing:

```powershell
wsl --install -d Ubuntu
wsl --set-default-version 2
wsl --list --verbose
```

After Ubuntu is installed, open Ubuntu once, finish user setup, then check GPU visibility:

```powershell
wsl nvidia-smi
```

Then use Linux-side checks from the repository:

```bash
bash scripts/00_preflight.sh
bash scripts/11_check_real_assets.sh
bash scripts/20_system_readiness.sh
```
"@

$md | Set-Content -LiteralPath $OutPath -Encoding UTF8
Write-Host "Wrote $OutPath"
Write-Host $md
