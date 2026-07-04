param(
    [string]$Python = "C:\Users\jiheo\miniconda3\envs\tca_map\python.exe",
    [string]$JsonReportPath = "reports\libero_hdf5_reader_report.json",
    [string]$MarkdownReportPath = "reports\libero_hdf5_reader_report.md"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "LIBERO HDF5 reader dependency check"
Write-Host "Repo root: $RepoRoot"
Write-Host "This script checks h5py availability only. It does not install packages, download data, run GPU jobs, train, rollout, import simulators or heavy VLA models, access tokens, or execute OpenVLA-OFT."

$dangerousGates = @(
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_OPENVLA",
    "ALLOW_CLOUD_HANDOFF"
)
$setGates = @()
foreach ($gate in $dangerousGates) {
    if ([Environment]::GetEnvironmentVariable($gate) -eq "1") {
        $setGates += $gate
    }
}
if ($setGates.Count -gt 0) {
    Write-Host ("Refusing HDF5 reader check while execution gates are set: " + ($setGates -join ", "))
    exit 20
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Python interpreter not found: $Python"
    exit 1
}

$jsonFullPath = if ([System.IO.Path]::IsPathRooted($JsonReportPath)) { $JsonReportPath } else { Join-Path $RepoRoot $JsonReportPath }
$markdownFullPath = if ([System.IO.Path]::IsPathRooted($MarkdownReportPath)) { $MarkdownReportPath } else { Join-Path $RepoRoot $MarkdownReportPath }

$code = @"
import importlib.util
import json
import platform
from pathlib import Path

json_path = Path(r'''$jsonFullPath''')
md_path = Path(r'''$markdownFullPath''')
spec = importlib.util.find_spec("h5py")
available = spec is not None
version = None
error = None
if available:
    try:
        import h5py  # type: ignore
        version = getattr(h5py, "__version__", None)
    except Exception as exc:
        available = False
        error = str(exc)

report = {
    "schema_version": "tca-map-libero-hdf5-reader-v0",
    "policy": {
        "check_only": True,
        "installs_performed": False,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "training_performed": False,
        "rollouts_performed": False,
        "simulator_executed": False,
        "heavy_model_imports_performed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
    },
    "python": {
        "version": platform.python_version(),
        "platform": platform.platform(),
    },
    "h5py": {
        "available": available,
        "version": version,
        "error": error,
    },
    "ready_for_libero_hdf5_interface_read": available,
    "recommended_next_step": (
        "Run scripts/48_plan_libero_offline_interface_smoke.ps1."
        if available
        else "Run a separate dependency risk assessment before installing h5py; do not train, rollout, or import simulators."
    ),
}
json_path.parent.mkdir(parents=True, exist_ok=True)
md_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
md_path.write_text(
    "\\n".join([
        "# LIBERO HDF5 Reader Report",
        "",
        f"- h5py available: `{available}`",
        f"- h5py version: `{version}`",
        f"- ready for LIBERO HDF5 interface read: `{available}`",
        "",
        report["recommended_next_step"],
        "",
    ]),
    encoding="utf-8",
)
print(json.dumps(report, indent=2, sort_keys=True))
"@

$code | & $Python -
exit $LASTEXITCODE
