"""Official LIBERO dataset acquisition risk gate and downloader.

This module is intentionally narrow: it can only target the already-recorded
official LIBERO Hugging Face dataset source and the approved local asset root.
It does not import simulators or VLA models.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

OFFICIAL_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
OFFICIAL_SOURCE_URL = f"https://huggingface.co/datasets/{OFFICIAL_REPO_ID}"
DEFAULT_TARGET = Path("C:/assets/data/libero")
DEFAULT_CACHE = Path("C:/assets/hf_home")
DEFAULT_CONFIG = Path("configs/libero_robosuite_sources.yaml")
EXPECTED_SIZE_GB = 100.0
LIBERO_DOWNLOAD_BUDGET_GB = 180.0
MIN_FREE_AFTER_GB = 250.0


def disk_free_gb(path: Path) -> float:
    root = Path(path.anchor or Path.cwd().anchor)
    usage = shutil.disk_usage(root)
    return round(usage.free / (1024**3), 3)


def directory_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return round(total / (1024**3), 3)


def count_dataset_files(path: Path) -> dict[str, Any]:
    extensions = {".hdf5", ".h5", ".npz", ".pkl", ".json", ".jsonl"}
    files: list[str] = []
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file() and item.suffix.lower() in extensions:
                files.append(str(item))
                if len(files) >= 20:
                    break
    return {
        "files_detected": bool(files),
        "sample_files": files,
        "sample_limit": 20,
    }


def validate_recorded_source(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {"valid": False, "reason": f"missing source config: {config_path}"}
    text = config_path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "official_source_url_present": OFFICIAL_SOURCE_URL in text,
        "target_path_present": "C:/assets/data/libero" in text,
        "token_required_false": "token_required: false" in text,
        "license_click_through_required_false": "license_click_through_required: false" in text,
        "payment_required_false": "payment_required: false" in text,
        "expected_size_recorded": "expected_size_gb: 100.0" in text,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "reason": "recorded source metadata matches expected official LIBERO dataset" if all(checks.values()) else "recorded source metadata is incomplete",
    }


def estimate_remote_size_gb(repo_id: str = OFFICIAL_REPO_ID) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return {"available": False, "size_gb": None, "error": f"huggingface_hub unavailable: {exc}"}
    try:
        info = HfApi().repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True, token=False)
    except Exception as exc:
        return {"available": False, "size_gb": None, "error": str(exc)}
    sizes = [sibling.size for sibling in getattr(info, "siblings", []) if getattr(sibling, "size", None) is not None]
    if not sizes:
        return {"available": False, "size_gb": None, "error": "Hugging Face API returned no file sizes"}
    return {
        "available": True,
        "size_gb": round(sum(sizes) / (1024**3), 3),
        "file_count_with_sizes": len(sizes),
        "method": "Hugging Face repo_info files_metadata sum",
    }


def build_risk_report(
    target: Path = DEFAULT_TARGET,
    cache: Path = DEFAULT_CACHE,
    config_path: Path = DEFAULT_CONFIG,
    remote_size_check: bool = False,
) -> dict[str, Any]:
    recorded_source = validate_recorded_source(config_path)
    remote_size = estimate_remote_size_gb() if remote_size_check else {"available": False, "size_gb": None, "skipped": True}
    expected_size = remote_size.get("size_gb") if remote_size.get("available") else EXPECTED_SIZE_GB
    free_before = disk_free_gb(target)
    free_after = round(free_before - float(expected_size), 3)
    target_normalized = str(target).replace("/", "\\").lower()
    source_official = recorded_source["valid"]

    stop_reasons: list[str] = []
    if not source_official:
        stop_reasons.append(recorded_source["reason"])
    if OFFICIAL_SOURCE_URL != f"https://huggingface.co/datasets/{OFFICIAL_REPO_ID}":
        stop_reasons.append("official source constant mismatch")
    if float(expected_size) > LIBERO_DOWNLOAD_BUDGET_GB:
        stop_reasons.append(f"expected size {expected_size} GB exceeds {LIBERO_DOWNLOAD_BUDGET_GB} GB LIBERO-only budget")
    if free_after < MIN_FREE_AFTER_GB:
        stop_reasons.append(f"free disk after estimate {free_after} GB would be below {MIN_FREE_AFTER_GB} GB")
    if not target_normalized.startswith("c:\\assets\\data\\libero"):
        stop_reasons.append("target path is outside C:\\assets\\data\\libero")

    download_tool = {"huggingface_hub_available": False}
    try:
        import huggingface_hub  # noqa: F401

        download_tool["huggingface_hub_available"] = True
    except Exception as exc:
        download_tool["error"] = str(exc)

    if not download_tool["huggingface_hub_available"]:
        stop_reasons.append("huggingface_hub download tool is unavailable")

    decision = "proceed" if not stop_reasons else "stop"
    return {
        "schema_version": "tca-map-libero-data-acquisition-v0",
        "task": "official LIBERO dataset acquisition",
        "source": {
            "repo_id": OFFICIAL_REPO_ID,
            "source_url": OFFICIAL_SOURCE_URL,
            "official_or_documented": source_official,
            "recorded_source_config": str(config_path),
            "recorded_source_validation": recorded_source,
            "license": "apache-2.0",
            "token_required": False,
            "login_required": False,
            "license_click_through_required": False,
            "payment_required": False,
        },
        "target": {
            "target_path": str(target),
            "cache_path": str(cache),
            "libero_root": "C:/assets/repos/LIBERO",
            "robosuite_root": "C:/assets/repos/robosuite",
        },
        "size": {
            "expected_size_gb": expected_size,
            "recorded_expected_size_gb": EXPECTED_SIZE_GB,
            "remote_size": remote_size,
            "size_estimation_method": "recorded repo source-resolution report; optional HF API check when enabled",
        },
        "budgets": {
            "libero_download_budget_gb": LIBERO_DOWNLOAD_BUDGET_GB,
            "min_free_after_gb": MIN_FREE_AFTER_GB,
            "disk_free_before_gb": free_before,
            "disk_free_after_estimate_gb": free_after,
        },
        "policy": {
            "risk_assessment_only": True,
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "simulator_executed": False,
            "rollouts_performed": False,
            "heavy_model_imports_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "external_upload_performed": False,
        },
        "download_tool": download_tool,
        "expected_runtime": "dataset acquisition may take many minutes depending on network throughput",
        "stop_reasons": stop_reasons,
        "decision": decision,
        "reason": "risk assessment is green for official LIBERO dataset acquisition" if decision == "proceed" else "; ".join(stop_reasons),
    }


def acquire_dataset(target: Path, cache: Path) -> dict[str, Any]:
    if os.environ.get("ALLOW_DOWNLOADS") != "1":
        raise RuntimeError("ALLOW_DOWNLOADS=1 is required for this task-local LIBERO acquisition")
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    started = time.time()
    snapshot_path = snapshot_download(
        repo_id=OFFICIAL_REPO_ID,
        repo_type="dataset",
        local_dir=str(target),
        cache_dir=str(cache),
        token=False,
        resume_download=True,
    )
    elapsed = round(time.time() - started, 3)
    return {
        "download_attempted": True,
        "downloads_performed": True,
        "snapshot_path": snapshot_path,
        "elapsed_seconds": elapsed,
        "target_size_gb": directory_size_gb(target),
        "dataset_probe": count_dataset_files(target),
    }


def write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LIBERO Data Acquisition Report",
        "",
        f"- decision: `{report['decision']}`",
        f"- source: `{report['source']['source_url']}`",
        f"- target: `{report['target']['target_path']}`",
        f"- expected size GB: `{report['size']['expected_size_gb']}`",
        f"- budget GB: `{report['budgets']['libero_download_budget_gb']}`",
        f"- disk free before GB: `{report['budgets']['disk_free_before_gb']}`",
        f"- disk free after estimate GB: `{report['budgets']['disk_free_after_estimate_gb']}`",
        f"- minimum free after GB: `{report['budgets']['min_free_after_gb']}`",
        f"- downloads performed: `{report['policy']['downloads_performed']}`",
        f"- rollout performed: `{report['policy']['rollouts_performed']}`",
        f"- OpenVLA-OFT executed: `{report['policy']['openvla_oft_executed']}`",
        "",
        report["reason"],
        "",
    ]
    if report.get("acquisition"):
        lines.extend(
            [
                "## Acquisition",
                f"- elapsed seconds: `{report['acquisition'].get('elapsed_seconds')}`",
                f"- target size GB: `{report['acquisition'].get('target_size_gb')}`",
                f"- dataset files detected: `{report['acquisition'].get('dataset_probe', {}).get('files_detected')}`",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report-json", default="reports/libero_data_acquisition_report.json")
    parser.add_argument("--report-md", default="reports/libero_data_acquisition_report.md")
    parser.add_argument("--remote-size-check", action="store_true")
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()

    target = Path(args.target)
    cache = Path(args.cache)
    report = build_risk_report(
        target=target,
        cache=cache,
        config_path=Path(args.config),
        remote_size_check=args.remote_size_check,
    )
    if args.acquire:
        if report["decision"] != "proceed":
            write_reports(report, Path(args.report_json), Path(args.report_md))
            print(json.dumps(report, indent=2, sort_keys=True))
            raise SystemExit(30)
        try:
            acquisition = acquire_dataset(target=target, cache=cache)
            report["acquisition"] = acquisition
            report["policy"]["downloads_performed"] = True
            report["decision"] = "acquired"
            report["reason"] = "official LIBERO dataset acquisition completed"
        except Exception as exc:
            report["acquisition_error"] = str(exc)
            report["decision"] = "stop"
            report["reason"] = f"acquisition failed: {exc}"
            write_reports(report, Path(args.report_json), Path(args.report_md))
            print(json.dumps(report, indent=2, sort_keys=True))
            raise SystemExit(40)

    write_reports(report, Path(args.report_json), Path(args.report_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
