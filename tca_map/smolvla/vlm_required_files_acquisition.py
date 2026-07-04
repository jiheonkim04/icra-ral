"""Acquire the bounded SmolVLM2 files required for VLM-enabled SmolVLA checks.

This module is deliberately narrow: it downloads selected files from the
official Hugging FaceTB dependency only after the metadata risk planner has
passed and the task-local download gate is set. It never loads models, runs
inference, trains, rolls out, touches OpenVLA-OFT, or reads tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


SOURCE_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DOWNLOAD_GATE = "ALLOW_DOWNLOADS"
MAX_EXPECTED_SIZE_GB = 8.0
MIN_FREE_AFTER_GB = 250.0
MAX_RUNTIME_SECONDS = 1800
TARGET_ROOT = Path("C:/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
HF_HOME = Path("C:/assets/hf_home")

FORBIDDEN_GATES = [
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
    "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
    "ALLOW_WSL_SMOLVLA_SINGLE_ACTION",
    "ALLOW_VLM_ENABLED_LOAD_SMOKE",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _disk_free_gb(path: Path) -> float:
    path = path.expanduser()
    anchor = path.anchor or Path.cwd().anchor or str(Path.cwd())
    return round(shutil.disk_usage(anchor).free / (1024**3), 3)


def _size_gb(size_bytes: int | float | None) -> float:
    if size_bytes is None:
        return 0.0
    return float(size_bytes) / (1024**3)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-10:],
    }


def _dedupe_file_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry.get("rfilename") or "").replace("\\", "/").strip("/")
        if not name or name in seen:
            continue
        seen.add(name)
        copied = dict(entry)
        copied["rfilename"] = name
        result.append(copied)
    return result


def required_files_from_risk_report(risk: dict[str, Any]) -> list[dict[str, Any]]:
    files = risk.get("files") or {}
    return _dedupe_file_entries(
        list(files.get("root_safetensors") or [])
        + list(files.get("config_tokenizer_processor_files") or [])
    )


def _expected_size_gb(files: list[dict[str, Any]]) -> float:
    return round(sum(_size_gb(item.get("size")) for item in files), 3)


def _all_sizes_known(files: list[dict[str, Any]]) -> bool:
    return bool(files) and all(item.get("size") for item in files)


def _existing_files(target_root: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    present = []
    for item in files:
        rel = str(item["rfilename"])
        path = target_root / rel
        if path.exists() and path.is_file():
            present.append(
                {
                    "rfilename": rel,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "expected_size": item.get("size"),
                }
            )
    return present


def _target_size_gb(target_root: Path) -> float:
    if not target_root.exists():
        return 0.0
    total = 0
    for path in target_root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return round(total / (1024**3), 3)


def _download_required_files(
    *,
    source_repo: str,
    files: list[dict[str, Any]],
    target_root: Path,
    hf_home: Path,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001 - exact missing tool is reported.
        raise RuntimeError(
            "Missing download tool: huggingface_hub is required for this acquisition runner."
        ) from exc

    os.environ["HF_HOME"] = str(hf_home)
    acquired = []
    for item in files:
        rel = str(item["rfilename"])
        local_path = hf_hub_download(
            repo_id=source_repo,
            filename=rel,
            repo_type="model",
            local_dir=str(target_root),
            force_download=False,
            token=False,
        )
        path = Path(local_path)
        acquired.append(
            {
                "rfilename": rel,
                "path": str(path),
                "size": path.stat().st_size if path.exists() else None,
                "expected_size": item.get("size"),
            }
        )
    return acquired


Downloader = Callable[..., list[dict[str, Any]]]


def build_report(args: argparse.Namespace, downloader: Downloader = _download_required_files) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    risk_report_path = Path(args.risk_report)
    target_root = Path(args.target_root)
    hf_home = Path(args.hf_home)

    report: dict[str, Any] = {
        "evidence_label": "vlm_required_files_acquisition",
        "vlm_required_files_acquisition_passed": False,
        "decision": "stop",
        "ready_for_bounded_vlm_enabled_load_smoke_plan": False,
        "ready_for_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "task_local_gate_required": f"{DOWNLOAD_GATE}=1",
            "download_gate_set": _env_flag(DOWNLOAD_GATE),
            "downloads_performed": False,
            "download_source_restricted": True,
            "installs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "simulator_environment_created": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "forbidden_gates_set": [name for name in FORBIDDEN_GATES if _env_flag(name)],
        },
        "claims": {
            "standard_success_claimed": False,
            "benchmark_success_claimed": False,
            "counterfactual_robustness_claimed": False,
            "sota_claimed": False,
            "paper_grade_claim_made": False,
        },
        "risk_assessment": {
            "task": "Acquire required SmolVLM2 files for later VLM-enabled SmolVLA load-smoke planning",
            "command": "scripts\\112_acquire_vlm_required_files.ps1",
            "source_repo": args.source_repo,
            "source_url": f"https://huggingface.co/{args.source_repo}",
            "official_documented_source": args.source_repo == SOURCE_REPO,
            "expected_runtime_minutes": 30,
            "expected_ram_gb": 2,
            "expected_vram_gb": 0,
            "simulator_will_run": False,
            "rollout_will_run": False,
            "training_will_run": False,
            "model_load_will_run": False,
            "target_path": str(target_root),
            "cache_path": str(hf_home),
            "stop_condition": (
                "Stop if source differs, metadata risk did not pass, token/login/license/payment is required, "
                "size is unknown or above budget, disk-after budget is below 250GB, download tooling is missing, "
                "or any execution gate beyond ALLOW_DOWNLOADS is set."
            ),
            "fallback_plan": "Keep load_vlm_weights=false diagnostics and plan a cloud handoff or smaller metadata-only analysis.",
        },
        "paths": {
            "risk_report": str(risk_report_path),
            "target_root": str(target_root),
            "hf_home": str(hf_home),
        },
        "source": {
            "repo_id": args.source_repo,
            "expected_repo_id": SOURCE_REPO,
            "token_login_license_payment_required": None,
        },
        "files": {
            "required": [],
            "present_before": [],
            "acquired": [],
            "present_after": [],
        },
        "runtime": {
            "started_at_unix": started,
            "elapsed_sec": None,
            "target_size_before_gb": _target_size_gb(target_root),
            "target_size_after_gb": None,
        },
        "error": None,
        "recommended_next_step": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["risk_assessment"]["decision"] = "stop"
        report["risk_assessment"]["reason"] = reason
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        report["runtime"]["elapsed_sec"] = round(time.monotonic() - started, 3)
        report["runtime"]["target_size_after_gb"] = _target_size_gb(target_root)
        return report, code

    if not report["policy"]["download_gate_set"]:
        return block(f"{DOWNLOAD_GATE}=1 is required only for this bounded acquisition task.", 2)
    if report["policy"]["forbidden_gates_set"]:
        return block("Forbidden gate(s) set: " + ", ".join(report["policy"]["forbidden_gates_set"]), 3)
    if args.source_repo != SOURCE_REPO:
        return block(f"Unexpected source repo: {args.source_repo}", 4)
    if not risk_report_path.exists():
        return block(f"VLM risk report is missing: {risk_report_path}", 5)

    try:
        risk = _read_json(risk_report_path)
        report["risk_report_summary"] = {
            "decision": risk.get("decision"),
            "ready_for_vlm_weight_acquisition_plan": risk.get("ready_for_vlm_weight_acquisition_plan"),
            "source": risk.get("source"),
        }
        source = risk.get("source") or {}
        report["source"]["token_login_license_payment_required"] = bool(
            source.get("token_login_license_payment_required")
        )
        files = required_files_from_risk_report(risk)
        expected_size = _expected_size_gb(files)
        free_before = _disk_free_gb(target_root)
        free_after = round(free_before - expected_size, 3)
        report["files"]["required"] = files
        report["files"]["present_before"] = _existing_files(target_root, files)
        report["risk_assessment"].update(
            {
                "expected_new_disk_gb": expected_size,
                "current_free_disk_gb": free_before,
                "free_disk_after_estimate_gb": free_after,
                "max_expected_size_gb": MAX_EXPECTED_SIZE_GB,
                "min_free_after_gb": MIN_FREE_AFTER_GB,
            }
        )

        if risk.get("decision") != "proceed" or not risk.get("ready_for_vlm_weight_acquisition_plan"):
            return block("VLM risk report did not authorize acquisition.", 6)
        if not bool(source.get("official_source")):
            return block("Risk report does not mark the source as official.", 7)
        if source.get("repo_id") != SOURCE_REPO:
            return block("Risk report source repo does not match the acquisition source.", 8)
        if source.get("private") or source.get("gated") or report["source"]["token_login_license_payment_required"]:
            return block("VLM source requires token/login/license/payment or is gated/private.", 9)
        if not _all_sizes_known(files):
            return block("Required file size metadata is incomplete.", 10)
        if expected_size > MAX_EXPECTED_SIZE_GB:
            return block(f"Expected file size exceeds budget: {expected_size}GB", 11)
        if free_after < MIN_FREE_AFTER_GB:
            return block(f"Free disk after acquisition would be below {MIN_FREE_AFTER_GB}GB: {free_after}GB", 12)

        target_root.mkdir(parents=True, exist_ok=True)
        hf_home.mkdir(parents=True, exist_ok=True)
        acquired = downloader(source_repo=args.source_repo, files=files, target_root=target_root, hf_home=hf_home)
        report["files"]["acquired"] = acquired
        report["files"]["present_after"] = _existing_files(target_root, files)
        missing = sorted(
            set(str(item["rfilename"]) for item in files)
            - set(str(item["rfilename"]) for item in report["files"]["present_after"])
        )
        if missing:
            return block("Required files are still missing after acquisition: " + ", ".join(missing), 13)

        elapsed = time.monotonic() - started
        if elapsed > MAX_RUNTIME_SECONDS:
            return block("VLM required file acquisition exceeded the 30 minute runtime budget.", 14)

        report["policy"]["downloads_performed"] = True
        report["vlm_required_files_acquisition_passed"] = True
        report["decision"] = "acquisition_complete"
        report["ready_for_bounded_vlm_enabled_load_smoke_plan"] = True
        report["risk_assessment"]["decision"] = "proceed"
        report["risk_assessment"]["reason"] = (
            "Official public source, bounded file list, known size, no token/login/license/payment gate, "
            "and disk-after budget remained green."
        )
        report["runtime"]["elapsed_sec"] = round(elapsed, 3)
        report["runtime"]["target_size_after_gb"] = _target_size_gb(target_root)
        report["recommended_next_step"] = (
            "Plan a bounded VLM-enabled load smoke. Do not load VLM weights until that separate risk plan passes."
        )
        return report, 0
    except Exception as exc:  # noqa: BLE001 - exact acquisition failure matters.
        report["error"] = _compact_error(exc)
        reason = f"VLM required file acquisition failed: {type(exc).__name__}: {exc}"
        report["risk_assessment"]["decision"] = "stop"
        report["risk_assessment"]["reason"] = reason
        report["recommended_next_step"] = "Resolve the acquisition blocker before any VLM-enabled load smoke."
        report["runtime"]["elapsed_sec"] = round(time.monotonic() - started, 3)
        report["runtime"]["target_size_after_gb"] = _target_size_gb(target_root)
        return report, 15


def write_markdown(report: dict[str, Any], path: Path) -> None:
    risk = report.get("risk_assessment") or {}
    lines = [
        "# VLM Required Files Acquisition Report",
        "",
        f"- decision: {report.get('decision')}",
        f"- acquisition passed: {report.get('vlm_required_files_acquisition_passed')}",
        f"- source: {risk.get('source_repo')}",
        f"- target: {risk.get('target_path')}",
        f"- cache: {risk.get('cache_path')}",
        f"- expected new disk GB: {risk.get('expected_new_disk_gb')}",
        f"- free disk after estimate GB: {risk.get('free_disk_after_estimate_gb')}",
        f"- downloads performed: {(report.get('policy') or {}).get('downloads_performed')}",
        f"- heavy imports/model load/inference/training/rollout/OpenVLA: false",
        f"- ready for bounded VLM-enabled load-smoke plan: {report.get('ready_for_bounded_vlm_enabled_load_smoke_plan')}",
        "",
        "This report is acquisition/readiness evidence only. It is not a model-load result, rollout result, benchmark result, or paper-grade claim.",
        "",
        "## Recommended Next Step",
        "",
        str(report.get("recommended_next_step")),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--risk-report", default="reports/vlm_enabled_loading_risk_plan_report.json")
    parser.add_argument("--target-root", default=str(TARGET_ROOT))
    parser.add_argument("--hf-home", default=str(HF_HOME))
    parser.add_argument("--json-report", default="reports/vlm_required_files_acquisition_report.json")
    parser.add_argument("--markdown-report", default="reports/vlm_required_files_acquisition_report.md")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    json_report = Path(args.json_report)
    md_report = Path(args.markdown_report)
    json_report.parent.mkdir(parents=True, exist_ok=True)
    json_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
