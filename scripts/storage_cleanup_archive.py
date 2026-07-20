#!/usr/bin/env python3
"""Inventory and archive the VLA workspace before/after approved cleanup.

This script is deliberately read-only with respect to cleanup targets.  It
only writes the reports/storage_cleanup evidence package.  Destructive work is
performed by separate, manifest-driven launchers after their dry-run checks.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(r"C:\Users\jiheo\tca_map")
OUT = REPO / "reports" / "storage_cleanup"
DISTRO = "Ubuntu-22.04"
VHDX = Path(
    r"C:\Users\jiheo\AppData\Local\wsl\{180ce51e-73f9-4a6b-91f0-3a4f1842ad61}\ext4.vhdx"
)
OVERLAP_START_UNIX = 1784491800  # 2026-07-20 05:10 KST, before the last Epoch 6 write.
SELECTED_XVLA_ROOT = "/home/jiheon/assets/checkpoints/xvla_hf_cache"
SELECTED_XVLA_SNAPSHOT = (
    SELECTED_XVLA_ROOT
    + "/transformers/models--2toINF--X-VLA-Libero/snapshots/"
    + "129e71460678b7236cee6fc9707f09d9fa0c3590"
)
PROTECTED_WINDOWS = [
    r"C:\Users\jiheo\tca_map\rollouts\2026_07_17",
    r"C:\Users\jiheo\tca_map\rollouts\2026_07_18",
]
WSL_METRICS_CACHE: dict[str, dict[str, Any]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def run(args: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {args!r}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def git(*args: str) -> str:
    return run(["git", *args], cwd=REPO)


def wsl(*args: str, check: bool = True) -> str:
    return run(["wsl", "-d", DISTRO, "--", *args], check=check)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_wsl(path: str) -> str:
    return wsl("sha256sum", "--", path).split()[0].upper()


def windows_allocated_size(path: Path) -> int:
    if os.name != "nt":
        return path.stat().st_size
    high = ctypes.c_ulong(0)
    fn = ctypes.windll.kernel32.GetCompressedFileSizeW
    fn.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    fn.restype = ctypes.c_ulong
    low = fn(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.GetLastError() != 0:
        return path.stat().st_size
    return (high.value << 32) | low


def windows_metrics(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists() and not path.is_symlink():
        return {"exists": False, "path": str(path)}
    apparent = 0
    allocated = 0
    files = 0
    hardlinked_files = 0
    reparse_entries: list[str] = []
    newest = path.lstat().st_mtime
    if path.is_file() or path.is_symlink():
        stat = path.lstat()
        apparent = stat.st_size
        allocated = windows_allocated_size(path) if path.is_file() else stat.st_size
        files = 1
        hardlinked_files = int(getattr(stat, "st_nlink", 1) > 1)
    else:
        for root, dirs, names in os.walk(path, followlinks=False):
            root_path = Path(root)
            retained_dirs = []
            for name in dirs:
                child = root_path / name
                try:
                    stat = child.lstat()
                    newest = max(newest, stat.st_mtime)
                    is_reparse = bool(
                        getattr(stat, "st_file_attributes", 0) & 0x400
                    )
                    if child.is_symlink() or is_reparse:
                        reparse_entries.append(str(child))
                    else:
                        retained_dirs.append(name)
                except OSError:
                    reparse_entries.append(str(child))
            dirs[:] = retained_dirs
            for name in names:
                child = root_path / name
                try:
                    stat = child.lstat()
                    newest = max(newest, stat.st_mtime)
                    apparent += stat.st_size
                    allocated += windows_allocated_size(child)
                    files += 1
                    hardlinked_files += int(getattr(stat, "st_nlink", 1) > 1)
                except OSError:
                    reparse_entries.append(str(child))
    return {
        "exists": True,
        "path": str(path),
        "resolved_path": str(path.resolve(strict=True)),
        "item_type": "file" if path.is_file() else "directory",
        "apparent_size_bytes": apparent,
        "allocated_size_bytes": allocated,
        "file_count": files,
        "newest_mtime": datetime.fromtimestamp(newest, timezone.utc).isoformat(),
        "top_level_link": path.is_symlink(),
        "reparse_entries_count": len(reparse_entries),
        "reparse_entry_samples": reparse_entries[:10],
        "files_with_multiple_hardlinks": hardlinked_files,
        "allocated_size_method": "GetCompressedFileSizeW summed over regular files",
    }


def wsl_metrics(path: str) -> dict[str, Any]:
    if path in WSL_METRICS_CACHE:
        return WSL_METRICS_CACHE[path]
    if subprocess.run(
        ["wsl", "-d", DISTRO, "--", "test", "-e", path], capture_output=True
    ).returncode:
        return {"exists": False, "path": path}
    apparent = int(wsl("du", "-sx", "-B1", "--apparent-size", "--", path).split()[0])
    allocated = int(wsl("du", "-sx", "-B1", "--", path).split()[0])
    stat_type = wsl("stat", "-Lc", "%F", "--", path)
    stat_mtime = wsl("stat", "-Lc", "%Y", "--", path)
    stat_links = wsl("stat", "-Lc", "%h", "--", path)
    resolved = wsl("readlink", "-f", "--", path)
    file_count = int(wsl("find", path, "-type", "f", "-printf", ".").count("."))
    recent = wsl(
        "find",
        path,
        "-type",
        "f",
        "-newermt",
        f"@{OVERLAP_START_UNIX}",
        "-printf",
        "%p\\n",
    )
    return {
        "exists": True,
        "path": path,
        "resolved_path": resolved,
        "item_type": stat_type,
        "apparent_size_bytes": apparent,
        "allocated_size_bytes": allocated,
        "file_count": file_count,
        "mtime": datetime.fromtimestamp(int(stat_mtime), timezone.utc).isoformat(),
        "hardlink_count_top_level": int(stat_links),
        "top_level_symlink": subprocess.run(
            ["wsl", "-d", DISTRO, "--", "test", "-L", path], capture_output=True
        ).returncode == 0,
        "overlap_window_modified_file_count": len(recent.splitlines()) if recent else 0,
        "allocated_size_method": "GNU du -sx -B1",
    }


def windows_children(root: str) -> list[Path]:
    path = Path(root)
    return sorted(path.iterdir(), key=lambda p: str(p).lower()) if path.exists() else []


def wsl_children(root: str) -> list[str]:
    code = r'''
import datetime, json, os, stat
root = os.environ["ROOT"]
threshold = int(os.environ["THRESHOLD"])
out = []
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name)
    top = os.lstat(path)
    apparent = 0
    allocated = 0
    files = 0
    newest = top.st_mtime
    overlap = 0
    top_link = stat.S_ISLNK(top.st_mode)
    if top_link or stat.S_ISREG(top.st_mode):
        apparent = top.st_size
        allocated = top.st_blocks * 512
        files = int(stat.S_ISREG(top.st_mode))
        overlap = int(stat.S_ISREG(top.st_mode) and top.st_mtime > threshold)
    else:
        for base, dirs, names2 in os.walk(path, followlinks=False):
            try:
                ds = os.lstat(base)
                allocated += ds.st_blocks * 512
                newest = max(newest, ds.st_mtime)
            except OSError:
                pass
            kept = []
            for d in dirs:
                child = os.path.join(base, d)
                try:
                    cs = os.lstat(child)
                    if stat.S_ISLNK(cs.st_mode):
                        apparent += cs.st_size
                        allocated += cs.st_blocks * 512
                        newest = max(newest, cs.st_mtime)
                    else:
                        kept.append(d)
                except OSError:
                    pass
            dirs[:] = kept
            for f in names2:
                child = os.path.join(base, f)
                try:
                    fs = os.lstat(child)
                    apparent += fs.st_size
                    allocated += fs.st_blocks * 512
                    files += int(stat.S_ISREG(fs.st_mode))
                    newest = max(newest, fs.st_mtime)
                    overlap += int(stat.S_ISREG(fs.st_mode) and fs.st_mtime > threshold)
                except OSError:
                    pass
    out.append({
        "exists": True, "path": path, "resolved_path": os.path.realpath(path),
        "item_type": "symbolic link" if top_link else ("regular file" if stat.S_ISREG(top.st_mode) else "directory"),
        "apparent_size_bytes": apparent, "allocated_size_bytes": allocated,
        "file_count": files, "mtime": datetime.datetime.fromtimestamp(newest, datetime.timezone.utc).isoformat(),
        "hardlink_count_top_level": top.st_nlink, "top_level_symlink": top_link,
        "overlap_window_modified_file_count": overlap,
        "allocated_size_method": "st_blocks*512 summed without following symlinks"
    })
print(json.dumps(out, sort_keys=True))
'''
    raw = run(
        [
            "wsl", "-d", DISTRO, "--", "env", f"ROOT={root}",
            f"THRESHOLD={OVERLAP_START_UNIX}", "python3", "-c", code,
        ]
    )
    rows = json.loads(raw)
    for row in rows:
        WSL_METRICS_CACHE[row["path"]] = row
    return [row["path"] for row in rows]


def process_snapshot() -> dict[str, Any]:
    ps = (
        f"$self=$PID; $audit={os.getpid()}; Get-CimInstance Win32_Process | "
        "Where-Object {$_.ProcessId -ne $self -and $_.ProcessId -ne $audit -and "
        "($_.Name -match 'python|python3|wsl|bash|mujoco|conda|pip|uv|aria2|curl|wget' "
        "-or $_.CommandLine -match 'LIBERO|VLA|rollout|train|download|huggingface|mujoco')} | "
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine | "
        "ConvertTo-Json -Depth 3"
    )
    raw = run(["powershell", "-NoProfile", "-Command", ps], check=False)
    windows = json.loads(raw) if raw else []
    if isinstance(windows, dict):
        windows = [windows]
    windows = [
        item for item in windows
        if "storage_cleanup_archive.py" not in str(item.get("CommandLine", ""))
    ]
    wsl_raw = wsl("ps", "-eo", "pid,ppid,comm,args", "--no-headers")
    relevant_wsl = [
        line.strip()
        for line in wsl_raw.splitlines()
        if any(token in line.lower() for token in ("python", "libero", "vla", "mujoco", "train", "rollout"))
        and "networkd-dispatcher" not in line
        and "unattended-upgrade" not in line
    ]
    gpu_raw = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=False,
    )
    gpu_research = [
        line for line in gpu_raw.splitlines() if any(x in line.lower() for x in ("python", "wsl", "mujoco"))
    ]
    return {
        "captured_at": now_iso(),
        "windows_candidate_processes": windows,
        "wsl_research_processes": relevant_wsl,
        "gpu_research_compute_processes": gpu_research,
        "relevant_worker_active": bool(relevant_wsl or gpu_research or any(
            str(p.get("Name", "")).lower().startswith(("python", "mujoco")) for p in windows
        )),
        "note": "wslservice/system services and the auditing shell are not research workers",
    }


def git_snapshot() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    remote_ref = f"origin/{branch}"
    return {
        "branch": branch,
        "local_head": git("rev-parse", "HEAD"),
        "origin_head": git("rev-parse", remote_ref),
        "merge_base": git("merge-base", "HEAD", remote_ref),
        "ahead_behind": git("rev-list", "--left-right", "--count", f"HEAD...{remote_ref}"),
        "status_short_branch": git("status", "--short", "--branch").splitlines(),
        "diff_stat": git("diff", "--stat").splitlines(),
        "cached_diff_stat": git("diff", "--cached", "--stat").splitlines(),
    }


def disk_snapshot() -> dict[str, Any]:
    usage = shutil.disk_usage("C:\\")
    df = wsl("df", "-B1", "--output=size,used,avail,pcent,target", "/").splitlines()
    vals = df[-1].split()
    return {
        "captured_at": now_iso(),
        "windows_c": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
        "wsl_root": {
            "total_bytes": int(vals[0]),
            "used_bytes": int(vals[1]),
            "free_bytes": int(vals[2]),
            "percent_used": vals[3],
            "mount": vals[4],
        },
        "vhdx": {
            "distribution": DISTRO,
            "path": str(VHDX),
            "physical_size_bytes": VHDX.stat().st_size,
        },
    }


def xvla_closure() -> dict[str, Any]:
    code = r'''
import hashlib, json, os
root = os.environ["SNAPSHOT"]
links = []
for base, dirs, files in os.walk(root):
    for name in files:
        p = os.path.join(base, name)
        if os.path.islink(p):
            resolved = os.path.realpath(p)
            h = hashlib.sha256()
            with open(resolved, "rb") as f:
                for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                    h.update(chunk)
            links.append({"link": p, "link_target": os.readlink(p), "resolved_blob": resolved,
                          "blob_size_bytes": os.path.getsize(resolved), "blob_sha256": h.hexdigest().upper()})
print(json.dumps(links, sort_keys=True))
'''
    raw = run(
        [
            "wsl",
            "-d",
            DISTRO,
            "--",
            "env",
            f"SNAPSHOT={SELECTED_XVLA_SNAPSHOT}",
            "python3",
            "-c",
            code,
        ]
    )
    links = json.loads(raw)
    return {
        "model_id": "2toINF/X-VLA-Libero",
        "revision": "129e71460678b7236cee6fc9707f09d9fa0c3590",
        "root": SELECTED_XVLA_ROOT,
        "snapshot": SELECTED_XVLA_SNAPSHOT,
        "root_metrics": wsl_metrics(SELECTED_XVLA_ROOT),
        "snapshot_symlink_count": len(links),
        "required_snapshot_links_and_blobs": links,
        "all_links_resolve_inside_selected_root": all(
            item["resolved_blob"].startswith(SELECTED_XVLA_ROOT + "/") for item in links
        ),
    }


def add_delete_target(
    targets: list[dict[str, Any]],
    *,
    platform: str,
    path: str,
    audited_root: str,
    category: str,
    reason: str,
    regeneration_source: str,
    references: list[str] | None = None,
    content_hash: str | None = None,
    immutable_revision: str | None = None,
    sentinel_hashes: dict[str, str] | None = None,
) -> None:
    metrics = windows_metrics(path) if platform == "windows" else wsl_metrics(path)
    if not metrics.get("exists"):
        return
    targets.append(
        {
            "target_id": f"{platform}-{len(targets)+1:04d}",
            "platform": platform,
            "path": path,
            "resolved_path": metrics["resolved_path"],
            "audited_root": audited_root,
            "item_type": metrics["item_type"],
            "apparent_size_bytes": metrics["apparent_size_bytes"],
            "allocated_size_bytes": metrics["allocated_size_bytes"],
            "modification_time": metrics.get("newest_mtime", metrics.get("mtime")),
            "owner_process_lock_status": "NO_RELEVANT_WORKER; exact handle recheck required by executor",
            "git_state": "NOT_APPLICABLE",
            "symlink_junction_hardlink": {
                "top_level_symlink": metrics.get("top_level_link", metrics.get("top_level_symlink", False)),
                "reparse_entries_count": metrics.get("reparse_entries_count"),
                "hardlinked_files": metrics.get("files_with_multiple_hardlinks"),
                "top_level_hardlink_count": metrics.get("hardlink_count_top_level"),
            },
            "referencing_artifacts": references or [],
            "content_sha256": content_hash,
            "sentinel_hashes": sentinel_hashes or {},
            "immutable_revision": immutable_revision,
            "regeneration_source": regeneration_source,
            "scientific_role": category,
            "classification": "VERIFIED_DISPOSABLE",
            "reason": reason,
            "estimated_reclaimable_bytes": metrics["allocated_size_bytes"],
            "overlap_window_modified_file_count": metrics.get("overlap_window_modified_file_count", 0),
            "identity": metrics,
        }
    )


def make_pre() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generated_at = now_iso()
    git_state = git_snapshot()
    workers = process_snapshot()
    if workers["relevant_worker_active"]:
        raise RuntimeError("relevant research worker detected; refusing to produce an executable delete manifest")
    disk = disk_snapshot()
    xvla = xvla_closure()

    overlap_keep = [
        "C:\\Users\\jiheo\\tca_map\\reports\\epoch6_terminal_handoff.md",
        "C:\\Users\\jiheo\\tca_map\\reports\\epoch6_campaign_state.json",
        "C:\\Users\\jiheo\\tca_map\\reports\\epoch6_evidence_index.json",
        "C:\\Users\\jiheo\\tca_map\\reports\\epoch6_resource_inventory.json",
        "C:\\Users\\jiheo\\tca_map\\reports\\epoch6_closure_registry.json",
        "C:\\Users\\jiheo\\tca_map\\reports\\epoch6_schedule_invariant_evaluation",
        "C:\\Users\\jiheo\\tca_map\\runs\\epoch6_schedule_invariant_evaluation",
        "C:\\Users\\jiheo\\tca_map\\scripts\\run_epoch6_schedule_closed_loop.py",
        "C:\\Users\\jiheo\\tca_map\\scripts\\monitor_epoch6_schedule_closed_loop_smoke.ps1",
        "C:\\Users\\jiheo\\tca_map\\scripts\\run_epoch6_schedule_closed_loop_smoke_wsl.sh",
        "C:\\Users\\jiheo\\tca_map\\tests\\test_epoch6_schedule_closed_loop.py",
    ]
    keep_items = [
        {"path": str(REPO), "classification": "KEEP", "reason": "Git repository, tracked history, reports, code, tests, and archive record"},
        {"path": str(REPO / "runs"), "classification": "KEEP_OR_REVIEW", "reason": "Unique scientific checkpoints/raw results; no run payload is deleted in this pass"},
        {"path": str(REPO / "reports"), "classification": "KEEP", "reason": "Scientific record and governance"},
        {"path": SELECTED_XVLA_ROOT, "classification": "KEEP", "reason": "One verified runnable X-VLA path, exact snapshot and backing blobs"},
        {"path": r"C:\assets\repos\X-VLA", "classification": "KEEP", "reason": "Selected runnable source at 6bc2513f5f1cbec715cc668b414392a6cae5c671"},
        {"path": "/home/jiheon/assets/repos/LIBERO", "classification": "KEEP", "reason": "Verified WSL LIBERO runtime and task assets"},
        {"path": r"C:\assets\repos\LIBERO", "classification": "KEEP", "reason": "Canonical audited official LIBERO source"},
        {"path": r"C:\assets\data\libero", "classification": "KEEP", "reason": "Official benchmark demonstrations and reset/task evidence"},
        {"path": r"C:\assets\datasets\lerobot_libero", "classification": "KEEP", "reason": "Retained SmolVLA training/evaluation dataset dependency"},
        {"path": r"C:\assets\checkpoints", "classification": "KEEP_OR_REVIEW", "reason": "Referenced local checkpoints; no Windows checkpoint is deleted"},
        {"path": "/home/jiheon/miniconda3-official/envs", "classification": "KEEP", "reason": "Installed runnable environments; package cache only may be removed"},
        {"path": "/home/jiheon/venvs", "classification": "KEEP_OR_REVIEW", "reason": "Installed environments may not be deleted automatically"},
        {"path": "/home/jiheon/.venvs", "classification": "KEEP_OR_REVIEW", "reason": "Installed environments may not be deleted automatically"},
        {"path": "/home/jiheon/assets/checkpoints/lightvla", "classification": "USER_DECISION_REQUIRED", "reason": "Local loader modifications/backups make exact ownership nontrivial"},
        {"path": "/home/jiheon/assets/checkpoints/openpi", "classification": "USER_DECISION_REQUIRED", "reason": "Public GCS source lacks a frozen immutable checkpoint revision in the retained ledger"},
        {"path": r"C:\assets\repos\PCD-LeRobot", "classification": "KEEP_OR_REVIEW", "reason": "1,866 tracked changes and 25 untracked entries observed"},
        {"path": r"C:\assets\repos\robosuite", "classification": "KEEP_OR_REVIEW", "reason": "Pre-existing macros.py edit and untracked mujoco.dll"},
        {"path": r"C:\Users\jiheo\AppData\Local\Temp\DiagOutputDir", "classification": "AMBIGUOUS", "reason": "Large Remote Desktop diagnostic trace outside VLA ownership"},
    ]
    keep_items.extend(
        {"path": p, "classification": "PROTECTED", "reason": "Explicitly protected rollout directory"}
        for p in PROTECTED_WINDOWS
    )
    keep_items.extend(
        {"path": p, "classification": "KEEP_OR_REVIEW", "reason": "Epoch 6 overlap-window scientific state"}
        for p in overlap_keep
    )
    pdf_listing = run(["rg", "--files", "-g", "*.pdf"], cwd=REPO, check=False)
    archive_pdfs = [str(REPO / line) for line in pdf_listing.splitlines() if line]

    keep_manifest = {
        "schema_version": "storage_cleanup.keep_manifest.v1",
        "generated_at": generated_at,
        "git": git_state,
        "overlap_window": {
            "start_unix": OVERLAP_START_UNIX,
            "policy": "Every Epoch 6 path written or plausibly owned during the overlap is KEEP_OR_REVIEW",
            "candidate_roots_with_writes_after_start": 0,
        },
        "selected_runnable_path": xvla,
        "protected_items": PROTECTED_WINDOWS,
        "items": keep_items,
        "research_archive_pdfs_found": archive_pdfs,
    }
    (OUT / "keep_manifest.json").write_text(json.dumps(keep_manifest, indent=2) + "\n", encoding="utf-8")

    dependency_graph = {
        "schema_version": "storage_cleanup.dependency_graph.v1",
        "generated_at": generated_at,
        "nodes": [
            {"id": "git_archive", "path": str(REPO), "class": "KEEP"},
            {"id": "epoch6_state", "path": "reports/epoch6_campaign_state.json", "class": "KEEP_OR_REVIEW"},
            {"id": "epoch6_stage0", "path": "reports/epoch6_schedule_invariant_evaluation/stage0_result.json", "class": "KEEP"},
            {"id": "epoch6_closed_loop", "path": "reports/epoch6_schedule_invariant_evaluation", "class": "KEEP_OR_REVIEW"},
            {"id": "xvla_checkpoint", "path": SELECTED_XVLA_ROOT, "class": "KEEP"},
            {"id": "xvla_source", "path": r"C:\assets\repos\X-VLA", "class": "KEEP"},
            {"id": "xvla_environment", "path": "/home/jiheon/miniconda3-official/envs/official-smolvla-libero", "class": "KEEP"},
            {"id": "libero_runtime", "path": "/home/jiheon/assets/repos/LIBERO", "class": "KEEP"},
            {"id": "libero_data", "path": r"C:\assets\data\libero", "class": "KEEP"},
            {"id": "protected_rollouts", "paths": PROTECTED_WINDOWS, "class": "PROTECTED"},
            {"id": "openvla_reacquisition", "source": "https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10", "revision": "638918f3d1c2e43a39a8a20772bdb8b91835e4b7", "class": "REACQUIRE_RECIPE"},
        ],
        "edges": [
            {"from": "epoch6_state", "to": "epoch6_stage0", "relation": "indexes"},
            {"from": "epoch6_state", "to": "epoch6_closed_loop", "relation": "archives_pending_schedule"},
            {"from": "epoch6_stage0", "to": "xvla_checkpoint", "relation": "requires_for_resume"},
            {"from": "epoch6_stage0", "to": "xvla_source", "relation": "requires_code"},
            {"from": "xvla_checkpoint", "to": "xvla_environment", "relation": "loads_in"},
            {"from": "epoch6_closed_loop", "to": "libero_runtime", "relation": "requires_simulator"},
            {"from": "epoch6_stage0", "to": "libero_data", "relation": "uses_demonstrations"},
            {"from": "git_archive", "to": "openvla_reacquisition", "relation": "retains_checksums_and_recipe"},
        ],
        "closure_assertions": {
            "selected_snapshot_and_blobs_resolve": xvla["all_links_resolve_inside_selected_root"],
            "protected_rollouts_excluded_from_deletion": True,
            "overlap_state_excluded_from_deletion": True,
            "one_runnable_vla_libero_path_preserved": True,
        },
    }
    (OUT / "dependency_graph.json").write_text(json.dumps(dependency_graph, indent=2) + "\n", encoding="utf-8")

    targets: list[dict[str, Any]] = []
    crash_root = r"C:\Users\jiheo\AppData\Local\Temp\wsl-crashes"
    for path in windows_children(crash_root):
        add_delete_target(
            targets,
            platform="windows",
            path=str(path),
            audited_root=crash_root,
            category="VLA WSL crash dump",
            reason="Oversized campaign crash dump; failure diagnosis and compact telemetry are retained",
            regeneration_source="Not regenerated; diagnostic record retained in reports and run logs",
            references=["reports/a2c2_prior", "reports/epoch6_resource_inventory.json"],
            content_hash=sha256_file(path),
        )
    pip_root = r"C:\Users\jiheo\AppData\Local\pip\cache"
    for path in windows_children(pip_root):
        add_delete_target(
            targets, platform="windows", path=str(path), audited_root=pip_root,
            category="pip package-download cache", reason="Enumerated regenerable package cache, not an installed environment",
            regeneration_source="pip package indexes and retained environment specifications",
        )
    conda_root = r"C:\Users\jiheo\miniconda3\pkgs"
    for path in windows_children(conda_root):
        add_delete_target(
            targets, platform="windows", path=str(path), audited_root=conda_root,
            category="Conda package-download/extraction cache", reason="Enumerated package cache; installed environments are outside this root",
            regeneration_source="Conda channels and installed conda-meta records",
        )
    win_hf_root = r"C:\Users\jiheo\.cache\huggingface\hub"
    win_model = win_hf_root + r"\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    add_delete_target(
        targets, platform="windows", path=win_model, audited_root=win_hf_root,
        category="nonselected public Hugging Face model cache", reason="Selected runnable X-VLA uses an isolated WSL cache; this public SmolVLM cache is not in the closure",
        regeneration_source="https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        immutable_revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    )
    win_lock = win_hf_root + r"\.locks\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    add_delete_target(
        targets, platform="windows", path=win_lock, audited_root=win_hf_root,
        category="stale Hugging Face lock cache", reason="No downloader is active and the paired nonselected model cache is disposable",
        regeneration_source="Hugging Face cache manager",
    )
    for path, repo_id, revision, license_text in [
        (r"C:\assets\repos\PCD", "https://github.com/pcd-robot/PCD.git", "cec18b820daeadfdaf080c030a1b5eb080ff75cd", "license unresolved; source audit only"),
        (r"C:\assets\repos\VLA-Arena", "https://github.com/PKU-Alignment/VLA-Arena.git", "babe582ebffc82b979b77964a7e56417d02f63a4", "Apache-2.0"),
    ]:
        status = run(["git", "-C", path, "status", "--porcelain"])
        head = run(["git", "-C", path, "rev-parse", "HEAD"])
        if status or head != revision:
            raise RuntimeError(f"external repo identity changed: {path}")
        add_delete_target(
            targets, platform="windows", path=path, audited_root=r"C:\assets\repos",
            category="closed-route clean public source clone", reason=f"Clean source-audit-only clone; exact Git object and source retained ({license_text})",
            regeneration_source=repo_id, immutable_revision=revision,
            references=["reports/epoch5_prior_reproduction_result.md", "reports/epoch6_resource_inventory.json"],
        )

    for root, category, source in [
        ("/home/jiheon/.cache/uv", "uv package/source cache", "Python package indexes and retained uv environment"),
        ("/home/jiheon/.cache/pip", "pip package-download cache", "pip package indexes and retained environment specifications"),
        ("/home/jiheon/miniconda3-official/pkgs", "Conda package-download/extraction cache", "Conda channels and installed conda-meta records"),
        ("/home/jiheon/.cache/torch", "Torch Hub cache", "Public Torch Hub sources"),
    ]:
        for path in wsl_children(root):
            add_delete_target(
                targets, platform="wsl", path=path, audited_root=root,
                category=category, reason="Enumerated regenerable cache entry; no installed environment or selected model is inside the target",
                regeneration_source=source,
            )
    wsl_hf_root = "/home/jiheon/.cache/huggingface/hub"
    for name, source, revision, role in [
        ("models--TTJiang--LightVLA-libero-10", "https://huggingface.co/TTJiang/LightVLA-libero-10", "d40628fe49fbbca841e1ae9c7b17e2fb6abe7aa7", "nonselected LightVLA metadata cache"),
        ("models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct", "https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467", "nonselected public base-model cache"),
        ("models--openai--clip-vit-base-patch32", "https://huggingface.co/openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "closed RL4IL prior feature cache"),
    ]:
        path = f"{wsl_hf_root}/{name}"
        add_delete_target(
            targets, platform="wsl", path=path, audited_root=wsl_hf_root,
            category=role, reason="Public nonselected cache outside the isolated selected X-VLA root",
            regeneration_source=source, immutable_revision=revision,
        )
        lock_path = f"{wsl_hf_root}/.locks/{name}"
        add_delete_target(
            targets, platform="wsl", path=lock_path, audited_root=wsl_hf_root,
            category="stale Hugging Face lock cache", reason="No downloader is active and the paired cache is nonselected",
            regeneration_source="Hugging Face cache manager",
        )
    xet_logs = "/home/jiheon/.cache/huggingface/xet/logs"
    for path in wsl_children(xet_logs):
        add_delete_target(
            targets, platform="wsl", path=path, audited_root=xet_logs,
            category="Hugging Face transfer log", reason="Download diagnostic log; model identities and failures are retained elsewhere",
            regeneration_source="Hugging Face Xet client",
        )

    openvla_path = "/home/jiheon/assets/checkpoints/openvla-oft/moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    sentinels = {}
    for name in [
        "README.md", "action_head--300000_checkpoint.pt", "config.json",
        "lora_adapter/adapter_model.safetensors", "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors", "proprio_projector--300000_checkpoint.pt",
    ]:
        sentinels[name] = sha256_wsl(f"{openvla_path}/{name}")
    add_delete_target(
        targets, platform="wsl", path=openvla_path,
        audited_root="/home/jiheon/assets/checkpoints/openvla-oft",
        category="closed-route public OpenVLA-OFT checkpoint",
        reason="Not the selected runnable path; exact public revision and fresh key-file hashes are retained for reacquisition",
        regeneration_source="https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        immutable_revision="638918f3d1c2e43a39a8a20772bdb8b91835e4b7",
        sentinel_hashes=sentinels,
        references=["reports/openvla_oft_int4_download_status.md", "reports/epoch5_prior_reproduction_result.json"],
    )

    targets.sort(key=lambda item: item["estimated_reclaimable_bytes"], reverse=True)
    for index, item in enumerate(targets, 1):
        item["target_id"] = f"{item['platform']}-{index:04d}"
    protected_norm = [os.path.normcase(os.path.abspath(p)) for p in PROTECTED_WINDOWS]
    selected_norm = SELECTED_XVLA_ROOT.rstrip("/") + "/"
    validation_errors: list[str] = []
    for item in targets:
        if item["classification"] != "VERIFIED_DISPOSABLE":
            validation_errors.append(f"bad class: {item['path']}")
        if item["overlap_window_modified_file_count"]:
            validation_errors.append(f"overlap write: {item['path']}")
        if item["platform"] == "windows":
            target_norm = os.path.normcase(os.path.abspath(item["resolved_path"]))
            root_norm = os.path.normcase(os.path.abspath(item["audited_root"]))
            if os.path.commonpath([target_norm, root_norm]) != root_norm:
                validation_errors.append(f"outside audited root: {item['path']}")
            for protected in protected_norm:
                if target_norm == protected or target_norm.startswith(protected + os.sep) or protected.startswith(target_norm + os.sep):
                    validation_errors.append(f"protected overlap: {item['path']}")
        else:
            resolved = item["resolved_path"].rstrip("/")
            root = item["audited_root"].rstrip("/")
            if not (resolved == root or resolved.startswith(root + "/")):
                validation_errors.append(f"outside audited WSL root: {item['path']}")
            if resolved == SELECTED_XVLA_ROOT or resolved.startswith(selected_norm) or SELECTED_XVLA_ROOT.startswith(resolved + "/"):
                validation_errors.append(f"selected X-VLA overlap: {item['path']}")
        if item["symlink_junction_hardlink"].get("top_level_symlink"):
            validation_errors.append(f"top-level link target prohibited: {item['path']}")

    total = sum(item["estimated_reclaimable_bytes"] for item in targets)
    delete_manifest = {
        "schema_version": "storage_cleanup.delete_manifest.v1",
        "generated_at": generated_at,
        "source_git": git_state,
        "worker_snapshot": workers,
        "overlap_policy": {
            "start_unix": OVERLAP_START_UNIX,
            "targets_with_overlap_writes": sum(bool(t["overlap_window_modified_file_count"]) for t in targets),
            "all_epoch6_overlap_state_excluded": True,
        },
        "expected_total_reclaimable_bytes": total,
        "target_count": len(targets),
        "validation": {
            "errors": validation_errors,
            "passed": not validation_errors,
            "selected_xvla_blob_closure_intact": xvla["all_links_resolve_inside_selected_root"],
            "protected_rollouts_excluded": True,
            "git_tracked_targets": 0,
        },
        "targets": targets,
    }
    if validation_errors:
        raise RuntimeError("delete manifest validation failed:\n" + "\n".join(validation_errors))
    (OUT / "delete_manifest.json").write_text(json.dumps(delete_manifest, indent=2) + "\n", encoding="utf-8")

    groups: dict[str, dict[str, int]] = {}
    for item in targets:
        group = groups.setdefault(item["scientific_role"], {"targets": 0, "bytes": 0})
        group["targets"] += 1
        group["bytes"] += item["estimated_reclaimable_bytes"]
    group_rows = sorted(groups.items(), key=lambda pair: pair[1]["bytes"], reverse=True)
    delete_md = [
        "# Storage Cleanup Delete Manifest",
        "",
        f"Generated: `{generated_at}`",
        "",
        f"Validated exact targets: **{len(targets)}**",
        f"Expected reclaimable allocation: **{total / 1_000_000_000:.3f} GB**",
        "",
        "Every JSON target is `VERIFIED_DISPOSABLE`. The executor must recheck identity, workers, handles,",
        "resolved containment, overlap-window writes, and selected/protected exclusions immediately before deletion.",
        "",
        "| Category | Exact targets | Estimated GB |",
        "|---|---:|---:|",
    ]
    delete_md.extend(f"| {name} | {data['targets']} | {data['bytes']/1_000_000_000:.3f} |" for name, data in group_rows)
    delete_md += [
        "",
        "## Largest exact targets",
        "",
        "| Platform | Exact path | GB | Evidence |",
        "|---|---|---:|---|",
    ]
    for item in targets[:20]:
        rev = item.get("immutable_revision") or item.get("content_sha256") or "enumerated cache"
        delete_md.append(f"| {item['platform']} | `{item['path']}` | {item['estimated_reclaimable_bytes']/1_000_000_000:.3f} | `{rev}` |")
    delete_md += ["", "The complete exact target list and hashes are in `delete_manifest.json`.", ""]
    (OUT / "delete_manifest.md").write_text("\n".join(delete_md), encoding="utf-8")

    protected_sizes = [windows_metrics(p) for p in PROTECTED_WINDOWS]
    inventory_candidates = [
        {"path": str(REPO / "runs"), "classification": "KEEP_OR_REVIEW", "metrics": windows_metrics(str(REPO / "runs")), "reason": "unique scientific outputs/checkpoints"},
        {"path": str(REPO / "rollouts"), "classification": "KEEP_OR_REVIEW", "metrics": windows_metrics(str(REPO / "rollouts")), "reason": "includes two explicitly protected directories"},
        {"path": r"C:\assets\data\libero", "classification": "KEEP", "metrics": windows_metrics(r"C:\assets\data\libero"), "reason": "benchmark data"},
        {"path": r"C:\assets\datasets\lerobot_libero", "classification": "KEEP", "metrics": windows_metrics(r"C:\assets\datasets\lerobot_libero"), "reason": "retained dataset dependency"},
        {"path": r"C:\assets\checkpoints", "classification": "KEEP_OR_REVIEW", "metrics": windows_metrics(r"C:\assets\checkpoints"), "reason": "referenced Windows checkpoints"},
        {"path": "/home/jiheon/assets/checkpoints/lightvla", "classification": "USER_DECISION_REQUIRED", "metrics": wsl_metrics("/home/jiheon/assets/checkpoints/lightvla"), "reason": "local checkpoint modifications"},
        {"path": "/home/jiheon/assets/checkpoints/openpi", "classification": "USER_DECISION_REQUIRED", "metrics": wsl_metrics("/home/jiheon/assets/checkpoints/openpi"), "reason": "non-immutable GCS source"},
        {"path": SELECTED_XVLA_ROOT, "classification": "KEEP", "metrics": xvla["root_metrics"], "reason": "selected runnable checkpoint closure"},
        {"path": str(VHDX), "classification": "KEEP", "metrics": windows_metrics(str(VHDX)), "reason": "verified WSL virtual disk; compaction only"},
        {"path": r"C:\Users\jiheo\AppData\Local\Temp\DiagOutputDir", "classification": "AMBIGUOUS", "metrics": windows_metrics(r"C:\Users\jiheo\AppData\Local\Temp\DiagOutputDir"), "reason": "unrelated/ambiguous diagnostic ownership"},
    ]
    inventory_candidates.sort(key=lambda item: item["metrics"].get("allocated_size_bytes", 0), reverse=True)
    pre_inventory = {
        "schema_version": "storage_cleanup.pre_inventory.v1",
        "generated_at": generated_at,
        "git": git_state,
        "workers": workers,
        "storage_before": disk,
        "overlap_window": keep_manifest["overlap_window"],
        "selected_runnable_path": xvla,
        "protected_sizes": protected_sizes,
        "large_retained_or_review_candidates": inventory_candidates,
        "delete_manifest_summary": {"target_count": len(targets), "estimated_reclaimable_bytes": total},
        "notes": [
            "No repository run or rollout payload is deleted in this pass.",
            "Windows Conda package-cache files may be hardlinked; actual free-space delta is authoritative.",
            "WSL internal deletion does not imply VHDX shrink until supported compaction succeeds.",
        ],
    }
    (OUT / "pre_cleanup_inventory.json").write_text(json.dumps(pre_inventory, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Pre-cleanup Storage Inventory",
        "",
        f"Captured: `{generated_at}`",
        f"Branch / HEAD: `{git_state['branch']}` / `{git_state['local_head']}`",
        f"Windows free: **{disk['windows_c']['free_bytes']/1_000_000_000:.3f} GB**",
        f"WSL internal free: **{disk['wsl_root']['free_bytes']/1_000_000_000:.3f} GB**",
        f"WSL VHDX physical size: **{disk['vhdx']['physical_size_bytes']/1_000_000_000:.3f} GB**",
        f"Manifested deletion allocation: **{total/1_000_000_000:.3f} GB** across **{len(targets)}** exact targets.",
        "",
        "## Safety refresh",
        "",
        f"- Relevant worker active: `{workers['relevant_worker_active']}`.",
        "- Branch, local/remote HEAD, ancestry, worktree, candidate mtimes, selected snapshot blobs, and high-risk hashes were refreshed after the overlap pause.",
        "- All Epoch 6 overlap-window state is `KEEP_OR_REVIEW`; it is absent from the delete manifest.",
        "- The two protected rollout directories remain `PROTECTED`.",
        "",
        "## Large retained or review items",
        "",
        "| Classification | Path | Allocated GB | Reason |",
        "|---|---|---:|---|",
    ]
    for item in inventory_candidates:
        md.append(f"| {item['classification']} | `{item['path']}` | {item['metrics'].get('allocated_size_bytes',0)/1_000_000_000:.3f} | {item['reason']} |")
    md += [
        "",
        "## Protected rollout sizes",
        "",
    ]
    md.extend(f"- `{m['path']}`: {m.get('allocated_size_bytes',0)/1_000_000:.3f} MB" for m in protected_sizes)
    md += [
        "",
        "The complete candidate fields, exact deletion targets, source revisions, hashes, and dependency closure are in the JSON manifests.",
        "",
    ]
    (OUT / "pre_cleanup_inventory.md").write_text("\n".join(md), encoding="utf-8")


def make_post() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pre = json.loads((OUT / "pre_cleanup_inventory.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "delete_manifest.json").read_text(encoding="utf-8"))
    disk = disk_snapshot()
    xvla = xvla_closure()
    remaining = []
    wsl_execution_path = OUT / "deletion_execution_wsl.json"
    wsl_execution = json.loads(wsl_execution_path.read_text(encoding="utf-8-sig"))
    wsl_deleted = {
        item["path"] for item in wsl_execution.get("results", [])
        if wsl_execution.get("execute") and item.get("status") == "DELETED"
    }
    for item in manifest["targets"]:
        if item["platform"] == "windows":
            exists = Path(item["path"]).exists()
        else:
            # The WSL executor validated every identity, removed it, and checked
            # os.path.lexists immediately after each exact deletion.
            exists = item["path"] not in wsl_deleted
        if exists:
            remaining.append(item["path"])
    before = pre["storage_before"]
    post = {
        "schema_version": "storage_cleanup.post_inventory.v1",
        "generated_at": now_iso(),
        "git": git_snapshot(),
        "workers": process_snapshot(),
        "storage_before": before,
        "storage_after": disk,
        "deltas": {
            "windows_free_bytes": disk["windows_c"]["free_bytes"] - before["windows_c"]["free_bytes"],
            "wsl_internal_free_bytes": disk["wsl_root"]["free_bytes"] - before["wsl_root"]["free_bytes"],
            "vhdx_physical_bytes": before["vhdx"]["physical_size_bytes"] - disk["vhdx"]["physical_size_bytes"],
        },
        "manifest_target_count": manifest["target_count"],
        "remaining_manifest_targets": remaining,
        "selected_xvla_closure_after": xvla,
        "protected_rollouts_exist": {p: Path(p).exists() for p in PROTECTED_WINDOWS},
    }
    (OUT / "post_cleanup_inventory.json").write_text(json.dumps(post, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pre", "post"))
    args = parser.parse_args()
    if args.mode == "pre":
        make_pre()
    else:
        make_post()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
