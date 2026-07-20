#!/usr/bin/env python3
"""Validate and delete only WSL targets enumerated in the cleanup manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path


MANIFEST = Path("/mnt/c/Users/jiheo/tca_map/reports/storage_cleanup/delete_manifest.json")
LOG = Path("/mnt/c/Users/jiheo/tca_map/reports/storage_cleanup/deletion_execution_wsl.json")
SELECTED = "/home/jiheon/assets/checkpoints/xvla_hf_cache"
OVERLAP_START = 1784491800


def digest(path: str) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def metrics(path: str) -> dict:
    top = os.lstat(path)
    apparent = 0
    allocated = 0
    files = 0
    newest = top.st_mtime
    overlap = 0
    if stat.S_ISLNK(top.st_mode) or stat.S_ISREG(top.st_mode):
        apparent = top.st_size
        allocated = top.st_blocks * 512
        files = int(stat.S_ISREG(top.st_mode))
        overlap = int(stat.S_ISREG(top.st_mode) and top.st_mtime > OVERLAP_START)
    else:
        for base, dirs, names in os.walk(path, followlinks=False):
            ds = os.lstat(base)
            allocated += ds.st_blocks * 512
            newest = max(newest, ds.st_mtime)
            retained = []
            for name in dirs:
                child = os.path.join(base, name)
                cs = os.lstat(child)
                if stat.S_ISLNK(cs.st_mode):
                    apparent += cs.st_size
                    allocated += cs.st_blocks * 512
                    newest = max(newest, cs.st_mtime)
                else:
                    retained.append(name)
            dirs[:] = retained
            for name in names:
                child = os.path.join(base, name)
                fs = os.lstat(child)
                apparent += fs.st_size
                allocated += fs.st_blocks * 512
                files += int(stat.S_ISREG(fs.st_mode))
                newest = max(newest, fs.st_mtime)
                overlap += int(stat.S_ISREG(fs.st_mode) and fs.st_mtime > OVERLAP_START)
    return {
        "apparent_size_bytes": apparent,
        "allocated_size_bytes": allocated,
        "file_count": files,
        "newest_mtime": newest,
        "overlap_writes": overlap,
    }


def relevant_workers() -> list[dict]:
    workers = []
    self_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == self_pid:
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            comm = (proc / "comm").read_text(errors="replace").strip()
        except OSError:
            continue
        lowered = f"{comm} {command}".lower()
        if any(word in lowered for word in ("libero", "mujoco", "rollout", "train", "download", "huggingface", "vla")):
            if "invoke_wsl_storage_cleanup.py" not in lowered:
                workers.append({"pid": int(proc.name), "comm": comm, "command": command})
        elif comm.startswith("python") and comm not in {"python3"}:
            workers.append({"pid": int(proc.name), "comm": comm, "command": command})
    return workers


def open_handles(targets: list[dict]) -> list[dict]:
    roots = [(item["path"].rstrip("/"), item["target_id"]) for item in targets]
    found = []
    self_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == self_pid:
            continue
        fd_root = proc / "fd"
        try:
            fds = list(fd_root.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                value = os.readlink(fd)
            except OSError:
                continue
            for root, target_id in roots:
                if value == root or value.startswith(root + "/"):
                    found.append({"pid": int(proc.name), "fd": str(fd), "path": value, "target_id": target_id})
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    if not manifest["validation"]["passed"]:
        raise SystemExit("manifest validation is false")
    workers = relevant_workers()
    if workers:
        raise SystemExit("relevant WSL worker detected: " + json.dumps(workers))
    targets = [item for item in manifest["targets"] if item["platform"] == "wsl"]
    handles = open_handles(targets)
    if handles:
        raise SystemExit("open handles detected: " + json.dumps(handles))
    validated = []
    for item in targets:
        path = item["path"]
        if item["classification"] != "VERIFIED_DISPOSABLE":
            raise SystemExit(f"bad class: {path}")
        if not os.path.lexists(path):
            raise SystemExit(f"target disappeared: {path}")
        if os.path.islink(path):
            raise SystemExit(f"top-level symlink refused: {path}")
        resolved = os.path.realpath(path).rstrip("/")
        expected = item["resolved_path"].rstrip("/")
        root = os.path.realpath(item["audited_root"]).rstrip("/")
        if resolved != expected:
            raise SystemExit(f"resolved identity changed: {path}")
        if not resolved.startswith(root + "/"):
            raise SystemExit(f"outside audited root: {path}")
        if resolved == SELECTED or resolved.startswith(SELECTED + "/") or SELECTED.startswith(resolved + "/"):
            raise SystemExit(f"selected X-VLA overlap: {path}")
        current = metrics(path)
        if current["overlap_writes"]:
            raise SystemExit(f"overlap-window write detected: {path}")
        # Direct-child cache metrics were captured via st_blocks; standalone targets via du.
        size_delta = abs(current["allocated_size_bytes"] - int(item["allocated_size_bytes"]))
        apparent_delta = abs(current["apparent_size_bytes"] - int(item["apparent_size_bytes"]))
        if size_delta > 1024 * 1024 or apparent_delta > 1024 * 1024:
            raise SystemExit(f"target size changed: {path}; {current} vs manifest")
        for relative, expected_hash in item.get("sentinel_hashes", {}).items():
            actual = digest(os.path.join(path, relative))
            if actual != expected_hash:
                raise SystemExit(f"sentinel hash changed: {path}/{relative}")
        revision = item.get("immutable_revision")
        ref = os.path.join(path, "refs", "main")
        if revision and os.path.isfile(ref):
            if Path(ref).read_text().strip() != revision:
                raise SystemExit(f"public cache revision changed: {path}")
        validated.append({
            "target_id": item["target_id"], "path": path,
            "allocated_size_bytes": item["allocated_size_bytes"],
            "status": "VALIDATED", "open_handle_check": "PASS",
        })
    results = []
    if args.execute:
        for item in validated:
            path = item["path"]
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
            if os.path.lexists(path):
                raise SystemExit(f"deletion verification failed: {path}")
            results.append({**item, "status": "DELETED"})
        subprocess.run(["sync"], check=True)
    else:
        results = [{**item, "status": "DRY_RUN_VALIDATED"} for item in validated]
    payload = {
        "schema_version": "storage_cleanup.wsl_execution.v1",
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "execute": args.execute,
        "target_count": len(results),
        "allocated_size_bytes": sum(int(item["allocated_size_bytes"]) for item in results),
        "workers": workers,
        "open_handles": handles,
        "results": results,
    }
    LOG.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("timestamp", "execute", "target_count", "allocated_size_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
