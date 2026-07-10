from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "training_manifest.json",
    "eval_preprocessor_postprocessor_refs.json",
    "source_repro_lock.yaml",
    "sha256_manifest.json",
}


class CheckpointManifestError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _read_json(manifest_path)
    seeds = manifest.get("seeds") or []
    if [int(seed.get("seed")) for seed in seeds] != [11, 22, 33]:
        raise CheckpointManifestError("central manifest must list seeds [11, 22, 33] in order")
    seen_paths: set[str] = set()
    for seed_info in seeds:
        seed = int(seed_info["seed"])
        if seed_info.get("status") != "CHECKPOINT_COMPLETE_VERIFIED":
            raise CheckpointManifestError(f"seed {seed} status is not verified: {seed_info.get('status')}")
        root = Path(seed_info["checkpoint_path"])
        if str(root).lower() in seen_paths:
            raise CheckpointManifestError(f"duplicate checkpoint path: {root}")
        seen_paths.add(str(root).lower())
        if root.name != f"seed_{seed}":
            raise CheckpointManifestError(f"seed {seed} path does not end in seed_{seed}: {root}")
        missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
        if missing:
            raise CheckpointManifestError(f"seed {seed} missing required files: {missing}")
        bundle_manifest = _read_json(root / "sha256_manifest.json")
        bundle_files = bundle_manifest.get("files") or {}
        for relative, metadata in bundle_files.items():
            file_path = root / relative
            if not file_path.is_file():
                raise CheckpointManifestError(f"seed {seed} manifest references missing file: {relative}")
            expected = str(metadata.get("sha256", "")).upper()
            actual = _sha256_file(file_path)
            if expected != actual:
                raise CheckpointManifestError(f"seed {seed} checksum mismatch for {relative}: {expected} != {actual}")
        central_files = seed_info.get("file_hashes") or {}
        for relative, metadata in central_files.items():
            expected = str(metadata.get("sha256", "")).upper()
            actual = _sha256_file(root / relative)
            if expected != actual:
                raise CheckpointManifestError(f"seed {seed} central checksum mismatch for {relative}: {expected} != {actual}")
        disk_reload = seed_info.get("disk_reload") or {}
        if disk_reload.get("loaded_from_disk") is not True:
            raise CheckpointManifestError(f"seed {seed} disk reload is not proven")
        if not str(disk_reload.get("model_parameter_device", "")).startswith("cuda"):
            raise CheckpointManifestError(f"seed {seed} disk reload did not report CUDA parameters")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate official SmolVLA LoRA checkpoint bundles.")
    parser.add_argument("path", nargs="?", default="reports/official_smolvla_lora_checkpoint_manifest.json")
    args = parser.parse_args()
    manifest = validate_manifest(args.path)
    print(f"OFFICIAL_SMOLVLA_LORA_CHECKPOINTS_OK final_decision={manifest.get('final_decision')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
