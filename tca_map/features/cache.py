"""Feature-cache interface scaffolding.

This module is intentionally dependency-light. It validates the cache contract
using dummy hidden tokens, not SmolVLA imports or model execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from tca_map.datasets import make_dummy_samples
from tca_map.models import DummyAdapter

FEATURE_CACHE_SCHEMA_VERSION = "tca-map-feature-cache-v0"


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_feature_record(sample: dict, hidden_tokens: list[float], adapter_name: str = "dummy") -> dict:
    record = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "sample_id": sample["sample_id"],
        "dataset_version": sample.get("dataset_version", "unknown"),
        "adapter_name": adapter_name,
        "instruction": sample["instruction"],
        "target": sample["target"],
        "distractor": sample.get("distractor"),
        "expert_action": sample["expert_action"],
        "hidden_tokens": [float(value) for value in hidden_tokens],
        "hidden_dim": len(hidden_tokens),
        "observation_summary": {
            "rgb_shape": sample.get("observation", {}).get("rgb_shape"),
            "candidate_object_count": len(sample.get("observation", {}).get("candidate_objects", [])),
        },
        "policy": {
            "dummy_cache": adapter_name == "dummy",
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "openvla_oft_executed": False,
        },
    }
    record["record_sha256"] = _stable_hash(record)
    return record


def write_feature_cache(
    records: Iterable[dict],
    output_dir: Path,
    metadata: dict | None = None,
    overwrite: bool = False,
) -> dict:
    record_list = list(records)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Feature cache output is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.jsonl"
    manifest_path = output_dir / "manifest.json"

    with features_path.open("w", encoding="utf-8") as handle:
        for record in record_list:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")

    hidden_dims = sorted({int(record.get("hidden_dim", 0)) for record in record_list})
    manifest = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "record_count": len(record_list),
        "hidden_dims": hidden_dims,
        "features_path": features_path.name,
        "metadata": metadata or {},
        "policy": {
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "openvla_oft_executed": False,
        },
    }
    manifest["manifest_sha256"] = _stable_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def write_dummy_feature_cache(output_dir: Path, max_samples: int = 4, overwrite: bool = False) -> dict:
    adapter = DummyAdapter()
    records = []
    for sample in make_dummy_samples(count=max_samples):
        hidden_tokens = adapter.get_hidden_tokens(sample["observation"], sample["instruction"])
        records.append(build_feature_record(sample, hidden_tokens, adapter_name=adapter.name))
    return write_feature_cache(
        records,
        output_dir=output_dir,
        metadata={
            "source": "dummy-libero",
            "purpose": "feature-cache interface validation only",
            "paper_grade_result": False,
        },
        overwrite=overwrite,
    )


def validate_feature_cache(output_dir: Path) -> dict:
    manifest_path = output_dir / "manifest.json"
    features_path = output_dir / "features.jsonl"
    errors: list[str] = []

    if not manifest_path.exists():
        errors.append("missing manifest.json")
    if not features_path.exists():
        errors.append("missing features.jsonl")

    manifest = {}
    records: list[dict] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if features_path.exists():
        records = [json.loads(line) for line in features_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if manifest.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        errors.append("unexpected manifest schema_version")
    if manifest.get("record_count") != len(records):
        errors.append("manifest record_count does not match features.jsonl")

    for index, record in enumerate(records):
        if record.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
            errors.append(f"record {index} has unexpected schema_version")
        if record.get("hidden_dim") != len(record.get("hidden_tokens", [])):
            errors.append(f"record {index} hidden_dim mismatch")
        policy = record.get("policy", {})
        for key in [
            "downloads_performed",
            "gpu_jobs_performed",
            "heavy_model_imports_performed",
            "model_load_performed",
            "model_inference_performed",
            "training_performed",
            "rollouts_performed",
            "openvla_oft_executed",
        ]:
            if policy.get(key) is not False:
                errors.append(f"record {index} policy {key} must be false")

    return {
        "valid": not errors,
        "errors": errors,
        "manifest": manifest,
        "record_count": len(records),
    }


def plan_feature_cache(report_path: Path, output_dir: Path, write_dummy_cache: bool = False) -> dict:
    manifest = None
    validation = None
    if write_dummy_cache:
        manifest = write_dummy_feature_cache(output_dir=output_dir, overwrite=True)
        validation = validate_feature_cache(output_dir)

    report = {
        "policy": {
            "planning_only": not write_dummy_cache,
            "dummy_cache_written": write_dummy_cache,
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "openvla_oft_executed": False,
        },
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "default_output_dir": str(output_dir),
        "cache_files": ["manifest.json", "features.jsonl"],
        "real_smolvla_feature_cache_blockers": [
            "runtime packages missing until separately approved install",
            "ALLOW_HEAVY_IMPORT=1 required before actual model load",
            "no model inference, training, rollouts, simulator, or OpenVLA-OFT allowed in this scaffold",
        ],
        "dummy_manifest": manifest,
        "dummy_validation": validation,
        "recommended_next_step": (
            "Use dummy cache tests for interface work. Stop before real SmolVLA feature extraction until runtime install and heavy import gates are explicitly approved."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-path", default="reports/feature_cache_plan_report.json")
    parser.add_argument("--output-dir", default="runs/feature_cache/dummy_contract")
    parser.add_argument("--write-dummy-cache", action="store_true")
    args = parser.parse_args()

    report = plan_feature_cache(
        report_path=Path(args.report_path),
        output_dir=Path(args.output_dir),
        write_dummy_cache=args.write_dummy_cache,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
