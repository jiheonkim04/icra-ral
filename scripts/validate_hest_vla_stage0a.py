"""Independently reload and validate the completed HEST-VLA Stage 0A artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.hest_vla import (  # noqa: E402
    ACTION_DIM,
    ARM_DIM,
    HORIZON,
    PROPOSAL_HASH,
    Stage0ADecisionInputs,
    canonical_json_sha256,
    chunk_sha256,
    classify_stage0a,
    parse_sha256_registry,
    validate_manifest,
)


REPORT_ROOT = REPO_ROOT / "reports" / "hest_vla"


def _read_json(name: str) -> dict[str, Any]:
    return json.loads((REPORT_ROOT / name).read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate() -> dict[str, Any]:
    manifest = _read_json("stage_0a_pair_manifest.json")
    partial = _read_json("stage_0a_partial.json")
    result = _read_json("stage_0a_result.json")
    status = _read_json("stage_0a_status.json")
    heartbeat = _read_json("stage_0a_heartbeat.json")
    rows = list(partial["rows"])
    manifest_rows = list(manifest["rows"])
    audit = validate_manifest(manifest_rows, rows)

    manifest_payload = dict(manifest)
    persisted_manifest_hash = str(manifest_payload.pop("manifest_hash"))
    recomputed_manifest_hash = canonical_json_sha256(manifest_payload)
    proposal_path = REPORT_ROOT / "researcher_proposal.md"
    registry_hash = parse_sha256_registry((REPORT_ROOT / "proposal_hash.txt").read_text(encoding="utf-8"))
    proposal_hash_recomputed = _sha256(proposal_path)

    invalid_support_counts = {
        policy: sum(not bool(row["support_valid"][policy]) for row in rows)
        for policy in ("base", "spline_proxy", "hest", "no_endpoint", "moving_average")
    }
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    validation_invalid_support_counts = {
        policy: sum(not bool(row["support_valid"][policy]) for row in validation_rows)
        for policy in invalid_support_counts
    }

    persisted_chunk_errors: list[str] = []
    for row in rows:
        path = Path(str(row["persisted_chunk_path"]))
        if not path.is_file():
            persisted_chunk_errors.append(f"missing:{row['window_key']}")
            continue
        chunk = np.load(path, allow_pickle=False)
        if chunk.shape != (HORIZON, ACTION_DIM) or not np.isfinite(chunk).all():
            persisted_chunk_errors.append(f"shape_or_finite:{row['window_key']}")
            continue
        if chunk_sha256(chunk) != row["output_sha256"]["hest"]:
            persisted_chunk_errors.append(f"output_hash:{row['window_key']}")
        if _sha256(path) != row["persisted_chunk_sha256"]:
            persisted_chunk_errors.append(f"file_hash:{row['window_key']}")

    comparator_max = {
        "spline_proxy": max(float(row["hest_spline_proxy_max_abs_delta"]) for row in validation_rows),
        "no_endpoint": max(float(row["hest_no_endpoint_max_abs_delta"]) for row in validation_rows),
        "moving_average": max(float(row["hest_moving_average_max_abs_delta"]) for row in validation_rows),
    }
    manifest_ok = (
        audit["manifest_row_count"] == 160
        and audit["partial_row_count"] == 160
        and audit["duplicate_manifest_key_count"] == 0
        and audit["duplicate_partial_key_count"] == 0
        and audit["missing_manifest_key_count"] == 0
        and audit["extra_partial_key_count"] == 0
        and audit["partition_overlap_count"] == 0
        and bool(audit["key_sets_equal"])
    )
    inputs = Stage0ADecisionInputs(
        proposal_hash_ok=proposal_hash_recomputed == registry_hash == PROPOSAL_HASH,
        manifest_audit_ok=manifest_ok,
        source_finite_shape_ok=all(
            bool(row["source_shape_valid"]) and float(row["source_finite_fraction"]) == 1.0 for row in rows
        ),
        arm_support_noncollapsed=bool(
            np.all(np.asarray(result["summary"]["discovery_arm_ranges"], dtype=np.float64) > 1e-8)
        ),
        validation_transition_count=sum(bool(row["gripper_transition"]) for row in validation_rows),
        endpoint_max_error=max(float(row["endpoint_max_abs_error"]) for row in rows),
        first_action_max_error=max(float(row["first_action_max_abs_error"]) for row in rows),
        gripper_max_error=max(float(row["gripper_max_abs_error"]) for row in rows),
        all_variant_support_valid=all(
            all(bool(valid) for valid in row["support_valid"].values()) for row in rows
        ),
        acting_fraction=float(
            np.mean([float(row["hest_base_arm_max_abs_delta"]) > 1e-8 for row in validation_rows])
        ),
        median_energy_reduction=float(
            np.median([float(row["hest_energy_reduction"]) for row in validation_rows])
        ),
        comparator_distinct=all(value > 1e-10 for value in comparator_max.values()),
        roundtrip_max_error=max(float(row["roundtrip_max_abs_error"]) for row in rows),
        exception_count=int(partial["exception_count"]),
    )
    recomputed_decision = classify_stage0a(inputs)
    exit_code_text = (REPORT_ROOT / "stage_0a_exit_code.txt").read_text(encoding="utf-8").strip()
    artifact_integrity_ok = (
        inputs.proposal_hash_ok
        and persisted_manifest_hash == recomputed_manifest_hash
        and partial["manifest_hash"] == persisted_manifest_hash
        and result["manifest_hash"] == persisted_manifest_hash
        and manifest_ok
        and not persisted_chunk_errors
        and int(partial["completed_window_count"]) == 160
        and int(result["completed_window_count"]) == 160
        and int(partial["exception_count"]) == 0
        and status["status"] == "completed"
        and heartbeat["status"] == "completed"
        and exit_code_text == "0"
        and recomputed_decision == result["final_decision"]
    )
    return {
        "method": "HEST-VLA",
        "stage": "0A",
        "artifact_integrity_ok": artifact_integrity_ok,
        "proposal_hash_recomputed": proposal_hash_recomputed,
        "proposal_hash_registry": registry_hash,
        "manifest_hash_persisted": persisted_manifest_hash,
        "manifest_hash_recomputed": recomputed_manifest_hash,
        "manifest_audit": audit,
        "completed_window_count": len(rows),
        "planned_window_count": 160,
        "exception_count": int(partial["exception_count"]),
        "status": status["status"],
        "heartbeat_status": heartbeat["status"],
        "exit_code": exit_code_text,
        "persisted_chunk_error_count": len(persisted_chunk_errors),
        "persisted_chunk_errors": persisted_chunk_errors,
        "invalid_support_counts": invalid_support_counts,
        "validation_invalid_support_counts": validation_invalid_support_counts,
        "all_variant_support_valid": inputs.all_variant_support_valid,
        "acting_fraction": inputs.acting_fraction,
        "median_energy_reduction": inputs.median_energy_reduction,
        "recomputed_decision": recomputed_decision,
        "persisted_decision": result["final_decision"],
        "stage_0b_allowed": False,
        "valid_scientific_result": False,
    }


def main() -> int:
    validation = validate()
    _write_json(REPORT_ROOT / "stage_0a_independent_validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["artifact_integrity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
