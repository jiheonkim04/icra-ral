"""Run the frozen AMP-VLA Stage 0 action-manifold development audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.amp_vla import (  # noqa: E402
    ACTION_DIM,
    CHUNK_SIZE,
    LATENT_DIMS,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    amp_row_key,
    canonical_json_sha256,
    classify_stage0,
    json_default,
)


POLICY_PROBE = "amp_stage0_action_manifold_projection"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report = Path(args.report_root)
    run = Path(args.run_root)
    return {
        "report": report,
        "run": run,
        "serializer_preflight": report / "stage_0_serializer_preflight.json",
        "manifest": report / "stage_0_manifest.json",
        "partial": report / "stage_0_partial.json",
        "result_json": report / "stage_0_result.json",
    }


def _serializer_preflight(path: Path) -> dict[str, Any]:
    manifest_row = {
        "partition": "validation",
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": 8,
        "frame_index": 3,
        "latent_dim": LATENT_DIMS[0],
        "policy_probe": POLICY_PROBE,
    }
    manifest_row["row_key"] = amp_row_key(manifest_row)
    fixture = {
        "method": "AMP-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_row": manifest_row,
        "action_chunk": np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32),
        "manifold": {
            "latent_dim": LATENT_DIMS[0],
            "mean_preview": np.zeros(4, dtype=np.float32),
            "component_preview": np.eye(2, 4, dtype=np.float32),
            "coordinate_std_preview": np.ones(4, dtype=np.float32),
        },
        "projection_diagnostic": {
            "base_manifold_distance": np.float32(1.0),
            "clipped_base_manifold_distance": np.float32(0.9),
            "amp_manifold_distance": np.float32(0.5),
        },
        "decision_inputs": Stage0DecisionInputs(
            proposal_hash_ok=True,
            serializer_preflight_ok=True,
            official_prior_asset_check_persisted=True,
            manifest_integrity_ok=True,
            source_alignment_ok=True,
            feature_action_proprio_finite_aligned=True,
            minimum_discovery_windows=512,
            minimum_validation_windows=128,
            all_tasks_reported=True,
            maximum_validation_task_fraction=0.25,
            coordinate_variance_all_positive=True,
            manifold_reconstruction_relative_improvement=0.10,
            manifold_reconstruction_absolute_huber_improvement=0.0,
            coordinate_probe_relative_improvement=0.05,
            coordinate_probe_absolute_huber_improvement=0.0,
            abot_proxy_headroom_relative_improvement=0.05,
            abot_proxy_headroom_absolute_huber_improvement=0.0,
            clipping_explains_projection=False,
            projection_path_distinct=True,
            finite_objectives_and_gradients=True,
            amp_gradient_nonzero=True,
            gradient_ratio_at_most_100=True,
            frozen_parameter_gradient_count=0,
            identity_max_error=0.0,
            base_hash_unchanged=True,
            checkpoint_reload_ok=True,
            action_validity_ok=True,
            exception_count=0,
        ).__dict__,
    }
    fixture["decision"] = classify_stage0(Stage0DecisionInputs(**fixture["decision_inputs"]))
    fixture_hash = canonical_json_sha256(fixture)
    _write_json(path, {"fixture": fixture, "fixture_hash": fixture_hash, "written_at": _utc_now()})
    parsed = _read_json(path)
    reproduced = canonical_json_sha256(parsed["fixture"])
    result = {
        **parsed,
        "parsed": True,
        "reproduced_hash": reproduced,
        "passed": reproduced == fixture_hash and parsed.get("fixture_hash") == fixture_hash,
    }
    _write_json(path, result)
    if not result["passed"]:
        raise RuntimeError("AMP serializer preflight hash did not reproduce")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serializer-preflight", action="store_true")
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "amp_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "amp_vla" / "stage0"))
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.serializer_preflight:
        result = _serializer_preflight(paths["serializer_preflight"])
        print(f"AMP serializer preflight passed: {paths['serializer_preflight']} {result['fixture_hash']}")
        return 0
    raise SystemExit("AMP full Stage 0 execution path is not launched before runner validation is complete")


if __name__ == "__main__":
    raise SystemExit(main())
