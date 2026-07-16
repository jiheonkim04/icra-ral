"""Run DCCG-VLA Stage 0 implementation preflight utilities."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.dccg_vla import (  # noqa: E402
    ACTION_DIM,
    FEATURE_COUNT,
    HORIZON,
    POLICY_ROWS,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    action_validity_summary,
    apply_dccg_guidance,
    canonical_json_sha256,
    classify_stage0,
    coherence_energy,
    coherence_features,
    deployment_bin_key,
    dccg_row_key,
    fit_demo_statistics,
    gradient_smoke,
    json_default,
    no_demo_calibration_stats,
    smoothing_simple_killer,
    validate_manifest,
)


POLICY_PROBE = "dccg_stage0_demonstration_calibrated_coherence_guidance"
CONFIG_LABEL = "dccg_frozen_stage0_c0"
REPORT_ROOT = REPO_ROOT / "reports" / "dccg_vla"
RUN_ROOT = REPO_ROOT / "runs" / "dccg_vla" / "stage0"
REQUIRED_SOURCE_DOCS = (
    REPORT_ROOT / "researcher_proposal.md",
    REPORT_ROOT / "reviewer_attack.md",
    REPORT_ROOT / "researcher_rebuttal.md",
    REPORT_ROOT / "mathematical_mechanism_audit.md",
    REPORT_ROOT / "preregistration.md",
    REPORT_ROOT / "prototype_protocol.md",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def _manifest_row(split: str, policy: str = "dccg_full") -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split,
        "task_suite": "libero_goal",
        "task_id": "libero_goal/task_5",
        "demo_id": 30 if split == "validation" else 0,
        "window_start": 12,
        "bin_key": "libero_goal|q1|t1|r0|g1|c1",
        "policy": policy,
        "probe_label": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
    }
    row["row_key"] = dccg_row_key(row)
    return row


def _synthetic_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(20263600)
    base = np.zeros((4, HORIZON, ACTION_DIM), dtype=np.float64)
    demo = base.copy()
    for idx in range(4):
        demo[idx, :, 0] = np.linspace(0.0, 0.04 + idx * 0.004, HORIZON)
        demo[idx, :, 1] = 0.01 * np.sin(np.linspace(0.0, np.pi, HORIZON) + idx)
        demo[idx, 18:24, 6] = 0.35
    jitter = demo.copy()
    jitter[:, :, 0] += rng.normal(scale=0.015, size=(4, HORIZON))
    bin_keys = [deployment_bin_key(chunk, task_family="libero_goal", queue_index=12) for chunk in demo]
    features = coherence_features(demo)
    stats = fit_demo_statistics(features, bin_keys)
    global_stats = no_demo_calibration_stats(features)
    gradient = gradient_smoke(jitter[:1], [bin_keys[0]], stats)
    dccg, _ = apply_dccg_guidance(jitter[:1], gradient["gradient"], [1.0], gamma=0.10)
    smoothing = smoothing_simple_killer(jitter[:1])
    gate_fraction = 0.25
    decision_inputs = Stage0DecisionInputs(
        proposal_hash_ok=True,
        serializer_preflight_ok=True,
        official_prior_asset_check_persisted=True,
        preflight_passed=True,
        manifest_integrity_ok=True,
        source_alignment_ok=True,
        action_semantics_ok=True,
        base_chunks_valid=True,
        features_noncollapsed=True,
        bins_noncollapsed=True,
        enough_discovery_windows=True,
        enough_validation_windows=True,
        validation_task_coverage_ok=True,
        maximum_validation_task_fraction=0.25,
        gate_activation_fraction=gate_fraction,
        base_acg_headroom=0.05,
        dccg_differs_from_base=True,
        dccg_differs_from_acg=True,
        dccg_differs_from_ablation=True,
        dccg_differs_from_smoothing=bool(np.max(np.abs(dccg - smoothing)) > 0.0),
        finite_nonzero_gradients=bool(gradient["finite_nonzero_gradients"]),
        exact_base_passthrough_ok=True,
        gripper_event_preservation_ok=True,
        normalized_action_validity_ok=True,
        postprocessed_action_validity_ok=True,
        clean_retention_ok=True,
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        closed_loop_experiment_happened=False,
        simulator_load_count=0,
        training_happened=False,
        validation_search_happened=False,
        exception_count=0,
    )
    manifest = [_manifest_row("discovery", "smolvla_base"), _manifest_row("validation", "dccg_full")]
    partial = [{"row_key": row["row_key"]} for row in manifest]
    return {
        "method": "DCCG-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "horizon": HORIZON,
        "action_dimension": ACTION_DIM,
        "feature_count": FEATURE_COUNT,
        "policy_rows": POLICY_ROWS,
        "config_label": CONFIG_LABEL,
        "probe_label": POLICY_PROBE,
        "manifest_row": manifest[0],
        "manifest_summary": validate_manifest(manifest, partial),
        "coherence_features": features,
        "coherence_energy": coherence_energy(jitter[:1], [bin_keys[0]], stats),
        "global_energy": coherence_energy(jitter[:1], ["global"], global_stats),
        "gradient_norm_mean": gradient["gradient_norm_mean"],
        "action_delta_summary": action_delta_summary(jitter[:1], dccg),
        "action_validity_summary": action_validity_summary(dccg),
        "decision_inputs": decision_inputs,
        "decision": classify_stage0(decision_inputs),
    }


def _serializer_preflight(path: Path) -> dict[str, Any]:
    fixture = _synthetic_fixture()
    fixture_hash = canonical_json_sha256(fixture)
    payload = {
        "method": "DCCG-VLA",
        "passed": True,
        "created_utc": _utc_now(),
        "fixture": fixture,
        "fixture_hash": fixture_hash,
        "parsed": True,
        "reproduced_hash": fixture_hash,
    }
    _write_json(path, payload)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    persisted_hash = canonical_json_sha256(persisted["fixture"])
    persisted["reproduced_hash"] = persisted_hash
    persisted["passed"] = persisted_hash == persisted["fixture_hash"]
    _write_json(path, persisted)
    return persisted


def _preflight(report_root: Path) -> dict[str, Any]:
    proposal_hash_file = report_root / "proposal_hash.txt"
    proposal_hash_ok = proposal_hash_file.is_file() and proposal_hash_file.read_text(encoding="utf-8").strip() == PROPOSAL_HASH
    missing_docs = [str(path) for path in REQUIRED_SOURCE_DOCS if not path.is_file()]
    payload = {
        "method": "DCCG-VLA",
        "created_utc": _utc_now(),
        "proposal_hash": PROPOSAL_HASH,
        "proposal_hash_ok": proposal_hash_ok,
        "missing_required_docs": missing_docs,
        "preflight_passed": proposal_hash_ok and not missing_docs,
        "implementation_stage": "preflight_only",
    }
    _write_json(report_root / "stage_0_preflight.json", payload)
    return payload


def _write_static_contract_artifacts(report_root: Path) -> None:
    _write_json(
        report_root / "stage_0_official_prior_asset_check.json",
        {
            "method": "DCCG-VLA",
            "created_utc": _utc_now(),
            "closest_prior": "ACG",
            "official_repository": "https://github.com/DAVIAN-Robotics/ACG",
            "official_assets_checked": False,
            "local_policy_2_label": "acg_official_proxy",
            "transparent_proxy_required_if_official_assets_unavailable": True,
        },
    )
    _write_json(
        report_root / "stage_0_action_semantics.json",
        {
            "method": "DCCG-VLA",
            "created_utc": _utc_now(),
            "model_native_action_shape": [HORIZON, ACTION_DIM],
            "postprocessor_or_unnormalizer_class": "official SmolVLA/LIBERO action postprocessor required at full Stage 0",
            "gripper_convention": "preserve hard transition counts and sign-change timing",
            "final_action_validity_definition": "shape [50,7], finite entries, official postprocessor validity, and Base-relative group caps",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", default=str(REPORT_ROOT))
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--serializer-preflight", action="store_true")
    args = parser.parse_args(argv)
    report_root = Path(args.report_root)
    if not report_root.is_absolute():
        report_root = REPO_ROOT / report_root
    report_root.mkdir(parents=True, exist_ok=True)
    if args.serializer_preflight:
        _serializer_preflight(report_root / "stage_0_serializer_preflight.json")
        return 0
    _preflight(report_root)
    _write_static_contract_artifacts(report_root)
    _serializer_preflight(report_root / "stage_0_serializer_preflight.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
