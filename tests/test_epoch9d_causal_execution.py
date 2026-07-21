from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.adjudicate_epoch9d_causal_panel import adjusted_hc3, paired_t_interval


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def test_epoch9d_adjusted_effect_preserves_positive_exact_pair_signal() -> None:
    bases = {}
    rows = []
    for offset, identity in enumerate(range(56, 72)):
        bases[identity] = {
            "base_identity_id": identity,
            "spatial_stratum": f"spatial_{offset % 4}",
            "probe_order": ["front", "back"] if offset % 2 == 0 else ["back", "front"],
            "candidate_initial_xyz_eval_only": {"back": [-0.17 + 0.002 * offset, 0.03 + 0.002 * (offset % 4), 0.898]},
            "candidate_initial_lane_margin_m_eval_only": {"back": 0.011 + 0.0001 * offset},
        }
        rows.append({"base_identity_id": identity, "mass_contrast_m": 0.004 + 0.0001 * (offset % 3)})
    adjusted = adjusted_hc3(rows, bases)
    interval = paired_t_interval(np.asarray([row["mass_contrast_m"] for row in rows]))
    assert adjusted["estimate_m"] > 0
    assert adjusted["hc3_95_interval_m"][0] > 0
    assert interval[0] > 0


def test_epoch9d_execution_seal_binds_every_executable_after_build() -> None:
    seal_path = REPORTS / "epoch9d_causal_execution_seal.json"
    if not seal_path.exists():
        return
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["outcomes_accessed_before_seal"] is False
    assert seal["frozen_counts"] == {
        "base_states": 16,
        "primary_assignments": 32,
        "candidate_probes": 64,
        "sham_rows": 16,
    }
    for path_key, hash_key in (
        ("causal_protocol_path", "causal_protocol_sha256"),
        ("runner_path", "runner_sha256"),
        ("host_wrapper_path", "host_wrapper_sha256"),
        ("original_epoch9b_runner_path", "original_epoch9b_runner_sha256"),
        ("original_controller_freeze_path", "original_controller_freeze_sha256"),
        ("calibration_path", "calibration_sha256"),
    ):
        assert sha256(ROOT / seal[path_key]) == seal[hash_key]
    adjudicator_hash = sha256(ROOT / seal["adjudicator_path"])
    if adjudicator_hash != seal["adjudicator_sha256"]:
        repair = json.loads(
            (REPORTS / "epoch9d_causal_adjudication_parser_repair.json").read_text(encoding="utf-8")
        )
        assert repair["original_sealed_adjudicator_sha256"] == seal["adjudicator_sha256"]
        assert repair["repaired_adjudicator_sha256"] == adjudicator_hash
        assert repair["scientific_fields_changed"] is False


def test_epoch9d_no_causal_outcome_exists_before_execution_seal_commit() -> None:
    result = REPORTS / "epoch9d_causal_panel/result.json"
    preflight = REPORTS / "epoch9d_causal_execution_preflight.json"
    if not (REPORTS / "epoch9d_causal_execution_seal.json").exists():
        assert not result.exists()
        assert not preflight.exists()
