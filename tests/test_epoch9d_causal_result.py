from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_epoch9d_causal_signal_go_passes_every_frozen_gate() -> None:
    adjudication = load(REPORTS / "epoch9d_causal_panel_adjudication.json")
    assert adjudication["decision"] == "CAUSAL_SIGNAL_GO"
    assert adjudication["causal_signal_go"] is True
    assert adjudication["near_miss_replication_eligible"] is False
    assert all(adjudication["gates"].values())
    counts = adjudication["counts"]
    assert counts["finite_bounded_actions"] == 64
    assert counts["intended_contact_or_excitation"] == 64
    assert counts["both_candidates_excited"] == 32
    assert counts["rank_correct"] == 28
    assert counts["rank_by_heavy_position"] == {
        "front": {"correct": 15, "total": 16},
        "back": {"correct": 13, "total": 16},
    }
    assert counts["exact_pair_correct_flips"] == 12
    for key in ("collisions", "identity_swaps", "falls", "workspace_exits", "unrecoverable_track_losses"):
        assert counts[key] == 0


def test_epoch9d_causal_effect_and_controls_match_sealed_adjudication() -> None:
    adjudication = load(REPORTS / "epoch9d_causal_panel_adjudication.json")
    effect = adjudication["paired_mass_intervention"]
    assert effect["mean_m"] == 0.006593329847616967
    assert effect["positive_pairs"] == 15
    assert effect["negative_pairs"] == 1
    assert effect["zero_pairs"] == 0
    assert effect["one_sided_exact_sign_test_p"] == 0.0002593994140625
    assert effect["paired_student_t_95_interval_m"][0] > 0
    assert effect["adjusted_position_lane_order"]["hc3_95_interval_m"][0] > 0
    sham = adjudication["sham_control"]
    assert sham["pairs"] == 8
    assert sham["sampled_contact_rows"] == 0
    assert sham["collision_rows"] == 0
    assert sham["prediction_flips"] == 0
    assert sham["paired_student_t_95_interval_m"][0] <= 0 <= sham["paired_student_t_95_interval_m"][1]


def test_epoch9d_causal_raw_rows_traces_hashes_and_resources_are_complete() -> None:
    adjudication = load(REPORTS / "epoch9d_causal_panel_adjudication.json")
    result_path = ROOT / adjudication["raw_result"]["path"]
    result = load(result_path)
    assert sha256(result_path) == adjudication["raw_result"]["sha256"]
    assert len(result["rows"]) == 48
    assert sum(row["row_type"] == "PRIMARY_ASSIGNMENT" for row in result["rows"]) == 32
    assert sum(row["row_type"] == "SHAM_CONTROL" for row in result["rows"]) == 16
    assert len({row["row_key"] for row in result["rows"]}) == 48
    assert all(row["completed"] and row["exception"] is None for row in result["rows"])
    trace_records = []
    for row in result["rows"]:
        if row["row_type"] == "PRIMARY_ASSIGNMENT":
            trace_records.extend(row["probe_audits"].values())
        else:
            trace_records.append(row["probe_audit"])
    assert len(trace_records) == 80
    assert all(sha256(ROOT / record["trace_path"]) == record["trace_sha256"] for record in trace_records)
    resources = adjudication["resource_summary"]
    assert resources["peak_host_ram_percent"] < 82.0
    assert resources["wsl_swap_used_peak_bytes"] == 0
    assert resources["process_max_rss_bytes"] < 3 * 1024**3
    assert adjudication["validation_accessed"] is False
    assert adjudication["confirmation_accessed"] is False
