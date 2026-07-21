from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DECISION = "EPOCH9E_NONDRAG_DISENGAGEMENT_FROZEN_NO_GO_ACTIVE_ROUTE_CLOSED"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_terminal_state_stops_at_exact_scoped_joint_failure() -> None:
    state = json.loads((REPORTS / "epoch9e_campaign_state.json").read_text(encoding="utf-8"))
    handoff = json.loads((REPORTS / "epoch9e_terminal_handoff.json").read_text(encoding="utf-8"))
    assert state["terminal_state"] == handoff["terminal_state"] == DECISION
    assert state["episode_accounting"]["joint_primary_assignments"] == {
        "planned": 24,
        "launched": 2,
        "completed": 1,
        "protocol_failed": 1,
        "unexecuted": 22,
    }
    assert state["episode_accounting"]["joint_sham_rows"]["unexecuted"] == 12
    assert state["statistics"]["complete_exact_pairs"] == 0
    assert state["statistics"]["paired_contrast_m"] is None
    assert state["validation_accessed"] is False
    assert state["confirmation_accessed"] is False
    assert state["estimator_development_started"] is False
    assert state["official_evaluation_started"] is False
    assert state["paper_status"] == "PAPER_NOT_AUTHORIZED"
    assert state["paper_paths"] == []


def test_terminal_evidence_index_hashes_every_entry() -> None:
    index = json.loads((REPORTS / "epoch9e_evidence_index.json").read_text(encoding="utf-8"))
    assert index["terminal_decision"] == DECISION
    assert index["entry_count"] == len(index["entries"])
    for row in index["entries"]:
        path = ROOT / row["path"]
        assert path.is_file()
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]
    assert index["validation_accessed"] is False
    assert index["confirmation_accessed"] is False


def test_terminal_trace_disclosure_retains_failed_response_window() -> None:
    state = json.loads((REPORTS / "epoch9e_campaign_state.json").read_text(encoding="utf-8"))
    traces = state["continuous_trace_disclosure"]
    assert len(traces) == 4
    incomplete = [row for row in traces if not row["frozen_response_window_complete"]]
    assert len(incomplete) == 1
    assert incomplete[0]["path"].endswith("epoch9e_joint_base_20261134_assignment_B_back.npz")
    assert incomplete[0]["frozen_response_window_steps"] == 0
    assert all(row["full_continuous_arrays_preserved_in_npz"] for row in traces)


def test_terminal_protected_manifests_and_seals_remain_exact() -> None:
    state = json.loads((REPORTS / "epoch9e_campaign_state.json").read_text(encoding="utf-8"))
    protected = {row["path"]: row for row in state["protected_untracked_manifests"]}
    assert protected["rollouts/2026_07_17/"] == {
        "path": "rollouts/2026_07_17/",
        "file_count": 27,
        "total_bytes": 5_143_751,
        "manifest_sha256": "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54",
    }
    assert protected["rollouts/2026_07_18/"] == {
        "path": "rollouts/2026_07_18/",
        "file_count": 10,
        "total_bytes": 924_633,
        "manifest_sha256": "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00",
    }
    assert state["identities"]["validation_demo_ids_sealed"] == [40, 41, 42, 43, 44]
    assert state["identities"]["confirmation_demo_ids_sealed"] == [45, 46, 47, 48, 49]
    assert state["active_epoch9_scientific_workers"] == []
