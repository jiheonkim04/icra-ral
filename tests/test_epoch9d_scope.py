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


def protected_snapshot(path: Path) -> tuple[int, int, str]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    lines = [
        f"{str(value.relative_to(ROOT)).replace(chr(92), '/')}\t{value.stat().st_size}\t{sha256(value)}"
        for value in files
    ]
    manifest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper()
    return len(files), sum(value.stat().st_size for value in files), manifest


def test_epoch9d_scope_correction_preserves_historical_records() -> None:
    state = json.loads((REPORTS / "epoch9d_campaign_state.json").read_text(encoding="utf-8"))
    scope = (REPORTS / "epoch9d_scope_correction.md").read_text(encoding="utf-8")
    assert state["program_status"] in {
        "ACTIVE_DYNAMIC_PROBE_CAUSAL_SIGNAL_AND_TASK_HEADROOM_UNRESOLVED",
        "ACTIVE_DYNAMIC_PROBE_SIGNAL_CONFIRMED_TASK_PRESERVATION_UNRESOLVED",
        "ACTIVE_DYNAMIC_PROBE_SIGNAL_CONFIRMED_TASK_PRESERVATION_NOT_ACHIEVED",
    }
    assert "ACTIVE_DYNAMIC_PROBE_CAUSAL_SIGNAL_AND_TASK_HEADROOM_UNRESOLVED" in scope
    assert state["paper_status"] == "PAPER_NOT_AUTHORIZED"
    assert state["validation_accessed"] is False
    assert state["confirmation_accessed"] is False
    assert "does not delete, edit, relabel, or supersede any historical row" in scope
    for record in state["historical_terminal_records_preserved"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]


def test_epoch9d_identity_allocations_are_literal_fresh_and_disjoint() -> None:
    state = json.loads((REPORTS / "epoch9d_campaign_state.json").read_text(encoding="utf-8"))
    inventory = json.loads((REPORTS / "epoch9d_identity_seed_inventory.json").read_text(encoding="utf-8"))
    allocation = state["identity_and_seed_allocations"]
    assert inventory["epoch9_largest_previously_used_numeric_development_identity_M"] == 39
    assert allocation["sealed_validation_source_demo_identities"] == list(range(40, 45))
    assert allocation["sealed_confirmation_source_demo_identities"] == list(range(45, 50))
    allocated: list[int] = []
    for key, values in allocation.items():
        if key.endswith("generated_identity_ids"):
            allocated.extend(values)
    assert min(allocated) == 50
    assert len(allocated) == len(set(allocated))
    assert set(allocated).isdisjoint(range(40, 50))
    used_seeds = set(inventory["seed_values"])
    for low, high in allocation["generator_seed_ranges"].values():
        assert used_seeds.isdisjoint(range(low, high + 1))


def test_epoch9d_protected_rollouts_match_frozen_snapshots() -> None:
    expected = {
        "rollouts/2026_07_17/": (27, 5143751, "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54"),
        "rollouts/2026_07_18/": (10, 924633, "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00"),
    }
    for relative, frozen in expected.items():
        assert protected_snapshot(ROOT / relative.rstrip("/")) == frozen
