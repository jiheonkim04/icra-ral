from __future__ import annotations

import json
from pathlib import Path

from scripts.run_epoch10b_manifest_erratum import (
    CORRECTED_PREREG_PATH,
    ORIGINAL_PREREG_PATH,
    _is_finite,
    _key_ledger_sha256,
    _union_preregistration,
)


def test_union_manifest_has_128_primary_and_retains_original_16_reverse_states() -> None:
    original = json.loads(ORIGINAL_PREREG_PATH.read_text(encoding="utf-8"))
    corrected = json.loads(CORRECTED_PREREG_PATH.read_text(encoding="utf-8"))
    union = _union_preregistration(original, corrected)
    by_id = {row["state_id"]: row for row in union["states"]}
    assert len(union["states"]) == len(by_id) == 128
    assert sum(row["reverse_order_duplicate"] for row in union["states"]) == 16
    assert by_id["libero_spatial|task_0|demo_8|frame_60|transport_goal"]["reverse_order_duplicate"] is False
    assert by_id["libero_spatial|task_0|demo_8|frame_61|transport_goal"]["reverse_order_duplicate"] is True


def test_key_ledger_is_order_independent_and_newline_bound() -> None:
    left = _key_ledger_sha256(["b", "a", "a"])
    right = _key_ledger_sha256(["a", "b"])
    assert left == right
    assert left == "911169ddaaf146aff539f58c26c489af3b892dff0fe283c1c264c65ae5aa59a2"


def test_recursive_finite_check_rejects_nonfinite_scientific_values() -> None:
    assert _is_finite({"state": [0.0, 1.0], "score": 0.25})
    assert not _is_finite({"state": [0.0, float("nan")]})


def test_frozen_decision_sidecar_is_present() -> None:
    sidecar = Path("reports/epoch10b_erratum_frozen_decision.sha256")
    assert sidecar.is_file()
    assert sidecar.read_text(encoding="utf-8").split()[0] == (
        "cab840da177eaf99a9fa9f34b9814adc7464a273f512215bb7566ea7468a64a0"
    )
