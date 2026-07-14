from __future__ import annotations

import numpy as np
import pytest

from tca_map.smolvla.evostate_vla import (
    EvoStateConfig,
    audit_evostate_records,
    build_transition_pairs,
    damped_inverse_correction,
    validate_inference_fields,
)


def _record(task: str, identity: int, step: int, state: list[float], action: list[float]) -> dict:
    return {
        "task_key": task,
        "identity": identity,
        "step": step,
        "split": "acquisition",
        "state": state,
        "action": action,
        "previous_action": [0.0] * 7,
        "chunk_index_fraction": float(step % 4) / 4.0,
        "success": True,
    }


def test_build_transition_pairs_requires_consecutive_steps() -> None:
    records = [
        _record("libero_spatial/task_4", 20260901, 0, [0.0] * 8, [0.1] * 7),
        _record("libero_spatial/task_4", 20260901, 1, [0.1] * 8, [0.2] * 7),
        _record("libero_spatial/task_4", 20260901, 3, [0.3] * 8, [0.3] * 7),
    ]

    pairs = build_transition_pairs(records)

    assert len(pairs) == 1
    assert pairs[0]["step"] == 0
    assert np.allclose(pairs[0]["delta_state"], np.full(8, 0.1))


def test_validate_inference_fields_rejects_privileged_keys() -> None:
    with pytest.raises(ValueError, match="privileged EvoState inference fields"):
        validate_inference_fields({"state": [0.0] * 8, "success": True})


def test_damped_inverse_correction_clips_l2_norm() -> None:
    b = np.eye(8, 7, dtype=np.float64)
    mismatch = np.ones((3, 8), dtype=np.float64)

    correction = damped_inverse_correction(b, mismatch, damping=1e-2, delta_max=0.2)

    assert correction.shape == (3, 7)
    assert np.all(np.linalg.norm(correction, axis=1) <= 0.200001)


def test_audit_detects_forbidden_development_identity() -> None:
    records = []
    for step in range(4):
        state = [float(step + i) for i in range(8)]
        action = [float(step + i) / 10.0 for i in range(7)]
        records.append(_record("libero_spatial/task_4", 20260999, step, state, action))
    cfg = EvoStateConfig(
        train_identities=(20260999,),
        validation_identities=(),
        forbidden_development_identities=(20260999,),
        min_transition_pairs=1,
        min_task_transition_pairs=1,
    )

    report = audit_evostate_records(records, cfg)

    assert report["final_decision"].startswith("AUDIT_STOP")
    assert any("forbidden development identities" in reason for reason in report["hard_stop_reasons"])
