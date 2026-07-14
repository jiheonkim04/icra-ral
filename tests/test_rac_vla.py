from __future__ import annotations

import numpy as np
import pytest

from tca_map.smolvla.rac_vla import (
    RACConfig,
    audit_rac_records,
    build_consequence_pairs,
    build_labeled_examples,
    calibration_residual,
    inverse_residual,
    perturb_action,
    run_validation_search,
    validate_inference_fields,
)


def _record(task: str, identity: int, step: int, state: list[float], action: list[float]) -> dict:
    return {
        "task_key": task,
        "identity": identity,
        "step": step,
        "state": state,
        "action": action,
        "previous_action": [0.0] * 7,
        "chunk_index_fraction": float(step % 4) / 4.0,
        "success": True,
    }


def test_perturbation_and_inverse_are_bounded() -> None:
    action = np.array([0.4, -0.2, 0.1, 0.01, 0.02, 0.03, -0.8], dtype=np.float64)

    perturbed = perturb_action(action, 1)
    residual = inverse_residual(action, 1)
    direct_residual = calibration_residual(action, 1)

    assert perturbed[0] == pytest.approx(0.4 / 0.65)
    assert residual.shape == (7,)
    assert np.allclose(residual, direct_residual)
    assert np.linalg.norm(residual) <= 0.200001


def test_validate_inference_fields_rejects_privileged_keys() -> None:
    with pytest.raises(ValueError, match="privileged RAC inference fields"):
        validate_inference_fields({"state": [0.0] * 8, "identity": 20260901})


def test_build_labeled_examples_uses_history_horizon() -> None:
    records = [
        _record("libero_spatial/task_4", 20260901, 0, [0.0] * 8, [0.1] * 7),
        _record("libero_spatial/task_4", 20260901, 1, [0.1] * 8, [0.2] * 7),
        _record("libero_spatial/task_4", 20260901, 2, [0.2] * 8, [0.3] * 7),
    ]

    pairs = build_consequence_pairs(records)
    examples = build_labeled_examples(pairs, RACConfig(history_horizon=2, min_consequence_pairs=1, min_task_pairs=1))

    assert len(pairs) == 2
    assert examples["labels"].shape == (5,)
    assert examples["features"]["full"].shape[0] == 5


def test_audit_detects_forbidden_identity() -> None:
    records = []
    for step in range(4):
        records.append(
            _record(
                "libero_spatial/task_4",
                20260999,
                step,
                [float(step + i) for i in range(8)],
                [float(step + i) / 10.0 for i in range(7)],
            )
        )
    cfg = RACConfig(
        train_identities=(20260999,),
        validation_identities=(),
        forbidden_development_identities=(20260999,),
        min_consequence_pairs=1,
        min_task_pairs=1,
    )

    report = audit_rac_records(records, cfg)

    assert report["final_decision"].startswith("AUDIT_STOP")
    assert any("forbidden development identities" in reason for reason in report["hard_stop_reasons"])


def test_validation_search_returns_six_configs_for_valid_records() -> None:
    records = []
    for identity in list(range(20260901, 20260911)) + list(range(20260911, 20260917)):
        for step in range(6):
            base = float(identity - 20260900) * 0.01 + float(step) * 0.1
            records.append(
                _record(
                    "libero_spatial/task_4" if identity % 2 else "libero_10/task_4",
                    identity,
                    step,
                    [base + float(i) * 0.01 for i in range(8)],
                    [base + float(i) * 0.02 for i in range(7)],
                )
            )

    report = run_validation_search(records)

    assert report["tried_config_count"] == 6
    assert "selected_config" in report
