import numpy as np
import pytest

from tca_map.smolvla.fang_vla import (
    FANGAuditConfig,
    audit_fang_records,
    build_fang_feature,
    compute_gate_targets,
    records_to_arrays,
    split_development_records,
    standardize_train_validation,
    validate_inference_fields,
)


def _record(task: str, identity: int, step: int, success: bool, action_shift: float) -> dict:
    return {
        "split": "unit",
        "task_key": task,
        "identity": identity,
        "step": step,
        "success": success,
        "state": np.full(8, float(identity % 10) * 0.01 + step * 0.001).tolist(),
        "action": [action_shift, 0.1 * action_shift, 0.0, 0.0, 0.0, 0.0, 0.0],
        "previous_action": [0.0] * 7,
        "chunk_index_fraction": 0.25,
    }


def _healthy_records() -> list[dict]:
    rows = []
    for task_index, task in enumerate(("libero_spatial/task_4", "libero_10/task_4")):
        offset = float(task_index) * 0.05
        for identity in (1, 2, 3, 4):
            success = identity in (1, 2)
            action = 0.8 + offset if success else -0.8 - offset
            for step in range(6):
                rows.append(_record(task, 20260900 + identity, step, success, action + step * 0.01))
        for identity in (11, 12, 13, 14):
            success = identity in (11, 12)
            action = 0.75 + offset if success else -0.75 - offset
            for step in range(6):
                rows.append(_record(task, 20260900 + identity, step, success, action + step * 0.01))
    return rows


def test_build_fang_feature_shape() -> None:
    feature = build_fang_feature(
        state=np.zeros(8),
        action=np.zeros(7),
        previous_action=np.zeros(7),
        chunk_index_fraction=0.5,
        task_key="libero_spatial/task_4",
    )

    assert feature.shape == (25,)
    assert feature[-2:].tolist() == [1.0, 0.0]


def test_validate_inference_fields_rejects_privileged_keys() -> None:
    with pytest.raises(ValueError, match="privileged FANG"):
        validate_inference_fields({"reward": 1.0, "state_vector": [0.0]})


def test_audit_passes_healthy_development_records() -> None:
    report = audit_fang_records(
        _healthy_records(),
        FANGAuditConfig(
            train_identities=tuple(range(20260901, 20260905)),
            validation_identities=tuple(range(20260911, 20260915)),
            forbidden_confirmatory_identities=tuple(range(20260920, 20260925)),
            min_class_rows=10,
            min_action_field_separation=0.05,
        ),
    )

    assert report["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert report["validation_action_field_separation"]["median"] > 1.0
    assert report["duplicate_development_keys"] == 0


def test_gate_targets_are_bounded_and_noncollapsed_for_separated_records() -> None:
    config = FANGAuditConfig(
        train_identities=tuple(range(20260901, 20260905)),
        validation_identities=tuple(range(20260911, 20260915)),
        forbidden_confirmatory_identities=tuple(range(20260920, 20260925)),
        min_class_rows=10,
        min_action_field_separation=0.05,
    )
    splits = split_development_records(_healthy_records(), config)
    train = records_to_arrays(splits["train"])
    validation = records_to_arrays(splits["validation"])
    standardized = standardize_train_validation(train["features"], validation["features"], config)

    targets = compute_gate_targets(
        train_features_std=standardized["train_features"],
        train_actions=train["actions"],
        train_tasks=train["tasks"],
        train_labels=train["labels"],
        query_features_std=standardized["validation_features"],
        query_tasks=validation["tasks"],
        config=config,
    )

    assert targets["targets"].shape == (len(splits["validation"]),)
    assert np.all(targets["targets"] >= 0.0)
    assert np.all(targets["targets"] <= 1.0)
    assert float(np.mean(targets["targets"])) > 0.1


def test_audit_stops_on_duplicates_and_confirmatory_overlap() -> None:
    rows = _healthy_records()
    duplicate = dict(rows[0])
    rows.append(duplicate)
    rows.append(_record("libero_spatial/task_4", 20260921, 99, True, 0.5))

    report = audit_fang_records(
        rows,
        FANGAuditConfig(
            train_identities=tuple(range(20260901, 20260905)),
            validation_identities=tuple(range(20260911, 20260915)),
            forbidden_confirmatory_identities=tuple(range(20260920, 20260925)),
            min_class_rows=10,
        ),
    )

    assert report["final_decision"].startswith("AUDIT_STOP")
    assert "DATA_FAILURE" in report["failure_classifications"]
    assert report["duplicate_development_keys"] == 1
    assert report["forbidden_confirmatory_identities_present"] == [20260921]
