import numpy as np
import pytest

from tca_map.smolvla.cavm_vla import (
    CAVMConfig,
    apply_cavm_action,
    build_cavm_key,
    fit_cavm_memory,
    validate_inference_fields,
)


def _record(task: str, identity: int, success: bool, state_shift: float, action_shift: float) -> dict:
    state = np.full(8, state_shift, dtype=float)
    action = np.array([action_shift, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    previous = np.zeros(7, dtype=float)
    return {
        "task_key": task,
        "identity": identity,
        "success": success,
        "state": state.tolist(),
        "action": action.tolist(),
        "previous_action": previous.tolist(),
        "chunk_index_fraction": 0.0,
    }


def test_build_cavm_key_has_expected_shape() -> None:
    key = build_cavm_key(
        state=np.zeros(8),
        action=np.zeros(7),
        previous_action=np.zeros(7),
        chunk_index_fraction=0.25,
        task_key="libero_spatial/task_4",
    )

    assert key.shape == (25,)
    assert key[-2:].tolist() == [1.0, 0.0]


def test_validate_inference_fields_rejects_privileged_keys() -> None:
    with pytest.raises(ValueError, match="privileged CAVM"):
        validate_inference_fields({"object_pose": [0.0], "state_vector": [0.0]})


def test_fit_memory_passes_when_success_and_failure_separate() -> None:
    acquisition = []
    calibration = []
    for task_index, task in enumerate(["libero_spatial/task_4", "libero_10/task_4"]):
        base = float(task_index)
        acquisition.extend(
            [
                _record(task, 1 + task_index * 10, True, base, 0.8),
                _record(task, 2 + task_index * 10, True, base + 0.01, 0.75),
                _record(task, 3 + task_index * 10, False, base, -0.8),
                _record(task, 4 + task_index * 10, False, base + 0.01, -0.75),
            ]
        )
        calibration.append(_record(task, 20 + task_index, True, base + 0.005, 0.7))

    memory = fit_cavm_memory(acquisition, calibration, CAVMConfig(min_gateable_fraction=0.5))

    assert memory["final_decision"] == "STAGE_1_PROCEED_TO_STAGE_2A"
    assert memory["calibration_metrics"]["median_success_failure_separation"] > 1.0
    assert memory["episode_counts"]["libero_spatial/task_4"]["success"] == 2


def test_fit_memory_kills_without_failure_mixture() -> None:
    acquisition = []
    calibration = []
    for task_index, task in enumerate(["libero_spatial/task_4", "libero_10/task_4"]):
        base = float(task_index)
        acquisition.extend(
            [
                _record(task, 1 + task_index * 10, True, base, 0.8),
                _record(task, 2 + task_index * 10, True, base + 0.01, 0.75),
            ]
        )
        calibration.append(_record(task, 20 + task_index, True, base + 0.005, 0.7))

    memory = fit_cavm_memory(acquisition, calibration)

    assert memory["final_decision"] == "STAGE_0_PERMANENT_KILL_NO_CONTRASTIVE_MEMORY"
    assert memory["calibration_metrics"]["hard_kill_reasons"]


def test_full_cavm_uses_failure_direction_beyond_no_contrast() -> None:
    acquisition = []
    calibration = []
    for task_index, task in enumerate(["libero_spatial/task_4", "libero_10/task_4"]):
        base = float(task_index)
        acquisition.extend(
            [
                _record(task, 1 + task_index * 10, True, base, 0.8),
                _record(task, 2 + task_index * 10, True, base + 0.01, 0.75),
                _record(task, 3 + task_index * 10, False, base, -0.8),
                _record(task, 4 + task_index * 10, False, base + 0.01, -0.75),
            ]
        )
        calibration.append(_record(task, 20 + task_index, True, base + 0.005, 0.7))
    memory = fit_cavm_memory(acquisition, calibration, CAVMConfig(min_gateable_fraction=0.5))

    kwargs = {
        "memory": memory,
        "state": np.zeros(8),
        "action": np.zeros(7),
        "previous_action": np.zeros(7),
        "chunk_index_fraction": 0.0,
        "task_key": "libero_spatial/task_4",
    }
    full_action, full_diag = apply_cavm_action(variant="cavm_full", **kwargs)
    ablation_action, ablation_diag = apply_cavm_action(variant="cavm_no_contrast_ablation", **kwargs)

    assert full_diag["gate"] > 0.0
    assert ablation_diag["gate"] > 0.0
    assert full_action[0] > ablation_action[0]
