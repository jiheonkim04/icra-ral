import copy

import numpy as np
import pytest

from tca_map.epoch7_persistent_completion import (
    adjudicate_result,
    branch_signature,
    collapsed_mechanism,
    neutral_action,
    persistence_summary,
    validate_protocol,
)


def protocol_fixture():
    tasks = [
        {
            "task_id": index,
            "task_name": f"task_{index}",
            "mechanism": "placement",
            "hdf5": f"task_{index}.hdf5",
            "hdf5_sha256": "a" * 64,
        }
        for index in range(10)
    ]
    return {
        "schema_version": "epoch7.persistent_completion.problem_protocol.v1",
        "status": "FROZEN_BEFORE_TASK_PERSISTENCE_OUTCOMES",
        "policy_loaded_or_queried": False,
        "ours_authorized": False,
        "tasks": tasks,
        "hold_contract": {"steps": 30, "pose_delta": [0.0] * 6},
        "gates": {
            "execution": {
                "completed_tasks_min": 10,
                "exception_count_max": 0,
                "cold_repeat_task_ids": [0, 8],
            },
            "coverage": {
                "native_success_tasks_min": 8,
                "required_mechanisms": [
                    "placement",
                    "containment_or_insertion",
                    "planar_push",
                    "articulation",
                ],
            },
            "problem": {
                "immediate_persistence_failure_tasks_min": 3,
                "failure_mechanisms_min": 2,
                "failure_fraction_among_native_success_min": 0.2,
                "single_predicate_explanation_fraction_max": 2 / 3,
            },
            "headroom": {"suffix_recovered_tasks_min": 2, "suffix_recovered_mechanisms_min": 2},
        },
    }


def result_row(task_id, mechanism, task_name, immediate_persistent=True, suffix_persistent=True):
    return {
        "task_id": task_id,
        "task_name": task_name,
        "mechanism": mechanism,
        "completed": True,
        "finite_actions": True,
        "error": None,
        "cold_repeat_required": task_id in (0, 8),
        "cold_repeat_match": True if task_id in (0, 8) else None,
        "branches": {
            "immediate_neutral_hold": {
                "native_success": True,
                "persistent_success": immediate_persistent,
                "error": None,
            },
            "expert_suffix_then_hold": {
                "native_success": True,
                "persistent_success": suffix_persistent,
                "error": None,
            },
            "last_action_repeat": {"error": None},
        },
    }


def test_neutral_action_latches_gripper_sign():
    assert np.array_equal(neutral_action([1, 2, 3, 4, 5, 6, 0.2]), [0, 0, 0, 0, 0, 0, 1])
    assert np.array_equal(neutral_action([1, 2, 3, 4, 5, 6, -0.2]), [0, 0, 0, 0, 0, 0, -1])
    with pytest.raises(ValueError):
        neutral_action([0] * 6)


def test_persistence_summary_is_strict_all_steps():
    passed = persistence_summary([True] * 30, 30)
    failed = persistence_summary([True] * 7 + [False] + [True] * 22, 30)
    assert passed["persistent_success"] is True
    assert failed["persistent_success"] is False
    assert failed["first_hold_failure_index"] == 7
    assert failed["final_hold_success"] is True


def test_protocol_validation_rejects_changed_hold():
    protocol = protocol_fixture()
    assert validate_protocol(protocol) == []
    changed = copy.deepcopy(protocol)
    changed["hold_contract"]["steps"] = 29
    assert "primary hold length must remain 30" in validate_protocol(changed)


def test_mechanism_collapse():
    assert collapsed_mechanism("composite_articulation_containment") == "articulation"
    assert collapsed_mechanism("insertion") == "containment_or_insertion"
    assert collapsed_mechanism("placement") == "placement"


def test_adjudication_passes_only_with_diverse_gap_and_headroom():
    protocol = protocol_fixture()
    rows = [
        result_row(0, "articulation", "open_drawer", False, True),
        result_row(1, "placement", "put_bowl_on_stove", False, True),
        result_row(2, "containment", "put_item_in_bowl", False, False),
        result_row(3, "planar_push", "push_plate", True, True),
    ]
    rows.extend(result_row(i, "placement", f"other_{i}") for i in range(4, 10))
    summary = adjudicate_result(protocol, rows)
    assert summary["decision"] == "PROBLEM_VERIFIED_STRONG_COMPARATOR_RESIDUAL"
    no_headroom = copy.deepcopy(rows)
    for row in no_headroom:
        row["branches"]["expert_suffix_then_hold"]["persistent_success"] = False
    assert adjudicate_result(protocol, no_headroom)["decision"] == "NO_LEGAL_HEADROOM"


def test_branch_signature_excludes_timing_noise():
    branch = {
        "native_success": True,
        "first_success_step": 5,
        "hold_success_trace": [True, False],
        "persistent_success": False,
        "final_hold_success": False,
        "error": None,
        "wall_seconds": 12.3,
    }
    assert "wall_seconds" not in branch_signature(branch)
