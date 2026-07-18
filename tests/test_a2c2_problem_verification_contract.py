from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_a2c2_problem_verification import _episode_bounds


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_a2c2_problem_verification_contract() -> None:
    protocol = json.loads((ROOT / "reports/a2c2_prior/problem_verification_protocol.json").read_text(encoding="utf-8"))
    assert protocol["fidelity_label"] == "MECHANISM_FAITHFUL_A2C2_LOCAL_PORT"
    assert protocol["not_official_reproduction"] is True
    assert protocol["prior_training_budget"]["job_classification"] == "PRIOR_MODULE_TRAINING"
    assert protocol["prior_training_budget"]["not_vla_training"] is True
    assert protocol["prior_training_budget"]["optimizer_steps"] == 40000
    assert protocol["prior_training_budget"]["microbatch"] == 4
    assert protocol["evaluation_panel"]["official_init_state_ids"] == [0, 1, 2, 3, 4]
    assert [task["task_id"] for task in protocol["evaluation_panel"]["tasks"]] == [0, 4, 8]
    assert protocol["evaluation_panel"]["episodes_per_condition"] == 15
    assert protocol["conditions"]["BASE_STANDARD_E10_D0"] == {
        "execution_horizon": 10,
        "inference_delay": 0,
        "role": "Base competence",
    }
    assert protocol["conditions"]["BASE_DELAYED_E40_D10"]["execution_horizon"] == 40
    assert protocol["conditions"]["BASE_DELAYED_E40_D10"]["inference_delay"] == 10
    assert protocol["ours_boundary"]["ours_candidates_allowed_before_verified_residual"] == 0
    assert len(protocol["training_data"]["episodes_by_global_task_index"]) == 10
    assert sum(len(value) for value in protocol["training_data"]["episodes_by_global_task_index"].values()) == 40
    assert set(protocol["frozen_decision_rules"]["allowed_final_decisions"]) == {
        "VERIFIED_PRIOR_RESIDUAL",
        "PRIOR_SATURATES_PROBLEM",
        "BASE_NOT_COMPETENT",
        "NO_REPEATABLE_PROBLEM",
        "NO_DIAGNOSTIC_HEADROOM",
        "PRIOR_INFRASTRUCTURE_BLOCKED",
        "EVALUATION_INVALID",
    }


def test_protocol_code_hashes_match_frozen_sources() -> None:
    protocol = json.loads((ROOT / "reports/a2c2_prior/problem_verification_protocol.json").read_text(encoding="utf-8"))
    for key in ("module", "runner"):
        path = ROOT / protocol["implementation"][key]
        observed = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        assert observed == protocol["implementation"][f"{key}_sha256"]


def test_episode_bounds_use_lerobot_044_subset_local_indices() -> None:
    class Dataset:
        episodes = [13, 11]
        hf_dataset = {"episode_index": [11, 11, 13, 13, 13]}

        def __len__(self) -> int:
            return 5

    assert _episode_bounds(Dataset()) == [(0, 2), (2, 5)]


def test_episode_bounds_reject_noncontiguous_duplicate_episode() -> None:
    class Dataset:
        episodes = [11, 13]
        hf_dataset = {"episode_index": [11, 13, 11]}

        def __len__(self) -> int:
            return 3

    with pytest.raises(RuntimeError, match="missing, duplicated, or noncontiguous"):
        _episode_bounds(Dataset())
