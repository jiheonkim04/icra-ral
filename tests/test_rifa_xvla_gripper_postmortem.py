from __future__ import annotations

import numpy as np

from tca_map.rifa_xvla.gripper_postmortem import (
    exact_live_row_index,
    analyze_action_semantics,
)
from tca_map.rifa_xvla.stage0 import load_frozen_contract


def _plan(gripper_scores: list[float]) -> np.ndarray:
    plan = np.zeros((len(gripper_scores), 20), dtype=np.float32)
    plan[:, 3] = 1.0
    plan[:, 7] = 1.0
    plan[:, 9] = np.asarray(gripper_scores, dtype=np.float32)
    return plan


def test_exact_failing_live_row_index_and_seed_basis() -> None:
    contract = load_frozen_contract()
    assert exact_live_row_index(contract) == 9
    assert int(contract["training_budget"]["seed"]) + 3000 + exact_live_row_index(contract) == 20263727


def test_threshold_crossing_is_classified_as_postprocess_discontinuity() -> None:
    base = _plan([0.1, 0.5001, 0.2])
    full = _plan([0.1, 0.4999, 0.2])
    ablation = _plan([0.1, 0.4998, 0.2])
    result = analyze_action_semantics(base, full, ablation)
    assert result["decision"] == "RIFA_GRIPPER_POSTPROCESS_DISCONTINUITY_CONFIRMED"
    assert result["chunk_index"] == 0
    assert result["action_index_within_chunk"] == 1
    assert result["max_abs_2_caused_by_sign_threshold_discontinuity"] is True
    assert result["full_and_ablation_same_gripper_whole_chunk"] is True
    assert result["policies_at_flip"]["BASE"]["final_discretized_gripper_action"] == 1.0
    assert result["policies_at_flip"]["RIFA_XVLA"]["final_discretized_gripper_action"] == -1.0


def test_missing_raw_gripper_signal_is_reported_unavailable() -> None:
    plan = np.zeros((2, 9), dtype=np.float32)
    result = analyze_action_semantics(plan, plan, plan)
    assert result["decision"] == "RIFA_GRIPPER_INTERNAL_SIGNAL_UNAVAILABLE"
