from __future__ import annotations

from pathlib import Path

import pytest

from tca_map.xvla_task1.closed_loop_residual_eval import (
    ClosedLoopResidualConfig,
    _build_frozen_manifest,
    _decide,
    _policy_specs_from_labels,
    identity_to_initial_state_index,
    parse_policy_labels,
)
from tca_map.xvla_task1.training_spec import build_br_xvla_training_spec


def test_identity_20260727_maps_to_frozen_task1_initial_state() -> None:
    assert identity_to_initial_state_index(20260727) == 16


def test_parse_policy_labels_rejects_unregistered_policy() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_policy_labels("br_xvla_primary,new_method")


def test_frozen_manifest_declares_single_residual_no_retuning(tmp_path: Path) -> None:
    spec = build_br_xvla_training_spec()
    config = ClosedLoopResidualConfig(
        spec_path=tmp_path / "spec.json",
        output_root=tmp_path / "closed_loop",
        training_output_root=tmp_path / "training",
        policy_labels=("xvla_prior_base", "br_xvla_primary", "uniform_xvla_ablation"),
    )
    policies = _policy_specs_from_labels(config, spec)

    manifest = _build_frozen_manifest(config, spec, policies)

    assert manifest["status"] == "FROZEN_BEFORE_RESULT_INSPECTION"
    assert manifest["reset_identities"] == [20260727]
    assert manifest["initial_state_indices"] == {"20260727": 16}
    assert manifest["training_happened_at_manifest_write"] is False
    assert manifest["optimizer_step_happened_at_manifest_write"] is False
    assert manifest["closed_loop_ours_evaluation_happened_at_manifest_write"] is False
    assert manifest["selection_rules"]["retuning_from_this_result_allowed"] is False
    assert manifest["selection_rules"]["broader_confirmatory_evaluation_allowed_by_this_manifest"] is False
    assert [policy["label"] for policy in manifest["policy_specs"]] == [
        "xvla_prior_base",
        "br_xvla_primary",
        "uniform_xvla_ablation",
    ]


def test_closed_loop_decision_requires_primary_to_beat_uniform_when_available() -> None:
    rows = {
        "xvla_prior_base": [{"completed": True, "success": False}],
        "br_xvla_primary": [{"completed": True, "success": True}],
        "uniform_xvla_ablation": [{"completed": True, "success": False}],
    }
    success, decision, summary = _decide(rows, [])

    assert success is True
    assert decision == "BR_XVLA_CLOSED_LOOP_RESIDUAL_PASS_BEATS_ABLATION"
    assert summary["xvla_prior_failure_reproduced"] is True
    assert summary["primary_beats_uniform_ablation"] is True


def test_closed_loop_decision_flags_nondecisive_when_uniform_also_succeeds() -> None:
    rows = {
        "xvla_prior_base": [{"completed": True, "success": False}],
        "br_xvla_primary": [{"completed": True, "success": True}],
        "uniform_xvla_ablation": [{"completed": True, "success": True}],
    }
    success, decision, summary = _decide(rows, [])

    assert success is True
    assert decision == "BR_XVLA_CLOSED_LOOP_RESIDUAL_PASS_NOT_ABLATION_DECISIVE"
    assert summary["primary_beats_uniform_ablation"] is False


def test_closed_loop_decision_blocks_if_prior_failure_not_reproduced() -> None:
    rows = {
        "xvla_prior_base": [{"completed": True, "success": True}],
        "br_xvla_primary": [{"completed": True, "success": True}],
        "uniform_xvla_ablation": [{"completed": True, "success": False}],
    }
    success, decision, summary = _decide(rows, [])

    assert success is False
    assert decision == "BR_XVLA_CLOSED_LOOP_RESIDUAL_PRIOR_FAILURE_NOT_REPRODUCED"
    assert summary["xvla_prior_failure_reproduced"] is False
