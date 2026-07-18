from __future__ import annotations

import json
from pathlib import Path

import torch

from tca_map.action_consistent_missing_view_distillation.adapter import (
    ActionConsistentMissingViewAdapter,
    adapter_parameter_count,
    state_dict_parameter_count,
)
from tca_map.action_consistent_missing_view_distillation.spec import (
    DEFAULT_SPEC,
    load_frozen_method_spec,
)


def test_frozen_spec_validates_and_matches_exact_parameter_counts() -> None:
    spec = load_frozen_method_spec()
    module_spec = spec["trainable_module"]
    adapter = ActionConsistentMissingViewAdapter(
        hidden_size=module_spec["hidden_size"],
        bottleneck_dim=module_spec["bottleneck_dim"],
        wrist_token_count=module_spec["wrist_token_count"],
        wrist_token_dim=module_spec["wrist_token_dim"],
        residual_scale=module_spec["residual_scale"],
    )
    assert adapter_parameter_count(adapter) == 434_816
    assert state_dict_parameter_count(adapter.inference_state_dict()) == 279_808


def test_zero_initialization_and_clean_path_are_exact() -> None:
    torch.manual_seed(7)
    adapter = ActionConsistentMissingViewAdapter(
        hidden_size=32,
        bottleneck_dim=8,
        wrist_token_count=4,
        wrist_token_dim=16,
    )
    hidden = torch.randn(2, 5, 32)

    clean, reconstruction, _ = adapter(
        hidden,
        torch.zeros(2, 1),
        compute_reconstruction=True,
    )
    assert clean is hidden
    assert reconstruction is None
    assert torch.equal(clean, hidden)

    dropout, reconstruction, _ = adapter(
        hidden,
        torch.ones(2, 1),
        compute_reconstruction=True,
    )
    assert torch.equal(dropout, hidden)
    assert reconstruction is not None
    assert torch.count_nonzero(reconstruction) == 0


def test_reconstruction_prediction_is_not_an_action_residual_input() -> None:
    torch.manual_seed(11)
    adapter = ActionConsistentMissingViewAdapter(
        hidden_size=32,
        bottleneck_dim=8,
        wrist_token_count=4,
        wrist_token_dim=16,
    )
    hidden = torch.randn(2, 5, 32)
    _, reconstruction, _ = adapter(
        hidden,
        torch.ones(2, 1),
        compute_reconstruction=True,
    )
    assert reconstruction is not None
    reconstruction.square().mean().backward()
    assert adapter.action_residual_output.weight.grad is None
    assert adapter.reconstruction_output.weight.grad is not None


def test_inference_export_omits_every_reconstruction_parameter() -> None:
    adapter = ActionConsistentMissingViewAdapter()
    exported = adapter.inference_state_dict()
    assert exported
    assert not any(
        key.startswith(adapter.RECONSTRUCTION_PREFIXES)
        for key in exported
    )


def test_spec_is_machine_readable_without_outcome_fields() -> None:
    raw = json.loads(Path(DEFAULT_SPEC).read_text(encoding="utf-8"))
    assert raw["data_splits"]["confirmatory_outcomes_accessed"] is False
    assert raw["execution_boundaries"]["current_stage"] == "METHOD_SPECIFICATION_FROZEN_NOT_TRAINED"
    assert len(raw["data_splits"]["stage_a"]["initial_reset_identities_per_task"]) == 3
    assert raw["data_splits"]["stage_b"]["initial_paired_failure_episode_rows_per_policy"] == 60
    assert raw["data_splits"]["stage_b"]["single_expansion_paired_failure_episode_rows_per_policy"] == 80
    assert len(raw["data_splits"]["stage_b"]["failure_conditions"]) == 3
    assert raw["execution_boundaries"]["second_backbone_required_for_paper_candidate_go"] is False
    assert raw["execution_boundaries"]["camera_only_validation_required_for_paper_candidate_go"] is False
    assert "result" not in raw
