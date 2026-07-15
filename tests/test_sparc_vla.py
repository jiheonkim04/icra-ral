from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tca_map.smolvla.sparc_vla import (
    APERTURES,
    HIDDEN_WIDTH,
    PROPOSAL_HASH,
    SparcPostResidualAdapter,
    action_safety,
    activation_key,
    aggregate_covariances,
    aggregate_mean_conceptors,
    apply_token_operator,
    compute_conceptor,
    conceptor_and_not,
    equal_episode_covariance,
    episode_key,
    manifest_audit,
    quota,
    retained_energy,
    select_aperture,
    tensor_sha256,
    validate_conceptor,
)


def _diagonal_covariance(scale: float = 1.0) -> np.ndarray:
    values = np.linspace(0.0, scale, HIDDEN_WIDTH, dtype=np.float64)
    return np.diag(values)


def test_proposal_hash_matches_frozen_file() -> None:
    path = Path(__file__).parents[1] / "reports" / "sparc_vla" / "proposal_hash.txt"
    assert path.read_text(encoding="utf-8").strip() == PROPOSAL_HASH


def test_equal_episode_covariance_is_invariant_to_within_episode_duplication() -> None:
    first = np.zeros((1, HIDDEN_WIDTH), dtype=np.float64)
    second = np.ones((1, HIDDEN_WIDTH), dtype=np.float64)
    mean_a, covariance_a = equal_episode_covariance([first, second])
    mean_b, covariance_b = equal_episode_covariance([np.repeat(first, 20, axis=0), second])
    assert np.array_equal(mean_a, mean_b)
    assert np.array_equal(covariance_a, covariance_b)


def test_conceptor_matches_closed_form_eigenvalues_and_quota_is_monotone() -> None:
    covariance = _diagonal_covariance()
    small = compute_conceptor(covariance, 0.1)
    large = compute_conceptor(covariance, 10.0)
    expected = np.diag(covariance) / (np.diag(covariance) + 100.0)
    assert np.allclose(np.diag(small), expected)
    assert quota(large) > quota(small)
    assert validate_conceptor(large)["effective_rank_ge_0_1"] > 0


def test_and_not_is_finite_bounded_and_suppresses_failure_energy() -> None:
    success_covariance = np.diag(np.r_[np.ones(120), np.zeros(HIDDEN_WIDTH - 120)])
    failure_covariance = np.diag(np.r_[np.zeros(60), np.ones(60), np.zeros(HIDDEN_WIDTH - 120)])
    success = compute_conceptor(success_covariance, 1.0)
    failure = compute_conceptor(failure_covariance, 1.0)
    combined = conceptor_and_not(success, failure)
    validate_conceptor(combined)
    assert retained_energy(combined, failure_covariance) < retained_energy(success, failure_covariance)


def test_covariance_aggregate_is_not_mean_conceptor_on_nonidentical_spectra() -> None:
    first = _diagonal_covariance(1.0)
    second = np.diag(np.linspace(2.0, 0.0, HIDDEN_WIDTH, dtype=np.float64))
    aggregate_then_regularize = compute_conceptor(aggregate_covariances([first, second]), 1.0)
    regularize_then_aggregate = aggregate_mean_conceptors([first, second], 1.0)
    assert not np.allclose(aggregate_then_regularize, regularize_then_aggregate)


def test_aperture_selection_obeys_band_and_tie_break() -> None:
    values = {alpha: 0.95 for alpha in APERTURES}
    values[0.5] = 0.84
    values[1.0] = 0.86
    selected = select_aperture(values)
    assert selected["selected_aperture"] == 0.5
    assert selected["inside_band"] is True


def test_token_operator_is_identity_at_zero_and_acts_per_token() -> None:
    hidden = np.ones((1, 50, HIDDEN_WIDTH), dtype=np.float32)
    conceptor = np.eye(HIDDEN_WIDTH, dtype=np.float64)
    assert np.array_equal(apply_token_operator(hidden, conceptor, 0.0), hidden)
    conceptor[0, 0] = 0.0
    acted = apply_token_operator(hidden, conceptor, 0.1)
    assert acted.shape == hidden.shape
    assert np.allclose(acted[..., 0], 0.9)
    assert np.array_equal(acted[..., 1:], hidden[..., 1:])


def test_tensor_and_manifest_hashes_are_stable_and_duplicates_fail() -> None:
    value = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert tensor_sha256(value) == tensor_sha256(value.copy())
    episode = {
        "partition": "discovery",
        "policy": "base",
        "suite": "libero_10",
        "task_id": 0,
        "reset_seed": 20261901,
    }
    key = episode_key(episode)
    activation = activation_key(
        {"episode_key": key, "replan_index": 0, "residual_site": 11, "denoising_step": 0}
    )
    assert "replan=0" in activation
    assert manifest_audit([key], [key])["passed"] is True
    assert manifest_audit([key], [key, key])["duplicate_observed_count"] == 1


def test_action_safety_reports_components_and_rejects_large_delta() -> None:
    base = np.zeros((50, 7), dtype=np.float32)
    small = base.copy()
    small[:, 0] = 0.01
    audit = action_safety(base, small)
    assert audit["passed"] is True
    assert audit["translation_delta_l2_p95"] == pytest.approx(0.01)
    large = base.copy()
    large[:, :3] = 0.5
    assert action_safety(base, large)["passed"] is False


def test_post_residual_adapter_captures_and_mutates_the_shared_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = torch.nn.Identity()

    class Expert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(16)])

    class VLMWithExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_expert = Expert()

    class Inner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vlm_with_expert = VLMWithExpert()

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()

    policy = Policy()
    adapter = SparcPostResidualAdapter(policy, 11, expected_steps=1)
    adapter.register()
    hidden = torch.ones((1, 50, HIDDEN_WIDTH), dtype=torch.float32)
    with torch.no_grad():
        policy.model.vlm_with_expert.lm_expert.layers[12].input_layernorm(hidden)
    adapter.assert_complete()
    assert adapter.captures[0].delta_norm == 0.0

    conceptor = np.eye(HIDDEN_WIDTH, dtype=np.float64)
    conceptor[0, 0] = 0.0
    adapter.configure(conceptor, beta=0.1)
    adapter.reset_capture()
    acted = torch.ones((1, 50, HIDDEN_WIDTH), dtype=torch.float32)
    with torch.no_grad():
        policy.model.vlm_with_expert.lm_expert.layers[12].input_layernorm(acted)
    adapter.assert_complete()
    assert torch.allclose(acted[..., 0], torch.full_like(acted[..., 0], 0.9))
    assert adapter.captures[0].delta_norm > 0.0

    path = tmp_path / "adapter.npz"
    adapter.save(path)
    adapter.remove()
    loaded = SparcPostResidualAdapter.load(policy, path)
    loaded.register()
    loaded.reset_capture()
    reloaded = torch.ones((1, 50, HIDDEN_WIDTH), dtype=torch.float32)
    with torch.no_grad():
        policy.model.vlm_with_expert.lm_expert.layers[12].input_layernorm(reloaded)
    loaded.remove()
    assert torch.equal(acted, reloaded)
    metadata = json.loads(str(np.load(path, allow_pickle=False)["metadata"].item()))
    assert metadata["proposal_hash"] == PROPOSAL_HASH
