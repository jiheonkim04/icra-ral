import numpy as np
import pytest
import torch

from tca_map.smolvla.nice_vla import (
    LOW_RANK,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    TinyCovariance,
    TinyResidualMean,
    action_validity,
    auroc_average_ranks,
    classify_stage0a,
    condition_vector,
    conformal_threshold,
    covariance_nll,
    dense_innovation_reference,
    deterministic_pca_basis,
    discovery_gripper_deadband,
    episode_cluster_score,
    innovation_terms,
    mean_cosine_loss,
    nearest_rank_quantile,
    pair_key,
    passthrough_queue_action,
    validate_manifest,
)


def test_proposal_hash_is_frozen() -> None:
    assert PROPOSAL_HASH == "898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A"


def test_condition_vector_has_frozen_shape_and_transition() -> None:
    action = torch.tensor([[0.3, 0.4, 0.0, 0.0, 0.0, 0.2, 0.6]])
    previous = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]])
    result = condition_vector(action, previous, 0.4)
    assert result.shape == (1, 18)
    assert result[0, 14].item() == pytest.approx(0.5)
    assert result[0, 15].item() == pytest.approx(0.2)
    assert result[0, 16].item() == pytest.approx(0.6)
    assert result[0, 17].item() == 1.0


def test_discovery_deadband_uses_nonzero_median() -> None:
    actions = np.zeros((5, 7), dtype=np.float32)
    actions[:, 6] = [0.0, 0.0, 0.2, 0.6, 0.6]
    assert discovery_gripper_deadband([actions]) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        discovery_gripper_deadband([np.zeros((5, 7))])


def test_tiny_mean_shapes_and_gradient() -> None:
    torch.manual_seed(1)
    model = TinyResidualMean(12)
    visual = torch.randn(3, 5, 12)
    action = torch.randn(3, 7)
    target = torch.randn(3, 5, 12)
    prediction = model(visual, action)
    loss = mean_cosine_loss(prediction, target)
    loss.backward()
    assert prediction.shape == target.shape
    assert torch.isfinite(loss)
    assert sum(float(parameter.grad.norm()) for parameter in model.parameters() if parameter.grad is not None) > 0.0


@pytest.mark.parametrize("rank", [0, LOW_RANK])
def test_tiny_covariance_is_bounded(rank: int) -> None:
    torch.manual_seed(2)
    model = TinyCovariance(12, rank=rank)
    diagonal, low_rank = model(torch.randn(3, 5, 12), torch.randn(3, 18))
    assert diagonal.shape == (3, 60)
    assert torch.all(diagonal >= 1e-6)
    assert torch.all(diagonal <= 1e2)
    assert (low_rank is None) == (rank == 0)
    if low_rank is not None:
        assert low_rank.shape == (3, 8)


def test_diagonal_innovation_matches_dense_reference() -> None:
    torch.manual_seed(3)
    residual = torch.randn(2, 3, 4, dtype=torch.float64)
    diagonal = torch.rand(2, 12, dtype=torch.float64) + 0.2
    _, mahal, logdet = innovation_terms(residual, diagonal)
    dense_mahal, dense_logdet = dense_innovation_reference(residual, diagonal)
    assert torch.allclose(mahal, dense_mahal, atol=1e-10, rtol=1e-10)
    assert torch.allclose(logdet, dense_logdet, atol=1e-10, rtol=1e-10)


def test_low_rank_innovation_matches_dense_reference() -> None:
    torch.manual_seed(4)
    residual = torch.randn(2, 3, 4, dtype=torch.float64)
    diagonal = torch.rand(2, 12, dtype=torch.float64) + 0.2
    basis, _ = torch.linalg.qr(torch.randn(12, 3, dtype=torch.float64))
    low_rank = torch.rand(2, 3, dtype=torch.float64) + 0.1
    _, mahal, logdet = innovation_terms(residual, diagonal, basis=basis, rank_variance=low_rank)
    dense_mahal, dense_logdet = dense_innovation_reference(
        residual, diagonal, basis=basis, rank_variance=low_rank
    )
    assert torch.allclose(mahal, dense_mahal, atol=1e-9, rtol=1e-9)
    assert torch.allclose(logdet, dense_logdet, atol=1e-9, rtol=1e-9)


def test_covariance_gradient_does_not_reach_detached_residual() -> None:
    torch.manual_seed(5)
    residual = torch.randn(3, 4, 6, requires_grad=True)
    raw = torch.randn(3, 24, requires_grad=True)
    diagonal = torch.nn.functional.softplus(raw) + 1e-6
    loss = covariance_nll(residual.detach(), diagonal)
    loss.backward()
    assert residual.grad is None
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.norm() > 0


def test_deterministic_pca_basis_is_orthonormal_and_signed() -> None:
    torch.manual_seed(6)
    residuals = torch.randn(16, 4, 5)
    first = deterministic_pca_basis(residuals)
    second = deterministic_pca_basis(residuals)
    assert first.shape == (20, 8)
    assert torch.allclose(first.T @ first, torch.eye(8), atol=1e-5)
    assert torch.equal(first, second)
    for column in range(8):
        pivot = torch.argmax(torch.abs(first[:, column]))
        assert first[pivot, column] >= 0


def test_nearest_rank_and_episode_cluster_score() -> None:
    values = list(range(1, 21))
    assert nearest_rank_quantile(values, 0.90) == 18.0
    assert episode_cluster_score(values) == 18.0
    with pytest.raises(ValueError):
        episode_cluster_score(values[:15])


def test_conformal_threshold_is_finite_sample_and_task_balanced() -> None:
    result = conformal_threshold({"a": [1, 2, 3], "b": [4, 5, 6]}, 0.90)
    assert result["one_indexed_rank"] == 6
    assert result["threshold"] == 6.0
    tied = conformal_threshold({"a": [1, 2], "b": [2, 2]}, 0.50)
    assert tied["one_indexed_rank"] == 3
    assert tied["threshold"] == 2.0
    with pytest.raises(ValueError):
        conformal_threshold({"a": [1], "b": [2, 3]}, 0.90)


def test_manifest_detects_duplicates_missing_and_extras() -> None:
    base = {
        "suite": "libero_10",
        "task_identity": "libero_10/task_1",
        "source_path": "/x/a.hdf5",
        "demo_id": 0,
        "frame_t": 1,
        "frame_t_plus_10": 11,
        "partition": "discovery",
    }
    other = {**base, "frame_t": 2, "frame_t_plus_10": 12}
    complete = [{**base, "pair_key": pair_key(base)}, {**other, "pair_key": pair_key(other)}]
    assert validate_manifest([base, other], complete)["passed"]
    duplicate = validate_manifest([base, other], [complete[0], complete[0]])
    assert duplicate["duplicate_result_key_count"] == 1
    assert duplicate["missing_manifest_key_count"] == 1

    validation = {**base, "partition": "validation_calibration", "frame_t": 3, "frame_t_plus_10": 13}
    validation_result = [{**validation, "pair_key": pair_key(validation)}]
    assert validate_manifest(
        [validation], validation_result, allowed_partitions=("discovery", "validation_calibration")
    )["passed"]


def test_monitor_disabled_passthrough_is_exact() -> None:
    queue = np.arange(35, dtype=np.float32).reshape(5, 7)
    copied, action = passthrough_queue_action(queue)
    assert np.array_equal(copied, queue)
    assert np.array_equal(action, queue[0])
    copied[0, 0] = -1
    assert queue[0, 0] == 0


def test_action_validity_reports_component_groups() -> None:
    result = action_validity(np.zeros((4, 7), dtype=np.float32))
    assert result["inside_fraction"] == 1.0
    assert result["translation_inside_fraction"] == 1.0
    assert result["rotation_inside_fraction"] == 1.0
    assert result["gripper_inside_fraction"] == 1.0


def test_auroc_uses_average_ranks_for_ties() -> None:
    assert auroc_average_ranks([0.0, 0.0], [1.0, 1.0]) == 1.0
    assert auroc_average_ranks([1.0, 1.0], [0.0, 0.0]) == 0.0
    assert auroc_average_ranks([1.0, 1.0], [1.0, 1.0]) == 0.5


def test_stage0_decision_requires_every_gate() -> None:
    values = Stage0DecisionInputs(
        completed_pairs=128,
        planned_pairs=128,
        exception_count=0,
        manifest_passed=True,
        source_passed=True,
        latent_passed=True,
        gradient_passed=True,
        algebra_passed=True,
        calibration_passed=True,
        passthrough_passed=True,
        reload_passed=True,
        action_validity_passed=True,
        forbidden_reads_zero=True,
    )
    assert classify_stage0a(values) == "NICE_STAGE_0A_PASS_STAGE_0B_ALLOWED"
    assert classify_stage0a(Stage0DecisionInputs(**{**values.__dict__, "algebra_passed": False})) == (
        "NICE_STAGE_0A_IMPLEMENTATION_FAILURE"
    )
    assert classify_stage0a(Stage0DecisionInputs(**{**values.__dict__, "source_passed": False})) == (
        "NICE_STAGE_0A_DATA_FAILURE"
    )
