import math

import pytest

from tca_map.inference import (
    distributional_tca_select_inference,
    heatmap_entropy,
    heatmap_js,
    heatmap_kl,
    sample_heatmap_candidates,
    score_candidate_distributional,
)


def _action_heatmap():
    return {
        "values": [0.05, 0.2, 0.65, 0.1, 0.03],
        "actions": ["open", "lift", "place", "push", "idle"],
        "target_indices": [0, 1, 2, 0, 1],
    }


def _target_heatmap():
    return {"values": [0.1, 0.2, 0.7]}


def test_kl_zero_for_identical_heatmaps():
    heatmap = [0.1, 0.7, 0.2]
    assert heatmap_kl(heatmap, heatmap) == pytest.approx(0.0, abs=1e-9)


def test_kl_positive_for_different_heatmaps():
    assert heatmap_kl([0.8, 0.1, 0.1], [0.1, 0.8, 0.1]) > 0.0


def test_js_finite_and_symmetric():
    first = [0.7, 0.2, 0.1]
    second = [0.1, 0.2, 0.7]
    first_to_second = heatmap_js(first, second)
    second_to_first = heatmap_js(second, first)
    assert math.isfinite(first_to_second)
    assert first_to_second == pytest.approx(second_to_first)


def test_entropy_finite():
    entropy = heatmap_entropy([0.2, 0.3, 0.5])
    assert math.isfinite(entropy)
    assert entropy > 0.0


def test_candidate_scoring_works_without_privileged_metadata():
    heatmap = _action_heatmap()
    candidate = sample_heatmap_candidates(heatmap, K=4)[0]
    score = score_candidate_distributional(
        candidate,
        full_action_heatmap=heatmap,
        full_target_heatmap=_target_heatmap(),
    )
    assert isinstance(score, float)
    assert math.isfinite(score)


def test_candidate_scoring_works_with_masked_heatmap():
    heatmap = _action_heatmap()
    masked = {"values": [0.2, 0.2, 0.2, 0.2, 0.2]}
    candidate = sample_heatmap_candidates(heatmap, K=4)[0]
    score = score_candidate_distributional(
        candidate,
        full_action_heatmap=heatmap,
        masked_action_heatmap=masked,
        full_target_heatmap=_target_heatmap(),
    )
    assert math.isfinite(score)


def test_candidate_scoring_works_with_multiple_negative_heatmaps():
    heatmap = _action_heatmap()
    negatives = [
        {"values": [0.6, 0.1, 0.1, 0.1, 0.1]},
        {"values": [0.05, 0.7, 0.05, 0.1, 0.1]},
    ]
    negative_targets = [
        {"values": [0.7, 0.2, 0.1]},
        {"values": [0.2, 0.7, 0.1]},
    ]
    candidate = sample_heatmap_candidates(heatmap, K=4)[0]
    score = score_candidate_distributional(
        candidate,
        full_action_heatmap=heatmap,
        negative_action_heatmaps=negatives,
        full_target_heatmap=_target_heatmap(),
        negative_target_heatmaps=negative_targets,
    )
    assert math.isfinite(score)


def test_default_k4_selection_works_on_dummy_heatmaps():
    result = distributional_tca_select_inference(
        _action_heatmap(),
        target_heatmap=_target_heatmap(),
        masked_action_heatmap={"values": [0.2, 0.2, 0.2, 0.2, 0.2]},
        negative_action_heatmaps=[{"values": [0.7, 0.1, 0.1, 0.05, 0.05]}],
    )
    assert result["K"] == 4
    assert len(result["candidates"]) == 4
    assert result["selected"] is not None
    assert result["external_verifier_used"] is False
    assert result["privileged_inference_used"] is False


def test_no_external_verifier_path_is_required():
    result = distributional_tca_select_inference(_action_heatmap(), target_heatmap=_target_heatmap())
    assert result["selected"] is not None

    with pytest.raises(ValueError, match="external verifiers"):
        distributional_tca_select_inference(
            _action_heatmap(),
            target_heatmap=_target_heatmap(),
            external_verifier=object(),
        )
