import pytest

from tca_map.inference import (
    sample_heatmap_candidates,
    score_candidate_target_consistency,
    tca_select_inference,
)


def dummy_action_heatmap():
    return {
        "values": [0.1, 0.8, 0.2, 0.6, 0.05],
        "actions": [
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.0, 1.0],
            [0.0, 0.1, 0.0, 1.0],
            [0.1, 0.1, 0.0, 1.0],
            [0.0, 0.0, 0.1, 1.0],
        ],
        "target_indices": [0, 1, 0, 1, 2],
    }


def dummy_target_heatmap():
    return {"scores": [0.1, 1.0, 0.2], "top_index": 1}


def test_k_candidate_sampling_works():
    candidates = sample_heatmap_candidates(dummy_action_heatmap(), K=4, temperature=0.5)
    assert len(candidates) == 4
    assert candidates[0]["index"] == 1
    assert all("probability" in candidate for candidate in candidates)


def test_no_privileged_metadata_required_for_consistency_score():
    candidate = sample_heatmap_candidates(dummy_action_heatmap(), K=1)[0]
    score = score_candidate_target_consistency(candidate, dummy_action_heatmap(), dummy_target_heatmap())
    assert score > 0


def test_privileged_metadata_is_rejected():
    candidate = sample_heatmap_candidates(dummy_action_heatmap(), K=1)[0]
    with pytest.raises(ValueError):
        score_candidate_target_consistency(
            candidate,
            dummy_action_heatmap(),
            dummy_target_heatmap(),
            metadata={"simulator_state": {"object_pose": [0, 0, 0]}},
        )


def test_default_k4_tca_select_path_works_on_dummy_heatmaps():
    result = tca_select_inference(
        action_heatmap=dummy_action_heatmap(),
        target_heatmap=dummy_target_heatmap(),
        full_heatmap=dummy_action_heatmap(),
        masked_heatmap={"values": [0.1, 0.2, 0.2, 0.2, 0.05]},
    )
    assert result["K"] == 4
    assert result["temperature"] == 0.5
    assert result["selected"] is not None
    assert result["external_verifier_used"] is False
    assert result["privileged_inference_used"] is False


def test_external_verifier_is_rejected():
    with pytest.raises(ValueError):
        tca_select_inference(
            action_heatmap=dummy_action_heatmap(),
            target_heatmap=dummy_target_heatmap(),
            external_verifier=lambda candidate: 1.0,
        )
