import numpy as np

from tca_map.rl4il_prior.action_oracle import (
    ActionOracleConfig,
    action_sequence_distance,
    oracle_index_for_candidates,
    pairwise_action_distance_matrix,
    resample_action_sequence,
)


def test_resample_action_sequence_preserves_rank_and_endpoints():
    actions = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    out = resample_action_sequence(actions, steps=5)
    assert out.shape == (5, 2)
    np.testing.assert_allclose(out[0], actions[0])
    np.testing.assert_allclose(out[-1], actions[-1])


def test_action_distance_zero_for_identical_sequences():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    assert action_sequence_distance(actions, actions) == 0.0


def test_action_distance_uses_length_penalty_for_same_constant_motion():
    a = np.ones((4, 2), dtype=np.float32)
    b = np.ones((8, 2), dtype=np.float32)
    dist = action_sequence_distance(a, b, ActionOracleConfig(resample_steps=4, length_penalty_weight=0.5))
    assert dist > 0.0


def test_pairwise_matrix_is_symmetric_and_oracle_excludes_self():
    actions = [
        np.zeros((4, 2), dtype=np.float32),
        np.ones((4, 2), dtype=np.float32) * 0.1,
        np.ones((4, 2), dtype=np.float32) * 5.0,
    ]
    matrix = pairwise_action_distance_matrix(actions)
    np.testing.assert_allclose(matrix, matrix.T)
    assert oracle_index_for_candidates(0, [0, 1, 2], matrix) == 1
