import numpy as np
import pytest

from tca_map.smolvla.dicd_vla import (
    DICDConfig,
    assert_no_privileged_inference_fields,
    build_dicd_examples,
    direct_chunk_index_action,
    file_sha256,
    load_dicd_checkpoint,
    make_dicd_features,
    predict_dicd_action,
    save_dicd_checkpoint,
    train_dicd_adapter,
)


def _chunk(offset: float = 0.0) -> np.ndarray:
    rows = []
    for index in range(8):
        rows.append([offset + index * 0.1, 0.2, -0.1, 0.3, 0.0, -0.2, -1.0])
    return np.asarray(rows, dtype=np.float32)


def test_direct_chunk_index_clamps_to_available_chunk() -> None:
    config = DICDConfig()
    chunk = _chunk()

    assert np.allclose(direct_chunk_index_action(chunk, 2, config), chunk[2].reshape(1, -1))
    assert np.allclose(direct_chunk_index_action(chunk[:3], 99, config), chunk[2].reshape(1, -1))


def test_features_encode_history_when_enabled() -> None:
    config = DICDConfig()
    chunk = _chunk()
    history = [np.ones(7, dtype=np.float32), np.ones(7, dtype=np.float32) * 2.0]

    with_history = make_dicd_features(chunk, history=history, delay=2, step_fraction=0.5, config=config)
    without_history = make_dicd_features(chunk, history=history, delay=2, step_fraction=0.5, config=config, use_history=False)

    assert len(with_history) == config.input_dim
    assert len(without_history) == config.input_dim
    assert not np.allclose(with_history, without_history)


def test_training_has_gradients_and_decreasing_loss() -> None:
    config = DICDConfig(hidden_dim=32)
    chunks = [_chunk(index * 0.03) for index in range(12)]
    executed = [direct_chunk_index_action(chunk, 2, config).reshape(-1) + 0.05 for chunk in chunks]
    examples = build_dicd_examples(chunks, executed, delay=2, config=config)

    model, stats = train_dicd_adapter(examples, config=config, epochs=80, lr=3e-3, seed=7)

    assert stats["finite_gradients"] is True
    assert stats["max_grad_norm"] > 0.0
    assert stats["loss_decreased"] is True
    assert stats["final_loss"] < stats["initial_loss"]
    assert predict_dicd_action(model, examples[0].features).shape == (1, 7)


def test_checkpoint_roundtrip_preserves_predictions(tmp_path) -> None:
    config = DICDConfig(hidden_dim=16)
    chunks = [_chunk(index * 0.02) for index in range(8)]
    executed = [direct_chunk_index_action(chunk, 1, config).reshape(-1) for chunk in chunks]
    examples = build_dicd_examples(chunks, executed, delay=1, config=config)
    model, stats = train_dicd_adapter(examples, config=config, epochs=20, lr=3e-3, seed=3)
    before = predict_dicd_action(model, examples[2].features)
    checkpoint = tmp_path / "dicd.pt"

    save_dicd_checkpoint(checkpoint, model, stats)
    loaded, loaded_stats = load_dicd_checkpoint(checkpoint)
    after = predict_dicd_action(loaded, examples[2].features)

    assert file_sha256(checkpoint)
    assert loaded_stats["example_count"] == stats["example_count"]
    assert np.allclose(before, after)


def test_privileged_inference_fields_are_rejected() -> None:
    assert_no_privileged_inference_fields(["observation", "instruction", "action_chunk", "delay"])

    with pytest.raises(ValueError):
        assert_no_privileged_inference_fields(["observation", "sim_state"])
