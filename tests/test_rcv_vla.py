import hashlib

import numpy as np
import pytest

from tca_map.smolvla.rcv_vla import (
    RCVConfig,
    action_disagreement,
    assert_no_privileged_inference_fields,
    build_rcv_features,
    load_verifier,
    predict_replan_probability,
    save_verifier,
    task_one_hot,
    train_verifier,
)


def _record(index: int, *, split: str) -> dict[str, object]:
    state_flag = 1.0 if index % 2 else -1.0
    disagreement = 0.9 if state_flag > 0 else 0.1
    return {
        "split": split,
        "state": [state_flag] + [0.0] * 7,
        "queued_action": [0.2] * 7,
        "previous_action": [0.0] * 7,
        "chunk_index_fraction": 0.25,
        "task_key": "libero_spatial/task_4",
        "disagreement": disagreement,
    }


def test_feature_shapes_and_task_one_hot() -> None:
    kwargs = {
        "state": [0.0] * 8,
        "queued_action": [0.1] * 7,
        "previous_action": [0.0] * 7,
        "chunk_index_fraction": 0.5,
        "task_key": "libero_10/task_4",
    }

    full = build_rcv_features(**kwargs, include_context=True)
    no_context = build_rcv_features(**kwargs, include_context=False)

    assert full.shape == (25,)
    assert no_context.shape == (10,)
    assert np.allclose(task_one_hot("libero_10/task_4"), [0.0, 1.0])


def test_action_disagreement_is_l1_mean() -> None:
    queued = [0.0, 1.0, -1.0, 0.5, 0.0, 0.0, 0.0]
    fresh = [1.0, 1.0, -0.5, -0.5, 0.0, 0.0, 0.0]

    assert action_disagreement(queued, fresh) == pytest.approx((1.0 + 0.0 + 0.5 + 1.0) / 7.0)


def test_privileged_verifier_fields_rejected() -> None:
    with pytest.raises(ValueError):
        assert_no_privileged_inference_fields(["observation.images.camera1", "queued_action"])


def test_train_verifier_uses_context_when_context_is_predictive() -> None:
    records = [_record(index, split="train") for index in range(20)]
    records += [_record(index, split="calibration") for index in range(20, 30)]
    config = RCVConfig(disagreement_quantile=0.5, max_epochs=250, learning_rate=0.1, seed=13)

    full = train_verifier(records, include_context=True, config=config)
    no_context = train_verifier(records, include_context=False, config=config, tau_train=full["tau_train"])

    assert full["tau_train"] == pytest.approx(0.5)
    assert full["calibration_metrics"]["balanced_accuracy"] > 0.95
    assert no_context["calibration_metrics"]["balanced_accuracy"] <= 0.55
    probability = predict_replan_probability(
        full,
        state=[1.0] + [0.0] * 7,
        queued_action=[0.2] * 7,
        previous_action=[0.0] * 7,
        chunk_index_fraction=0.25,
        task_key="libero_spatial/task_4",
    )
    assert probability > full["theta_train"]


def test_verifier_checkpoint_round_trips(tmp_path) -> None:
    records = [_record(index, split="train") for index in range(12)]
    records += [_record(index, split="calibration") for index in range(12, 18)]
    verifier = train_verifier(records, include_context=True, config=RCVConfig(max_epochs=50))
    path = tmp_path / "verifier.json"

    save_verifier(path, verifier)
    loaded = load_verifier(path)

    assert loaded["schema_version"] == "rcv_logistic_verifier_v1"
    assert loaded["weights"] == verifier["weights"]


def test_preregistered_proposal_hash_matches_file() -> None:
    proposal = "reports/rcv_vla/researcher_proposal.md"
    digest = hashlib.sha256(open(proposal, "rb").read()).hexdigest().upper()

    assert digest == "86044E841D178DB5AA485B7D12B01FF8E4274CBDFDCDAC7D427477BF0646F26F"
