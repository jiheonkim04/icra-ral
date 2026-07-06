import json
from argparse import Namespace
from pathlib import Path

import pytest

from tca_map.smolvla.online_7d_diagnostic_head import (
    _classify_fixed_prior_rollout_support,
    build_report,
    readiness_gate,
    train_online_7d_heads,
)


def _write_demo(path: Path, offset: float, target_bias: float = 0.0) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = 12
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(12, 7), dtype="f4")
        obs = demo.create_group("obs")
        ee_states = obs.create_dataset("ee_states", shape=(12, 6), dtype="f4")
        for step in range(12):
            phase = step / 11.0
            actions[step, :] = [
                offset + phase,
                offset - 0.5 * phase,
                target_bias + 0.25 * phase,
                0.1 * phase,
                -0.2 * phase,
                0.3 * phase,
                -1.0 if step < 8 else 1.0,
            ]
            ee_states[step, :] = [
                offset + 0.01 * step,
                target_bias + 0.02 * step,
                0.03 * step,
                0.04 * step,
                -0.05 * step,
                0.06 * step,
            ]


def _write_manifest(tmp_path: Path) -> tuple[Path, Path]:
    rollout = tmp_path / "rollout_positive.hdf5"
    rollout_cf = tmp_path / "rollout_counterfactual.hdf5"
    train_counter = tmp_path / "train_counter.hdf5"
    train_positive = tmp_path / "train_positive.hdf5"
    train_counter_2 = tmp_path / "train_counter_2.hdf5"
    _write_demo(rollout, 0.0, 0.0)
    _write_demo(rollout_cf, 0.4, 1.0)
    _write_demo(train_counter, 0.8, 1.0)
    _write_demo(train_positive, 0.2, 0.0)
    _write_demo(train_counter_2, 1.0, 1.0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "counterfactual_pairs": [
                    {
                        "pair_id": "libero_10:rollout__vs__rollout_cf",
                        "suite": "libero_10",
                        "positive_task_id": "rollout_task",
                        "counterfactual_task_id": "rollout_cf_task",
                        "positive_instruction": "put the moka pot on the stove",
                        "counterfactual_instruction": "put the black bowl in the drawer",
                        "positive_demo_file": str(rollout),
                        "counterfactual_demo_file": str(rollout_cf),
                    },
                    {
                        "pair_id": "libero_10:leaky_rollout_path__vs__train_counter",
                        "suite": "libero_10",
                        "positive_task_id": "rollout_task_duplicate",
                        "counterfactual_task_id": "train_counter_task",
                        "positive_instruction": "put the moka pot on the stove",
                        "counterfactual_instruction": "put the black bowl in the drawer",
                        "positive_demo_file": str(rollout),
                        "counterfactual_demo_file": str(train_counter),
                    },
                    {
                        "pair_id": "libero_10:train_positive__vs__train_counter_2",
                        "suite": "libero_10",
                        "positive_task_id": "train_positive_task",
                        "counterfactual_task_id": "train_counter_2_task",
                        "positive_instruction": "turn on the stove near the moka pot",
                        "counterfactual_instruction": "place the black bowl in the drawer",
                        "positive_demo_file": str(train_positive),
                        "counterfactual_demo_file": str(train_counter_2),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest, rollout


def _args(manifest: Path, json_report: Path, md_report: Path) -> Namespace:
    return Namespace(
        manifest=str(manifest),
        report_json=str(json_report),
        report_md=str(md_report),
        smolvla_ckpt="unused",
        checkpoint_root="unused",
        hf_home="unused",
        libero_root="unused",
        robosuite_root="unused",
        max_steps=6,
        train_max_steps=8,
        sample_stride=2,
        camera_size=64,
        device="cpu",
    )


def test_train_online_7d_heads_excludes_rollout_demo_from_training(tmp_path):
    manifest, rollout = _write_manifest(tmp_path)

    models, meta = train_online_7d_heads(manifest, max_steps=6, train_max_steps=8, stride=2)

    assert meta["rollout_demo_excluded_from_training"] is True
    assert str(rollout) not in meta["train_demo_paths"]
    assert meta["eval_demo_paths"] == [str(rollout)]
    assert meta["train_sample_count"] > 0
    assert meta["eval_sample_count"] == 6
    gate = readiness_gate(models)
    assert gate["status"] == "green"
    assert gate["ready_for_bounded_matched_init_rollout"] is True
    for variant in ("actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d"):
        assert models[variant]["offline_metrics"]["action_dim"] == 7
        assert models[variant]["leakage_audit"]["uses_same_or_future_hdf5_action_at_inference"] is False


def test_build_report_without_rollout_gate_trains_but_does_not_rollout(tmp_path, monkeypatch):
    manifest, rollout = _write_manifest(tmp_path)
    monkeypatch.delenv("ALLOW_ONLINE_7D_DIAGNOSTIC_HEAD_ROLLOUT", raising=False)
    report = build_report(_args(manifest, tmp_path / "report.json", tmp_path / "report.md"))

    assert report["decision"] == "trained_7d_heads_rollout_gate_not_set"
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["loss_computed"] is True
    assert report["policy"]["rollout_happened"] is False
    assert report["training"]["rollout_demo_excluded_from_training"] is True
    assert str(rollout) not in report["training"]["train_demo_paths"]
    assert report["rollout_readiness_gate"]["status"] == "green"
    assert report["result"]["fixed_prior_tca_valid_rollout_support"] is False


def test_support_classifier_keeps_partial_target_movement_below_valid_support():
    actionmap = {
        "valid_closed_loop_online_rollout": True,
        "reward_sum": 0.0,
        "final_success": False,
        "target_directed_movement_score": -0.02,
    }
    fixed = {
        "valid_closed_loop_online_rollout": True,
        "reward_sum": 0.0,
        "final_success": False,
        "target_directed_movement_score": -0.01,
    }

    support = _classify_fixed_prior_rollout_support(actionmap, fixed)

    assert support["fixed_prior_tca_valid_rollout_support"] is False
    assert support["fixed_prior_tca_partial_target_movement_support"] is True
    assert support["blocker_classification"] == "online_7d_head_partial_target_movement_no_success"


def test_support_classifier_accepts_reward_or_success_gain_only():
    actionmap = {"valid_closed_loop_online_rollout": True, "reward_sum": 0.0, "final_success": False}
    fixed = {"valid_closed_loop_online_rollout": True, "reward_sum": 1.0, "final_success": False}

    support = _classify_fixed_prior_rollout_support(actionmap, fixed)

    assert support["fixed_prior_tca_valid_rollout_support"] is True
    assert support["fixed_prior_tca_partial_target_movement_support"] is False
    assert support["blocker_classification"] is None
