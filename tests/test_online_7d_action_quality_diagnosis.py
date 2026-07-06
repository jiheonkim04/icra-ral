import json
from argparse import Namespace
from pathlib import Path

import pytest

from tca_map.smolvla.online_7d_action_quality_diagnosis import build_action_quality_report


def _write_demo(path: Path, offset: float, target_bias: float = 0.0) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = 16
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(16, 7), dtype="f4")
        obs = demo.create_group("obs")
        ee_states = obs.create_dataset("ee_states", shape=(16, 6), dtype="f4")
        for step in range(16):
            phase = step / 15.0
            actions[step, :] = [
                offset + 0.5 * phase,
                offset - 0.25 * phase,
                target_bias + 0.125 * phase,
                0.05 * phase,
                -0.1 * phase,
                0.15 * phase,
                -1.0 if step < 10 else 1.0,
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
    train_positive = tmp_path / "train_positive.hdf5"
    train_counter = tmp_path / "train_counter.hdf5"
    _write_demo(rollout, 0.0, 0.0)
    _write_demo(rollout_cf, 0.6, 1.0)
    _write_demo(train_positive, 0.2, 0.0)
    _write_demo(train_counter, 0.9, 1.0)
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
                        "pair_id": "libero_10:train_positive__vs__train_counter",
                        "suite": "libero_10",
                        "positive_task_id": "train_positive_task",
                        "counterfactual_task_id": "train_counter_task",
                        "positive_instruction": "turn on the stove near the moka pot",
                        "counterfactual_instruction": "place the black bowl in the drawer",
                        "positive_demo_file": str(train_positive),
                        "counterfactual_demo_file": str(train_counter),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest, rollout


def _fake_rollout_report(path: Path) -> None:
    def variant(name: str, score: float, l2: float) -> dict:
        return {
            "variant": name,
            "reward_sum": 0.0,
            "final_success": False,
            "valid_closed_loop_online_rollout": name not in {"hdf5_expert_replay_exact_init"},
            "target_directed_movement_score": score,
            "action_stats": {
                "gripper_range": {"min": -1.0, "max": -0.5, "mean": -0.75},
                "rotation_range": {"max_abs": 0.2},
                "translation_range": {"max_abs": 0.9},
            },
            "expert_match": {"mean_l2": l2},
            "action_provenance": [
                {"l2_to_hdf5_expert_same_timestep": l2 + 0.01 * step, "uses_future_hdf5_action": False}
                for step in range(6)
            ],
        }

    path.write_text(
        json.dumps(
            {
                "decision": "bounded_online_7d_head_rollout_completed",
                "policy": {
                    "rollout_happened": True,
                    "heavy_model_imports_performed": True,
                    "model_load_performed": True,
                    "model_inference_performed": True,
                },
                "rollout_results": [
                    variant("native_smolvla_online_policy", -0.2, 1.5),
                    variant("hdf5_expert_replay_exact_init", -0.01, 0.0),
                    variant("actionmap_7d", -0.02, 1.0),
                    variant("fixed_prior_tca_7d", -0.01, 0.99),
                    variant("hard_learned_target_tca_7d", -0.03, 1.2),
                ],
            }
        ),
        encoding="utf-8",
    )


def _args(manifest: Path, rollout_report: Path, tmp_path: Path) -> Namespace:
    return Namespace(
        manifest=str(manifest),
        online_7d_report=str(rollout_report),
        report_json=str(tmp_path / "report.json"),
        report_md=str(tmp_path / "report.md"),
        max_steps=6,
        train_max_steps=10,
        sample_stride=2,
        teacher_max_steps=12,
    )


def test_action_quality_report_contains_required_diagnostics(tmp_path, monkeypatch):
    manifest, rollout_demo = _write_manifest(tmp_path)
    rollout_report = tmp_path / "online_rollout.json"
    _fake_rollout_report(rollout_report)
    monkeypatch.delenv("ALLOW_DOWNLOADS", raising=False)
    monkeypatch.delenv("ALLOW_GPU_TRAINING", raising=False)
    monkeypatch.delenv("ALLOW_OPENVLA_OFT", raising=False)

    report = build_action_quality_report(_args(manifest, rollout_report, tmp_path))

    assert report["decision"] == "online_7d_action_quality_diagnosis_completed"
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["loss_computed"] is True
    assert report["policy"]["lora_training_performed"] is False
    assert report["policy"]["rollout_happened"] is True
    assert report["data"]["rollout_demo_path"] == str(rollout_demo)
    assert report["data"]["rollout_demo_excluded_from_training"] is True

    diff = report["action_difference_audit"]["actionmap_vs_fixed_prior_tca"]
    assert "per_step_action_l2" in diff
    assert "meaningfully_different" in diff
    assert report["supervised_action_quality_breakdown"]["mean_action_baseline"]["metrics"]["sample_count"] == 6
    assert "fixed_prior_tca_7d" in report["teacher_forced_trajectory_diagnostic"]["variants"]
    assert report["closed_loop_failure_diagnosis"]["fixed_prior_tca_valid_rollout_support"] is False
    assert report["closed_loop_failure_diagnosis"]["fixed_prior_tca_partial_target_movement_support"] is True
    assert report["conclusion"]["recommended_next_milestone"] in {
        "A. bounded improved-head matched-init rollout",
        "B. gripper/rotation calibration",
        "C. target-prior conditioning redesign",
        "D. paper-readiness package with honest rollout caveat",
    }


def test_action_quality_report_refuses_forbidden_download_gate(tmp_path, monkeypatch):
    manifest, _rollout_demo = _write_manifest(tmp_path)
    rollout_report = tmp_path / "missing.json"
    monkeypatch.setenv("ALLOW_DOWNLOADS", "1")

    report = build_action_quality_report(_args(manifest, rollout_report, tmp_path))

    assert report["result"]["passed"] is False
    assert "ALLOW_DOWNLOADS" in report["result"]["blocked_reason"]
