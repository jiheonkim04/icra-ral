import json
from argparse import Namespace
from pathlib import Path

import pytest

from tca_map.smolvla.online_7d_head_redesign_gate import build_redesign_gate_report


def _write_demo(path: Path, offset: float, target_bias: float = 0.0) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = 18
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(18, 7), dtype="f4")
        obs = demo.create_group("obs")
        ee_states = obs.create_dataset("ee_states", shape=(18, 6), dtype="f4")
        for step in range(18):
            phase = step / 17.0
            actions[step, :] = [
                offset + 0.45 * phase,
                offset - 0.20 * phase,
                target_bias + 0.10 * phase,
                0.04 * phase,
                -0.07 * phase,
                0.12 * phase,
                -1.0 if step < 11 else 1.0,
            ]
            ee_states[step, :] = [
                offset + 0.01 * step,
                target_bias + 0.02 * step,
                0.03 * step,
                0.04 * step,
                -0.05 * step,
                0.06 * step,
            ]


def _write_manifest(tmp_path: Path) -> Path:
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
    return manifest


def _args(manifest: Path, tmp_path: Path) -> Namespace:
    return Namespace(
        manifest=str(manifest),
        report_json=str(tmp_path / "report.json"),
        report_md=str(tmp_path / "report.md"),
        max_steps=6,
        train_max_steps=10,
        sample_stride=2,
        teacher_max_steps=12,
        mlp_steps=8,
    )


def test_redesign_gate_report_contains_required_variants_and_gate(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    monkeypatch.delenv("ALLOW_DOWNLOADS", raising=False)
    monkeypatch.delenv("ALLOW_OPENVLA_OFT", raising=False)

    report = build_redesign_gate_report(_args(manifest, tmp_path))

    assert report["decision"] == "bounded_7d_head_redesign_gate_completed"
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["loss_computed"] is True
    assert report["policy"]["lora_training_performed"] is False
    assert report["policy"]["rollout_happened"] is False
    assert report["data"]["rollout_demo_excluded_from_training"] is True
    variants = report["stage2_head_variants"]
    for name in [
        "mean_action_baseline",
        "actionmap_7d",
        "fixed_prior_tca_7d",
        "normalized_fixed_prior_tca_7d",
        "split_fixed_prior_tca_7d",
        "small_cpu_mlp_fixed_prior_tca_7d",
        "fixed_prior_tca_mean_residual_7d",
        "phase_aware_fixed_prior_tca_7d",
    ]:
        assert name in variants
    assert variants["native_smolvla_learned_residual_7d"]["status"] == "not_evaluated"
    assert "why_mean_baseline_beats_previous_heads" in report["stage1_mean_baseline_diagnosis"]
    assert report["stage4_rollout_gate"]["status"] in {"red", "green"}
    assert report["conclusion"]["recommended_next_milestone"] in {
        "A. bounded improved-head matched-init rollout",
        "B. gripper/rotation calibration",
        "C. target-prior conditioning redesign",
        "D. paper-readiness package with honest rollout caveat",
    }


def test_redesign_gate_refuses_forbidden_download_gate(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    monkeypatch.setenv("ALLOW_DOWNLOADS", "1")

    report = build_redesign_gate_report(_args(manifest, tmp_path))

    assert report["result"]["passed"] is False
    assert "ALLOW_DOWNLOADS" in report["result"]["blocked_reason"]
