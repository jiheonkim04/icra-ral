import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from tca_map.smolvla import libero_action_stat_subset_audit as audit


def _write_plan(path: Path):
    path.write_text(
        json.dumps(
            {
                "ready_for_libero_action_stat_audit": True,
                "audit_summary": {
                    "action_mean_range": {"max_abs": 120.0},
                    "action_std_range": {"max": 50.0},
                    "action_stat_prefixes": ["so100"],
                    "policy_action_shape": [6],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_hdf5(path: Path):
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset(
            "actions",
            data=np.asarray([[0.0, 0.1, -0.1, 0.0, 0.0, 0.0, -1.0], [0.5, -0.5, 0.2, 0.0, 0.1, -0.1, 1.0]], dtype=np.float32),
        )


def _args(tmp_path: Path):
    plan = tmp_path / "plan.json"
    data = tmp_path / "data"
    _write_plan(plan)
    _write_hdf5(data / "libero_10" / "demo.hdf5")
    return argparse.Namespace(
        plan_report=str(plan),
        libero_data_root=str(data),
        max_files=5,
        max_actions_per_file=500,
        report_path=str(tmp_path / "report.json"),
        markdown_report_path=str(tmp_path / "report.md"),
    )


def test_libero_action_stat_audit_confirms_scale_and_dim_mismatch(tmp_path, monkeypatch):
    for gate in audit.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = audit.build_report(_args(tmp_path))

    assert code == 0
    assert report["libero_action_stat_subset_audit_passed"] is True
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["comparison_to_checkpoint"]["scale_mismatch_confirmed"] is True
    assert report["comparison_to_checkpoint"]["dimension_mismatch_confirmed"] is True
    assert report["ready_for_rollout_scaling"] is False


def test_libero_action_stat_audit_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report, code = audit.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUTS" in report["recommended_next_step"]
    assert report["policy"]["rollouts_performed"] is False


def test_libero_action_stat_audit_requires_plan_authorization(tmp_path, monkeypatch):
    for gate in audit.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    Path(args.plan_report).write_text(json.dumps({"ready_for_libero_action_stat_audit": False}), encoding="utf-8")

    report, code = audit.build_report(args)

    assert code != 0
    assert report["decision"] == "stop"
    assert "did not authorize" in report["recommended_next_step"]
