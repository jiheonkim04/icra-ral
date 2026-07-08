import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tca_map.contactset_vla.diagnostic import VARIANTS, build_contactset_vla_diagnostic


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "200_contactset_vla_diagnostic.ps1"


def _write_demo(path: Path, *, offset: float, safety_y: float) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = 56
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["language"] = "put the mug on the plate while avoiding the spoon"
        actions = demo.create_dataset("actions", shape=(steps, 7), dtype="f4")
        states = demo.create_dataset("states", shape=(steps, 12), dtype="f4")
        obs = demo.create_group("obs")
        ee_pos = obs.create_dataset("ee_pos", shape=(steps, 3), dtype="f4")
        ee_ori = obs.create_dataset("ee_ori", shape=(steps, 3), dtype="f4")
        gripper = obs.create_dataset("gripper_states", shape=(steps, 2), dtype="f4")
        mug = obs.create_dataset("mug_pos", shape=(steps, 3), dtype="f4")
        plate = obs.create_dataset("plate_pos", shape=(steps, 3), dtype="f4")
        spoon = obs.create_dataset("spoon_pos", shape=(steps, 3), dtype="f4")
        source = np.asarray([0.25 + offset, 0.02, 0.92], dtype=np.float64)
        dest = np.asarray([0.52 + offset, 0.18, 0.92], dtype=np.float64)
        safety = np.asarray([0.38 + offset, safety_y, 0.92], dtype=np.float64)
        for step in range(steps):
            phase = step / float(steps - 1)
            closed = step >= steps // 2
            eef = np.asarray([0.12 + 0.42 * phase + offset, 0.02 + 0.16 * phase, 1.00], dtype=np.float64)
            target = dest if closed else source
            avoid = eef - safety
            avoid[2] = 0.0
            avoid = avoid / max(0.05, np.linalg.norm(avoid))
            translation = 0.55 * (target - eef) + 0.10 * avoid + np.asarray([0.0, 0.0, 0.025 if closed else -0.01])
            actions[step, :] = [
                translation[0],
                translation[1],
                translation[2],
                0.02 * np.sin(phase),
                -0.01 * np.cos(phase),
                0.015 if closed else -0.015,
                1.0 if closed else -1.0,
            ]
            states[step, :] = 0.0
            ee_pos[step, :] = eef
            ee_ori[step, :] = [3.14, 0.0, 0.0]
            gripper[step, :] = [0.01, -0.01] if closed else [0.04, -0.04]
            mug[step, :] = source
            plate[step, :] = dest
            spoon[step, :] = safety


def test_contactset_vla_diagnostic_runs_required_variants(tmp_path):
    data_root = tmp_path / "libero"
    _write_demo(data_root / "libero_10" / "demo_a.hdf5", offset=0.00, safety_y=0.10)
    _write_demo(data_root / "libero_10" / "demo_b.hdf5", offset=0.04, safety_y=0.24)

    report = build_contactset_vla_diagnostic(
        libero_data_root=data_root,
        max_demos=2,
        max_action_steps=48,
        feature_width=32,
        ridge=1e-3,
    )

    assert report["schema_version"] == "contactset-vla-diagnostic-v1"
    assert report["policy"]["training_performed"] is True
    assert report["model"]["loss_computed"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert set(report["variants"]) == set(VARIANTS)
    assert report["data"]["source_object_points_observable"] is True
    assert report["data"]["destination_points_observable"] is True
    assert report["data"]["full_contact_set_observable"] is True
    assert report["data"]["uses_eval_label_leakage"] is False
    for payload in report["variants"].values():
        metrics = payload["metrics"]
        assert metrics["action_l2"] is not None
        assert metrics["translation_l2"] is not None
        assert metrics["rotation_l2"] is not None
        assert metrics["gripper_error"] is not None
        assert metrics["contact_placement_consistency"] is not None
    assert report["decision"]["decision"] in {"continue", "kill", "blocked"}
    assert "single_point_action_l2" in report["decision"]
    assert "contact_set_action_l2" in report["decision"]


def test_contactset_vla_runner_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for ContactSet-VLA runner tests")
    data_root = tmp_path / "libero"
    _write_demo(data_root / "libero_10" / "demo_a.hdf5", offset=0.00, safety_y=0.10)

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-LiberoDataRoot",
            str(data_root),
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 21
    assert "ALLOW_TINY_TRAINING=1" in (result.stdout + result.stderr)


def test_contactset_vla_runner_outputs_json(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for ContactSet-VLA runner tests")
    data_root = tmp_path / "libero"
    _write_demo(data_root / "libero_10" / "demo_a.hdf5", offset=0.00, safety_y=0.10)
    report_json = tmp_path / "report.json"

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-LiberoDataRoot",
            str(data_root),
            "-MaxDemos",
            "1",
            "-MaxActionSteps",
            "32",
            "-FeatureWidth",
            "32",
            "-JsonReportPath",
            str(report_json),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "ALLOW_TINY_TRAINING": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report_json.read_text(encoding="utf-8-sig"))
    assert data["schema_version"] == "contactset-vla-diagnostic-v1"
    assert set(data["variants"]) == set(VARIANTS)

