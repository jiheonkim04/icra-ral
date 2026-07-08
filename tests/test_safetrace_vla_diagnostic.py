import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tca_map.safetrace_vla.diagnostic import build_safetrace_vla_diagnostic


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "220_safetrace_vla_diagnostic.ps1"


def _write_demo(path: Path, *, offset: float) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = 52
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["language"] = "put the mug on the plate while avoiding the spoon"
        actions = demo.create_dataset("actions", shape=(steps, 7), dtype="f4")
        dones = demo.create_dataset("dones", shape=(steps,), dtype="u1")
        rewards = demo.create_dataset("rewards", shape=(steps,), dtype="u1")
        obs = demo.create_group("obs")
        ee_pos = obs.create_dataset("ee_pos", shape=(steps, 3), dtype="f4")
        grip = obs.create_dataset("gripper_states", shape=(steps, 2), dtype="f4")
        mug = obs.create_dataset("mug_pos", shape=(steps, 3), dtype="f4")
        plate = obs.create_dataset("plate_pos", shape=(steps, 3), dtype="f4")
        spoon = obs.create_dataset("spoon_pos", shape=(steps, 3), dtype="f4")
        dest = np.asarray([0.72 + offset, 0.18, 0.90], dtype=np.float64)
        safety = np.asarray([0.36 + offset, 0.08, 0.93], dtype=np.float64)
        for step in range(steps):
            phase = step / float(steps - 1)
            eef = np.asarray([0.12 + 0.58 * phase + offset, 0.02 + 0.17 * phase, 0.96], dtype=np.float64)
            closed = 24 <= step < 40
            released = step >= 40
            if step < 10:
                obj = np.asarray([0.18 + offset, 0.02, 0.90], dtype=np.float64)
            elif step < 24:
                obj = eef + np.asarray([0.0, 0.0, -0.04], dtype=np.float64)
            elif step < 40:
                obj = eef + np.asarray([0.0, 0.0, -0.035], dtype=np.float64)
            else:
                obj = np.asarray([0.58 + offset, 0.12, 0.88], dtype=np.float64)
            target = dest if closed else obj
            trans = 0.45 * (target - eef) + np.asarray([0.01, 0.0, 0.0])
            actions[step, :] = [trans[0], trans[1], trans[2], 0.0, 0.0, 0.0, 1.0 if closed else -1.0]
            ee_pos[step, :] = eef
            grip[step, :] = [0.01, -0.01] if closed else [0.04, -0.04]
            mug[step, :] = obj
            plate[step, :] = dest
            spoon[step, :] = safety
        dones[-1] = 1
        rewards[-1] = 1


def test_safetrace_vla_diagnostic_kills_when_safety_only_matches(tmp_path):
    data_root = tmp_path / "libero"
    _write_demo(data_root / "libero_10" / "demo_a.hdf5", offset=0.0)
    _write_demo(data_root / "libero_10" / "demo_b.hdf5", offset=0.04)

    report = build_safetrace_vla_diagnostic(
        libero_data_root=data_root,
        libero_root=tmp_path / "LIBERO",
        max_demos=2,
        max_action_steps=48,
        chunk=8,
    )

    assert report["schema_version"] == "safetrace-vla-state1-diagnostic-v1"
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["decision"]["real_temporal_metric_produced"] is True
    assert report["temporal_metrics"]["nonzero_violations_or_risk_exposure"] is True
    assert report["preference_pairs"]["valid_pair_count"] > 0
    assert report["preference_pairs"]["nontrivial_pair_count"] > 0
    assert report["decision"]["final_output"] == "KILL"
    assert report["decision"]["safety_only_matches_safetrace"] is True


def test_safetrace_vla_runner_outputs_json(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for SafeTrace-VLA runner tests")
    data_root = tmp_path / "libero"
    _write_demo(data_root / "libero_10" / "demo_a.hdf5", offset=0.0)
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
            "-LiberoRoot",
            str(tmp_path / "LIBERO"),
            "-MaxDemos",
            "1",
            "-MaxActionSteps",
            "44",
            "-JsonReportPath",
            str(report_json),
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

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report_json.read_text(encoding="utf-8-sig"))
    assert data["schema_version"] == "safetrace-vla-state1-diagnostic-v1"
    assert data["decision"]["final_output"] == "KILL"

