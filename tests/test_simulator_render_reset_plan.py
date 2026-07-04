import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "58_plan_simulator_render_reset.ps1"


def _run_planner(tmp_path, extra_env=None, import_report=None, render_report=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the simulator render/reset-step planner")

    json_report = tmp_path / "simulator_render_reset_plan_report.json"
    md_report = tmp_path / "simulator_render_reset_plan_report.md"
    import_report_path = tmp_path / "bounded_simulator_import_smoke_report.json"
    render_report_path = tmp_path / "bounded_simulator_render_smoke_report.json"
    if import_report is not None:
        import_report_path.write_text(json.dumps(import_report), encoding="utf-8")
    if render_report is not None:
        render_report_path.write_text(json.dumps(render_report), encoding="utf-8")

    libero_root = tmp_path / "LIBERO"
    libero_data_root = tmp_path / "libero_data"
    robosuite_root = tmp_path / "robosuite"
    data_root = tmp_path / "data"
    for path in (libero_root, libero_data_root, robosuite_root, data_root):
        path.mkdir()

    env = os.environ.copy()
    for key in (
        "ALLOW_SIMULATOR_RENDER_SMOKE",
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_TINY_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_TINY_TRAINING",
    ):
        env.pop(key, None)
    env.update(
        {
            "LIBERO_ROOT": str(libero_root),
            "LIBERO_DATA_ROOT": str(libero_data_root),
            "ROBOSUITE_ROOT": str(robosuite_root),
            "DATA_ROOT": str(data_root),
        }
    )
    env.update(extra_env or {})

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-RuntimePlatform",
            "linux",
            "-PathsFile",
            str(tmp_path / "missing_paths.local.yaml"),
            "-ImportSmokeReportPath",
            str(import_report_path),
            "-RenderSmokeReportPath",
            str(render_report_path),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    return result, json.loads(result.stdout[start:]), json_report, md_report


def test_render_reset_plan_stops_without_import_report(tmp_path):
    result, report, json_report, md_report = _run_planner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["policy"]["planning_only"] is True
    assert report["policy"]["render_smoke_performed"] is False
    assert report["policy"]["reset_step_smoke_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["ready_for_bounded_render_smoke_plan"] is False
    assert any("import smoke report" in reason for reason in report["stop_reasons"])
    assert json_report.exists()
    assert md_report.exists()


def test_render_reset_plan_allows_later_render_plan_after_import_pass(tmp_path):
    result, report, _, _ = _run_planner(
        tmp_path,
        import_report={"bounded_simulator_import_smoke_passed": True},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_bounded_render_smoke_plan"] is True
    assert report["ready_for_bounded_reset_step_smoke_plan"] is False
    assert report["ready_for_rollout"] is False
    assert report["policy"]["simulator_imports_performed"] is False
    assert report["policy"]["render_smoke_performed"] is False
    assert report["policy"]["simulator_environment_steps_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["risk_assessment"]["expected_vram_gb"] == 0


def test_render_reset_plan_allows_later_reset_step_plan_after_render_pass(tmp_path):
    result, report, _, _ = _run_planner(
        tmp_path,
        import_report={"bounded_simulator_import_smoke_passed": True},
        render_report={"bounded_simulator_render_smoke_passed": True},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_bounded_render_smoke_plan"] is True
    assert report["ready_for_bounded_reset_step_smoke_plan"] is True
    assert report["ready_for_rollout"] is False
    assert report["policy"]["reset_step_smoke_performed"] is False
    assert report["policy"]["rollouts_performed"] is False


def test_render_reset_plan_refuses_execution_gates(tmp_path):
    result, report, _, _ = _run_planner(
        tmp_path,
        extra_env={"ALLOW_SIMULATOR_RENDER_SMOKE": "1"},
        import_report={"bounded_simulator_import_smoke_passed": True},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_SIMULATOR_RENDER_SMOKE" in report["dangerous_execution_gates_set"]
    assert report["policy"]["render_smoke_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
