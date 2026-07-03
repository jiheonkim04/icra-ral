import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET = REPO_ROOT / "configs" / "compute_budget.yaml"
HEAD_ONLY_CONFIGS = [
    REPO_ROOT / "configs" / "actionmap_head_only_lowcompute.yaml",
    REPO_ROOT / "configs" / "tca_map_head_only_lowcompute.yaml",
]
SCRIPT = REPO_ROOT / "scripts" / "30_enforce_compute_budget.ps1"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_local_pilot_step_budget_is_300():
    budget = _load_yaml(BUDGET)

    assert budget["limits"]["max_local_pilot_steps_initial"] == 300

    for path in HEAD_ONLY_CONFIGS:
        config = _load_yaml(path)
        assert config["training"]["max_steps"] <= 300
        assert config["training"]["train_backbone"] is False
        assert config["run"]["rollouts_allowed"] is False
        assert config["openvla_oft"]["enabled"] is False


def test_compute_budget_checker_reports_300_step_limit():
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the compute budget checker")

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["limits"]["max_local_pilot_steps_initial"] == 300
    assert report["gpu_jobs_performed"] is False
    assert report["downloads_performed"] is False
    assert report["rollouts_performed"] is False
