import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_offline_counterfactual_split import build_offline_counterfactual_split


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "51_build_libero_offline_counterfactual_split.ps1"


def _write_fake_task(root: Path, suite: str, task_name: str, language: str, obj: str, target: str) -> None:
    suite_root = root / "libero" / "libero" / "bddl_files" / suite
    suite_root.mkdir(parents=True, exist_ok=True)
    task_path = suite_root / f"{task_name}.bddl"
    task_path.write_text(
        "\n".join(
            [
                "(define (problem FAKE_LIBERO)",
                "  (:domain robosuite)",
                f"  (:language {language})",
                "  (:obj_of_interest",
                f"    {obj}",
                f"    {target}",
                "  )",
                "  (:goal",
                f"    (And (On {obj} {target}))",
                "  )",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    info_path = suite_root / "tasks_info.txt"
    existing = info_path.read_text(encoding="utf-8") if info_path.exists() else ""
    info_path.write_text(existing + f"libero/bddl_files/{suite}/{task_name}.bddl\n", encoding="utf-8")


def _write_demo(data_root: Path, suite: str, task_name: str) -> None:
    h5py = pytest.importorskip("h5py")
    suite_root = data_root / suite
    suite_root.mkdir(parents=True, exist_ok=True)
    with h5py.File(suite_root / f"{task_name}_demo.hdf5", "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.create_dataset("actions", shape=(3, 7), dtype="f4")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", shape=(3, 8, 8, 3), dtype="u1")


def test_build_offline_counterfactual_split_matches_bddl_to_hdf5(tmp_path):
    libero_root = tmp_path / "LIBERO"
    data_root = tmp_path / "data"
    _write_fake_task(
        libero_root,
        "libero_object",
        "pick_soup_to_basket",
        "Pick up the alphabet soup and place it in the basket",
        "alphabet_soup_1",
        "basket_1",
    )
    _write_fake_task(
        libero_root,
        "libero_object",
        "pick_milk_to_basket",
        "Pick up the milk and place it in the basket",
        "milk_1",
        "basket_1",
    )
    _write_demo(data_root, "libero_object", "pick_soup_to_basket")
    _write_demo(data_root, "libero_object", "pick_milk_to_basket")

    report = build_offline_counterfactual_split(
        libero_root=libero_root,
        libero_data_root=data_root,
        suites=["libero_object"],
        max_tasks_per_suite=4,
        max_demo_files=4,
        max_counterfactual_pairs=4,
    )

    assert report["decision"] == "proceed"
    assert report["ready_for_tiny_offline_counterfactual_split"] is True
    assert report["ready_for_tiny_offline_actionmap_tca_comparison"] is True
    assert report["ready_for_rollout"] is False
    assert report["matched_task_count"] == 2
    assert report["counterfactual_pair_count"] >= 1
    assert report["counterfactual_pairs"][0]["offline_proxy_only"] is True
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["simulator_executed"] is False


def test_offline_counterfactual_split_script_refuses_execution_gates(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for offline counterfactual split script tests")

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "ALLOW_ROLLOUTS": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "Refusing offline counterfactual split builder" in (result.stdout + result.stderr)


def test_offline_counterfactual_split_script_outputs_json_with_temp_paths(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for offline counterfactual split script tests")

    libero_root = tmp_path / "LIBERO"
    data_root = tmp_path / "data"
    _write_fake_task(libero_root, "libero_object", "pick_soup_to_basket", "Pick soup to basket", "alphabet_soup_1", "basket_1")
    _write_fake_task(libero_root, "libero_object", "pick_milk_to_basket", "Pick milk to basket", "milk_1", "basket_1")
    _write_demo(data_root, "libero_object", "pick_soup_to_basket")
    _write_demo(data_root, "libero_object", "pick_milk_to_basket")

    paths_file = tmp_path / "paths.local.yaml"
    paths_file.write_text(
        f"assets:\n  libero_root: \"{libero_root}\"\n  libero_data_root: \"{data_root}\"\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "split.yaml"
    config_file.write_text("suites:\n  - libero_object\nmax_tasks_per_suite: 4\nmax_demo_files: 4\nmax_counterfactual_pairs: 4\n", encoding="utf-8")

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-PathsFile",
            str(paths_file),
            "-Config",
            str(config_file),
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

    assert result.returncode == 0, result.stderr
    start = result.stdout.find("{")
    assert start >= 0
    report = json.loads(result.stdout[start:])
    assert report["ready_for_tiny_offline_counterfactual_split"] is True
    assert report["counterfactual_pair_count"] >= 1
    assert report["policy"]["downloads_performed"] is False
