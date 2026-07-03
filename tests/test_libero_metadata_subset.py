import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_metadata_subset import build_libero_metadata_subset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "47_build_libero_metadata_subset.ps1"


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


def test_build_libero_metadata_subset_from_bddl_without_importing_simulator(tmp_path):
    libero_root = tmp_path / "LIBERO"
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_fake_task(
        libero_root,
        "libero_spatial",
        "pick_bowl_on_stove",
        "Pick the black bowl on the stove and place it on the plate",
        "akita_black_bowl_1",
        "plate_1",
    )
    _write_fake_task(
        libero_root,
        "libero_spatial",
        "pick_bowl_on_cabinet",
        "Pick the black bowl on the cabinet and place it on the plate",
        "akita_black_bowl_1",
        "plate_1",
    )

    report = build_libero_metadata_subset(
        libero_root=libero_root,
        libero_data_root=data_root,
        suites=["libero_spatial"],
        max_tasks_per_suite=4,
        max_counterfactual_pairs=4,
    )

    assert report["ready_for_metadata_only_subset"] is True
    assert report["ready_for_real_dataset_interface_smoke"] is False
    assert report["ready_for_rollout"] is False
    assert report["selected_task_count"] == 2
    assert report["counterfactual_pair_count"] >= 1
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["simulator_executed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["tasks"][0]["metadata_only"] is True


def test_libero_metadata_subset_script_refuses_execution_gates(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for metadata subset script tests")

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
        env={**os.environ, "ALLOW_DOWNLOADS": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "Refusing metadata-only builder" in (result.stdout + result.stderr)


def test_libero_metadata_subset_script_outputs_json_with_temp_paths(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for metadata subset script tests")

    libero_root = tmp_path / "LIBERO"
    data_root = tmp_path / "data"
    data_root.mkdir()
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
    paths_file = tmp_path / "paths.local.yaml"
    paths_file.write_text(
        f"assets:\n  libero_root: \"{libero_root}\"\n  libero_data_root: \"{data_root}\"\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "subset.yaml"
    config_file.write_text("suites:\n  - libero_object\nmax_tasks_per_suite: 4\nmax_counterfactual_pairs: 4\n", encoding="utf-8")

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
    assert report["ready_for_metadata_only_subset"] is True
    assert report["ready_for_real_dataset_interface_smoke"] is False
    assert report["selected_task_count"] == 2
    assert report["policy"]["downloads_performed"] is False
