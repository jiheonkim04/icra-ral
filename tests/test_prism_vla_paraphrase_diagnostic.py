import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.prism_vla.paraphrase_diagnostic import build_prism_vla_diagnostic


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "190_prism_vla_paraphrase_diagnostic.ps1"


def _write_task(root: Path, suite: str, task_name: str, language: str, obj: str, target: str) -> None:
    suite_root = root / "libero" / "libero" / "bddl_files" / suite
    suite_root.mkdir(parents=True, exist_ok=True)
    task_path = suite_root / f"{task_name}.bddl"
    task_path.write_text(
        "\n".join(
            [
                "(define (problem PRISM_FAKE_LIBERO)",
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


def _write_demo(data_root: Path, suite: str, task_name: str, offset: float) -> None:
    h5py = pytest.importorskip("h5py")
    path = data_root / suite / f"{task_name}_demo.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        actions = demo.create_dataset("actions", shape=(4, 7), dtype="f4")
        for row in range(4):
            actions[row, :] = offset + row * 0.02


def _write_metadata(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "high,mid,low,eval,batch_idx,new_instruction,original_instruction,structural_similarity,keyword_similarity",
                "obj,lexical,synonym,0,0,set the dish on the plate,put the bowl on the plate,0.65,0.55",
                "act,lexical,synonym,0,1,place the bowl on the plate,put the bowl on the plate,0.82,0.78",
                "comp,lexical+structural,synonym+coordination,0,2,find the dish and set it on the plate,put the bowl on the plate,0.48,0.50",
                "obj,lexical,synonym,1,0,activate the cooktop,turn on the stove,0.62,0.52",
                "act,lexical,synonym,1,1,switch on the stove,turn on the stove,0.86,0.82",
                "comp,lexical+structural,synonym+subordination,1,2,turn on the cooktop so it heats,turn on the stove,0.50,0.50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    libero_root = tmp_path / "LIBERO"
    data_root = tmp_path / "data"
    metadata = tmp_path / "libero_para_metadata.csv"
    _write_task(
        libero_root,
        "libero_goal",
        "put_the_bowl_on_the_plate",
        "put the bowl on the plate",
        "bowl_1",
        "plate_1",
    )
    _write_task(
        libero_root,
        "libero_goal",
        "turn_on_the_stove",
        "turn on the stove",
        "stove_1",
        "knob_1",
    )
    _write_demo(data_root, "libero_goal", "put_the_bowl_on_the_plate", 0.10)
    _write_demo(data_root, "libero_goal", "turn_on_the_stove", 0.65)
    _write_metadata(metadata)
    return libero_root, data_root, metadata


def test_prism_vla_diagnostic_runs_required_variants(tmp_path):
    libero_root, data_root, metadata = _fixture_roots(tmp_path)

    report = build_prism_vla_diagnostic(
        libero_root=libero_root,
        libero_data_root=data_root,
        libero_para_metadata_csv=metadata,
        max_tasks=2,
        max_paraphrases_per_task=3,
        max_action_steps=4,
        max_steps=16,
        learning_rate=0.10,
        feature_width=48,
    )

    assert report["schema_version"] == "prism-vla-paraphrase-diagnostic-v1-heldout"
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["loss_computed"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["data"]["official_libero_para_metadata_used"] is True
    assert set(report["variants"]) == {
        "base_no_paraphrase_training",
        "simple_paraphrase_augmentation",
        "canonicalization_only",
        "prism_vla_consistency",
        "prism_vla_plus_canonicalization",
        "difficulty_weighted_prism",
        "counterfactual_sensitive_prism",
    }
    split_audit = report["data"]["split_audit"]
    assert split_audit["official_split_used"] is False
    assert split_audit["group_leakage_detected"] is False
    assert split_audit["train_paraphrase_group_count"] > 0
    assert split_audit["heldout_paraphrase_group_count"] > 0
    assert split_audit["heldout_object_group_count"] > 0
    assert report["data"]["train_paraphrase_count"] > 0
    assert report["data"]["heldout_paraphrase_count"] > 0
    assert report["real_vla_adapter_diagnostic"]["happened"] is False
    for payload in report["variants"].values():
        assert payload["loss_curve"]
        assert payload["clean"]["continuous_proxy_score"] is not None
        assert payload["paraphrase"]["action_trajectory_divergence"] is not None
        assert payload["syntactic_variation"]["count"] >= 0
        assert payload["counterfactual_sensitivity"]["pair_count"] == 1
    assert report["decision"]["decision"] in {"continue", "continue_reframe_canonicalized_prism", "kill"}
    assert "canonicalization_only_metric" in report["decision"]


def test_prism_vla_diagnostic_falls_back_to_local_paraphrases(tmp_path):
    libero_root, data_root, _metadata = _fixture_roots(tmp_path)
    missing_metadata = tmp_path / "missing.csv"

    report = build_prism_vla_diagnostic(
        libero_root=libero_root,
        libero_data_root=data_root,
        libero_para_metadata_csv=missing_metadata,
        max_tasks=2,
        max_paraphrases_per_task=2,
        max_action_steps=4,
        max_steps=8,
        learning_rate=0.10,
        feature_width=48,
    )

    assert report["data"]["official_libero_para_metadata_used"] is False
    assert report["data"]["local_exploratory_paraphrases_used"] is True
    assert report["data"]["selected_paraphrase_count"] >= 2
    assert report["data"]["heldout_paraphrase_count"] >= 1
    assert report["data"]["split_audit"]["group_leakage_detected"] is False
    assert report["model"]["real_vla_model_metric_produced"] is False
    assert report["data"]["real_dataset_metric_produced"] is True


def test_prism_vla_runner_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for PRISM-VLA runner tests")

    libero_root, data_root, metadata = _fixture_roots(tmp_path)
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-LiberoRoot",
            str(libero_root),
            "-LiberoDataRoot",
            str(data_root),
            "-LiberoParaMetadataCsv",
            str(metadata),
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


def test_prism_vla_runner_refuses_dangerous_gates(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for PRISM-VLA runner tests")

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
        env={**os.environ, "ALLOW_TINY_TRAINING": "1", "ALLOW_DOWNLOADS": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "Refusing PRISM-VLA diagnostic" in (result.stdout + result.stderr)


def test_prism_vla_runner_outputs_json(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for PRISM-VLA runner tests")

    libero_root, data_root, metadata = _fixture_roots(tmp_path)
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-LiberoRoot",
            str(libero_root),
            "-LiberoDataRoot",
            str(data_root),
            "-LiberoParaMetadataCsv",
            str(metadata),
            "-MaxTasks",
            "2",
            "-MaxParaphrasesPerTask",
            "3",
            "-MaxActionSteps",
            "4",
            "-MaxSteps",
            "8",
            "-FeatureWidth",
            "48",
            "-JsonReportPath",
            str(tmp_path / "report.json"),
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

    assert result.returncode == 0, result.stderr
    start = result.stdout.find("{")
    assert start >= 0
    report = json.loads(result.stdout[start:])
    assert report["policy"]["training_performed"] is True
    assert report["variants"]["counterfactual_sensitive_prism"]["loss_curve"]
    assert report["data"]["evaluation_split"] == "deterministic_heldout_paraphrase_group_split"
    assert (tmp_path / "report.md").exists()
