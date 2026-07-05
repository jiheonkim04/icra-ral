import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.datasets.libero_offline_lora_comparison import run_libero_offline_lora_comparison


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "53_compare_libero_offline_lora.ps1"


def _clean_env(extra=None):
    env = os.environ.copy()
    for gate in [
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_CLOUD_HANDOFF",
    ]:
        env.pop(gate, None)
    env.update(extra or {})
    return env


def _write_demo(path: Path, offset: float) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        actions = demo.create_dataset("actions", shape=(4, 7), dtype="f4")
        for row in range(4):
            actions[row, :] = offset + row * 0.01


def _write_manifest(tmp_path: Path, pair_count: int = 2) -> Path:
    pairs = []
    for index in range(pair_count):
        positive = tmp_path / "data" / "libero_10" / f"positive_{index}_demo.hdf5"
        counter = tmp_path / "data" / "libero_10" / f"counter_{index}_demo.hdf5"
        _write_demo(positive, 0.1 + index * 0.05)
        _write_demo(counter, 0.4 + index * 0.05)
        pairs.append(
            {
                "pair_id": f"libero_10:positive_{index}__vs__counter_{index}",
                "positive_demo_file": str(positive),
                "counterfactual_demo_file": str(counter),
                "positive_instruction": "pick the soup can",
                "counterfactual_instruction": "pick the milk carton",
            }
        )
    manifest = {
        "ready_for_tiny_offline_counterfactual_split": True,
        "counterfactual_pairs": pairs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_head_only_reference(tmp_path: Path, train_pair_ids, eval_pair_ids) -> Path:
    reference = {
        "comparison": {"conclusion": "weakens_tca_map"},
        "split": {
            "train_pair_ids": train_pair_ids,
            "eval_pair_ids": eval_pair_ids,
        },
        "arms": {
            "actionmap_head_only": {"evaluation_metrics": {"standard_proxy_score": 0.4, "wrong_target_proxy_rate": 0.5}},
            "tca_map_head_only": {"evaluation_metrics": {"standard_proxy_score": 0.0, "wrong_target_proxy_rate": 1.0}},
            "tca_map_distributional_select": {"evaluation_metrics": {"standard_proxy_score": 0.0, "wrong_target_proxy_rate": 1.0}},
        },
    }
    path = tmp_path / "head_only.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    return path


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_libero_offline_lora_comparison_builds_required_arms_and_checks(tmp_path):
    manifest = _write_manifest(tmp_path, pair_count=2)
    train_ids = ["libero_10:positive_0__vs__counter_0"]
    eval_ids = ["libero_10:positive_1__vs__counter_1"]
    head_only = _write_head_only_reference(tmp_path, train_ids, eval_ids)
    report = run_libero_offline_lora_comparison(
        manifest_path=manifest,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        head_only_report_path=head_only,
        max_pairs=2,
        max_action_steps=4,
        max_steps=4,
        max_samples=4,
        rank=2,
        require_training_gate=False,
    )

    assert report["libero_offline_lora_comparison_passed"] is True
    assert report["ready_for_rollout"] is False
    assert report["policy"]["real_dataset_used"] is True
    assert report["policy"]["training_performed"] is True
    assert report["policy"]["lora_training_performed"] is True
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert {arm["arm"] for arm in report["arms"]} == {
        "actionmap_lora",
        "tca_map_lora",
        "tca_map_lora_distributional_select",
    }
    assert report["sanity_checks"]["target_labels_aligned"] is True
    assert report["sanity_checks"]["wrong_target_proxy_not_inverted"] is True
    assert report["sanity_checks"]["target_conditioning_non_constant"] is True
    assert report["sanity_checks"]["same_split_as_head_only"] is True
    assert report["sanity_checks"]["tca_select_candidate_scores_degenerate"] is False
    for arm in report["arms"]:
        assert arm["trainable_lora_parameter_count"] > 0
        assert arm["loss_curve"]
        assert arm["initial_loss"] >= 0.0
        assert arm["final_loss"] >= 0.0
        assert "standard_proxy_score" in arm["evaluation_metrics"]


def test_libero_offline_lora_script_requires_training_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for LIBERO offline LoRA comparison script tests")

    manifest = _write_manifest(tmp_path)
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ManifestPath",
            str(manifest),
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 21
    assert "ALLOW_TINY_TRAINING=1" in (result.stdout + result.stderr)


def test_libero_offline_lora_script_runs_with_bounded_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for LIBERO offline LoRA comparison script tests")

    manifest = _write_manifest(tmp_path, pair_count=2)
    head_only = _write_head_only_reference(
        tmp_path,
        ["libero_10:positive_0__vs__counter_0"],
        ["libero_10:positive_1__vs__counter_1"],
    )
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ManifestPath",
            str(manifest),
            "-HeadOnlyReportPath",
            str(head_only),
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
            "-MaxPairs",
            "2",
            "-MaxActionSteps",
            "4",
            "-MaxSteps",
            "4",
            "-MaxSamples",
            "4",
            "-Rank",
            "2",
        ],
        cwd=REPO_ROOT,
        env=_clean_env({"ALLOW_TINY_TRAINING": "1"}),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["libero_offline_lora_comparison_passed"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["sanity_checks"]["same_split_as_head_only"] is True


def test_libero_offline_lora_script_refuses_dangerous_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for LIBERO offline LoRA comparison script tests")

    manifest = _write_manifest(tmp_path)
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ManifestPath",
            str(manifest),
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env=_clean_env({"ALLOW_TINY_TRAINING": "1", "ALLOW_DOWNLOADS": "1"}),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "dangerous gates" in (result.stdout + result.stderr)
