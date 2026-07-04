import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "66_plan_libero_policy_rollout_readiness.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for LIBERO policy rollout readiness tests")
    return exe


def _make_local_paths(tmp_path):
    libero_root = tmp_path / "LIBERO"
    robosuite_root = tmp_path / "robosuite"
    data_root = tmp_path / "libero_data"
    ckpt_root = tmp_path / "smolvla"
    hf_home = tmp_path / "hf_home"
    dependency_root = hf_home / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"
    for path in (libero_root, robosuite_root, data_root, ckpt_root, dependency_root):
        path.mkdir(parents=True, exist_ok=True)
    (ckpt_root / "config.json").write_text("{}", encoding="utf-8")
    (ckpt_root / "model.safetensors").write_text("marker", encoding="utf-8")
    (ckpt_root / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    return libero_root, robosuite_root, data_root, ckpt_root, hf_home


def _run_script(tmp_path, extra_env=None, extra_args=None, runtime_probe=None, make_paths=True):
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps({"bounded_libero_robosuite_diagnostic_rollout_passed": True}),
        encoding="utf-8",
    )
    json_report = tmp_path / "report.json"
    md_report = tmp_path / "report.md"

    if make_paths:
        libero_root, robosuite_root, data_root, ckpt_root, hf_home = _make_local_paths(tmp_path)
    else:
        libero_root = tmp_path / "missing_LIBERO"
        robosuite_root = tmp_path / "missing_robosuite"
        data_root = tmp_path / "missing_data"
        ckpt_root = tmp_path / "missing_smolvla"
        hf_home = tmp_path / "missing_hf_home"

    probe_path = ""
    if runtime_probe is not None:
        probe = tmp_path / "wsl_probe.json"
        probe.write_text(json.dumps(runtime_probe), encoding="utf-8")
        probe_path = str(probe)

    env = os.environ.copy()
    for key in (
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_SIMULATOR_RENDER_SMOKE",
        "ALLOW_TINY_ROLLOUT",
        "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})

    args = [
        _powershell(),
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCRIPT),
        "-DiagnosticReportPath",
        str(diagnostic),
        "-LiberoRoot",
        str(libero_root),
        "-RobosuiteRoot",
        str(robosuite_root),
        "-LiberoDataRoot",
        str(data_root),
        "-SmolVlaCheckpoint",
        str(ckpt_root),
        "-HfHome",
        str(hf_home),
        "-JsonReportPath",
        str(json_report),
        "-MarkdownReportPath",
        str(md_report),
    ]
    if probe_path:
        args.extend(["-WslRuntimeProbeReportPath", probe_path])
    else:
        args.append("-SkipLiveWslProbe")
    args.extend(extra_args or [])

    result = subprocess.run(
        args,
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


def _ready_probe():
    modules = {
        "torch": True,
        "torchvision": True,
        "transformers": True,
        "lerobot": True,
        "safetensors": True,
        "huggingface_hub": True,
        "accelerate": True,
        "num2words": True,
        "draccus": True,
        "datasets": True,
        "imageio": True,
        "diffusers": True,
        "serial": True,
        "deepdiff": True,
        "av": True,
        "einops": True,
    }
    return {
        "ok": True,
        "python": "3.10.12",
        "module_specs": modules,
        "heavy_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "gpu_jobs_performed": False,
        "training_performed": False,
        "rollouts_performed": False,
        "openvla_oft_executed": False,
    }


def test_policy_rollout_readiness_reports_reduce_scope_without_wsl_runtime(tmp_path):
    result, report, json_report, md_report = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "reduce_scope"
    assert report["ready_for_tiny_learned_policy_rollout_plan"] is True
    assert report["ready_for_tiny_learned_policy_rollout_execution"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_policy_rollout_readiness_allows_green_wsl_only_topology_with_probe(tmp_path):
    result, report, _, _ = _run_script(tmp_path, runtime_probe=_ready_probe())

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_tiny_learned_policy_rollout_execution"] is True
    assert report["ready_for_benchmark_rollout"] is True
    assert report["ready_for_paper_claim"] is False
    assert report["topologies"]["wsl_only_policy_and_sim"]["ready"] is True
    assert report["topologies"]["windows_policy_wsl_sim_bridge"]["ready"] is False


def test_policy_rollout_readiness_refuses_execution_gates(tmp_path):
    result, report, _, _ = _run_script(
        tmp_path,
        extra_env={"ALLOW_POLICY_ROLLOUT": "1"},
        runtime_probe=_ready_probe(),
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_POLICY_ROLLOUT" in report["dangerous_execution_gates_set"]
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["model_inference_performed"] is False


def test_policy_rollout_readiness_stops_when_local_paths_missing(tmp_path):
    result, report, _, _ = _run_script(tmp_path, make_paths=False, runtime_probe=_ready_probe())

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["ready_for_tiny_learned_policy_rollout_plan"] is False
    assert any("required local path" in reason for reason in report["stop_reasons"])
