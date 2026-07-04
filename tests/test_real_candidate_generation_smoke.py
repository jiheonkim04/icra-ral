import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tca_map.smolvla import real_candidate_generation_smoke as smoke


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "133_bounded_real_candidate_generation_smoke.ps1"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path: Path) -> argparse.Namespace:
    plan = tmp_path / "plan.json"
    runtime = tmp_path / "runtime.json"
    load = tmp_path / "load.json"
    single = tmp_path / "single.json"
    _write_json(
        plan,
        {
            "real_candidate_generation_smoke_plan_passed": True,
            "ready_for_real_candidate_generation_smoke_implementation": True,
            "ready_for_real_candidate_generation_smoke_execution": False,
        },
    )
    _write_json(runtime, {"runtime_dependencies": {"ready_for_load_only_runtime": True}})
    _write_json(load, {"result": {"passed": True}})
    _write_json(single, {"result": {"passed": True}})
    return argparse.Namespace(
        smolvla_ckpt=str(tmp_path / "smolvla"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        hf_home=str(tmp_path / "hf_home"),
        plan_report=str(plan),
        runtime_deps_report=str(runtime),
        load_only_report=str(load),
        single_sample_report=str(single),
        report_path=str(tmp_path / "report.json"),
        markdown_report_path=str(tmp_path / "report.md"),
        device="cpu",
        task="pick up the object",
        candidate_count=4,
        heatmap_grid=8,
        temperature=0.5,
    )


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for gate in smoke.REQUIRED_GATES + smoke.FORBIDDEN_GATES:
        env.pop(gate, None)
    env.update(extra or {})
    return env


def _json_from_stdout(stdout: str) -> dict:
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_candidate_heatmap_builder_is_low_resolution_and_selectable():
    action_heatmap, masked_heatmap, negative_heatmap, target_heatmap = smoke.build_candidate_heatmaps(
        [0.1, -0.2, 0.3, 0.0, 0.05, -0.05],
        candidate_count=4,
        heatmap_grid=8,
    )

    assert action_heatmap["low_resolution"] is True
    assert action_heatmap["coarse_to_fine_ready"] is True
    assert len(action_heatmap["candidates"]) == 4
    assert masked_heatmap["candidates"][0]["logit"] < action_heatmap["candidates"][0]["logit"]
    assert negative_heatmap["candidates"][1]["target_index"] == 1

    selection = smoke.distributional_tca_select_inference(
        action_heatmap=action_heatmap,
        target_heatmap=target_heatmap,
        masked_action_heatmap=masked_heatmap,
        negative_action_heatmaps=[negative_heatmap],
        K=4,
        temperature=0.5,
        metadata=None,
        external_verifier=None,
    )
    assert selection["external_verifier_used"] is False
    assert selection["privileged_inference_used"] is False
    assert selection["selected"]["target_index"] == 0


def test_real_candidate_generation_smoke_requires_all_gates(tmp_path, monkeypatch):
    for gate in smoke.REQUIRED_GATES + smoke.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = smoke.build_report(_args(tmp_path))

    assert code == 2
    assert report["real_candidate_generation_smoke_passed"] is False
    assert "ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1" in report["result"]["blocked_reason"]
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False


def test_real_candidate_generation_smoke_refuses_forbidden_gate_before_loading(tmp_path, monkeypatch):
    for gate in smoke.REQUIRED_GATES:
        monkeypatch.setenv(gate, "1")
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report, code = smoke.build_report(_args(tmp_path))

    assert code == 3
    assert "ALLOW_ROLLOUTS" in report["policy"]["forbidden_gates_set"]
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False


def test_real_candidate_generation_smoke_script_refuses_without_gates(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        return

    args = _args(tmp_path)
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-PlanReportPath",
            args.plan_report,
            "-RuntimeDepsReportPath",
            args.runtime_deps_report,
            "-LoadOnlyReportPath",
            args.load_only_report,
            "-SingleSampleReportPath",
            args.single_sample_report,
            "-ReportPath",
            args.report_path,
            "-MarkdownReportPath",
            args.markdown_report_path,
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    report = _json_from_stdout(result.stdout)
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["downloads_performed"] is False
    assert Path(args.report_path).exists()
    assert Path(args.markdown_report_path).exists()
