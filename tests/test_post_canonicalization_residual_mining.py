import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.tg7d_adapter import residual_mining as residual


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "235_post_canonicalization_residual_mining.ps1"


def _minimal_report(path: Path) -> None:
    def metrics(action_l2, translation=0.3, rotation=0.08, gripper=0.4):
        return {
            "action_l2": action_l2,
            "translation_l2": translation,
            "rotation_l2": rotation,
            "gripper_error": gripper,
            "gripper_accuracy": 0.8,
            "sample_count": 2,
            "per_dim_mae": [0.1, 0.1, 0.2, 0.01, 0.02, 0.03, gripper],
        }

    report = {
        "method_gate": {
            "variants": {
                "mean_action": {"eval_metrics": {"clean": metrics(0.9), "heldout_paraphrase": metrics(0.9), "object_lexical": metrics(0.9)}},
                "small_mlp": {"eval_metrics": {"clean": metrics(0.62), "heldout_paraphrase": metrics(0.62), "object_lexical": metrics(0.62)}},
                "ridge": {"eval_metrics": metrics(0.58)},
                "standard_smolvla_7d_lora_adapter": {"eval_metrics": {"clean": metrics(0.60), "heldout_paraphrase": metrics(0.60), "object_lexical": metrics(0.60)}},
                "canonicalization_only": {"eval_metrics": {"clean": metrics(0.588), "heldout_paraphrase": metrics(0.587), "object_lexical": metrics(0.587)}},
                "tg7d_adapter": {
                    "eval_metrics": {"clean": metrics(0.735), "heldout_paraphrase": metrics(0.740), "object_lexical": metrics(0.744)},
                    "counterfactual_sensitivity": {"prediction_delta_l2": 0.06},
                },
                "oracle_target_upper_bound": {"eval_metrics": {"clean": metrics(0.72), "heldout_paraphrase": metrics(0.72), "object_lexical": metrics(0.72)}},
            }
        }
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_final_decisions_are_exact():
    assert residual.FINAL_DECISIONS == {
        "GO_RESIDUAL_METHOD_DESIGN",
        "STOP_LANGUAGE_TARGET_ROUTE",
        "NEED_OFFICIAL_LIBERO_PARA_BENCHMARK",
        "KILL_CANONICALIZATION_DOMINATED",
        "NO_VALID_RESIDUAL_METRIC",
    }


def test_residual_summary_kills_when_canonicalization_best(tmp_path):
    report_path = tmp_path / "tg7d.json"
    _minimal_report(report_path)
    source = json.loads(report_path.read_text(encoding="utf-8"))
    table = residual._variant_table(source)
    action = residual._action_dimension_breakdown(source)
    summary = residual._residual_summary(source, table, action)
    decision, _ = residual._decide(summary, source)
    assert decision == "KILL_CANONICALIZATION_DOMINATED"
    assert summary["method_worthy_residual"] is False
    assert summary["oracle_headroom_exists"] is False


def test_runner_refuses_training_gate(tmp_path):
    powershell = pytest.importorskip("shutil").which("powershell")
    if powershell is None:
        pytest.skip("PowerShell not available")
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ReportPath",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "ALLOW_TG7D_ADAPTER_TRAINING": "1"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert result.returncode == 20
    assert "Refusing residual mining" in (result.stdout + result.stderr)
