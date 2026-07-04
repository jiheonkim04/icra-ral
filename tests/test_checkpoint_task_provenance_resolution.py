import argparse
import json
from pathlib import Path

from tca_map.smolvla import checkpoint_task_provenance_resolution as audit


def _write_checkpoint(root: Path, *, so100_readme=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [6]},
                    "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
                    "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
                    "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [6]}},
                "normalization_mapping": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
                "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "load_vlm_weights": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
                    },
                    {
                        "registry_name": "normalizer_processor",
                        "config": {
                            "features": {
                                "observation.state": {"type": "STATE", "shape": [6]},
                                "action": {"type": "ACTION", "shape": [6]},
                            }
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {"features": {"action": {"type": "ACTION", "shape": [6]}}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    readme = "Intended use: Base model to fine tune on your specific use case. lerobot-record --robot.type=so100_follower"
    if not so100_readme:
        readme = "Intended use: Base model to fine tune on your specific use case."
    (root / "README.md").write_text(readme, encoding="utf-8")


def _write_plan(path: Path, *, ready=True):
    path.write_text(
        json.dumps({"ready_for_checkpoint_task_provenance_resolution": ready}),
        encoding="utf-8",
    )


def _write_stat_audit(path: Path, *, passed=True, prefixes=None, scale=True, dim=True):
    if prefixes is None:
        prefixes = ["so100", "so100-blue"]
    path.write_text(
        json.dumps(
            {
                "libero_action_stat_subset_audit_passed": passed,
                "libero_action_stats": {"count": 2500, "dim": 7, "max_abs": 1.0},
                "comparison_to_checkpoint": {
                    "checkpoint_action_stat_prefixes": prefixes,
                    "checkpoint_action_mean_max_abs": 125.0 if scale else 0.6,
                    "checkpoint_action_std_max": 59.0 if scale else 0.2,
                    "scale_mismatch_confirmed": scale,
                    "dimension_mismatch_confirmed": dim,
                },
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path):
    checkpoint = tmp_path / "smolvla"
    plan = tmp_path / "plan.json"
    stat = tmp_path / "stat.json"
    _write_checkpoint(checkpoint)
    _write_plan(plan)
    _write_stat_audit(stat)
    return argparse.Namespace(
        checkpoint_root=str(checkpoint),
        normalized_plan_report=str(plan),
        libero_action_stat_report=str(stat),
        report_path=str(tmp_path / "report.json"),
        markdown_report_path=str(tmp_path / "report.md"),
    )


def test_audit_blocks_learned_policy_rollout_scaling_for_so100_libero_mismatch(tmp_path, monkeypatch):
    for gate in audit.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = audit.build_report(_args(tmp_path))

    assert code == 0
    assert report["checkpoint_task_provenance_resolution_passed"] is True
    assert report["decision"] == "no_go_learned_policy_rollout_scaling"
    assert report["current_checkpoint_libero_rollout_evidence_valid"] is False
    assert report["ready_for_offline_head_tca_pivot"] is True
    assert report["ready_for_libero_aligned_checkpoint_source_plan"] is True
    assert report["ready_for_rollout_scaling"] is False


def test_audit_requires_authorized_plan(tmp_path, monkeypatch):
    for gate in audit.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_plan(Path(args.normalized_plan_report), ready=False)

    report, code = audit.build_report(args)

    assert code != 0
    assert report["decision"] == "stop"
    assert "did not authorize" in report["recommended_next_step"]


def test_audit_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report, code = audit.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUTS" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False


def test_audit_routes_to_normalized_probe_plan_for_non_so100_mismatch(tmp_path, monkeypatch):
    for gate in audit.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_checkpoint(Path(args.checkpoint_root), so100_readme=False)
    _write_stat_audit(Path(args.libero_action_stat_report), prefixes=["unknown"], scale=True, dim=True)

    report, code = audit.build_report(args)

    assert code == 0
    assert report["decision"] == "reduce_scope"
    assert report["selected_next_step"] == "plan_bounded_normalized_action_space_probe"
    assert report["ready_for_rollout_scaling"] is False
