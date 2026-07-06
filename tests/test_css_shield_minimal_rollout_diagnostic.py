from argparse import Namespace

import numpy as np

from tca_map.css_shield.minimal_rollout_diagnostic import apply_shield, assess_action, build_report


def _obs():
    return {
        "robot0_eef_pos": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        "moka_pot_1_pos": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "black_bowl_1_pos": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    }


def _args(tmp_path):
    return Namespace(
        manifest=str(tmp_path / "missing.json"),
        report_json=str(tmp_path / "report.json"),
        report_md=str(tmp_path / "report.md"),
        smolvla_ckpt="unused",
        checkpoint_root="unused",
        hf_home="unused",
        libero_root="unused",
        robosuite_root="unused",
        max_steps=5,
        camera_size=64,
        max_translation_norm=0.2,
        proposal_source="synthetic_counterfactual_probe",
        device="cpu",
    )


def test_assess_action_detects_wrong_target_direction():
    action = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, -1.0]

    report = assess_action(action, _obs(), "put the moka pot on the stove", "put the black bowl in the drawer")

    assert report["semantic_proxy_available"] is True
    assert report["wrong_target_action"] is True
    assert report["unsafe_action"] is True
    assert report["object_position_keys"]["target_key"] == "moka_pot_1_pos"
    assert report["object_position_keys"]["wrong_key"] == "black_bowl_1_pos"


def test_full_shield_redirects_wrong_target_action_toward_target():
    action = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, -1.0]

    shielded, meta = apply_shield(action, _obs(), "put the moka pot on the stove", "put the black bowl in the drawer", "full_css_shield")

    assert meta["intervened"] is True
    assert meta["before"]["wrong_target_action"] is True
    assert meta["after"]["wrong_target_action"] is False
    assert shielded[0] > 0.0
    assert abs(shielded[1]) < 1e-9


def test_safety_only_damps_excessive_translation_without_semantic_redirect():
    action = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, -1.0]

    shielded, meta = apply_shield(action, _obs(), "put the moka pot on the stove", "put the black bowl in the drawer", "safety_only")

    assert meta["intervention"] == "damp"
    assert meta["after"]["unsafe_action"] is False
    assert shielded[1] > 0.0


def test_build_report_requires_task_local_rollout_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_CSS_SHIELD_ROLLOUT", raising=False)

    report = build_report(_args(tmp_path))

    assert report["result"]["passed"] is False
    assert "ALLOW_CSS_SHIELD_ROLLOUT" in report["result"]["blocked_reason"]
