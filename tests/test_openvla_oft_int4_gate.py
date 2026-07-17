from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
RUNS = REPO_ROOT / "runs" / "openvla_oft_int4"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_quantized_hard_slice_result_is_not_full_precision_claim() -> None:
    result = _read_json(REPORTS / "openvla_oft_quantized_hard_slice_result.json")

    assert result["decision"] == "FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE"
    assert result["training_happened"] is False
    assert result["full_bf16_attempted"] is False
    assert result["libero_pro_justified"] is False
    assert "quantized" in result["quantization_limitation"].lower()
    assert "not claimed" in result["quantization_limitation"].lower()


def test_exact_manifest_uses_only_frozen_twenty_episode_design() -> None:
    manifest = _read_json(REPORTS / "openvla_oft_quantized_hard_slice_manifest.json")

    assert manifest["reset_identities"] == [20260711, 20260712, 20260713, 20260714, 20260715]
    assert manifest["episode_budget"] == {
        "max_total": 40,
        "openvla_int4_planned": 20,
        "smolvla_exact_planned": 20,
    }
    assert manifest["exact_init_sha_match_between_policies"] is True
    assert len(manifest["episodes"]) == 20
    assert {(item["suite"], item["task_id"]) for item in manifest["episodes"]} == {
        ("libero_spatial", 4),
        ("libero_10", 4),
        ("libero_spatial", 2),
        ("libero_10", 2),
    }
    assert {item["initial_state_index"] for item in manifest["episodes"]} == {0, 1, 2, 3, 4}
    assert all(item["initial_state_sha256"] == item["smolvla_initial_state_sha256"] for item in manifest["episodes"])


def test_rollouts_have_videos_no_offload_and_expected_success_pattern() -> None:
    openvla = _read_json(RUNS / "hard_slice_openvla_int4.json")
    smolvla = _read_json(RUNS / "hard_slice_smolvla_exact.json")

    assert openvla["success"] is True
    assert openvla["completed_episode_count"] == 20
    assert openvla["successful_episode_count"] == 20
    assert openvla["offload_status"] == "NO_CPU_OR_DISK_OFFLOAD_DETECTED"
    assert len([item for item in openvla["episodes"] if item.get("video_path")]) == 20

    assert smolvla["success"] is True
    assert smolvla["completed_episode_count"] == 20
    assert smolvla["successful_episode_count"] == 11
    assert len([item for item in smolvla["episodes"] if item.get("video_path")]) == 20


def test_decision_report_blocks_libero_pro_and_method_design() -> None:
    decision = (REPORTS / "openvla_oft_quantized_cross_backbone_decision.md").read_text(encoding="utf-8")

    assert "FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE" in decision
    assert "INT4 is quantized" in decision
    assert "full-precision OpenVLA-OFT claim" in decision
    assert "LIBERO-PRO justified now: `false`" in decision
    assert "stop method design" in decision


def test_epoch5_residual_manifest_controls_are_explicit_and_matched() -> None:
    from tca_map import openvla_oft_int4_gate as openvla
    from tca_map.smolvla import exact_hard_slice_rollout as smolvla

    task_specs = "libero_10:8:epoch5_two_moka_pots,libero_10:9:epoch5_microwave_close"
    reset_ids = "20260716,20260717"

    assert openvla._parse_task_specs(task_specs) == [
        {"suite": "libero_10", "task_id": 8, "role": "epoch5_two_moka_pots"},
        {"suite": "libero_10", "task_id": 9, "role": "epoch5_microwave_close"},
    ]
    assert smolvla._parse_task_specs(task_specs) == openvla._parse_task_specs(task_specs)
    assert openvla._parse_reset_identities(reset_ids) == [20260716, 20260717]
    assert smolvla._parse_reset_identities(reset_ids) == [20260716, 20260717]

    assert openvla._identity_to_initial_state_index(20260711) == 0
    assert openvla._identity_to_initial_state_index(20260716) == 5
    assert smolvla._identity_to_initial_state_index(20260760) == 49

    with pytest.raises(ValueError):
        openvla._identity_to_initial_state_index(20260761)
    with pytest.raises(ValueError):
        smolvla._parse_reset_identities("20260716,20260716")


def test_openvla_optional_json_numpy_shim_has_patch_and_spec(monkeypatch) -> None:
    import sys

    from tca_map import openvla_oft_int4_gate as openvla

    monkeypatch.delitem(sys.modules, "json_numpy", raising=False)
    original_find_spec = openvla.importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "json_numpy":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(openvla.importlib.util, "find_spec", fake_find_spec)

    used = openvla.install_openvla_optional_import_shims()

    assert used == ["json_numpy"]
    assert sys.modules["json_numpy"].__spec__ is not None
    assert sys.modules["json_numpy"].__spec__.name == "json_numpy"
    assert callable(sys.modules["json_numpy"].patch)
