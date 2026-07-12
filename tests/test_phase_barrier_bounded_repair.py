import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def _json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def _sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_bounded_manifest_matches_result_counts():
    manifest = _json("phase_barrier_bounded_repair_manifest.json")
    result = _json("phase_barrier_bounded_repair_result.json")

    assert manifest["planned_episode_count"] == 100
    assert manifest["episodes_per_policy"] == 20
    assert manifest["eval_identities"] == result["eval_manifest"]["eval_identities"]
    assert result["eval_manifest"]["planned_episodes"] == 100
    assert len(result["episodes"]) == 100
    assert set(manifest["variants"]) == set(result["variants"])


def test_bounded_result_reuses_original_phasebarrier_checkpoint():
    original = _json("phase_barrier_vla_prototype_result.json")
    result = _json("phase_barrier_bounded_repair_result.json")
    invalid = _json("phase_barrier_bounded_repair_invalid_retrained_result.json")

    assert result["final_decision"] == "PHASEBARRIER_COMPONENT_NOT_USEFUL"
    assert result["runner_internal_decision"] == "PHASE_BARRIER_VALID_KILL"
    assert result["training_happened"] is False
    assert result["training_reused_from"] == "reports/phase_barrier_vla_prototype_result.json"
    assert result["train"]["positive_label_count"] == original["train"]["positive_label_count"] == 8
    assert result["train"]["phase_model"]["weights"] == original["train"]["phase_model"]["weights"]
    assert result["train"]["no_phase_model"]["weights"] == original["train"]["no_phase_model"]["weights"]
    assert result["training_checkpoint_identity"]["phase_model_sha256"] == _sha(original["train"]["phase_model"])
    assert invalid["train"]["positive_label_count"] != original["train"]["positive_label_count"]
    assert invalid["invalidated_by_checkpoint_identity_check"] is True


def test_bounded_decision_and_mechanism_activation_are_consistent():
    result = _json("phase_barrier_bounded_repair_result.json")
    adjudication = result["bounded_adjudication"]
    by_variant = result["summary"]["by_variant"]

    assert adjudication["decision"] == "PHASEBARRIER_COMPONENT_NOT_USEFUL"
    assert by_variant["phase_barrier_full"]["successes"] == 0
    assert by_variant["phase_barrier_full"]["total"] == 20
    assert by_variant["phase_barrier_no_phase_ablation"]["successes"] == 9
    assert by_variant["frozen_smolvla"]["successes"] == 8
    assert by_variant["phase_barrier_full"]["mean_action_delta_norm"] > 0.10
    assert adjudication["mechanism"]["full_shaped_episode_count"] == 20
    assert adjudication["mechanism"]["lower_bound_contact_transport_shaped_steps_total"] > 0
    assert adjudication["paired"]["phase_barrier_no_phase_ablation"]["full_wins"] == 0
    assert adjudication["paired"]["phase_barrier_no_phase_ablation"]["full_losses"] == 9


def test_bounded_no_privileged_inference_and_official_checkpoint_route():
    result = _json("phase_barrier_bounded_repair_result.json")
    audit = result["policy_load_audit"]
    source = (REPO_ROOT / "scripts" / "run_phase_barrier_vla_prototype.py").read_text(encoding="utf-8")

    assert audit["policy_class"] == "SmolVLAPolicy"
    assert audit["policy_name"] == "frozen_base"
    assert audit["old_custom_libero_7d_route_used"] is False
    assert audit["peft"]["used"] is False
    assert "action_feature_dict(base_action, eef=eef" in source
    assert "_transform_action(" in source
