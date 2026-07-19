from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "reports" / "a2c2_prior"


def load(name: str) -> dict:
    return json.loads((REPORT_ROOT / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_fidelity_audit_preserves_v1_and_is_outcome_independent() -> None:
    audit = load("fidelity_gap_audit_result.json")
    v1 = audit["preserved_v1_evidence"]

    assert audit["primary_decision"] == "A2C2_OBJECTIVE_FIDELITY_DEFECT_FOUND"
    assert audit["audit_outcome_independent_of_v1_results"] is True
    assert audit["empirical_execution_happened"] is False
    assert audit["training_happened"] is False
    assert audit["rollout_happened"] is False
    assert audit["ours_designed_or_executed"] is False
    assert v1["fidelity_label"] == "MECHANISM_FAITHFUL_A2C2_LOCAL_PORT"
    assert v1["completed_rows"] == 45
    assert (v1["base_standard_successes"], v1["base_delayed_successes"], v1["prior_delayed_successes"]) == (
        10,
        4,
        3,
    )
    assert v1["decision"] == "A2C2_PRIOR_NO_LOCAL_IMPROVEMENT"
    assert v1["old_panel_reusable_for_tuning"] is False
    assert v1["paper_disproved"] is False
    assert sha256(REPO_ROOT / v1["runner"]) == v1["runner_sha256"]
    assert sha256(REPO_ROOT / v1["module"]) == v1["module_sha256"]


def test_audit_pins_public_author_pair_and_full_dataset() -> None:
    audit = load("fidelity_gap_audit_result.json")
    artifacts = audit["primary_sources"]["official_author_artifacts"]
    base = artifacts["paired_base"]
    prior = artifacts["paper_structural_match_prior"]
    dataset = artifacts["paper_structural_match_dataset"]

    assert base["repo_id"] == "k1000dai/smolvla_libero_spatial_scratch"
    assert base["revision"] == "caa0efcb24e261574c824366526c5775d3664cac"
    assert base["model_sha256"] == "45F3B6FC1B8AE0B7CF3AB8EBD22336AB23EB3798A8BFEF027F5D45596C45A9BE"
    assert prior["repo_id"] == "k1000dai/residual_transformer_libero_spatial_add_vlm_context"
    assert prior["revision"] == "9c89cca4aae8eecc42a20084ef414ff74f94ba05"
    assert prior["model_sha256"] == "85D00523E8273A4141E288E4F6692224D50AAF8DF99AD8CCF7E72EE7BF3AB712"
    assert prior["architecture"]["vlm_hidden_dim"] == 960
    assert prior["architecture"]["encoder_layers"] == 6
    assert dataset["revision"] == "13291992d094884988a5b71a8d3cb6613e31c116"
    assert (dataset["episodes"], dataset["frames"], dataset["tasks"]) == (432, 52970, 10)
    assert dataset["features"]["vla_actions"] == [50, 7]
    assert dataset["features"]["vlm_hidden"] == [960]


def test_audit_records_only_primary_source_objective_defects() -> None:
    audit = load("fidelity_gap_audit_result.json")
    defects = {row["id"]: row for row in audit["objective_fidelity_defects"]}

    assert set(defects) == {
        "A2C2_FD1_WRONG_PAIRED_BASE",
        "A2C2_FD2_MISSED_PUBLIC_PAPER_STRUCTURAL_MATCH_CHECKPOINT",
        "A2C2_FD3_LIVE_RGB_ORIENTATION",
    }
    assert all(row["proven_independently_of_outcomes"] for row in defects.values())
    assert audit["adjudication_gates"] == {
        "concrete_official_mismatch_proven": True,
        "not_inferred_from_poor_results": True,
        "correction_uniquely_determined_by_primary_sources": True,
        "correction_depends_on_observed_45_outcomes": False,
        "old_tasks_or_outcomes_used_for_tuning": False,
        "delay_condition_metrics_or_decision_rules_need_change": False,
        "v1_preserved_as_version_1": True,
    }
    orientation = next(row for row in audit["audit_rows"] if row["item"] == "live image orientation")
    reset = next(row for row in audit["audit_rows"] if row["item"] == "official initial state and stabilization")
    queue = next(row for row in audit["audit_rows"] if row["item"] == "queue construction")
    assert "wrong orientation" in orientation["exact_mismatch"]
    assert reset["exact_mismatch"] is None
    assert queue["exact_mismatch"] is None


def test_corrected_protocol_is_hash_frozen_and_uses_new_identities() -> None:
    protocol_path = REPORT_ROOT / "fidelity_corrected_protocol.json"
    protocol = load("fidelity_corrected_protocol.json")
    frozen_hash = (REPORT_ROOT / "fidelity_corrected_protocol.sha256").read_text(encoding="utf-8").split()[0]
    verification = protocol["verification_panel"]

    assert sha256(protocol_path) == frozen_hash == "06C817027FAB240E1892C8F344731960B0FF2EB1AC5A2BDB389EF3F2BC863F34"
    assert protocol["implementation_label"] == "A2C2_FIDELITY_CORRECTED_LOCAL_PORT"
    assert protocol["official_reproduction_claim"] is False
    assert protocol["only_fidelity_correction_iteration"] == 1
    assert protocol["second_fidelity_correction_allowed"] is False
    assert protocol["pinned_artifacts"]["full_dataset_download_required"] is False
    assert protocol["pinned_artifacts"]["correction_head_retraining"] is False
    assert [row["task_id"] for row in verification["tasks"]] == [0, 4, 8]
    assert verification["official_init_state_ids"] == [5, 6, 7, 8, 9]
    assert set(verification["official_init_state_ids"]).isdisjoint(verification["old_v1_init_state_ids"])
    assert protocol["development_smoke"]["official_init_state_id"] == 10
    assert protocol["development_smoke"]["task_id"] == 2
    assert verification["planned_rows"] == 45
    assert verification["identity_overlap_with_v1"] == 0
    assert verification["identity_overlap_with_development_smoke"] == 0


def test_corrected_protocol_preserves_decisions_resources_and_ours_boundary() -> None:
    protocol = load("fidelity_corrected_protocol.json")
    rules = protocol["decision_rules"]
    resources = protocol["resource_contract"]

    assert rules["allowed_final_decisions"] == [
        "CORRECTED_A2C2_PRIOR_IMPROVES_AND_LEAVES_RESIDUAL",
        "CORRECTED_A2C2_PRIOR_SATURATES_DELAY",
        "CORRECTED_A2C2_PRIOR_NO_IMPROVEMENT",
        "CORRECTED_A2C2_BASE_NOT_COMPETENT",
        "CORRECTED_A2C2_EVALUATION_INVALID",
        "CORRECTED_A2C2_IMPLEMENTATION_OR_RESOURCE_FAILURE",
    ]
    assert resources["temporary_wsl_memory_gib"] == 12
    assert resources["swap_gib"] == 0
    assert resources["full_backbone_residencies"] == 1
    assert resources["repeat_8_10_12_resource_qualification"] is False
    assert protocol["additional_prior_authorized_before_decision"] is False
    assert protocol["ours_authorized_before_decision"] is False
    assert protocol["paper_package_authorized_before_paper_candidate_go"] is False
