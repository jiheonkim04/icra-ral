from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports" / "a2c2_prior"


def _load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def test_action_semantics_authority_and_historical_result_are_frozen() -> None:
    authorization = _load("official_action_semantics_continuation_authorization.json")
    historical = _load("fidelity_corrected_actual_path_smoke_result.json")

    assert authorization["authority_sha256"] == (
        "CDC674DB9E0EFDC85F3529FA7387D4E3A9BD31DF91A66B0D9B87C1279DA6C0B0"
    )
    assert authorization["authorized_change_count"] == 1
    assert authorization["active_continuation_state"] == (
        "A2C2_OFFICIAL_ACTION_SEMANTICS_CORRECTION_AUTHORIZED"
    )
    assert authorization["historical_result_preserved"]["role"] == (
        "HISTORICAL_LOCAL_STRICT_RAW_BOUND_GATE_RESULT"
    )
    assert historical["final_decision"] == "CORRECTED_A2C2_EVALUATION_INVALID"
    assert historical["verification_panel_started"] is False


def test_official_action_path_audit_is_complete_and_exact() -> None:
    audit = _load("official_action_semantics_audit_result.json")

    assert audit["final_decision"] == "OFFICIAL_ACTION_SEMANTICS_VERIFIED"
    assert audit["action_normalization"]["base_prior_statistics_exact_match"] is True
    assert audit["official_environment_path"]["action_dimension"] == 7
    assert audit["official_environment_path"]["action_spec_low"] == [-1.0] * 7
    assert audit["official_environment_path"]["action_spec_high"] == [1.0] * 7
    assert audit["native_handling"]["external_evaluator_clip"] is False
    assert audit["native_handling"]["same_post_policy_path_for_base_and_prior"] is True
    assert audit["audit_checks"]["effective_action_observable_passively"] is True
    assert audit["audit_checks"]["action_path_implementation_defect_found"] is False


def test_preregistered_protocol_hash_and_practical_rule_are_frozen() -> None:
    protocol_path = REPORTS / "official_action_semantics_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_hash = (REPORTS / "official_action_semantics_protocol.sha256").read_text(
        encoding="utf-8"
    ).split()[0]

    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest().upper() == expected_hash
    assert protocol["frozen_before_new_semantics_smoke"] is True
    assert protocol["scientific_legality"]["raw_nominal_bound_exceedance"] == (
        "required diagnostic, not automatic invalidity"
    )
    assert protocol["scientific_legality"]["external_wrapper_clipping"] is False
    assert protocol["practical_prior_instability_rule"] == {
        "purpose": "Do not call a negligible accepted raw excursion prior-specific instability.",
        "development_identities_required": 2,
        "reproducible_on_every_identity": True,
        "prior_minus_matched_base_max_exceedance_min": 0.05,
        "and_one_of": {
            "prior_minus_base_raw_element_exceedance_fraction_min": 0.02,
            "prior_minus_base_native_arm_clip_step_fraction_min": 0.10,
        },
        "outcome_used_to_set_rule": False,
    }
    assert protocol["outcome_suppressed_smoke"]["planned_technical_traces"] == 4
    assert protocol["outcome_suppressed_smoke"]["scientific_rows"] == 0
    assert protocol["scientific_panel_after_smoke_pass_only"]["planned_rows"] == 45


def test_corrected_runner_does_not_clip_policy_actions_externally() -> None:
    runner = (REPO_ROOT / "scripts" / "run_a2c2_fidelity_corrected.py").read_text(
        encoding="utf-8"
    )

    # The one np.clip call is the unchanged quaternion-to-axis-angle conversion,
    # not a policy-action transform. Every policy action goes directly to env.step.
    assert runner.count("np.clip(") == 1
    assert "torch.clamp" not in runner
    assert "env.step(np.clip" not in runner
    assert "env.step(torch.clamp" not in runner
    assert "obs, _, done, _ = env.step(action)" in runner
    assert "obs, _, _, _ = env.step(action)" in runner


def test_host_monitor_classifies_the_new_mode_as_a_smoke() -> None:
    monitor = (REPO_ROOT / "scripts" / "monitor_a2c2_fidelity_corrected.ps1").read_text(
        encoding="utf-8"
    )

    assert '[ValidateSet("smoke", "semantics_smoke", "panel")]' in monitor
    assert '($Mode -in @("smoke", "semantics_smoke"))' in monitor
    assert "CORRECTED_A2C2_OFFICIAL_SEMANTICS_SMOKE_PASS" in monitor


def test_host_monitor_separates_pagefile_counter_drift_from_write_activity() -> None:
    monitor = (REPO_ROOT / "scripts" / "monitor_a2c2_fidelity_corrected.ps1").read_text(
        encoding="utf-8"
    )

    assert "$peakPagesOutputPerSec" in monitor
    assert "$pagefileWriteActivity" in monitor
    assert "$pagefileCounterDriftWithoutWrites" in monitor
    assert "-not $memoryReleaseVerified -or $pagefileWriteActivity" in monitor
    assert "-not $memoryReleaseVerified -or $pagefileGrowthMiB -gt 0" not in monitor


def test_valid_official_semantics_smoke_opens_only_the_frozen_panel() -> None:
    result = _load("official_action_semantics_smoke_result.json")

    assert result["final_decision"] == "CORRECTED_A2C2_OFFICIAL_SEMANTICS_SMOKE_PASS"
    assert result["execution"]["technical_traces"] == 4
    assert result["execution"]["total_simulator_steps"] == 320
    assert result["execution"]["scientific_episode_rows"] == 0
    assert result["execution"]["task_success_inspected_by_runner"] is False
    assert result["execution"]["task_success_persisted"] is False
    assert result["execution"]["task_success_counted"] is False
    assert all(trace["action_semantics_valid"] for trace in result["traces"])
    assert result["native_path_summary"]["external_action_clip_added"] is False
    assert result["native_path_summary"]["native_arm_clip_step_count"] == 0
    assert result["practical_prior_instability"]["reproducible"] is False
    assert result["resources"]["swap_total_bytes"] == 0
    assert result["resources"]["pagefile_current_growth_mib"] == 0
    assert result["next_action"] == "The unchanged 45-row corrected panel is now authorized."


def test_panel_telemetry_failure_is_preserved_and_candidate_is_quarantined() -> None:
    failure = _load("official_action_semantics_panel_host_telemetry_failed_attempt.json")

    assert failure["completed_scientific_rows"] == 45
    assert failure["internal_candidate_decision"] == "CORRECTED_A2C2_NO_REPEATABLE_DELAY_GAP"
    assert failure["internal_candidate_decision_adopted"] is False
    assert failure["host_monitor_decision"] == "A2C2_CORRECTED_HOST_FAIL_MEMORY_OR_PAGEFILE"
    assert failure["root_cause"]["peak_page_writes_per_sec"] == 0
    assert failure["root_cause"]["peak_pages_output_per_sec"] == 0
    assert failure["root_cause"]["wsl_swap_total_bytes"] == 0
    assert failure["repair"]["repair_count_for_this_root"] == 1
    assert failure["repair"]["required_rerun"].startswith("identical frozen 45-row panel")


def test_host_telemetry_repair_protocol_is_hashed_before_identical_rerun() -> None:
    protocol_path = REPORTS / "official_action_semantics_panel_host_telemetry_repair_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_hash = (
        REPORTS / "official_action_semantics_panel_host_telemetry_repair_protocol.sha256"
    ).read_text(encoding="utf-8").split()[0]

    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest().upper() == expected_hash
    assert protocol["frozen_before_identical_rerun"] is True
    assert protocol["repair_count_for_root"] == 1
    assert protocol["identical_rerun"]["planned_rows"] == 45
    assert protocol["identical_rerun"]["start_from_zero_rows"] is True
    assert protocol["identical_rerun"]["model_action_path_and_adjudicator_unchanged"] is True


def test_accepted_corrected_panel_closes_on_no_repeatable_delay_gap() -> None:
    result_path = REPORTS / "official_action_semantics_corrected_panel_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_hash = (REPORTS / "official_action_semantics_corrected_panel_result.sha256").read_text(
        encoding="utf-8"
    ).split()[0]

    assert hashlib.sha256(result_path.read_bytes()).hexdigest().upper() == expected_hash
    assert result["final_decision"] == "CORRECTED_A2C2_NO_REPEATABLE_DELAY_GAP"
    assert result["accepted_execution"]["host_decision"] == "A2C2_CORRECTED_HOST_PANEL_PASS"
    assert result["accepted_execution"]["completed_scientific_rows"] == 45
    assert result["accepted_execution"]["duplicate_scientific_keys"] == 0
    assert result["accepted_execution"]["prior_module_forward_count"] == 2148
    assert result["results"]["successes"] == {
        "BASE_STANDARD_E10_D0": 11,
        "BASE_DELAYED_E40_D10": 9,
        "PRIOR_DELAYED_E40_D10": 9,
    }
    assert result["results"]["gates"]["manifest_valid"] is True
    assert result["results"]["gates"]["base_competent"] is True
    assert result["results"]["gates"]["repeatable_delay_gap"] is False
    assert result["results"]["repeatable_delay_gap_rule"]["observed_standard_minus_delayed"] == 2
    assert result["raw_action_diagnostics"]["all_nominal_exceedance_events_persisted"] is True
    assert result["controller_native_diagnostics"]["all_controller_actions_accepted"] is True
    assert result["resources"]["pagefile_write_activity"] is False
    assert result["resources"]["wsl_swap_total_bytes"] == 0
    assert result["repeatability"]["first_attempt_and_rerun_scientific_subset_mismatch_count"] == 0
    assert result["route"]["close_claim_specific_thesis"] is True
    assert result["route"]["additional_prior_authorized"] is False
    assert result["route"]["ours_authorized"] is False
