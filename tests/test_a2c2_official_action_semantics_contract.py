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
