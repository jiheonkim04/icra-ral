from tca_map.css_shield.phase2_autopilot import assess_native_action_report, audit_diagnostic_package


def test_phase2_native_assessment_rejects_safety_only_tie(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_path = tmp_path / "native.json"
    report_path.write_text(
        """
{
  "result": {"passed": true},
  "policy": {"rollout_happened": true, "model_inference_performed": true},
  "proposal_source": {"used": "native_smolvla", "native": {"available": true}},
  "comparison": {
    "full_vs_safety_only_wrong_target_rate_reduction": 0.0,
    "full_vs_safety_only_unsafe_rate_reduction": 0.0,
    "full_vs_clipping_wrong_target_rate_reduction": 0.0,
    "full_vs_clipping_unsafe_rate_reduction": 0.8,
    "utility_drop_vs_no_shield": 0.0
  },
  "variants": [
    {"shield_variant": "no_shield", "wrong_target_action_rate_after": 0.0, "step_records": [1, 2, 3]},
    {"shield_variant": "full_css_shield", "wrong_target_action_rate_after": 0.0, "intervention_rate": 0.5, "false_positive_intervention_rate": 0.0, "step_records": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
  ]
}
""",
        encoding="utf-8",
    )

    result = assess_native_action_report(report_path)

    assert result["continue"] is False
    assert result["decision"] == "kill_or_reframe"
    assert "safety-only" in result["reason"]


def test_phase2_audit_continues_when_first_package_has_nontrivial_signal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "css_shield_ral_strength_check.json").write_text('{"continue": true}', encoding="utf-8")
    (reports / "css_shield_autopilot_state.json").write_text(
        '{"rollout_happened": true, "native_smolvla_inference_happened": true}',
        encoding="utf-8",
    )
    (reports / "css_shield_state4_scale_summary.json").write_text(
        """
{
  "continue": true,
  "full_vs_safety_wrong_target_delta": 0.58,
  "full_vs_clipping_wrong_target_delta": 0.58,
  "full_vs_clipping_unsafe_delta": 0.24,
  "full_intervention_rate": 0.58,
  "full_false_positive_rate": 0.0
}
""",
        encoding="utf-8",
    )

    audit = audit_diagnostic_package()

    assert audit["continue"] is True
    assert audit["full_shield_beats_safety_only"] is True
    assert audit["native_smolvla_action_evidence"] is True
