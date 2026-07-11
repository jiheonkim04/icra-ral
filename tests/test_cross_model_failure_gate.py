import json
from pathlib import Path


REQUIRED_REPORTS = [
    "reports/cross_backbone_candidate_audit.md",
    "reports/cross_benchmark_candidate_audit.md",
    "reports/openvla_oft_local_feasibility.md",
    "reports/second_vla_selection.md",
    "reports/second_benchmark_selection.md",
    "reports/cross_model_failure_manifest.json",
    "reports/cross_model_failure_result.md",
    "reports/cross_model_failure_result.json",
    "reports/cross_model_visual_annotations.json",
    "reports/cross_model_failure_generality.md",
    "reports/cross_model_latest_work_comparison.md",
    "reports/cross_model_method_readiness_decision.md",
]


def test_cross_model_gate_reports_exist():
    missing = [path for path in REQUIRED_REPORTS if not Path(path).exists()]

    assert missing == []


def test_cross_model_manifest_freezes_separate_mechanisms_and_limits():
    manifest = json.loads(Path("reports/cross_model_failure_manifest.json").read_text(encoding="utf-8"))

    assert manifest["selected_second_backbone"]["name"] == "OpenVLA-OFT"
    assert manifest["selected_second_backbone"]["download_status"] == "not_downloaded"
    assert manifest["selected_second_backbone"]["checkpoint_size_gib"] == 14.845
    assert manifest["selected_second_benchmark"]["name"] == "LIBERO-PRO"
    assert manifest["evaluation_protocol"]["total_max_episodes"] <= 96

    mechanisms = {item["short_name"]: item for item in manifest["mechanisms"]}
    assert set(mechanisms) == {"stable_grasp", "long_horizon_compounding"}
    assert mechanisms["stable_grasp"]["must_remain_separate_from"] == [
        "libero10_multi_object_long_horizon_compounding"
    ]
    assert mechanisms["long_horizon_compounding"]["must_remain_separate_from"] == [
        "spatial_drawer_bowl_stable_grasp_extraction"
    ]


def test_cross_model_result_is_blocked_before_download_or_rollout():
    result = json.loads(Path("reports/cross_model_failure_result.json").read_text(encoding="utf-8"))

    assert result["downloads_happened"] is False
    assert result["training_happened"] is False
    assert result["rollout_happened"] is False
    assert result["method_implemented"] is False
    assert result["episodes_completed"] == 0
    assert result["final_decision"] == "SECOND_BACKBONE_OR_BENCHMARK_BLOCKED"
    assert result["exact_next_implementation_prompt"] is None


def test_cross_model_visual_annotations_empty_schema_is_valid():
    annotations = json.loads(Path("reports/cross_model_visual_annotations.json").read_text(encoding="utf-8"))

    assert annotations["videos_reviewed"] == 0
    assert annotations["annotations"] == []
    assert annotations["not_run_reason"] == "SECOND_BACKBONE_DOWNLOAD_APPROVAL_REQUIRED"


def test_latest_work_comparison_contains_required_route_checks():
    text = Path("reports/cross_model_latest_work_comparison.md").read_text(encoding="utf-8")

    for term in [
        "GraspCorrect",
        "CRAFT",
        "HapticVLA",
        "UniTacVLA",
        "VLA-Reasoner",
        "AFIL",
        "FAR",
        "ProgressVLA",
        "ProgVLA",
        "VLA-Corrector",
    ]:
        assert term in text
