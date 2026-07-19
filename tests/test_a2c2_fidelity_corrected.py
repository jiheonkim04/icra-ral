from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass
import json

from tca_map.smolvla.a2c2_fidelity_corrected import (
    ALLOWED_FINAL_DECISIONS,
    CONDITIONS,
    EVAL_TASK_IDS,
    VERIFICATION_INIT_STATE_IDS,
    adjudicate_panel,
    noise_seed,
    phase_feature,
    refresh_action_plan,
    rotate_live_rgb_180,
)


def test_runner_config_snapshot_supports_historical_author_dataclasses() -> None:
    import importlib.util

    runner_path = Path(__file__).resolve().parents[1] / "scripts" / "run_a2c2_fidelity_corrected.py"
    spec = importlib.util.spec_from_file_location("a2c2_corrected_runner", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    @dataclass
    class HistoricalConfig:
        device: str = "cuda"
        width: int = 512

    assert module.config_snapshot(HistoricalConfig()) == {"device": "cuda", "width": 512}


def test_live_rgb_rotation_matches_official_double_axis_reverse() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rotated = rotate_live_rgb_180(image)

    np.testing.assert_array_equal(rotated, image[::-1, ::-1])
    assert rotated.flags.c_contiguous
    roundtrip = rotate_live_rgb_180(rotated)
    np.testing.assert_array_equal(roundtrip, image)


def test_released_phase_and_noise_schedule_are_deterministic() -> None:
    np.testing.assert_allclose(phase_feature(0), np.array([0.0, 1.0], dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(phase_feature(49), np.array([0.0, 1.0], dtype=np.float32), atol=1e-6)
    assert noise_seed(0, 5, 0) == noise_seed(0, 5, 0)
    assert noise_seed(0, 5, 0) != noise_seed(0, 5, 1)
    assert noise_seed(0, 5, 0) != noise_seed(0, 6, 0)


def test_queue_refresh_exactly_matches_official_first_and_later_slices() -> None:
    old = [f"old_{index}" for index in range(50)]
    new = [f"new_{index}" for index in range(50)]
    first_plan, pending = refresh_action_plan(
        new_entries=old,
        pending_entries=[],
        execution_horizon=40,
        inference_delay=10,
        first_chunk=True,
    )
    assert first_plan == old[:40]
    assert pending == old[40:50]

    later_plan, later_pending = refresh_action_plan(
        new_entries=new,
        pending_entries=pending,
        execution_horizon=40,
        inference_delay=10,
        first_chunk=False,
    )
    assert later_plan == old[40:50] + new[10:40]
    assert later_pending == new[40:50]
    assert len(later_plan) == 40


def make_rows(
    *,
    clean_success: set[tuple[int, int]],
    delayed_success: set[tuple[int, int]],
    prior_success: set[tuple[int, int]],
) -> list[dict]:
    outcomes = {
        "BASE_STANDARD_E10_D0": clean_success,
        "BASE_DELAYED_E40_D10": delayed_success,
        "PRIOR_DELAYED_E40_D10": prior_success,
    }
    rows = []
    for condition in CONDITIONS:
        for task_id in EVAL_TASK_IDS:
            for init_state_id in VERIFICATION_INIT_STATE_IDS:
                with_prior = condition == "PRIOR_DELAYED_E40_D10"
                rows.append(
                    {
                        "condition": condition,
                        "task_id": task_id,
                        "official_init_state_id": init_state_id,
                        "success": (task_id, init_state_id) in outcomes[condition],
                        "action_finite": True,
                        "action_legal": True,
                        "base_model_forward_count": 1,
                        "prior_module_forward_count": 1 if with_prior else 0,
                        "prior_mean_abs_correction": 0.01 if with_prior else 0.0,
                        "exception": None,
                    }
                )
    return rows


def test_corrected_adjudicator_returns_no_improvement_without_posthoc_rescue() -> None:
    clean = {(task, init) for task in EVAL_TASK_IDS for init in (5, 6, 7)}
    delayed = {(task, 5) for task in EVAL_TASK_IDS}
    rows = make_rows(clean_success=clean, delayed_success=delayed, prior_success=set(delayed))
    result = adjudicate_panel(rows)

    assert result["final_decision"] == "CORRECTED_A2C2_PRIOR_NO_IMPROVEMENT"
    assert result["successes"] == {
        "BASE_STANDARD_E10_D0": 9,
        "BASE_DELAYED_E40_D10": 3,
        "PRIOR_DELAYED_E40_D10": 3,
    }
    assert result["gates"]["manifest_valid"] is True
    assert result["gates"]["repeatable_delay_gap"] is True
    assert result["gates"]["prior_improves"] is False


def test_corrected_adjudicator_distinguishes_residual_and_saturation() -> None:
    clean = {(task, init) for task in EVAL_TASK_IDS for init in (5, 6, 7)}
    delayed = {(task, 5) for task in EVAL_TASK_IDS}
    leaves_residual = delayed | {(0, 6), (4, 6)}
    residual_result = adjudicate_panel(
        make_rows(clean_success=clean, delayed_success=delayed, prior_success=leaves_residual)
    )
    assert residual_result["final_decision"] == "CORRECTED_A2C2_PRIOR_IMPROVES_AND_LEAVES_RESIDUAL"
    assert residual_result["gates"]["prior_improves"] is True
    assert residual_result["gates"]["residual_remains"] is True

    saturating = set(clean)
    saturation_result = adjudicate_panel(
        make_rows(clean_success=clean, delayed_success=delayed, prior_success=saturating)
    )
    assert saturation_result["final_decision"] == "CORRECTED_A2C2_PRIOR_SATURATES_DELAY"
    assert saturation_result["gates"]["prior_saturates"] is True


def test_corrected_adjudicator_distinguishes_base_invalid_and_resource_failures() -> None:
    clean_not_competent = {(0, init) for init in (5, 6, 7)} | {(4, init) for init in (5, 6, 7)}
    delayed = {(0, 5)}
    base_result = adjudicate_panel(
        make_rows(
            clean_success=clean_not_competent,
            delayed_success=delayed,
            prior_success=set(delayed),
        )
    )
    assert base_result["final_decision"] == "CORRECTED_A2C2_BASE_NOT_COMPETENT"

    valid_clean = {(task, init) for task in EVAL_TASK_IDS for init in (5, 6, 7)}
    rows = make_rows(clean_success=valid_clean, delayed_success={(0, 5)}, prior_success={(0, 5)})
    invalid_result = adjudicate_panel(rows[:-1])
    assert invalid_result["final_decision"] == "CORRECTED_A2C2_EVALUATION_INVALID"

    resource_result = adjudicate_panel([], infrastructure_failure=True)
    assert resource_result["final_decision"] == "CORRECTED_A2C2_IMPLEMENTATION_OR_RESOURCE_FAILURE"
    assert resource_result["final_decision"] in ALLOWED_FINAL_DECISIONS


def test_illegal_action_invalidates_the_corrected_panel() -> None:
    clean = {(task, init) for task in EVAL_TASK_IDS for init in (5, 6, 7)}
    delayed = {(task, 5) for task in EVAL_TASK_IDS}
    rows = make_rows(clean_success=clean, delayed_success=delayed, prior_success=set(delayed))
    rows[0]["action_legal"] = False

    result = adjudicate_panel(rows)
    assert result["final_decision"] == "CORRECTED_A2C2_EVALUATION_INVALID"
    assert result["gates"]["manifest_valid"] is False


def test_runner_and_downloader_pin_the_frozen_path_without_training() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (repo_root / "scripts" / "run_a2c2_fidelity_corrected.py").read_text(encoding="utf-8")
    downloader = (repo_root / "scripts" / "download_a2c2_corrected_assets.ps1").read_text(encoding="utf-8")
    launcher = (repo_root / "scripts" / "run_a2c2_fidelity_corrected_wsl.sh").read_text(encoding="utf-8")

    assert 'choices=("metadata_preflight", "smoke", "panel", "adjudicate")' in runner
    assert "A2C2_FIDELITY_CORRECTED_LOCAL_PORT" not in runner  # imported from the frozen helper
    assert "predict_action_chunk" in runner
    assert "rotate_live_rgb_180" in runner
    assert "refresh_action_plan" in runner
    assert "optimizer" not in runner.lower()
    assert ".backward(" not in runner
    assert "smolvla_libero_spatial_scratch" in downloader
    assert "residual_transformer_libero_spatial_add_vlm_context" in downloader
    assert "libero-spatial-smolvla-add-vlm-context" not in downloader
    assert "a2c2-libero-checkpoint-compat-c197a01/src" in launcher
    assert "a2c2-libero/src" not in launcher


def test_runner_requires_strict_checkpoint_compatible_author_source() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (repo_root / "scripts" / "run_a2c2_fidelity_corrected.py").read_text(encoding="utf-8")

    assert "CHECKPOINT_COMPATIBLE_COMMIT" in runner
    assert 'prior_projection_shape != [512, 512]' in runner
    assert '"strict_safetensor_load": True' in runner
    assert "strict=False" not in runner
    assert "config_snapshot(base_config)" in runner


def test_corrected_actual_path_result_stops_before_scientific_panel() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (repo_root / "reports" / "a2c2_prior" / "fidelity_corrected_actual_path_smoke_result.json")
        .read_text(encoding="utf-8")
    )

    assert result["final_decision"] == "CORRECTED_A2C2_EVALUATION_INVALID"
    assert result["implementation_label"] == "A2C2_FIDELITY_CORRECTED_LOCAL_PORT"
    assert result["verification_panel_started"] is False
    assert result["scientific_episode_rows"] == 0
    assert result["training_happened"] is False
    assert result["ours_designed_or_executed"] is False
    assert result["completed_actual_path_run"]["conditions"]["BASE_STANDARD_E10_D0"]["raw_action_legal"] is False
    assert result["completed_actual_path_run"]["conditions"]["PRIOR_DELAYED_E40_D10"]["raw_action_legal"] is False
    assert result["frozen_route_adjudication"]["additional_prior_authorized"] is False
