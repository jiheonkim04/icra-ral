import numpy as np

from tca_map.smolvla.official_wsl_libero_rollout import (
    RENAME_MAP,
    STATIC_MIX_CLASSIFICATION,
    alpha_zero_static_mix,
    choose_final_decision,
    static_mix_duplicate_records,
    write_reports,
)


def test_alpha_zero_static_mix_is_exactly_frozen_base():
    base = np.array([1.0, -2.0, 3.5, 4.25], dtype=np.float32)
    lora = np.array([-9.0, 8.0, 7.0, -6.0], dtype=np.float32)

    mixed = alpha_zero_static_mix(base, lora, alpha=0.0)

    assert np.array_equal(mixed, base)


def test_static_mix_duplicates_are_classified_not_run():
    records = static_mix_duplicate_records()

    assert records == {
        "static_mix_seed_11": STATIC_MIX_CLASSIFICATION,
        "static_mix_seed_22": STATIC_MIX_CLASSIFICATION,
        "static_mix_seed_33": STATIC_MIX_CLASSIFICATION,
    }


def test_rollout_decision_prioritizes_cpu_fallback():
    assert choose_final_decision({"cpu_fallback_bug": True}) == "CPU_FALLBACK_BUG"


def test_rollout_decision_ready_after_full_pilot():
    report = {
        "smoke": {"all_policies_executed": True},
        "pilot": {"executed": True, "planned_episodes": 48, "completed_episodes": 48},
    }

    assert choose_final_decision(report) == "OFFICIAL_ROLLOUT_BASELINE_READY"


def test_rollout_decision_ready_for_pilot_only_report():
    report = {
        "smoke": {},
        "pilot": {"executed": True, "planned_episodes": 48, "completed_episodes": 48},
    }

    assert choose_final_decision(report) == "OFFICIAL_ROLLOUT_BASELINE_READY"


def test_official_rename_map_preserves_saved_checkpoint_camera_keys():
    assert RENAME_MAP == {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }


def test_write_reports_does_not_write_empty_pilot_placeholder(tmp_path):
    report = {
        "final_decision": "CANONICAL_BASELINES_READY_NEEDS_MORE_ROLLOUT",
        "mode": "smoke",
        "static_mix_duplicate_runs_skipped": True,
        "old_custom_libero_7d_route_used": False,
        "smoke": {"results": [{"policy": "frozen_base"}], "errors": []},
        "pilot": {"executed": False},
    }

    write_reports(report, tmp_path)

    assert (tmp_path / "official_libero_rollout_smoke_result.json").exists()
    assert not (tmp_path / "official_libero_rollout_pilot_result.json").exists()
