from __future__ import annotations

from pathlib import Path

from tca_map.smolvla.official_libero_lora_seed_repro import (
    REQUIRED_BUNDLE_FILES,
    _checkpoint_dir_for_seed,
    _checkpoint_manifest,
    _choose_checkpoint_regen_decision,
    _compare_against_prior,
)
from tca_map.smolvla.official_libero_routing_design_gate import action_dim_oracle_rows


def _metric(value: float) -> dict:
    return {
        "action_l2_mean": value,
        "task_balanced_action_l2_mean": value,
        "translation_l2_mean": value,
        "rotation_l2_mean": value,
        "gripper_abs_mean": value,
    }


def _seed_summary(seed: int, *, lora: float, static: float) -> dict:
    return {
        "seed": seed,
        "record_count": 2800,
        "metrics": {
            "frozen_base": _metric(0.085),
            "rank4_lora": _metric(lora),
            "mean_action_prior": _metric(1.0),
            "frame_oracle": _metric(0.069),
            "task_oracle": _metric(0.081),
            "moira_style_instruction_task_router": _metric(0.088),
            "static_mix_val_selected": _metric(static),
        },
        "analysis": {"static_is_best_realistic": static < min(0.085, lora, 1.0, 0.088)},
        "checkpoint_bundle": {
            "status": "CHECKPOINT_COMPLETE_VERIFIED",
            "checkpoint_path": f"C:/assets/checkpoints/smolvla_libero_lora/rank4/seed_{seed}",
            "adapter_model_sha256": f"ADAPTER{seed}",
            "adapter_config_sha256": f"CONFIG{seed}",
            "disk_reload": {"loaded_from_disk": True, "model_parameter_device": "cuda:0"},
            "file_hashes": {name: {"sha256": f"{seed}-{name}", "size_bytes": 1} for name in REQUIRED_BUNDLE_FILES},
        },
    }


def _prior(seed_values: dict[int, tuple[float, float]]) -> dict:
    summaries = []
    for seed, (lora, static) in seed_values.items():
        summaries.append(_seed_summary(seed, lora=lora, static=static))
    return {
        "seed_summaries": summaries,
        "aggregate": {
            "baseline_summary": {
                "rank4_lora": {"action_l2": {"mean": 0.088, "std": 0.003}},
                "static_mix_val_selected": {"action_l2": {"mean": 0.081, "std": 0.003}},
                "frame_oracle": {"action_l2": {"mean": 0.069, "std": 0.002}},
                "task_oracle": {"action_l2": {"mean": 0.081, "std": 0.002}},
            }
        },
    }


def test_checkpoint_dir_for_seed_is_seed_isolated() -> None:
    root = Path("C:/assets/checkpoints/smolvla_libero_lora/rank4")

    assert _checkpoint_dir_for_seed(root, 11) != _checkpoint_dir_for_seed(root, 22)
    assert _checkpoint_dir_for_seed(root, 11).name == "seed_11"
    assert _checkpoint_dir_for_seed(root, 33).name == "seed_33"


def test_comparison_passes_when_metrics_within_frozen_tolerance() -> None:
    old = _prior({11: (0.084, 0.077), 22: (0.090, 0.081), 33: (0.090, 0.084)})
    new = [
        _seed_summary(11, lora=0.0845, static=0.0775),
        _seed_summary(22, lora=0.0905, static=0.0815),
        _seed_summary(33, lora=0.0905, static=0.0845),
    ]

    comparison = _compare_against_prior(prior_result=old, seed_summaries=new, tolerance=0.002)

    assert comparison["tolerance_pass"] is True
    assert comparison["static_mix_conclusion_preserved"] is True


def test_checkpoint_regen_decision_blocks_metric_drift() -> None:
    report = {
        "seed_summaries": [
            _seed_summary(11, lora=0.10, static=0.077),
            _seed_summary(22, lora=0.10, static=0.081),
            _seed_summary(33, lora=0.10, static=0.084),
        ],
        "reproduction_comparison": {"tolerance_pass": False, "static_mix_conclusion_preserved": True},
    }

    assert _choose_checkpoint_regen_decision(report) == "LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT"


def test_checkpoint_manifest_records_all_seed_paths() -> None:
    report = {
        "date": "2026-07-10 KST",
        "final_decision": "LORA_CHECKPOINTS_REGENERATED_AND_VERIFIED",
        "paths": {"checkpoint_output_root": "C:/assets/checkpoints/smolvla_libero_lora/rank4"},
        "seed_summaries": [
            _seed_summary(11, lora=0.084, static=0.077),
            _seed_summary(22, lora=0.090, static=0.081),
            _seed_summary(33, lora=0.090, static=0.084),
        ],
        "preflight": {},
    }

    manifest = _checkpoint_manifest(report)

    assert manifest["checksum_status"] == "RECORDED"
    assert [item["seed"] for item in manifest["seeds"]] == [11, 22, 33]
    assert len({item["checkpoint_path"] for item in manifest["seeds"]}) == 3


def test_action_dim_oracle_allows_missing_eval_loss_for_no_eval_loss_regen() -> None:
    base = {
        "episode_index": 1,
        "frame_index": 2,
        "task_index": 3,
        "per_dim_abs": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "gripper_sign_match": False,
        "eval_loss": None,
        "range_violation_count": 2,
    }
    lora = {
        **base,
        "per_dim_abs": [0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1],
        "gripper_sign_match": True,
        "eval_loss": None,
        "range_violation_count": 1,
    }

    rows = action_dim_oracle_rows([base], [lora])

    assert rows[0]["eval_loss"] is None
    assert rows[0]["gripper_sign_match"] is True
