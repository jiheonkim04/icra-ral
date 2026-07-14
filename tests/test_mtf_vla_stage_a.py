from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from tca_map.smolvla.mtf_vla_stage_a import (
    STAGE_A_POLICY_ORDER,
    STAGE_A_RESET_SEEDS,
    STAGE_B_RESET_SEEDS,
    build_stage_a_manifest,
    build_stage_b_manifest,
    validate_stage_a_manifest,
    validate_stage_b_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path: Path, task_manifest: Path, checkpoint_manifest: Path) -> Namespace:
    return Namespace(
        date="2026-07-14",
        base_path="/mnt/c/assets/checkpoints/smolvla_libero",
        lora_root="/mnt/c/Users/jiheo/tca_map/runs/mtf_vla_checkpoints/mtf_r20_ret100",
        libero_config_dir="/home/jiheon/.libero",
        wsl_repo_root="/mnt/c/Users/jiheo/tca_map",
        official_task_manifest=str(task_manifest),
        checkpoint_manifest=str(checkpoint_manifest),
        stage_a_manifest=str(tmp_path / "stage_a_manifest.json"),
        stage_a_manifest_md=str(tmp_path / "stage_a_manifest.md"),
        stage_a_output=str(tmp_path / "stage_a_result.json"),
        stage_a_md=str(tmp_path / "stage_a_result.md"),
        stage_a_partial_output=str(tmp_path / "stage_a_partial_result.json"),
        stage_a_preflight_output=str(tmp_path / "stage_a_preflight.json"),
        stage_b_manifest=str(tmp_path / "stage_b_manifest.json"),
        stage_b_manifest_md=str(tmp_path / "stage_b_manifest.md"),
        stage_b_output=str(tmp_path / "stage_b_result.json"),
        stage_b_md=str(tmp_path / "stage_b_result.md"),
        stage_b_partial_output=str(tmp_path / "stage_b_partial_result.json"),
    )


def _task_manifest() -> dict:
    return {
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": index,
                "instruction": f"instruction {index}",
                "suite_task_count": 20,
                "selection_rule": "fixture",
            }
            for index in range(20)
        ]
    }


def _checkpoint_manifest() -> dict:
    variants = []
    for variant in STAGE_A_POLICY_ORDER:
        if variant == "frozen_smolvla":
            continue
        variants.append(
            {
                "variant": variant,
                "checkpoint_path": f"runs\\mtf_vla_checkpoints\\mtf_r20_ret100\\{variant}\\seed_101",
                "adapter_model_sha256": "A" * 64,
                "adapter_config_sha256": "B" * 64,
                "disk_reload": True,
                "seed": 101,
                "training_event_count": 10,
                "validation_action_l2_mean": 0.1,
                "adapter_minus_base_action_l2_p95": 0.2,
            }
        )
    return {
        "final_decision": "MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY",
        "stage_a_allowed": True,
        "checkpoint_root": "runs/mtf_vla_checkpoints/mtf_r20_ret100",
        "variant_count": 4,
        "variants": variants,
    }


def _stage_a_result() -> dict:
    return {
        "final_decision": "MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED",
        "completed_episode_count": 50,
        "summary": {"exception_count": 0},
    }


def test_build_stage_a_manifest_freezes_five_policy_ten_pair_design(tmp_path: Path) -> None:
    task_manifest = tmp_path / "official_tasks.json"
    checkpoint_manifest = tmp_path / "checkpoints.json"
    _write_json(task_manifest, _task_manifest())
    _write_json(checkpoint_manifest, _checkpoint_manifest())

    manifest = build_stage_a_manifest(_args(tmp_path, task_manifest, checkpoint_manifest))

    assert manifest["final_decision"] == "MTF_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert manifest["policy_order"] == STAGE_A_POLICY_ORDER
    assert manifest["stage_a_reset_seeds"] == STAGE_A_RESET_SEEDS
    assert manifest["stage_a_pair_count_per_policy"] == 10
    assert manifest["planned_episode_count"] == 50
    assert len(manifest["tasks"]) == 5
    assert [task["source_official_task_manifest_index"] for task in manifest["tasks"]] == [0, 4, 8, 12, 16]
    assert manifest["closed_loop_experiment_happened"] is False
    assert manifest["confirmatory_test_tuning_happened"] is False
    assert len({episode["episode_id"] for episode in manifest["episodes"]}) == 50
    for policy in STAGE_A_POLICY_ORDER:
        policy_pairs = {episode["pair_id"] for episode in manifest["episodes"] if episode["policy"] == policy}
        assert policy_pairs == {pair["pair_id"] for pair in manifest["pairs"]}


def test_validate_stage_a_manifest_rejects_policy_drift(tmp_path: Path) -> None:
    task_manifest = tmp_path / "official_tasks.json"
    checkpoint_manifest = tmp_path / "checkpoints.json"
    _write_json(task_manifest, _task_manifest())
    _write_json(checkpoint_manifest, _checkpoint_manifest())
    manifest = build_stage_a_manifest(_args(tmp_path, task_manifest, checkpoint_manifest))
    manifest["policies"][0]["policy"] = "renamed_base"

    with pytest.raises(ValueError, match="policies"):
        validate_stage_a_manifest(manifest)


def test_build_stage_a_manifest_requires_verified_checkpoints(tmp_path: Path) -> None:
    task_manifest = tmp_path / "official_tasks.json"
    checkpoint_manifest = tmp_path / "checkpoints.json"
    _write_json(task_manifest, _task_manifest())
    bad = _checkpoint_manifest()
    bad["stage_a_allowed"] = False
    _write_json(checkpoint_manifest, bad)

    with pytest.raises(ValueError, match="does not allow Stage A"):
        build_stage_a_manifest(_args(tmp_path, task_manifest, checkpoint_manifest))


def test_build_stage_b_manifest_freezes_all_task_forty_pair_design(tmp_path: Path) -> None:
    task_manifest = tmp_path / "official_tasks.json"
    checkpoint_manifest = tmp_path / "checkpoints.json"
    stage_a_result = tmp_path / "stage_a_result.json"
    _write_json(task_manifest, _task_manifest())
    _write_json(checkpoint_manifest, _checkpoint_manifest())
    _write_json(stage_a_result, _stage_a_result())
    args = _args(tmp_path, task_manifest, checkpoint_manifest)
    args.stage_a_output = str(stage_a_result)

    manifest = build_stage_b_manifest(args)

    assert manifest["final_decision"] == "MTF_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert manifest["policy_order"] == STAGE_A_POLICY_ORDER
    assert manifest["stage_b_reset_seeds"] == STAGE_B_RESET_SEEDS
    assert manifest["stage_b_pair_count_per_policy"] == 40
    assert manifest["planned_episode_count"] == 200
    assert len(manifest["tasks"]) == 20
    assert manifest["identity_overlap_verification"]["overlap_with_stage_a_reset_seeds"] == 0
    assert manifest["stage_a_result"]["final_decision"] == "MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    validate_stage_b_manifest(manifest)
    for policy in STAGE_A_POLICY_ORDER:
        policy_pairs = {episode["pair_id"] for episode in manifest["episodes"] if episode["policy"] == policy}
        assert policy_pairs == {pair["pair_id"] for pair in manifest["pairs"]}
