import json
from pathlib import Path

import numpy as np

from tca_map.smolvla.marc_vla import (
    MARCConfig,
    audit_marc_records,
    build_marc_records,
    compute_disagreement_labels,
    run_validation_search,
)
from tca_map.smolvla.marc_vla_stage_a import (
    STAGE_A_POLICY_ORDER,
    STAGE_A_RESET_SEEDS,
    _sha256_payload,
    _task_index_for_task,
    _task_index_map_from_artifact,
    validate_stage_a_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MARC = REPO_ROOT / "reports" / "marc_vla"


def _record(index: int, *, split: str, task_index: int, residual_scale: float) -> dict:
    phase = (index % 25) / 24.0
    base = np.asarray(
        [
            0.1 * np.sin(phase + task_index),
            0.1 * np.cos(phase),
            -0.05 + 0.01 * task_index,
            0.02 * phase,
            -0.03,
            0.04,
            -1.0 if phase < 0.5 else 1.0,
        ],
        dtype=np.float64,
    )
    residual = residual_scale * np.asarray([phase, 1.0 - phase, 0.2, 0.1, -0.1, 0.05, 0.5 if phase > 0.5 else -0.2])
    target = base + residual
    return {
        "split": split,
        "task": f"task {task_index}",
        "task_index": task_index,
        "episode_index": index // 25,
        "frame_index": index % 25,
        "sample_id": f"{split}-{task_index}-{index}",
        "normalized_phase": phase,
        "state": [phase, task_index, residual_scale, 0.0, 0.1, -0.1, 0.2, -0.2],
        "base_action": base.tolist(),
        "target_action": target.tolist(),
        "mean_action": [0.0] * 7,
        "lora_action": (base + 0.5 * residual).tolist(),
    }


def _fixture_records() -> list[dict]:
    rows = []
    for split, count, offset in (("train", 180, 0), ("val", 80, 1000), ("test", 80, 2000)):
        for i in range(count):
            task_index = i % 4
            residual_scale = 0.01 + 0.02 * ((i + task_index) % 5)
            rows.append(_record(offset + i, split=split, task_index=task_index, residual_scale=residual_scale))
    return rows


def test_marc_disagreement_labels_are_train_quantile_based() -> None:
    rows = _fixture_records()
    labeled, thresholds = compute_disagreement_labels(build_marc_records(rows), MARCConfig(min_scoreable_records=100))

    assert "disagreement_l2_quantile_0_60" in thresholds
    train_labels = [row["disagreement_label"] for row in labeled if row["split"] == "train"]
    assert 0.2 < sum(train_labels) / len(train_labels) < 0.6


def test_marc_stage_0_audit_fixture_passes_without_rollout() -> None:
    report = audit_marc_records(
        _fixture_records(),
        config=MARCConfig(
            min_scoreable_records=100,
            min_positive_count=10,
            min_negative_count=10,
            max_task_positive_share=0.4,
        ),
    )

    assert report["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert report["closed_loop_experiment_happened"] is False
    assert report["training_happened"] is False
    assert report["confirmatory_test_tuning_happened"] is False
    assert report["split_overlap"] == {"train_reserved": 0, "train_validation": 0, "validation_reserved": 0}
    assert report["hard_stop_reasons"] == []
    assert report["base_action_validity"] == 1.0
    assert report["gate_probe_summary"]["accuracy_margin"] >= 0.02


def test_marc_validation_search_fixture_selects_one_of_six_configs(tmp_path) -> None:
    report = run_validation_search(
        _fixture_records(),
        output_dir=tmp_path / "checkpoints",
        config=MARCConfig(
            min_scoreable_records=100,
            min_positive_count=10,
            min_negative_count=10,
            max_task_positive_share=0.4,
            validation_epochs=120,
        ),
    )

    assert report["final_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert report["closed_loop_experiment_happened"] is False
    assert report["confirmatory_test_tuning_happened"] is False
    assert report["tried_config_count"] == 6
    selected = report["selected_config"]
    assert selected["config_id"] in {item["config_id"] for item in report["tried_configs"]}
    assert selected["checkpoint_reload_max_abs_diff"] == 0.0
    assert selected["initial_delta_p95"] <= 1e-6
    assert selected["validation_metrics"]["action_validity"] == 1.0
    for item in report["tried_configs"]:
        assert (REPO_ROOT / item["checkpoint_path"]).exists() or Path(item["checkpoint_path"]).exists()


def test_marc_repo_artifacts_when_present_are_consistent() -> None:
    audit_path = MARC / "development_audit.json"
    validation_path = MARC / "validation_search.json"
    if not audit_path.exists() or not validation_path.exists():
        return

    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))

    assert audit["method"] == "MARC-VLA"
    assert audit["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert audit["closed_loop_experiment_happened"] is False
    assert validation["method"] == "MARC-VLA"
    assert validation["final_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert validation["tried_config_count"] == 6
    assert validation["selected_config"]["initial_delta_p95"] <= 1e-6


def test_marc_policy_identities_are_disk_reloadable_when_present() -> None:
    manifest_path = MARC / "policy_checkpoint_manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    assert manifest["final_decision"] == "MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert manifest["stage_a_allowed"] is True
    assert manifest["closed_loop_experiment_happened"] is False
    assert manifest["confirmatory_test_identities_used"] is False
    assert manifest["policy_identities"] == [
        "frozen_smolvla",
        "openvla_oft_l1_proxy",
        "marc_full",
        "marc_no_disagreement_gate_ablation",
        "static_l1_mixture_baseline",
    ]
    variants = {row["variant"]: row for row in manifest["variant_results"]}
    assert set(variants) == {
        "openvla_oft_l1_proxy",
        "marc_full",
        "marc_no_disagreement_gate_ablation",
        "static_l1_mixture_baseline",
    }
    for row in variants.values():
        assert row["final_decision"] == "MARC_POLICY_CHECKPOINT_VERIFIED"
        assert row["disk_reload"] is True
        assert row["initial_delta_p95"] == 0.0
        assert row["validation"]["action_validity"] == 1.0
        for filename in row["required_files"]:
            assert (REPO_ROOT / row["checkpoint_path"] / filename).exists()

    assert variants["marc_full"]["gate_metrics"]["accuracy_margin"] >= 0.02
    assert manifest["distinction"]["marc_full_vs_openvla_oft_l1_proxy_mean_l2"] > 1e-6
    assert manifest["distinction"]["marc_full_vs_marc_no_disagreement_gate_ablation_mean_l2"] > 1e-6
    assert manifest["distinction"]["marc_full_vs_static_l1_mixture_baseline_mean_l2"] > 1e-6


def test_marc_stage_a_manifest_when_present_is_frozen_and_paired() -> None:
    manifest_path = MARC / "stage_a_manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    validate_stage_a_manifest(manifest)

    assert manifest["final_decision"] == "MARC_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert manifest["closed_loop_experiment_happened"] is False
    assert manifest["confirmatory_test_tuning_happened"] is False
    assert manifest["confirmatory_test_identities_used_for_training_or_validation"] is False
    assert manifest["planned_episode_count"] == 50
    assert manifest["stage_a_pair_count_per_policy"] == 10
    assert manifest["stage_a_reset_seeds"] == STAGE_A_RESET_SEEDS == [20261209, 20261210]
    assert manifest["policy_order"] == STAGE_A_POLICY_ORDER
    assert [row["policy"] for row in manifest["policies"]] == STAGE_A_POLICY_ORDER
    assert manifest["identity_overlap_verification"]["duplicate_evaluation_keys"] == 0
    assert manifest["identity_overlap_verification"]["identical_task_reset_pairs_across_policies"] is True

    policy_records = {row["policy"]: row for row in manifest["policies"]}
    assert (
        policy_records["openvla_oft_l1_proxy"]["proxy_or_reproduction_label"]
        == "faithful_transparent_local_proxy_not_official_openvla_oft_reproduction"
    )
    assert policy_records["openvla_oft_l1_proxy"]["disk_reload"] is True
    assert policy_records["marc_full"]["disk_reload"] is True
    assert policy_records["marc_no_disagreement_gate_ablation"]["disk_reload"] is True
    assert policy_records["static_l1_mixture_baseline"]["disk_reload"] is True

    canonical_payload = {key: value for key, value in manifest.items() if key != "canonical_payload_sha256"}
    assert manifest["canonical_payload_sha256"] == _sha256_payload(canonical_payload)

    pair_sets = {}
    for policy in STAGE_A_POLICY_ORDER:
        rows = [row for row in manifest["episodes"] if row["policy"] == policy]
        assert len(rows) == 10
        pair_sets[policy] = {(row["suite"], row["task_id"], row["reset_seed"]) for row in rows}
    assert len({tuple(sorted(values)) for values in pair_sets.values()}) == 1

    expected_task_indices = {
        "libero_spatial/task_0": 34,
        "libero_spatial/task_8": 36,
        "libero_object/task_6": 27,
        "libero_goal/task_4": 18,
        "libero_10/task_2": 3,
    }
    stable_index_map = _task_index_map_from_artifact(REPO_ROOT / "reports" / "official_smolvla_stable_prediction_artifact.json")
    observed = {
        f"{task['suite']}/task_{task['task_id']}": _task_index_for_task(task, stable_index_map)
        for task in manifest["tasks"]
    }
    assert observed == expected_task_indices
