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
