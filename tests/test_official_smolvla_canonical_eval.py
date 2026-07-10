import copy
from pathlib import Path

import numpy as np
import pytest

from tca_map.smolvla.official_canonical_eval import (
    CanonicalEvalError,
    aggregate_seed_metrics,
    canonical_rng_seed,
    immutable_frame_identity,
    select_static_alpha,
    validate_checkpoint_manifest,
)
from tca_map.smolvla.official_libero_failure_mining import _metric_row


def _sample(label=0):
    return {
        "split": "val",
        "sample_id": "val_ep1_frame2",
        "dataset_local_index": 12,
        "dataset_global_index": 102,
        "episode_index": 1,
        "frame_index": 2,
        "episode_length": 50,
        "task_index": 7,
        "target_action": [label],
        "action_l2": 999,
    }


def test_canonical_rng_seed_uses_immutable_frame_identity_not_label_fields():
    first = _sample(label=0)
    second = _sample(label=1)
    assert immutable_frame_identity(first) == immutable_frame_identity(second)
    assert canonical_rng_seed(101, first) == canonical_rng_seed(101, second)
    assert canonical_rng_seed(101, first) != canonical_rng_seed(202, first)
    moved = copy.deepcopy(first)
    moved["frame_index"] = 3
    assert canonical_rng_seed(101, first) != canonical_rng_seed(101, moved)


def test_static_alpha_selection_uses_validation_only():
    records = []
    for split, target in [("val", 0.5), ("test", 1.0)]:
        for eval_seed in [101, 202]:
            records.append(
                {
                    "split": split,
                    "sample_id": f"{split}_{eval_seed}",
                    "dataset_local_index": eval_seed,
                    "dataset_global_index": eval_seed,
                    "episode_index": 0,
                    "frame_index": eval_seed,
                    "episode_length": 100,
                    "task_index": 0,
                    "task": "task",
                    "phase": "early",
                    "normalized_phase": 0.1,
                    "eval_seed": eval_seed,
                    "canonical_rng_seed": eval_seed,
                    "base_action": [0.0],
                    "lora_action": [1.0],
                    "target_action": [target],
                    "base_action_l2": abs(0.0 - target),
                    "lora_action_l2": abs(1.0 - target),
                }
            )

    selected = select_static_alpha(
        records,
        [101, 202],
        action_min=np.asarray([-10.0], dtype=np.float32),
        action_max=np.asarray([10.0], dtype=np.float32),
        grid=[0.0, 0.5, 1.0],
    )

    assert selected["selected_alpha"] == 0.5
    assert selected["selection_split"] == "val"
    assert selected["test_metrics_influence_selection"] is False


def test_aggregate_seed_metrics_reports_mean_and_std_over_eval_seeds():
    rows = []
    action_min = np.asarray([-10.0], dtype=np.float32)
    action_max = np.asarray([10.0], dtype=np.float32)
    for eval_seed, pred in [(101, [0.0]), (202, [2.0])]:
        rows.append(
            _metric_row(
                sample_meta={
                    "sample_id": str(eval_seed),
                    "dataset_local_index": eval_seed,
                    "dataset_global_index": eval_seed,
                    "episode_index": 0,
                    "frame_index": eval_seed,
                    "episode_length": 10,
                    "task_index": 0,
                    "task": "task",
                    "phase": "early",
                    "split": "test",
                    "eval_seed": eval_seed,
                    "canonical_rng_seed": eval_seed,
                },
                pred=np.asarray(pred, dtype=np.float32),
                target=np.asarray([1.0], dtype=np.float32),
                eval_loss=None,
                action_min=action_min,
                action_max=action_max,
            )
        )

    metrics = aggregate_seed_metrics(rows, [101, 202], seed=7)

    assert metrics["action_l2_mean"] == 1.0
    assert metrics["action_l2_mean_std_over_eval_seeds"] == 0.0
    assert metrics["per_seed"]["101"]["sample_count"] == 1
    assert metrics["per_task"]["0"]["eval_seed_count"] == 2


def test_checkpoint_manifest_validator_rejects_missing_required_bundle(tmp_path):
    root = tmp_path / "seed_11"
    root.mkdir()
    manifest = {
        "seeds": [
            {
                "seed": 11,
                "status": "CHECKPOINT_COMPLETE_VERIFIED",
                "checkpoint_path": str(root),
                "file_hashes": {},
                "disk_reload": {"loaded_from_disk": True, "loaded_policy_type": "PeftModel"},
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    with pytest.raises(CanonicalEvalError) as exc:
        validate_checkpoint_manifest(Path(path), [11])

    assert exc.value.code == "CHECKPOINT_IDENTITY_FAILED"
