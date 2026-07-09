from tca_map.smolvla.official_libero_stable_protocol import _choose_decision, sample_frame_offsets


def test_sample_frame_offsets_are_deterministic_and_cover_edges():
    assert sample_frame_offsets(10, 4) == [0, 3, 6, 9]
    assert sample_frame_offsets(3, 10) == [0, 1, 2]
    assert sample_frame_offsets(0, 5) == []


def test_choose_decision_requires_larger_prediction_artifact_when_manifest_is_good():
    manifest = {
        "summary": {
            "frame_counts": {"train": 1200, "val": 400, "test": 1200},
            "leakage_checks": {
                "episode_disjoint_train_val": True,
                "episode_disjoint_train_test": True,
                "episode_disjoint_val_test": True,
            },
        }
    }
    artifact_plan = {"status": "planned_not_generated"}

    assert _choose_decision(manifest, artifact_plan) == "NEEDS_LARGER_PREDICTION_ARTIFACT"


def test_choose_decision_rejects_small_or_leaky_manifest():
    small = {
        "summary": {
            "frame_counts": {"train": 100, "val": 50, "test": 100},
            "leakage_checks": {
                "episode_disjoint_train_val": True,
                "episode_disjoint_train_test": True,
                "episode_disjoint_val_test": True,
            },
        }
    }
    leaky = {
        "summary": {
            "frame_counts": {"train": 1200, "val": 400, "test": 1200},
            "leakage_checks": {
                "episode_disjoint_train_val": False,
                "episode_disjoint_train_test": True,
                "episode_disjoint_val_test": True,
            },
        }
    }

    assert _choose_decision(small, {"status": "planned_not_generated"}) == "NEEDS_TASK_BALANCED_SPLIT"
    assert _choose_decision(leaky, {"status": "planned_not_generated"}) == "NEEDS_TASK_BALANCED_SPLIT"
