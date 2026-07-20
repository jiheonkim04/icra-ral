import json
from pathlib import Path

import numpy as np

from tca_map.epoch7_selective_language_grounding import (
    cag_guidance,
    canonicalize_instruction,
    character_ngrams,
    counter_cosine,
    iter_pair_specs,
    load_json,
    normalized_text,
    select_pair_specs,
    summarize_episodes,
    validate_protocol,
)
from scripts.run_epoch7_semantic_canonicalizer_preflight import metadata_row_to_bddl
from scripts.run_epoch7_language_grounding_base import build_semantic_episode_plan
from scripts.run_epoch7_method_partition_freeze import (
    hard_negative_map,
    split_demo_paths,
    split_paraphrase_rows,
)
from scripts.run_epoch7_base_action_energy_falsifier import (
    decode_official_image_bytes,
    official_libero_left_action,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "reports/epoch7_selective_language_grounding/problem_verification_protocol.json"


def test_frozen_protocol_contains_30_unique_pairs_and_no_ours() -> None:
    protocol = load_json(PROTOCOL_PATH)
    specs = list(iter_pair_specs(protocol))

    assert validate_protocol(protocol) == []
    assert protocol["ours_authorized"] is False
    assert len(specs) == 30
    assert len({spec["pair_id"] for spec in specs}) == 30
    assert specs[0]["pair_id"] == "eval0_act"
    assert specs[-1]["pair_id"] == "eval9_comp"


def test_pair_selection_is_frozen_and_outcome_independent() -> None:
    protocol = load_json(PROTOCOL_PATH)
    selected = select_pair_specs(protocol, ["eval9_comp", "eval0_obj"], None)

    assert [spec["pair_id"] for spec in selected] == ["eval0_obj", "eval9_comp"]
    assert selected[0]["seed"] == 8
    assert selected[0]["initial_state_index"] == 1
    assert "success" not in selected[0]
    assert "reward" not in selected[0]


def test_character_trigram_canonicalizer_and_frozen_tie_break() -> None:
    catalog = {
        0: "open the middle drawer of the cabinet",
        9: "turn on the stove",
    }
    exact = canonicalize_instruction("turn on the stove", catalog)
    tie = canonicalize_instruction("", catalog)

    assert normalized_text("Turn-on, THE stove!") == "turn on the stove"
    assert character_ngrams("abc", 3) == {"abc": 1}
    assert counter_cosine(character_ngrams("abc"), character_ngrams("abc")) == 1.0
    assert exact["selected_eval_id"] == 9
    assert exact["selected_instruction"] == "turn on the stove"
    assert tie["selected_eval_id"] == 9


def test_result_summary_uses_only_completed_episodes() -> None:
    summary = summarize_episodes(
        [
            {"condition": "canonical", "completed": True, "success": True},
            {"condition": "canonical", "completed": True, "success": False},
            {"condition": "paraphrase", "completed": True, "success": False},
            {"condition": "paraphrase", "completed": False, "success": True},
        ]
    )

    assert summary["completed_episodes"] == 3
    assert summary["successful_episodes"] == 1
    assert summary["by_condition"]["canonical"]["success_rate"] == 0.5
    assert summary["by_condition"]["paraphrase"]["success_rate"] == 0.0


def test_protocol_json_round_trips() -> None:
    protocol = load_json(PROTOCOL_PATH)
    assert json.loads(json.dumps(protocol))["schema_version"] == "epoch7.problem_verification_protocol.v1"


def test_cag_prior_is_frozen_without_authorizing_ours() -> None:
    protocol = load_json(PROTOCOL_PATH)
    prior = protocol["prior"]

    assert prior["formula"] == "a_empty + omega * (a_cond - a_empty)"
    assert prior["omega"] == 1.5
    assert prior["sequential_branches"] is True
    assert prior["shared_rng_state"] is True
    assert prior["one_model_resident"] is True
    assert protocol["ours_authorized"] is False


def test_cag_equation_has_the_reference_parameterization() -> None:
    assert cag_guidance(conditional=2.0, unconditional=1.0, omega=0.0) == 1.0
    assert cag_guidance(conditional=2.0, unconditional=1.0, omega=1.0) == 2.0
    assert cag_guidance(conditional=2.0, unconditional=1.0, omega=1.5) == 2.5


def test_metadata_row_to_bddl_matches_benchmark_naming() -> None:
    assert metadata_row_to_bddl(
        {"high": "act", "mid": "lexical", "low": "addition_deletion", "eval": "0", "batch_idx": "3"}
    ) == "act_lexical_addition_deletion_eval0_ver3.bddl"
    assert metadata_row_to_bddl(
        {
            "high": "comp",
            "mid": "lexical+pragmatical",
            "low": "addition_deletion+hint",
            "eval": "2",
            "batch_idx": "7",
        }
    ) == "comp_lexical+pragmatical_addition_deletion+hint_eval2_ver7.bddl"


def test_semantic_control_mismatch_plan_uses_mapping_not_outcomes() -> None:
    protocol = load_json(PROTOCOL_PATH)
    specs = list(iter_pair_specs(protocol))[:2]
    mapping = {
        "frozen_discovery_panel": {
            "predictions": [
                {
                    "pair_id": specs[0]["pair_id"],
                    "mapping_correct": True,
                    "predicted_instruction": specs[0]["canonical_instruction"],
                },
                {
                    "pair_id": specs[1]["pair_id"],
                    "mapping_correct": False,
                    "predicted_instruction": "turn on the stove",
                },
            ]
        }
    }
    plan = build_semantic_episode_plan(specs, mapping, mismatches_only=True)

    assert len(plan) == 1
    assert plan[0]["pair_id"] == specs[1]["pair_id"]
    assert plan[0]["instruction"] == "turn on the stove"
    assert "success" not in plan[0]
    assert "reward" not in plan[0]


def test_method_demo_partition_is_disjoint_and_reserves_eight_per_task() -> None:
    paths = [Path("task_demo") / f"demo_{index}.hdf5" for index in range(37)]
    split = split_demo_paths(paths)

    assert len(split["train"]) == 29
    assert len(split["validation"]) == 4
    assert len(split["confirmatory"]) == 4
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["confirmatory"]))
    assert not (set(split["validation"]) & set(split["confirmatory"]))
    assert split == split_demo_paths(reversed(paths))


def test_method_paraphrase_partition_is_deterministic_and_disjoint() -> None:
    rows = [
        {
            "eval": "0",
            "high": "act",
            "mid": "lexical",
            "low": "addition_deletion",
            "batch_idx": str(index),
            "original_instruction": "turn on the stove",
            "new_instruction": f"switch the stove on variant {index}",
        }
        for index in range(20)
    ]
    split = split_paraphrase_rows(rows)

    assert [len(split[name]) for name in ("train", "validation", "confirmatory")] == [14, 3, 3]
    ids = [{row["new_instruction"] for row in split[name]} for name in split]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert split == split_paraphrase_rows(reversed(rows))


def test_hard_negative_map_uses_text_only_and_excludes_factual_intent() -> None:
    canonical = {
        0: "put the bowl on the stove",
        1: "put the bowl on the plate",
        2: "turn on the stove",
    }
    negatives = hard_negative_map(canonical)

    assert negatives[0]["eval_id"] == 1
    assert negatives[0]["eval_id"] != 0
    assert "success" not in negatives[0]
    assert "reward" not in negatives[0]


def test_standalone_official_image_decoder_avoids_dataset_package_import() -> None:
    import cv2

    image = np.zeros((8, 9, 3), dtype=np.uint8)
    image[:, :, 1] = 127
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    decoded = decode_official_image_bytes(encoded.tobytes())

    assert decoded.size == (9, 8)
    assert np.asarray(decoded).shape == (8, 9, 3)


def test_official_libero_action_thresholds_gripper_before_bce() -> None:
    stored = np.zeros((3, 10), dtype=np.float32)
    stored[:, 9] = [-1.0, 0.0, 1.0]

    converted = official_libero_left_action(stored)

    assert converted.shape == (3, 10)
    assert converted[:, 9].tolist() == [0.0, 0.0, 1.0]
    assert np.array_equal(converted[:, :9], stored[:, :9])
