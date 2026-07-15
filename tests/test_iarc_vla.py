import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tca_map.smolvla.iarc_vla import (
    CONFLICT_COSINE_THRESHOLD,
    CONTEXT_PREFIX,
    PERTURBATION_FAMILIES,
    PROPOSAL_HASH,
    PerturbationSpec,
    classify_stage0,
    flatten_gradients,
    parameter_manifest,
    partition_stage0_manifest,
    perturb_image,
    perturb_instruction,
    perturb_raw_sample,
    perturbation_spec,
    project_clean_gradient,
    sorted_trainable_parameters,
    unflatten_vector,
    value_hash,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _row(split: str, task: int, rank: int, phase: float) -> dict[str, object]:
    episode = {"train": 100, "val": 200, "test": 300}[split] + task
    frame = rank * 10
    return {
        "split": split,
        "task_index": task,
        "episode_index": episode,
        "episode_length": 100,
        "frame_index": frame,
        "normalized_phase": phase,
        "sample_id": f"{split}_task{task}_rank{rank}",
        "dataset_global_index": ({"train": 0, "val": 1000, "test": 2000}[split] + task * 10 + rank),
        "task": f"task {task}",
    }


def _manifest(task_count: int = 4) -> dict[str, object]:
    train = []
    val = []
    test = []
    for task in range(task_count):
        train.extend(
            [
                _row("train", task, 0, 0.1),
                _row("train", task, 1, 0.5),
                _row("train", task, 2, 0.4),
                _row("train", task, 3, 0.6),
            ]
        )
        val.extend([_row("val", task, 0, 0.2), _row("val", task, 1, 0.5)])
        test.append(_row("test", task, 0, 0.5))
    return {"splits": {"train": train, "val": val, "test": test}}


def test_partition_uses_stable_midphase_ranks_without_overlap() -> None:
    partitions = partition_stage0_manifest(_manifest())

    assert [float(row["normalized_phase"]) for row in partitions["micro_fit"]] == [0.5] * 4
    assert [float(row["normalized_phase"]) for row in partitions["conflict_audit"]] == [0.4] * 4
    assert [float(row["normalized_phase"]) for row in partitions["one_check"]] == [0.6] * 4
    assert [float(row["normalized_phase"]) for row in partitions["validation"]] == [0.5] * 4
    identities = [
        str(row["sample_id"])
        for rows in partitions.values()
        for row in rows
    ]
    assert len(identities) == len(set(identities))


def test_partition_rejects_cross_partition_frame_overlap() -> None:
    manifest = _manifest(task_count=1)
    manifest["splits"]["val"][1]["episode_index"] = manifest["splits"]["train"][1]["episode_index"]
    manifest["splits"]["val"][1]["frame_index"] = manifest["splits"]["train"][1]["frame_index"]

    with pytest.raises(ValueError, match="frame overlap"):
        partition_stage0_manifest(manifest)


def test_task_balanced_family_and_severity_assignment_is_exact() -> None:
    specs = [
        perturbation_spec(
            {"task_index": task, "sample_id": f"audit_{task}"},
            partition="conflict_audit",
            sorted_task_indices=range(40),
        )
        for task in range(40)
    ]

    assert {family: sum(spec.family == family for spec in specs) for family in PERTURBATION_FAMILIES} == {
        family: 10 for family in PERTURBATION_FAMILIES
    }
    for family in PERTURBATION_FAMILIES:
        assert [spec.severity_index for spec in specs if spec.family == family] == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


def test_image_and_text_perturbations_are_deterministic_and_shape_preserving() -> None:
    image = torch.linspace(0.0, 1.0, 3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    gaussian = PerturbationSpec("gaussian_sensor_noise", 0, 0.02, 7)
    translated = PerturbationSpec("image_translation", 0, 2, 8, "right")

    first = perturb_image(image, gaussian, camera_key="observation.images.image")
    second = perturb_image(image, gaussian, camera_key="observation.images.image")
    shifted = perturb_image(image, translated, camera_key="observation.images.image")
    assert torch.equal(first, second)
    assert first.shape == image.shape
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
    assert shifted.shape == image.shape
    assert torch.equal(shifted[..., :, :2], image[..., :, :1].expand(-1, -1, 2))

    instruction = "pick up the mug"
    repeated = perturb_instruction(instruction, PerturbationSpec("instruction_repetition", 1, 2, 9))
    wrapped = perturb_instruction(instruction, PerturbationSpec("context_wrapper", 1, 2, 10))
    assert repeated == "pick up the mug ; pick up the mug ; pick up the mug"
    assert wrapped == f"{CONTEXT_PREFIX} {CONTEXT_PREFIX} {instruction}"


def test_raw_sample_perturbation_preserves_action_and_nonallowlisted_inputs() -> None:
    sample = {
        "observation.images.image": torch.full((3, 8, 8), 0.5),
        "observation.images.image2": torch.full((3, 8, 8), 0.25),
        "observation.state": torch.arange(8, dtype=torch.float32),
        "action": torch.arange(350, dtype=torch.float32).reshape(50, 7),
        "action_is_pad": torch.zeros(50, dtype=torch.bool),
        "task": "put the cup on the plate",
        "episode_index": torch.tensor(3),
    }
    spec = PerturbationSpec("gaussian_sensor_noise", 1, 0.05, 11)
    changed = perturb_raw_sample(sample, spec)

    assert value_hash(changed["action"]) == value_hash(sample["action"])
    assert value_hash(changed["observation.state"]) == value_hash(sample["observation.state"])
    assert value_hash(changed["action_is_pad"]) == value_hash(sample["action_is_pad"])
    assert value_hash(changed["episode_index"]) == value_hash(sample["episode_index"])
    assert changed["task"] == sample["task"]
    assert not torch.equal(changed["observation.images.image"], sample["observation.images.image"])


@pytest.mark.parametrize(
    ("clean", "robust", "expected", "status"),
    [
        ([1.0, 0.0], [1.0, 0.0], [1.0, 0.0], "agreeing_or_orthogonal"),
        ([-1.0, 0.0], [1.0, 0.0], [0.0, 0.0], "projected_conflict"),
        ([-1.0, 1.0], [1.0, 0.0], [0.0, 1.0], "projected_conflict"),
        ([0.0, 1.0], [1.0, 0.0], [0.0, 1.0], "agreeing_or_orthogonal"),
    ],
)
def test_projection_exact_cases(clean: list[float], robust: list[float], expected: list[float], status: str) -> None:
    result = project_clean_gradient(torch.tensor(clean), torch.tensor(robust))

    assert result["status"] == status
    assert torch.allclose(result["projected_gradient"], torch.tensor(expected), atol=1e-7, rtol=0.0)
    assert result["constraint_passed"] is True
    if status == "agreeing_or_orthogonal":
        assert torch.equal(result["projected_gradient"], torch.tensor(clean))


def test_projection_floor_nonfinite_and_positive_scale_invariance() -> None:
    below = project_clean_gradient(torch.tensor([1.0, 2.0]), torch.tensor([1e-8, 0.0]))
    assert below["status"] == "robust_gradient_below_floor"
    assert below["projected_gradient"] is None
    with pytest.raises(ValueError, match="nonfinite"):
        project_clean_gradient(torch.tensor([float("nan")]), torch.tensor([1.0]))

    clean = torch.tensor([-2.0, 3.0, 1.0])
    robust = torch.tensor([4.0, 0.0, 0.0])
    original = project_clean_gradient(clean, robust)
    scaled = project_clean_gradient(clean, robust * 7.5)
    assert torch.allclose(original["projected_gradient"], scaled["projected_gradient"], atol=1e-6, rtol=0.0)
    assert original["cosine"] < CONFLICT_COSINE_THRESHOLD


def test_flatten_unflatten_uses_sorted_names_and_exact_shapes() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.z = torch.nn.Parameter(torch.zeros(2, 2))
            self.a = torch.nn.Parameter(torch.zeros(3))

    model = Toy()
    named = sorted_trainable_parameters(model)
    assert [name for name, _ in named] == ["a", "z"]
    model.a.grad = torch.tensor([1.0, 2.0, 3.0])
    model.z.grad = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
    vector = flatten_gradients(named)
    manifest = parameter_manifest(named)
    restored = unflatten_vector(vector, manifest)

    assert torch.equal(vector, torch.arange(1.0, 8.0))
    assert tuple(restored["a"].shape) == (3,)
    assert tuple(restored["z"].shape) == (2, 2)
    assert torch.equal(torch.cat([restored["a"].reshape(-1), restored["z"].reshape(-1)]), vector)


def _healthy_summary(**overrides: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "partition_health": True,
        "perturbation_health": True,
        "action_target_health": True,
        "preflight_passed": True,
        "shared_draw_health": True,
        "lora_only_trainable": True,
        "identity_passed": True,
        "checkpoint_reload_passed": True,
        "base_unchanged": True,
        "mechanism_invariants_passed": True,
        "action_validity_passed": True,
        "memory_passed": True,
        "confirmatory_sealed": True,
        "gradient_health": True,
        "subset_fit_passed": True,
        "conflict_count": 4,
        "conflict_family_count": 2,
    }
    summary.update(overrides)
    return summary


def test_classification_obeys_direct_pass_one_check_and_false_negative_priority() -> None:
    assert classify_stage0(_healthy_summary()) == "IARC_STAGE_0A_PASS_HEADROOM_PENDING"
    assert (
        classify_stage0(_healthy_summary(conflict_count=3, conflict_family_count=2))
        == "IARC_STAGE_0A_UNDERPOWERED_ONE_CHECK_ALLOWED"
    )
    assert (
        classify_stage0(_healthy_summary(conflict_count=6, conflict_family_count=1))
        == "IARC_STAGE_0A_UNDERPOWERED_ONE_CHECK_ALLOWED"
    )
    assert (
        classify_stage0(_healthy_summary(conflict_count=0, conflict_family_count=0))
        == "IARC_DESIGN_FAILURE_NONACTING_MECHANISM"
    )
    assert (
        classify_stage0(_healthy_summary(perturbation_health=False, conflict_count=0))
        == "IARC_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_summary(gradient_health=False, conflict_count=0))
        == "IARC_LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT"
    )
    assert (
        classify_stage0(_healthy_summary(checkpoint_reload_passed=False, conflict_count=0))
        == "IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )


def test_frozen_artifact_contract_matches_proposal_hash_and_seals_test_rows() -> None:
    assert (REPO_ROOT / "reports" / "iarc_vla" / "proposal_hash.txt").read_text(encoding="utf-8").strip() == PROPOSAL_HASH
    state = json.loads((REPO_ROOT / "reports" / "autonomous_until_paper_state.json").read_text(encoding="utf-8-sig"))
    contract = state["epoch_4_cycle_16_iarc_pre_stage_0a"]
    assert contract["stage_0a_fit_rows"] == 40
    assert contract["stage_0a_audit_rows"] == 40
    assert contract["stage_0a_validation_rows"] == 40
    assert contract["confirmatory_rows_decoded_max"] == 0
    assert contract["policy_order"] == [
        "smolvla_base",
        "strong_vla_transparent_proxy",
        "iarc_vla_full",
        "iarc_unprojected_joint_replay_ablation",
        "standard_lora_clean_only",
    ]
