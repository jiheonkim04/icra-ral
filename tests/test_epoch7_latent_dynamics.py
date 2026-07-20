from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tca_map.epoch7_latent_dynamics import (
    adjudicate_discovery,
    apply_intervention,
    body_descendant_ids,
    compare_observations,
    iter_episode_specs,
    load_json,
    target_contact_state,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "reports/epoch7_latent_dynamics_attribution/discovery_protocol.json"


class FakeModel:
    def __init__(self) -> None:
        self.body_names = ["world", "plate_1_main", "plate_child", "robot0_gripper", "wrong_object"]
        self.body_parentid = np.array([0, 0, 1, 0, 0])
        self.body_mass = np.array([0.0, 2.0, 0.5, 1.0, 1.0])
        self.body_inertia = np.ones((5, 3), dtype=float)
        self.nbody = len(self.body_names)
        self.joint_names = ["drawer_joint"]
        self.jnt_dofadr = np.array([0])
        self.njnt = 1
        self.nv = 1
        self.dof_damping = np.array([5.0])
        self.geom_names = ["plate_geom", "plate_child_geom", "gripper_geom", "wrong_geom"]
        self.geom_bodyid = np.array([1, 2, 3, 4])
        self.geom_contype = np.array([1, 1, 1, 1])
        self.geom_friction = np.ones((4, 3), dtype=float)
        self.ngeom = len(self.geom_names)

    def body_name2id(self, name: str) -> int:
        return self.body_names.index(name)

    def body_id2name(self, index: int) -> str:
        return self.body_names[index]

    def joint_name2id(self, name: str) -> int:
        return self.joint_names.index(name)

    def joint_id2name(self, index: int) -> str:
        return self.joint_names[index]

    def geom_id2name(self, index: int) -> str:
        return self.geom_names[index]


def test_protocol_is_frozen_and_has_24_outcome_free_episode_specs() -> None:
    protocol = load_json(PROTOCOL_PATH)
    specs = list(iter_episode_specs(protocol))

    assert validate_protocol(protocol) == []
    assert protocol["ours_authorized"] is False
    assert len(specs) == 24
    assert specs[0]["episode_id"] == "eval0_state0_standard"
    assert specs[-1]["episode_id"] == "eval9_state2_latent_dynamics_intervention"
    assert specs[0]["seed"] == specs[1]["seed"]
    assert "success" not in specs[0]


def test_body_descendants_and_exact_target_contact_exclude_wrong_object() -> None:
    model = FakeModel()
    target_contact = SimpleNamespace(geom1=2, geom2=1)
    wrong_contact = SimpleNamespace(geom1=2, geom2=3)

    assert body_descendant_ids(model, "plate_1_main") == {1, 2}
    positive = target_contact_state(SimpleNamespace(model=model, data=SimpleNamespace(ncon=1, contact=[target_contact])), "plate_1_main")
    negative = target_contact_state(SimpleNamespace(model=model, data=SimpleNamespace(ncon=1, contact=[wrong_contact])), "plate_1_main")

    assert positive["target_contact"] is True
    assert negative["target_contact"] is False
    assert len(negative["other_gripper_contacts"]) == 1


def test_interventions_change_only_named_arrays_and_factor_one_is_noop() -> None:
    model = FakeModel()
    untouched_mass = model.body_mass.copy()
    friction = apply_intervention(
        model,
        {
            "axis": "target_contact_friction",
            "body_name": "plate_1_main",
            "components": [0, 1, 2],
            "collision_geoms_only": True,
            "factor": 0.25,
        },
    )
    assert friction["changed_values"] == 6
    assert np.allclose(model.geom_friction[:2], 0.25)
    assert np.allclose(model.geom_friction[2:], 1.0)
    assert np.array_equal(model.body_mass, untouched_mass)

    noop = apply_intervention(
        model,
        {"axis": "joint_damping", "joint_name": "drawer_joint", "factor": 4.0},
        factor_override=1.0,
    )
    assert noop["changed_values"] == 0
    assert model.dof_damping.tolist() == [5.0]

    mass = apply_intervention(
        model,
        {"axis": "target_mass", "body_name": "plate_1_main", "factor": 8.0},
    )
    assert mass["changed_values"] == 4
    assert model.body_mass[1] == 16.0
    assert np.allclose(model.body_inertia[1], 8.0)


def test_observation_equivalence_requires_exact_images_and_tight_proprio() -> None:
    standard = {
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((4, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
    }
    same = {key: value.copy() for key, value in standard.items()}
    changed = {key: value.copy() for key, value in standard.items()}
    changed["agentview_image"][0, 0, 0] = 1

    assert compare_observations(standard, same)["eligible"] is True
    assert compare_observations(standard, changed)["eligible"] is False


def test_discovery_adjudication_requires_cross_family_contact_failures() -> None:
    protocol = load_json(PROTOCOL_PATH)
    episodes = []
    for spec in iter_episode_specs(protocol):
        row = dict(spec, completed=True, success=True, target_contact_any=True)
        if spec["condition"] == "latent_dynamics_intervention" and spec["eval_id"] in (0, 2) and spec["state_index"] < 2:
            row["success"] = False
        episodes.append(row)
    experts = [
        {"eval_id": task["eval_id"], "standard_success": True, "intervention_success": True}
        for task in protocol["tasks"]
    ]
    result = adjudicate_discovery(protocol, episodes, experts)

    assert result["standard_wins"] == 4
    assert result["contact_preserving_failures"] == 4
    assert result["paired_drop_percentage_points"] > 20
    assert result["pass"] is True

