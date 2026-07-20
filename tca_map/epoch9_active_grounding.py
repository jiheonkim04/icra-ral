"""Leakage-safe experiment design utilities for Epoch 9 active grounding."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


PARTITION_DEMO_INDICES: dict[str, tuple[int, ...]] = {
    "development": tuple(range(30, 40)),
    "validation": tuple(range(40, 45)),
    "confirmation": tuple(range(45, 50)),
}

# Both the latent-property direction and the response contrast are balanced.
# The 2x/4x conditions prevent the model from being evaluated only on the
# easiest endpoint contrast calibrated during controller development.
MASS_ASSIGNMENTS: tuple[tuple[float, float], ...] = (
    (8.0, 1.0),
    (1.0, 8.0),
    (4.0, 2.0),
    (2.0, 4.0),
)
PROBE_ORDERS: tuple[tuple[str, str], ...] = (
    ("front", "back"),
    ("back", "front"),
)

LEGAL_TRACE_FIELDS: tuple[str, ...] = (
    "phase",
    "action",
    "eef_pos",
    "eef_quat",
    "controller_goal_pos",
    "controller_error",
    "rgb_diff_32",
)


def build_episode_specs(partition: str) -> list[dict[str, Any]]:
    """Return the complete predeclared physical-pair grid for a partition."""

    if partition not in PARTITION_DEMO_INDICES:
        raise ValueError(f"unknown partition: {partition}")
    specs: list[dict[str, Any]] = []
    for demo_index in PARTITION_DEMO_INDICES[partition]:
        for front_mass, back_mass in MASS_ASSIGNMENTS:
            for probe_order in PROBE_ORDERS:
                heavier_slot = "front" if front_mass > back_mass else "back"
                specs.append(
                    {
                        "episode_id": (
                            f"{partition}_demo{demo_index}_front{front_mass:g}_"
                            f"back{back_mass:g}_{probe_order[0]}-first"
                        ),
                        "partition": partition,
                        "demo_index": demo_index,
                        "front_mass_factor": front_mass,
                        "back_mass_factor": back_mass,
                        "probe_order": list(probe_order),
                        "heavier_slot_training_label": heavier_slot,
                        "lighter_slot_training_label": "back" if heavier_slot == "front" else "front",
                    }
                )
    validate_episode_specs(specs)
    return specs


def validate_episode_specs(specs: Iterable[dict[str, Any]]) -> None:
    """Reject duplicate, unbalanced, or cross-partition episode specifications."""

    rows = list(specs)
    episode_ids = [str(row["episode_id"]) for row in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("duplicate episode id")
    for row in rows:
        partition = str(row["partition"])
        if partition not in PARTITION_DEMO_INDICES:
            raise ValueError(f"unknown partition: {partition}")
        if int(row["demo_index"]) not in PARTITION_DEMO_INDICES[partition]:
            raise ValueError("demo identity crosses its declared partition")
        if tuple(row["probe_order"]) not in PROBE_ORDERS:
            raise ValueError("invalid probe order")
        front = float(row["front_mass_factor"])
        back = float(row["back_mass_factor"])
        if (front, back) not in MASS_ASSIGNMENTS or front == back:
            raise ValueError("invalid or non-relational mass assignment")


def validate_partition_disjointness() -> None:
    """Assert that reset identities cannot leak across experiment partitions."""

    partitions = {name: set(values) for name, values in PARTITION_DEMO_INDICES.items()}
    names = tuple(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if partitions[left] & partitions[right]:
                raise ValueError(f"partition overlap: {left} and {right}")


validate_partition_disjointness()
