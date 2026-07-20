#!/usr/bin/env python3
"""Freeze the Epoch 9B v2 lane and task-preservation gate before outcomes."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json

OUTPUT_JSON = REPO_ROOT / "reports/epoch9b_v2_task_preservation_protocol.json"
OUTPUT_MD = REPO_ROOT / "reports/epoch9b_v2_task_preservation_protocol.md"
PAIRED_V1 = REPO_ROOT / "reports/epoch9_relational_probe_dataset/development/paired_v1/result.json"
_BOWL_XML_WINDOWS = Path(
    "C:/assets/repos/LIBERO/libero/libero/assets/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml"
)
_BOWL_XML_WSL = Path(
    "/mnt/c/assets/repos/LIBERO/libero/libero/assets/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml"
)
BOWL_XML = _BOWL_XML_WINDOWS if _BOWL_XML_WINDOWS.exists() else _BOWL_XML_WSL
LEGACY_REFERENCE_M = 0.03
V2_ABSOLUTE_DISPLACEMENT_LIMIT_M = 0.05


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def collision_planar_radius(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    maximum = 0.0
    boxes = 0
    for geom in root.findall(".//geom"):
        if geom.get("type") != "box" or geom.get("group", "0") != "0":
            continue
        boxes += 1
        position = np.asarray([float(value) for value in geom.get("pos", "0 0 0").split()])
        size = np.asarray([float(value) for value in geom.get("size", "").split()])
        quaternion = np.asarray([float(value) for value in geom.get("quat", "1 0 0 0").split()])
        rotation = _quaternion_matrix_wxyz(quaternion)
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            corner = position + rotation @ (size * np.asarray(signs))
            maximum = max(maximum, float(np.linalg.norm(corner[:2])))
    return {
        "source": str(path).replace("\\", "/"),
        "source_sha256": sha256(path),
        "collision_box_count": int(boxes),
        "maximum_planar_collision_radius_m": float(maximum),
        "maximum_planar_collision_diameter_m": float(2.0 * maximum),
        "method": "maximum XY radius over every rotated vertex of every group-0 collision box",
    }


def clean_reset_geometry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_slot: dict[str, dict[int, list[list[float]]]] = {
        "front": defaultdict(list),
        "back": defaultdict(list),
    }
    # Only the first probe sees an untouched paired scene. Repeated mass/order
    # assignments provide an exact repeatability check without using 40..49.
    for row in payload["rows"]:
        probe = row["probes"][0]
        by_slot[probe["slot"]][int(row["demo_index"])].append(probe["initial_target_eval_only"])
    result: dict[str, Any] = {}
    for slot, identities in by_slot.items():
        identity_means = []
        repeat_spans = []
        for demo_index, rows in sorted(identities.items()):
            values = np.asarray(rows, dtype=np.float64)
            identity_means.append(np.mean(values, axis=0))
            repeat_spans.append(float(np.max(np.ptp(values, axis=0))))
        array = np.asarray(identity_means, dtype=np.float64)
        result[slot] = {
            "development_demo_indices": sorted(int(value) for value in identities),
            "identity_count": int(array.shape[0]),
            "center_xyz_min_m": [float(value) for value in np.min(array, axis=0)],
            "center_xyz_max_m": [float(value) for value in np.max(array, axis=0)],
            "center_xyz_span_m": [float(value) for value in np.ptp(array, axis=0)],
            "center_xyz_sample_std_m": [float(value) for value in np.std(array, axis=0, ddof=1)],
            "maximum_within_identity_repeat_span_m": float(max(repeat_spans)),
        }
    return {
        "source": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": sha256(path),
        "untouched_first_probe_only": True,
        "validation_or_confirmation_identity_used": False,
        "by_slot": result,
    }


def feasibility_manifest() -> list[dict[str, Any]]:
    combinations = list(itertools.product(("front", "back"), ("front-first", "back-first"), ("heaviest", "lightest")))
    front_x_permutation = [(5 * index + 3) % 24 for index in range(24)]
    front_y_permutation = [(7 * index + 1) % 24 for index in range(24)]
    back_x_permutation = [(11 * index + 5) % 24 for index in range(24)]
    back_y_permutation = [(13 * index + 7) % 24 for index in range(24)]
    rows: list[dict[str, Any]] = []
    for repeat in range(3):
        for combination_index, (heavy_slot, order, instruction) in enumerate(combinations):
            index = repeat * len(combinations) + combination_index
            unit = lambda permutation: (float(permutation[index]) + 0.5) / 24.0
            front_xy = [0.075 + 0.05 * unit(front_x_permutation), 0.125 + 0.05 * unit(front_y_permutation)]
            back_xy = [-0.175 + 0.05 * unit(back_x_permutation), 0.025 + 0.05 * unit(back_y_permutation)]
            light_slot = "back" if heavy_slot == "front" else "front"
            rows.append(
                {
                    "scene_id": f"epoch9b_dev_scene_{index:03d}",
                    "partition": "DEVELOPMENT_FEASIBILITY",
                    "generator_seed": int(910000 + index),
                    "source_state_demo_index": 30,
                    "fresh_scene_construction": "replace both candidate free-joint XY positions with the frozen in-region coordinates before settling",
                    "candidate_initial_xy_m": {"front": front_xy, "back": back_xy},
                    "mass_factor": {heavy_slot: 8.0, light_slot: 1.0},
                    "heavy_slot": heavy_slot,
                    "probe_order": order.split("-")[0:1] + (["back"] if order == "front-first" else ["front"]),
                    "instruction_property": instruction,
                    "completion_target_slot_eval_only": heavy_slot if instruction == "heaviest" else light_slot,
                }
            )
    return rows


def balance_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        values: dict[str, int] = defaultdict(int)
        for row in rows:
            value = row[field]
            key = "/".join(value) if isinstance(value, list) else str(value)
            values[key] += 1
        return dict(sorted(values.items()))

    joint: dict[str, int] = defaultdict(int)
    for row in rows:
        joint[f"{row['heavy_slot']}|{'-'.join(row['probe_order'])}|{row['instruction_property']}"] += 1
    scene_ids = [row["scene_id"] for row in rows]
    return {
        "scene_count": len(rows),
        "unique_scene_count": len(set(scene_ids)),
        "heavy_slot_counts": counts("heavy_slot"),
        "probe_order_counts": counts("probe_order"),
        "instruction_property_counts": counts("instruction_property"),
        "joint_factorial_counts": dict(sorted(joint.items())),
        "all_eight_factor_cells_equal": len(set(joint.values())) == 1 and len(joint) == 8,
        "sealed_integer_demo_indices_present": any(
            int(row["source_state_demo_index"]) in range(40, 50) for row in rows
        ),
    }


def protocol() -> dict[str, Any]:
    geometry = collision_planar_radius(BOWL_XML)
    resets = clean_reset_geometry(PAIRED_V1)
    manifest = feasibility_manifest()
    balance = balance_audit(manifest)
    return {
        "schema_version": "epoch9b.v2_task_preservation_protocol.v1",
        "frozen_at": timestamp(),
        "status": "FROZEN_BEFORE_ANY_EPOCH9B_CONTROLLER_OUTCOME",
        "branch": "codex/epoch9b-adaptive-probe-paper-continuation",
        "source_checkpoint": "d805e0a84fc8bcb720ecfd0cafeeda153aa603b4",
        "evidence_partition": "DEVELOPMENT",
        "sealed_identity_contract": {
            "validation_demo_indices": list(range(40, 45)),
            "confirmation_demo_indices": list(range(45, 50)),
            "access_authorized": False,
        },
        "geometric_basis": geometry,
        "clean_reset_basis": resets,
        "v2_absolute_displacement_rule": {
            "limit_m": V2_ABSOLUTE_DISPLACEMENT_LIMIT_M,
            "legacy_v1_reference_m_reported_only": LEGACY_REFERENCE_M,
            "threshold_set_from_epoch9_v1_0419m_outcome": False,
            "justification": [
                "0.05 m is below the independently reconstructed 0.053949 m bowl collision radius, limiting a probe to less than one object radius.",
                "The largest clean coordinate span over untouched development resets is 0.045015 m; the cap covers the task's natural placement scale while the lane rule separately prevents cross-lane motion.",
                "Expanding the observed initial XY extrema by 0.05 m only along the intended +x nudge and by 0.01 m cross-lane yields the frozen rectangles below.",
                "The absolute limit is subordinate to official post-probe task recoverability: meeting distance alone never makes a probe valid.",
            ],
        },
        "safe_center_lanes_m": {
            "coordinate_frame": "LIBERO world XY; evaluation only",
            "intended_nudge_axis": "+x for both candidates",
            "front": {"x": [0.065, 0.180], "y": [0.115, 0.175]},
            "back": {"x": [-0.185, -0.070], "y": [0.015, 0.090]},
            "lane_membership_checked_continuously": True,
            "candidate_pair_minimum_center_separation_m": float(2.0 * geometry["maximum_planar_collision_radius_m"] + 0.005),
        },
        "reachable_center_envelope_m": {
            "x": [-0.195, 0.190],
            "y": [0.005, 0.185],
            "z": [0.888, 0.915],
            "verification": "pose-adaptive scripted completion must grasp from the observed post-probe pose and achieve official task success",
        },
        "per_probe_validity": {
            "required": [
                "intended candidate physically contacted or excited (simulator contact is evaluation-only)",
                "all raw 7-D actions finite and within [-1,1]",
                "candidate center remains inside its frozen lane at every sampled step",
                "candidate displacement <= 0.05 m and is reported continuously",
                "candidate remains upright with center z in the reachable envelope",
                "no candidate-candidate or candidate-distractor collision, identity swap, fall, or workspace exit",
                "audited RGB tracker remains on the intended instance with template quality >= 0.50",
                "end effector returns within 0.05 m of episode-neutral pose and final z >= 1.10 m",
            ],
            "visual_change_reporting": "pixel dx, dy, magnitude, confidence, changed support, and continuous trajectory; no binary-only concealment",
        },
        "per_scene_task_preservation": {
            "required": [
                "both candidate identities remain trackable and distinct after sequential probing",
                "both candidates remain inside lane and reachable envelopes",
                "pose-adaptive scripted oracle completes the property-conditioned task with official env.check_success",
            ],
            "oracle_privilege_boundary": "post-probe simulator pose may route the scripted headroom controller only; it is forbidden to the probe controller and learned inference",
            "canonical_vla_endpoint": "separately run oracle-routed canonical X-VLA from the same post-probe state using the selected front/back task language; report success without substituting it for the oracle gate",
        },
        "minimal_feasibility_go": {
            "finite_bounded_actions": "48/48 probes",
            "intended_candidate_contact": ">=46/48 probes",
            "both_candidates_contacted": ">=22/24 scenes",
            "lane_and_reachability": "48/48 candidates",
            "label_blind_heavy_rank": ">=20/24 overall and >=10/12 in each heavy-slot stratum",
            "post_probe_oracle_completion": ">=20/24 scenes",
            "canonical_vla_completion": "reported separately as practical headroom; no threshold selected from its outcomes",
            "position_and_order_shortcut": "ranking gate must pass independently in each heavy-slot stratum and be reported by probe order",
        },
        "controller_information_boundary": {
            "allowed": ["agentview RGB", "ordinary robot proprioception", "commanded/executed actions", "elapsed time", "internal history"],
            "forbidden": ["simulator object pose", "mass/property label", "force", "reward", "segmentation", "success", "oracle target identity"],
            "identical_policy": "same gains, contact detector, micro-impulse magnitude, and stopping rule for both slots after applying the preregistered slot-frame transform",
        },
        "feasibility_manifest": manifest,
        "manifest_balance_audit": balance,
    }


def write_markdown(value: dict[str, Any]) -> None:
    lanes = value["safe_center_lanes_m"]
    limit = value["v2_absolute_displacement_rule"]
    geometry = value["geometric_basis"]
    resets = value["clean_reset_basis"]["by_slot"]
    lines = [
        "# Epoch 9B v2 Task-Preservation Protocol",
        "",
        f"Frozen: {value['frozen_at']}",
        "",
        f"Status: `{value['status']}`",
        "",
        "This rule was frozen before any Epoch 9B controller outcome. The old 3 cm v1 gate remains unchanged "
        "for every historical v1 result and is retained as a reported reference only.",
        "",
        "## Geometric basis",
        "",
        f"The bowl's collision geometry has a maximum planar radius of "
        f"`{geometry['maximum_planar_collision_radius_m']:.6f} m`. Across untouched first-probe states from "
        f"development demos 30..39, front reset XY spans are `{resets['front']['center_xyz_span_m'][:2]}` "
        f"and back spans are `{resets['back']['center_xyz_span_m'][:2]}`.",
        "",
        f"The v2 absolute displacement limit is `{limit['limit_m']:.3f} m`, below one collision radius. "
        "It is coupled to lane membership and task completion and was not selected from the old 4.19 cm result.",
        "",
        "| slot | center-x lane (m) | center-y lane (m) |",
        "|---|---:|---:|",
        f"| front | {lanes['front']['x']} | {lanes['front']['y']} |",
        f"| back | {lanes['back']['x']} | {lanes['back']['y']} |",
        "",
        "## Validity rule",
        "",
    ]
    lines.extend(f"- {item}" for item in value["per_probe_validity"]["required"])
    lines.extend(
        [
            "",
            "At scene level, both instances must remain trackable and reachable and the pose-adaptive scripted "
            "oracle must complete the property-conditioned task under the official success predicate. The retained "
            "canonical X-VLA path is run separately from the identical post-probe state as a practical-headroom "
            "endpoint.",
            "",
            "## Frozen 24-scene panel",
            "",
            "The manifest contains 24 fresh procedural development scenes and 48 probes. Heavy slot, probe order, "
            "and heaviest/lightest instruction are a fully crossed 2x2x2 design with three scenes per cell. Both "
            "candidate placements are independently permuted inside their original BDDL regions. The task generator "
            "uses development demo 30 only as a robot/fixture base state, replaces both candidate free-joint XY "
            "coordinates, and never reads identities 40..49.",
            "",
            "The complete manifest, exact gates, continuous metrics, and information boundary are in the companion JSON.",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite frozen Epoch 9B v2 protocol")
    value = protocol()
    audit = value["manifest_balance_audit"]
    if audit["scene_count"] != 24 or not audit["all_eight_factor_cells_equal"] or audit["sealed_integer_demo_indices_present"]:
        raise RuntimeError(f"invalid feasibility manifest balance: {audit}")
    atomic_write_json(OUTPUT_JSON, value)
    write_markdown(value)
    print(json.dumps({"status": value["status"], "scenes": audit["scene_count"], "balanced": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
