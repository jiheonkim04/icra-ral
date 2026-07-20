#!/usr/bin/env python3
"""Collect legal same-scene paired probe trajectories for Epoch 9.

Mass and simulator contact/pose are used only to construct interventions and
evaluate completed episodes.  The frozen controller branches only on fixed
slot calibration, RGB/proprioception, its own commands, and elapsed phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_epoch9_probe_controller_development import (
    BDDL_ROOT,
    DATA_ROOT,
    LIBERO_ROOT,
    SLOT_CALIBRATION,
    TASKS,
    _gray,
    _trace_step,
    feedback_move,
    load_env_class,
    make_env,
    read_demo,
    template_shift,
)
from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json
from tca_map.epoch9_active_grounding import LEGAL_TRACE_FIELDS, build_episode_specs

DEFAULT_PROTOCOL_PATH = ROOT / "reports/epoch9_active_grounding_protocol.json"
OUTPUT_ROOT = ROOT / "reports/epoch9_relational_probe_dataset"
SOURCE_TASK = TASKS["front"]
BODY_BY_SLOT = {"front": TASKS["front"]["body"], "back": TASKS["back"]["body"]}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_frozen_inputs(partition: str, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    controller = ROOT / "scripts/run_epoch9_probe_controller_development.py"
    if sha256(controller) != protocol["frozen_probe_controller"]["source_sha256"]:
        raise RuntimeError("frozen controller source hash no longer matches protocol")
    source_hdf5 = DATA_ROOT / SOURCE_TASK["hdf5"]
    if sha256(source_hdf5) != protocol["source_artifacts"]["hdf5_sha256"]:
        raise RuntimeError("source HDF5 hash no longer matches protocol")
    source_bddl = BDDL_ROOT / SOURCE_TASK["bddl"]
    if sha256(source_bddl) != protocol["source_artifacts"]["bddl_sha256"]:
        raise RuntimeError("source BDDL hash no longer matches protocol")
    paired_controller = protocol.get("paired_probe_controller")
    if paired_controller and sha256(Path(__file__).resolve()) != paired_controller["runner_sha256"]:
        raise RuntimeError("paired probe runner hash no longer matches repair protocol")
    if partition == "validation":
        freeze = ROOT / "reports/epoch9_model_freeze.json"
        if not freeze.exists() or json.loads(freeze.read_text(encoding="utf-8")).get("status") != "FROZEN_BEFORE_VALIDATION":
            raise RuntimeError("validation access requires a complete model freeze record")
    if partition == "confirmation":
        adjudication = ROOT / "reports/epoch9_validation_adjudication.json"
        if not adjudication.exists() or not json.loads(adjudication.read_text(encoding="utf-8")).get(
            "confirmation_authorized", False
        ):
            raise RuntimeError("confirmation remains sealed until validation authorizes it")
    return protocol


def new_trace() -> dict[str, list[Any]]:
    return {field: [] for field in LEGAL_TRACE_FIELDS} | {"target_contact_eval": []}


def persist_legal_trace(path: Path, trace: dict[str, list[Any]]) -> None:
    """Persist only deployable observations; privileged metrics stay in JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            phase=np.asarray(trace["phase"]),
            action=np.asarray(trace["action"], dtype=np.float32),
            eef_pos=np.asarray(trace["eef_pos"], dtype=np.float32),
            eef_quat=np.asarray(trace["eef_quat"], dtype=np.float32),
            controller_goal_pos=np.asarray(trace["controller_goal_pos"], dtype=np.float32),
            controller_error=np.asarray(trace["controller_error"], dtype=np.float32),
            rgb_diff_32=np.asarray(trace["rgb_diff_32"], dtype=np.int16),
        )
    temporary.replace(path)


def feedback_hold_with_gripper(
    env: Any,
    observation: dict[str, Any],
    target: np.ndarray,
    *,
    phase: str,
    steps: int,
    gripper: float,
    trace: dict[str, list[Any]],
    initial_gray_small: np.ndarray,
    eval_target_body: str,
) -> dict[str, Any]:
    """Hold a position while preserving an explicitly frozen gripper command."""

    for _ in range(steps):
        current = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((np.asarray(target) - current) / 0.04, -1.0, 1.0).astype(np.float32)
        action[6] = float(gripper)
        observation, _, _, _ = env.step(action)
        _trace_step(trace, env, observation, action, phase, initial_gray_small, eval_target_body)
    return observation


def run_frozen_v12_probe(
    env: Any,
    observation: dict[str, Any],
    *,
    slot: str,
    episode_id: str,
    trace_root: Path,
    push_scale: float,
    gripper_command: float,
    contact_override: list[float] | None,
    return_variant: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the exact v12 mechanics once without reading privileged state for actions."""

    import cv2
    from PIL import Image

    body = BODY_BY_SLOT[slot]
    trace = new_trace()
    initial_frame = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
    initial_gray_small = cv2.resize(_gray(initial_frame), (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    initial_target_eval = np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy()

    calibration = SLOT_CALIBRATION[slot]
    contact = np.asarray(calibration["contact_eef"], dtype=np.float64).copy()
    if contact_override is not None:
        contact = np.asarray(contact_override, dtype=np.float64)
    elif slot == "front":
        contact = np.asarray([0.050, 0.169, 0.926], dtype=np.float64)
    above = contact.copy()
    above[2] = 1.02
    pushed = contact + float(push_scale) * np.asarray(calibration["push_delta"], dtype=np.float64)
    phase_reached: dict[str, bool] = {}
    for target, phase in ((above, "approach_above"), (contact, "approach_contact"), (pushed, "probe_inward")):
        observation, phase_reached[phase] = feedback_move(
            env,
            observation,
            target,
            gripper=float(gripper_command),
            phase=phase,
            trace=trace,
            initial_gray_small=initial_gray_small,
            tolerance_m=0.025 if phase == "approach_contact" else 0.012,
            eval_target_body=body,
        )
    observation = feedback_hold_with_gripper(
        env,
        observation,
        pushed,
        phase="probe_hold",
        steps=8,
        gripper=float(gripper_command),
        trace=trace,
        initial_gray_small=initial_gray_small,
        eval_target_body=body,
    )

    prehome = np.asarray([initial_eef[0], initial_eef[1], 1.02], dtype=np.float64)
    if return_variant == "clearance_first":
        withdraw_clear = contact.copy()
        withdraw_clear[1] += 0.04 if slot == "front" else -0.04
        lift_clear = withdraw_clear.copy()
        lift_clear[2] = 1.02
        central_high = np.asarray([-0.05, 0.02, 1.02], dtype=np.float64)
        return_path = (
            (contact, "withdraw_contact"),
            (withdraw_clear, "withdraw_clear"),
            (lift_clear, "lift_clear"),
            (central_high, "return_central_high"),
            (prehome, "return_prehome"),
            (initial_eef, "return_neutral"),
        )
    elif return_variant == "legacy_low_clearance":
        low_above = contact.copy()
        low_above[2] = 0.97
        retreat_side = np.asarray([contact[0], 0.08 if slot == "front" else 0.0, 0.98], dtype=np.float64)
        central = np.asarray([-0.05, 0.02, 0.98], dtype=np.float64)
        return_path = (
            (contact, "withdraw_contact"),
            (low_above, "withdraw_low_above"),
            (retreat_side, "return_retreat_side"),
            (central, "return_central"),
            (prehome, "return_prehome"),
            (initial_eef, "return_neutral"),
        )
    else:
        raise ValueError(f"unknown return variant: {return_variant}")
    for target, phase in return_path:
        observation, phase_reached[phase] = feedback_move(
            env,
            observation,
            target,
            gripper=float(gripper_command),
            phase=phase,
            trace=trace,
            initial_gray_small=initial_gray_small,
            tolerance_m=0.004 if phase == "return_neutral" else 0.012,
            eval_target_body=body,
        )

    final_frame = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
    final_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    final_target_eval = np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy()
    actions = np.asarray(trace["action"], dtype=np.float32)
    trace_path = trace_root / "traces" / f"{episode_id}_{slot}.npz"
    persist_legal_trace(trace_path, trace)
    frame_root = trace_root / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    for label, frame in (("initial", initial_frame), ("final", final_frame)):
        destination = frame_root / f"{episode_id}_{slot}_{label}.png"
        temporary = destination.with_suffix(".tmp.png")
        Image.fromarray(frame).save(temporary)
        temporary.replace(destination)
    probe_result = {
        "slot": slot,
        "frozen_push_scale": float(push_scale),
        "frozen_gripper_command": float(gripper_command),
        "frozen_contact_eef": contact.tolist(),
        "frozen_return_variant": return_variant,
        "phase_reached": phase_reached,
        "steps": len(actions),
        "finite_bounded_actions": bool(
            actions.size and np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.0 + 1e-7
        ),
        "target_contact_eval_only": bool(any(trace["target_contact_eval"])),
        "initial_eef": initial_eef.tolist(),
        "final_eef": final_eef.tolist(),
        "final_eef_displacement_m": float(np.linalg.norm(final_eef - initial_eef)),
        "initial_target_eval_only": initial_target_eval.tolist(),
        "final_target_eval_only": final_target_eval.tolist(),
        "target_displacement_m_eval_only": float(np.linalg.norm(final_target_eval - initial_target_eval)),
        "visual_return_estimate": template_shift(initial_frame, final_frame, slot),
        "trace_path": str(trace_path.relative_to(ROOT)).replace("\\", "/"),
        "trace_fields": list(LEGAL_TRACE_FIELDS),
        "privileged_fields_in_trace": [],
    }
    return observation, probe_result


def run_episode(
    env_class: Any,
    spec: dict[str, Any],
    trace_root: Path,
    push_scale_by_slot: dict[str, float],
    gripper_command: float,
    contact_override_by_slot: dict[str, list[float]],
    return_variant: str,
) -> dict[str, Any]:
    env = None
    row = dict(spec) | {
        "timestamp": timestamp(),
        "exception": None,
        "online_inputs": [
            "fixed slot calibration",
            "agentview RGB",
            "end-effector proprioception",
            "executed action and controller-command history",
            "elapsed controller phase",
        ],
        "forbidden_online_inputs_used": [],
    }
    try:
        init_state, _ = read_demo(SOURCE_TASK, int(spec["demo_index"]))
        env, observation = make_env(env_class, SOURCE_TASK, init_state)
        initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        initial_targets = {
            slot: np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy()
            for slot, body in BODY_BY_SLOT.items()
        }
        for slot in ("front", "back"):
            factor = float(spec[f"{slot}_mass_factor"])
            if factor != 1.0:
                apply_intervention(
                    env.sim.model,
                    {"axis": "target_mass", "body_name": BODY_BY_SLOT[slot], "arrays": ["body_mass", "body_inertia"], "factor": factor},
                )
        env.sim.forward()
        probes: list[dict[str, Any]] = []
        for slot in spec["probe_order"]:
            observation, probe = run_frozen_v12_probe(
                env,
                observation,
                slot=str(slot),
                episode_id=str(spec["episode_id"]),
                trace_root=trace_root,
                push_scale=float(push_scale_by_slot[str(slot)]),
                gripper_command=float(gripper_command),
                contact_override=contact_override_by_slot.get(str(slot)),
                return_variant=return_variant,
            )
            probes.append(probe)
        final_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        final_targets = {
            slot: np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64).copy()
            for slot, body in BODY_BY_SLOT.items()
        }
        row.update(
            {
                "completed": True,
                "probes": probes,
                "episode_initial_eef_eval_only": initial_eef.tolist(),
                "episode_final_eef_eval_only": final_eef.tolist(),
                "episode_final_eef_displacement_m_eval_only": float(np.linalg.norm(final_eef - initial_eef)),
                "candidate_final_displacement_m_eval_only": {
                    slot: float(np.linalg.norm(final_targets[slot] - initial_targets[slot])) for slot in BODY_BY_SLOT
                },
                "simulator_state_used_for_actions": False,
                "expert_actions_used_online": False,
                "mass_used_online": False,
            }
        )
    except Exception as exc:  # pragma: no cover - simulator runtime evidence
        row.update({"completed": False, "exception": f"{type(exc).__name__}: {exc}"})
    finally:
        if env is not None:
            env.close()
    return row


def summarize(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    completed = [row for row in rows if row.get("completed")]
    probes = [probe for row in completed for probe in row.get("probes", [])]
    visual = [float(np.hypot(p["visual_return_estimate"]["dx"], p["visual_return_estimate"]["dy"])) for p in probes]
    candidate_displacements = [
        float(value)
        for row in completed
        for value in row.get("candidate_final_displacement_m_eval_only", {}).values()
    ]
    summary = {
        "expected_episodes": expected,
        "recorded_episodes": len(rows),
        "completed_episodes": len(completed),
        "exception_count": sum(row.get("exception") is not None for row in rows),
        "probe_count": len(probes),
        "contact_fraction": float(np.mean([p["target_contact_eval_only"] for p in probes])) if probes else 0.0,
        "bounded_action_fraction": float(np.mean([p["finite_bounded_actions"] for p in probes])) if probes else 0.0,
        "max_candidate_final_displacement_m": max(candidate_displacements, default=float("inf")),
        "max_episode_final_eef_displacement_m": max(
            (float(row["episode_final_eef_displacement_m_eval_only"]) for row in completed), default=float("inf")
        ),
        "max_visual_return_residual_pixels": max(visual, default=float("inf")),
    }
    summary["execution_gate_pass"] = bool(
        len(rows) == expected
        and len(completed) == expected
        and summary["exception_count"] == 0
        and summary["contact_fraction"] == 1.0
        and summary["bounded_action_fraction"] == 1.0
        and summary["max_candidate_final_displacement_m"] <= 0.03
        and summary["max_episode_final_eef_displacement_m"] <= 0.05
        and summary["max_visual_return_residual_pixels"] <= 5.0
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("development", "validation", "confirmation"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--demo-indices", help="comma-separated identity filter; development diagnostics only")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = verify_frozen_inputs(args.partition, protocol_path)
    specs = list(protocol["episode_specs"][args.partition])
    if args.demo_indices:
        if args.partition != "development":
            raise RuntimeError("identity filtering is allowed only for development diagnostics")
        selected_indices = {int(value) for value in args.demo_indices.split(",")}
        specs = [spec for spec in specs if int(spec["demo_index"]) in selected_indices]
    if args.max_episodes is not None:
        specs = specs[: args.max_episodes]
    run_root = OUTPUT_ROOT / args.partition / args.run_id
    output = run_root / "result.json"
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
    else:
        result = {
            "schema_version": "epoch9.relational_probe_dataset.v1",
            "timestamp": timestamp(),
            "status": "RUNNING",
            "evidence_class": args.partition.upper(),
            "partition": args.partition,
            "run_id": args.run_id,
            "protocol_path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "protocol_status": protocol["status"],
            "rows": [],
        }
    recorded = {row["episode_id"] for row in result["rows"]}
    pending = [spec for spec in specs if spec["episode_id"] not in recorded]
    env_class = load_env_class()
    push_scale_by_slot = {
        slot: float(value)
        for slot, value in protocol.get(
            "paired_probe_controller",
            {"push_scale_by_slot": {"front": 2.0 / 3.0, "back": 2.0 / 3.0}},
        )["push_scale_by_slot"].items()
    }
    gripper_command = float(protocol.get("paired_probe_controller", {}).get("gripper_command", 1.0))
    contact_override_by_slot = {
        str(slot): [float(component) for component in value]
        for slot, value in protocol.get("paired_probe_controller", {}).get("contact_override_by_slot", {}).items()
    }
    return_variant = str(
        protocol.get("paired_probe_controller", {}).get("return_variant", "legacy_low_clearance")
    )
    for spec in pending:
        result["rows"].append(
            run_episode(
                env_class,
                spec,
                run_root,
                push_scale_by_slot,
                gripper_command,
                contact_override_by_slot,
                return_variant,
            )
        )
        result["summary"] = summarize(result["rows"], len(specs))
        atomic_write_json(output, result)
    result["summary"] = summarize(result["rows"], len(specs))
    result["status"] = "FINISHED" if len(result["rows"]) == len(specs) else "RUNNING"
    atomic_write_json(output, result)
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
